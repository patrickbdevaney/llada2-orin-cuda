#!/usr/bin/env python3
"""s07b: pack everything that is not a routed expert into the engine directory (format i8 tiles, moe_i8.cu):

  nonexpert_L{li}.bin : qkv [3072, 2048], dense [2048, 2048], gate_up (shared expert, or layer 0's dense MLP; rows
                        permuted by packlib.gu_row_perm), down -- from s05n's codes at --bits (rotated bases, see
                        s05n_nonexpert.py), each 4 KiB aligned
  lm_head.bin         : lm_head [157184, 2048] at --lm-bits
  small.bin           : fp32 norms (input, post-attention, q, k per layer; final), router weights (bf16) and expert
                        bias (fp32) per MoE layer, the residual-rotation signs d (Q = diag(d) H / sqrt(2048))
  embed.bin           : word embeddings fp16 [157184, 2048] -- read from NVMe by the host (mmap), never resident
  model.json          : config, bits, and {name: [offset, bytes, dtype, shape]} for every tensor above
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import packlib as PL
import quantlib as QL
from s03_teacher_pass import MODEL, Weights, load_classes

WORK = Path(os.environ.get("LLADA_WORK", Path.home() / "models/llada-work"))
ALIGN = 4096


class Writer:
    def __init__(self, path: Path):
        self.f = open(path, "wb")
        self.off = 0
        self.index = {}

    def add(self, name: str, arr: np.ndarray, dtype: str, shape):
        pad = (-self.off) % ALIGN
        self.f.write(b"\0" * pad)
        self.off += pad
        b = np.ascontiguousarray(arr).tobytes()
        self.index[name] = [self.off, len(b), dtype, list(shape)]
        self.f.write(b)
        self.off += len(b)

    def close(self):
        self.f.close()
        return self.index


def tiles(cz, bits, perm=None):
    c, s, z = (t.numpy() for t in cz)
    if perm is not None:
        c, s, z = c[perm], s[perm], z[perm]
    parts = []
    for r0 in range(0, c.shape[0], 4096):           # strips are independent: chunk rows (lm_head has 157k)
        cc, ss, zz = c[r0:r0 + 4096], s[r0:r0 + 4096], z[r0:r0 + 4096]
        blob = PL.pack_matrix_i8(cc, ss, zz, bits)
        c2, s2, z2 = PL.unpack_matrix_i8(blob, *cc.shape, bits)
        assert (c2 == cc).all() and (s2.view(np.uint16) == ss.view(np.uint16)).all() and (z2 == zz).all()
        parts.append(blob)
    return np.concatenate(parts), list(c.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--lm-bits", type=int, default=8)
    ap.add_argument("--out", default=str(Path.home() / "models/llada2mini-orin"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg, *_ = load_classes()
    W = Weights()
    codes = WORK / "s05n" / "codes"
    files = {}
    for li in range(cfg.num_hidden_layers):
        cz = torch.load(codes / f"L{li}_b{a.bits}.pt")
        w = Writer(out / f"nonexpert_L{li}.bin")
        I = cfg.intermediate_size if li < cfg.first_k_dense_replace else cfg.moe_intermediate_size
        for name, perm in (("qkv", None), ("dense", None), ("gate_up", PL.gu_row_perm(I)), ("down", None)):
            blob, shape = tiles(cz[name], a.bits, perm)
            w.add(name, blob, f"i8tiles{a.bits}", shape)
        files[f"nonexpert_L{li}.bin"] = w.close()
        print(f"L{li} non-expert packed ({files[f'nonexpert_L{li}.bin']['down'][0] / 1e6:.1f} MB+)", flush=True)
    cz = torch.load(codes / f"lm_head_b{a.lm_bits}.pt")
    w = Writer(out / "lm_head.bin")
    blob, shape = tiles(cz, a.lm_bits)
    w.add("lm_head", blob, f"i8tiles{a.lm_bits}", shape)
    files["lm_head.bin"] = w.close()

    w = Writer(out / "small.bin")
    f32 = lambda t: t.float().numpy().astype(np.float32)          # noqa: E731
    for li in range(cfg.num_hidden_layers):
        p = f"model.layers.{li}."
        for k in ("input_layernorm.weight", "post_attention_layernorm.weight", "attention.query_layernorm.weight",
                  "attention.key_layernorm.weight"):
            t = W.get(p + k)
            w.add(f"L{li}.{k}", f32(t), "f32", t.shape)
        if li >= cfg.first_k_dense_replace:
            t = W.get(p + "mlp.gate.weight")
            w.add(f"L{li}.router", t.to(torch.bfloat16).view(torch.int16).numpy(), "bf16", t.shape)
            t = W.get(p + "mlp.gate.expert_bias")
            w.add(f"L{li}.expert_bias", f32(t), "f32", t.shape)
    t = W.get("model.norm.weight")
    w.add("final_norm", f32(t), "f32", t.shape)
    Q = QL.residual_rotation(cfg.hidden_size, device="cpu")
    d = (Q[:, 0] * np.sqrt(cfg.hidden_size)).numpy().astype(np.float32)   # first column of diag(d) H is d / sqrt(n)
    assert np.allclose(np.abs(d), 1.0)
    w.add("signs", d, "f32", d.shape)
    files["small.bin"] = w.close()

    if not (out / "embed.bin").exists():
        e = W.get("model.word_embeddings.weight").to(torch.float16)
        e.numpy().tofile(out / "embed.bin")
        files["embed.bin"] = {"embed": [0, e.numel() * 2, "f16", list(e.shape)]}
        del e
    else:
        files["embed.bin"] = {"embed": [0, cfg.vocab_size * cfg.hidden_size * 2, "f16", [cfg.vocab_size, cfg.hidden_size]]}
    meta = {"config": {k: getattr(cfg, k) for k in (
        "num_hidden_layers", "hidden_size", "intermediate_size", "moe_intermediate_size", "num_experts",
        "num_experts_per_tok", "expert_capacity", "block_size", "num_attention_heads", "num_key_value_heads",
        "head_dim", "rotary_dim", "rope_theta", "rms_norm_eps", "routed_scaling_factor", "vocab_size",
        "first_k_dense_replace")},
        "mask_id": 156895, "nonexpert_bits": a.bits, "lm_bits": a.lm_bits, "kv_format": "4c/4",
        "rotation": {"residual_seed": 20261024, "kv_seed": 20261025, "v_folded": True},
        "files": files}
    json.dump(meta, open(out / "model.json", "w"), indent=1)
    print("model.json written")


if __name__ == "__main__":
    main()
