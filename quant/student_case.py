#!/usr/bin/env python3
"""The quantized model as s06 simulates it (alloc experts through the rotated W·A8 path, s05n 8-bit non-experts,
4c/4 KV for committed blocks), run over a prompt case in the engine's serving layout: blocks processed with the
block-causal mask, each block attending to the quantized K/V of earlier blocks and its own fresh K/V.
Writes h_L*.bin / topk_L*.bin like tests/gen_layer_case.py, so engine/tests/test_layers <dir> self compares the
engine against this simulation instead of the BF16 teacher.

  python student_case.py <prompt.bin> <out dir> [--alloc npy] [--no-kv] [--teacher]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import quantlib as QL
from s03_teacher_pass import DEV, WORK, Weights, build_layer, load_classes, route
from s06_arbitrate import QuantExperts, fake_quant, rotated_moe

ap = argparse.ArgumentParser()
ap.add_argument("prompt")
ap.add_argument("out")
ap.add_argument("--alloc", default=str(WORK / "s05" / "alloc" / "32768_int4_noprune.npy"))
ap.add_argument("--no-kv", action="store_true")
ap.add_argument("--kv", default="4c4", choices=["4c4", "k8v4"], help="4c4: K 4-bit per channel per block, V 4-bit per "
                "token; k8v4: K int8 per token (symmetric) with int8 Q per row, V 4-bit per token, fresh K int8 too")
ap.add_argument("--teacher", action="store_true", help="BF16 everything (layout check only)")
ap.add_argument("--uniform", type=int, default=0, help="every expert at this many bits (s05 bank) instead of --alloc")
ap.add_argument("--readout", action="store_true", help="also print the first-step top-3 at the block's masked positions")
a = ap.parse_args()
torch.set_grad_enabled(False)

pr = np.fromfile(a.prompt, dtype=np.int32).astype(np.int64)
ctx = len(pr) // 32 * 32
L = ctx + 32
ids = np.full(L, 156895, np.int64)
ids[: len(pr)] = pr
nb = L // 32
cfg, Layer, Rotary, _ = load_classes()
from transformers.modeling_utils import AttentionInterface  # noqa: E402
from transformers.integrations.sdpa_attention import sdpa_attention_forward  # noqa: E402

KVQ = not (a.no_kv or a.teacher)
R = QL.residual_rotation(cfg.head_dim, seed=20261025, device=DEV)


def block_attention(module, query, key, value, attention_mask, **kw):
    """query/key/value [B, h, L, D]. Block i reads 4c/4-quantized K/V of blocks < i and its own fresh K/V."""
    if not KVQ:
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)
    B_, hk, L_, D = key.shape
    fresh_k = key
    if a.kv == "k8v4":
        def sym8(x):
            sc = (x.float().abs().amax(-1, keepdim=True) / 127).clamp_min(1e-8).half().float()
            return (torch.round(x.float() / sc).clamp(-127, 127) * sc).to(x.dtype)
        kq = sym8(key)
        fresh_k = kq
        query = sym8(query)
    else:
        kt = key.float().reshape(B_, hk, L_ // 32, 32, D).transpose(-1, -2)
        kq = fake_quant(kt, 4, 32).transpose(-1, -2).reshape(key.shape).to(key.dtype)
    vr = value.float() @ R                                            # V quantized in the engine's folded basis
    vq = (fake_quant(vr, 4, D) @ R.T).to(value.dtype)
    outs = []
    for i in range(L_ // 32):
        r = slice(i * 32, (i + 1) * 32)
        k_i = torch.cat([kq[..., : i * 32, :], fresh_k[..., r, :]], dim=-2)
        v_i = torch.cat([vq[..., : i * 32, :], value[..., r, :]], dim=-2)
        o, _ = sdpa_attention_forward(module, query[..., r, :], k_i, v_i, None, **kw)
        outs.append(o)
    return torch.cat(outs, dim=1), None                                # [B, q, heads, D]


AttentionInterface.register("blockq", block_attention)
cfg._attn_implementation = "blockq"
W = Weights()
rot = Rotary(cfg).to(DEV)
blk = torch.arange(L, device=DEV) // 32
mask = torch.where(blk[None, :] <= blk[:, None], 0.0, float("-inf")).to(torch.bfloat16)[None, None]
pos = torch.arange(L, device=DEV)[None]
x = W.get("model.word_embeddings.weight")[torch.from_numpy(ids)].to(DEV, torch.bfloat16)[None]
bits_all = np.load(a.alloc).astype(np.int8)
if a.uniform:
    bits_all[:] = a.uniform
moe_layers = [l for l in range(cfg.num_hidden_layers) if l >= cfg.first_k_dense_replace]
Q = QL.residual_rotation(cfg.hidden_size)
H512 = QL.hadamard(cfg.moe_intermediate_size)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
ids[ctx:].astype(np.int32).tofile(out / "block_ids.bin")
ids[:ctx].astype(np.int32).tofile(out / "ctx_ids.bin")
for li in range(cfg.num_hidden_layers):
    layer = build_layer(cfg, Layer, W, li)
    if not a.teacher:
        params = dict(layer.named_parameters())
        for k, v in torch.load(WORK / "s05n" / "eff" / f"L{li}_b8.pt").items():
            params[k].data.copy_(v.to(DEV))
    sel = {}
    if li in moe_layers:
        mlp = layer.mlp
        gamma = layer.post_attention_layernorm.weight.float()
        if a.teacher:
            from s03_teacher_pass import moe_forward
            fwd = lambda xx, _m=mlp: moe_forward(_m, xx, {})              # noqa: E731
        else:
            mi = moe_layers.index(li)
            qe = QuantExperts(WORK / "s05" / "q" / f"L{li}", bits_all[mi], 128)
            for ex in mlp.experts:
                for p_ in ex.parameters():
                    p_.data = torch.empty(0, device=DEV, dtype=p_.dtype)
            fwd = lambda xx, _m=mlp, _q=qe, _g=gamma: rotated_moe(_m, xx, _g, Q, H512, _q.weights, 8)  # noqa: E731

        def wrapped(xx, _f=fwd, _m=mlp):
            topk, _, _ = route(_m.gate, xx.reshape(-1, xx.shape[-1]))
            sel["topk"] = topk[ctx:].cpu().numpy().astype(np.int32)
            return _f(xx), None
        mlp.forward = wrapped
    x = layer(x, attention_mask=mask, position_ids=pos, position_embeddings=rot(x, pos))[0]
    x[0, ctx:].float().cpu().numpy().tofile(out / f"h_L{li}.bin")
    if sel:
        sel["topk"].tofile(out / f"topk_L{li}.bin")
    print(f"layer {li}: block |h| rms {x[0, ctx:].float().pow(2).mean().sqrt():.3f}", flush=True)
    del layer
    torch.cuda.empty_cache()
if a.readout:
    from tokenizers import Tokenizer
    from s03_teacher_pass import MODEL
    from s03_teacher_pass import load_classes as _lc
    _, _, _, RMSNorm = _lc()
    norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    norm.weight.data = W.get("model.norm.weight")
    norm = norm.to(DEV, torch.bfloat16)
    head = W.get("lm_head.weight").to(DEV, torch.bfloat16).float()
    p = (norm(x[0, ctx:]).float() @ head.T).softmax(-1)
    v, i = p.topk(3, dim=-1)
    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    for j in range(len(pr) - ctx, 32):
        print(f"  pos {j:2d}: " + "  ".join(f"{tok.id_to_token(int(i[j, k]))!r} {float(v[j, k]):.3f}" for k in range(3)))
json.dump({"ctx": ctx, "layers": cfg.num_hidden_layers, "row": -1}, open(out / "case.json", "w"))
