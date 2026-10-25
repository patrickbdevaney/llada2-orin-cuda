#!/usr/bin/env python3
"""Integration case for the engine's first layers: the BF16 HF reference (CPU) over [context C tokens | one 32-token
block, half of it masked] with the block-causal mask the engine implements, capturing per layer
  K/V of the context (post q/k-norm + RoPE; V rotated into the engine's folded basis v R),
  the block's hidden state after each layer, and the MoE layers' top-8 routing of the block tokens."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

Q_DIR = Path(__file__).resolve().parents[2] / "quant"
sys.path.insert(0, str(Q_DIR))
import quantlib as QL  # noqa: E402
import s03_teacher_pass as S3  # noqa: E402

S3.DEV = "cpu"
ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--ctx", type=int, default=256)
ap.add_argument("--out", default=str(Path(__file__).parent / "layers"))
ap.add_argument("--prompt", default=None, help="int32 prompt ids: context = its full blocks, block = the rest + masks")
a = ap.parse_args()
torch.set_grad_enabled(False)
cfg, Layer, Rotary, _ = S3.load_classes()
from transformers.modeling_utils import AttentionInterface  # noqa: E402
from transformers.integrations.sdpa_attention import sdpa_attention_forward  # noqa: E402
cap = {}


def capture(module, query, key, value, attention_mask, **kw):
    cap["k"], cap["v"] = key.detach().clone(), value.detach().clone()
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)


AttentionInterface.register("cap", capture)
cfg._attn_implementation = "cap"
W = S3.Weights()
cdir = S3.WORK / "corpus"
tokens = np.load(cdir / "tokens.npy")
rows = [json.loads(l) for l in open(cdir / "rows.jsonl")]
r = next(x for x in rows if x["heldout"] and x["bucket"] == "code")
if a.prompt:
    pr = np.fromfile(a.prompt, dtype=np.int32).astype(np.int64)
    a.ctx = len(pr) // 32 * 32
    L = a.ctx + 32
    ids = np.full(L, 156895, np.int64)
    ids[: len(pr)] = pr
    r = {"row": -1}
else:
    L = a.ctx + 32
    ids = tokens[r["offset"]: r["offset"] + L].astype(np.int64).copy()
    rng = np.random.default_rng(0)
    blk = np.arange(a.ctx, L)
    ids[blk[rng.random(32) < 0.5]] = 156895
nb = L // 32
m = torch.full((L, L), float("-inf"))
for i in range(nb):
    m[i * 32:(i + 1) * 32, : (i + 1) * 32] = 0
m = m.to(torch.bfloat16)[None, None]
rot = Rotary(cfg)
x = W.get("model.word_embeddings.weight")[torch.from_numpy(ids)].to(torch.bfloat16)[None]
pos = torch.arange(L)[None]
R = QL.residual_rotation(cfg.head_dim, seed=20261025, device="cpu")
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
ids[a.ctx:].astype(np.int32).tofile(out / "block_ids.bin")
ids[: a.ctx].astype(np.int32).tofile(out / "ctx_ids.bin")
for li in range(a.layers):
    layer = S3.build_layer(cfg, Layer, W, li)
    if hasattr(layer.mlp, "gate"):
        mlp = layer.mlp
        sel = {}

        def fwd(xx, _m=mlp):
            h = xx.reshape(-1, xx.shape[-1])
            topk, w, _ = S3.route(_m.gate, h)
            sel["topk"] = topk[a.ctx:].numpy().astype(np.int32)
            return S3.moe_forward(_m, xx, {}), None
        mlp.forward = fwd
    x = layer(x, attention_mask=m, position_ids=pos, position_embeddings=rot(x, pos))[0]
    k = cap["k"][0, :, : a.ctx].transpose(0, 1).float()                  # [C, 4, 128]
    v = (cap["v"][0, :, : a.ctx].transpose(0, 1).float() @ R)              # folded basis
    k.half().numpy().tofile(out / f"k_L{li}.bin")
    v.half().numpy().tofile(out / f"v_L{li}.bin")
    x[0, a.ctx:].float().numpy().tofile(out / f"h_L{li}.bin")
    if "sel" in dir() and sel:
        sel["topk"].tofile(out / f"topk_L{li}.bin")
    print(f"layer {li}: block |h| rms {x[0, a.ctx:].float().pow(2).mean().sqrt():.3f}", flush=True)
    del layer
json.dump({"ctx": a.ctx, "layers": a.layers, "row": r["row"]}, open(out / "case.json", "w"))
