#!/usr/bin/env python3
"""BF16 teacher probe: first denoising step of the first generation block for a prompt (int32 ids file, as the
engine's bench_gen reads). Streams the full model layer by layer (s03 helpers), with the block-causal mask the
engine uses, and prints per masked position the teacher's top-3 tokens and probabilities.

  python teacher_probe.py <prompt.bin> [--steps N]

--steps N > 1 continues the block with plain greedy threshold decoding (M2T only, threshold 0.5, at least one
token per step) to show what the teacher would write; each step is a full streamed pass.
"""
import argparse
import struct
import time

import numpy as np
import torch

from s03_teacher_pass import DEV, Weights, build_layer, load_classes, log

BLOCK, MASK = 32, 156895


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--steps", type=int, default=1)
    a = ap.parse_args()
    b = open(a.prompt, "rb").read()
    prompt = list(struct.unpack(f"{len(b) // 4}i", b))
    full = len(prompt) // BLOCK
    L = (full + 1) * BLOCK
    ids = prompt + [MASK] * (L - len(prompt))
    gen0 = len(prompt)
    cfg, Layer, Rotary, RMSNorm = load_classes()
    W = Weights()
    rot = Rotary(cfg).to(DEV)
    blk = torch.arange(L, device=DEV) // BLOCK
    mask = torch.where(blk[None, :] <= blk[:, None], 0.0, float("-inf")).to(torch.bfloat16)[None, None]
    pos = torch.arange(L, device=DEV)[None]
    norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    norm.weight.data = W.get("model.norm.weight")
    norm = norm.to(DEV, torch.bfloat16)
    from tokenizers import Tokenizer
    from s03_teacher_pass import MODEL
    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    for step in range(a.steps):
        t0 = time.time()
        emb = W.get("model.word_embeddings.weight")
        x = emb[torch.tensor(ids)].to(DEV, torch.bfloat16)[None]
        del emb
        for li in range(cfg.num_hidden_layers):
            layer = build_layer(cfg, Layer, W, li)
            with torch.no_grad():
                x = layer(x, attention_mask=mask, position_ids=pos, position_embeddings=rot(x, pos))[0]
            del layer
            torch.cuda.empty_cache()
        head = W.get("lm_head.weight").to(DEV, torch.bfloat16)
        with torch.no_grad():
            p = (norm(x[0, gen0:]).float() @ head.float().T).softmax(-1)
        del head
        torch.cuda.empty_cache()
        v, i = p.topk(3, dim=-1)
        log(f"step {step}: {time.time() - t0:.0f}s")
        for j in range(L - gen0):
            mark = "*" if ids[gen0 + j] == MASK else " "
            print(f"  {mark}pos {j:2d}: " + "  ".join(f"{tok.id_to_token(int(i[j, k]))!r} {float(v[j, k]):.3f}" for k in range(3)))
        masked = [j for j in range(L - gen0) if ids[gen0 + j] == MASK]
        if not masked or step + 1 == a.steps:
            break
        hi = [j for j in masked if float(v[j, 0]) > 0.5] or [max(masked, key=lambda j: float(v[j, 0]))]
        for j in hi:
            ids[gen0 + j] = int(i[j, 0])
        print("  text so far:", repr(tok.decode(ids[gen0:], skip_special_tokens=False).replace("<|mask|>", "_")))


if __name__ == "__main__":
    main()
