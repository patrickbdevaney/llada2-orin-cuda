#!/usr/bin/env python3
"""s05n: GPTQ for everything that is not a routed expert -- attention qkv / dense, shared experts, the dense layer-0
MLP and lm_head -- at --bits (default 4 and 8), in the bases the engine will run:

  qkv     W' = (W γ_in) Q, input x' = n_in Q                     (n = RMS-normalised, no γ; Q as for the experts)
          V rows per KV head folded with R (v' = v R): attention output is then a R per head, at no runtime cost
  dense   W' = W blockdiag(R)                                     input a' = a R per head
  shared  gate/up (W γ_post) Q, down Qᵀ W H512                      as the routed experts (output in the Q basis)
  dense0  gate/up (W γ_post) Q, down unrotated (5120 is not a power of two)
  lm_head W' = (W γ_final) Q                                      Hessian from noisy-stream (decoded) positions only
R = KV rotation (128-point random-sign Hadamard, seed 20261025; the same R s06's +kv:...h arms use).

Hessians come from the BF16 teacher streamed over the same training rows as s05 (--row-stride), clean and noisy
streams; every 4th row is held back to measure each matrix's relative output error
  rel = tr(ΔW H_eval ΔWᵀ) / tr(W H_eval Wᵀ).
Outputs per layer (WORK/s05n):
  codes/L{li}_b{b}.pt   {name: (codes u8, scale f16, zero u8)} in the rotated basis      (engine packer)
  eff/L{li}_b{b}.pt     {state-dict key: bf16 weight mapped back to the original basis} (s06 +ne: arms)
  codes|eff/lm_head_b{b}.pt, report.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import quantlib as QL
from s03_teacher_pass import (DEV, WORK, Weights, batches, build_layer, load_classes, log, moe_forward,
                              noisy_ids, stream_mask)

KV_SEED = 20261025


class Hess:
    def __init__(self):
        self.h = {}

    def add(self, key: str, x: torch.Tensor, ev: torch.Tensor):
        """x [N, C] fp32, ev [N] bool (eval token)."""
        for part, sel in (("tr", ~ev), ("ev", ev)):
            xs = x[sel]
            if xs.shape[0]:
                k = (key, part)
                if k not in self.h:
                    self.h[k] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
                self.h[k].addmm_(xs.T, xs)

    def get(self, key, part):
        return self.h[(key, part)]


def rel_err(W, Wq, He):
    d = (Wq - W).float()
    return float(((d @ He) * d).sum() / ((W.float() @ He) * W.float()).sum())


def quantize(W, Htr, bits, group, prep=None):
    """W [R, C] fp32 -> (dequantized [R, C], (codes, scale, zero)); rows chunked (lm_head has 157k)."""
    prep = prep if prep is not None else QL.prepare(Htr[None])
    outs, cs, ss, zs = [], [], [], []
    for r0 in range(0, W.shape[0], 8192):
        Q_, c, s, z = QL.gptq(W[None, r0:r0 + 8192], None, bits, group, prep=prep, codes=True)
        outs.append(Q_[0]); cs.append(c[0].cpu()); ss.append(s[0].cpu()); zs.append(z[0].cpu())
    return torch.cat(outs), (torch.cat(cs), torch.cat(ss), torch.cat(zs)), prep


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(WORK / "corpus"))
    ap.add_argument("--row-stride", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--group", type=int, default=128)
    a = ap.parse_args()

    out = WORK / "s05n"
    for d in ("codes", "eff"):
        (out / d).mkdir(parents=True, exist_ok=True)
    cdir = Path(a.corpus)
    tokens, elig = np.load(cdir / "tokens.npy"), np.load(cdir / "elig.npy")
    rows = [json.loads(l) for l in open(cdir / "rows.jsonl")]
    rows = [r for r in rows if not r["heldout"]][:: a.row_stride]
    off2, o = {}, 0
    for r in rows:
        off2[r["row"]] = o
        o += 2 * r["length"]
    T2, H = o, 2048
    evrow = {r["row"]: (i % 4 == 1) for i, r in enumerate(rows)}
    log(f"{len(rows)} training rows ({sum(evrow.values())} for error eval), {T2:,} stream tokens")

    hsd = WORK / "hsn"
    hsd.mkdir(parents=True, exist_ok=True)
    mm = {k: np.lib.format.open_memmap(hsd / f"{k}.npy", mode="w+", dtype=np.int16, shape=(T2, H)) for k in "ab"}
    cfg, Layer, Rotary, RMSNorm = load_classes()
    W = Weights()
    rot = Rotary(cfg).to(DEV)
    Q = QL.residual_rotation(H)
    R = QL.residual_rotation(cfg.head_dim, seed=KV_SEED)
    H512 = QL.hadamard(cfg.moe_intermediate_size)
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    report = {"bits": a.bits, "group": a.group, "layers": {}}

    noisy = {r["row"]: noisy_ids(tokens[r["offset"]: r["offset"] + r["length"]],
                                 elig[r["offset"]: r["offset"] + r["length"]], 1_000_003 * r["row"] + 17)
             for r in rows}
    emb = W.get("model.word_embeddings.weight").to(DEV, torch.bfloat16)
    for r in rows:
        ids = np.concatenate([tokens[r["offset"]: r["offset"] + r["length"]], noisy[r["row"]][0]])
        mm["a"][off2[r["row"]]: off2[r["row"]] + 2 * r["length"]] = \
            emb[torch.from_numpy(ids.astype(np.int64)).to(DEV)].view(torch.int16).cpu().numpy()
    del emb

    masks: dict[int, torch.Tensor] = {}
    for li in range(cfg.num_hidden_layers):
        t0 = time.time()
        src, dst = mm["a" if li % 2 == 0 else "b"], mm["b" if li % 2 == 0 else "a"]
        layer = build_layer(cfg, Layer, W, li)
        att, mlp = layer.attention, layer.mlp
        g_in = layer.input_layernorm.weight.float()
        g_post = layer.post_attention_layernorm.weight.float()
        is_moe = hasattr(mlp, "gate")
        hs = Hess()
        cur_ev = {}

        def hook(key, fn):
            def pre(mod, args):
                x = args[0].reshape(-1, args[0].shape[-1]).float()
                hs.add(key, fn(x), cur_ev["m"])
            return pre

        hooks = [att.query_key_value.register_forward_pre_hook(hook("qkv", lambda x: (x / g_in) @ Q)),
                 att.dense.register_forward_pre_hook(hook("dense", lambda x: (x.view(-1, nh, hd) @ R).view(-1, nh * hd)))]
        sm = mlp.shared_experts if is_moe else mlp
        hooks.append(sm.gate_proj.register_forward_pre_hook(hook("gu", lambda x: (x / g_post) @ Q)))
        hooks.append(sm.down_proj.register_forward_pre_hook(
            hook("down", (lambda x: x @ H512) if is_moe else (lambda x: x))))
        if is_moe:
            mlp.forward = lambda x, _m=mlp: (moe_forward(_m, x, {}), None)
        for L, rs in batches(rows, a.max_tokens):
            if L not in masks:
                masks.clear()
                masks[L] = stream_mask(L, DEV)
            cur_ev["m"] = torch.tensor(np.repeat([evrow[r["row"]] for r in rs], 2 * L), device=DEV)
            x = torch.stack([torch.from_numpy(np.asarray(src[off2[r["row"]]: off2[r["row"]] + 2 * L]))
                             for r in rs]).to(DEV).view(torch.bfloat16)
            pos = torch.arange(L, device=DEV).repeat(2).unsqueeze(0).expand(len(rs), -1)
            y = layer(x, attention_mask=masks[L][None, None].expand(len(rs), 1, -1, -1),
                      position_ids=pos, position_embeddings=rot(x, pos))[0]
            for j, r in enumerate(rs):
                dst[off2[r["row"]]: off2[r["row"]] + 2 * L] = y[j].contiguous().view(torch.int16).cpu().numpy()
        dst.flush()
        for h_ in hooks:
            h_.remove()

        # ---- rotated-basis weights
        qkv = att.query_key_value.weight.float()
        Wq = (qkv * g_in) @ Q
        v0 = (nh + nkv) * hd
        Wq[v0:] = (R.T @ Wq[v0:].view(nkv, hd, H)).reshape(nkv * hd, H)
        mats = {"qkv": (Wq, "qkv"),
                "dense": ((att.dense.weight.float().view(H, nh, hd) @ R).view(H, nh * hd), "dense")}
        g_, u_, d_ = sm.gate_proj.weight.float(), sm.up_proj.weight.float(), sm.down_proj.weight.float()
        mats["gate_up"] = (torch.cat([(g_ * g_post) @ Q, (u_ * g_post) @ Q]), "gu")
        mats["down"] = ((Q.T @ d_ @ H512) if is_moe else d_, "down")
        lrep = {}
        for b in a.bits:
            codes, eff = {}, {}
            for name, (Wr, hk) in mats.items():
                Wdq, cz, _ = quantize(Wr, hs.get(hk, "tr"), b, a.group)
                lrep[f"{name}_b{b}"] = rel_err(Wr, Wdq, hs.get(hk, "ev"))
                codes[name] = cz
                # back to the original basis for s06
                if name == "qkv":
                    Wdq[v0:] = (R @ Wdq[v0:].view(nkv, hd, H)).reshape(nkv * hd, H)
                    eff["attention.query_key_value.weight"] = (Wdq @ Q.T) / g_in
                elif name == "dense":
                    eff["attention.dense.weight"] = (Wdq.view(H, nh, hd) @ R.T).reshape(H, nh * hd)
                elif name == "gate_up":
                    I_ = Wdq.shape[0] // 2
                    pre = "mlp.shared_experts." if is_moe else "mlp."
                    eff[pre + "gate_proj.weight"] = (Wdq[:I_] @ Q.T) / g_post
                    eff[pre + "up_proj.weight"] = (Wdq[I_:] @ Q.T) / g_post
                else:
                    pre = "mlp.shared_experts." if is_moe else "mlp."
                    eff[pre + "down_proj.weight"] = (Q @ Wdq @ H512) if is_moe else Wdq
                del Wdq
            torch.save(codes, out / "codes" / f"L{li}_b{b}.pt")
            torch.save({k: v.to(torch.bfloat16).cpu() for k, v in eff.items()}, out / "eff" / f"L{li}_b{b}.pt")
            del codes, eff
        report["layers"][str(li)] = lrep
        json.dump(report, open(out / "report.json", "w"), indent=1)
        log(f"layer {li}: " + " ".join(f"{k} {v:.2e}" for k, v in lrep.items()) + f" ({time.time() - t0:.0f}s)")
        mats = hs = Wq = att = mlp = sm = layer_ = None
        del layer
        gc.collect()
        torch.cuda.empty_cache()

    # ---- lm_head: Hessian of the final normed state at noisy-stream positions
    final = mm["a" if cfg.num_hidden_layers % 2 == 0 else "b"]
    gf = W.get("model.norm.weight").to(DEV).float()
    hs = Hess()
    eps = cfg.rms_norm_eps
    for r in rows:
        n0 = off2[r["row"]] + r["length"]
        x = torch.from_numpy(np.asarray(final[n0: n0 + r["length"]])).to(DEV).view(torch.bfloat16).float()
        n = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps)
        hs.add("lm", n @ Q, torch.full((x.shape[0],), evrow[r["row"]], device=DEV))
    sl = W.files[W.where["lm_head.weight"]].get_slice("lm_head.weight")
    V = sl.get_shape()[0]
    for b in a.bits:
        prep = None
        cs, err_n, err_d, effs = [], 0.0, 0.0, []
        He = hs.get("lm", "ev")
        for r0 in range(0, V, 16384):
            Wc = (sl[r0:r0 + 16384].to(DEV).float() * gf) @ Q
            Wdq, cz, prep = quantize(Wc, hs.get("lm", "tr"), b, a.group, prep)
            d = Wdq - Wc
            err_n += float(((d @ He) * d).sum()); err_d += float(((Wc @ He) * Wc).sum())
            cs.append(cz)
            effs.append(((Wdq @ Q.T) / gf).to(torch.bfloat16).cpu())
            del Wc, Wdq, d
        torch.save(tuple(torch.cat([c[i] for c in cs]) for i in range(3)), out / "codes" / f"lm_head_b{b}.pt")
        torch.save({"lm_head.weight": torch.cat(effs)}, out / "eff" / f"lm_head_b{b}.pt")
        report[f"lm_head_b{b}"] = err_n / err_d
        log(f"lm_head b{b}: rel {err_n / err_d:.2e}")
        del cs, effs
        gc.collect()
    json.dump(report, open(out / "report.json", "w"), indent=1)
    log("s05n done")


if __name__ == "__main__":
    main()
