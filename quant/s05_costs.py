#!/usr/bin/env python3
"""s05a: per-expert cost table for the joint prune / bit-width ILP (ProQuant-style, arXiv 2610.30779).

For every routed expert e of every MoE layer, on calibration tokens actually routed to it:
  cost[e, b, s]  = sum_t g_t^2 * || f_e(x_t) - fhat_e,b(x_t) ||^2      b in --bits (GPTQ on rotated weights)
  prune[e, s]    = sum_t g_t^2 * || f_e(x_t) ||^2                       (b = 0: the expert is removed)
split by stream s (0 clean / 1 noisy) and scaled from the evaluation sample to the expert's full routed count.
Both are energies of the same perturbation of the layer's MoE output, so they are directly comparable; the ILP
(s05_alloc.py) divides each layer's costs by that layer's residual-stream energy (also banked here) so that a
perturbation is judged relative to the signal it lands on.

Teacher path: the BF16 model, streamed one layer at a time (as s03) over a subset of TRAINING rows
(--row-stride; held-out rows stay untouched for s06). The MoE input of every token and its block-routed top-8
are written to NVMe per layer (WORK/s05/x.npy, ~3.4 GB at stride 3); experts are then processed in batches,
each with up to --cap tokens: 75% build the GPTQ Hessian, 25% measure the cost (no overfit flattery).

The quantizer is quantlib.gptq: asymmetric b-bit, group 128, clipped, true-sequential inside the expert
(the down-proj Hessian is built from the QUANTIZED gate/up output).
"""
from __future__ import annotations

import argparse
import gc
from concurrent.futures import ThreadPoolExecutor
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


def expert_weights(mlp, ids: list[int]):
    g = torch.stack([mlp.experts[e].gate_proj.weight for e in ids]).float()
    u = torch.stack([mlp.experts[e].up_proj.weight for e in ids]).float()
    d = torch.stack([mlp.experts[e].down_proj.weight for e in ids]).float()
    return g, u, d                      # [B, 512, 2048], [B, 512, 2048], [B, 2048, 512]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(WORK / "corpus"))
    ap.add_argument("--row-stride", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--cap", type=int, default=6144, help="max sampled routed tokens per expert")
    ap.add_argument("--batch-experts", type=int, default=12)
    ap.add_argument("--rotate", choices=["had", "none"], default="had")
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="only these MoE layers (ablations)")
    ap.add_argument("--tag", default="", help="output subdir suffix (ablations)")
    a = ap.parse_args()

    out = WORK / f"s05{a.tag}"
    (out / "cost").mkdir(parents=True, exist_ok=True)
    cdir = Path(a.corpus)
    tokens, elig = np.load(cdir / "tokens.npy"), np.load(cdir / "elig.npy")
    rows = [json.loads(l) for l in open(cdir / "rows.jsonl")]
    rows = [r for r in rows if not r["heldout"]][:: a.row_stride]
    off2, o = {}, 0
    for r in rows:
        off2[r["row"]] = o
        o += 2 * r["length"]
    T2, H = o, 2048
    log(f"{len(rows)} training rows, {T2:,} stream tokens (clean + noisy)")

    hsd = WORK / "hs5"
    hsd.mkdir(parents=True, exist_ok=True)
    mm = {k: np.lib.format.open_memmap(hsd / f"{k}.npy", mode="r+" if (hsd / f"{k}.npy").exists() else "w+",
                                        dtype=np.float16, shape=(T2, H)) for k in ("a", "b")}
    xmm = np.lib.format.open_memmap(out / "x.npy", mode="w+", dtype=np.float16, shape=(T2, H))
    imm = np.lib.format.open_memmap(out / "idx.npy", mode="w+", dtype=np.int16, shape=(T2, 8))
    wmm = np.lib.format.open_memmap(out / "w.npy", mode="w+", dtype=np.float16, shape=(T2, 8))
    stm = np.zeros(T2, np.int8)
    for r in rows:
        stm[off2[r["row"]] + r["length"]: off2[r["row"]] + 2 * r["length"]] = 1

    state_p = out / "state.json"
    state = json.load(open(state_p)) if state_p.exists() else {"next_layer": -1}
    cfg, Layer, Rotary, _ = load_classes()
    W = Weights()
    rot = Rotary(cfg).to(DEV)
    E = cfg.num_experts
    Q = QL.residual_rotation(H) if a.rotate == "had" else None
    json.dump({"rotate": a.rotate, "rotation_seed": 20261024, "group": a.group, "bits": a.bits,
               "basis": "gate/up: (W*gamma_post) @ Q ; down: Q.T @ W @ H512 ; x' = (x/gamma) @ Q ; z' = z @ H512"},
              open(out / "q_meta.json", "w"), indent=1)
    H512 = QL.hadamard(cfg.moe_intermediate_size) if a.rotate == "had" else None

    noisy = {r["row"]: noisy_ids(tokens[r["offset"]: r["offset"] + r["length"]],
                                 elig[r["offset"]: r["offset"] + r["length"]], 1_000_003 * r["row"] + 17)[0]
             for r in rows}                                   # same masks as s03 (same seeds)
    if state["next_layer"] < 0:
        emb = W.get("model.word_embeddings.weight").to(DEV, torch.bfloat16)
        for r in rows:
            ids = np.concatenate([tokens[r["offset"]: r["offset"] + r["length"]], noisy[r["row"]]])
            mm["a"][off2[r["row"]]: off2[r["row"]] + 2 * r["length"]] = \
                emb[torch.from_numpy(ids.astype(np.int64)).to(DEV)].half().cpu().numpy()
        del emb
        mm["a"].flush()
        state = {"next_layer": 0}
        json.dump(state, open(state_p, "w"))

    masks: dict[int, torch.Tensor] = {}
    for li in range(state["next_layer"], cfg.num_hidden_layers):
        t0 = time.time()
        src, dst = mm["a" if li % 2 == 0 else "b"], mm["b" if li % 2 == 0 else "a"]
        layer = build_layer(cfg, Layer, W, li)
        is_moe = hasattr(layer.mlp, "gate")
        cursor = [0]
        resid = np.zeros(2)

        def capture(h, topk_idx, w):
            n = h.shape[0]
            c = cursor[0]
            xmm[c:c + n] = h.half().cpu().numpy()
            imm[c:c + n] = topk_idx.to(torch.int16).cpu().numpy()
            wmm[c:c + n] = w.half().cpu().numpy()
            cursor[0] = c + n

        if is_moe:
            mlp = layer.mlp
            ctx = {"capture": capture}
            mlp.forward = lambda x, _m=mlp, _c=ctx: (moe_forward(_m, x, _c), None)
        order = []                                            # stream-token order as captured
        for L, rs in batches(rows, a.max_tokens):
            if L not in masks:                                # batches arrive sorted by length: keep one mask
                masks.clear()                                 # (a 2Lx2L bf16 mask is 512 MiB at L = 8192)
                masks[L] = stream_mask(L, DEV)
            x = torch.stack([torch.from_numpy(np.asarray(src[off2[r["row"]]: off2[r["row"]] + 2 * L]))
                             for r in rs]).to(DEV, torch.bfloat16)
            pos = torch.arange(L, device=DEV).repeat(2).unsqueeze(0).expand(len(rs), -1)
            with torch.no_grad():
                y = layer(x, attention_mask=masks[L][None, None].expand(len(rs), 1, -1, -1),
                          position_ids=pos, position_embeddings=rot(x, pos))[0]
            e2 = (y.float() ** 2).sum(-1).view(len(rs), 2, L).sum((0, 2)).double().cpu().numpy()
            resid += e2
            for j, r in enumerate(rs):
                dst[off2[r["row"]]: off2[r["row"]] + 2 * L] = y[j].half().cpu().numpy()
                order.append(np.arange(off2[r["row"]], off2[r["row"]] + 2 * L))
        dst.flush()
        for m_ in (src, dst):
            fd = os.open(m_.filename, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)

        if is_moe and (a.layers is None or li in a.layers):
            xmm.flush(); imm.flush(); wmm.flush()
            st = stm[np.concatenate(order)]                   # stream flag in capture order
            n_tok = cursor[0]
            idx = np.asarray(imm[:n_tok]).astype(np.int64)
            gw = np.asarray(wmm[:n_tok]).astype(np.float32)
            flat = idx.reshape(-1)
            srt = np.argsort(flat, kind="stable")
            bounds = np.searchsorted(flat[srt], np.arange(E + 1))
            gamma = layer.post_attention_layernorm.weight.float()
            nb = len(a.bits)
            cost = np.zeros((E, nb, 2))
            prune = np.zeros((E, 2))
            n_routed = np.zeros((E, 2))
            n_sample = np.zeros(E, np.int64)
            method = np.zeros(E, np.int8)                     # 1 gptq, 0 rtn (too few tokens)
            rng = np.random.default_rng(li)
            # Code bank: every expert at every bit-width, in the rotated basis (s06 assembles any allocation from it,
            # s07 packs the chosen codes for the CUDA engine). ~2.4 GB per layer on NVMe, uint8 codes unpacked.
            qdir = out / "q" / f"L{li}"
            qdir.mkdir(parents=True, exist_ok=True)
            I = cfg.moe_intermediate_size
            shapes = {"gu": ((E, 2 * I, H), np.uint8), "gu_s": ((E, 2 * I, H // a.group), np.float16),
                      "gu_z": ((E, 2 * I, H // a.group), np.uint8), "d": ((E, H, I), np.uint8),
                      "d_s": ((E, H, I // a.group), np.float16), "d_z": ((E, H, I // a.group), np.uint8)}
            qbank = {b: {k: np.lib.format.open_memmap(qdir / f"b{b}_{k}.npy", mode="w+", dtype=dt, shape=sh)
                         for k, (sh, dt) in shapes.items()} for b in a.bits}
            t_e = time.time()
            def load_batch(ids):
                """Sample each expert's routed tokens and read their MoE inputs with ONE sequential scan of x.npy
                (random 4 KB row reads from a 3 GB file the page cache cannot hold ran at ~3k faults/s, ~9 min
                per layer; a chunked scan streams at NVMe speed)."""
                picks = []
                for e in ids:
                    slots = srt[bounds[e]: bounds[e + 1]]
                    tok = slots // 8
                    for s_ in (0, 1):
                        n_routed[e, s_] = int((st[tok] == s_).sum())
                    if len(slots) > a.cap:
                        slots = np.sort(rng.choice(slots, a.cap, replace=False))
                    n_sample[e] = len(slots)
                    picks.append(slots)
                need = np.unique(np.concatenate([p_ // 8 for p_ in picks])) if picks else np.zeros(0, np.int64)
                rows_ = np.empty((len(need), H), np.float16)
                CH = 1 << 16
                for c0 in range(0, n_tok, CH):
                    lo_, hi_ = np.searchsorted(need, [c0, min(n_tok, c0 + CH)])
                    if hi_ > lo_:
                        rows_[lo_:hi_] = np.asarray(xmm[c0:min(n_tok, c0 + CH)])[need[lo_:hi_] - c0]
                samp = []
                for e, slots in zip(ids, picks):
                    tok = slots // 8
                    xs = torch.from_numpy(rows_[np.searchsorted(need, tok)]).to(DEV).float()
                    if Q is not None:
                        xs = (xs / gamma) @ Q
                    xs = xs.half()                            # rotated x is O(1): fp16 halves the batch footprint
                    g_t = torch.from_numpy(gw.reshape(-1)[slots]).to(DEV)
                    perm = rng.permutation(len(slots))
                    n_tr = max(1, int(0.75 * len(slots))) if len(slots) >= 64 else len(slots)
                    tr, ev = perm[:n_tr], (perm[n_tr:] if len(slots) >= 64 else perm)
                    samp.append((xs, g_t, torch.from_numpy(st[tok]).to(DEV), tr, ev))
                del rows_
                Hgu = torch.stack([s_[0][s_[3]].float().T @ s_[0][s_[3]].float() for s_ in samp])
                return samp, Hgu

            # pipeline: the next batch's CPU Cholesky runs on a worker thread while the GPU quantizes this one
            starts = list(range(0, E, a.batch_experts))
            pool = ThreadPoolExecutor(1)
            nxt_ids = list(range(starts[0], min(E, starts[0] + a.batch_experts)))
            nxt_samp, nxt_H = load_batch(nxt_ids)
            nxt_fut = pool.submit(QL.prepare, nxt_H)
            for bi_, e0 in enumerate(starts):
                ids, samp, Hgu, fut = nxt_ids, nxt_samp, nxt_H, nxt_fut
                if bi_ + 1 < len(starts):
                    e1 = starts[bi_ + 1]
                    nxt_ids = list(range(e1, min(E, e1 + a.batch_experts)))
                    nxt_samp, nxt_H = load_batch(nxt_ids)
                    nxt_fut = pool.submit(QL.prepare, nxt_H)
                if e0 and (e0 // a.batch_experts) % max(1, 64 // a.batch_experts) == 0:
                    log(f"layer {li}: {e0}/{E} experts, {time.time() - t_e:.0f}s")
                g_w, u_w, d_w = expert_weights(mlp, ids)
                if Q is not None:
                    g_w = (g_w * gamma) @ Q
                    u_w = (u_w * gamma) @ Q
                    d_w = Q.T @ d_w @ H512
                prep_gu = fut.result()                        # one factorisation, reused for every bit-width
                for bi, b in enumerate(a.bits):
                    use_gptq = [len(s[3]) >= 256 for s in samp]
                    Wgu = torch.cat([g_w, u_w], 1)
                    Wgu_q, cgu, sgu, zgu = QL.gptq(Wgu, None, b, a.group, prep=prep_gu, codes=True)
                    for k, ok in enumerate(use_gptq):
                        if not ok:
                            r_ = QL.rtn(Wgu[k:k + 1], b, a.group, codes=True)
                            Wgu_q[k], cgu[k], sgu[k], zgu[k] = (t[0] for t in r_)
                    gq, uq = Wgu_q[:, :512], Wgu_q[:, 512:]
                    zs = []
                    for k, s in enumerate(samp):
                        xk = s[0].float()
                        z = F.silu(xk @ gq[k].T) * (xk @ uq[k].T)
                        zs.append(z @ H512 if H512 is not None else z)
                    Hd = torch.stack([z[s[3]].T @ z[s[3]] for z, s in zip(zs, samp)])
                    Wd_q, cd, sd, zd = QL.gptq(d_w, Hd, b, a.group, codes=True)
                    for k, ok in enumerate(use_gptq):
                        if not ok:
                            r_ = QL.rtn(d_w[k:k + 1], b, a.group, codes=True)
                            Wd_q[k], cd[k], sd[k], zd[k] = (t[0] for t in r_)
                    qb = qbank[b]
                    for name, t in (("gu", cgu), ("gu_s", sgu), ("gu_z", zgu), ("d", cd), ("d_s", sd), ("d_z", zd)):
                        qb[name][ids[0]: ids[-1] + 1] = t.cpu().numpy()
                    for k, (e, s) in enumerate(zip(ids, samp)):
                        xs, g_t, st_t, tr, ev = s
                        xe = xs[ev].float()
                        f = (F.silu(xe @ g_w[k].T) * (xe @ u_w[k].T)) @ (H512 if H512 is not None else
                                                                          torch.eye(512, device=DEV))
                        f = f @ d_w[k].T
                        fq = zs[k][ev] @ Wd_q[k].T
                        g2 = g_t[ev] ** 2
                        for sidx in (0, 1):
                            m = st_t[ev] == sidx
                            n_ev = int(m.sum())
                            if n_ev == 0:
                                continue
                            scale = n_routed[e, sidx] / n_ev
                            cost[e, bi, sidx] = float((g2[m] * ((f[m] - fq[m]) ** 2).sum(-1)).sum()) * scale
                            if bi == 0:
                                prune[e, sidx] = float((g2[m] * (f[m] ** 2).sum(-1)).sum()) * scale
                        method[e] = int(use_gptq[k])
                del Hgu, Hd, samp, zs, prep_gu
                torch.cuda.empty_cache()
            pool.shutdown()
            for qb in qbank.values():
                for m_ in qb.values():
                    m_.flush()
            del qbank
            if not (np.isfinite(cost).all() and np.isfinite(prune).all()):
                raise FloatingPointError(f"non-finite costs at layer {li}")
            np.savez(out / "cost" / f"L{li}.npz", cost=cost, prune=prune, n_routed=n_routed, n_sample=n_sample,
                     method=method, bits=np.array(a.bits), resid=resid, group=a.group, rotate=a.rotate)
            rel = cost.sum((0, 2)) / max(prune.sum(), 1e-30)
            log(f"layer {li}: costs done; layer distortion vs prune-all: "
                + " ".join(f"{b}b {r:.4f}" for b, r in zip(a.bits, rel))
                + f"; rtn experts {int((method == 0).sum())}")
        # every reference to the layer must go before the next one loads (mlp, its forward closure, the pipeline's
        # look-ahead batch): `del layer` alone left layer 1's 1.6 GB alive under layer 2 and the OOM killer fired
        mlp = ctx = samp = Hgu = nxt_samp = nxt_H = nxt_fut = fut = prep_gu = g_w = u_w = d_w = None
        del layer
        gc.collect()
        torch.cuda.empty_cache()
        state = {"next_layer": li + 1}
        json.dump(state, open(state_p, "w"))
        log(f"layer {li} done in {time.time() - t0:.0f}s (gpu peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB)")
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
