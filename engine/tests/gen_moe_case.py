#!/usr/bin/env python3
"""Build a routed-expert test case from the real s05 code bank: one decode block (T tokens, top-8 inside a pool of
48 experts of one layer at their allocated bits), packed in the engine tile format, plus an fp32 numpy reference
of acc = sum_k w_k * (SiLU(x' g'ᵀ) * (x' u'ᵀ)) H512 d'ᵀ in the rotated basis. numpy only: the GPU stays free."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "quant"))
import packlib as PL  # noqa: E402

WORK = Path.home() / "models/llada-work"


def dequant(c, s, z):
    return (c.astype(np.float32) - np.repeat(z, 128, -1).astype(np.float32)) * np.repeat(s.astype(np.float32), 128, -1)


def a8(v):
    """Symmetric per-128-group int8 as the engine kernels quantize: s = max|v| / 127 (fp32), round half to even."""
    g = v.reshape(*v.shape[:-1], -1, 128).astype(np.float32)
    s = np.maximum(np.abs(g).max(-1, keepdims=True) / np.float32(127), np.float32(1e-30)).astype(np.float32)
    q = np.clip(np.rint(g / s), -127, 127)
    return (q * s).reshape(v.shape).astype(np.float32)


def hadamard(n):
    H = np.ones((1, 1), np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--alloc", default=str(WORK / "s05/alloc/32768_int4_noprune.npy"))
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--pool", type=int, default=48)
    ap.add_argument("--out", default=str(Path(__file__).parent / "case"))
    ap.add_argument("--fmt", choices=["f16", "i8"], default="f16", help="tile format; i8 also writes the A8 reference")
    a = ap.parse_args()
    rng = np.random.default_rng(a.layer)
    bits = np.load(a.alloc)[a.layer - 1]
    q = WORK / "s05/q" / f"L{a.layer}"
    banks = {b: {k: np.load(q / f"b{b}_{k}.npy", mmap_mode="r") for k in ("gu", "gu_s", "gu_z", "d", "d_s", "d_z")}
             for b in (2, 3, 4)}
    alive = np.flatnonzero(bits > 0)
    # a pool that contains every bit-width present in the layer
    pool = []
    for b in (2, 3, 4):
        cand = alive[bits[alive] == b]
        pool += list(rng.choice(cand, min(len(cand), 4), replace=False))
    rest = np.setdiff1d(alive, pool)
    pool = np.array(pool + list(rng.choice(rest, a.pool - len(pool), replace=False)))
    E, I, H = len(bits), 512, 2048
    perm = PL.gu_row_perm(I)
    blobs, gu_off, d_off, off = [], np.zeros(E, np.uint64), np.zeros(E, np.uint64), 0
    W = {}
    for e in pool:
        b = int(bits[e])
        bk = banks[b]
        cgu, sgu, zgu = np.asarray(bk["gu"][e]), np.asarray(bk["gu_s"][e]), np.asarray(bk["gu_z"][e])
        cd, sd, zd = np.asarray(bk["d"][e]), np.asarray(bk["d_s"][e]), np.asarray(bk["d_z"][e])
        W[e] = (dequant(cgu, sgu, zgu), dequant(cd, sd, zd))
        pk = PL.pack_matrix_i8 if a.fmt == "i8" else PL.pack_matrix
        g = pk(cgu[perm], sgu[perm], zgu[perm], b)
        d = pk(cd, sd, zd, b)
        gu_off[e] = off; blobs.append(g); off += g.size
        d_off[e] = off; blobs.append(d); off += d.size
    blob = np.concatenate(blobs)
    T, K = a.tokens, 8
    x = (rng.standard_normal((T, H)) * 1.0).astype(np.float16)
    idx = np.stack([rng.choice(pool, K, replace=False) for _ in range(T)]).astype(np.int32)
    w = rng.uniform(0.05, 0.6, (T, K)).astype(np.float32)
    H512 = hadamard(I)
    acc = np.zeros((T, H), np.float32)
    xf = x.astype(np.float32)
    if a.fmt == "i8":
        xf = a8(xf)
    for t in range(T):
        for k in range(K):
            gu, d = W[int(idx[t, k])]
            h = gu @ xf[t]
            z = (h[:I] / (1 + np.exp(-h[:I]))) * h[I:]
            z = z.astype(np.float16).astype(np.float32) @ H512
            z = a8(z) if a.fmt == "i8" else z.astype(np.float16).astype(np.float32)
            acc[t] += w[t, k] * (d @ z)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    blob.tofile(out / "blob.bin"); gu_off.tofile(out / "gu_off.bin"); d_off.tofile(out / "d_off.bin")
    bits.astype(np.int8).tofile(out / "bits.bin"); x.tofile(out / "x.bin"); idx.tofile(out / "idx.bin")
    w.tofile(out / "w.bin"); acc.tofile(out / "ref.bin")
    hist = {b: int((bits[pool] == b).sum()) for b in (2, 3, 4)}
    json.dump({"E": E, "T": T, "K": K, "H": H, "I": I, "blob_bytes": int(blob.size), "pool_bits": hist},
              open(out / "case.json", "w"))
    print(f"case: layer {a.layer}, pool {len(pool)} {hist}, {blob.size / 1e6:.1f} MB packed, T={T}")


if __name__ == "__main__":
    main()
