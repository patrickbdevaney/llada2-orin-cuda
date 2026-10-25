#!/usr/bin/env python3
"""s05b: joint expert pruning + bit-width allocation under the Orin memory budget, swept over context length.

For each (context, KV format) the bytes left for routed experts are
    budget = usable - fixed - context * kv_bytes_per_token
and an exact MILP (scipy/HiGHS) picks b(e) in --choices for every routed expert:
    minimise   sum_e  D[e, b(e)]
    s.t.       sum_e  bytes(b(e)) <= budget
               per layer: #pruned <= rho * E     (kept >= block-routing pool capacity 48)
               protected experts (s04 protect.npz: Super-Experts, top-pf per bucket) are never pruned
D is the s05a energy of the MoE-output perturbation divided by that layer's residual-stream energy, so it reads
as a relative perturbation of the signal it lands on, and clean (prefill/context) and noisy (decode) streams
are mixed by --w-noisy.

Output: a table (one row per context x KV format: objective, average expert bpw, pruned fraction, bit
histogram) and alloc/<ctx>_<kv>.npy [n_moe_layers, E] int8 bit-widths (0 = pruned) for s06 arbitration.
The objective is a proxy; s06 replays the chosen points against the teacher's held-out top-32 to decide.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

import quantlib as QL

WORK = Path(os.environ.get("LLADA_WORK", Path.home() / "models/llada-work"))
P_EXPERT = 3 * 2048 * 512
KV_BYTES = {"bf16": 40 * 1024, "int8": 21094, "int4": 10854}     # per token, scales included
FIXED = {"cuda_ctx": 0.35e9, "nonexpert_w4": 0.311e9 * QL.bytes_per_param(4), "lm_head_int8": 0.322e9 * 1.031,
         "workspace": 0.15e9, "server": 0.2e9}


def load(costdir: Path, w: float):
    files = sorted(costdir.glob("L*.npz"), key=lambda p: int(p.stem[1:]))
    layers, D, bits_list = [], [], None
    for p in files:
        z = np.load(p)
        bits_list = [int(b) for b in z["bits"]]
        resid = np.maximum(z["resid"], 1e-30)
        sw = np.array([1 - w, w]) / resid                         # per-stream normalisation
        q = (z["cost"] * sw).sum(-1)                              # [E, nb]
        pr = (z["prune"] * sw).sum(-1)                            # [E]
        D.append(np.concatenate([pr[:, None], q], 1))             # columns: 0, bits...
        layers.append(int(p.stem[1:]))
    return layers, np.stack(D), [0] + bits_list, int(z["group"])


def solve(D, choices, all_bits, budget, rho, protect, group):
    Lr, E, _ = D.shape
    cols = [all_bits.index(b) for b in choices]
    nb = len(cols)
    c = D[:, :, cols].reshape(-1)
    n = c.size
    byt = np.array([P_EXPERT * QL.bytes_per_param(b, group) for b in choices])
    rows, cidx, vals, lo, hi = [], [], [], [], []
    r = 0
    for v in range(Lr * E):                                      # one choice per expert
        rows += [r] * nb; cidx += list(range(v * nb, v * nb + nb)); vals += [1.0] * nb
        lo.append(1); hi.append(1); r += 1
    rows += [r] * n; cidx += list(range(n)); vals += list(np.tile(byt, Lr * E))
    lo.append(0); hi.append(budget); r += 1
    ub = np.ones(n)
    if 0 in choices:
        z0 = choices.index(0)
        for li in range(Lr):                                     # per-layer prune cap
            rows += [r] * E; cidx += [(li * E + e) * nb + z0 for e in range(E)]; vals += [1.0] * E
            lo.append(0); hi.append(int(rho * E)); r += 1
            for e in protect.get(li, []):
                ub[(li * E + e) * nb + z0] = 0
    A = coo_matrix((vals, (rows, cidx)), shape=(r, n)).tocsr()
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(n), bounds=Bounds(0, ub),
               options={"time_limit": 120, "mip_rel_gap": 1e-4})
    if res.x is None:
        return None
    x = res.x.reshape(Lr, E, nb).argmax(-1)
    return np.array(choices)[x].astype(np.int8), float(res.fun)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", default=str(WORK / "s05" / "cost"))
    ap.add_argument("--protect", default=str(WORK / "s04" / "protect.npz"))
    ap.add_argument("--usable", type=float, default=6.6e9)
    ap.add_argument("--w-noisy", type=float, default=0.5)
    ap.add_argument("--rho", type=float, default=0.25)
    ap.add_argument("--choices", type=int, nargs="+", default=[0, 2, 3, 4])
    ap.add_argument("--contexts", type=int, nargs="+", default=[4096, 8192, 16384, 32768, 65536])
    ap.add_argument("--kv", nargs="+", default=["int8", "int4"])
    ap.add_argument("--prune-weight", type=float, default=1.0,
                    help="multiply the b=0 (prune) cost. s06 measured pruning at ~3.5x the proxy-to-KL ratio of 4-bit")
    ap.add_argument("--bit-weight", type=str, nargs="*", default=[],
                    help="per-bit cost multipliers as b=w pairs, e.g. 2=6 3=1.5 (low bits compound super-linearly)")
    ap.add_argument("--tag", default="", help="suffix for alloc file names")
    a = ap.parse_args()

    layers, D, all_bits, group = load(Path(a.cost), a.w_noisy)
    D[:, :, 0] *= a.prune_weight
    for bw in a.bit_weight:
        b_, w_ = str(bw).split("=") if "=" in str(bw) else (None, None)
        D[:, :, all_bits.index(int(b_))] *= float(w_)
    Lr, E, _ = D.shape
    protect = {}
    if Path(a.protect).exists():
        pz = np.load(a.protect)
        protect = {i: pz[f"L{li}"].tolist() for i, li in enumerate(layers) if f"L{li}" in pz}
    fixed = sum(FIXED.values())
    out = Path(a.cost).parent / "alloc"
    out.mkdir(exist_ok=True)
    print(f"{Lr} MoE layers, E={E}, fixed {fixed / 1e9:.2f} GB, usable {a.usable / 1e9:.2f} GB, "
          f"choices {a.choices}, rho {a.rho}, protected {sum(len(v) for v in protect.values())}")

    ref = {}
    for b in [x for x in all_bits if x > 0]:
        ref[b] = float(D[:, :, all_bits.index(b)].sum())
    print("uniform references (objective; bytes): " + "; ".join(
        f"{b}b {v:.4f} {Lr * E * P_EXPERT * QL.bytes_per_param(b, group) / 1e9:.2f} GB" for b, v in ref.items()))

    table = []
    print(f"{'ctx':>6} {'kv':>5} {'budget':>7} {'obj':>8} {'bpw':>5} {'pruned':>7}  hist(0/2/3/4)")
    for kv in a.kv:
        for ctx in a.contexts:
            budget = a.usable - fixed - ctx * KV_BYTES[kv]
            if budget <= 0:
                continue
            r = solve(D, a.choices, all_bits, budget, a.rho, protect, group)
            if r is None:
                print(f"{ctx:>6} {kv:>5} {budget / 1e9:6.2f}G infeasible")
                continue
            bits, obj = r
            used = sum(P_EXPERT * QL.bytes_per_param(int(b), group) for b in bits.reshape(-1))
            hist = {b: int((bits == b).sum()) for b in (0, 2, 3, 4)}
            bpw = used * 8 / (Lr * E * P_EXPERT)
            row = {"ctx": ctx, "kv": kv, "budget_gb": budget / 1e9, "objective": obj, "expert_bpw": bpw,
                   "pruned_frac": hist[0] / (Lr * E), "hist": hist}
            table.append(row)
            np.save(out / f"{ctx}_{kv}{a.tag}.npy", bits)
            print(f"{ctx:>6} {kv:>5} {budget / 1e9:6.2f}G {obj:8.4f} {bpw:5.2f} {hist[0] / (Lr * E):7.3f}  "
                  f"{hist[0]}/{hist[2]}/{hist[3]}/{hist[4]}")
    json.dump({"prune_weight": a.prune_weight, "bit_weight": a.bit_weight, "table": table, "uniform": ref, "fixed": FIXED, "usable": a.usable, "w_noisy": a.w_noisy,
               "rho": a.rho, "choices": a.choices}, open(out / f"table{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
