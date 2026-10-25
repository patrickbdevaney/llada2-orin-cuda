#!/usr/bin/env python3
"""s04: protection lists and pure-pruning candidate keep-sets from the s03 banks (numpy, CPU only).

Two products:
  1. protect.npz: per MoE layer, experts that must never be pruned (and are therefore never offered the
     b = 0 choice in the s05 bit-allocation ILP):
       * Super-Experts: fmax above the model-wide P99.5 and >= max/10 (DeepSeek/GLM guard; LLaDA's late layers
         reach |f| in the 1000s, the massive-outlier sites the dLLM PTQ study flagged);
       * the top `pf` fraction of every bucket pool by REAP-sum, pf read off the HOPE knee (sweep_pf).
  2. keepsets/<arm>_k<n_keep>.npy [n_moe_layers, E] bool: pure-pruning arms at 10/15/20/25% for s06
     arbitration against the teacher. They are the "prune-only" baselines the joint ILP has to beat.

Streams. The bank keeps clean (prefill/context) and noisy (decode block) statistics separately. Saliency is
w * noisy + (1 - w) * clean, w = --w-noisy (default 0.5): decode routing decides speed and the tokens the
executor emits; clean routing decides how well the prompt (tool schemas, files) is read.

Arms: hope_sum (pf knee), reap_sum, mass, freq, random0..2 (anchors: a criterion has to beat them).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import select_lib as S

WORK = Path(os.environ.get("LLADA_WORK", Path.home() / "models/llada-work"))
RATIOS = (0.10, 0.15, 0.20, 0.25)
POOL_CAPACITY = 48          # block-routing pool size: every layer must keep at least this many experts


def combined(bank: dict, w: float) -> dict:
    """Collapse the stream axis: [2, n_buckets, E] -> [n_buckets, E], [2, E, E] -> [E, E]."""
    sw = np.array([1.0 - w, w])
    out = {k: np.tensordot(sw, bank[k], axes=1) for k in ("cnt", "gat", "gsq", "nrm", "nsq", "sum", "sq")}
    out["F_sum"] = np.tensordot(sw, bank["F_sum"], axes=1)
    out["F_cnt"] = np.tensordot(sw, bank["F_cnt"], axes=1)
    out["fmax"] = bank["fmax"]
    out["buckets"] = bank["buckets"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=str(WORK / "s03" / "bank"))
    ap.add_argument("--out", default=str(WORK / "s04"))
    ap.add_argument("--w-noisy", type=float, default=0.5)
    ap.add_argument("--pf", type=float, default=None, help="fix protect fraction (default: HOPE knee at 20%)")
    a = ap.parse_args()

    bank_dir, out = Path(a.bank), Path(a.out)
    (out / "keepsets").mkdir(parents=True, exist_ok=True)
    files = sorted(bank_dir.glob("L*.npz"), key=lambda p: int(p.stem[1:]))
    layers = [int(p.stem[1:]) for p in files]
    banks = [combined({k: v for k, v in np.load(p).items()}, a.w_noisy) for p in files]
    E = banks[0]["sum"].shape[1]
    buckets = [str(b) for b in banks[0]["buckets"]]
    print(f"{len(layers)} MoE layers {layers[0]}..{layers[-1]}, E={E}, buckets={buckets}")

    # Super-Experts, model-wide threshold
    fm = np.concatenate([b["fmax"] for b in banks])
    thr = max(np.percentile(fm, 99.5), fm.max() / 10)
    guard = [np.flatnonzero(b["fmax"] >= thr) for b in banks]
    print(f"super-expert threshold {thr:.1f}: {sum(len(g) for g in guard)} experts "
          f"in layers {[layers[i] for i, g in enumerate(guard) if len(g)]}")

    per_layer = []
    for b, g in zip(banks, guard):
        P = b["sum"]                                           # [n_buckets, E] REAP-sum per pool
        per_layer.append({"F": b["F_sum"], "P": P, "live": P.sum(1) > 0, "guard": g,
                          "crit": S.criteria(b)})

    # pf knee at the 20% ratio (the expected operating region), then reused for every ratio
    n_keep20 = E - round(E * 0.20)
    sweep = S.sweep_pf(per_layer, n_keep20, "sum")
    pf = a.pf if a.pf is not None else sweep["knee"]
    print(f"HOPE pf knee {sweep['knee']} (using {pf})")

    protect = {}
    for li, d in zip(layers, per_layer):
        protect[f"L{li}"] = np.union1d(d["guard"], S.pool_protect(d["P"], pf, d["live"])).astype(np.int32)
    np.savez(out / "protect.npz", **protect)

    summary = {"layers": layers, "E": E, "buckets": buckets, "w_noisy": a.w_noisy, "pf": pf,
               "super_threshold": float(thr), "pf_sweep": sweep, "arms": {}}
    for r in RATIOS:
        n_keep = max(POOL_CAPACITY, E - round(E * r))
        for arm in ("hope_sum", "reap_sum", "mass", "freq", "random0", "random1", "random2"):
            keep = np.zeros((len(layers), E), bool)
            ret, ic = [], []
            for i, (li, d) in enumerate(zip(layers, per_layer)):
                prot = protect[f"L{li}"] if arm.startswith("hope") else d["guard"]
                if arm == "hope_sum":
                    pr = S.prune_set("hope", n_keep=n_keep, protect=prot, F=d["F"])
                elif arm.startswith("random"):
                    pr = S.prune_set("random", n_keep=n_keep, protect=prot, E=E, seed=int(arm[-1]) * 1000 + li)
                else:
                    pr = S.prune_set(arm, n_keep=n_keep, protect=prot, crit=d["crit"])
                keep[i] = True
                keep[i, pr] = False
                ret.append(S.retention(d["P"], pr))
                ic.append(S.icost(d["F"], pr))
            R = np.stack(ret).mean(0)
            name = f"{arm}_k{n_keep}"
            np.save(out / "keepsets" / f"{name}.npy", keep)
            summary["arms"][name] = {"ratio": r, "n_keep": n_keep, "icost": float(np.mean(ic)),
                                     "retention": dict(zip(buckets, map(float, R))),
                                     "worst_bucket": buckets[int(np.argmin(R))], "worst": float(R.min())}
            print(f"  {name:18s} icost {np.mean(ic):.4f}  worst {buckets[int(np.argmin(R))]} {R.min():.3f}  "
                  + " ".join(f"{b[:5]} {v:.3f}" for b, v in zip(buckets, R)))
    json.dump(summary, open(out / "summary.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
