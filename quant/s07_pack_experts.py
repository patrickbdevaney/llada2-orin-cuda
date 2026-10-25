#!/usr/bin/env python3
"""s07a: pack the chosen allocation's routed experts from the s05 code bank into the engine tile format.

Out: <out>/experts_L{li}.bin per MoE layer -- for each expert in id order that is not pruned, its gate_up matrix
[2I, H] (rows permuted by packlib.gu_row_perm) followed by its down matrix [H, I], each 4 KiB aligned -- and
<out>/experts.json with per layer, per expert {bits, gu_off, d_off} (byte offsets in that layer's file; bits 0 =
pruned, offsets -1), plus the basis/rotation metadata the engine must apply (q_meta.json).
Each matrix is unpacked again and compared with the bank (codes, scales, zeros) before it is written.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import packlib as PL

WORK = Path(os.environ.get("LLADA_WORK", Path.home() / "models/llada-work"))
ALIGN = 4096


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc", default=str(WORK / "s05/alloc/32768_int4_noprune.npy"))
    ap.add_argument("--qbank", default=str(WORK / "s05"))
    ap.add_argument("--out", default=str(Path.home() / "models/llada2mini-orin"))
    ap.add_argument("--first-moe-layer", type=int, default=1)
    ap.add_argument("--only", type=int, nargs="*", default=None, help="pack only these layers (tests)")
    ap.add_argument("--fmt", choices=["f16", "i8"], default="i8",
                    help="tile format: i8 = engine W·A8 path (moe_i8.cu); f16 = moe_gemm.cu (reference path)")
    a = ap.parse_args()
    bits = np.load(a.alloc).astype(np.int8)
    meta = json.load(open(Path(a.qbank) / "q_meta.json"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    Lr, E = bits.shape
    I, H = 512, 2048
    perm = PL.gu_row_perm(I)
    pack, unpack = (PL.pack_matrix_i8, PL.unpack_matrix_i8) if a.fmt == "i8" else (PL.pack_matrix, PL.unpack_matrix)
    table = {"alloc": str(a.alloc), "group": PL.GROUP, "format": f"packlib tiles {a.fmt}", "I": I, "H": H, "E": E,
             "q_meta": meta, "layers": {}}
    total = 0
    for mi in range(Lr):
        li = mi + a.first_moe_layer
        if a.only is not None and li not in a.only:
            continue
        t0 = time.time()
        q = Path(a.qbank) / "q" / f"L{li}"
        bank = {b: {k: np.load(q / f"b{b}_{k}.npy", mmap_mode="r") for k in ("gu", "gu_s", "gu_z", "d", "d_s", "d_z")}
                for b in sorted(set(int(x) for x in bits[mi]) - {0})}
        ents, off = [], 0
        tmp = out / f"experts_L{li}.bin.part"
        with open(tmp, "wb") as f:
            for e in range(E):
                b = int(bits[mi, e])
                if b == 0:
                    ents.append({"bits": 0, "gu_off": -1, "d_off": -1})
                    continue
                bk = bank[b]
                rec = {"bits": b}
                for name, (c, s, z, rows) in {
                        "gu": (bk["gu"][e], bk["gu_s"][e], bk["gu_z"][e], perm),
                        "d": (bk["d"][e], bk["d_s"][e], bk["d_z"][e], None)}.items():
                    c, s, z = np.asarray(c), np.asarray(s), np.asarray(z)
                    if rows is not None:
                        c, s, z = c[rows], s[rows], z[rows]
                    blob = pack(c, s, z, b)
                    c2, s2, z2 = unpack(blob, *c.shape, b)
                    if not ((c2 == c).all() and (s2.view(np.uint16) == s.view(np.uint16)).all() and (z2 == z).all()):
                        raise RuntimeError(f"pack roundtrip mismatch L{li} e{e} {name}")
                    pad = (-off) % ALIGN
                    f.write(b"\0" * pad)
                    off += pad
                    rec[f"{name}_off"] = off
                    f.write(blob.tobytes())
                    off += blob.size
                ents.append(rec)
        tmp.rename(out / f"experts_L{li}.bin")
        total += off
        hist = {int(b): int((bits[mi] == b).sum()) for b in (0, 2, 3, 4)}
        table["layers"][str(li)] = {"bytes": off, "hist": hist, "experts": ents}
        print(f"L{li}: {off / 1e6:.1f} MB {hist} in {time.time() - t0:.0f}s", flush=True)
    table["total_bytes"] = total
    json.dump(table, open(out / "experts.json", "w"))
    print(f"total {total / 1e9:.3f} GB")


if __name__ == "__main__":
    main()
