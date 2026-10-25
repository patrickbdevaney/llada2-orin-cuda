#!/usr/bin/env python3
"""s07d: split the 8-bit lm_head into two 4-bit planes for screened readout (engine: lm_head screening).

8-bit code c = 16 h + l with value s (c - z). The engine streams only the HIGH plane each step,
    approx = s (16 h - zs),   zs = clip(round(z - 7.5), 0, 255)   (the low nibble replaced by its mean),
keeps every token's top candidates, and rescores those rows exactly from both planes. Both planes are ordinary
packlib "i8" 4-bit tiles (16 x 128, 1072 B): HIGH with (s, zs), LOW with (s, z) -- the low plane's group
parameters are the exact ones. Same bytes as the 8-bit head plus one 48 B parameter block per tile.

  lm_head_split.bin = [HIGH tiles of all strips][LOW tiles of all strips]   (strip-major, as every matrix)

  python s07d_lm_split.py [--dir ~/models/llada2mini-orin]
"""
import argparse
from pathlib import Path

import numpy as np

import packlib as PL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path.home() / "models/llada2mini-orin"))
    a = ap.parse_args()
    d = Path(a.dir)
    idx = [l.split() for l in open(d / "engine.idx") if l.startswith("T ") and "lm_head" in l]
    ent = next(x for x in idx if x[1] == "lm_head.bin" and x[2] == "lm_head")
    off, nbytes, R, K = int(ent[3]), int(ent[4]), int(ent[6]), int(ent[7])
    assert ent[5] == "i8tiles8", ent
    blob = np.memmap(d / "lm_head.bin", dtype=np.uint8, mode="r", offset=off, shape=(nbytes,))
    tb8, tb4 = PL.tile_bytes(8), PL.tile_bytes(4)
    T = K // PL.GROUP
    strip8 = T * tb8
    S = R // 16
    assert S * strip8 == nbytes
    out = d / "lm_head_split.bin"
    hi_bytes = S * T * tb4
    with open(out, "wb") as f:
        f.truncate(2 * hi_bytes)
    mm = np.memmap(out, dtype=np.uint8, mode="r+", shape=(2 * hi_bytes,))
    CH = 256                                                 # strips per chunk (4096 rows)
    for s0 in range(0, S, CH):
        s1 = min(S, s0 + CH)
        c, sc, z = PL.unpack_matrix_i8(np.asarray(blob[s0 * strip8: s1 * strip8]), (s1 - s0) * 16, K, 8)
        hi, lo = c >> 4, c & 15
        zs = np.clip(np.rint(z.astype(np.float32) - 7.5), 0, 255).astype(np.uint8)
        mm[s0 * T * tb4: s1 * T * tb4] = PL.pack_matrix_i8(hi, sc, zs, 4)
        mm[hi_bytes + s0 * T * tb4: hi_bytes + s1 * T * tb4] = PL.pack_matrix_i8(lo, sc, z, 4)
        if s0 == 0:                                          # self-check on the first chunk
            ch, _, _ = PL.unpack_matrix_i8(np.asarray(mm[: (s1 - s0) * T * tb4]), (s1 - s0) * 16, K, 4)
            cl, _, zl = PL.unpack_matrix_i8(np.asarray(mm[hi_bytes: hi_bytes + (s1 - s0) * T * tb4]), (s1 - s0) * 16, K, 4)
            assert ((ch.astype(np.int32) << 4) + cl == c).all() and (zl == z).all()
    mm.flush()
    print(f"{out}: {2 * hi_bytes} bytes ({R} x {K}, high plane {hi_bytes} + low plane {hi_bytes})")
    with open(d / "engine.idx", "a") as f:
        f.write(f"T lm_head_split.bin lm_head_hi 0 {hi_bytes} i8tiles4 {R} {K}\n")
        f.write(f"T lm_head_split.bin lm_head_lo {hi_bytes} {hi_bytes} i8tiles4 {R} {K}\n")
        f.write(f"F lm_head_split.bin {2 * hi_bytes}\n")


if __name__ == "__main__":
    main()
