#!/usr/bin/env python3
"""Synthetic dense W·A8 case: random b-bit codes for an [R, K] matrix (format i8), T tokens, fp32 reference
Y = A8(x) @ dequant(W)ᵀ. Exercises the 8-bit path and NT = 4 (32-token) items that dense projections use."""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "quant"))
import packlib as PL  # noqa: E402
from gen_moe_case import a8  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=3072); ap.add_argument("--k", type=int, default=2048)
ap.add_argument("--bits", type=int, default=8); ap.add_argument("--tokens", type=int, default=32)
ap.add_argument("--out", default=str(Path(__file__).parent / "dense"))
a = ap.parse_args()
rng = np.random.default_rng(a.bits)
R, K, b = a.rows, a.k, a.bits
c = rng.integers(0, 2 ** b, (R, K), dtype=np.uint8)
s = (rng.uniform(0.5, 1.5, (R, K // 128)) * 2.0 / 2 ** b * 0.05).astype(np.float16)
z = rng.integers(2 ** b // 2 - 2, 2 ** b // 2 + 2, (R, K // 128)).astype(np.uint8)
Wd = (c.astype(np.float32) - np.repeat(z, 128, -1)) * np.repeat(s.astype(np.float32), 128, -1)
x = rng.standard_normal((a.tokens, K)).astype(np.float16)
ref = a8(x.astype(np.float32)) @ Wd.T
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
PL.pack_matrix_i8(c, s, z, b).tofile(out / "blob.bin")
x.tofile(out / "x.bin"); ref.astype(np.float32).tofile(out / "ref.bin")
np.array([R, K, b, a.tokens], np.int32).tofile(out / "dims.bin")
print(f"dense case {R}x{K} {b}-bit, T={a.tokens}")
