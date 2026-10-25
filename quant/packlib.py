"""Engine tile format for the mixed 2/3/4-bit routed experts (what engine/moe_gemm.cu reads).

A matrix [R, K] (R % 16 == 0, K % 128 == 0, one quant group = 128 columns) is cut into strips of 16 rows; a strip
is a sequence of K/128 tiles, each 16 rows x 128 columns = one group per row. A tile is laid out so that a warp
computing mma.m16n8k16 with the WEIGHTS as the A operand (swap-AB: weight rows are M, tokens are N) loads its
whole A fragment for 8 k16-steps with coalesced 16-byte loads and dequantizes it with masks, no shuffles:

  lane l holds rows r0 = l/4 and r0+8, columns c0 = 2(l%4) + {0,1} and c0 + 8 + {0,1} of each k16 step, i.e.
  for step j the 8 values  (r0,c0) (r0,c0+1) | (r0+8,c0) (r0+8,c0+1) | (r0,c0+8) (r0,c0+9) | (r0+8,c0+8) (r0+8,c0+9)
  = 4 pairs, pair p = 4j + i (i = 0..3), first/second element = low/high half of the half2 register.

  4-bit: 8 words per lane; word j = step j, pair i: first at nibble i, second at nibble i+4
         -> half2 = ((w >> 4i) & 0x000F000F) | 0x64006400       (fp16 1024 + code)
  2-bit: 4 words per lane; word w = steps 2w, 2w+1 (pairs i = 0..7): first at slot i, second at slot i+8
         -> ((w >> 2i) & 0x00030003)
  3-bit: the 2-bit layout for the low two bits, plus a 1-bit plane of 2 words per lane; word w = steps 4w..4w+3
         (pairs i = 0..15): first at bit i, second at bit i+16 -> ((h >> i) & 0x00010001) << 2

  each plane is stored [uint4 index][lane] so a warp's load of one uint4 index is 512 contiguous bytes; the 3-bit
  high plane is [uint2 index 0][lane] (256 B). Tile weight bytes: 1024 (4b), 768 (3b), 512 (2b), then 48 B of
  group parameters: 16 fp16 scales (rows 0..15) followed by 16 u8 zeros. Dequant: (code - zero) * scale.

Strips of a matrix are contiguous, tiles of a strip contiguous, so one strip is one sequential stream.
gate_up is packed with rows permuted so strip s holds gate rows 8s..8s+7 as rows 0..7 and the matching up rows as
rows 8..15: each lane then has gate and up accumulators of the same row and tokens in registers for SiLU(g)*u.
"""
from __future__ import annotations

import numpy as np

GROUP = 128
META_BYTES = 48
TILE_W_BYTES = {2: 512, 3: 768, 4: 1024, 8: 2048}   # 8: format "i8" only


def tile_bytes(bits: int) -> int:
    return TILE_W_BYTES[bits] + META_BYTES


def matrix_bytes(R: int, K: int, bits: int) -> int:
    return (R // 16) * (K // GROUP) * tile_bytes(bits)


def _lane_values(codes: np.ndarray) -> np.ndarray:
    """codes [S, T, 16, 128] -> [S, T, 32 lanes, 32 pairs, 2] in fragment order."""
    lane = np.arange(32)
    r0, c0 = lane // 4, (lane % 4) * 2
    j = np.arange(8)
    # per pair i of a step: (row offset, col offset)
    pr = np.array([0, 8, 0, 8])
    pc = np.array([0, 0, 8, 8])
    rows = r0[:, None, None] + pr[None, None, :]                       # [32, 1, 4]
    cols = c0[:, None, None] + 16 * j[None, :, None] + pc[None, None, :]   # [32, 8, 4]
    rows = np.broadcast_to(rows, (32, 8, 4)).reshape(32, 32)
    cols = cols.reshape(32, 32)
    first = codes[:, :, rows, cols]                                      # [S, T, 32, 32]
    second = codes[:, :, rows, cols + 1]
    return np.stack([first, second], -1).astype(np.uint32)


def _words(v: np.ndarray, bits: int) -> list[np.ndarray]:
    """v [S, T, 32 lanes, 32 pairs, 2] (codes < 2^bits) -> planes, each [S, T, n_uint4_or_uint2, 32 lanes, words]."""
    S, T = v.shape[:2]
    if bits == 4:
        p = v.reshape(S, T, 32, 8, 4, 2)                                  # word j, pair i
        sh = 4 * np.arange(4, dtype=np.uint32)
        w = ((p[..., 0] << sh) | (p[..., 1] << (sh + 16))).sum(-1, dtype=np.uint64).astype(np.uint32)  # [S,T,32,8]
        return [w.reshape(S, T, 32, 2, 4).transpose(0, 1, 3, 2, 4)]       # [S,T,2 uint4,32,4]
    lo = v & 3
    p = lo.reshape(S, T, 32, 4, 8, 2)                                     # word w, pair i (0..7)
    sh = 2 * np.arange(8, dtype=np.uint32)
    w = ((p[..., 0] << sh) | (p[..., 1] << (sh + 16))).sum(-1, dtype=np.uint64).astype(np.uint32)      # [S,T,32,4]
    planes = [w.reshape(S, T, 32, 1, 4).transpose(0, 1, 3, 2, 4)]         # [S,T,1 uint4,32,4]
    if bits == 3:
        hi = v >> 2
        p = hi.reshape(S, T, 32, 2, 16, 2)
        sh = np.arange(16, dtype=np.uint32)
        h = ((p[..., 0] << sh) | (p[..., 1] << (sh + 16))).sum(-1, dtype=np.uint64).astype(np.uint32)  # [S,T,32,2]
        planes.append(h.reshape(S, T, 32, 1, 2).transpose(0, 1, 3, 2, 4))  # [S,T,1 uint2,32,2]
    return planes


def pack_matrix(codes: np.ndarray, scale: np.ndarray, zero: np.ndarray, bits: int) -> np.ndarray:
    """codes uint8 [R, K], scale fp16 [R, K/128], zero uint8 [R, K/128] -> uint8 blob of matrix_bytes(R, K, bits)."""
    R, K = codes.shape
    assert R % 16 == 0 and K % GROUP == 0 and int(codes.max(initial=0)) < 2 ** bits
    S, T = R // 16, K // GROUP
    c = codes.reshape(S, 16, T, GROUP).transpose(0, 2, 1, 3)              # [S, T, 16, 128]
    planes = _words(_lane_values(c), bits)
    w = np.ascontiguousarray(np.concatenate([p.reshape(S, T, -1) for p in planes], -1)).view(np.uint8)  # [S, T, wb]
    sc = scale.astype(np.float16).reshape(S, 16, T).transpose(0, 2, 1).copy().view(np.uint8).reshape(S, T, 32)
    zz = zero.astype(np.uint8).reshape(S, 16, T).transpose(0, 2, 1)
    out = np.concatenate([w, sc, zz], -1)
    assert out.shape[-1] == tile_bytes(bits)
    return np.ascontiguousarray(out).reshape(-1)


def gu_row_perm(I: int) -> np.ndarray:
    """Row order for the packed gate_up [2I, K] (gate rows 0..I-1, up rows I..2I-1): strip s = gate 8s..8s+7, up 8s..8s+7."""
    s = np.arange(I // 8)[:, None]
    k = np.arange(8)[None, :]
    return np.concatenate([8 * s + k, I + 8 * s + k], 1).reshape(-1)


def unpack_matrix(blob: np.ndarray, R: int, K: int, bits: int):
    """Inverse of pack_matrix (reference for tests): -> codes [R, K], scale [R, K/128], zero [R, K/128]."""
    S, T = R // 16, K // GROUP
    tb = blob.reshape(S, T, tile_bytes(bits))
    wb = TILE_W_BYTES[bits]
    words = tb[..., :wb].copy().view(np.uint32)                           # [S, T, wb/4]
    sc = tb[..., wb:wb + 32].copy().view(np.float16).reshape(S, T, 16)
    zz = tb[..., wb + 32:wb + 48].reshape(S, T, 16)
    if bits == 4:
        w = words.reshape(S, T, 2, 32, 4).transpose(0, 1, 3, 2, 4).reshape(S, T, 32, 8, 1)
        sh = 4 * np.arange(4, dtype=np.uint32)
        first, second = (w >> sh) & 0xF, (w >> (sh + 16)) & 0xF           # [S,T,32,8,4]
    else:
        lo = words[..., :512 // 4].reshape(S, T, 1, 32, 4).transpose(0, 1, 3, 2, 4).reshape(S, T, 32, 4, 1)
        sh = 2 * np.arange(8, dtype=np.uint32)
        first, second = (lo >> sh) & 3, (lo >> (sh + 16)) & 3             # [S,T,32,4,8]
        if bits == 3:
            hi = words[..., 512 // 4:].reshape(S, T, 1, 32, 2).transpose(0, 1, 3, 2, 4).reshape(S, T, 32, 2, 1)
            sh = np.arange(16, dtype=np.uint32)
            hf, hs = (hi >> sh) & 1, (hi >> (sh + 16)) & 1                # [S,T,32,2,16]
            first = first.reshape(S, T, 32, 32) | (hf.reshape(S, T, 32, 32) << 2)
            second = second.reshape(S, T, 32, 32) | (hs.reshape(S, T, 32, 32) << 2)
    first, second = first.reshape(S, T, 32, 32), second.reshape(S, T, 32, 32)
    lane = np.arange(32)
    r0, c0 = lane // 4, (lane % 4) * 2
    pr, pc = np.array([0, 8, 0, 8]), np.array([0, 0, 8, 8])
    rows = np.broadcast_to(r0[:, None, None] + pr[None, None, :], (32, 8, 4)).reshape(32, 32)
    cols = (c0[:, None, None] + 16 * np.arange(8)[None, :, None] + pc[None, None, :]).reshape(32, 32)
    c = np.zeros((S, T, 16, GROUP), np.uint8)
    c[:, :, rows, cols] = first
    c[:, :, rows, cols + 1] = second
    codes = c.transpose(0, 2, 1, 3).reshape(R, K)
    return codes, sc.transpose(0, 2, 1).reshape(R, T), zz.transpose(0, 2, 1).reshape(R, T)


# ---------------------------------------------------------------------------------------------------------------
# Format "i8" (v2): for mma.m16n8k32.row.col.s32.u8.s8.s32 -- weights enter the tensor cores as raw u8 codes,
# activations as int8 (per token, per 128-group symmetric scales). A tile is still 16 rows x 128 columns = one
# group, 4 k32-steps. Lane l (g = l/4, t = l%4) holds for step j the 4 registers
#   a0: row g,   cols 32j + 4t + 0..3      a1: row g+8, same cols
#   a2: row g,   cols 32j + 16 + 4t + 0..3 a3: row g+8, same cols
# each register = 4 codes in its 4 bytes. Words are laid out so ONE shift + ONE mask yields a whole register:
#   4-bit: words 2j, 2j+1 of the lane; word 2j: byte b low nibble = a0[b], high nibble = a1[b]; word 2j+1 likewise
#          a2/a3       -> a0 = w & 0x0F0F0F0F, a1 = (w >> 4) & 0x0F0F0F0F
#   2-bit: word j: bits 2r..2r+1 of byte b = a_r[b]            -> a_r = (w >> 2r) & 0x03030303
#   3-bit: low two bits as 2-bit; high plane words h (2 per lane): word j/2, bit i = (j%2)*4 + r of byte b = top bit
#          of a_r[b]                                              -> a_r |= ((h >> i) & 0x01010101) << 2
#   8-bit: uint4 j of the lane = (a0, a1, a2, a3) of step j, stored [uint4 j][lane] (2048 B)  -> no extraction
# Plane storage, tile bytes and the 48 B of group parameters are as in format v1.
# ---------------------------------------------------------------------------------------------------------------
def _lane_codes_i8(codes: np.ndarray) -> np.ndarray:
    """codes [S, T, 16, 128] -> [S, T, 32 lanes, 4 steps, 4 regs, 4 bytes] in m16n8k32 A-fragment order."""
    lane = np.arange(32)
    g, t = lane // 4, lane % 4
    j, r, b = np.arange(4), np.arange(4), np.arange(4)
    rows = g[:, None, None, None] + 8 * (r[None, None, :, None] % 2)                       # a1, a3: row g+8
    cols = 32 * j[None, :, None, None] + 16 * (r[None, None, :, None] // 2) + 4 * t[:, None, None, None] + b[None, None, None, :]
    rows = np.broadcast_to(rows, (32, 4, 4, 4))
    cols = np.broadcast_to(cols, (32, 4, 4, 4))
    return codes[:, :, rows, cols].astype(np.uint32)


def pack_matrix_i8(codes: np.ndarray, scale: np.ndarray, zero: np.ndarray, bits: int) -> np.ndarray:
    R, K = codes.shape
    assert R % 16 == 0 and K % GROUP == 0 and int(codes.max(initial=0)) < 2 ** bits
    S, T = R // 16, K // GROUP
    v = _lane_codes_i8(codes.reshape(S, 16, T, GROUP).transpose(0, 2, 1, 3))   # [S,T,32,4j,4r,4b]
    bsh = 8 * np.arange(4, dtype=np.uint32)
    if bits == 8:
        w = (v << bsh).sum(-1, dtype=np.uint64).astype(np.uint32)                      # [S,T,32,4j,4r]
        planes = [w.transpose(0, 1, 3, 2, 4)]                                          # [S,T,4 uint4,32,4]
    elif bits == 4:
        lo = (v[:, :, :, :, 0::2, :] << bsh) .sum(-1, dtype=np.uint64)             # r = 0, 2 -> low nibbles
        hi = (v[:, :, :, :, 1::2, :] << (bsh + 4)).sum(-1, dtype=np.uint64)       # r = 1, 3 -> high nibbles
        w = (lo | hi).astype(np.uint32).reshape(S, T, 32, 8)                       # word 2j + (r // 2)
        planes = [w.reshape(S, T, 32, 2, 4).transpose(0, 1, 3, 2, 4)]
    else:
        lo2 = v & 3
        rsh = 2 * np.arange(4, dtype=np.uint32)
        w = ((lo2 << (bsh[None, :] + rsh[:, None])).sum((-1, -2), dtype=np.uint64)).astype(np.uint32)  # [S,T,32,4j]
        planes = [w.reshape(S, T, 32, 1, 4).transpose(0, 1, 3, 2, 4)]
        if bits == 3:
            hb = (v >> 2) & 1                                                      # [S,T,32,4j,4r,4b]
            hb = hb.reshape(S, T, 32, 2, 2, 4, 4)                                  # word j//2, (j%2), r, b
            ish = (4 * np.arange(2, dtype=np.uint32))[:, None] + np.arange(4, dtype=np.uint32)[None, :]   # i = (j%2)*4 + r
            h = (hb << (bsh[None, None, :] + ish[:, :, None])).sum((-1, -2, -3), dtype=np.uint64).astype(np.uint32)
            planes.append(h.reshape(S, T, 32, 1, 2).transpose(0, 1, 3, 2, 4))
    w = np.ascontiguousarray(np.concatenate([p.reshape(S, T, -1) for p in planes], -1)).view(np.uint8)
    sc = scale.astype(np.float16).reshape(S, 16, T).transpose(0, 2, 1).copy().view(np.uint8).reshape(S, T, 32)
    zz = zero.astype(np.uint8).reshape(S, 16, T).transpose(0, 2, 1)
    out = np.concatenate([w, sc, zz], -1)
    assert out.shape[-1] == tile_bytes(bits)
    return np.ascontiguousarray(out).reshape(-1)


def unpack_matrix_i8(blob: np.ndarray, R: int, K: int, bits: int):
    S, T = R // 16, K // GROUP
    tb = blob.reshape(S, T, tile_bytes(bits))
    wb = TILE_W_BYTES[bits]
    words = tb[..., :wb].copy().view(np.uint32)
    sc = tb[..., wb:wb + 32].copy().view(np.float16).reshape(S, T, 16)
    zz = tb[..., wb + 32:wb + 48].reshape(S, T, 16)
    bsh = 8 * np.arange(4, dtype=np.uint32)
    v = np.zeros((S, T, 32, 4, 4, 4), np.uint32)                                    # lane, j, r, b
    if bits == 8:
        w = words.reshape(S, T, 4, 32, 4).transpose(0, 1, 3, 2, 4)                     # [S,T,32,4j,4r]
        v[:] = (w[..., None] >> bsh) & 0xFF
    elif bits == 4:
        w = words.reshape(S, T, 2, 32, 4).transpose(0, 1, 3, 2, 4).reshape(S, T, 32, 4, 2)   # [.., j, r//2]
        for rr in range(4):
            v[:, :, :, :, rr, :] = (w[..., rr // 2, None] >> (bsh + 4 * (rr % 2))) & 0xF
    else:
        w = words[..., :128].reshape(S, T, 1, 32, 4).transpose(0, 1, 3, 2, 4).reshape(S, T, 32, 4)
        for rr in range(4):
            v[:, :, :, :, rr, :] = (w[..., None] >> (bsh + 2 * rr)) & 3
        if bits == 3:
            h = words[..., 128:].reshape(S, T, 1, 32, 2).transpose(0, 1, 3, 2, 4).reshape(S, T, 32, 2)
            for j in range(4):
                for rr in range(4):
                    i = (j % 2) * 4 + rr
                    v[:, :, :, j, rr, :] |= ((h[..., j // 2, None] >> (bsh + i)) & 1) << 2
    lane = np.arange(32)
    g, t = lane // 4, lane % 4
    j, r, b = np.arange(4), np.arange(4), np.arange(4)
    rows = np.broadcast_to(g[:, None, None, None] + 8 * (r[None, None, :, None] % 2), (32, 4, 4, 4))
    cols = np.broadcast_to(32 * j[None, :, None, None] + 16 * (r[None, None, :, None] // 2) + 4 * t[:, None, None, None]
                           + b[None, None, None, :], (32, 4, 4, 4))
    c = np.zeros((S, T, 16, GROUP), np.uint8)
    c[:, :, rows, cols] = v
    return c.transpose(0, 2, 1, 3).reshape(R, K), sc.transpose(0, 2, 1).reshape(R, T), zz.transpose(0, 2, 1).reshape(R, T)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for b in (2, 3, 4):
        R, K = 64, 512
        cd = rng.integers(0, 2 ** b, (R, K), dtype=np.uint8)
        s = rng.standard_normal((R, K // GROUP)).astype(np.float16)
        z = rng.integers(0, 2 ** b, (R, K // GROUP), dtype=np.uint8)
        blob = pack_matrix(cd, s, z, b)
        assert blob.size == matrix_bytes(R, K, b)
        c2, s2, z2 = unpack_matrix(blob, R, K, b)
        assert (c2 == cd).all() and (s2 == s).all() and (z2 == z).all(), b
        blob = pack_matrix_i8(cd, s, z, b)
        if b == 4:                                                                     # also exercise i8-only 8-bit
            c8 = rng.integers(0, 256, (R, K), dtype=np.uint8)
            b8 = pack_matrix_i8(c8, s, z, 8)
            assert b8.size == matrix_bytes(R, K, 8)
            assert (unpack_matrix_i8(b8, R, K, 8)[0] == c8).all()
        assert blob.size == matrix_bytes(R, K, b)
        c2, s2, z2 = unpack_matrix_i8(blob, R, K, b)
        assert (c2 == cd).all() and (s2 == s).all() and (z2 == z).all(), ("i8", b)
    p = gu_row_perm(512)
    assert sorted(p) == list(range(1024)) and p[8] == 512 and p[16] == 8
    print("packlib roundtrip ok")
