"""Quantization primitives for the LLaDA2.2-mini expert pipeline (torch, GPU).

Format (what the Orin kernels will read): asymmetric uniform b-bit codes, b in {2, 3, 4}, one (fp16 scale,
u8 zero) per `group` consecutive input columns of each output row -> 0.1875 bpw of overhead at group 128.

GPTQ (Frantar et al.), batched over experts, lazy 128-column blocks, group parameters fitted when the group is
reached from the CURRENT error-updated weights, with a per-row clipping search (shrink grid) weighted by the
Hessian diagonal. No act-order: group order stays contiguous so the kernel never needs a g_idx gather.

Rotation (QuaRot-style, exact in floating point):
  residual stream  x' = x_noγ Q,  W_gu' = (W_gu diag γ) Q          Q = diag(±1) H_2048 / sqrt(2048)
  expert down in   z' = z H_512,  W_down' = Qᵀ W_down H_512         H_512 symmetric orthonormal (online FWHT)
Output norms are rotation-invariant, so errors are measured in the rotated space.
"""
from __future__ import annotations

import math

import torch

CLIP_GRID = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)
FP16_MIN_SCALE = 6.2e-5          # smallest normal fp16; an all-zero group then encodes as exact zeros


def bytes_per_param(bits: int, group: int = 128) -> float:
    return 0.0 if bits == 0 else (bits + 24.0 / group) / 8.0


def hadamard(n: int, device="cuda") -> torch.Tensor:
    assert n & (n - 1) == 0
    H = torch.ones(1, 1, device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def residual_rotation(n: int = 2048, seed: int = 20261024, device="cuda") -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    d = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float().to(device)
    return d[:, None] * hadamard(n, device)             # diag(d) @ H  (orthogonal)


def _fit(W: torch.Tensor, imp: torch.Tensor, maxq: int):
    """W [B, R, g], imp [B, g] -> scale, zero [B, R] (asymmetric, clipped), minimising imp-weighted MSE."""
    wmin, wmax = W.amin(-1), W.amax(-1)
    best_err, best_s, best_z = None, None, None
    for a in CLIP_GRID:
        lo, hi = (wmin * a).clamp_max(0), (wmax * a).clamp_min(0)       # range holds 0: zero point in [0, maxq]
        s = ((hi - lo) / maxq).clamp_min(FP16_MIN_SCALE)   # scales are stored fp16: a 1e-8 floor became 0 -> 0/0
        z = torch.round(-lo / s).clamp(0, maxq)
        q = torch.clamp(torch.round(W / s[..., None]) + z[..., None], 0, maxq)
        err = (((q - z[..., None]) * s[..., None] - W) ** 2 * imp[:, None, :]).sum(-1)
        if best_err is None:
            best_err, best_s, best_z = err, s, z
        else:
            m = err < best_err
            best_err = torch.where(m, err, best_err)
            best_s, best_z = torch.where(m, s, best_s), torch.where(m, z, best_z)
    return best_s, best_z


def prepare(H: torch.Tensor, damp: float = 0.01):
    """H [B, C, C] fp32 (sum x xᵀ) -> (U upper Cholesky of H⁻¹, Hessian diagonal, dead-column mask), on H's device.
    The factorisation runs on the CPU in float64 (JetPack 6's cuSOLVER 11.6 lacks the symbols torch 2.10's CUDA
    linalg needs), ONE matrix at a time: a batch of 8 at 2048² in fp64 is ~1.1 GB of CPU buffers, and on the Orin
    that is the same RAM as the GPU heap -- it swapped the process out (1,350 major faults/s) in the first s05 run."""
    dev = H.device
    B, C, _ = H.shape
    U = torch.empty(B, C, C, dtype=torch.float32, device=dev)
    diag_out = torch.empty(B, C, dtype=torch.float32, device=dev)
    dead_out = torch.empty(B, C, dtype=torch.bool, device=dev)
    for b in range(B):
        Hc = H[b].detach().double().cpu()
        diag = torch.diagonal(Hc)
        dead = diag <= 0
        diag[dead] = 1.0
        mean = diag.mean()
        Hc += torch.eye(C, dtype=Hc.dtype) * (damp * mean)
        for attempt in range(6):
            L, info = torch.linalg.cholesky_ex(Hc)
            if int(info) == 0:
                break
            Hc += torch.eye(C, dtype=Hc.dtype) * (damp * mean * 10 ** attempt)
        else:
            raise RuntimeError("H not positive definite")
        U[b] = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True).float().to(dev)
        diag_out[b] = torch.diagonal(Hc).float().to(dev)
        dead_out[b] = dead.to(dev)
        del Hc, L
    return U, diag_out, dead_out


def gptq(W: torch.Tensor, H: torch.Tensor | None, bits: int, group: int = 128, damp: float = 0.01,
         block: int = 128, prep=None, codes: bool = False):
    """W [B, R, C] (any float), H [B, C, C] fp32 (sum x xᵀ) or prep = prepare(H) to reuse one factorisation
    across bit-widths. Returns the dequantized weights [B, R, C] fp32, and with codes=True also
    (codes uint8 [B, R, C], scale fp16 [B, R, C/group], zero uint8 [B, R, C/group])."""
    W = W.float().clone()
    B, R, C = W.shape
    assert C % group == 0 and block % group == 0
    maxq = 2 ** bits - 1
    U, imp, dead = prep if prep is not None else prepare(H, damp)
    W[dead.unsqueeze(1).expand_as(W)] = 0.0
    Q = torch.empty_like(W)
    if codes:
        qc = torch.empty(B, R, C, dtype=torch.uint8, device=W.device)
        sc = torch.empty(B, R, C // group, dtype=torch.float16, device=W.device)
        zc = torch.empty(B, R, C // group, dtype=torch.uint8, device=W.device)
    for c0 in range(0, C, block):
        c1 = min(C, c0 + block)
        W1 = W[:, :, c0:c1].clone()
        Err = torch.empty_like(W1)
        s = z = None
        for k in range(c1 - c0):
            c = c0 + k
            if c % group == 0:
                s, z = _fit(W1[:, :, k:k + group], imp[:, c:c + group], maxq)
                s = s.half().float()                     # the kernel stores fp16 scales: quantize against those
                if codes:
                    sc[:, :, c // group], zc[:, :, c // group] = s.half(), z.to(torch.uint8)
            w = W1[:, :, k]
            q = torch.clamp(torch.round(w / s) + z, 0, maxq)
            if codes:
                qc[:, :, c] = q.to(torch.uint8)
            dq = (q - z) * s
            Q[:, :, c] = dq
            err = (w - dq) / U[:, c, c][:, None]
            Err[:, :, k] = err
            if k + 1 < c1 - c0:
                W1[:, :, k + 1:] -= err[:, :, None] * U[:, None, c, c + 1:c1]
        if c1 < C:
            W[:, :, c1:] -= torch.bmm(Err, U[:, c0:c1, c1:])
    if not torch.isfinite(Q).all():
        raise FloatingPointError("gptq produced non-finite weights")
    return (Q, qc, sc, zc) if codes else Q


def dequant(qc: torch.Tensor, sc: torch.Tensor, zc: torch.Tensor, group: int = 128) -> torch.Tensor:
    """codes [..., R, C] uint8, scale/zero [..., R, C/group] -> weights [..., R, C] fp32."""
    s = sc.float().repeat_interleave(group, -1)
    z = zc.float().repeat_interleave(group, -1)
    return (qc.float() - z) * s


def rtn(W: torch.Tensor, bits: int, group: int = 128, codes: bool = False):
    """Round-to-nearest with the same clipped asymmetric groups (fallback for experts with too few tokens)."""
    W = W.float()
    B, R, C = W.shape
    out = torch.empty_like(W)
    qc = torch.empty(B, R, C, dtype=torch.uint8, device=W.device)
    sc = torch.empty(B, R, C // group, dtype=torch.float16, device=W.device)
    zc = torch.empty(B, R, C // group, dtype=torch.uint8, device=W.device)
    maxq = 2 ** bits - 1
    ones = torch.ones(B, group, device=W.device)
    for c0 in range(0, C, group):
        g = W[:, :, c0:c0 + group]
        s, z = _fit(g, ones, maxq)
        s = s.half().float()
        q = torch.clamp(torch.round(g / s[..., None]) + z[..., None], 0, maxq)
        out[:, :, c0:c0 + group] = (q - z[..., None]) * s[..., None]
        qc[:, :, c0:c0 + group] = q.to(torch.uint8)
        sc[:, :, c0 // group], zc[:, :, c0 // group] = s.half(), z.to(torch.uint8)
    return (out, qc, sc, zc) if codes else out
