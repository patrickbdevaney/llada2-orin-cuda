#!/usr/bin/env python3
"""s06: replay candidate compressed models against the BF16 teacher on the HELD-OUT rows.

Every decision is measured as agreement with the unpruned teacher, never perplexity (DeepSeek/GLM REAP lesson):
at each noisy masked position of a held-out row (the decode state the model meets in serving)
  top1   student argmax == teacher argmax
  kl32   KL(teacher || student) over the teacher's top-32 ids (both renormalised over those ids)
  truth  student argmax == the real token
per bucket and overall, with 95% row-clustered bootstrap CIs and a paired McNemar z for top-1 against
--reference. Held-out rows never touched s03 saliency or s05 calibration.

Arms (repeat --arm):
  teacher            original experts; when present it is also the REFERENCE (in-run replay, no storage noise)
  rotid              original experts through the rotated expert path, unquantized (rotation exactness)
  keep:<npy>         s04 keep-set [n_moe_layers, E] bool; pruned experts get expert_bias = -inf, so neither
                     the block pool nor the token top-8 can select them (the serving rule)
  alloc:<npy>        s05 allocation [n_moe_layers, E] int8 bit-widths, 0 = pruned; codes from the s05 bank
  uniform:<b>        every expert at b bits from the s05 bank, nothing pruned
  uniform:<b>@<l,..> only those MoE layers quantized; elsewhere bits = -1 (original BF16 expert)
Name an arm with name=spec, e.g. --arm a16k=alloc:.../16384_int8.npy
Any arm can add +ne:<b>[/<lm_b>]: attention, shared experts and the dense layer-0 MLP replaced by s05n's b-bit GPTQ
weights (mapped back from the rotated basis), and lm_head by its lm_b-bit version when given.
alloc/uniform arms can add +act:<b>: routed-expert inputs x' and z' integer-quantized per token and 128-group
(symmetric, b bits), the int8 tensor-core path of the engine.
Any arm can add a KV-cache format: <spec>+kv:<Kbits>[c]/<Vbits>[h][g<group>], e.g. teacher+kv:4/4h; c = keys per channel over 32-token blocks. The clean-stream
(cached context) K and V are fake-quantized per token and head (asymmetric, fp16 scale and min per group, default
group = head_dim); h = an orthogonal 128-point rotation (random signs times Hadamard) of q and k after RoPE and of v
(undone on the output), so attention is unchanged in exact arithmetic. The noisy stream stands for the block
being denoised, which serving recomputes fresh every step, so its own K and V stay BF16. Clean queries also read
quantized clean keys, which is pessimistic for prefill.

Only routed experts (and, with +kv:, the KV cache) change. Attention weights, shared experts, dense layer 0 and
lm_head stay BF16 here; the full serving configuration is checked by the CUDA engine's own gates.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import quantlib as QL
from s03_teacher_pass import (DEV, WORK, Weights, batches, build_layer, load_classes, log, moe_forward,
                              noisy_ids, route, stream_mask)


def parse_arm(spec: str):
    name, _, body = spec.partition("=") if "=" in spec else ("", "", spec)
    body, *mods = body.split("+")
    mods = dict(m.split(":", 1) for m in mods)
    kv, ne, act = mods.pop("kv", ""), mods.pop("ne", ""), mods.pop("act", "")
    if mods:
        raise ValueError(f"unknown arm modifiers {list(mods)}")
    kind, _, arg = body.partition(":")
    name = name or (kind if not arg else f"{kind}:{Path(arg).stem if kind != 'uniform' else arg}") + \
        (f"+kv{kv}" if kv else "") + (f"+ne{ne}" if ne else "") + (f"+act{act}" if act else "")
    arm = {"name": name, "kind": kind, "arg": arg}
    if act:
        arm["act"] = int(act)
    if ne:
        b_, _, lm = ne.partition("/")
        arm["ne"], arm["ne_lm"] = int(b_), (int(lm) if lm else None)
    if kv:
        m = re.fullmatch(r"(\d+)(c?)/(\d+)(h?)(?:g(\d+))?", kv)
        if not m:
            raise ValueError(f"bad kv spec {kv!r}")
        arm["kv"] = {"kb": int(m[1]), "kchan": bool(m[2]), "vb": int(m[3]), "rot": bool(m[4]), "g": int(m[5] or 0)}
    return arm


def fake_quant(x: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """Per-group asymmetric uniform quantization of the last dim with fp16 scale and fp16 min, as the KV cache stores it."""
    if bits >= 16:
        return x
    sh = x.shape
    xf = x.float().reshape(*sh[:-1], sh[-1] // group, group)
    mn, mx = xf.amin(-1, keepdim=True), xf.amax(-1, keepdim=True)
    maxq = 2 ** bits - 1
    s = ((mx - mn) / maxq).clamp_min(QL.FP16_MIN_SCALE).half().float()
    mn = mn.half().float()
    q = torch.round((xf - mn) / s).clamp(0, maxq)
    return (q * s + mn).reshape(sh).to(x.dtype)


_KV_ROT = {}


def kvq_attention(module, query, key, value, attention_mask, **kw):
    """sdpa with the clean half of K/V (the cached context) fake-quantized per module._kvq; None = plain sdpa."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    spec = getattr(module, "_kvq", None)
    if spec is None:
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)
    D, L = key.shape[-1], key.shape[-2] // 2
    g = spec["g"] or D
    R = None
    if spec["rot"]:
        if D not in _KV_ROT:
            _KV_ROT[D] = QL.residual_rotation(D, seed=20261025, device=key.device)
        R = _KV_ROT[D]
        dt = key.dtype                                      # rotate in fp32: a bf16 product alone flips near-ties
        query, key, value = query.float() @ R, key.float() @ R, value.float() @ R
    else:
        dt = key.dtype
        key, value = key.clone(), value.clone()
    if spec["kchan"]:          # KIVI-style: per channel over each committed 32-token block (one scale/min per block x channel)
        kc = key[..., :L, :]
        B_, h_, _, _ = kc.shape
        kt = kc.reshape(B_, h_, L // 32, 32, D).transpose(-1, -2)          # [..., blocks, D, 32]
        key[..., :L, :] = fake_quant(kt, spec["kb"], 32).transpose(-1, -2).reshape(kc.shape)
    else:
        key[..., :L, :] = fake_quant(key[..., :L, :], spec["kb"], g)
    value[..., :L, :] = fake_quant(value[..., :L, :], spec["vb"], g)
    if R is not None:                                       # attention itself in fp32 for the rotated arms
        out, w = sdpa_attention_forward(module, query, key, value, attention_mask.float() if
                                        attention_mask is not None and attention_mask.is_floating_point() else attention_mask, **kw)
        return (out @ R.T).to(dt), w                        # [B, q, heads, D]: undo the value rotation
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)


class QuantExperts:
    """One layer's experts for an alloc/uniform arm: uint8 codes + fp16 scales on the GPU, dequantized per expert
    on use (the full dequantized layer would be 1.6 GB on top of the BF16 teacher layer)."""

    def __init__(self, qdir: Path, bits: np.ndarray, group: int):
        self.bits, self.group, self.t = bits, group, {}
        for b in sorted(set(int(x) for x in bits) - {0, -1}):
            sel = np.flatnonzero(bits == b)
            for k in ("gu", "gu_s", "gu_z", "d", "d_s", "d_z"):
                arr = np.load(qdir / f"b{b}_{k}.npy", mmap_mode="r")
                for e in sel:
                    self.t.setdefault(int(e), {})[k] = torch.from_numpy(np.ascontiguousarray(arr[e])).to(DEV)

    def weights(self, e: int):
        d = self.t[e]
        gu = QL.dequant(d["gu"], d["gu_s"], d["gu_z"], self.group)
        return gu[: gu.shape[0] // 2], gu[gu.shape[0] // 2:], QL.dequant(d["d"], d["d_s"], d["d_z"], self.group)


def act_quant(x: torch.Tensor, bits: int | None, group: int = 128) -> torch.Tensor:
    """Symmetric per-token, per-group integer activations (the engine's W·A8 path); None = unchanged."""
    if not bits:
        return x
    xf = x.float().reshape(*x.shape[:-1], x.shape[-1] // group, group)
    qmax = 2 ** (bits - 1) - 1
    s = (xf.abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-12)
    return (torch.round(xf / s).clamp(-qmax, qmax) * s).reshape(x.shape)


def rotated_moe(mlp, x, gamma, Q, H512, weights_of, act_bits=None):
    """MoE forward with experts in the rotated basis: x' = (x/γ)Q, z' = z H512, f = (z' Wdᵀ) Qᵀ.
    act_bits: integer-quantize x' and z' per token and 128-group, as the engine's int8 tensor-core path does
    (x' is first rounded to fp16, the engine's storage of it)."""
    gate = mlp.gate
    B, T, H = x.shape
    h = x.reshape(-1, H)
    topk_idx, w, _ = route(gate, h)
    N, K = topk_idx.shape
    xr = (h.float() / gamma) @ Q
    if act_bits:
        xr = act_quant(xr.half().float(), act_bits)
    flat = topk_idx.reshape(-1)
    order = flat.argsort()
    counts = torch.bincount(flat, minlength=gate.num_experts).tolist()
    acc = torch.zeros(N, H, dtype=torch.float32, device=h.device)
    wflat = w.reshape(-1)
    start = 0
    for e, n in enumerate(counts):
        if n:
            sl = order[start:start + n]
            tok = sl // K
            g, u, d = weights_of(e)
            xe = xr[tok]
            f = act_quant((F.silu(xe @ g.T) * (xe @ u.T)).half().float() @ H512, act_bits) @ d.T
            acc.index_add_(0, tok, f * wflat[sl].unsqueeze(-1))
            start += n
    y = (acc @ Q.T).to(h.dtype) + mlp.shared_experts(h)
    return y.view(B, T, H)


def bootstrap(per_row_num: np.ndarray, per_row_den: np.ndarray, n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_row_num), size=(n, len(per_row_num)))
    stats = per_row_num[idx].sum(1) / np.maximum(per_row_den[idx].sum(1), 1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(WORK / "corpus"))
    ap.add_argument("--teacher", default=str(WORK / "s03" / "teacher_heldout_top32.npz"))
    ap.add_argument("--qbank", default=str(WORK / "s05"))
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--reference", default=None, help="arm name for paired McNemar (default: first non-teacher)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--out", default=str(WORK / "s06"))
    a = ap.parse_args()

    arms = [parse_arm(s) for s in a.arm]
    out = Path(a.out)
    (out / "hs").mkdir(parents=True, exist_ok=True)
    cdir = Path(a.corpus)
    tokens, elig = np.load(cdir / "tokens.npy"), np.load(cdir / "elig.npy")
    rows = [json.loads(l) for l in open(cdir / "rows.jsonl")]
    rows = [r for r in rows if r["heldout"]]
    off2, o = {}, 0
    for r in rows:
        off2[r["row"]] = o
        o += 2 * r["length"]
    T2, H = o, 2048
    log(f"{len(rows)} held-out rows, {T2:,} stream tokens, arms: {[x['name'] for x in arms]}")

    cfg, Layer, Rotary, RMSNorm = load_classes()
    from transformers.modeling_utils import AttentionInterface
    AttentionInterface.register("kvq", kvq_attention)
    cfg._attn_implementation = "kvq"
    W = Weights()
    rot = Rotary(cfg).to(DEV)
    E = cfg.num_experts
    moe_layers = [l for l in range(cfg.num_hidden_layers) if l >= cfg.first_k_dense_replace]
    qmeta = json.load(open(Path(a.qbank) / "q_meta.json")) if (Path(a.qbank) / "q_meta.json").exists() else None
    group = qmeta["group"] if qmeta else 128
    Q = QL.residual_rotation(H)
    H512 = QL.hadamard(cfg.moe_intermediate_size)

    for arm in arms:
        if arm["kind"] == "keep":
            arm["keep"] = np.load(arm["arg"])
        elif arm["kind"] == "alloc":
            arm["bits"] = np.load(arm["arg"]).astype(np.int8)
        elif arm["kind"] == "uniform":
            b_, _, ls = arm["arg"].partition("@")
            arm["bits"] = np.full((len(moe_layers), E), int(b_), np.int8)
            if ls:
                only = {int(x) for x in ls.split(",")}
                for i, l in enumerate(moe_layers):
                    if l not in only:
                        arm["bits"][i] = -1
        if "bits" in arm or "keep" in arm:
            assert (arm.get("bits", arm.get("keep"))).shape == (len(moe_layers), E), arm["name"]
        # hidden states as raw bf16 bits (int16): lossless vs the compute dtype, and a quantized arm whose residual
        # stream grows past fp16's 65,504 (the teacher already reaches ~38k) cannot turn into a silent inf
        arm["mm"] = {k: np.lib.format.open_memmap(out / "hs" / f"{arm['name'].replace(':', '_').replace('/', '_')}_{k}.npy",
                                                   mode="w+", dtype=np.int16, shape=(T2, H)) for k in ("a", "b")}

    noisy = {r["row"]: noisy_ids(tokens[r["offset"]: r["offset"] + r["length"]],
                                 elig[r["offset"]: r["offset"] + r["length"]], 1_000_003 * r["row"] + 17)
             for r in rows}
    emb = W.get("model.word_embeddings.weight").to(DEV, torch.bfloat16)
    for r in rows:
        ids = np.concatenate([tokens[r["offset"]: r["offset"] + r["length"]], noisy[r["row"]][0]])
        e_ = emb[torch.from_numpy(ids.astype(np.int64)).to(DEV)].view(torch.int16).cpu().numpy()
        for arm in arms:
            arm["mm"]["a"][off2[r["row"]]: off2[r["row"]] + 2 * r["length"]] = e_
    del emb

    masks: dict[int, torch.Tensor] = {}
    NE = WORK / "s05n" / "eff"
    for li in range(cfg.num_hidden_layers):
        t0 = time.time()
        layer = build_layer(cfg, Layer, W, li)
        ne_b = sorted({x["ne"] for x in arms if x.get("ne")})
        params = dict(layer.named_parameters())
        ne_eff = {b: torch.load(NE / f"L{li}_b{b}.pt") for b in ne_b}
        ne_orig = {k: params[k].detach().clone() for k in (ne_eff[ne_b[0]] if ne_b else {})}
        is_moe = li in moe_layers
        mi = moe_layers.index(li) if is_moe else None
        mlp = layer.mlp
        bias0 = mlp.gate.expert_bias.clone() if is_moe else None
        gamma = layer.post_attention_layernorm.weight.float()
        # arms that need the original experts first, then free them before the quantized arms
        ordered = sorted(arms, key=lambda x: x["kind"] in ("alloc", "uniform") and is_moe
                         and not (x["bits"][mi] == -1).all())
        freed = False
        for arm in ordered:
            src = arm["mm"]["a" if li % 2 == 0 else "b"]
            dst = arm["mm"]["b" if li % 2 == 0 else "a"]
            layer.attention._kvq = arm.get("kv")
            for k, v in (ne_eff[arm["ne"]] if arm.get("ne") else ne_orig).items():
                params[k].data.copy_(v.to(DEV))
            if is_moe:
                mlp.gate.expert_bias.copy_(bias0)
                kind = arm["kind"]
                if kind in ("alloc", "uniform") and (arm["bits"][mi] == -1).all():
                    fwd = lambda x: moe_forward(mlp, x, {})                                    # noqa: E731
                elif kind in ("alloc", "uniform"):
                    if (arm["bits"][mi] == -1).any():
                        raise ValueError("mixing BF16 (-1) and quantized experts inside one layer is not supported")
                    if not freed:
                        for ex in mlp.experts:
                            for p_ in ex.parameters():
                                p_.data = torch.empty(0, device=DEV, dtype=p_.dtype)
                        gc.collect(); torch.cuda.empty_cache()
                        freed = True
                    bits = arm["bits"][mi]
                    mlp.gate.expert_bias[torch.from_numpy(np.flatnonzero(bits == 0)).to(DEV)] = float("-inf")
                    qe = QuantExperts(Path(a.qbank) / "q" / f"L{li}", bits, group)
                    fwd = lambda x, _q=qe, _ab=arm.get("act"): rotated_moe(mlp, x, gamma, Q, H512, _q.weights, _ab)  # noqa: E731
                elif kind == "rotid":
                    def wo(e):
                        ex = mlp.experts[e]
                        return ((ex.gate_proj.weight.float() * gamma) @ Q, (ex.up_proj.weight.float() * gamma) @ Q,
                                Q.T @ ex.down_proj.weight.float() @ H512)
                    fwd = lambda x: rotated_moe(mlp, x, gamma, Q, H512, wo)                    # noqa: E731
                else:
                    if kind == "keep":
                        mlp.gate.expert_bias[torch.from_numpy(np.flatnonzero(~arm["keep"][mi])).to(DEV)] = float("-inf")
                    fwd = lambda x: moe_forward(mlp, x, {})                                    # noqa: E731
                mlp.forward = lambda x, _f=fwd: (_f(x), None)
            for L, rs in batches(rows, a.max_tokens):
                if L not in masks:
                    masks.clear()
                    masks[L] = stream_mask(L, DEV)
                x = torch.stack([torch.from_numpy(np.asarray(src[off2[r["row"]]: off2[r["row"]] + 2 * L]))
                                 for r in rs]).to(DEV).view(torch.bfloat16)
                pos = torch.arange(L, device=DEV).repeat(2).unsqueeze(0).expand(len(rs), -1)
                with torch.no_grad():
                    y = layer(x, attention_mask=masks[L][None, None].expand(len(rs), 1, -1, -1),
                              position_ids=pos, position_embeddings=rot(x, pos))[0]
                if not torch.isfinite(y).all():
                    raise RuntimeError(f"non-finite output: arm {arm['name']} layer {li}")
                for j, r in enumerate(rs):
                    dst[off2[r["row"]]: off2[r["row"]] + 2 * L] = y[j].contiguous().view(torch.int16).cpu().numpy()
            dst.flush()
            if is_moe and arm["kind"] in ("alloc", "uniform") and "qe" in dir():
                del qe
        mlp = fwd = qe = params = ne_eff = ne_orig = None         # drop every reference before the next layer loads
        del layer
        gc.collect()
        torch.cuda.empty_cache()
        log(f"layer {li}: {len(arms)} arms in {time.time() - t0:.0f}s")

    # ---- readout and metrics
    norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    norm.weight.data = W.get("model.norm.weight")
    norm = norm.to(DEV, torch.bfloat16)
    # fp32 readout: bf16 logits (~0.125 resolution near 20) tie often enough that argmax and top-k disagreed on
    # ~5% of positions for the teacher against ITSELF in the first full run
    head_cur = {"key": "unset", "t": None}

    def head_for(lm_b):
        """fp32 lm_head (base or s05n lm_b-bit), one resident at a time: each is 1.29 GB."""
        if head_cur["key"] != lm_b:
            head_cur["t"] = None
            gc.collect(); torch.cuda.empty_cache()
            # mmap + chunked upload: a whole CPU copy (0.64 GB) next to the 1.29 GB fp32 GPU copy was OOM-killed
            w_ = W.get("lm_head.weight") if lm_b is None else \
                torch.load(WORK / "s05n" / "eff" / f"lm_head_b{lm_b}.pt", mmap=True)["lm_head.weight"]
            t_ = torch.empty(w_.shape, dtype=torch.float32, device=DEV)
            for r0 in range(0, w_.shape[0], 16384):
                t_[r0:r0 + 16384] = w_[r0:r0 + 16384].to(DEV).float()
            head_cur["key"], head_cur["t"] = lm_b, t_
            del w_
        return head_cur["t"]

    head = head_for(None)
    final = "a" if cfg.num_hidden_layers % 2 == 0 else "b"

    # Reference distribution. With a `teacher` arm in this run, its own replay is the reference: same storage,
    # batching and kernels as every other arm, so the comparison carries no replay noise (against s03's saved
    # top-32 the teacher itself only reached ~0.957 top-1 on the smoke row: near-tie masked positions flip under
    # fp16-vs-bf16 storage and batching). Without one, fall back to s03's saved top-32.
    t_arm = next((x for x in arms if x["kind"] == "teacher"), None)
    if t_arm is not None:
        ref = {"row": [], "pos": [], "ids": [], "logp": []}
        for r in rows:
            pos = np.flatnonzero(noisy[r["row"]][1])
            for c0 in range(0, pos.size, 256):
                pc = pos[c0:c0 + 256]
                hs = torch.from_numpy(np.asarray(t_arm["mm"][final][off2[r["row"]] + r["length"] + pc])).to(DEV).view(torch.bfloat16)
                v, i = (norm(hs).float() @ head.T).log_softmax(-1).topk(32, dim=-1)
                ref["row"].append(np.full(pc.size, r["row"], np.int32)); ref["pos"].append(pc.astype(np.int32))
                ref["ids"].append(i.cpu().numpy().astype(np.int32)); ref["logp"].append(v.cpu().numpy())
        tz = {k: np.concatenate(v) for k, v in ref.items()}
        log("reference: in-run teacher arm")
    else:
        tz = dict(np.load(a.teacher))
        log("reference: s03 saved top-32 (carries replay noise; add a teacher arm)")
    t_key = {(int(r), int(p)): i for i, (r, p) in enumerate(zip(tz["row"], tz["pos"]))}
    buckets = sorted({r["bucket"] for r in rows})
    res, flags = {}, {}
    for arm in sorted(arms, key=lambda x: x.get("ne_lm") or 0):
        head = head_for(arm.get("ne_lm"))
        per = {"row": [], "bucket": [], "n": [], "top1": [], "kl": [], "truth": []}
        fl = []
        for r in rows:
            L = r["length"]
            masked = noisy[r["row"]][1]
            pos = np.flatnonzero(masked)
            keys = [t_key.get((r["row"], int(p))) for p in pos]
            if not pos.size or any(k is None for k in keys):
                continue
            t1s, kls, trs, fls = 0.0, 0.0, 0.0, []
            for c0 in range(0, pos.size, 256):                   # chunked: full-vocab logits are ~0.6 MB/position
                pc, kc = pos[c0:c0 + 256], keys[c0:c0 + 256]
                hs = torch.from_numpy(np.asarray(arm["mm"][final][off2[r["row"]] + L + pc])).to(DEV).view(torch.bfloat16)
                with torch.no_grad():
                    lp = (norm(hs).float() @ head.T).log_softmax(-1)
                ids = torch.from_numpy(tz["ids"][kc].astype(np.int64)).to(DEV)
                P = torch.from_numpy(tz["logp"][kc].astype(np.float32)).to(DEV).softmax(-1)
                logq = lp.gather(1, ids).log_softmax(-1)
                kls += float((P * (P.clamp_min(1e-30).log() - logq)).sum())
                am = lp.argmax(-1)
                t1 = (am == ids[:, 0])
                t1s += float(t1.sum())
                trs += float((am.cpu().numpy() == tokens[r["offset"] + pc]).sum())
                fls.append(t1.cpu().numpy())
                del lp
            fl.append(np.concatenate(fls))
            per["row"].append(r["row"]); per["bucket"].append(r["bucket"]); per["n"].append(pos.size)
            per["top1"].append(t1s); per["kl"].append(kls); per["truth"].append(trs)
        flags[arm["name"]] = np.concatenate(fl)
        n = np.array(per["n"], float)
        out_a = {"overall": {}}
        for m in ("top1", "kl", "truth"):
            v = np.array(per[m])
            out_a["overall"][m] = float(v.sum() / n.sum())
            out_a["overall"][m + "_ci"] = bootstrap(v, n)
        for b in buckets:
            sel = np.array([x == b for x in per["bucket"]])
            if sel.any():
                out_a[b] = {m: float(np.array(per[m])[sel].sum() / n[sel].sum()) for m in ("top1", "kl", "truth")}
        out_a["positions"] = int(n.sum())
        res[arm["name"]] = out_a

    res = {x["name"]: res[x["name"]] for x in arms}          # report in --arm order
    refname = a.reference or next((x["name"] for x in arms if x["kind"] != "teacher"), arms[0]["name"])
    for name, f in flags.items():
        if name != refname and len(f) == len(flags[refname]):
            b_ = int((f & ~flags[refname]).sum()); c_ = int((~f & flags[refname]).sum())
            res[name]["mcnemar_vs_" + refname] = {"only_this": b_, "only_ref": c_,
                                                 "z": (b_ - c_) / max(1.0, (b_ + c_) ** 0.5)}
    json.dump(res, open(out / "results.json", "w"), indent=1)
    hdr = f"| arm | top-1 | kl32 | truth | " + " | ".join(f"{b} top-1" for b in buckets) + " |"
    lines = [hdr, "|" + "---|" * (4 + len(buckets))]
    for name, r in res.items():
        o_ = r["overall"]
        lines.append(f"| {name} | {o_['top1']:.4f} [{o_['top1_ci'][0]:.3f}, {o_['top1_ci'][1]:.3f}] | {o_['kl']:.4f} | "
                     f"{o_['truth']:.4f} | " + " | ".join(f"{r.get(b, {}).get('top1', float('nan')):.3f}" for b in buckets) + " |")
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
