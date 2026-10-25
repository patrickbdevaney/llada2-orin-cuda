#!/usr/bin/env python3
"""s03: BF16 teacher pass over the s02 corpus, one decoder layer at a time (the 32.5 GB checkpoint never
sits in the Orin's 8 GB; each layer is ~1.65 GB in BF16).

Two streams per row, in ONE forward (the block-diffusion training layout):
  clean  [0, L)    the row's tokens, block-causal attention (32-token blocks, bidirectional inside a block)
                   == how prompt / committed context is processed in serving
  noisy  [L, 2L)   the same tokens with eligible positions masked (MASK_ID) at a ratio r ~ U(0,1) drawn per
                   block; noisy block i attends clean blocks < i and itself
                   == the current block of a decode step
Block routing groups consecutive 32-token blocks; both halves are 32-aligned, so each half routes exactly as
it would in serving.

Banked per MoE layer (training rows only; held-out rows never touch an accumulator), by stream x bucket:
  cnt, gat, gsq, nrm, nsq, sum (g*||f||), sq   [2, n_buckets, E]   REAP / frequency / mass / EAN family
  pool                                          [2, E]              block-pool (top-48) memberships
  fmax                                          [E]                 max |f| (Super-Expert guard)
  F_sum, F_cnt                                  [2, E, E]           HOPE co-activation of g*||f||
Teacher outputs: for the noisy masked positions of every row, the argmax (sanity: masked-token accuracy) and,
for held-out rows, top-32 ids + log-probs (the arbitration target for s04 keep-sets and later quant gates).

Hidden states ping-pong on NVMe as fp16 memmaps (WORK/hs/{a,b}.bin, ~9.8 GB each at 1.2M tokens).
Resumable per layer (WORK/s03/state.json).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODEL = Path(os.environ.get("LLADA_MODEL", Path.home() / "models/LLaDA2.2-mini"))
WORK = Path(os.environ.get("LLADA_WORK", Path.home() / "models/llada-work"))
MASK_ID = 156895
BLOCK = 32
TOPK_CACHE = 32
DEV = "cuda"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_classes():
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    cfg._attn_implementation = "sdpa"
    ref = "modeling_llada2_moe."
    get = lambda n: get_class_from_dynamic_module(ref + n, MODEL)   # noqa: E731
    return cfg, get("LLaDA2MoeDecoderLayer"), get("LLaDA2MoeRotaryEmbedding"), get("LLaDA2MoeRMSNorm")


class Weights:
    """Tensor lookup across the safetensors shards (mmap; nothing is read until asked)."""

    def __init__(self):
        from safetensors import safe_open
        idx = json.load(open(MODEL / "model.safetensors.index.json"))["weight_map"]
        self.where = idx
        self.files = {f: safe_open(str(MODEL / f), "pt", device="cpu") for f in set(idx.values())}
        self.gpu = {}

    def get(self, name: str) -> torch.Tensor:
        return self.files[self.where[name]].get_tensor(name)

    def prefix(self, p: str, device: str = "cpu") -> dict[str, torch.Tensor]:
        """One tensor at a time straight to `device` (never a whole CPU copy of the layer next to its GPU copy:
        on the Orin both live in the same 7.4 GiB)."""
        return {k[len(p):]: self.get(k).to(device, torch.bfloat16) for k in self.where if k.startswith(p)}


def build_layer(cfg, Layer, W: Weights, i: int):
    with torch.device("meta"):
        layer = Layer(cfg, i)
    sd = W.prefix(f"model.layers.{i}.", DEV)
    sd = {k: (v.float() if k.endswith("expert_bias") else v) for k, v in sd.items()}
    layer.load_state_dict(sd, strict=True, assign=True)
    for name, buf in layer.named_buffers():          # non-persistent buffers (none expected) stay on meta otherwise
        if buf.is_meta:
            raise RuntimeError(f"unloaded buffer {name}")
    return layer.eval()


def stream_mask(L: int, device) -> torch.Tensor:
    """Additive [2L, 2L] mask for the clean+noisy layout."""
    # Filled quadrant by quadrant, one query block-row at a time: no [2L, 2L] bool temporaries (at L = 8192 the
    # mask alone is 512 MiB, and the Orin's GPU heap is its system RAM).
    nb = L // BLOCK
    m = torch.full((2 * L, 2 * L), float("-inf"), dtype=torch.bfloat16, device=device)
    for i in range(nb):
        r = slice(i * BLOCK, (i + 1) * BLOCK)
        m[r, : (i + 1) * BLOCK] = 0                                   # clean q -> clean k: block-causal
        m[L + i * BLOCK: L + (i + 1) * BLOCK, : i * BLOCK] = 0        # noisy q -> clean k: earlier blocks
        m[L + i * BLOCK: L + (i + 1) * BLOCK, L + i * BLOCK: L + (i + 1) * BLOCK] = 0   # noisy -> own block
    return m


def noisy_ids(tok: np.ndarray, elig: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    L = len(tok)
    r = np.repeat(rng.random(L // BLOCK), BLOCK)
    masked = (elig.astype(bool)) & (rng.random(L) < r)
    out = tok.copy()
    out[masked] = MASK_ID
    return out, masked


class Bank:
    def __init__(self, n_buckets: int, E: int):
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=DEV)   # noqa: E731
        self.cnt, self.gat, self.gsq, self.nrm, self.nsq, self.sum, self.sq = (
            z(2, n_buckets, E) for _ in range(7))
        self.pool = z(2, E)
        self.fmax = torch.zeros(E, dtype=torch.float32, device=DEV)
        self.F_sum, self.F_cnt = z(2, E, E), z(2, E, E)
        self.E = E

    def save(self, path: Path, buckets: list[str]):
        np.savez(path, buckets=np.array(buckets), **{k: getattr(self, k).cpu().numpy() for k in
                 ("cnt", "gat", "gsq", "nrm", "nsq", "sum", "sq", "pool", "fmax", "F_sum", "F_cnt")})


def route(gate, h: torch.Tensor):
    """LLaDA2MoeGate with block routing: -> topk_idx [N, K], weights [N, K] fp32, pool_idx [nb, C]."""
    logits = F.linear(h.float(), gate.weight.float())
    scores = torch.sigmoid(logits)
    s_route = scores + gate.expert_bias
    N, E = s_route.shape
    nb = N // gate.block_size
    blk = s_route.view(nb, gate.block_size, E).max(1).values
    pool_idx = torch.topk(blk, k=gate.expert_capacity, dim=-1).indices          # [nb, C]
    allowed = torch.zeros(nb, E, dtype=torch.bool, device=h.device).scatter_(1, pool_idx, True)
    allowed = allowed.repeat_interleave(gate.block_size, 0)
    topk_idx = torch.topk(s_route.masked_fill(~allowed, float("-inf")), k=gate.top_k, dim=-1).indices
    w = torch.gather(scores, 1, topk_idx)
    w = w / (w.sum(-1, keepdim=True) + 1e-20) * gate.routed_scaling_factor          # [N, K] fp32
    return topk_idx, w, pool_idx


def moe_forward(mlp, x: torch.Tensor, ctx: dict) -> torch.Tensor:
    """Re-implements LLaDA2MoeSparseMoeBlock.forward (block routing included) and banks statistics.
    ctx: bank (or None), stream [N] int (0 clean / 1 noisy), bucket [N] int, train [N] bool,
    capture (optional callable(h, topk_idx, w) that sees the MoE input and routing)."""
    gate = mlp.gate
    B, T, H = x.shape
    h = x.reshape(-1, H)
    topk_idx, w, pool_idx = route(gate, h)
    N, E = h.shape[0], gate.num_experts
    nb = N // gate.block_size
    if ctx.get("capture") is not None:
        ctx["capture"](h, topk_idx, w)

    K = topk_idx.shape[1]
    flat = topk_idx.reshape(-1)
    # Expert loop without materialising [N*K, H] copies: gather each expert's tokens, accumulate its weighted
    # output in place, and keep only the per-slot output norm and max (all the bank needs).
    order = flat.argsort()
    counts = torch.bincount(flat, minlength=E).tolist()
    acc = torch.zeros(N, H, dtype=torch.float32, device=h.device)
    fnorm = torch.empty(N * K, dtype=torch.float32, device=h.device)
    famax = torch.zeros(E, dtype=torch.float32, device=h.device)
    wflat = w.reshape(-1)
    start = 0
    for e, n in enumerate(counts):
        if n:
            sl = order[start:start + n]
            tok = sl // K
            out = mlp.experts[e](h[tok]).float()
            fnorm[sl] = out.norm(dim=-1)
            famax[e] = out.abs().max()
            acc.index_add_(0, tok, out * wflat[sl].unsqueeze(-1))
            del out
            start += n
    y = acc.to(h.dtype) + mlp.shared_experts(h)
    del acc
    fn = fnorm.view(N, K)

    bank = ctx.get("bank")
    if bank is not None:
        tr = ctx["train"]
        bank.fmax = torch.maximum(bank.fmax, famax)
        st, bu = ctx["stream"], ctx["bucket"]
        for s in (0, 1):
            sel = tr & (st == s)
            if not sel.any():
                continue
            ids, ww, nn_ = topk_idx[sel], w[sel], fn[sel]
            bb = bu[sel].unsqueeze(1).expand_as(ids)
            lin = (bb * E + ids).reshape(-1)
            gv, nv = ww.reshape(-1).double(), nn_.reshape(-1).double()
            sv = gv * nv
            for name, v in (("cnt", torch.ones_like(gv)), ("gat", gv), ("gsq", gv * gv), ("nrm", nv),
                            ("nsq", nv * nv), ("sum", sv), ("sq", sv * sv)):
                getattr(bank, name)[s].view(-1).index_add_(0, lin, v)
            sk = (ww * nn_).double()
            pair = (sk[:, :, None] * sk[:, None, :]).reshape(-1)
            pidx = (ids[:, :, None] * E + ids[:, None, :]).reshape(-1)
            bank.F_sum[s].view(-1).index_add_(0, pidx, pair)
            bank.F_cnt[s].view(-1).index_add_(0, pidx, torch.ones_like(pair))
        # pool memberships per block (a block is train iff its tokens are)
        blk_tr = tr.view(nb, gate.block_size)[:, 0]
        blk_st = st.view(nb, gate.block_size)[:, 0]
        for s in (0, 1):
            ps = pool_idx[blk_tr & (blk_st == s)].reshape(-1)
            bank.pool[s].index_add_(0, ps, torch.ones_like(ps, dtype=torch.float64))
    return y.view(B, T, H)


def batches(rows: list[dict], max_tokens: int):
    """Group rows of equal length into batches of <= max_tokens clean tokens."""
    by_len: dict[int, list[dict]] = {}
    for r in rows:
        by_len.setdefault(r["length"], []).append(r)
    for L, rs in sorted(by_len.items()):
        per = max(1, max_tokens // L)
        for i in range(0, len(rs), per):
            yield L, rs[i:i + per]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(WORK / "corpus"))
    ap.add_argument("--max-tokens", type=int, default=4096, help="clean tokens per layer call (x2 streams)")
    ap.add_argument("--limit-rows", type=int, default=0, help="debug: use only the first N rows")
    a = ap.parse_args()

    cdir = Path(a.corpus)
    tokens = np.load(cdir / "tokens.npy")
    elig = np.load(cdir / "elig.npy")
    rows = [json.loads(l) for l in open(cdir / "rows.jsonl")]
    if a.limit_rows:
        rows = rows[: a.limit_rows]
    buckets = sorted({r["bucket"] for r in rows})
    bidx = {b: i for i, b in enumerate(buckets)}
    out = WORK / "s03"
    (out / "bank").mkdir(parents=True, exist_ok=True)
    hsd = WORK / "hs"
    hsd.mkdir(parents=True, exist_ok=True)

    # flat 2L layout per row on disk: row r occupies [2*offset_r, 2*offset_r + 2L)
    T2 = 2 * sum(r["length"] for r in rows)
    off2, o = {}, 0
    for r in rows:
        off2[r["row"]] = o
        o += 2 * r["length"]
    H = 2048
    mm = {k: np.lib.format.open_memmap(hsd / f"{k}.npy", mode="r+" if (hsd / f"{k}.npy").exists() else "w+",
                                        dtype=np.float16, shape=(T2, H)) for k in ("a", "b")}

    state_p = out / "state.json"
    state = json.load(open(state_p)) if state_p.exists() else {"next_layer": -1}
    cfg, Layer, Rotary, RMSNorm = load_classes()
    W = Weights()
    rot = Rotary(cfg).to(DEV)
    n_layers = cfg.num_hidden_layers
    E = cfg.num_experts

    noisy_masks = {}
    for r in rows:
        tk = tokens[r["offset"]: r["offset"] + r["length"]]
        el = elig[r["offset"]: r["offset"] + r["length"]]
        noisy_masks[r["row"]] = noisy_ids(tk, el, seed=1_000_003 * r["row"] + 17)

    def row_ids(r):
        tk = tokens[r["offset"]: r["offset"] + r["length"]]
        return np.concatenate([tk, noisy_masks[r["row"]][0]])

    # ---- embeddings -> stream a
    if state["next_layer"] < 0:
        emb = W.get("model.word_embeddings.weight").to(DEV, torch.bfloat16)
        for r in rows:
            ids = torch.from_numpy(row_ids(r).astype(np.int64)).to(DEV)
            mm["a"][off2[r["row"]]: off2[r["row"]] + 2 * r["length"]] = emb[ids].half().cpu().numpy()
        del emb
        mm["a"].flush()
        state = {"next_layer": 0}
        json.dump(state, open(state_p, "w"))
        log(f"embeddings written: {T2:,} stream tokens")

    masks: dict[int, torch.Tensor] = {}
    for li in range(state["next_layer"], n_layers):
        src, dst = mm["a" if li % 2 == 0 else "b"], mm["b" if li % 2 == 0 else "a"]
        t0 = time.time()
        layer = build_layer(cfg, Layer, W, li)
        is_moe = hasattr(layer.mlp, "gate")
        bank = Bank(len(buckets), E) if is_moe else None
        if is_moe:
            mlp = layer.mlp
            ctx: dict = {}
            mlp.forward = lambda x, _m=mlp, _c=ctx: (moe_forward(_m, x, _c), None)
        n_done = 0
        for L, rs in batches(rows, a.max_tokens):
            if L not in masks:                                # batches arrive sorted by length: keep one mask
                masks.clear()                                 # (a 2Lx2L bf16 mask is 512 MiB at L = 8192)
                masks[L] = stream_mask(L, DEV)
            x = torch.stack([torch.from_numpy(np.asarray(src[off2[r["row"]]: off2[r["row"]] + 2 * L]))
                             for r in rs]).to(DEV, torch.bfloat16)
            pos = torch.arange(L, device=DEV).repeat(2).unsqueeze(0).expand(len(rs), -1)
            pe = rot(x, pos)
            if is_moe:
                ctx.update(bank=bank,
                           stream=torch.arange(2 * L, device=DEV).ge(L).long().repeat(len(rs)),
                           bucket=torch.tensor([bidx[r["bucket"]] for r in rs], device=DEV).repeat_interleave(2 * L),
                           train=torch.tensor([not r["heldout"] for r in rs], device=DEV).repeat_interleave(2 * L))
            with torch.no_grad():
                y = layer(x, attention_mask=masks[L][None, None].expand(len(rs), 1, -1, -1),
                          position_ids=pos, position_embeddings=pe)[0]
            if not torch.isfinite(y).all():
                raise RuntimeError(f"non-finite output at layer {li}")
            yh = y.half().cpu().numpy()
            for j, r in enumerate(rs):
                dst[off2[r["row"]]: off2[r["row"]] + 2 * L] = yh[j]
            n_done += len(rs)
        dst.flush()
        for m_ in (src, dst):
            fd = os.open(m_.filename, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        if bank is not None:
            bank.save(out / "bank" / f"L{li}.npz", buckets)
        mlp = ctx = None                                          # drop every reference before the next layer loads
        del layer, bank
        gc.collect()
        torch.cuda.empty_cache()
        state = {"next_layer": li + 1}
        json.dump(state, open(state_p, "w"))
        rss = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
        log(f"layer {li}: {n_done} rows in {time.time() - t0:.0f}s"
            f" (gpu peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, rss {rss:.2f} GiB)")
        torch.cuda.reset_peak_memory_stats()

    # ---- final norm + lm_head on noisy masked positions
    if state["next_layer"] == n_layers and not (out / "teacher_done.json").exists():
        last = mm["a" if n_layers % 2 == 0 else "b"]
        norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        norm.weight.data = W.get("model.norm.weight")
        norm = norm.to(DEV, torch.bfloat16)
        head = W.get("lm_head.weight").to(DEV, torch.bfloat16)
        acc = {b: [0, 0] for b in buckets}
        held = {"row": [], "pos": [], "ids": [], "logp": []}
        for r in rows:
            L = r["length"]
            ntk, masked = noisy_masks[r["row"]]
            pos = np.flatnonzero(masked)
            if pos.size == 0:
                continue
            # chunked: [positions x 157,184] fp32 logits for a long row would be ~2 GB, twice with log_softmax
            for c0 in range(0, pos.size, 256):
                pc = pos[c0:c0 + 256]
                hs = torch.from_numpy(np.asarray(last[off2[r["row"]] + L + pc])).to(DEV, torch.bfloat16)
                with torch.no_grad():
                    lg = (norm(hs) @ head.T).float()
                truth = torch.from_numpy(tokens[r["offset"] + pc].astype(np.int64)).to(DEV)
                acc[r["bucket"]][0] += int((lg.argmax(-1) == truth).sum())
                acc[r["bucket"]][1] += pc.size
                if r["heldout"]:
                    v, i = lg.log_softmax(-1).topk(TOPK_CACHE, dim=-1)
                    held["row"].append(np.full(pc.size, r["row"], np.int32))
                    held["pos"].append(pc.astype(np.int32))
                    held["ids"].append(i.cpu().numpy().astype(np.int32))
                    held["logp"].append(v.cpu().numpy().astype(np.float16))
                del lg
        np.savez(out / "teacher_heldout_top32.npz", **{k: np.concatenate(v) for k, v in held.items() if v})
        res = {b: {"masked": n, "top1_vs_truth": c / max(1, n)} for b, (c, n) in acc.items()}
        json.dump(res, open(out / "teacher_done.json", "w"), indent=1)
        log("teacher masked-token accuracy: " + ", ".join(f"{b} {v['top1_vs_truth']:.3f}" for b, v in res.items()))


if __name__ == "__main__":
    main()
