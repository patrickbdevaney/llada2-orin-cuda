#!/usr/bin/env python3
"""s02: executor-weighted calibration corpus for LLaDA2.2-mini.

Sources, provenance policy (no Claude-written text) and licence rules come from the DeepSeek-V4.1 REAP run
(ds_corpus_spec.py, vendored). What changes here:
  * the token mix, weighted for a fast agent executor (tool calls, code, patches) with general/math/science
    kept as guards, and multilingual as a small collapse guard (the Ling tokenizer is CJK-heavy);
  * rendering through LLaDA2.2's own no-think chat template (reasoning traces are dropped by the template,
    which is what the model sees in serving);
  * a per-token `elig` mask marking assistant spans (all tokens of raw documents). s03 masks only eligible
    tokens in the noisy stream, so routing is calibrated on the decode states the model actually meets;
  * 2,048-token rows (s03 doubles each row into a clean + noisy stream), longctx rows up to 8,192.

Output (WORK/corpus): tokens.npy [T] int32, elig.npy [T] uint8, rows.jsonl (row, bucket, offset, length,
heldout), sources.json, manifest.json (written last: marks completion).

RUN:  python s02_corpus.py --dry-run | --probe | (build)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ds_corpus_spec as ds  # noqa: E402

MODEL = Path(os.environ.get("LLADA_MODEL", Path.home() / "models/LLaDA2.2-mini"))
WORK = Path(os.environ.get("LLADA_WORK", Path.home() / "models/llada-work"))
OUT = WORK / "corpus"

SEQ_LEN = 2048
LONG_MIN, LONG_MAX = 6144, 8192
TOTAL_TOKENS = 1_200_000          # ~37k routed slots per expert on average (x2 with the noisy stream)
HELDOUT_FRACTION = 0.10

# Executor mix (token shares). Vision and finance are dropped (LLaDA2.2-mini is text-only and is not a finance
# worker); their share goes to agentic and code.
TOKEN_TARGET = {
    "agentic": 0.26,
    "code": 0.24,
    "general": 0.18,
    "math": 0.10,
    "science": 0.06,
    "longctx": 0.10,
    "multilingual": 0.06,
}
assert abs(sum(TOKEN_TARGET.values()) - 1.0) < 1e-9
BUCKETS = list(TOKEN_TARGET)

LONG_SOURCES = [
    ds.S("emozilla/pg19", None, "train", 0.5, ds._text("text"), licence="apache-2.0 (public-domain books)",
         generator="human"),
    ds.S("nebius/SWE-rebench", None, "test", 0.5, ds._qa(["problem_statement"], ["patch"]), licence="cc-by-4.0",
         generator="human"),
]


def pool_sources(bucket: str) -> list[dict]:
    srcs = LONG_SOURCES if bucket == "longctx" else ds.SOURCES[bucket]
    return [s for s in srcs if not s["hf_id"].startswith(("local:", "git:"))]   # no selfgen traces yet


class Renderer:
    """Document -> (ids, elig) through LLaDA2.2's tokenizer and chat template."""
    ASSIST = re.compile(r"<role>assistant</role>(.*?)(<\|role_end\|>|$)", re.S)

    def __init__(self):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    def render(self, doc) -> tuple[list[int], list[int]] | None:
        if doc is None:
            return None
        if isinstance(doc, str):
            ids = self.tok(doc, add_special_tokens=False)["input_ids"]
            return ids, [1] * len(ids)
        msgs, tools = [], None
        for m in doc:
            m = dict(m)
            if "tools" in m:                                   # ds._xlam puts tools on the user turn
                tools = [t.get("function", t) for t in m.pop("tools")]
                tools = [{"type": "function", "function": t} for t in tools]
            m.pop("reasoning_content", None)                    # no-think template drops it anyway
            msgs.append(m)
        text = self.tok.apply_chat_template(msgs, tools=tools, tokenize=False)
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        spans = [(m.start(1), m.end(2)) for m in self.ASSIST.finditer(text)]   # content + <|role_end|>
        elig = [int(any(a <= s < b for a, b in spans)) for s, _ in enc["offset_mapping"]]
        if not any(elig):
            return None
        return enc["input_ids"], elig


def stream(src: dict):
    from datasets import load_dataset
    kw = {"split": src["split"], "streaming": True}
    return iter(load_dataset(src["hf_id"], src["config"], **kw) if src["config"]
                else load_dataset(src["hf_id"], **kw))


def url_filter(src: dict):
    st = src.get("status", "")
    for key, pat in (("SCIENCE_URL", ds.SCIENCE_URL), ("KNOWLEDGE_URL", ds.KNOWLEDGE_URL),
                     ("FINANCE_URL", ds.FINANCE_URL)):
        if key in st:
            return re.compile(pat)
    return None


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_pool(bucket: str, budget: int, R: Renderer, rep: dict, allow_unverified: bool,
               max_read: int | None = None) -> list[dict]:
    long_pool = bucket == "longctx"
    srcs = [s for s in pool_sources(bucket) if ds.usable(s, allow_unverified)[0]]
    for s in pool_sources(bucket):
        ok, why = ds.usable(s, allow_unverified)
        if not ok:
            log(f"{bucket}: exclude {s['hf_id']} ({why})")
    wsum = sum(s["weight"] for s in srcs)
    rows: list[dict] = []
    cur = None
    deficit = 0.0
    for si, s in enumerate(srcs):
        name = s["hf_id"] + (f"/{s['config']}" if s["config"] else "") + f":{s['split']}"
        w_left = sum(x["weight"] for x in srcs[si:])
        quota = int(budget * s["weight"] / wsum + deficit * s["weight"] / w_left)
        deficit -= deficit * s["weight"] / w_left
        rec = rep.setdefault(name, {"bucket": bucket, "licence": s["licence"], "generator": s["generator"],
                                    "read": 0, "kept": 0, "tokens": 0})
        flt, got, t0 = url_filter(s), 0, time.time()
        try:
            for row in stream(s):
                if got >= quota or (max_read and rec["read"] >= max_read):
                    break
                rec["read"] += 1
                if flt is not None and not flt.search(str(row.get("url", ""))):
                    continue
                try:
                    out = R.render(s["fn"](row))
                except Exception:
                    out = None
                if not out:
                    if rec["read"] > 2000 and rec["kept"] == 0:
                        break
                    continue
                ids, elig = out
                if long_pool:
                    if len(ids) < LONG_MIN:
                        continue
                    n = min(LONG_MAX, len(ids), max(LONG_MIN, quota - got))
                    n -= n % 32                                   # block-aligned for block routing
                    rows.append({"tokens": ids[:n], "elig": elig[:n], "docs": [name]})
                    got += n
                else:
                    ids, elig = ids[: min(2 * SEQ_LEN, quota - got)], elig[: min(2 * SEQ_LEN, quota - got)]
                    pos = 0
                    while pos < len(ids):
                        if cur is None or len(cur["tokens"]) == SEQ_LEN:
                            cur = {"tokens": [], "elig": [], "docs": []}
                            rows.append(cur)
                        k = min(SEQ_LEN - len(cur["tokens"]), len(ids) - pos)
                        cur["tokens"] += ids[pos:pos + k]
                        cur["elig"] += elig[pos:pos + k]
                        cur["docs"].append(name)
                        pos += k
                        got += k
                rec["kept"] += 1
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        rec["tokens"] += got
        rec["seconds"] = round(time.time() - t0, 1)
        deficit += max(0, quota - got)
        log(f"{bucket}: {name} +{got:,} tok (quota {quota:,}) kept {rec['kept']}/{rec['read']}"
            + (f" ERROR {rec['error']}" if "error" in rec else ""))
    if not long_pool:
        if rows and len(rows[-1]["tokens"]) < SEQ_LEN:
            rows.pop()
        rows = rows[: max(1, budget // SEQ_LEN)]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probe", action="store_true", help="tiny build (few rows per source) to check yields")
    ap.add_argument("--total", type=int, default=TOTAL_TOKENS)
    ap.add_argument("--allow-unverified", action="store_true")
    a = ap.parse_args()

    budgets = {b: int(a.total * TOKEN_TARGET[b]) for b in BUCKETS}
    if a.dry_run:
        for b in BUCKETS:
            print(f"{b:12s} {budgets[b]:>9,} tok")
            for s in pool_sources(b):
                print(f"    {'OK ' if ds.usable(s, a.allow_unverified)[0] else 'NO '} {s['weight']:.2f} {s['hf_id']}"
                      f" [{s['licence']}; {s['generator']}]")
        return

    out = OUT if not a.probe else WORK / "corpus_probe"
    out.mkdir(parents=True, exist_ok=True)
    R = Renderer()
    rep: dict = {}
    pools = {}
    for b in BUCKETS:
        bud = budgets[b] if not a.probe else (LONG_MAX if b == "longctx" else 2 * SEQ_LEN)
        pools[b] = build_pool(b, bud, R, rep, a.allow_unverified, max_read=50 if a.probe else None)
        log(f"{b}: {len(pools[b])} rows, {sum(len(r['tokens']) for r in pools[b]):,} tok")

    # held-out = last 10% of each pool's rows; rows interleaved smoothly so partial passes see the mix
    counts = {b: len(pools[b]) for b in BUCKETS if pools[b]}
    order = ds.interleave(counts) if counts else []
    nxt = {b: 0 for b in counts}
    toks, elig, recs, off = [], [], [], 0
    for i, b in enumerate(order):
        r = pools[b][nxt[b]]
        held = nxt[b] >= int(round(len(pools[b]) * (1 - HELDOUT_FRACTION)))
        nxt[b] += 1
        toks.append(np.asarray(r["tokens"], np.int32))
        elig.append(np.asarray(r["elig"], np.uint8))
        recs.append({"row": i, "bucket": b, "offset": off, "length": len(r["tokens"]), "heldout": held,
                     "docs": sorted(set(r["docs"]))})
        off += len(r["tokens"])
    np.save(out / "tokens.npy", np.concatenate(toks) if toks else np.zeros(0, np.int32))
    np.save(out / "elig.npy", np.concatenate(elig) if elig else np.zeros(0, np.uint8))
    with open(out / "rows.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    json.dump(rep, open(out / "sources.json", "w"), indent=1)
    achieved = {b: sum(r["length"] for r in recs if r["bucket"] == b) / max(1, off) for b in BUCKETS}
    json.dump({"tokens": off, "rows": len(recs), "achieved": achieved, "target": TOKEN_TARGET,
               "seq_len": SEQ_LEN, "elig_frac": float(np.concatenate(elig).mean()) if elig else 0.0},
              open(out / "manifest.json", "w"), indent=1)
    log(f"corpus done: {len(recs)} rows, {off:,} tokens, mix " +
        ", ".join(f"{b} {v:.3f}" for b, v in achieved.items()))


if __name__ == "__main__":
    main()
