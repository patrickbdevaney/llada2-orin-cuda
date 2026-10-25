#!/usr/bin/env python3
"""OpenAI-compatible HTTP front end for the LLaDA2.2-mini Orin engine (engine/libllada.so).

All model compute -- prefill, block diffusion decoding, KV cache, sampling -- is in the CUDA engine. This process
only renders the model's own chat template (jinja2 + the model's tool_declaration_ts.py), tokenizes with the Rust
`tokenizers` library (the model's tokenizer.json, identical to HF), parses tool calls, and speaks HTTP. It never
imports torch, so it stays small next to the engine on the shared 7.4 GiB.

  POST /v1/chat/completions   messages, tools, max_tokens, stream, (llada) threshold / editing_threshold /
                              max_post_steps / steps (per-step unmasking floor = 32 / steps tokens)
  GET  /v1/models, /health, /stats

Run:  python server.py --model ~/models/LLaDA2.2-mini --engine ~/models/llada2mini-orin --port 8000
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jinja2
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
TOOL_RE = re.compile(r"<\|tool_call_begin\|>(.*?)<\|tool_call_argument_begin\|>(.*?)<\|tool_call_end\|>", re.S)
CB = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p)


def deep_sort(o):
    if isinstance(o, dict):
        return {k: deep_sort(v) for k, v in sorted(o.items())}
    if isinstance(o, list):
        return [deep_sort(x) for x in o]
    return o


class Model:
    def __init__(self, model_dir: Path, engine_dir: Path, lib: Path, max_ctx: int):
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

        def raise_exception(msg):
            raise ValueError(msg)

        def tojson(x, ensure_ascii=False, indent=None):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent)

        env.globals["raise_exception"] = raise_exception
        env.filters["tojson"] = tojson
        self.template = env.from_string((model_dir / "chat_template.jinja").read_text())
        spec = importlib.util.spec_from_file_location("tool_declaration_ts", model_dir / "tool_declaration_ts.py")
        self.tools_ts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.tools_ts)
        self.special = {t: self.tok.token_to_id(t) for t in ("<|endoftext|>", "<|role_end|>")}
        self.lib = ctypes.CDLL(str(lib))
        self.lib.llada_init.restype = ctypes.c_void_p
        self.lib.llada_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.llada_generate.restype = ctypes.c_int
        self.lib.llada_generate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
                                            ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int,
                                            ctypes.POINTER(ctypes.c_int), ctypes.c_int, CB, ctypes.c_void_p]
        self.lib.llada_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        t0 = time.time()
        self.h = self.lib.llada_init(str(engine_dir).encode(), max_ctx)
        self.load_s = time.time() - t0
        self.lock = threading.Lock()

    def render(self, messages, tools=None) -> str:
        kw = {}
        tools = deep_sort(tools) if tools else None
        if tools:
            kw["tools_ts_str"] = self.tools_ts.encode_tools_to_typescript_style(tools)
        return self.template.render(messages=messages, tools=tools, add_generation_prompt=True, **kw)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    def stats(self) -> dict:
        s = (ctypes.c_double * 8)()
        self.lib.llada_stats(self.h, s)
        return {"steps": s[0], "blocks": s[1], "commits": s[2], "prefix_blocks_reused_last": s[3],
                "weights_gb": s[4], "kv_gb": s[5], "max_ctx": int(s[6]), "load_s": self.load_s}

    def generate(self, prompt_ids, max_new, opts, on_text=None):
        """-> (ids, timings). on_text(delta) streams decoded text block by block."""
        arr = (ctypes.c_int * len(prompt_ids))(*prompt_ids)
        out = (ctypes.c_int * max(1, max_new + 64))()
        got: list[int] = []
        sent = [""]

        def cb(ptr, n, _user):
            got.extend(ptr[i] for i in range(n))
            if on_text:
                text = self.decode(got)
                if text.endswith("�"):            # incomplete UTF-8 sequence: wait for the next block
                    return
                on_text(text[len(sent[0]):])
                sent[0] = text

        cbf = CB(cb)
        with self.lock:
            t0 = time.time()
            n = self.lib.llada_generate(self.h, arr, len(prompt_ids), max_new, opts["threshold"],
                                        opts["editing_threshold"], opts["max_post_steps"], opts["steps"], out, len(out),
                                        cbf, None)
            dt = time.time() - t0
        return list(out[:n]), dt


def split_output(model: Model, ids: list[int]):
    """Generated ids -> (content, tool_calls) in OpenAI form."""
    text = model.decode(ids)
    for t in ("<|endoftext|>", "<|role_end|>"):
        text = text.split(t)[0] if t == "<|role_end|>" else text.replace(t, "")
    calls = []
    if "<|tool_calls_section_begin|>" in text:
        text, sect = text.split("<|tool_calls_section_begin|>", 1)
        sect = sect.split("<|tool_calls_section_end|>")[0]
        for i, (tid, args) in enumerate(TOOL_RE.findall(sect)):
            name = tid.split("functions.", 1)[-1].rsplit(":", 1)[0]
            calls.append({"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                          "function": {"name": name, "arguments": args.strip()}})
    return text.strip(), calls


class Handler(BaseHTTPRequestHandler):
    model: Model = None
    name = "llada2.2-mini-orin"
    defaults = {"threshold": 0.3, "editing_threshold": 0.0, "max_post_steps": 16, "steps": 16}   # fast preset
    # (model card: threshold 0.5, steps 32 -- see --reference; 44/44 on tests/eval_knobs.py either way)
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/health", "/v1/health"):
            return self._json(200, {"ok": True})
        if self.path == "/v1/models":
            return self._json(200, {"object": "list", "data": [{"id": self.name, "object": "model", "owned_by": "local"}]})
        if self.path == "/stats":
            return self._json(200, self.model.stats())
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            return self._json(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            m = self.model
            prompt = m.encode(m.render(req["messages"], req.get("tools")))
            max_new = int(req.get("max_tokens") or req.get("max_completion_tokens") or 1024)
            d = Handler.defaults
            opts = {"threshold": float(req.get("threshold", d["threshold"])),
                    "editing_threshold": float(req.get("editing_threshold", d["editing_threshold"])),
                    "max_post_steps": int(req.get("max_post_steps", d["max_post_steps"])),
                    "steps": int(req.get("steps", d["steps"]))}
        except Exception as e:                                   # noqa: BLE001
            return self._json(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
        rid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            def send(delta, finish=None):
                ch = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": self.name,
                      "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                self.wfile.write(f"data: {json.dumps(ch)}\n\n".encode())
                self.wfile.flush()

            send({"role": "assistant"})
            state = {"tool": False}

            def on_text(d):
                if state["tool"]:
                    return
                if "<|tool_calls_section_begin|>" in d:          # tool calls are sent whole at the end
                    d = d.split("<|tool_calls_section_begin|>")[0]
                    state["tool"] = True
                d = d.replace("<|endoftext|>", "").split("<|role_end|>")[0]
                if d:
                    send({"content": d})

            ids, dt = m.generate(prompt, max_new, opts, on_text)
            content, calls = split_output(m, ids)
            if calls:
                send({"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]})
            send({}, "tool_calls" if calls else ("stop" if len(ids) < max_new else "length"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
            return
        ids, dt = m.generate(prompt, max_new, opts)
        content, calls = split_output(m, ids)
        msg = {"role": "assistant", "content": content or None}
        if calls:
            msg["tool_calls"] = calls
        self._json(200, {"id": rid, "object": "chat.completion", "created": created, "model": self.name,
                         "choices": [{"index": 0, "message": msg,
                                      "finish_reason": "tool_calls" if calls else ("stop" if len(ids) < max_new else "length")}],
                         "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(ids),
                                   "total_tokens": len(prompt) + len(ids)},
                         "timing": {"generate_s": dt, "tok_per_s": len(ids) / dt if dt else None,
                                    "prefix_blocks_reused": m.stats()["prefix_blocks_reused_last"]}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "models/LLaDA2.2-mini"), help="HF dir (tokenizer, template)")
    ap.add_argument("--engine", default=str(Path.home() / "models/llada2mini-orin"), help="packed engine weights")
    ap.add_argument("--lib", default=str(ROOT / "engine" / "libllada.so"))
    ap.add_argument("--max-ctx", type=int, default=32768)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reference", action="store_true",
                    help="model-card decode preset threshold 0.5, steps 32 (default is the fast preset 0.3 / 16)")
    ap.add_argument("--fast", action="store_true", help="fast preset threshold 0.3, steps 16 (the default; kept for scripts)")
    ap.add_argument("--threshold", type=float, default=None, help="default unmasking threshold (default 0.3)")
    ap.add_argument("--steps", type=int, default=None, help="default per-block step schedule (default 16)")
    a = ap.parse_args()
    if a.reference:
        Handler.defaults.update(threshold=0.5, steps=32)
    if a.threshold is not None:
        Handler.defaults["threshold"] = a.threshold
    if a.steps is not None:
        Handler.defaults["steps"] = a.steps
    Handler.model = Model(Path(a.model), Path(a.engine), Path(a.lib), a.max_ctx)
    print(f"engine loaded in {Handler.model.load_s:.1f}s: {Handler.model.stats()}, decode defaults {Handler.defaults}",
          flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
