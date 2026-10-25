#!/usr/bin/env python3
"""Benchmark prompts for bench_gen, rendered with the model's chat template (same path as server.py) and written as
int32 token files: tests/prompts/<name>.bin."""
import json, struct, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "server"))
import jinja2
from tokenizers import Tokenizer
import importlib.util

M = Path.home() / "models/LLaDA2.2-mini"
OUT = Path(__file__).resolve().parent / "prompts"
tok = Tokenizer.from_file(str(M / "tokenizer.json"))
env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None: json.dumps(x, ensure_ascii=ensure_ascii, indent=indent)
tpl = env.from_string((M / "chat_template.jinja").read_text())
spec = importlib.util.spec_from_file_location("tts", M / "tool_declaration_ts.py")
tts = importlib.util.module_from_spec(spec); spec.loader.exec_module(tts)


def render(messages, tools=None):
    kw = {"tools_ts_str": tts.encode_tools_to_typescript_style(tools)} if tools else {}
    return tpl.render(messages=messages, tools=tools, add_generation_prompt=True, **kw)


tools = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file from the workspace.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "run_shell", "description": "Run a shell command and return stdout.",
     "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}, "timeout_s": {"type": "integer"}},
                    "required": ["cmd"]}}},
]
src = (M / "modeling_llada2_moe.py").read_text()
prompts = {
    "code": [{"role": "user", "content": "Write a Python function that parses an ISO-8601 date string without using "
              "datetime, with validation, and add three unit tests."}],
    "agent": [{"role": "system", "content": "You are a coding agent. Use the tools to answer."},
              {"role": "user", "content": "Find which Python files in src/ import requests and count them."}],
    "chat": [{"role": "user", "content": "Explain the difference between TCP and UDP for a new engineer, with one "
              "example use case each."}],
    "math": [{"role": "user", "content": "A tank fills at 3 liters per minute and drains at 1.2 liters per minute. "
              "Starting empty, how long until it holds 45 liters? Show the steps."}],
    "longctx": [{"role": "user", "content": "Here is a source file:\n\n```python\n" + src[:12000] +
                 "\n```\n\nSummarize what the decoder layer does, in five bullet points."}],
}
OUT.mkdir(exist_ok=True)
for name, msgs in prompts.items():
    ids = tok.encode(render(msgs, tools if name == "agent" else None), add_special_tokens=False).ids
    (OUT / f"{name}.bin").write_bytes(struct.pack(f"{len(ids)}i", *ids))
    print(name, len(ids))
