#!/usr/bin/env python3
"""Probe prompts for the context-path investigation: where the user's question lands relative to committed blocks."""
import struct, sys
from pathlib import Path
sys.argv = [sys.argv[0]]
exec(open(Path(__file__).parent / "make_prompts.py").read().split("prompts = {")[0])
sysmsg = ("You are a helpful assistant for a small engineering team. Answer clearly and concisely, prefer concrete "
          "examples, and say when you are unsure. Keep answers under two hundred words unless asked for more detail.")
q = "Explain the difference between TCP and UDP for a new engineer, with one example use case each."
probes = {
    "p_sys_chat": [{"role": "system", "content": sysmsg}, {"role": "user", "content": q}],
    "p_sys2_chat": [{"role": "system", "content": sysmsg + " " + sysmsg}, {"role": "user", "content": q}],
    "p_agent_nosys": [{"role": "user", "content": "Find which Python files in src/ import requests and count them."}],
}
for name, msgs in probes.items():
    ids = tok.encode(render(msgs, tools if "agent" in name else None), add_special_tokens=False).ids
    (OUT / f"{name}.bin").write_bytes(struct.pack(f"{len(ids)}i", *ids))
    print(name, len(ids), "-> committed blocks", len(ids) // 32, "+", len(ids) % 32)
