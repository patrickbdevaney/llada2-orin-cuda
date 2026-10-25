#!/usr/bin/env python3
"""s07c: flatten model.json (s07b) and experts.json (s07a) into engine.idx, a line format the C++ loader reads
without a JSON parser:
  C <key> <value>                                     config scalars
  T <file> <name> <offset> <bytes> <dtype> <d0> [d1]  a tensor in a weight file
  E <layer> <expert> <bits> <gu_off> <d_off>          a routed expert inside experts_L<layer>.bin (bits 0 = pruned)
  F <file> <bytes>                                     file sizes (the loader checks them)
"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "models/llada2mini-orin")
m = json.load(open(d / "model.json"))
x = json.load(open(d / "experts.json"))
lines = [f"C {k} {v}" for k, v in m["config"].items()]
lines += [f"C mask_id {m['mask_id']}", f"C nonexpert_bits {m['nonexpert_bits']}", f"C lm_bits {m['lm_bits']}"]
for f, ents in m["files"].items():
    for name, (off, nb, dt, shape) in ents.items():
        lines.append(f"T {f} {name} {off} {nb} {dt} " + " ".join(str(s) for s in shape))
for li, L in x["layers"].items():
    lines.append(f"F experts_L{li}.bin {L['bytes']}")
    for e, ent in enumerate(L["experts"]):
        lines.append(f"E {li} {e} {ent['bits']} {ent['gu_off']} {ent['d_off']}")
assert x["format"] == "packlib tiles i8", x["format"]
(d / "engine.idx").write_text("\n".join(lines) + "\n")
print(f"engine.idx: {len(lines)} lines")
