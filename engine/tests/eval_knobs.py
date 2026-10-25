#!/usr/bin/env python3
"""Speed/quality of the decode knobs on automatically scored tasks, through the real serving path (server.Model ->
libllada.so). Tasks are generated here, not taken from a corpus:
  math  two-step arithmetic word problems with exact integer answers ("Answer: N")
  code  small Python functions, scored by running hidden asserts in a subprocess (5 s timeout)
  tool  tool-use prompts: a well-formed call to the expected function with its required arguments
Each config is run on the same tasks; reports accuracy per kind and aggregate decode tok/s.

  python tests/eval_knobs.py [--configs 32:0.5,16:0.5,8:0.5,32:0.3] [--n-math 24]
"""
import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))
import server  # noqa: E402


def math_tasks(n, seed=7):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        k = i % 4
        if k == 0:
            a, b = rng.randint(2, 9), rng.randint(3, 12)
            c = (a * b // 10 + 1) * 10 + rng.choice([0, 10, 20])
            q = f"A shop sells pencils at ${a} each. Tom buys {b} pencils and pays with a ${c} bill. How much change does he get?"
            ans = c - a * b
        elif k == 1:
            a, b, c = rng.randint(20, 90), rng.randint(2, 6), rng.randint(5, 30)
            q = f"A bus carries {a} passengers. At the first stop, the number of passengers is multiplied by {b} as a group boards, then {c} people get off. How many passengers are on the bus?"
            ans = a * b - c
        elif k == 2:
            a, b = rng.randint(3, 12), rng.randint(2, 9)
            c = rng.randint(1, a * b - 1)
            q = f"Each box holds {a} apples. There are {b} full boxes and {c} apples are eaten. How many apples are left?"
            ans = a * b - c
        else:
            a, b, c = rng.randint(40, 120), rng.randint(2, 5), rng.randint(3, 9)
            q = f"A runner runs {a} meters per minute for {b} minutes, then {c} meters more. How many meters did the runner cover in total?"
            ans = a * b + c
        out.append({"kind": "math", "prompt": q + " Think briefly, then end with a line 'Answer: <number>'.", "ans": ans})
    return out


CODE = [
    ("digit_sum", "takes a non-negative integer n and returns the sum of its decimal digits",
     ["assert digit_sum(0) == 0", "assert digit_sum(12345) == 15", "assert digit_sum(909) == 18"]),
    ("reverse_words", "takes a string and returns the words in reverse order, joined by single spaces",
     ["assert reverse_words('a b c') == 'c b a'", "assert reverse_words('hello world') == 'world hello'"]),
    ("count_vowels", "returns the number of vowels (a, e, i, o, u, case-insensitive) in a string",
     ["assert count_vowels('Hello') == 2", "assert count_vowels('xyz') == 0", "assert count_vowels('AEIOU') == 5"]),
    ("is_prime", "returns True if the integer n is prime and False otherwise",
     ["assert is_prime(2) and is_prime(97)", "assert not is_prime(1) and not is_prime(91)"]),
    ("gcd", "returns the greatest common divisor of two positive integers a and b without using math.gcd",
     ["assert gcd(12, 18) == 6", "assert gcd(17, 5) == 1", "assert gcd(100, 75) == 25"]),
    ("flatten", "takes a list that may contain nested lists to any depth and returns a flat list of the non-list items in order",
     ["assert flatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]", "assert flatten([]) == []"]),
    ("run_length", "returns the run-length encoding of a string as a list of (char, count) tuples",
     ["assert run_length('aaabcc') == [('a', 3), ('b', 1), ('c', 2)]", "assert run_length('') == []"]),
    ("second_largest", "returns the second largest distinct value in a list of integers, or None if there is none",
     ["assert second_largest([3, 1, 4, 4]) == 3", "assert second_largest([5, 5]) is None"]),
    ("is_palindrome", "returns True if a string is a palindrome ignoring case and non-alphanumeric characters",
     ["assert is_palindrome('A man, a plan, a canal: Panama')", "assert not is_palindrome('hello')"]),
    ("merge_sorted", "merges two sorted lists of integers into one sorted list without using sort",
     ["assert merge_sorted([1, 4, 9], [2, 3, 10]) == [1, 2, 3, 4, 9, 10]", "assert merge_sorted([], [1]) == [1]"]),
    ("word_freq", "returns a dict mapping each lowercase word in a string to its count, splitting on whitespace",
     ["assert word_freq('a B a') == {'a': 2, 'b': 1}", "assert word_freq('') == {}"]),
    ("binary_search", "returns the index of target in a sorted list, or -1 if absent",
     ["assert binary_search([1, 3, 5, 7], 5) == 2", "assert binary_search([1, 3], 4) == -1"]),
]


def code_tasks():
    return [{"kind": "code", "name": n,
             "prompt": f"Write a Python function `{n}` that {d}. Reply with only the code in one ```python block.",
             "tests": t} for n, d, t in CODE]


TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file from the workspace.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "run_shell", "description": "Run a shell command and return stdout.",
     "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}},
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "send_email", "description": "Send an email.",
     "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"},
                    "body": {"type": "string"}}, "required": ["to", "subject", "body"]}}},
]
TOOL_TASKS = [
    ("Show me the contents of config/settings.yaml.", "read_file", ["path"]),
    ("How much free disk space is there? Use the shell.", "run_shell", ["cmd"]),
    ("What's the weather like in Lisbon right now?", "get_weather", ["city"]),
    ("Email bob@example.com with subject 'Standup' saying I will be 10 minutes late.", "send_email", ["to", "subject", "body"]),
    ("List the files in the current directory.", "run_shell", ["cmd"]),
    ("Open README.md and tell me what it says.", "read_file", ["path"]),
    ("Is it raining in Tokyo?", "get_weather", ["city"]),
    ("Count the lines of Python code under src/ using a shell command.", "run_shell", ["cmd"]),
]


def tool_tasks():
    return [{"kind": "tool", "prompt": p, "fn": f, "req": r} for p, f, r in TOOL_TASKS]


def score(m, task, ids):
    content, calls = server.split_output(m, ids)
    if task["kind"] == "math":
        g = re.findall(r"Answer:\s*\$?(-?[\d,]+)", content)
        return bool(g) and int(g[-1].replace(",", "")) == task["ans"]
    if task["kind"] == "code":
        blk = re.findall(r"```(?:python)?\n(.*?)```", content, re.S)
        if not blk:
            return False
        src = blk[0] + "\n" + "\n".join(task["tests"]) + "\nprint('OK')\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(src)
        try:
            r = subprocess.run([sys.executable, "-I", f.name], capture_output=True, text=True, timeout=5)
            return r.stdout.strip().endswith("OK")
        except subprocess.TimeoutExpired:
            return False
    ok = False
    for c in calls:
        try:
            args = json.loads(c["function"]["arguments"])
        except Exception:                                                   # noqa: BLE001
            continue
        ok = ok or (c["function"]["name"] == task["fn"] and all(k in args and args[k] for k in task["req"]))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="32:0.5,16:0.5,8:0.5,32:0.3")
    ap.add_argument("--n-math", type=int, default=24)
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--show-fail", action="store_true", help="print the prompt and output of every failed task")
    a = ap.parse_args()
    m = server.Model(Path.home() / "models/LLaDA2.2-mini", Path.home() / "models/llada2mini-orin",
                     ROOT / "engine" / "libllada.so", a.max_ctx)
    tasks = math_tasks(a.n_math) + code_tasks() + tool_tasks()
    print(f"{len(tasks)} tasks", flush=True)
    for cfg in a.configs.split(","):
        steps, thr = cfg.split(":")
        opts = {"threshold": float(thr), "editing_threshold": 0.0, "max_post_steps": 16, "steps": int(steps)}
        res, ntok, secs = {}, 0, 0.0
        for t in tasks:
            msgs = [{"role": "user", "content": t["prompt"]}]
            prompt = m.encode(m.render(msgs, TOOLS if t["kind"] == "tool" else None))
            ids, dt = m.generate(prompt, 768, opts)
            ntok += len(ids); secs += dt
            ok = score(m, t, ids)
            res.setdefault(t["kind"], []).append(ok)
            if a.show_fail and not ok:
                print(f"--- FAIL ({t['kind']}): {t['prompt']}\n{m.tok.decode(ids)}\n---", flush=True)
        summary = "  ".join(f"{k} {sum(v)}/{len(v)}" for k, v in res.items())
        print(f"steps {steps:>2} threshold {thr}: {summary}  | {ntok} tokens in {secs:.1f}s = {ntok / secs:.1f} tok/s",
              flush=True)


if __name__ == "__main__":
    main()
