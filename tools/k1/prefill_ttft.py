#!/usr/bin/env python3
"""K1 cold-prefill timing on :8001 (cache-proof: random salt at the start of every prompt), max_tokens=1.

Reports prompt tokens, TTFT and tok/s per (length, rep). Prompts are natural text (evalkit corpus) so the token
count per word is realistic. usage: prefill_ttft.py --lengths 30000,64000,128000 --reps 3 --out file.json
"""

from __future__ import annotations

import argparse
import http.client
import json
import random
import time

CORPUS = "/home/kevin/Desktop/qwen38-evalkit/corpus/corpus_128000tok.txt"


def prompt(tokens, rng):
    words = open(CORPUS).read().split()
    n = int(tokens / 3.64)
    out = []
    while len(out) < n:
        s = rng.randint(0, len(words) - 1)
        out += words[s:s + min(n - len(out), 4000)]
    salt = " ".join(f"{rng.getrandbits(40):x}" for _ in range(16))
    return f"[{salt}]\n" + " ".join(out) + "\n\nSummarise the text above in one word."


def ttft(p):
    c = http.client.HTTPConnection("127.0.0.1", 8001, timeout=7200)
    body = dict(model="qwen-local", messages=[{"role": "user", "content": p}], max_tokens=1, temperature=0,
                chat_template_kwargs={"enable_thinking": False})
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json", "X-Client": "bench-k1"})
    d = json.loads(c.getresponse().read())
    dt = time.time() - t0
    return d["usage"]["prompt_tokens"], dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="30000,64000,128000")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--arm", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = random.Random(time.time_ns())
    rows = []
    ttft(prompt(2000, rng))  # warm
    for L in (int(x) for x in a.lengths.split(",")):
        for r in range(a.reps):
            n, dt = ttft(prompt(L, rng))
            row = dict(arm=a.arm, target=L, prompt_tokens=n, ttft_s=round(dt, 2), tok_s=round(n / dt, 1))
            rows.append(row)
            print(json.dumps(row), flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
