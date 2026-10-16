#!/usr/bin/env python3
"""Smoke check against a running vLLM OpenAI server: greedy completions twice, assert determinism + finite logprobs.

Writes a JSON with PASS/FAIL, generated text, token ids, per-request latency and decode tok/s (speed is
measurable from the first token on). Usage: smoke_generate.py --port 8011 --out res.json [--model NAME]
"""
import argparse
import json
import math
import sys
import time
import urllib.request

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Explain in one sentence why the sky is blue.",
]


def post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=32)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    models = json.loads(urllib.request.urlopen(base + "/v1/models", timeout=30).read())
    model = models["data"][0]["id"]
    res = {"model": model, "runs": [], "pass": True}
    for prompt in PROMPTS:
        outs = []
        for rep in range(2):
            body = {"model": model, "prompt": prompt, "max_tokens": a.max_tokens, "temperature": 0.0,
                    "logprobs": 1, "ignore_eos": True}
            out, dt = post(base + "/v1/completions", body)
            ch = out["choices"][0]
            lps = (ch.get("logprobs") or {}).get("token_logprobs") or []
            finite = all(lp is not None and math.isfinite(lp) for lp in lps)
            n = (out.get("usage") or {}).get("completion_tokens", len(lps))
            outs.append({"text": ch["text"], "tokens": (ch.get("logprobs") or {}).get("tokens"), "finite": finite,
                         "latency_s": round(dt, 3), "completion_tokens": n,
                         "tok_s": round(n / dt, 2) if dt > 0 else None})
        same = outs[0]["text"] == outs[1]["text"]
        ok = same and all(o["finite"] for o in outs) and outs[0]["completion_tokens"] > 0
        res["pass"] &= ok
        res["runs"].append({"prompt": prompt, "deterministic": same, "pass": ok, "outputs": outs})
        print(("PASS" if ok else "FAIL"), repr(prompt), repr(outs[0]["text"][:80]), outs[0]["tok_s"], "tok/s", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("RESULT", "PASS" if res["pass"] else "FAIL")
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
