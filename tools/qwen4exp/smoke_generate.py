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
    "Question: What is 2 + 2?\nAnswer:",
]
# Content sanity: a deterministic, finite model can still be broken (wrong head order, missing
# norm offset). Each listed prompt must produce at least one of its words (case-insensitive).
EXPECT = {
    "The capital of France is": ["paris"],
    "def fibonacci(n):": ["return", "fib"],
    "Question: What is 2 + 2?\nAnswer:": ["4", "four"],
}
# Teacher-forced fluency probe: mean NLL (nats/token) of this plain paragraph via echo+logprobs.
# Written for this check (no third-party text). A broken model scores far worse than a healthy
# one; arms are compared against each other (--nll-ref), never against a fixed number.
NLL_TEXT = (
    "Every morning the baker opens the small shop on the corner before the sun comes up. "
    "She lights the ovens, mixes flour, water and salt, and shapes the loaves by hand. "
    "By seven o'clock the first customers arrive, and the smell of fresh bread fills the street. "
    "Most of them buy the same thing every day, and she knows many of them by name."
)


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
    ap.add_argument("--no-content", action="store_true",
                    help="random-weight smoke (--load-format dummy, synthetic GGUF): skip the content check")
    ap.add_argument("--nll-ref", default=None, help="smoke.json of a reference arm: report the NLL delta")
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
        want = EXPECT.get(prompt)
        content_ok = True if (want is None or a.no_content) else any(w in outs[0]["text"].lower() for w in want)
        ok = same and all(o["finite"] for o in outs) and outs[0]["completion_tokens"] > 0 and content_ok
        res["pass"] &= ok
        res["runs"].append({"prompt": prompt, "deterministic": same, "content_ok": content_ok, "pass": ok,
                            "outputs": outs})
        print(("PASS" if ok else "FAIL"), repr(prompt), repr(outs[0]["text"][:80]), outs[0]["tok_s"], "tok/s", flush=True)
    # fluency probe (reported, compared arm to arm; it never gates by itself)
    try:
        out, _ = post(base + "/v1/completions", {"model": model, "prompt": NLL_TEXT, "max_tokens": 1,
                                                 "temperature": 0.0, "echo": True, "logprobs": 1})
        lps = [lp for lp in (out["choices"][0].get("logprobs") or {}).get("token_logprobs") or []
               if lp is not None]
        res["nll_text_tokens"] = len(lps)
        res["nll_nats_per_token"] = round(-sum(lps) / len(lps), 4) if lps else None
    except Exception as e:  # noqa: BLE001 - servers without prompt logprobs still pass the gate
        res["nll_error"] = repr(e)[:200]
    if a.nll_ref and res.get("nll_nats_per_token") is not None:
        ref = json.load(open(a.nll_ref)).get("nll_nats_per_token")
        if ref is not None:
            res["nll_delta_vs_ref"] = round(res["nll_nats_per_token"] - ref, 4)
    print("NLL", res.get("nll_nats_per_token"), "delta", res.get("nll_delta_vs_ref"), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("RESULT", "PASS" if res["pass"] else "FAIL")
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
