#!/usr/bin/env python3
"""Decode-throughput sweep against a running vLLM OpenAI server: C1/C2/C4/C8/C16 streams, TTFT, 3 reps each.

Every stream sends the same-shaped request (prompt ~PROMPT_TOKENS of varied text, max_tokens N, ignore_eos, greedy),
all streams start together; per concurrency we report aggregate decode tok/s, per-stream tok/s, TTFT p50/max,
the spread over reps, and errors. Stops early at the first concurrency that errors or exceeds --max-step-s.
Usage: bench_concurrency.py --port 8011 --out res.json [--conc 1,2,4,8,16] [--tokens 256] [--reps 3]
"""
import argparse
import concurrent.futures as cf
import json
import statistics
import sys
import time
import urllib.request

TOPICS = ["rust borrow checker", "the French revolution", "photosynthesis", "TCP congestion control",
          "the Riemann hypothesis", "sourdough baking", "Byzantine fault tolerance", "jazz harmony",
          "plate tectonics", "compiler register allocation", "the Silk Road", "CRISPR", "black holes",
          "double-entry bookkeeping", "transformer attention", "medieval castles"]


def stream_one(base, model, i, tokens, timeout):
    body = {"model": model, "prompt": f"Write a detailed technical essay about {TOPICS[i % len(TOPICS)]}.\n\n",
            "max_tokens": tokens, "temperature": 0.0, "ignore_eos": True, "stream": True}
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    n = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            ch = json.loads(line[5:])["choices"][0]
            if ch.get("text"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n += 1
    t1 = time.perf_counter()
    return {"ttft": ttft, "tokens": n, "decode_s": t1 - t0 - (ttft or 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conc", default="1,2,4,8,16")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-step-s", type=float, default=300)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    model = json.loads(urllib.request.urlopen(base + "/v1/models", timeout=30).read())["data"][0]["id"]
    stream_one(base, model, 0, 16, 600)  # warm-up (graphs, Triton JIT)
    res = {"model": model, "tokens": a.tokens, "rows": []}
    for c in [int(x) for x in a.conc.split(",")]:
        reps = []
        err = None
        for _ in range(a.reps):
            t0 = time.perf_counter()
            try:
                with cf.ThreadPoolExecutor(c) as ex:
                    outs = list(ex.map(lambda i: stream_one(base, model, i, a.tokens, a.max_step_s), range(c)))
            except Exception as e:  # noqa: BLE001
                err = repr(e)
                break
            wall = time.perf_counter() - t0
            toks = sum(o["tokens"] for o in outs)
            reps.append({"agg_tok_s": toks / wall, "per_stream_tok_s": statistics.mean(
                o["tokens"] / o["decode_s"] for o in outs if o["decode_s"] > 0),
                "ttft_p50": statistics.median(o["ttft"] or 0 for o in outs),
                "ttft_max": max(o["ttft"] or 0 for o in outs), "wall_s": wall})
        row = {"conc": c, "error": err, "reps": reps}
        if reps:
            agg = [r["agg_tok_s"] for r in reps]
            row.update(agg_tok_s=statistics.median(agg), agg_spread=max(agg) - min(agg),
                       per_stream_tok_s=statistics.median(r["per_stream_tok_s"] for r in reps),
                       ttft_p50=statistics.median(r["ttft_p50"] for r in reps))
        res["rows"].append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "reps"}), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
        if err:
            break
    ok = bool(res["rows"]) and res["rows"][0].get("error") is None
    print("RESULT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
