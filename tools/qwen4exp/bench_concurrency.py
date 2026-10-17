#!/usr/bin/env python3
"""Concurrency ramp against a running vLLM OpenAI server: find the measured breaking point, no fixed ceiling.

Starting at --start streams, the concurrency doubles while aggregate decode throughput keeps rising by more than
the run-to-run spread (--reps runs per point) and the engine keeps serving (no errors, no timeouts); then it bisects
between the last rising point and the first non-rising / failing point. Every measured point is reported (the whole
curve), with the per-stream SLO marked: decode >= --slo-tok-s tok/s per stream and TTFT <= --slo-ttft-s
(--slo-ttft-long-s for prompts of 28K+ tokens).

Each stream sends the same-shaped request (varied topic, --prompt-tokens of filler, --tokens output, ignore_eos,
greedy); all streams of a point start together. By default the filler is the SAME text for every stream, i.e. one
shared prefix (agent lanes sharing a system prompt: the prefix cache serves it once). --distinct-context gives every
stream of every rep its own filler from the first token (no prefix sharing at all: each lane pays its full context in
KV), which is the memory-wall case; run both to bracket real traffic.
Usage: bench_concurrency.py --port 8011 --out res.json [--start 1] [--tokens 256] [--reps 3] [--prompt-tokens 0]
"""
import argparse
import concurrent.futures as cf
import json
import random
import statistics
import sys
import time
import urllib.request

TOPICS = ["rust borrow checker", "the French revolution", "photosynthesis", "TCP congestion control",
          "the Riemann hypothesis", "sourdough baking", "Byzantine fault tolerance", "jazz harmony",
          "plate tectonics", "compiler register allocation", "the Silk Road", "CRISPR", "black holes",
          "double-entry bookkeeping", "transformer attention", "medieval castles"]
FILLER = ("The quick brown fox jumps over the lazy dog while the committee reviews the quarterly report. ")
WORDS = ("amber basalt cedar delta ember fjord granite harbor iris juniper kelp lagoon mesa nectar orchid prairie "
         "quartz reef sierra tundra umber valley willow xenon yarrow zephyr anchor beacon canyon dune estuary glacier "
         "heron inlet jetty knoll ledge marsh oasis pinnacle quarry ridge summit thicket upland vista wharf").split()


def _filler(a, i):
    """--prompt-tokens of filler; shared by default, unique per (rep, stream) with --distinct-context."""
    if not a.prompt_tokens:
        return ""
    if not a.distinct_context:
        return FILLER * (a.prompt_tokens // 20)
    rng = random.Random(f"{a.nonce}-{getattr(a, 'rep', 0)}-{i}")
    # ~1.3 tokens per word for this vocabulary; the session tag leads so no two streams share a first block
    words = [rng.choice(WORDS) for _ in range(int(a.prompt_tokens / 1.3))]
    return f"[session {a.nonce}-{getattr(a, 'rep', 0)}-{i}] " + " ".join(words) + "."


def stream_one(base, model, i, a):
    pad = _filler(a, i)
    body = {"model": model, "prompt": f"{pad}\nWrite a detailed technical essay about {TOPICS[i % len(TOPICS)]} "
            f"(variant {i}).\n\n", "max_tokens": a.tokens, "temperature": 0.0, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    usage_tokens = None
    with urllib.request.urlopen(req, timeout=a.request_timeout_s) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            msg = json.loads(line[5:])
            if msg.get("usage"):
                usage_tokens = msg["usage"].get("completion_tokens")
            if not msg.get("choices"):
                continue
            if msg["choices"][0].get("text"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks += 1
    # The server's own completion count, not the SSE chunk count. A busy API server coalesces several tokens into
    # one chunk (W2c fp16-state ramp: every C5+ point read as "short output" at 64 free lanes, no preemption).
    n = usage_tokens if usage_tokens is not None else chunks
    return {"ttft": ttft, "tokens": n, "chunks": chunks, "decode_s": time.perf_counter() - t0 - (ttft or 0)}


def measure(base, model, c, a):
    reps, err = [], None
    # One unmeasured rep at this concurrency first. A new batch shape JIT-compiles its Triton kernels, which made
    # W1 v10's first C4 rep run at 27 tok/s against 43-48 for the next two. That 21 tok/s spread exceeded the
    # 20 tok/s gain over C2, so a linear curve read as "not rising" and the ramp stopped at C3.
    warm_s = None
    if a.warmup_tokens > 0:
        wa = argparse.Namespace(**{**vars(a), "tokens": a.warmup_tokens, "rep": f"c{c}warm"})
        t0 = time.perf_counter()
        try:
            with cf.ThreadPoolExecutor(c) as ex:
                list(ex.map(lambda i: stream_one(base, model, i, wa), range(c)))
        except Exception as e:  # noqa: BLE001 - failing even the warm-up is the breaking point
            return {"conc": c, "error": "warm-up: " + repr(e)[:280], "reps": []}
        warm_s = time.perf_counter() - t0
    for rep_i in range(a.reps):
        a.rep = f"c{c}r{rep_i}"  # distinct-context fillers never repeat across points or reps
        t0 = time.perf_counter()
        try:
            with cf.ThreadPoolExecutor(c) as ex:
                outs = list(ex.map(lambda i: stream_one(base, model, i, a), range(c)))
        except Exception as e:  # noqa: BLE001 - any failure is the breaking point
            err = repr(e)[:300]
            break
        wall = time.perf_counter() - t0
        if any(o["tokens"] < a.tokens for o in outs):
            err = "short output (preemption or abort)"
        reps.append({"agg_tok_s": sum(o["tokens"] for o in outs) / wall,
                     "per_stream_tok_s": statistics.median(o["tokens"] / o["decode_s"] for o in outs if o["decode_s"] > 0),
                     "ttft_p50": statistics.median(o["ttft"] or 0 for o in outs),
                     "ttft_max": max(o["ttft"] or 0 for o in outs), "wall_s": wall})
        if err:
            break
    row = {"conc": c, "error": err, "reps": reps, "warmup_s": warm_s}
    if reps:
        agg = [r["agg_tok_s"] for r in reps]
        row.update(agg_tok_s=statistics.median(agg), agg_spread=max(agg) - min(agg),
                   per_stream_tok_s=statistics.median(r["per_stream_tok_s"] for r in reps),
                   ttft_p50=statistics.median(r["ttft_p50"] for r in reps),
                   ttft_max=max(r["ttft_max"] for r in reps))
        slo_ttft = a.slo_ttft_long_s if a.prompt_tokens >= 28000 else a.slo_ttft_s
        row["slo_ok"] = row["per_stream_tok_s"] >= a.slo_tok_s and row["ttft_max"] <= slo_ttft
    return row


def rises(prev, cur):
    """cur is a real gain over prev: above both points' run-to-run spread."""
    if cur.get("error") or "agg_tok_s" not in cur:
        return False
    noise = max(prev.get("agg_spread", 0.0), cur.get("agg_spread", 0.0))
    return cur["agg_tok_s"] - prev["agg_tok_s"] > noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--prompt-tokens", type=int, default=0)
    ap.add_argument("--distinct-context", action="store_true",
                    help="unique filler per stream and rep (no prefix sharing): the KV memory-wall case")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup-tokens", type=int, default=16,
                    help="output tokens of the unmeasured warm-up rep at each concurrency (0 = none)")
    ap.add_argument("--request-timeout-s", type=float, default=900)
    ap.add_argument("--slo-tok-s", type=float, default=30.0)
    ap.add_argument("--slo-ttft-s", type=float, default=2.0)
    ap.add_argument("--slo-ttft-long-s", type=float, default=8.0)
    ap.add_argument("--points", default="", help="explicit comma list instead of the ramp (e.g. for a latency map)")
    a = ap.parse_args()
    a.nonce = f"{time.time_ns():x}"
    base = f"http://127.0.0.1:{a.port}"
    model = json.loads(urllib.request.urlopen(base + "/v1/models", timeout=30).read())["data"][0]["id"]
    stream_one(base, model, 0, argparse.Namespace(**{**vars(a), "tokens": 16}))  # warm-up (graphs, Triton JIT)
    res = {"model": model, "args": vars(a), "curve": []}
    done: dict[int, dict] = {}

    def point(c):
        if c not in done:
            done[c] = measure(base, model, c, a)
            res["curve"] = [done[k] for k in sorted(done)]
            print(json.dumps({k: v for k, v in done[c].items() if k != "reps"}), flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
        return done[c]

    if a.points:
        for c in [int(x) for x in a.points.split(",")]:
            point(c)
    else:
        good = point(a.start)
        if not good.get("error"):
            c = a.start
            while True:  # double while it still rises
                nxt = point(c * 2)
                if not rises(done[c], nxt):
                    lo, hi = c, c * 2
                    break
                c *= 2
            while hi - lo > 1:  # bisect between the last rising point and the first non-rising one
                mid = (lo + hi) // 2
                if rises(done[lo], point(mid)):
                    lo = mid
                else:
                    hi = mid
            res["breaking_point"] = {"best_conc": lo, "first_non_rising_conc": hi,
                                     "best_agg_tok_s": done[lo].get("agg_tok_s")}
    slo = [r["conc"] for r in res["curve"] if r.get("slo_ok")]
    res["slo_max_conc"] = max(slo) if slo else None
    json.dump(res, open(a.out, "w"), indent=1)
    print("BREAKING", json.dumps(res.get("breaking_point")), "SLO_MAX", res["slo_max_conc"])
    ok = bool(res["curve"]) and not res["curve"][0].get("error")
    print("RESULT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
