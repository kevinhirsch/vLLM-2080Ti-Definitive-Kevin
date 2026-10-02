#!/usr/bin/env python3
"""bench_mixed.py -- prefill-vs-decode contention benchmark against the engine (:8001 direct).

Reproduces the Halo-like shape that matters for the 3584-token step budget: N long-ish decode streams running
while unique ~30K-token prompts prefill (each in 3568-token aligned chunks). Reports, per run:
  - decode stream throughput during the prefill window (tok/s aggregate and per stream), worst inter-token gap
  - long-prompt TTFT, needle accuracy (quality guard: a unique 6-digit code hidden in each 30K prompt)
  - engine spec-decode acceptance delta (from /metrics)
Run on a quiesced engine (gateway drained) for clean numbers. No LLM judge; deterministic scoring.
"""
import argparse, json, random, re, statistics, threading, time, urllib.request, http.client, sys

ENGINE = ("127.0.0.1", 8001)
WORDS = ("amber basalt cedar delta ember fjord garnet harbor indigo jasper kelp lumen marble nectar onyx pewter quartz "
         "russet sable tundra umber velvet willow xenon yarrow zephyr").split()


def para(rng, n_words):
    return " ".join(rng.choice(WORDS) + str(rng.randint(0, 999)) for _ in range(n_words))


def long_prompt(rng, target_tokens, code):
    # ~1.0 token/word-ish for this vocab; build then let engine report real count.
    body = []
    n = target_tokens // 4
    pos = rng.randint(n // 5, 4 * n // 5)
    for i in range(n):
        if i == pos:
            body.append(f"IMPORTANT FACT: the secret code is {code}.")
        body.append(para(rng, 3) + ".")
    return " ".join(body) + "\n\nQuestion: what is the secret code? Answer with the 6-digit number only."


def post_stream(body, on_chunk, timeout=900):
    conn = http.client.HTTPConnection(*ENGINE, timeout=timeout)
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json", "X-Client": "bench-ef"})
    r = conn.getresponse()
    buf = b""
    while True:
        chunk = r.read1(65536)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1)
            for line in ev.split(b"\n"):
                if line.startswith(b"data: ") and line != b"data: [DONE]":
                    try:
                        on_chunk(json.loads(line[6:]))
                    except Exception:
                        pass
    conn.close()


def metrics():
    c = http.client.HTTPConnection(*ENGINE, timeout=10)
    c.request("GET", "/metrics")
    t = c.getresponse().read().decode()
    def g(name):
        return sum(float(x.split()[-1]) for x in t.splitlines() if x.startswith(name + "{"))
    return {k: g("vllm:" + k) for k in ("spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total",
                                         "spec_decode_num_accepted_tokens_total", "num_preemptions_total",
                                         "prefix_cache_hits_total", "prefix_cache_queries_total")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoders", type=int, default=6)
    ap.add_argument("--longs", type=int, default=3)
    ap.add_argument("--long-tokens", type=int, default=30000)
    ap.add_argument("--decode-tokens", type=int, default=1200)
    ap.add_argument("--long-gap", type=float, default=6.0, help="seconds between long-prompt submissions")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prefill-first", action="store_true", help="start the long prefills first, decoders join 4s later (older prefill row precedes newer decode rows in the scheduler's running list)")
    ap.add_argument("--out")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    m0 = metrics()
    res = {"decoders": [], "longs": []}
    lock = threading.Lock()
    t_start = time.time()

    def decoder(i):
        prompt = f"Stream {i}. " + para(random.Random(a.seed * 100 + i), 300) + "\nWrite a long numbered list of 400 short facts about rivers, one per line."
        body = {"model": "qwen-local", "messages": [{"role": "user", "content": prompt}], "max_tokens": a.decode_tokens,
                "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False}}
        times, text = [], []
        usage = {}
        t_sub = time.time()
        def on(ch):
            if ch.get("usage"):
                usage.update(ch["usage"])
            for c in ch.get("choices", []):
                d = (c.get("delta") or {}).get("content")
                if d:
                    times.append(time.time())
                    text.append(d)
        post_stream(body, on)
        with lock:
            res["decoders"].append({"i": i, "ttft": (times[0] - t_sub) if times else None, "times": times, "text": "".join(text)[:400], "n_chunks": len(times),
                                    "completion_tokens": usage.get("completion_tokens")})

    def longer(j):
        code = "%06d" % rng.randint(0, 999999)
        prompt = long_prompt(random.Random(a.seed * 1000 + j), a.long_tokens, code)
        body = {"model": "qwen-local", "messages": [{"role": "user", "content": prompt}], "max_tokens": 48, "temperature": 0,
                "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
        t0 = time.time(); first = [None]; text = []; usage = {}
        def on(ch):
            if ch.get("usage"):
                usage.update(ch["usage"])
            for c in ch.get("choices", []):
                d = (c.get("delta") or {}).get("content")
                if d:
                    if first[0] is None:
                        first[0] = time.time()
                    text.append(d)
        post_stream(body, on)
        out = "".join(text)
        with lock:
            res["longs"].append({"j": j, "ttft": (first[0] - t0) if first[0] else None, "ok": code in out,
                                 "prompt_tokens": usage.get("prompt_tokens"), "answer": out[:60]})

    ths = [threading.Thread(target=decoder, args=(i,)) for i in range(a.decoders)]
    lts = []
    if a.prefill_first:
        t_long0 = time.time()
        for j in range(a.longs):
            t = threading.Thread(target=longer, args=(j,)); t.start(); lts.append(t)
            if j == 0:
                time.sleep(4)
                for dt in ths:
                    dt.start(); time.sleep(0.25)
            time.sleep(a.long_gap)
    else:
        for t in ths:
            t.start(); time.sleep(0.25)
        time.sleep(6)  # let decode reach steady state
        t_long0 = time.time()
        for j in range(a.longs):
            t = threading.Thread(target=longer, args=(j,)); t.start(); lts.append(t); time.sleep(a.long_gap)
    for t in lts:
        t.join()
    t_long1 = time.time()
    for t in ths:
        t.join()
    m1 = metrics()
    # decode throughput inside the long-prefill window
    win = []
    gaps = []
    for d in res["decoders"]:
        ts = [x for x in d["times"] if t_long0 <= x <= t_long1]
        win.append(len(ts))
        allts = d["times"]
        g = max((b - c for c, b in zip(allts, allts[1:]) if t_long0 <= c <= t_long1), default=0)
        gaps.append(g)
    dur = t_long1 - t_long0
    drafts = m1["spec_decode_num_draft_tokens_total"] - m0["spec_decode_num_draft_tokens_total"]
    acc = m1["spec_decode_num_accepted_tokens_total"] - m0["spec_decode_num_accepted_tokens_total"]
    summ = {
        "prefill_window_s": round(dur, 1),
        "decode_chunks_in_window_total": sum(win),
        "decode_chunks_per_s_in_window": round(sum(win) / dur, 2),
        "decode_chunks_per_stream_per_s": round(statistics.mean(win) / dur, 3) if win else None,
        "decode_worst_gap_s": round(max(gaps), 2) if gaps else None,
        "decode_median_worst_gap_s": round(statistics.median(gaps), 2) if gaps else None,
        "decoder_ttft_s_max": round(max((d["ttft"] for d in res["decoders"] if d.get("ttft") is not None), default=0), 1),
        "decoder_ttft_s_median": round(statistics.median([d["ttft"] for d in res["decoders"] if d.get("ttft") is not None] or [0]), 1),
        "long_ttft_s": [round(x["ttft"], 1) if x["ttft"] else None for x in sorted(res["longs"], key=lambda x: x["j"])],
        "needle_ok": f"{sum(1 for x in res['longs'] if x['ok'])}/{len(res['longs'])}",
        "long_prompt_tokens": [x["prompt_tokens"] for x in sorted(res["longs"], key=lambda x: x["j"])],
        "spec_accept_rate": round(acc / drafts, 3) if drafts else None,
        "preemptions": m1["num_preemptions_total"] - m0["num_preemptions_total"],
        "total_s": round(time.time() - t_start, 1),
        "decoder_text_heads": [d["text"][:60] for d in sorted(res["decoders"], key=lambda d: d["i"])],
    }
    print(json.dumps(summ, indent=1))
    if a.out:
        json.dump({"summary": summ, "raw": {"decoders": [{k: v for k, v in d.items() if k != "times"} for d in res["decoders"]], "longs": res["longs"]}}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
