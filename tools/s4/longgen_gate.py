#!/usr/bin/env python3
"""Lane S4: long-generation quality gate for the GDN state dtype (fp32 / fp16-RNE / fp16-SR).
K8 measured that a fp16 round-to-nearest state drifts with decode length (KL 0.0005 -> 0.0038 over 256 tokens).  This gate looks at THOUSANDS of generated tokens:
for each of P prompts x S seeds it generates up to --max-tokens (default 4096) at temperature 0.7 / top_p 0.95 with a fixed per-request seed, no thinking,
and records the chosen-token logprob of every token.  Per config it reports, by 512-token bucket: mean chosen-token logprob; plus distinct-4-gram ratio of the last
1024 tokens, the fraction of runs that end in a repetition loop, mean length, and the fraction that ended with EOS.  A drifting state shows as a trend of the
bucket means away from the fp32-state config, falling distinct-4gram ratio, or loops.   Usage:  longgen_gate.py run TAG   |   longgen_gate.py cmp TAG_A TAG_B [TAG_C]"""
import json, sys, os, time, threading, http.client, statistics, collections
OUT = "/home/kevin/projects/lanes/s4/longgen"
PROMPTS = [
    "Write a long, detailed essay on the history and engineering of the printing press, with numbered sections, covering Gutenberg, the spread across Europe, typography, economics and social effects. Keep going in depth.",
    "Write a long short-story with many chapters about a lighthouse keeper who discovers a letter in a bottle every morning. Develop characters and subplots at length.",
    "Write a complete Python module implementing an LRU cache, a trie, a priority queue and a rate limiter with extensive docstrings, type hints and a long test-suite using unittest. Include all code.",
    "Derive step by step, with every algebraic step shown, the closed forms for the sum of the first n squares, cubes and fourth powers, then prove them by induction, then generalize with Bernoulli numbers. Be very thorough.",
    "Explain in great depth how a modern CPU executes instructions: fetch, decode, rename, schedule, execute, retire, caches, branch prediction, prefetching, memory ordering. Use many subsections and examples.",
    "Write a detailed technical design document for a distributed job scheduler: requirements, architecture, data model, failure modes, consistency, observability, rollout plan, with many tables and subsections.",
    "Write an extensive tutorial on SQL window functions with at least twenty worked examples, each with schema, query, result and explanation.",
    "Compose a long dialogue between a physicist and a philosopher about the nature of time, with at least fifty turns and substantive arguments in each turn.",
]
def post(body, timeout=3600):
    c = http.client.HTTPConnection("127.0.0.1", 8001, timeout=timeout)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json", "X-Client": "s4-longgen"})
    r = c.getresponse(); return json.loads(r.read())
def run(tag, maxtok=4096, seeds=(1, 2, 3)):
    os.makedirs(OUT, exist_ok=True)
    c = http.client.HTTPConnection("127.0.0.1", 8001, timeout=10); c.request("GET", "/v1/models"); model = json.loads(c.getresponse().read())["data"][0]["id"]
    res, lk = [], threading.Lock()
    def one(pi, sd):
        d = post(dict(model=model, messages=[{"role": "user", "content": PROMPTS[pi]}], max_tokens=maxtok, temperature=0.7, top_p=0.95, seed=sd * 100 + pi,
                      logprobs=True, top_logprobs=0, chat_template_kwargs={"enable_thinking": False}))
        ch = d["choices"][0]; toks = [t["token"] for t in ch["logprobs"]["content"]]; lps = [t["logprob"] for t in ch["logprobs"]["content"]]
        with lk: res.append(dict(p=pi, seed=sd, finish=ch["finish_reason"], toks=toks, lps=lps))
    t0 = time.time(); th = [threading.Thread(target=one, args=(pi, sd)) for sd in seeds for pi in range(len(PROMPTS))]
    # 8 at a time (estate-like concurrency)
    for i in range(0, len(th), 8):
        for t in th[i:i + 8]: t.start()
        for t in th[i:i + 8]: t.join()
    json.dump(res, open(f"{OUT}/{tag}.json", "w")); print(tag, "runs", len(res), "wall_s", round(time.time() - t0), "tokens", sum(len(r["toks"]) for r in res))
def stats(res):
    B = 512; nb = 8; bsum = [[] for _ in range(nb)]
    d4, loops, lens, eos = [], 0, [], 0
    for r in res:
        lens.append(len(r["toks"])); eos += r["finish"] == "stop"
        for i, lp in enumerate(r["lps"]):
            b = i // B
            if b < nb: bsum[b].append(lp)
        tail = r["toks"][-1024:]
        g = [tuple(tail[i:i + 4]) for i in range(max(0, len(tail) - 3))]
        if g: d4.append(len(set(g)) / len(g))
        t = r["toks"][-1024:]; loop = False
        for blk in (16, 32, 64):
            if len(t) >= 4 * blk and all(t[-blk * (k + 1):len(t) - blk * k] == t[-blk:] for k in range(1, 4)): loop = True
        loops += loop
    return dict(bucket_lp=[round(statistics.mean(x), 4) if x else None for x in bsum], d4_last1024=round(statistics.mean(d4), 4) if d4 else None,
                loops=f"{loops}/{len(res)}", mean_len=round(statistics.mean(lens)), eos=f"{eos}/{len(res)}")
def cmp(tags):
    S = {t: stats(json.load(open(f"{OUT}/{t}.json"))) for t in tags}
    print("bucket (tokens)    " + "  ".join(f"{i*512:>5}-{(i+1)*512:<5}" for i in range(8)))
    for t, s in S.items(): print(f"{t:14s} lp   " + "  ".join(f"{(x if x is not None else float('nan')):12.4f}" for x in s["bucket_lp"]))
    base = tags[0]
    for t in tags[1:]:
        print(f"{t} minus {base}: " + "  ".join(f"{(a-b):12.4f}" if a is not None and b is not None else "         n/a" for a, b in zip(S[t]["bucket_lp"], S[base]["bucket_lp"])))
    for t, s in S.items(): print(t, {k: v for k, v in s.items() if k != "bucket_lp"})
if __name__ == "__main__":
    if sys.argv[1] == "run": run(sys.argv[2], int(os.environ.get("MAXTOK", 4096)))
    else: cmp(sys.argv[2:])
