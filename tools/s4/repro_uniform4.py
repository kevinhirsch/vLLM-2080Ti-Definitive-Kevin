#!/usr/bin/env python3
"""Lane S4: GPU repro of the 2026-10-03 06:27:29 engine death (vllm-project/vllm#53051 class).
A batch whose rows ALL have 1 + num_speculative_tokens (= 4) scheduled tokens is shape-classified as uniform spec-decode and replayed on the
FULL cudagraph captured for pure spec rows.  If one of those rows is a NON-spec row (a request whose uncached prompt tail is exactly 4 tokens)
while at least one other row is a spec-decode row, the live GDN metadata (non_spec_token_indx = 4 elements) no longer matches the captured one
(0 elements): both TP workers raise `CUDAGRAPH-REFRESH mismatch dst shape=(0,) src shape=(4,)` and the engine dies.  A lone 4-token tail has
spec_sequence_masks None and is replayed silently on stale metadata (state loss, #53051).
Fixed engine (guard on, VLLM_UNIFORM_DECODE_NONSPEC_GUARD default 1): the step falls back to PIECEWISE and everything completes.
Usage (idle engine / gateway offline window): repro_uniform4.py --block 3568 (plain stack) | --block 1856 (fp16 SSM)
Steps: (1) A = X(k*block)+100 ids caches the aligned blocks; (2) C = long generation (spec rows, q=4 each step) is started; (3) B = X+[tail ids] is sent while C decodes."""
import argparse, json, random, http.client, time, sys, threading
ap = argparse.ArgumentParser(); ap.add_argument("--block", type=int, default=3568); ap.add_argument("--k", type=int, default=2); ap.add_argument("--tail", type=int, default=4)
ap.add_argument("--tries", type=int, default=6); ap.add_argument("--seed", type=int, default=int(time.time()) % 100000)
a = ap.parse_args(); r = random.Random(a.seed)
def post(ids, max_tokens=8):
    c = http.client.HTTPConnection("127.0.0.1", 8001, timeout=300)
    c.request("POST", "/v1/completions", json.dumps(dict(model="qwen-local", prompt=ids, max_tokens=max_tokens, temperature=0)), {"Content-Type": "application/json", "X-Client": "s4-repro"})
    resp = c.getresponse(); d = resp.read()
    try: j = json.loads(d); u = j.get("usage", {}); return resp.status, u.get("prompt_tokens"), (u.get("prompt_tokens_details") or {}).get("cached_tokens"), j["choices"][0]["text"][:30]
    except Exception: return resp.status, None, None, d[:120]
rnd = lambda n: [r.randint(1000, 100000) for _ in range(n)]
dead = False
for t in range(a.tries):
    X = rnd(a.k * a.block)
    print("try", t, "A (X+100):", post(X + rnd(100))[:3])
    res = {}
    def long_c():
        try: res["C"] = post(rnd(300), max_tokens=400)[:3]
        except Exception as e: res["C"] = "EXC %r" % e
    th = threading.Thread(target=long_c); th.start(); time.sleep(0.4 + 0.25 * t)   # C is decoding (q=4 spec rows) when B arrives
    try: res["B"] = post(X + rnd(a.tail))
    except Exception as e: res["B"] = "EXC %r" % e
    th.join(); print("   B (tail %d):" % a.tail, res.get("B"), "| C:", res.get("C"))
    if not isinstance(res.get("B"), tuple) or res["B"][0] != 200: dead = True; break
print("RESULT:", "ENGINE ERROR/DEATH" if dead else "all completed")
sys.exit(3 if dead else 0)
