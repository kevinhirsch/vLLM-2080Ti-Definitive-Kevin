#!/usr/bin/env python3
"""Lane LP arm B1: cache-proof cold prefill rate. One ~28K-token recorded estate body, a random salt prepended to the first
message (so no prefix-cache page can hit), max_tokens=1, N reps; prints prompt_tokens / wall per rep. Usage: cold_prefill.py [N] [out.json]"""
import glob, http.client, json, os, random, sys, time
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
FR = os.environ.get("ESTATE_FR", "/home/kevin/projects/lanes/s2-speed/fr")
files = sorted(p for p in glob.glob(FR + "/*.json") if 26000 <= int(os.path.basename(p).split("_")[1].replace("tok.json", "")) <= 31000)
assert files, "no 26-31K body in " + FR
c = http.client.HTTPConnection("127.0.0.1", 8001, timeout=10); c.request("GET", "/v1/models"); M = json.loads(c.getresponse().read())["data"][0]["id"]
res = []
for i in range(N):
    b = json.load(open(files[i % len(files)]))
    for k in ("max_completion_tokens", "stream_options", "store", "thinking_token_budget", "reasoning_effort", "n", "logprobs", "top_logprobs"): b.pop(k, None)
    salt = "".join(random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(48))
    m0 = b["messages"][0]
    if isinstance(m0.get("content"), str): m0["content"] = f"[{salt}] " + m0["content"]
    else: b["messages"].insert(0, {"role": "system", "content": f"[{salt}]"})
    b.update(model=M, max_tokens=1, stream=False, temperature=0)
    t = time.time()
    cn = http.client.HTTPConnection("127.0.0.1", 8001, timeout=1800)
    cn.request("POST", "/v1/chat/completions", json.dumps(b), {"Content-Type": "application/json", "X-Client": "lp-b1"})
    r = json.loads(cn.getresponse().read()); dt = time.time() - t
    u = r.get("usage", {}); pt = u.get("prompt_tokens", 0); cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    res.append({"file": os.path.basename(files[i % len(files)]), "prompt_tokens": pt, "cached": cached, "wall_s": dt, "tok_s": (pt - (cached or 0)) / dt})
    print(f"B1 rep{i}: {pt} tok cached {cached} wall {dt:.2f}s -> {(pt-(cached or 0))/dt:.0f} tok/s", flush=True)
if len(sys.argv) > 2: json.dump(res, open(sys.argv[2], "w"), indent=1)
print("B1 median tok/s", sorted(r["tok_s"] for r in res)[len(res) // 2])
