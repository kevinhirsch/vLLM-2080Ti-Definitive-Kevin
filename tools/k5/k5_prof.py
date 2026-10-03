#!/usr/bin/env python3
"""K5: torch-profile steady decode on :8001 (engine booted with --profiler-config).  usage: k5_prof.py STREAMS [TOKENS]
Warms the streams first (so the profile holds only decode steps), then profiles a window of decode with STREAMS
concurrent short-prompt requests (temp 0, ignore_eos)."""
import http.client
import json
import sys
import threading
import time

S = int(sys.argv[1])
NTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 96


def call(m, p, body=None, timeout=900):
    c = http.client.HTTPConnection("127.0.0.1", 8001, timeout=timeout)
    t0 = time.time()
    c.request(m, p, json.dumps(body) if body is not None else None, {"Content-Type": "application/json"})
    r = c.getresponse()
    d = r.read()
    return r.status, round(time.time() - t0, 3), d


M = json.loads(call("GET", "/v1/models")[2])["data"][0]["id"]
res = [None] * S


def one(i, n, tag):
    b = dict(model=M, messages=[{"role": "user", "content": f"K5 {tag} {i}: write a long, detailed essay about the history of printing, section {i}."}],
             max_tokens=n, min_tokens=n, temperature=0, ignore_eos=True, chat_template_kwargs={"enable_thinking": False})
    st, w, d = call("POST", "/v1/chat/completions", b)
    u = json.loads(d).get("usage", {})
    res[i] = (st, w, u.get("completion_tokens"))


def wave(n, tag):
    th = [threading.Thread(target=one, args=(i, n, tag)) for i in range(S)]
    t0 = time.time()
    [t.start() for t in th]
    [t.join() for t in th]
    return time.time() - t0


wave(32, "warm")
# start long requests, profile a window in the middle of their decode
th = [threading.Thread(target=one, args=(i, NTOK + 400, "prof")) for i in range(S)]
t0 = time.time()
[t.start() for t in th]
time.sleep(4.0)
print("start", call("POST", "/start_profile")[:2], flush=True)
time.sleep(1.5)
print("stop", call("POST", "/stop_profile")[:2], flush=True)
[t.join() for t in th]
wall = time.time() - t0
tok = sum(r[2] or 0 for r in res)
print(json.dumps(dict(streams=S, wall_s=round(wall, 2), tokens=tok, agg_tok_s=round(tok / wall, 1), status=[r[0] for r in res])))
