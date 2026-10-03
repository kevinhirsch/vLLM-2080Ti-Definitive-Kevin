#!/usr/bin/env python3
"""Lane DFT live A/B on the running engine (:8001): natural-text decode x3 and the warm 12-body estate pass x3 (after one discarded prime pass),
with per-position speculative acceptance read from /metrics deltas. usage: live_ab.py LABEL OUT.json  (ESTATE_FR must point at s2-speed/fr)"""
import json, http.client, os, subprocess, sys, time, statistics
label, out = sys.argv[1], sys.argv[2]
ESTATE = "/home/kevin/Desktop/wt-integrate/tools/s2-bench/estate_load.py"
FR = os.path.expanduser("~/projects/lanes/s2-speed/fr")
def get(path, body=None, timeout=900):
    c = http.client.HTTPConnection("127.0.0.1", 8001, timeout=timeout)
    if body is None: c.request("GET", path)
    else: c.request("POST", path, json.dumps(body), {"Content-Type": "application/json", "X-Client": "dft-live-ab"})
    return c.getresponse().read()
def met():
    t = get("/metrics").decode(); d = {}
    for ln in t.splitlines():
        if ln.startswith("vllm:spec_decode_num_drafts_total"): d["drafts"] = float(ln.split()[-1])
        if ln.startswith("vllm:spec_decode_num_draft_tokens_total"): d["draft_tokens"] = float(ln.split()[-1])
        if ln.startswith("vllm:spec_decode_num_accepted_tokens_total"): d["accepted"] = float(ln.split()[-1])
        if ln.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total"): d[f"pos{ln.split('position=\"')[1].split('\"')[0]}"] = float(ln.split()[-1])
    return d
def delta(a, b):
    dr = max(b["drafts"] - a["drafts"], 1)
    p = [b[f"pos{k}"] - a[f"pos{k}"] for k in range(3)]
    return dict(drafts=int(dr), pos_abs=[round(x / dr, 4) for x in p], pos_cond=[round(p[0] / dr, 4), round(p[1] / max(p[0], 1), 4), round(p[2] / max(p[1], 1), 4)],
                accept_len=round(1 + sum(p) / dr, 3))
MODEL = json.loads(get("/v1/models"))["data"][0]["id"]
res = dict(label=label, t=time.strftime("%F %T"))
# natural text
for w in range(3):
    get("/v1/chat/completions", dict(model=MODEL, messages=[{"role": "user", "content": f"Warm-up {w}: explain how a bicycle works in 150 words."}], max_tokens=150, temperature=0, chat_template_kwargs={"enable_thinking": False}))
nat, m0 = [], met()
for i in range(3):
    t0 = time.time()
    r = json.loads(get("/v1/chat/completions", dict(model=MODEL, messages=[{"role": "user", "content": f"Write a detailed essay about the history of topic number {i}: the printing press, with numbered sections."}],
                                                    max_tokens=400, min_tokens=400, temperature=0, chat_template_kwargs={"enable_thinking": False})))
    nat.append(r["usage"]["completion_tokens"] / (time.time() - t0))
res["natural_tok_s"] = [round(x, 1) for x in nat]; res["natural_accept"] = delta(m0, met())
# estate pass: prime + 3 reps
env = dict(os.environ, ESTATE_FR=FR)
def estate(tag):
    o = f"/tmp/dft_estate_{label}_{tag}.json"
    subprocess.run([sys.executable, ESTATE, "--n", "12", "--max-tokens", "256", "--out", o], env=env, capture_output=True, timeout=1500)
    return json.load(open(o))["summary"]
estate("prime")
reps = []
for i in range(3):
    m1 = met(); s = estate(f"r{i}"); s["accept"] = delta(m1, met()); reps.append(s)
res["estate_wall_s"] = [r["wall_s"] for r in reps]; res["estate_agg_tok_s"] = [r["agg_out_tok_s"] for r in reps]
res["estate_med_dec_tps"] = [r["med_dec_tps"] for r in reps]; res["estate_accept"] = [r["accept"] for r in reps]; res["estate_reps"] = reps
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("estate_reps",)}))
