#!/usr/bin/env python3
"""CR2: re-score continuation pairs with the ENGINE's prompt length (cached_actual+computed_actual)
instead of the gateway's estimate (ptok). usage: truepairs.py DIR B LO HI"""
import glob, hashlib, json, os, sys, collections
d = sys.argv[1]; B = int(sys.argv[2]); lo, hi = float(sys.argv[3]), float(sys.argv[4])
tel = {}
for f in sorted(glob.glob(os.path.expanduser("~/.local/share/vllm-qwen27b/telemetry/requests-2026100*.jsonl")))[-2:]:
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        if r.get("request_body_sha256"): tel.setdefault(r["request_body_sha256"], []).append(r)
items = []
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    raw = open(f, "rb").read()
    try: j = json.loads(raw)
    except Exception: continue
    r = tel.get(hashlib.sha256(raw).hexdigest(), [None])[-1]
    if r is None: continue
    t0 = r["t"] - r["duration"]
    if lo <= t0 < hi: items.append((t0, [json.dumps(m, sort_keys=True) for m in j.get("messages") or []], r))
items.sort(key=lambda x: x[0])
c = collections.Counter(); est_err = []; lost = collections.Counter()
for i, (t0, ms, r) in enumerate(items):
    if r.get("route") != "local" or r.get("cached_actual") is None: continue
    best = None
    for t02, ms2, r2 in items[:i]:
        if r2["t"] <= t0 and len(ms2) < len(ms) and ms[:len(ms2)] == ms2 and (best is None or len(ms2) > len(best[1])): best = (t02, ms2, r2)
    if not best: continue
    q = best[2]
    if q.get("route") != "local" or q.get("computed_actual") is None: continue
    true_pp = q["cached_actual"] + q["computed_actual"]
    est_err.append(q["ptok"] - true_pp)
    E_est = ((q["ptok"] - 4) // B) * B; E_true = ((true_pp - 4) // B) * B; ca = r["cached_actual"]
    k_est = "ok" if ca >= E_est else ("one_short" if ca >= E_est - B else "deeper")
    k_true = "ok" if ca >= E_true else ("one_short" if ca >= E_true - B else "deeper")
    c[(k_est, k_true)] += 1
    lost["est"] += max(0, E_est - ca); lost["true"] += max(0, E_true - ca)
est_err.sort()
print(json.dumps({"pairs": sum(c.values()), "(by_estimate, by_engine_length)": {f"{a}->{b}": n for (a, b), n in c.items()},
                  "lost_tokens": dict(lost),
                  "ptok_estimate_minus_engine_len": {"min": est_err[0], "p50": est_err[len(est_err)//2], "max": est_err[-1],
                   "share_overestimated": round(sum(e > 0 for e in est_err) / len(est_err), 2)}}, indent=1))
