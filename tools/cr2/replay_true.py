#!/usr/bin/env python3
"""CR2: copy-mode tail-publish replay with ENGINE prompt lengths (cached_actual+computed_actual), not ptok estimates."""
import glob, hashlib, json, os, sys
d = sys.argv[1]; U = int(sys.argv[2]); lo, hi = float(sys.argv[3]), float(sys.argv[4]); tps = 1279.0
tel = {}
for f in sorted(glob.glob(os.path.expanduser("~/.local/share/vllm-qwen27b/telemetry/requests-2026100*.jsonl")))[-2:]:
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        if r.get("request_body_sha256"): tel[r["request_body_sha256"]] = r
items = []
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    raw = open(f, "rb").read()
    try: j = json.loads(raw)
    except Exception: continue
    r = tel.get(hashlib.sha256(raw).hexdigest())
    if r and lo <= r["t"] - r["duration"] < hi: items.append(([json.dumps(m, sort_keys=True) for m in j["messages"]], r))
items.sort(key=lambda x: x[1]["t"] - x[1]["duration"])
L = lambda r: r["cached_actual"] + r["computed_actual"]
all_comp = rec = rec_est = cont = n = 0
for i, (ms, r) in enumerate(items):
    if r["route"] != "local" or r.get("computed_actual") is None: continue
    comp = r["computed_actual"]; all_comp += comp
    best = None
    for ms2, r2 in items[:i]:
        if r2["t"] <= r["t"] - r["duration"] and len(ms2) < len(ms) and ms[:len(ms2)] == ms2 and (best is None or len(ms2) > len(best[0])):
            best = (ms2, r2)
    if not best or best[1]["route"] != "local" or best[1].get("computed_actual") is None: continue
    q = best[1]; n += 1; cont += comp
    rec += max(0, ((L(q) - 4) // U) * U - r["cached_actual"])
    rec_est += max(0, ((q["ptok"] - 4) // U) * U - r["cached_actual"])
print(json.dumps({"U": U, "local_computed_engine": all_comp, "continuations": n, "continuation_computed": cont,
  "recovered_tokens_engine_len": rec, "recovered_tokens_if_ptok_estimate": rec_est,
  "recovered_pct_of_all_local_computed": round(100 * rec / max(1, all_comp), 1),
  "recovered_pct_of_continuation_computed": round(100 * rec / max(1, cont), 1),
  "avg_recovered_per_continuation": round(rec / max(1, n)), "prefill_s_saved": round(rec / tps, 1)}, indent=1))
