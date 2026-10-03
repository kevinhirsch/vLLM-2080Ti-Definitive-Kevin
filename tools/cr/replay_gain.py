#!/usr/bin/env python3
"""Lane CR: expected recovery from copy-mode tail publication, replayed on ground-truth pairs.
For every captured local continuation whose true predecessor (pure message prefix) was local:
  today   : computed = p - cached_actual
  copy U  : the hit lands at floor((pp - g)/U)*U  (g = generation-prompt tokens re-rendered, 4)
            unless today's hit is already deeper.
Reports recovered tokens / today's computed, and prefill-seconds at the measured rate.
usage: replay_gain.py DIR [U] [prefill_tps]"""
import glob, hashlib, json, os, sys
d = sys.argv[1]; U = int(sys.argv[2]) if len(sys.argv) > 2 else 64; tps = float(sys.argv[3]) if len(sys.argv) > 3 else 1279.0
tel = {}
for f in sorted(glob.glob(os.path.expanduser("~/.local/share/vllm-qwen27b/telemetry/requests-2026100*.jsonl")))[-1:]:
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
    if r: items.append(([json.dumps(m, sort_keys=True) for m in j["messages"]], r))
items.sort(key=lambda x: x[1]["t"] - x[1]["duration"])
all_comp = rec = cont = n = 0
for i, (ms, r) in enumerate(items):
    if r["route"] != "local" or r.get("cached_actual") is None: continue
    comp = r["ptok"] - r["cached_actual"]; all_comp += comp
    best = None
    for ms2, r2 in items[:i]:
        if r2["t"] <= r["t"] - r["duration"] and len(ms2) < len(ms) and ms[:len(ms2)] == ms2 and (best is None or r2["ptok"] > best["ptok"]):
            best = r2
    if not best or best["route"] != "local": continue
    n += 1; cont += comp
    new_hit = ((best["ptok"] - 4) // U) * U
    rec += max(0, new_hit - r["cached_actual"])
print(json.dumps({"local_requests_computed": all_comp, "continuations": n, "continuation_computed": cont,
                  "recovered_tokens": rec, "recovered_pct_of_all_computed": round(100 * rec / max(1, all_comp), 1),
                  "recovered_pct_of_continuation_computed": round(100 * rec / max(1, cont), 1),
                  "avg_recovered_per_continuation": round(rec / max(1, n)),
                  "prefill_seconds_saved_at_%d_tps" % tps: round(rec / tps, 1)}, indent=1))
