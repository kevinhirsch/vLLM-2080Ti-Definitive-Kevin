#!/usr/bin/env python3
"""Lane CR: ground-truth continuation pairs from captured bodies joined to telemetry by sha256.
For each body whose longest-prefix predecessor is a PURE message prefix, report the engine's
cached tokens against the block-aligned expectation. usage: body_pairs.py DIR [B]"""
import glob, hashlib, json, os, sys, collections
d = sys.argv[1]; B = int(sys.argv[2]) if len(sys.argv) > 2 else 1856
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
    sha = hashlib.sha256(raw).hexdigest()
    items.append((os.path.basename(f), [json.dumps(m, sort_keys=True) for m in j.get("messages") or []], tel.get(sha, [None])[-1]))
out = collections.Counter(); rows = []
for i, (fn, ms, r) in enumerate(items):
    best = None
    for fn2, ms2, r2 in items[:i]:
        if len(ms2) < len(ms) and ms[:len(ms2)] == ms2 and (best is None or len(ms2) > len(best[1])):
            best = (fn2, ms2, r2)
    if not best or r is None or best[2] is None:
        out["unjoined_or_no_pred"] += 1; continue
    q = best[2]
    p, pp = r["ptok"], q["ptok"]; ca = r.get("cached_actual")
    E = ((pp - 4) // B) * B
    gap = (r["t"] - r["duration"]) - q["t"]
    if r.get("route") != "local":
        k = "cur_remote"
    elif q.get("route") != "local":
        k = "pred_remote"
    elif ca is None:
        k = "no_cached_field"
    elif ca >= E - B // 4:
        k = "hit_ok"
    elif ca >= B:
        k = "partial"
    else:
        k = "miss"
    out[k] += 1
    rows.append((fn, k, p, pp, ca, E, round(gap, 1), r.get("client"), r.get("pm_credit"), q.get("route"), r.get("route")))
for x in rows: print(*x)
print(dict(out))
