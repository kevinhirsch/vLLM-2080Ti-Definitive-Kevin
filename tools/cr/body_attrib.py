#!/usr/bin/env python3
"""Lane CR: GROUND-TRUTH attribution of local prefill on captured bodies (>=15K-token requests).
Each local request's computed tokens (ptok - cached) split into:
  new      : tokens appended since its true predecessor (inherent)
  tail     : predecessor prefix past its last block boundary (structural to align-mode GDN)
  one_block: an extra whole block short of the predecessor's last boundary
  lost_other: deeper miss (evict / restart / remote predecessor)
  cold     : no captured predecessor sharing the system prompt
  sibling_tail: shares a prefix with an earlier SIBLING session (not a continuation): uncached
             part of the shared prefix
usage: body_attrib.py DIR [B] [--since EPOCH]"""
import glob, hashlib, json, os, sys, collections
d = sys.argv[1]; B = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 1856
tel = {}
for f in sorted(glob.glob(os.path.expanduser("~/.local/share/vllm-qwen27b/telemetry/requests-2026100*.jsonl")))[-2:]:
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        if r.get("request_body_sha256"): tel[r["request_body_sha256"]] = r
items = []
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    if f.endswith(".resp.json"): continue
    raw = open(f, "rb").read()
    try: j = json.loads(raw)
    except Exception: continue
    ms = [json.dumps(m, sort_keys=True) for m in j.get("messages") or []]
    r = tel.get(hashlib.sha256(raw).hexdigest())
    if r is None: continue
    cl = [0]
    for m in ms: cl.append(cl[-1] + len(m))
    items.append((r["t"] - r["duration"], ms, cl, r))
items.sort(key=lambda x: x[0])
tot = collections.Counter(); n = collections.Counter()
for i, (t0, ms, cl, r) in enumerate(items):
    if r.get("route") != "local" or r.get("cached_actual") is None: continue
    p, ca = r["ptok"], r["cached_actual"]; comp = max(0, p - ca); tot["computed"] += comp; n["req"] += 1
    best = None
    for t02, ms2, cl2, r2 in items[:i]:
        if r2["t"] > t0: continue           # predecessor must have finished
        k = 0
        while k < min(len(ms), len(ms2)) and ms[k] == ms2[k]: k += 1
        if k == 0: continue
        pure = k == len(ms2) and k < len(ms)
        shared_tok = r2["ptok"] if pure else int(cl[k] / max(1, cl[-1]) * p)
        if best is None or shared_tok > best[0]: best = (shared_tok, pure, r2)
    if best is None:
        tot["cold"] += comp; n["cold"] += 1; continue
    st, pure, r2 = best
    E = ((st - 4) // B) * B
    if not pure:
        tot["sibling"] += comp; n["sibling"] += 1; tot["sibling_lost"] += max(0, E - ca); continue
    new = max(0, p - st); tail = st - E; lost = max(0, E - ca)
    tot["new"] += new; tot["tail"] += min(tail, max(0, p - ca - new))
    n["cont"] += 1
    if lost == 0: n["ok"] += 1
    elif r2.get("route") != "local": tot["lost_pred_remote"] += lost; n["pred_remote"] += 1
    elif lost <= B + B // 4: tot["one_block"] += lost; n["one_block"] += 1
    else: tot["lost_other"] += lost; n["lost_other"] += 1
c = tot["computed"]
print(json.dumps({"requests": n["req"], "computed_tok": c, "counts": dict(n),
                  "share_of_computed": {k: round(v / max(1, c), 3) for k, v in tot.items() if k != "computed"}}, indent=1))
