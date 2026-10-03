#!/usr/bin/env python3
"""Lane R2: how much local prefill is spent on tool-loop continuations that LOST their cache?

Reads the gateway request telemetry (read-only). A request is a "continuation" when the same client sent a
request <=300 s earlier whose prompt is 3.5K..12K tokens shorter than this one; it is "lost" when cached_actual
is more than one page below the predecessor's replay boundary. Heuristic (same-client look-alikes can inflate
it); the share of tool-result previews in the lost set is printed as a plausibility check.
usage: lost_continuation.py [--since 'YYYY-MM-DD HH:MM'] [--until ...] [--client halo-hermes]
"""
import argparse, collections, datetime, glob, json, os

BS = 3568
ap = argparse.ArgumentParser()
ap.add_argument("--since"); ap.add_argument("--until"); ap.add_argument("--client")
a = ap.parse_args()
ts = lambda s: datetime.datetime.strptime(s, "%Y-%m-%d %H:%M").timestamp() if s else None
lo, hi = ts(a.since), ts(a.until)
rows = []
for f in sorted(glob.glob(os.path.expanduser("~/.local/share/vllm-qwen27b/telemetry/requests-*.jsonl"))):
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        if r.get("route") == "local" and r.get("status") == 200 and r.get("cached_actual") is not None and r.get("ptok"):
            if (lo is None or r["t"] >= lo) and (hi is None or r["t"] < hi) and (not a.client or r["client"] == a.client):
                rows.append(r)
rows.sort(key=lambda r: r["t"])
recent = collections.defaultdict(list)
tot = lost = ok = cold = 0; lostn = okn = 0; tool_prev = 0
for r in rows:
    c, t, p, ca = r["client"], r["t"], r["ptok"], r["cached_actual"]
    comp = p - ca; tot += comp
    cand = [(tp, pp) for tp, pp in recent[c] if t - tp <= 300 and pp <= p and p - pp < 12000 and pp > 2 * BS]
    if cand:
        pp = max(cand)[1]; ex = ((pp - 1) // BS) * BS
        if ca >= ex - BS: ok += comp; okn += 1
        else:
            lost += comp; lostn += 1
            if r["preview"].startswith("<untrusted_tool_result"): tool_prev += 1
    else: cold += comp
    recent[c].append((t, p)); recent[c] = [x for x in recent[c] if t - x[0] <= 300]
print(f"requests={len(rows)} computed={tot/1e6:.2f}M continuation-ok={ok/1e6:.2f}M ({okn}) LOST={lost/1e6:.2f}M ({lostn}, "
      f"{100*lost/max(tot,1):.0f}% of computed, {100*tool_prev/max(lostn,1):.0f}% with tool-result preview) other={cold/1e6:.2f}M")
