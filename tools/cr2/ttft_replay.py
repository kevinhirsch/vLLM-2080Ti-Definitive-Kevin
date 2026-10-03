#!/usr/bin/env python3
"""CR2/L101: what does prefix-aware short-first buy? Telemetry replay (read-only).
Classes of LOCAL requests by engine length L=cached+computed and uncached C=computed:
  short      : L <= B            -> already eligible for the short-first yield
  warm_short : L > B, C <= B     -> eligible ONLY with VLLM_SCHED_SHORT_FIRST_PREFIX_AWARE=1
'blocked' = arrived at the engine while another local request's prefill had > 1 block of
compute left. Estimated new TTFT for blocked warm_short = its own compute time + the
yield-step overhead measured on blocked short requests (median ttft - own compute)."""
import bisect, glob, json, os, re, statistics as st, sys
RATE = float(os.environ.get("RATE", 1279.0))
gens = []
for line in open(sys.argv[1]):
    t = float(line.split()[0]); m = re.search(r"block size to (\d+)", line)
    if m: blk = int(m.group(1))
    if "GPU KV cache size" in line: gens.append((t, blk))
gstart = [g[0] for g in gens]
rows = []
for f in sys.argv[2:]:
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        if r.get("route") != "local" or r.get("status") != 200 or r.get("computed_actual") is None or not r.get("ttft"): continue
        t0 = r["t"] - r["duration"]; gi = bisect.bisect_right(gstart, t0) - 1
        if gi < 0: continue
        arr = t0 + (r.get("waited") or 0.0)
        rows.append(dict(arr=arr, ft=arr + r["ttft"], C=r["computed_actual"], L=r["computed_actual"] + r["cached_actual"],
                         B=gens[gi][1], gi=gi, ttft=r["ttft"], cl=r.get("client")))
rows.sort(key=lambda x: x["arr"])
def blocked(x):
    for y in rows:
        if y is x or y["gi"] != x["gi"] or y["arr"] >= x["arr"]: continue
        if y["ft"] > x["arr"] and y["C"] > y["B"] and (y["ft"] - x["arr"]) * RATE > y["B"]: return True
    return False
out = {}
for cls, pred in (("short", lambda x: x["L"] <= x["B"]), ("warm_short", lambda x: x["L"] > x["B"] and x["C"] <= x["B"])):
    xs = [x for x in rows if pred(x)]
    b = [x for x in xs if blocked(x)]
    out[cls] = {"n": len(xs), "blocked": len(b),
                "ttft_blocked_p50": round(st.median([x["ttft"] for x in b]), 2) if b else None,
                "ttft_blocked_p90": round(sorted(x["ttft"] for x in b)[int(0.9 * len(b))], 2) if b else None,
                "ttft_unblocked_p50": round(st.median([x["ttft"] for x in xs if x not in b]), 2) if len(xs) > len(b) else None,
                "excess_blocked_p50": round(st.median([x["ttft"] - x["C"] / RATE for x in b]), 2) if b else None}
short_b = [x for x in rows if x["L"] <= x["B"] and blocked(x)]
yield_overhead = st.median([x["ttft"] - x["C"] / RATE for x in short_b]) if short_b else 1.0
wb = [x for x in rows if x["L"] > x["B"] and x["C"] <= x["B"] and blocked(x)]
new = [min(x["ttft"], x["C"] / RATE + yield_overhead) for x in wb]
span_h = (rows[-1]["arr"] - rows[0]["arr"]) / 3600 if rows else 1
out["warm_short_blocked_estimate"] = {
    "yield_overhead_s(from blocked short)": round(yield_overhead, 2),
    "ttft_now_p50": round(st.median([x["ttft"] for x in wb]), 2) if wb else None,
    "ttft_new_p50": round(st.median(new), 2) if new else None,
    "ttft_now_p90": round(sorted(x["ttft"] for x in wb)[int(0.9 * len(wb))], 2) if wb else None,
    "ttft_new_p90": round(sorted(new)[int(0.9 * len(new))], 2) if new else None,
    "seconds_saved_total": round(sum(x["ttft"] for x in wb) - sum(new), 1),
    "per_hour": round(len(wb) / span_h, 1), "hours": round(span_h, 1)}
print(json.dumps(out, indent=1))
