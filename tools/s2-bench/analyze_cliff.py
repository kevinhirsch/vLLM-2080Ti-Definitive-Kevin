#!/usr/bin/env python3
"""analyze_cliff.py TAG -- per pass: wall, TTFT med, cached frac, ΔPC hit-rate, ΔPreempt, KV max, GPU clocks/throttle during the pass."""
import json,sys,csv,glob,os
tag=sys.argv[1]
rows=list(csv.DictReader(open(f"{tag}_sampler.csv")))
def f(x):
    try: return float(x)
    except: return None
meta=json.load(open(f"{tag}_passes.json"))   # list of {name,t0,t1,file}
for m in meta:
    d=json.load(open(m["file"])); r=d["reqs"]; s=d["summary"]
    seg=[x for x in rows if m["t0"]<=float(x["t"])<=m["t1"]]
    if not seg: print(m["name"],"no samples"); continue
    h0,h1=f(seg[0]["pc_hits"]),f(seg[-1]["pc_hits"]); q0,q1=f(seg[0]["pc_queries"]),f(seg[-1]["pc_queries"])
    pre=f(seg[-1]["preempt"])-f(seg[0]["preempt"]); kvmax=max(f(x["kv_usage"]) or 0 for x in seg)
    acc=(f(seg[-1]["spec_acc"])-f(seg[0]["spec_acc"]))/max(1,(f(seg[-1]["spec_drafts"])-f(seg[0]["spec_drafts"])))
    sm0=sum(f(x["g0_sm"]) for x in seg)/len(seg); sm1=sum(f(x["g1_sm"]) for x in seg)/len(seg)
    pw0=sum(f(x["g0_w"]) for x in seg)/len(seg); pw1=sum(f(x["g1_w"]) for x in seg)/len(seg)
    t0=max(f(x["g0_c"]) for x in seg); t1=max(f(x["g1_c"]) for x in seg)
    thr=set(x["g0_thr"].strip() for x in seg)|set(x["g1_thr"].strip() for x in seg)
    cf=sum(x["cached"] or 0 for x in r)/max(1,sum(x["prompt"] for x in r))
    print(f"{m['name']:10s} wall={s['wall_s']:6.1f}s ttft_med={s['med_ttft']} cached={cf:4.2f} ΔPChit={(h1-h0)/max(1,(q1-q0)):4.2f} preempt={pre:.0f} kvmax={kvmax:4.2f} accept/draft={acc:4.2f} sm_MHz={sm0:.0f}/{sm1:.0f} W={pw0:.0f}/{pw1:.0f} maxT={t0:.0f}/{t1:.0f} thr={sorted(thr)}")
