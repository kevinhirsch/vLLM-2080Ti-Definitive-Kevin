#!/usr/bin/env python3
"""K5: per-decode-step anatomy from a vLLM torch-profiler trace (rank*.pt.trace.json.gz).
For each steady decode step (gpu_user_annotation execute_context_0(0)_generation_*): period to the next step start,
GPU kernel busy (union) inside the period, idle = period - busy, and busy split into classes.  usage: k5_steps.py TRACE"""
import collections
import gzip
import json
import statistics
import sys

ev = json.load(gzip.open(sys.argv[1]))["traceEvents"]
ann = sorted([e for e in ev if e.get("ph") == "X" and e.get("cat") == "gpu_user_annotation" and "generation_" in e.get("name", "")],
             key=lambda e: e["ts"])
ks = sorted([k for k in ev if k.get("ph") == "X" and k.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")], key=lambda k: k["ts"])


def cls(n):
    if "marlin::Marlin" in n:
        return "marlin"
    if "cross_device_reduce" in n or "nccl" in n.lower():
        return "allreduce"
    if "delta_rule" in n or "causal_conv1d" in n or "gdn_mtp_sm75" in n:
        return "gdn"
    if n.startswith("tq_") or "_tq_" in n:
        return "attn"
    if "cutlass" in n or "splitKreduce" in n or "sgemm" in n or "gemm" in n.lower():
        return "small_gemm"
    if "Memcpy" in n or "Memset" in n:
        return "memcpy"
    if "sampl" in n.lower() or "topk" in n.lower() or "SoftMax" in n or "argmax" in n.lower():
        return "sampler"
    return "glue"


rows = []
for i in range(len(ann) - 1):
    a = ann[i]
    if "context_0(0)" not in a["name"]:
        continue
    t0, t1 = a["ts"], ann[i + 1]["ts"]
    if t1 - t0 > 200000:  # not steady (>200 ms)
        continue
    sel = [k for k in ks if t0 <= k["ts"] < t1]
    busy, cs, ce = 0.0, None, None
    C = collections.Counter()
    for k in sel:
        s, e = k["ts"], min(k["ts"] + k["dur"], t1)
        C[cls(k["name"])] += k["dur"]
        if ce is None or s > ce:
            if ce is not None:
                busy += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if ce is not None:
        busy += ce - cs
    rows.append(dict(name=a["name"], period=t1 - t0, fwd=a["dur"], busy=busy, n=len(sel), cls=C))
if not rows:
    print("no steady decode steps")
    sys.exit(1)
med = lambda xs: statistics.median(xs)
P, B = med([r["period"] for r in rows]), med([r["busy"] for r in rows])
out = dict(trace=sys.argv[1].split("/")[-1], steps=len(rows), step_name=collections.Counter(r["name"] for r in rows).most_common(2),
           period_ms=round(P / 1e3, 2), target_fwd_ms=round(med([r["fwd"] for r in rows]) / 1e3, 2), busy_ms=round(B / 1e3, 2),
           idle_ms=round((P - B) / 1e3, 2), idle_pct=round(100 * (P - B) / P, 1), kernels_per_step=int(med([r["n"] for r in rows])),
           class_ms={c: round(med([r["cls"].get(c, 0) for r in rows]) / 1e3, 2) for c in
                     ("marlin", "glue", "allreduce", "gdn", "attn", "small_gemm", "memcpy", "sampler")})
print(json.dumps(out))
