#!/usr/bin/env python
"""Lane K9: FlashInfer fa2 ragged prefill on sm_75 at our TP2 per-rank shapes (12 q heads, 2 kv heads, hd 256),
use_fp16_qk_reduction False (today, QK+PV on HMMA.F32 half-rate path) vs True (QK on HMMA.F16 full rate).
Interleaved A/B, warmup, N reps; reports median/min/max ms and TFLOPS; plus error vs fp32 reference on given q/k/v.
Run with a private FLASHINFER_WORKSPACE_BASE so the engine's JIT cache is untouched."""
import argparse, json, os, statistics, time
import torch, flashinfer
ap = argparse.ArgumentParser()
ap.add_argument("--q", type=int, default=3632); ap.add_argument("--kv", default="3632,14336,28672")
ap.add_argument("--reps", type=int, default=7); ap.add_argument("--json", default="")
ap.add_argument("--qscale", type=float, default=1.0)
a = ap.parse_args()
dev = torch.device("cuda"); Hq, Hk, D = 12, 2, 256; sm = 1.0 / D ** 0.5
ws = torch.empty(128 << 20, dtype=torch.uint8, device=dev)
def mk(f16qk, q_len, kv_len):
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="fa2")
    w.plan(torch.tensor([0, q_len], dtype=torch.int32, device=dev), torch.tensor([0, kv_len], dtype=torch.int32, device=dev),
           Hq, Hk, D, causal=True, sm_scale=sm, q_data_type=torch.float16, kv_data_type=torch.float16, use_fp16_qk_reduction=f16qk)
    return w
def ref_attn(q, k, v):  # fp32, bottom-right causal, GQA
    qf, kf, vf = q.float(), k.float(), v.float(); ql, kl = q.shape[0], k.shape[0]
    kf = kf.repeat_interleave(Hq // Hk, 1); vf = vf.repeat_interleave(Hq // Hk, 1)
    out = torch.empty_like(qf)
    for s in range(0, ql, 512):
        e = min(ql, s + 512)
        lg = torch.einsum("qhd,khd->hqk", qf[s:e], kf) * sm
        qi = torch.arange(s, e, device=dev)[:, None] + (kl - ql); ki = torch.arange(kl, device=dev)[None]
        lg.masked_fill_((ki > qi)[None], float("-inf"))
        out[s:e] = torch.einsum("hqk,khd->qhd", lg.softmax(-1), vf)
    return out
res = []
g = torch.Generator(device=dev).manual_seed(0)
for kv_len in [int(x) for x in a.kv.split(",")]:
    q = (torch.randn(a.q, Hq, D, device=dev, generator=g) * a.qscale).half()
    k = torch.randn(kv_len, Hk, D, device=dev, generator=g).half(); v = torch.randn(kv_len, Hk, D, device=dev, generator=g).half()
    W = {f: mk(f, a.q, kv_len) for f in (False, True)}
    outs = {f: W[f].run(q, k, v) for f in W}; torch.cuda.synchronize()
    ref = ref_attn(q, k, v)
    err = {f: ((outs[f].float() - ref).norm() / ref.norm()).item() for f in W}
    nonfinite = {f: int((~torch.isfinite(outs[f])).sum()) for f in W}
    # causal FLOPs: 4*D*Hq*sum_i(visible kv)
    vis = sum(min(kv_len, kv_len - a.q + i + 1) for i in range(a.q)); flops = 4 * D * Hq * vis
    t = {False: [], True: []}
    for f in W:
        for _ in range(3): W[f].run(q, k, v)
    torch.cuda.synchronize()
    for r in range(a.reps):
        for f in ((False, True) if r % 2 == 0 else (True, False)):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); W[f].run(q, k, v); e1.record(); e1.synchronize(); t[f].append(e0.elapsed_time(e1))
    row = {"q": a.q, "kv": kv_len, "qscale": a.qscale}
    for f in W:
        nm = "f16qk" if f else "f32"
        row[nm] = {"med_ms": statistics.median(t[f]), "min_ms": min(t[f]), "max_ms": max(t[f]), "tflops_med": flops / statistics.median(t[f]) / 1e9,
                   "rel_err": err[f], "nonfinite": nonfinite[f]}
    row["speedup_med"] = row["f32"]["med_ms"] / row["f16qk"]["med_ms"]
    res.append(row); print(json.dumps(row), flush=True)
    del W, q, k, v, ref, outs; torch.cuda.empty_cache()
if a.json: json.dump(res, open(a.json, "w"), indent=1)
