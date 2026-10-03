#!/usr/bin/env python
"""Lane K9: fa75 prefill (K1 kernel) QK16 two-level variants vs K1's fp32-QK variants: error vs an fp32 reference and
interleaved speed, at TP2 per-rank shapes (12 q heads / 2 kv heads, hd 256, bottom-right causal).
Data: real post-norm q/k/v rows of a full-attention layer (U2 CPU reference), resampled to long contexts with RoPE
re-applied at the new positions; 'peak' additionally plants keys aligned with the queries (worst-case sharp softmax)."""
import argparse, json, math, os, statistics, sys
import torch
sys.path.insert(0, "/home/kevin/Desktop/wt-integrate/tools/u2")
import ref_dump as R
from vllm.v1.attention.ops import fa75_prefill as FA
ap = argparse.ArgumentParser()
ap.add_argument("--shapes", default="3632:3632,3632:32768,3632:65536")
ap.add_argument("--variants", default="2,3,10,11,19,27,35")
ap.add_argument("--data", default="real")  # rand | real | peak
ap.add_argument("--layer", type=int, default=3); ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--noref", action="store_true"); ap.add_argument("--json", default="")
a = ap.parse_args()
dev = torch.device("cuda"); Hq, Hk, D = 12, 2, 256; scale = D ** -0.5
inv, ms = R.yarn_inv_freq_and_mscale()
def rope(x, pos):  # x [T,H,256] fp32, neox on first 64 dims
    fr = torch.outer(pos.float(), inv); emb = torch.cat([fr, fr], -1); c, s = (emb.cos() * ms)[:, None], (emb.sin() * ms)[:, None]
    xr, xp = x[..., :64], x[..., 64:]; h = 32
    return torch.cat([xr * c + torch.cat([-xr[..., h:], xr[..., :h]], -1) * s, xp], -1)
real = torch.load("/home/kevin/projects/lanes/k9/acts/qkv_real.pt")["qkv"] if a.data != "rand" else None
def make(Tq, Tkv, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if a.data == "rand":
        return (torch.randn(Tq, Hq, D, generator=g).half().to(dev), torch.randn(Tkv, Hk, D, generator=g).half().to(dev),
                torch.randn(Tkv, Hk, D, generator=g).half().to(dev))
    L = real[a.layer]; N = L["q"].shape[0]
    qi = torch.randint(0, N, (Tq,), generator=g); ki = torch.randint(0, N, (Tkv,), generator=g)
    q = L["q"][qi][:, :Hq].float(); k = L["k"][ki][:, :Hk].float(); v = L["v"][ki][:, :Hk].contiguous().to(dev)  # rank-0 heads
    if a.data == "peak":  # plant 64 keys per kv head equal to a (scaled) query of its group: logits near |q||k| bound
        for j in torch.randint(0, Tkv, (64,), generator=g).tolist():
            src = q[torch.randint(0, Tq, (1,), generator=g).item()]
            for hk in range(Hk): k[j, hk] = src[hk * (Hq // Hk)] / src[hk * (Hq // Hk)].norm() * L["k"][:, hk].float().norm(dim=-1).max()
    pos_k = torch.arange(Tkv); pos_q = pos_k[Tkv - Tq:]
    return rope(q, pos_q).half().to(dev), rope(k, pos_k).half().to(dev), v
def ref_attn(q, k, v):  # fp32, per q head, chunked rows: small footprint next to the live engine
    Tq, Tkv = q.shape[0], k.shape[0]; out = torch.empty(Tq, Hq, D, device=dev)
    step = max(16, int(2 ** 22 // Tkv))
    kidx = torch.arange(Tkv, device=dev)[None]
    for h in range(Hq):
        kf = k[:, h // (Hq // Hk)].float(); vf = v[:, h // (Hq // Hk)].float()
        for s0 in range(0, Tq, step):
            e = min(Tq, s0 + step)
            lg = (q[s0:e, h].float() @ kf.T).mul_(scale)
            qi = torch.arange(s0, e, device=dev)[:, None] + (Tkv - Tq)
            lg.masked_fill_(kidx > qi, float("-inf"))
            out[s0:e, h] = torch.softmax(lg, -1) @ vf; del lg
        del kf, vf
    return out
var = [int(x) for x in a.variants.split(",")]
rows = []
for sh in a.shapes.split(","):
    Tq, Tkv = (int(x) for x in sh.split(":"))
    q, k, v = make(Tq, Tkv)
    vis = Tq * (Tkv - Tq) + Tq * (Tq + 1) / 2; flops = 4 * vis * D * Hq
    ref = None if a.noref else ref_attn(q, k, v)
    o = torch.empty_like(q)
    row = {"Tq": Tq, "Tkv": Tkv, "data": a.data, "layer": a.layer, "v": {}}
    for vv in var:
        FA.fa75_prefill(q, k, v, scale=scale, out=o, variant=vv); torch.cuda.synchronize()
        d = {"nonfinite": int((~torch.isfinite(o)).sum())}
        if ref is not None:
            d["rel"] = ((o.float() - ref).norm() / ref.norm()).item()
            d["maxabs"] = (o.float() - ref).abs().max().item()
        row["v"][vv] = d
    if ref is not None:
        row["fp16_round_rel"] = ((ref.half().float() - ref).norm() / ref.norm()).item()
    for vv in var:
        for _ in range(2): FA.fa75_prefill(q, k, v, scale=scale, out=o, variant=vv)
    t = {vv: [] for vv in var}
    for r in range(a.reps):
        order = var if r % 2 == 0 else var[::-1]
        for vv in order:
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); FA.fa75_prefill(q, k, v, scale=scale, out=o, variant=vv); e1.record(); e1.synchronize()
            t[vv].append(e0.elapsed_time(e1))
    base = statistics.median(t[var[0]])
    for vv in var:
        m = statistics.median(t[vv])
        row["v"][vv].update({"med_ms": round(m, 3), "min_ms": round(min(t[vv]), 3), "max_ms": round(max(t[vv]), 3),
                             "tflops": round(flops / m / 1e9, 1), "speedup_vs_first": round(base / m, 3)})
    rows.append(row)
    print(json.dumps({"Tq": Tq, "Tkv": Tkv, "data": a.data, "round": row.get("fp16_round_rel")}), flush=True)
    for vv in var:
        print(f"   v{vv:<3d} {json.dumps(row['v'][vv])}", flush=True)
    del q, k, v, ref, o; torch.cuda.empty_cache()
if a.json: json.dump(rows, open(a.json, "w"), indent=1)
