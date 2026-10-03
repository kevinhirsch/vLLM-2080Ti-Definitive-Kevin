#!/usr/bin/env python3
"""K5: correctness + microbench of the sm_75 fused GDN post-conv MTP decode kernel vs the stock fp16 path.

Reference = what production runs today on fp16/sm_75: Triton fused_sigmoid_gating_delta_rule_update (null id PAD=-1)
followed by RMSNormGated.forward_static (norm_before_gate, silu).  Shapes = Qwen3.8-27B per TP rank (H=8, HV=24,
K=V=128), MTP-3 (W=4 rows/request).  Needs ~200 MB of GPU memory.  Run with PYTHONPATH=<wt-k5>.
Exit 0 = PASS.
"""
import json
import os
import sys
import time

import torch

from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.mamba.gdn import gdn_sm75_cuda as k5
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import fused_sigmoid_gating_delta_rule_update

H, HV, K, V, W = 8, 24, 128, 128, 4
QKV = 2 * H * K + HV * V
dev = torch.device("cuda")
ext = k5._load()
torch.manual_seed(0)


def make(N, sdt, invalid_req=None):
    T = N * W
    buf = (torch.randn(T, QKV + HV * V, device=dev) * 0.5).half()
    mixed, z = buf[:, :QKV], buf[:, QKV:].reshape(T, HV, V)
    ba = torch.randn(T, 2 * HV, device=dev).half()
    b, a = ba.chunk(2, dim=-1)
    A_log = (torch.randn(HV, device=dev) * 0.5).float()
    dt_bias = torch.randn(HV, device=dev).half()
    nslots = 2 + N * W
    state = (torch.randn(nslots, HV, V, K, device=dev) * 0.05).to(sdt)
    sidx = (torch.arange(N * W, device=dev, dtype=torch.int32) + 1).view(N, W).contiguous()
    if invalid_req is not None:
        sidx[invalid_req] = -1
    cu = torch.arange(0, T + 1, W, device=dev, dtype=torch.int32)
    nacc = torch.randint(1, W + 1, (N,), device=dev, dtype=torch.int32)
    w = (1 + 0.1 * torch.randn(V, device=dev)).half()
    return dict(dt_f32=dt_bias.float(), w_f32=w.float(), mixed=mixed, z=z, a=a, b=b, A_log=A_log, dt_bias=dt_bias, state=state, sidx=sidx, cu=cu, nacc=nacc,
                w=w, T=T, N=N)


def stock(d, state):
    T = d["T"]
    q, k_, v = torch.split(d["mixed"], [H * K, H * K, HV * V], dim=-1)
    fused = torch.cat([q.reshape(-1), k_.reshape(-1), v.reshape(-1)])
    q = fused[: T * H * K].view(1, T, H, K)
    k_ = fused[T * H * K: 2 * T * H * K].view(1, T, H, K)
    v = fused[2 * T * H * K:].view(1, T, HV, V)
    o, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=d["A_log"], a=d["a"].contiguous(), b=d["b"].contiguous(), dt_bias=d["dt_bias"], q=q, k=k_, v=v,
        initial_state=state, inplace_final_state=True, cu_seqlens=d["cu"], ssm_state_indices=d["sidx"],
        num_accepted_tokens=d["nacc"], use_qk_l2norm_in_kernel=True, null_block_id=-1)
    core = torch.zeros(T, HV, V, device=dev, dtype=torch.half)
    core[:T] = o.squeeze(0)
    y = RMSNormGated.forward_static(core.reshape(-1, V), d["z"].reshape(-1, V), d["w"], 1e-6, torch.half, None, True,
                                    "silu")
    return y.view(T, HV, V)


def fused(d, state, variant=2):
    out = torch.zeros(d["T"], HV, V, device=dev, dtype=torch.half)
    ext.gdn_mtp(d["mixed"], d["a"], d["b"], d["A_log"], d["dt_f32"], d["sidx"], d["cu"], d["nacc"], state, d["z"],
                d["w_f32"], out, K ** -0.5, 1e-6, -1, False, variant)
    return out


def graph_time(fn, iters=300):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    t = []
    for _ in range(5):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        t.append(e0.elapsed_time(e1) * 1000 / iters)
    return sorted(t)


res = {"cases": [], "bench": []}
ok = True
for sdt in (torch.float16, torch.float32):
    for N, inv in ((1, None), (4, 2), (12, None), (16, 5)):
        d = make(N, sdt, inv)
        for var in (1, 2):
            s_ref, s_k5 = d["state"].clone(), d["state"].clone()
            y_ref = stock(d, s_ref)
            y_k5 = fused(d, s_k5, var)
            torch.cuda.synchronize()
            dy = (y_ref.float() - y_k5.float()).abs()
            rel = (dy.max() / y_ref.float().abs().max().clamp_min(1e-6)).item()
            ds = (s_ref.float() - s_k5.float()).abs().max().item()
            smag = s_ref.float().abs().max().item()
            nan = bool(torch.isnan(y_k5).any() or torch.isnan(s_k5.float()).any())
            tol_y = 2e-2
            tol_s = 2e-3 * smag + (1e-3 if sdt == torch.float16 else 1e-5)
            good = (not nan) and rel < tol_y and ds < tol_s
            if inv is not None:
                good = good and float(y_k5[inv * W:(inv + 1) * W].abs().max()) == 0.0
            ok &= good
            row = dict(variant=var, state=str(sdt).split(".")[-1], N=N, invalid_req=inv,
                       out_maxabs=round(dy.max().item(), 5), out_rel=round(rel, 5), state_maxabs=round(ds, 6),
                       state_absmax=round(smag, 4), nan=nan, pass_=good)
            res["cases"].append(row)
            print(row, flush=True)
for sdt in (torch.float16, torch.float32):
    for N in (1, 4, 12, 16):
        d = make(N, sdt)
        st = d["state"].clone()
        ts = graph_time(lambda: stock(d, st))
        t1 = graph_time(lambda: fused(d, st, 1))
        tk = graph_time(lambda: fused(d, st, 2))
        row = dict(state=str(sdt).split(".")[-1], N=N, stock_us=round(ts[2], 2), k5v1_us=round(t1[2], 2),
                   k5v2_us=round(tk[2], 2), speedup_v2=round(ts[2] / tk[2], 2),
                   saved_us_per_step_48_layers=round((ts[2] - tk[2]) * 48, 1))
        res["bench"].append(row)
        print(row, flush=True)
res["pass"] = bool(ok)
out = os.environ.get("K5_OUT", "/home/kevin/projects/lanes/k5/test_gdn_mtp.json")
json.dump(res, open(out, "w"), indent=1)
print("PASS" if ok else "FAIL", out)
sys.exit(0 if ok else 1)
