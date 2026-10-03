#!/usr/bin/env python3
"""K5 step 2: batched TurboQuant spec-continuation attention vs the production per-request loop.

Builds a real k3v4_nc cache with the production store kernel (Hadamard rotation, Lloyd-Max centroids), then runs
TurboQuantAttentionImpl._spec_decode_attention_raw_current with the loop (K5 off) and with the batched path (K5 on)
on the same inputs; compares outputs, then CUDA-graph-times both.  Shapes = Qwen3.8-27B per TP rank attention
(Hq=12, Hk=2, D=256), MTP-3 (QL=4).  Needs <~400 MB.  Run with PYTHONPATH=<wt-k5>.  Exit 0 = PASS."""
import json
import os
import sys
import types

import torch

from vllm.model_executor.layers.quantization.turboquant.centroids import get_centroids
from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
from vllm.v1.attention.backends import turboquant_attn as ta
from vllm.v1.attention.ops import k5_tq_batched as k5
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store

dev = torch.device("cuda")
torch.manual_seed(0)
HQ, HK, D, QL, BS = 12, 2, 256, 4, int(os.getenv("K5_BS", "1856"))
cfg = TurboQuantConfig.from_cache_dtype("turboquant_k3v4_nc", D)
H = ta._build_hadamard(D, "cuda")
cent = get_centroids(D, cfg.centroid_bits).to(device=dev, dtype=torch.float32)
cs, _ = cent.sort()
mid = (cs[:-1] + cs[1:]) / 2

impl = types.SimpleNamespace(tq_config=cfg, scale=D ** -0.5, max_num_kv_splits=int(os.getenv("VLLM_TURBOQUANT_MAX_KV_SPLITS", "128")))
impl._spec_continuation_decode_attention = types.MethodType(ta.TurboQuantAttentionImpl._spec_continuation_decode_attention, impl)
impl._k5_spec_batched = types.MethodType(ta.TurboQuantAttentionImpl._k5_spec_batched, impl)
run = types.MethodType(ta.TurboQuantAttentionImpl._spec_decode_attention_raw_current, impl)


def make(N, ctx):
    blocks_per = [(c + BS - 1) // BS for c in ctx]
    nb = sum(blocks_per) + 1
    kv = torch.zeros(nb, BS, HK, cfg.slot_size_aligned, dtype=torch.uint8, device=dev)
    maxb = max(blocks_per)
    bt = torch.zeros(N, maxb, dtype=torch.int32, device=dev)
    nxt = 1
    for i, b in enumerate(blocks_per):
        bt[i, :b] = torch.arange(nxt, nxt + b, dtype=torch.int32, device=dev)
        nxt += b
    for i, c in enumerate(ctx):  # store the whole context (prefix + current chunk), as do_kv_cache_update does
        pos = torch.arange(c, device=dev)
        slots = (bt[i, pos // BS].long() * BS + pos % BS).to(torch.int32)
        k = torch.randn(c, HK, D, device=dev).half()
        v = torch.randn(c, HK, D, device=dev).half()
        triton_turboquant_store(k, v, kv, slots, H, mid, mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
                                value_quant_bits=cfg.effective_value_quant_bits, key_fp8=cfg.key_fp8)
    seq_lens = torch.tensor(ctx, dtype=torch.int32, device=dev)
    qsl = torch.arange(0, N * QL + 1, QL, dtype=torch.int32)
    md = types.SimpleNamespace(query_start_loc_cpu=qsl, query_start_loc=qsl.to(dev), seq_lens=seq_lens, block_table=bt,
                               max_seq_len=max(ctx), max_query_len=QL)
    q = (torch.randn(N * QL, HQ, D, device=dev) * 0.5).half()
    kc = torch.randn(N * QL, HK, D, device=dev).half()
    vc = torch.randn(N * QL, HK, D, device=dev).half()
    return q, kc, vc, kv, md


def call(on, args):
    k5._ENABLED = on
    q, kc, vc, kv, md = args
    return run(q, kc, vc, kv, md, H, cent, H)


def gtime(fn, iters=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / iters)
    return sorted(ts)[2]


res = {"cases": []}
ok = True
for N, ctx_each in ((1, 2000), (4, 2000), (12, 2000), (4, 28000), (12, 8000)):
    ctx = [ctx_each + 37 * i for i in range(N)]
    args = make(N, ctx)
    ref = call(False, args)
    new = call(True, args)
    torch.cuda.synchronize()
    diff = (ref.float() - new.float()).abs().max().item()
    nan = bool(torch.isnan(new).any())
    good = (not nan) and diff <= 1e-3
    ok &= good
    t_loop = gtime(lambda: call(False, args))
    t_bat = gtime(lambda: call(True, args))
    row = dict(N=N, ctx=ctx_each, maxabs=diff, bitwise=bool(diff == 0.0), nan=nan, loop_us=round(t_loop, 1),
               batched_us=round(t_bat, 1), saved_us_per_step_16_layers=round(16 * (t_loop - t_bat), 1), pass_=good)
    res["cases"].append(row)
    print(row, flush=True)
    del args
    torch.cuda.empty_cache()
res["pass"] = bool(ok)
out = os.environ.get("K5_OUT2", "/home/kevin/projects/lanes/k5/test_tq_batched.json")
json.dump(res, open(out, "w"), indent=1)
print("PASS" if ok else "FAIL", out)
sys.exit(0 if ok else 1)
