#!/usr/bin/env python
"""Lane K8: time the production GDN spec-decode recurrent kernel (fused_sigmoid_gating_delta_rule_update) at TP2 shapes
(per rank: H=8 qk heads, HV=24 v heads, K=V=128, MTP3 => 4 tokens/seq, 1 state read + 4 per-token state writes).
Tiny VRAM footprint (prod engine owns the GPUs).  Usage: CUDA_VISIBLE_DEVICES=1 python bench_gdn_decode.py"""
import os, sys, time, json
sys.path.insert(0, "/home/kevin/Desktop/wt-k8")
import torch
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import fused_sigmoid_gating_delta_rule_update as fn

H, HV, K, V, TS = 8, 24, 128, 128, 4
dev = "cuda"


def bench(N, dtype, reps=60, warm=10):
    slots = N * TS + 2
    st = torch.randn(slots, HV, V, K, device=dev).to(dtype) * 0.1
    q = torch.randn(1, N * TS, H, K, device=dev, dtype=torch.float16)
    k = torch.randn(1, N * TS, H, K, device=dev, dtype=torch.float16)
    v = torch.randn(1, N * TS, HV, V, device=dev, dtype=torch.float16)
    a = torch.randn(N * TS, HV, device=dev, dtype=torch.float16)
    b = torch.randn(N * TS, HV, device=dev, dtype=torch.float16)
    A_log = torch.randn(HV, device=dev)
    dtb = torch.randn(HV, device=dev)
    cu = torch.arange(N + 1, device=dev, dtype=torch.int32) * TS
    idx = (torch.arange(N * TS, device=dev, dtype=torch.int32) + 1).reshape(N, TS)
    nacc = torch.randint(1, TS + 1, (N,), device=dev, dtype=torch.int32)
    call = lambda: fn(A_log=A_log, a=a, b=b, dt_bias=dtb, q=q, k=k, v=v, initial_state=st, inplace_final_state=True,
                      cu_seqlens=cu, ssm_state_indices=idx, num_accepted_tokens=nacc, use_qk_l2norm_in_kernel=True,
                      null_block_id=-1)
    for _ in range(warm):
        call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); call(); e1.record(); e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000)
    ts.sort()
    byt = N * (1 + TS) * HV * V * K * st.element_size()
    return ts[len(ts) // 2], ts[0], byt


if __name__ == "__main__":
    for dt in (torch.float16, torch.float32):
        for N in (1, 4, 12, 16):
            med, mn, byt = bench(N, dt)
            print(f"{str(dt):14s} N={N:2d} median {med:7.1f} us min {mn:7.1f} us  state bytes {byt/1e6:6.1f} MB  -> {byt/med/1e3:6.1f} GB/s (median)  x48 layers = {med*48/1e3:.2f} ms/step", flush=True)
        print(torch.cuda.max_memory_allocated() / 2**20, "MiB peak")
