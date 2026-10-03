#!/usr/bin/env python3
"""EF2 / L134 discriminating probe for the TurboQuant continuation-prefill Xid 31 class.

Run once per venv (legacy tree = FlashInfer 0.6.8, integrate tree = FlashInfer 0.6.18).
Every part is single-GPU and small (< 1.5 GiB). Intended to run inside a GPU window,
preferably under compute-sanitizer memcheck with PYTORCH_NO_CUDA_MEMORY_CACHING=1 so
that every tensor is its own cudaMalloc and any out-of-bounds access is reported
precisely instead of only when it happens to cross into unmapped VA (FAULT_PDE).

Parts
  race    Proves (or refutes) the legacy pinned-indptr race: one TQ impl reuses a single
          pinned CPU indptr buffer for every FlashInfer plan(); plan() copies it to the
          device with non_blocking=True. With the stream busy, the next request's write
          into the same buffer lands before the DMA executes, so the first wrapper's
          device indptr holds the SECOND request's lengths while its host-side schedule
          was built from the first. Shapes are the 10-01 14:40 step: continuation q=843,
          kv=40091 followed by a first chunk q=1087. Also runs the integrate-tree pattern
          (a fresh pinned tensor per plan) as the control.
  poison  Runs the race-corrupted wrapper (plan for q=843/kv=40091, device indptr
          [0,1087]) on tensors of the planned shape. Under memcheck this reports the
          out-of-bounds access that the production faults hit. Never run it without
          the sanitizer (it may raise a real Xid).
  shapes  Correctly planned FlashInfer fa2 ragged prefill (Hq=12, Hk=2, D=256, fp16,
          causal) for the exact fault shapes plus a sweep, and the prefix-combine legs
          (non-causal prefix + causal current). Checks finiteness, and checks values
          against an fp32 reference where that fits in memory.
  dequant _tq_full_dequant_kv (integrate tree) at the fault shapes with the block table
          ending on the LAST block of an exactly sized pool, writing the S3 strided
          (rows, Hk, D) layout.

Output: one JSON line per check on stdout, then a summary line {"part": ..., "ok": bool}.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

HQ, HK, D = 12, 2, 256  # per-rank heads of Qwen3.x-27B at TP=2
FAULT_SHAPES = [  # (q_len, kv_len) of the TQ continuation rows in the fault steps
    (843, 40091),   # 2026-10-01 14:40 (legacy, block 3568, mixed)
    (1087, 1087),   # 2026-10-01 14:40 first-chunk row in the same step
    (3568, 32112),  # 2026-10-01 20:45 (legacy, block 3568, mixed)
    (1856, 7424),   # 2026-10-02 12:00 (legacy, block 1856, mixed)
    (1856, 66816),  # 2026-10-02 12:49 (legacy, block 1856, pure prefill)
]


def emit(**kw):
    print(json.dumps(kw, default=str), flush=True)


def fi():
    import flashinfer
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
    return flashinfer.__version__, BatchPrefillWithRaggedKVCacheWrapper


def make_wrapper(ws):
    _, W = fi()
    return W(ws, "NHD", backend="fa2")


def plan(w, qo, kv, causal=True):
    w.plan(
        qo_indptr=qo, kv_indptr=kv, num_qo_heads=HQ, num_kv_heads=HK, head_dim_qk=D,
        causal=causal, sm_scale=D ** -0.5, q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )


def busy(ms: float = 200.0):
    # Keep the stream busy so a later host write races the queued H2D copy.
    torch.cuda._sleep(int(ms * 1e6))  # ~1 GHz cycles -> roughly ms


def part_race(ws) -> bool:
    ok = True
    for trial in range(5):
        # Legacy pattern: one pinned buffer per impl, mutated per call.
        qo_cpu = torch.empty(2, dtype=torch.int32, pin_memory=True)
        kv_cpu = torch.empty(2, dtype=torch.int32, pin_memory=True)
        torch.cuda.synchronize()
        busy()
        qo_cpu[0], qo_cpu[1] = 0, 843
        kv_cpu[0], kv_cpu[1] = 0, 40091
        wa = make_wrapper(ws)
        plan(wa, qo_cpu, kv_cpu)  # enqueues non_blocking H2D from qo_cpu/kv_cpu
        # Next request in the same _prefill_attention loop (first chunk q=1087):
        qo_cpu[1] = 1087
        kv_cpu[1] = 1087
        torch.cuda.synchronize()
        dev = (wa._qo_indptr_buf.tolist(), wa._kv_indptr_buf.tolist())
        legacy_corrupt = dev != ([0, 843], [0, 40091])
        # Integrate pattern: fresh pinned tensor per plan, never mutated.
        busy()
        qo2 = torch.tensor([0, 843], dtype=torch.int32, pin_memory=True)
        kv2 = torch.tensor([0, 40091], dtype=torch.int32, pin_memory=True)
        wb = make_wrapper(ws)
        plan(wb, qo2, kv2)
        qo3 = torch.tensor([0, 1087], dtype=torch.int32, pin_memory=True)  # next request
        del qo2, kv2
        torch.cuda.synchronize()
        dev2 = (wb._qo_indptr_buf.tolist(), wb._kv_indptr_buf.tolist())
        integrate_corrupt = dev2 != ([0, 843], [0, 40091])
        emit(part="race", trial=trial, legacy_device_indptr=dev, legacy_corrupt=legacy_corrupt,
             integrate_device_indptr=dev2, integrate_corrupt=integrate_corrupt)
        ok = ok and not integrate_corrupt
        del qo3
    return ok


def part_poison(ws) -> bool:
    qo_cpu = torch.empty(2, dtype=torch.int32, pin_memory=True)
    kv_cpu = torch.empty(2, dtype=torch.int32, pin_memory=True)
    torch.cuda.synchronize()
    busy()
    qo_cpu[0], qo_cpu[1], kv_cpu[0], kv_cpu[1] = 0, 843, 0, 40091
    w = make_wrapper(ws)
    plan(w, qo_cpu, kv_cpu)
    qo_cpu[1] = 1087
    kv_cpu[1] = 1087
    torch.cuda.synchronize()
    emit(part="poison", device_indptr=(w._qo_indptr_buf.tolist(), w._kv_indptr_buf.tolist()))
    q = torch.randn(843, HQ, D, dtype=torch.float16, device="cuda")
    k = torch.randn(40091, HK, D, dtype=torch.float16, device="cuda")
    v = torch.randn(40091, HK, D, dtype=torch.float16, device="cuda")
    out = w.run(q, k, v)
    torch.cuda.synchronize()
    emit(part="poison", ran=True, out_shape=list(out.shape))
    return True


def ref_attn(q, k, v, causal_offset):
    qf = q.float().transpose(0, 1)  # (Hq, q, D)
    kf = k.float().repeat_interleave(HQ // HK, dim=1).transpose(0, 1)
    vf = v.float().repeat_interleave(HQ // HK, dim=1).transpose(0, 1)
    s = (qf @ kf.transpose(1, 2)) * D ** -0.5
    if causal_offset is not None:
        qi = torch.arange(q.shape[0], device=q.device)[:, None] + causal_offset
        ki = torch.arange(k.shape[0], device=q.device)[None, :]
        s = s.masked_fill(ki > qi, float("-inf"))
    return (torch.softmax(s, -1) @ vf).transpose(0, 1)


def run_one(ws, qlen, kvlen, causal=True, check_ref=False):
    qo = torch.tensor([0, qlen], dtype=torch.int32, pin_memory=True)
    kv = torch.tensor([0, kvlen], dtype=torch.int32, pin_memory=True)
    w = make_wrapper(ws)
    plan(w, qo, kv, causal=causal)
    q = torch.randn(qlen, HQ, D, dtype=torch.float16, device="cuda")
    k = torch.randn(kvlen, HK, D, dtype=torch.float16, device="cuda")
    v = torch.randn(kvlen, HK, D, dtype=torch.float16, device="cuda")
    out = w.run(q, k, v)
    torch.cuda.synchronize()
    rec = dict(q=qlen, kv=kvlen, causal=causal, finite=bool(torch.isfinite(out).all()))
    if check_ref:
        r = ref_attn(q, k, v, (kvlen - qlen) if causal else None)
        rec["max_abs_err"] = float((out.float() - r).abs().max())
    return rec


def part_shapes(ws, sweep: int, seed: int) -> bool:
    ok = True
    for qlen, kvlen in FAULT_SHAPES:
        rec = run_one(ws, qlen, kvlen, check_ref=qlen * kvlen <= 1087 * 8192)
        emit(part="shapes", kind="fault", **rec)
        ok = ok and rec["finite"] and rec.get("max_abs_err", 0.0) < 2e-2
    # prefix-combine legs (integrate default above 20,480 tokens)
    for qlen, kvlen in [(1856, 64960), (3568, 28544), (843, 39248)]:
        a = run_one(ws, qlen, kvlen, causal=False)
        b = run_one(ws, qlen, qlen, causal=True)
        emit(part="shapes", kind="prefix_combine", prefix=a, current=b)
        ok = ok and a["finite"] and b["finite"]
    g = torch.Generator().manual_seed(seed)
    for _ in range(sweep):
        qlen = int(torch.randint(129, 3633, (1,), generator=g))
        kvlen = qlen + int(torch.randint(0, 70000, (1,), generator=g))
        rec = run_one(ws, qlen, kvlen)
        emit(part="shapes", kind="sweep", **rec)
        ok = ok and rec["finite"]
    return ok


def part_dequant() -> bool:
    from vllm.v1.attention.ops.triton_turboquant_decode import _tq_full_dequant_kv
    import triton

    mse_bits, vqb = 3, 4  # turboquant_k3v4_nc
    mse_bytes = math.ceil(D * mse_bits / 8)
    kps = mse_bytes + 2
    val_data = math.ceil(D * vqb / 8)
    slot = kps + val_data + 4
    ok = True
    for bs, cached, qlen in [(3568, 39248, 843), (3568, 28544, 3568), (1856, 5568, 1856),
                             (1856, 64960, 1856), (3568, 3568 * 3 + 7, 200)]:
        pages = math.ceil(cached / bs)
        num_blocks = pages + 2
        kv_cache = torch.randint(0, 255, (num_blocks, bs, HK, slot), dtype=torch.uint8,
                                 device="cuda")
        # fp16 norms/scales must be finite: write small values into those byte slots
        kv_cache[..., mse_bytes:mse_bytes + 2] = torch.tensor([0, 60], dtype=torch.uint8,
                                                             device="cuda")
        kv_cache[..., kps + val_data:kps + val_data + 4] = torch.tensor(
            [0, 32, 0, 0], dtype=torch.uint8, device="cuda")
        # block table ends on the LAST block of the pool (pool-end overrun check)
        bt = torch.zeros(1, pages + 4, dtype=torch.int32, device="cuda")
        bt[0, :pages] = torch.arange(num_blocks - pages, num_blocks, dtype=torch.int32)
        alloc_len = pages * bs
        rows = max(alloc_len, math.ceil((cached + qlen) / bs) * bs)
        k_rows = torch.empty(rows, HK, D, dtype=torch.float16, device="cuda")
        v_rows = torch.empty(rows, HK, D, dtype=torch.float16, device="cuda")
        kc = k_rows.permute(1, 0, 2).unsqueeze(0)
        vc = v_rows.permute(1, 0, 2).unsqueeze(0)
        cent = torch.linspace(-1, 1, 1 << mse_bits, device="cuda", dtype=torch.float32)
        _tq_full_dequant_kv[(alloc_len, HK)](
            kv_cache, bt, cent, kc, vc,
            kc.stride(0), kc.stride(1), kc.stride(2), vc.stride(0), vc.stride(1), vc.stride(2),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2), bt.stride(0),
            HEAD_DIM=D, BLOCK_SIZE=bs, NUM_KV_HEADS=HK, MSE_BYTES=mse_bytes, KPS=kps,
            VQB=vqb, VAL_DATA_BYTES=val_data, MSE_BITS=mse_bits, KEY_FP8=0,
            BLOCK_D=triton.next_power_of_2(D), NORM_CORRECTION=1, FP8_FORMAT=0, num_warps=4,
        )
        torch.cuda.synchronize()
        fin = bool(torch.isfinite(k_rows[:cached]).all() and torch.isfinite(v_rows[:cached]).all())
        emit(part="dequant", bs=bs, cached=cached, q=qlen, rows=rows, num_blocks=num_blocks,
             finite=fin)
        ok = ok and fin
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["race", "poison", "shapes", "dequant"])
    ap.add_argument("--sweep", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    t0 = time.time()
    ver = None
    ws = None
    if a.part != "dequant":
        ver, _ = fi()
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    emit(part=a.part, flashinfer=ver, torch=torch.__version__, device=torch.cuda.get_device_name(),
         no_caching=os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"))
    if a.part == "race":
        ok = part_race(ws)
    elif a.part == "poison":
        ok = part_poison(ws)
    elif a.part == "shapes":
        ok = part_shapes(ws, a.sweep, a.seed)
    else:
        ok = part_dequant()
    emit(part=a.part, ok=ok, seconds=round(time.time() - t0, 1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
