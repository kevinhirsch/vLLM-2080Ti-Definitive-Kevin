#!/usr/bin/env python
"""Lane K7 / L88: is integer math worth it for lm_head / MTP at DECODE row counts?
Times the production int4 lm_head path (Marlin W4A16 g128, the live U2b arm) at M = decode rows (batch x (1+k)),
against the W4A4 s4 kernel (k7) and the weight-bandwidth floor (bytes / measured copy bandwidth).
Uses a vocab slice of N rows (default 32768 of the ~124k per-rank vocab) to stay small next to the live engine.
Usage: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/kevin/Desktop/wt-integrate/tools/u2 python tools/k7/head_decode_bench.py"""
import argparse, json, os, statistics, subprocess, sys
ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=32768); ap.add_argument("--K", type=int, default=5120)
ap.add_argument("--Ms", default="1,4,8,16,32,48,64,96,128,192,256")
ap.add_argument("--iters", type=int, default=40)
ap.add_argument("--min-free-mib", type=int, default=600); ap.add_argument("--cap-mib", type=int, default=450)
ap.add_argument("--json", default="/home/kevin/projects/lanes/k7/head_decode_bench.json")
a = ap.parse_args()
gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
free = int(subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).decode())
if free < a.min_free_mib:
    sys.exit(f"refusing: GPU{gpu} {free} MiB free")
import torch
torch.cuda.set_per_process_memory_fraction(a.cap_mib / (torch.cuda.get_device_properties(0).total_memory / 2**20))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k7ext
from _ctlayer import make_marlin_layer
E = k7ext.ext()
N, K = a.N, a.K
t = {"weight_packed": torch.randint(-2**31, 2**31 - 1, (N, K // 8), dtype=torch.int32),
     "weight_scale": (torch.rand(N, K // 128) * 1e-2).half(),
     "weight_zero_point": torch.randint(-2**31, 2**31 - 1, (N // 8, K // 128), dtype=torch.int32)}
l16, s16 = make_marlin_layer(t, N, K, device="cuda")
B4 = torch.randint(-128, 128, (N, K // 2), dtype=torch.int8, device="cuda"); sb = torch.rand(N, device="cuda")
# measured achievable read bandwidth (copy of a buffer larger than L2)
buf = torch.empty(64 * 2**20, dtype=torch.uint8, device="cuda"); dst = torch.empty_like(buf)


def timeit(fn):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); fn(); e.record(); e.synchronize(); return s.elapsed_time(e)


for _ in range(5): dst.copy_(buf)
cp = min(timeit(lambda: dst.copy_(buf)) for _ in range(20))
bw = 2 * buf.numel() / (cp * 1e-3)  # read+write bytes/s
wbytes = N * K / 2 + N * (K // 128) * 2
floor_ms = wbytes / bw * 1e3
print(f"copy bandwidth {bw/1e9:.0f} GB/s; int4 weight bytes {wbytes/2**20:.1f} MiB -> floor {floor_ms:.3f} ms", flush=True)
rows = []
for M in [int(m) for m in a.Ms.split(",")]:
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    fns = {"w4a16": lambda: s16.apply_weights(l16, x, None),
           "w4a4": lambda: E.w4a4_gemm(*E.act_quant_h128(x, 7.0), B4, sb, 4) if False else None}
    A4, sa = E.act_quant_h128(x, 7.0)
    best = {}
    for cfg in (4, 2, 0):
        try:
            E.w4a4_gemm(A4, B4, sa, sb, cfg)
            best[cfg] = min(timeit(lambda: E.w4a4_gemm(A4, B4, sa, sb, cfg)) for _ in range(6))
        except RuntimeError:
            pass
    cfg = min(best, key=best.get)
    fns = {"w4a16": lambda: s16.apply_weights(l16, x, None), "w4a4": lambda: E.w4a4_gemm(A4, B4, sa, sb, cfg),
           "aq": lambda: E.act_quant_h128(x, 7.0)}
    for f in fns.values():
        for _ in range(3): f()
    ts = {k: [] for k in fns}
    for _ in range(a.iters):
        for k, f in fns.items(): ts[k].append(timeit(f))
    r = {"M": M, "cfg": cfg, **{k: min(v) for k, v in ts.items()}, **{k + "_med": statistics.median(v) for k, v in ts.items()},
         "floor": floor_ms}
    r["w4a16_over_floor"] = r["w4a16"] / floor_ms
    r["w4a4_total"] = r["w4a4"] + r["aq"]
    rows.append(r)
    print(f"M={M:4d} w4a16 {r['w4a16']:.3f} ms ({r['w4a16_over_floor']:.2f}x floor, {2*M*N*K/r['w4a16']/1e9:5.1f} TF) | "
          f"w4a4 {r['w4a4']:.3f} + aq {r['aq']:.3f} (cfg{cfg}) | ratio w4a16/w4a4tot {r['w4a16']/r['w4a4_total']:.2f}x", flush=True)
json.dump({"N": N, "K": K, "bw_GBs": bw / 1e9, "floor_ms": floor_ms, "rows": rows,
           "gpu": subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=clocks.sm,utilization.gpu", "--format=csv,noheader"]).decode().strip()},
          open(a.json, "w"), indent=1)
