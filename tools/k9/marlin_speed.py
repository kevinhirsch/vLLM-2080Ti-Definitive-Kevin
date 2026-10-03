#!/usr/bin/env python
"""Lane K9: stock Marlin W4A16 (sm_75 f16-acc) GEMM throughput at TP2 per-rank prefill shapes vs the measured
f16-acc tensor roof (512 MAC/clk/SM x 68 SM x SM clock sampled via NVML during the timed loop)."""
import json, statistics, sys, threading, time
import torch, pynvml
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import awq_marlin_quantize
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
dev = torch.device("cuda"); ws = marlin_make_workspace_new(dev)
pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
SHAPES = {"gate_up": (5120, 17408), "down": (8704, 5120), "qkvz": (5120, 8192), "out_proj": (3072, 5120)}
MS = [512, 1024, 2048, 3632, 4096]
res = []
for nm, (k, n) in SHAPES.items():
    w = torch.randn(k, n, dtype=torch.half) * 0.02
    _, mq, ms, mzp = awq_marlin_quantize(w, scalar_types.uint4, 128)[:4]
    mq, ms, mzp = mq.to(dev), ms.to(dev), mzp.to(dev)
    for m in MS:
        x = torch.randn(m, k, device=dev, dtype=torch.half)
        f = lambda: ops.marlin_gemm(x, None, mq, None, ms, None, None, mzp, ws, scalar_types.uint4, m, n, k, use_fp32_reduce=True)
        for _ in range(5): f()
        torch.cuda.synchronize()
        ts, clk = [], []
        for r in range(7):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            for _ in range(10): f()
            e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) / 10)
            clk.append(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
        med = statistics.median(ts); mhz = statistics.median(clk)
        tf = 2 * m * n * k / med / 1e9; roof = 2 * 512 * 68 * mhz * 1e6 / 1e12
        row = {"shape": nm, "m": m, "k": k, "n": n, "med_ms": med, "min_ms": min(ts), "max_ms": max(ts), "tflops": tf, "sm_mhz": mhz,
               "roof_f16acc_tf": roof, "pct_roof": 100 * tf / roof}
        res.append(row); print(json.dumps(row), flush=True)
json.dump(res, open("/home/kevin/projects/lanes/k9/marlin_speed.json", "w"), indent=1)
