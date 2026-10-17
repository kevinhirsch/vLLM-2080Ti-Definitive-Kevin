#!/usr/bin/env python3
"""GPU microbenchmark: the HC W8A16 kernels against cuBLAS F16 on Flash-Next's real HC projection shapes.

No engine: one GPU, about 30 s. For each projection ([N, K] = [336, 10240] down incl. block-inject + pad, and
[10240, 320] up) and each token count M, times
  f16   torch.nn.functional.linear on the F16 weight (cuBLAS GEMV/GEMM): what the HC runs today without W8
  gemm  the tiled W8A16 kernel (_launch + the split-K sum), the only W8 path before the GEMV
  gemv  the one-token W8A16 kernel (_launch_gemv + sum), M <= _GEMV_MAX_M only
as the median of 5 batches of 200 calls after a warm-up, and reports us per call, the weight bytes each path reads
and the GB/s that implies, plus the max relative error of each W8 path against the dequantized F16 reference.
W2o's trace measured the gemm at ~110 us per call at M=1 (about 30 GB/s); the floor at ~515 GB/s is about 7 us.

usage: bench_hc_w8.py <out.json> [--m 1,2,4,8,16] [--iters 200]
PYTHONPATH must hold a vLLM release with vllm.models.qwen4_exp.nvidia.ops.hc_w8; run with the engine stopped.
"""
import argparse
import json
import statistics
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--m", default="1,2,4,8,16")
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()
    from vllm.models.qwen4_exp.nvidia import model as _model  # noqa: F401  (import order: avoids the qsa cycle)
    from vllm.models.qwen4_exp.nvidia.ops import hc_w8 as W

    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    g = torch.Generator(device="cpu").manual_seed(0)
    res = {"device": torch.cuda.get_device_name(0), "group": W.HC_W8_GROUP, "cases": []}

    def timed(fn) -> float:
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        batches = []
        for _ in range(5):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(a.iters):
                fn()
            end.record()
            torch.cuda.synchronize()
            batches.append(start.elapsed_time(end) * 1e3 / a.iters)
        return statistics.median(batches)

    for n, k in ((336, 10240), (10240, 320)):
        w = (torch.randn(n, k, generator=g) * 0.05).to(dev)
        q, s = W.quantize_w8_groupwise(w, W.HC_W8_GROUP)
        w16 = (q.float() * s.float().repeat_interleave(W.HC_W8_GROUP, dim=1)).half()  # what the W8 paths compute
        W._W8_WARMED.clear()
        W._warm_up(torch.zeros(1, k, dtype=torch.float16, device=dev), q, s, W.HC_W8_GROUP)
        for m in [int(x) for x in a.m.split(",")]:
            x = torch.randn(m, k, generator=g).half().to(dev)
            ref = torch.nn.functional.linear(x, w16).float()

            def f16():
                return torch.nn.functional.linear(x, w16)

            split = W._split_k(m, n, k, W._BLOCK_M, W._BLOCK_N, W.HC_W8_GROUP)

            def gemm():
                return W._launch(x, q, s, W.HC_W8_GROUP, split).sum(dim=0)

            row = {"n": n, "k": k, "m": m, "f16_bytes": n * k * 2, "w8_bytes": n * k + s.numel() * 2}
            for name, fn, nbytes in (("f16", f16, row["f16_bytes"]), ("gemm", gemm, row["w8_bytes"])):
                us = timed(fn)
                row[f"{name}_us"] = round(us, 2)
                row[f"{name}_gb_s"] = round(nbytes / (us * 1e-6) / 1e9, 1)
            row["gemm_max_rel_err"] = float(((gemm() - ref).abs().max() / ref.abs().max()).item())
            if m <= W._GEMV_MAX_M:
                gsplit, rows = W._gemv_split(n, k, W.HC_W8_GROUP), W._gemv_rows(m)

                def gemv():
                    return W._launch_gemv(x, q, s, W.HC_W8_GROUP, gsplit, rows).sum(dim=0)

                us = timed(gemv)
                row["gemv_us"] = round(us, 2)
                row["gemv_gb_s"] = round(row["w8_bytes"] / (us * 1e-6) / 1e9, 1)
                row["gemv_split"] = gsplit
                row["gemv_max_rel_err"] = float(((gemv() - ref).abs().max() / ref.abs().max()).item())
            res["cases"].append(row)
            print(json.dumps(row), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    worst = max((r.get("gemv_max_rel_err", 0.0) for r in res["cases"]), default=0.0)
    print("RESULT", "PASS" if worst < 1e-2 else f"FAIL (gemv max rel err {worst:.3g})")
    return 0 if worst < 1e-2 else 1


if __name__ == "__main__":
    sys.exit(main())
