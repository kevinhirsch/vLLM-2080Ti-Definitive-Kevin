#!/usr/bin/env python
"""Lane K9: Marlin W4A16 (sm_75 = fp16 accumulate over the whole K slice, HMMA.1688.F16) output error vs an fp32
reference on REAL captured linear inputs and REAL checkpoint weights at TP2 per-rank shard shapes; cuBLAS fp16
(fp32-accumulate) and pure fp16 output rounding as baselines; fp16 partial-sum headroom; activation scale-up stress."""
import argparse, json, os, sys
import torch
sys.path.insert(0, "/home/kevin/Desktop/wt-integrate/tools/u2")
import ref_dump as R
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import awq_marlin_quantize
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
ap = argparse.ArgumentParser(); ap.add_argument("--acts", nargs="+", default=["/home/kevin/projects/lanes/k9/acts/acts.pt"])
ap.add_argument("--json", default="/home/kevin/projects/lanes/k9/marlin_err.json"); ap.add_argument("--only", default="")
a = ap.parse_args()
dev = torch.device("cuda"); ws = marlin_make_workspace_new(dev)
ROWPAR = ("down_proj", "o_proj", "out_proj")
def rel(y, r): return ((y.float() - r).norm() / r.norm()).item()
out = []
for path in a.acts:
    caps = torch.load(path)["caps"]
    for name, c in caps.items():
        if a.only and a.only not in name: continue
        w = R.dequant_linear(R.MAIN_ST, name)  # [out, in] fp32
        N, K = w.shape
        w = w[:, : K // 2] if name.endswith(ROWPAR) else w[: N // 2]
        x = c["x"]; x = x[:, : K // 2] if name.endswith(ROWPAR) else x
        wk = w.T.contiguous().half()  # [k, n]
        k, n = wk.shape
        w_ref, mq, ms, mzp = awq_marlin_quantize(wk, scalar_types.uint4, 128)[:4]
        w_ref, mq, ms, mzp = w_ref.to(dev), mq.to(dev), ms.to(dev), mzp.to(dev)
        x16 = x.to(dev).half(); m = x16.shape[0]
        row = {"name": name, "m": m, "k": k, "n": n}
        for sc in (1.0, 8.0, 64.0):
            xs = (x16.float() * sc).half()
            ref = xs.float() @ w_ref.float()
            ym = ops.marlin_gemm(xs, None, mq, None, ms, None, None, mzp, ws, scalar_types.uint4, m, n, k, use_fp32_reduce=True)
            yc = xs @ w_ref.half()
            fin = torch.isfinite(ref.half())
            d = {"ref_absmax": ref.abs().max().item(), "marlin_rel": rel(ym, ref), "cublas_rel": rel(yc, ref), "round_rel": rel(ref.half(), ref),
                 "marlin_nonfinite": int((~torch.isfinite(ym)).sum()), "ref_overflow_fp16": int((~fin).sum()),
                 "marlin_maxabs_err_over_absmax": ((ym.float() - ref)[fin].abs().max() / ref.abs().max()).item()}
            if sc == 1.0:  # fp16 partial-sum headroom: max |prefix sum| at 64-K boundaries over 1024 output columns
                cols = torch.arange(0, n, max(1, n // 1024), device=dev)[:1024]
                xf = xs.float().view(m, k // 64, 64); wf = w_ref.float()[:, cols].T.contiguous().view(len(cols), k // 64, 64)
                part = torch.einsum("mgk,ngk->mng", xf, wf).cumsum(-1)
                d["partial_absmax"] = part.abs().max().item(); d["final_absmax_subset"] = part[..., -1].abs().max().item()
                del part, xf, wf
            row[f"x{int(sc)}"] = d
        out.append(row)
        s1 = row["x1"]
        print(f"{name.split('layers.')[1]:28s} m{m} k{k} n{n} | marlin {s1['marlin_rel']:.2e} cublas {s1['cublas_rel']:.2e} round {s1['round_rel']:.2e} | "
              f"partial/final {s1['partial_absmax']:.1f}/{s1['final_absmax_subset']:.1f} | x8 m {row['x8']['marlin_rel']:.2e} nf {row['x8']['marlin_nonfinite']} | "
              f"x64 m {row['x64']['marlin_rel']:.2e} nf {row['x64']['marlin_nonfinite']} refovf {row['x64']['ref_overflow_fp16']}", flush=True)
        del w_ref, mq, ms, mzp, x16; torch.cuda.empty_cache()
json.dump(out, open(a.json, "w"), indent=1)
