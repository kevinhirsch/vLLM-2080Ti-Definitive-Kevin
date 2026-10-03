#!/usr/bin/env python
"""Lane R2 -> S4: L55 microbench. READY TO RUN, NOT YET RUN (no GPU access in lane R2).  Engine must be stopped / offline window
(~1.5 GiB per GPU, saturates tensor cores).  Builds REAL Marlin layers exactly as the engine does (tools/u2/_ctlayer.py) and times, per
Qwen3.8-27B TP2 shape and prefill M:
  marlin        production Marlin GEMM (scheme.apply_weights)
  dq_kn/dq_nk   my Triton Marlin-layout -> dense fp16 dequant (tools/r2/marlin_dequant.py), output [K,N] or [N,K]
  mm16 / mm32   cublasGemmEx on the dequantised scratch, fp16-accumulate (COMPUTE_16F) / fp32-accumulate
  total16/32    dq_nk + mm16 / mm32  (= what a 'dequant to scratch for M>=1024' path costs per GEMM)
Correctness: dq_nk vs the U2b torch reference dequant of the SAME quantised weights (max |diff|, must be 0 or fp16 rounding), and rel L2 of
total16 output vs Marlin output.
Prints a per-chunk projection over 64 layers (linear layers only) and its share of the 2.8 s chunk wall (1,279 tok/s at M=3584).
Usage:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/r2/bench_l55.py --Ms 1024,2048,3584 --json /tmp/l55.json
Decision rule: go to an engine integration only if total16 <= 0.8 x marlin at M>=2048 on gate_up/down AND linear-layer projection >= 50% of the chunk.
"""
import argparse, ctypes, ctypes.util, glob, importlib.util, json, os, statistics, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); U2 = os.path.join(HERE, "..", "u2")
sys.path.insert(0, HERE); sys.path.insert(0, U2)
import marlin_dequant as md  # noqa: E402
spec = importlib.util.spec_from_file_location("u2hq", os.path.join(U2, "../../vllm/model_executor/layers/quantization/u2_headquant.py"))
hq = importlib.util.module_from_spec(spec); spec.loader.exec_module(hq)
from _ctlayer import make_marlin_layer  # noqa: E402

SHAPES = [("gate_up", 5120, 17408, 64), ("down", 8704, 5120, 64), ("gdn_qkvz", 5120, 8192, 48), ("gdn_out", 3072, 5120, 48),
          ("attn_qkv", 5120, 7168, 16), ("attn_o", 3072, 5120, 16)]
ap = argparse.ArgumentParser(); ap.add_argument("--Ms", default="1024,2048,3584"); ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--warmup", type=int, default=8); ap.add_argument("--json"); ap.add_argument("--smoke", action="store_true")
a = ap.parse_args(); dev = torch.device("cuda")
Ms = [64] if a.smoke else [int(m) for m in a.Ms.split(",")]; shapes = [("gate_up", 5120, 2048, 1)] if a.smoke else SHAPES


def _cublas():
    c = glob.glob(os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "*", "lib", "libcublas.so*")) + \
        glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libcublas.so*"))
    for p in sorted(c, reverse=True):
        try: return ctypes.CDLL(p)
        except OSError: pass
    return ctypes.CDLL(ctypes.util.find_library("cublas"))
cb = _cublas(); R16, C16, C32, TOP = 2, 64, 68, 99
cb.cublasGemmEx.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 5 + [ctypes.c_void_p] * 2 + [ctypes.c_int] * 2 + [ctypes.c_void_p, ctypes.c_int, ctypes.c_int] + \
    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
h1, h0, f1, f0 = ctypes.c_uint16(0x3C00), ctypes.c_uint16(0), ctypes.c_float(1), ctypes.c_float(0)


def gemm(x, w_nk, y, comp):  # y[M,N] = x[M,K] @ w_nk[N,K]^T
    M, K = x.shape; N = w_nk.shape[0]
    al, be = (ctypes.byref(h1), ctypes.byref(h0)) if comp == C16 else (ctypes.byref(f1), ctypes.byref(f0))
    st = cb.cublasGemmEx(ctypes.c_void_p(torch.cuda.current_blas_handle()), 1, 0, N, M, K, al, ctypes.c_void_p(w_nk.data_ptr()), R16, K,
                         ctypes.c_void_p(x.data_ptr()), R16, K, be, ctypes.c_void_p(y.data_ptr()), R16, N, comp, TOP)
    assert st == 0, st
    return y


def bench(fn):
    for _ in range(a.warmup): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(a.iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def ref_dequant(t, N, K):  # U2b torch reference, [N,K]
    sh = torch.arange(8, device=dev, dtype=torch.int32) * 4
    q = ((t["weight_packed"].unsqueeze(-1) >> sh) & 15).reshape(N, K // 128, 128)
    zp = ((t["weight_zero_point"].unsqueeze(1) >> sh.view(1, 8, 1)) & 15).reshape(N, -1)
    return ((q - zp.unsqueeze(-1)).to(torch.float16) * t["weight_scale"].unsqueeze(-1)).reshape(N, K)


rows = []; torch.manual_seed(0)
for name, K, N, nl in shapes:
    W = (torch.randn(N, K, device=dev) * 0.02).half()
    t = {k: v.to(dev) for k, v in hq.quantize_linear(W.cpu(), device="cuda").items()}
    layer, scheme = make_marlin_layer({k: v.cpu() for k, v in t.items()}, N, K, device="cuda")
    w_q, w_s, w_zp = scheme.kernel._get_weight_params(layer)
    zpa = w_zp if (w_zp is not None and w_zp.numel()) else None
    dq_ref = ref_dequant(t, N, K)
    scratch = torch.empty(N, K, device=dev, dtype=torch.float16)
    md.marlin_dequant_triton(w_q, w_s, zpa, K, N, 128, out=scratch, out_nk=True)
    maxdiff = float((scratch.float() - dq_ref.float()).abs().max())
    for M in Ms:
        x = torch.randn(M, K, device=dev).half(); x[:, :max(1, K // 512)] *= 30
        y16 = torch.empty(M, N, device=dev, dtype=torch.float16); y32 = torch.empty_like(y16)
        row = dict(shape=name, K=K, N=N, M=M, layers=nl, dq_maxdiff=maxdiff)
        row["marlin"] = bench(lambda: scheme.apply_weights(layer, x, None))
        row["dq_nk"] = bench(lambda: md.marlin_dequant_triton(w_q, w_s, zpa, K, N, 128, out=scratch, out_nk=True))
        row["mm16"] = bench(lambda: gemm(x, scratch, y16, C16)); row["mm32"] = bench(lambda: gemm(x, scratch, y32, C32))
        row["total16"] = row["dq_nk"] + row["mm16"]; row["total32"] = row["dq_nk"] + row["mm32"]
        ym = scheme.apply_weights(layer, x, None).float(); gemm(x, scratch, y16, C16)
        row["rel_l2_total16_vs_marlin"] = float((y16.float() - ym).norm() / ym.norm()); row["finite16"] = bool(torch.isfinite(y16).all())
        rows.append(row)
        print(f"{name:9s} M={M:5d} marlin {row['marlin']:7.3f} | dq {row['dq_nk']:6.3f} mm16 {row['mm16']:7.3f} mm32 {row['mm32']:7.3f} | "
              f"total16 {row['total16']:7.3f} ({row['marlin']/row['total16']:.2f}x) total32 {row['total32']:7.3f} ({row['marlin']/row['total32']:.2f}x) | "
              f"dq maxdiff {maxdiff:.1e} rel {row['rel_l2_total16_vs_marlin']:.1e} finite {row['finite16']}", flush=True)
    del W, t, layer, scratch; torch.cuda.empty_cache()
Mt = max(Ms); proj = {}
for r in rows:
    if r["M"] == Mt:
        for k in ("marlin", "total16", "total32"): proj[k] = proj.get(k, 0) + r[k] * r["layers"]
print(f"\nPROJECTION per GPU per {Mt}-token chunk, linear layers only (ms): {json.dumps({k: round(v, 1) for k, v in proj.items()})}")
chunk_ms = Mt / 1279.0 * 1000.0
print(f"chunk wall today ~{chunk_ms:.0f} ms (1,279 tok/s): marlin linear share {100*proj['marlin']/chunk_ms:.0f}%, path-16acc would cut the chunk by "
      f"{proj['marlin']-proj['total16']:.0f} ms = {100*(proj['marlin']-proj['total16'])/chunk_ms:.0f}% of wall")
if a.json: json.dump({"rows": rows, "projection_ms": proj}, open(a.json, "w"), indent=1)
