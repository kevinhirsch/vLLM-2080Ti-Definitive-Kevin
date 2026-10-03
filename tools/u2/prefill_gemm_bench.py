#!/usr/bin/env python
"""Lane U2 -> Lane S3: prefill GEMM microbench.  READY TO RUN, but ONLY while the engine is stopped / in an offline window
(it wants ~1.5 GiB per GPU and saturates the tensor cores; do not run next to the live engine).

Question: at prefill M = 3584 tokens (max-num-batched-tokens), is Marlin W4A16 leaving tensor-core throughput on the table
versus dense fp16 cuBLAS on a dequantized scratch weight, and what does fp16-ACCUMULATE (cublasGemmEx COMPUTE_16F) buy?
Turing GeForce rates (2080 Ti, 68 SM): fp16 tensor with fp32 accumulate = 1/2 rate (53.8 TF/s at 1545 MHz, RF's roof);
fp16 accumulate = full rate (107.5 TF/s).  Prefill measured: 1,279 tok/s = 68% of the fp32-acc roof (RF).

FINDING (code-read, csrc/libtorch_stable/quantization/marlin/marlin_template.h): on sm_75 Marlin ALREADY accumulates in fp16
(`use_fp16_accum` is true for fp16 activations with group-quantized int4), so the "fp16-accumulate" arm is mostly already unlocked inside
Marlin; what is left is (1) how close Marlin gets to the 107.5 TF/s fp16-acc roof at M=3584 (dequant ALU per 64-row block), (2) dense cuBLAS
on a dequantized scratch, (3) W4A8-INT8 (2x the fp16 roof).  Hence the arms below.

Arms (per shape, per M):
  marlin       production kernel: CompressedTensorsWNA16 -> MarlinLinearKernel (real vLLM objects, same packing as the engine)
  marlin_int8  (speed potential only: needs SYMMETRIC int4 weights, the checkpoint is asymmetric) W4A8-INT8 Marlin (VLLM_MARLIN_INPUT_DTYPE=int8, per-token int8 activations, int8 tensor cores: 215 TOPS peak on Turing = 2x the
               fp16-acc roof).  NOTE the engine reads that env GLOBALLY (decode too).  Naive per-token int8 with outlier channels is the quality risk.
  cublas32     dense fp16 weight, torch.matmul (cuBLAS, fp16 inputs, fp32 accumulate)           -- GEMM only
  cublas16acc  dense fp16 weight, cublasGemmEx(CUBLAS_COMPUTE_16F, GEMM_DEFAULT_TENSOR_OP)       -- GEMM only (fp16 accumulate)
  deq          int4 -> fp16 scratch dequant cost (torch ops here = upper bound; a fused kernel is bytes/BW:  0.5*N*K read + 2*N*K write)
  => arm totals:  deq+cublas32 , deq+cublas16acc  are what a "dequant to scratch for M>=512" design would cost per GEMM.
Reports ms (median of --iters, CUDA events), TFLOP/s, % of roof, and a per-prefill-chunk projection over all 64 layers.

Model shapes (TP=2 per GPU, K=input features, N=output features per partition):
  gate_up 5120->17408 (x64)   down 8704->5120 (x64)   [mlp, every layer]
  gdn in_proj_qkvz 5120->8192 (x48)   gdn out_proj 3072->5120 (x48)
  attn qkv 5120->7168 (x16)   attn o_proj 3072->5120 (x16)

Quality gate for the fp16-accumulate arm (run by this script, micro level, reported as PASS/FAIL):
  (a) no inf/nan in outputs for inputs drawn like real post-norm activations incl. outlier channels (--outlier-scale);
  (b) rel L2 error of the fp16-accumulate output vs an fp32 reference <= 4e-3 (= half a bf16 ulp, i.e. no worse than the rounding the
      model already applies between layers; the int4 weights' own error is ~10% rel, 25x larger), per shape; the ratio to the fp32-acc floor is printed;
  (c) max |y| headroom: max|y| < 20000 (fp16 max 65504) -- accumulate overflow risk grows with K and outlier channels.
  Engine-level gate (S3 does this, only if (a)-(c) PASS and speedup >= 1.15x on gate_up/down at M=3584):
  evalkit tool_call+code_exec+long_ctx must stay 60/60-equivalent (>=59/60: tool_call_020 is a known 95% item), needle_long.py --tokens 131000 and
  524K both correct, 6 frozen estate bodies greedy-replay diverging from stock only at near-tie wording, MTP acceptance (vllm:spec_decode_*)
  within -0.02 absolute, and CE (prompt_logprobs on tools/u2/ref windows) within +0.005 nats.  Rollback = env off.

Usage (inside the build venv, engine STOPPED):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/u2/prefill_gemm_bench.py --Ms 512,1024,2048,3584 --json /tmp/u2_gemm.json
Smoke (tiny, safe next to the engine only if >=1.2 GiB free):  ... --smoke
"""
import argparse, ctypes, ctypes.util, glob, importlib.util, json, os, statistics, subprocess, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location("u2hq", os.path.join(HERE, "../../vllm/model_executor/layers/quantization/u2_headquant.py"))
hq = importlib.util.module_from_spec(spec); spec.loader.exec_module(hq)

SHAPES = [  # name, K, N, layers
    ("gate_up", 5120, 17408, 64), ("down", 8704, 5120, 64),
    ("gdn_qkvz", 5120, 8192, 48), ("gdn_out", 3072, 5120, 48),
    ("attn_qkv", 5120, 7168, 16), ("attn_o", 3072, 5120, 16),
]
ROOT32, ROOT16 = 53.8, 107.5  # TF/s at 1545 MHz, 68 SMs

ap = argparse.ArgumentParser()
ap.add_argument("--Ms", default="512,1024,2048,3584"); ap.add_argument("--iters", type=int, default=40); ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--json"); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--outlier-scale", type=float, default=30.0)
ap.add_argument("--arms", default="marlin,marlin_int8,cublas32,cublas16acc,deq")
a = ap.parse_args()
arms = a.arms.split(",")
Ms = [64] if a.smoke else [int(m) for m in a.Ms.split(",")]
shapes = [("gate_up", 5120, 2048, 1)] if a.smoke else SHAPES
dev = torch.device("cuda")
if a.smoke:
    torch.cuda.set_per_process_memory_fraction(0.03)


# ---- cublasGemmEx with COMPUTE_16F via ctypes (torch exposes no fp16-accumulate switch) -------------------------------------
def _find_cublas():
    cands = glob.glob(os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "*", "lib", "libcublas.so*")) + \
            glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libcublas.so*"))
    for c in sorted(cands, reverse=True):
        try:
            return ctypes.CDLL(c)
        except OSError:
            pass
    return ctypes.CDLL(ctypes.util.find_library("cublas"))


_cublas = _find_cublas()
CUDA_R_16F, CUBLAS_COMPUTE_16F, CUBLAS_COMPUTE_32F, GEMM_DEFAULT_TENSOR_OP = 2, 64, 68, 99
CUBLAS_OP_N, CUBLAS_OP_T = 0, 1
_cublas.cublasGemmEx.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_one_h, _zero_h = ctypes.c_uint16(0x3C00), ctypes.c_uint16(0)      # __half 1.0, 0.0
_one_f, _zero_f = ctypes.c_float(1.0), ctypes.c_float(0.0)


def gemm_ex(x, w, y, compute):
    """y[M,N] = x[M,K] @ w[N,K]^T, all fp16 row-major.  Column-major view: y^T[N,M] = w[N,K] . x^T  (opA=T on w, opB=N on x)."""
    M, K = x.shape; N = w.shape[0]
    alpha, beta = (ctypes.byref(_one_h), ctypes.byref(_zero_h)) if compute == CUBLAS_COMPUTE_16F else (ctypes.byref(_one_f), ctypes.byref(_zero_f))
    st = _cublas.cublasGemmEx(ctypes.c_void_p(torch.cuda.current_blas_handle()), CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                              alpha, ctypes.c_void_p(w.data_ptr()), CUDA_R_16F, K, ctypes.c_void_p(x.data_ptr()), CUDA_R_16F, K,
                              beta, ctypes.c_void_p(y.data_ptr()), CUDA_R_16F, N, compute, GEMM_DEFAULT_TENSOR_OP)
    assert st == 0, f"cublasGemmEx status {st}"
    return y


def bench(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts), min(ts), max(ts)


def deq_to_fp16(t, N, K, out):
    """int4 pack-quantized -> fp16 [N,K] scratch (torch ops; fused kernel would be ~bytes/BW)."""
    sh = torch.arange(8, device=out.device, dtype=torch.int32) * 4
    q = ((t["weight_packed"].unsqueeze(-1) >> sh) & 15).reshape(N, K // 128, 128)
    zp = ((t["weight_zero_point"].unsqueeze(1) >> sh.view(1, 8, 1)) & 15).reshape(N, -1)
    out.view(N, K // 128, 128).copy_(((q - zp.unsqueeze(-1)).to(torch.float16) * t["weight_scale"].unsqueeze(-1)))
    return out


res = []
torch.manual_seed(0)
for (name, K, N, nlayers) in shapes:
    W = (torch.randn(N, K, device=dev) * 0.02).half()
    t = hq.quantize_linear(W.cpu(), device="cuda")
    t = {k: v.to(dev) for k, v in t.items()}
    layer = None
    if "marlin" in arms:
        from _ctlayer import make_marlin_layer
        layer, scheme = make_marlin_layer({k: v.cpu() for k, v in t.items()}, N, K, device="cuda")
    layer8 = None
    if "marlin_int8" in arms:
        from _ctlayer import make_marlin_layer
        os.environ["VLLM_MARLIN_INPUT_DTYPE"] = "int8"  # W4A8-INT8 Marlin (sm75 s8 kernels are compiled into this build); act_type is fixed at layer build
        try:
            # the int8-activation Marlin path asserts weight_type == uint4b8 (SYMMETRIC int4): the shipped checkpoint is asymmetric (zero points), so this arm
            # measures raw kernel SPEED on a symmetric re-quantization of the same random weights; using it for real needs a sym re-quant of every layer.
            t8 = hq.quantize_linear(W.cpu(), device="cuda", symmetric=True)
            layer8, scheme8 = make_marlin_layer({k: v.cpu() for k, v in t8.items()}, N, K, device="cuda", symmetric=True)
        finally:
            os.environ.pop("VLLM_MARLIN_INPUT_DTYPE", None)
    Wd = deq_to_fp16(t, N, K, torch.empty(N, K, device=dev, dtype=torch.float16))
    for M in Ms:
        x = torch.randn(M, K, device=dev).half()
        x[:, : max(1, K // 512)] *= a.outlier_scale  # a few outlier channels like real post-norm activations
        y32 = torch.empty(M, N, device=dev, dtype=torch.float16); y16 = torch.empty_like(y32)
        flops = 2.0 * M * N * K
        row = {"shape": name, "K": K, "N": N, "M": M, "layers": nlayers}
        def rec(arm, fn):
            med, mn, mx = bench(fn, a.iters, a.warmup)
            row[arm] = {"ms": med, "min": mn, "max": mx, "tflops": flops / med / 1e9}
        if "marlin" in arms:
            from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import CompressedTensorsWNA16
            rec("marlin", lambda: scheme.apply_weights(layer, x, None))
        if layer8 is not None:
            rec("marlin_int8", lambda: scheme8.apply_weights(layer8, x, None))
            ref8 = torch.matmul(x.float(), Wd.float().T); y8 = scheme8.apply_weights(layer8, x, None).float()
            row["marlin_int8_rel_err"] = ((y8 - ref8).norm() / ref8.norm()).item()
        if "cublas32" in arms:
            rec("cublas32", lambda: torch.matmul(x, Wd.T))
            rec("cublas32_ex", lambda: gemm_ex(x, Wd, y32, CUBLAS_COMPUTE_32F))
        if "cublas16acc" in arms:
            rec("cublas16acc", lambda: gemm_ex(x, Wd, y16, CUBLAS_COMPUTE_16F))
        if "deq" in arms:
            rec("deq_torch", lambda: deq_to_fp16(t, N, K, Wd))
            row["deq_fused_bound_ms"] = (0.5 * N * K + 2.0 * N * K) / 616e9 * 1e3
        # quality micro-gate for fp16 accumulate
        ref = torch.matmul(x.float(), Wd.float().T)
        y32 = gemm_ex(x, Wd, y32, CUBLAS_COMPUTE_32F).float(); y16f = gemm_ex(x, Wd, y16, CUBLAS_COMPUTE_16F).float()
        floor = ((y32 - ref).norm() / ref.norm()).item(); e16 = ((y16f - ref).norm() / ref.norm()).item()
        row["quality"] = {"rel_fp32acc": floor, "rel_fp16acc": e16, "finite": bool(torch.isfinite(y16f).all()), "max_abs_y": float(y16f.abs().max()),
                          "PASS": bool(torch.isfinite(y16f).all() and e16 <= 4e-3 and float(y16f.abs().max()) < 20000)}
        res.append(row)
        f = lambda arm: f"{row[arm]['ms']:7.3f}ms {row[arm]['tflops']:5.1f}TF" if arm in row else "   -   "
        print(f"{name:9s} M={M:5d} K={K:5d} N={N:5d} | marlin {f('marlin')} | cublas32 {f('cublas32')} | 16acc {f('cublas16acc')} | "
              f"int8 {f('marlin_int8')} | deq(torch) {row.get('deq_torch',{}).get('ms',0):6.3f}ms | q16: rel {e16:.2e} (floor {floor:.2e}) maxy {row['quality']['max_abs_y']:.0f} {'PASS' if row['quality']['PASS'] else 'FAIL'}", flush=True)
    del W, Wd, t, layer
    torch.cuda.empty_cache()

# projection per prefill chunk (sum over the 64 layers), only for the largest M
Mtop = max(Ms)
tot = {}
for r in res:
    if r["M"] != Mtop:
        continue
    for arm in ("marlin", "marlin_int8", "cublas32", "cublas16acc"):
        if arm in r:
            tot[arm] = tot.get(arm, 0) + r[arm]["ms"] * r["layers"]
    if "deq_fused_bound_ms" in r and "cublas32" in r:
        tot["deq_fused+cublas32"] = tot.get("deq_fused+cublas32", 0) + (r["deq_fused_bound_ms"] + r["cublas32"]["ms"]) * r["layers"]
        if "cublas16acc" in r:
            tot["deq_fused+cublas16acc"] = tot.get("deq_fused+cublas16acc", 0) + (r["deq_fused_bound_ms"] + r["cublas16acc"]["ms"]) * r["layers"]
print(f"\nPROJECTION per GPU per {Mtop}-token prefill chunk, linear layers only (ms): " + json.dumps({k: round(v, 1) for k, v in tot.items()}))
print("Read: chunk wall today ~ 2.8 s at 1,279 tok/s; the linear layers' share is the part these arms can change (GDN/attention/norms are separate).")
if a.json:
    json.dump({"rows": res, "projection_ms": tot, "roof_tflops": {"fp32acc": ROOT32, "fp16acc": ROOT16}}, open(a.json, "w"), indent=1)
