# SPDX-License-Identifier: Apache-2.0
"""Lane LP: W4A8-INT8 Marlin with per-(row, 128-K group) activation scales ("MX-style int8", sm_75 IMMA).

Env-gated: VLLM_LP_W4A8G=1 (together with VLLM_MARLIN_INPUT_DTYPE=int8, prefix scope VLLM_LP_INT8_SKIP/ONLY).
Why: per-token int8 activations lose 10-13% GEMM output accuracy on mlp.down_proj (SwiGLU outliers are dynamic, not static
channels); a float activation scale per (row, 128-K group) brings it to 1.0-1.4% (better than FP8 E4M3 per-token, 1.4-1.9%),
and it aligns with the weight group, so the kernel applies a_scale[row,g] * w_scale[g,col] (fp16 scales, no int16 rounding)
to each int32 group partial in fp32. Kernel: csrc_lp/marlin_g8 (vLLM marlin_template.h + LP_A8G patch), JIT-built once
into .deps/lp_marlin_g8_build. The activation quantizer is a Triton kernel.
"""
import os

import torch

_ext = None
_SRC = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../../csrc_lp/marlin_g8"))
_BUILD = os.path.realpath(os.environ.get("VLLM_LP_W4A8G_BUILD", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../../.deps/lp_marlin_g8_build")))


def enabled() -> bool:
    return os.environ.get("VLLM_LP_W4A8G", "0") == "1"


def _load():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load

        os.makedirs(_BUILD, exist_ok=True)
        _ext = load(name="lp_marlin_g8", sources=[os.path.join(_SRC, "lp_marlin_g8.cu")], build_directory=_BUILD,
                    extra_include_paths=[_SRC], verbose=False,
                    extra_cuda_cflags=["-O3", "-DLP_A8G", "--expt-relaxed-constexpr", "-std=c++17", "-lineinfo",
                                       "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                                       "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                                       "-gencode=arch=compute_75,code=sm_75"],
                    extra_cflags=["-O3", "-std=c++17"])
    return _ext


import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _quant_g128_kernel(x_ptr, q_ptr, s_ptr, stride_xm, K, G: tl.constexpr):
    row = tl.program_id(0)
    grp = tl.program_id(1)
    offs = grp * G + tl.arange(0, G)
    x = tl.load(x_ptr + row * stride_xm + offs).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x), 0), 1e-8) / 127.0
    q = tl.extra.cuda.libdevice.rint(x / s)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(q_ptr + row * K + offs, q.to(tl.int8))
    tl.store(s_ptr + row * (K // G) + grp, s)


def quant_g128(x: torch.Tensor):
    M, K = x.shape
    q = torch.empty((M, K), dtype=torch.int8, device=x.device)
    s = torch.empty((M, K // 128), dtype=torch.float32, device=x.device)
    if M > 0:
        _quant_g128_kernel[(M, K // 128)](x, q, s, x.stride(0), K, G=128)
    return q, s


@torch.library.custom_op("lp::w4a8g_gemm", mutates_args=())
def w4a8g_gemm(x: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor, w_zp: torch.Tensor, workspace: torch.Tensor,
               size_n: int) -> torch.Tensor:
    q, s = quant_g128(x)
    return _load().w4a8g_gemm(q, s, w_q, w_s, w_zp, workspace, size_n, True)


@w4a8g_gemm.register_fake
def _(x, w_q, w_s, w_zp, workspace, size_n):
    return x.new_empty((x.shape[0], size_n), dtype=torch.float16)
