# SPDX-License-Identifier: Apache-2.0
"""Lane LP: W4A8-INT8 Marlin with per-(row, 128-K group) activation scales ("MX-style int8", sm_75 IMMA).

Env (all default off): VLLM_LP_W4A8G=1 together with VLLM_MARLIN_INPUT_DTYPE=int8 (prefix scope VLLM_LP_INT8_SKIP/ONLY).
VLLM_LP_W4A8G_MODE = e (default) | g
  e: int-exponent mode (LP_A8E): act int8 with scale amax_row/127 * 2^-e[row,g], e in [0, EMAX]; the kernel shifts the stock int16
     weight group scale left by (EMAX - e) before the stock IMAD, so the epilogue costs what stock W4A8 costs. EMAX = VLLM_LP_A8E_EMAX
     (default 3), int16 weight scale levels = VLLM_LP_A8E_LEVEL (default 4096 >> EMAX, keeps the stock int32 overflow headroom).
  g: float mode (LP_A8G): fp32 act scale per (row, group) x real fp16 weight scale, fp32 FMA per group partial (more precise, slower).
Why: per-token int8 (stock W4A8) loses 10-13% GEMM accuracy on mlp.down_proj (dynamic SwiGLU outliers); per-group scales fix it.
Kernel: csrc_lp/marlin_g8 (vLLM marlin_template.h + LP_A8G/LP_A8E patch), JIT-built once into .deps/lp_marlin_*_build.
"""
import os

import torch
import triton
import triton.language as tl

_ROOT = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../.."))
_SRC = os.path.join(_ROOT, "csrc_lp/marlin_g8")
_exts = {}


def enabled() -> bool:
    return os.environ.get("VLLM_LP_W4A8G", "0") == "1"


def mode() -> str:
    return os.environ.get("VLLM_LP_W4A8G_MODE", "e")


def emax() -> int:
    return int(os.environ.get("VLLM_LP_A8E_EMAX", "3"))


def level() -> int:
    return int(os.environ.get("VLLM_LP_A8E_LEVEL", str(4096 >> emax())))


def load_ext(m: str | None = None, e: int | None = None):
    m = m or mode()
    e = emax() if e is None else e
    key = (m, e)
    if key not in _exts:
        from torch.utils.cpp_extension import load

        if m == "g":
            name, src, defs = "lp_marlin_g8", "lp_marlin_g8.cu", ["-DLP_A8G"]
        else:
            name, src, defs = f"lp_marlin_e8_x{e}", "lp_marlin_e8.cu", ["-DLP_A8E", f"-DLP_EMAX={e}"]
        build = os.path.realpath(os.environ.get("VLLM_LP_W4A8G_BUILD", os.path.join(_ROOT, ".deps", name + "_build")))
        os.makedirs(build, exist_ok=True)
        _exts[key] = load(name=name, sources=[os.path.join(_SRC, src)], build_directory=build, extra_include_paths=[_SRC],
                          verbose=False,
                          extra_cuda_cflags=["-O3", *defs, "--expt-relaxed-constexpr", "-std=c++17", "-lineinfo",
                                             "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                                             "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                                             "-gencode=arch=compute_75,code=sm_75"],
                          extra_cflags=["-O3", "-std=c++17"])
    return _exts[key]


@triton.jit
def _quant_g128_kernel(x_ptr, q_ptr, s_ptr, stride_xm, M, K, BM: tl.constexpr, GB: tl.constexpr):
    # float mode: one program = BM rows x GB groups of 128; per-(row,group) absmax / 127
    pm = tl.program_id(0)
    pg = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    cols = pg * GB * 128 + tl.arange(0, GB * 128)
    rmask = rows < M
    m2 = rmask[:, None] & (cols[None, :] < K)
    x = tl.load(x_ptr + rows[:, None] * stride_xm + cols[None, :], mask=m2, other=0.0).to(tl.float32)
    x3 = tl.reshape(x, (BM, GB, 128))
    s = tl.maximum(tl.max(tl.abs(x3), 2), 1e-8) / 127.0
    q = tl.extra.cuda.libdevice.rint(x3 / s[:, :, None])
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(q_ptr + rows[:, None] * K + cols[None, :], tl.reshape(q, (BM, GB * 128)).to(tl.int8), mask=m2)
    G = K // 128
    gidx = pg * GB + tl.arange(0, GB)
    tl.store(s_ptr + rows[:, None] * G + gidx[None, :], s, mask=rmask[:, None] & (gidx[None, :] < G))


@triton.jit
def _quant_e_kernel(x_ptr, q_ptr, e_ptr, amax_ptr, stride_xm, M, K, EMAX: tl.constexpr, BM: tl.constexpr, GB: tl.constexpr):
    # int-exponent mode, one program = BM rows x GB groups; row amax precomputed (amax_ptr, fp32 [M])
    pm = tl.program_id(0)
    pg = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    cols = pg * GB * 128 + tl.arange(0, GB * 128)
    rmask = rows < M
    m2 = rmask[:, None] & (cols[None, :] < K)
    G = K // 128
    gidx = pg * GB + tl.arange(0, GB)
    gmask = rmask[:, None] & (gidx[None, :] < G)
    amax = tl.maximum(tl.load(amax_ptr + rows, mask=rmask, other=1.0), 1e-8)
    x = tl.load(x_ptr + rows[:, None] * stride_xm + cols[None, :], mask=m2, other=0.0).to(tl.float32)
    x3 = tl.reshape(x, (BM, GB, 128))
    gm = tl.maximum(tl.max(tl.abs(x3), 2), 1e-30)
    e = tl.floor(tl.log2(amax[:, None] / gm))
    e = tl.minimum(tl.maximum(e, 0.0), EMAX * 1.0)
    sc = (amax[:, None] / 127.0) / tl.exp2(e)
    q = tl.extra.cuda.libdevice.rint(x3 / sc[:, :, None])
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(q_ptr + rows[:, None] * K + cols[None, :], tl.reshape(q, (BM, GB * 128)).to(tl.int8), mask=m2)
    tl.store(e_ptr + rows[:, None] * G + gidx[None, :], e.to(tl.uint8), mask=gmask)


def _gb(G):
    return 8 if G % 8 == 0 else (4 if G % 4 == 0 else (2 if G % 2 == 0 else 1))


def quant_g128(x: torch.Tensor):
    M, K = x.shape
    q = torch.empty((M, K), dtype=torch.int8, device=x.device)
    s = torch.empty((M, K // 128), dtype=torch.float32, device=x.device)
    if M > 0:
        G = K // 128
        _quant_g128_kernel[(triton.cdiv(M, 16), G // _gb(G))](x, q, s, x.stride(0), M, K, BM=16, GB=_gb(G))
    return q, s


def quant_e(x: torch.Tensor, wglob: float, e_max: int):
    M, K = x.shape
    G = K // 128
    q = torch.empty((M, K), dtype=torch.int8, device=x.device)
    e = torch.empty((M, G), dtype=torch.uint8, device=x.device)
    if M == 0:
        return q, e, torch.empty((0,), dtype=torch.float32, device=x.device)
    amax = x.abs().amax(dim=-1).float()
    _quant_e_kernel[(triton.cdiv(M, 16), G // _gb(G))](x, q, e, amax, x.stride(0), M, K, EMAX=e_max, BM=16, GB=_gb(G))
    r = amax.clamp(min=1e-8) * (wglob / 127.0 / (1 << e_max))
    return q, e, r


def process_scales_e(s_perm: torch.Tensor, lvl: int):
    """stock-style int16 group scales with `lvl` levels at the layer max; returns (int16-as-fp16 scales, float global)."""
    smax = s_perm.float().max()
    si = torch.round(s_perm.float() / smax * lvl).clamp(min=0).to(torch.int16).view(torch.float16)
    return si, float(smax / lvl)


@torch.library.custom_op("lp::w4a8g_gemm", mutates_args=())
def w4a8g_gemm(x: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor, w_zp: torch.Tensor, workspace: torch.Tensor,
               size_n: int, wglob: float, e_max: int, m: str) -> torch.Tensor:
    if m == "g":
        q, s = quant_g128(x)
        ones = torch.ones((x.shape[0],), dtype=torch.float32, device=x.device)
        return load_ext("g").gemm(q, ones, s, w_q, w_s, w_zp, workspace, size_n, True)
    q, e, r = quant_e(x, wglob, e_max)
    return load_ext("e", e_max).gemm(q, r, e, w_q, w_s, w_zp, workspace, size_n, True)


@w4a8g_gemm.register_fake
def _(x, w_q, w_s, w_zp, workspace, size_n, wglob, e_max, m):
    return x.new_empty((x.shape[0], size_n), dtype=torch.float16)
