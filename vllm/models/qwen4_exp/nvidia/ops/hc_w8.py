# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W8A16 (group-wise symmetric int8 weight, fp16 activation) GEMM for the HC projections.

Flash-Next's hyper-connection projections (``input_mix_weight_down[_block_inject]`` 10240 ->
320(+pad) and ``input_mix_weight_up`` 320 -> 10240, 100 modules) stay F16 in the EXL3 pack and
are replicated on every rank, so at one stream they are ~46% of the weight bytes each card reads
per decode token. Int8 with one fp16 scale per GROUP input elements halves those bytes (and the
VRAM they hold); measured on the real weights at GROUP=64: ~0.7% relative weight/output error,
min output cosine 0.99994. Opt-in: ``VLLM_QWEN4_HC_W8=1`` (see ``convert_hc_weights_w8``).

y[m, n] = sum_g s[n, g] * sum_{k in g} x[m, k] * q[n, k]: the scale multiplies each group's
fp32 partial dot, so the int8 -> fp16 operand conversion is exact and no per-element scaling is
needed. Split-K (deterministic: partials summed by torch, no atomics) keeps the narrow down
projection (N = 336) from leaving most SMs idle at small M.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

HC_W8_GROUP = 64
# Kill switch for the one-token GEMV path below (0: every M uses the tiled GEMM).
_GEMV = __import__("os").environ.get("VLLM_QWEN4_HC_W8_GEMV", "1") != "0"

# Only what varies at runtime: M (the token count; specializing it on divisibility by 16 compiled a new variant per
# batch-size class mid-serving, W2j saw 4) and x's alignment (x may be a view at any offset). N, K and the strides
# are per-projection constants: their divisibility is what lets Triton vectorize the int8 weight loads, which are
# nearly all the traffic. Unspecializing them too (W2l ramp-hcw8op) cut C1 from 23.4 to 15.2 tok/s.
_W8_NO_SPECIALIZE = ("x_ptr", "M")


@triton.jit(do_not_specialize=_W8_NO_SPECIALIZE)
def _w8a16_gemm_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_qn,
    stride_sn,
    stride_os,
    stride_om,
    GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, GROUP)
    m_ok = rm < M
    n_ok = rn < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_start = pid_k * K_PER_SPLIT
    for kk in range(0, K_PER_SPLIT, GROUP):
        k = k_start + kk
        k_ok = (k + rk) < K
        x = tl.load(
            x_ptr + rm[:, None] * stride_xm + (k + rk)[None, :],
            mask=m_ok[:, None] & k_ok[None, :],
            other=0.0,
        )
        q = tl.load(
            q_ptr + rn[None, :] * stride_qn + (k + rk)[:, None],
            mask=n_ok[None, :] & k_ok[:, None],
            other=0,
        )
        part = tl.dot(x, q.to(x.dtype))  # [BLOCK_M, BLOCK_N] fp32, int8 -> fp16 exact
        s = tl.load(s_ptr + rn * stride_sn + k // GROUP, mask=n_ok & (k < K), other=0.0)
        acc += part * s.to(tl.float32)[None, :]
    tl.store(
        out_ptr + pid_k * stride_os + rm[:, None] * stride_om + rn[None, :],
        acc,
        mask=m_ok[:, None] & n_ok[None, :],
    )


# One-token path (M <= _GEMV_MAX_M: decode at low concurrency). W2o traced the GEMM above at ~110 us per call at C1
# whatever the projection (3.4 MB, ~30 GB/s, 21.7 ms per token over 192 calls, against ~3.5 ms for cuBLAS on the
# same matrices in F16). Here each program reads [BLOCK_N, GROUP] weight tiles whose rows are GROUP contiguous int8
# (K is the contiguous axis in memory and of the tile), reduces with tl.sum instead of a 16-row tl.dot, and splits K
# only by divisors of the group count, so no K mask stops the loads from vectorizing.
_GEMV_MAX_M = 4
_GEMV_BLOCK_N = 32


@triton.jit(do_not_specialize=_W8_NO_SPECIALIZE)
def _w8a16_gemv_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_qn,
    stride_sn,
    stride_os,
    stride_om,
    GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    M_ROWS: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
) -> None:
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, GROUP)
    rm = tl.arange(0, M_ROWS)
    n_ok = rn < N
    m_ok = rm < M
    acc = tl.zeros((M_ROWS, BLOCK_N), dtype=tl.float32)
    k_start = pid_k * K_PER_SPLIT
    for kk in range(0, K_PER_SPLIT, GROUP):
        k = k_start + kk
        q = tl.load(q_ptr + rn[:, None] * stride_qn + (k + rk)[None, :], mask=n_ok[:, None], other=0)
        x = tl.load(x_ptr + rm[:, None] * stride_xm + (k + rk)[None, :], mask=m_ok[:, None], other=0.0)
        sc = tl.load(s_ptr + rn * stride_sn + k // GROUP, mask=n_ok, other=0.0).to(tl.float32)
        part = tl.sum(x.to(tl.float32)[:, None, :] * q.to(tl.float32)[None, :, :], axis=2)  # [M_ROWS, BLOCK_N]
        acc += part * sc[None, :]
    tl.store(
        out_ptr + pid_k * stride_os + rm[:, None] * stride_om + rn[None, :],
        acc,
        mask=m_ok[:, None] & n_ok[None, :],
    )


def _gemv_rows(m: int) -> int:
    """The M_ROWS bucket (a constexpr) for m <= _GEMV_MAX_M tokens: 1, 2 or 4."""
    return 1 if m <= 1 else (2 if m <= 2 else 4)


def _gemv_split(n: int, k: int, group: int) -> int:
    """The smallest divisor of the group count that gives about two programs per SM (68 on a 2080 Ti); every split
    then covers whole groups, so the kernel needs no K mask. The choice depends on the shape only, not on M."""
    groups = k // group
    programs = triton.cdiv(n, _GEMV_BLOCK_N)
    for split in range(1, groups + 1):
        if groups % split == 0 and programs * split >= 136:
            return split
    return groups


def _launch_gemv(x2: torch.Tensor, q: torch.Tensor, s: torch.Tensor, group: int, split: int, rows: int) -> torch.Tensor:
    m, k = x2.shape
    n = q.shape[0]
    partial = torch.empty((split, m, n), dtype=torch.float32, device=x2.device)
    _w8a16_gemv_kernel[(triton.cdiv(n, _GEMV_BLOCK_N), split)](
        x2,
        q,
        s,
        partial,
        m,
        n,
        k,
        x2.stride(0),
        q.stride(0),
        s.stride(0),
        partial.stride(0),
        partial.stride(1),
        GROUP=group,
        BLOCK_N=_GEMV_BLOCK_N,
        M_ROWS=rows,
        K_PER_SPLIT=(k // group) // split * group,
        num_warps=4,
    )
    return partial


def quantize_w8_groupwise(
    weight: torch.Tensor, group: int = HC_W8_GROUP
) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] float weight -> (int8 [N, K], fp16 scales [N, K // group]); symmetric, round to nearest."""
    n, k = weight.shape
    if k % group:
        raise ValueError(f"K={k} is not a multiple of the int8 group {group}")
    w = weight.float().view(n, k // group, group)
    scale = w.abs().amax(dim=2, keepdim=True).clamp_min(1e-8) / 127.0
    q = torch.round(w / scale).clamp_(-127, 127).to(torch.int8).view(n, k)
    return q.contiguous(), scale.view(n, k // group).to(torch.float16).contiguous()


def _split_k(m: int, n: int, k: int, block_m: int, block_n: int, group: int) -> int:
    """Split K until there are about two programs per SM (68 on a 2080 Ti), never empty."""
    programs = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
    groups = k // group
    split = 1
    while programs * split < 136 and split * 2 <= groups:
        split *= 2
    return split


_BLOCK_M = 16
_BLOCK_N = 64
_W8_WARMED: set = set()


def _launch(x2: torch.Tensor, q: torch.Tensor, s: torch.Tensor, group: int, split: int) -> torch.Tensor:
    m, k = x2.shape
    n = q.shape[0]
    k_per_split = triton.cdiv(k // group, split) * group
    partial = torch.empty((split, m, n), dtype=torch.float32, device=x2.device)
    grid = (triton.cdiv(m, _BLOCK_M), triton.cdiv(n, _BLOCK_N), split)
    _w8a16_gemm_kernel[grid](
        x2,
        q,
        s,
        partial,
        m,
        n,
        k,
        x2.stride(0),
        q.stride(0),
        s.stride(0),
        partial.stride(0),
        partial.stride(1),
        GROUP=group,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        K_PER_SPLIT=k_per_split,
        num_warps=4,
        num_stages=2,
    )
    return partial


def _warm_up(x2: torch.Tensor, q: torch.Tensor, s: torch.Tensor, group: int) -> int:
    """Compile every split-K variant this [N, K] projection can use, once per shape and dtype, at the first eager
    call (K_PER_SPLIT is a constexpr picked from the token count, so new batch sizes would otherwise JIT-compile
    mid-serving). Skipped while a CUDA graph is captured. Returns the number of launches."""
    n, k = q.shape
    key = (n, k, group, x2.dtype, str(x2.device))
    if key in _W8_WARMED:
        return 0
    if x2.is_cuda and torch.cuda.is_current_stream_capturing():
        return 0
    _W8_WARMED.add(key)
    groups = k // group
    splits, split = [], 1
    while True:  # every power of two _split_k can return: it doubles while split * 2 <= groups
        splits.append(split)
        if split * 2 > groups:
            break
        split *= 2
    seen = set()
    probe = x2.new_zeros((1, k))
    for sp in splits:
        k_per_split = triton.cdiv(groups, sp) * group
        if k_per_split in seen:
            continue
        seen.add(k_per_split)
        _launch(probe, q, s, group, sp)
    launches = len(seen)
    if _GEMV and k % group == 0:
        gsplit = _gemv_split(n, k, group)
        for rows in (1, 2, 4):  # every M_ROWS bucket _gemv_rows returns
            _launch_gemv(x2.new_zeros((rows, k)), q, s, group, gsplit, rows)
            launches += 1
    return launches


def _w8a16_linear_impl(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, group: int) -> torch.Tensor:
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    m, k = x2.shape
    n = q.shape[0]
    assert q.shape == (n, k) and s.shape == (n, k // group)
    _warm_up(x2, q, s, group)
    if _GEMV and m <= _GEMV_MAX_M:
        partial = _launch_gemv(x2, q, s, group, _gemv_split(n, k, group), _gemv_rows(m))
        split = partial.shape[0]
    else:
        split = _split_k(m, n, k, _BLOCK_M, _BLOCK_N, group)
        partial = _launch(x2, q, s, group, split)
    y = partial[0] if split == 1 else partial.sum(dim=0)  # deterministic reduction
    return y.to(x.dtype).reshape(*lead, n)


def _w8a16_linear_fake(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, group: int) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], q.shape[0]))


# Opaque to torch.compile, like every other HC op here (ops/hc.py, low_latency_gemm.py): the split-K choice is
# Python on the token count, which Dynamo would otherwise specialize or break the graph on. W2j measured the raw
# call path at C1 23.4 tok/s against 44.5 for the fp16 projections.
direct_register_custom_op(
    op_name="qwen4_exp_hc_w8a16_linear",
    op_func=_w8a16_linear_impl,
    fake_impl=_w8a16_linear_fake,
)


def w8a16_linear(
    x: torch.Tensor,
    q: torch.Tensor,
    s: torch.Tensor,
    group: int = HC_W8_GROUP,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """y = x @ dequant(q, s).T for x [..., K] fp16/bf16, q int8 [N, K], s fp16 [N, K // group]."""
    y = torch.ops.vllm.qwen4_exp_hc_w8a16_linear(x, q, s, group)
    return y if out_dtype is None or out_dtype == y.dtype else y.to(out_dtype)


class HCW8LinearMethod:
    """Drop-in ``quant_method`` for an HC Linear after ``convert_hc_weights_w8``."""

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None):
        y = w8a16_linear(x, layer.hc_w8_q, layer.hc_w8_s, HC_W8_GROUP, x.dtype)
        return y if bias is None else y + bias


__all__ = ["HC_W8_GROUP", "HCW8LinearMethod", "quantize_w8_groupwise", "w8a16_linear"]
