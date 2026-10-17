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

HC_W8_GROUP = 64


@triton.jit
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


def w8a16_linear(
    x: torch.Tensor,
    q: torch.Tensor,
    s: torch.Tensor,
    group: int = HC_W8_GROUP,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """y = x @ dequant(q, s).T for x [..., K] fp16/bf16, q int8 [N, K], s fp16 [N, K // group]."""
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    m, k = x2.shape
    n = q.shape[0]
    assert q.shape == (n, k) and s.shape == (n, k // group)
    block_m = 16
    block_n = 64
    split = _split_k(m, n, k, block_m, block_n, group)
    k_per_split = triton.cdiv(k // group, split) * group
    partial = torch.empty((split, m, n), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n), split)
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
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        K_PER_SPLIT=k_per_split,
        num_warps=4,
        num_stages=2,
    )
    y = partial[0] if split == 1 else partial.sum(dim=0)  # deterministic reduction
    return y.to(out_dtype or x.dtype).reshape(*lead, n)


class HCW8LinearMethod:
    """Drop-in ``quant_method`` for an HC Linear after ``convert_hc_weights_w8``."""

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None):
        y = w8a16_linear(x, layer.hc_w8_q, layer.hc_w8_s, HC_W8_GROUP, x.dtype)
        return y if bias is None else y + bias


__all__ = ["HC_W8_GROUP", "HCW8LinearMethod", "quantize_w8_groupwise", "w8a16_linear"]
