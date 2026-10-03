# SPDX-License-Identifier: Apache-2.0
"""Lane K5 (2026-10-03), decode-megakernel step 2: batched TurboQuant speculative-continuation attention.

`TurboQuantAttentionImpl._spec_decode_attention_raw_current` (VLLM_TURBOQUANT_SPEC_CONTINUATION_DECODE_FASTPATH=1,
production) loops over requests in Python: per request and per attention layer it launches the q-rotation GEMM, the
sm_75 GQA prefix kernel (stage1+stage2), ~10 small where/expand/fill kernels and the raw-chunk merge.  At N streams
that is ~18*N launches per layer (x16 layers), each a few microseconds of fixed GPU cost, and every GQA launch covers
one sequence only.  For the uniform verifier batch (every request has the same q_len, which is what the decode CUDA
graphs replay) the same math runs batched: ONE rotation GEMM, ONE GQA call over all sequences (the kernel already takes
S sequences x QL rows with a per-sequence prefix length), ONE where-pair and ONE merge launch.

Per-sequence math is unchanged (same kernels, same per-sequence prefix lengths and split partition), so outputs are
expected to be bitwise identical to the loop.  Disabled unless VLLM_K5_TQ_BATCHED=1.
"""

from __future__ import annotations

import os

import torch

from vllm.triton_utils import tl, triton

_ENABLED = os.getenv("VLLM_K5_TQ_BATCHED", "0") == "1"


def enabled() -> bool:
    return _ENABLED


@triton.jit
def _k5_merge_batched_kernel(
    Q_ptr, K_ptr, V_ptr, Prefix_out_ptr, Prefix_lse_ptr, Out_ptr,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_pot, stride_poh, stride_pod,
    stride_plt, stride_plh,
    stride_ot, stride_oh, stride_od,
    ATTN_SCALE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    SEQ_ROWS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Same math as _tq_spec_continuation_merge_kernel; rows are grouped into sequences of SEQ_ROWS rows and the
    causal raw-chunk attention of a row only sees the rows of its own sequence."""
    row = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // KV_GROUP_SIZE
    local = row % SEQ_ROWS
    base = row - local

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    q = tl.load(Q_ptr + row * stride_qt + q_head * stride_qh + d_offs * stride_qd, mask=d_mask, other=0.0).to(
        tl.float32
    )
    current_max = -float("inf")
    current_sum = 0.0
    current_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    token_offsets = tl.arange(0, BLOCK_T)
    for start in range(0, SEQ_ROWS, BLOCK_T):
        token_idx = start + token_offsets
        token_mask = (token_idx < SEQ_ROWS) & (token_idx <= local)
        kv_offsets = (base + token_idx)[:, None] * stride_kt + kv_head * stride_kh + d_offs[None, :] * stride_kd
        key = tl.load(K_ptr + kv_offsets, mask=token_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        scores = tl.sum(key * q[None, :], axis=1) * ATTN_SCALE
        scores = tl.where(token_mask, scores, -float("inf"))
        tile_max = tl.max(scores, axis=0)
        next_max = tl.maximum(current_max, tile_max)
        current_scale = tl.exp(current_max - next_max)
        weights = tl.exp(scores - next_max)
        current_sum = current_sum * current_scale + tl.sum(weights, axis=0)
        value_offsets = (base + token_idx)[:, None] * stride_vt + kv_head * stride_vh + d_offs[None, :] * stride_vd
        value = tl.load(V_ptr + value_offsets, mask=token_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        current_acc = current_acc * current_scale + tl.sum(value * weights[:, None], axis=0)
        current_max = next_max

    current_lse = current_max + tl.log(current_sum)
    current_out = current_acc / current_sum
    prefix_lse = tl.load(Prefix_lse_ptr + row * stride_plt + q_head * stride_plh).to(tl.float32)
    prefix_out = tl.load(
        Prefix_out_ptr + row * stride_pot + q_head * stride_poh + d_offs * stride_pod, mask=d_mask, other=0.0
    ).to(tl.float32)
    merged_max = tl.maximum(prefix_lse, current_lse)
    prefix_weight = tl.exp(prefix_lse - merged_max)
    current_weight = tl.exp(current_lse - merged_max)
    denom = prefix_weight + current_weight
    output = (prefix_out * prefix_weight + current_out * current_weight) / denom
    tl.store(Out_ptr + row * stride_ot + q_head * stride_oh + d_offs * stride_od, output, mask=d_mask)


def merge_batched(query, key_chunk, value_chunk, prefix_out, prefix_lse, scale, seq_rows, output):
    rows, hq, d = query.shape
    assert rows % seq_rows == 0 and seq_rows <= 128
    _k5_merge_batched_kernel[(rows, hq)](
        query, key_chunk, value_chunk, prefix_out, prefix_lse, output,
        query.stride(0), query.stride(1), query.stride(2),
        key_chunk.stride(0), key_chunk.stride(1), key_chunk.stride(2),
        value_chunk.stride(0), value_chunk.stride(1), value_chunk.stride(2),
        prefix_out.stride(0), prefix_out.stride(1), prefix_out.stride(2),
        prefix_lse.stride(0), prefix_lse.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        ATTN_SCALE=scale,
        HEAD_DIM=d,
        KV_GROUP_SIZE=hq // key_chunk.shape[1],
        SEQ_ROWS=seq_rows,
        BLOCK_T=min(16, triton.next_power_of_2(seq_rows)),
        BLOCK_D=triton.next_power_of_2(d),
        num_warps=4,
        num_stages=1,
    )
    return output
