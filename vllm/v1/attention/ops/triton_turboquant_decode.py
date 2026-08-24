# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fused TurboQuant decode attention.

Decode path: Triton stage1 (split-KV tiled attention scoring + value
accumulation) + stage2 (log-sum-exp reduction across splits).

Supports FP8 (E4M3) keys, 3-bit and 4-bit uniform quantized values.
"""

import math
import os
from typing import Any

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_decode_attention import (
    _fwd_kernel_stage2,
)

_FP8_FORMAT_CODE: dict[int, int] = {}
_FP8_FORMAT_OVERRIDE = os.getenv(
    "VLLM_TURBOQUANT_K8V4_FP8_FORMAT",
    "auto",
).strip().lower()
if _FP8_FORMAT_OVERRIDE not in ("", "auto", "e4b15", "e4nv", "e5"):
    raise ValueError(
        "VLLM_TURBOQUANT_K8V4_FP8_FORMAT must be one of: "
        "auto, e4b15, e4nv, e5"
    )


def _read_decode_block_kv() -> int:
    value = os.getenv("VLLM_TURBOQUANT_DECODE_BLOCK_KV", "2").strip()
    try:
        block_kv = int(value)
    except ValueError as err:
        raise ValueError(
            "VLLM_TURBOQUANT_DECODE_BLOCK_KV must be one of: 1, 2, 4, 8, 16"
        ) from err
    if block_kv not in (1, 2, 4, 8, 16):
        raise ValueError(
            "VLLM_TURBOQUANT_DECODE_BLOCK_KV must be one of: 1, 2, 4, 8, 16"
        )
    return block_kv


_DECODE_BLOCK_KV = _read_decode_block_kv()

# F-2: query-row KV-tile reuse for the stage-1 spec-verify decode kernel.
# When set, MTP verify steps whose query rows all share ONE sequence's cached
# prefix (the `_spec_continuation_decode_attention` prefix leg) dispatch to
# `_tq_decode_stage1_qtile`, which loads each KV tile once and applies all
# q rows to it, instead of re-scanning the whole KV per query row. Default
# OFF so prod ships it via a gated A/B window. See
# docs/f2-tq-depth-cost-research.md.
_STAGE1_QTILE = os.getenv("VLLM_TURBOQUANT_STAGE1_QTILE", "0") == "1"


def _fp8_format_code(device: int = 0) -> int:
    """Return 0=e4nv, 1=e4b15, 2=e5 for TQ raw FP8 key kernels."""
    if _FP8_FORMAT_OVERRIDE == "e4b15":
        return 1
    if _FP8_FORMAT_OVERRIDE == "e5":
        return 2
    if _FP8_FORMAT_OVERRIDE == "e4nv":
        if current_platform.is_cuda_alike():
            cap = torch.cuda.get_device_capability(device)
            if cap < (8, 9):
                raise ValueError(
                    "VLLM_TURBOQUANT_K8V4_FP8_FORMAT=e4nv is not supported "
                    f"on CUDA capability {cap[0]}.{cap[1]}; Triton supports "
                    "e4b15 and e5 on this architecture."
                )
        return 0
    if device not in _FP8_FORMAT_CODE:
        if current_platform.is_cuda_alike():
            cap = torch.cuda.get_device_capability(device)
            if cap < (8, 0):
                # SM75/Turing supports fp8e5 and fp8e4b15 in Triton. Raw
                # post-RoPE Qwen3.6 keys need the wider e5 range; e4b15
                # severely degrades long-context retrieval quality.
                _FP8_FORMAT_CODE[device] = 2
            else:
                _FP8_FORMAT_CODE[device] = 1 if cap < (8, 9) else 0
        else:
            _FP8_FORMAT_CODE[device] = 0
    return _FP8_FORMAT_CODE[device]


def _fp8_format_name(device: int = 0) -> str:
    return ("e4nv", "e4b15", "e5")[_fp8_format_code(device)]


def _use_fp8_e4b15(device: int = 0) -> int:
    return 1 if _fp8_format_code(device) == 1 else 0


# ---------------------------------------------------------------------------
# Stage 1: Fused TQ score + value accumulation (BLOCK_KV tiled)
# ---------------------------------------------------------------------------


@triton.jit
def _tq_decode_stage1(
    # Precomputed query projection
    Q_rot_ptr,  # [B, Hq, D] float32
    # Compressed KV cache (combined K+V)
    KV_cache_ptr,  # [num_blocks, block_size, Hk, padded_slot] uint8
    # Block table and sequence info
    Block_table_ptr,  # [B, max_num_blocks] int32
    Seq_lens_ptr,  # [B] int32
    # TQ parameters
    Centroids_ptr,  # [n_centroids] float32
    # Output (intermediate for stage2)
    Mid_o_ptr,  # [B, Hq, NUM_KV_SPLITS, D+1] float32
    # Strides
    stride_qb,
    stride_qh,  # Q strides: [B, Hq, D]
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,  # KV cache
    stride_bt_b,  # block_table stride per batch
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,  # mid_o strides
    # Constexpr dims
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # KV cache block_size (pages)
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,  # Hq // Hk
    # TQ layout constants
    MSE_BITS: tl.constexpr,  # 3 or 4
    MSE_BYTES: tl.constexpr,  # ceil(D * mse_bits / 8)
    KPS: tl.constexpr,  # key_packed_size
    VQB: tl.constexpr,  # value_quant_bits (4 or 8=FP8)
    VAL_DATA_BYTES: tl.constexpr,  # ceil(D * vqb / 8) or D for FP8
    # Score constants
    ATTN_SCALE: tl.constexpr,  # 1/sqrt(D)
    SLIDING_WINDOW: tl.constexpr,  # 0 disables windowing; otherwise left window
    # Block tile sizes
    BLOCK_D: tl.constexpr,  # next_power_of_2(HEAD_DIM)
    BLOCK_KV: tl.constexpr,  # tokens per tile (16)
    KEY_FP8: tl.constexpr,  # 1 if K is stored as FP8
    NORM_CORRECTION: tl.constexpr = 0,  # 1 = re-normalize centroids
    FP8_FORMAT: tl.constexpr = 0,  # 0=e4nv, 1=e4b15, 2=e5
):
    bid = tl.program_id(0)  # batch index
    hid = tl.program_id(1)  # q_head index
    sid = tl.program_id(2)  # kv_split index

    kv_head = hid // KV_GROUP_SIZE

    # Sequence length for this batch
    seq_len = tl.load(Seq_lens_ptr + bid)
    kv_start = 0
    if SLIDING_WINDOW > 0:
        kv_start = tl.maximum(0, seq_len - SLIDING_WINDOW)

    # KV split range
    active_len = seq_len - kv_start
    split_len = tl.cdiv(active_len, NUM_KV_SPLITS)
    split_start = kv_start + split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)

    if split_start >= split_end:
        return

    # Dimension offsets
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)

    # Load query vector: q_rot — [BLOCK_D] float32
    q_base = bid * stride_qb + hid * stride_qh
    q_rot = tl.load(Q_rot_ptr + q_base + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    # Precompute byte/bit index vectors for MSE gather loads
    if not KEY_FP8:
        mse_bit_off = d_offs * MSE_BITS
        mse_byte_idx = mse_bit_off // 8
        mse_bit_shift = mse_bit_off % 8
        mse_mask = (1 << MSE_BITS) - 1

    # Precompute value bit/byte index vectors (loop-invariant)
    if VQB == 3:
        val_bit_off = d_offs * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8

    # Online softmax accumulators
    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    bt_base = bid * stride_bt_b

    # ================================================================
    # TILED LOOP: process BLOCK_KV tokens per iteration
    # ================================================================
    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx,
            mask=kv_mask,
            other=0,
        ).to(tl.int64)

        slot_bases = (
            block_nums * stride_cache_block
            + page_off.to(tl.int64) * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )

        # ============================================================
        # COMPUTE ATTENTION SCORES: [BLOCK_KV]
        # ============================================================
        if KEY_FP8:
            k_addrs = slot_bases[:, None] + d_offs[None, :]
            k_raw = tl.load(
                KV_cache_ptr + k_addrs,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            )
            if FP8_FORMAT == 1:
                k_float = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
            elif FP8_FORMAT == 2:
                k_float = k_raw.to(tl.float8e5, bitcast=True).to(tl.float32)
            else:
                k_float = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            scores = (
                tl.sum(
                    tl.where(d_mask[None, :], q_rot[None, :] * k_float, 0.0),
                    axis=1,
                )
                * ATTN_SCALE
            )
            scores = tl.where(kv_mask, scores, -float("inf"))
        else:
            # MSE unpack + norms
            mse_addrs0 = slot_bases[:, None] + mse_byte_idx[None, :]
            mse_raw0 = tl.load(
                KV_cache_ptr + mse_addrs0,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            mse_raw1 = tl.load(
                KV_cache_ptr + mse_addrs0 + 1,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw16 = mse_raw0 | (mse_raw1 << 8)
            mse_idx = (raw16 >> mse_bit_shift[None, :]) & mse_mask

            # Centroid gather + dot product
            c_vals = tl.load(
                Centroids_ptr + mse_idx,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0.0,
            )

            # Norm correction: re-normalize centroid vector to unit norm
            if NORM_CORRECTION:
                c_norm_sq = tl.sum(
                    tl.where(d_mask[None, :], c_vals * c_vals, 0.0),
                    axis=1,
                )
                c_inv_norm = 1.0 / tl.sqrt(c_norm_sq + 1e-16)
                c_vals = c_vals * c_inv_norm[:, None]

            term1 = tl.sum(
                tl.where(d_mask[None, :], q_rot[None, :] * c_vals, 0.0),
                axis=1,
            )

            # Load norms (fp16 -> fp32): norms are at MSE_BYTES offset
            norm_bases = slot_bases + MSE_BYTES
            n_lo = tl.load(KV_cache_ptr + norm_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            n_hi = tl.load(KV_cache_ptr + norm_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            vec_norms = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

            scores = vec_norms * term1 * ATTN_SCALE
            scores = tl.where(kv_mask, scores, -float("inf"))

        # ============================================================
        # ONLINE SOFTMAX UPDATE (block-level)
        # ============================================================
        n_e_max = tl.maximum(tl.max(scores, 0), m_prev)
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max)

        # ============================================================
        # VALUE LOAD + DEQUANTIZE: [BLOCK_KV, BLOCK_D]
        # ============================================================
        val_bases = slot_bases + KPS

        if VQB == 3:
            val_addrs0 = val_bases[:, None] + val_byte_idx[None, :]
            val_raw0 = tl.load(
                KV_cache_ptr + val_addrs0,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            val_raw1 = tl.load(
                KV_cache_ptr + val_addrs0 + 1,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw16 = val_raw0 | (val_raw1 << 8)
            v_idx = ((raw16 >> val_bit_shift[None, :]) & 0x7).to(tl.float32)

            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_scales = (
                (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            )
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(
                tl.uint16
            )
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            values = v_idx * v_scales[:, None] + v_zeros[:, None]
        else:  # VQB == 4
            vb_idx = d_offs // 2
            vb_shift = (d_offs % 2) * 4
            val_addrs = val_bases[:, None] + vb_idx[None, :]
            val_raw = tl.load(
                KV_cache_ptr + val_addrs,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)

            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_scales = (
                (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            )
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(
                tl.uint16
            )
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            values = v_idx * v_scales[:, None] + v_zeros[:, None]

        # ============================================================
        # WEIGHTED VALUE ACCUMULATION
        # ============================================================
        acc = acc * re_scale + tl.sum(p[:, None] * values, 0)
        l_prev = l_prev * re_scale + tl.sum(p, 0)
        m_prev = n_e_max

    # Store partial result
    out_base = bid * stride_mid_b + hid * stride_mid_h + sid * stride_mid_s
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(Mid_o_ptr + out_base + d_offs, acc / safe_l, mask=d_mask)
    lse = m_prev + tl.log(safe_l)
    tl.store(Mid_o_ptr + out_base + HEAD_DIM, lse)


# ---------------------------------------------------------------------------
# Stage 1 (q-row-tiled): one KV tile load reused across ALL query rows
# ---------------------------------------------------------------------------
#
# F-2 attempt 2. Numerically identical per row to `_tq_decode_stage1`, but
# the query-row dimension moves from the grid into the program: grid is
# (Hq, NUM_KV_SPLITS) and each program loads a KV tile ONCE (K bytes +
# centroid gather + norms, V bytes), then applies every query row to it.
# Every row of a program shares ONE sequence's cached prefix (same seq_len,
# same block_table row) — the contract of the
# `_spec_continuation_decode_attention` prefix leg — so a tile's bytes are
# byte-identical for every row and one load serves all of them. This
# collapses the per-row full-KV re-scan that made q_len=4 cost ~3.7x
# q_len=1 in the per-row-grid kernel.
#
# STRUCTURE (vs attempt 1, commit 8edf274f1, which measured 150-158 vs the
# old kernel's 120-127 us/1K KV at q_len=4). Attempt 1 vectorized the q
# rows into a tensor dimension: [Q_BLOCK, BLOCK_KV, BLOCK_D]
# broadcast-products for the qk dot and the p*V accumulate, reduced over
# the last axis. The initial "register spill" theory is REFUTED by offline
# ptxas for SM75: attempt 1 compiles to 72 regs/thread at its launched
# num_warps=4, zero spills, zero smem. What the evidence does support:
# (1) the grid collapse (B, Hq, SPLITS)=3072 -> (Hq, SPLITS)=768 programs
# cuts the number of independent tile-walking instruction streams per SM
# (~16 resident CTAs/SM at the old kernel's 4K-reg footprint vs ~5-7
# here), degrading memory-latency hiding for this gather-latency-bound
# loop; and (2) each 2048-element 3D product + two-stage reduction sits on
# the tile loop's serial dependency chain as one long fused computation,
# adding per-iteration latency that the reduced CTA parallelism can no
# longer hide. This attempt keeps the same 768-program grid (that IS the
# 4x-traffic fix) but attacks (2): Q_LEN_ACTUAL is tl.constexpr (2..8) and
# the per-row work is fully UNROLLED at trace time — each row keeps its
# own q[BLOCK_D] vector and (m, l, acc[BLOCK_D]) online-softmax registers
# and runs the old kernel's exact per-row op sequence ([BLOCK_KV, BLOCK_D]
# -> [BLOCK_KV] masked dot, scalar max/exp rescale, [BLOCK_KV, BLOCK_D] ->
# [BLOCK_D] weighted-value reduction). The unrolled rows are MUTUALLY
# INDEPENDENT instruction chains interleavable by the warp scheduler —
# in-CTA ILP that attempt 1's monolithic 3D reductions could not expose.
# Residual under-parallelism has two launch-side antidotes to sweep in the
# GPU window: VLLM_TURBOQUANT_MAX_KV_SPLITS (grid size is proportional to
# splits) and VLLM_TURBOQUANT_DECODE_BLOCK_KV (memory-level parallelism
# per iteration; pair BLOCK_KV=4/8 with num_warps=4 — ptxas shows
# BLOCK_KV=8 spills at num_warps<=2).
#
# Offline ptxas -v (SM75, k3v4_nc, D=256, BLOCK_KV=2, q_len=4):
#   this kernel  num_warps=1: 254 regs 0-spill | =2: 151 regs 0-spill
#                num_warps=4:  93 regs 0-spill | q8 w2: 226 regs 0-spill
#   old kernel   num_warps=1: 128 regs 0-spill (16 CTAs/SM by regs)
#   attempt 1    num_warps=4:  72 regs 0-spill (slow anyway — see above)
#
# The per-row float op sequence (reduction shapes and order, softmax
# update order, accumulate order) matches `_tq_decode_stage1` exactly and
# the shared tile values are the same loads, so per-row results are
# bit-identical to the old kernel. mid_o keeps its [B, Hq, NUM_KV_SPLITS,
# D+1] layout with per-row writes at each row's own bid, and the exact set
# of written (row, head, split) entries matches the old kernel (identical
# split_start/split_end guard from the shared seq_len) — stage-2 reduce is
# unchanged and shared.


@triton.jit
def _tq_decode_stage1_qtile(
    # Precomputed query projection — [B, Hq, D] float32 (B == Q_LEN_ACTUAL
    # rows of ONE sequence)
    Q_rot_ptr,
    KV_cache_ptr,  # [num_blocks, block_size, Hk, padded_slot] uint8
    Block_table_ptr,  # [B, max_num_blocks] int32 (all rows identical)
    Seq_lens_ptr,  # [B] int32 (all identical)
    Centroids_ptr,  # [n_centroids] float32
    Mid_o_ptr,  # [B, Hq, NUM_KV_SPLITS, D+1] float32
    # Strides
    stride_qb,
    stride_qh,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_bt_b,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    # Constexpr dims
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    # TQ layout constants
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    # Score constants
    ATTN_SCALE: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    # Block tile sizes
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    KEY_FP8: tl.constexpr,
    # Query-row tiling
    Q_LEN_ACTUAL: tl.constexpr,  # real query rows, 2..8, fully unrolled
    NORM_CORRECTION: tl.constexpr = 0,
    FP8_FORMAT: tl.constexpr = 0,
):
    tl.static_assert(Q_LEN_ACTUAL >= 2, "qtile kernel requires >= 2 q rows")
    tl.static_assert(Q_LEN_ACTUAL <= 8, "q-row unroll is capped at 8 rows")

    hid = tl.program_id(0)  # q_head index
    sid = tl.program_id(1)  # kv_split index

    kv_head = hid // KV_GROUP_SIZE

    # All query rows share one sequence: seq_len / block_table from row 0.
    seq_len = tl.load(Seq_lens_ptr)
    kv_start = 0
    if SLIDING_WINDOW > 0:
        kv_start = tl.maximum(0, seq_len - SLIDING_WINDOW)

    active_len = seq_len - kv_start
    split_len = tl.cdiv(active_len, NUM_KV_SPLITS)
    split_start = kv_start + split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)

    if split_start >= split_end:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)

    # Per-row query vector + online-softmax state (m, l, acc). Q_LEN_ACTUAL
    # is constexpr: every `if Q_LEN_ACTUAL >= n` here and in the tile loop
    # below is resolved at trace time, so absent rows cost zero registers
    # and zero instructions.
    q_base = hid * stride_qh
    q_r0 = tl.load(
        Q_rot_ptr + q_base + d_offs, mask=d_mask, other=0.0
    ).to(tl.float32)
    m_0 = -float("inf")
    l_0 = 0.0
    acc_0 = tl.zeros([BLOCK_D], dtype=tl.float32)
    q_r1 = tl.load(
        Q_rot_ptr + stride_qb + q_base + d_offs, mask=d_mask, other=0.0
    ).to(tl.float32)
    m_1 = -float("inf")
    l_1 = 0.0
    acc_1 = tl.zeros([BLOCK_D], dtype=tl.float32)
    if Q_LEN_ACTUAL >= 3:
        q_r2 = tl.load(
            Q_rot_ptr + 2 * stride_qb + q_base + d_offs, mask=d_mask, other=0.0
        ).to(tl.float32)
        m_2 = -float("inf")
        l_2 = 0.0
        acc_2 = tl.zeros([BLOCK_D], dtype=tl.float32)
    if Q_LEN_ACTUAL >= 4:
        q_r3 = tl.load(
            Q_rot_ptr + 3 * stride_qb + q_base + d_offs, mask=d_mask, other=0.0
        ).to(tl.float32)
        m_3 = -float("inf")
        l_3 = 0.0
        acc_3 = tl.zeros([BLOCK_D], dtype=tl.float32)
    if Q_LEN_ACTUAL >= 5:
        q_r4 = tl.load(
            Q_rot_ptr + 4 * stride_qb + q_base + d_offs, mask=d_mask, other=0.0
        ).to(tl.float32)
        m_4 = -float("inf")
        l_4 = 0.0
        acc_4 = tl.zeros([BLOCK_D], dtype=tl.float32)
    if Q_LEN_ACTUAL >= 6:
        q_r5 = tl.load(
            Q_rot_ptr + 5 * stride_qb + q_base + d_offs, mask=d_mask, other=0.0
        ).to(tl.float32)
        m_5 = -float("inf")
        l_5 = 0.0
        acc_5 = tl.zeros([BLOCK_D], dtype=tl.float32)
    if Q_LEN_ACTUAL >= 7:
        q_r6 = tl.load(
            Q_rot_ptr + 6 * stride_qb + q_base + d_offs, mask=d_mask, other=0.0
        ).to(tl.float32)
        m_6 = -float("inf")
        l_6 = 0.0
        acc_6 = tl.zeros([BLOCK_D], dtype=tl.float32)
    if Q_LEN_ACTUAL >= 8:
        q_r7 = tl.load(
            Q_rot_ptr + 7 * stride_qb + q_base + d_offs, mask=d_mask, other=0.0
        ).to(tl.float32)
        m_7 = -float("inf")
        l_7 = 0.0
        acc_7 = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Precompute byte/bit index vectors for MSE gather loads
    if not KEY_FP8:
        mse_bit_off = d_offs * MSE_BITS
        mse_byte_idx = mse_bit_off // 8
        mse_bit_shift = mse_bit_off % 8
        mse_mask = (1 << MSE_BITS) - 1

    # Precompute value bit/byte index vectors (loop-invariant)
    if VQB == 3:
        val_bit_off = d_offs * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8

    bt_base = 0  # row 0 of block table (all rows identical)

    # ================================================================
    # TILED LOOP: load each BLOCK_KV tile ONCE, apply all q rows
    # ================================================================
    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx,
            mask=kv_mask,
            other=0,
        ).to(tl.int64)

        slot_bases = (
            block_nums * stride_cache_block
            + page_off.to(tl.int64) * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )

        # ============================================================
        # K TILE LOAD (ONCE, shared by all rows): [BLOCK_KV, BLOCK_D]
        #   k_mat      dequanted FP8 key or gathered centroid vector
        #   k_rowscale per-token score scale (vec_norm for MSE, 1 for FP8)
        # ============================================================
        if KEY_FP8:
            k_addrs = slot_bases[:, None] + d_offs[None, :]
            k_raw = tl.load(
                KV_cache_ptr + k_addrs,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            )
            if FP8_FORMAT == 1:
                k_mat = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
            elif FP8_FORMAT == 2:
                k_mat = k_raw.to(tl.float8e5, bitcast=True).to(tl.float32)
            else:
                k_mat = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            # Multiplying by 1.0 is exact in fp32: lets the per-row score
            # expression below be shared verbatim by both key paths.
            k_rowscale = tl.full([BLOCK_KV], 1.0, dtype=tl.float32)
        else:
            # MSE unpack + norms (loaded once, shared across rows)
            mse_addrs0 = slot_bases[:, None] + mse_byte_idx[None, :]
            mse_raw0 = tl.load(
                KV_cache_ptr + mse_addrs0,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            mse_raw1 = tl.load(
                KV_cache_ptr + mse_addrs0 + 1,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw16 = mse_raw0 | (mse_raw1 << 8)
            mse_idx = (raw16 >> mse_bit_shift[None, :]) & mse_mask

            # Centroid gather + (optional) norm correction
            c_vals = tl.load(
                Centroids_ptr + mse_idx,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            if NORM_CORRECTION:
                c_norm_sq = tl.sum(
                    tl.where(d_mask[None, :], c_vals * c_vals, 0.0),
                    axis=1,
                )
                c_inv_norm = 1.0 / tl.sqrt(c_norm_sq + 1e-16)
                c_vals = c_vals * c_inv_norm[:, None]
            k_mat = c_vals

            norm_bases = slot_bases + MSE_BYTES
            n_lo = tl.load(KV_cache_ptr + norm_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            n_hi = tl.load(KV_cache_ptr + norm_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            k_rowscale = (
                (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            )

        # ============================================================
        # VALUE LOAD + DEQUANTIZE (ONCE): [BLOCK_KV, BLOCK_D]
        # ============================================================
        val_bases = slot_bases + KPS

        if VQB == 3:
            val_addrs0 = val_bases[:, None] + val_byte_idx[None, :]
            val_raw0 = tl.load(
                KV_cache_ptr + val_addrs0,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            val_raw1 = tl.load(
                KV_cache_ptr + val_addrs0 + 1,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw16 = val_raw0 | (val_raw1 << 8)
            v_idx = ((raw16 >> val_bit_shift[None, :]) & 0x7).to(tl.float32)

            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_scales = (
                (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            )
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(
                tl.uint16
            )
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            values = v_idx * v_scales[:, None] + v_zeros[:, None]
        else:  # VQB == 4
            vb_idx = d_offs // 2
            vb_shift = (d_offs % 2) * 4
            val_addrs = val_bases[:, None] + vb_idx[None, :]
            val_raw = tl.load(
                KV_cache_ptr + val_addrs,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)

            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_scales = (
                (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            )
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(
                tl.uint16
            )
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            values = v_idx * v_scales[:, None] + v_zeros[:, None]

        # ============================================================
        # PER-ROW QK DOT + ONLINE SOFTMAX UPDATE (constexpr-unrolled;
        # each block is the old kernel's exact per-row op sequence)
        # ============================================================
        # ---- row 0 ----
        t_r = tl.sum(
            tl.where(d_mask[None, :], q_r0[None, :] * k_mat, 0.0), axis=1
        )
        s_r = k_rowscale * t_r * ATTN_SCALE
        s_r = tl.where(kv_mask, s_r, -float("inf"))
        m_new = tl.maximum(tl.max(s_r, 0), m_0)
        re_s = tl.exp(m_0 - m_new)
        p_r = tl.exp(s_r - m_new)
        acc_0 = acc_0 * re_s + tl.sum(p_r[:, None] * values, 0)
        l_0 = l_0 * re_s + tl.sum(p_r, 0)
        m_0 = m_new
        # ---- row 1 ----
        t_r = tl.sum(
            tl.where(d_mask[None, :], q_r1[None, :] * k_mat, 0.0), axis=1
        )
        s_r = k_rowscale * t_r * ATTN_SCALE
        s_r = tl.where(kv_mask, s_r, -float("inf"))
        m_new = tl.maximum(tl.max(s_r, 0), m_1)
        re_s = tl.exp(m_1 - m_new)
        p_r = tl.exp(s_r - m_new)
        acc_1 = acc_1 * re_s + tl.sum(p_r[:, None] * values, 0)
        l_1 = l_1 * re_s + tl.sum(p_r, 0)
        m_1 = m_new
        # ---- rows 2..7 (constexpr-guarded: absent rows compile away) ----
        if Q_LEN_ACTUAL >= 3:
            t_r = tl.sum(
                tl.where(d_mask[None, :], q_r2[None, :] * k_mat, 0.0), axis=1
            )
            s_r = k_rowscale * t_r * ATTN_SCALE
            s_r = tl.where(kv_mask, s_r, -float("inf"))
            m_new = tl.maximum(tl.max(s_r, 0), m_2)
            re_s = tl.exp(m_2 - m_new)
            p_r = tl.exp(s_r - m_new)
            acc_2 = acc_2 * re_s + tl.sum(p_r[:, None] * values, 0)
            l_2 = l_2 * re_s + tl.sum(p_r, 0)
            m_2 = m_new
        if Q_LEN_ACTUAL >= 4:
            t_r = tl.sum(
                tl.where(d_mask[None, :], q_r3[None, :] * k_mat, 0.0), axis=1
            )
            s_r = k_rowscale * t_r * ATTN_SCALE
            s_r = tl.where(kv_mask, s_r, -float("inf"))
            m_new = tl.maximum(tl.max(s_r, 0), m_3)
            re_s = tl.exp(m_3 - m_new)
            p_r = tl.exp(s_r - m_new)
            acc_3 = acc_3 * re_s + tl.sum(p_r[:, None] * values, 0)
            l_3 = l_3 * re_s + tl.sum(p_r, 0)
            m_3 = m_new
        if Q_LEN_ACTUAL >= 5:
            t_r = tl.sum(
                tl.where(d_mask[None, :], q_r4[None, :] * k_mat, 0.0), axis=1
            )
            s_r = k_rowscale * t_r * ATTN_SCALE
            s_r = tl.where(kv_mask, s_r, -float("inf"))
            m_new = tl.maximum(tl.max(s_r, 0), m_4)
            re_s = tl.exp(m_4 - m_new)
            p_r = tl.exp(s_r - m_new)
            acc_4 = acc_4 * re_s + tl.sum(p_r[:, None] * values, 0)
            l_4 = l_4 * re_s + tl.sum(p_r, 0)
            m_4 = m_new
        if Q_LEN_ACTUAL >= 6:
            t_r = tl.sum(
                tl.where(d_mask[None, :], q_r5[None, :] * k_mat, 0.0), axis=1
            )
            s_r = k_rowscale * t_r * ATTN_SCALE
            s_r = tl.where(kv_mask, s_r, -float("inf"))
            m_new = tl.maximum(tl.max(s_r, 0), m_5)
            re_s = tl.exp(m_5 - m_new)
            p_r = tl.exp(s_r - m_new)
            acc_5 = acc_5 * re_s + tl.sum(p_r[:, None] * values, 0)
            l_5 = l_5 * re_s + tl.sum(p_r, 0)
            m_5 = m_new
        if Q_LEN_ACTUAL >= 7:
            t_r = tl.sum(
                tl.where(d_mask[None, :], q_r6[None, :] * k_mat, 0.0), axis=1
            )
            s_r = k_rowscale * t_r * ATTN_SCALE
            s_r = tl.where(kv_mask, s_r, -float("inf"))
            m_new = tl.maximum(tl.max(s_r, 0), m_6)
            re_s = tl.exp(m_6 - m_new)
            p_r = tl.exp(s_r - m_new)
            acc_6 = acc_6 * re_s + tl.sum(p_r[:, None] * values, 0)
            l_6 = l_6 * re_s + tl.sum(p_r, 0)
            m_6 = m_new
        if Q_LEN_ACTUAL >= 8:
            t_r = tl.sum(
                tl.where(d_mask[None, :], q_r7[None, :] * k_mat, 0.0), axis=1
            )
            s_r = k_rowscale * t_r * ATTN_SCALE
            s_r = tl.where(kv_mask, s_r, -float("inf"))
            m_new = tl.maximum(tl.max(s_r, 0), m_7)
            re_s = tl.exp(m_7 - m_new)
            p_r = tl.exp(s_r - m_new)
            acc_7 = acc_7 * re_s + tl.sum(p_r[:, None] * values, 0)
            l_7 = l_7 * re_s + tl.sum(p_r, 0)
            m_7 = m_new

    # Store per-row partials at each row's own bid (old mid_o layout)
    out_base = hid * stride_mid_h + sid * stride_mid_s
    safe_l = tl.where(l_0 > 0.0, l_0, 1.0)
    tl.store(Mid_o_ptr + out_base + d_offs, acc_0 / safe_l, mask=d_mask)
    tl.store(Mid_o_ptr + out_base + HEAD_DIM, m_0 + tl.log(safe_l))
    safe_l = tl.where(l_1 > 0.0, l_1, 1.0)
    tl.store(
        Mid_o_ptr + stride_mid_b + out_base + d_offs,
        acc_1 / safe_l,
        mask=d_mask,
    )
    tl.store(
        Mid_o_ptr + stride_mid_b + out_base + HEAD_DIM, m_1 + tl.log(safe_l)
    )
    if Q_LEN_ACTUAL >= 3:
        safe_l = tl.where(l_2 > 0.0, l_2, 1.0)
        tl.store(
            Mid_o_ptr + 2 * stride_mid_b + out_base + d_offs,
            acc_2 / safe_l,
            mask=d_mask,
        )
        tl.store(
            Mid_o_ptr + 2 * stride_mid_b + out_base + HEAD_DIM,
            m_2 + tl.log(safe_l),
        )
    if Q_LEN_ACTUAL >= 4:
        safe_l = tl.where(l_3 > 0.0, l_3, 1.0)
        tl.store(
            Mid_o_ptr + 3 * stride_mid_b + out_base + d_offs,
            acc_3 / safe_l,
            mask=d_mask,
        )
        tl.store(
            Mid_o_ptr + 3 * stride_mid_b + out_base + HEAD_DIM,
            m_3 + tl.log(safe_l),
        )
    if Q_LEN_ACTUAL >= 5:
        safe_l = tl.where(l_4 > 0.0, l_4, 1.0)
        tl.store(
            Mid_o_ptr + 4 * stride_mid_b + out_base + d_offs,
            acc_4 / safe_l,
            mask=d_mask,
        )
        tl.store(
            Mid_o_ptr + 4 * stride_mid_b + out_base + HEAD_DIM,
            m_4 + tl.log(safe_l),
        )
    if Q_LEN_ACTUAL >= 6:
        safe_l = tl.where(l_5 > 0.0, l_5, 1.0)
        tl.store(
            Mid_o_ptr + 5 * stride_mid_b + out_base + d_offs,
            acc_5 / safe_l,
            mask=d_mask,
        )
        tl.store(
            Mid_o_ptr + 5 * stride_mid_b + out_base + HEAD_DIM,
            m_5 + tl.log(safe_l),
        )
    if Q_LEN_ACTUAL >= 7:
        safe_l = tl.where(l_6 > 0.0, l_6, 1.0)
        tl.store(
            Mid_o_ptr + 6 * stride_mid_b + out_base + d_offs,
            acc_6 / safe_l,
            mask=d_mask,
        )
        tl.store(
            Mid_o_ptr + 6 * stride_mid_b + out_base + HEAD_DIM,
            m_6 + tl.log(safe_l),
        )
    if Q_LEN_ACTUAL >= 8:
        safe_l = tl.where(l_7 > 0.0, l_7, 1.0)
        tl.store(
            Mid_o_ptr + 7 * stride_mid_b + out_base + d_offs,
            acc_7 / safe_l,
            mask=d_mask,
        )
        tl.store(
            Mid_o_ptr + 7 * stride_mid_b + out_base + HEAD_DIM,
            m_7 + tl.log(safe_l),
        )


# ---------------------------------------------------------------------------
# Pre-dequant kernel: Bulk dequant K (MSE+norms) and V to fp16
# ---------------------------------------------------------------------------


@triton.jit
def _tq_full_dequant_kv(
    KV_cache_ptr,
    Block_table_ptr,
    Centroids_ptr,
    K_out_ptr,  # [B, Hk, max_seq, D] float16
    V_out_ptr,  # [B, Hk, max_seq, D] float16
    stride_ko_b,
    stride_ko_h,
    stride_ko_s,
    stride_vo_b,
    stride_vo_h,
    stride_vo_s,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_bt_b,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    MSE_BITS: tl.constexpr,
    KEY_FP8: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NORM_CORRECTION: tl.constexpr = 0,
    FP8_FORMAT: tl.constexpr = 0,  # 0=e4nv, 1=e4b15, 2=e5
):
    """Full dequant: reconstruct K (MSE centroids * norm or FP8) and V to fp16."""
    pos = tl.program_id(0)
    bh = tl.program_id(1)
    bid = bh // NUM_KV_HEADS
    hid = bh % NUM_KV_HEADS

    page_idx = pos // BLOCK_SIZE
    page_off = pos % BLOCK_SIZE
    block_num = tl.load(Block_table_ptr + bid * stride_bt_b + page_idx).to(tl.int64)
    slot_base = (
        block_num * stride_cache_block
        + tl.cast(page_off, tl.int64) * stride_cache_pos
        + tl.cast(hid, tl.int64) * stride_cache_head
    )

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM

    # === K dequant ===
    ko_base = bid * stride_ko_b + hid * stride_ko_h + pos * stride_ko_s
    if KEY_FP8:
        k_raw = tl.load(KV_cache_ptr + slot_base + d_offs, mask=d_mask, other=0)
        if FP8_FORMAT == 1:
            k_recon = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
        elif FP8_FORMAT == 2:
            k_recon = k_raw.to(tl.float8e5, bitcast=True).to(tl.float32)
        else:
            k_recon = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        tl.store(K_out_ptr + ko_base + d_offs, k_recon.to(tl.float16), mask=d_mask)
    else:
        # MSE unpack (3-bit or 4-bit) + norms
        mse_bit_off = d_offs * MSE_BITS
        mse_byte_idx = mse_bit_off // 8
        mse_bit_shift = mse_bit_off % 8
        mse_umask = (1 << MSE_BITS) - 1

        mse_raw0 = tl.load(
            KV_cache_ptr + slot_base + mse_byte_idx, mask=d_mask, other=0
        ).to(tl.int32)
        mse_raw1 = tl.load(
            KV_cache_ptr + slot_base + mse_byte_idx + 1, mask=d_mask, other=0
        ).to(tl.int32)
        raw16_key = mse_raw0 | (mse_raw1 << 8)
        mse_idx = (raw16_key >> mse_bit_shift) & mse_umask

        k_mse = tl.load(Centroids_ptr + mse_idx, mask=d_mask, other=0.0)

        # Norm correction: re-normalize centroid vector to unit norm
        if NORM_CORRECTION:
            c_norm_sq = tl.sum(tl.where(d_mask, k_mse * k_mse, 0.0), axis=0)
            c_inv_norm = 1.0 / tl.sqrt(c_norm_sq + 1e-16)
            k_mse = k_mse * c_inv_norm

        # Norms at MSE_BYTES offset (no QJL bytes)
        norm_base = slot_base + MSE_BYTES
        n_lo = tl.load(KV_cache_ptr + norm_base).to(tl.uint16)
        n_hi = tl.load(KV_cache_ptr + norm_base + 1).to(tl.uint16)
        vec_norm = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

        k_recon = vec_norm * k_mse
        tl.store(K_out_ptr + ko_base + d_offs, k_recon.to(tl.float16), mask=d_mask)

    # === V dequant ===
    val_base = slot_base + KPS
    if VQB == 4:
        vb_idx = d_offs // 2
        vb_shift = (d_offs % 2) * 4
        val_raw = tl.load(KV_cache_ptr + val_base + vb_idx, mask=d_mask, other=0).to(
            tl.int32
        )
        v_idx = ((val_raw >> vb_shift) & 0xF).to(tl.float32)

        sc_base = val_base + VAL_DATA_BYTES
        sc_lo = tl.load(KV_cache_ptr + sc_base).to(tl.uint16)
        sc_hi = tl.load(KV_cache_ptr + sc_base + 1).to(tl.uint16)
        v_scale = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        zr_lo = tl.load(KV_cache_ptr + sc_base + 2).to(tl.uint16)
        zr_hi = tl.load(KV_cache_ptr + sc_base + 3).to(tl.uint16)
        v_zero = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        v_vals = v_idx * v_scale + v_zero
    elif VQB == 3:
        # 3-bit value unpack: 8 values per 3 bytes
        val_bit_off = d_offs * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8
        val_raw0 = tl.load(
            KV_cache_ptr + val_base + val_byte_idx, mask=d_mask, other=0
        ).to(tl.int32)
        val_raw1 = tl.load(
            KV_cache_ptr + val_base + val_byte_idx + 1, mask=d_mask, other=0
        ).to(tl.int32)
        raw16_val = val_raw0 | (val_raw1 << 8)
        v_idx = ((raw16_val >> val_bit_shift) & 0x7).to(tl.float32)

        sc_base = val_base + VAL_DATA_BYTES
        sc_lo = tl.load(KV_cache_ptr + sc_base).to(tl.uint16)
        sc_hi = tl.load(KV_cache_ptr + sc_base + 1).to(tl.uint16)
        v_scale = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        zr_lo = tl.load(KV_cache_ptr + sc_base + 2).to(tl.uint16)
        zr_hi = tl.load(KV_cache_ptr + sc_base + 3).to(tl.uint16)
        v_zero = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        v_vals = v_idx * v_scale + v_zero
    else:
        v_vals = tl.zeros([BLOCK_D], dtype=tl.float32)

    vo_base = bid * stride_vo_b + hid * stride_vo_h + pos * stride_vo_s
    tl.store(V_out_ptr + vo_base + d_offs, v_vals.to(tl.float16), mask=d_mask)


# ---------------------------------------------------------------------------
# Stage 2: Reuse from triton_decode_attention.py
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Launcher — cached constants + fused GEMM
# ---------------------------------------------------------------------------

_layout_cache: dict = {}


def _get_layout(D, mse_bits, value_quant_bits, key_packed_size):
    """Get cached layout constants."""
    key = (D, mse_bits, value_quant_bits, key_packed_size)
    cfg = _layout_cache.get(key)
    if cfg is None:
        val_data_bytes = math.ceil(D * value_quant_bits / 8)
        cfg = {
            "mse_bytes": math.ceil(D * mse_bits / 8),
            "val_data_bytes": val_data_bytes,
            "mse_bits": mse_bits,
            "n_centroids": 2**mse_bits,
            "BLOCK_D": triton.next_power_of_2(D),
        }
        _layout_cache[key] = cfg
    return cfg


def triton_turboquant_decode_attention(
    query: torch.Tensor,  # [B, Hq, D] — original query
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, padded_slot] uint8
    block_table: torch.Tensor,  # [B, max_num_blocks] int32
    seq_lens: torch.Tensor,  # [B] int32
    Pi: torch.Tensor,  # [D, D] float32
    centroids: torch.Tensor,  # [n_centroids] float32
    scale: float,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    key_fp8: bool = False,
    norm_correction: bool = False,
    PiT: torch.Tensor | None = None,  # [D, D] pre-computed Pi.T contiguous
    # Pre-allocated buffers (optional, avoids per-call allocation)
    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    buf_holder: Any = None,
    max_num_kv_splits: int = 32,  # fixed split count (must be constant for cudagraph)
    sliding_window: int = 0,
    qtile_same_seq: bool = False,  # F-2: all B rows share ONE sequence's prefix
) -> torch.Tensor:
    """Launch fused TQ decode attention (Triton stage1 + stage2).

    Returns: output tensor [B, Hq, D] in query's dtype.
    """
    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    device = query.device

    cfg = _get_layout(D, mse_bits, value_quant_bits, key_packed_size)

    # Compute q_rot = q @ Pi.T (rotated query for MSE key scoring)
    # FP8 path: pass query directly (float16); kernel casts inline.
    # MSE path: still needs external GEMM (cuBLAS), so q_rot is float32.
    if key_fp8:
        q_rot = query.contiguous()
    else:
        q_float = query.float()
        if PiT is None:
            PiT = Pi.T.contiguous()
        q_rot = (q_float @ PiT).contiguous()

    NUM_KV_SPLITS = max_num_kv_splits
    sliding_window = max(0, int(sliding_window))

    if (
        mid_o_buf is not None
        and mid_o_buf.shape[0] >= B
        and mid_o_buf.shape[2] >= NUM_KV_SPLITS
    ):
        mid_o = mid_o_buf[:B, :Hq, :NUM_KV_SPLITS, :]
    else:
        mid_o = torch.empty(
            B,
            Hq,
            NUM_KV_SPLITS,
            D + 1,
            dtype=torch.float32,
            device=device,
        )
        if buf_holder is not None:
            buf_holder._tq_mid_o_buf = mid_o

    # Stage 1: split-KV tiled attention scoring + value accumulation
    fp8_format = _fp8_format_code(device.index or 0)
    BLOCK_KV = _DECODE_BLOCK_KV

    # F-2 q-row-tiled path: only when the caller guarantees all B rows share
    # ONE sequence's cached prefix (the spec-verify prefix leg), the env flag
    # is on, and there are 2..8 rows to amortize over (the kernel unrolls
    # per-row state at trace time, capped at 8; MTP verify is K+1 = 4 rows).
    # Plain batch decode (`_decode_attention`, B = distinct sequences) never
    # sets qtile_same_seq, so it keeps the original per-row grid unchanged.
    if _STAGE1_QTILE and qtile_same_seq and 1 < B <= 8:
        grid_q = (Hq, NUM_KV_SPLITS)
        _tq_decode_stage1_qtile[grid_q](
            q_rot,
            kv_cache,
            block_table,
            seq_lens,
            centroids,
            mid_o,
            q_rot.stride(0),
            q_rot.stride(1),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_table.stride(0),
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            NUM_KV_HEADS=Hk,
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            KV_GROUP_SIZE=kv_group_size,
            MSE_BITS=mse_bits,
            MSE_BYTES=cfg["mse_bytes"],
            KPS=key_packed_size,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=cfg["val_data_bytes"],
            ATTN_SCALE=scale,
            SLIDING_WINDOW=sliding_window,
            BLOCK_D=cfg["BLOCK_D"],
            BLOCK_KV=BLOCK_KV,
            KEY_FP8=1 if key_fp8 else 0,
            Q_LEN_ACTUAL=B,
            NORM_CORRECTION=1 if norm_correction else 0,
            FP8_FORMAT=fp8_format,
            # Row-unrolled program. Offline ptxas (SM75, k3v4_nc, q_len=4,
            # BLOCK_KV=2): num_warps=1/2/4 -> 254/151/93 regs, all 0-spill.
            # Start at 2; sweep {1, 2, 4} x VLLM_TURBOQUANT_DECODE_BLOCK_KV
            # {2, 4, 8} x VLLM_TURBOQUANT_MAX_KV_SPLITS {32, 64} in the GPU
            # window if the first result lands within 20% of the 60 us/1K
            # bar. Constraint: BLOCK_KV=8 requires num_warps=4 (spills at
            # <=2); BLOCK_KV=16 spills even at 4 — do not sweep it.
            num_warps=2,
            num_stages=1,
        )
    else:
        grid = (B, Hq, NUM_KV_SPLITS)
        _tq_decode_stage1[grid](
            q_rot,
            kv_cache,
            block_table,
            seq_lens,
            centroids,
            mid_o,
            q_rot.stride(0),
            q_rot.stride(1),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_table.stride(0),
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            NUM_KV_HEADS=Hk,
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            KV_GROUP_SIZE=kv_group_size,
            MSE_BITS=mse_bits,
            MSE_BYTES=cfg["mse_bytes"],
            KPS=key_packed_size,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=cfg["val_data_bytes"],
            ATTN_SCALE=scale,
            SLIDING_WINDOW=sliding_window,
            BLOCK_D=cfg["BLOCK_D"],
            BLOCK_KV=BLOCK_KV,
            KEY_FP8=1 if key_fp8 else 0,
            NORM_CORRECTION=1 if norm_correction else 0,
            FP8_FORMAT=fp8_format,
            num_warps=1,
            num_stages=1,
        )

    # Stage 2: Reduce across KV splits
    # Output in query dtype — eliminates float16_copy kernel after stage2
    out_dtype = query.dtype
    if (
        output_buf is not None
        and output_buf.shape[0] >= B
        and output_buf.dtype == out_dtype
    ):
        output = output_buf[:B, :Hq, :D]
    else:
        output = torch.empty(B, Hq, D, dtype=out_dtype, device=device)
        if buf_holder is not None:
            buf_holder._tq_output_buf = output
    if lse_buf is not None and lse_buf.shape[0] >= B:
        lse = lse_buf[:B, :Hq]
    else:
        lse = torch.empty(B, Hq, dtype=torch.float32, device=device)
        if buf_holder is not None:
            buf_holder._tq_lse_buf = lse

    grid2 = (B, Hq)
    _fwd_kernel_stage2[grid2](
        mid_o,
        output,
        lse,
        seq_lens,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=cfg["BLOCK_D"],
        Lv=D,
        OUTPUT_FP16=1 if out_dtype == torch.float16 else 0,
        SLIDING_WINDOW=sliding_window,
        num_warps=4,
        num_stages=2,
    )

    return output  # already in query dtype
