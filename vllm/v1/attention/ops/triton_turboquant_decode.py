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
from vllm.v1.attention.ops import tq_gqa_cuda as _gqa_cuda
from vllm.v1.attention.ops import tq_imma_cuda as _imma_cuda
from vllm.v1.attention.ops.triton_decode_attention import (
    _fwd_kernel_stage2,
)


def _read_decode_block_kv() -> int:
    """Read the SM75 TQ decode tile size selected by the runtime profile.

    Turing's register/shared-memory balance is materially better with the
    validated two-token tile. Keep the value import-time constant so Triton
    can specialize the kernel, while allowing the launcher/profile to select
    another supported tile for experiments.
    """
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

# Keep this independent from the backend policy constant: this low-level
# helper has no backend import and is also exercised directly by GPU tests.
_SPEC_CONTINUATION_MAX_TOKENS = 128

_FP8_FORMAT_CODE: dict[int, int] = {}
_FP8_FORMAT_OVERRIDE = os.getenv(
    "VLLM_TURBOQUANT_K8V4_FP8_FORMAT", "auto"
).strip().lower()
if _FP8_FORMAT_OVERRIDE not in ("", "auto", "e4b15", "e4nv", "e5"):
    raise ValueError(
        "VLLM_TURBOQUANT_K8V4_FP8_FORMAT must be one of: "
        "auto, e4b15, e4nv, e5"
    )


def _fp8_format_code(device: int = 0) -> int:
    """Return the raw TQ FP8 key encoding: 0=e4nv, 1=e4b15, 2=e5."""
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
                # Qwen stores post-RoPE raw keys. On SM75 their dynamic range
                # requires e5; interpreting them as e4b15 loses retrieval
                # quality and collapses later MTP acceptance.
                _FP8_FORMAT_CODE[device] = 2
            else:
                _FP8_FORMAT_CODE[device] = 1 if cap < (8, 9) else 0
        else:
            _FP8_FORMAT_CODE[device] = 0
    return _FP8_FORMAT_CODE[device]


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

    # KV split range
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
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


@triton.jit
def _tq_spec_continuation_merge_kernel(
    # Raw continuation chunk.
    Q_ptr,  # [T, Hq, D]
    K_ptr,  # [T, Hk, D]
    V_ptr,  # [T, Hk, D]
    # Attention state over the compressed, committed prefix.
    Prefix_out_ptr,  # [T, Hq, D]
    Prefix_lse_ptr,  # [T, Hq]
    # Final prefix + raw-continuation attention state.
    Out_ptr,  # [T, Hq, D]
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_pot,
    stride_poh,
    stride_pod,
    stride_plt,
    stride_plh,
    stride_ot,
    stride_oh,
    stride_od,
    ATTN_SCALE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    CHUNK_LEN: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Merge compressed-prefix attention and causal raw current KV.

    One program owns one query row and one query head. Current KV is processed
    in small time tiles using online softmax, which keeps the register
    footprint bounded for ordinary continuation prefill and short verifier
    chunks.
    """
    row = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // KV_GROUP_SIZE

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    q = tl.load(
        Q_ptr + row * stride_qt + q_head * stride_qh + d_offs * stride_qd,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)

    # Online softmax state for raw causal attention over s <= row.
    current_max = -float("inf")
    current_sum = 0.0
    current_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    token_offsets = tl.arange(0, BLOCK_T)

    for start in range(0, CHUNK_LEN, BLOCK_T):
        token_idx = start + token_offsets
        token_mask = (token_idx < CHUNK_LEN) & (token_idx <= row)
        kv_offsets = (
            token_idx[:, None] * stride_kt
            + kv_head * stride_kh
            + d_offs[None, :] * stride_kd
        )
        key = tl.load(
            K_ptr + kv_offsets,
            mask=token_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(key * q[None, :], axis=1) * ATTN_SCALE
        scores = tl.where(token_mask, scores, -float("inf"))

        tile_max = tl.max(scores, axis=0)
        next_max = tl.maximum(current_max, tile_max)
        current_scale = tl.exp(current_max - next_max)
        weights = tl.exp(scores - next_max)
        current_sum = current_sum * current_scale + tl.sum(weights, axis=0)

        value_offsets = (
            token_idx[:, None] * stride_vt
            + kv_head * stride_vh
            + d_offs[None, :] * stride_vd
        )
        value = tl.load(
            V_ptr + value_offsets,
            mask=token_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        current_acc = current_acc * current_scale + tl.sum(
            value * weights[:, None], axis=0
        )
        current_max = next_max

    current_lse = current_max + tl.log(current_sum)
    current_out = current_acc / current_sum

    prefix_lse = tl.load(Prefix_lse_ptr + row * stride_plt + q_head * stride_plh).to(
        tl.float32
    )
    prefix_out = tl.load(
        Prefix_out_ptr + row * stride_pot + q_head * stride_poh + d_offs * stride_pod,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)

    merged_max = tl.maximum(prefix_lse, current_lse)
    prefix_weight = tl.exp(prefix_lse - merged_max)
    current_weight = tl.exp(current_lse - merged_max)
    denom = prefix_weight + current_weight
    output = (prefix_out * prefix_weight + current_out * current_weight) / denom
    tl.store(
        Out_ptr + row * stride_ot + q_head * stride_oh + d_offs * stride_od,
        output,
        mask=d_mask,
    )


def triton_turboquant_spec_continuation_merge(
    query: torch.Tensor,
    key_chunk: torch.Tensor,
    value_chunk: torch.Tensor,
    prefix_out: torch.Tensor,
    prefix_lse: torch.Tensor,
    scale: float,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Merge TQ prefix attention with exact causal current-chunk attention.

    ``do_kv_cache_update`` stores a speculative verifier's complete current
    chunk before its attention call. The compressed cache is intentionally
    lossy, so rereading those just-written rows can alter verifier logits. This
    helper keeps the committed prefix in TurboQuant and consumes only the
    current chunk from raw K/V. It is graph-safe and uses FP32 softmax state.

    When ``output`` aliases ``prefix_out`` the input buffer is overwritten
    after it has been read by each independent program. The backend uses that
    form to avoid another graph-captured allocation.
    """
    if query.ndim != 3 or key_chunk.ndim != 3 or value_chunk.ndim != 3:
        raise ValueError("TQ continuation tensors must have shape [T, H, D].")
    chunk_len, num_query_heads, head_dim = query.shape
    if (
        chunk_len <= 0
        or key_chunk.shape != value_chunk.shape
        or key_chunk.shape[0] != chunk_len
        or key_chunk.shape[2] != head_dim
        or num_query_heads % key_chunk.shape[1] != 0
    ):
        raise ValueError("Invalid TQ speculative-continuation tensor shapes.")
    if prefix_out.shape != query.shape or prefix_lse.shape != query.shape[:2]:
        raise ValueError("TQ prefix attention state has incompatible shape.")
    if chunk_len > _SPEC_CONTINUATION_MAX_TOKENS:
        raise ValueError(
            "TQ speculative-continuation merge supports chunks up to "
            f"{_SPEC_CONTINUATION_MAX_TOKENS} tokens."
        )

    if output is None:
        output = torch.empty_like(query)
    if output.shape != query.shape or output.dtype != query.dtype:
        raise ValueError("TQ continuation output has incompatible shape or dtype.")

    # A bounded raw-KV tile avoids the excessive register footprint of a
    # [128, D] materialization while still specializing the common T=8 case.
    block_t = min(16, triton.next_power_of_2(chunk_len))
    block_d = triton.next_power_of_2(head_dim)
    grid = (chunk_len, num_query_heads)
    _tq_spec_continuation_merge_kernel[grid](
        query,
        key_chunk,
        value_chunk,
        prefix_out,
        prefix_lse,
        output,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_chunk.stride(0),
        key_chunk.stride(1),
        key_chunk.stride(2),
        value_chunk.stride(0),
        value_chunk.stride(1),
        value_chunk.stride(2),
        prefix_out.stride(0),
        prefix_out.stride(1),
        prefix_out.stride(2),
        prefix_lse.stride(0),
        prefix_lse.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        ATTN_SCALE=scale,
        HEAD_DIM=head_dim,
        KV_GROUP_SIZE=num_query_heads // key_chunk.shape[1],
        CHUNK_LEN=chunk_len,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=1,
    )
    return output

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
) -> torch.Tensor:
    """Launch fused TQ decode attention (Triton stage1 + stage2).

    Returns: output tensor [B, Hq, D] in query's dtype.
    """
    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    device = query.device

    if (
        _imma_cuda.enabled()
        and query.dtype == torch.float16
        and block_table.stride(1) == 1
        and _imma_cuda.eligible(
            Hq=Hq, Hk=Hk, D=D, mse_bits=mse_bits, value_quant_bits=value_quant_bits,
            key_fp8=key_fp8, key_packed_size=key_packed_size, q_per_seq=1, block_size=block_size,
        )
    ):
        # lane K2: INT8 tensor-core (IMMA) decode straight on the TurboQuant codes. A stride-0 block table is the
        # continuation call (one request's verifier rows, incremental lengths): group its rows so each cached
        # token is decoded once per kv head for up to 32 // GQA rows.
        if block_table.stride(0) == 0 and B > 1:
            ql = max(1, min(B, 32 // kv_group_size))
            while B % ql:
                ql -= 1
        else:
            ql = 1
        out, _ = _imma_cuda.tq_imma_decode_attention(
            query, kv_cache, block_table, seq_lens, centroids, scale, norm_correction,
            q_per_seq=ql, num_seqs=B // ql, PiT=PiT if PiT is not None else Pi.T.contiguous(),
            output_buf=output_buf if output_buf is not None and output_buf.dtype == torch.float16 else None,
            lse_buf=lse_buf,
        )
        return out

    if (
        _gqa_cuda.enabled()
        and query.dtype == torch.float16
        and _gqa_cuda.gqa_eligible(
            Hq=Hq, Hk=Hk, D=D, mse_bits=mse_bits, value_quant_bits=value_quant_bits,
            key_fp8=key_fp8, key_packed_size=key_packed_size,
        )
    ):
        # lane S2: grouped sm_75 kernel (dequantizes each cached token once per CTA instead of once per query head)
        return _gqa_cuda.tq_gqa_decode_attention(
            query, kv_cache, block_table, seq_lens, Pi, centroids, scale, norm_correction,
            q_per_seq=1, num_splits=max_num_kv_splits, PiT=PiT, output_buf=output_buf, lse_buf=lse_buf, mid_buf=mid_o_buf,
        )

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
        num_warps=4,
        num_stages=2,
    )

    return output  # already in query dtype
