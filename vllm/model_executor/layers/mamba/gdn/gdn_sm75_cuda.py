# SPDX-License-Identifier: Apache-2.0
"""Lane K5 (2026-10-03): fused GatedDeltaNet post-conv MTP decode for sm_75 / fp16 (decode-megakernel step 1).

On the fp16 + sm_75 stack the GDN layer cannot use upstream's fused CUDA decode (bf16 + sm80 only), so each GDN layer
of an MTP verify step ran ~25 eager kernels (copies, Triton recurrence, scatter, a 9-kernel native RMSNormGated).
This replaces everything between the conv1d update and out_proj with one kernel (gdn_sm75_ext/gdn_mtp_sm75.cu).

Disabled unless VLLM_K5_GDN_FUSED=1.  Only used for pure spec-decode batches (no prefill, no plain decode rows);
everything else keeps the stock path.
"""

from __future__ import annotations

import os

import torch

_EXT = None
_ENABLED = os.getenv("VLLM_K5_GDN_FUSED", "0") == "1"
# 2 = reduction-light layout (8 lanes/row, single-pass algebra), 1 = upstream-style layout (32 lanes/row)
_VARIANT = int(os.getenv("VLLM_K5_GDN_VARIANT", "2"))


def enabled() -> bool:
    return _ENABLED


def _load():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        src = os.path.join(os.path.dirname(__file__), "gdn_sm75_ext", "gdn_mtp_sm75.cu")
        bd = os.environ.get("VLLM_K5_GDN_BUILD_DIR", os.path.expanduser("~/.cache/vllm-k5-gdn"))
        os.makedirs(bd, exist_ok=True)
        _EXT = load(
            name="gdn_mtp_sm75",
            sources=[src],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_75,code=sm_75"],
            build_directory=bd,
            verbose=False,
        )
    return _EXT


def eligible(*, head_k_dim: int, head_v_dim: int, num_k_heads: int, num_v_heads: int, act_dtype: torch.dtype,
             state_dtype: torch.dtype, interleaved: bool, activation: str, norm_before_gate: bool,
             group_size: int | None) -> bool:
    return (
        _ENABLED
        and not interleaved
        and head_k_dim == 128
        and head_v_dim == 128
        and act_dtype == torch.float16
        and state_dtype in (torch.float16, torch.float32)
        and num_v_heads % num_k_heads == 0
        and num_v_heads // num_k_heads in (1, 2, 3, 4, 8)
        and activation in ("silu", "sigmoid", "swish")
        and norm_before_gate
        and (group_size is None or group_size == head_v_dim)
    )


def gdn_mtp(
    mixed_qkv: torch.Tensor,  # [L, 2*H*128 + HV*128] fp16, post-conv (row stride free)
    a: torch.Tensor,  # [L, HV] fp16
    b: torch.Tensor,  # [L, HV] fp16
    A_log: torch.Tensor,  # [HV] fp32
    dt_bias: torch.Tensor,  # [HV] fp32|fp16
    state_indices: torch.Tensor,  # [N, W] int32
    cu_seqlens: torch.Tensor,  # [N+1] int32
    num_accepted: torch.Tensor,  # [N] int32
    state: torch.Tensor,  # [slots, HV, 128, 128] fp16|fp32, updated in place
    z: torch.Tensor,  # [L, HV, 128] fp16 output gate
    norm_weight: torch.Tensor,  # [128] fp16|fp32
    out: torch.Tensor,  # [L, HV, 128] fp16
    scale: float,
    eps: float,
    null_block_id: int,
    sigmoid_gate: bool,
    variant: int | None = None,
) -> None:
    _load().gdn_mtp(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted, state, z,
                    norm_weight, out, float(scale), float(eps), int(null_block_id), bool(sigmoid_gate),
                    int(_VARIANT if variant is None else variant))
