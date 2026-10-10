# SPDX-License-Identifier: Apache-2.0
"""Lane K5 (2026-10-03): fused GatedDeltaNet post-conv MTP decode for sm_75 / fp16 (decode-megakernel step 1).

On the fp16 + sm_75 stack the GDN layer cannot use upstream's fused CUDA decode (bf16 + sm80 only), so each GDN layer
of an MTP verify step ran ~25 eager kernels (copies, Triton recurrence, scatter, a 9-kernel native RMSNormGated).
This replaces everything between the conv1d update and out_proj with one kernel (gdn_sm75_ext/gdn_mtp_sm75.cu).

Disabled unless VLLM_K5_GDN_FUSED=1.  Only used for pure spec-decode batches (no prefill, no plain decode rows);
everything else keeps the stock path.
"""

from __future__ import annotations

import math
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
        and _VARIANT in (1, 2)
        and not interleaved
        and head_k_dim == 128
        and head_v_dim == 128
        and act_dtype == torch.float16
        and state_dtype in (torch.float16, torch.float32)
        and num_k_heads > 0
        and num_v_heads > 0
        and num_v_heads % num_k_heads == 0
        and num_v_heads // num_k_heads in (1, 2, 3, 4, 8)
        and activation in ("silu", "sigmoid", "swish")
        and norm_before_gate
        and (group_size is None or group_size == head_v_dim)
    )


def inputs_supported(
    *, mixed_qkv: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
    A_log: torch.Tensor, dt_bias: torch.Tensor, state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor, num_accepted: torch.Tensor, state: torch.Tensor,
    z: torch.Tensor, norm_weight: torch.Tensor, conv_state: torch.Tensor,
    conv_weight: torch.Tensor, conv_bias: torch.Tensor | None,
    num_requests: int, num_tokens: int, scale: float, eps: float,
) -> bool:
    """Metadata-only preflight, before convolution mutates either cache.

    No tensor contents, GPU queries or synchronizations are read here. The
    guarded convolution and GDN kernels check device-resident slot/range
    values. Unsupported shapes/layouts fall back before any state update.
    """
    tensors = (mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens,
               num_accepted, state, z, norm_weight, conv_state, conv_weight)
    if conv_bias is not None:
        tensors += (conv_bias,)
    if not all(t.is_cuda and t.device == mixed_qkv.device for t in tensors):
        return False
    if not (0 < num_requests < 2**31 - 1 and 0 < num_tokens <= 2**31 - 1
            and math.isfinite(scale) and scale > 0
            and math.isfinite(eps) and eps > 0):
        return False
    if not (mixed_qkv.dim() == 2 and mixed_qkv.dtype == torch.float16
            and mixed_qkv.size(0) == num_tokens and mixed_qkv.stride(1) == 1
            and mixed_qkv.stride(0) >= mixed_qkv.size(1)):
        return False
    if not (state.dim() == 4 and state.dtype in (torch.float16, torch.float32)
            and 0 < state.size(0) <= 2**31 - 1 and 0 < state.size(1) <= 65535
            and tuple(state.shape[2:]) == (128, 128)
            and tuple(state.stride()[1:]) == (128 * 128, 128, 1)
            and state.stride(0) >= state.size(1) * 128 * 128
            and state.data_ptr() % 16 == 0
            and state.stride(0) * state.element_size() % 16 == 0):
        return False
    hv = state.size(1)
    kw = mixed_qkv.size(1) - hv * 128
    if kw <= 0 or kw % 256:
        return False
    h = kw // 256
    if hv % h or hv // h not in (1, 2, 3, 4, 8):
        return False
    for t in (a, b):
        if not (t.dim() == 2 and t.dtype == torch.float16
                and tuple(t.shape) == (num_tokens, hv)
                and t.stride(1) == 1 and t.stride(0) >= hv):
            return False
    if not (z.dim() == 3 and z.dtype == torch.float16
            and tuple(z.shape) == (num_tokens, hv, 128)
            and tuple(z.stride()[1:]) == (128, 1)
            and z.stride(0) >= hv * 128):
        return False
    for t in (A_log, dt_bias):
        if not (t.dim() == 1 and t.numel() == hv and t.is_contiguous()):
            return False
    if A_log.dtype != torch.float32 or dt_bias.dtype not in (torch.float16, torch.float32):
        return False
    if not (norm_weight.dim() == 1 and norm_weight.numel() == 128
            and norm_weight.dtype in (torch.float16, torch.float32)
            and norm_weight.is_contiguous()):
        return False
    if not (state_indices.dim() == 2 and state_indices.dtype == torch.int32
            and state_indices.size(0) >= num_requests
            and 1 <= state_indices.size(1) <= 8
            and state_indices.is_contiguous()):
        return False
    for t, length in ((cu_seqlens, num_requests + 1), (num_accepted, num_requests)):
        if not (t.dim() == 1 and t.dtype == torch.int32
                and t.numel() >= length and t.is_contiguous()):
            return False
    width = state_indices.size(1)
    if num_tokens > num_requests * width:
        return False
    channels = mixed_qkv.size(1)
    if not (conv_state.dim() == 3 and conv_state.dtype == torch.float16
            and conv_state.size(0) == state.size(0)
            and conv_state.size(1) == channels
            and all(s > 0 for s in conv_state.stride())):
        return False
    cs, cc, ct = conv_state.stride()
    conv_length = conv_state.size(2)
    if not ((cc >= conv_length * ct or ct >= channels * cc)
            and cs >= (channels - 1) * cc + (conv_length - 1) * ct + 1):
        return False
    if not (conv_weight.dim() == 2 and conv_weight.dtype == torch.float16
            and conv_weight.size(0) == channels and conv_weight.stride(1) == 1
            and conv_weight.stride(0) >= conv_weight.size(1)
            and 2 <= conv_weight.size(1) <= 5
            and conv_state.size(2) >= conv_weight.size(1) - 1 + width - 1):
        return False
    if conv_bias is not None and not (conv_bias.dim() == 1
            and conv_bias.numel() == channels and conv_bias.dtype == torch.float16
            and conv_bias.is_contiguous()):
        return False
    return True


def gdn_mtp(
    mixed_qkv: torch.Tensor,  # [L, 2*H*128 + HV*128] fp16, post-conv (row stride free)
    a: torch.Tensor,  # [L, HV] fp16
    b: torch.Tensor,  # [L, HV] fp16
    A_log: torch.Tensor,  # [HV] fp32
    dt_bias: torch.Tensor,  # [HV] fp32
    state_indices: torch.Tensor,  # [N, W] int32
    cu_seqlens: torch.Tensor,  # [N+1] int32
    num_accepted: torch.Tensor,  # [N] int32
    state: torch.Tensor,  # [slots, HV, 128, 128] fp16|fp32, updated in place
    z: torch.Tensor,  # [L, HV, 128] fp16 output gate
    norm_weight: torch.Tensor,  # [128] fp32
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
