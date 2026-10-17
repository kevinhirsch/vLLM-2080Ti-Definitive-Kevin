# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8 main-cache contract for the Qwen4Exp QSA layers.

Pure torch, device-agnostic: the Triton write kernel and the sparse
attention kernel (``nvidia/ops/qsa.py``) implement exactly this math, and
``tools/selfcheck_int8_kv.py`` checks it on the CPU.

Contract
--------
The checkpoint ships one fp32 scalar per QSA layer and side
(``model-kvscales.safetensors``: ``layers.<i>.self_attn.{k,v}_scale``) under
the fp8 contract ``e4m3fn(x / s_fp8)`` with ``s_fp8 = max_abs / 448``.
The int8 cache reuses the same calibrated ``max_abs`` with the int8 range:

    s_int8 = max_abs / 127 = s_fp8 * 448 / 127
    q      = clamp(floor(x / s_int8 + 0.5), -127, 127)   (stored as torch.int8)
    x~     = q * s_int8                                  (inside the kernels)

Rounding is half-up (``floor(t + 0.5)`` in fp32) rather than half-to-even:
it differs from ``torch.round`` only at exact ties and is the same op
sequence in the Triton kernel, the torch reference and the interpreter.
``x / s`` is evaluated as ``x * (1 / s)`` on both sides. Values beyond the
calibrated ``max_abs`` saturate at +-127, exactly where the fp8 path would
saturate at +-448 (same ``max_abs``). ``-128`` is never produced, so the
grid is symmetric.
"""

from __future__ import annotations

import torch

INT8_KV_QMAX = 127
"""Largest stored magnitude; symmetric grid, -128 unused."""

FP8_E4M3_MAX = 448.0
"""Denominator of the checkpoint's fp8 scale contract (max_abs / 448)."""

INT8_KV_SCALE_FROM_FP8 = FP8_E4M3_MAX / INT8_KV_QMAX
"""s_int8 = s_fp8 * INT8_KV_SCALE_FROM_FP8 (448 / 127 ~= 3.5276)."""

INT8_KV_FALLBACK_MAX_ABS = 32.0
"""Assumed |x| bound for a layer whose checkpoint carries no scale. The
shipped scales imply max_abs between ~8 and ~36, so 32 keeps most rows out
of saturation while still resolving ~0.25 per step."""

INT8_KV_FALLBACK_SCALE = INT8_KV_FALLBACK_MAX_ABS / INT8_KV_QMAX


def int8_kv_scale_from_fp8_scale(fp8_scale: float) -> float:
    """Convert a checkpoint fp8 scale (max_abs / 448) to the int8 scale."""
    if not fp8_scale > 0.0:
        raise ValueError(f"fp8 KV scale must be positive, got {fp8_scale}")
    return float(fp8_scale) * INT8_KV_SCALE_FROM_FP8


def quantize_kv_int8(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Reference quantizer: fp32 math, floor(t + 0.5), clamp to +-127."""
    scaled = x.to(torch.float32) * (1.0 / float(scale))
    codes = torch.floor(scaled + 0.5).clamp_(-INT8_KV_QMAX, INT8_KV_QMAX)
    return codes.to(torch.int8)


def dequantize_kv_int8(
    q: torch.Tensor, scale: float, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Reference dequantizer: q * scale in fp32, then cast to ``dtype``."""
    return (q.to(torch.float32) * float(scale)).to(dtype)


__all__ = [
    "FP8_E4M3_MAX",
    "INT8_KV_FALLBACK_MAX_ABS",
    "INT8_KV_FALLBACK_SCALE",
    "INT8_KV_QMAX",
    "INT8_KV_SCALE_FROM_FP8",
    "dequantize_kv_int8",
    "int8_kv_scale_from_fp8_scale",
    "quantize_kv_int8",
]
