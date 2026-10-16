# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numpy dequantization references for qwen4exp GGUF tests (CPU)."""
from __future__ import annotations

import numpy as np


def dequant_q2_0(raw: np.ndarray, n_elems: int) -> np.ndarray:
    """Q2_0: 18-byte block = fp16 d + 16 bytes qs; weight j = d * (((qs[j//4] >> (2*(j%4))) & 3) - 1)."""
    raw = np.asarray(raw, dtype=np.uint8)
    block_bytes = 18
    n_blocks = n_elems // 64
    out = np.empty(n_elems, dtype=np.float32)
    for b in range(n_blocks):
        blk = raw[b * block_bytes:(b + 1) * block_bytes]
        d = np.frombuffer(blk[0:2], dtype=np.float16)[0].astype(np.float32)
        qs = blk[2:18]
        for j in range(64):
            val = int((qs[j // 4] >> (2 * (j % 4))) & 3) - 1
            out[b * 64 + j] = d * val
    return out


def dequant(raw_bytes: np.ndarray, type_name: str, shape) -> np.ndarray:
    """Dequantize raw bytes to float32 of `shape` (torch order, dims reversed)."""
    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    n_elems = int(np.prod(shape))
    if type_name == "Q2_0":
        flat = dequant_q2_0(raw, n_elems)
    elif type_name in ("F32", "F16", "BF16"):
        import numpy as _np

        dt = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}[type_name]
        arr = np.frombuffer(raw, dtype=dt)
        if type_name == "BF16":
            # upper 16 bits of float32
            bits = arr.astype(np.uint32) << 16
            flat = bits.view(np.float32).astype(np.float32)
        else:
            flat = arr.astype(np.float32)
    else:
        import gguf

        qt = gguf.GGMLQuantizationType[type_name]
        _, block_bytes = gguf.GGML_QUANT_SIZES[qt]
        flat = gguf.quants.dequantize(raw.reshape(-1, block_bytes), qt)
        flat = np.asarray(flat, dtype=np.float32).reshape(-1)
    # reshape to torch order: ggml dims are innermost-first, torch shape is reversed
    torch_shape = tuple(reversed(shape))
    return flat.reshape(torch_shape)
