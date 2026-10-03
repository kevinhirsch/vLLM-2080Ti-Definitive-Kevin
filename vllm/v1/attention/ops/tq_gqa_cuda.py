# SPDX-License-Identifier: Apache-2.0
"""Python wrapper for the sm_75 TurboQuant GQA/shared-prefix grouped decode kernel (lane S2, 2026-10-02).

See tq_gqa_ext/tq_gqa.cu.  Eligible only for the production layout: head_dim 256, 3-bit MSE keys (k3, norm
correction), 4-bit values, GQA group <= 16.  `gqa_eligible` tells the caller; everything else keeps using the stock
Triton kernel.  Disabled unless VLLM_TQ_GQA_CUDA=1.
"""

from __future__ import annotations

import os

import torch

_EXT = None
_ENABLED = os.getenv("VLLM_TQ_GQA_CUDA", "0") == "1"


def enabled() -> bool:
    return _ENABLED


def num_splits() -> int:
    return int(os.getenv("VLLM_TQ_GQA_SPLITS", "128"))


def _load():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        src = os.path.join(os.path.dirname(__file__), "tq_gqa_ext", "tq_gqa.cu")
        bd = os.environ.get("VLLM_TQ_GQA_BUILD_DIR", os.path.expanduser("~/.cache/vllm-tq-gqa"))
        os.makedirs(bd, exist_ok=True)
        _EXT = load(
            name="tq_gqa_sm75",
            sources=[src],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_75,code=sm_75"] + (["-lineinfo"] if os.getenv("VLLM_TQ_GQA_LINEINFO") else []),
            build_directory=bd,
            verbose=False,
        )
    return _EXT


def gqa_eligible(*, Hq: int, Hk: int, D: int, mse_bits: int, value_quant_bits: int, key_fp8: bool, key_packed_size: int) -> bool:
    return (
        not key_fp8
        and D == 256
        and mse_bits == 3
        and value_quant_bits == 4
        and key_packed_size == 98
        and Hk > 0
        and Hq % Hk == 0
        and (Hq // Hk) <= 16
    )


def tq_gqa_decode_attention(
    query: torch.Tensor,  # [S*QL, Hq, D] fp16, un-rotated
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, 230] uint8
    block_table: torch.Tensor,  # [S, max_blocks] int32
    seq_lens: torch.Tensor,  # [S] int32 (prefix length shared by the QL rows of a sequence)
    Pi: torch.Tensor,
    centroids: torch.Tensor,
    scale: float,
    norm_correction: bool,
    q_per_seq: int = 1,
    num_splits: int = 128,
    PiT: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    mid_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    R, Hq, D = query.shape
    S = block_table.shape[0]
    QL = q_per_seq
    assert R == S * QL and query.dtype == torch.float16
    Hk = kv_cache.shape[2]
    assert kv_cache.shape[1] >= 16, "tq_gqa kernel needs attention block_size >= 16"
    G = Hq // Hk
    QS = max(1, min(QL, 16 // G))
    if PiT is None:
        PiT = Pi.T.contiguous()
    q_rot = (query.float() @ PiT).contiguous()
    if (
        mid_buf is not None
        and mid_buf.shape[0] >= R
        and mid_buf.shape[1] >= Hq
        and mid_buf.shape[2] >= num_splits
        and mid_buf.shape[3] >= D + 1
    ):
        mid = mid_buf[:R, :Hq, :num_splits, :]  # caller's pre-reserved workspace (no extra allocation)
    else:
        mid = torch.empty(R, Hq, num_splits, D + 1, dtype=torch.float32, device=query.device)
    out = output_buf[:R, :Hq, :D] if output_buf is not None and output_buf.shape[0] >= R else torch.empty(R, Hq, D, dtype=query.dtype, device=query.device)
    lse = lse_buf[:R, :Hq] if lse_buf is not None and lse_buf.shape[0] >= R else torch.empty(R, Hq, dtype=torch.float32, device=query.device)
    _load().decode(q_rot, kv_cache, block_table, seq_lens, centroids, mid, out, lse, QL, QS, G, kv_cache.shape[1], num_splits, scale, 1 if norm_correction else 0)
    return out
