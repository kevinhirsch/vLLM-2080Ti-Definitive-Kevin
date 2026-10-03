# SPDX-License-Identifier: Apache-2.0
"""TurboQuant k3v4_nc decode attention on Turing INT8 tensor cores (lane K2, 2026-10-03).

See tq_imma_ext/tq_imma.cu for the math.  Same eligibility as tq_gqa (head_dim 256, 3-bit MSE keys, 4-bit values)
plus QL * GQA-group <= 32 and block_size >= 64.  Disabled unless VLLM_TQ_IMMA=1.

The kernel groups the QL rows of one sequence (MTP verifier rows) into one CTA; rows carry their own causal length
(`row_lens`), so the production continuation call (4 rows, lengths cached+1 .. cached+4, one block-table row) maps to
S=1, QL=4 without decoding the cache once per row.
"""

from __future__ import annotations

import functools
import os

import torch

_EXT = None
_ENABLED = os.getenv("VLLM_TQ_IMMA", "0") == "1"


def enabled() -> bool:
    return _ENABLED


def _load():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        src = os.path.join(os.path.dirname(__file__), "tq_imma_ext", "tq_imma.cu")
        bd = os.environ.get("VLLM_TQ_IMMA_BUILD_DIR", os.path.expanduser("~/.cache/vllm-tq-imma"))
        os.makedirs(bd, exist_ok=True)
        _EXT = load(
            name="tq_imma_sm75",
            sources=[src],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_75,code=sm_75"]
            + (["-lineinfo"] if os.getenv("VLLM_TQ_IMMA_LINEINFO") else []),
            build_directory=bd,
            verbose=False,
        )
    return _EXT


@functools.lru_cache(maxsize=8)
def int8_lut(centroids: tuple[float, ...], norm_correction: bool) -> tuple[int, int, float, float]:
    """Pick int8 levels n_i ~= centroid_i / s minimising the worst relative level error.

    With norm correction only the ratios matter (s cancels); without it s (returned as cscale) rescales.
    Returns (lut_lo, lut_hi, cscale, max_rel_err).
    """
    c = list(centroids)
    assert len(c) == 8
    cmax = max(abs(x) for x in c)
    best = None
    for top in range(64, 128):
        s = cmax / top
        n = [max(-127, min(127, round(x / s))) for x in c]
        if any(v == 0 for v in n):
            continue
        err = max(abs(v * s - x) / abs(x) for v, x in zip(n, c))
        if best is None or err < best[0]:
            best = (err, s, n)
    err, s, n = best
    b = [v & 0xFF for v in n]
    lo = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)
    hi = b[4] | (b[5] << 8) | (b[6] << 16) | (b[7] << 24)
    # pass as signed int64 holding the u32 bit pattern
    return lo, hi, s, err


def eligible(*, Hq: int, Hk: int, D: int, mse_bits: int, value_quant_bits: int, key_fp8: bool, key_packed_size: int,
             q_per_seq: int, block_size: int) -> bool:
    return (
        not key_fp8
        and D == 256
        and mse_bits == 3
        and value_quant_bits == 4
        and key_packed_size == 98
        and Hk > 0
        and Hq % Hk == 0
        and q_per_seq * (Hq // Hk) <= 32
        and block_size >= 64
    )


def default_splits(S: int, Hk: int, max_len_hint: int | None = None) -> int:
    """Split count: ~2 waves of 68 SMs (1 CTA/SM), bounded by 64-token chunks when the length is known."""
    env = os.getenv("VLLM_TQ_IMMA_SPLITS")
    if env:
        return int(env)
    ns = max(1, (2 * 68 + S * Hk - 1) // (S * Hk))
    if max_len_hint is not None:
        ns = max(1, min(ns, (max_len_hint + 255) // 256))
    return min(ns, 128)


_LUT_CACHE: dict = {}


def _lut_for(centroids: torch.Tensor, norm_correction: bool):
    key = (centroids.data_ptr(), centroids.device.index, bool(norm_correction))
    v = _LUT_CACHE.get(key)
    if v is None:
        if torch.cuda.is_current_stream_capturing():
            # no device->host read inside a CUDA graph capture: rebuild the (deterministic) Lloyd-Max table on host
            from vllm.model_executor.layers.quantization.turboquant.centroids import get_centroids

            vals = get_centroids(256, 3).tolist()
        else:
            vals = centroids.float().cpu().tolist()
        v = int8_lut(tuple(float(x) for x in vals), bool(norm_correction))
        _LUT_CACHE[key] = v
    return v


def qsplit_enabled() -> bool:
    # hi/lo int8 query planes: LSE error ~15x lower (needed when the LSE is merged with another partial), +8% time
    return os.getenv("VLLM_TQ_IMMA_QSPLIT", "1") == "1"


def tq_imma_decode_attention(
    query: torch.Tensor,  # [S*QL, Hq, D] fp16, un-rotated
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, slot] uint8
    block_table: torch.Tensor,  # [S, max_blocks] int32 (row stride may be 0 = shared table)
    row_lens: torch.Tensor,  # [S*QL] int32: causal length per query row
    centroids: torch.Tensor,
    scale: float,
    norm_correction: bool,
    q_per_seq: int = 1,
    num_splits: int | None = None,
    PiT: torch.Tensor | None = None,
    Pi: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    max_len_hint: int | None = None,
    num_seqs: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    R, Hq, D = query.shape
    S = num_seqs if num_seqs is not None else block_table.shape[0]
    Hk = kv_cache.shape[2]
    assert R == S * q_per_seq and query.dtype == torch.float16
    if PiT is None:
        PiT = Pi.T.contiguous()
    ns = num_splits or default_splits(S, Hk, max_len_hint)
    q_rot = (query.float() @ PiT).contiguous()
    dev = query.device
    q8 = torch.empty(R, Hq, 2 * D, dtype=torch.int8, device=dev)
    qs = torch.empty(R, Hq, dtype=torch.float32, device=dev)
    mid = torch.empty(R, Hq, ns, D + 8, dtype=torch.float32, device=dev)
    out = output_buf[:R, :Hq, :D] if output_buf is not None else torch.empty(R, Hq, D, dtype=torch.float16, device=dev)
    lse = lse_buf[:R, :Hq] if lse_buf is not None else torch.empty(R, Hq, dtype=torch.float32, device=dev)
    lo, hi, cscale, _ = _lut_for(centroids, norm_correction)
    rl = row_lens if (row_lens.dtype == torch.int32 and row_lens.is_contiguous()) else row_lens.to(torch.int32).contiguous()
    _load().decode(q_rot, kv_cache, block_table, rl, q8, qs, mid, out, lse, S, q_per_seq, ns, scale, cscale,
                   1 if norm_correction else 0, lo, hi, 1 if qsplit_enabled() else 0)
    return out, lse


def prefill_splits(Lq: int, Hq: int, Hk: int, C: int) -> int:
    """KV splits for the prefix kernel: enough CTAs for ~2 waves of 68 SMs (1 CTA/SM)."""
    env = os.getenv("VLLM_TQ_IMMA_PREFILL_SPLITS")
    if env:
        return int(env)
    ctas = (Lq * (Hq // Hk) + 63) // 64 * Hk
    ns = max(1, (2 * 68 + ctas - 1) // ctas)
    return max(1, min(ns, (C + 1023) // 1024, 32))


def prefill_enabled() -> bool:
    return os.getenv("VLLM_TQ_IMMA_PREFILL", "0") == "1"


def tq_imma_prefill_prefix_attention(
    query: torch.Tensor,  # [Lq, Hq, D] fp16, un-rotated (the continuation chunk's queries)
    kv_cache: torch.Tensor,
    block_table_row: torch.Tensor,  # [1, max_blocks] or [max_blocks] int32
    cached_len: int,
    centroids: torch.Tensor,
    scale: float,
    norm_correction: bool,
    PiT: torch.Tensor,
    num_splits: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention of every chunk query over the C cached tokens (non-causal: all precede the chunk).
    Returns (out fp16 [Lq, Hq, D], lse fp32 [Lq, Hq], natural log) for merging with the chunk's own attention."""
    Lq, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    dev = query.device
    ns = num_splits or prefill_splits(Lq, Hq, Hk, cached_len)
    q_rot = (query.float() @ PiT).contiguous()
    q8 = torch.empty(Lq, Hq, 2 * D, dtype=torch.int8, device=dev)
    qs = torch.empty(Lq, Hq, dtype=torch.float32, device=dev)
    out = torch.empty(Lq, Hq, D, dtype=torch.float16, device=dev)
    lse = torch.empty(Lq, Hq, dtype=torch.float32, device=dev)
    mid = torch.empty(Lq, Hq, ns, D + 8, dtype=torch.float32, device=dev) if ns > 1 else lse
    lo, hi, cscale, _ = _lut_for(centroids, norm_correction)
    bt = block_table_row.reshape(-1)
    if bt.dtype != torch.int32 or bt.stride(0) != 1:
        bt = bt.to(torch.int32).contiguous()
    _load().prefill_prefix(q_rot, kv_cache, bt, int(cached_len), q8, qs, out, lse, mid, ns, scale, cscale,
                           1 if norm_correction else 0, lo, hi, 1 if qsplit_enabled() else 0)
    return out, lse
