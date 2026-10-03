# SPDX-License-Identifier: Apache-2.0
"""Lane K1 (2026-10-03): sm_75 flash-attention prefill kernel for head_dim 256 (fa75-derived, see fa75_ext/).

Drop-in for the FlashInfer ``BatchPrefillWithRaggedKVCacheWrapper.run`` calls of the TurboQuant backend on Turing:
q [Tq, Hq, 256], k/v [Tkv, Hkv, 256] fp16 (any row/head stride, last dim contiguous), bottom-right causal, GQA,
optional varlen batch, optional LSE out (natural log, the convention of ``merge_attn_states``).

Disabled unless ``VLLM_TQ_FA75_PREFILL=1``. JIT-built with torch.utils.cpp_extension on first use
(build dir ``VLLM_TQ_FA75_BUILD_DIR``, default ~/.cache/vllm-k1fa).
"""

from __future__ import annotations

import os

import torch

_EXT = None
_ENABLED = os.getenv("VLLM_TQ_FA75_PREFILL", "0") == "1"
# variant bits: 1 = P V accumulates per key slice in fp16 (fa75 scheme, 2x HMMA rate), 2 = lazy O rescale,
# 4 = two interleaved Q K^T accumulator sets. Default 7 (fastest; passes the 1e-3 gate)
_VARIANT = int(os.getenv("VLLM_TQ_FA75_VARIANT", "7"))
_BN = int(os.getenv("VLLM_TQ_FA75_BN", "16"))
# Continuation prefill over the TurboQuant cache in pieces of this many cached tokens (rounded up to whole blocks),
# merged exactly by LSE inside the kernel. Bounds the dequant workspace to ~(segment + chunk) rows instead of
# max_model_len (1.07 GB/GPU at 524K for 2 KV heads). 0 = one piece (needs the max_model_len workspace).
_SEGMENT_TOKENS = int(os.getenv("VLLM_TQ_FA75_SEGMENT_TOKENS", "32768"))


def enabled() -> bool:
    return _ENABLED


def segment_rows(block_size: int) -> int:
    """Cached rows per continuation piece (whole blocks); 0 when segmentation is off or K1 is disabled."""
    if not _ENABLED or _SEGMENT_TOKENS <= 0:
        return 0
    return -(-_SEGMENT_TOKENS // block_size) * block_size


def workspace_rows(block_size: int, max_num_batched_tokens: int) -> int:
    """Dequant workspace rows the segmented continuation needs: one piece plus the largest chunk."""
    seg = segment_rows(block_size)
    return seg + -(-max_num_batched_tokens // block_size) * block_size if seg else 0


def _load():
    """Load the extension. A prebuilt .so whose stamp matches the source hash is imported directly (no nvcc/ninja
    in the serving process); otherwise JIT-build it once and stamp it. Prebuild before a boot with
    ``python -c "from vllm.v1.attention.ops import fa75_prefill as f; f._load()"`` (CUDA_VISIBLE_DEVICES='' is fine)."""
    global _EXT
    if _EXT is not None:
        return _EXT
    import hashlib
    import importlib.util

    src = os.path.join(os.path.dirname(__file__), "fa75_ext", "fa75_prefill.cu")
    bd = os.environ.get("VLLM_TQ_FA75_BUILD_DIR", os.path.expanduser("~/.cache/vllm-k1fa"))
    os.makedirs(bd, exist_ok=True)
    flags = ["-O3", "-gencode=arch=compute_75,code=sm_75", "--use_fast_math"]
    flags += ["-lineinfo"] if os.getenv("VLLM_TQ_FA75_LINEINFO") else []
    with open(src, "rb") as f:
        digest = hashlib.sha256(f.read() + " ".join(flags).encode() + torch.__version__.encode()).hexdigest()
    flags += ["-Xptxas=-v"] if os.getenv("VLLM_TQ_FA75_PTXAS_V") else []  # log only, not in the stamp
    so = os.path.join(bd, "k1fa_sm75.so")
    stamp = os.path.join(bd, "k1fa_sm75.stamp")
    try:
        with open(stamp) as f:
            fresh = f.read().strip() == digest and os.path.exists(so)
    except OSError:
        fresh = False
    if fresh:
        spec = importlib.util.spec_from_file_location("k1fa_sm75", so)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EXT = mod
        return _EXT
    from torch.utils.cpp_extension import load

    _EXT = load(
        name="k1fa_sm75",
        sources=[src],
        extra_cuda_cflags=flags,
        build_directory=bd,
        verbose=bool(os.getenv("VLLM_TQ_FA75_VERBOSE")),
    )
    with open(stamp + ".tmp", "w") as f:
        f.write(digest)
    os.replace(stamp + ".tmp", stamp)
    return _EXT


def eligible(q: torch.Tensor, k: torch.Tensor) -> bool:
    return (
        q.is_cuda
        and q.dtype == torch.float16
        and k.dtype == torch.float16
        and q.shape[-1] == 256
        and k.shape[-1] == 256
        and q.shape[1] % k.shape[1] == 0
    )


_MAX_SPLITS = int(os.getenv("VLLM_TQ_FA75_MAX_SPLITS", "4"))
_MIN_SPLIT_KEYS = int(os.getenv("VLLM_TQ_FA75_MIN_SPLIT_KEYS", "4096"))
_SLOTS: dict[int, int] = {}


def choose_splits(Tq: int, Tkv: int, Hq: int, device: torch.device) -> int:
    """KV splits that best fill whole waves of CTAs (2 resident per SM). Only for continuation-shaped calls where
    every query tile does about the same work (Tkv - Tq >= Tq); first chunks keep heaviest-first ordering."""
    if _MAX_SPLITS <= 1 or Tkv - Tq < Tq:
        return 1
    dev = device.index or 0
    slots = _SLOTS.get(dev)
    if slots is None:
        slots = _SLOTS[dev] = 2 * torch.cuda.get_device_properties(dev).multi_processor_count
    base = -(-Tq // 64) * Hq
    best, best_eff = 1, 0.0
    for S in range(1, _MAX_SPLITS + 1):
        if S > 1 and Tkv // S < _MIN_SPLIT_KEYS:
            break
        n = base * S
        eff = n / (-(-n // slots) * slots)
        if eff > best_eff + 0.02:  # a split must buy > 2% of wave efficiency (combine pass + partial traffic)
            best, best_eff = S, eff
    return best


def part_buffer_shapes(max_tokens: int, Hq: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Workspace shapes for the split-KV parts of one call of up to max_tokens query rows."""
    S = max(1, _MAX_SPLITS)
    return (S, max_tokens, Hq, 256), (S, max_tokens, Hq)


_CU_CACHE: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _single_cu(Tq: int, Tkv: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (Tq, Tkv, device.index or 0)
    cu = _CU_CACHE.get(key)
    if cu is None:
        if len(_CU_CACHE) > 256:
            _CU_CACHE.clear()
        cu = (
            torch.tensor([0, Tq], dtype=torch.int32, device=device),
            torch.tensor([0, Tkv], dtype=torch.int32, device=device),
        )
        _CU_CACHE[key] = cu
    return cu


def fa75_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    causal: bool = True,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    return_lse: bool = False,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    variant: int | None = None,
    bn: int | None = None,
    acc_o: torch.Tensor | None = None,
    acc_lse: torch.Tensor | None = None,
    acc_mode: int = 0,
    nsplit: int | None = None,
    part_o: torch.Tensor | None = None,
    part_lse: torch.Tensor | None = None,
):
    """Attention of q over k/v. Single request unless cu_seqlens_* (int32, on device) are given.

    lse, if requested, is fp32 [Tq, Hq] (any strides: pass ``lse=buf.transpose(0, 1)`` to fill an [Hq, Tq] buffer)
    in natural-log units.

    Segmented context (exact): acc_mode 1 = first piece writes acc_o (fp32 [Tq, Hq, 256]) and acc_lse (fp32 [Tq, Hq]);
    2 = middle piece merges into them; 3 = last piece merges and writes ``out`` (fp16). Pieces may come in any order;
    only the last one may be causal.

    Split-KV (wave balance, single request): nsplit None = automatic (``choose_splits``), 1 = off. With nsplit > 1
    the CTAs write fp32 parts to part_o [S, Tq, Hq, 256] / part_lse [S, Tq, Hq] (allocated here when not given) and a
    combine kernel merges them exactly.
    """
    ext = _load()
    if out is None:
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if return_lse and lse is None:
        lse = torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
    if cu_seqlens_q is None:
        cu_seqlens_q, cu_seqlens_k = _single_cu(q.shape[0], k.shape[0], q.device)
        max_seqlen_q = q.shape[0]
    assert cu_seqlens_k is not None and max_seqlen_q is not None
    S = 1
    if nsplit is None:
        S = choose_splits(q.shape[0], k.shape[0], q.shape[1], q.device) if cu_seqlens_q.numel() == 2 else 1
    else:
        S = max(1, int(nsplit))
    split_len = 0
    if S > 1:
        split_len = -(-(-(-k.shape[0] // S)) // 16) * 16
        if part_o is None:
            part_o = torch.empty((S, q.shape[0], q.shape[1], 256), dtype=torch.float32, device=q.device)
        if part_lse is None:
            part_lse = torch.empty((S, q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
    ext.fwd(
        q,
        k,
        v,
        out,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        int(max_seqlen_q),
        float(scale),
        bool(causal),
        int(bn if bn is not None else _BN),
        int(variant if variant is not None else _VARIANT),
        acc_o,
        acc_lse,
        int(acc_mode),
        int(S),
        int(split_len),
        part_o if S > 1 else None,
        part_lse if S > 1 else None,
    )
    if return_lse:
        return out, lse
    return out
