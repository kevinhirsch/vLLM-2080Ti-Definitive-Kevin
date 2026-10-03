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
# variant bits: 1 = P V accumulates per key slice in fp16 (fa75 scheme, 2x HMMA rate), 2 = lazy O rescale
_VARIANT = int(os.getenv("VLLM_TQ_FA75_VARIANT", "2"))
_BN = int(os.getenv("VLLM_TQ_FA75_BN", "16"))


def enabled() -> bool:
    return _ENABLED


def _load():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        src = os.path.join(os.path.dirname(__file__), "fa75_ext", "fa75_prefill.cu")
        bd = os.environ.get("VLLM_TQ_FA75_BUILD_DIR", os.path.expanduser("~/.cache/vllm-k1fa"))
        os.makedirs(bd, exist_ok=True)
        _EXT = load(
            name="k1fa_sm75",
            sources=[src],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_75,code=sm_75", "--use_fast_math"]
            + (["-lineinfo"] if os.getenv("VLLM_TQ_FA75_LINEINFO") else [])
            + (["-Xptxas=-v"] if os.getenv("VLLM_TQ_FA75_PTXAS_V") else []),
            build_directory=bd,
            verbose=bool(os.getenv("VLLM_TQ_FA75_VERBOSE")),
        )
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
):
    """Attention of q over k/v. Single request unless cu_seqlens_* (int32, on device) are given.

    lse, if requested, is fp32 [Tq, Hq] (any strides: pass ``lse=buf.transpose(0, 1)`` to fill an [Hq, Tq] buffer)
    in natural-log units.
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
    )
    if return_lse:
        return out, lse
    return out
