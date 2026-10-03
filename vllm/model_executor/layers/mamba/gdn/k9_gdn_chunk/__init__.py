# SPDX-License-Identifier: Apache-2.0
"""Lane K9 (L102): chunked gated-delta-rule prefill forward on sm_75 tensor cores (see gdn_chunk_sm75.cu).

Drop-in for FlashQLA legacy ``chunk_gated_delta_rule_fwd_legacy_varlen`` (same conventions: q/k [1,T,Hk,128] l2-normed,
v [1,T,Hv,128], g log-decay and beta fp32 [1,T,Hv], cu_seqlens int32, initial_state fp32 [N,Hv,128,128]).
Opt-in (VLLM_K9_GDN_CHUNK=1, default off); JIT-built on first use (build dir VLLM_K9_GDN_BUILD_DIR)."""
from __future__ import annotations

import os

import torch

_EXT = None
ENABLED = os.getenv("VLLM_K9_GDN_CHUNK", "0") == "1"
F16QK = os.getenv("VLLM_K9_GDN_F16QK", "0") == "1"  # fp16-acc Q K^T / K K^T: ~1% faster, 1.5-2x the output error


def _load():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        bd = os.environ.get("VLLM_K9_GDN_BUILD_DIR", os.path.expanduser("~/.cache/vllm-k9gdn"))
        os.makedirs(bd, exist_ok=True)
        _EXT = load(
            name="k9_gdn_chunk_sm75",
            sources=[os.path.join(os.path.dirname(__file__), "gdn_chunk_sm75.cu")],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_75,code=sm_75"]
            + (["-Xptxas=-v"] if os.getenv("VLLM_K9_GDN_PTXAS_V") else []),
            build_directory=bd,
            verbose=bool(os.getenv("VLLM_K9_GDN_VERBOSE")),
        )
    return _EXT


def gdn_chunk_fwd_varlen(q, k, v, g, beta, cu_seqlens, scale, initial_state, f16qk: bool | None = None):
    """Returns (output [1,T,Hv,128] fp16, final_state [N,Hv,128,128] fp32)."""
    ext = _load()
    o, st = ext.fwd(q[0], k[0], v[0], g[0].contiguous(), beta[0].contiguous(), cu_seqlens, initial_state.contiguous(),
                    float(scale), F16QK if f16qk is None else bool(f16qk))
    return o.unsqueeze(0), st
