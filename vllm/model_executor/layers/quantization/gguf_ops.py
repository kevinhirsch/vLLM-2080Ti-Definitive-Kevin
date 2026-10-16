# SPDX-License-Identifier: Apache-2.0
"""NM-9: GGUF (ggml) CUDA ops for the SM75 fork, as a JIT-built torch op library.

The prod build ships only the stable-ABI libraries, which contain no ggml kernels. The kernels in ``gguf_ext/``
(vLLM's csrc/quantization/gguf plus the NM-9 Q2_0 type 42 for dequantize / mmvq / moe_vec, and a guard that lets
expert-parallel ``topk_ids == -1`` contribute zero) use the regular torch C++ API, so they are built with
``torch.utils.cpp_extension`` and register ``torch.ops._nm9_ggml.*``.

Release builds prebuild the library at release time:
    release.py build ... --jit-ext vllm/model_executor/layers/quantization/gguf_ops.py:VLLM_NM9_GGML_BUILD_DIR
At runtime the prebuilt .so is loaded when its source stamp matches (no compiler needed).

Never call ``_load`` for the first time inside CUDA-graph capture; ``GGUFConfig`` loads it at model init.
"""

from __future__ import annotations

import glob
import hashlib
import os

import torch

_NAME = "nm9_ggml_ops"
_DIR = os.path.join(os.path.dirname(__file__), "gguf_ext")
_SOURCES = [os.path.join(_DIR, "nm9_ggml_ops.cpp"), os.path.join(_DIR, "gguf_kernel.cu")]
_CFLAGS = [
    "-O3",
    "-gencode=arch=compute_75,code=sm_75",
    "--use_fast_math",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
]
_LOADED = False

# ggml type ids (gguf-py GGMLQuantizationType + the agention fork's Q2_0 = 42).
F32, F16, Q4_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K = 0, 1, 2, 8, 10, 11, 12, 13, 14
IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, IQ1_M, BF16 = 16, 17, 18, 19, 20, 21, 22, 23, 29, 30
Q2_0 = 42
UNQUANTIZED_TYPES = {F32, F16, BF16}
# (block elements, block bytes) per type
BLOCK = {
    Q4_0: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), Q8_0: (32, 34),
    Q2_K: (256, 84), Q3_K: (256, 110), Q4_K: (256, 144), Q5_K: (256, 176), Q6_K: (256, 210),
    IQ2_XXS: (256, 66), IQ2_XS: (256, 74), IQ3_XXS: (256, 98), IQ1_S: (256, 50), IQ4_NL: (32, 18),
    IQ3_S: (256, 110), IQ2_S: (256, 82), IQ4_XS: (256, 136), IQ1_M: (256, 56), Q2_0: (64, 18),
}
DEQUANT_TYPES = set(BLOCK)
MMVQ_TYPES = set(BLOCK)
MMQ_TYPES = {Q4_0, 3, 6, 7, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K}
MOE_VEC_TYPES = set(BLOCK) - {IQ1_M} | {IQ1_M}


def quant_shape_to_byte_shape(shape: tuple[int, ...], qtype: int) -> tuple[int, ...]:
    """Logical (..., n) weight shape -> stored uint8 shape (..., n // block_elems * block_bytes)."""
    be, bb = BLOCK[qtype]
    if shape[-1] % be:
        raise ValueError(f"last dim {shape[-1]} not a multiple of the type-{qtype} block {be}")
    return (*shape[:-1], shape[-1] // be * bb)


def _build_dir() -> str:
    return os.environ.get("VLLM_NM9_GGML_BUILD_DIR", os.path.expanduser("~/.cache/vllm-nm9-ggml"))


def _source_stamp() -> str:
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(_DIR, "*"))):
        with open(p, "rb") as f:
            h.update(os.path.basename(p).encode() + b"\0" + f.read())
    h.update(repr((_CFLAGS, torch.__version__, torch.version.cuda)).encode())
    return h.hexdigest()


def _load() -> None:
    """Prebuilt .so with a matching stamp: loaded directly. Otherwise JIT-built once and stamped."""
    global _LOADED
    if _LOADED:
        return
    if hasattr(torch.ops, "_nm9_ggml") and hasattr(torch.ops._nm9_ggml, "ggml_dequantize"):
        _LOADED = True
        return
    if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("nm9 ggml ops must be loaded before CUDA graph capture")
    bd = _build_dir()
    so = os.path.join(bd, _NAME + ".so")
    stamp_path = os.path.join(bd, _NAME + ".stamp")
    stamp = _source_stamp()
    try:
        with open(stamp_path) as f:
            fresh = f.read().strip() == stamp and os.path.exists(so)
    except OSError:
        fresh = False
    if fresh:
        torch.ops.load_library(so)
    else:
        from torch.utils.cpp_extension import load

        os.makedirs(bd, exist_ok=True)
        load(
            name=_NAME,
            sources=_SOURCES,
            extra_include_paths=[_DIR],
            extra_cuda_cflags=_CFLAGS,
            build_directory=bd,
            is_python_module=False,
            verbose=False,
        )
        try:
            with open(stamp_path, "w") as f:
                f.write(stamp + "\n")
        except OSError:
            pass
    _register_fakes()
    _LOADED = True


_FAKES_DONE = False


def _register_fakes() -> None:
    """Shape functions so the ops trace under torch.compile / dynamo."""
    global _FAKES_DONE
    if _FAKES_DONE:
        return
    reg = torch.library.register_fake

    @reg("_nm9_ggml::ggml_dequantize")
    def _(W, type, m, n, dtype=None):
        return torch.empty((m, n), dtype=dtype or torch.float16, device=W.device)

    @reg("_nm9_ggml::ggml_mul_mat_vec_a8")
    def _(W, X, type, row):
        return torch.empty((X.shape[0], row), dtype=X.dtype, device=X.device)

    @reg("_nm9_ggml::ggml_mul_mat_a8")
    def _(W, X, type, row):
        return torch.empty((X.shape[0], row), dtype=X.dtype, device=X.device)

    @reg("_nm9_ggml::ggml_moe_a8")
    def _(X, W, sorted_token_ids, expert_ids, num_tokens_post_padded, type, row, top_k, tokens):
        return torch.empty((tokens * top_k, row), dtype=X.dtype, device=X.device)

    @reg("_nm9_ggml::ggml_moe_a8_vec")
    def _(X, W, topk_ids, top_k, type, row, tokens):
        return torch.empty((tokens * top_k, row), dtype=X.dtype, device=X.device)

    _FAKES_DONE = True


# ---- thin wrappers (same signatures as the old vllm._custom_ops ggml_* functions) ----
def ggml_dequantize(W: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype | None = None):
    _load()
    return torch.ops._nm9_ggml.ggml_dequantize(W, quant_type, m, n, dtype)


def ggml_mul_mat_vec_a8(W: torch.Tensor, X: torch.Tensor, quant_type: int, row: int):
    _load()
    return torch.ops._nm9_ggml.ggml_mul_mat_vec_a8(W, X, quant_type, row)


def ggml_mul_mat_a8(W: torch.Tensor, X: torch.Tensor, quant_type: int, row: int):
    _load()
    return torch.ops._nm9_ggml.ggml_mul_mat_a8(W, X, quant_type, row)


def ggml_moe_a8(X, W, sorted_token_ids, expert_ids, num_tokens_post_padded, quant_type, row, top_k, tokens):
    _load()
    return torch.ops._nm9_ggml.ggml_moe_a8(
        X, W, sorted_token_ids, expert_ids, num_tokens_post_padded, quant_type, row, top_k, tokens
    )


def ggml_moe_a8_vec(X, W, topk_ids, top_k, quant_type, row, tokens):
    _load()
    return torch.ops._nm9_ggml.ggml_moe_a8_vec(X, W, topk_ids, top_k, quant_type, row, tokens)


def ggml_moe_get_block_size(quant_type: int) -> int:
    _load()
    return torch.ops._nm9_ggml.ggml_moe_get_block_size(quant_type)
