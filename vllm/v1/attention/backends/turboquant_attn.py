# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant attention backend for vLLM.

Prefill: Standard scaled dot-product attention on uncompressed K/V,
         then quantize K and store K+V into combined cache slot.
Decode:  Compute TQ attention scores from compressed cache,
         unpack FP16 values, softmax + weighted sum.

Cache layout (no leading 2 dimension):
  (num_blocks, block_size, num_kv_heads, slot_size)
  where slot_size = key_packed_size + value_fp16_size

Per-head per-position slot layout:
  [key_packed (kps bytes) | value_fp16 (D*2 bytes)]
  For turboquant_k3v4_nc head_dim=256: [100 bytes key | 512 bytes value] = 612
"""

import contextlib
from collections import OrderedDict
import functools
import math
import os
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from packaging.version import Version

from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.utils.ef_fence import fence as _ef_fence
from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
# FlyDSL TurboQuant decode (AMD gfx950). Auto-selected when FlyDSL is available
# for eligible layers (SoA store + FlyDSL decode + SoA-aware continuation);
# non-gfx950 or ineligible layers use the SoA Triton decode.
from vllm.v1.attention.ops.flydsl_turboquant_decode import (
    flydsl_turboquant_decode_attention,
    is_flydsl_available,
    is_flydsl_gqa6_available,
)
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.attention.ops import tq_gqa_cuda as _gqa_cuda
from vllm.v1.attention.ops import tq_imma_cuda as _imma_cuda
from vllm.v1.attention.ops.triton_turboquant_decode import (
    _fp8_format_code,
    _tq_full_dequant_kv,
    triton_turboquant_decode_attention,
    triton_turboquant_spec_continuation_merge,
)
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

logger = init_logger(__name__)

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
if _HAS_FLASH_ATTN:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

try:
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
except ImportError:
    BatchPrefillWithRaggedKVCacheWrapper = None  # type: ignore[assignment]


def _flashinfer_version() -> Version | None:
    try:
        return Version(version("flashinfer-python"))
    except (PackageNotFoundError, ValueError):
        return None


_FLASHINFER_VERSION = _flashinfer_version()

# Continuation prefill: for small continuation chunks (q_len ≤ threshold),
# use the TQ decode kernel directly instead of full-dequant + flash_attn.
# do_kv_cache_update already stored all tokens to TQ cache, so the decode
# kernel can read them efficiently. This avoids O(cached_len) dequant work
# per continuation, eliminating the O(N²/chunk_size) collapse at long context.
_CONTINUATION_DECODE_THRESHOLD = 128
# [FORK][LANE S3] tokens per in-place inverse-rotation matmul in _continuation_prefill (bounds its temporary).
_TQ_CONT_ROT_CHUNK = int(os.getenv("VLLM_TQ_CONT_ROT_CHUNK", "16384"))
_SPEC_CONTINUATION_DECODE_FASTPATH = (
    os.getenv("VLLM_TURBOQUANT_SPEC_CONTINUATION_DECODE_FASTPATH", "0") == "1"
)
_TQ_CUDAGRAPH_SPEC_DECODE_SAFE = (
    os.getenv("VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE", "0") == "1"
)
_TQ_CUDAGRAPH_SPEC_PREFIX_ROWS = (
    os.getenv("VLLM_TURBOQUANT_CUDAGRAPH_SPEC_PREFIX_ROWS", "0") == "1"
)


def _tq_max_kv_splits(configured_splits: int) -> int:
    """Resolve a graph-stable TQ split count from the optional override."""
    value = os.getenv("VLLM_TURBOQUANT_MAX_KV_SPLITS", "").strip()
    if not value:
        return configured_splits
    try:
        splits = int(value)
    except ValueError as err:
        raise ValueError(
            "VLLM_TURBOQUANT_MAX_KV_SPLITS must be a positive integer"
        ) from err
    if splits <= 0:
        raise ValueError("VLLM_TURBOQUANT_MAX_KV_SPLITS must be a positive integer")
    return splits


def _normalize_tq_prefix_combine_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on", "force", "always"):
        return "on"
    if normalized in ("0", "false", "no", "off", "disable", "disabled"):
        return "off"
    if normalized in ("", "auto"):
        return "auto"
    raise ValueError(
        "VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE must be one of "
        "off, on, auto, 0, or 1"
    )


_TQ_CONTINUATION_PREFIX_COMBINE_MODE = _normalize_tq_prefix_combine_mode(
    os.getenv("VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE", "auto")
)
_TQ_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS = max(
    0,
    int(
        os.getenv(
            "VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS", "20480"
        )
    ),
)


# [FORK][LANE K1] FlashInfer returns the attention LSE in base 2 (prefill.cuh: log2(d) + m with the scores
# pre-scaled by log2(e)); merge_attn_states weights the two partial outputs with exp(lse), i.e. natural log.
# Without the conversion the prefix/current mix is e^(1.44 dLSE) instead of e^dLSE: 2.5-12% normwise output
# error measured vs an fp32 reference (tools/k1/test_fa75_prefill.py --pc-only). 0 restores the old behaviour.
_TQ_PREFIX_COMBINE_LSE_LN2 = os.getenv("VLLM_TURBOQUANT_PREFIX_COMBINE_LSE_LN2", "1") == "1"
_LN2 = math.log(2.0)


def _tq_merge_flashinfer_partials(
    prefix_out: torch.Tensor,  # (q_len, Hq, D)
    prefix_lse: torch.Tensor,  # (q_len, Hq) fp32, FlashInfer (base 2)
    current_out: torch.Tensor,
    current_lse: torch.Tensor,
    lse_base2: bool | None = None,
) -> torch.Tensor:
    """Merge two FlashInfer partial attentions with merge_attn_states (natural-log LSE)."""
    if _TQ_PREFIX_COMBINE_LSE_LN2 if lse_base2 is None else lse_base2:
        prefix_lse = prefix_lse * _LN2
        current_lse = current_lse * _LN2
    merged_out = torch.empty_like(prefix_out)
    merge_attn_states(
        merged_out,
        prefix_out,
        prefix_lse.transpose(0, 1).contiguous(),
        current_out,
        current_lse.transpose(0, 1).contiguous(),
    )
    return merged_out


def _tq_continuation_prefix_combine_enabled(seq_len: int) -> bool:
    if _TQ_CONTINUATION_PREFIX_COMBINE_MODE == "on":
        return True
    if _TQ_CONTINUATION_PREFIX_COMBINE_MODE == "auto":
        return seq_len >= _TQ_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS
    return False


def _normalize_turboquant_flashinfer_backend(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return "fa2"
    return normalized or "fa2"


_DEFAULT_TQ_FI_BACKEND = _normalize_turboquant_flashinfer_backend(
    os.getenv("VLLM_TURBOQUANT_FLASHINFER_BACKEND", "fa2")
)
_DEFAULT_TQ_FI_PREFILL = os.getenv("VLLM_TURBOQUANT_USE_FLASHINFER_PREFILL", "1") == "1"
_TQ_REQUIRE_FLASHINFER_PREFILL = (
    os.getenv("VLLM_TURBOQUANT_REQUIRE_FLASHINFER_PREFILL", "0") == "1"
)
_DEFAULT_TQ_FI_PLAN_CACHE = (
    os.getenv("VLLM_TURBOQUANT_FLASHINFER_PREFILL_PLAN_CACHE", "1") == "1"
)
_TQ_FI_PREFILL_PLAN_CACHE_MAXSIZE = max(
    1,
    int(
        os.getenv(
            "VLLM_TURBOQUANT_FLASHINFER_PREFILL_PLAN_CACHE_MAXSIZE",
            "16",
        )
    ),
)
_TQ_FI_PREFILL_CUDAGRAPH_SAFE = (
    os.getenv("VLLM_TURBOQUANT_FLASHINFER_PREFILL_CUDAGRAPH_SAFE", "0") == "1"
)
_SM75_TQ_FI_PREFILL_MIN_QUERY_LEN = int(
    os.getenv("VLLM_TURBOQUANT_SM75_FLASHINFER_PREFILL_MIN_QUERY_LEN", "1")
)
_SM75_TQ_FI_CONTINUATION_MIN_QUERY_LEN = int(
    os.getenv("VLLM_TURBOQUANT_SM75_FLASHINFER_CONTINUATION_MIN_QUERY_LEN", "1")
)
_TQ_FI_PREFILL_WORKSPACES: dict[tuple[str, str], torch.Tensor] = {}
_TQ_FI_PREFILL_WRAPPERS: OrderedDict[tuple[Any, ...], Any] = OrderedDict()


def _soa_imports():
    """Lazy import of the HIP-free SoA Triton subset (store / dequant / decode).

    Kept lazy so the default (FlyDSL-off) path never imports these modules.
    """
    from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_decode import (
        _tq_full_dequant_kv as soa_dequant,
    )
    from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_store import (
        triton_turboquant_store as soa_store,
    )
    from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_unified_attention import (  # noqa: E501
        triton_turboquant_decode_attention_soa as soa_decode,
    )

    return soa_store, soa_dequant, soa_decode


def _prepare_tq_flashinfer_prefill_wrapper_cache(cache_key: tuple[Any, ...]):
    """Return a cached wrapper and make room for a new plan when needed.

    CUDA-graph-safe wrappers own indptr buffers referenced by captured graphs;
    without a graph-destruction callback they must remain live for the process
    lifetime. The ordinary path can evict the least-recently-used wrapper so
    long-context plan keys do not consume unbounded GPU workspace.
    """
    wrapper = _TQ_FI_PREFILL_WRAPPERS.get(cache_key)
    if wrapper is not None:
        _TQ_FI_PREFILL_WRAPPERS.move_to_end(cache_key)
        return wrapper

    if _TQ_FI_PREFILL_CUDAGRAPH_SAFE:
        logger.warning_once(
            "TurboQuant FlashInfer prefill plan cache eviction is disabled "
            "when CUDA-graph-safe wrappers are enabled; set "
            "VLLM_TURBOQUANT_FLASHINFER_PREFILL_CUDAGRAPH_SAFE=0 to allow "
            "bounded wrapper caching"
        )
    elif len(_TQ_FI_PREFILL_WRAPPERS) >= _TQ_FI_PREFILL_PLAN_CACHE_MAXSIZE:
        # Evict before constructing/planning the replacement. Each wrapper owns
        # an auxiliary GPU workspace, so eviction after planning briefly needs
        # maxsize + 1 workspaces and can defeat the OOM protection.
        _TQ_FI_PREFILL_WRAPPERS.popitem(last=False)
    return None


def _build_hadamard(d: int, device_str: str) -> torch.Tensor:
    """Orthonormal Hadamard matrix (Sylvester construction), cached per (d, device).

    Precomputed D×D matrix enables matmul-based WHT — single cuBLAS GEMM
    instead of log2(D) butterfly kernel launches. 64KB for D=128.
    """
    # Normalize device string so "cuda" and "cuda:0" hit the same cache entry.
    return _build_hadamard_cached(d, str(torch.device(device_str)))


@functools.cache
def _build_hadamard_cached(d: int, device_str: str) -> torch.Tensor:
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(torch.device(device_str))


def _normalize_cuda_device(device: torch.device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _get_shared_flashinfer_prefill_workspace(
    device: torch.device,
    backend: str,
) -> torch.Tensor:
    device = _normalize_cuda_device(device)
    key = (str(device), backend)
    workspace = _TQ_FI_PREFILL_WORKSPACES.get(key)
    if workspace is None:
        workspace = torch.empty(
            envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,
            dtype=torch.uint8,
            device=device,
        )
        _TQ_FI_PREFILL_WORKSPACES[key] = workspace
    return workspace


def _get_or_plan_tq_flashinfer_prefill_wrapper(
    device: torch.device,
    plan_key: tuple[Any, ...],
    plan_kwargs: dict[str, Any],
):
    """Plan outside model forward so FlashInfer remains out of CUDA graphs."""
    if BatchPrefillWithRaggedKVCacheWrapper is None:
        return None
    if not _DEFAULT_TQ_FI_PLAN_CACHE:
        wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(
                envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,
                dtype=torch.uint8,
                device=device,
            ),
            "NHD",
            backend=_DEFAULT_TQ_FI_BACKEND,
        )
        wrapper.plan(**plan_kwargs)
        return wrapper

    norm_device = _normalize_cuda_device(device)
    cache_key = (str(norm_device), _DEFAULT_TQ_FI_BACKEND, *plan_key)
    wrapper = _prepare_tq_flashinfer_prefill_wrapper_cache(cache_key)
    if wrapper is None:
        workspace = _get_shared_flashinfer_prefill_workspace(
            norm_device, _DEFAULT_TQ_FI_BACKEND
        )
        wrapper_kwargs: dict[str, Any] = {"backend": _DEFAULT_TQ_FI_BACKEND}
        if _TQ_FI_PREFILL_CUDAGRAPH_SAFE:
            qo_indptr = plan_kwargs.get("qo_indptr")
            kv_indptr = plan_kwargs.get("kv_indptr")
            if qo_indptr is None or kv_indptr is None:
                raise RuntimeError("FlashInfer cudagraph-safe prefill requires indptr buffers")
            wrapper_kwargs.update(
                {
                    "use_cuda_graph": True,
                    "qo_indptr_buf": torch.empty_like(qo_indptr, device=norm_device),
                    "kv_indptr_buf": torch.empty_like(kv_indptr, device=norm_device),
                }
            )
        wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            workspace,
            "NHD",
            **wrapper_kwargs,
        )
        wrapper.plan(**plan_kwargs)
        _TQ_FI_PREFILL_WRAPPERS[cache_key] = wrapper
    return wrapper


class TurboQuantAttentionBackend(AttentionBackend):
    """Attention backend using TurboQuant KV-cache compression."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "turboquant_k8v4",
        "turboquant_4bit_nc",
        "turboquant_k3v4_nc",
        "turboquant_3bit_nc",
    ]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """TurboQuant packs K+V into one slot per head."""
        if spec.state_content_bytes is not None or not spec.kv_quant_mode.is_turboquant:
            return spec
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        # KVQuantMode member names mirror the preset strings.
        tq = TurboQuantConfig.from_cache_dtype(
            spec.kv_quant_mode.name.lower(), spec.head_size
        )
        return replace(spec, state_content_bytes=tq.slot_size_aligned)

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # The Triton store/decode paths consume explicit B/N/H strides after
        # converting the logical [B, H, N, C] view to [B, N, H, C]. They can
        # therefore also use block-compact BLHNC, which is required when this
        # compressed target cache shares an allocator with DFlash FP16 pages
        # of a different HNC shape.
        return (KVCacheLayout.LBNHC, KVCacheLayout.BLHNC)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # The Triton store/decode kernels derive page addressing from the
        # runtime cache shape. Hybrid Mamba models may align the manager page
        # to their much larger state page, so accepting any 16-token multiple
        # avoids an invalid virtual split across a padded TQ page.
        return [MultipleOf(16)]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        # Per-request speculative decode slices are planned from CPU query
        # boundaries. Adaptive verification can redistribute verifier rows
        # only in the device boundaries, which would pair rows with another
        # request's sequence length and block table.
        return False

    @staticmethod
    def get_impl_cls() -> type["TurboQuantAttentionImpl"]:
        return TurboQuantAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["TurboQuantMetadataBuilder"]:
        return TurboQuantMetadataBuilder

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return False
        return kv_cache_dtype.startswith("turboquant_")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # head_size from spec is effective_head_size (padded_slot//2),
        # not the model's actual head_dim. Accept any positive value.
        return head_size > 0


@dataclass
class TurboQuantMetadata(AttentionMetadata):
    """Metadata for TurboQuant attention."""

    seq_lens: torch.Tensor  # (num_reqs,) — total context length per request
    slot_mapping: torch.Tensor  # (num_tokens,) — cache slot for each token
    block_table: torch.Tensor  # (num_reqs, max_num_blocks)
    query_start_loc: torch.Tensor  # (num_reqs + 1,) — cu_seqlens for queries
    num_actual_tokens: int = 0  # actual tokens (excluding padding)
    max_query_len: int = 0  # longest query in batch
    max_seq_len: int = 0  # longest context in batch
    is_prefill: bool = False
    num_decodes: int = 0  # number of decode requests (first in batch)
    num_decode_tokens: int = 0  # tokens from decode requests
    # CUDA graph capture uses this graph-safe multi-token continuation path
    # when the scheduler classifies speculative MTP tokens as decode work.
    force_spec_decode: bool = False
    # CPU-resident copies used by the prefill path for per-request iteration
    # without per-step D2H syncs.
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None
    # FlashInfer wrappers are planned by the metadata builder. Keeping plan()
    # out of AttentionImpl.forward avoids leaking a Python/JIT operation into
    # the compiled model or its CUDA graph capture.
    flashinfer_first_chunk_wrapper: Any | None = None
    flashinfer_first_chunk_wrappers: dict[int, Any] | None = None
    flashinfer_continuation_wrappers: dict[int, Any] | None = None
    flashinfer_prefix_combine_wrappers: dict[int, tuple[Any, Any]] | None = None


class TurboQuantMetadataBuilder(AttentionMetadataBuilder[TurboQuantMetadata]):
    """Builds TurboQuantMetadata from scheduler output."""

    kv_cache_spec: AttentionSpec
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(
            1, supports_spec_as_decode=_TQ_CUDAGRAPH_SPEC_DECODE_SAFE
        )
        self._device = torch.device(device)
        self._flashinfer_prefill_enabled = (
            _DEFAULT_TQ_FI_PREFILL
            and BatchPrefillWithRaggedKVCacheWrapper is not None
            and current_platform.is_cuda()
        )
        if _TQ_REQUIRE_FLASHINFER_PREFILL and not self._flashinfer_prefill_enabled:
            raise RuntimeError(
                "TurboQuant fast route requires the FlashInfer prefill backend; "
                "install a supported FlashInfer build or disable the fast route."
            )
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self._flashinfer_num_qo_heads = model_config.get_num_attention_heads(
            parallel_config
        )
        self._flashinfer_num_kv_heads = kv_cache_spec.num_kv_heads
        self._flashinfer_head_dim = kv_cache_spec.head_size
        self._flashinfer_dtype = model_config.dtype
        self._flashinfer_scale = self._flashinfer_head_dim**-0.5
        self._max_num_kv_splits = _tq_max_kv_splits(
            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        )
        self._reserve_workspace()

    def _plan_flashinfer_prefill_wrappers(
        self,
        cam: CommonAttentionMetadata,
        num_decodes: int,
    ) -> tuple[
        Any | None,
        dict[int, Any] | None,
        dict[int, Any] | None,
        dict[int, tuple[Any, Any]] | None,
    ]:
        """Prepare raw-K/V FlashInfer wrappers before model forward."""
        if (
            not self._flashinfer_prefill_enabled
            or cam.max_query_len <= 0
            or cam.query_start_loc_cpu is None
            or cam.seq_lens_cpu_upper_bound is None
        ):
            return None, None, None, None

        qsl = cam.query_start_loc_cpu
        seq_lens = cam.seq_lens_cpu_upper_bound
        q_lens = qsl[1:] - qsl[:-1]
        num_reqs = q_lens.shape[0]
        Hq = self._flashinfer_num_qo_heads
        Hk = self._flashinfer_num_kv_heads
        D = self._flashinfer_head_dim
        dtype = self._flashinfer_dtype
        window_left = -1

        # A complete first chunk can use one batched ragged plan. The explicit
        # CPU check prevents treating a continuation's raw K/V suffix as its
        # entire context.
        if num_decodes == 0 and torch.equal(q_lens, seq_lens[:num_reqs]):
            plan_key = (
                "batch_first_chunk",
                Hq,
                Hk,
                D,
                window_left,
                str(dtype),
                tuple(int(x) for x in qsl.tolist()),
            )
            wrapper = _get_or_plan_tq_flashinfer_prefill_wrapper(
                self._device,
                plan_key,
                {
                    "qo_indptr": qsl,
                    "kv_indptr": qsl,
                    "num_qo_heads": Hq,
                    "num_kv_heads": Hk,
                    "head_dim_qk": D,
                    "causal": True,
                    "window_left": window_left,
                    "sm_scale": self._flashinfer_scale,
                    "pos_encoding_mode": "NONE",
                    "q_data_type": dtype,
                    "kv_data_type": dtype,
                    "seq_lens": q_lens,
                    "seq_lens_q": q_lens,
                    "max_token_per_sequence": cam.max_query_len,
                    "max_sequence_kv": cam.max_seq_len,
                },
            )
            return wrapper, None, None, None

        # Mixed batches can contain complete first chunks after one or more
        # decode requests. Plan those requests independently; the batched
        # ragged plan above is only valid when every request is first-chunk.
        first_chunk_wrappers: dict[int, Any] = {}
        wrappers: dict[int, Any] = {}
        prefix_combine_wrappers: dict[int, tuple[Any, Any]] = {}
        prefix_combine_supported = (
            getattr(self.kv_cache_spec, "sliding_window", None) is None
        )
        pin_memory = self._device.type == "cuda" and torch.cuda.is_available()
        for request_idx in range(num_decodes, num_reqs):
            q_len = int(q_lens[request_idx])
            seq_len = int(seq_lens[request_idx])
            if q_len == seq_len and q_len > 0:
                qo_indptr = torch.tensor(
                    [0, q_len], dtype=torch.int32, pin_memory=pin_memory
                )
                first_chunk_wrappers[request_idx] = (
                    _get_or_plan_tq_flashinfer_prefill_wrapper(
                        self._device,
                        (
                            "mixed_first_chunk",
                            Hq,
                            Hk,
                            D,
                            window_left,
                            str(dtype),
                            q_len,
                        ),
                        {
                            "qo_indptr": qo_indptr,
                            "kv_indptr": qo_indptr,
                            "num_qo_heads": Hq,
                            "num_kv_heads": Hk,
                            "head_dim_qk": D,
                            "causal": True,
                            "window_left": window_left,
                            "sm_scale": self._flashinfer_scale,
                            "pos_encoding_mode": "NONE",
                            "q_data_type": dtype,
                            "kv_data_type": dtype,
                            "seq_lens": torch.tensor(
                                [q_len], dtype=torch.int32
                            ),
                            "seq_lens_q": torch.tensor(
                                [q_len], dtype=torch.int32
                            ),
                            "max_token_per_sequence": q_len,
                            "max_sequence_kv": q_len,
                        },
                    )
                )
                continue
            if (
                q_len <= _CONTINUATION_DECODE_THRESHOLD
                or q_len < _SM75_TQ_FI_CONTINUATION_MIN_QUERY_LEN
                or q_len > seq_len
            ):
                continue
            cached_len = seq_len - q_len
            qo_indptr = torch.tensor(
                [0, q_len], dtype=torch.int32, pin_memory=pin_memory
            )
            if prefix_combine_supported and _tq_continuation_prefix_combine_enabled(
                seq_len
            ):
                prefix_kv_indptr = torch.tensor(
                    [0, cached_len], dtype=torch.int32, pin_memory=pin_memory
                )
                prefix_wrapper = _get_or_plan_tq_flashinfer_prefill_wrapper(
                    self._device,
                    (
                        "continuation_prefix_combine_prefix",
                        Hq,
                        Hk,
                        D,
                        window_left,
                        str(dtype),
                        q_len,
                        cached_len,
                    ),
                    {
                        "qo_indptr": qo_indptr,
                        "kv_indptr": prefix_kv_indptr,
                        "num_qo_heads": Hq,
                        "num_kv_heads": Hk,
                        "head_dim_qk": D,
                        "causal": False,
                        "window_left": window_left,
                        "sm_scale": self._flashinfer_scale,
                        "pos_encoding_mode": "NONE",
                        "q_data_type": dtype,
                        "kv_data_type": torch.float16,
                        "seq_lens": torch.tensor([cached_len], dtype=torch.int32),
                        "seq_lens_q": torch.tensor([q_len], dtype=torch.int32),
                        "max_token_per_sequence": q_len,
                        "max_sequence_kv": cached_len,
                    },
                )
                current_kv_indptr = torch.tensor(
                    [0, q_len], dtype=torch.int32, pin_memory=pin_memory
                )
                current_wrapper = _get_or_plan_tq_flashinfer_prefill_wrapper(
                    self._device,
                    (
                        "continuation_prefix_combine_current",
                        Hq,
                        Hk,
                        D,
                        window_left,
                        str(dtype),
                        q_len,
                    ),
                    {
                        "qo_indptr": qo_indptr,
                        "kv_indptr": current_kv_indptr,
                        "num_qo_heads": Hq,
                        "num_kv_heads": Hk,
                        "head_dim_qk": D,
                        "causal": True,
                        "window_left": window_left,
                        "sm_scale": self._flashinfer_scale,
                        "pos_encoding_mode": "NONE",
                        "q_data_type": dtype,
                        "kv_data_type": dtype,
                        "seq_lens": torch.tensor([q_len], dtype=torch.int32),
                        "seq_lens_q": torch.tensor([q_len], dtype=torch.int32),
                        "max_token_per_sequence": q_len,
                        "max_sequence_kv": q_len,
                    },
                )
                if prefix_wrapper is not None and current_wrapper is not None:
                    prefix_combine_wrappers[request_idx] = (
                        prefix_wrapper,
                        current_wrapper,
                    )
                    continue
                if _TQ_REQUIRE_FLASHINFER_PREFILL:
                    raise RuntimeError(
                        "TurboQuant fast route could not plan FlashInfer "
                        "continuation prefix-combine wrappers."
                    )
            kv_indptr = torch.tensor(
                [0, seq_len], dtype=torch.int32, pin_memory=pin_memory
            )
            plan_key = (
                "continuation",
                Hq,
                Hk,
                D,
                window_left,
                str(dtype),
                q_len,
                seq_len,
            )
            wrappers[request_idx] = _get_or_plan_tq_flashinfer_prefill_wrapper(
                self._device,
                plan_key,
                {
                    "qo_indptr": qo_indptr,
                    "kv_indptr": kv_indptr,
                    "num_qo_heads": Hq,
                    "num_kv_heads": Hk,
                    "head_dim_qk": D,
                    "causal": True,
                    "window_left": window_left,
                    "sm_scale": self._flashinfer_scale,
                    "pos_encoding_mode": "NONE",
                    "q_data_type": dtype,
                    "kv_data_type": dtype,
                    "seq_lens": torch.tensor([seq_len], dtype=torch.int32),
                    "seq_lens_q": torch.tensor([q_len], dtype=torch.int32),
                    "max_token_per_sequence": q_len,
                    "max_sequence_kv": seq_len,
                },
            )
        return (
            None,
            first_chunk_wrappers or None,
            wrappers or None,
            prefix_combine_wrappers or None,
        )

    def _reserve_workspace(self) -> None:
        if not is_workspace_manager_initialized():
            return

        scheduler_config = self.vllm_config.scheduler_config
        model_config = self.vllm_config.model_config
        parallel_config = self.vllm_config.parallel_config

        max_num_reqs = scheduler_config.max_num_seqs
        num_heads = model_config.get_num_attention_heads(parallel_config)
        num_kv_heads = self.kv_cache_spec.num_kv_heads
        head_size = self.kv_cache_spec.head_size
        max_num_splits = self._max_num_kv_splits

        current_workspace_manager().get_simultaneous(
            ((max_num_reqs, num_heads, max_num_splits, head_size + 1), torch.float32),
            ((max_num_reqs, num_heads, head_size), model_config.dtype),
            ((max_num_reqs, num_heads), torch.float32),
        )

        reserve_continuation_prefill = (
            scheduler_config.enable_chunked_prefill
            and scheduler_config.max_num_batched_tokens > _CONTINUATION_DECODE_THRESHOLD
        )
        if not reserve_continuation_prefill:
            return

        # [FORK][LANE S3] the continuation path dequantizes straight into the
        # (rows, Hk, D) K/V it hands to attention (no second full-context copy),
        # so reserve exactly that once, at max_model_len, inside the profiled
        # workspace instead of paying for context-sized temporaries at runtime.
        alloc_len = round_up(
            max(1, model_config.max_model_len - 1), self.kv_cache_spec.block_size
        )
        cache_buf_shape = (alloc_len, num_kv_heads, head_size)
        current_workspace_manager().get_simultaneous(
            (cache_buf_shape, torch.float16),
            (cache_buf_shape, torch.float16),
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TurboQuantMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        if (
            _TQ_CUDAGRAPH_SPEC_DECODE_SAFE
            and 1 < attn_metadata.max_query_len <= _CONTINUATION_DECODE_THRESHOLD
        ):
            # Capture the MTP continuation through the raw-current-K/V path.
            # Keep the shared warmup sequence length at one: GDN/Mamba reads
            # the same CommonAttentionMetadata, and changing it to q_len
            # makes its SM75 capture address speculative state that was not
            # allocated for the dummy batch. TurboQuant clamps its local
            # prefix length to zero below, so it does not need a synthetic
            # q_len-sized prefix here.
            attn_metadata.force_spec_decode = True
            attn_metadata.seq_lens.fill_(1)
            return attn_metadata
        # Set seq_lens to 1 so CUDA graph capture is fast
        # (real seq_lens are filled at replay time).
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        """Build TurboQuantMetadata from common attention metadata."""
        cam = common_attn_metadata

        # The model runner guarantees decodes come first. Keep short requests
        # still in prefill on the prefill side when that state is available.
        assert self.reorder_batch_threshold is not None
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            cam,
            decode_threshold=self.reorder_batch_threshold,
            treat_short_extends_as_decodes=cam.is_prefilling is None,
        )
        (
            first_chunk_wrapper,
            first_chunk_wrappers,
            continuation_wrappers,
            prefix_combine_wrappers,
        ) = (
            self._plan_flashinfer_prefill_wrappers(cam, num_decodes)
        )

        return TurboQuantMetadata(
            seq_lens=cam.seq_lens,
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            is_prefill=(cam.max_query_len > 1),
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc_cpu=cam.query_start_loc_cpu,
            seq_lens_cpu=cam.seq_lens_cpu_upper_bound,
            flashinfer_first_chunk_wrapper=first_chunk_wrapper,
            flashinfer_first_chunk_wrappers=first_chunk_wrappers,
            flashinfer_continuation_wrappers=continuation_wrappers,
            flashinfer_prefix_combine_wrappers=prefix_combine_wrappers,
        )


class TurboQuantAttentionImpl(AttentionImpl["TurboQuantMetadata"]):
    """TurboQuant attention implementation.

    Vectorized PyTorch: batch quantize/store, vectorized bit-unpack
    decode with einsum scores and value gather.
    """

    supports_quant_query_input: bool = False

    # Lazily populated before cudagraph capture (FlyDSL decode path only).
    _arange_cache: torch.Tensor
    _cu_2: torch.Tensor

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        self.tq_config = TurboQuantConfig.from_cache_dtype(kv_cache_dtype, head_size)

        # Pre-compute kernel constants from config (avoid repeated arithmetic)
        cfg = self.tq_config
        self._mse_bytes = (
            math.ceil(head_size * cfg.key_mse_bits / 8)
            if not cfg.key_fp8
            else head_size
        )
        self._val_data_bytes = math.ceil(head_size * cfg.effective_value_quant_bits / 8)
        self._n_centroids = cfg.n_centroids if not cfg.key_fp8 else 1

        self._fi_prefill_workspace: torch.Tensor | None = None
        self._fi_prefill_backend = _DEFAULT_TQ_FI_BACKEND
        self._use_flashinfer_prefill = (
            _DEFAULT_TQ_FI_PREFILL
            and BatchPrefillWithRaggedKVCacheWrapper is not None
            and current_platform.is_cuda()
        )
        self._prefill_sliding_window = -1 if sliding_window is None else int(sliding_window)

        # FlashInfer fa2 is available for the SM75 build. Prefer it for raw
        # K/V prefill and continuation instead of routing those paths to SDPA.
        self.fa_version = (
            None
            if self._use_flashinfer_prefill
            else get_flash_attn_version(head_size=head_size)
        )
        if self._use_flashinfer_prefill:
            capability = current_platform.get_device_capability()
            cap_str = capability.as_version_str() if capability is not None else "unknown"
            logger.info_once(
                "TurboQuant prefill is using FlashInfer backend=%s on CUDA "
                "capability %s (flashinfer=%s).",
                self._fi_prefill_backend,
                cap_str,
                _FLASHINFER_VERSION or "unknown",
            )

        # The split count determines both the decode grid and CUDA-graph
        # workspace shape, so it must match the metadata builder's value.
        vllm_config = get_current_vllm_config()
        self.max_num_kv_splits = _tq_max_kv_splits(
            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        )
        if self.max_num_kv_splits != (
            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        ):
            logger.info_once(
                "TurboQuant decode max KV splits overridden to %s by "
                "VLLM_TURBOQUANT_MAX_KV_SPLITS.",
                self.max_num_kv_splits,
            )

        # FlyDSL decode state. Auto-enabled on gfx950 when FlyDSL is available.
        self.sliding_window = sliding_window
        self.sinks = kwargs.get("sinks")
        # Cache max_model_len now (config is available at __init__ but NOT
        # during CUDA-graph capture when _ensure_on_device is re-entered).
        self._max_model_len = vllm_config.model_config.max_model_len
        # SoA store is required by the FlyDSL decode/continuation path, so it
        # tracks FlyDSL availability (single switch for the whole pipeline).
        self._use_flydsl = is_flydsl_available()
        self._soa_store = self._use_flydsl

    def _flash_attn_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        # fa_utils.get_flash_attn_version() returns None on backends that
        # should not pass an explicit fa_version kwarg.
        if self.fa_version is None:
            return flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
            )
        return flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            fa_version=self.fa_version,
        )

    def _ensure_on_device(self, layer, device):
        """One-time derivation of TQ buffers (rotation matrix, midpoints).

        The Hadamard rotation is shared across all layers: random sign
        flips do not improve Lloyd-Max quantization quality because the
        quantizer is symmetric around zero (sign-flipping a coordinate
        maps it to the mirror centroid with identical distortion).
        """
        if self._soa_store:
            # CUDA-graph capture safety for the FlyDSL decode path on ROCm.
            # (1) Pre-allocate _arange_cache / _cu_2 BEFORE any capture; lazy
            #     allocation during graph replay lands in the HIP graph memory
            #     pool and yields stale addresses (GPU fault / garbage).
            # (2) Pre-warm the WorkspaceManager to its max size before capture
            #     so mid-capture growth cannot invalidate pointers baked into
            #     already-captured batch sizes.
            _max_len = self._max_model_len
            _already_ok = (
                hasattr(self, "_arange_cache")
                and self._arange_cache.device.type == str(device).split(":")[0]
                and self._arange_cache.shape[0] >= _max_len + 2
            )
            if not _already_ok:
                self._arange_cache = torch.arange(
                    0, _max_len + 2, device=device, dtype=torch.int32
                )
            if not hasattr(self, "_cu_2") or self._cu_2.device != torch.device(device):
                self._cu_2 = torch.zeros(2, device=device, dtype=torch.int32)
            if (
                is_workspace_manager_initialized()
                and not current_workspace_manager().is_locked()
            ):
                B_max = self._max_capture_batch_size()
                D = self.head_size
                Hq = self.num_heads
                S = self.max_num_kv_splits
                _pre_warm_bytes = (
                    B_max * Hq * (S * (D + 1) + D) * 4  # fp32 mid_o + fp32 lse
                    + B_max * Hq * D * 2  # query-dtype output (bf16 = 2 B)
                    + 512  # alignment padding
                )
                with contextlib.suppress(AssertionError):
                    current_workspace_manager().get_simultaneous(
                        ((_pre_warm_bytes,), torch.uint8)
                    )

        if not hasattr(layer, "_tq_cached"):
            D = self.head_size

            # Pure Hadamard: orthonormal + symmetric (H = H^T), enabling
            # in-kernel butterfly fusion and trivial inverse for continuation.
            H = _build_hadamard(D, str(device))
            layer._tq_PiT = H
            layer._tq_Pi = H
            # fp16 copy for rotation in continuation prefill path
            layer._tq_Pi_half = H.to(torch.float16)

            # Centroids for Lloyd-Max quantization.
            layer._tq_centroids = get_centroids(D, self.tq_config.centroid_bits).to(
                device=device, dtype=torch.float32
            )

            c_sorted, _ = layer._tq_centroids.sort()
            layer._tq_midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
            layer._tq_cached = True

    def _max_capture_batch_size(self) -> int:
        """Largest decode batch we might see at runtime (for workspace pre-warm).

        Take max(cudagraph_capture_sizes, scheduler.max_num_seqs): a forward
        pass exceeding the largest captured graph size falls back to eager,
        but the workspace is locked and that eager path can still hit batch
        sizes up to max_num_seqs. Falls back to 1024 if config is unavailable.
        """
        try:
            cfg = get_current_vllm_config()
            candidates: list[int] = []
            sizes = cfg.compilation_config.cudagraph_capture_sizes
            if sizes:
                candidates.append(int(max(sizes)))
            sched = getattr(cfg, "scheduler_config", None)
            if sched is not None and getattr(sched, "max_num_seqs", None):
                candidates.append(int(sched.max_num_seqs))
            if candidates:
                return max(candidates)
        except Exception:  # noqa: BLE001
            pass
        return 1024

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Store compressed K/V into the combined TQ cache.

        Called as a separate custom op (unified_kv_cache_update) BEFORE
        the attention forward, matching FlashAttention's split pattern.
        slot_mapping is already sliced to num_actual_tokens by the caller.
        """
        N = slot_mapping.shape[0]
        if N <= 0:
            return

        device = key.device
        self._ensure_on_device(layer, device)

        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        # (B, H, N, C) -> (B, N, H, C) for TQ kernels
        kv_cache = kv_cache.transpose(1, 2)
        self._store_kv(k, v, kv_cache, slot_mapping, layer)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "TurboQuantMetadata",
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]

        if output is None:
            output = torch.zeros(
                num_tokens,
                self.num_heads * self.head_size,
                dtype=query.dtype,
                device=query.device,
            )

        if attn_metadata is None:
            return output.fill_(0)

        # (B, H, N, C) -> (B, N, H, C) for TQ kernels
        kv_cache = kv_cache.transpose(1, 2)

        # Slice to actual tokens
        N = attn_metadata.num_actual_tokens
        if N <= 0:
            return output.fill_(0)

        q = query[:N].view(N, self.num_heads, self.head_size)

        # Get TQ buffers, ensure on device (one-time migration).
        # Use Any-typed alias for dynamic _tq_* attrs set by _ensure_on_device.
        tq_layer: Any = layer
        device = q.device
        self._ensure_on_device(tq_layer, device)
        Pi = tq_layer._tq_Pi
        PiT = tq_layer._tq_PiT
        centroids = tq_layer._tq_centroids

        # Compute attention (KV cache was already updated by do_kv_cache_update)
        # With reorder_batch_threshold=1, decodes come first in the batch.
        # num_decodes/num_decode_tokens from metadata give the split point.
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        # [FORK][LANE EF] steps that carry prefill/continuation tokens are long:
        # fence so a stream fault is attributed to the stage that caused it.
        _ef_fence_on = N > num_decode_tokens and not getattr(
            attn_metadata, "force_spec_decode", False
        )
        _ef_ctx = lambda: {  # noqa: E731
            "layer": getattr(layer, "layer_name", None),
            "N": N,
            "num_decodes": num_decodes,
            "num_decode_tokens": num_decode_tokens,
            "max_query_len": attn_metadata.max_query_len,
            "max_seq_len": attn_metadata.max_seq_len,
            "q_lens": (
                attn_metadata.query_start_loc_cpu[1:]
                - attn_metadata.query_start_loc_cpu[:-1]
            ).tolist(),
            "seq_lens": attn_metadata.seq_lens_cpu.tolist(),
        }
        if _ef_fence_on:
            _ef_fence("tq:pre-attn(kv-store+previous-layers)", _ef_ctx)

        if attn_metadata.force_spec_decode:
            if (
                _SPEC_CONTINUATION_DECODE_FASTPATH
                and attn_metadata.max_query_len > 1
            ):
                k = key[:N].view(N, self.num_kv_heads, self.head_size)
                v = value[:N].view(N, self.num_kv_heads, self.head_size)
                attn_out = self._spec_decode_attention_raw_current(
                    q, k, v, kv_cache, attn_metadata, Pi, centroids, PiT
                )
            else:
                attn_out = self._spec_decode_attention(
                    q, kv_cache, attn_metadata, Pi, centroids, PiT
                )
        elif not attn_metadata.is_prefill:
            # Pure decode batch — fast path
            attn_out = self._decode_attention(
                q, kv_cache, attn_metadata, Pi, centroids, PiT, layer
            )
        elif num_decodes == 0:
            # Pure prefill batch
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out = self._prefill_attention(
                q,
                k,
                v,
                kv_cache,
                attn_metadata,
                Pi,
                centroids,
                PiT,
                layer=layer,
            )
        elif num_decode_tokens >= N or num_decodes >= attn_metadata.seq_lens.shape[0]:
            # With spec-as-decode enabled, a uniform MTP continuation has no
            # prefill tail even though its query length is greater than one.
            # Each query needs an incrementing sequence length for causal
            # attention, so it cannot use the regular decode path directly.
            if _SPEC_CONTINUATION_DECODE_FASTPATH:
                k = key[:N].view(N, self.num_kv_heads, self.head_size)
                v = value[:N].view(N, self.num_kv_heads, self.head_size)
                attn_out = self._spec_decode_attention_raw_current(
                    q, k, v, kv_cache, attn_metadata, Pi, centroids, PiT
                )
            else:
                attn_out = self._spec_decode_attention(
                    q, kv_cache, attn_metadata, Pi, centroids, PiT
                )
        else:
            # Mixed batch: decodes first (guaranteed by reorder_batch).
            attn_out = torch.empty(
                N, self.num_heads, self.head_size, device=device, dtype=q.dtype
            )

            # --- Decode portion (first num_decodes requests) ---
            # Use full-batch max_seq_len as safe upper bound (no GPU sync).
            decode_qsl_cpu = (
                attn_metadata.query_start_loc_cpu[: num_decodes + 1]
                if attn_metadata.query_start_loc_cpu is not None
                else None
            )
            decode_qsl = attn_metadata.query_start_loc[: num_decodes + 1]
            if decode_qsl_cpu is not None:
                decode_max_query_len = int(
                    (decode_qsl_cpu[1:] - decode_qsl_cpu[:-1]).max()
                )
            else:
                decode_max_query_len = int(
                    (decode_qsl[1:] - decode_qsl[:-1]).max().item()
                )
            decode_meta = TurboQuantMetadata(
                seq_lens=attn_metadata.seq_lens[:num_decodes],
                slot_mapping=attn_metadata.slot_mapping[:num_decode_tokens],
                block_table=attn_metadata.block_table[:num_decodes],
                query_start_loc=decode_qsl,
                num_actual_tokens=num_decode_tokens,
                max_query_len=decode_max_query_len,
                max_seq_len=attn_metadata.max_seq_len,
                is_prefill=False,
                query_start_loc_cpu=decode_qsl_cpu,
                seq_lens_cpu=(
                    attn_metadata.seq_lens_cpu[:num_decodes]
                    if attn_metadata.seq_lens_cpu is not None
                    else None
                ),
            )
            if decode_max_query_len > 1:
                # A late prefill makes this a genuinely mixed batch. The
                # verifier rows must retain their raw K/V; rereading them
                # from the just-written TurboQuant cache changes speculative
                # logits. Keep the ordinary pure-spec graph route unchanged.
                k = key[:N].view(N, self.num_kv_heads, self.head_size)
                v = value[:N].view(N, self.num_kv_heads, self.head_size)
                attn_out[:num_decode_tokens] = (
                    self._spec_decode_attention_raw_current(
                        q[:num_decode_tokens],
                        k[:num_decode_tokens],
                        v[:num_decode_tokens],
                        kv_cache,
                        decode_meta,
                        Pi,
                        centroids,
                        PiT,
                    )
                )
            else:
                attn_out[:num_decode_tokens] = self._decode_attention(
                    q[:num_decode_tokens],
                    kv_cache,
                    decode_meta,
                    Pi,
                    centroids,
                    PiT,
                    layer,
                )

            if _ef_fence_on:
                _ef_fence("tq:mixed-decode-portion", _ef_ctx)

            # --- Prefill portion (remaining requests) ---
            # CRITICAL: use prefill-specific max_seq_len so flash_attn's
            # fast path (max_query_len == max_seq_len) triggers for
            # first-chunk prefills. Using full-batch max_seq_len breaks
            # this because decode requests inflate max_seq_len.
            prefill_seq_lens = attn_metadata.seq_lens[num_decodes:]
            # Use the CPU-resident `seq_lens` upper-bound from the metadata
            # (populated in the builder) to compute the prefill sub-batch
            # max without a GPU→CPU sync.
            if attn_metadata.seq_lens_cpu is not None:
                prefill_max_seq = int(attn_metadata.seq_lens_cpu[num_decodes:].max())
            else:
                prefill_max_seq = attn_metadata.max_seq_len
            prefill_qsl = (
                attn_metadata.query_start_loc[num_decodes:] - num_decode_tokens
            )
            prefill_qsl_cpu = None
            if attn_metadata.query_start_loc_cpu is not None:
                prefill_qsl_cpu = (
                    attn_metadata.query_start_loc_cpu[num_decodes:] - num_decode_tokens
                )
            prefill_query_start_loc = (
                prefill_qsl_cpu if prefill_qsl_cpu is not None else prefill_qsl
            )
            prefill_max_query_len = int(
                (prefill_query_start_loc[1:] - prefill_query_start_loc[:-1])
                .max()
                .item()
            )
            prefill_meta = TurboQuantMetadata(
                seq_lens=prefill_seq_lens,
                slot_mapping=attn_metadata.slot_mapping[num_decode_tokens:N],
                block_table=attn_metadata.block_table[num_decodes:],
                query_start_loc=prefill_qsl,
                num_actual_tokens=N - num_decode_tokens,
                max_query_len=prefill_max_query_len,
                max_seq_len=prefill_max_seq,
                is_prefill=True,
                query_start_loc_cpu=prefill_qsl_cpu,
                seq_lens_cpu=attn_metadata.seq_lens_cpu[num_decodes:]
                if attn_metadata.seq_lens_cpu is not None
                else None,
                flashinfer_first_chunk_wrappers=(
                    {
                        request_idx - num_decodes: wrapper
                        for request_idx, wrapper in (
                            attn_metadata.flashinfer_first_chunk_wrappers or {}
                        ).items()
                        if request_idx >= num_decodes
                    }
                    or None
                ),
                flashinfer_continuation_wrappers=(
                    {
                        request_idx - num_decodes: wrapper
                        for request_idx, wrapper in (
                            attn_metadata.flashinfer_continuation_wrappers or {}
                        ).items()
                        if request_idx >= num_decodes
                    }
                    or None
                ),
                flashinfer_prefix_combine_wrappers=(
                    {
                        request_idx - num_decodes: wrappers
                        for request_idx, wrappers in (
                            attn_metadata.flashinfer_prefix_combine_wrappers or {}
                        ).items()
                        if request_idx >= num_decodes
                    }
                    or None
                ),
            )
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out[num_decode_tokens:] = self._prefill_attention(
                q[num_decode_tokens:],
                k[num_decode_tokens:],
                v[num_decode_tokens:],
                kv_cache,
                prefill_meta,
                Pi,
                centroids,
                PiT,
                layer=layer,
            )

        if _ef_fence_on:
            _ef_fence("tq:prefill-portion", _ef_ctx)

        # Write into output buffer: attn_out is (N, Hq, D)
        # output may be 2D (N, Hq*D) or 3D (N, Hq, D)
        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    def _spec_decode_attention(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run a multi-token speculative continuation as causal decodes.

        This compressed-cache implementation remains available when the
        raw-current continuation fastpath is explicitly disabled.
        """
        qsl_cpu = attn_metadata.query_start_loc_cpu
        qsl = (
            qsl_cpu.tolist()
            if qsl_cpu is not None
            else attn_metadata.query_start_loc.tolist()
        )
        num_reqs = attn_metadata.seq_lens.shape[0]
        output = torch.empty_like(query)

        max_seq = max(attn_metadata.max_seq_len, attn_metadata.max_query_len)
        arange_cache: torch.Tensor | None = getattr(self, "_arange_cache", None)
        if arange_cache is None or arange_cache.shape[0] <= max_seq:
            arange_cache = torch.arange(
                max_seq + 1,
                device=query.device,
                dtype=attn_metadata.seq_lens.dtype,
            )
            self._arange_cache = arange_cache

        for request_idx in range(num_reqs):
            q_start = qsl[request_idx]
            q_end = qsl[request_idx + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            rel_seq_lens = arange_cache[1 : q_len + 1]
            seq_lens = (
                attn_metadata.seq_lens[request_idx : request_idx + 1]
                - q_len
                + rel_seq_lens
            ).contiguous()
            block_table = (
                attn_metadata.block_table[request_idx : request_idx + 1]
                .expand(q_len, -1)
                .contiguous()
            )
            output[q_start:q_end] = triton_turboquant_decode_attention(
                query=query[q_start:q_end],
                kv_cache=kv_cache,
                block_table=block_table,
                seq_lens=seq_lens,
                Pi=Pi,
                centroids=centroids,
                scale=self.scale,
                mse_bits=self.tq_config.key_mse_bits,
                key_packed_size=self.tq_config.key_packed_size,
                value_quant_bits=self.tq_config.effective_value_quant_bits,
                key_fp8=self.tq_config.key_fp8,
                norm_correction=self.tq_config.norm_correction,
                PiT=PiT,
                max_num_kv_splits=self.max_num_kv_splits,
            ).to(query.dtype)

        return output

    def _spec_decode_attention_raw_current(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run speculative continuation decode with raw current K/V.

        The cache update precedes attention. The just-written verifier chunk is
        lossy in TurboQuant, so a request interrupted by late prefill attends to
        the committed prefix from TQ and the current verifier chunk from raw
        K/V. Pure and mixed MTP batches must use this same attention math to
        avoid changing verifier logits when a late prefill joins the batch.
        """
        qsl_cpu = attn_metadata.query_start_loc_cpu
        qsl = (
            qsl_cpu.tolist()
            if qsl_cpu is not None
            else attn_metadata.query_start_loc.tolist()
        )
        num_reqs = attn_metadata.seq_lens.shape[0]
        output = torch.empty_like(query)

        max_seq = max(attn_metadata.max_seq_len, attn_metadata.max_query_len)
        arange_cache: torch.Tensor | None = getattr(self, "_arange_cache", None)
        if arange_cache is None or arange_cache.shape[0] <= max_seq:
            arange_cache = torch.arange(
                max_seq + 1,
                device=query.device,
                dtype=attn_metadata.seq_lens.dtype,
            )
            self._arange_cache = arange_cache

        for request_idx in range(num_reqs):
            q_start = qsl[request_idx]
            q_end = qsl[request_idx + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            q_seq = query[q_start:q_end]
            k_seq = key[q_start:q_end]
            v_seq = value[q_start:q_end]
            cached_len = (
                attn_metadata.seq_lens[request_idx : request_idx + 1]
                - q_len
            ).clamp_min(0)
            out = self._spec_continuation_decode_attention(
                q_seq,
                k_seq,
                v_seq,
                kv_cache,
                attn_metadata.block_table[request_idx : request_idx + 1],
                cached_len,
                Pi,
                centroids,
                PiT,
                arange_cache,
            )
            if out is None:
                raise RuntimeError(
                    "TurboQuant speculative decode could not construct the "
                    "raw-current verifier attention path."
                )
            output[q_start:q_end] = out.to(query.dtype)

        return output

    # ------------------------------------------------------------------ #
    #  Store K/V into combined cache (vectorized)                         #
    # ------------------------------------------------------------------ #
    def _store_kv(
        self,
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        slot_mapping: torch.Tensor,
        layer: Any,
    ):
        """Quantize + store via fused Triton kernel."""
        if self._soa_store:
            # SoA layout (data region + metadata region separated per block),
            # required by the FlyDSL decode kernel. Pure-Triton store; the
            # cache tensor shape is identical to the default AoS store, only
            # the within-block byte convention differs.
            soa_store, _, _ = _soa_imports()
            soa_store(
                key=key,
                value=value,
                kv_cache=kv_cache,
                slot_mapping=slot_mapping,
                PiT=layer._tq_PiT,
                midpoints=layer._tq_midpoints,
                mse_bits=self.tq_config.key_mse_bits,
                key_packed_size=self.tq_config.key_packed_size,
                value_quant_bits=self.tq_config.effective_value_quant_bits,
                key_fp8=self.tq_config.key_fp8,
                centroids=layer._tq_centroids,
                norm_correction=self.tq_config.norm_correction,
            )
            return
        triton_turboquant_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            layer._tq_PiT,
            layer._tq_midpoints,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            key_fp8=self.tq_config.key_fp8,
        )

    # ------------------------------------------------------------------ #
    #  Prefill: SDPA on raw Q/K/V with causal mask                        #
    # ------------------------------------------------------------------ #
    def _prefill_attention(
        self,
        query: torch.Tensor,  # (N, Hq, D)
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: Any = None,
    ) -> torch.Tensor:
        N, Hq, D = query.shape

        # The builder plans FlashInfer before the model forward. The compiled
        # forward only launches the prepared kernel, preserving CUDA-graph
        # capture for decode/spec-decode.
        if attn_metadata.flashinfer_first_chunk_wrapper is not None:
            return attn_metadata.flashinfer_first_chunk_wrapper.run(query, key, value)

        if (
            _TQ_REQUIRE_FLASHINFER_PREFILL
            and attn_metadata.max_query_len == attn_metadata.max_seq_len
            and not attn_metadata.flashinfer_first_chunk_wrappers
        ):
            raise RuntimeError(
                "TurboQuant fast route could not plan FlashInfer prefill; "
                "refusing FlashAttention/SDPA fallback."
            )

        # Fast path: use flash_attn for first-chunk prefills (all K/V in batch).
        # max_query_len == max_seq_len means no request has prior cached KV.
        # Both are Python ints — no GPU sync.
        if _HAS_FLASH_ATTN and attn_metadata.max_query_len == attn_metadata.max_seq_len:
            return self._flash_attn_varlen(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.max_query_len,
            )

        # Continuation or no flash_attn: per-request attention.
        # For continuation chunks (seq_len > q_len), we must attend to
        # previously cached K/V from the TQ cache, not just the current
        # chunk's raw K/V.
        Hk = key.shape[1]
        use_gqa = Hk < Hq
        query_start_loc = attn_metadata.query_start_loc
        num_reqs = query_start_loc.shape[0] - 1

        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)

        # Prefer the CPU-resident copies from the metadata if populated —
        # otherwise `.tolist()` on GPU tensors forces a synchronizing copy.
        if attn_metadata.query_start_loc_cpu is not None:
            qsl = attn_metadata.query_start_loc_cpu.tolist()
        else:
            qsl = query_start_loc.tolist()
        if attn_metadata.seq_lens_cpu is not None:
            seq_lens_list = attn_metadata.seq_lens_cpu.tolist()
        else:
            seq_lens_list = attn_metadata.seq_lens.tolist()

        # Pre-allocate cu_seqlens for single-request flash_attn calls
        # to avoid per-request host→device tensor creation.
        if not hasattr(self, "_cu_2"):
            self._cu_2 = torch.zeros(2, device=query.device, dtype=torch.int32)
        # Cache arange on self (avoid per-call kernel launch).
        _max_seq = attn_metadata.max_seq_len
        _ac: torch.Tensor | None = getattr(self, "_arange_cache", None)
        if _ac is None or _ac.shape[0] <= _max_seq:
            _ac = torch.arange(
                0, _max_seq + 1, device=query.device, dtype=attn_metadata.seq_lens.dtype
            )
            self._arange_cache = _ac
        _arange_cache: torch.Tensor = _ac

        # [FORK][LANE K2] one INT8 tensor-core call for every short continuation (MTP verifier rows) in the batch
        # instead of one decode call per request.
        done = self._imma_batched_continuations(
            query, kv_cache, attn_metadata, qsl, seq_lens_list, num_reqs, PiT, Pi, centroids, output
        ) if _imma_cuda.enabled() and not self._soa_store else ()

        for i in range(num_reqs):
            if i in done:
                continue
            q_start = qsl[i]
            q_end = qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            seq_len = seq_lens_list[i]
            q_seq = query[q_start:q_end]  # (q_len, Hq, D)
            k_seq = key[q_start:q_end]  # (q_len, Hk, D)
            v_seq = value[q_start:q_end]  # (q_len, Hk, D)

            if q_len == seq_len:
                # First-chunk prefill: all K/V are in the current batch.
                first_chunk_wrapper = (
                    attn_metadata.flashinfer_first_chunk_wrappers.get(i)
                    if attn_metadata.flashinfer_first_chunk_wrappers
                    else None
                )
                if first_chunk_wrapper is not None:
                    out = first_chunk_wrapper.run(q_seq, k_seq, v_seq)
                elif _HAS_FLASH_ATTN:
                    # Assign to slice to avoid gpu/cpu sync.
                    self._cu_2[1:2] = q_len
                    cu = self._cu_2
                    out = self._flash_attn_varlen(
                        q=q_seq,
                        k=k_seq,
                        v=v_seq,
                        cu_seqlens_q=cu,
                        cu_seqlens_k=cu,
                        max_seqlen_q=q_len,
                        max_seqlen_k=q_len,
                    )
                else:
                    q_t = q_seq.transpose(0, 1).contiguous()
                    k_t = k_seq.transpose(0, 1).contiguous()
                    v_t = v_seq.transpose(0, 1).contiguous()
                    out = F.scaled_dot_product_attention(
                        q_t,
                        k_t,
                        v_t,
                        is_causal=True,
                        scale=self.scale,
                        enable_gqa=use_gqa,
                    ).transpose(0, 1)
                output[q_start:q_end] = out.to(query.dtype)
            else:
                # Continuation chunk: tokens already stored to TQ cache
                # by do_kv_cache_update. Use decode kernel directly to
                # avoid O(cached_len) full-dequant per continuation.
                # For large continuations, fall back to _continuation_prefill.
                cached_len = seq_len - q_len
                if q_len <= _CONTINUATION_DECODE_THRESHOLD:
                    # Fast path: treat each query as a decode request
                    # with incremental seq_lens for causal masking.
                    # Slice from pre-built arange (no kernel launch)
                    synth_seq_lens = _arange_cache[cached_len + 1 : seq_len + 1]
                    synth_bt = attn_metadata.block_table[i : i + 1].expand(q_len, -1)
                    if self._soa_store:
                        # The cache was written in SoA layout (always, for FlyDSL),
                        # so it MUST be read with the SoA-aware decode. The
                        # default AoS decode reads k_norm/v_scale/v_zero from
                        # the wrong offsets in a SoA cache -> garbage
                        # cached-prefix output ->
                        # accuracy collapse on every multi-turn / prefix-cached
                        # (APC) request. Continuation stays on the Triton SoA
                        # path even when FlyDSL is the main decode kernel
                        # (FlyDSL is decode-batch only).
                        out = self._dispatch_decode_soa(
                            query=q_seq,
                            kv_cache=kv_cache,
                            block_table=synth_bt,
                            seq_lens=synth_seq_lens,
                            Pi=Pi,
                            centroids=centroids,
                            scale=self.scale,
                            mse_bits=self.tq_config.key_mse_bits,
                            key_packed_size=self.tq_config.key_packed_size,
                            value_quant_bits=(
                                self.tq_config.effective_value_quant_bits
                            ),
                            value_packed_size=self.tq_config.value_packed_size,
                            max_seq_len=int(seq_len),
                            key_fp8=self.tq_config.key_fp8,
                            norm_correction=self.tq_config.norm_correction,
                            PiT=PiT,
                            sinks=self.sinks,
                            sliding_window=self.sliding_window,
                        )
                    else:
                        # The current MTP chunk is already present in the
                        # compressed cache, but reading it back changes the
                        # speculative logits. Keep its raw K/V and merge it
                        # with attention over the compressed prefix.
                        if _SPEC_CONTINUATION_DECODE_FASTPATH and q_len > 1:
                            out = self._spec_continuation_decode_attention(
                                q_seq,
                                k_seq,
                                v_seq,
                                kv_cache,
                                attn_metadata.block_table[i : i + 1],
                                cached_len,
                                Pi,
                                centroids,
                                PiT,
                                _arange_cache,
                            )
                        else:
                            out = None

                        if out is None:
                            out = triton_turboquant_decode_attention(
                                query=q_seq,
                                kv_cache=kv_cache,
                                block_table=synth_bt,
                                seq_lens=synth_seq_lens,
                                Pi=Pi,
                                centroids=centroids,
                                scale=self.scale,
                                mse_bits=self.tq_config.key_mse_bits,
                                key_packed_size=self.tq_config.key_packed_size,
                                value_quant_bits=(
                                    self.tq_config.effective_value_quant_bits
                                ),
                                key_fp8=self.tq_config.key_fp8,
                                norm_correction=self.tq_config.norm_correction,
                                PiT=PiT,
                            )
                else:
                    # Large continuation: dequant cached K/V and use
                    # flash_attn for better throughput.
                    if (
                        _TQ_REQUIRE_FLASHINFER_PREFILL
                        and (
                            (
                                attn_metadata.flashinfer_continuation_wrappers is None
                                or i
                                not in attn_metadata.flashinfer_continuation_wrappers
                            )
                            and (
                                attn_metadata.flashinfer_prefix_combine_wrappers
                                is None
                                or i
                                not in attn_metadata.flashinfer_prefix_combine_wrappers
                            )
                        )
                    ):
                        raise RuntimeError(
                            "TurboQuant fast route could not plan FlashInfer "
                            "continuation prefill; refusing FlashAttention/SDPA "
                            "fallback."
                        )
                    out = self._continuation_prefill(
                        layer,
                        q_seq,
                        k_seq,
                        v_seq,
                        kv_cache,
                        attn_metadata.block_table[i : i + 1],
                        cached_len,
                        seq_len,
                        Pi,
                        centroids,
                        flashinfer_wrapper=(
                            attn_metadata.flashinfer_continuation_wrappers.get(i)
                            if attn_metadata.flashinfer_continuation_wrappers
                            else None
                        ),
                        flashinfer_prefix_combine_wrappers=(
                            attn_metadata.flashinfer_prefix_combine_wrappers.get(i)
                            if attn_metadata.flashinfer_prefix_combine_wrappers
                            else None
                        ),
                    )
                output[q_start:q_end] = out.to(query.dtype)

        return output

    def _imma_batched_continuations(
        self, query, kv_cache, attn_metadata, qsl, seq_lens_list, num_reqs, PiT, Pi, centroids, output
    ) -> set[int]:
        """Lane K2: batch all continuation requests with q_len <= _CONTINUATION_DECODE_THRESHOLD whose rows fit one
        IMMA CTA (q_len * GQA <= 32) into one tq_imma call per q_len.  Same math as the per-request continuation
        decode (each row attends to the TQ cache up to its own causal length).  Returns the request ids handled."""
        Hq, D = query.shape[1], query.shape[2]
        Hk = kv_cache.shape[2]
        if not _imma_cuda.eligible(
            Hq=Hq, Hk=Hk, D=D, mse_bits=self.tq_config.key_mse_bits,
            value_quant_bits=self.tq_config.effective_value_quant_bits, key_fp8=self.tq_config.key_fp8,
            key_packed_size=self.tq_config.key_packed_size, q_per_seq=1, block_size=kv_cache.shape[1],
        ) or query.dtype != torch.float16:
            return set()
        G = Hq // Hk
        groups: dict[int, list[int]] = {}
        for i in range(num_reqs):
            q_len = qsl[i + 1] - qsl[i]
            if q_len <= 0 or q_len >= seq_lens_list[i] or q_len > _CONTINUATION_DECODE_THRESHOLD:
                continue
            if _SPEC_CONTINUATION_DECODE_FASTPATH and q_len > 1:
                continue
            if q_len * G > 32:
                continue
            groups.setdefault(q_len, []).append(i)
        done: set[int] = set()
        dev = query.device
        for q_len, reqs in groups.items():
            n = len(reqs)
            rows = [r for i in reqs for r in range(qsl[i], qsl[i] + q_len)]
            lens = [seq_lens_list[i] - q_len + 1 + j for i in reqs for j in range(q_len)]
            contiguous = rows == list(range(rows[0], rows[0] + len(rows)))
            host = torch.tensor(lens + reqs + rows, dtype=torch.int32, pin_memory=True)
            meta = host.to(dev, non_blocking=True)
            row_lens = meta[: n * q_len]
            req_idx = meta[n * q_len: n * q_len + n].long()
            q_rows = query[rows[0]: rows[0] + len(rows)] if contiguous else query.index_select(0, meta[n * q_len + n:].long())
            bt = attn_metadata.block_table.index_select(0, req_idx)
            out, _ = _imma_cuda.tq_imma_decode_attention(
                q_rows, kv_cache, bt, row_lens, centroids, self.scale, self.tq_config.norm_correction,
                q_per_seq=q_len, num_seqs=n, PiT=PiT if PiT is not None else Pi.T.contiguous(),
                max_len_hint=max(lens),
            )
            if contiguous:
                output[rows[0]: rows[0] + len(rows)] = out.to(output.dtype)
            else:
                output.index_copy_(0, meta[n * q_len + n:].long(), out.to(output.dtype))
            done.update(reqs)
        return done

    def _spec_continuation_decode_attention(
        self,
        query: torch.Tensor,
        key_chunk: torch.Tensor,
        value_chunk: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cached_len: int | torch.Tensor,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None,
        arange_cache: torch.Tensor,
    ) -> torch.Tensor | None:
        """Merge compressed-prefix attention with raw MTP continuation KV.

        The cache update runs before attention, so reading the current
        speculative tokens from the quantized cache is lossy. Computing the
        prefix once with the TQ kernel and the raw current chunk with an
        online-softmax merge preserves MTP quality without full prefix dequant.
        """
        if isinstance(cached_len, int) and cached_len <= 0:
            return None

        q_len, Hq, D = query.shape
        Hk = key_chunk.shape[1]
        if Hk <= 0 or Hq % Hk != 0:
            return None

        if isinstance(cached_len, int):
            prefix_seq_lens = torch.full(
                (q_len,),
                cached_len,
                dtype=arange_cache.dtype,
                device=query.device,
            )
            has_prefix = cached_len > 0
        else:
            # The capture warmup has no cached prefix. Keep that zero length:
            # its block table does not name a valid TQ page yet. The Triton
            # stage still evaluates page addresses even for seq_len=0, so a
            # sentinel entry can poison the captured graph with NaNs. Select
            # a legal zero page for capture and switch to the replay page
            # table once the real prefix length becomes positive.
            # The decode kernel indexes seq_lens as a contiguous B-element
            # array. ``expand`` alone leaves one backing element with stride
            # zero, so rows after the first read outside its storage.
            prefix_seq_lens = cached_len.reshape(1).expand(q_len).contiguous()
            has_prefix = cached_len > 0
        safe_block_table = block_table
        if not isinstance(cached_len, int):
            safe_block_table = torch.where(
                has_prefix.reshape(1, 1),
                block_table,
                torch.zeros_like(block_table),
            )
        prefix_lse = torch.empty(q_len, Hq, dtype=torch.float32, device=query.device)
        prefix_out = torch.empty_like(query)
        if (
            _gqa_cuda.enabled()
            and query.dtype == torch.float16
            and _gqa_cuda.gqa_eligible(
                Hq=Hq, Hk=Hk, D=D, mse_bits=self.tq_config.key_mse_bits,
                value_quant_bits=self.tq_config.effective_value_quant_bits,
                key_fp8=self.tq_config.key_fp8, key_packed_size=self.tq_config.key_packed_size,
            )
        ):
            # lane S2: all q_len speculative rows share the same cached prefix -> one grouped kernel call
            # (each cached token dequantized once, scored against q_len * GQA-group rows on tensor cores).
            _gqa_cuda.tq_gqa_decode_attention(
                query, kv_cache, safe_block_table, prefix_seq_lens[:1].to(torch.int32), Pi, centroids,
                self.scale, self.tq_config.norm_correction, q_per_seq=q_len,
                num_splits=self.max_num_kv_splits, PiT=PiT, output_buf=prefix_out, lse_buf=prefix_lse,
            )
        elif _TQ_CUDAGRAPH_SPEC_PREFIX_ROWS and not isinstance(cached_len, int):
            # All MTP candidates attend to the same compressed prefix. On
            # SM75, capture the fixed-width candidates as individual B=1 TQ
            # decodes to avoid the B=4 materialized page-table path. This is
            # still fully captured and replayed by CUDA Graph; only the graph
            # topology differs.
            prefix_seq_len = cached_len.reshape(1)
            for row in range(q_len):
                triton_turboquant_decode_attention(
                    query=query[row : row + 1],
                    kv_cache=kv_cache,
                    block_table=safe_block_table,
                    seq_lens=prefix_seq_len,
                    Pi=Pi,
                    centroids=centroids,
                    scale=self.scale,
                    mse_bits=self.tq_config.key_mse_bits,
                    key_packed_size=self.tq_config.key_packed_size,
                    value_quant_bits=self.tq_config.effective_value_quant_bits,
                    key_fp8=self.tq_config.key_fp8,
                    norm_correction=self.tq_config.norm_correction,
                    PiT=PiT,
                    output_buf=prefix_out[row : row + 1],
                    lse_buf=prefix_lse[row : row + 1],
                    max_num_kv_splits=self.max_num_kv_splits,
                )
        else:
            prefix_bt = safe_block_table.expand(q_len, -1).contiguous()
            triton_turboquant_decode_attention(
                query=query,
                kv_cache=kv_cache,
                block_table=prefix_bt,
                seq_lens=prefix_seq_lens,
                Pi=Pi,
                centroids=centroids,
                scale=self.scale,
                mse_bits=self.tq_config.key_mse_bits,
                key_packed_size=self.tq_config.key_packed_size,
                value_quant_bits=self.tq_config.effective_value_quant_bits,
                key_fp8=self.tq_config.key_fp8,
                norm_correction=self.tq_config.norm_correction,
                PiT=PiT,
                output_buf=prefix_out,
                lse_buf=prefix_lse,
                max_num_kv_splits=self.max_num_kv_splits,
            )
        if not isinstance(has_prefix, bool):
            prefix_out = torch.where(
                has_prefix.reshape(1, 1, 1),
                prefix_out,
                torch.zeros_like(prefix_out),
            )
            prefix_lse = torch.where(
                has_prefix.reshape(1, 1),
                prefix_lse,
                torch.full_like(prefix_lse, float("-inf")),
            )

        return triton_turboquant_spec_continuation_merge(
            query,
            key_chunk,
            value_chunk,
            prefix_out,
            prefix_lse,
            self.scale,
            # The prefix state has been consumed by every program when its
            # matching output row is written, so it doubles as the final
            # output buffer without an additional graph-captured allocation.
            output=prefix_out,
        )

    def _continuation_prefill(
        self,
        layer: Any,
        query: torch.Tensor,  # (q_len, Hq, D)
        key_chunk: torch.Tensor,  # (q_len, Hk, D)
        val_chunk: torch.Tensor,  # (q_len, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        block_table: torch.Tensor,  # (1, max_num_blocks)
        cached_len: int,
        seq_len: int,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        flashinfer_wrapper: Any | None = None,
        flashinfer_prefix_combine_wrappers: tuple[Any, Any] | None = None,
    ) -> torch.Tensor:
        """Handle continuation chunk by dequanting cached K/V from TQ cache.

        Dequants previously cached K/V, concatenates with the current
        chunk's raw K/V, then runs flash_attn with causal masking.
        """
        q_len, Hq, D = query.shape
        Hk = key_chunk.shape[1]
        device = query.device
        block_size = kv_cache.shape[1]
        BLOCK_D = triton.next_power_of_2(D)

        # [FORK][LANE K2] prefix attention straight on the TQ codes with INT8 tensor cores (no full dequant, no
        # rotation pass, no fp16 FlashInfer over the prefix); the chunk's own causal part stays on FlashInfer.
        if (
            flashinfer_prefix_combine_wrappers is not None
            and _imma_cuda.prefill_enabled()
            and query.dtype == torch.float16
            and not self._soa_store
            and cached_len > 0
            and _imma_cuda.eligible(
                Hq=Hq, Hk=Hk, D=D, mse_bits=self.tq_config.key_mse_bits,
                value_quant_bits=self.tq_config.effective_value_quant_bits, key_fp8=self.tq_config.key_fp8,
                key_packed_size=self.tq_config.key_packed_size, q_per_seq=1, block_size=block_size,
            )
        ):
            _, current_wrapper = flashinfer_prefix_combine_wrappers
            prefix_out, prefix_lse = _imma_cuda.tq_imma_prefill_prefix_attention(
                query, kv_cache, block_table[:1], cached_len, centroids, self.scale,
                self.tq_config.norm_correction, layer._tq_PiT,
            )
            current_out = torch.empty_like(query)
            current_lse = torch.empty((q_len, Hq), dtype=torch.float32, device=device)
            current_out, current_lse = current_wrapper.run(
                query, key_chunk, val_chunk, out=current_out, lse=current_lse, return_lse=True
            )
            logger.info_once("TurboQuant continuation prefix via IMMA (lane K2): cached_len=%s q_len=%s",
                             cached_len, q_len)
            # K2 LSE is natural-log, FlashInfer's is base 2: express both in base 2 and merge as such.
            return _tq_merge_flashinfer_partials(
                prefix_out, prefix_lse * (1.0 / _LN2), current_out, current_lse, lse_base2=True
            )

        mse_bytes = self._mse_bytes
        val_data_bytes = self._val_data_bytes

        # Dequant cached K/V from TQ cache.
        # [FORK][LANE S3] The dequant kernel writes straight into the
        # (rows, Hk, D) buffers that attention reads (strided output), the
        # inverse key rotation is applied in place in bounded token chunks,
        # and the buffers come from the WorkspaceManager (reserved at
        # max_model_len and counted by memory profiling). The previous code
        # also allocated k_flat, k_full and v_full per call, each
        # cached_len*Hk*D*2 bytes (268 MB at 262K tokens, 537 MB at 524K) with
        # ~0.3 GiB free, which OOM-killed the engine at long context.
        alloc_len = math.ceil(cached_len / block_size) * block_size
        rows = max(alloc_len, math.ceil(seq_len / block_size) * block_size)
        k_rows, v_rows = current_workspace_manager().get_simultaneous(
            ((rows, Hk, D), torch.float16),
            ((rows, Hk, D), torch.float16),
        )
        # Dequant layout view (1, Hk, rows, D) over the (rows, Hk, D) buffer.
        k_cached = k_rows.permute(1, 0, 2).unsqueeze(0)
        v_cached = v_rows.permute(1, 0, 2).unsqueeze(0)

        grid = (alloc_len, 1 * Hk)
        if self._soa_store:
            # SoA-aware dequant: read the data/metadata-separated SoA cache
            # written by the SoA store. Constants must match the store side.
            _, soa_dequant, _ = _soa_imports()
            from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_decode import (
                _use_fp8_e4b15,
            )
            key_fp8 = self.tq_config.key_fp8
            key_data_bytes = D if key_fp8 else mse_bytes
            data_bytes_per_slot = key_data_bytes + val_data_bytes
            meta_region_offset = block_size * Hk * data_bytes_per_slot
            num_soa_fields = 2 if key_fp8 else 3
            soa_k_norm = 0
            soa_v_scale = 0 if key_fp8 else 1
            soa_v_zero = 1 if key_fp8 else 2
            kv_cache_u16 = kv_cache.view(torch.uint16)
            soa_dequant[grid](
                kv_cache,
                kv_cache_u16,
                block_table,
                centroids,
                k_cached,
                v_cached,
                k_cached.stride(0),
                k_cached.stride(1),
                k_cached.stride(2),
                v_cached.stride(0),
                v_cached.stride(1),
                v_cached.stride(2),
                kv_cache.stride(0),
                block_table.stride(0),
                HEAD_DIM=D,
                BLOCK_SIZE=block_size,
                NUM_KV_HEADS=Hk,
                MSE_BYTES=mse_bytes,
                VQB=self.tq_config.effective_value_quant_bits,
                VAL_DATA_BYTES=val_data_bytes,
                MSE_BITS=self.tq_config.key_mse_bits,
                KEY_FP8=1 if key_fp8 else 0,
                KEY_DATA_BYTES=key_data_bytes,
                META_REGION_OFFSET=meta_region_offset,
                NUM_SOA_FIELDS=num_soa_fields,
                SOA_K_NORM=soa_k_norm,
                SOA_V_SCALE=soa_v_scale,
                SOA_V_ZERO=soa_v_zero,
                BLOCK_D=BLOCK_D,
                NORM_CORRECTION=1 if self.tq_config.norm_correction else 0,
                FP8_E4B15=_use_fp8_e4b15(device.index or 0),
                num_warps=4,
            )
        else:
            _tq_full_dequant_kv[grid](
                kv_cache,
                block_table,
                centroids,
                k_cached,
                v_cached,
                k_cached.stride(0),
                k_cached.stride(1),
                k_cached.stride(2),
                v_cached.stride(0),
                v_cached.stride(1),
                v_cached.stride(2),
                kv_cache.stride(0),
                kv_cache.stride(1),
                kv_cache.stride(2),
                block_table.stride(0),
                HEAD_DIM=D,
                BLOCK_SIZE=block_size,
                NUM_KV_HEADS=Hk,
                MSE_BYTES=mse_bytes,
                KPS=self.tq_config.key_packed_size,
                VQB=self.tq_config.effective_value_quant_bits,
                VAL_DATA_BYTES=val_data_bytes,
                MSE_BITS=self.tq_config.key_mse_bits,
                KEY_FP8=1 if self.tq_config.key_fp8 else 0,
                BLOCK_D=BLOCK_D,
                NORM_CORRECTION=1 if self.tq_config.norm_correction else 0,
                FP8_FORMAT=_fp8_format_code(device.index or 0),
                num_warps=4,
            )

        # Inverse-rotate MSE keys back to original space, in place, in
        # bounded chunks (temporary <= _TQ_CONT_ROT_CHUNK*Hk*D*2 bytes).
        if not self.tq_config.key_fp8:
            Pi_half = layer._tq_Pi_half
            for c0 in range(0, cached_len, _TQ_CONT_ROT_CHUNK):
                c1 = min(c0 + _TQ_CONT_ROT_CHUNK, cached_len)
                blk = k_rows[c0:c1]
                blk.copy_((blk.reshape(-1, D) @ Pi_half).reshape(c1 - c0, Hk, D))

        qdtype = query.dtype
        if qdtype == torch.float16:
            k_full = k_rows[:seq_len]
            v_full = v_rows[:seq_len]
            k_full[cached_len:] = key_chunk
            v_full[cached_len:] = val_chunk
            k_cached_trim = k_rows[:cached_len]
            v_cached_trim = v_rows[:cached_len]
        else:
            k_full = k_rows[:seq_len].to(qdtype)
            v_full = v_rows[:seq_len].to(qdtype)
            k_full[cached_len:] = key_chunk
            v_full[cached_len:] = val_chunk
            k_cached_trim = k_full[:cached_len]
            v_cached_trim = v_full[:cached_len]

        if flashinfer_prefix_combine_wrappers is not None:
            prefix_wrapper, current_wrapper = flashinfer_prefix_combine_wrappers
            prefix_out = torch.empty_like(query)
            prefix_lse = torch.empty(
                (q_len, Hq), dtype=torch.float32, device=device
            )
            prefix_out, prefix_lse = prefix_wrapper.run(
                query,
                k_cached_trim,
                v_cached_trim,
                out=prefix_out,
                lse=prefix_lse,
                return_lse=True,
            )
            current_out = torch.empty_like(query)
            current_lse = torch.empty(
                (q_len, Hq), dtype=torch.float32, device=device
            )
            current_out, current_lse = current_wrapper.run(
                query,
                key_chunk,
                val_chunk,
                out=current_out,
                lse=current_lse,
                return_lse=True,
            )
            merged_out = _tq_merge_flashinfer_partials(
                prefix_out, prefix_lse, current_out, current_lse
            )
            logger.info_once(
                "TurboQuant continuation prefix-combine path used: "
                "mode=%s min_tokens=%s seq_len=%s cached_len=%s q_len=%s",
                _TQ_CONTINUATION_PREFIX_COMBINE_MODE,
                _TQ_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS,
                seq_len,
                cached_len,
                q_len,
            )
            return merged_out

        if flashinfer_wrapper is not None:
            return flashinfer_wrapper.run(query, k_full, v_full)

        # Attention: q_len queries attending to seq_len K/V with causal mask
        if _HAS_FLASH_ATTN:
            # Reuse pre-allocated cu_seqlens (avoid host→device transfer)
            if not hasattr(self, "_cu_2_q"):
                self._cu_2_q = torch.zeros(2, device=device, dtype=torch.int32)
                self._cu_2_k = torch.zeros(2, device=device, dtype=torch.int32)
            # Assigning to slice uses fill_ which avoids cpu/gpu sync.
            self._cu_2_q[1:2] = q_len
            self._cu_2_k[1:2] = seq_len
            cu_seqlens_q = self._cu_2_q
            cu_seqlens_k = self._cu_2_k
            return self._flash_attn_varlen(
                q=query,
                k=k_full,
                v=v_full,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=q_len,
                max_seqlen_k=seq_len,
            )
        else:
            # SDPA fallback: expand KV for GQA, build causal mask
            q_t = query.transpose(0, 1).unsqueeze(0)  # (1, Hq, q_len, D)
            k_t = k_full.transpose(0, 1).unsqueeze(0)  # (1, Hk, seq_len, D)
            v_t = v_full.transpose(0, 1).unsqueeze(0)  # (1, Hk, seq_len, D)
            # Build causal mask: query position p can attend to K position j
            # where j <= cached_len + p (p is 0-indexed within chunk)
            q_pos = torch.arange(q_len, device=device).unsqueeze(1) + cached_len
            k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
            mask = k_pos <= q_pos  # (q_len, seq_len)
            out = F.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                attn_mask=mask,
                scale=self.scale,
                enable_gqa=(Hk < Hq),
            )  # (1, Hq, q_len, D)
            return out[0].transpose(0, 1)  # (q_len, Hq, D)

    # ------------------------------------------------------------------ #
    #  Decode: Triton TQ decode attention                                 #
    # ------------------------------------------------------------------ #
    def _decode_attention(
        self,
        query: torch.Tensor,  # (B, Hq, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        # Acquire shared decode scratch buffers from WorkspaceManager.
        # Layers execute sequentially so one set of buffers is sufficient.
        # Falls back to kernel-internal allocation if workspace unavailable.
        B = query.shape[0]
        D = self.head_size
        S = self.max_num_kv_splits
        Hq = self.num_heads
        mid_o_buf = output_buf = lse_buf = None
        if is_workspace_manager_initialized():
            # output_buf in query dtype — matches the in-kernel fp16 cast in stage2.
            mid_o_buf, output_buf, lse_buf = (
                current_workspace_manager().get_simultaneous(
                    ((B, Hq, S, D + 1), torch.float32),
                    ((B, Hq, D), query.dtype),
                    ((B, Hq), torch.float32),
                )
            )

        if self._use_flydsl:
            # FlyDSL decode (gfx950, MSE-key, HEAD_SIZE=128, GQA in {6, 8, 16}).
            # GQA-6 routes to the MiniMax sibling kernel. Ineligible layers fall
            # back to SoA Triton decode.
            _gqa = self.num_kv_groups
            flydsl_gqa_ok = (_gqa in (8, 16)) or (
                _gqa == 6 and is_flydsl_gqa6_available()
            )
            flydsl_eligible = (
                not self.tq_config.key_fp8
                and self.tq_config.key_mse_bits == 4
                and self.tq_config.effective_value_quant_bits == 4
                and self.head_size == 128
                and flydsl_gqa_ok
                and self.sinks is None
                and not (self.sliding_window and self.sliding_window > 0)
            )
            if flydsl_eligible:
                return flydsl_turboquant_decode_attention(
                    query=query,
                    kv_cache=kv_cache,
                    block_table=attn_metadata.block_table,
                    seq_lens=attn_metadata.seq_lens,
                    Pi=Pi,
                    centroids=centroids,
                    scale=self.scale,
                    mse_bits=self.tq_config.key_mse_bits,
                    key_packed_size=self.tq_config.key_packed_size,
                    value_quant_bits=self.tq_config.effective_value_quant_bits,
                    value_packed_size=self.tq_config.value_packed_size,
                    max_seq_len=attn_metadata.max_seq_len,
                    key_fp8=self.tq_config.key_fp8,
                    norm_correction=self.tq_config.norm_correction,
                    PiT=PiT,
                    mid_o_buf=mid_o_buf,
                    output_buf=output_buf,
                    lse_buf=lse_buf,
                    buf_holder=layer,
                    max_num_kv_splits=self.max_num_kv_splits,
                    sinks=self.sinks,
                )
            logger.warning_once(
                "TurboQuant FlyDSL ineligible (key_fp8=%s mse_bits=%s vqb=%s "
                "head_size=%s num_kv_groups=%s sinks=%s) -> SoA Triton decode",
                self.tq_config.key_fp8,
                self.tq_config.key_mse_bits,
                self.tq_config.effective_value_quant_bits,
                self.head_size,
                self.num_kv_groups,
                self.sinks is not None,
            )
            return self._dispatch_decode_soa(
                query=query,
                kv_cache=kv_cache,
                block_table=attn_metadata.block_table,
                seq_lens=attn_metadata.seq_lens,
                Pi=Pi,
                centroids=centroids,
                scale=self.scale,
                mse_bits=self.tq_config.key_mse_bits,
                key_packed_size=self.tq_config.key_packed_size,
                value_quant_bits=self.tq_config.effective_value_quant_bits,
                value_packed_size=self.tq_config.value_packed_size,
                max_seq_len=attn_metadata.max_seq_len,
                key_fp8=self.tq_config.key_fp8,
                norm_correction=self.tq_config.norm_correction,
                PiT=PiT,
                mid_o_buf=mid_o_buf,
                output_buf=output_buf,
                lse_buf=lse_buf,
                buf_holder=layer,
                max_num_kv_splits=self.max_num_kv_splits,
                sinks=self.sinks,
                sliding_window=self.sliding_window,
            )

        result = triton_turboquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            Pi=Pi,
            centroids=centroids,
            scale=self.scale,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            key_fp8=self.tq_config.key_fp8,
            norm_correction=self.tq_config.norm_correction,
            PiT=PiT,
            mid_o_buf=mid_o_buf,
            output_buf=output_buf,
            lse_buf=lse_buf,
            buf_holder=layer,
            max_num_kv_splits=self.max_num_kv_splits,
        )
        return result

    def _dispatch_decode_soa(self, **kwargs):
        """SoA-aware Triton decode — fallback for FlyDSL-ineligible layers.

        The SoA decode launcher accepts a subset of the FlyDSL kwargs; filter to
        the params it accepts and raise if a *meaningful* (non-None) kwarg would
        be silently dropped (so we never mask a real feature gap).
        """
        import inspect

        _, _, soa_decode = _soa_imports()
        accepted = set(inspect.signature(soa_decode).parameters)
        dropped = [k for k, v in kwargs.items() if k not in accepted and v is not None]
        if dropped:
            raise NotImplementedError(
                f"SoA decode does not support kwargs {sorted(dropped)}"
            )
        return soa_decode(**{k: v for k, v in kwargs.items() if k in accepted})
