# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA QSA owner with Triton kernels."""

from __future__ import annotations

from typing import ClassVar, cast

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding, get_rope
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.utils.torch_utils import (
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

from ..common.qsa_cache import QSAForwardMetadata
from ..common.qsa_int8 import (
    INT8_KV_FALLBACK_SCALE,
    int8_kv_scale_from_fp8_scale,
)
from . import model
from .indexer_qsa import QSAIndexer


_QSA_DTYPES = (torch.bfloat16, torch.float16)
# "int8": per-layer symmetric int8 main cache (thecyberdevl 0026), dequantized
# inside the QSA Triton kernel; the side caches stay in the model dtype.
_QSA_KV_CACHE_DTYPES = ("auto", "bfloat16", "float16", "int8")

logger = init_logger(__name__)

# Opt-in K/V amax recorder for checkpoints that carry no k_scale/v_scale (EXL3
# packs): run once with a 16-bit cache and VLLM_QWEN4_QSA_KV_AMAX_DUMP=<path>; each
# rank writes <path>.rank<N>.json at exit, and tools/qwen4exp/kvscales_from_amax.py
# turns those into model-kvscales.safetensors for --kv-cache-dtype int8.
_KV_AMAX_DUMP = __import__("os").environ.get("VLLM_QWEN4_QSA_KV_AMAX_DUMP", "")
_KV_AMAX_OWNERS: list = []
# TP rank of this process's owners, looked up when the first one registers: the
# atexit dump runs after WorkerProc.shutdown() has destroyed the model-parallel
# groups, where the lookup asserts (every rank fell back to rank 0 and the last
# writer overwrote the others' K/V heads).
_KV_AMAX_RANK: list[int] = []


def _kv_amax_register(owner) -> None:
    if not _KV_AMAX_RANK:
        try:
            from vllm.distributed import get_tensor_model_parallel_rank

            _KV_AMAX_RANK.append(get_tensor_model_parallel_rank())
        except Exception:  # noqa: BLE001  (no TP group: a single-process build)
            _KV_AMAX_RANK.append(0)
    _KV_AMAX_OWNERS.append(owner)


def _dump_kv_amax() -> None:
    import json
    import os

    if not _KV_AMAX_OWNERS:
        return
    rank = _KV_AMAX_RANK[0] if _KV_AMAX_RANK else 0
    table = {
        owner.layer_name: [float(owner._kv_amax[0]), float(owner._kv_amax[1])]
        for owner in _KV_AMAX_OWNERS
        if owner._kv_amax is not None
    }
    path = f"{_KV_AMAX_DUMP}.rank{rank}.json"
    with open(path, "w") as fh:
        json.dump(table, fh, indent=1, sort_keys=True)
    logger.info("QSA K/V amax for %d layers written to %s", len(table), path)


if _KV_AMAX_DUMP:
    __import__("atexit").register(_dump_kv_amax)


class Qwen4ExpQSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """Flash metadata supporting uniform decode and target-verify graphs."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class Qwen4ExpQSAFlashAttentionBackend(FlashAttentionBackend):
    """FullAttentionSpec backend used by the merged QSA owner."""

    # The sparse attention itself is Triton (``ops/qsa.py``) and the KV write is
    # ``reshape_and_cache_flash`` from ``_C``; neither needs the FlashAttention
    # binary, so FP16 on Turing (SM75, no BF16) is supported.
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16, torch.float16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "float16",
        "int8",
    ]

    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_QSA_TRITON"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # QSA consumes manager pages directly and does not use FA4 paged attention.
        return [MultipleOf(16)]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # No FlashAttention kernel is launched (Triton tl.dot fp16 MMA: SM75+).
        return capability >= DeviceCapability(7, 5)

    @staticmethod
    def get_impl_cls() -> type[Qwen4ExpQSAFlashAttentionImpl]:
        return Qwen4ExpQSAFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Qwen4ExpQSAMetadataBuilder]:
        return Qwen4ExpQSAMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
    """Run paged sparse GQA with the QSA Triton kernel."""

    supports_dcp: bool = False
    supports_pcp: bool = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.dcp_world_size != 1:
            raise NotImplementedError(
                "Qwen4Exp QSA does not support decode context parallelism"
            )
        if self.kv_cache_dtype not in _QSA_KV_CACHE_DTYPES:
            raise NotImplementedError(
                "Qwen4Exp QSA requires a 16-bit or --kv-cache-dtype int8 main KV cache"
            )
        self.kv_int8 = self.kv_cache_dtype == "int8"
        self.supports_quant_query_input = False

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # FlashAttentionImpl.do_kv_cache_update calls the reshape_and_cache_flash that
        # flash_attn.py imports only when flash-attn is available, which it never is on
        # SM75. On Turing the first K/V write hit a NameError, right after the profile
        # run. The scatter is vLLM's own cache kernel, the one fa_utils binds on CUDA, so
        # call it directly with the same (B, H, N, 2D) -> 2 x (B, N, H, D) views.
        from vllm._custom_ops import reshape_and_cache_flash

        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        token_to_req: torch.Tensor,
        use_prefill_config: bool,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QSA does not support fused output quantization")
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError("QSA does not support ALiBi or attention sinks")
        if self.sliding_window != (-1, -1):
            raise NotImplementedError("QSA does not support sliding-window attention")

        num_tokens = attn_metadata.num_actual_tokens
        output.zero_()
        if num_tokens == 0:
            return output

        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        logical_indices = topk_buffer[:num_tokens]
        token_to_req = token_to_req[:num_tokens]
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        k_scale = v_scale = 1.0
        if query.dtype not in _QSA_DTYPES:
            raise NotImplementedError("Qwen4Exp QSA requires a BF16/FP16 query")
        if self.kv_int8:
            if key_cache.dtype != torch.int8:
                raise NotImplementedError(
                    "Qwen4Exp QSA int8 mode requires an INT8 main KV cache"
                )
            k_scale, v_scale = layer.qsa_int8_scales()
        elif key_cache.dtype != query.dtype:
            raise NotImplementedError("Qwen4Exp QSA requires matching BF16/FP16 Q/K/V")

        from .ops.qsa import qsa_sparse_paged_attention

        qsa_sparse_paged_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            use_prefill_config,
            output[:num_tokens],
            k_scale,
            v_scale,
        )
        return output


class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA index side branch."""

    supports_dcp = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen4Exp QSA requires a paged KV cache")
        if model_config.dtype not in _QSA_DTYPES:
            raise NotImplementedError("Qwen4Exp QSA requires a BF16 or FP16 model")
        if cache_config.cache_dtype not in _QSA_KV_CACHE_DTYPES:
            raise NotImplementedError(
                "Qwen4Exp QSA requires a 16-bit or --kv-cache-dtype int8 main KV cache"
            )
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support KV quantization")
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA does not support context parallelism"
            )
        if not getattr(config, "is_causal", True):
            raise NotImplementedError("Qwen4Exp QSA requires causal decoder attention")

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        # Decode/verify batches have at most 1 + num_spec query tokens per
        # request; use_prefill_config (max_query_len > this) steers the
        # config table. Shorter batches take the decode profile — harmless,
        # the difference is tile-shape tuning, not correctness.
        self._max_decode_query_len = 1 + vllm_config.num_speculative_tokens
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support dual-chunk RoPE")
        # Qwen4Exp full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=model.without_modelopt_fp4(quant_config),
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        mm_config = model_config.multimodal_config
        text_only = mm_config is None or mm_config.language_model_only
        mrope_section = getattr(self.rotary_emb, "mrope_section", None)
        supports_mrope = bool(
            type(self.rotary_emb) is MRotaryEmbedding
            and mrope_section
            and len(mrope_section) == 3
            and sum(mrope_section) == self.rotary_emb.rotary_dim // 2
            and getattr(self.rotary_emb, "mrope_interleaved", False)
        )
        supports_dtype = getattr(self.rotary_emb, "dtype", None) in (
            torch.float16,
            torch.bfloat16,
        )
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_cuda()
            and supports_dtype
            and (text_only or supports_mrope)
        )

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = cache_config.cache_dtype
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        # Q (and the kernel's upcast K/V operands) always use the model dtype.
        self.attn_dtype = model_config.dtype
        self.kv_int8 = self.kv_cache_dtype == "int8"
        if self.kv_int8:
            if self.kv_cache_torch_dtype != torch.int8:
                raise RuntimeError("int8 KV cache resolved to a non-int8 dtype")
        elif self.kv_cache_torch_dtype != model_config.dtype:
            raise NotImplementedError(
                "Qwen4Exp QSA requires the KV cache dtype to equal the model dtype "
                "(or --kv-cache-dtype int8)"
            )
        # int8 storage scales, derived once from the loaded fp8-contract
        # _k_scale/_v_scale buffers (finalize_kv_cache_scales).
        self._int8_scales: tuple[float, float] | None = None
        # device-side running amax of the stored K and V rows (opt-in recorder)
        self._kv_amax: torch.Tensor | None = None
        if _KV_AMAX_DUMP and not self.kv_int8:
            _kv_amax_register(self)
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)

        self.attn_backend = Qwen4ExpQSAFlashAttentionBackend
        self.impl = Qwen4ExpQSAFlashAttentionImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            None,
            None,
            self.kv_cache_dtype,
            None,
            AttentionType.DECODER,
            None,
        )
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        # PACKED selection buffer: the trailing column holds each row's
        # valid-entry count (written by the expand kernel) — never a token
        # index; the sparse attention kernel reads it as its loop bound.
        # MTP skip_topk steps reuse rows frozen from step 0; the count is
        # a row column, so compaction/reuse keep it paired with the content.
        self.register_buffer(
            "topk_indices_buffer",
            torch.empty(
                max_tokens,
                self.indexer.packed_output_width,
                dtype=torch.int32,
            ),
            persistent=False,
        )

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def finalize_kv_cache_scales(self) -> tuple[float, float]:
        """Derive the int8 storage scales from the loaded fp8 scales (thecyberdevl 0026).

        Checkpoint ``k_scale``/``v_scale`` follow the fp8 contract ``s = max_abs / 448``;
        the int8 cache stores ``round(x / s_int8)`` with ``s_int8 = s * 448 / 127``.
        A buffer still at its 1.0 default means the loader left it alone: the
        layer's scale is then read from ``model-kvscales.safetensors``, and only if
        that fails does the fixed fallback (|x| <= 32 assumed) apply, logged.
        Called once after weight loading; the values are host constants passed to
        the Triton kernels (``_run_qsa`` runs outside CUDA-graph capture).
        """
        if self._int8_scales is not None:
            return self._int8_scales
        scales: list[float] = []
        for side in ("k", "v"):
            fp8_scale = float(getattr(self, f"_{side}_scale").item())
            if fp8_scale == 1.0 or not fp8_scale > 0.0:
                file_scale = _kv_scale_from_checkpoint_file(self.layer_name, side)
                if file_scale is not None and file_scale > 0.0:
                    fp8_scale = file_scale
            if fp8_scale == 1.0 or not fp8_scale > 0.0:
                logger.warning(
                    "%s: no checkpoint %s_scale; int8 KV cache uses the fixed "
                    "fallback scale %.6f",
                    self.layer_name,
                    side,
                    INT8_KV_FALLBACK_SCALE,
                )
                scales.append(INT8_KV_FALLBACK_SCALE)
            else:
                scales.append(int8_kv_scale_from_fp8_scale(fp8_scale))
        self._int8_scales = (scales[0], scales[1])
        logger.info(
            "%s: int8 KV cache scales k=%.6f v=%.6f",
            self.layer_name,
            scales[0],
            scales[1],
        )
        return self._int8_scales

    def qsa_int8_scales(self) -> tuple[float, float]:
        """(k_scale, v_scale) of the int8 main cache; finalizes lazily."""
        if self._int8_scales is None:
            return self.finalize_kv_cache_scales()
        return self._int8_scales

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    @eager_break_during_capture
    def _run_qsa(
        self,
        projected_qk: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            return
        main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        side_metadata = cast(
            QSAForwardMetadata,
            metadata[self.indexer.raw_key_cache.prefix],
        )
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")
        selected = self.indexer(
            projected_qk,
            positions,
            self.topk_indices_buffer[:num_tokens],
        )
        if selected.shape != (num_tokens, self.indexer.packed_output_width):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        if self.kv_int8:
            # The cached rows are the same post-rotary keys the fp16 path stores.
            from .ops.qsa import qsa_quantize_store_kv

            k_scale, v_scale = self.qsa_int8_scales()
            key_cache, value_cache = self.kv_cache.transpose(1, 2).split(
                self.head_dim, dim=-1
            )
            qsa_quantize_store_kv(
                key,
                value,
                key_cache,
                value_cache,
                main_metadata.slot_mapping,
                k_scale,
                v_scale,
            )
        else:
            impl.do_kv_cache_update(
                self,
                key,
                value,
                self.kv_cache,
                main_metadata.slot_mapping,
            )
            if _KV_AMAX_DUMP and num_tokens:
                cur = torch.stack(
                    (
                        key[:num_tokens].abs().amax().float(),
                        value[:num_tokens].abs().amax().float(),
                    )
                )
                self._kv_amax = (
                    cur if self._kv_amax is None else torch.maximum(self._kv_amax, cur)
                )
        impl.forward_qsa(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=side_metadata.token_to_req,
            use_prefill_config=main_metadata.max_query_len > self._max_decode_query_len,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        num_tokens = hidden_states.shape[0]
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        attn_output = torch.empty_like(query)
        # Keep the index projection outside the eager break.
        projected_qk, _ = self.indexer.index_qk_proj(hidden_states)
        self._run_qsa(
            projected_qk,
            positions,
            query,
            key,
            value,
            attn_output,
        )
        flat_output = attn_output.view(num_tokens, -1)
        if gate is not None:
            flat_output = flat_output * torch.sigmoid(gate)
        output, _ = self.o_proj(flat_output)
        return output


def finalize_qsa_kv_cache_scales(module: nn.Module) -> None:
    """Derive int8 cache scales for every QSA owner under ``module`` (no-op for 16-bit)."""
    for submodule in module.modules():
        if isinstance(submodule, Qwen4ExpQSAAttention) and submodule.kv_int8:
            submodule.finalize_kv_cache_scales()


_KV_SCALE_FILE_CACHE: dict = {}


def _kv_scale_from_checkpoint_file(layer_name: str, side: str):
    """Per-layer fp8 K/V scale from <model>/model-kvscales.safetensors, or None."""
    import os
    import re

    m = re.search(r"layers\.(\d+)\.", layer_name)
    if m is None:
        return None
    try:
        from vllm.config import get_current_vllm_config

        model_dir = get_current_vllm_config().model_config.model
    except Exception:  # noqa: BLE001
        return None
    path = os.path.join(model_dir, "model-kvscales.safetensors")
    if not os.path.isfile(path):
        return None
    table = _KV_SCALE_FILE_CACHE.get(path)
    if table is None:
        table = {}
        try:
            from safetensors import safe_open

            with safe_open(path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    table[k] = float(f.get_tensor(k).float().reshape(-1)[0].item())
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read %s: %s", path, exc)
        _KV_SCALE_FILE_CACHE[path] = table
    i = m.group(1)
    for key in (
        f"model.language_model.layers.{i}.self_attn.{side}_scale",
        f"language_model.model.layers.{i}.self_attn.{side}_scale",
        f"model.layers.{i}.self_attn.{side}_scale",
    ):
        if key in table:
            return table[key]
    return None


__all__ = [
    "QSAIndexer",
    "finalize_qsa_kv_cache_scales",
    "Qwen4ExpQSAAttention",
    "Qwen4ExpQSAFlashAttentionBackend",
    "Qwen4ExpQSAFlashAttentionImpl",
]
