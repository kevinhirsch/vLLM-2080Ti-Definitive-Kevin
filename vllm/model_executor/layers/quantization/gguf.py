# SPDX-License-Identifier: Apache-2.0
"""GGUF (llama.cpp) weights for the SM75 fork: linear, embedding / LM head and routed-expert methods.

Upstream vLLM moved GGUF out of tree, and its old methods relied on uninitialised parameters filled through
the generic weight loaders. Here the per-module ggml types are known before the model is built (the GGUF
header is read by :class:`GGUFConfig`), so every quantized parameter is allocated with its exact byte shape
and is written directly by the GGUF model loader (``model_loader/gguf_loader.py``), including TP / EP
slicing. Float tensors in the same file go through the model's normal ``load_weights``.

A linear module may mix ggml types across its logical output shards (e.g. ``qkv_proj`` = IQ4_XS q + Q6_K k +
Q4_K v): consecutive shards of one type form one *segment* with its own ``qweight_<i>`` parameter, and the
forward concatenates the segment outputs.

Kernels: ``gguf_ops`` (JIT-built ggml CUDA kernels incl. the Q2_0 type). Decode uses MMVQ / ``moe_vec`` (int8
dp4a); K-quants use MMQ for larger batches; I-quants and Q2_0 fall back to dequantize + GEMM for large batches.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization import gguf_ops as ops
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


# --------------------------------------------------------------------------------------------- matmul
def _mmvq_max_tokens(qtype: int, rows: int) -> int:
    """Batch size up to which the per-row MMVQ kernel beats the alternatives (old vLLM heuristic)."""
    if qtype in ops.MMQ_TYPES:
        return 2 if rows > 5120 else 6
    return 8 if rows > 5120 else 16


def _gguf_mul_mat(x: torch.Tensor, qweight: torch.Tensor, qweight_type: int, rows: int) -> torch.Tensor:
    if x.shape[0] == 0:
        return torch.empty(0, rows, dtype=x.dtype, device=x.device)
    if x.shape[0] <= _mmvq_max_tokens(qweight_type, rows):
        return ops.ggml_mul_mat_vec_a8(qweight, x, qweight_type, rows)
    if qweight_type in ops.MMQ_TYPES:
        return ops.ggml_mul_mat_a8(qweight, x, qweight_type, rows)
    be, bb = ops.BLOCK[qweight_type]
    weight = ops.ggml_dequantize(qweight, qweight_type, rows, qweight.shape[1] // bb * be, x.dtype)
    return x @ weight.T


def _gguf_mul_mat_fake(x, qweight, qweight_type, rows):
    return torch.empty(x.shape[0], rows, dtype=x.dtype, device=x.device)


direct_register_custom_op(
    op_name="gguf_mul_mat",
    op_func=_gguf_mul_mat,
    fake_impl=_gguf_mul_mat_fake,
)


def gguf_mul_mat(x: torch.Tensor, qweight: torch.Tensor, qweight_type: int, rows: int) -> torch.Tensor:
    return torch.ops.vllm.gguf_mul_mat(x, qweight, qweight_type, rows)


def _gguf_moe(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_type: int,
    w2_type: int,
) -> torch.Tensor:
    """Routed experts with ggml ``moe_vec`` kernels. ``topk_ids`` are LOCAL expert ids, -1 = not on this rank
    (contributes zero). Decode-oriented: each (token, expert) pair re-reads its expert rows."""
    num_tokens = x.shape[0]
    if num_tokens == 0:
        return torch.zeros_like(x)
    top_k = topk_ids.shape[1]
    n13 = w13.shape[1]
    hidden = w2.shape[1]
    ids = topk_ids.to(torch.int32).contiguous()
    gate_up = ops.ggml_moe_a8_vec(x.contiguous(), w13, ids, top_k, w13_type, n13, num_tokens)
    d = n13 // 2
    act = torch.nn.functional.silu(gate_up[:, :d]) * gate_up[:, d:]
    out = ops.ggml_moe_a8_vec(act.contiguous(), w2, ids, 1, w2_type, hidden, num_tokens * top_k)
    out = out.view(num_tokens, top_k, hidden) * topk_weights.view(num_tokens, top_k, 1).to(out.dtype)
    return out.sum(dim=1)


def _gguf_moe_fake(x, w13, w2, topk_weights, topk_ids, w13_type, w2_type):
    return torch.empty_like(x)


direct_register_custom_op(op_name="gguf_moe", op_func=_gguf_moe, fake_impl=_gguf_moe_fake)


# --------------------------------------------------------------------------------------------- config
class GGUFConfig(QuantizationConfig):
    """``quantization_config: {"quant_method": "gguf", "gguf_file": "<first shard>.gguf"}`` in config.json.

    ``specs`` maps a vLLM module prefix to its GGUF layout (``LinearSpec`` / ``EmbedSpec`` / ``MoESpec`` from
    ``model_loader/gguf_qwen4exp.py``); modules without a spec are float in the GGUF and run unquantized."""

    def __init__(self, gguf_file: str | None = None, specs: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.gguf_file = gguf_file
        self._specs = specs

    def __repr__(self) -> str:
        return f"GGUFConfig(gguf_file={self.gguf_file!r})"

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "gguf"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 61  # dp4a

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> GGUFConfig:
        return cls(gguf_file=config.get("gguf_file"))

    @property
    def specs(self) -> dict[str, Any]:
        if self._specs is None:
            if not self.gguf_file:
                raise ValueError("GGUFConfig needs quantization_config.gguf_file (or explicit specs)")
            from vllm.model_executor.model_loader.gguf_qwen4exp import quant_specs

            self._specs = quant_specs(self.gguf_model())
        return self._specs

    def gguf_model(self):
        """The parsed GGUF header (cached; the tokenizer arrays make parsing take a moment)."""
        if getattr(self, "_gguf_model", None) is None:
            from vllm.model_executor.model_loader.gguf_reader import GGUFModel

            self._gguf_model = GGUFModel(self.gguf_file)
        return self._gguf_model

    def _spec(self, prefix: str):
        specs = self.specs
        if prefix in specs:
            return specs[prefix]
        if prefix.endswith(".routed_experts"):
            return specs.get(prefix[: -len(".routed_experts")])
        return None

    def get_qwen4exp_ple_method(self, prefix: str) -> QuantizeMethodBase | None:
        """PLE n-gram table hook (``ngram_embedding.py``): the GGUF's IQ4_NL ``per_layer_token_embd`` stays in
        the file and is gathered on the host (``ple_gguf.py``). ``None`` (no table in the GGUF, or no GGUF
        file, e.g. ``--load-format dummy`` smoke tests) = the regular unquantized table."""
        if not self.gguf_file:
            return None
        from vllm.models.qwen4_exp.nvidia.ple_gguf import Qwen4ExpPLEGGUFTableMethod

        info = self.gguf_model().tensors.get("per_layer_token_embd.weight")
        if info is None:
            return None
        return Qwen4ExpPLEGGUFTableMethod(info.path, info.offset, info.dims[1], info.dims[0], info.ggml_type)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        from vllm.model_executor.model_loader.gguf_qwen4exp import EmbedSpec, LinearSpec, MoESpec

        spec = self._spec(prefix)
        if isinstance(layer, LinearBase):
            if isinstance(spec, LinearSpec):
                return GGUFLinearMethod(spec)
            return UnquantizedLinearMethod()
        if isinstance(layer, VocabParallelEmbedding):
            if isinstance(spec, EmbedSpec):
                return GGUFEmbeddingMethod(spec)
            return UnquantizedEmbeddingMethod()
        if isinstance(layer, RoutedExperts):
            if isinstance(spec, MoESpec):
                return GGUFMoEMethod(spec, layer.moe_config)
            return UnquantizedFusedMoEMethod(layer.moe_config)
        return None


def _uint8_param(*shape: int) -> Parameter:
    return Parameter(torch.zeros(*shape, dtype=torch.uint8), requires_grad=False)


def _row_bytes(cols: int, qtype: int, what: str) -> int:
    be, bb = ops.BLOCK[qtype]
    if cols % be:
        raise ValueError(f"{what}: {cols} input columns are not a multiple of the ggml type-{qtype} block {be}")
    return cols // be * bb


# --------------------------------------------------------------------------------------------- linear
class GGUFLinearMethod(LinearMethodBase):
    def __init__(self, spec) -> None:
        self.spec = spec

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        n = len(output_partition_sizes)
        if set(self.spec.shard_types) != set(range(n)):
            raise ValueError(
                f"GGUF linear: shards {sorted(self.spec.shard_types)} do not cover the {n} output shards"
            )
        segments = []  # (param name, qtype, row_start, rows, first_shard, last_shard)
        row = 0
        for s, size in enumerate(output_partition_sizes):
            t = self.spec.shard_types[s]
            if segments and segments[-1][1] == t:
                name, _, start, rows, first, _ = segments[-1]
                segments[-1] = (name, t, start, rows + size, first, s)
            else:
                segments.append((f"qweight_{len(segments)}", t, row, size, s, s))
            row += size
        for name, t, _, rows, _, _ in segments:
            p = _uint8_param(rows, _row_bytes(input_size_per_partition, t, name))
            set_weight_attrs(p, {"ggml_type": t, "gguf": True})
            layer.register_parameter(name, p)
        layer.gguf_segments = segments
        layer.gguf_output_partition_sizes = list(output_partition_sizes)
        layer.gguf_input_size_per_partition = input_size_per_partition

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        outs = [
            gguf_mul_mat(x2, getattr(layer, name), t, rows) for name, t, _, rows, _, _ in layer.gguf_segments
        ]
        y = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)
        if bias is not None:
            y = y + bias
        return y.reshape(*shape[:-1], y.shape[-1])


def load_gguf_linear(layer: torch.nn.Module, shards: tuple[int, ...], rows: torch.Tensor) -> None:
    """Write the FULL (unsharded) GGUF rows of logical ``shards`` (concatenated in order) into ``layer``.

    Column-parallel layers take this rank's rows of every shard (QKV k/v shards honour KV-head replication);
    row-parallel layers take this rank's input columns (quant-block aligned)."""
    tp_rank = getattr(layer, "tp_rank", 0)
    tp_size = getattr(layer, "tp_size", 1)
    parts = layer.gguf_output_partition_sizes
    in_pp = layer.gguf_input_size_per_partition
    if isinstance(layer, RowParallelLinear) and tp_size > 1:
        t = layer.gguf_segments[0][1]
        nb = _row_bytes(in_pp, t, "row-parallel slice")
        rows = rows[:, tp_rank * nb : (tp_rank + 1) * nb]
    row0 = 0
    for s in shards:
        if isinstance(layer, (ReplicatedLinear, RowParallelLinear)) or tp_size == 1:
            ranks, rank = 1, 0
        elif isinstance(layer, QKVParallelLinear) and s in (1, 2):
            ranks = tp_size // layer.num_kv_head_replicas
            rank = tp_rank // layer.num_kv_head_replicas
        else:
            ranks, rank = tp_size, tp_rank
        full = parts[s] * ranks
        mine = rows[row0 + rank * parts[s] : row0 + (rank + 1) * parts[s]]
        row0 += full
        seg = next(g for g in layer.gguf_segments if g[4] <= s <= g[5])
        name, _, seg_start, _, _, _ = seg
        dst = getattr(layer, name)
        off = sum(parts[:s]) - seg_start
        if mine.shape[1] != dst.shape[1]:
            raise ValueError(f"{name}: row bytes {mine.shape[1]} != {dst.shape[1]}")
        dst.data[off : off + parts[s]].copy_(mine)
    if row0 != rows.shape[0]:
        raise ValueError(f"GGUF linear: consumed {row0} of {rows.shape[0]} rows for shards {shards}")


# --------------------------------------------------------------------------------------------- embedding
class GGUFEmbeddingMethod(QuantizeMethodBase):
    """Quantized token embedding / LM head (vocab-parallel rows)."""

    def __init__(self, spec) -> None:
        self.spec = spec

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        t = self.spec.ggml_type
        rows = sum(output_partition_sizes)
        p = _uint8_param(rows, _row_bytes(input_size_per_partition, t, "embedding"))
        set_weight_attrs(p, {"ggml_type": t, "gguf": True})
        layer.register_parameter("qweight", p)
        layer.gguf_type = t
        layer.gguf_hidden = input_size_per_partition
        layer.params_dtype = params_dtype

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        y = gguf_mul_mat(x.reshape(-1, x.shape[-1]), layer.qweight, layer.gguf_type, layer.qweight.shape[0])
        if bias is not None:
            y = y + bias
        return y.reshape(*x.shape[:-1], y.shape[-1])

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        flat = input_.reshape(-1)
        q = torch.index_select(layer.qweight, 0, flat)
        out = ops.ggml_dequantize(q, layer.gguf_type, flat.shape[0], layer.gguf_hidden, layer.params_dtype)
        return out.view(*input_.shape, layer.gguf_hidden)


def load_gguf_embedding(layer: torch.nn.Module, rows: torch.Tensor) -> None:
    idx = layer.shard_indices
    start, end = idx.org_vocab_start_index, idx.org_vocab_end_index
    layer.qweight.data[: end - start].copy_(rows[start:end])


# --------------------------------------------------------------------------------------------- MoE
class GGUFMoEMethod(FusedMoEMethodBase):
    """Routed experts stored as 3-D ggml tensors [E, rows, row_bytes]: one ggml type for w13, one for w2."""

    def __init__(self, spec, moe) -> None:
        super().__init__(moe)
        self.spec = spec

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        t13, t2 = self.spec.w13_type, self.spec.w2_type
        w13 = _uint8_param(num_experts, 2 * intermediate_size_per_partition, _row_bytes(hidden_size, t13, "w13"))
        w2 = _uint8_param(num_experts, hidden_size, _row_bytes(intermediate_size_per_partition, t2, "w2"))
        for p, t in ((w13, t13), (w2, t2)):
            set_weight_attrs(p, {"ggml_type": t, "gguf": True})
        layer.register_parameter("w13_qweight", w13)
        layer.register_parameter("w2_qweight", w2)
        layer.gguf_w13_type, layer.gguf_w2_type = t13, t2
        layer.gguf_intermediate = intermediate_size_per_partition

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig | None:
        return FUSED_MOE_UNQUANTIZED_CONFIG

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        expert_map = getattr(layer, "expert_map", None)
        if expert_map is not None:
            topk_ids = expert_map[topk_ids.long()]
        return torch.ops.vllm.gguf_moe(
            x,
            layer.w13_qweight,
            layer.w2_qweight,
            topk_weights,
            topk_ids,
            layer.gguf_w13_type,
            layer.gguf_w2_type,
        )


def gguf_local_experts(layer: RoutedExperts) -> list[tuple[int, int]]:
    """(global expert id, local slot) pairs owned by this rank."""
    expert_map = getattr(layer, "expert_map", None)
    if expert_map is None:
        return [(e, e) for e in range(layer.w13_qweight.shape[0])]
    em = expert_map.cpu().tolist()
    return [(g, local) for g, local in enumerate(em) if local >= 0]


def load_gguf_experts(layer: RoutedExperts, part: str, data: torch.Tensor) -> None:
    """``data``: full GGUF expert tensor as uint8 [E_global, rows, row_bytes] (gate/up: rows = I, down: rows = H)."""
    tp_rank = layer.moe_config.tp_rank
    tp_size = layer.moe_config.tp_size
    inter = layer.gguf_intermediate
    for g, local in gguf_local_experts(layer):
        src = data[g]
        if part in ("w1", "w3"):
            mine = src[tp_rank * inter : (tp_rank + 1) * inter] if tp_size > 1 else src
            off = 0 if part == "w1" else inter
            layer.w13_qweight.data[local, off : off + inter].copy_(mine)
        elif part == "w2":
            if tp_size > 1:
                nb = _row_bytes(inter, layer.gguf_w2_type, "w2 slice")
                src = src[:, tp_rank * nb : (tp_rank + 1) * nb]
            layer.w2_qweight.data[local].copy_(src)
        else:
            raise ValueError(part)
