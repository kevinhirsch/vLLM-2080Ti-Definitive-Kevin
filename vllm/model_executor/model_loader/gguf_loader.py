# SPDX-License-Identifier: Apache-2.0
"""``--load-format gguf``: llama.cpp GGUF checkpoints (currently the ``qwen4exp`` architecture).

The model directory holds ``config.json`` (+ tokenizer) generated from the GGUF header by
``tools/qwen4exp/gguf_to_config.py``; its ``quantization_config`` is ``{"quant_method": "gguf", "gguf_file":
<first shard>}``. ``--model-loader-extra-config '{"gguf_file": ...}'`` overrides the file.

Float tensors are transformed back to HF semantics (``gguf_qwen4exp.py``) and fed through the model's own
``load_weights``; quantized tensors are copied byte-exact (row-permuted where llama.cpp reordered V heads) into
the GGUF quant-method parameters with this rank's TP / EP slice. The GGUF is memory-mapped: no host copy of the
70 GiB file is made, pages stream from the page cache straight into the device copies.
"""

from __future__ import annotations

import time

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

logger = init_logger(__name__)


def gguf_file_for(model_config: ModelConfig, load_config: LoadConfig) -> str:
    extra = load_config.model_loader_extra_config or {}
    path = extra.get("gguf_file")
    if path is None:
        qc = getattr(model_config.hf_config, "quantization_config", None) or {}
        path = qc.get("gguf_file") if isinstance(qc, dict) else getattr(qc, "gguf_file", None)
    if not path:
        raise ValueError(
            "GGUF load format: set quantization_config.gguf_file in config.json or "
            "--model-loader-extra-config '{\"gguf_file\": \"<first shard>.gguf\"}'"
        )
    return path


class GGUFModelLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra = load_config.model_loader_extra_config or {}
        unknown = set(extra) - {"gguf_file"}
        if unknown:
            raise ValueError(f"unknown GGUF model_loader_extra_config keys: {sorted(unknown)}")

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # local files only

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        path = gguf_file_for(model_config, self.load_config)
        stats = load_qwen4exp_gguf(model, model_config, path)
        logger.info(
            "Loaded GGUF %s: %d float tensors, %d quantized tensors (%.2f GiB) in %.1f s",
            path, stats["float"], stats["quant"], stats["quant_bytes"] / 2**30, stats["seconds"],
        )


def _hf_weights(gm, plans, nk, nv, device):
    """HF-named float tensors (and host-dequantized ones) for ``model.load_weights``."""
    from vllm.model_executor.model_loader import gguf_qwen4exp as Q

    pending: dict[str, dict[int, torch.Tensor]] = {}
    for p in plans:
        if p.kind == "hf":
            yield p.target, Q.float_tensor(gm, p, nk, nv)
        elif p.kind == "concat_hf":
            parts = pending.setdefault(p.target, {})
            parts[p.shards[0]] = Q.float_tensor(gm, p, nk, nv)
            if len(parts) == 2:
                yield p.target, torch.cat([parts[0], parts[1]], 0)
                del pending[p.target]
        elif p.kind == "dequant_hf":
            yield p.target, _dequant(gm, p, nk, nv, device)
    if pending:
        raise ValueError(f"incomplete concatenated tensors: {sorted(pending)}")
    kv = gm.kv
    for layer in kv.get("qwen4exp.ple.layers", []):
        base = f"model.layers.{layer}.ple.ple_embedding."
        for name, values in Q.ple_constants(gm).items():
            yield base + name, torch.tensor(values, dtype=torch.long)


def _dequant(gm, p, nk, nv, device):
    """Quantized -> fp16 via the ggml dequant kernel on the device (CPU: numpy gguf reference)."""
    from vllm.model_executor.model_loader import gguf_qwen4exp as Q

    rows = Q.quant_rows(gm, p, nk, nv)
    out_rows, cols = p.shape[0], p.shape[-1]
    if device.type == "cuda":
        from vllm.model_executor.layers.quantization import gguf_ops

        q = rows.reshape(out_rows, -1).to(device)
        w = gguf_ops.ggml_dequantize(q, p.ggml_type, out_rows, cols, torch.float16)
    else:  # CPU tests
        import gguf as gguf_py  # type: ignore[import-not-found]

        w = torch.from_numpy(
            gguf_py.quants.dequantize(rows.numpy(), gguf_py.GGMLQuantizationType(p.ggml_type))
        ).reshape(out_rows, cols).to(torch.float16)
    return Q.apply_transforms(w, p, nk, nv)


def load_qwen4exp_gguf(model: nn.Module, model_config: ModelConfig, path: str) -> dict:
    from vllm.model_executor.layers.quantization.gguf import (
        load_gguf_embedding,
        load_gguf_experts,
        load_gguf_linear,
    )
    from vllm.model_executor.model_loader import gguf_qwen4exp as Q
    from vllm.model_executor.model_loader.gguf_reader import GGML_TYPES, GGUFModel

    t0 = time.perf_counter()
    gm = GGUFModel(path)
    plans = Q.plan(gm)
    cfg = model_config.hf_text_config
    nk, nv = int(cfg.linear_num_key_heads), int(cfg.linear_num_value_heads)
    device = next(model.parameters()).device

    loaded = set(model.load_weights(_hf_weights(gm, plans, nk, nv, device)))

    modules = dict(model.named_modules())
    written: set[str] = set()
    quant_bytes = 0
    n_quant = 0
    for p in plans:
        if p.kind not in ("qlinear", "qembed", "qmoe"):
            continue
        n_quant += 1
        quant_bytes += gm.tensors[p.gguf_name].nbytes
        if p.kind == "qlinear":
            layer = modules[p.target]
            be, bb = GGML_TYPES[p.ggml_type][1:]
            rows = Q.quant_rows(gm, p, nk, nv).reshape(-1, p.shape[-1] // be * bb)
            load_gguf_linear(layer, p.shards, rows)
            written.update(f"{p.target}.{seg[0]}" for seg in layer.gguf_segments)
        elif p.kind == "qembed":
            layer = modules[p.target]
            load_gguf_embedding(layer, Q.quant_rows(gm, p, nk, nv))
            written.add(f"{p.target}.qweight")
        else:
            name = p.target + ".routed_experts" if p.target + ".routed_experts" in modules else p.target
            layer = modules[name]
            load_gguf_experts(layer, p.moe_part, Q.quant_rows(gm, p, nk, nv))
            written.add(f"{name}.{'w2' if p.moe_part == 'w2' else 'w13'}_qweight")

    missing = [
        n for n, _ in model.named_parameters()
        if n not in loaded and n not in written and not _is_ple_table(n)
    ]
    if missing:
        raise ValueError(f"GGUF load left {len(missing)} parameters unset, e.g. {missing[:8]}")
    return {
        "float": sum(p.kind in ("hf", "concat_hf", "dequant_hf") for p in plans),
        "quant": n_quant,
        "quant_bytes": quant_bytes,
        "seconds": time.perf_counter() - t0,
    }


def _is_ple_table(name: str) -> bool:
    # The n-gram table is owned by its PLE method (host gather from the GGUF), never loaded as a parameter.
    return ".ple.ple_embedding.ngram_embedding." in name
