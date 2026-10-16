# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config shim that maps a Qwen4Exp (``qwen4exp``) GGUF model to vLLM's
``Qwen4ExpTextConfig``.

The GGUF key/value pairs use a flat ``qwen4exp.*`` naming scheme; this
module translates them into the (nested, HuggingFace-style) keyword
arguments expected by :class:`vllm.models.qwen4_exp.config.Qwen4ExpTextConfig`
(a subclass of transformers' ``Qwen3NextConfig``).
"""

from __future__ import annotations

from typing import Any

from vllm.model_executor.model_loader.gguf_reader import GGUFModel

# gguf key prefix for all Qwen4Exp-specific values
_PREFIX = "qwen4exp."

# the base (unigram) size used to seed the per-ngram-head vocabulary
# tables; each head's actual size is the next prime after this base.
_NGRAM_VOCAB_SIZE_BASE = 20_000_000


def _is_prime(n: int) -> bool:
    """Return whether ``n`` is a prime number (trial division)."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    if n % 3 == 0:
        return n == 3
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _next_prime_after(n: int) -> int:
    """Return the smallest prime strictly greater than ``n``."""
    candidate = n + 1
    if candidate <= 2:
        return 2
    if candidate % 2 == 0:
        candidate += 1
    while not _is_prime(candidate):
        candidate += 2
    return candidate


def qwen4exp_text_config(model: GGUFModel) -> dict[str, Any]:
    """Build the keyword arguments for vLLM's ``Qwen4ExpTextConfig``.

    Args:
        model: A parsed GGUF model whose ``general.architecture``
            must be ``"qwen4exp"``.

    Returns:
        A dict of keyword arguments suitable for
        ``Qwen4ExpTextConfig(**result)``.

    Raises:
        ValueError: If the architecture is not ``qwen4exp``, if any
            nonzero attention compress ratio differs from the maximum
            (used as ``indexer_compress_ratio``), if nonzero compress
            ratios are not found exactly on the full-attention layers,
            or if the verified ``ngram_vocab_size_base`` does not match the
            first PLE head vocabulary size.
    """
    architecture = model.kv.get("general.architecture")
    if architecture != "qwen4exp":
        raise ValueError(
            f"expected general.architecture == 'qwen4exp', got {architecture!r}"
        )

    kv = model.kv
    A = _PREFIX

    head_dim = kv[A + "attention.key_length"]
    num_hidden_layers = kv[A + "block_count"]
    full_attention_interval = kv[A + "full_attention_interval"]

    # per-layer attention type: full attention every
    # ``full_attention_interval`` layers, linear attention otherwise
    layer_types = [
        "full_attention" if (i + 1) % full_attention_interval == 0
        else "linear_attention"
        for i in range(num_hidden_layers)
    ]

    # indexer compress ratio: the maximum per-layer compress ratio
    compress_ratios = kv[A + "attention.compress_ratios"]
    indexer_compress_ratio = max(compress_ratios)
    for i, ratio in enumerate(compress_ratios):
        if ratio != 0 and ratio != indexer_compress_ratio:
            raise ValueError(
                f"attention.compress_ratios[{i}] = {ratio} differs from "
                f"the maximum {indexer_compress_ratio}"
            )
    # nonzero compress ratios must line up exactly with the
    # full-attention layers
    full_attention_positions = {
        i for i, layer_type in enumerate(layer_types) if layer_type == "full_attention"
    }
    nonzero_positions = {i for i, ratio in enumerate(compress_ratios) if ratio != 0}
    if nonzero_positions != full_attention_positions:
        raise ValueError(
            "nonzero attention.compress_ratios are not found exactly on the "
            f"full-attention layers: nonzero at {sorted(nonzero_positions)}, "
            f"full attention at {sorted(full_attention_positions)}"
        )

    # verify the ngram vocabulary base against the first PLE head
    head_vocab_sizes = kv[A + "ple.head_vocab_sizes"]
    verified_first = _next_prime_after(_NGRAM_VOCAB_SIZE_BASE - 1)
    if verified_first != head_vocab_sizes[0]:
        raise ValueError(
            f"ngram_vocab_size_base verification failed: "
            f"_next_prime_after({_NGRAM_VOCAB_SIZE_BASE - 1}) = {verified_first} "
            f"!= ple.head_vocab_sizes[0] = {head_vocab_sizes[0]}"
        )

    return {
        "vocab_size": model.tensors["token_embd.weight"].dims[1],
        "hidden_size": kv[A + "embedding_length"],
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": kv[A + "attention.head_count"],
        "num_key_value_heads": kv[A + "attention.head_count_kv"],
        "head_dim": head_dim,
        "max_position_embeddings": kv[A + "context_length"],
        "rms_norm_eps": kv[A + "attention.layer_norm_rms_epsilon"],
        "num_experts": kv[A + "expert_count"],
        "num_experts_per_tok": kv[A + "expert_used_count"],
        "moe_intermediate_size": kv[A + "expert_feed_forward_length"],
        "shared_expert_intermediate_size": kv[A + "expert_shared_feed_forward_length"],
        "intermediate_size": kv[A + "expert_feed_forward_length"],
        "norm_topk_prob": True,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "hidden_act": "silu",
        "linear_conv_kernel_dim": kv[A + "ssm.conv_kernel"],
        "linear_key_head_dim": kv[A + "ssm.state_size"],
        "linear_value_head_dim": kv[A + "ssm.inner_size"] // kv[A + "ssm.time_step_rank"],
        "linear_num_key_heads": kv[A + "ssm.group_count"],
        "linear_num_value_heads": kv[A + "ssm.time_step_rank"],
        "layer_types": layer_types,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": kv[A + "rope.freq_base"],
            "partial_rotary_factor": kv[A + "rope.dimension_count"] / head_dim,
            "mrope_section": kv[A + "rope.dimension_sections"][:3],
            "mrope_interleaved": True,
        },
        "hc_count": kv[A + "hyper_connection.count"],
        "hc_lowrank": kv[A + "hyper_connection.low_rank"],
        "indexer_n_heads": kv[A + "attention.indexer.head_count"],
        "indexer_kv_heads": 1,
        "indexer_head_dim": kv[A + "attention.indexer.key_length"],
        "indexer_budget": kv[A + "attention.indexer.top_k"],
        "indexer_compress_ratio": indexer_compress_ratio,
        "ple_layer_ids": [i + 1 for i in kv[A + "ple.layers"]],
        "ple_embed_dim": (
            kv[A + "embedding_length_per_layer_input"]
            * (kv[A + "ple.ngram_size"] - 1)
            * kv[A + "ple.heads_per_ngram"]
        ),
        "ple_conv_kernel_size": kv[A + "ple.conv_kernel"],
        "ngram_size": kv[A + "ple.ngram_size"],
        "heads_per_ngram": kv[A + "ple.heads_per_ngram"],
        "ngram_vocab_size_base": _NGRAM_VOCAB_SIZE_BASE,
        "make_ngram_vocab_size_divisible_by": 128,
        "eos_token_id": kv[A + "ple.eos_token_id"],
        "tie_word_embeddings": False,
        "mtp_num_hidden_layers": 0,
    }


def qwen4exp_ple_constants(model: GGUFModel) -> dict[str, Any]:
    """Extract the PLE (per-layer embedding) constants from a model.

    Args:
        model: A parsed GGUF model.

    Returns:
        A dict with the per-layer embedding hash multipliers, the
        per-ngram-head vocabulary offsets and sizes, the total
        number of table rows, and the embedding (row) dimension.
    """
    kv = model.kv
    A = _PREFIX
    ple_tensor = model.tensors["per_layer_token_embd.weight"]
    return {
        "layer_multipliers": kv[A + "ple.layer_multipliers"],
        "head_offsets": kv[A + "ple.head_offsets"],
        "head_vocab_sizes": kv[A + "ple.head_vocab_sizes"],
        "table_rows": ple_tensor.dims[1],
        "row_dim": ple_tensor.dims[0],
    }


__all__ = [
    "qwen4exp_text_config",
    "qwen4exp_ple_constants",
]
