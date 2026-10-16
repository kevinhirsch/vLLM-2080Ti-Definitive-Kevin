#!/usr/bin/env python3
"""Write a tiny, structurally complete Qwen4Exp (Qwen3.8-Flash-Next) text config for smoke tests.

Every real code path is exercised (GDN with a 3:1 V/K head ratio, QSA attention with head_dim 256 and the
fused 128-dim indexer, hyper-connections, one PLE layer, routed + gated shared experts), at sizes that fit
one GPU next to nothing else. Pair it with ``--load-format dummy``.

Usage: make_tiny_config.py OUT_DIR [--tokenizer-dir DIR]
The tokenizer must have the Qwen3.5/3.8 248320-token vocabulary (e.g. any Qwen3.8 checkpoint).
"""
import argparse
import json
import os
import shutil

TINY = {
    "architectures": ["Qwen4ExpForCausalLM"],
    "model_type": "qwen4_exp_text",
    "vocab_size": 248320,
    "hidden_size": 256,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "max_position_embeddings": 8192,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_key_heads": 4,
    "linear_num_value_heads": 12,
    "num_experts": 16,
    "num_experts_per_tok": 4,
    "moe_intermediate_size": 128,
    "shared_expert_intermediate_size": 128,
    "norm_topk_prob": True,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25},
    "hc_count": 4,
    "hc_lowrank": 32,
    "indexer_n_heads": 4,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 128,
    "indexer_budget": 2048,
    "indexer_compress_ratio": 4,
    "ple_layer_ids": [2],
    "ple_conv_kernel_size": 4,
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "ngram_vocab_size_base": 4096,
    "make_ngram_vocab_size_divisible_by": 128,
    "eos_token_id": 248044,
    "tie_word_embeddings": False,
    "mtp_num_hidden_layers": 0,
    "torch_dtype": "float16",
}

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir")
    ap.add_argument("--tokenizer-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "config.json"), "w") as f:
        json.dump(TINY, f, indent=2)
    for name in TOKENIZER_FILES:
        src = os.path.join(a.tokenizer_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(a.out_dir, name))
    print(a.out_dir)


if __name__ == "__main__":
    main()
