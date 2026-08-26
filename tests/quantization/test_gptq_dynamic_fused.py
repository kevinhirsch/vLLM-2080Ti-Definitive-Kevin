# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused-layer dynamic-rule matching for GPTQ (`dynamic` from GPTQModel).

GPTQModel `dynamic` rules are written against checkpoint module names
(q_proj/k_proj/v_proj/...). vLLM fuses those into qkv_proj/gate_up_proj, so
rule matching against the raw vLLM prefix silently drops per-layer overrides:
the layer is then created at the base config's bits and weight loading dies
with a shape-mismatch assert. `dynamic_match_prefix` translates fused names
to their unfused shards for rule matching (mirroring what
`is_layer_gptq_quantized` already does for quantize-membership).

Regression context: a Qwen3.5-family mixed 4/8-bit self-quant with
`dynamic = {"...self_attn.(q_proj|k_proj|v_proj|o_proj)...": {"bits": 8}}`
failed to boot — qkv_proj allocated int4-shaped params against int8-packed
checkpoint tensors.
"""

import pytest

from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    dynamic_match_prefix,
    get_dynamic_override,
    override_config,
)

ATTN_IDS = "(3|7|11)"
UNFUSED_RULE = (
    rf"^language_model\.model\.layers\.{ATTN_IDS}"
    r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj)(\.|$)"
)

PACKED_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def _make_config(dynamic):
    config = GPTQMarlinConfig.from_config(
        {
            "bits": 4,
            "group_size": 128,
            "desc_act": False,
            "sym": True,
            "lm_head": False,
            "dynamic": dynamic,
        }
    )
    config.packed_modules_mapping = dict(PACKED_MAPPING)
    return config


def test_fused_qkv_matches_unfused_bits_rule():
    config = _make_config({UNFUSED_RULE: {"bits": 8}})
    fused = "language_model.model.layers.3.self_attn.qkv_proj"
    match = dynamic_match_prefix(config, fused)
    assert match == "language_model.model.layers.3.self_attn.q_proj"
    assert get_dynamic_override(config, match) == {"bits": 8}
    override_config(config, match)
    assert config.weight_bits == 8
    assert config.pack_factor == 4  # 32 // 8


def test_unfused_o_proj_still_matches_directly():
    config = _make_config({UNFUSED_RULE: {"bits": 8}})
    o_proj = "language_model.model.layers.3.self_attn.o_proj"
    assert dynamic_match_prefix(config, o_proj) == o_proj
    assert get_dynamic_override(config, o_proj) == {"bits": 8}


def test_non_matching_layer_keeps_base_bits():
    config = _make_config({UNFUSED_RULE: {"bits": 8}})
    fused = "language_model.model.layers.4.self_attn.qkv_proj"
    assert dynamic_match_prefix(config, fused) == fused
    assert get_dynamic_override(config, fused) is None
    override_config(config, fused)
    assert config.weight_bits == 4


def test_fused_name_rule_takes_precedence_over_shards():
    fused = "language_model.model.layers.3.self_attn.qkv_proj"
    fused_rule = (
        r"^language_model\.model\.layers\.3\.self_attn\.qkv_proj(\.|$)"
    )
    config = _make_config({fused_rule: {"bits": 8}, UNFUSED_RULE: {"bits": 4}})
    # a rule matching the fused vLLM name directly wins; no shard translation
    assert dynamic_match_prefix(config, fused) == fused
    assert get_dynamic_override(config, fused) == {"bits": 8}


def test_conflicting_shard_rules_raise():
    config = _make_config(
        {
            r"^language_model\.model\.layers\.3\.self_attn\.q_proj(\.|$)": {
                "bits": 8
            },
        }
    )
    fused = "language_model.model.layers.3.self_attn.qkv_proj"
    with pytest.raises(ValueError, match="fused layer"):
        dynamic_match_prefix(config, fused)


def test_negative_shard_rules_skip_fused_module():
    config = _make_config(
        {
            r"-:^language_model\.model\.layers\.3\.self_attn"
            r"\.(q_proj|k_proj|v_proj)(\.|$)": {},
        }
    )
    fused = "language_model.model.layers.3.self_attn.qkv_proj"
    match = dynamic_match_prefix(config, fused)
    assert get_dynamic_override(config, match) is False


def test_no_dynamic_is_identity():
    config = _make_config({})
    fused = "language_model.model.layers.3.self_attn.qkv_proj"
    assert dynamic_match_prefix(config, fused) == fused


def test_gate_up_fusion_translates():
    config = _make_config(
        {
            r"^model\.layers\.0\.mlp\.(gate_proj|up_proj)(\.|$)": {
                "group_size": 64
            },
        }
    )
    fused = "model.layers.0.mlp.gate_up_proj"
    match = dynamic_match_prefix(config, fused)
    assert match == "model.layers.0.mlp.gate_proj"
    assert get_dynamic_override(config, match) == {"group_size": 64}
