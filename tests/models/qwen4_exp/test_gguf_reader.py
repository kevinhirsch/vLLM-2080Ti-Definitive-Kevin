# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the GGUF reader and the Qwen4Exp config shim.

These tests operate on header-only copies of the real two-shard
Qwen3.8-Flash-Next GGUF (no tensor data is present), so they never
read tensor bytes beyond the header.  The module is skipped when
the header files are not available.
"""

from __future__ import annotations

import os
import sys

import pytest

# ---------------------------------------------------------------------------
# locate the header files (skip the whole module if they are absent)
# ---------------------------------------------------------------------------
_DEFAULT_HEADERS_DIR = "/home/kevin/nm9-work/headers"
HEADERS_DIR = os.environ.get("NM9_GGUF_HEADERS", _DEFAULT_HEADERS_DIR)

_SHARD1_NAME = "flashnext-hdr-00001-of-00002.gguf"
_SHARD2_NAME = "flashnext-hdr-00002-of-00002.gguf"


def _headers_available() -> bool:
    for directory in {HEADERS_DIR, _DEFAULT_HEADERS_DIR}:
        if os.path.isfile(os.path.join(directory, _SHARD1_NAME)):
            return True
    return False


if not _headers_available():
    pytest.skip(
        "GGUF header files not found (looked in "
        f"{HEADERS_DIR!r} and {_DEFAULT_HEADERS_DIR!r})",
        allow_module_level=True,
    )

from vllm.model_executor.model_loader.gguf_qwen4exp_config import (
    qwen4exp_ple_constants,
    qwen4exp_text_config,
)
from vllm.model_executor.model_loader.gguf_reader import GGUFModel

# the reference reader is copied into this directory so the tests
# can compare the new reader against a known-correct implementation;
# make sure this directory is importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from gguf_index_reference import read as reference_read  # noqa: E402


def _shard1_path() -> str:
    return os.path.join(HEADERS_DIR, _SHARD1_NAME)


def _shard2_path() -> str:
    return os.path.join(HEADERS_DIR, _SHARD2_NAME)


@pytest.fixture(scope="module")
def model() -> GGUFModel:
    """A GGUFModel opened from the first shard (spans both shards)."""
    return GGUFModel(_shard1_path())


# ---------------------------------------------------------------------------
# 1. shard discovery + tensor table
# ---------------------------------------------------------------------------
def test_finds_two_shards(model: GGUFModel) -> None:
    assert len(model.files) == 2
    assert model.files[0].path == _shard1_path()
    assert model.files[1].path == _shard2_path()


def test_tensor_count_matches_kv(model: GGUFModel) -> None:
    assert len(model.tensors) == model.kv["split.tensors.count"]


def test_per_layer_token_embd_in_shard2(model: GGUFModel) -> None:
    info = model.tensors["per_layer_token_embd.weight"]
    assert info.dims == (160, 320001536)
    assert info.ggml_type == 20
    # it lives in the second shard
    assert "per_layer_token_embd.weight" in model.files[1].tensors
    assert model.files[1].tensors["per_layer_token_embd.weight"] is info


# ---------------------------------------------------------------------------
# 2. compare every tensor against the reference reader
# ---------------------------------------------------------------------------
def test_matches_reference_reader(model: GGUFModel) -> None:
    shard_paths = [_shard1_path(), _shard2_path()]
    for path, file in zip(shard_paths, model.files):
        _kv, reference_tensors, reference_data_start = reference_read(path)
        assert len(reference_tensors) == len(file.tensors)
        for reference_tensor in reference_tensors:
            name = reference_tensor["name"]
            info = file.tensors[name]
            assert info.name == reference_tensor["name"]
            assert info.dims == tuple(reference_tensor["dims"])
            assert info.ggml_type == reference_tensor["type"]
            # the reference offset is relative; the reader stores the
            # absolute offset (data_start + relative offset)
            expected_offset = reference_data_start + reference_tensor["offset"]
            assert info.offset == expected_offset
            assert info.nbytes == reference_tensor["bytes"]


# ---------------------------------------------------------------------------
# 3. the config values
# ---------------------------------------------------------------------------
def test_config_values(model: GGUFModel) -> None:
    d = qwen4exp_text_config(model)
    assert d["hidden_size"] == 2560
    assert d["num_hidden_layers"] == 48
    # 12 full-attention layers at 0-based indices 3, 7, ..., 47
    full_attention_indices = [
        i for i, layer_type in enumerate(d["layer_types"]) if layer_type == "full_attention"
    ]
    assert full_attention_indices == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]
    assert d["linear_value_head_dim"] == 128
    assert d["linear_num_value_heads"] == 48
    assert d["linear_num_key_heads"] == 16
    assert d["head_dim"] == 256
    assert d["num_experts"] == 512
    assert d["num_experts_per_tok"] == 10
    assert d["vocab_size"] == 248320
    assert d["ple_layer_ids"] == [2]
    assert d["ple_embed_dim"] == 2560
    assert d["indexer_compress_ratio"] == 4
    assert d["rope_parameters"]["partial_rotary_factor"] == 0.25
    assert d["rope_parameters"]["mrope_section"] == [11, 11, 10]


# ---------------------------------------------------------------------------
# 4. the dict builds a real Qwen4ExpTextConfig
# ---------------------------------------------------------------------------
def test_builds_real_config(model: GGUFModel) -> None:
    from vllm.models.qwen4_exp.config import Qwen4ExpTextConfig

    d = qwen4exp_text_config(model)
    cfg = Qwen4ExpTextConfig(**d)
    assert cfg.layers_block_type.count("attention") == 12


# ---------------------------------------------------------------------------
# 5. vLLM's own hash constants equal the GGUF ones
# ---------------------------------------------------------------------------
def test_hash_constants_match_gguf(model: GGUFModel) -> None:
    from vllm.models.qwen4_exp.config import Qwen4ExpTextConfig
    from vllm.models.qwen4_exp.nvidia.ngram_embedding import (
        Qwen4ExpNGramEmbedding,
    )

    cls = Qwen4ExpNGramEmbedding
    d = qwen4exp_text_config(model)
    cfg = Qwen4ExpTextConfig(**d)
    ple = qwen4exp_ple_constants(model)

    # per-head vocabulary layout
    sizes, offsets, total = cls._make_vocab_layout(
        ngram_vocab_size_base=cfg.ngram_vocab_size_base,
        ngram_heads=16,
        ple_dense_layer_id=0,
    )
    assert sizes == ple["head_vocab_sizes"]
    assert offsets == ple["head_offsets"]
    padded_total = ((total + 127) // 128) * 128
    assert padded_total == ple["table_rows"]

    # per-layer hash multipliers
    gguf_multipliers = ple["layer_multipliers"]
    multipliers_id0 = cls._make_layer_multipliers(
        ngram_size=3,
        unigram_vocab_size=248320,
        seed=1234,
        ple_dense_layer_id=0,
    )
    multipliers_id1 = cls._make_layer_multipliers(
        ngram_size=3,
        unigram_vocab_size=248320,
        seed=1234,
        ple_dense_layer_id=1,
    )
    matching_id = None
    if multipliers_id0 == gguf_multipliers:
        matching_id = 0
    elif multipliers_id1 == gguf_multipliers:
        matching_id = 1
    assert matching_id is not None, (
        "neither ple_dense_layer_id=0 nor ple_dense_layer_id=1 "
        f"reproduces the GGUF layer_multipliers {gguf_multipliers}; "
        f"id=0 gave {multipliers_id0}, id=1 gave {multipliers_id1}"
    )
    assert matching_id == 0, (
        f"expected ple_dense_layer_id=0 to match the GGUF "
        f"layer_multipliers {gguf_multipliers}, but id=0 gave "
        f"{multipliers_id0} and id=1 gave {multipliers_id1}"
    )


# ---------------------------------------------------------------------------
# 6. raw() on a float tensor of a header-only file raises
# ---------------------------------------------------------------------------
def test_raw_float_beyond_eof_raises(model: GGUFModel) -> None:
    float_tensor_names = [
        name for name, info in model.tensors.items() if info.is_float
    ]
    assert float_tensor_names, "expected at least one float tensor"
    name = float_tensor_names[0]
    with pytest.raises(Exception):
        raw_bytes = model.raw(name)
        # force numpy to actually read the (out-of-bounds) data
        _ = raw_bytes[0]
