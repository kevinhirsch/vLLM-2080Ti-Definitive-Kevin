# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic tiny qwen4exp GGUF writer (CPU tests).

Writes a GGUF v3 file whose tensor names, ggml dims and quant types
follow ``real_tensor_table.txt`` scaled to the TINY shape.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# type name -> (ggml type id, block_elems, block_bytes)
GGML = {
    "F32": (0, 1, 4),
    "F16": (1, 1, 2),
    "BF16": (30, 1, 2),
    "Q8_0": (8, 32, 34),
    "Q4_K": (12, 256, 144),
    "Q5_K": (13, 256, 176),
    "Q6_K": (14, 256, 210),
    "IQ2_XXS": (16, 256, 66),
    "IQ2_XS": (17, 256, 74),
    "IQ3_XXS": (18, 256, 98),
    "IQ4_NL": (20, 32, 18),
    "IQ3_S": (21, 256, 110),
    "IQ2_S": (22, 256, 82),
    "IQ4_XS": (23, 256, 136),
    "Q2_0": (42, 64, 18),
}

_FLOAT_TYPES = ("F32", "F16", "BF16")
# types whose single `d` scale is at block bytes 0..1
_SCALE_D0 = (
    "Q8_0", "IQ4_NL", "Q2_0", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_XS",
)


def _f16_bytes(x: float) -> bytes:
    return np.float16(x).tobytes()


def random_blocks(type_name: str, n_elems: int, rng: np.random.Generator) -> bytes:
    """Random block bytes for a type; fp16 scale fields are finite small."""
    _type_id, block_elems, block_bytes = GGML[type_name]
    n_blocks = n_elems // block_elems

    if type_name in _FLOAT_TYPES:
        values = rng.standard_normal(n_elems) * 0.02
        if type_name == "F32":
            return values.astype(np.float32).tobytes()
        if type_name == "F16":
            return values.astype(np.float16).tobytes()
        # BF16 = upper 16 bits of float32
        bits = values.astype(np.float32).view(np.uint32)
        return (bits >> 16).astype(np.uint16).tobytes()

    out = bytearray()
    for _ in range(n_blocks):
        block = np.array(rng.integers(0, 256, size=block_bytes, dtype=np.uint8), dtype=np.uint8)
        if type_name in _SCALE_D0:
            block[0:2] = np.frombuffer(_f16_bytes(rng.uniform(0.001, 0.02)), dtype=np.uint8)
        elif type_name in ("Q4_K", "Q5_K"):
            block[0:2] = np.frombuffer(_f16_bytes(rng.uniform(0.001, 0.02)), dtype=np.uint8)
            block[2:4] = np.frombuffer(_f16_bytes(rng.uniform(0.001, 0.02)), dtype=np.uint8)
        elif type_name == "Q6_K":
            block[208:210] = np.frombuffer(_f16_bytes(rng.uniform(0.001, 0.02)), dtype=np.uint8)
        out.extend(block)
    return bytes(out)


def _pack_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _pack_kv_value(val) -> bytes:
    if isinstance(val, bool):
        return struct.pack("<I?", 7, val)
    if isinstance(val, int):
        if 0 <= val < 2**32:
            return struct.pack("<II", 4, val)
        if val < 0:
            return struct.pack("<Iq", 11, val)
        return struct.pack("<IQ", 10, val)
    if isinstance(val, float):
        return struct.pack("<If", 6, val)
    if isinstance(val, str):
        return struct.pack("<I", 8) + _pack_str(val)
    if isinstance(val, (list, tuple)):
        if all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in val):
            vals = [int(v) for v in val]
            if any(v >= 2**31 for v in vals):
                body = struct.pack("<I", 11)
                body += struct.pack("<Q", len(vals)) + b"".join(struct.pack("<q", v) for v in vals)
            else:
                body = struct.pack("<I", 5)
                body += struct.pack("<Q", len(vals)) + b"".join(struct.pack("<i", v) for v in vals)
            return struct.pack("<I", 9) + body
        if all(isinstance(v, str) for v in val):
            body = struct.pack("<I", 8)
            body += struct.pack("<Q", len(val))
            body += b"".join(_pack_str(v) for v in val)
            return struct.pack("<I", 9) + body
    raise ValueError(f"unsupported GGUF kv value: {val!r}")


def write_gguf(
    path: str | Path,
    kv: dict,
    tensors: list[tuple[str, list[int], str, bytes]],
) -> None:
    """GGUF v3 writer (32-byte alignment)."""
    path = Path(path)
    alignment = 32

    offset = 0
    infos: list[tuple[str, list[int], int, int, int]] = []
    for name, dims, type_name, data in tensors:
        type_id = GGML[type_name][0]
        padded = (len(data) + alignment - 1) // alignment * alignment
        infos.append((name, dims, type_id, offset, padded))
        offset += padded

    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", len(kv)))
        for key, val in kv.items():
            f.write(_pack_str(key))
            f.write(_pack_kv_value(val))
        for name, dims, type_id, off, _padded in infos:
            f.write(_pack_str(name))
            f.write(struct.pack("<I", len(dims)))
            for d in dims:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", type_id))
            f.write(struct.pack("<Q", off))
        pos = f.tell()
        pad = (-pos) % alignment
        if pad:
            f.write(b"\x00" * pad)
        for i, (name, dims, type_id, off, padded) in enumerate(infos):
            data = tensors[i][3]
            f.write(data)
            if len(data) < padded:
                f.write(b"\x00" * (padded - len(data)))


# ---------------------------------------------------------------------------
# TINY model shape
# ---------------------------------------------------------------------------

TINY = dict(
    n_layer=4,
    n_embd=256,
    n_head=4,
    n_head_kv=2,
    head_dim=256,
    n_expert=16,
    n_expert_used=4,
    expert_ff=128,
    shexp_ff=128,
    ssm_conv=4,
    ssm_state=128,
    ssm_group=4,
    ssm_time_step_rank=12,
    ssm_inner=12 * 128,
    hc_count=4,
    hc_lowrank=32,
    indexer_heads=4,
    indexer_dim=128,
    indexer_top_k=2048,
    ple_layers=[1],
    ngram_size=3,
    heads_per_ngram=8,
    ple_row_dim=32,
    PLE_E=512,
    vocab=1024,
    ple_table_rows=4096,
)


def _is_norm_name(name: str) -> bool:
    base = name.rsplit(".", 1)[0]
    return base.endswith("_norm") or base.endswith("norm")


def _tensor_data(name: str, dims: list[int], type_name: str, rng: np.random.Generator) -> bytes:
    n_elems = int(np.prod(dims))
    if type_name == "F32" and _is_norm_name(name):
        values = 1.0 + rng.standard_normal(n_elems) * 0.01
        return values.astype(np.float32).tobytes()
    if type_name == "F32" and name.endswith("ssm_a"):
        values = -np.exp(rng.uniform(-1.0, 1.0, size=n_elems))
        return values.astype(np.float32).tobytes()
    return random_blocks(type_name, n_elems, rng)


# ---------------------------------------------------------------------------
# TINY tensor table (names, ggml dims, types)
# ---------------------------------------------------------------------------

def _tiny_tensor_table() -> list[tuple[str, list[int], str]]:
    H = TINY["n_embd"]
    HC = H * TINY["hc_count"]
    R = TINY["hc_lowrank"]
    vocab = TINY["vocab"]
    ple_row_dim = TINY["ple_row_dim"]
    PLE_E = TINY["PLE_E"]
    n_expert = TINY["n_expert"]
    expert_ff = TINY["expert_ff"]
    shexp_ff = TINY["shexp_ff"]
    head_dim = TINY["head_dim"]
    n_head = TINY["n_head"]
    n_head_kv = TINY["n_head_kv"]
    ssm_rank = TINY["ssm_time_step_rank"]
    ssm_inner = TINY["ssm_inner"]
    ssm_state = TINY["ssm_state"]
    ssm_conv = TINY["ssm_conv"]
    indexer_heads = TINY["indexer_heads"]
    indexer_dim = TINY["indexer_dim"]
    hc_count = TINY["hc_count"]

    q2_0 = "Q2_0"
    iq2xxs = "IQ2_XXS"
    iq4nl = "IQ4_NL"

    # ffn_down_shexp type: expert down type if its block divides shexp_ff else IQ4_NL
    down_type = q2_0
    if shexp_ff % GGML[down_type][1] != 0:
        down_type = iq4nl

    tensors: list[tuple[str, list[int], str]] = []
    # global
    tensors.append(("output.weight", [H, vocab], "Q5_K"))
    tensors.append(("output_hc_down.weight", [HC, R], "BF16"))
    tensors.append(("output_hc_norm.weight", [HC], "F32"))
    tensors.append(("output_hc_up.weight", [R, HC], "BF16"))
    tensors.append(("token_embd.weight", [H, vocab], "IQ3_S"))
    tensors.append(("per_layer_token_embd.weight", [ple_row_dim, TINY["ple_table_rows"]], iq4nl))

    for N in range(TINY["n_layer"]):
        p = f"blk.{N}."
        # hc
        for tag in ("attn", "ffn"):
            tensors.append((p + f"hc_{tag}_down.weight", [HC, R], "BF16"))
            tensors.append((p + f"hc_{tag}_up.weight", [R, HC], "BF16"))
            tensors.append((p + f"hc_{tag}_inject.weight", [HC, hc_count], "BF16"))
            tensors.append((p + f"hc_{tag}_norm.weight", [HC], "F32"))
        # ffn shared
        tensors.append((p + "ffn_gate_inp.weight", [H, n_expert], "BF16"))
        tensors.append((p + "ffn_gate_inp_shexp.weight", [H], "BF16"))
        tensors.append((p + "ffn_gate_shexp.weight", [H, shexp_ff], "Q4_K"))
        tensors.append((p + "ffn_up_shexp.weight", [H, shexp_ff], "Q4_K"))
        tensors.append((p + "ffn_down_shexp.weight", [shexp_ff, H], down_type))
        tensors.append((p + "ffn_gate_exps.weight", [H, expert_ff, n_expert], iq2xxs))
        tensors.append((p + "ffn_up_exps.weight", [H, expert_ff, n_expert], iq2xxs))
        tensors.append((p + "ffn_down_exps.weight", [expert_ff, H, n_expert], q2_0))

        if N < TINY["n_layer"] - 1:
            # linear attention
            tensors.append((p + "attn_qkv.weight", [H, 2 * 4 * 128 + ssm_inner], "Q4_K"))
            tensors.append((p + "attn_gate.weight", [H, ssm_inner], "IQ4_XS"))
            tensors.append((p + "ssm_a", [ssm_rank], "F32"))
            tensors.append((p + "ssm_alpha.weight", [H, ssm_rank], "BF16"))
            tensors.append((p + "ssm_beta.weight", [H, ssm_rank], "BF16"))
            tensors.append((p + "ssm_conv1d.weight", [ssm_conv, 2 * 4 * 128 + ssm_inner], "F32"))
            tensors.append((p + "ssm_dt.bias", [ssm_rank], "F32"))
            tensors.append((p + "ssm_norm.weight", [ssm_state], "F32"))
            tensors.append((p + "ssm_out.weight", [ssm_inner, H], "Q6_K"))
        else:
            # full attention
            tensors.append((p + "attn_q.weight", [H, n_head * head_dim * 2], "IQ4_XS"))
            tensors.append((p + "attn_k.weight", [H, n_head_kv * head_dim], "Q6_K"))
            tensors.append((p + "attn_v.weight", [H, n_head_kv * head_dim], "Q4_K"))
            tensors.append((p + "attn_output.weight", [n_head * head_dim, H], "Q5_K"))
            tensors.append((p + "attn_q_norm.weight", [head_dim], "F32"))
            tensors.append((p + "attn_k_norm.weight", [head_dim], "F32"))
            tensors.append((p + "indexer.q_proj.weight", [H, indexer_heads * indexer_dim], "BF16"))
            tensors.append((p + "indexer.k_proj.weight", [H, indexer_dim], "BF16"))
            tensors.append((p + "indexer.q_norm.weight", [indexer_dim], "F32"))
            tensors.append((p + "indexer.k_norm.weight", [indexer_dim], "F32"))

        if N in TINY["ple_layers"]:
            tensors.append((p + "ple_key.weight", [PLE_E, HC], "BF16"))
            tensors.append((p + "ple_value.weight", [PLE_E, H], "BF16"))
            tensors.append((p + "ple_norm_key.weight", [HC], "F32"))
            tensors.append((p + "ple_norm_query.weight", [HC], "F32"))
            tensors.append((p + "ple_norm_conv.weight", [HC], "F32"))
            tensors.append((p + "ple_conv1d.weight", [4, HC], "F16"))

    return tensors


# ---------------------------------------------------------------------------
# KV metadata
# ---------------------------------------------------------------------------

def _tiny_kv() -> dict:
    kv = {
        "general.architecture": "qwen4exp",
        "general.alignment": 32,
        "qwen4exp.block_count": TINY["n_layer"],
        "qwen4exp.context_length": 8192,
        "qwen4exp.attention.layer_norm_rms_epsilon": 1e-6,
        "qwen4exp.rope.freq_base": 10000000.0,
        "qwen4exp.ple.eos_token_id": 248044,
        "qwen4exp.ple.image_token_id": 248056,
        "qwen4exp.embedding_length": TINY["n_embd"],
        "qwen4exp.attention.head_count": TINY["n_head"],
        "qwen4exp.attention.head_count_kv": TINY["n_head_kv"],
        "qwen4exp.attention.key_length": TINY["head_dim"],
        "qwen4exp.attention.value_length": TINY["head_dim"],
        "qwen4exp.expert_count": TINY["n_expert"],
        "qwen4exp.expert_used_count": TINY["n_expert_used"],
        "qwen4exp.expert_feed_forward_length": TINY["expert_ff"],
        "qwen4exp.expert_shared_feed_forward_length": TINY["shexp_ff"],
        "qwen4exp.ssm.conv_kernel": TINY["ssm_conv"],
        "qwen4exp.ssm.state_size": TINY["ssm_state"],
        "qwen4exp.ssm.group_count": TINY["ssm_group"],
        "qwen4exp.ssm.time_step_rank": TINY["ssm_time_step_rank"],
        "qwen4exp.ssm.inner_size": TINY["ssm_inner"],
        "qwen4exp.full_attention_interval": 4,
        "qwen4exp.rope.dimension_count": 64,
        "qwen4exp.hyper_connection.count": TINY["hc_count"],
        "qwen4exp.hyper_connection.low_rank": TINY["hc_lowrank"],
        "qwen4exp.attention.indexer.head_count": TINY["indexer_heads"],
        "qwen4exp.attention.indexer.key_length": TINY["indexer_dim"],
        "qwen4exp.attention.indexer.top_k": TINY["indexer_top_k"],
        "qwen4exp.ple.layers": TINY["ple_layers"],
        "qwen4exp.ple.ngram_size": TINY["ngram_size"],
        "qwen4exp.ple.heads_per_ngram": TINY["heads_per_ngram"],
        "qwen4exp.ple.conv_kernel": 4,
        "qwen4exp.embedding_length_per_layer_input": TINY["ple_row_dim"],
    }
    # arrays
    kv["qwen4exp.attention.compress_ratios"] = [0, 0, 0, 4]
    kv["qwen4exp.rope.dimension_sections"] = [11, 11, 10, 0]
    kv["qwen4exp.ple.layer_multipliers"] = [1, 2, 3]
    # 16 ints each covering 4096 rows (256 each)
    kv["qwen4exp.ple.head_offsets"] = [i * 256 for i in range(16)]
    kv["qwen4exp.ple.head_vocab_sizes"] = [256] * 16
    return kv


# ---------------------------------------------------------------------------
# Public writer
# ---------------------------------------------------------------------------

def write_tiny_qwen4exp(
    path: str | Path,
    seed: int = 0,
    expert_types: tuple[str, str] = ("IQ2_XXS", "Q2_0"),
    shard: bool = False,
) -> dict:
    path = Path(path)
    rng = np.random.default_rng(seed)

    table = _tiny_tensor_table()
    # apply expert_types
    n_expert = TINY["n_expert"]
    tensors: list[tuple[str, list[int], str, bytes]] = []
    name_map: dict[str, tuple[list[int], str]] = {}
    for name, dims, type_name in table:
        base = name.rsplit(".", 1)[0]
        if base.endswith("ffn_gate_exps") or base.endswith("ffn_up_exps"):
            type_name = expert_types[0]
        elif base.endswith("ffn_down_exps"):
            type_name = expert_types[1]
        data = _tensor_data(name, dims, type_name, rng)
        tensors.append((name, dims, type_name, data))
        name_map[name] = (dims, type_name)

    kv = _tiny_kv()

    if not shard:
        write_gguf(path, kv, tensors)
        return {"paths": [str(path)], "tensors": name_map}

    # sharded: two files
    stem = path.with_suffix("")
    p0 = stem.parent / f"{stem.name}-00001-of-00002.gguf"
    p1 = stem.parent / f"{stem.name}-00002-of-00002.gguf"
    total = len(tensors)
    ple_name = "per_layer_token_embd.weight"
    shard0 = [t for t in tensors if t[0] != ple_name]
    shard1 = [t for t in tensors if t[0] == ple_name]
    kv0 = dict(kv)
    kv0["split.no"] = 0
    kv0["split.count"] = 2
    kv0["split.tensors.count"] = total
    kv1 = dict(kv)
    kv1["split.no"] = 1
    kv1["split.count"] = 2
    kv1["split.tensors.count"] = total
    write_gguf(p0, kv0, shard0)
    write_gguf(p1, kv1, shard1)
    return {"paths": [str(p0), str(p1)], "tensors": name_map}
