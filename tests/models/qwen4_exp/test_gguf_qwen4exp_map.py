# SPDX-License-Identifier: Apache-2.0
"""CPU tests: qwen4exp GGUF -> vLLM plan, the inverse of llama.cpp's converter transforms, TP/EP slicing."""

import os

import numpy as np
import pytest
import torch

from vllm.model_executor.model_loader import gguf_qwen4exp as Q

HEADERS = os.environ.get("NM9_GGUF_HEADERS", "/home/kevin/projects/lanes/coexec/nm9/headers")
REAL = os.path.join(HEADERS, "flashnext-hdr-00001-of-00002.gguf")


def _llamacpp_reorder_v_heads(t, dim, num_k_heads, num_v_per_k, head_dim):
    """Verbatim logic of llama.cpp conversion/qwen.py _LinearAttentionVReorderBase._reorder_v_heads."""
    shape = list(t.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1:]
    t = t.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return t.permute(*perm).contiguous().reshape(*shape)


@pytest.mark.parametrize("nk,nv,hd", [(16, 48, 128), (4, 12, 8), (2, 2, 4)])
def test_vhead_perm_inverts_llamacpp_rows(nk, nv, hd):
    grouped = torch.randn(nv * hd, 5)
    tiled = _llamacpp_reorder_v_heads(grouped, 0, nk, nv // nk, hd)
    back = Q.permute_v_rows(tiled, nk, nv, hd)
    assert torch.equal(back, grouped)


def test_vhead_perm_qkv_prefix_untouched():
    nk, nv, hd = 4, 12, 8
    qk = torch.randn(2 * nk * hd, 3)
    v = torch.randn(nv * hd, 3)
    tiled = torch.cat([qk, _llamacpp_reorder_v_heads(v, 0, nk, nv // nk, hd)])
    back = Q.permute_v_rows(tiled, nk, nv, hd)
    assert torch.equal(back[: qk.shape[0]], qk) and torch.equal(back[qk.shape[0]:], v)


def test_transforms_match_converter():
    nk, nv, hd = 4, 12, 8
    # out_proj columns
    w = torch.randn(7, nv * hd)
    p = Q.TensorPlan("x", "dequant_hf", "t", 0, tuple(w.shape), transforms=(Q.VPERM_COLS,), vperm_head=hd)
    tiled = _llamacpp_reorder_v_heads(w, 1, nk, nv // nk, hd)
    assert torch.equal(Q.apply_transforms(tiled, p, nk, nv), w)
    # A_log: GGUF stores -exp(A_log), V-reordered
    a_log = torch.randn(nv)
    ssm_a = -torch.exp(_llamacpp_reorder_v_heads(a_log.unsqueeze(-1), 0, nk, nv // nk, 1).squeeze(-1))
    p = Q.TensorPlan("x", "hf", "t", 0, (nv,), transforms=(Q.A_LOG, Q.VPERM_ROWS), vperm_head=1)
    assert torch.allclose(Q.apply_transforms(ssm_a, p, nk, nv), a_log, atol=1e-6)
    # conv1d: HF [C,1,K] -> squeeze -> V channels reordered
    conv = torch.randn(2 * nk * hd + nv * hd, 1, 4)
    d = conv.squeeze()
    qk, v = d[: 2 * nk * hd], d[2 * nk * hd:]
    stored = torch.cat([qk, _llamacpp_reorder_v_heads(v, 0, nk, nv // nk, hd)])
    p = Q.TensorPlan("x", "hf", "t", 0, tuple(stored.shape), transforms=(Q.VPERM_CONV, Q.UNSQUEEZE1), vperm_head=hd)
    assert torch.equal(Q.apply_transforms(stored, p, nk, nv), conv)
    # Gemma norms: stored w + 1
    w = torch.randn(16)
    p = Q.TensorPlan("x", "hf", "t", 0, (16,), transforms=(Q.NORM_MINUS1,))
    assert torch.allclose(Q.apply_transforms(w + 1, p, nk, nv), w, atol=1e-6)


@pytest.mark.skipif(not os.path.exists(REAL), reason="real GGUF header copies not available")
def test_real_header_plan_is_complete():
    from vllm.model_executor.model_loader.gguf_reader import GGUFModel

    gm = GGUFModel(REAL)
    plans = Q.plan(gm)
    assert len(plans) == len(gm.tensors)
    kinds = {}
    for p in plans:
        kinds[p.kind] = kinds.get(p.kind, 0) + 1
    # 48 MoE layers, 36 GDN ssm_out dequantized, 12 QSA layers with q/k concatenated indexer
    assert kinds["qmoe"] == 48 * 3
    assert kinds["dequant_hf"] == 36
    assert kinds["concat_hf"] == 24
    assert kinds["ple_table"] == 1
    specs = Q.quant_specs(gm, plans)
    from vllm.model_executor.model_loader.gguf_qwen4exp import EmbedSpec, LinearSpec, MoESpec

    assert isinstance(specs["model.embed_tokens"], EmbedSpec) and isinstance(specs["lm_head"], EmbedSpec)
    qkv = specs["model.layers.3.self_attn.qkv_proj"]
    assert isinstance(qkv, LinearSpec) and set(qkv.shard_types) == {0, 1, 2}
    gdn = specs["model.layers.0.linear_attn.in_proj_qkvz"]
    assert set(gdn.shard_types) == {0, 1, 2, 3} and gdn.shard_types[0] == gdn.shard_types[2]
    assert "model.layers.0.linear_attn.out_proj" not in specs  # dequantized to fp16
    moe = [s for s in specs.values() if isinstance(s, MoESpec)]
    assert len(moe) == 48 and {s.w2_type for s in moe} == {20, 42}
    # every quantized linear input dim is a multiple of its block, also when split for TP2 row-parallel
    for p in plans:
        if p.kind in ("qlinear", "qmoe", "qembed"):
            from vllm.model_executor.model_loader.gguf_reader import GGML_TYPES

            be = GGML_TYPES[p.ggml_type][1]
            assert p.shape[-1] % be == 0, p.gguf_name
            if p.target.endswith(("o_proj", "down_proj")) or p.moe_part == "w2":
                assert (p.shape[-1] // 2) % be == 0, f"TP2 split of {p.gguf_name}"


class _FakeColumn(torch.nn.Module):
    pass


def test_load_gguf_linear_tp_slicing():
    from vllm.model_executor.layers.linear import QKVParallelLinear
    from vllm.model_executor.layers.quantization.gguf import load_gguf_linear

    # QKV with 2 KV heads at TP4 -> KV replicated twice; q, k, v of different types/segments
    layer = QKVParallelLinear.__new__(QKVParallelLinear)
    torch.nn.Module.__init__(layer)
    layer.tp_size, layer.tp_rank, layer.num_kv_head_replicas = 4, 3, 2
    layer.gguf_output_partition_sizes = [6, 2, 2]
    layer.gguf_input_size_per_partition = 32
    layer.gguf_segments = [("qweight_0", 23, 0, 6, 0, 0), ("qweight_1", 14, 6, 4, 1, 2)]
    layer.qweight_0 = torch.nn.Parameter(torch.zeros(6, 3, dtype=torch.uint8), requires_grad=False)
    layer.qweight_1 = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.uint8), requires_grad=False)
    q = torch.arange(24 * 3, dtype=torch.uint8).reshape(24, 3)  # 4 ranks x 6 rows
    load_gguf_linear(layer, (0,), q)
    assert torch.equal(layer.qweight_0.data, q[18:24])
    kvt = (100 + torch.arange(8 * 3)).to(torch.uint8).reshape(8, 3)  # k: 2 heads x 2 rows, v: same
    load_gguf_linear(layer, (1, 2), kvt)
    # rank 3 with 2 replicas -> kv shard rank 1: k rows 2:4, v rows 4+2:4+4
    assert torch.equal(layer.qweight_1.data[:2], kvt[2:4])
    assert torch.equal(layer.qweight_1.data[2:], kvt[6:8])
