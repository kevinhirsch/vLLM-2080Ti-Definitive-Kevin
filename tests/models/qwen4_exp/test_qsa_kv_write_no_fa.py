"""CPU test: the QSA impl's K/V write must not depend on flash-attn being importable.

On SM75 flash_attn.py never imports reshape_and_cache_flash (it is guarded by
is_flash_attn_varlen_func_available()), so the inherited FlashAttentionImpl.do_kv_cache_update
raised NameError on the first real K/V write. The override calls vllm._custom_ops directly;
here the op is replaced by a torch scatter with the kernel's slot semantics and the cache rows
are checked.
"""
import types

import torch

import vllm._custom_ops as ops
from vllm.models.qwen4_exp.nvidia import model as _model  # noqa: F401  (imports qsa without the cycle)
from vllm.models.qwen4_exp.nvidia import qsa


def test_kv_write_without_flash_attn(monkeypatch):
    blocks, heads, block_size, dim = 3, 2, 4, 8
    kv_cache = torch.zeros(blocks, heads, block_size, 2 * dim)
    key = torch.randn(5, heads, dim)
    value = torch.randn(5, heads, dim)
    slots = torch.tensor([0, 5, 6, 11, 2])
    seen = {}

    def fake_op(k, v, kc, vc, slot_mapping, kv_cache_dtype, k_scale, v_scale):
        seen["dtype"] = kv_cache_dtype
        for i, s in enumerate(slot_mapping.tolist()):
            kc[s // block_size, s % block_size] = k[i]
            vc[s // block_size, s % block_size] = v[i]

    monkeypatch.setattr(ops, "reshape_and_cache_flash", fake_op)
    impl = types.SimpleNamespace(head_size=dim, kv_cache_dtype="auto", attn_type=qsa.AttentionType.DECODER)
    layer = types.SimpleNamespace(_k_scale=torch.ones(()), _v_scale=torch.ones(()))
    qsa.Qwen4ExpQSAFlashAttentionImpl.do_kv_cache_update(impl, layer, key, value, kv_cache, slots)

    assert seen["dtype"] == "auto"
    for i, s in enumerate(slots.tolist()):
        b, n = divmod(s, block_size)
        assert torch.equal(kv_cache[b, :, n, :dim], key[i])
        assert torch.equal(kv_cache[b, :, n, dim:], value[i])
    assert "do_kv_cache_update" in qsa.Qwen4ExpQSAFlashAttentionImpl.__dict__
