"""CPU test: Qwen4Exp post-load transforms run once, after the whole checkpoint (NM-9 W2h).

AutoWeightsLoader calls a submodule's ``load_weights`` once per contiguous run of its weight
names. The EXL3 Flash-Next pack sorts into three runs of ``model.language_model.*``
(``lm_head`` in shard 5 and the n-gram table file split them), so a transform at the end of
``Qwen4ExpModel.load_weights`` ran after the first run only. With ``VLLM_QWEN4_HC_W8=1`` it
quantized later layers' not-yet-loaded HC weights, emptied their parameters, and the next run
failed: "Attempted to load weight (torch.Size([320, 10240])) into parameter (torch.Size([0]))".
The int8 KV-scale finalizer memoizes at its first call, so it had the same hazard. Both now run
from the top-level model's ``process_weights_after_loading``, which vLLM calls once.
"""
import contextlib
import os
import tempfile
import types

import pytest
import torch
from torch import nn

from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.models.qwen4_exp.nvidia import model as M  # noqa: F401  (import order: avoids the qsa cycle)
from vllm.models.qwen4_exp.nvidia import hyperconnection as H
from vllm.models.qwen4_exp.nvidia import mtp as MTP
from vllm.models.qwen4_exp.nvidia.ops.hc_w8 import HC_W8_GROUP, quantize_w8_groupwise

@pytest.fixture()
def cpu_dist():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    fd, path = tempfile.mkstemp()
    os.close(fd)
    try:
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                         distributed_init_method=f"file://{path}", backend="gloo")
            initialize_model_parallel(1, 1)
            yield
        cleanup_dist_env_and_memory()
    finally:
        with contextlib.suppress(OSError):  # the FileStore removes it itself
            os.unlink(path)


class _Inner(nn.Module):
    """Stands in for Qwen4ExpModel: a decoder layer's HC and the final HC mixer, loaded by AutoWeightsLoader."""

    def __init__(self, convert_in_load_weights: bool):
        super().__init__()
        # lora 64 keeps the up projection's K a multiple of the int8 group; 64 + 4 pads to 80
        cfg = H.HyperConnectionConfig(hc_count=4, hidden_size=64, hc_lowrank=64, params_dtype=torch.float16)
        self.layers = nn.ModuleList([nn.ModuleDict({"attn_hc": H.GatedResidual(cfg, use_combine=True)})])
        # the final mixer has no combine: a plain ReplicatedLinear down projection, as in the pack, whose
        # [320, 10240] weight is the one W2h failed on (default_weight_loader into the emptied parameter)
        self.hyper_connection_mixer = H.GatedResidual(cfg, use_combine=False)
        self.convert_in_load_weights = convert_in_load_weights
        self.load_calls = 0

    def hc_modules(self):
        return {"layers.0.attn_hc": self.layers[0]["attn_hc"], "hyper_connection_mixer": self.hyper_connection_mixer}

    def load_weights(self, weights):
        self.load_calls += 1
        loaded = AutoWeightsLoader(self).load_weights(weights)
        if self.convert_in_load_weights:  # the pre-fix placement
            H.convert_hc_weights_w8(self)
        return loaded


class _Root(nn.Module):
    """Stands in for the top-level model: the inner model plus an lm_head between its runs."""

    def __init__(self, convert_in_load_weights: bool):
        super().__init__()
        self.model = _Inner(convert_in_load_weights)
        self.lm_head = nn.Linear(8, 8, bias=False)

    def load_weights(self, weights):
        return AutoWeightsLoader(self).load_weights(weights)


def _split_checkpoint(root: _Root, seed: int = 0):
    """The pack's order: a decoder layer's HC, then lm_head, then the final mixer (two model.* runs)."""
    g = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, p in root.named_parameters():
        tensors[name] = (torch.randn(p.shape, generator=g) * 0.05).to(p.dtype)
    order = ([n for n in tensors if n.startswith("model.layers.0.")] + ["lm_head.weight"]
             + [n for n in tensors if n.startswith("model.hyper_connection_mixer.")])
    assert len(order) == len(tensors)
    return [(n, tensors[n]) for n in order], tensors


def test_loader_calls_submodule_once_per_run(cpu_dist):
    root = _Root(convert_in_load_weights=False)
    weights, _ = _split_checkpoint(root)
    root.load_weights(iter(weights))
    assert root.model.load_calls == 2  # the hazard: a submodule's load_weights is not "after loading"


def test_old_placement_reproduces_w2h_failure(cpu_dist, monkeypatch):
    monkeypatch.setenv("VLLM_QWEN4_HC_W8", "1")
    root = _Root(convert_in_load_weights=True)
    weights, _ = _split_checkpoint(root)
    with pytest.raises(AssertionError, match=r"into parameter \(torch.Size\(\[0\]\)\)"):
        root.load_weights(iter(weights))


def test_model_level_hook_converts_every_layer(cpu_dist, monkeypatch):
    monkeypatch.setenv("VLLM_QWEN4_HC_W8", "1")
    root = _Root(convert_in_load_weights=False)
    weights, tensors = _split_checkpoint(root)
    root.load_weights(iter(weights))
    M.finalize_qwen4_exp_weights(root)  # what the top-level process_weights_after_loading runs
    converted = 0
    for prefix, hc in root.model.hc_modules().items():  # the mixer is the run loaded after lm_head
        for name in ("input_mix_weight_down_block_inject", "input_mix_weight_down", "input_mix_weight_up"):
            lin = getattr(hc, name, None)
            if lin is None:
                continue
            w = tensors[f"model.{prefix}.{name}.weight"].clone()
            if name == "input_mix_weight_down_block_inject" and hc.pad_size:
                w[hc.lora_rank + hc.hc_count:] = 0  # padding rows are zeroed before quantizing
            q, s = quantize_w8_groupwise(w, HC_W8_GROUP)
            assert lin.weight.numel() == 0, (prefix, name)
            assert torch.equal(lin.hc_w8_q, q) and torch.equal(lin.hc_w8_s, s), (prefix, name)
            converted += 1
    assert converted == 4
    mixer_up = root.model.hyper_connection_mixer.input_mix_weight_up
    before = mixer_up.hc_w8_q.clone()
    M.finalize_qwen4_exp_weights(root)  # idempotent: a second call changes nothing
    assert torch.equal(mixer_up.hc_w8_q, before)


def _recorders(monkeypatch):
    calls = []
    for mod in (M, MTP):  # MTP used to import both names itself
        monkeypatch.setattr(mod, "finalize_qsa_kv_cache_scales", lambda m: calls.append(("kv", m)), raising=False)
        monkeypatch.setattr(mod, "convert_hc_weights_w8", lambda m: calls.append(("hc", m)) or 0, raising=False)
    return calls


class _Stub(nn.Module):
    def __init__(self):
        super().__init__()
        self._qsa_layer_ids = ()
        self.is_fused_shared_expert_enabled = False
        self.config = types.SimpleNamespace(num_experts=0)
        self.hf_to_vllm_mapper = WeightsMapper()
        self.hyper_connection_mixer = None


@pytest.mark.parametrize("load_weights", [M.Qwen4ExpModel.load_weights,
                                          MTP.Qwen4ExpMultiTokenPredictor.load_weights])
def test_inner_load_weights_runs_no_transform(monkeypatch, load_weights):
    calls = _recorders(monkeypatch)
    load_weights(_Stub(), iter([]))
    assert calls == []


@pytest.mark.parametrize("cls", [M.Qwen4ExpForCausalLM, M.Qwen4ExpForConditionalGeneration, MTP.Qwen4ExpMTP])
def test_top_level_hook_runs_each_transform_once(monkeypatch, cls):
    calls = _recorders(monkeypatch)
    stub = _Stub()
    cls.process_weights_after_loading(stub)
    assert calls == [("kv", stub), ("hc", stub)]
