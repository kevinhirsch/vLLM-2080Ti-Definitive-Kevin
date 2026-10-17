"""CPU test: each TP rank's QSA K/V amax dump keeps its own file name at exit.

The dump runs from atexit, after WorkerProc.shutdown() has destroyed the model-parallel groups; looking the rank up
then asserts, and every rank used to fall back to rank 0, so the last writer overwrote the others' K/V heads (2 KV
heads over TP2: one head per rank, so one head per layer was lost). The rank is now taken when the owner registers.
"""
import json

import pytest
import torch

import vllm.distributed as dist
from vllm.models.qwen4_exp.nvidia import model as _model  # noqa: F401  (import order: avoids the qsa cycle)
from vllm.models.qwen4_exp.nvidia import qsa as Q


class _Owner:
    def __init__(self, name, k, v):
        self.layer_name = name
        self._kv_amax = torch.tensor([k, v])


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "_KV_AMAX_DUMP", str(tmp_path / "kvamax"))
    monkeypatch.setattr(Q, "_KV_AMAX_OWNERS", [])
    monkeypatch.setattr(Q, "_KV_AMAX_RANK", [])
    return tmp_path


def _groups_destroyed():
    raise AssertionError("tensor model parallel group is not initialized")


def test_rank_taken_at_registration_survives_group_teardown(recorder, monkeypatch):
    monkeypatch.setattr(dist, "get_tensor_model_parallel_rank", lambda: 1)
    Q._kv_amax_register(_Owner("model.language_model.layers.3.self_attn.attn", 12.5, 3.25))
    Q._kv_amax_register(_Owner("model.language_model.layers.7.self_attn.attn", 8.0, 2.0))
    monkeypatch.setattr(dist, "get_tensor_model_parallel_rank", _groups_destroyed)  # what atexit sees
    Q._dump_kv_amax()
    assert sorted(p.name for p in recorder.iterdir()) == ["kvamax.rank1.json"]
    table = json.loads((recorder / "kvamax.rank1.json").read_text())
    assert table == {"model.language_model.layers.3.self_attn.attn": [12.5, 3.25],
                     "model.language_model.layers.7.self_attn.attn": [8.0, 2.0]}


def test_no_tp_group_at_registration_writes_rank0(recorder, monkeypatch):
    monkeypatch.setattr(dist, "get_tensor_model_parallel_rank", _groups_destroyed)
    Q._kv_amax_register(_Owner("model.language_model.layers.3.self_attn.attn", 1.0, 1.0))
    Q._dump_kv_amax()
    assert [p.name for p in recorder.iterdir()] == ["kvamax.rank0.json"]


def test_owner_without_samples_is_left_out(recorder, monkeypatch):
    monkeypatch.setattr(dist, "get_tensor_model_parallel_rank", lambda: 0)
    idle = _Owner("model.language_model.layers.11.self_attn.attn", 0.0, 0.0)
    idle._kv_amax = None  # registered, never saw a token
    Q._kv_amax_register(idle)
    Q._kv_amax_register(_Owner("model.language_model.layers.3.self_attn.attn", 4.0, 5.0))
    Q._dump_kv_amax()
    assert json.loads((recorder / "kvamax.rank0.json").read_text()) == {
        "model.language_model.layers.3.self_attn.attn": [4.0, 5.0]}
