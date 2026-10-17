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


def test_capture_time_eager_call_is_not_recorded(monkeypatch):
    """add_eager runs the QSA op once while capturing, on buffers no kernel wrote yet: garbage must not enter."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    owner = _Owner("model.language_model.layers.3.self_attn.attn", 0.0, 0.0)
    owner._kv_amax = None
    garbage = torch.full((4, 1, 8), float("nan"))
    monkeypatch.setattr(BreakableCUDAGraphCapture._tls, "active", object(), raising=False)
    Q._kv_amax_record(owner, garbage, garbage, 4)
    assert owner._kv_amax is None
    monkeypatch.setattr(BreakableCUDAGraphCapture._tls, "active", None)
    k = torch.tensor([[[1.0, -3.0]], [[2.0, 0.5]], [[99.0, 99.0]]])  # row 2 is padding past num_tokens
    v = torch.tensor([[[0.25, -0.5]], [[0.1, 0.2]], [[99.0, 99.0]]])
    Q._kv_amax_record(owner, k, v, 2)
    assert owner._kv_amax.tolist() == [3.0, 0.5]
    Q._kv_amax_record(owner, k * 2, v, 2)
    assert owner._kv_amax.tolist() == [6.0, 0.5]


def test_nan_from_a_real_step_still_propagates():
    owner = _Owner("model.language_model.layers.3.self_attn.attn", 0.0, 0.0)
    owner._kv_amax = None
    Q._kv_amax_record(owner, torch.ones(2, 1, 4), torch.ones(2, 1, 4), 2)
    Q._kv_amax_record(owner, torch.full((2, 1, 4), float("nan")), torch.ones(2, 1, 4), 2)
    assert torch.isnan(owner._kv_amax[0]) and owner._kv_amax[1] == 1.0


def test_scale_builder_refuses_nan_in_any_rank_file(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    tool = Path(__file__).resolve().parents[3] / "tools" / "qwen4exp" / "kvscales_from_amax.py"
    good = {"language_model.model.layers.3.self_attn.attn": [4.0, 2.0]}
    bad = {"language_model.model.layers.3.self_attn.attn": [float("nan"), 2.0]}
    (tmp_path / "kvamax.rank0.json").write_text(json.dumps(bad))  # json writes NaN; max(0.0, nan) would keep 0.0
    (tmp_path / "kvamax.rank1.json").write_text(json.dumps(good))
    run = subprocess.run([sys.executable, str(tool), str(tmp_path / "kvamax"), str(tmp_path)],
                         capture_output=True, text=True)
    assert run.returncode != 0 and "non-finite" in run.stderr
    assert not (tmp_path / "model-kvscales.safetensors").exists()
    (tmp_path / "kvamax.rank0.json").write_text(json.dumps(good))
    run = subprocess.run([sys.executable, str(tool), str(tmp_path / "kvamax"), str(tmp_path)],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "1 layers from 2 rank file(s)" in run.stdout
