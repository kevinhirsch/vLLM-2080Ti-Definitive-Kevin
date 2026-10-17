"""CPU test: host-table PLE prefetch during breakable CUDA-graph capture does no gather and no CUDA work.

At capture time the eager break runs on capture-time buffers: the n-gram ids come from a captured,
not executed segment, so they are garbage (W2d: "PLE row id out of range ... -2^63"). A gather thread's
CUDA calls during the next segment's capture also invalidate it (W2c: cuEventSynchronize from another
thread). The capture-time call must do nothing but record the token count and a finished future;
_finalize then copies don't-care rows into the graph-owned output. Replays run the real path.
"""
import types

import torch

from vllm.models.qwen4_exp.nvidia import model as _model  # noqa: F401  (imports ple_host without the cycle)
from vllm.models.qwen4_exp.nvidia import ple_host


class _Forbidden:
    """Any attribute access means the capture-time call touched CUDA or the gather pool."""

    def __getattr__(self, name):
        raise AssertionError(f"capture-time start_prefetch used {name}")


def _fake(**kw):
    return types.SimpleNamespace(_pending=None, _pending_tokens=0, _stream=_Forbidden(),
                                 _executor=_Forbidden(), _ids_dev=_Forbidden(), _ids_host=_Forbidden(), **kw)


def test_capture_time_prefetch_is_a_noop(monkeypatch):
    monkeypatch.setattr(ple_host.BreakableCUDAGraphCapture, "is_active", classmethod(lambda cls: True))
    me = _fake()
    ids = torch.full((3, 16), -(2**63), dtype=torch.int64)  # garbage, as in a captured-but-unexecuted segment
    start = getattr(ple_host.Qwen4ExpPLEHostEmbedding.start_prefetch, "__wrapped__",
                    ple_host.Qwen4ExpPLEHostEmbedding.start_prefetch)
    start(me, torch.empty(3, 4), ids)
    assert me._pending is ple_host._DONE and me._pending.done()
    assert me._pending_tokens == 3


def test_capture_time_finalize_fills_graph_output(monkeypatch):
    monkeypatch.setattr(ple_host.BreakableCUDAGraphCapture, "is_active", classmethod(lambda cls: True))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: types.SimpleNamespace(wait_stream=lambda s: None))
    me = types.SimpleNamespace(_pending=ple_host._DONE, _pending_tokens=2, _stream=None,
                               _prefetch_out=torch.arange(12.0).view(4, 3))
    out = torch.full((3, 3), 7.0)
    fin = getattr(ple_host.Qwen4ExpPLEHostEmbedding._finalize, "__wrapped__",
                  ple_host.Qwen4ExpPLEHostEmbedding._finalize)
    fin(me, out)
    assert me._pending is None
    assert torch.equal(out[:2], me._prefetch_out[:2]) and torch.equal(out[2:], torch.zeros(1, 3))
