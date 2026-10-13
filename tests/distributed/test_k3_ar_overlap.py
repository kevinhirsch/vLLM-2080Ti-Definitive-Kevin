# SPDX-License-Identifier: Apache-2.0
"""CPU tests for lane K3 (L78) GEMM/all-reduce overlap: chunking, gating, and the chunked data flow
(with CUDA stream/event/collective primitives faked; the real overlap needs the GPU window)."""
import contextlib
import types

import pytest
import torch

from vllm.distributed import k3_ar_overlap as ov


@pytest.mark.parametrize("n", [1, 7, 8, 9, 1024, 1816, 3584, 3632, 4097])
@pytest.mark.parametrize("k", [2, 3, 4])
def test_chunk_bounds_cover_exactly_and_aligned(n, k):
    b = ov.chunk_bounds(n, k)
    assert b[0][0] == 0 and b[-1][1] == n
    assert all(b[i][1] == b[i + 1][0] for i in range(len(b) - 1))
    assert all(x < y for x, y in b)
    assert all((y - x) % 8 == 0 for x, y in b[:-1])  # 16-byte multiples for fp16 custom AR
    assert len(b) <= k


def test_use_plain_gates(monkeypatch):
    monkeypatch.delenv("VLLM_K3_AR_OVERLAP_MIN_TOKENS", raising=False)
    assert ov.use_plain(1023, 2, False, 2)      # below min tokens (decode / graph sizes)
    assert not ov.use_plain(1024, 2, False, 2)
    assert ov.use_plain(3584, 2, True, 2)       # CUDA graph capture
    assert ov.use_plain(3584, 2, False, 3)      # non-2D input
    monkeypatch.setenv("VLLM_K3_AR_OVERLAP_MIN_TOKENS", "16")
    assert ov.use_plain(15, 2, False, 2) and not ov.use_plain(16, 2, False, 2)


def test_env_gate_default_off(monkeypatch):
    monkeypatch.delenv("VLLM_K3_AR_OVERLAP", raising=False)
    assert not ov.enabled()
    monkeypatch.setenv("VLLM_K3_AR_OVERLAP", "1")
    assert ov.enabled()


def test_chunked_dataflow_matches_plain(monkeypatch):
    """Chunked GEMM + per-chunk 'all-reduce' (2-rank sum emulated as x2) equals the whole-tensor result."""
    monkeypatch.setenv("VLLM_K3_AR_OVERLAP_MIN_TOKENS", "16")
    monkeypatch.setenv("VLLM_K3_AR_OVERLAP_CHUNKS", "3")
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, s: None, raising=False)
    fake_stream = types.SimpleNamespace(wait_event=lambda e: None)
    main = types.SimpleNamespace(wait_stream=lambda s: None)
    fake_event = type("E", (), {"record": lambda self, s: None})
    monkeypatch.setattr(torch.cuda, "Event", fake_event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: main)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    monkeypatch.setattr(ov, "_comm_stream", lambda dev: fake_stream)
    calls = []

    def fake_ar(part, out, stream):
        calls.append(tuple(part.shape))
        out.copy_(part * 2)

    monkeypatch.setattr(ov, "all_reduce_into", fake_ar)
    torch.manual_seed(0)
    W = torch.randn(32, 24)
    layer = types.SimpleNamespace(
        tp_rank=0, skip_bias_add=False, bias=None, output_size=24,
        quant_method=types.SimpleNamespace(apply=lambda l, x, b: x @ W))
    ov._LAYERS["t"] = lambda: layer
    x = torch.randn(100, 32)
    out = ov.k3_row_linear_ar_impl(x, "t")
    torch.testing.assert_close(out, (x @ W) * 2)
    assert sum(c[0] for c in calls) == 100 and len(calls) == 3
    # below threshold -> plain path is taken
    called = []
    monkeypatch.setattr(ov, "plain", lambda l, xx, b: called.append(xx.shape[0]) or xx)
    ov.k3_row_linear_ar_impl(x[:8], "t")
    assert called == [8]


def test_bench_parent_passes_one_shared_rdv(monkeypatch, tmp_path):
    """The 10-03 window failure: workers defaulted --rdv to their own pid, so ranks never met."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "k3_bench", __import__("os").path.join(__import__("os").path.dirname(ov.__file__), "../../tools/k3/bench_allreduce.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    argvs = []

    class P:
        def __init__(self, argv, **kw):
            argvs.append(argv)
            self.pid = 1

        def poll(self):
            return 0

    monkeypatch.setattr(mod.subprocess, "Popen", P)
    monkeypatch.setattr(sys, "argv", ["b", "--mode", "check", "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit):
        mod.main()
    rdvs = [a[a.index("--rdv") + 1] for a in argvs]
    assert len(rdvs) == 2 and rdvs[0] == rdvs[1]
    assert [a[a.index("--worker") + 1] for a in argvs] == ["0", "1"]
