"""CPU tests for the opt-in UVA host-table PLE (VLLM_QWEN4_PLE_UVA=1).

The UVA gather kernel runs under TRITON_INTERPRET on CPU tensors (on the GPU the table is a CUDA view of
pinned host memory). Ids outside the table are clamped: at capture time and for padded slots they are not
real hashes. The MemAvailable guard refuses to pin a table that does not fit. With UVA on, the method picks
the UVA embedding and the config hook no longer forces PIECEWISE (the lookup is capturable GPU work). Every PLE
embedding class a method can select is instantiable (W2i: the UVA class missed the abstract start_prefetch and
the engine failed to build the model).
"""
import os
import types

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models import config as mcfg
from vllm.models.qwen4_exp.nvidia import model as _model  # noqa: F401  (imports ple_host without the cycle)
from vllm.models.qwen4_exp.nvidia import ple_host


def test_uva_gather_rows_clamped():
    torch.manual_seed(0)
    table = torch.randint(-30000, 30000, (100, 41), dtype=torch.int16)
    ids = torch.tensor([0, 7, 99, -5, 150, 42], dtype=torch.int64)
    out = torch.zeros(6, 41, dtype=torch.int16)
    ple_host.uva_gather_rows(table, ids, out)
    want = table[ids.clamp(0, 99)]
    assert torch.equal(out, want)


def test_mem_guard_refuses(monkeypatch, tmp_path):
    monkeypatch.setattr(ple_host, "_mem_available_bytes", lambda: 1 << 20)
    path = tmp_path / "t.bin"
    path.write_bytes(b"\0" * (4096 * 3))
    with pytest.raises(RuntimeError, match="VLLM_QWEN4_PLE_UVA"):
        ple_host.HostRegisteredTable(str(path), 0, 100_000, 82)  # 8.2 MB table, 1 MiB available


def test_method_and_config_follow_the_switch(monkeypatch, tmp_path):
    method = object.__new__(ple_host.Qwen4ExpPLEEXL3TableMethod)
    (tmp_path / "ngram_embedding.safetensors").write_bytes(b"")
    vc = types.SimpleNamespace(
        model_config=types.SimpleNamespace(quantization="exl3", model=str(tmp_path),
                                           hf_text_config=types.SimpleNamespace(ple_layer_ids=[2])),
        compilation_config=types.SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE))
    monkeypatch.setenv("VLLM_QWEN4_PLE_UVA", "0")
    assert method.embedding_cls is ple_host.Qwen4ExpPLEHostEmbedding
    monkeypatch.setenv("VLLM_QWEN4_PLE_UVA", "1")
    assert method.embedding_cls is ple_host.Qwen4ExpPLEUVAEmbedding
    mcfg._qwen4_exp_piecewise_for_host_ple(vc)
    assert vc.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE



def test_every_ple_embedding_class_is_concrete():
    import inspect

    from vllm.models.qwen4_exp.nvidia import ngram_embedding as NE

    base = NE.Qwen4ExpPLEEmbedding
    classes = {c for m in (ple_host, NE) for _, c in inspect.getmembers(m, inspect.isclass)
               if issubclass(c, base) and c is not base}
    assert {ple_host.Qwen4ExpPLEUVAEmbedding, ple_host.Qwen4ExpPLEHostEmbedding} <= classes
    for c in classes:
        assert not c.__abstractmethods__, (c.__name__, sorted(c.__abstractmethods__))


def test_uva_embedding_never_prefetches():
    cls = ple_host.Qwen4ExpPLEUVAEmbedding
    assert cls.supports_prefetch is False  # the n-gram layer then calls forward(ngram_ids), never start_prefetch
    assert cls.start_prefetch(object(), torch.empty(0), torch.empty(0)) is None
