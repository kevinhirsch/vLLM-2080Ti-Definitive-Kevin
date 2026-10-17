"""CPU test: a Qwen4Exp checkpoint with a host-table PLE never gets FULL CUDA graphs.

The host row gather runs as eager breaks between breakable PIECEWISE segments; inside a FULL graph
the gather thread waits on a captured event (907 CUDA_ERROR_CAPTURED_EVENT, NM-9 W2e). So the config
hook drops FULL / FULL_DECODE_ONLY / FULL_AND_PIECEWISE (and the unset default) to PIECEWISE, and
leaves NONE, PIECEWISE and device-resident tables alone.
"""
import types

import pytest

from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models import config as mcfg


def _cfg(tmp_path, quant="exl3", mode=CUDAGraphMode.FULL_AND_PIECEWISE, ple=(2,), table=True):
    if table:
        (tmp_path / "ngram_embedding.safetensors").write_bytes(b"")
    model_config = types.SimpleNamespace(quantization=quant, model=str(tmp_path),
                                         hf_text_config=types.SimpleNamespace(ple_layer_ids=list(ple)))
    return types.SimpleNamespace(model_config=model_config,
                                 compilation_config=types.SimpleNamespace(cudagraph_mode=mode))


@pytest.mark.parametrize("mode", [None, CUDAGraphMode.FULL, CUDAGraphMode.FULL_DECODE_ONLY,
                                  CUDAGraphMode.FULL_AND_PIECEWISE])
def test_exl3_host_table_forces_piecewise(tmp_path, mode):
    vc = _cfg(tmp_path, mode=mode)
    mcfg._qwen4_exp_piecewise_for_host_ple(vc)
    assert vc.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE])
def test_eager_and_piecewise_kept(tmp_path, mode):
    vc = _cfg(tmp_path, mode=mode)
    mcfg._qwen4_exp_piecewise_for_host_ple(vc)
    assert vc.compilation_config.cudagraph_mode == mode


def test_device_table_untouched(tmp_path):
    no_file = tmp_path / "no_table"
    no_file.mkdir()
    cases = (_cfg(no_file, table=False),       # EXL3 pack without a streamed table
             _cfg(tmp_path, quant=None),       # unquantized: device/pinned table
             _cfg(tmp_path, ple=()))           # no PLE layers
    for vc in cases:
        mcfg._qwen4_exp_piecewise_for_host_ple(vc)
        assert vc.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE


def test_gguf_host_table_forces_piecewise(tmp_path):
    vc = _cfg(tmp_path, quant="gguf", table=False)
    mcfg._qwen4_exp_piecewise_for_host_ple(vc)
    assert vc.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE
