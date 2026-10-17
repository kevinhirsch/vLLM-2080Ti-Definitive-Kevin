"""CPU test: Qwen4Exp's nm9 profiler scopes exist only while a torch profiler runs.

tools/qwen4exp/latency_map.py attributes generic kernels (cuBLAS/EXL3 GEMVs, copies, top-k gating) to the
``nm9:<component>`` scope around them. Outside a profiler the helper must be a no-op, so serving and CUDA-graph
capture never see a record_function.
"""
import contextlib

import torch

from vllm.models.qwen4_exp.nvidia import model as qmodel


def test_scope_is_noop_without_profiler():
    assert isinstance(qmodel._profile_scope("hyperconn"), contextlib.nullcontext)


def test_scope_records_under_profiler():
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        with qmodel._profile_scope("hyperconn"):
            torch.ones(4).add_(1)
    names = {e.name for e in prof.events()}
    assert "nm9:hyperconn" in names, sorted(names)[:20]


def test_decoder_layer_scopes_every_component():
    import inspect

    src = inspect.getsource(qmodel)
    for comp in ("hyperconn", "gdn", "qsa_attention", "moe_experts", "ple_ngram", "lm_head"):
        assert f'_profile_scope("{comp}")' in src, comp
