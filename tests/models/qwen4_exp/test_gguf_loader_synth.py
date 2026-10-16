# SPDX-License-Identifier: Apache-2.0
"""End-to-end CPU test of the qwen4exp GGUF loader on a synthetic tiny GGUF (real quant types).

Builds the real vLLM Qwen4Exp model (CPU tensors, TP1) from a config generated from the GGUF header, loads it
with ``load_qwen4exp_gguf`` and checks: every parameter is written, quantized parameters hold the exact GGUF bytes
(V-head rows reordered where llama.cpp reordered them), float parameters hold the inverse-transformed values.
"""

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from tests.models.qwen4_exp import gguf_numpy_ref as ref
from tests.models.qwen4_exp import gguf_synth as synth


def _cpu_build(model_dir):
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import set_default_torch_dtype

    vcfg = EngineArgs(model=model_dir, dtype="float16", load_format="gguf", enforce_eager=True,
                      skip_tokenizer_init=True, max_model_len=1024, max_num_batched_tokens=1024,
                      mamba_cache_mode="align", engram_config={"cpu_offload": False}).create_engine_config()
    type(current_platform).current_device = classmethod(lambda cls: torch.device("cpu"))

    class _NoStream:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, name):
            return lambda *a, **k: None

    torch.cuda.Stream = _NoStream
    torch.cuda.Event = _NoStream
    with set_current_vllm_config(vcfg):
        init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                     distributed_init_method=f"file://{tempfile.mktemp()}", backend="gloo")
        initialize_model_parallel(1, 1)
        with set_default_torch_dtype(torch.float16), torch.device("cpu"):
            model = initialize_model(vllm_config=vcfg)
    return vcfg, model


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    from vllm.model_executor.layers.quantization import gguf as gguf_q
    from vllm.model_executor.model_loader.gguf_loader import load_qwen4exp_gguf
    from vllm.model_executor.model_loader.gguf_qwen4exp_config import qwen4exp_text_config
    from vllm.model_executor.model_loader.gguf_reader import GGUFModel

    d = tmp_path_factory.mktemp("tinygguf")
    res = synth.write_tiny_qwen4exp(str(d / "tiny.gguf"), seed=3, shard=True)
    first = sorted(res["paths"])[0]
    gm = GGUFModel(first)
    # the synthetic table is tiny (16 heads x 256 rows), not vLLM's 20M-prime hash layout
    from vllm.model_executor.model_loader import gguf_qwen4exp_config as C

    saved = (C._NGRAM_VOCAB_SIZE_BASE, C._next_prime_after)
    C._NGRAM_VOCAB_SIZE_BASE, C._next_prime_after = 256, (lambda n: n + 1)
    try:
        cfg = qwen4exp_text_config(gm)
    finally:
        C._NGRAM_VOCAB_SIZE_BASE, C._next_prime_after = saved
    cfg.update({"architectures": ["Qwen4ExpForCausalLM"], "model_type": "qwen4_exp_text", "torch_dtype": "float16",
                "quantization_config": {"quant_method": "gguf", "gguf_file": first}})
    (d / "config.json").write_text(json.dumps(cfg))
    orig = gguf_q.GGUFConfig.get_qwen4exp_ple_method
    gguf_q.GGUFConfig.get_qwen4exp_ple_method = lambda self, prefix: None  # host-gather PLE needs CUDA
    try:
        vcfg, model = _cpu_build(str(d))
        stats = load_qwen4exp_gguf(model, vcfg.model_config, first)
    finally:
        gguf_q.GGUFConfig.get_qwen4exp_ple_method = orig
    return gm, model, stats, cfg


def test_every_parameter_written(loaded):
    gm, model, stats, _ = loaded
    assert stats["quant"] > 0 and stats["float"] > 0


def test_quantized_bytes_exact(loaded):
    gm, model, _, cfg = loaded
    mods = dict(model.named_modules())
    # full-attention qkv: q shard segment holds attn_q bytes exactly (TP1)
    full = cfg["layer_types"].index("full_attention")
    layer = mods[f"model.layers.{full}.self_attn.qkv_proj"]
    q = np.asarray(gm.raw(f"blk.{full}.attn_q.weight"))
    seg0 = layer.gguf_segments[0]
    assert np.array_equal(getattr(layer, seg0[0]).numpy()[: seg0[3]].reshape(-1)[: q.size], q)
    # routed experts: w13 = gate rows then up rows of each expert; w2 = down
    ex = mods["model.layers.0.mlp.experts.routed_experts"]
    gate = np.asarray(gm.raw("blk.0.ffn_gate_exps.weight")).reshape(ex.w13_qweight.shape[0], -1)
    inter = ex.w13_qweight.shape[1] // 2
    assert np.array_equal(ex.w13_qweight[:, :inter].numpy().reshape(gate.shape), gate)
    down = np.asarray(gm.raw("blk.0.ffn_down_exps.weight"))
    assert np.array_equal(ex.w2_qweight.numpy().reshape(-1), down)


def test_float_transforms(loaded):
    gm, model, _, cfg = loaded
    params = dict(model.named_parameters())
    lin = cfg["layer_types"].index("linear_attention")
    nk, nv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    ssm_a = np.asarray(gm.array(f"blk.{lin}.ssm_a"), dtype=np.float32)
    idx = (np.arange(nv) % (nv // nk)) * nk + np.arange(nv) // (nv // nk)
    want = np.log(-ssm_a)[idx]
    got = params[f"model.layers.{lin}.linear_attn.A_log"].detach().float().numpy()
    assert np.allclose(got, want, atol=1e-5)
    norm = np.asarray(gm.array("output_hc_norm.weight"), dtype=np.float32)
    got = params["model.hyper_connection_mixer.hc_norm.weight"].detach().float().numpy()
    assert np.allclose(got, norm - 1.0, atol=2e-3)
    # GDN out_proj: dequantized on the host, input columns back in grouped V-head order
    w = ref.dequant(np.asarray(gm.raw(f"blk.{lin}.ssm_out.weight")).tobytes(),
                    gm.tensors[f"blk.{lin}.ssm_out.weight"].type_name, gm.tensors[f"blk.{lin}.ssm_out.weight"].dims)
    hd = cfg["linear_value_head_dim"]
    w = w.reshape(w.shape[0], nv, hd)[:, idx].reshape(w.shape[0], -1)
    got = params[f"model.layers.{lin}.linear_attn.out_proj.weight"].detach().float().numpy()
    assert np.allclose(got, w, atol=1e-2, rtol=1e-2)
