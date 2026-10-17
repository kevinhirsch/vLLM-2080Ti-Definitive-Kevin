# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.warmup.qwen_triton_warmup import (
    _FLA_POST_CONV_WARMUP_LENGTHS,
    _QwenGDNWarmupConfig,
    _warm_causal_conv1d_fwd_kernel,
    _warm_fused_post_conv_kernel,
    _warm_gated_rms_norm_kernel,
)
from vllm.platforms import current_platform


def _cuda_gdn_config() -> _QwenGDNWarmupConfig:
    h, hv, k, v = 2, 2, 16, 16
    conv_kernel_size = 4
    conv_dim = 2 * h * k + hv * v
    device = torch.device("cuda")
    conv_state = torch.empty(
        (8, conv_dim, conv_kernel_size - 1),
        dtype=torch.bfloat16,
        device=device,
    )
    return _QwenGDNWarmupConfig(
        h=h,
        hv=hv,
        k=k,
        v=v,
        conv_kernel_size=conv_kernel_size,
        conv_state=conv_state,
        conv_dtype=conv_state.dtype,
        norm_weight_dtype=torch.bfloat16,
        norm_before_gate=True,
        norm_activation="silu",
        a_log=torch.zeros(hv, dtype=torch.float32, device=device),
        dt_bias=torch.zeros(hv, dtype=torch.float32, device=device),
        state_stride_token=hv * v * k,
        state_dtype=torch.float32,
    )


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA is required")
def test_qwen_gdn_prefill_warmup_kernels_compile_on_gpu() -> None:
    config = _cuda_gdn_config()
    device = torch.device("cuda")
    _warm_gated_rms_norm_kernel(
        device, config, max_num_tokens=16, x_dtype=config.conv_dtype
    )
    _warm_causal_conv1d_fwd_kernel(device, config)
    _warm_fused_post_conv_kernel(device, config)
    assert _FLA_POST_CONV_WARMUP_LENGTHS == (1, 2, 16)
    torch.accelerator.synchronize(device)


# ---- CPU tests: the warm-up compiles the variants serving actually uses ----------------------------------------------


def _cpu_gdn_config() -> _QwenGDNWarmupConfig:
    h, hv, k, v = 2, 2, 16, 16
    conv_kernel_size = 4
    conv_dim = 2 * h * k + hv * v
    conv_state = torch.empty((8, conv_dim, conv_kernel_size - 1), dtype=torch.float16)
    return _QwenGDNWarmupConfig(
        h=h, hv=hv, k=k, v=v, conv_kernel_size=conv_kernel_size, conv_state=conv_state,
        conv_dtype=conv_state.dtype, norm_weight_dtype=torch.float16, norm_before_gate=True,
        norm_activation="silu", a_log=torch.zeros(hv), dt_bias=torch.zeros(hv),
        state_stride_token=hv * v * k, state_dtype=torch.float16,
    )


def test_flash_next_model_types_are_warmed() -> None:
    """NM-9 W2p: qwen4_exp was missing, so no GDN kernel was warmed for Flash-Next."""
    from vllm.model_executor.warmup.qwen_triton_warmup import _QWEN_MODEL_TYPES

    assert {"qwen4_exp", "qwen4_exp_text"} <= _QWEN_MODEL_TYPES


def test_warmups_pass_the_null_block_id_serving_passes(monkeypatch) -> None:
    """null_block_id is a constexpr of both kernels: a warm-up with another value compiles a variant nothing uses."""
    import vllm.model_executor.layers.mamba.ops.causal_conv1d as conv_ops
    import vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating as fsg
    from vllm.model_executor.warmup.qwen_triton_warmup import (
        _warm_fused_sigmoid_gating_delta_rule_update_kernel,
    )
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID

    seen = {}
    monkeypatch.setattr(conv_ops, "causal_conv1d_fn", lambda *a, **kw: seen.setdefault("conv", kw))
    monkeypatch.setattr(fsg, "fused_sigmoid_gating_delta_rule_update", lambda **kw: seen.setdefault("update", kw))
    config = _cpu_gdn_config()
    _warm_causal_conv1d_fwd_kernel(torch.device("cpu"), config)
    _warm_fused_sigmoid_gating_delta_rule_update_kernel(torch.device("cpu"), config)
    assert seen["conv"]["null_block_id"] == PAD_SLOT_ID
    assert bool((seen["conv"]["cache_indices"] == PAD_SLOT_ID).all())  # the warm-up touches no real cache block
    assert seen["update"]["null_block_id"] == PAD_SLOT_ID
    assert bool((seen["update"]["ssm_state_indices"] == PAD_SLOT_ID).all())
    assert seen["update"]["initial_state"].stride(0) == config.state_stride_token  # the real layout's constexpr stride


def test_every_serving_call_passes_pad_slot_id() -> None:
    """The warm-up mirrors serving: if a serving call of either kernel ever passes another null_block_id, this fails
    and the warm-up must warm that value too."""
    import ast
    import inspect

    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn

    tree = ast.parse(inspect.getsource(gdn))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", ""))
             in ("causal_conv1d_fn", "fused_sigmoid_gating_delta_rule_update")]
    assert calls
    for c in calls:
        kw = {k.arg: k.value for k in c.keywords}
        assert "null_block_id" in kw and getattr(kw["null_block_id"], "id", None) == "PAD_SLOT_ID", ast.unparse(c)[:120]
