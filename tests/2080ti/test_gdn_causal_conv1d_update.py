# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
@pytest.mark.parametrize("dim", [4096, 5120], ids=["qwen36-35b", "qwen36-27b"])
@pytest.mark.parametrize("state_slot", [0, 13], ids=["slot0", "nonzero-slot"])
def test_gdn_single_token_causal_conv_update_matches_torch(
    dim: int, state_slot: int
) -> None:
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )
    from vllm.model_executor.layers.mamba.ops.cpu.causal_conv1d import (
        causal_conv1d_update_torch,
    )
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID

    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.float16
    width = 4
    state_len = width - 1
    num_cache_lines = 17
    weight = torch.randn(dim, width, device=device, dtype=dtype) * 0.1
    bias = torch.randn(dim, device=device, dtype=dtype) * 0.1
    conv_state = (
        torch.randn(
            num_cache_lines,
            dim,
            state_len,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    untouched_state = conv_state.clone()
    state_indices = torch.tensor([state_slot], device=device, dtype=torch.int32)
    reference_state = conv_state[state_slot : state_slot + 1].clone()

    for step in range(64):
        x = torch.randn(1, dim, device=device, dtype=dtype) * 0.1
        actual = causal_conv1d_update(
            x.clone(),
            conv_state,
            weight,
            bias,
            "silu",
            conv_state_indices=state_indices,
            null_block_id=PAD_SLOT_ID,
            validate_data=True,
        )
        expected = causal_conv1d_update_torch(
            x.unsqueeze(-1),
            reference_state,
            weight,
            bias,
            "silu",
        ).squeeze(-1)

        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-3,
            atol=1e-3,
            msg=lambda msg: f"decode step {step}: {msg}",
        )
        torch.testing.assert_close(
            conv_state[state_slot : state_slot + 1],
            reference_state,
            rtol=0,
            atol=0,
            msg=lambda msg: f"conv state after decode step {step}: {msg}",
        )

    torch.testing.assert_close(
        conv_state[:state_slot], untouched_state[:state_slot], rtol=0, atol=0
    )
    torch.testing.assert_close(
        conv_state[state_slot + 1 :],
        untouched_state[state_slot + 1 :],
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
def test_gdn_recurrent_decode_accepts_state_slot_zero() -> None:
    from vllm.model_executor.layers.fla.ops import (
        fused_recurrent_gated_delta_rule_packed_decode,
        fused_sigmoid_gating_delta_rule_update,
    )
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID

    torch.manual_seed(11)
    device = torch.device("cuda")
    dtype = torch.float16
    heads = value_heads = 1
    key_dim = value_dim = 16

    q = torch.randn(heads, key_dim, device=device, dtype=dtype) * 0.1
    k = torch.randn(heads, key_dim, device=device, dtype=dtype) * 0.1
    v = torch.randn(value_heads, value_dim, device=device, dtype=dtype) * 0.1
    a = torch.randn(1, value_heads, device=device, dtype=dtype) * 0.1
    b = torch.randn(1, value_heads, device=device, dtype=dtype) * 0.1
    A_log = torch.randn(value_heads, device=device, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(value_heads, device=device, dtype=dtype) * 0.1
    state_indices = torch.tensor([0], device=device, dtype=torch.int32)
    initial_state = (
        torch.randn(
            2,
            value_heads,
            value_dim,
            key_dim,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    untouched_state = initial_state[1].clone()

    q_ref = torch.nn.functional.normalize(q.float(), dim=-1)
    k_ref = torch.nn.functional.normalize(k.float(), dim=-1)
    h_ref = initial_state[0].float()
    gate = -torch.exp(A_log.float()) * torch.nn.functional.softplus(
        a[0].float() + dt_bias.float()
    )
    h_ref = h_ref * torch.exp(gate)[:, None, None]
    v_ref = v.float() - (h_ref * k_ref[:, None, :]).sum(dim=-1)
    v_ref = v_ref * torch.sigmoid(b[0].float())[:, None]
    h_ref = h_ref + v_ref[:, :, None] * k_ref[:, None, :]
    out_ref = (h_ref * q_ref[:, None, :]).sum(dim=-1) * key_dim**-0.5

    packed_state = initial_state.clone()
    packed_out = torch.zeros(
        1, 1, value_heads, value_dim, device=device, dtype=dtype
    )
    mixed_qkv = torch.cat([q.flatten(), k.flatten(), v.flatten()]).unsqueeze(0)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=key_dim**-0.5,
        initial_state=packed_state,
        out=packed_out,
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
        null_block_id=PAD_SLOT_ID,
    )

    torch.testing.assert_close(
        packed_out[0, 0], out_ref, rtol=2e-3, atol=2e-3, check_dtype=False
    )
    torch.testing.assert_close(
        packed_state[0], h_ref, rtol=2e-3, atol=2e-3, check_dtype=False
    )
    torch.testing.assert_close(packed_state[1], untouched_state, rtol=0, atol=0)

    recurrent_state = initial_state.clone()
    recurrent_out, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q.reshape(1, 1, heads, key_dim),
        k=k.reshape(1, 1, heads, key_dim),
        v=v.reshape(1, 1, value_heads, value_dim),
        initial_state=recurrent_state,
        inplace_final_state=True,
        cu_seqlens=torch.tensor([0, 1], device=device, dtype=torch.int32),
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
        null_block_id=PAD_SLOT_ID,
    )

    torch.testing.assert_close(
        recurrent_out[0, 0], out_ref, rtol=2e-3, atol=2e-3, check_dtype=False
    )
    torch.testing.assert_close(
        recurrent_state[0], h_ref, rtol=2e-3, atol=2e-3, check_dtype=False
    )
    torch.testing.assert_close(recurrent_state[1], untouched_state, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
def test_gdn_padding_sentinel_skips_only_padded_state() -> None:
    from vllm.model_executor.layers.fla.ops import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )
    from vllm.model_executor.layers.mamba.ops.cpu.causal_conv1d import (
        causal_conv1d_update_torch,
    )
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID

    torch.manual_seed(19)
    device = torch.device("cuda")
    dtype = torch.float16
    dim = 64
    width = 4
    state_len = width - 1
    weight = torch.randn(dim, width, device=device, dtype=dtype) * 0.1
    bias = torch.randn(dim, device=device, dtype=dtype) * 0.1
    conv_state = torch.randn(2, dim, state_len, device=device, dtype=dtype) * 0.1
    untouched_conv_state = conv_state.clone()
    state_indices = torch.tensor([0, PAD_SLOT_ID], device=device, dtype=torch.int32)
    x = torch.randn(2, dim, device=device, dtype=dtype) * 0.1
    reference_state = conv_state[:1].clone()
    expected = causal_conv1d_update_torch(
        x[:1].unsqueeze(-1), reference_state, weight, bias, "silu"
    ).squeeze(-1)
    actual = causal_conv1d_update(
        x.clone(),
        conv_state,
        weight,
        bias,
        "silu",
        conv_state_indices=state_indices,
        null_block_id=PAD_SLOT_ID,
        validate_data=True,
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actual[1], x[1], rtol=0, atol=0)
    torch.testing.assert_close(conv_state[0], reference_state[0], rtol=0, atol=0)
    torch.testing.assert_close(conv_state[1], untouched_conv_state[1], rtol=0, atol=0)

    heads = value_heads = 1
    key_dim = value_dim = 16
    mixed_qkv = torch.randn(
        2, 2 * heads * key_dim + value_heads * value_dim, device=device, dtype=dtype
    ) * 0.1
    a = torch.randn(2, value_heads, device=device, dtype=dtype) * 0.1
    b = torch.randn(2, value_heads, device=device, dtype=dtype) * 0.1
    A_log = torch.randn(value_heads, device=device, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(value_heads, device=device, dtype=dtype) * 0.1
    recurrent_state = torch.randn(
        2, value_heads, value_dim, key_dim, device=device, dtype=dtype
    ) * 0.1
    untouched_recurrent_state = recurrent_state[1].clone()
    recurrent_out = torch.zeros(
        2, 1, value_heads, value_dim, device=device, dtype=dtype
    )
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=key_dim**-0.5,
        initial_state=recurrent_state,
        out=recurrent_out,
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
        null_block_id=PAD_SLOT_ID,
    )

    torch.testing.assert_close(
        recurrent_state[1], untouched_recurrent_state, rtol=0, atol=0
    )
    torch.testing.assert_close(
        recurrent_out[1], torch.zeros_like(recurrent_out[1]), rtol=0, atol=0
    )
