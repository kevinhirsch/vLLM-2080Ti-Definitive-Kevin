# SPDX-License-Identifier: Apache-2.0
"""Fail-closed behaviour of the GDN/conv spec-decode kernels for bad accepted counts.

Port of the kernel-side regressions from vllm-project/vllm#50021 (lane WR).
A stale / zero / too-large ``num_accepted_tokens`` used to be turned into an
unbounded array index inside the kernel and dereferenced (Xid 13 / Xid 31 on
SM75 under hybrid GDN + MTP + prefix caching). The kernels must now treat such
a row as invalid: zero output, state untouched.

Runs on GPU when one is free, otherwise on CPU under the Triton interpreter
(``TRITON_INTERPRET=1``, set below before triton is imported). Set
``WR_TEST_DEVICE=cpu`` to force the interpreter even when CUDA is visible (the
live 2080 Ti engine owns the GPUs).
"""

import os

if os.environ.get("WR_TEST_DEVICE", "").lower() == "cpu" or not (
    os.environ.get("WR_TEST_DEVICE", "").lower() == "cuda"
):
    os.environ.setdefault("TRITON_INTERPRET", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule,
    fused_sigmoid_gating_delta_rule_update,
)

DEVICE = "cuda" if os.environ.get("WR_TEST_DEVICE", "").lower() == "cuda" else "cpu"


def _gdn_inputs(num_tokens=4, hk=2, hv=4, d=16):
    g = torch.Generator(device="cpu").manual_seed(0)
    r = lambda *s: torch.rand(*s, generator=g).to(DEVICE)  # noqa: E731
    q = r(1, num_tokens, hk, d)
    k = r(1, num_tokens, hk, d)
    v = r(1, num_tokens, hv, d)
    a = r(num_tokens, hv)
    b = r(num_tokens, hv)
    A_log = r(hv)
    dt_bias = r(hv)
    state = r(num_tokens + 1, hv, d, d)
    idx = torch.arange(1, num_tokens + 1, dtype=torch.int32).view(1, num_tokens)
    cu = torch.tensor([0, num_tokens], dtype=torch.int32)
    return q, k, v, a, b, A_log, dt_bias, state, idx.to(DEVICE), cu.to(DEVICE), r


def _run_recurrent(accepted: int):
    q, k, v, a, b, A_log, dt_bias, state, idx, cu, r = _gdn_inputs()
    s = state.clone()
    out, s = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=r(1, q.shape[1], v.shape[2]),
        beta=r(1, q.shape[1], v.shape[2]),
        initial_state=s,
        inplace_final_state=True,
        ssm_state_indices=idx,
        cu_seqlens=cu,
        num_accepted_tokens=torch.tensor([accepted], dtype=torch.int32).to(DEVICE),
    )
    return out, s, state


def _run_sigmoid(accepted: int):
    q, k, v, a, b, A_log, dt_bias, state, idx, cu, _ = _gdn_inputs()
    s = state.clone()
    out, s = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=s,
        inplace_final_state=True,
        ssm_state_indices=idx,
        cu_seqlens=cu,
        num_accepted_tokens=torch.tensor([accepted], dtype=torch.int32).to(DEVICE),
    )
    return out, s, state


@pytest.mark.parametrize("runner", [_run_recurrent, _run_sigmoid])
@pytest.mark.parametrize("accepted", [0, 5, 1000])
def test_gdn_invalid_accepted_count_fails_closed(runner, accepted):
    out, state, state0 = runner(accepted)
    torch.testing.assert_close(out, torch.zeros_like(out))
    torch.testing.assert_close(state, state0)


@pytest.mark.skipif(
    DEVICE != "cuda",
    reason="valid-path GDN math uses tl helpers the Triton interpreter cannot run",
)
@pytest.mark.parametrize("runner", [_run_recurrent, _run_sigmoid])
@pytest.mark.parametrize("accepted", [1, 2, 4])
def test_gdn_valid_accepted_count_still_computes(runner, accepted):
    out, state, state0 = runner(accepted)
    assert torch.count_nonzero(out) > 0
    assert not torch.equal(state, state0)
    assert torch.isfinite(out).all()


def _conv(accepted: int, seqlen=3, width=4, dim=64, batch=1):
    g = torch.Generator(device="cpu").manual_seed(1)
    r = lambda *s: torch.randn(*s, generator=g).to(DEVICE)  # noqa: E731
    x = r(batch, dim, seqlen)
    weight = r(dim, width)
    conv_state = r(3, dim, width - 1 + seqlen - 1)
    before = conv_state.clone()
    idx = torch.tensor([1], dtype=torch.int32).to(DEVICE)
    acc = torch.tensor([accepted], dtype=torch.int32).to(DEVICE)
    out = torch.full_like(x, torch.nan)
    res = causal_conv1d_update(
        x, conv_state, weight, conv_state_indices=idx, num_accepted_tokens=acc, out=out
    )
    return res, conv_state, before


@pytest.mark.parametrize("accepted", [0, 4, 1000])
def test_causal_conv1d_update_invalid_accepted_count_fails_closed(accepted):
    res, state, before = _conv(accepted)
    torch.testing.assert_close(res, torch.zeros_like(res))
    torch.testing.assert_close(state, before)


@pytest.mark.parametrize("accepted", [1, 2, 3])
def test_causal_conv1d_update_valid_accepted_count_updates(accepted):
    res, state, before = _conv(accepted)
    assert torch.isfinite(res).all()
    assert not torch.equal(state, before)
