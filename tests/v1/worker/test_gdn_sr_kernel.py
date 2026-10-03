# SPDX-License-Identifier: Apache-2.0
"""Lane S4: fused_sigmoid_gating_delta_rule_update with VLLM_GDN_SR (GPU).
(a) SR off: deterministic and identical run-to-run; (b) SR on: every stored element is within one fp16 ulp of the fp32-state result, differs across
seeds, and the mean over many seeds converges on the fp32-state result much more tightly than a single round-to-nearest store."""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
T, N, HV, H, K, V = 4, 3, 4, 2, 128, 128


def _inputs(seed=0):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    ntok = N * T
    d = dict(
        q=torch.randn(1, ntok, H, K, device=dev, generator=g, dtype=torch.float16),
        k=torch.randn(1, ntok, H, K, device=dev, generator=g, dtype=torch.float16),
        v=torch.randn(1, ntok, HV, V, device=dev, generator=g, dtype=torch.float16),
        a=torch.randn(ntok, HV, device=dev, generator=g, dtype=torch.float16),
        b=torch.randn(ntok, HV, device=dev, generator=g, dtype=torch.float16),
        A_log=torch.randn(HV, device=dev, generator=g, dtype=torch.float32) * 0.1,
        dt_bias=torch.randn(HV, device=dev, generator=g, dtype=torch.float16),
        cu=torch.arange(0, ntok + 1, T, device=dev, dtype=torch.int32),
        idx=torch.arange(N * T, device=dev, dtype=torch.int32).view(N, T),
        nacc=torch.ones(N, device=dev, dtype=torch.int32),
        state16=(torch.randn(N * T + 2, HV, V, K, device=dev, generator=g) * 0.05).half(),
    )
    return d


def _run(d, state, sr: bool, seed_val: int = 0):
    from vllm.third_party.flash_linear_attention.ops import sr_convert
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import fused_sigmoid_gating_delta_rule_update

    os.environ["VLLM_GDN_SR"] = "1" if sr else "0"
    if sr:
        sr_convert.get_sr_seed(state.device).fill_(seed_val)
    st = state.clone()
    fused_sigmoid_gating_delta_rule_update(
        d["A_log"], d["a"], d["b"], d["dt_bias"], d["q"], d["k"], d["v"], initial_state=st, inplace_final_state=True, cu_seqlens=d["cu"],
        ssm_state_indices=d["idx"], num_accepted_tokens=d["nacc"], use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    return st[: N * T].float()


def test_sr_off_is_deterministic():
    d = _inputs()
    assert torch.equal(_run(d, d["state16"], False), _run(d, d["state16"], False))


def test_sr_on_neighbours_and_unbiased_vs_fp32_state():
    d = _inputs()
    ref32 = _run(d, d["state16"].float(), False)  # same fp16-representable initial state, exact fp32 stores
    rne = _run(d, d["state16"], False)
    reps = 96
    outs = torch.stack([_run(d, d["state16"], True, 1000 + r) for r in range(reps)])
    ulp = ref32.abs().clamp_min(2.0 ** -14) * 2.0 ** -10  # fp16 spacing (upper bound, within a binade)
    assert float(((outs - ref32).abs() / ulp).max()) <= 1.02  # every SR value is a neighbour of the fp32 value
    assert float((outs[0] != outs[1]).float().mean()) > 0.05  # randomness actually varies with the seed
    err_sr = ((outs.mean(0) - ref32).abs() / ulp).mean()
    err_rne = ((rne - ref32).abs() / ulp).mean()
    assert float(err_sr) < 0.35 * float(err_rne), (float(err_sr), float(err_rne))


def _run_packed(state, sr: bool, seed_val: int = 0):
    from vllm.third_party.flash_linear_attention.ops import sr_convert
    from vllm.third_party.flash_linear_attention.ops.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode

    os.environ["VLLM_GDN_SR"] = "1" if sr else "0"
    if sr:
        sr_convert.get_sr_seed(state.device).fill_(seed_val)
    dev = state.device
    g = torch.Generator(device=dev).manual_seed(5)
    B = 6
    mixed = torch.randn(B, 2 * H * K + HV * V, device=dev, generator=g, dtype=torch.float16)
    a = torch.randn(B, HV, device=dev, generator=g, dtype=torch.float16)
    b = torch.randn(B, HV, device=dev, generator=g, dtype=torch.float16)
    A_log = torch.randn(HV, device=dev, generator=g, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(HV, device=dev, generator=g, dtype=torch.float16)
    out = torch.empty(B, 1, HV, V, device=dev, dtype=torch.float16)
    st = state.clone()
    idx = torch.arange(B, device=dev, dtype=torch.int32)
    fused_recurrent_gated_delta_rule_packed_decode(mixed, a, b, A_log, dt_bias, K ** -0.5, st, out, idx, use_qk_l2norm_in_kernel=True)
    torch.cuda.synchronize()
    return st[:B].float()


def test_packed_decode_sr_neighbours_and_unbiased():
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(11)
    s16 = (torch.randn(8, HV, V, K, device=dev, generator=g) * 0.05).half()
    ref32 = _run_packed(s16.float(), False)
    rne = _run_packed(s16, False)
    outs = torch.stack([_run_packed(s16, True, 500 + r) for r in range(96)])
    ulp = ref32.abs().clamp_min(2.0 ** -14) * 2.0 ** -10
    assert float(((outs - ref32).abs() / ulp).max()) <= 1.02
    err_sr = ((outs.mean(0) - ref32).abs() / ulp).mean()
    err_rne = ((rne - ref32).abs() / ulp).mean()
    assert float(err_sr) < 0.35 * float(err_rne), (float(err_sr), float(err_rne))
