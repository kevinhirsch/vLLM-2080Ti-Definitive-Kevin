# SPDX-License-Identifier: Apache-2.0
"""[FORK][LANE K1] TurboQuant continuation prefix-combine: FlashInfer returns a base-2 LSE, merge_attn_states needs
natural log. The merged output must match an fp32 softmax reference at 20K+ context (the 'auto' threshold)."""

import math

import pytest
import torch

flashinfer = pytest.importorskip("flashinfer")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (7, 5),
    reason="needs an sm_75+ CUDA device",
)


def _ref(q, k, v, scale, causal):
    """fp32 reference per head in row chunks. q [T,Hq,D], k/v [S,Hk,D] -> out [T,Hq,D] fp32, lse [T,Hq] (ln)."""
    T, Hq, D = q.shape
    S, Hk, _ = k.shape
    g = Hq // Hk
    out = torch.empty(T, Hq, D, dtype=torch.float32, device=q.device)
    lse = torch.empty(T, Hq, dtype=torch.float32, device=q.device)
    cols = torch.arange(S, device=q.device)[None, :]
    rc = max(1, (4 << 20) // (4 * S))
    for h in range(Hq):
        kh, vh = k[:, h // g].float(), v[:, h // g].float()
        for r0 in range(0, T, rc):
            r1 = min(T, r0 + rc)
            s = (q[r0:r1, h].float() @ kh.T) * scale
            if causal:
                rows = torch.arange(r0, r1, device=q.device)[:, None] + (S - T)
                s.masked_fill_(cols > rows, float("-inf"))
            lse[r0:r1, h] = torch.logsumexp(s, -1)
            out[r0:r1, h] = torch.softmax(s, -1) @ vh
    return out, lse


_KEEP = []  # FlashInfer wrappers own their int workspace: keep them alive past their async run


def _fi(ws, q, k, v, scale, causal):
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="fa2")
    w.plan(
        torch.tensor([0, q.shape[0]], dtype=torch.int32),
        torch.tensor([0, k.shape[0]], dtype=torch.int32),
        q.shape[1],
        k.shape[1],
        q.shape[2],
        causal=causal,
        sm_scale=scale,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    _KEEP.append(w)
    return w.run(q, k, v, return_lse=True)


@pytest.mark.parametrize("cached, q_len, q_mult", [(20480, 512, 1.0), (20480, 512, 4.0), (24576, 384, 8.0)])
def test_prefix_combine_merge_matches_fp32_reference(cached, q_len, q_mult):
    from vllm.v1.attention.backends.turboquant_attn import _tq_merge_flashinfer_partials

    dev = "cuda"
    D, Hq, Hk = 256, 6, 1
    scale = D**-0.5
    g = torch.Generator(device=dev).manual_seed(cached + q_len)
    q = (torch.randn(q_len, Hq, D, device=dev, generator=g) * q_mult).half()
    k = torch.randn(cached + q_len, Hk, D, device=dev, generator=g).half()
    v = torch.randn(cached + q_len, Hk, D, device=dev, generator=g).half()
    ref, ref_lse = _ref(q, k, v, scale, True)

    ws = torch.empty(64 << 20, dtype=torch.uint8, device=dev)
    po, pl = _fi(ws, q, k[:cached], v[:cached], scale, False)  # prefix: whole cache, non-causal
    so, sl = _fi(ws, q, k[cached:], v[cached:], scale, True)  # current chunk: causal

    # FlashInfer's LSE is base 2: x ln2 equals the natural-log reference
    _, ref_pl = _ref(q, k[:cached], v[:cached], scale, False)
    assert (pl.float() * math.log(2) - ref_pl).abs().max().item() < 1e-3

    def relerr(x):
        return ((x.float() - ref).norm() / ref.norm()).item()

    fixed = _tq_merge_flashinfer_partials(po, pl.clone(), so, sl.clone(), lse_base2=True)
    old = _tq_merge_flashinfer_partials(po, pl.clone(), so, sl.clone(), lse_base2=False)
    assert relerr(fixed) < 1e-3, relerr(fixed)
    # the unconverted merge (pre-fix behaviour) is measurably wrong, so this test would have caught it
    assert relerr(old) > 1e-2, relerr(old)
    # default (env unset) is the fixed behaviour
    assert relerr(_tq_merge_flashinfer_partials(po, pl.clone(), so, sl.clone())) < 1e-3
    torch.cuda.synchronize()
    _KEEP.clear()
