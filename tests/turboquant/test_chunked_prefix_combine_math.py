# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RANK-4 backlog ("chunked continuation-dequant, +93K pool tokens"):
pure-torch, CPU-only proof that chunking the cached-prefix attention and
folding the partial results together with repeated log-sum-exp (LSE)
merges reproduces the same attention output as the existing single-shot
("unchunked") computation, up to floating-point summation-order rounding.

This validates the *algorithm* used by
``TurboQuantAttentionImpl._continuation_prefix_combine_chunked`` in
vllm/v1/attention/backends/turboquant_attn.py, independent of the actual
GPU kernels (triton dequant, flashinfer prefill, the triton
merge_attn_states kernel) it calls in production -- none of those are
importable/runnable without a CUDA device, so this test reimplements the
same two formulas in plain torch (fp32, CPU):

  * naive numerically-stable softmax attention (stands in for flashinfer's
    ``BatchPrefillWithRaggedKVCacheWrapper.run(..., return_lse=True)``)
  * the LSE-rescale merge (mirrors
    vllm/v1/attention/ops/triton_merge_attn_states.py:57-175's
    ``merge_attn_states_kernel``, dispatched from
    vllm/v1/attention/ops/merge_attn_states.py:9 -- "Implements section 2.2
    of https://www.arxiv.org/pdf/2501.01005")

and checks three things against random small tensors:

  1. the EXISTING (unchunked) prefix-combine algorithm (prefix attention
     non-causal + current-chunk attention causal + one merge) equals a
     ground-truth single softmax over the whole causally-masked
     concatenation -- i.e. this test's model of "what production already
     does" is itself correct before trusting the chunked variant against
     it;
  2. the NEW chunked algorithm (fold N chunk-partials via repeated merges,
     then merge with the current chunk) equals that same ground truth;
  3. the NEW chunked algorithm equals the EXISTING unchunked algorithm
     directly, at the tightest tolerance.

Chunk boundaries are produced by the real
``_tq_chunked_prefix_plan`` (the function
``_continuation_prefix_combine_chunked`` actually calls), so this is an
end-to-end proof of the exact partition the production code will use, not
a hand-picked one.
"""

import math

import pytest
import torch

from vllm.v1.attention.backends.turboquant_attn import _tq_chunked_prefix_plan

pytestmark = pytest.mark.cpu_test

torch.manual_seed(0)


def _merge(prefix_out, prefix_lse, suffix_out, suffix_lse):
    """Pure-torch reimplementation of merge_attn_states_kernel's math.

    Shapes here use flashinfer's native token-major convention, (T, H, D)
    for outputs and (T, H) for LSE, rather than the Triton kernel's
    (H, T)-contiguous layout the real GPU call site needs -- the merge is
    purely elementwise over (T, H), so the axis order used for this
    reference doesn't change the result, only the real kernel's raw
    pointer arithmetic cares about it.
    """
    max_lse = torch.maximum(prefix_lse, suffix_lse)
    p = torch.exp(prefix_lse - max_lse)
    s = torch.exp(suffix_lse - max_lse)
    denom = p + s
    out_lse = torch.log(denom) + max_lse
    p_scale = (p / denom).unsqueeze(-1)
    s_scale = (s / denom).unsqueeze(-1)
    out = prefix_out * p_scale + suffix_out * s_scale
    return out, out_lse


def _attn(q, k, v, scale, causal_offset):
    """Naive numerically-stable softmax attention, fp32, CPU.

    q: (Tq, H, D); k, v: (Tk, H, D).
    causal_offset=None -> fully unmasked: every query attends to every key
    (the "cached prefix" case -- every cached position strictly precedes
    every new query position, so nothing is masked).
    causal_offset=c -> query i may attend to key j iff j <= c + i (the
    "current new-token chunk" case uses c=0 for local causal masking; the
    "ground truth" full computation uses c=cached_len, i.e. each new
    query's true absolute position).
    Returns (out (Tq,H,D), lse (Tq,H)) -- flashinfer's native convention.
    """
    scores = torch.einsum("qhd,khd->hqk", q, k) * scale
    if causal_offset is not None:
        Tq, Tk = q.shape[0], k.shape[0]
        qi = torch.arange(Tq).view(1, Tq, 1)
        kj = torch.arange(Tk).view(1, 1, Tk)
        mask = kj <= (causal_offset + qi)
        scores = scores.masked_fill(~mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)  # (H, Tq)
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hqk,khd->qhd", probs, v)
    return out, lse.transpose(0, 1)  # (Tq,H,D), (Tq,H)


def _reference_full(q, k_prefix, v_prefix, k_cur, v_cur, scale, cached_len):
    """Ground truth: one softmax over the whole concatenated K/V under a
    global causal mask. This is what continuation-prefill is DEFINED to
    compute, independent of which internal algorithm produces it."""
    k_full = torch.cat([k_prefix, k_cur], dim=0)
    v_full = torch.cat([v_prefix, v_cur], dim=0)
    out, _ = _attn(q, k_full, v_full, scale, causal_offset=cached_len)
    return out


def _prefix_combine_unchunked(q, k_prefix, v_prefix, k_cur, v_cur, scale):
    """Mirrors the EXISTING unchunked prefix-combine branch in
    _continuation_prefill (turboquant_attn.py): whole-prefix attention
    (non-causal) + current-chunk attention (causal) + one merge_attn_states
    call."""
    prefix_out, prefix_lse = _attn(q, k_prefix, v_prefix, scale, causal_offset=None)
    cur_out, cur_lse = _attn(q, k_cur, v_cur, scale, causal_offset=0)
    out, _ = _merge(prefix_out, prefix_lse, cur_out, cur_lse)
    return out


def _prefix_combine_chunked(q, k_prefix, v_prefix, k_cur, v_cur, scale, plan):
    """Mirrors the NEW _continuation_prefix_combine_chunked algorithm: fold
    chunk-partial (out, lse) pairs sequentially via the same merge, then
    merge the accumulated prefix result with the current chunk. `plan`
    comes from the real _tq_chunked_prefix_plan."""
    acc_out = acc_lse = None
    for chunk_start_tok, chunk_len, _alloc_len, _start_page, _pages_needed in plan:
        k_c = k_prefix[chunk_start_tok : chunk_start_tok + chunk_len]
        v_c = v_prefix[chunk_start_tok : chunk_start_tok + chunk_len]
        c_out, c_lse = _attn(q, k_c, v_c, scale, causal_offset=None)
        if acc_out is None:
            acc_out, acc_lse = c_out, c_lse
        else:
            acc_out, acc_lse = _merge(acc_out, acc_lse, c_out, c_lse)
    cur_out, cur_lse = _attn(q, k_cur, v_cur, scale, causal_offset=0)
    out, _ = _merge(acc_out, acc_lse, cur_out, cur_lse)
    return out


def _make_tensors(cached_len, q_len, H, D, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(q_len, H, D, generator=g, dtype=torch.float32)
    k_prefix = torch.randn(cached_len, H, D, generator=g, dtype=torch.float32)
    v_prefix = torch.randn(cached_len, H, D, generator=g, dtype=torch.float32)
    k_cur = torch.randn(q_len, H, D, generator=g, dtype=torch.float32)
    v_cur = torch.randn(q_len, H, D, generator=g, dtype=torch.float32)
    return q, k_prefix, v_prefix, k_cur, v_cur


CASES = [
    # (cached_len, q_len, block_size, chunk_tokens, H, D)
    (17, 3, 16, 8, 2, 8),
    (63, 4, 16, 16, 2, 8),
    (64, 4, 16, 16, 2, 8),
    (65, 4, 16, 16, 2, 8),
    (200, 5, 16, 32, 4, 8),
    (200, 5, 16, 200, 4, 8),  # chunk == whole prefix (1 chunk, degenerate)
    (200, 5, 16, 100_000, 4, 8),  # chunk >> whole prefix (1 chunk)
    (3567, 6, 16, 512, 2, 16),
    (3568, 6, 16, 512, 2, 16),
    (3569, 6, 16, 512, 2, 16),
    (3568, 6, 16, 16384, 2, 16),  # chunk >> prefix, still block-boundary case
    (41498, 4, 16, 16384, 2, 8),
    (85976, 4, 16, 16384, 2, 8),
]


@pytest.mark.parametrize(
    "cached_len,q_len,block_size,chunk_tokens,H,D", CASES
)
def test_unchunked_prefix_combine_matches_reference(
    cached_len, q_len, block_size, chunk_tokens, H, D
):
    """Sanity baseline: this test file's model of the EXISTING production
    algorithm must itself match ground truth, or the chunked-vs-unchunked
    comparison below would be meaningless."""
    scale = 1.0 / math.sqrt(D)
    q, k_prefix, v_prefix, k_cur, v_cur = _make_tensors(
        cached_len, q_len, H, D, seed=cached_len * 1000 + q_len
    )
    got = _prefix_combine_unchunked(q, k_prefix, v_prefix, k_cur, v_cur, scale)
    ref = _reference_full(q, k_prefix, v_prefix, k_cur, v_cur, scale, cached_len)
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "cached_len,q_len,block_size,chunk_tokens,H,D", CASES
)
def test_chunked_prefix_combine_matches_reference(
    cached_len, q_len, block_size, chunk_tokens, H, D
):
    """The claim: chunking the prefix and folding partials via repeated
    LSE merges reproduces the single-softmax ground truth."""
    scale = 1.0 / math.sqrt(D)
    q, k_prefix, v_prefix, k_cur, v_cur = _make_tensors(
        cached_len, q_len, H, D, seed=cached_len * 1000 + q_len
    )
    plan = _tq_chunked_prefix_plan(cached_len, block_size, chunk_tokens)
    got = _prefix_combine_chunked(q, k_prefix, v_prefix, k_cur, v_cur, scale, plan)
    ref = _reference_full(q, k_prefix, v_prefix, k_cur, v_cur, scale, cached_len)
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "cached_len,q_len,block_size,chunk_tokens,H,D", CASES
)
def test_chunked_matches_unchunked_directly(
    cached_len, q_len, block_size, chunk_tokens, H, D
):
    """Direct equivalence at the tightest tolerance: the chunked path must
    be numerically indistinguishable from the unchunked path it replaces,
    not just both individually close to ground truth."""
    scale = 1.0 / math.sqrt(D)
    q, k_prefix, v_prefix, k_cur, v_cur = _make_tensors(
        cached_len, q_len, H, D, seed=cached_len * 1000 + q_len
    )
    plan = _tq_chunked_prefix_plan(cached_len, block_size, chunk_tokens)
    chunked = _prefix_combine_chunked(q, k_prefix, v_prefix, k_cur, v_cur, scale, plan)
    unchunked = _prefix_combine_unchunked(q, k_prefix, v_prefix, k_cur, v_cur, scale)
    assert torch.allclose(chunked, unchunked, rtol=1e-5, atol=1e-6)


def test_many_small_chunks_still_matches():
    """Stress the fold with many chunks (chunk_tokens == block_size, the
    smallest legal chunk) rather than the usual 1-3 chunks in CASES, to
    exercise the sequential-merge accumulation more than a couple of times."""
    cached_len, q_len, block_size, H, D = 400, 4, 16, 2, 8
    scale = 1.0 / math.sqrt(D)
    q, k_prefix, v_prefix, k_cur, v_cur = _make_tensors(
        cached_len, q_len, H, D, seed=12345
    )
    plan = _tq_chunked_prefix_plan(cached_len, block_size, block_size)
    assert len(plan) == cached_len // block_size  # 25 single-page chunks
    chunked = _prefix_combine_chunked(q, k_prefix, v_prefix, k_cur, v_cur, scale, plan)
    ref = _reference_full(q, k_prefix, v_prefix, k_cur, v_cur, scale, cached_len)
    assert torch.allclose(chunked, ref, rtol=1e-4, atol=1e-5)


def test_merge_is_order_independent_associative_combine():
    """Merging three partials in a different grouping/order must still
    agree (up to fp rounding) -- checks the associativity property the
    whole chunking design leans on, independent of the specific chunk
    plan/attention shapes used elsewhere in this file."""
    g = torch.Generator().manual_seed(7)
    T, H, D = 3, 2, 4
    outs = [torch.randn(T, H, D, generator=g) for _ in range(3)]
    lses = [torch.randn(T, H, generator=g) for _ in range(3)]

    # ((0 merge 1) merge 2)
    o01, l01 = _merge(outs[0], lses[0], outs[1], lses[1])
    left_assoc, _ = _merge(o01, l01, outs[2], lses[2])

    # (0 merge (1 merge 2))
    o12, l12 = _merge(outs[1], lses[1], outs[2], lses[2])
    right_assoc, _ = _merge(outs[0], lses[0], o12, l12)

    assert torch.allclose(left_assoc, right_assoc, rtol=1e-5, atol=1e-6)
