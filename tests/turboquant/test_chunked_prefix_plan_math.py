# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RANK-4 backlog ("chunked continuation-dequant, +93K pool tokens"):
unit tests for the pure chunk-range/pages/alloc arithmetic in
``_tq_chunked_prefix_plan`` (vllm/v1/attention/backends/turboquant_attn.py).

This is the math ``TurboQuantAttentionImpl._continuation_prefix_combine_chunked``
uses to decide, for a given cached prefix length, exactly which
[start_tok, start_tok+chunk_len) range of the TQ cache to dequantize per
chunk and which block_table page range backs it. No CUDA/triton/flashinfer
is exercised here -- only the host-side partition arithmetic, which is
exactly what the tq-crash sibling lane's "bounds-explicit" coordination ask
is about: get the range/page math right and provably in-bounds *before*
anything is asserted or clamped against real tensors at launch time.

Boundary cached_len values (3567/3568/3569, 41498, 85976) are the ones
called out in the RANK-4 task brief -- picked to straddle a block-size
boundary and to land inside the 41K-86K range the tq-crash lane is
investigating for the illegal-memory-access crash in this same code path.
"""

import pytest

from vllm.v1.attention.backends.turboquant_attn import _tq_chunked_prefix_plan

pytestmark = pytest.mark.cpu_test

BOUNDARY_CACHED_LENS = [1, 2, 15, 16, 17, 3567, 3568, 3569, 41498, 85976, 524288]
BLOCK_SIZES = [1, 16, 32, 64, 128]
CHUNK_TOKENS = [1, 15, 16, 17, 4096, 8192, 16384, 100_000, 10_000_000]


def _check_plan_invariants(
    plan: list[tuple[int, int, int, int, int]],
    cached_len: int,
    block_size: int,
    chunk_tokens: int,
) -> None:
    assert len(plan) > 0, "plan must cover a positive cached_len with >=1 chunk"

    chunk_pages_expected = max(1, chunk_tokens // block_size)
    chunk_tokens_eff = chunk_pages_expected * block_size

    # No gaps, no overlaps, exact coverage of [0, cached_len).
    running = 0
    for idx, (start_tok, chunk_len, alloc_len, start_page, pages_needed) in enumerate(
        plan
    ):
        is_last = idx == len(plan) - 1

        assert start_tok == running, (
            f"chunk {idx}: expected start_tok={running}, got {start_tok} "
            f"(gap or overlap)"
        )
        assert chunk_len > 0, f"chunk {idx}: chunk_len must be positive"

        # Every chunk boundary is exactly page-aligned: start_tok is always
        # an integer multiple of chunk_tokens_eff, which is itself an exact
        # multiple of block_size, by construction (i * chunk_pages *
        # block_size for integer i, chunk_pages).
        assert start_tok % block_size == 0, (
            f"chunk {idx}: start_tok={start_tok} not page-aligned to "
            f"block_size={block_size}"
        )
        assert start_page == start_tok // block_size

        # All but the last chunk are exactly the effective chunk size;
        # only the last may be a (shorter) remainder.
        if not is_last:
            assert chunk_len == chunk_tokens_eff, (
                f"chunk {idx}/{len(plan) - 1}: non-final chunk_len={chunk_len} "
                f"!= chunk_tokens_eff={chunk_tokens_eff}"
            )
        else:
            assert chunk_len <= chunk_tokens_eff

        # alloc_len rounds chunk_len up to a whole number of pages, and pads
        # by less than one page -- mirrors the unchunked branch's
        # alloc_len = ceil(cached_len / block_size) * block_size.
        assert pages_needed == -(-chunk_len // block_size)  # ceil div
        assert alloc_len == pages_needed * block_size
        assert alloc_len >= chunk_len
        assert alloc_len - chunk_len < block_size

        running += chunk_len

    assert running == cached_len, (
        f"plan covers {running} tokens, expected exactly cached_len={cached_len}"
    )

    # num_chunks matches ceil(cached_len / chunk_tokens_eff).
    import math

    assert len(plan) == max(1, math.ceil(cached_len / chunk_tokens_eff))


@pytest.mark.parametrize("cached_len", BOUNDARY_CACHED_LENS)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("chunk_tokens", CHUNK_TOKENS)
def test_plan_invariants_hold(cached_len, block_size, chunk_tokens):
    plan = _tq_chunked_prefix_plan(cached_len, block_size, chunk_tokens)
    _check_plan_invariants(plan, cached_len, block_size, chunk_tokens)


@pytest.mark.parametrize("cached_len", [3567, 3568, 3569, 41498, 85976])
def test_boundary_values_exact_shape_block_size_16(cached_len):
    """Named regression pin for the exact values called out in the RANK-4
    brief, at vLLM's common default block_size=16, chunk=16384 (the value
    named in the brief as the example production setting)."""
    plan = _tq_chunked_prefix_plan(cached_len, 16, 16384)
    _check_plan_invariants(plan, cached_len, 16, 16384)
    # 16384 is already a multiple of 16, so chunk_tokens_eff == 16384 exactly
    # -- no rounding-down surprise at this specific, documented setting.
    full_chunks = cached_len // 16384
    remainder = cached_len % 16384
    expected_num_chunks = full_chunks + (1 if remainder else 0)
    assert len(plan) == expected_num_chunks
    if remainder:
        assert plan[-1][1] == remainder
    else:
        assert plan[-1][1] == 16384


def test_single_chunk_when_chunk_size_covers_whole_prefix():
    """chunk_tokens >= cached_len must degenerate to exactly one chunk
    spanning the whole prefix -- the algorithmic no-op case that should
    reduce to the same single dequant+attend call the unchunked branch
    already makes."""
    for cached_len in (1, 100, 3568, 85976):
        plan = _tq_chunked_prefix_plan(cached_len, 16, 10_000_000)
        assert len(plan) == 1
        start_tok, chunk_len, alloc_len, start_page, pages_needed = plan[0]
        assert start_tok == 0
        assert chunk_len == cached_len
        assert start_page == 0


def test_chunk_tokens_rounds_down_to_block_multiple():
    """chunk_tokens not a multiple of block_size rounds DOWN (never up --
    rounding up could make a chunk's page range reach past what the
    caller's block_table bounds check has already validated for the
    overall cached_len)."""
    # block_size=16, chunk_tokens=17 -> chunk_pages=max(1,17//16)=1 ->
    # chunk_tokens_eff=16, not 32.
    plan = _tq_chunked_prefix_plan(100, 16, 17)
    assert plan[0][1] == 16  # first (non-final) chunk_len == chunk_tokens_eff


def test_chunk_tokens_smaller_than_block_size_floors_to_one_page():
    # chunk_pages = max(1, chunk_tokens // block_size) guards against a
    # zero-token chunk (and hence an infinite loop) when chunk_tokens <
    # block_size.
    plan = _tq_chunked_prefix_plan(100, 16, 1)
    _check_plan_invariants(plan, 100, 16, 1)
    assert plan[0][1] == 16


@pytest.mark.parametrize(
    "cached_len,block_size,chunk_tokens",
    [(0, 16, 16384), (-1, 16, 16384), (100, 0, 16384), (100, 16, 0), (100, 16, -5)],
)
def test_invalid_inputs_raise(cached_len, block_size, chunk_tokens):
    with pytest.raises(AssertionError):
        _tq_chunked_prefix_plan(cached_len, block_size, chunk_tokens)
