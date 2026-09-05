# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""tq-continuation-crash lane: unit tests for the pure index math and
block-table clamp helpers behind TurboQuant's continuation-prefill dequant
launch (`_continuation_prefill` -> `_tq_full_dequant_kv`) and the decode
kernel family (`_tq_decode_stage1[_qtile]`).

No torch/CUDA device is exercised — every function under test here is
either plain Python arithmetic or a CPU tensor op (`torch.Tensor.clamp`
works identically on CPU and CUDA). This deliberately cannot test the
Triton kernel launches themselves (that needs a GPU, which this lane is
not permitted to allocate — see
/home/kevin/projects/lanes/tq-crash/REPORT.md for what the 23:30 GPU
repro window is expected to show for each mechanism); it pins the index
math and clamp semantics the fix and instrumentation commits depend on.

See /home/kevin/projects/lanes/tq-crash/ANALYSIS.md and MECHANISMS.md for
the investigation this supports.
"""

import math

import pytest
import torch

from vllm.v1.attention.backends.turboquant_attn import (
    _tq_block_ids_out_of_range,
    _tq_clamp_block_table_pages,
    _tq_clamp_cached_len_to_width,
    _tq_compute_pages_and_alloc_len,
)

# Production config this lane investigated (see ANALYSIS.md):
#   block_size=3568, max_num_batched_tokens=3584, max_model_len=524288
PROD_BLOCK_SIZE = 3568
PROD_MAX_MODEL_LEN = 524288
PROD_WIDTH = math.ceil(PROD_MAX_MODEL_LEN / PROD_BLOCK_SIZE)  # 147

# The five flight-recorder payload token counts from the 2026-09-03 03:3x
# crash window (ANALYSIS.md Section 4), used here as a live-numbers
# regression: whatever these compute to must match the ANALYSIS.md table.
FLIGHT_RECORDER_PTOK = [41498, 55597, 57126, 75767, 85976]


# --------------------------------------------------------------------------
# _tq_compute_pages_and_alloc_len
# --------------------------------------------------------------------------


def test_pages_and_alloc_len_zero_cached_len():
    assert _tq_compute_pages_and_alloc_len(0, 32) == (0, 0)
    assert _tq_compute_pages_and_alloc_len(-5, 32) == (0, 0)


def test_pages_and_alloc_len_exact_multiple():
    # cached_len exactly on a block boundary: pages == cached_len/block_size,
    # alloc_len == cached_len (no padding).
    pages, alloc_len = _tq_compute_pages_and_alloc_len(128, 32)
    assert (pages, alloc_len) == (4, 128)


def test_pages_and_alloc_len_rounds_up():
    # cached_len not a multiple of block_size -- the common case in
    # production, since block_size (3568) != max_num_batched_tokens (3584).
    pages, alloc_len = _tq_compute_pages_and_alloc_len(100, 32)
    assert pages == 4  # ceil(100/32)
    assert alloc_len == 128  # 4*32
    assert alloc_len >= 100


@pytest.mark.parametrize("cached_len", [1, 31, 32, 33, 3567, 3568, 3569, 999983])
def test_pages_and_alloc_len_invariants(cached_len):
    block_size = PROD_BLOCK_SIZE
    pages, alloc_len = _tq_compute_pages_and_alloc_len(cached_len, block_size)
    # ANALYSIS.md Section 1.3's core safety argument: alloc_len is always a
    # block_size multiple that is >= cached_len, and every position up to
    # alloc_len-1 maps to a page index strictly less than `pages` -- i.e.
    # the "grid spans alloc_len, not cached_len" padding never walks past
    # the pages this function reports.
    assert alloc_len % block_size == 0
    assert alloc_len >= cached_len
    assert pages * block_size == alloc_len
    last_pos = alloc_len - 1
    assert last_pos // block_size < pages


def test_pages_and_alloc_len_matches_analysis_table():
    # Cross-check against the table in ANALYSIS.md Section 4.
    expected = {
        41498: (12, 42816),
        55597: (16, 57088),
        57126: (17, 60656),
        75767: (22, 78496),
        85976: (25, 89200),
    }
    for ptok, (exp_pages, exp_alloc_len) in expected.items():
        pages, alloc_len = _tq_compute_pages_and_alloc_len(ptok, PROD_BLOCK_SIZE)
        assert (pages, alloc_len) == (exp_pages, exp_alloc_len), ptok


def test_flight_recorder_payloads_never_exceed_width():
    # ANALYSIS.md Section 4's headline finding: none of the five real
    # crash-window payloads come anywhere near tripping the pages>width
    # structural clamp (width=147 at the production block_size/max_model_len).
    for ptok in FLIGHT_RECORDER_PTOK:
        pages, _ = _tq_compute_pages_and_alloc_len(ptok, PROD_BLOCK_SIZE)
        assert pages <= PROD_WIDTH
        assert pages < PROD_WIDTH // 4  # not even close


# --------------------------------------------------------------------------
# _tq_clamp_cached_len_to_width (the always-on structural guard)
# --------------------------------------------------------------------------


def test_clamp_cached_len_no_op_when_within_width():
    cached_len, seq_len, was_clamped = _tq_clamp_cached_len_to_width(
        cached_len=1000, q_len=50, block_size=32, width=100
    )
    assert cached_len == 1000
    assert seq_len == 1050
    assert was_clamped is False


def test_clamp_cached_len_zero_passthrough():
    cached_len, seq_len, was_clamped = _tq_clamp_cached_len_to_width(
        cached_len=0, q_len=50, block_size=32, width=4
    )
    assert (cached_len, seq_len, was_clamped) == (0, 50, False)


def test_clamp_cached_len_clamps_when_pages_exceed_width():
    # pages = ceil(5000/32) = 157 > width=4 -> clamp to width*block_size.
    cached_len, seq_len, was_clamped = _tq_clamp_cached_len_to_width(
        cached_len=5000, q_len=10, block_size=32, width=4
    )
    assert was_clamped is True
    assert cached_len == 4 * 32
    # The seq_len == cached_len + q_len invariant the caller's k_full/v_full
    # sizing depends on (ANALYSIS.md Section 1.3) must hold post-clamp too.
    assert seq_len == cached_len + 10


def test_clamp_cached_len_boundary_exact_width():
    # pages == width exactly must NOT clamp (only pages > width should).
    block_size, width = 32, 4
    cached_len = block_size * width  # pages == width exactly
    out_cached_len, seq_len, was_clamped = _tq_clamp_cached_len_to_width(
        cached_len=cached_len, q_len=1, block_size=block_size, width=width
    )
    assert was_clamped is False
    assert out_cached_len == cached_len


@pytest.mark.parametrize(
    "cached_len,q_len,block_size,width",
    [
        (5000, 10, 32, 4),
        (999983, 7, PROD_BLOCK_SIZE, PROD_WIDTH),  # near-ceiling, prod sizing
        (PROD_BLOCK_SIZE * (PROD_WIDTH + 1), 3, PROD_BLOCK_SIZE, PROD_WIDTH),
    ],
)
def test_clamp_cached_len_result_always_fits_width(cached_len, q_len, block_size, width):
    out_cached_len, seq_len, _ = _tq_clamp_cached_len_to_width(
        cached_len, q_len, block_size, width
    )
    pages_after = math.ceil(out_cached_len / block_size) if out_cached_len > 0 else 0
    assert pages_after <= width
    assert seq_len == out_cached_len + q_len


# --------------------------------------------------------------------------
# _tq_block_ids_out_of_range
# --------------------------------------------------------------------------


def test_block_ids_out_of_range_empty():
    assert _tq_block_ids_out_of_range([], 100) == (False, -1, -1)


def test_block_ids_out_of_range_all_valid():
    out_of_range, mn, mx = _tq_block_ids_out_of_range([0, 5, 10, 99], 100)
    assert (out_of_range, mn, mx) == (False, 0, 99)


def test_block_ids_out_of_range_negative():
    out_of_range, mn, mx = _tq_block_ids_out_of_range([-1, 0, 5], 100)
    assert out_of_range is True
    assert mn == -1


def test_block_ids_out_of_range_too_large():
    out_of_range, mn, mx = _tq_block_ids_out_of_range([0, 5, 100], 100)
    assert out_of_range is True  # 100 is not < num_blocks=100
    assert mx == 100


def test_block_ids_out_of_range_exactly_at_bound_is_valid():
    # num_blocks=100 means valid ids are [0, 99]; 99 must NOT be flagged.
    out_of_range, _, _ = _tq_block_ids_out_of_range([99], 100)
    assert out_of_range is False


# --------------------------------------------------------------------------
# _tq_clamp_block_table_pages (the CPU/CUDA-agnostic tensor clamp used by
# the unconditional fix in _continuation_prefill)
# --------------------------------------------------------------------------


def test_clamp_block_table_pages_noop_shape_and_dtype():
    bt = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int32)
    out = _tq_clamp_block_table_pages(bt, pages=3, num_blocks=50)
    assert out.shape == bt.shape
    assert out.dtype == bt.dtype


def test_clamp_block_table_pages_clamps_only_first_pages_columns():
    bt = torch.tensor([[0, 1, 2, 999, -5, 7]])
    out = _tq_clamp_block_table_pages(bt, pages=5, num_blocks=50)
    # First 5 columns clamped into [0, 49]; column 5 (value 7, untouched
    # because pages=5 means indices 0..4) must pass through unchanged.
    assert out.tolist() == [[0, 1, 2, 49, 0, 7]]


def test_clamp_block_table_pages_is_a_no_op_on_already_valid_ids():
    # "Clearly bounds-safe": the whole point of making this unconditional
    # is that it must not change anything when nothing was wrong.
    bt = torch.tensor([[0, 1, 2, 3, 10, 49]])
    out = _tq_clamp_block_table_pages(bt, pages=6, num_blocks=50)
    assert out.tolist() == bt.tolist()


def test_clamp_block_table_pages_never_mutates_caller_tensor():
    bt = torch.tensor([[0, 1, 2, 999]])
    original = bt.clone()
    _tq_clamp_block_table_pages(bt, pages=4, num_blocks=50)
    assert torch.equal(bt, original), "helper must not mutate the input in place"


def test_clamp_block_table_pages_zero_pages_returns_input_unchanged():
    bt = torch.tensor([[0, 1, 999]])
    out = _tq_clamp_block_table_pages(bt, pages=0, num_blocks=50)
    assert out is bt  # identity: no clone when there is nothing to clamp


def test_clamp_block_table_pages_zero_num_blocks_returns_input_unchanged():
    bt = torch.tensor([[0, 1, 999]])
    out = _tq_clamp_block_table_pages(bt, pages=2, num_blocks=0)
    assert out is bt


def test_clamp_block_table_pages_multi_row():
    bt = torch.tensor([[0, 999, 2], [-1, 5, 999]])
    out = _tq_clamp_block_table_pages(bt, pages=3, num_blocks=10)
    assert out.tolist() == [[0, 9, 2], [0, 5, 9]]


# --------------------------------------------------------------------------
# Behavioral equivalence: old (gated, detect-then-clamp) vs new
# (unconditional) semantics for the continuation-prefill safety property.
#
# This pins the claim in the FIX A commit message: the new unconditional
# clamp must produce bitwise-identical output to the old
# VLLM_TURBOQUANT_CONTINUATION_BOUNDS_CHECK=1 code path in both the
# "nothing wrong" case and the "something was out of range" case -- it only
# removes the host sync used to *decide* whether to clamp, not the clamp's
# own correctness.
# --------------------------------------------------------------------------


def _old_gated_clamp(block_table: torch.Tensor, pages_eff: int, num_blocks: int) -> torch.Tensor:
    """Reference re-implementation of the pre-fix behavior (see git history
    of turboquant_attn.py, commit "guard: env-gated block_table bounds
    check..."): only clone+clamp when a violation is actually found.
    """
    if pages_eff <= 0:
        return block_table
    bt_slice = block_table[0, :pages_eff]
    bt_max = int(bt_slice.max().item())
    bt_min = int(bt_slice.min().item())
    if bt_min < 0 or bt_max >= num_blocks:
        safe = block_table.clone()
        safe[0, :pages_eff].clamp_(0, num_blocks - 1)
        return safe
    return block_table


@pytest.mark.parametrize(
    "row,pages_eff,num_blocks",
    [
        ([0, 1, 2, 3], 4, 50),  # all valid -> old path returns input unchanged
        ([0, 1, 2, 999], 4, 50),  # one OOB -> old path clamps
        ([-1, -2, 5, 6], 4, 50),  # negative ids -> old path clamps
        ([49, 49, 49, 49], 4, 50),  # exactly at the boundary -> valid, no clamp
    ],
)
def test_new_clamp_matches_old_gated_clamp_values(row, pages_eff, num_blocks):
    bt = torch.tensor([row])
    old = _old_gated_clamp(bt, pages_eff, num_blocks)
    new = _tq_clamp_block_table_pages(bt, pages_eff, num_blocks)
    assert old.tolist() == new.tolist()
