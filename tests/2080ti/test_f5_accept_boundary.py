# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""F-5 (backlog RANK 2, f5-forced-k lane): unit tests for the pure-Python
mamba align-mode block-boundary detector in
``vllm.v1.worker.f5_accept_boundary``.

No torch/CUDA required -- every function under test operates on plain
ints/dicts/sequences. These tests pin two things:

  1. ``compute_curr_state_idx`` reproduces
     ``vllm.v1.worker.mamba_utils.preprocess_mamba``'s original inline
     formula exactly (it was extracted from, not rewritten against, that
     function -- see the worked corner case in that function's own
     docstring/comment, reproduced as a test case below).
  2. ``batch_has_pending_boundary_crossing`` correctly OR-reduces per-request
     crossing predictions across a batch, including the edge cases
     (brand-new request, request missing from one of the two mappings,
     request whose prior block is the synthetic "no computed tokens yet"
     sentinel) that a careless reimplementation would get wrong.

This validates the shape/keying logic only -- it says nothing about
whether refusing FULL-graph replay on a detected boundary actually
prevents the doubled-token defect on real hardware. That requires the
forced-K probe (PROBE.md) on a GPU, which this test suite does not have
access to; see LANE/REPORT.md.
"""

from vllm.v1.worker.f5_accept_boundary import (
    batch_has_pending_boundary_crossing,
    compute_curr_state_idx,
    crosses_boundary,
    request_crosses_boundary_this_step,
)


def _original_formula(
    num_computed_tokens: int, num_scheduled_tokens: int, block_size: int, num_speculative_blocks: int
) -> int:
    """Direct transcription of the original inline computation this module
    replaced in ``mamba_utils.preprocess_mamba``, kept here so a future
    change to either copy is caught by a diff in test behavior rather than
    silently drifting apart."""
    from vllm.utils.math_utils import cdiv

    num_blocks = (
        cdiv(num_computed_tokens + num_scheduled_tokens, block_size) + num_speculative_blocks
    )
    return num_blocks - 1 - num_speculative_blocks


class TestComputeCurrStateIdx:
    def test_matches_original_inline_formula_broadly(self) -> None:
        for num_computed in (0, 1, 3, 4, 5, 15, 16, 17, 63, 64, 1000):
            for num_scheduled in (1, 2, 3, 4):
                for block_size in (4, 16, 32):
                    for num_spec_blocks in (0, 1, 2, 3):
                        got = compute_curr_state_idx(
                            num_computed, num_scheduled, block_size, num_spec_blocks
                        )
                        want = _original_formula(
                            num_computed, num_scheduled, block_size, num_spec_blocks
                        )
                        assert got == want, (
                            f"mismatch at computed={num_computed}, "
                            f"scheduled={num_scheduled}, block_size={block_size}, "
                            f"num_spec_blocks={num_spec_blocks}: {got} != {want}"
                        )

    def test_worked_corner_case_from_mamba_utils_docstring(self) -> None:
        # From preprocess_mamba's own comment: block_size=4,
        # num_speculative_tokens=2, request [A,B,C] + 2 draft tokens.
        # Block 0: [A,B,C,draft1], Block 1: [draft2, TOFILL, TOFILL, TOFILL],
        # Block 2/3: speculative blocks. Running state saved at block 1.
        # 3 prompt tokens + 2 draft tokens scheduled this step = 5 new tokens
        # total context length (num_computed_tokens=0 before this step).
        curr = compute_curr_state_idx(
            num_computed_tokens=0,
            num_scheduled_tokens=5,
            block_size=4,
            num_speculative_blocks=2,
        )
        assert curr == 1

    def test_monotonic_nondecreasing_in_computed_tokens(self) -> None:
        block_size, num_spec_blocks, num_scheduled = 16, 3, 1
        prev = compute_curr_state_idx(0, num_scheduled, block_size, num_spec_blocks)
        for num_computed in range(1, 200):
            curr = compute_curr_state_idx(
                num_computed, num_scheduled, block_size, num_spec_blocks
            )
            assert curr >= prev
            prev = curr


class TestCrossesBoundary:
    def test_none_never_crosses(self) -> None:
        assert crosses_boundary(None, 0) is False
        assert crosses_boundary(None, 5) is False

    def test_sentinel_minus_one_never_crosses(self) -> None:
        # -1 is preprocess_mamba's synthetic "brand new / zero computed
        # tokens" sentinel: (0 - 1) // block_size == -1 for any block_size.
        assert crosses_boundary(-1, 0) is False
        assert crosses_boundary(-1, 3) is False

    def test_same_block_does_not_cross(self) -> None:
        assert crosses_boundary(4, 4) is False

    def test_different_block_crosses(self) -> None:
        assert crosses_boundary(4, 5) is True
        assert crosses_boundary(10, 3) is True  # direction doesn't matter


class TestRequestCrossesBoundaryThisStep:
    def test_returns_curr_state_idx_alongside_flag(self) -> None:
        crosses, curr = request_crosses_boundary_this_step(
            prev_state_idx=0,
            num_computed_tokens=15,
            num_scheduled_tokens=2,
            block_size=16,
            num_speculative_blocks=0,
        )
        assert curr == compute_curr_state_idx(15, 2, 16, 0)
        assert crosses == crosses_boundary(0, curr)


class TestBatchHasPendingBoundaryCrossing:
    def test_empty_batch_is_false(self) -> None:
        assert (
            batch_has_pending_boundary_crossing(
                req_ids=[],
                num_computed_tokens_by_req={},
                num_scheduled_tokens_by_req={},
                mamba_state_idx={},
                block_size=16,
                num_speculative_blocks=3,
            )
            is False
        )

    def test_no_request_crosses(self) -> None:
        # Every request comfortably mid-block: computed=1, scheduled=1,
        # block_size=16 -> curr_state_idx == 0 for all, matching prior.
        req_ids = ["r1", "r2", "r3"]
        got = batch_has_pending_boundary_crossing(
            req_ids=req_ids,
            num_computed_tokens_by_req={r: 1 for r in req_ids},
            num_scheduled_tokens_by_req={r: 1 for r in req_ids},
            mamba_state_idx={r: 0 for r in req_ids},
            block_size=16,
            num_speculative_blocks=0,
        )
        assert got is False

    def test_single_request_crossing_flips_whole_batch(self) -> None:
        req_ids = ["r1", "r2", "r3"]
        # r2 crosses a block boundary this step: computed=16, scheduled=1,
        # block_size=16 -> cdiv(17,16)=2 -> curr_state_idx == 1, but its
        # recorded prev is 0 (16 computed tokens exactly fill block 0, so
        # curr_state_idx stays 0 up to and including computed==16 with
        # scheduled==0; adding one more scheduled token is what pushes the
        # 17th token into block 1).
        computed = {"r1": 1, "r2": 16, "r3": 1}
        scheduled = {"r1": 1, "r2": 1, "r3": 1}
        prev = {"r1": 0, "r2": 0, "r3": 0}
        got = batch_has_pending_boundary_crossing(
            req_ids=req_ids,
            num_computed_tokens_by_req=computed,
            num_scheduled_tokens_by_req=scheduled,
            mamba_state_idx=prev,
            block_size=16,
            num_speculative_blocks=0,
        )
        assert got is True

    def test_brand_new_request_absent_from_mamba_state_idx_does_not_cross(
        self,
    ) -> None:
        # A request with no entry in mamba_state_idx yet (its first mamba
        # step) must not itself trigger a refusal -- there is nothing
        # "stale" to have captured yet.
        got = batch_has_pending_boundary_crossing(
            req_ids=["new_req"],
            num_computed_tokens_by_req={"new_req": 0},
            num_scheduled_tokens_by_req={"new_req": 40},
            mamba_state_idx={},  # no prior entry
            block_size=16,
            num_speculative_blocks=3,
        )
        assert got is False

    def test_request_missing_from_computed_tokens_mapping_is_skipped(self) -> None:
        # Defensive: a req_id present in the batch but absent from the
        # computed-tokens mapping (should not happen in practice, since
        # both are derived from the same scheduled batch) must not raise.
        got = batch_has_pending_boundary_crossing(
            req_ids=["ghost"],
            num_computed_tokens_by_req={},
            num_scheduled_tokens_by_req={"ghost": 1},
            mamba_state_idx={"ghost": 0},
            block_size=16,
            num_speculative_blocks=0,
        )
        assert got is False

    def test_missing_scheduled_tokens_defaults_to_zero(self) -> None:
        # num_scheduled_tokens_by_req.get(req_id, 0) fallback: a request
        # with computed tokens and a prior block but no scheduled-tokens
        # entry (defensive default) should not crash the arithmetic.
        got = batch_has_pending_boundary_crossing(
            req_ids=["r1"],
            num_computed_tokens_by_req={"r1": 16},
            num_scheduled_tokens_by_req={},
            mamba_state_idx={"r1": 0},
            block_size=16,
            num_speculative_blocks=0,
        )
        # computed=16, scheduled=0 -> cdiv(16,16)=1 -> curr=0 == prev -> no cross
        assert got is False
