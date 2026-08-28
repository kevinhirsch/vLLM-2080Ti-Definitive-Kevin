# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Post-port expected behavior for vLLM PR #53479 (partial-prefix-cache hits /
align-mode Mamba dense boundary materialization) on frontier-pastnative-20260816.

TDD scaffolding written during F-1 PORT PREP (see
docs/f1-partial-prefix-hits-research.md and the F-1 port-prep correspondence
table, `correspondence.md`, produced alongside this file but not committed to
this branch). These tests express the intended POST-PORT behavior of
`Scheduler._mamba_block_aligned_split`; the ones that depend on code that does
not exist yet are marked `pytest.mark.xfail(reason="pre-port")` and must start
passing (with the marker removed) once the actual port lands. This file is
read-only against the pre-port function today -- no production code is
touched here.

Three groups, matching the F-1 port-prep task brief:
  1. Boundary-stop materialization per chunk (the core #53479 fix: a state
     must materialize at EVERY crossed block boundary, not only where a
     scheduling step's budget happens to run out).
  2. MTP-retention gate interplay (this fork's existing
     retain_mamba_align_mtp_cache_block, per docs/mtp-retention-invariant.md,
     plus the post-port retention-interval-aware replay/eagle-reach stops).
  3. EAGLE back-off cases. Whether the ported shape keeps this fork's narrow
     MTP-only relaxation or adopts upstream's unconditional back-off removal
     is an OPEN design decision (see correspondence.md) -- deliberately NOT
     presupposed here. Only the current, passing shape is pinned.

All expected values in the xfail tests were derived by hand-simulating the
proposed post-port algorithm (correspondence.md Sec B3's trimmed 4-element
`stops` tuple: `boundary_stop, replay_boundary, eagle_reach,
last_cache_position` -- our fork has no equivalent of upstream's
`tail_boundary`/shared-prefix stops, see correspondence.md) and then verified
by actually running each case against today's pre-port code to confirm it
fails for the expected reason (AssertionError, not a coincidental pass) rather
than merely asserted from upstream's own numbers, which use a much larger
block size than these tests and don't transfer directly.
"""

from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.scheduler import Scheduler


def _stub_scheduler(
    *,
    block_size: int = 16,
    use_eagle: bool = False,
    retain_mamba_align_mtp_cache_block: bool = False,
    mamba_retention_interval=None,  # post-port attr; unread by pre-port code
    mamba_eagle_reach_margin: int = 0,  # post-port attr; unread by pre-port code
) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        use_eagle=use_eagle,
        retain_mamba_align_mtp_cache_block=retain_mamba_align_mtp_cache_block,
        mamba_retention_interval=mamba_retention_interval,
        mamba_eagle_reach_margin=mamba_eagle_reach_margin,
    )


def _request(num_tokens: int, num_computed_tokens: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        num_prompt_tokens=num_tokens,
        num_tokens=num_tokens,
        num_computed_tokens=num_computed_tokens,
    )


def _split(scheduler, request, num_new_tokens: int) -> int:
    return Scheduler._mamba_block_aligned_split(scheduler, request, num_new_tokens)


def _run_to_completion(scheduler, request, prompt_len: int) -> list[int]:
    """Replay the split call-by-call, as the real scheduler does, until the
    whole prompt is consumed. Returns the sequence of chunk-end positions."""
    ends: list[int] = []
    while request.num_computed_tokens < prompt_len:
        remaining = prompt_len - request.num_computed_tokens
        num_new = _split(scheduler, request, remaining)
        assert num_new > 0, (
            f"split returned {num_new} <= 0 at num_computed_tokens="
            f"{request.num_computed_tokens} -- would infinite-loop the real scheduler"
        )
        request.num_computed_tokens += num_new
        ends.append(request.num_computed_tokens)
    return ends


# ---------------------------------------------------------------------------
# Group 1: boundary-stop materialization per chunk (the core #53479 fix)
# ---------------------------------------------------------------------------


def test_current_behavior_large_budget_jumps_past_intermediate_boundaries() -> None:
    """PRE-PORT, PASSING TODAY: pins the sparse-state bug itself. A single
    scheduler step whose budget spans multiple block boundaries jumps
    straight to the boundary nearest the budget limit -- intermediate
    boundaries never become chunk ends, so no state is ever materialized
    there (docs/f1-partial-prefix-hits-research.md Sec 3: "only one Mamba
    state block is recorded per scheduler step"). This is the mirror image of
    test_boundary_stop_materializes_one_block_at_a_time below: when the port
    lands, THIS test's assertion should invert (64, not 16) -- re-check both
    when that happens, not just the new one."""
    block_size = 16
    scheduler = _stub_scheduler(block_size=block_size, use_eagle=False)
    request = _request(num_tokens=10 * block_size)
    # Budget large enough to span 4 full blocks in one step.
    assert _split(scheduler, request, 4 * block_size) == 4 * block_size


def test_boundary_stop_materializes_one_block_at_a_time() -> None:
    """POST-PORT: with dense retention (mamba_retention_interval=None, this
    fork's default), a chunk must stop at the VERY NEXT block boundary
    regardless of how much scheduling budget is available, so a state
    materializes at every crossed boundary rather than only where the budget
    happens to run out (vLLM #53479 "boundary_stop"). Mirrors upstream
    tests/v1/core/test_mamba_align_chunk_split.py::
    test_split_stops_at_every_boundary_without_checkpoints."""
    block_size = 16
    # FORK DEVIATION (documented in scheduler.py): retention None keeps the
    # legacy multi-block chunking; the #53479 every-block materialization is
    # opt-in via retention <= block_size. These tests exercise the opt-in arm.
    scheduler = _stub_scheduler(
        block_size=block_size, use_eagle=False,
        mamba_retention_interval=block_size,
    )
    request = _request(num_tokens=10 * block_size)
    assert _split(scheduler, request, 4 * block_size) == block_size


def test_boundary_stop_advances_one_block_per_call_across_full_prefill() -> None:
    """POST-PORT: replaying the split call-by-call over a full unaligned
    prefill must land on every block boundary in turn, then the tail --
    never skipping ahead to a later boundary just because the budget would
    allow it. Mirrors upstream's test_split_stops_at_every_boundary_without_
    checkpoints, scaled to a block size where the unaligned tail (7) is
    smaller than one block, matching this fork's own convention (see
    test_mamba_align_prefill_split_keeps_intermediate_chunks_aligned in
    test_prefix_caching.py)."""
    block_size = 16
    prompt_len = 3 * block_size + 7
    # FORK DEVIATION (documented in scheduler.py): retention None keeps the
    # legacy multi-block chunking; the #53479 every-block materialization is
    # opt-in via retention <= block_size. These tests exercise the opt-in arm.
    scheduler = _stub_scheduler(
        block_size=block_size, use_eagle=False,
        mamba_retention_interval=block_size,
    )
    request = _request(num_tokens=prompt_len)
    ends = _run_to_completion(scheduler, request, prompt_len)
    assert ends == [block_size, 2 * block_size, 3 * block_size, prompt_len]


def test_exact_block_aligned_prompt_gets_a_state_above_the_lookup_cap() -> None:
    """POST-PORT regression for the second half of the doc's "sparse states"
    bug: an EXACTLY block-aligned prompt must still materialize a state
    reachable by an identical-repeat lookup capped at num_tokens - 1, not
    only at num_tokens (research doc Sec 1, defect 1, second sentence)."""
    block_size = 16
    prompt_len = 4 * block_size  # exactly aligned
    # FORK DEVIATION (documented in scheduler.py): retention None keeps the
    # legacy multi-block chunking; the #53479 every-block materialization is
    # opt-in via retention <= block_size. These tests exercise the opt-in arm.
    scheduler = _stub_scheduler(
        block_size=block_size, use_eagle=False,
        mamba_retention_interval=block_size,
    )
    request = _request(num_tokens=prompt_len)
    ends = _run_to_completion(scheduler, request, prompt_len)
    # The boundary one block below the (aligned) prompt end must be a chunk
    # end too, not just the final one -- the position a capped
    # (num_tokens - 1) lookup can actually reach.
    assert (prompt_len - block_size) in ends


# ---------------------------------------------------------------------------
# Group 2: MTP-retention gate interplay (docs/mtp-retention-invariant.md)
# ---------------------------------------------------------------------------


def test_mtp_retention_gate_unaffected_by_absent_post_port_attrs() -> None:
    """PRE-PORT, PASSING TODAY: this fork's existing MTP retention gate
    (retain_mamba_align_mtp_cache_block) must keep behaving exactly as
    docs/mtp-retention-invariant.md specifies, independent of whether the
    post-port retention-interval attributes exist -- the invariant's safety
    argument (proposer statelessness, not dense boundary materialization)
    does not depend on #53479 at all. Numbers match the doc's own worked
    example (also pinned by tests/2080ti/test_mamba_align_mtp_prefix_cache.py)."""
    block_size = 4
    scheduler = _stub_scheduler(
        block_size=block_size, use_eagle=True, retain_mamba_align_mtp_cache_block=True
    )
    request = _request(num_tokens=14)
    assert _split(scheduler, request, 14) == 12  # tail retained (doc's example)

    scheduler_disabled = _stub_scheduler(
        block_size=block_size, use_eagle=True, retain_mamba_align_mtp_cache_block=False
    )
    assert _split(scheduler_disabled, request, 14) == 8  # legacy EAGLE back-off


def test_default_retention_keeps_the_eagle_reachable_state_post_port() -> None:
    """POST-PORT: under sparse retention (mamba_retention_interval=0) with a
    positive mamba_eagle_reach_margin, the split must end a chunk at the
    eagle-reachable boundary (num_prompt_tokens - 1 - margin, block-floored)
    as well as the replay boundary, so a state exists there for a pruned
    EAGLE/MTP lookup to find -- otherwise sparse retention only keeps the
    replay boundary and a repeat request misses (upstream #53479, jschmied's
    GB10 confirmation: request 3 -> 2). Verified decision-agnostic w.r.t. the
    open back-off design question (correspondence.md Sec B2): the expected
    `ends` sequence is identical whether or not the legacy EAGLE back-off is
    also kept, because eagle_reach and the (possibly backed-off)
    last_cache_position coincide here regardless."""
    block_size = 16
    prompt_len = 4 * block_size + 7
    scheduler = _stub_scheduler(
        block_size=block_size,
        use_eagle=True,
        mamba_retention_interval=0,
        mamba_eagle_reach_margin=block_size,
    )
    request = _request(num_tokens=prompt_len)
    ends = _run_to_completion(scheduler, request, prompt_len)
    # eagle-reachable boundary, replay boundary, tail -- nothing else, since
    # sparse retention (0) discards every other boundary_stop.
    assert ends == [3 * block_size, 4 * block_size, prompt_len]


def test_sparse_retention_without_eagle_margin_matches_current_chunking() -> None:
    """NEGATIVE CONTROL, PASSING BOTH PRE- AND POST-PORT (deliberately not
    xfail): isolates that eagle_reach specifically -- not sparse retention on
    its own -- is what test_default_retention_keeps_the_eagle_reachable_
    state_post_port discriminates on. With mamba_eagle_reach_margin=0 (no
    speculative decoding, or no pruned attention group), post-port's
    replay_boundary = (num_prompt_tokens - 1) // block_size * block_size is
    numerically identical to pre-port's non-eagle last_cache_position =
    round_down(num_tokens, block_size), because num_tokens == num_prompt_tokens
    throughout initial prefill (no output tokens yet). So sparse retention
    (mamba_retention_interval=0) with no eagle margin produces the exact same
    chunk-end sequence as today's code -- there is nothing extra to retain
    beyond what the pre-port function already naturally stops at. This was
    initially miswritten as an xfail (asserting the post-port-only 2-entry
    shape) and caught as a strict-xfail XPASS failure when actually run
    against pre-port code -- left in as a real, verified invariant rather
    than deleted, since it precisely isolates what does and doesn't require
    the port."""
    block_size = 16
    prompt_len = 4 * block_size + 7
    scheduler = _stub_scheduler(
        block_size=block_size,
        use_eagle=False,
        mamba_retention_interval=0,
        mamba_eagle_reach_margin=0,
    )
    request = _request(num_tokens=prompt_len)
    ends = _run_to_completion(scheduler, request, prompt_len)
    assert ends == [4 * block_size, prompt_len]


# ---------------------------------------------------------------------------
# Group 3: EAGLE back-off cases (open design decision -- pins CURRENT shape
# only; deliberately no xfail twin, see module docstring and
# correspondence.md Sec B2 for why)
# ---------------------------------------------------------------------------


def test_true_eagle_back_off_still_applies_today() -> None:
    """PRE-PORT, PASSING TODAY: true EAGLE (retain_mamba_align_mtp_cache_block
    =False, i.e. every non-MTP eagle-family method on this fork today) must
    still pay the one-block back-off -- the invariant doc's load-bearing
    safety property for EAGLE's shifted resume point. Whatever the post-port
    shape turns out to be (keep our narrow MTP gate vs. adopt upstream's
    unconditional removal backed by dense boundary materialization -- an
    OPEN decision per correspondence.md Sec B2, not resolved by this test
    file), true EAGLE must never silently lose this protection without the
    compensating dense-boundary-state guarantee landing atomically alongside
    it. No xfail twin: which post-port shape is correct is itself the
    decision, not a fact this file can assert in advance. An earlier draft of
    this file had a "post-port" xfail test here asserting the back-off is
    gone; it was dropped after hand-verification showed it XPASSed today by
    coincidence (an aligned 3-block prompt makes the legacy back-off boundary
    and the dense-materialization boundary land on the same position) --
    exactly the kind of false-confidence result a strict xfail is supposed to
    catch, and did."""
    block_size = 16
    scheduler = _stub_scheduler(block_size=block_size, use_eagle=True)
    request = _request(num_tokens=3 * block_size)
    # Aligned prompt: legacy back-off drops one full block below the end.
    assert _split(scheduler, request, 3 * block_size) == 2 * block_size
