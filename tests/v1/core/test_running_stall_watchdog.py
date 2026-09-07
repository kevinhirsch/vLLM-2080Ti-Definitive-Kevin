# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Lane running-stall-watchdog (task: "RUNNING-loop progress watchdog").
# Unit tests for SchedulerConfig.running_stall_timeout_s: any RUNNING
# request that makes no observable forward progress (num_computed_tokens
# or num_output_tokens increasing) for longer than the timeout is logged
# ("RUNNING-STALL", one WARNING line) and aborted (FINISHED_ERROR) via
# the same finish_requests() path used for client cancellation/errors.
# See Scheduler._abort_stalled_running_requests and the progress-stamp in
# Scheduler.update_from_output for the implementation under test.
#
# Import-bootstrap deviation from upstream test-tree conventions
# (mirrors tests/v1/core/test_skipped_waiting_deadlock.py from lane
# port-54348): this fork's tests/ tree has no __init__.py anywhere, so
# pytest's default "prepend" import mode collects each test file with no
# package context, and a `from .utils import ...` relative import would
# raise ImportError. Bootstrap sys.path instead of adding tree-wide
# __init__.py files for one test file.

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(__file__))

from utils import create_requests, create_scheduler  # noqa: E402

from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402
from vllm.v1.request import RequestStatus  # noqa: E402

BLOCK_SIZE = 16


def _admit_one(scheduler, num_tokens=BLOCK_SIZE, max_tokens=1000, req_id="r0"):
    """Create+add a single request and run one schedule() to admit it into
    RUNNING, scheduling its whole (single-block) prompt in one chunk so
    the very next update_from_output can already deliver its first
    sampled token (mirrors real decode-after-prefill behavior)."""
    request = create_requests(
        1,
        num_tokens=num_tokens,
        ignore_eos=True,
        max_tokens=max_tokens,
        block_size=BLOCK_SIZE,
        req_ids=[req_id],
    )[0]
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    assert request.status == RequestStatus.RUNNING
    return request, scheduler_output


def _confirm_step(scheduler, scheduler_output, request, new_token_id=1):
    """Feed back a ModelRunnerOutput for `request`'s just-scheduled step,
    producing exactly one new sampled token (real, confirmed decode
    progress) -- as opposed to the speculative num_computed_tokens bump
    _update_after_schedule already made before this is called."""
    model_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[new_token_id]],
    )
    scheduler.update_from_output(scheduler_output, model_output)


@patch("time.monotonic")
def test_disabled_by_default_is_a_no_op(mock_monotonic):
    """running_stall_timeout_s defaults to 0.0: a RUNNING request that
    never makes any observable progress must never be touched, no matter
    how much (mocked) wall-clock time passes."""
    mock_monotonic.return_value = 0.0
    scheduler = create_scheduler(
        max_num_seqs=10,
        max_num_batched_tokens=1000,
        num_blocks=100,
        block_size=BLOCK_SIZE,
    )
    assert scheduler.scheduler_config.running_stall_timeout_s == 0.0
    request, _ = _admit_one(scheduler)

    # Absurdly far in the future, zero confirmed progress in between.
    mock_monotonic.return_value = 1_000_000.0
    scheduler.schedule()

    assert request.status == RequestStatus.RUNNING
    assert request in scheduler.running


@patch("time.monotonic")
def test_stalled_running_request_is_aborted_with_log_line(mock_monotonic):
    """A RUNNING request with zero confirmed progress is left alone right
    up to the timeout, then aborted (FINISHED_ERROR, freed, removed from
    self.running) the first step it's found older than the timeout, with
    a single machine-greppable "RUNNING-STALL" WARNING log line carrying
    the request id, age, num_computed_tokens, prompt length, and the
    current running/waiting counts."""
    mock_monotonic.return_value = 0.0
    scheduler = create_scheduler(
        max_num_seqs=10,
        max_num_batched_tokens=1000,
        num_blocks=100,
        block_size=BLOCK_SIZE,
        running_stall_timeout_s=30.0,
    )
    request, _ = _admit_one(scheduler)
    assert request.last_progress_time == 0.0
    assert request.num_prompt_tokens == BLOCK_SIZE

    # Just under the timeout: untouched.
    mock_monotonic.return_value = 29.9
    scheduler.schedule()
    assert request.status == RequestStatus.RUNNING
    assert request in scheduler.running

    # Just over the timeout, still zero confirmed progress: aborted.
    mock_monotonic.return_value = 30.1
    with patch("vllm.v1.core.sched.scheduler.logger") as mock_logger:
        scheduler.schedule()
        assert mock_logger.warning.call_count == 1
        fmt, *args = mock_logger.warning.call_args[0]
        assert fmt.startswith("RUNNING-STALL")
        logged = fmt % tuple(args)
        assert "RUNNING-STALL" in logged
        assert request.request_id in logged
        assert "age_s=30.1" in logged
        assert f"num_computed_tokens={request.num_computed_tokens}" in logged
        assert f"num_prompt_tokens={BLOCK_SIZE}" in logged
        assert "running=1" in logged  # counts snapshot as-of the check
        assert "waiting=0" in logged

    assert request.status == RequestStatus.FINISHED_ERROR
    assert request not in scheduler.running
    assert len(scheduler.running) == 0
    assert request.request_id not in scheduler.requests  # KV/blocks freed


@patch("time.monotonic")
def test_steady_slow_progress_is_never_aborted(mock_monotonic):
    """A request that keeps making confirmed progress (one new token per
    round-trip) at intervals shorter than the timeout must never be
    aborted, even after cumulative elapsed time far exceeds the
    timeout -- each confirmation resets the clock."""
    mock_monotonic.return_value = 0.0
    timeout_s = 10.0
    scheduler = create_scheduler(
        max_num_seqs=10,
        max_num_batched_tokens=1000,
        num_blocks=100,
        block_size=BLOCK_SIZE,
        running_stall_timeout_s=timeout_s,
    )
    request, scheduler_output = _admit_one(scheduler, max_tokens=1000)

    t = 0.0
    for step in range(6):
        t += timeout_s - 1  # always < timeout since the last confirmation
        mock_monotonic.return_value = t
        _confirm_step(scheduler, scheduler_output, request, new_token_id=1)
        assert request.status == RequestStatus.RUNNING
        scheduler_output = scheduler.schedule()
        assert request.status == RequestStatus.RUNNING

    # Total elapsed (6 * 9 = 54s) is well past the 10s timeout, yet the
    # request is alive because progress kept resetting the clock.
    assert t > timeout_s
    assert request in scheduler.running
    assert request.num_output_tokens == 6


@patch("time.monotonic")
def test_long_chunked_prefill_counts_as_progress(mock_monotonic):
    """A long prompt split into multiple chunked-prefill steps must not
    be aborted partway through: num_computed_tokens growing each step
    (even with zero sampled tokens, since it's not the final chunk) is
    itself progress. Regression guard for the task's explicit warning:
    'never abort a request legitimately waiting on prefill work still
    being scheduled.'"""
    mock_monotonic.return_value = 0.0
    timeout_s = 10.0
    # 3 blocks of prompt, budget of 1 block/step -> 3 prefill steps.
    # max_model_len must be set explicitly here: create_scheduler()
    # defaults it to max_num_batched_tokens, which would otherwise be
    # smaller than this test's 3-block prompt.
    scheduler = create_scheduler(
        max_num_seqs=10,
        max_num_batched_tokens=BLOCK_SIZE,
        max_model_len=BLOCK_SIZE * 3 + 100,
        num_blocks=100,
        block_size=BLOCK_SIZE,
        running_stall_timeout_s=timeout_s,
    )
    request, scheduler_output = _admit_one(
        scheduler, num_tokens=BLOCK_SIZE * 3, max_tokens=1000
    )
    assert request.is_prefill_chunk  # first chunk only, more prompt left

    t = 0.0
    for _ in range(2):  # 2 more chunks to finish the 3-block prompt
        t += timeout_s - 1
        mock_monotonic.return_value = t
        # Chunked prefill: model returns no sampled token for this req.
        model_output = ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[]],
        )
        scheduler.update_from_output(scheduler_output, model_output)
        assert request.status == RequestStatus.RUNNING
        scheduler_output = scheduler.schedule()
        assert request.status == RequestStatus.RUNNING

    assert request.num_computed_tokens == BLOCK_SIZE * 3
    assert not request.is_prefill_chunk
    assert request in scheduler.running


@patch("time.monotonic")
def test_preempted_then_resumed_gets_fresh_clock(mock_monotonic):
    """A preempted-then-resumed request must not inherit its clock from
    before preemption: resuming it resets last_progress_time, so a long
    gap while PREEMPTED (waiting, not running) doesn't retroactively
    count against the RUNNING-stall timeout once it's back."""
    mock_monotonic.return_value = 0.0
    timeout_s = 30.0
    scheduler = create_scheduler(
        max_num_seqs=10,
        max_num_batched_tokens=1000,
        num_blocks=100,
        block_size=BLOCK_SIZE,
        running_stall_timeout_s=timeout_s,
    )
    request, _ = _admit_one(scheduler)
    assert request.last_progress_time == 0.0

    # Preempt well before any stall timeout would fire (simulates
    # resource pressure from another request, not a stall).
    mock_monotonic.return_value = 5.0
    scheduler.running.remove(request)
    scheduler._preempt_request(request, 5.0)
    # Mirrors Scheduler.reset_prefix_cache's own manual-preemption path:
    # forcing preemption+resumption outside a normal schedule() cycle
    # must also clear this, or _make_cached_request_data's "a resumed
    # request wasn't in the prior step's scheduled set" invariant trips
    # on stale bookkeeping from the admission step two schedule() calls
    # ago -- a test-harness artifact of skipping a cycle, not something
    # a real preemption (which always happens inside schedule()) hits.
    scheduler.prev_step_scheduled_req_ids.clear()
    assert request.status == RequestStatus.PREEMPTED

    # A long time passes while PREEMPTED (not RUNNING) -- more than the
    # timeout since the ORIGINAL admission at t=0.
    mock_monotonic.return_value = 500.0
    scheduler.schedule()

    # Only request in the queue with ample blocks/budget: resumed.
    assert request.status == RequestStatus.RUNNING
    assert request.last_progress_time == 500.0  # fresh, not the old 0.0
    assert request.last_progress_num_computed_tokens == 0

    # 20s later (< 30s timeout since the t=500 resume): must survive.
    # Bug check: if the clock had NOT been reset, age would be computed
    # as 520 - 0 = 520s > 30s and this would wrongly abort.
    mock_monotonic.return_value = 520.0
    scheduler.schedule()
    assert request.status == RequestStatus.RUNNING
    assert request in scheduler.running
