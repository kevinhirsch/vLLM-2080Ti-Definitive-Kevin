# SPDX-License-Identifier: Apache-2.0
"""[FORK][LANE EF2] Short-first interleave: decision helper.

Evidence: with mamba-align every non-final prefill chunk is one 3568-token block, so a running 28K prefill takes the whole
3584-token step budget each step and a newly arrived short request waits out the entire prefill (decoder TTFT median
19-21 s). The helper decides, per step, whether the long prefill sits the step out; the step after is always normal.
"""

import types
from types import SimpleNamespace

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

BLOCK = 3568


def _req(num_tokens, computed=0, prompt=None, status=RequestStatus.WAITING):
    return SimpleNamespace(
        num_tokens=num_tokens,
        num_computed_tokens=computed,
        num_prompt_tokens=prompt if prompt is not None else num_tokens,
        status=status,
    )


def _sched(running, waiting, skipped=(), enabled=True, last=False, max_run=1):
    s = SimpleNamespace(
        short_first_enabled=enabled,
        _short_first_streak=1 if last else 0,
        short_first_max_run=max_run,
        cache_config=SimpleNamespace(block_size=BLOCK),
        running=list(running),
        waiting=list(waiting),
        skipped_waiting=list(skipped),
    )
    s._prefill_remaining = types.MethodType(Scheduler._prefill_remaining, s)
    return s


def _go(s):
    return Scheduler._short_first_should_yield(s)


LONG_RUNNING = _req(28000, computed=7136)  # 20.8K of prefill left
DECODE = _req(900, computed=899, prompt=300)  # decode phase: computed = num_tokens-1


def test_yields_when_short_waits_behind_running_long_prefill():
    assert _go(_sched([LONG_RUNNING, DECODE], [_req(900)]))


def test_never_two_yield_steps_in_a_row():
    assert not _go(_sched([LONG_RUNNING], [_req(900)], last=True))


def test_no_yield_without_a_waiting_short():
    assert not _go(_sched([LONG_RUNNING], []))
    assert not _go(_sched([LONG_RUNNING], [_req(40000)]))  # a waiting LONG is not a reason


def test_no_yield_without_a_running_long_prefill():
    assert not _go(_sched([DECODE], [_req(900)]))
    # a prefill whose remainder fits one block is a tail: it completes this step, nothing to yield
    assert not _go(_sched([_req(28000, computed=28000 - 3000)], [_req(900)]))


def test_short_in_skipped_waiting_counts_and_cached_prefix_ignored():
    assert _go(_sched([LONG_RUNNING], [_req(40000)], skipped=[_req(500)]))
    # blocked statuses (remote KV etc.) do not count
    assert not _go(_sched([LONG_RUNNING], [_req(500, status=RequestStatus.WAITING_FOR_REMOTE_KVS)]))


def test_kill_switch():
    assert not _go(_sched([LONG_RUNNING], [_req(900)], enabled=False))


def test_scan_is_bounded():
    assert not _go(_sched([LONG_RUNNING], [_req(40000)] * 16 + [_req(500)]))


def test_max_consecutive_yields_is_configurable():
    s = _sched([LONG_RUNNING], [_req(900)], max_run=3)
    s._short_first_streak = 2
    assert _go(s)
    s._short_first_streak = 3
    assert not _go(s)


# ---------------------------------------------------------------- prefill/decode time slicing
def _ps(running, share=0.75, enabled=True):
    s = SimpleNamespace(
        prefill_share=share,
        prefill_share_enabled=enabled,
        _ps_last_t=100.0,
        _ps_last_had_chunk=True,
        _ps_chunk_secs=0.0,
        _ps_cooldown_until=0.0,
        running=list(running),
    )
    s._prefill_remaining = types.MethodType(Scheduler._prefill_remaining, s)
    return s


def _cool(s, now):
    return Scheduler._prefill_share_cooldown(s, now)


def test_cooldown_after_a_chunk_step_gives_decode_rows_their_share():
    s = _ps([LONG_RUNNING, DECODE], share=0.75)
    # previous step (a chunk) lasted 3.3 s; share 0.75 -> long chunks sit out 3.3*0.25/0.75 = 1.1 s
    assert _cool(s, 103.3)
    assert abs(s._ps_cooldown_until - (103.3 + 1.1)) < 1e-6
    assert _cool(s, 104.0)  # still inside the window (no new chunk happened)
    assert not _cool(s, 104.5)  # window over -> next step may carry a chunk


def test_no_cooldown_without_decode_rows_or_when_disabled():
    assert not _cool(_ps([LONG_RUNNING]), 103.3)  # nothing to protect
    assert not _cool(_ps([LONG_RUNNING, DECODE], enabled=False), 103.3)


def test_chunk_estimate_is_smoothed_and_clamped():
    s = _ps([LONG_RUNNING, DECODE], share=0.5)
    _cool(s, 100.0 + 3.0)
    assert abs(s._ps_chunk_secs - 3.0) < 1e-9
    s._ps_last_t, s._ps_last_had_chunk = 200.0, True
    _cool(s, 200.0 + 999.0)  # an idle gap must not poison the estimate
    assert s._ps_chunk_secs <= 0.5 * 3.0 + 0.5 * 10.0 + 1e-9
