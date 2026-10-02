# SPDX-License-Identifier: Apache-2.0
"""[FORK][LANE EF] Scheduler progress-invariant guard.

Evidence case: 2026-10-01 17:03:23 the engine died with "repeats may not
contain negative values". Scheduler dump of the step: request 9d24dc5d had
num_computed_tokens=30729, num_output_tokens=1, one scheduled spec token and
num_scheduled_tokens=-2, i.e. num_computed_tokens = num_tokens + 3. The guard
detects the counter disagreement the step it happens and queues the request
for preempt+recompute instead of letting it reach np.repeat().
"""

import types
from types import SimpleNamespace

from vllm.v1.core.sched.scheduler import Scheduler


def _sched(enabled=True):
    s = SimpleNamespace(
        progress_guard_enabled=enabled, _heal_queue=[], num_progress_violations=0
    )
    s._record_progress_violation = types.MethodType(
        Scheduler._record_progress_violation, s
    )
    return s


def _req(computed, num_tokens, prompt=30725, outputs=1):
    return SimpleNamespace(
        request_id="r",
        num_computed_tokens=computed,
        num_tokens=num_tokens,
        num_prompt_tokens=prompt,
        output_token_ids=[1] * outputs,
        use_structured_output=True,
    )


def _check(s, r, pre_computed, pre_nt, n_sched=4, n_spec=3, n_gen=0):
    return Scheduler._check_progress_invariant(
        s, r, pre_computed, pre_nt, n_sched, n_spec, n_gen
    )


def test_healthy_decode_row_passes():
    s = _sched()
    # decode row: before step computed=nt-1; scheduled 4 -> pre_computed=nt+3;
    # 2 accepted + 1 bonus => nt grows by 3, computed rolled back to nt-1.
    r = _req(computed=30727 + 3 - 1, num_tokens=30727 + 3)
    assert _check(s, r, pre_computed=30727 + 3, pre_nt=30727, n_gen=3)
    assert s._heal_queue == [] and s.num_progress_violations == 0


def test_mid_prefill_chunk_not_checked():
    s = _sched()
    r = _req(computed=7136, num_tokens=30725, outputs=0)
    assert _check(s, r, pre_computed=7136, pre_nt=30725, n_sched=3568, n_spec=0)
    assert s.num_progress_violations == 0


def test_final_prefill_chunk_passes():
    s = _sched()
    r = _req(computed=30725, num_tokens=30726)  # one sampled token appended
    assert _check(s, r, pre_computed=30725, pre_nt=30725, n_sched=2000, n_spec=0, n_gen=1)


def test_missing_sample_overshoot_is_healed_17_03_case():
    # Step scheduled 1+3 tokens for a request with nt=30726 (1 output), the
    # worker returned no tokens for the row, so no rejection roll-back ran:
    # computed stays at pre_computed = 30725 + 4 = 30729 while num_tokens is
    # unchanged -> next step would schedule num_tokens_with_spec - computed = -2.
    s = _sched()
    r = _req(computed=30729, num_tokens=30726)
    assert not _check(s, r, pre_computed=30729, pre_nt=30726, n_gen=0)
    assert s._heal_queue == [r]
    assert s.num_progress_violations == 1


def test_gap_after_decode_is_flagged():
    s = _sched()
    r = _req(computed=100, num_tokens=110)
    assert not _check(s, r, pre_computed=108, pre_nt=105, n_gen=5)
    assert s.num_progress_violations == 1


def test_disabled_guard_counts_but_does_not_queue():
    s = _sched(enabled=False)
    r = _req(computed=30729, num_tokens=30726)
    assert not _check(s, r, pre_computed=30729, pre_nt=30726)
    assert s._heal_queue == [] and s.num_progress_violations == 1
