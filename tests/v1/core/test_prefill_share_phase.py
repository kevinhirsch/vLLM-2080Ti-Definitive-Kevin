# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A sub-block prefill must count toward the measured prefill wall-time share."""
from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.scheduler import Scheduler

pytestmark = pytest.mark.cpu_test


def scheduler(exact=True):
    result = Scheduler.__new__(Scheduler)
    result.prefill_share_exact = exact
    result.cache_config = SimpleNamespace(block_size=3568)
    result.requests = {
        "cold": SimpleNamespace(num_prompt_tokens=72000, num_tokens=72000,
                                num_computed_tokens=1024),
        "warm": SimpleNamespace(num_prompt_tokens=32000, num_tokens=32000,
                                num_computed_tokens=31872),
        "decode": SimpleNamespace(num_prompt_tokens=32000, num_tokens=32005,
                                  num_computed_tokens=32004),
        "replay": SimpleNamespace(num_prompt_tokens=16000, num_tokens=32100,
                                  num_computed_tokens=32000),
    }
    return result


@pytest.mark.parametrize("chunk", [128, 512, 1024, 3568])
def test_prefill_phase_covers_sub_block_quanta(chunk):
    assert scheduler()._step_has_prefill({"cold": chunk})


def test_warm_tail_is_prefill_but_speculative_decode_is_not():
    actual = scheduler()
    assert actual._step_has_prefill({"warm": 128, "decode": 4})
    assert not actual._step_has_prefill({"decode": 4})


def test_unscheduled_prefill_does_not_charge_decode_only_step():
    assert not scheduler()._step_has_prefill({"cold": 0, "decode": 4})
    assert not scheduler()._step_has_prefill({})


def test_resumed_output_history_replay_is_prefill_work():
    assert scheduler()._step_has_prefill({"replay": 99})


def test_default_control_keeps_original_block_classifier():
    actual = scheduler(exact=False)
    assert not actual._step_has_prefill({"cold": 1024})
    assert actual._step_has_prefill({"cold": 3568})


def test_sub_block_chunk_activates_wall_time_cooldown():
    actual = scheduler()
    actual.prefill_share_enabled = True
    actual.prefill_share = 0.75
    actual._ps_last_had_chunk = actual._step_has_prefill({"cold": 512, "decode": 4})
    actual._ps_last_t = 10.0
    actual._ps_chunk_secs = 0.0
    actual._ps_cooldown_until = 0.0
    actual.running = [actual.requests["cold"], actual.requests["decode"]]
    assert actual._prefill_share_cooldown(10.6)
    assert actual._ps_chunk_secs == pytest.approx(0.6)
    assert actual._ps_cooldown_until == pytest.approx(10.8)
    assert not actual._prefill_share_cooldown(10.8)
