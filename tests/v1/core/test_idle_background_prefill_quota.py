# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_SCHED_IDLE_BACKGROUND_PREFILL_QUOTA: the batch-wide bound on positive-priority prompt work while no decoder is
ready ("Kevin first": background prefills cannot fill the step an urgent prompt needs).

Real scheduler/KV manager with the fake CPU worker of the mixed-quota tests. Scheduling contracts only.
"""
import pytest

from vllm.v1.core.sched.scheduler import Scheduler

from . import test_prefill_batch_barrier_mamba_align as fixtures
from .test_mixed_prefill_quota import _step
from .test_prefill_batch_barrier_mamba_align import _build_scheduler, _request

pytestmark = pytest.mark.cpu_test

BACKGROUND, URGENT = 5, 0          # vLLM priority: lower runs first; the gateway sends background as positive


def _idle_scheduler(monkeypatch, quota, mixed_quota=0):
    base = _build_scheduler(max_num_batched_tokens=3632, max_num_seqs=8)
    base.scheduler_config.policy = "priority"
    monkeypatch.setenv("VLLM_SCHED_IDLE_BACKGROUND_PREFILL_QUOTA", str(quota))
    if mixed_quota:
        monkeypatch.setenv("VLLM_SCHED_MIXED_PREFILL_QUOTA", str(mixed_quota))
        monkeypatch.setenv("VLLM_SCHED_MIXED_PREFILL_PRIORITY_ONLY", "1")
    s = Scheduler(vllm_config=base.vllm_config, kv_cache_config=base.kv_cache_config,
                  structured_output_manager=base.structured_output_manager, block_size=base.block_size,
                  log_stats=True)
    s.num_prefill_lookahead = fixtures.NUM_SPEC
    s.use_eagle = True
    s.use_eagle_block_drop = True
    s.prefill_batch_barrier = False
    s.short_first_enabled = False
    s.prefill_share_enabled = False
    return s


def _req(rid, n, priority):
    r = _request(rid, n)
    r.priority = priority
    return r


@pytest.mark.parametrize("quota", [256, 512])
def test_background_prompt_work_shares_one_idle_quota(monkeypatch, quota):
    s = _idle_scheduler(monkeypatch, quota)
    for i, n in enumerate((800, 700, 900)):
        s.add_request(_req(f"bg{i}", n, BACKGROUND))
    _, work, _ = _step(s)
    assert 0 < sum(work.values()) <= quota          # a batch-wide quota, not one quota per request


def test_urgent_prompt_is_not_capped_by_the_idle_quota(monkeypatch):
    s = _idle_scheduler(monkeypatch, 512)
    bg, urgent = _req("bg0", 1500, BACKGROUND), _req("kevin0", 1500, URGENT)
    s.add_request(bg)
    s.add_request(urgent)
    _, work, _ = _step(s)
    assert work.get(urgent.request_id, 0) > 512
    assert work.get(bg.request_id, 0) <= 512


def test_background_alone_is_capped_but_completes(monkeypatch):
    s = _idle_scheduler(monkeypatch, 512)
    bg = _req("bg0", 1500, BACKGROUND)
    s.add_request(bg)
    for _ in range(20):
        idle, prefilling = not s._runnable_decode_for_quota(False), not bg.num_output_tokens
        _, work, _ = _step(s)
        if idle and prefilling:
            assert sum(work.values()) <= 512
        if bg.num_output_tokens:
            break
    assert bg.num_output_tokens, "a capped background prompt must still finish its prefill"


def test_ready_decoder_hands_over_to_the_mixed_quota(monkeypatch):
    s = _idle_scheduler(monkeypatch, 512, mixed_quota=256)
    dec = _req("decode0", 31, URGENT)
    s.add_request(dec)
    _step(s)
    assert dec.num_output_tokens == 1
    bg = _req("bg0", 1500, BACKGROUND)
    s.add_request(bg)
    out, work, _ = _step(s)
    assert dec.request_id in out.num_scheduled_tokens           # the ready decoder keeps its step
    assert work.get(bg.request_id, 0) <= 256                    # the mixed quota (256) governs, not the idle one (512)


@pytest.mark.parametrize("value", ["-1", "invalid"])
def test_invalid_idle_quota_fails_at_construction(monkeypatch, value):
    base = _build_scheduler()
    base.scheduler_config.policy = "priority"
    monkeypatch.setenv("VLLM_SCHED_IDLE_BACKGROUND_PREFILL_QUOTA", value)
    with pytest.raises(ValueError):
        Scheduler(vllm_config=base.vllm_config, kv_cache_config=base.kv_cache_config,
                  structured_output_manager=base.structured_output_manager, block_size=base.block_size)


def test_idle_quota_requires_priority_scheduling(monkeypatch):
    base = _build_scheduler()
    monkeypatch.setenv("VLLM_SCHED_IDLE_BACKGROUND_PREFILL_QUOTA", "512")
    with pytest.raises(ValueError, match="requires priority scheduling"):
        Scheduler(vllm_config=base.vllm_config, kv_cache_config=base.kv_cache_config,
                  structured_output_manager=base.structured_output_manager, block_size=base.block_size)
