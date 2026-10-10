# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-wide prompt work bounds, including hybrid sub-block/tail progress.

These use the real scheduler/KV manager and a fake CPU worker. They establish
scheduling contracts, not GDN numerical equivalence or an engine speedup.
"""
from math import ceil
from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from . import test_prefill_batch_barrier_mamba_align as fixtures
from .test_prefill_batch_barrier_mamba_align import (
    _build_scheduler,
    _model_output,
    _request,
)

pytestmark = pytest.mark.cpu_test


def _scheduler(monkeypatch, quota=512):
    monkeypatch.setenv("VLLM_SCHED_MIXED_PREFILL_QUOTA", str(quota))
    result = _build_scheduler(max_num_batched_tokens=3632, max_num_seqs=8)
    # CR2 cohort barriers are a separate candidate, not a prerequisite here.
    result.prefill_batch_barrier = False
    result.short_first_enabled = False
    result.prefill_share_enabled = False
    return result


def _step(scheduler):
    before = {r.request_id: r.num_computed_tokens for r in scheduler.running}
    output = scheduler.schedule()
    # Waiting APC admission can credit cached tokens. The scheduler's computed
    # position after schedule minus its query length is the true query start.
    work = {
        rid: scheduler._prefill_work(
            scheduler.requests[rid],
            scheduler.requests[rid].num_computed_tokens - n,
            n,
        )
        for rid, n in output.num_scheduled_tokens.items()
    }
    if output.num_scheduled_tokens:
        scheduler.update_from_output(output, _model_output(scheduler, output))
    return output, work, before


def _decoder(scheduler):
    request = _request("decode0", 31)
    scheduler.add_request(request)
    _step(scheduler)
    assert request.num_output_tokens == 1
    assert scheduler._runnable_decode_for_quota(False)
    return request


@pytest.mark.parametrize("quota", [256, 512])
def test_running_and_waiting_prefills_share_one_quota(monkeypatch, quota):
    scheduler = _scheduler(monkeypatch, quota)
    decoder = _decoder(scheduler)
    first = _request("running0", 800)
    second = _request("waiting1", 700)
    scheduler.add_request(first)
    out, work, _ = _step(scheduler)
    assert sum(work.values()) == quota
    assert first in scheduler.running
    scheduler.add_request(second)
    saw_shared_tail = False
    for _ in range(10):
        was_decode = not decoder.is_finished()
        out, work, _ = _step(scheduler)
        if was_decode:
            assert sum(work.values()) <= quota
            assert decoder.request_id in out.num_scheduled_tokens
        if work.get(first.request_id, 0) and work.get(second.request_id, 0):
            saw_shared_tail = True
        if first.num_output_tokens and second.num_output_tokens:
            break
    assert saw_shared_tail, "a running tail and waiting prompt never shared a step"
    assert first.num_output_tokens and second.num_output_tokens


def test_many_short_waiting_prefills_do_not_each_get_the_quota(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    _decoder(scheduler)
    prompts = [_request(f"short{i}", 200) for i in range(5)]
    for request in prompts:
        scheduler.add_request(request)
    output, work, _ = _step(scheduler)
    assert sum(work.values()) == 512
    assert len([n for n in work.values() if n]) == 3
    assert sum(output.num_scheduled_tokens.values()) < 3632


@pytest.mark.parametrize("quota", [0, 256, 512])
def test_no_decode_keeps_larger_prefill_batches(monkeypatch, quota):
    scheduler = _scheduler(monkeypatch, quota)
    request = _request("cold0", 10000)
    scheduler.add_request(request)
    output, work, _ = _step(scheduler)
    assert output.num_scheduled_tokens[request.request_id] == 3200
    assert sum(work.values()) > 512


def test_disabled_quota_preserves_mixed_baseline(monkeypatch):
    scheduler = _scheduler(monkeypatch, 0)
    _decoder(scheduler)
    scheduler.add_request(_request("cold1", 10000))
    _, work, _ = _step(scheduler)
    assert sum(work.values()) == 3200


@pytest.mark.parametrize("quota", [256, 512])
@pytest.mark.parametrize("cache_block", [1600, 3568])
def test_sub_block_prefill_crosses_cache_boundaries_without_starving(
    monkeypatch, quota, cache_block
):
    monkeypatch.setattr(fixtures, "MAMBA_BLOCK", cache_block)
    monkeypatch.setattr(fixtures, "ATTN_BLOCK", cache_block)
    scheduler = _scheduler(monkeypatch, quota)
    scheduler.num_prefill_lookahead = 3
    decoder = _decoder(scheduler)
    # Keep a decode row active until the entire prompt has been admitted.
    decoder.max_tokens = 256
    prompt_len = 3 * cache_block + 332
    request = _request("long1", prompt_len)
    scheduler.add_request(request)
    starts = []
    for _ in range(80):
        output, work, before = _step(scheduler)
        assert output.num_scheduled_tokens
        assert sum(work.values()) <= quota
        assert decoder.request_id in output.num_scheduled_tokens
        if request.request_id in work:
            starts.append((before.get(request.request_id, 0),
                           work[request.request_id]))
        if request.num_output_tokens:
            break
    assert request.num_output_tokens, "prefill starved below the Mamba block size"
    ends = [start + n for start, n in starts]
    assert cache_block in ends and 2 * cache_block in ends
    assert len(starts) <= ceil(prompt_len / quota) + 4


def test_prompt_work_counts_replayed_history_but_not_draft_padding():
    request = SimpleNamespace(num_prompt_tokens=100, num_tokens=181)
    assert Scheduler._prefill_work(request, 160, 24) == 20
    assert Scheduler._prefill_work(request, 180, 4) == 0
    scheduler = Scheduler.__new__(Scheduler)
    assert scheduler._limit_mixed_prefill(request, 160, 24, 20) == 24
    assert scheduler._limit_mixed_prefill(request, 160, 24, 19) == 19


@pytest.mark.parametrize("reason", ["cadence", "max_tokens", "context", "no_query"])
def test_unschedulable_decode_does_not_throttle_prefill(monkeypatch, reason):
    scheduler = _scheduler(monkeypatch)
    decoder = _decoder(scheduler)
    if reason == "cadence":
        decoder.next_decode_eligible_step = scheduler.current_step + 100
    elif reason == "max_tokens":
        decoder.num_output_placeholders = 1
        decoder.max_tokens = 1
    elif reason == "context":
        decoder.num_computed_tokens = scheduler.max_model_len
    else:
        decoder.num_computed_tokens = decoder.num_tokens_with_spec
    assert not scheduler._runnable_decode_for_quota(False)
    scheduler.add_request(_request("cold1", 10000))
    _, work, _ = _step(scheduler)
    assert sum(work.values()) == 3200


def test_warm_waiting_first_token_is_not_yet_decode_priority(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    first = _request("waiting0", 200)
    second = _request("waiting1", 700)
    scheduler.add_request(first)
    scheduler.add_request(second)
    assert not scheduler._runnable_decode_for_quota(False)
    _, work, _ = _step(scheduler)
    assert sum(work.values()) == 900


def test_prefix_cache_credit_is_not_charged_as_new_prompt_work(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    donor = _request("donor0", 5000)
    donor.max_tokens = 1
    scheduler.add_request(donor)
    for _ in range(10):
        _step(scheduler)
        if donor.is_finished():
            break
    assert donor.is_finished()
    _decoder(scheduler)
    warm = _request("warm0", 5200)  # Identical constant-token cached prefix.
    scheduler.add_request(warm)
    output, work, _ = _step(scheduler)
    query_start = (
        warm.num_computed_tokens - output.num_scheduled_tokens[warm.request_id]
    )
    assert query_start >= 1600, "fixture did not exercise an APC hit"
    assert work[warm.request_id] == 512
    assert sum(work.values()) == 512


@pytest.mark.parametrize(
    "start,budget,junction,expected",
    [(1536, 512, 0, 64), (0, 512, 256, 256), (0, 256, 0, 256)],
)
def test_quota_preserves_mandatory_state_stops(
    monkeypatch, start, budget, junction, expected
):
    scheduler = _scheduler(monkeypatch)
    request = _request("split0", 5000)
    request.num_computed_tokens = start
    request.shared_prefix_boundary = junction
    scheduler.mamba_fine_grained_prefix_cache = bool(junction)
    assert scheduler._mamba_block_aligned_split(
        request, budget, prefill_chunk_limit=budget
    ) == expected


def test_decode_tail_keeps_full_draft_group_when_prompt_quota_is_zero(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    decoder = _decoder(scheduler)
    request = _request("prompt1", 10000)
    scheduler.add_request(request)
    _step(scheduler)
    # Put the running prompt first to spend the full cap before the decoder.
    scheduler.running.reverse()
    decoder.spec_token_ids = [3, 4, 5]
    # This test covers scheduler query shape only, without fabricating a
    # worker's three-token acceptance record in the one-lookahead fixture.
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[request.request_id] == 512
    assert output.num_scheduled_tokens[decoder.request_id] == 4


def test_quota_requires_explicit_chunked_prefill(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    scheduler.scheduler_config.enable_chunked_prefill = False
    with pytest.raises(ValueError, match="requires chunked prefill"):
        Scheduler(
            vllm_config=scheduler.vllm_config,
            kv_cache_config=scheduler.kv_cache_config,
            structured_output_manager=scheduler.structured_output_manager,
            block_size=scheduler.block_size,
        )


@pytest.mark.parametrize("value", ["-1", "invalid"])
def test_invalid_quota_fails_at_construction(monkeypatch, value):
    monkeypatch.setenv("VLLM_SCHED_MIXED_PREFILL_QUOTA", value)
    with pytest.raises(ValueError):
        _build_scheduler()


def test_priority_preemption_refunds_already_scheduled_prompt_work(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    scheduler.policy = SchedulingPolicy.PRIORITY
    decoder = _decoder(scheduler)
    low = _request("low1", 200)
    high = _request("high0", 700)
    later = _request("later1", 700)
    # Admit both without consuming the following step's prompt quota.
    for request in (low, high, later):
        request.status = RequestStatus.RUNNING
        scheduler.requests[request.request_id] = request
        scheduler.running.append(request)
    low.priority, high.priority, later.priority, decoder.priority = 10, 0, 5, -1
    real_allocate = scheduler.kv_cache_manager.allocate_slots
    failed = False

    def allocate(request, *args, **kwargs):
        nonlocal failed
        if request is high and not failed:
            failed = True
            return None
        return real_allocate(request, *args, **kwargs)

    monkeypatch.setattr(scheduler.kv_cache_manager, "allocate_slots", allocate)
    output, work, _ = _step(scheduler)
    assert failed and low.num_preemptions == 1
    assert low.request_id not in output.num_scheduled_tokens
    assert sum(work.values()) <= 512
    # 200 was reserved for low before its preemption. The high request's
    # already clipped chunk is 312; refund allows a later running prompt.
    assert output.num_scheduled_tokens[high.request_id] == 312
    assert output.num_scheduled_tokens[later.request_id] == 200
    assert sum(work.values()) == 512
