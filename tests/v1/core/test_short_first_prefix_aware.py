# SPDX-License-Identifier: Apache-2.0
"""[FORK][LANE CR2] Short-first judges a waiting request by its UNCACHED tokens.

Live evidence (L101): the short-first yield tested ``num_tokens - num_computed_tokens``
of waiting requests, and ``num_computed_tokens`` stays 0 until admission. A 30K warm
tool-loop continuation with ~1K uncached tokens therefore never counted as short and
waited out every cold long prefill ahead of it. With
VLLM_SCHED_SHORT_FIRST_PREFIX_AWARE=1 the decision probes the local prefix cache.

These tests drive the REAL Scheduler + KVCacheManager (hybrid full-attention + Mamba
"align" groups, CPU only).
"""

import os

import pytest
import torch

from vllm.config import (
    CacheConfig,
    ModelConfig,
    ObservabilityConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test

MODEL = os.environ.get("VLLM_SCHED_TEST_MODEL", "facebook/opt-125m")
_LOCAL_MODEL = os.path.isdir(MODEL)
EOS = 50256
BLOCK = 256
BUDGET = BLOCK + 16  # production shape: the step budget is one block plus a little
MAX_LEN = 16384


def _build(monkeypatch, prefix_aware: bool) -> Scheduler:
    monkeypatch.setenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    monkeypatch.setenv("VLLM_SCHED_SHORT_FIRST", "1")
    monkeypatch.setenv("VLLM_SCHED_SHORT_FIRST_RUN", "1")
    monkeypatch.setenv("VLLM_SCHED_PREFILL_SHARE", "1.0")
    monkeypatch.setenv("VLLM_SCHED_SHORT_FIRST_PREFIX_AWARE", "1" if prefix_aware else "0")
    model_config = ModelConfig(
        model=MODEL,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=not _LOCAL_MODEL,
        max_model_len=MAX_LEN,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=8,
        max_num_batched_tokens=BUDGET,
        max_model_len=MAX_LEN,
        enable_chunked_prefill=True,
        long_prefill_token_threshold=0,
        watermark=0.0,
        is_encoder_decoder=False,
    )
    cache_config = CacheConfig(
        block_size=BLOCK,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=True,
        mamba_cache_mode="align",
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        observability_config=ObservabilityConfig(),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=2048,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=BLOCK,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    cache_config.num_gpu_blocks = kv_cache_config.num_blocks
    register_all_kvcache_specs(vllm_config)
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def _request(rid: str, tokens: list[int], max_tokens: int = 2) -> Request:
    init_none_hash(sha256)
    sp = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    sp.update_from_generation_config({}, EOS)
    return Request(
        request_id=rid,
        prompt_token_ids=tokens,
        sampling_params=sp,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )


def _step(s: Scheduler):
    out = s.schedule()
    ids, sampled = [], []
    for rid in out.num_scheduled_tokens:
        r = s.requests[rid]
        ids.append(rid)
        sampled.append([7] if r.num_computed_tokens >= r.num_tokens else [])
    s.update_from_output(
        out,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={rid: i for i, rid in enumerate(ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    return out


SESSION = [1000 + i % 977 for i in range(6 * BLOCK + 100)]  # 6 blocks + 100-token tail
COLD = [5 + i % 991 for i in range(12 * BLOCK)]


def _warm_then_cold(s: Scheduler) -> tuple[Request, Request]:
    """Turn 1 of a session finishes (its prefix is cached), then a cold 12-block
    prefill starts. Returns (cold, continuation) with the continuation queued."""
    t1 = _request("turn1", SESSION)
    s.add_request(t1)
    for _ in range(40):
        _step(s)
        if t1.is_finished():
            break
    assert t1.is_finished()
    cold = _request("cold", COLD)
    s.add_request(cold)
    _step(s)
    assert cold.num_computed_tokens == BLOCK, "cold prefill should be mid-prompt"
    # Turn 2: the whole of turn 1's prompt plus a 150-token tool result.
    t2 = _request("turn2", SESSION + [3] * 150)
    s.add_request(t2)
    return cold, t2


def test_probe_matches_admission_hit(monkeypatch):
    s = _build(monkeypatch, prefix_aware=True)
    cold, t2 = _warm_then_cold(s)
    hit = s.kv_cache_manager.probe_prefix_cache_hit(t2)
    assert hit == 6 * BLOCK
    _, admitted_hit, _ = s.kv_cache_manager.get_computed_blocks(t2)
    assert admitted_hit == hit
    # read-only: probing does not take block references or move the LRU
    free = s.kv_cache_manager.block_pool.get_num_free_blocks()
    s.kv_cache_manager.probe_prefix_cache_hit(t2)
    assert s.kv_cache_manager.block_pool.get_num_free_blocks() == free


def test_warm_continuation_overtakes_running_cold_prefill(monkeypatch):
    s = _build(monkeypatch, prefix_aware=True)
    cold, t2 = _warm_then_cold(s)
    out = _step(s)
    assert "cold" not in out.num_scheduled_tokens, "cold chunk should yield one step"
    assert out.num_scheduled_tokens.get("turn2") == len(t2.prompt_token_ids) - 6 * BLOCK
    assert s.num_short_first_prefix_yields == 1
    out = _step(s)  # never two yields in a row: the cold prefill resumes
    assert out.num_scheduled_tokens.get("cold") == BLOCK


def test_default_off_keeps_previous_behavior(monkeypatch):
    s = _build(monkeypatch, prefix_aware=False)
    cold, t2 = _warm_then_cold(s)
    out = _step(s)
    assert out.num_scheduled_tokens.get("cold") == BLOCK
    assert "turn2" not in out.num_scheduled_tokens
    assert s.num_short_first_prefix_yields == 0


def test_warm_but_long_uncached_does_not_yield(monkeypatch):
    """A continuation whose uncached remainder exceeds a block is not short."""
    s = _build(monkeypatch, prefix_aware=True)
    cold, t2 = _warm_then_cold(s)
    s.finish_requests("turn2", s.requests["turn2"].status.FINISHED_ABORTED)
    t3 = _request("turn3", SESSION + [4] * (2 * BLOCK))
    s.add_request(t3)
    out = _step(s)
    assert out.num_scheduled_tokens.get("cold") == BLOCK
    assert s.num_short_first_prefix_yields == 0
