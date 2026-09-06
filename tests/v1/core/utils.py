# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# NOTE: This is a lane-local, trimmed-down port of upstream
# `tests/v1/core/utils.py` (as of vllm-project/vllm PR #54348,
# https://github.com/vllm-project/vllm/pull/54348). It intentionally
# drops upstream helper surface (KV-connector mocking, EC-connector,
# speculative-config, multimodal hashing, watermark, etc.) that either
# does not exist on this fork's older/heavily-patched scheduler snapshot,
# or is not exercised by the ported blocked-waiting-deadlock regression
# tests. Only the subset needed by
# tests/v1/core/test_skipped_waiting_deadlock.py is implemented.

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
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

EOS_TOKEN_ID = 50256


def create_scheduler(
    model: str = "facebook/opt-125m",
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    enable_chunked_prefill: bool = True,
    enable_prefix_caching: bool = False,
    long_prefill_token_threshold: int = 0,
    num_blocks: int = 10000,
    block_size: int = 16,
    max_model_len: int | None = None,
) -> Scheduler:
    """Create a minimal scheduler under test (CPU-only, no real model).

    Trimmed subset of upstream's create_scheduler(): only the parameters
    used by the ported blocked-waiting-deadlock regression tests.
    """
    if max_model_len is None:
        max_model_len = max_num_batched_tokens

    model_config = ModelConfig(
        model=model,
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=False,
        max_model_len=max_model_len,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        long_prefill_token_threshold=long_prefill_token_threshold,
        enable_chunked_prefill=enable_chunked_prefill,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=enable_prefix_caching,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        observability_config=ObservabilityConfig(),
    )

    kv_cache_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], kv_cache_spec)],
    )
    cache_config.num_gpu_blocks = num_blocks

    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )
    return scheduler


_none_hash_initialized = False


def create_requests(
    num_requests: int,
    num_tokens: int = 10,
    ignore_eos: bool = False,
    max_tokens: int = 16,
    block_size: int = 16,
    req_ids: list[str] | None = None,
) -> list[Request]:
    global _none_hash_initialized
    if not _none_hash_initialized:
        init_none_hash(sha256)
        _none_hash_initialized = True

    block_hasher = get_request_block_hasher(block_size, sha256)
    sampling_params = SamplingParams(
        ignore_eos=ignore_eos,
        max_tokens=max_tokens,
    )
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    requests = []

    if req_ids:
        assert len(req_ids) == num_requests
    else:
        req_ids = [f"{i}" for i in range(num_requests)]

    for i in range(num_requests):
        prompt_token_ids = [i] * num_tokens
        request = Request(
            request_id=req_ids[i],
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            pooling_params=None,
            mm_features=None,
            block_hasher=block_hasher,
        )
        requests.append(request)
    return requests
