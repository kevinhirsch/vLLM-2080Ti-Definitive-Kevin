# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

from vllm.v1.core.sched.scheduler import Scheduler


def _split_with_mtp_cache_retention(enabled: bool, num_tokens: int = 14) -> int:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.use_eagle = True
    scheduler.retain_mamba_align_mtp_cache_block = enabled
    scheduler.cache_config = SimpleNamespace(block_size=4)
    # [FORK] vllm-project/vllm#54076: fake scheduler needs its own
    # mamba_state_block_size now that _mamba_block_aligned_split reads it
    # instead of cache_config.block_size directly; same value here since
    # this test's scenario is homogeneous (single KV-cache-group geometry).
    scheduler.mamba_state_block_size = 4
    # Pre-existing gap (unrelated to #54076/#53802): this fixture predates
    # the F-1 retention-interval feature and never set these; dense/off
    # matches the fork's actual default and this test's original intent.
    scheduler.mamba_retention_interval = None
    scheduler.mamba_eagle_reach_margin = 0
    request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=num_tokens,
        num_tokens=num_tokens,
    )
    return scheduler._mamba_block_aligned_split(request, num_new_tokens=14)


def test_mamba_align_mtp_can_retain_final_cache_boundary() -> None:
    """MTP should not discard a complete Mamba block before the prompt tail."""
    # The legacy EAGLE rule drops the final complete block (12 -> 8).
    assert _split_with_mtp_cache_retention(enabled=False) == 8
    # MTP still computes the two-token tail, so the 12-token state is valid.
    assert _split_with_mtp_cache_retention(enabled=True) == 12
    # Without a tail, preserve EAGLE's safety block.
    #
    # [FORK] vllm-project/vllm#53802: the highest position any future lookup
    # can register a hit at is round_down(num_tokens - 1, block_size) = 8
    # here (KVCacheManager.get_computed_blocks caps lookups at
    # num_tokens - 1 = 11), not round_down(num_tokens, block_size) = 12.
    # EAGLE's one-block safety margin is measured from that corrected
    # boundary (8), landing on 4 -- not from the pre-#53802 value (12),
    # which happened to coincide with the lookup cap itself and so gave a
    # "safety" margin of zero. Expectation updated 8 -> 4 to match.
    assert _split_with_mtp_cache_retention(enabled=True, num_tokens=12) == 4
