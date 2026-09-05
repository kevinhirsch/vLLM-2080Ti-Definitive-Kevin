# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

from vllm.v1.core.sched.scheduler import Scheduler


def _split_with_mtp_cache_retention(enabled: bool, num_tokens: int = 14) -> int:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.use_eagle = True
    scheduler.retain_mamba_align_mtp_cache_block = enabled
    scheduler.cache_config = SimpleNamespace(block_size=4)
    # [FORK][LANE f1-provenance] (/home/kevin/projects/lanes/f1-provenance)
    # This stub predates the 2026-08-27 port (25436c5a0, vLLM #53479) that
    # added these two attributes as unconditionally-read Scheduler state
    # (see _mamba_block_aligned_split's "stops" tuple below the retain_
    # final_mtp_block branch this test targets). Set them to the SAME
    # defaults the real Scheduler.__init__ computes:
    #   mamba_retention_interval = coordinator.retention_interval, i.e.
    #     envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL, which defaults to None
    #     (dense caching -- this fork's default, unset in this test env).
    #   mamba_eagle_reach_margin = coordinator.eagle_reach_margin, which
    #     for a hybrid model is "one full attention block" (see
    #     HybridKVCacheCoordinator.eagle_reach_margin) -- here that's this
    #     same block_size=4, since this fork requires full-attention and
    #     Mamba-align block sizes to match.
    # With retention=None, `boundary_stop`/`replay_boundary`/`eagle_reach`
    # all collapse to inert values regardless of the margin (see the `if
    # retention is not None:` guard), so this test's assertions do not
    # actually depend on the margin's specific value -- it is still set to
    # its realistic default rather than an arbitrary placeholder, so this
    # stub keeps matching real Scheduler state as that function evolves.
    scheduler.mamba_retention_interval = None
    scheduler.mamba_eagle_reach_margin = scheduler.cache_config.block_size
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
    assert _split_with_mtp_cache_retention(enabled=True, num_tokens=12) == 8
