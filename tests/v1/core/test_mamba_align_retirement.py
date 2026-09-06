# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[FORK] Adapted port of vllm-project/vllm#55450's regression test
("[Bugfix][Core] Retire Mamba states across null gaps").

Upstream's PR overrides a shared `_remove_blocks_in_range` helper on its V2
`MambaManager`. This fork's `SingleTypeKVCacheManager` has no such helper --
the equivalent top-down, stop-at-first-null scan is inlined directly in the
base class's `remove_skipped_blocks`. `MambaManager._retire_align_blocks`
(vllm/v1/core/single_type_kv_cache_manager.py) replaces that scan for align
mode only, so this test targets it directly rather than a
`_remove_blocks_in_range` override that does not exist on this fork.

CPU-only: pure Python block-list bookkeeping, no CUDA/model involved.
"""

import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import MambaSpec


def _make_manager(block_size: int = 4) -> tuple[MambaManager, BlockPool]:
    spec = MambaSpec(
        block_size=block_size,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    pool = BlockPool(num_gpu_blocks=8, enable_caching=False, hash_block_size=block_size)
    manager = MambaManager(
        spec,
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
    )
    return manager, pool


def test_align_retirement_skips_over_null_gaps_instead_of_stopping():
    """A null block that is NOT the result of a previous retirement call
    (here: a leading gap, standing in for one `find_longest_cache_hit`
    inserts on a cache hit, or one sparse retention leaves behind) must not
    stop the scan before it reaches a real block that still needs freeing.
    """
    manager, pool = _make_manager()
    old, committed, in_flight = pool.get_new_blocks(3)
    manager.req_to_blocks["r"] = [old, pool.null_block, committed, in_flight]

    # Idempotent across repeat calls (this runs every decode step): the
    # second call must be a no-op, not a re-scan or a double-free.
    for _ in range(2):
        manager.remove_skipped_blocks("r", num_computed_tokens=12)
        assert manager.req_to_blocks["r"] == [
            pool.null_block,
            pool.null_block,
            committed,
            in_flight,
        ]
        assert old.ref_cnt == 0
        assert committed.ref_cnt == in_flight.ref_cnt == 1
        assert pool.get_num_free_blocks() == 5

    # A fresh request reusing the same id after `free()` must not inherit
    # the previous request's retirement high-water mark.
    manager.free("r")
    manager.req_to_blocks["r"] = pool.get_new_blocks(2)
    manager.remove_skipped_blocks("r", num_computed_tokens=5)
    assert manager.req_to_blocks["r"][0].is_null
    assert manager.req_to_blocks["r"][1].ref_cnt == 1


def test_align_retirement_matches_base_scan_when_no_gaps_present():
    """No-gap case: behaviour must be unchanged from the base class's scan
    (same end state as the pre-existing generic
    `SingleTypeKVCacheManager.remove_skipped_blocks`).
    """
    manager, pool = _make_manager()
    a, b, c, d = pool.get_new_blocks(4)
    manager.req_to_blocks["r"] = [a, b, c, d]

    manager.remove_skipped_blocks("r", num_computed_tokens=9)  # skip 2 blocks

    assert manager.req_to_blocks["r"] == [pool.null_block, pool.null_block, c, d]
    assert a.ref_cnt == 0
    assert b.ref_cnt == 0


def test_non_align_mamba_cache_mode_is_untouched():
    """mamba_cache_mode != "align" must keep calling the base class's scan
    (this fix is align-mode only, per upstream's own scope)."""
    spec = MambaSpec(
        block_size=4,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="none",
    )
    pool = BlockPool(num_gpu_blocks=8, enable_caching=False, hash_block_size=4)
    manager = MambaManager(
        spec,
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
    )
    assert not hasattr(manager, "_num_retired_blocks")
    a, b = pool.get_new_blocks(2)
    manager.req_to_blocks["r"] = [a, b]
    manager.remove_skipped_blocks("r", num_computed_tokens=5)
    assert manager.req_to_blocks["r"][0].is_null
    assert a.ref_cnt == 0
