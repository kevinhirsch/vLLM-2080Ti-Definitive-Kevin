# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace


from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.v1.core.sched.scheduler import Scheduler



def test_hybrid_mamba_external_hit_is_aligned_and_does_not_crash():
    """External hybrid hit is block-aligned and accepted by Mamba scheduling."""
    offload_scheduler = object.__new__(OffloadingConnectorScheduler)
    offload_scheduler._sliding_window_groups = (1,)
    offload_scheduler._lookup_groups = (0, 1)
    offload_scheduler._mamba_align_size = 16
    offload_scheduler._blocks_being_loaded = None

    class _AllHitManager:
        def lookup(self, key, req_context):
            return True

    offload_scheduler.manager = _AllHitManager()
    offload_scheduler.config = SimpleNamespace(
        kv_group_configs=(
            SimpleNamespace(
                offloaded_block_size=32, sliding_window_size_in_blocks=None
            ),
            SimpleNamespace(offloaded_block_size=16, sliding_window_size_in_blocks=1),
        )
    )
    request = SimpleNamespace(num_tokens=33, request_id="unit")
    state = SimpleNamespace(
        req=request,
        req_context=None,
        num_locally_computed_tokens=0,
        group_states=(
            SimpleNamespace(offload_keys=[b"fa0", b"fa1"]),
            SimpleNamespace(offload_keys=[b"m0", b"m1", b"m2"]),
        ),
    )

    # 33-token request is reduced to the 32-token Mamba boundary.
    assert offload_scheduler._lookup(state) == 32

    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        use_eagle=False,
        retain_mamba_align_mtp_cache_block=False,
    )
    model_request = SimpleNamespace(
        num_computed_tokens=0, num_prompt_tokens=33, num_tokens=34
    )
    # External KV tokens must be accepted by the local split calculation.
    assert (
        Scheduler._mamba_block_aligned_split(
            scheduler,
            model_request,
            20,
            num_external_computed_tokens=16,
        )
        == 16
    )


def test_hybrid_hit_realigns_after_non_divisor_group():
    """A group whose offloaded_block_size is not a divisor of the Mamba
    alignment must not report an unaligned hit that includes a Mamba state
    beyond it.

    Group 0 (full attention) uses offloaded_block_size=12; group 1 (Mamba,
    align mode) uses offloaded_block_size=16. 12 is not a divisor of 16, so the
    per-group ``min(max_hit, len_keys * block_size)`` tightening lands the
    boundary on a multiple of 12 that is not a multiple of 16. Unless the
    boundary is re-aligned to the Mamba block after *each* per-group constraint,
    ``_lookup`` reports 24 tokens (not aligned to 16) while the Mamba group's
    key lookup already reached the block covering tokens 16..32 -- a recurrent
    state beyond the reported hit, which would restore a Mamba state
    inconsistent with the attention prefix. With the fix the hit is rounded
    back to the 16-token boundary and the Mamba lookup stops there.
    """
    # Record how far into the Mamba (group 1) keys the lookup actually reached.
    mamba_blocks_looked_up: list[int] = []

    class _RecordingAllHitManager:
        def lookup(self, key, req_context):
            if isinstance(key, bytes) and key.startswith(b"m"):
                mamba_blocks_looked_up.append(int(key[1:]))
            return True

    offload_scheduler = object.__new__(OffloadingConnectorScheduler)
    offload_scheduler._sliding_window_groups = (1,)
    offload_scheduler._lookup_groups = (0, 1)
    offload_scheduler._mamba_align_size = 16
    offload_scheduler._blocks_being_loaded = None
    offload_scheduler.manager = _RecordingAllHitManager()
    offload_scheduler.config = SimpleNamespace(
        kv_group_configs=(
            # full attention; 12 is NOT a divisor of the Mamba alignment (16)
            SimpleNamespace(
                offloaded_block_size=12, sliding_window_size_in_blocks=None
            ),
            # Mamba align group; its block size is the alignment
            SimpleNamespace(offloaded_block_size=16, sliding_window_size_in_blocks=1),
        )
    )
    request = SimpleNamespace(num_tokens=34, request_id="unit")
    state = SimpleNamespace(
        req=request,
        req_context=None,
        num_locally_computed_tokens=0,
        group_states=(
            SimpleNamespace(offload_keys=[b"fa0", b"fa1"]),
            SimpleNamespace(offload_keys=[b"m0", b"m1", b"m2"]),
        ),
    )

    num_hit_tokens = offload_scheduler._lookup(state)

    mamba_align = offload_scheduler._mamba_align_size
    # Tokens covered by the Mamba group's key lookup (block idx + 1) * block_size.
    assert mamba_blocks_looked_up, "Mamba group was never looked up"
    mamba_covered_tokens = (max(mamba_blocks_looked_up) + 1) * mamba_align

    # 1) The reported hit must be aligned to the Mamba boundary (pre-fix: 24).
    assert num_hit_tokens % mamba_align == 0, (
        f"unaligned hit {num_hit_tokens} is not a multiple of {mamba_align}"
    )
    # 2) The Mamba state included must correspond to <= the reported hit, i.e.
    #    no recurrent state beyond the attention prefix we report as hit.
    assert mamba_covered_tokens <= num_hit_tokens, (
        f"Mamba lookup reached {mamba_covered_tokens} tokens, beyond the "
        f"reported hit of {num_hit_tokens}"
    )
    # Concretely, the fix reports the 16-token boundary, not the unaligned 24.
    assert num_hit_tokens == 16
