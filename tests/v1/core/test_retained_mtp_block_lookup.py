# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LANE f1-lookup (/home/kevin/projects/lanes/f1-lookup): read-side lookup
tests for VLLM_PREFIX_CACHE_USE_RETAINED_MTP_BLOCK.

Context: HybridKVCacheCoordinator.find_longest_cache_hit unconditionally
drops the last matched block for any EAGLE/MTP-affected attention group
(vllm-project/vllm#43650 -- see test_prefix_caching.py::
test_full_attention_prefix_cache_eagle_regression /
test_hybrid_mamba_eagle_does_not_reuse_lookahead_state for the existing,
unchanged-by-this-lane behavior). Separately, VLLM_MAMBA_ALIGN_RETAIN_MTP_
CACHE_BLOCK (docs/mtp-retention-invariant.md) already makes the WRITE side
keep a genuinely valid extra aligned Mamba boundary for method="mtp". This
file tests the READ side: VLLM_PREFIX_CACHE_USE_RETAINED_MTP_BLOCK, gated by
BOTH that env AND the coordinator's mtp_retain_active flag, lets the lookup
count that retained block as a hit instead of re-dropping it.

Warm-up chunk shapes (IMPORTANT, empirically verified against this exact
worktree -- see LANE/MAP.md "Test fixture note"): MambaManager's align-mode
allocate_new_blocks only hashes a REAL, lookup-reachable block at wherever
each individual allocate_slots() call's cumulative token count lands; any
block boundary a single call jumps PAST (without landing exactly on it)
gets a null placeholder with no hash at all (this is the pre-#53479 "sparse
states" behavior pinned by test_partial_prefix_boundary_stops.py Group 1,
orthogonal to this lane). This fork's production geometry has
max_num_batched_tokens (~3584) barely larger than the block size (~3568),
so in production every scheduler step lands on (approximately) one more
block boundary -- "legacy chunking already materialises ~every boundary"
per the lane's task brief. These tests reproduce that per-block-landing
shape explicitly (chunks of exactly one block each) rather than one large
chunk, so the fixture's write-time cache state matches what the real
scheduler produces, not an artifact of an unrealistically large single
test call:
  - "retained" shape: block, block, block, block, tail
    (Scheduler._mamba_block_aligned_split with retain_final_mtp_block=True
    would stop each step at the next block boundary up to and including
    the tail-adjacent one, same as it does with retain=False -- retention
    changes WHERE THE LAST boundary is allowed to land, at
    round_down(num_tokens, block_size) instead of one block earlier -- not
    how the earlier boundaries are chunked.)
  - "not retained" shape: block, block, block, (block+tail)
    (the legacy EAGLE back-off: last_cache_position lands one block earlier,
    so the final chunk folds the would-be 4th block together with the
    unaligned tail into one call, and that 4th-block boundary is therefore
    never individually hashed at all.)

Every test explicitly sets/deletes the env var via monkeypatch (vllm.envs
does a fresh os.environ read on every attribute access -- no caching, no
reload needed, verified against this exact worktree before writing these
tests) so tests never depend on ambient process environment.

See LANE/MAP.md for the file:line trace and LANE/REPORT.md for the residual
risk this change does NOT close (block-hash lookups carry no per-block
provenance).
"""

import torch
import pytest

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

ENV = "VLLM_PREFIX_CACHE_USE_RETAINED_MTP_BLOCK"
BLOCK = 16
# 4 blocks worth of prompt (64 tokens) plus a 7-token unaligned tail = 71.
TOKEN_IDS = [i for i in range(4) for _ in range(BLOCK)] + [4] * 7
RETAINED_SHAPE_CHUNKS = (BLOCK, BLOCK, BLOCK, BLOCK, 7)
NOT_RETAINED_SHAPE_CHUNKS = (BLOCK, BLOCK, BLOCK, BLOCK + 7)


@pytest.fixture(autouse=True)
def _init_hash() -> None:
    init_none_hash(sha256)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Belt-and-suspenders: every test sets this explicitly too, but make sure
    # no test can leak the env var to another via ambient process state.
    monkeypatch.delenv(ENV, raising=False)


def _make_request(request_id: str, token_ids: list[int], block_size: int) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        sampling_params=SamplingParams(max_tokens=17),
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )


def _full_spec(block_size: int) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32
    )


def _mamba_align_spec(block_size: int) -> MambaSpec:
    return MambaSpec(
        block_size=block_size,
        shapes=(1, 1),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )


def _sliding_spec(block_size: int, window: int) -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=window,
    )


def _make_manager(
    block_size: int,
    *,
    use_eagle: bool,
    mtp_retain_active: bool = False,
    extra_group: KVCacheGroupSpec | None = None,
) -> KVCacheManager:
    groups = [
        KVCacheGroupSpec(["full"], _full_spec(block_size)),
        KVCacheGroupSpec(["mamba"], _mamba_align_spec(block_size)),
    ]
    if extra_group is not None:
        groups.append(extra_group)
    return KVCacheManager(
        KVCacheConfig(num_blocks=100, kv_cache_tensors=[], kv_cache_groups=groups),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
        use_eagle=use_eagle,
        mtp_retain_active=mtp_retain_active,
    )


def _warm(
    manager: KVCacheManager,
    token_ids: list[int],
    chunks: tuple[int, ...],
    request_id: str = "first",
) -> None:
    assert sum(chunks) == len(token_ids)
    req = _make_request(request_id, token_ids, BLOCK)
    computed, num_computed = manager.get_computed_blocks(req)
    assert num_computed == 0
    for num_new_tokens in chunks:
        blocks = manager.allocate_slots(req, num_new_tokens, num_computed, computed)
        assert blocks is not None
        req.num_computed_tokens += num_new_tokens
        computed, num_computed = None, 0
    manager.free(req)


# ---------------------------------------------------------------------------
# (a) env unset -> identical to today's behavior, even with mtp_retain_active
#     True at the coordinator level and a genuinely-retained write-time
#     boundary underneath.
# ---------------------------------------------------------------------------


def test_env_unset_matches_baseline_even_with_retain_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _make_manager(BLOCK, use_eagle=True, mtp_retain_active=True)
    _warm(manager, TOKEN_IDS, RETAINED_SHAPE_CHUNKS)

    monkeypatch.delenv(ENV, raising=False)
    second = _make_request("second", TOKEN_IDS, BLOCK)
    computed, num_computed = manager.get_computed_blocks(second)

    assert num_computed == 3 * BLOCK
    assert [len(g) for g in computed.blocks] == [3, 3]


def test_env_set_without_mtp_retain_active_matches_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Double-gate check: the read-side env alone (model not using MTP+align,
    so the coordinator's mtp_retain_active is False) must not change
    anything -- it's a no-op unless the write-side mechanism is also active
    for this model."""
    manager = _make_manager(BLOCK, use_eagle=True, mtp_retain_active=False)
    _warm(manager, TOKEN_IDS, RETAINED_SHAPE_CHUNKS)

    monkeypatch.setenv(ENV, "1")
    second = _make_request("second", TOKEN_IDS, BLOCK)
    computed, num_computed = manager.get_computed_blocks(second)

    assert num_computed == 3 * BLOCK
    assert [len(g) for g in computed.blocks] == [3, 3]


# ---------------------------------------------------------------------------
# (b) env set AND mtp_retain_active -> hit length increases by exactly one
#     block where the block is genuinely hash-cached in every group (the
#     "retained" write shape), and never past the actual hashed prefix
#     otherwise (the "not retained" write shape leaves nothing extra to
#     recover, and a differently-hashed second request leaves nothing extra
#     to recover either).
# ---------------------------------------------------------------------------


def test_env_set_and_retain_active_recovers_one_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write side actually retained the boundary (RETAINED_SHAPE_CHUNKS: each
    of the 4 blocks is its own allocate_slots call, so MambaManager's
    align-mode bookkeeping hashes a real block at every one of them, not
    just the final landing spot -- see module docstring). With both gates
    on, the lookup must recover exactly that block: 4, not more."""
    manager = _make_manager(BLOCK, use_eagle=True, mtp_retain_active=True)
    _warm(manager, TOKEN_IDS, RETAINED_SHAPE_CHUNKS)

    monkeypatch.setenv(ENV, "1")
    second = _make_request("second", TOKEN_IDS, BLOCK)
    computed, num_computed = manager.get_computed_blocks(second)

    assert num_computed == 4 * BLOCK
    assert [len(g) for g in computed.blocks] == [4, 4]
    # Never past the hashed prefix: the request has exactly 71 tokens (4
    # blocks + a 7-token unaligned tail); a 5th block was never hashed, so
    # recovery cannot manufacture one.
    assert num_computed < len(TOKEN_IDS)


def test_env_set_but_write_side_never_retained_recovers_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The critical safety-net case: mtp_retain_active is True at the
    coordinator level (a caller could set it whenever the model uses
    method="mtp" + align mode, independent of whether any *specific*
    request's own boundary was actually written under retention), but THIS
    request's write-time chunking used the legacy (non-retained) shape --
    the 4th block was truly never hashed in the mamba group (only in
    full-attention). The relaxation must not, and cannot, manufacture a hit
    for a block nothing ever cached: the cross-group fixed point falls back
    to whatever mamba's own hash chain actually supports (3 blocks), exactly
    like the pre-existing #43650 cross-group capping already does for
    sparse-retention mismatches."""
    manager = _make_manager(BLOCK, use_eagle=True, mtp_retain_active=True)
    _warm(manager, TOKEN_IDS, NOT_RETAINED_SHAPE_CHUNKS)

    monkeypatch.setenv(ENV, "1")
    second = _make_request("second", TOKEN_IDS, BLOCK)
    computed, num_computed = manager.get_computed_blocks(second)

    assert num_computed == 3 * BLOCK
    assert [len(g) for g in computed.blocks] == [3, 3]


def test_recovery_never_exceeds_the_actual_hashed_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second, hash-content-driven angle on the same 'never past the hashed
    prefix' requirement: the second request's own token content diverges
    from the first's after block 3, so blocks 1-3 hash-match but block 4
    cannot, regardless of what got cached. This holds even though
    mtp_retain_active and the env are both on and the first request's block
    4 genuinely is fully cached (for the *first* request's own content)."""
    manager = _make_manager(BLOCK, use_eagle=True, mtp_retain_active=True)
    _warm(manager, TOKEN_IDS, RETAINED_SHAPE_CHUNKS)

    # Same first 3 blocks, then genuinely different content.
    second_ids = TOKEN_IDS[: 3 * BLOCK] + [99] * (BLOCK + 7)
    monkeypatch.setenv(ENV, "1")
    second = _make_request("second", second_ids, BLOCK)
    computed, num_computed = manager.get_computed_blocks(second)

    assert num_computed == 3 * BLOCK
    assert [len(g) for g in computed.blocks] == [3, 3]


def test_pure_mamba_unitary_coordinator_unaffected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defense in depth: UnitaryKVCacheCoordinator (no full-attention group,
    e.g. a pure-Mamba/linear-attn model) never receives mtp_retain_active at
    all -- get_kv_cache_coordinator only forwards it to HybridKVCacheCoordinator
    (see vllm/v1/core/kv_cache_coordinator.py::get_kv_cache_coordinator). The
    new env being set must not change pure-Mamba eagle-drop behavior, which
    is the vllm-project/vllm#43650 fix pinned by test_prefix_caching.py::
    test_pure_mamba_prefix_cache_eagle_drop."""
    groups = [
        KVCacheGroupSpec(
            ["mamba"],
            MambaSpec(
                block_size=BLOCK,
                shapes=(1, 1),
                dtypes=(torch.float32,),
                mamba_cache_mode="all",
            ),
        )
    ]
    manager = KVCacheManager(
        KVCacheConfig(num_blocks=100, kv_cache_tensors=[], kv_cache_groups=groups),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=BLOCK,
        use_eagle=True,
        mtp_retain_active=True,  # even if a caller mistakenly passed this
    )
    _warm(manager, TOKEN_IDS, (len(TOKEN_IDS),))

    monkeypatch.setenv(ENV, "1")
    second = _make_request("second", TOKEN_IDS, BLOCK)
    computed, num_computed = manager.get_computed_blocks(second)

    assert num_computed == 3 * BLOCK
    assert [len(g) for g in computed.blocks] == [3]


# ---------------------------------------------------------------------------
# (c) the coordinator's iterative fixed-point loop still converges.
#
# This fork's real production model (Mamba/GDN + one full-attention group)
# is always "is_simple_hybrid" (single pass; see HybridKVCacheCoordinator.
# find_longest_cache_hit) -- tests above already exercise that path,
# including its one shrink-and-settle branch
# (test_env_set_but_write_side_never_retained_recovers_nothing forces a
# shrink from the relaxed full-attention candidate back down to what mamba
# actually supports). The genuinely iterative (>1 pass) branch only triggers
# with 3+ distinct attention-group specs, which this fork's real model never
# produces. This test is a synthetic, structural check of the algorithm
# itself (full-attention + sliding-window + mamba-align, all independently
# eagle-affected where supported), not a claim about any real model shape:
# it asserts termination and a self-consistent final hit_length across an
# A/B of the env, not exact per-group block contents.
# ---------------------------------------------------------------------------


def test_coordinator_fixed_point_converges_with_three_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extra = KVCacheGroupSpec(["sliding"], _sliding_spec(BLOCK, BLOCK))
    manager = _make_manager(
        BLOCK, use_eagle=True, mtp_retain_active=True, extra_group=extra
    )
    _warm(manager, TOKEN_IDS, RETAINED_SHAPE_CHUNKS)

    monkeypatch.delenv(ENV, raising=False)
    baseline_req = _make_request("baseline", TOKEN_IDS, BLOCK)
    computed_off, num_computed_off = manager.get_computed_blocks(baseline_req)

    monkeypatch.setenv(ENV, "1")
    relaxed_req = _make_request("relaxed", TOKEN_IDS, BLOCK)
    computed_on, num_computed_on = manager.get_computed_blocks(relaxed_req)

    # Termination: reaching these assertions at all (pytest's own default
    # per-test wall time, no explicit timeout needed for a bounded, in-
    # process loop over 3 tiny groups) already demonstrates the fixed point
    # does not hang. Self-consistency: whatever hit_length the function
    # settles on, the full-attention group's returned block count must
    # agree with it exactly (final-truncation bookkeeping stays correct
    # under the >1-attention-group loop, not only the simple-hybrid path).
    for computed, num_computed in ((computed_off, num_computed_off), (computed_on, num_computed_on)):
        assert num_computed % BLOCK == 0
        assert len(computed.blocks[0]) == num_computed // BLOCK
        assert num_computed <= 4 * BLOCK

    assert num_computed_off == 3 * BLOCK
    # The relaxation must be at least as good as baseline, and in this
    # fixture (sliding-window group also fully covers the 4th block, window
    # == block size) it recovers it fully, same as the plain 2-group case.
    assert num_computed_on >= num_computed_off
    assert num_computed_on == 4 * BLOCK
