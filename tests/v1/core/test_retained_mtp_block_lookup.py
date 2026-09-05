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
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)
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
    num_gpu_blocks: int = 100,
) -> KVCacheManager:
    groups = [
        KVCacheGroupSpec(["full"], _full_spec(block_size)),
        KVCacheGroupSpec(["mamba"], _mamba_align_spec(block_size)),
    ]
    if extra_group is not None:
        groups.append(extra_group)
    return KVCacheManager(
        KVCacheConfig(num_blocks=num_gpu_blocks, kv_cache_tensors=[], kv_cache_groups=groups),
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
    4 genuinely is fully cached (for the *first* request's own content).

    [UPDATED by LANE f1-provenance -- /home/kevin/projects/lanes/
    f1-provenance] Before provenance checking, this asserted 3 blocks (48):
    the OLD f1-lookup relaxation skipped the eagle-pop for ANY candidate
    once the coarse (env + mtp_retain_active) gates were on, regardless of
    which specific block it landed on -- so block 2 (an ordinary,
    non-boundary prefill block that merely happens to be THIS lookup's
    deepest hash match, since content diverges at block 3) escaped the
    drop right along with the one genuinely-provenanced boundary. That was
    exactly the residual risk LANE/REPORT.md (f1-lookup) flagged: "it
    relaxes the drop for ANY eagle-affected candidate once both flags are
    on, not only the specific retained one." Block 2 was never marked
    KVCacheBlock.retained_mtp_boundary (only the true tail, block 3 of the
    FIRST request's own write, ever is -- see
    test_provenance_bit_set_only_by_retain_path), so the read side now
    correctly falls back to the ordinary eagle-drop for it, exactly as it
    would with the relaxation off entirely: 3 hash-matched blocks (0,1,2),
    minus the unproven deepest one (2), leaves 2."""
    manager = _make_manager(BLOCK, use_eagle=True, mtp_retain_active=True)
    _warm(manager, TOKEN_IDS, RETAINED_SHAPE_CHUNKS)

    # Same first 3 blocks, then genuinely different content.
    second_ids = TOKEN_IDS[: 3 * BLOCK] + [99] * (BLOCK + 7)
    monkeypatch.setenv(ENV, "1")
    second = _make_request("second", second_ids, BLOCK)
    computed, num_computed = manager.get_computed_blocks(second)

    # Never past the diverging block (3), AND never past the last block this
    # lookup can actually prove safe: block 2 hash-matches but isn't the
    # provenanced boundary, so it is dropped too, same as plain (non-relaxed)
    # eagle behavior would drop it.
    assert num_computed == 2 * BLOCK
    assert [len(g) for g in computed.blocks] == [2, 2]


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


# ---------------------------------------------------------------------------
# LANE f1-provenance (/home/kevin/projects/lanes/f1-provenance) --
# KVCacheBlock.retained_mtp_boundary: write-side correctness + lifecycle.
#
# f1-lookup's own REPORT.md flagged the exact gap closed here: the coarse
# (env + mtp_retain_active) gates alone cannot tell "the one boundary
# retain_final_mtp_block protects" apart from an ordinary decode-time
# boundary that happens to be some lookup's last match -- and
# test_recovery_never_exceeds_the_actual_hashed_prefix above is direct,
# empirical proof the gap was real: before this bit existed, that test's
# second request recovered 3 blocks (48 tokens), silently keeping an
# ordinary, non-boundary block (index 2) alive purely because the coarse
# gates were on. It now correctly recovers only 2 (see that test's updated
# docstring for the full account).
#
# These tests drive genuine post-prefill decode steps
# (Request.append_output_token_ids + further allocate_slots calls, no
# Scheduler needed) to prove the bit distinguishes retained-prefill-tail
# commits from decode-time ones. LANE/DESIGN.md has the full write-side
# argument; in short: ordinary decode's cache_blocks call always targets
# num_tokens_to_cache == request.num_tokens exactly (the running total
# after the just-appended token), while the retain-path commit is the only
# one that ever deliberately stops SHORT of request.num_tokens, at exactly
# round_down(request.num_tokens, block_size) -- so the two can never be
# confused structurally, not just empirically.
# ---------------------------------------------------------------------------


def _prefill_request(manager: KVCacheManager, request_id: str, token_ids: list[int]) -> Request:
    """Warm a request through RETAINED_SHAPE_CHUNKS's per-block chunking
    (see module docstring) WITHOUT freeing it, so the caller can continue
    driving it into decode and/or inspect its own blocks directly."""
    assert sum(RETAINED_SHAPE_CHUNKS) == len(token_ids)
    req = _make_request(request_id, token_ids, BLOCK)
    computed, num_computed = manager.get_computed_blocks(req)
    assert num_computed == 0
    for num_new_tokens in RETAINED_SHAPE_CHUNKS:
        blocks = manager.allocate_slots(req, num_new_tokens, num_computed, computed)
        assert blocks is not None
        req.num_computed_tokens += num_new_tokens
        computed, num_computed = None, 0
    return req


def _decode_steps(manager: KVCacheManager, req: Request, num_steps: int) -> None:
    """Simulate `num_steps` of ordinary, non-speculative one-token decode:
    append a fresh output token then cache up to the request's new (one
    token longer) num_tokens -- the shape every real decode step takes
    once num_computed_tokens has caught up to num_tokens - 1 after
    prefill (see the module-level comment above for why this can never
    satisfy the write-side marking condition)."""
    for i in range(num_steps):
        req.append_output_token_ids(1000 + i)
        blocks = manager.allocate_slots(req, 1, 0, None)
        assert blocks is not None
        req.num_computed_tokens += 1


def test_provenance_bit_set_only_by_retain_path() -> None:
    """(TASK 2a) The bit is set on exactly the one block
    Scheduler._mamba_block_aligned_split's retain_final_mtp_block branch
    protects (block index 3, tokens 48-64 -- the tail of the original
    71-token prompt, landed on exactly by RETAINED_SHAPE_CHUNKS's 4th
    chunk) for BOTH eagle-affected groups it is committed in (full-
    attention AND mamba-align both independently satisfy the identical
    write-side condition for their own req_to_blocks, since both share the
    same request.num_tokens/block_size and the same
    manager.mtp_retain_active -- see HybridKVCacheCoordinator.
    verify_and_split_kv_cache_groups propagating it onto every manager).
    It is never set on any ordinary prefill block before it (positions
    0-2), nor on any decode-time commit after it (position 4, landed on by
    9 single-token decode steps continuing the SAME request from 71 to 80
    tokens -- crossing a full new block boundary post-prefill, exactly the
    kind of boundary f1-lookup's residual-risk note worried about)."""
    manager = _make_manager(BLOCK, use_eagle=True, mtp_retain_active=True)
    req = _prefill_request(manager, "writer", TOKEN_IDS)

    full_mgr, mamba_mgr = manager.coordinator.single_type_managers
    # Save direct block references (not list indices) before decode: Mamba's
    # align-mode running-state bookkeeping (MambaManager.remove_skipped_
    # blocks / last_state_block_idx) frees and null-replaces its OWN older
    # "current state" slot as soon as a newer one supersedes it during
    # ordinary decode -- a pre-existing mechanic wholly unrelated to
    # provenance (the freed block keeps its hash+bit; see BlockPool.
    # free_blocks, which never touches either). Checking the SAVED
    # reference (rather than re-indexing req_to_blocks after decode) is
    # what actually verifies the bit itself is stable, independent of that
    # unrelated recycling.
    retained_block = {}
    for mgr, label in ((full_mgr, "full"), (mamba_mgr, "mamba")):
        blocks = mgr.req_to_blocks[req.request_id]
        assert blocks[3].retained_mtp_boundary is True, label
        for i in (0, 1, 2):
            assert blocks[i].retained_mtp_boundary is False, (label, i)
        retained_block[label] = blocks[3]

    # Continue the SAME request into 9 steps of ordinary one-token decode:
    # 71 -> 80 tokens, crossing the block-4 boundary (tokens 64-80) as an
    # everyday decode-time commit, not a prompt-chunking decision.
    _decode_steps(manager, req, num_steps=9)
    assert req.num_tokens == 80

    for mgr, label in ((full_mgr, "full"), (mamba_mgr, "mamba")):
        # The originally-marked block is still marked (decode elsewhere in
        # the request never spuriously clears an unrelated block's bit).
        assert retained_block[label].retained_mtp_boundary is True, label
        # Whatever NEW block(s) decode caused to be committed do not carry
        # it. The live req_to_blocks list is the right place to look for
        # full-attention (it keeps every block for the request's whole
        # lifetime), but NOT for mamba: align-mode's own running-state
        # bookkeeping (remove_skipped_blocks/last_state_block_idx) has by
        # now freed-and-null-replaced its slot-3 entry in favor of a newer
        # "current state" block (an unrelated, pre-existing mechanic -- see
        # the comment above `retained_block = {}`), so the live list
        # legitimately contains ZERO marked blocks for mamba at this point.
        # Either way, the marked set found live can never be anything OTHER
        # than the original boundary block -- that is what would indicate a
        # decode-time block wrongly picking up the bit.
        marked = {
            b.block_id
            for b in mgr.req_to_blocks[req.request_id]
            if not b.is_null and b.retained_mtp_boundary
        }
        assert marked <= {retained_block[label].block_id}, label

    manager.free(req)


def test_read_side_drops_decode_time_block_but_keeps_retained_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(TASK 2b) End-to-end read-side proof, both halves in one lookup: warm
    a request through the retained prefill tail (block 3, provenanced) and
    then 9 ordinary decode steps past it (block 4, NOT provenanced -- see
    test_provenance_bit_set_only_by_retain_path). A second, fresh request
    with the SAME full 80-token history hash-matches all 5 blocks; with
    both env gates on the coordinator's candidate last block is block 4
    (no bit) so it falls back to the ordinary eagle-drop for it -- but
    that drop only removes ONE block, leaving block 3 (the genuinely
    retained one) counted. Net: exactly 4 blocks (64 tokens), not 5 (the
    un-provenanced decode boundary rejected) and not 3 (the provenanced
    boundary correctly kept, unlike the pre-provenance behavior)."""
    manager = _make_manager(BLOCK, use_eagle=True, mtp_retain_active=True)
    req = _prefill_request(manager, "writer", TOKEN_IDS)
    _decode_steps(manager, req, num_steps=9)
    assert req.num_tokens == 80
    manager.free(req)

    monkeypatch.setenv(ENV, "1")
    reader_ids = TOKEN_IDS + [1000 + i for i in range(9)]
    reader = _make_request("reader", reader_ids, BLOCK)
    computed, num_computed = manager.get_computed_blocks(reader)

    assert num_computed == 4 * BLOCK
    assert [len(g) for g in computed.blocks] == [4, 4]


def test_reset_hash_clears_provenance_bit() -> None:
    """(TASK 2c, unit) Direct test of the exact choke point
    (kv_cache_utils.py KVCacheBlock.reset_hash): the ONLY place a block's
    hash is ever reset also clears retained_mtp_boundary in the same call,
    which is what makes staleness structurally impossible -- see
    LANE/DESIGN.md's "why the bit cannot go stale" argument (the
    block_hash setter asserts the current hash is None, so a block can
    never get a NEW hash without reset_hash running first)."""
    block = KVCacheBlock(block_id=5)
    block.block_hash = make_block_hash_with_group_id(BlockHash(b"y" * 32), 0)
    block.retained_mtp_boundary = True

    block.reset_hash()

    assert block.block_hash is None
    assert block.retained_mtp_boundary is False


def test_provenance_bit_cleared_on_pool_recycle() -> None:
    """(TASK 2c, integration) The same choke point exercised through the
    real recycling path: BlockPool.get_new_blocks -> _maybe_evict_cached_
    block -> reset_hash, when the LRU free-queue candidate still carries a
    hash from a previous, now-freed tenant. Bypasses KVCacheManager/Mamba-
    align complexity entirely -- a focused BlockPool-level test."""
    pool = BlockPool(num_gpu_blocks=3, enable_caching=True, hash_block_size=BLOCK)
    # Hold both usable blocks (block_id 0 is the reserved null block) so
    # neither is sitting idle-free ahead of the one we care about in the
    # LRU queue once we free it.
    block, held = pool.get_new_blocks(2)
    block_hash = make_block_hash_with_group_id(BlockHash(b"x" * 32), 0)
    block.block_hash = block_hash
    pool.cached_block_hash_to_block.insert(block_hash, block)
    block.retained_mtp_boundary = True

    pool.free_blocks([block])  # ref_cnt -> 0; back on the free queue, still hashed+marked
    assert block.block_hash is not None
    assert block.retained_mtp_boundary is True

    # `held` is still allocated, so `block` is the only free candidate --
    # get_new_blocks() must evict (reset_hash) it before handing it back.
    [recycled] = pool.get_new_blocks(1)

    assert recycled.block_id == block.block_id
    assert recycled.block_hash is None
    assert recycled.retained_mtp_boundary is False


def test_stress_pool_recycling_never_leaks_stale_provenance_bit() -> None:
    """(TASK 2d) Cycle a deliberately tight pool (8 blocks -- 1 null + 7
    usable, versus ~9-10 needed per request across both groups) through 30
    distinct requests, each independently exercising the retain path. Each
    iteration's own block 3 must carry the bit (the mechanism still works
    under constant eviction pressure) and, critically, block 1 (an
    ordinary, never-boundary position) must NEVER carry it -- if
    reset_hash() ever failed to clear a recycled block's stale bit from a
    PRIOR occupant's own boundary commit, this is exactly the check that
    would catch a physical block_id being reused across iterations (block
    1 in iteration N could easily be the very block_id that served as
    iteration N-3's block 3) while silently keeping iteration N-3's True."""
    manager = _make_manager(
        BLOCK, use_eagle=True, mtp_retain_active=True, num_gpu_blocks=8
    )
    full_mgr, mamba_mgr = manager.coordinator.single_type_managers
    reused_block_ids: set[int] = set()

    for i in range(30):
        # Distinct content per iteration so hashes never collide across
        # requests (which would turn this into a cache-hit test instead of
        # a fresh-write-every-time recycling stress test).
        ids = [2000 + i] * (4 * BLOCK) + [77] * 7
        req = _make_request(f"req{i}", ids, BLOCK)
        computed, num_computed = manager.get_computed_blocks(req)
        assert num_computed == 0  # distinct content: never a cache hit
        for num_new_tokens in RETAINED_SHAPE_CHUNKS:
            blocks = manager.allocate_slots(req, num_new_tokens, num_computed, computed)
            assert blocks is not None, f"pool exhausted at iteration {i}"
            req.num_computed_tokens += num_new_tokens
            computed, num_computed = None, 0

        for mgr in (full_mgr, mamba_mgr):
            rb = mgr.req_to_blocks[req.request_id]
            reused_block_ids.add(rb[1].block_id)
            assert rb[1].retained_mtp_boundary is False, (
                f"stale bit leaked onto block_id={rb[1].block_id} "
                f"(an ordinary, never-boundary position) at iteration {i}"
            )
            assert rb[3].retained_mtp_boundary is True, (
                f"retain path failed to mark its own boundary at iteration {i}"
            )
        manager.free(req)

    # Confirm this was actually a recycling stress test, not 30 iterations
    # each getting fresh, never-before-used blocks (which would make the
    # "never leaks" assertions above vacuous).
    assert len(reused_block_ids) < 30, (
        "pool never forced block reuse -- widen the stress by shrinking "
        "num_gpu_blocks in _make_manager for this test"
    )
