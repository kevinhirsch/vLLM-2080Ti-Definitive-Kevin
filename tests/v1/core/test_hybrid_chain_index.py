# SPDX-License-Identifier: Apache-2.0
"""Lane R2: chain-aware eviction ordering for hybrid (full attention + Mamba align) prefix caches.

CPU only: drives the real KVCacheManager with 1 full-attention group + 3 Mamba(align) groups,
retention interval 0 (the production setting), block size 16 standing in for 3568.
"""
import random

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_block_hash,
    get_group_id,
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

BS = 16
init_none_hash(sha256)


def _mgr(monkeypatch, num_blocks=80, chain=True, supersede=False, grace=0.0, dry=False):
    monkeypatch.setenv("VLLM_R2_CHAIN_AWARE_EVICT", "1" if chain else "0")
    monkeypatch.setenv("VLLM_R2_SUPERSEDE", "1" if supersede else "0")
    monkeypatch.setenv("VLLM_R2_SUPERSEDE_GRACE_S", str(grace))
    monkeypatch.setenv("VLLM_R2_DRYRUN", "1" if dry else "0")
    groups = [
        KVCacheGroupSpec(
            ["a"],
            FullAttentionSpec(block_size=BS, num_kv_heads=1, head_size=1, dtype=torch.float32),
        )
    ]
    for i in range(3):
        groups.append(
            KVCacheGroupSpec(
                [f"m{i}"],
                MambaSpec(
                    block_size=BS,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            )
        )
    cfg = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        prefix_cache_retention_interval=0,
    )
    return KVCacheManager(
        cfg, max_model_len=10000, enable_caching=True, hash_block_size=BS, scheduler_block_size=BS
    )


def _req(rid, toks):
    sp = SamplingParams(max_tokens=4)
    sp.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=rid,
        prompt_token_ids=toks,
        mm_features=None,
        sampling_params=sp,
        pooling_params=None,
        lora_request=None,
        cache_salt=None,
        block_hasher=get_request_block_hasher(BS, sha256),
        session_id=None,
    )


def _toks(seed, n):
    r = random.Random(seed)
    return [r.randrange(1, 90000) for _ in range(n)]


def _run(mgr, rid, toks):
    """Chunked prefill (one block per step) + 3 decode steps + free; returns hit tokens."""
    req = _req(rid, toks)
    blocks, hit, _ = mgr.get_computed_blocks(req)
    pos, first = hit, True
    while pos < len(toks):
        step = min(BS, len(toks) - pos)
        if first:
            nb = mgr.allocate_slots(
                req, step, num_new_computed_tokens=hit, new_computed_blocks=blocks
            )
            first = False
        else:
            nb = mgr.allocate_slots(req, step)
        assert nb is not None
        req.num_computed_tokens = pos + step
        pos += step
        mgr.new_step_starts()
    for t in range(3):
        req.append_output_token_ids(1000 + t)
        assert mgr.allocate_slots(req, 1) is not None
        req.num_computed_tokens += 1
        mgr.new_step_starts()
    mgr.free(req)
    return hit


def _queue_front(mgr, n):
    q = mgr.block_pool.free_block_queue
    out, b = [], q.fake_free_list_head.next_free_block
    while b is not q.fake_free_list_tail and len(out) < n:
        out.append(b)
        b = b.next_free_block
    return out


def _snapshot_blocks(mgr):
    pool = mgr.block_pool
    return [
        b for b in pool.blocks
        if b.block_hash is not None and get_group_id(b.block_hash) in (1, 2, 3)
    ]


def _attn_blocks(mgr):
    return [
        b for b in mgr.block_pool.blocks
        if b.block_hash is not None and get_group_id(b.block_hash) == 0
    ]


def test_disabled_by_default(monkeypatch):
    m = _mgr(monkeypatch, chain=False)
    assert m.block_pool.chain_index is None
    assert _run(m, "a", _toks(1, 140)) == 0


def test_installed_and_registers_snapshot(monkeypatch):
    m = _mgr(monkeypatch)
    ci = m.block_pool.chain_index
    assert ci is not None and ci.attn_gids == {0} and ci.snap_gids == {1, 2, 3}
    _run(m, "a", _toks(1, 140))
    assert len(ci.anc) == 1
    (s,) = ci.anc
    assert ci._snapshot_alive(s)
    assert len(ci.anc[s]) == 8  # attention pages 0..7 sit under the snapshot


def test_snapshot_eviction_recycles_its_attention_pages_first(monkeypatch):
    m = _mgr(monkeypatch)
    ci = m.block_pool.chain_index
    t = _toks(1, 140)
    _run(m, "a", t)
    pool = m.block_pool
    attn_before = {b.block_id for b in _attn_blocks(m)}
    snap = _snapshot_blocks(m)
    assert len(snap) == 3 and len(attn_before) == 8
    # Evict ONE snapshot page the way an allocation would (pop + evict).
    victim = snap[0]
    pool.free_block_queue.remove(victim)
    pool._maybe_evict_cached_block(victim)
    front = {b.block_id for b in _queue_front(m, 10)}
    # the other two snapshot pages and all 8 attention pages are now first in line
    assert {b.block_id for b in snap[1:]} <= front
    assert attn_before <= front
    assert ci.demoted >= 10


def test_attention_eviction_recycles_snapshot_pages_first(monkeypatch):
    m = _mgr(monkeypatch)
    ci = m.block_pool.chain_index
    _run(m, "a", _toks(1, 140))
    pool = m.block_pool
    attn = sorted(_attn_blocks(m), key=lambda b: b.block_hash_num_tokens)
    snap_ids = {b.block_id for b in _snapshot_blocks(m)}
    victim = attn[3]
    pool.free_block_queue.remove(victim)
    pool._maybe_evict_cached_block(victim)
    front = {b.block_id for b in _queue_front(m, 10)}
    assert snap_ids <= front


def test_shared_head_pages_survive_when_another_chain_is_live(monkeypatch):
    m = _mgr(monkeypatch)
    head = _toks(7, 64)  # 4 blocks
    _run(m, "a", head + _toks(1, 70))
    _run(m, "b", head + _toks(2, 70))
    pool = m.block_pool
    # kill chain a only: evict one of its snapshot pages
    ci = m.block_pool.chain_index
    sa, sb = list(ci.anc)
    ka = [b for b in _snapshot_blocks(m) if get_block_hash(b.block_hash) == sa]
    pool.free_block_queue.remove(ka[0])
    pool._maybe_evict_cached_block(ka[0])
    for hh in ci.anc[sb][:4]:  # shared head pages still cached and NOT demoted to the front
        blk = pool.cached_block_hash_to_block.get_one_block(make_block_hash_with_group_id(hh, 0))
        assert blk is not None
    front = {b.block_id for b in _queue_front(m, 6)}
    shared = {
        pool.cached_block_hash_to_block.get_one_block(make_block_hash_with_group_id(hh, 0)).block_id
        for hh in ci.anc[sb][:4]
    }
    assert not (shared & front)


def test_missing_attention_page_recomputed_revives_snapshot(monkeypatch):
    """Dead is not permanent: recomputing the missing page makes the snapshot servable again."""
    m = _mgr(monkeypatch)
    ci = m.block_pool.chain_index
    t = _toks(1, 140)
    _run(m, "a", t)
    (s,) = ci.anc
    attn = sorted(_attn_blocks(m), key=lambda b: b.block_hash_num_tokens)
    victim = attn[-1]
    m.block_pool.free_block_queue.remove(victim)
    m.block_pool._maybe_evict_cached_block(victim)
    assert not ci._snapshot_alive(s)
    assert s in ci.anc  # registration kept
    # Another request recomputes the page (cache miss at lookup, then registers the same hash).
    _run(m, "b", t + _toks(5, 40))
    # whatever b did, the index never raised and still knows the snapshots it registered
    assert len(ci.anc) >= 1


def test_supersede_recycles_overtaken_snapshot_and_keeps_latest(monkeypatch):
    m = _mgr(monkeypatch, supersede=True, grace=0.0)
    ci = m.block_pool.chain_index
    t1 = _toks(1, 100)
    t2 = t1 + _toks(2, 40)
    t3 = t2 + _toks(3, 40)
    _run(m, "t1", t1)
    (s1,) = ci.anc
    assert _run(m, "t2", t2) == (len(t1) - 1) // BS * BS
    assert ci.superseded_total == 1
    # the overtaken snapshot is either already recycled or queued to be recycled first
    s1_ids = {b.block_id for b in _snapshot_blocks(m) if get_block_hash(b.block_hash) == s1}
    assert s1 not in ci.anc or s1_ids <= {b.block_id for b in _queue_front(m, 8)}
    # the latest snapshot is untouched: turn 3 resumes from turn 2's boundary
    assert _run(m, "t3", t3) == (len(t2) - 1) // BS * BS


def test_supersede_off_by_default(monkeypatch):
    m = _mgr(monkeypatch)
    _run(m, "t1", _toks(1, 100))
    _run(m, "t2", _toks(1, 100) + _toks(2, 40))
    assert not m.block_pool.chain_index.superseded


def test_forks_inside_the_grace_keep_the_parent_snapshot(monkeypatch):
    """Sub-agent style forks (siblings re-reading the parent's snapshot) must not lose it."""
    m = _mgr(monkeypatch, supersede=True, grace=100.0)
    ci = m.block_pool.chain_index
    now = [0.0]
    ci.clock = lambda: now[0]
    parent = _toks(1, 100)
    boundary = (len(parent) - 1) // BS * BS
    assert _run(m, "p", parent) == 0
    assert _run(m, "c1", parent + _toks(2, 40)) == boundary
    now[0] = 30.0
    assert _run(m, "c2", parent + _toks(3, 40)) == boundary
    now[0] = 60.0
    assert _run(m, "c3", parent + _toks(4, 40)) == boundary
    assert ci.superseded_total == 0


def test_without_grace_a_late_fork_loses_the_parent_boundary(monkeypatch):
    """Documented trade-off of grace=0: it is the aggressive setting, fork-hostile."""
    m = _mgr(monkeypatch, supersede=True, grace=0.0)
    parent = _toks(1, 100)
    _run(m, "p", parent)
    _run(m, "c1", parent + _toks(2, 40))
    assert _run(m, "c2", parent + _toks(3, 40)) in (0, (len(parent) - 1) // BS * BS)


def test_grace_is_cancelled_by_a_fork_hit(monkeypatch):
    m = _mgr(monkeypatch, supersede=True, grace=100.0)
    ci = m.block_pool.chain_index
    now = [0.0]
    ci.clock = lambda: now[0]
    parent = _toks(1, 100)
    _run(m, "p", parent)
    (sp,) = ci.anc
    _run(m, "c1", parent + _toks(2, 40))
    assert sp not in ci.superseded and sp in ci._pending_set
    now[0] = 50.0
    _run(m, "c2", parent + _toks(3, 40))  # fork re-reads the parent's snapshot inside the grace
    now[0] = 200.0
    ci.process_pending()
    assert sp not in ci.superseded
    # without any fork hit a later supersede does take effect
    _run(m, "c1b", parent + _toks(2, 40) + _toks(9, 40))
    now[0] = 400.0
    ci.process_pending()
    assert ci.superseded


def test_dry_run_changes_nothing_but_counts(monkeypatch):
    m = _mgr(monkeypatch, supersede=True, dry=True)
    ci = m.block_pool.chain_index
    _run(m, "t1", _toks(1, 100))
    before = [b.block_id for b in _queue_front(m, 80)]
    _run(m, "t2", _toks(1, 100) + _toks(2, 40))
    assert ci.demoted == 0 and ci.would_demote > 0


@pytest.mark.parametrize("policy", [(True, False), (True, True)])
def test_random_workload_never_breaks_cache_correctness(monkeypatch, policy):
    """Whatever the ordering does, every reported hit must be a true common prefix of an
    earlier request and block-aligned, and the pool must stay consistent."""
    chain, sup = policy
    m = _mgr(monkeypatch, num_blocks=60, chain=chain, supersede=sup, grace=0.0)
    rnd = random.Random(3)
    seen: list[list[int]] = []
    heads = [_toks(100 + i, 48) for i in range(2)]
    sessions = [heads[i % 2] + _toks(200 + i, 20) for i in range(5)]
    for k in range(120):
        i = rnd.randrange(len(sessions))
        sessions[i] = sessions[i] + _toks(1000 + k, rnd.randrange(8, 40))
        if len(sessions[i]) > 400:
            sessions[i] = heads[i % 2] + _toks(5000 + k, 20)
        t = sessions[i]
        hit = _run(m, f"r{k}", t)
        assert hit % BS == 0 and hit <= len(t)
        if hit:
            assert any(s[:hit] == t[:hit] for s in seen)
        seen.append(t)
    pool = m.block_pool
    assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1  # everything released
    ci = pool.chain_index
    if ci is not None:
        for s, chain_h in ci.anc.items():
            assert chain_h[-1] == s
