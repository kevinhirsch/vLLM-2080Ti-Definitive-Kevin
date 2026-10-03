# SPDX-License-Identifier: Apache-2.0
"""Chain-aware eviction ordering for the hybrid (full-attention + Mamba/GDN) prefix cache.

Why (Lane R2, 2026-10-03). A hit on a hybrid model needs BOTH a recurrent-state snapshot at
the hit boundary (every Mamba group) AND every full-attention page from block 0 up to that
boundary.  The shared free queue is plain LRU over blocks, so it is blind to that coupling:

  * when a snapshot page is reused, the attention pages that only that snapshot could ever
    unlock stay resident as dead weight until they age out;
  * when an attention page is reused, every snapshot above it is unreachable, but its
    snapshot pages (one per Mamba group) also stay resident.

Dead pages crowd out live ones and, under cyclic replay, turn a pool that is "almost big
enough" into a 0 % hit cliff (see tools/r2/cyclic.py).  This is the leaf-pruning half of
Marconi (arXiv 2411.19379): never keep a prefix node that no checkpoint can serve.

What this does: tracks, per snapshot, its ancestor attention hashes (a radix-tree edge list
without a tree), and when a cache entry is evicted it moves the pages that just became
unreachable to the FRONT of the free queue so they are recycled next.  Hashes and contents
are untouched: only the eviction ORDER changes, so a mis-judgement can cost a hit but can
never serve wrong data.  Off unless VLLM_R2_CHAIN_AWARE_EVICT=1 (and only for a shared pool
whose groups all use the hash block size).
"""
from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING, Callable

from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    KVCacheBlock,
    get_block_hash,
    get_group_id,
    make_block_hash_with_group_id,
)

from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool


class HybridChainIndex:
    def __init__(self, pool: "BlockPool", attn_gids: set[int], snap_gids: set[int]):
        self.pool = pool
        self.attn_gids = frozenset(attn_gids)
        self.snap_gids = frozenset(snap_gids)
        self._attn_gid = min(attn_gids)
        # snapshot hash -> ancestor attention hashes (block 0 .. snapshot block, inclusive)
        self.anc: dict[BlockHash, tuple[BlockHash, ...]] = {}
        # attention hash -> snapshot hashes whose chain passes through it
        self.desc: dict[BlockHash, set[BlockHash]] = {}
        self.demoted = 0
        self.would_demote = 0
        self.log_every = 500
        self.superseded_total = 0
        self._registered = 0
        self.snapshots_killed = 0
        # dry-run: track but never reorder (used by tools/r2 to measure dead weight)
        self.dry_run = False
        # Opt-in second rule: a snapshot overtaken by a deeper one on the same lineage
        # (and not a branch point) is demoted. Off by default (VLLM_R2_SUPERSEDE=1).
        self.supersede = False
        self.superseded: set[BlockHash] = set()
        # Fork guard: a superseded snapshot is only recycled first after this grace
        # period without a hit (a sub-agent fork re-reads its parent's snapshot).
        self.supersede_grace_s = 0.0
        self.clock: Callable[[], float] = time.monotonic
        self._pending: deque[tuple[float, BlockHash]] = deque()
        self._touched_since: dict[BlockHash, bool] = {}
        self._pending_set: set[BlockHash] = set()

    # ---- registration -------------------------------------------------
    def on_snapshot_cached(self, h: BlockHash, chain: tuple[BlockHash, ...]) -> None:
        if h in self.anc:
            return
        self.anc[h] = chain
        for a in chain:
            self.desc.setdefault(a, set()).add(h)
        self._registered += 1
        if self._registered % self.log_every == 0:
            self.log_stats()
        if self.supersede:
            # idx == len-2 is the adjacent boundary: an identical resend and a longer
            # sibling of the same prompt resume one unit apart and BOTH are reachable
            # (get_replay_boundaries), so only snapshots at least two units below qualify.
            for idx in range(len(chain) - 3, -1, -1):
                j = chain[idx]
                if j in self.anc and j not in self.superseded and not self._is_branch_point(j, idx):
                    if self.supersede_grace_s > 0:
                        if j not in self._pending_set:
                            self._pending_set.add(j)
                            self._touched_since.pop(j, None)
                            self._pending.append((self.clock() + self.supersede_grace_s, j))
                    else:
                        self._mark_superseded(j)
        self.process_pending()

    def _mark_superseded(self, j: BlockHash) -> None:
        self.superseded_total += 1
        self.superseded.add(j)
        for g in self.snap_gids:
            self._demote(self._cached(j, g))

    def process_pending(self) -> None:
        now = self.clock()
        pending = self._pending
        while pending and pending[0][0] <= now:
            _, j = pending.popleft()
            self._pending_set.discard(j)
            if j in self.anc and j not in self.superseded and not self._touched_since.pop(j, False):
                self._mark_superseded(j)

    def _is_branch_point(self, a: BlockHash, idx: int) -> bool:
        """True if snapshots below ``a`` diverge right after it (a shared junction)."""
        nxt = None
        for s in self.desc.get(a, ()):
            c = self.anc.get(s)
            if c is None or len(c) <= idx + 1 or c[idx] != a:
                continue
            if nxt is None:
                nxt = c[idx + 1]
            elif c[idx + 1] != nxt:
                return True
        return False

    def on_touch(self, blocks) -> None:
        """A cache hit on a superseded snapshot proves it is a branch point: keep it."""
        if not self.superseded and not self._pending:
            return
        for blk in blocks:
            if blk.block_hash is not None and get_group_id(blk.block_hash) in self.snap_gids:
                h = get_block_hash(blk.block_hash)
                self.superseded.discard(h)
                if h in self._pending_set:
                    self._touched_since[h] = True

    def evict_first(self, block: KVCacheBlock) -> bool:
        """Free-path hook: a released superseded snapshot page is recycled first."""
        if not self.supersede or block.block_hash is None or self.dry_run:
            return False
        if get_group_id(block.block_hash) not in self.snap_gids:
            return False
        return get_block_hash(block.block_hash) in self.superseded

    # ---- liveness ------------------------------------------------------
    def _cached(self, h: BlockHash, gid: int) -> KVCacheBlock | None:
        return self.pool.cached_block_hash_to_block.get_one_block(
            make_block_hash_with_group_id(h, gid)
        )

    def _snapshot_alive(self, s: BlockHash) -> bool:
        chain = self.anc.get(s)
        if chain is None:
            return False
        for g in self.snap_gids:
            if self._cached(s, g) is None:
                return False
        for a in chain:
            if self._cached(a, self._attn_gid) is None:
                return False
        return True

    def _demote(self, blk: KVCacheBlock | None) -> None:
        # Only blocks sitting in the free queue can be reordered; in-use blocks are freed
        # later through the normal path.
        if blk is None or blk.is_null or blk.ref_cnt != 0 or blk.prev_free_block is None:
            return
        if self.dry_run:
            self.would_demote += 1
            return
        q = self.pool.free_block_queue
        if q.fake_free_list_head.next_free_block is blk:
            return
        q.remove(blk)
        q.prepend_n([blk])
        self.demoted += 1

    def _forget_snapshot(self, s: BlockHash) -> None:
        chain = self.anc.pop(s, None)
        self.superseded.discard(s)
        if chain is None:
            return
        for a in chain:
            d = self.desc.get(a)
            if d is not None:
                d.discard(s)
                if not d:
                    del self.desc[a]

    def _demote_unreachable(self, s: BlockHash) -> None:
        """Snapshot ``s`` can no longer be served: recycle its pages first, and the
        attention pages that no other servable snapshot needs.  Registration is kept:
        recomputing a missing attention page makes ``s`` servable again (lazy liveness)."""
        chain = self.anc.get(s)
        if chain is None:
            return
        self.snapshots_killed += 1
        for g in self.snap_gids:
            self._demote(self._cached(s, g))
        # Prefix property: a page with a live descendant snapshot makes all its own
        # ancestors live too, so the walk from the leaf stops at the first live page.
        for a in reversed(chain):
            if self._has_live_desc(a):
                break
            self._demote(self._cached(a, self._attn_gid))

    def _has_live_desc(self, a: BlockHash) -> bool:
        d = self.desc.get(a)
        if not d:
            return False
        for s in tuple(d):
            if self._snapshot_alive(s):
                return True
        return False

    # ---- eviction events ----------------------------------------------
    def on_evicted(self, removed: list[BlockHashWithGroupId]) -> None:
        for key in removed:
            h = get_block_hash(key)
            gid = get_group_id(key)
            if self._cached(h, gid) is not None:
                continue  # a duplicate copy of this entry is still cached
            if gid in self.snap_gids:
                if h in self.anc:
                    self._demote_unreachable(h)
                    self._forget_snapshot(h)
            elif gid in self.attn_gids:
                d = self.desc.get(h)
                if d:
                    for s in tuple(d):
                        self._demote_unreachable(s)

    def clear(self) -> None:
        self.anc.clear()
        self.desc.clear()
        self.superseded.clear()
        self._pending.clear()
        self._pending_set.clear()
        self._touched_since.clear()

    def log_stats(self) -> None:
        d = self.dead_weight()
        logger.info(
            "R2 chain index%s: snapshots=%d (live %d) hashed pages: attn %d (dead %d) snap %d (dead %d); "
            "demoted=%d would_demote=%d superseded=%d killed=%d",
            " [DRY-RUN]" if self.dry_run else "", len(self.anc), d["live_snapshots"], d["attn"],
            d["attn_dead"], d["snap"], d["snap_dead"], self.demoted, self.would_demote,
            self.superseded_total, self.snapshots_killed)

    # ---- diagnostics (not used on the hot path) ------------------------
    def dead_weight(self) -> dict[str, int]:
        """Count cached pages that no live snapshot can ever serve."""
        pool = self.pool
        live_attn: set[BlockHash] = set()
        live_snap = 0
        for s, chain in self.anc.items():
            if self._snapshot_alive(s):
                live_snap += 1
                live_attn.update(chain)
        n_attn = n_attn_dead = n_snap = n_snap_dead = 0
        free_attn_dead = 0
        for b in pool.blocks:
            if b.block_hash is None:
                continue
            h, g = get_block_hash(b.block_hash), get_group_id(b.block_hash)
            if g in self.attn_gids:
                n_attn += 1
                if h not in live_attn:
                    n_attn_dead += 1
            elif g in self.snap_gids:
                n_snap += 1
                if h not in self.anc or not self._snapshot_alive(h):
                    n_snap_dead += 1
        return dict(attn=n_attn, attn_dead=n_attn_dead, snap=n_snap, snap_dead=n_snap_dead,
                    live_snapshots=live_snap)
