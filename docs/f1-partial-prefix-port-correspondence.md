# Correspondence table — PR #53479 (current diff) vs. `frontier-pastnative-20260816`

**Scope decision (flagging prominently, not improvising silently):** this table
targets the CURRENT upstream diff fetched today
(`kamb-code/vllm@b99d152af44e765c181150a2d473daddd2c9d3ab`, saved verbatim to
`upstream-hunks-current.diff`), not the simpler two-hunk version described in
`docs/f1-partial-prefix-hits-research.md` (2026-08-24). See `hunk-drift-notes.md`
for the full diff between the two. The research doc's description is materially
stale — building this table against it would target a version of the PR that no
longer exists upstream. All "our tree" references are to the
`f1-partial-prefix-port` worktree, i.e. `frontier-pastnative-20260816` HEAD
(`1b2010744`), read-only.

Everything below is a **proposal for the senior agent to evaluate**, not applied
to any source file. Every location is `file:line` as of this writing; re-check
before acting since neither tree is frozen.

---

## A. `vllm/v1/core/kv_cache_coordinator.py`

### A1. Base-class `eagle_reach_margin` property (upstream, new)

Upstream adds, on the base `KVCacheCoordinator`:
```python
@property
def eagle_reach_margin(self) -> int:
    """... 0 without speculative decoding or without a pruned attention group."""
    return 0
```

**Our tree:** `KVCacheCoordinator` base class at
`vllm/v1/core/kv_cache_coordinator.py:74`. No `eagle_reach_margin` concept
anywhere in this file today (confirmed via grep across the whole file — zero
hits). **No equivalent; straightforward to add** — the default-0 base property
has no dependency on anything we're missing. Low uncertainty.

### A2. `HybridKVCacheCoordinator._eagle_margin` + `eagle_reach_margin` property (upstream, new)

Upstream's `HybridKVCacheCoordinator.__init__` gains a loop wiring
`manager.eagle_reach_margin = self.eagle_reach_margin` onto every single-type
manager, plus:
```python
def _eagle_margin(self, manager_cls, group_block_size: int) -> int:
    return (
        self.hash_block_size
        if self.enable_partial_hash_hits
        and manager_cls.supports_fine_grained_hash_lookup
        and group_block_size > self.hash_block_size
        else group_block_size
    )

@property
def eagle_reach_margin(self) -> int:
    for spec, group_ids, manager_cls, use_eagle in self.attention_groups:
        if use_eagle and not isinstance(spec, MambaSpec):
            return self._eagle_margin(manager_cls, ...)
    return 0
```
This is a **refactor**, not new logic: upstream's `find_longest_cache_hit`
already computed this exact expression inline (`eagle_margin = self.
hash_block_size if self.enable_partial_hash_hits and manager_cls.
supports_fine_grained_hash_lookup and group_block_size > self.hash_block_size
else group_block_size`) before this PR; #53479 just lifts it into a reusable
method.

**Our tree:** `HybridKVCacheCoordinator` at
`vllm/v1/core/kv_cache_coordinator.py:451`. `verify_and_split_kv_cache_groups`
builds `self.attention_groups` as a plain `list[tuple[KVCacheSpec, list[int],
type[SingleTypeKVCacheManager]]]` (`kv_cache_coordinator.py:496-522`) — **note
the 3-tuple, not upstream's 4-tuple** (upstream's `for spec, group_ids,
manager_cls, use_eagle in self.attention_groups` unpacks 4 values; ours only
ever unpacks 3, e.g. `find_longest_cache_hit`'s `for idx, (spec, group_ids,
manager_cls) in enumerate(self.attention_groups)` at line ~589). Our
`find_longest_cache_hit` (`kv_cache_coordinator.py:546-639`) instead recomputes
`idx in self.eagle_attn_group_indices` per-call against a precomputed
`self.eagle_attn_group_indices: set[int]` (line ~537-542). **Gap: no
equivalent.** Upstream stores per-group eagle membership as a 4th tuple element;
we store it as a separate index set. A literal port of `eagle_reach_margin`
would need to either (a) iterate `self.eagle_attn_group_indices` against
`self.attention_groups` by position instead of unpacking a 4th tuple field, or
(b) widen our tuple to 4 elements (touches `verify_and_split_kv_cache_groups`
and both call sites). (a) is the smaller diff.

**Bigger gap — the margin formula itself has no basis in our tree.** Our
`find_longest_cache_hit`'s actual margin computation
(`kv_cache_coordinator.py:605-611`):
```python
drop_eagle_block = (
    idx in self.eagle_attn_group_indices and idx not in eagle_verified
    and spec.supports_eagle_cache_peek
)
_max_length = curr_hit_length
if drop_eagle_block:
    _max_length = min(curr_hit_length + spec.block_size, max_cache_hit_length)
```
This is **unconditionally one full `spec.block_size`** — there is no
`hash_block_size`-scaled fine-grained variant at all. Confirmed by grep: `enable_
partial_hash_hits` and `supports_fine_grained_hash_lookup` do not exist anywhere
in this fork (`grep -rn` across `vllm/v1/core/*.py` and `kv_cache_interface.py`
returns zero hits for either). Our fork never got the upstream "partial hash
hits" / fine-grained prefix matching feature that `_eagle_margin`'s conditional
depends on. **Adaptation sketch (medium-high uncertainty):** don't port
`_eagle_margin`'s conditional at all — our fork's `eagle_reach_margin` can only
ever mean "one full attention block," i.e.:
```python
@property
def eagle_reach_margin(self) -> int:
    for idx, (spec, group_ids, manager_cls) in enumerate(self.attention_groups):
        if idx in self.eagle_attn_group_indices and not isinstance(spec, MambaSpec):
            return spec.block_size
    return 0
```
This preserves the *isinstance(spec, MambaSpec)* exclusion upstream added
(so the mamba group itself never reports a margin against itself), using
`isinstance` since our fork gates the mamba exclusion via
`spec.supports_eagle_cache_peek` (see `single_type_kv_cache_manager.py:891-902`
docstring) rather than upstream's explicit isinstance check inside
`find_longest_cache_hit` proper — worth double-checking these two exclusion
mechanisms actually agree in all cases (they should, since `MambaSpec.
supports_eagle_cache_peek` is hardcoded `False`, but I have not exhaustively
proven no other spec type could diverge). **Flagging as uncertain**: whether
"one full block, unconditionally" is the *correct* value to feed into the
scheduler's new `eagle_reach` stop, or whether it silently over-retains /
under-retains relative to what our `find_longest_cache_hit` actually drops in
edge cases (e.g. `is_simple_hybrid` early-break path, `eagle_verified` set
interactions across iterations) — I did not trace every branch of our
`find_longest_cache_hit`'s fixed-point loop against this value.

### A3. Refactor call site (upstream: `find_longest_cache_hit` calls `self._eagle_margin(...)` instead of inlining it)

Trivial once A2 is resolved; not a separate porting decision.

---

## B. `vllm/v1/core/sched/scheduler.py` — `_mamba_block_aligned_split` and its `__init__` inputs

### B1. New `__init__` attributes (upstream `scheduler.py:345-354` in the fetched head file)

```python
self.mamba_retention_interval = kv_cache_config.prefix_cache_retention_interval
self.mamba_eagle_reach_margin = (
    self.kv_cache_manager.coordinator.eagle_reach_margin
    if self.need_mamba_block_aligned_split else 0
)
```

**Our tree target:** right after our existing `retain_mamba_align_mtp_cache_block`
block, `vllm/v1/core/sched/scheduler.py:255-259`.

**Gap 1 (small, mechanical):** upstream reads `kv_cache_config.
prefix_cache_retention_interval` — a `KVCacheConfig` field. Our fork has **no
such field**; our coordinator instead reads the env var directly
(`self.retention_interval = envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL`,
`kv_cache_coordinator.py:130`). **Adaptation (low uncertainty, this is the clean
path):** don't add a `KVCacheConfig` field — reuse what's already exposed:
```python
self.mamba_retention_interval = self.kv_cache_manager.coordinator.retention_interval
```
This is available at the same point in `__init__` (the coordinator is
constructed before this block runs; confirmed `self.kv_cache_manager` exists
earlier in `__init__`, same pattern the existing `envs.
VLLM_MAMBA_ALIGN_RETAIN_MTP_CACHE_BLOCK` read at line 258 already relies on
being post-construction).

**Gap 2:** `self.mamba_eagle_reach_margin` depends on `coordinator.
eagle_reach_margin` existing (§A2) — straightforward once A2 lands, same
conditional structure upstream uses (`if self.need_mamba_block_aligned_split
else 0`) transfers directly onto our `self.need_mamba_block_aligned_split`
(already present, `scheduler.py:252-254`).

### B2. Hunk 1 — back-off deletion (upstream `scheduler.py:421-425` in fetched head)

Upstream deletes the `if self.use_eagle: last_cache_position -= block_size`
back-off unconditionally.

**Our exact equivalent:** `vllm/v1/core/sched/scheduler.py:300-301`:
```python
if self.use_eagle and not retain_final_mtp_block:
    last_cache_position = max(last_cache_position - block_size, 0)
```
**This is NOT a literal match** — ours already has a conditional exception
(`retain_final_mtp_block`, gated on `retain_mamba_align_mtp_cache_block` i.e.
`method == "mtp"` + `VLLM_MAMBA_ALIGN_RETAIN_MTP_CACHE_BLOCK`, `scheduler.py:
296-299`), which is the open design decision the plan digest already surfaced
(keep our narrow MTP-only relaxation vs. adopt upstream's unconditional
removal). **Not re-litigating that decision here** (per the plan digest, "not
mine to resolve") — flagging the mechanical fact that upstream's hunk 1 is a
strict superset of what we already have: if hunk 1 + hunk 2 land as designed
(with the retention-aware boundary/replay/eagle-reach stops backing it), our
`retain_final_mtp_block`/`retain_mamba_align_mtp_cache_block` gate becomes dead
code (subsumed), per the research doc's own §4 analysis. Low uncertainty on the
mechanics; the uncertainty is entirely the product decision, already flagged
upstream in the plan digest.

### B3. Hunk 2 — the `stops` tuple redesign (upstream `scheduler.py:447-501` in fetched head, full function reproduced below for reference)

This is the core of the port and the biggest gap. Full upstream function,
fetched from the current PR head (not the diff snippet — the surrounding
pre-existing context matters):

```python
def _mamba_block_aligned_split(self, request, num_new_tokens,
                                num_new_local_computed_tokens=0,
                                num_external_computed_tokens=0) -> int:
    start = (request.num_computed_tokens + num_new_local_computed_tokens
             + num_external_computed_tokens)
    prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
    if start >= prefill_end:
        return num_new_tokens

    block_size = self.cache_config.block_size
    last_cache_position = request.num_tokens - request.num_tokens % block_size
    # (back-off deleted here -- see B2)

    end = start + num_new_tokens
    use_internal_checkpoint = (
        self.mamba_has_prefill_checkpoint_blocks and start % block_size == 0
    )
    if use_internal_checkpoint:
        last_cache_position = 0
    if end < prefill_end and not use_internal_checkpoint:
        max_prefill_tokens = self.max_num_scheduled_tokens
        long_prefill_threshold = self.scheduler_config.long_prefill_token_threshold
        if long_prefill_threshold > 0:
            max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)
        aligned_end = end // block_size * block_size
        if aligned_end > start or block_size <= max_prefill_tokens:
            end = aligned_end

    next_block_boundary = (start // block_size + 1) * block_size
    tail_boundary = (
        request.num_prompt_tokens // self.hash_block_size * self.hash_block_size
        if self.mamba_partial_cache_hit else 0
    )
    retention = self.mamba_retention_interval
    if use_internal_checkpoint or retention == 0:
        boundary_stop = 0
    elif retention is None or retention <= block_size:
        boundary_stop = next_block_boundary
    else:
        boundary_stop = (start // retention + 1) * retention
    replay_boundary = eagle_reach = 0
    if retention is not None and not use_internal_checkpoint:
        replay_end = request.num_prompt_tokens - 1
        replay_boundary = replay_end // block_size * block_size
        if self.mamba_eagle_reach_margin > 0:
            eagle_reach = max(
                (replay_end - self.mamba_eagle_reach_margin) // block_size * block_size, 0
            )
    stops = (
        boundary_stop,
        replay_boundary,
        eagle_reach,
        last_cache_position,
        tail_boundary if last_cache_position < tail_boundary < request.num_prompt_tokens else 0,
        start + (request.shared_prefix_boundary - start) // block_size * block_size
            if start < request.shared_prefix_boundary < end else 0,
    )
    end = min((s for s in stops if start < s < end), default=end)
    return max(end - start, 0)
```

**Our exact current function** (full body, `vllm/v1/core/sched/scheduler.py:
267-311`):

```python
def _mamba_block_aligned_split(self, request, num_new_tokens,
                                num_new_local_computed_tokens=0,
                                num_external_computed_tokens=0) -> int:
    num_computed_tokens = (request.num_computed_tokens
        + num_new_local_computed_tokens + num_external_computed_tokens)
    prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
    if num_computed_tokens < prefill_end:
        block_size = self.cache_config.block_size
        last_cache_position = round_down(request.num_tokens, block_size)
        retain_final_mtp_block = (
            self.retain_mamba_align_mtp_cache_block
            and last_cache_position < request.num_tokens
        )
        if self.use_eagle and not retain_final_mtp_block:
            last_cache_position = max(last_cache_position - block_size, 0)

        chunk_end = num_computed_tokens + num_new_tokens
        if num_computed_tokens < last_cache_position:
            chunk_end = min(round_down(chunk_end, block_size), last_cache_position)
        elif chunk_end < prefill_end:
            chunk_end = round_down(chunk_end, block_size)

        num_new_tokens = max(chunk_end - num_computed_tokens, 0)
    return num_new_tokens
```

**Line-by-line correspondence:**

| Upstream concept | Our equivalent (file:line) | Status |
|---|---|---|
| `start = num_computed_tokens + ...` | `num_computed_tokens = ...` (`scheduler.py:268-272`) | Same value, different name (`start` vs `num_computed_tokens`). Cosmetic. |
| `prefill_end = max(...)` | identical, `scheduler.py:283` | Exact match. |
| `if start >= prefill_end: return` (early return) | inverted as `if num_computed_tokens < prefill_end:` wrapping the rest (`scheduler.py:284`) | Logically equivalent, structured as guard vs. wrap. Any port must preserve one of these two equivalent control-flow shapes — trivial. |
| `last_cache_position = request.num_tokens - request.num_tokens % block_size` | `last_cache_position = round_down(request.num_tokens, block_size)` (`scheduler.py:290`) | **Confirmed identical.** `round_down(x, y) = (x // y) * y` (`vllm/utils/math_utils.py:25-27`) — equal to `x - x % y` for non-negative integers by the standard floor-division identity. No uncertainty. |
| back-off (`if self.use_eagle: ... -= block_size`) | see B2 (`scheduler.py:296-301`) — **already conditional on our fork**, not a straight deletion target | Design-decision gap, not mechanical. |
| `end = start + num_new_tokens` | `chunk_end = num_computed_tokens + num_new_tokens` (`scheduler.py:303`) | Same value, different name. |
| `use_internal_checkpoint = self.mamba_has_prefill_checkpoint_blocks and start % block_size == 0` | **NO EQUIVALENT** | **Gap.** `self.mamba_has_prefill_checkpoint_blocks` does not exist anywhere in our `scheduler.py` (confirmed via grep — zero hits for `prefill_checkpoint`, `internal_checkpoint`, `has_prefill_checkpoint`). Depends on `MambaSpec.num_prefill_checkpoint_blocks`, which also doesn't exist in our `kv_cache_interface.py` (confirmed via grep — zero hits). This is upstream #52789 (internal prefill checkpoints), confirmed absent per the research doc and independently re-confirmed here. **Adaptation: `use_internal_checkpoint` is always `False` on our fork** — every place upstream branches on it collapses to the `else`/`False` arm. This matches the plan digest's inference ("the doc doesn't say this explicitly, but it falls out of the diff") — I'm now confirming it directly by exhaustive grep rather than inference: there is no path by which our fork could construct a non-`False` value for this without first porting #52789 in full (a separate, larger prerequisite, out of scope here). **Low uncertainty on the conclusion, medium uncertainty on scope** — I have not audited whether #52789 has other, non-obvious footprint elsewhere in the file the way `shared_prefix_boundary` turned out to (see B3 below) — flagging as a spot-check item for whoever ports this. |
| the `end < prefill_end` re-alignment block (long-prefill threshold clamp) | **NO EQUIVALENT — this entire block is absent from our function.** | **Gap, independent of #53479.** Our function has no chunked-prefill-budget-vs-block-size reconciliation at all; it always rounds `chunk_end` down to `block_size` once past `last_cache_position` (`scheduler.py:308-309`), with no `long_prefill_token_threshold`/`max_num_scheduled_tokens` interaction. This logic is **not part of the #53479 diff** (present in upstream before and after this PR) but is a real structural gap this correspondence table should flag since a straight copy-paste of upstream's function would silently pull in unrelated behavior change (a chunk-size clamp we don't have) alongside the #53479-specific pieces. **Recommend NOT porting this block as part of #53479** — it's out of scope for this specific fix and deserves its own separate research/port pass if wanted. Flagging as a scope boundary, not a #53479 gap per se. |
| `next_block_boundary = (start // block_size + 1) * block_size` | **NO EQUIVALENT (as a named value)** — the *behavior* exists inline via `round_down(chunk_end, block_size)` at `scheduler.py:306,309`, but there is no standalone "next boundary from `start`" computation | **Gap.** Would need to be introduced fresh as part of the `stops`-tuple reimplementation (see below). Mechanical, low uncertainty. |
| `tail_boundary` (`mamba_partial_cache_hit`) | **NO EQUIVALENT.** `mamba_partial_cache_hit` does not exist in our `scheduler.py` (confirmed via grep). Depends on `self.hash_block_size < self.block_size` and `coordinator.enable_partial_hash_hits` — the latter confirmed absent from our fork entirely (§A2). | **Out of scope for #53479 itself** (this stop is pre-existing upstream, not part of this PR's diff) — but flagging because a literal transplant of the `stops` tuple would break immediately without it. **Recommend: omit this stop entirely** when reimplementing against our shape (equivalent to always `0`, i.e. our fork's fine-grained partial-tail registration doesn't exist, so there is nothing for this stop to protect). |
| the Marconi `shared_prefix_boundary` stop | **NO EQUIVALENT**, and deeper than a missing scheduler field: `Request.shared_prefix_boundary` does not exist on our `vllm/v1/request.py` (confirmed via grep), and our `KVCacheManager.get_computed_blocks` (`vllm/v1/core/kv_cache_manager.py:183`) returns a **2-tuple** `tuple[KVCacheBlocks, int]`, not upstream's 3-tuple `(blocks, num_local, shared_prefix_boundary)`. | **Out of scope for #53479 itself** (pre-existing upstream), but structurally the deepest gap in the whole table — porting this stop for real would mean changing a public method's return arity and threading a new `Request` field through the entire scheduling path, matching the plan digest's note that `origin/vllm-2080ti-definitive-0.2.x` has this ("Marconi junction stop") and `frontier-pastnative-20260816` does not. **Recommend: omit this stop too**, same as `tail_boundary` — always `0` on our fork, consistent with 0.2.x being the "closer landing zone" if that ever becomes the serving line (plan digest §5, not re-litigated here). |
| `boundary_stop` (retention-aware, new in #53479) | **This is the actual #53479 payload for our shape.** No direct equivalent, but our fork's `MambaManager.reachable_block_mask` (`single_type_kv_cache_manager.py:1128-1181`) already independently encodes "how sparse is the retained state grid" via `retention_interval`/`segment_tokens`/`per_segment` — **the scheduler-side and manager-side retention concepts are currently NOT unified on our fork** (the scheduler doesn't know about `retention_interval` at all today; only the coordinator/manager do). Adaptation sketch: introduce `self.mamba_retention_interval` (B1) and reproduce upstream's 3-way `boundary_stop` branch verbatim — the branch logic itself has no fork-specific obstruction, it only needs `next_block_boundary` (mechanical, see above) and `use_internal_checkpoint` (always `False` per above, which simplifies the branch to just `if retention == 0: boundary_stop = 0 elif retention is None or retention <= block_size: boundary_stop = next_block_boundary else: boundary_stop = (start // retention + 1) * retention` — the `use_internal_checkpoint or` clause is dead weight we could drop or keep as a no-op for forward-compatibility with a future #52789 port). **Medium uncertainty**: whether `start` (upstream's variable) should map to our `num_computed_tokens` for this computation given our current early-return structure guards on `num_computed_tokens < prefill_end` rather than upstream's `start >= prefill_end` early-return — I believe these are equivalent (same variable, different name, per row 3 above) but have not traced every call site (`scheduler.py:404-405`, `677-678`, mirrored at `1030-1031` in the upstream head file for the async/resumed-request path) to confirm both trees call this function identically at every site. |
| `replay_boundary`, `eagle_reach` (new in #53479) | **This is the other #53479 payload.** No equivalent at all in our scheduler. Depends on B1 (`self.mamba_eagle_reach_margin`, itself depending on §A2's medium-high-uncertainty margin formula). Mechanical once B1/A2 land — the arithmetic (`replay_end // block_size * block_size`, the `max(..., 0)` clamp) has no fork-specific obstruction. | Gap, but a clean one — flagged uncertainty is entirely upstream in A2, not here. |
| final `stops = (...)` tuple + `min(...)` reduction | **Reimplement fresh**, restricted to the subset our fork actually has: `(boundary_stop, replay_boundary, eagle_reach, last_cache_position)` — a 4-tuple, dropping `tail_boundary` and the `shared_prefix_boundary` term (both structurally absent, see above rows). | This is the concrete proposal: our ported `stops` tuple should NOT be a literal copy of upstream's 6-tuple; it should be the 4 elements our fork can actually support, computed as upstream computes them, with the other two omitted (equivalent to being permanently `0` / never firing). **Flagging as the single highest-value, highest-confidence proposal in this document** — everything upstream needs for the #53479 fix specifically (as opposed to the pre-existing `tail_boundary`/Marconi machinery) is present in this 4-tuple. |
| `end = min((s for s in stops if start < s < end), default=end)` / `return max(end - start, 0)` | our `chunk_end = min(round_down(chunk_end, block_size), last_cache_position)` / `elif chunk_end < prefill_end: chunk_end = round_down(...)` / `return max(chunk_end - num_computed_tokens, 0)` (`scheduler.py:304-311`) | **Structural rewrite required**, not a tweak. Upstream's shape is "compute all candidate stop positions, take the minimum one strictly inside `(start, end)`, clamp to that." Ours is "conditionally round down OR cap at `last_cache_position`, branching on whether we're before or after `last_cache_position`." These produce equivalent results for the cases both shapes handle today, but upstream's generalizes to N stops uniformly while ours is hand-specialized for exactly one (`last_cache_position`). **This is the crux of the reimplementation**: adopting the `stops`-tuple pattern (even the trimmed 4-element version) means replacing our two-branch `if/elif` with upstream's `min(...)`-over-candidates pattern. Low uncertainty that this is *necessary*; medium uncertainty on whether the trimmed 4-tuple version is fully behavior-preserving for every existing test in `test_prefix_caching.py` (`test_mamba_align_prefill_split_keeps_intermediate_chunks_aligned`, `test_mamba_align_eagle_split_stops_at_reusable_boundary`) without `retention=None` (our default) collapsing `boundary_stop` back to exactly `next_block_boundary`, which needs to be **exhaustively checked against those two existing tests**, not just asserted. I did not hand-simulate every branch of both existing tests against the proposed rewrite — flagging this as the top item for the senior agent to verify before landing, and it's exactly what `test_partial_prefix_boundary_stops.py` (§ Task 4 deliverable) is scaffolded to check once real code exists to run it against. |

---

## C. `vllm/v1/core/single_type_kv_cache_manager.py`

### C1. Base class `eagle_reach_margin: int = 0` + `_reachable_boundaries` refactor

Upstream adds a class attribute and refactors `cache_blocks`'s inline
`reachable_boundaries = [request.num_prompt_tokens - 1]` (+ conditional
`shared_prefix_boundary` append) into an overridable `_reachable_boundaries(self,
request)` instance method.

**Our tree: structurally incompatible, not just "missing."** Our
`reachable_block_mask` (`single_type_kv_cache_manager.py:334-353` base,
`:1128-1181` `MambaManager` override) is a **`@classmethod`**, called from
`cache_blocks` (`:277-330`) with explicit params (`retention_interval`,
`num_prompt_tokens`, etc.) rather than reading instance state. A `@classmethod`
has no `self` and cannot read a per-instance `self.eagle_reach_margin` the way
upstream's new `MambaManager._reachable_boundaries(self, request)` does.
**This is the highest-uncertainty structural gap in the whole table.**

Two adaptation paths, both viable, presenting trade-offs I'm not resolving here:

- **(a) Thread `eagle_reach_margin` as an explicit classmethod parameter**,
  mirroring how `retention_interval`/`num_prompt_tokens` already flow in: add
  `eagle_reach_margin: int = 0` to `reachable_block_mask`'s signature (base and
  `MambaManager` override), and have `cache_blocks` (`:277-330`, an *instance*
  method, so it CAN read `self.eagle_reach_margin`) pass it through explicitly:
  `block_mask = self.reachable_block_mask(..., eagle_reach_margin=self.
  eagle_reach_margin)`. Smaller diff, preserves the classmethod boundary,
  consistent with our fork's existing style of explicit params over instance
  reads at this boundary.
- **(b) Convert `reachable_block_mask` from classmethod to instance method.**
  Bigger diff (touches the base class contract and any other override), closer
  to upstream's actual shape, but not obviously required — nothing else in our
  fork's `reachable_block_mask` needs instance state today, and changing the
  method's binding type is the kind of change that should probably get its own
  review rather than ride in with #53479.

**I lean toward (a)** but am flagging this as genuinely uncertain — it's a
judgment call about matching upstream's shape vs. minimizing diff surface, and
I don't have enough visibility into whether a future upstream sync would make
(a) or (b) less painful to reconcile later. Either way, `eagle_reach_margin`
still needs to land on the coordinator (§A) and be threaded down to whichever
shape is chosen here.

### C2. `MambaManager`-specific retention logic — partial overlap already exists

This is the most interesting finding in this section: **our fork's
`reachable_block_mask` already implements roughly half of what upstream's new
`MambaManager._reachable_boundaries` adds**, independently, as part of the
pre-existing `#45845` retention-interval port. Our version
(`single_type_kv_cache_manager.py:1171-1179`):

```python
# (2) Replay boundary. `find_longest_cache_hit` caps hits at
# `num_prompt - 1`, so an exact prompt replay lands on the latest
# fine-aligned boundary. Sparse retention would otherwise skip its
# state, so keep it explicitly.
if num_prompt_tokens is not None:
    latest = (num_prompt_tokens - 1) // alignment_tokens * alignment_tokens
    boundary_block = latest // block_size - 1
    if start_block <= boundary_block < end_block:
        mask[boundary_block - start_block] = True
```

already retains the **replay boundary** (`num_prompt_tokens - 1`, floored)
under sparse retention — functionally the same intent as upstream's new base
`_reachable_boundaries`'s `[request.num_prompt_tokens - 1]`. **What's actually
missing is only the second half**: upstream's `MambaManager._reachable_boundaries`
override additionally retains `request.num_prompt_tokens - 1 -
self.eagle_reach_margin` when `eagle_reach_margin > 0` — the EAGLE/MTP-adjusted
retention point. Our mask has no equivalent second retained position at all.

**Adaptation sketch:** extend part (2) of our `reachable_block_mask` (right
after the existing replay-boundary block, `:1179`) with a third clause:
```python
# (3) EAGLE/MTP-reachable boundary (ported from vLLM #53479): under a
# pruned speculative lookup, the deepest reusable state sits
# eagle_reach_margin tokens below the replay boundary.
if num_prompt_tokens is not None and eagle_reach_margin:  # new param, see C1
    eagle_latest = max(num_prompt_tokens - 1 - eagle_reach_margin, 0)
    eagle_latest = eagle_latest // alignment_tokens * alignment_tokens
    eagle_boundary_block = eagle_latest // block_size - 1
    if start_block <= eagle_boundary_block < end_block:
        mask[eagle_boundary_block - start_block] = True
```
Mechanical once C1's parameter-threading question is settled. Low uncertainty
on the arithmetic (mirrors the existing block 2 pattern closely); the only
open question is the same C1 plumbing question.

**Practically inert today regardless of how it's ported**: `reachable_block_mask`
returns `None` (dense, no masking at all) whenever `retention_interval is None`
(`:1148-1150`), which is our fork's default (`envs.py:269`,
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL = None`). This whole section (C2) only
activates if `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` is ever set to `0` or a
segment value in our deployment — which, per `hunk-drift-notes.md` §6, it
currently is not. Flagging so the senior agent doesn't over-prioritize this
half of the port if the retention interval is staying dense in production;
it's real but currently dormant, same shape as the original #53479-inertness
finding the research doc made about upstream, just one layer further down our
stack.

---

## Summary: what's genuinely new-and-needed vs. out-of-scope-but-would-break-a-literal-copy

**In scope for a #53479 port onto our fork (the actual fix):**
- A1 (`eagle_reach_margin` base property) — mechanical.
- A2 (`HybridKVCacheCoordinator.eagle_reach_margin`) — needs the simplified,
  no-fine-grained-hash-hits formula (medium-high uncertainty on the exact
  right value).
- B1 (`mamba_retention_interval`/`mamba_eagle_reach_margin` scheduler init) —
  mechanical, reuse `coordinator.retention_interval` directly.
- B3's `boundary_stop`/`replay_boundary`/`eagle_reach` computation and the
  trimmed 4-element `stops` tuple + `min(...)`-reduction rewrite — the real
  payload; mostly mechanical, one item (exhaustive test-compatibility check)
  flagged high-priority.
- C1/C2 (`MambaManager` retention-mask extension) — mechanical once C1's
  parameter-threading question is settled; currently dormant given our dense
  default.
- B2's back-off deletion — mechanical, but gated on the open MTP-gate-vs-
  upstream-unconditional product decision the plan digest already surfaces.

**Out of scope (pre-existing upstream machinery this PR's diff does not
touch, but a literal copy-paste would require anyway) — recommend omitting
these when reimplementing against our shape:**
- `use_internal_checkpoint` / `mamba_has_prefill_checkpoint_blocks` (#52789) —
  always `False` on our fork; omit, don't stub.
- The long-prefill-threshold chunk-size clamp block — unrelated behavior,
  don't pull in.
- `tail_boundary` / `mamba_partial_cache_hit` — depends on fine-grained hash
  hits our fork doesn't have; omit (equivalent to always 0).
- `shared_prefix_boundary` / Marconi stop — deepest gap (touches `Request` and
  `get_computed_blocks`'s return arity); omit (equivalent to always 0).

## Top 3 highest-uncertainty spots (as requested)

1. **A2 — the `eagle_reach_margin` formula for our fork.** Upstream's version
   is conditioned on `enable_partial_hash_hits`/`supports_fine_grained_hash_lookup`,
   neither of which exists on our fork; I've proposed collapsing it to
   "unconditionally one full `spec.block_size`" (matching our fork's actual,
   simpler `find_longest_cache_hit` drop behavior) but have not exhaustively
   traced our coordinator's fixed-point loop (`eagle_verified` set,
   `is_simple_hybrid` early break) to prove this value is correct in every
   case, only the common case.
2. **B3 — the `stops`-tuple rewrite's test-compatibility.** I've proposed
   trimming upstream's 6-element tuple to the 4 elements our fork can support
   and rewriting our two-branch conditional into the `min(...)`-over-candidates
   pattern, and I believe (but have not hand-simulated exhaustively) that this
   preserves both existing tests in `test_prefix_caching.py`
   (`test_mamba_align_prefill_split_keeps_intermediate_chunks_aligned`,
   `test_mamba_align_eagle_split_stops_at_reusable_boundary`) when
   `mamba_retention_interval=None` (our default, which should make `boundary_stop`
   always equal `next_block_boundary`, reproducing today's dense behavior). This
   needs to be verified against real code, not just this table.
3. **C1 — classmethod-vs-instance-method for `reachable_block_mask`.** A real
   architectural fork-in-the-road (pun noted) between minimizing diff size
   (thread `eagle_reach_margin` as an explicit param, path (a)) and matching
   upstream's shape more closely for easier future syncs (convert to instance
   method, path (b)). I lean (a) but this is a judgment call for the senior
   agent / Kevin, not something I resolved.

