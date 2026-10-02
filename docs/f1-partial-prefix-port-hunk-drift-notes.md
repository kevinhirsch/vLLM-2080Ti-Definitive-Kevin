# Hunk drift notes — PR #53479 vs. research doc (2026-08-24) vs. live fetch (2026-08-26)

Source of truth for this file: `gh pr diff 53479 --repo vllm-project/vllm`, saved
verbatim to `upstream-hunks-current.diff` in this directory (1547 lines), plus the
full upstream head files fetched read-only into `upstream-head/` (from
`kamb-code/vllm@b99d152af44e765c181150a2d473daddd2c9d3ab`, the current PR head SHA)
for context around the diff hunks. Compared against
`docs/f1-partial-prefix-hits-research.md` (written 2026-08-24) and the plan digest.

## Bottom line

**The PR has been substantially rewritten since the research doc was written.**
This is not "the same two hunks, lightly edited" — it is a different, larger design.
Anyone porting from the research doc's description alone would be porting a version
of this PR that no longer exists upstream. `correspondence.md` in this directory
targets the CURRENT diff (fetched today), not the doc's description — see the note
at the top of that file.

## 1. CI / review state (live, 2026-08-26 fetch)

- `state`: OPEN, `mergeable`: MERGEABLE, `reviewDecision`: REVIEW_REQUIRED —
  unchanged from the research doc. Still zero maintainer/codeowner review.
- `pre-run-check` (pre-commit workflow): **FAILURE**, ran `19:51:50Z`-`19:51:56Z`
  (a 6-second run — consistent with a fast lint/pre-commit-style gate, not the
  full test suite). **Correction after double-checking my own first pass on
  this:** I initially misread this as stale/pre-dating the current head. It is
  not — `gh pr view --json commits` shows exactly 2 commits on this PR, and the
  current head (`b99d152`, "retention-aware boundary stops; keep the EAGLE...")
  has `committedDate: 2026-08-24T19:20:51Z`, ~31 minutes before this check ran;
  kamb-code's "Pushed (`b99d152`)" comment lands at `19:52:00Z`, 4 seconds after
  the check completed — consistent with the local commit (19:20:51) being
  pushed to GitHub around 19:51:5x (triggering CI immediately) and the comment
  following right after. **So this check most likely DID run against the
  current head** and failed fast — almost certainly a lint/pre-commit issue,
  not a logic failure, but still an unresolved red X on the current diff.
- `pre-commit` job: SKIPPED. DCO: SUCCESS. Meta Internal-Only Changes Check:
  SUCCESS. Mergify Summary: SUCCESS. readthedocs build: SUCCESS.
- No full upstream test-suite CI run recorded (still needs a maintainer `/ci run`).
- Last activity: `2026-08-25T07:06:31Z`.
- **Verdict: still a moving spike, now more so.** Treat as reference-only, not a
  stable cherry-pick target — same conclusion as the research doc, stronger now
  given the scope grew and CI hasn't re-validated the new scope at all.

## 2. Last 3 review comments (chronological, verbatim source: PR #53479)

1. **kamb-code, 2026-08-24T19:52:00Z** — "Pushed (`b99d152`): both pieces, with
   tests." Describes two new mechanisms: (a) "Retention-aware stops" — the split
   now stops only where the retention mask will hash a state; (b) "EAGLE-reachable
   state kept" — the hybrid coordinator exposes the margin its lookup drops,
   `MambaManager` retains a state at `num_prompt_tokens - 1 - margin`. Includes
   worked carve examples and a prediction for jschmied's rig (first hit on
   request 2, both test prompts).
2. **jschmied, 2026-08-25T03:35:30Z** — ran `b99d152` on GB10 hardware, **both
   predicted cells land** (request 3→2 for both a 7,292-tok unaligned prompt and a
   6,400-tok aligned prompt, default install, retention untouched). Three caveats
   recorded: block size is 1,600 not 1,648 (corrects an earlier report); `usage.
   prompt_tokens_details.cached_tokens` reads 0 unless `--enable-prompt-tokens-
   details` is passed (used `prefix_cache_hits_total` instead); tree is `19c935190`
   (~1 commit off `8d6b1832`, neither touching these paths).
3. **kamb-code, 2026-08-25T07:06:31Z** — confirms the `cached_tokens` flag
   default explains jschmied's zero reading; confirms `use_internal_checkpoint`
   stays constantly `False` under MTP on any tree (untested by jschmied's run,
   but doesn't affect this PR's MTP-relevant path).

Full comment bodies (not just these three) are visible via `gh pr view 53479
--repo vllm-project/vllm --json comments` if the senior agent wants the earlier
falsification rounds referenced in the research doc's own §1.

## 3. Scope explosion: 1 file / 2 hunks -> 3 files, much more surface

Research doc (§1, verbatim): *"The fix — the only non-test file touched is
`vllm/v1/core/sched/scheduler.py`, inside `_mamba_block_aligned_split`, two
hunks."* **This is no longer true.** The current diff touches three production
files:

- `vllm/v1/core/sched/scheduler.py` — still the primary file, but the hunk 2
  region is far larger than described (see §5 below).
- `vllm/v1/core/kv_cache_coordinator.py` — **new**, not mentioned in the research
  doc at all. Adds an `eagle_reach_margin` property (base class: constant `0`;
  `HybridKVCacheCoordinator`: computed via a new `_eagle_margin` helper refactored
  out of the pre-existing inline `find_longest_cache_hit` computation), and wires
  `manager.eagle_reach_margin = self.eagle_reach_margin` onto every single-type
  manager at coordinator `__init__` time.
- `vllm/v1/core/single_type_kv_cache_manager.py` — **new**, not mentioned in the
  research doc. Adds a class-level `eagle_reach_margin: int = 0` attribute to the
  base `SingleTypeKVCacheManager`; refactors the inline `reachable_boundaries`
  computation in `cache_blocks` into an overridable `_reachable_boundaries(self,
  request)` method; `MambaManager` overrides it to append `request.
  num_prompt_tokens - 1 - self.eagle_reach_margin` as an extra retained boundary
  when the margin is positive.

Also touches test files the research doc did already flag as absent from our
tree (`test_mamba_align_chunk_split.py`, the `prefix_cache/` dir,
`test_prefix_caching.py::test_hybrid_cache_mamba_align_shared_prefix_detection`),
plus, newly, `tests/v1/kv_connector/unit/offloading_connector/{test_scheduler.py,
utils.py}` gets a *third* round of unrelated changes bundled in (see §7).

## 4. Hunk 1 (back-off removal): essentially unchanged in substance

The literal deletion — `if self.use_eagle: last_cache_position -= block_size` —
is still exactly what's removed. The diff hunk header shifted from
`@@ -408,12 +408,11 @@` (doc) to `@@ -408,12 +418,11 @@` (current) purely because
the new `__init__` block (§3, `mamba_retention_interval`/`mamba_eagle_reach_margin`,
~10 lines) was inserted earlier in the file. **No semantic drift here** — this
part of the research doc's description still holds.

## 5. Hunk 2 (stops tuple): NOT "same shape, minor tweak" — fully redesigned

This is the part of the research doc that is now actively misleading if read as
a literal spec.

**Doc's description:** the `stops` tuple's relevant entry changes from
`next_block_boundary if start % block_size != 0 else 0` to `next_block_boundary
if not use_internal_checkpoint else 0` — i.e. a single boolean-gated value,
"stop at next boundary on every chunk unless internal checkpoints exist."

**Current actual diff:** the single entry is replaced by *three* new entries,
each independently computed:

```python
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
    last_cache_position,       # pre-existing, unchanged
    tail_boundary if ... else 0,               # pre-existing, unchanged
    <shared_prefix_boundary term> if ... else 0,  # pre-existing, unchanged
)
```

The new design is retention-interval-aware (dense / segment / sparse all produce
different `boundary_stop` cadences) and explicitly materializes two more
positions (`replay_boundary`, `eagle_reach`) that a sparse retention mask would
otherwise discard. This directly folds in the exact mechanism the research doc's
§1 flagged as a *separate, upstream-review-thread-discovered* gating problem
(`prefix_cache_retention_interval` silently discarding the new states) — the PR
author closed that gap **inside this same PR**, not by leaving it as an external
prerequisite. See §6 for what this means for our sequencing-risk assessment.

Practically: `next_block_boundary`, `tail_boundary`, and the
`shared_prefix_boundary`-based term are **pre-existing** in upstream's function
(not part of this PR's diff — confirmed by reading the full current upstream file,
not just the diff) — the PR only touches the back-off line and the first tuple
entry, then adds two new entries after it. Our fork has *none* of those
pre-existing entries either (see `correspondence.md` §B) — our function predates
the entire `stops`-tuple refactor, so this is a reimplementation against our
shape regardless of which version of the PR we target.

## 6. Correction to the plan digest: the retention-interval prerequisite has ALREADY LANDED on our target branch

**This is the single most important correction in this file.** The plan digest
states (and the research doc's §5 states, consistent with it):

> "We already have an unmerged port of that exact mechanism: local branch
> `feat-retention-interval` (commit `3277ffb3c`, ports upstream #45845), NOT
> merged into `frontier-pastnative-20260816`."

**This is stale.** Verified directly against the `f1-partial-prefix-port`
worktree (branched from `frontier-pastnative-20260816` HEAD today):

- `vllm/v1/core/kv_cache_coordinator.py:130` — `self.retention_interval =
  envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL` — present.
- `vllm/v1/core/single_type_kv_cache_manager.py` — `MambaManager.
  reachable_block_mask` (a full sparse-retention implementation, classmethod,
  `retention_interval`/`num_prompt_tokens`-aware) — present, ~line 1128-1181.
- `vllm/envs.py:269` — `VLLM_PREFIX_CACHE_RETENTION_INTERVAL: int | None = None`
  — confirms the dense default the digest describes.
- `git merge-base --is-ancestor 3277ffb3c HEAD` on `frontier-pastnative-20260816`
  returns false (that exact commit hash is genuinely not an ancestor) — **but**
  `git log --oneline` shows `283936fa6 feat(kv-cache): port
  VLLM_PREFIX_CACHE_RETENTION_INTERVAL for Mamba/GDN groups (upstream #45845)`
  and, further back, a merge commit `04dd3f7ae merge feat-retention-interval:
  #45845 retention-interval port (dense default per F-1 research) + 4 stowaway
  Xid31 guards` — i.e. the same content landed under different commit hashes,
  almost certainly because the branch was rebased onto v0.1.17 after the merge
  (rebases replay commits under new hashes; the plan digest's cited hash predates
  that rebase). The merge commit message's "dense default per F-1 research"
  confirms this was landed *in direct response to* this research thread's own
  finding — i.e. the gap the doc flagged has already been closed on this branch.

**Consequence:** Plan digest risk #2 ("Inert without the retention-interval
prerequisite landing first/together... a sequencing risk, not a logic risk") is
**no longer an open risk for `frontier-pastnative-20260816`** — the prerequisite
is in place, dense-by-default, today. It doesn't make the #53479 port unnecessary
(our fork's mask already retains the *replay* boundary under sparse retention,
independently, but has no equivalent of the PR's new *eagle-reach* boundary — see
`correspondence.md` §C for the precise gap), but the "no visible movement on a
default install" failure mode the research doc worried about does not apply here:
our default is dense, not sparse, so nothing is being silently discarded today.
Step 2 of the plan digest's step-list ("Land `feat-retention-interval`... FIRST/
simultaneously") is **already done** and can be struck from the pre-port
checklist; step 3 ("Confirm `use_internal_checkpoint`/#52789 is genuinely a no-op
gap") is separately re-confirmed true in `correspondence.md` §B.

This does not touch or contradict plan digest risk #1 (upstream immaturity) or
risk #3 (EAGLE atomicity) — both still fully apply, and risk #3 if anything gets
*more* pointed now that "atomic" means 3 files instead of 1.

## 7. Bundled unrelated fixes (issue #52735) — now a larger share of the diff

The research doc already flagged that the `offloading_connector` test changes
"adjust reconciliation expectations for a full-attention `eagle_verified` group,
not the mamba path itself." The current diff confirms that and shows it grew: on
top of the two existing test-expectation edits
(`test_non_eagle_tighten_clears_eagle_verified`,
`test_full_attn_store_excludes_trailing_decode_block`,
`test_sw_store_excludes_trailing_decode_block`), there are now two **entirely
new test classes** — `TestSharedGroupMTPOffload` and
`TestMambaHybridOffloadServing` — regression-testing issue #52735 ("shared-group
MTP models" / drafter-group annotation collapse in the offloading connector).
This is explicitly a second, unrelated bugfix riding in the same PR. Our fork
does not carry the `offloading_connector` KV connector at all (confirmed absent
in the earlier research), so none of this touches our port surface — noting it
only because it further inflates the "hunk 1 + hunk 2, ~15 lines" framing that
is no longer accurate for the PR as a whole.

## 8. Net effect on the plan digest

- Step-list step 2 (land retention-interval prerequisite): **done**, strike it.
- Step-list step 3 (confirm `use_internal_checkpoint` is a no-op gap): **still
  true**, re-confirmed independently in `correspondence.md`.
- Step-list step 5 ("port hunk 1 + hunk 2 atomically... reimplementation against
  our tree, not a cherry-pick"): **still the right framing, but the target
  shape to reimplement against is now 3 files / ~6 new concepts, not 1 file /
  1 conditional.** Whoever ports this needs `correspondence.md`, not the
  original research doc, as the line-level target.
- Top risk #2 (sequencing / inert-without-retention-interval): **resolved on
  this branch**, downgrade or remove from the risk list.
- Top risks #1 (upstream immaturity) and #3 (EAGLE atomicity): **unchanged,
  both still apply**, #3 arguably strengthened by the larger surface.
