# F-1 phase 1: partial-prefix-hit research — vLLM PR #53479 vs. our align-mode fork

Research only. No code changed, no engine run. Scope: read vllm-project/vllm
PR #53479 ("Mamba align: materialize a state at every boundary and drop the
speculative back-off"), map it onto our tree, and assess whether/how it should
be ported.

Repo: `weicj/vLLM-2080Ti-Definitive` (this checkout), branch
`frontier-pastnative-20260816`. Serving config: Qwen3.8-27B, hybrid
GDN/full-attention, `--mamba-cache-mode align`, prefix caching on, MTP spec
decode (`deploy/bin/serve-qwen38-mtp-requal-fg.sh`,
`deploy/bin/serve-qwen-8001.sh`).

## 1. PR #53479 — exact mechanism, review status, CI

**Author:** `kamb-code` (Kam Basra), AI-assisted (Claude Code), opened
2026-08-23T19:41Z against `vllm-project/vllm` `main`. **State at research
time (2026-08-24):** OPEN, `MERGEABLE`, `reviewDecision: REVIEW_REQUIRED` —
no maintainer/codeowner review yet, only community discussion. CI: PR-triggered
`pre-run-check` is **FAILING**; full upstream CI has not been run (fork PRs
need a maintainer `/ci run`, which hasn't happened). Labels: `bug`,
`kv-connector`, `scheduler`, `kv-cache-manager`.

**The bug (two coupled defects), per the PR body:**

1. **Sparse states.** Align-mode Mamba states materialize only at *chunk
   ends* (a scheduler step boundary), not at every block boundary a chunk
   crosses. A quiet (single-step) prefill of a long prompt produces one deep
   mid-prompt chunk end; a sibling sharing anything less than that single
   position gets zero reuse, even on an idle server (measured: shared 2,000
   of 4,278 tokens → 0 hit). An exactly block-aligned prompt gets zero reuse
   even on an *identical* repeat, because its only state sits at
   `num_tokens`, above the `num_tokens - 1` lookup cap.
2. **The EAGLE/MTP one-block back-off.** `last_cache_position -= block_size`
   under `use_eagle` was sized for 16-token attention blocks; align blocks
   are 544–2,128 tokens on the PR's reference hardware (per our own
   `docs/mtp-retention-invariant.md`, "~2-4K at our shapes"), so the whole
   speculative family (`eagle/eagle3/mtp/dflash/dspark`) forfeits one entire
   align block of reusable prefix on every resumed request.

**The fix — the only non-test file touched is `vllm/v1/core/sched/scheduler.py`,
inside `_mamba_block_aligned_split`, two hunks (`gh pr diff 53479`, hunk
headers `@@ -408,12 +408,11 @@` and `@@ -442,9 +441,13 @@`):**

- Hunk 1: unconditionally deletes `if self.use_eagle: last_cache_position -=
  block_size`. No more back-off, for any `use_eagle` method (EAGLE, EAGLE3,
  MTP, dflash).
- Hunk 2: in the `stops` tuple, changes
  `next_block_boundary if start % block_size != 0 else 0` (stop only when the
  chunk *starts* mid-block) to `next_block_boundary if not
  use_internal_checkpoint else 0` (stop at the next boundary on *every*
  chunk, unless the model has internal prefill checkpoints — #52789 —
  which make mid-block states independently recoverable). This is what
  makes hunk 1 safe: with a state at every crossed boundary, a lookup capped
  one block short by the (now-removed) back-off falls back to the *previous*
  boundary's state instead of missing outright.

Both hunks are context-coupled: hunk 1's safety argument depends on hunk 2
being in effect. Test diff: 3 new regression files under
`tests/v1/core/prefix_cache/` + edits to `tests/v1/core/test_mamba_align_chunk_split.py`,
`tests/v1/core/test_prefix_caching.py`, and
`tests/v1/kv_connector/unit/offloading_connector/{test_scheduler.py,utils.py}`
(the last two adjust reconciliation expectations for a full-attention
`eagle_verified` group, not the mamba path itself). Cited prior art:
`#48815` (opt-in, `mtp`-only, keeps the back-off for aligned prompts — this PR
removes it unconditionally), `#52244` (a narrower replay-landing stop this PR
subsumes), `#52371` (lookup-side pins, unaffected, complementary), `#50897`
(lookahead hashing, disjoint files, complementary).

**Review status — the substantive finding, not a rubber stamp.** The comment
thread (all today, `jschmied` running measurements on GB10 production
hardware, `kamb-code` predicting/reconciling) went through several rounds of
falsification before landing on the real gate: **`prefix_cache_retention_interval`
defaults to `0` upstream** (`config/cache.py`,
`_get_prefix_cache_retention_interval`), and per its own docstring `0`
"retains only semantic checkpoints, including the latest replay boundary and
shared-prefix junctions." In `MambaManager.reachable_block_mask`,
`retention_interval == 0` skips the dense-segment branch entirely, so this
PR's newly-materialized per-boundary states are filed and then **immediately
discarded** by the retention mask before they're ever hashed/found. Measured,
production code, only the interval changed: first-hit request count on a
7,292-token prompt went from 3 (default) to 2 (`retention_interval =
block_size`); this PR is invisible on a default install without also setting
that env var. This is flagged in the thread as something worth stating
explicitly in the PR description — as of research time it is not yet
reflected there.

## 2. Port-surface table

The PR's only production hunk is the single function
`_mamba_block_aligned_split` in `vllm/v1/core/sched/scheduler.py`. Reading the
allocate/state-materialization path it depends on
(`vllm/v1/core/single_type_kv_cache_manager.py::MambaManager`,
`vllm/v1/worker/mamba_utils.py::postprocess_mamba`) confirms *why* it works:
`MambaManager.allocate_new_blocks` allocates and records at most one new state
block per scheduler step (`self.last_state_block_idx[request_id] = ...`), and
`postprocess_mamba` writes the recurrent state to exactly one
`dest_block_idx = aligned_new_computed_tokens // block_size - 1` per step —
one state materializes per *chunk*, confirming the PR's diagnosis. Neither of
those files is touched by the PR; the fix is entirely about *where the
scheduler decides to stop a chunk*, not the storage mechanism.

| Upstream (their file:hunk) | Our tree (file:region) | Status |
|---|---|---|
| `scheduler.py` hunk 1 — unconditional back-off removal (upstream `_mamba_block_aligned_split`, `use_eagle` check deleted) | `vllm/v1/core/sched/scheduler.py:300-301` — `if self.use_eagle and not retain_final_mtp_block: last_cache_position = max(last_cache_position - block_size, 0)` | **Conflict.** We already have a narrower, MTP-only conditional relaxation of this exact back-off (`retain_mamba_align_mtp_cache_block` / `VLLM_MAMBA_ALIGN_RETAIN_MTP_CACHE_BLOCK`, added for reasons documented in `docs/mtp-retention-invariant.md`). Upstream's version is unconditional (also covers true EAGLE/EAGLE3/dflash) and its safety argument requires hunk 2. Cannot apply as a literal patch — needs a design decision on whether to keep our narrower MTP-specific gate or replace it. |
| `scheduler.py` hunk 2 — `stops` tuple, `next_block_boundary if not use_internal_checkpoint else 0` | `vllm/v1/core/sched/scheduler.py:303-311` — `chunk_end = min(round_down(chunk_end, block_size), last_cache_position)` / `elif chunk_end < prefill_end: chunk_end = round_down(chunk_end, block_size)` | **Needs adaptation (large).** Our current function predates the `stops`-tuple refactor entirely: no `next_block_boundary`, no `use_internal_checkpoint`, no per-boundary stop logic at all — it rounds the *whole* chunk down to one block boundary in one step, which is precisely the sparse-state bug. `use_internal_checkpoint` is gated on upstream #52789, which is **absent from this branch** (and from `origin/vllm-2080ti-definitive-0.2.x`) — a real prerequisite gap, not a naming difference. `origin/vllm-2080ti-definitive-0.2.x` already carries the `stops`-tuple shape (see §5) and is the closer landing zone. |
| `tests/v1/core/test_mamba_align_chunk_split.py` (existing, extended) | — | **Not applicable.** File does not exist in this repo's test tree at all. |
| `tests/v1/core/prefix_cache/test_eagle_mamba_drop_cost.py`, `test_eagle_mamba_shared_system_prompt.py` (new) | — | **Not applicable.** No `tests/v1/core/prefix_cache/` directory in this fork. |
| `tests/v1/core/prefix_cache/test_mamba_align_sparse_keys.py` (new) | — | **Out of scope for this fix, but a separate latent bug worth flagging.** Covers `emit_cached_block_events` publishing wrong/invented mamba keys under `kv_cache_report_mode="full"`, including an `IndexError` crash on unaligned prompts. Distinct defect, same sparse-key root cause. Only matters to us if/when we turn on KV-cache-event reporting. |
| `tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py` (modified) | — | Not applicable — file absent. |
| `tests/v1/core/test_prefix_caching.py::test_hybrid_cache_mamba_align_shared_prefix_detection` (modified) | `tests/v1/core/test_prefix_caching.py` (file present) | **Not applicable.** File exists but this specific test function does not — our trimmed suite predates it. |
| `tests/v1/kv_connector/unit/offloading_connector/{test_scheduler.py,utils.py}` (modified) | — | **Not applicable.** We don't carry the `offloading_connector` KV connector or its tests. |
| — | `tests/2080ti/test_mamba_align_mtp_prefix_cache.py` | Our own fork-local regression for the MTP retention gate (`_split_with_mtp_cache_retention`). Model for how we'd need to author fresh tests for any ported fix, since upstream's harness doesn't transplant. |

Net: the port surface is nominally "one function, two hunks," but neither of
our two branches has the scaffolding those hunks assume, and the corresponding
upstream test harness doesn't exist in our tree at all. This is a
reimplementation against our shape, not a cherry-pick.

## 3. Workload analysis — Claude-Code-style agentic traffic (10K → 200K growing contexts)

Per the **current** code (`scheduler.py:267-313`), only one Mamba state block
is recorded per scheduler step (confirmed via `MambaManager.allocate_new_blocks`
+ `postprocess_mamba`, §2). With our batching budgets far larger than one
align block (~2-4K tokens per `docs/mtp-retention-invariant.md`), a single
prefill step routinely spans several block boundaries and checkpoints only
the last one it lands on — the store-side half of the PR's "sparse states"
diagnosis applies directly to us.

- **Diverged siblings (new session/subagent sharing a system prompt/repo
  preamble, then branching):** today this is **effectively exact-repeat-only**.
  A sibling that shares less than the one deep chunk-end position the
  original prefill happened to stop at gets **zero** reuse — matching the
  PR's own repro (shared 2,000/4,278 → 0) and its "shared system prompt"
  test (24/40 shared → 0 reconciled hit). Multiple Claude-Code
  subagents/parallel turns fanned out from a common context are the shape
  most exposed to this: block-granular divergence is common, exact-position
  divergence is not.
- **Same-session multi-turn continuation (turn N's prompt = turn N-1's cached
  context + new tail):** this is the more forgiving case, since there's no
  divergence — the request is literally the same lineage extending forward,
  resuming each turn from wherever the previous turn's chunk ended. It still
  pays the EAGLE/MTP one-block back-off *every turn* unless
  `VLLM_MAMBA_ALIGN_RETAIN_MTP_CACHE_BLOCK` is set (it is **not** set in
  either live serve script, confirmed by grep) — i.e., today's production
  config re-prefills one align block (~2-4K tokens) of already-seen context
  on every turn's resume, exactly the tax `docs/mtp-retention-invariant.md`
  was written to justify removing for MTP specifically.
- **Qualitative hit-rate by turn N:** at low N (10K-ish, few blocks total),
  the "only one deep checkpoint" limitation is a large fraction of the
  prompt — a diverged sibling at turn 1 realistically gets ~0% reuse today,
  and #53479 would raise that toward "up to the deepest shared block
  boundary," a large relative jump when there are only a handful of
  boundaries. At high N (200K, dozens of blocks), the *relative* miss from
  sparse states shrinks as a fraction of context, but the *absolute* tax
  stays pinned near one block's worth of tokens per divergence/resume event
  — so the win from #53479 is best read as "avoids ~1 block of re-prefill
  per turn/branch point," roughly constant in absolute tokens rather than
  proportional to session depth.
- Caveat that constrains all of the above (§1, §4): none of this shows up in
  practice unless the retention-interval mechanism keeps more than "semantic
  checkpoints." See §4/§5 — this is precisely the axis our fork already has
  an (unmerged) opinion on.

## 4. Interaction with our `VLLM_MAMBA_ALIGN_RETAIN_MTP_CACHE_BLOCK` gate

Complementary in *intent*, not simply additive in *implementation*. Both
attack "the EAGLE/MTP back-off forfeits a full align block," but:

- **Ours** (`scheduler.py:296-301`, `retain_mamba_align_mtp_cache_block`) is
  narrow and self-contained: gated to `method == "mtp"` only, and its safety
  proof (`docs/mtp-retention-invariant.md`) does **not** depend on dense
  per-boundary state materialization — it rests on MTP's proposer being
  stateless and needing only the uncached tail's own hidden state, unlike
  EAGLE's shifted (`hit-1`) resume point. It is independently safe today,
  with or without #53479's hunk 2.
- **Theirs** removes the back-off unconditionally, including for true EAGLE,
  where our own doc explicitly says the back-off is *load-bearing*
  ("EAGLE needs this because its cache hits are shifted by one... the only
  consistent boundary is one block earlier"). Upstream's justification for
  extending the removal to true EAGLE is exactly hunk 2: with a state at
  every crossed boundary, a lookup capped short by the missing back-off
  falls back one boundary instead of missing. **Porting hunk 1 without hunk
  2 would reintroduce the miss the back-off was invented to prevent for
  EAGLE-proper traffic** — the two must land atomically.
- **Combined semantics if both are ported correctly:** our MTP-only gate
  becomes a strict subset of upstream's — once dense per-boundary states
  exist and the back-off is unconditionally gone, `retain_mamba_align_mtp_cache_block`
  is redundant for the MTP case (upstream's version already retains it, for
  every `use_eagle` method, not just MTP) and could be retired in favor of
  the upstream shape. Until then, they should **not** be merged naively:
  keep ours (MTP-scoped, independently proven, already tested in
  `tests/2080ti/test_mamba_align_mtp_prefix_cache.py`) as-is, and treat
  porting #53479 as a full replacement of that code region, not a layer on
  top of it.

## 5. Does `origin/vllm-2080ti-definitive-0.2.x` already have this?

No equivalent of #53479, but a materially different — and in one respect
more advanced — starting point. `git merge-base` between this branch and
`origin/vllm-2080ti-definitive-0.2.x` shows massive divergence (~19,484
commits ahead on 0.2.x, ~182 commits ahead here — different major upstream
sync lines), so this needed a direct read of 0.2.x's `_mamba_block_aligned_split`
rather than a diff:

- **No `VLLM_MAMBA_ALIGN_RETAIN_MTP_CACHE_BLOCK`** — grep for `RETAIN_MTP` in
  0.2.x's `scheduler.py` and `envs.py` returns nothing. The MTP retention fix
  is `frontier-pastnative-20260816`-only.
- **No #53479-equivalent** — 0.2.x's back-off is still the plain, unconditional
  `if self.use_eagle: last_cache_position = max(last_cache_position - block_size, 0)`,
  and its `stops` tuple still reads `next_block_boundary if start % block_size
  != 0 else 0` — i.e., 0.2.x is at the same *pre-#53479* upstream shape, not
  ahead of it. It also has no `use_internal_checkpoint` (#52789 is absent
  there too).
- **But it already has the `stops`-tuple scaffolding** (`next_block_boundary`,
  plus two stops this branch lacks entirely: `tail_boundary` for
  `mamba_partial_cache_hit` partial-tail registration, and a
  `shared_prefix_boundary` "Marconi" junction stop) — making 0.2.x the
  structurally closer landing zone for porting #53479's hunk 2 than this
  branch, if/when 0.2.x becomes the serving line.
- **It already carries `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`** — confirmed
  present in 0.2.x's `single_type_kv_cache_manager.py` and `envs.py`
  (`reachable_block_mask`, same shape as the `feat-retention-interval` branch
  off this repo, §below). This is the exact mechanism §1's review thread
  found gating #53479's real-world benefit upstream.

Separately (not `origin/vllm-2080ti-definitive-0.2.x`, but directly relevant
and worth surfacing): **this repo already has an unmerged port of that same
retention-interval mechanism**, on local branch `feat-retention-interval`
(commit `3277ffb3c`, "port `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` for
Mamba/GDN groups, upstream #45845"), not merged into
`frontier-pastnative-20260816`. Notably, **our port's default is the opposite
of upstream's**: ours defaults `VLLM_PREFIX_CACHE_RETENTION_INTERVAL = None`
→ dense (cache every block, current unchanged behavior), where upstream
effectively defaults to `0` → sparse ("semantic checkpoints only") — the
exact setting §1's reviewers showed zeroes out #53479's gains on a default
install. `docs/UPSTREAM-PR-PLAN.md` already lists this as "PR-3... medium
acceptance odds... ported but not yet independently validated here" in
Kevin's own upstreaming queue, written before today's discovery that it's
also the load-bearing prerequisite for #53479 to matter in production.

## 6. Verdict

**Port feasibility: medium.** The upstream diff is small (one function, two
coupled hunks, ~15 lines), but it doesn't apply cleanly to either of our
branches: this branch (`frontier-pastnative-20260816`) lacks the entire
`stops`-tuple refactor the hunks are written against (closer to a rewrite
than a patch here); `origin/vllm-2080ti-definitive-0.2.x` has the tuple
scaffolding but is missing the `use_internal_checkpoint` (#52789)
prerequisite. Either path also requires reconciling with our own
`VLLM_MAMBA_ALIGN_RETAIN_MTP_CACHE_BLOCK` gate (§4) and authoring a fresh
test harness, since none of the upstream PR's test files exist in our
(heavily trimmed) test tree.

**Top 3 risks:**

1. **Upstream immaturity.** Opened <24h before this research, zero maintainer
   review (`REVIEW_REQUIRED`), pre-commit CI currently failing, and the
   comment thread is still actively finding and reconciling new behavior
   (retention-interval gating was only nailed down hours before this
   research ran). Porting now means importing logic that hasn't survived
   upstream review; treat as a spike/reference implementation to track, not
   a stable target to cherry-pick today.
2. **The fix is inert without a companion change we already half-have.**
   §1/§5: #53479 alone buys nothing on a default install because
   `prefix_cache_retention_interval` defaults to sparse upstream. Landing
   #53479 without first (or simultaneously) landing our own
   `feat-retention-interval` port (with its dense default) onto whatever
   branch serves it would reproduce the exact "no visible movement" result
   upstream's own reviewer measured on GB10 hardware — a sequencing risk,
   not a logic risk.
3. **Atomicity with our EAGLE-proper safety invariant.** §4: our narrower
   `retain_mamba_align_mtp_cache_block` gate is safe on its own reasoning;
   upstream's unconditional back-off removal is only safe together with
   dense per-boundary materialization. A partial/staged port (e.g., taking
   hunk 1 first because it's the smaller diff) would silently reintroduce a
   real cache-miss bug for true-EAGLE traffic that the original back-off
   existed to prevent.

**Bench protocol to prove the win — multi-turn replay TTFT per turn:**

- **Setup:** live serving config (`mamba-cache-mode=align`, prefix caching
  on, MTP k=3, Qwen3.8-27B), one vLLM instance, no other traffic.
- **Workload, two arms of the traffic model:**
  - *(a) Diverged siblings* — N sessions sharing a common repo/system
    preamble (~2K-8K tokens), each branching into a distinct task from turn 1.
    Targets what #53479 is supposed to fix.
  - *(b) Single-session growth* — one session, ~20 turns, context growing
    10K → 200K, each turn resubmitting previous context + a new tail.
    Targets the back-off/resume cost `VLLM_MAMBA_ALIGN_RETAIN_MTP_CACHE_BLOCK`
    already addresses, to isolate it from #53479's contribution.
- **Metric:** TTFT per turn, plus vLLM's own prefix-cache-hit token count per
  request (`PrefixCacheStats` / the `prefix_cache_hit_rate` metric) broken
  out by turn index N, so a TTFT delta can be attributed to cache mechanics
  rather than noise.
- **Arms to compare:** (1) current code, baseline; (2) +
  `VLLM_MAMBA_ALIGN_RETAIN_MTP_CACHE_BLOCK=1` alone; (3) + ported #53479
  (both hunks, atomically) alone; (4) (3) + our `feat-retention-interval`
  port set dense (`VLLM_PREFIX_CACHE_RETENTION_INTERVAL` ≤ block size) — the
  arm that isolates whether the retention-interval gate found in §1 is the
  actual bottleneck on our stack, not just upstream's.
- Per [[Benchmark Rigor]]: warm up, 3+ reps per arm per turn, report median
  and spread, not a single-shot number.
