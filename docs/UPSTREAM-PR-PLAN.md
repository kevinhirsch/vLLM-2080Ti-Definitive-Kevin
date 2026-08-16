# Upstream PR Plan

**PLAN ONLY — nothing posted upstream without Kevin's approval.**

This document maps which frontier-session changes are candidates for upstream
(`weicj/vLLM-2080Ti-Definitive`) and how to separate them into clean,
independently reviewable PRs. It is a planning artifact. No branch here has been
pushed and no PR or issue has been opened anywhere. `origin` is upstream and is
push-forbidden by standing rule; the fork's own remote is `kevin`
(`kevinhirsch/vLLM-2080Ti-Definitive-Kevin`).

Branch/commit references match [Frontier Changelog 2026-08-16](frontier-changelog-20260816.md).

## Ordering rationale

Submit smallest and highest-acceptance first, so review load and merge risk grow
gradually rather than landing as one large diff. Recommended order: PR-1, PR-2,
PR-3, PR-4.

| PR | Change | Source branch | Type | Acceptance odds |
|---|---|---|---|---|
| PR-1 | `pages > width` guard + write-side bounds assert | `fix-xid31-guard` | Bugfix | High |
| PR-2 | SM75 `CUSTOM_ALL_REDUCE_MAX_SIZES` entry | `exp040-allreduce-32mib` | Config | Medium |
| PR-3 | Retention-interval port | `feat-retention-interval` | Parity | Medium |
| PR-4 | RoPE cache-size override | `exp017-rope-extend` | Feature | Low without rework |

## PR-1 — pages>width structural guard + write-side bounds assert

Smallest, most defensible, highest acceptance. A structural early-out and a
write-side assertion that catch an out-of-bounds block-table condition — a pure
safety/correctness bugfix with no behavior change on healthy paths.

- **Cherry-pick:** `db65d0a` (always-on `pages > width` early-out + reader-guard
  width fix), `27c064a` (write-side OOB assert in `BlockTable.append_row`),
  `22f4c2b` (shrink `seq_len` with `cached_len` in the early-out).
- **Cleanup first:** drop the env-gated reader-side probe `28458c6`
  (`VLLM_TURBOQUANT_CONTINUATION_BOUNDS_CHECK`) — it is fork-local
  instrumentation, not a fix, and would invite scope questions. Keep the PR to
  the always-on structural guard and the write-side assert. Frame it as
  defensive bounds-checking mirroring the existing `:998` sibling early-out;
  do not tie it to the unresolved Xid31 crash in the PR narrative.
- **Risk of rejection:** low. Small, self-contained, mirrors existing code.
  Main reviewer ask is likely to justify the assert has no hot-path cost.

## PR-2 — SM75 CUSTOM_ALL_REDUCE_MAX_SIZES entry

Add the missing `"7.5"` entry so SM75 gets an explicit custom-allreduce cap
instead of silently taking the `8MiB` default and falling back to NCCL for
larger allreduces.

- **Cherry-pick:** `6a82ec3`. Prefer expressing the change as a `"7.5"` key in
  `CUSTOM_ALL_REDUCE_MAX_SIZES` (`all_reduce_utils.py`) rather than only raising
  the constructor default in `custom_all_reduce.py`, so it is a table addition,
  not a default change that touches every architecture.
- **Cleanup first:** upstream will want their own bench validation on real SM75
  hardware before accepting a cap value. Attach our data honestly: correct at
  `32MiB` (needle 20k/40k/80k), decode unchanged, **prefill unchanged** — the
  change is correctness/routing, not a proven speedup. See the SM75 dataset
  section below. Do not claim a perf win the data does not support.
- **Risk of rejection:** medium. The mechanism is uncontroversial; the specific
  `32MiB` value needs their hardware confirmation. Offer the value as a starting
  point and defer to their bench.

## PR-3 — retention-interval port (#45845 parity)

Port of upstream vLLM #45845 (with #43447 scaffolding) extending
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL` to Mamba/GDN cache groups.

- **Cherry-pick:** `3277ffb`.
- **Cleanup first:** rebase off the Xid31 guard chain so the PR carries only the
  retention-interval change (its current parent is `22f4c2b`). Confirm naming
  and default match upstream #45845 exactly so it reads as parity, not a fork
  divergence. Add a route validation on this fork before submitting — it is
  ported but not yet independently validated here.
- **Risk of rejection:** medium, mostly on parity/validation rather than
  concept. Cleaner if #45845 has already landed upstream and this is a
  cache-group extension of it.

## PR-4 — RoPE cache-size override (rework before submitting)

Lets the RoPE cos/sin cache be sized past `config.max_position_embeddings` so
hybrid models can serve past native without an out-of-bounds attention read.

- **Cherry-pick:** `721ad64` (the one-line `get_rope` change in
  `qwen3_next.py`).
- **Cleanup first — do not submit as-is.** The current form keys off an
  undocumented env (`VLLM_ROPE_MAX_POSITION`). Rework into a proper config-driven
  knob before submitting: a CLI/config field (or derive the RoPE cache ceiling
  from the resolved `--max-model-len` / YaRN parameters) rather than an env read
  buried in the model file. Pair it with the deep-needle past-native validation
  from [Past-Native Context](past-native-context.md) and state the caveats
  (evalkit and soak pending). Note the hybrid rationale — 16/64 RoPE layers —
  since that is why the extension is well-behaved.
- **Risk of rejection:** high in current form (env-driven, single-model,
  experimental), moderate after rework into a config knob with validation. This
  is the least mature candidate; treat it as last.

## SM75 validation dataset (attachable to PR-2)

We can attach the SM75 custom-allreduce evidence gathered this session:

- Correctness: deep-needle retrieval correct at 20k / 40k / 80k with the
  `32MiB` cap.
- Decode throughput: unchanged vs the `8MiB`/NCCL-fallback baseline.
- Prefill throughput: unchanged (3-rep) — recorded as a negative so upstream
  does not over-read the change.

This dataset supports "correct and safe to route on the custom kernel," not "a
speedup." Present it that way.

## What is explicitly out of scope for upstream

- Xid31 env-gated reader probe (`28458c6`) — fork-local instrumentation.
- The stacked session branch `frontier-pastnative-20260816` as a unit — it mixes
  three unrelated concerns; upstream gets the isolated single-purpose branches.
- The `merged-v0115-regression` bank — internal checkpoint, not a contribution.
