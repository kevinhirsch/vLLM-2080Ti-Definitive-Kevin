# Frontier session 3 — 2026-08-17 (overnight + morning)

Continues `frontier-session2-20260816.md`. Focus: upstream PR #111 review hardening,
the >450K MTP boundary hunt, the scoped-drafter v2 win, and community-PR mining.

## 1. PR #111 (workspace reserve) — two review rounds, both improved the patch
- **Round 1 (double-count):** whether the workspace arena lands in the profiled torch
  peak depends on allocation timing (load-time `process_weights_after_loading` vs lazy
  first-touch) — so `available_memory` varied boot-to-boot by ~1.1 GiB (we had observed
  this as an unexplained 470K-vs-615K token pool variance, including spurious
  max_model_len rejections). Fix: `determine_available_memory` adds back the arena
  measured at profile time so the explicit reserve applies exactly once. **Verified:
  identical 741,146-token pools across consecutive boots (+~125K vs the double-counted
  boots).**
- **Round 2 (shared arena):** the arena also hosts decode/DCP/moe reservations, and the
  reserve is 0 for non-turboquant caches — a full-arena add-back could inflate the KV
  budget. Fix: add back `min(arena, reserve)` under the same gating the sizing path
  uses (`_turboquant_prefill_workspace_reserve_bytes`).

## 2. The >450K-with-MTP boundary (open, precisely characterized)
The Path A fix moved the crash boundary from (352K, 380K] to **(444,346, ~451,500]**
actual tokens — but a crash class above ~450K persists with spec decoding enabled.
Nine hypotheses eliminated by measurement, including: free-margin exhaustion (crashes
with 3.2 GB free), draft rope-cache sizing (draft shares `Qwen3NextAttention`, inherits
the extension), cudagraph-specificity (reproduces under `enforce_eager` +
`CUDA_LAUNCH_BLOCKING`), and a 3584×128 chunk-table theory (455K < 458,752 crashes).
Surviving profile: **request-size-gated allocation overflowing a fixed structure,
spec-config-dependent, in the prefill path, faulting at allocator-layout-dependent
depths** (Xid 31 FAULT_PDE). Production is fenced below it (gateway per-request cap
440,000). Next: allocation-logging instrumentation, not more probes.

## 3. Scoped re-emission drafter v2 — 811.5 tok/s on verbatim spans
The v1 A/B refuted itself (−75% on its target regime); per-position acceptance data
fingerprinted the bug: the drafter's merge runs **before** bookkeeping commits the
step's sampled tokens, so draft position 0 proposed the token just emitted (instant
reject, cascade zeros). Fixed by splicing sampled tokens into the match needle
(`_effective_tail`); reproduced-then-fixed in a pure-python unit test (2/5 → 5/5).
- **Re-A/B (PIECEWISE both sides, K=16 pipeline, MTP chain capped K=2 vs S4 on):**
  quote **365 → 811.5 tok/s** (mean 7.44 accepted/step), mixed 101.5 (no harm),
  generation 66.1 (unchanged). 27B on 2×2080 Ti.
- **Bonus:** sizing the spec pipeline to K=16 lets the MTP head chain deep even with
  the drafter off — mixed 72 → 102 tok/s free, no cost to pure generation.
- Ship blockers (open): K=16 memory profiling underestimates by ~5 GiB; K=16 inflates
  the mamba align block 3488 → 3856 (mnbt floor moves). Branch `feat-s4-scoped-drafter`.

## 4. Community-PR mining (open PRs on this repo)
- **#92** (MTP mamba align cache retention) and **#108** (custom-AR profiling IPC
  leak) cherry-pick cleanly — staged on `steal-pr108-pr92` for window testing.
- **#100**'s thinking-budget reset targets runaway-think loops (in-engine forcing
  beats gateway trimming) — queued test-first. **#109**'s 3D segmented softmax may
  lift long-context decode independently — queued test-first.
- **#106** int8 KV: ~2.3× our 3.5-bit KV footprint (pool would halve) — not a
  replacement, but a near-lossless **quality oracle** for measuring what 3-bit K
  costs at depth. **#103** is unmergeable as posted (whole-file reversions +
  truncated payloads); its intent ships in #92/#100. **#80** DFlash replaces MTP and
  has no Qwen3.8 draft checkpoint — watch only.

## 5. Named-tool-choice review batch (branch `claude/qwen-3.8-27b-tuning-isbc7k`)
Fixed and pushed: truncated named tool calls no longer report
`finish_reason="tool_calls"` (three paths; `length` preserved — clients must not
execute truncated argument JSON), fallback deduplicated onto
`extract_named_tool_call_streaming`, MTP-requal bench hard-fails on contaminated
spec counters (drafts delta bounded by the probe's own tokens) and on missing
/metrics evidence, garble signatures single-sourced. +2 permanent truncation
regression tests.
