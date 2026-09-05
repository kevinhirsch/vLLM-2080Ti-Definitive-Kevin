# vLLM-2080Ti-Definitive — Kevin's fork

A consolidated, production-hardened lineage of
[weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive)
serving **Qwen3.8-27B (GPTQ-Int4 self-quant) on 2× RTX 2080 Ti 22 GB (SM75, TP=2, NVLink)**
as a fully local agentic-coding backend (multimodal, tool-calling, 524K context).

Every number below is **measured on that hardware**, with the shape and date it was
measured at. Negative results are kept on purpose — they are load-bearing.

**Sync state (2026-08-27):** this fork contains **all** upstream commits through the
**v0.1.17 base** (rebase tip `315866591`; prod tip `f103d4818` carries tools-only
commits on top — see `CHANGELOG.md`). Upstream has accepted
[#111](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/111) (turboquant workspace
reserve) and, confirmed via the v0.1.17 changelog entry,
[#125](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/125) (named-tool streaming
correctness) from this lineage;
[#126](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/126) (MTP Mamba-align
retention) was closed at the maintainer's request to be split; the open PRs from this lineage are
[#131](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/131) (spec-verify reserve + DP mask) and
[#137](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/137) (align-allocate tolerate) — max 2 open PRs upstream at a time, by policy.

## Live production config (deployed, gated 2026-08-27)

| Axis | Value | Notes |
|---|---|---|
| Context | **524,288 max-model-len** (YaRN 2×) | 500K gateway fence; 490K uncached prefill needle-verified (2026-08-17) |
| KV pool | **637,560 tokens** | k3v4_nc at MTP K=3; down from 690,366 under K=2 (EXP-048, 2026-08-24: **-7.6%** pool cost for the K=3 decode win below — local context-lane guarantee held throughout) |
| KV precision | **3-bit K / 4-bit V** (`turboquant_k3v4_nc`) | validated 2026-08-20: needle-at-depth A/B vs 8-bit K = **zero retrieval cost, 1.9× capacity** (131K d10/50/90 + 240K d50, 4/4 = 4/4) |
| Mamba cache | `mamba_cache_mode align` | base hybrid-Mamba caching strategy (distinct from the opt-in `VLLM_MAMBA_ALIGN_RETAIN_MTP_CACHE_BLOCK` retention knob below, which layers on top of it) |
| Decode (single) | **~73–80 tok/s** (MTP K=3, acceptance ~3.16–3.38) | quiet-hour full-serve-path re-bench 2026-08-24 EOD: 13.05ms mean / 12.53ms median TPOT = 76.6 / 79.4 tok/s, accept 3.16, evalkit 45/45; same-day EXP-048 A/B window: 72.9 mean / 79.8 median tok/s, accept 3.38. Supersedes the K=2 predecessor (59–68 tok/s, 2026-08-16/17/23) |
| Decode (aggregate) | **364.7 tok/s @ 16 lanes** (agent-shaped) | prod shape 2026-08-17 under K=2/690,366-pool; **not yet re-verified** under the current K=3/637,560-pool config — 24 lanes regresses (knee mapped) |
| Vision | **ON** (2026-08-23) | images ≤4/prompt, auto-downscaled to ~1 MP (~976 img tokens); reads fine screenshot text end-to-end through the gateway |
| Spec decode | MTP K=3 (`VLLM_MTP_DRAFT_CAP=3`, dense-FP16 draft load) | K=2→K=3 flip shipped 2026-08-24 (below); K=8 chained REFUTED at prod shape (agg −43%, 2026-08-18) — K=4 plateau probe queued (upstream's Qwen3.6 data says the ladder plateaus at K=3; not yet verified on 3.8) |

Vision ceiling: the transformers-5.13 qwen3_5 processor grid-mismatches above ~1024
image tokens (≈1 MP) — `max_pixels 1000000` sits just under it. Native full-res needs a
processor patch (open item).

**Methodology, so the numbers above are honest:** single-stream decode records are
**median-of-3** (a single evalkit/throughput run is ±1 stochastic — the same champion
config measured {44,45} on different nights); speed claims are gated against a
**same-hour paired baseline** (never a historical bar — "one sample a millisecond below
a bar is not enough to disqualify"), quality against **median-of-3 evalkit ≥44/45**.
Different bench *shapes* are never mixed: e.g. the 2026-08-26 v0.1.17-rebase regression
gate (bench-style 4K/128, single-stream, paired same-hour) measured rebase median
**28.57 tok/s** vs champion **29.30** (97.5%, evalkit median 44 {43,44,45}) — a
*parity* check for the rebase itself, not comparable to the quiet-box full-serve-path
figures above (different harness, different concurrency/warm-cache conditions).

## Route conformance (M-8)

Smoke-tested at each kv/align/GDN patch wave, independent of the tuned prod profile
above (8K max-model-len, mnbt 3584, seqs 2, noMTP, PIECEWISE graphs, deterministic
probe — full transcript in `frontier-queue/results/m8_conformance.txt`):

| Route | Config | Result | Pool @8K |
|---|---|---|---|
| FP16 KV | (default) | **PASS** | 174,441 |
| INT8 KV | `int8_per_token_head` | **PASS** | 246,442 |
| FP8 weights | `--quantization fp8` | SKIP | no local FP8 checkpoint to test against |
| CPU offload | `--cpu-offload-gb` | SKIP | fork documents no offload serve route; no claim made either way |

Prod (`turboquant_k3v4_nc` + MTP K=3) is continuously validated by the ship gates
above, not this tier — this table exists to keep the *other* routes honest about
what has and hasn't actually been run. Full detail: `docs/m8-conformance-20260826.md`.

## Known issues

- **Nightly continuation-prefill crash under 40K+ shared-prefix streams.** Currently
  **mitigated by gateway serialisation** (the keepalive-shim queues rather than
  parallelizes requests that would collide on this path) rather than root-caused —
  treat as an open item, not a closed one.
- **F-5 FULL-graph residual.** The validated lock-before-capture fix for the
  FULL-graph capture race lives on branch `f5-lock-before-capture` and is **not
  merged** — a residual doubled-token mechanism remains open under investigation.
  `VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=0` stays set in prod as the deadlock guard
  until this lands.

## What this fork lineage adds over upstream v0.1.17

**Correctness / stability (root-caused, regression-tested):**
- Xid31 crash class **fixed at source** (KV-pool overcommit → workspace-before-KV-sizing;
  the accepted #111) — 380–490K prefills clean where they deterministically crashed;
  plus an env-gated write-side block-table bounds guard and a 451-line Xid31
  instrumentation module (`VLLM_TQ_XID31_TRACE`, default off)
- **Spec-decode verify workspace reserve** — profiling only exercises K=1 width; large-K
  boots OOM'd post-profiling. Reserved at KV sizing (0.355 GiB at prod shape, K=2/seqs16)
- **MTP Mamba-align retention** (`VLLM_MAMBA_ALIGN_RETAIN_MTP_CACHE_BLOCK`, default off;
  PR #126) — stops re-prefilling a ~block_size (3536-token) prefix block per multi-turn
  continuation
- **Named-tool streaming**: fallback dedup + truncated calls keep `finish_reason:"length"`
  with partials surfaced per OpenAI semantics (PR #125)
- numpy≥2.0 `np.uint64(negative)` fix in the S4 rolling hash (+regression tests)
- Config-time hard-fails for async-incompatible drafter combos (suffix overlay, S4-v2)

**Serving primitives (env-gated, default off, hardware-proven):**
- **Upstream #53479 port** (F-1, prefix-cache retention-interval sparsification for
  Mamba/GDN): merged into the v0.1.17 base but shipped **dormant**
  (`VLLM_PREFIX_CACHE_RETENTION_INTERVAL` unset preserves legacy chunking
  byte-for-byte) — redundant at our current mnbt≈block geometry, kept for parity and
  as the foundation for a future `eagle_reach_margin` change. No serving-behavior
  change vs the pre-port engine.
- **GDN snapshot / pin / restore / fork** (`/tq/*` HTTP API, stages 0–6): byte-exact
  hybrid-state snapshot + prefill-once/fork-N — the substrate for best-of-N serving.
  Per-child SSE streaming on `/tq/fork2`.
- **S4 scoped re-emission drafter**: FSM-gated verbatim-copy drafting.
  **811.5 tok/s on copy-shaped spans measured 2026-08-17 at its A/B shape**
  (65536 ctx / 4 seqs / util 0.75 / K=16+cap2, mean accept 7.44). Honest status:
  that config **crashes at current HEAD** (`num_required_blocks` assertion — under RCA);
  v3 GPU-merge async-on is **decisively negative** (copy −42–58%, scoped positions get
  zero acceptance under async — `docs/exp039-v3-gpu-merge.md §10–11`); current-HEAD
  clean datapoint is v3-async-off at ~150 tok/s quote vs ~78 baseline. Not prod-eligible
  until re-proven.

**Ops (`deploy/`, `tools/`):**
- Capacity gateway (`keepalive-shim`, :8000): local-first routing with DeepSeek
  failover (thinking force-disabled on remote — tool-using histories no longer 400),
  budget/admission control, tiny fast-lane, think-budget + repetition guards,
  background no-think, flight recorder, live dashboard + 34 contract tests.
  Telemetry endpoints: `/gateway/dashboard` (real-time lanes-in-use,
  local-vs-overflow %, per-GPU util/VRAM, recent-request feed), `/gateway/stats`
  (same data as JSON), and — added 2026-09-04 — the "Research & Lanes" admin
  section (`/gateway/lanes`, `/gateway/research/{id}`; lane roots configured in
  `~/.local/share/vllm-qwen27b/lanes.dirs`); this addition is admin-surface only,
  the front-door proxy behavior and engine are unchanged.
- `tools/update-from-remote.sh` — status / `--apply` (ff-only) / `--restart`
  (safe engine window behind the failover)
- systemd units + watchdog, banked known-good serve configs, model switcher

**Verdict ledger (measured, kept honest):**
- 3-bit-K: **keep** (zero retrieval cost, 1.9× pool) · retention-interval port
  (#45845-style, EXP-022 2026-08-16): **no benefit** at our 3536-token Mamba blocks
  (OFF=ON pool) — the same finding is why the now-landed #53479 port above ships
  dormant rather than enabled · K=8 MTP chain: **refuted** at prod shape ·
  EXL2/EXL3: **rejected** (no qwen3_5 arch / no SM75) · suffix overlay:
  **reverted** (−35% single on novel text)

## Quantization

GPTQ-Int4 self-quant via gptqmodel 7.3.2 (group 128, sym, desc_act=false); vision tower
kept FP16 (loads when vision flags are on). AWQ kernels are not SM75-usable; NVFP4 needs
Blackwell. Open experiment: per-module mixed-precision GPTQ (8-bit on the 16 full-attn
layers) — the "variable bitrate" play, ship-gated on an eval win.

## Docs

`docs/` carries the experiment ledgers (EXP-038 fork primitive, EXP-039 drafter design +
v3 verdicts, MTP requalification kit, Qwen3.8 harness addendum, PROMPT-HEAD-ABI in the
localflow consolidation). `deploy/docs/` has the ops runbooks.

## Branch map

- **`vllm-2080ti-definitive-0.1.x-kevin`** — default; fully consolidated (all 5 fork PRs
  + upstream v0.1.17 base). `frontier-pastnative-20260816` tracks it in lockstep (local
  prod checkout, currently at tip `f103d4818`).
- Worktree branches (`feat-*`, `exp*`, `steal-*`) — experiment lineages; all merged or
  superseded by the default, kept for the worktrees that pin them.
- Rollback: tag `pre-consolidation-rollback`.

## Hardware notes

Modded 22 GB 2080 Tis, NVLink NV2 active (~51 GB/s) — GPU1's x4 PCIe is not a TP
bottleneck. No ECC: soak new configs before trusting them (12-minute passes lie; use
30-minute mixed soaks).

## Upstream & license

This repository is a fork of [vLLM](https://github.com/vllm-project/vllm),
licensed under the Apache License 2.0 (see `LICENSE`), by way of the SM75
groundwork in [weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive)
(linked at the top of this document).
