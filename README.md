# vLLM-2080Ti-Definitive — Kevin's fork

A consolidated, production-hardened lineage of
[weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive)
serving **Qwen3.8-27B (GPTQ-Int4 self-quant) on 2× RTX 2080 Ti 22 GB (SM75, TP=2, NVLink)**
as a fully local agentic-coding backend (multimodal, tool-calling, 524K context).

Every number below is **measured on that hardware**, with the shape and date it was
measured at. Negative results are kept on purpose — they are load-bearing.

**Sync state (2026-08-23):** this fork contains **all** upstream commits through the
v0.1.16 prep head (`b24d829`). Upstream has accepted
[#111](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/111) (turboquant workspace
reserve) from this lineage;
[#125](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/125) (named-tool streaming
correctness) and
[#126](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/126) (MTP Mamba-align
retention) are open.

## Live production config (deployed, gated 2026-08-23)

| Axis | Value | Notes |
|---|---|---|
| Context | **524,288 max-model-len** (YaRN 2×) | 500K gateway fence; 490K uncached prefill needle-verified (2026-08-17) |
| KV pool | **690,366 tokens** | k3v4_nc; delta vs prior 801K fully attributed: vision tower +2.0 GiB, spec-verify reserve 0.355 GiB, TQ reserve single-counted |
| KV precision | **3-bit K / 4-bit V** (`turboquant_k3v4_nc`) | validated 2026-08-20: needle-at-depth A/B vs 8-bit K = **zero retrieval cost, 1.9× capacity** (131K d10/50/90 + 240K d50, 4/4 = 4/4) |
| Decode (single) | **~59–68 tok/s** (MTP K=2, acceptance ~3.0) | 59 = 1-shot deploy gate 2026-08-23; 65–68 = 3-rep benches 2026-08-16/17 |
| Decode (aggregate) | **364.7 tok/s @ 16 lanes** (agent-shaped) | prod shape 2026-08-17; 24 lanes regresses (knee mapped) |
| Vision | **ON** (2026-08-23) | images ≤4/prompt, auto-downscaled to ~1 MP (~976 img tokens); reads fine screenshot text end-to-end through the gateway |
| Spec decode | MTP K=2 (`VLLM_MTP_DRAFT_CAP=2`, dense-FP16 draft load) | K=8 chained REFUTED at prod shape (agg −43%, 2026-08-18) — K=2 strictly better |

Vision ceiling: the transformers-5.13 qwen3_5 processor grid-mismatches above ~1024
image tokens (≈1 MP) — `max_pixels 1000000` sits just under it. Native full-res needs a
processor patch (open item).

## What this fork lineage adds over upstream v0.1.16

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
  background no-think, flight recorder, live dashboard + 34 contract tests
- `tools/update-from-remote.sh` — status / `--apply` (ff-only) / `--restart`
  (safe engine window behind the failover)
- systemd units + watchdog, banked known-good serve configs, model switcher

**Verdict ledger (measured, kept honest):**
- 3-bit-K: **keep** (zero retrieval cost, 1.9× pool) · retention-interval port
  (#45845-style): **no benefit** at our 3536-token Mamba blocks (OFF=ON pool) ·
  K=8 MTP chain: **refuted** at prod shape · EXL2/EXL3: **rejected** (no qwen3_5 arch /
  no SM75) · suffix overlay: **reverted** (−35% single on novel text)

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
  + upstream v0.1.16 prep). `frontier-pastnative-20260816` tracks it in lockstep (local
  prod checkout).
- Worktree branches (`feat-*`, `exp*`, `steal-*`) — experiment lineages; all merged or
  superseded by the default, kept for the worktrees that pin them.
- Rollback: tag `pre-consolidation-rollback`.

## Hardware notes

Modded 22 GB 2080 Tis, NVLink NV2 active (~51 GB/s) — GPU1's x4 PCIe is not a TP
bottleneck. No ECC: soak new configs before trusting them (12-minute passes lie; use
30-minute mixed soaks).
