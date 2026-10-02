# Frontier session 2 — 2026-08-16 evening (Qwen3.8-27B, 2×2080 Ti SM75, TP=2)

Continuation of `frontier-changelog-20260816.md`. Four shipped results, one proven
serving primitive, one refutation, one open crash class. All measured (3-rep where
timing matters), all reproducible from the branches below.

## 1. k3v4_nc demolishes the context VRAM wall — SHIPPED
`--kv-cache-dtype turboquant_k3v4_nc` (3-bit K / 4-bit V) at `--max-model-len 524288`
gives a **758,046-token KV pool** (vs ~374K estimated max on k8v4). The prior 352K
wall (VRAM-budget crowding, see session-1 EXP-041) does not exist in this dtype regime.
- **Gotcha (align-mode floor):** 3-bit K packs fewer bytes/token, so mamba align mode
  needs 3488 tokens/block to byte-match the GDN page → boot asserts unless
  `--max-num-batched-tokens ≥ 3488`. We ship 3584. Any KV-dtype change must re-check
  this floor.
- Proven: 465,693 then **491,563-token uncached needle prefills** served correctly
  (needle retrieved from 90% depth, coherent output). Quality: 100-item evalkit at
  per-category parity with the k8v4 baseline (3-bit K holds at least to 128K-context
  items; the one −1 category delta was shown to be sampling noise via 3 reps).

## 2. The MTP "collapse" was a silently mis-loaded draft — SHIPPED (+54% decode)
The GPTQ layer matcher (`is_layer_gptq_quantized`) keys on bare projection names
(`q_proj`…), so the MTP draft decoder (`mtp.layers.0.self_attn.q_proj`) is built as
GPTQMarlin — but the checkpoint stores all 15 `mtp.*` tensors dense BF16 (GPTQ skips
the head). Result: the draft loads against missing `qweight` → silently degraded
drafts → the historical near-zero acceptance that got MTP turned off on this class of
setup.
- **Fix:** `VLLM_QWOPUS_MTP_BF16_DRAFT=1` (forces the whole draft dense; FP16 on SM75
  via `--dtype half` cast on load — `qwen3_5_mtp.py:88-137`).
- Measured with the fix, K=2, `cudagraph FULL_AND_PIECEWISE [4]`:
  **acceptance p0=0.786 / p1=0.563** (D=5623), **single-stream 43.5 → 67.1 tok/s
  (+54%)**, agg unchanged, evalkit parity, spec decoding lossless (greedy token-exact
  vs MTP-off modulo cudagraph-mode numeric tie-breaks).
- **Open crash class:** with MTP ON, requests in (352K, 380K] reliably fault (Xid 31
  FAULT_PDE, both GPUs). NOT cudagraph-specific (reproduces under
  `enforce_eager` + `CUDA_LAUNCH_BLOCKING=1`), NOT global VRAM exhaustion (3.2 GB free
  at death), NOT a fixed position boundary (death at 52s fresh-boot vs 457-478s aged).
  Production is fenced (gateway per-request cap = 352000 = proven coverage). Hunt
  continues (compute-sanitizer minimized repro queued).

## 3. Concurrency knee mapped — SHIPPED at the peak (16 lanes)
Agent-shaped traffic (~1K prompt / 300 gen), 3-rep per rung, on the shipped config:
`max-num-seqs` **8 → 268 agg tok/s (p50 8.7s) · 12 → 334 (10.5s) · 16 → 369
(+37%, 12.7s) · 24 → 308 REGRESSES (17.7s)**. Peak is exactly 16 on this box; past it
scheduler/spec overhead inverts the curve. KV pool unchanged across rungs.

## 4. EXP-038: snapshot → pin → restore → FORK, all proven on hardware (branch `feat-gdn-snapshot`)
The hybrid-state serving primitive: prefill once, continue N times.
- **Stage-0:** park a live sequence's GDN recurrent state (48 layers × conv+temporal =
  96 tensors) into a park pool via the align-mode `batch_memcpy` path —
  `torch.equal` byte-exact on both TP ranks.
- **Stage-1:** pin the request's KV blocks scheduler-side (`block_pool.touch()` via new
  env-gated EngineCore utility RPCs) — blocks survive the source request's free with
  ref_cnt ≥ 1.
- **Stage-2 (zero-kernel restore):** pin mid-generation → source freed → cache churned
  with filler prompts → resubmit identical token_ids → **num_cached_tokens 2080/2112**
  (rode the pinned cache, no prefix recompute) → 64-token continuation byte-exact with
  logprobs matching. `num_cached_tokens` is the restore-vs-recompute discriminator;
  block-table intersection is NOT a valid observable (partial attn blocks never
  cache-hit; mamba state is copied out of cache into a fresh running slot by design).
- **Stage-3 (fork):** one pinned handle → two co-scheduled children. Both cache-hit,
  64 greedy tokens byte-exact 3-way, shared attn block ref-bumped, **distinct** mamba
  running slots per child. Max logprob delta 1.16e-02 = batch-shape numerics (reference
  batch=1 vs children batch=2; GPU reductions are not batch-invariant) — tokens are the
  correctness signal.
- Everything env-gated (`VLLM_TQ_GDN_SNAPSHOT`, default off, inert in production).
  Drivers: `tools/tq_gdn_snapshot_stage{0,1,2,3}.py` (double-guarded, throwaway-engine
  only). Next: a scheduler-level `fork(handle, n)` API without resubmit.

## 5. Refuted (so nobody retries it): suffix overlay stacked on MTP
`VLLM_SUFFIX_OVERLAY=1` on top of MTP-K2: single 67.1 → 43.6 (−35%), agg8 237 → 166
(−30%) on generation-shaped work — overlay drafts are rejected constantly on novel
text and drag the spec pipeline. Reverted. Verbatim re-emission (44.9% of agent-shaped
output by census) is real but replay-shaped; the exploit is a scoped FSM-gated drafter
(planned), not a blanket overlay.

## Branch index (this session)
- `feat-gdn-snapshot` — EXP-038 stages 0-3, all passing. Candidate upstream RFC.
- `frontier-pastnative-20260816` — serving branch (rope-cache sizing + bounds guard + docs).
- `feat-kv-workspace-reserve` — boot-time workspace accounting (upstream-worthy safety fix).
- `feat-retention-interval` — port of upstream #45845 for Mamba/GDN groups.
