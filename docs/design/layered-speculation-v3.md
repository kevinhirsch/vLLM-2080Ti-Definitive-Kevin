# Layered Speculation v3 — MTP + ngram_gpu GPU tensor merge (build plan)

**Goal:** best-of-both speculative decoding on 2×2080Ti: MTP's learned drafts for novel text, GPU prompt-lookup for repetitive/agentic spans — **with async scheduling and CUDA graphs kept on** (the constraint that killed v1/v2).

## Measured evidence (2026-08-13, Qwen3.6-27B-GPTQ, prose/echo/codegen tok/s)
| config | prose | echo | codegen | async |
|---|---|---|---|---|
| champion: mtp3 + graphs | **83.7** | 105.7 | **86.1** | ON |
| suffix16 + graphs (method=suffix) | 44.0 | 121.2 | 49.1 | off |
| v1 overlay (mtp8 + suffix CPU) | 55.9 | 25.2 | 64.8 | off |
| v2 conditional (suffix-first, mtp cap 3) | 56.3 | 38.8 | 59.5 | off |
| **ngram_gpu@8 + graphs** | 43.2 | **118.6** | 51.6 | **ON** |

**Conclusions:** (1) CPU drafts force async scheduling off ≈ −40% on this box — unrecoverable by skipping the 1-layer MTP head (v2 proved it). (2) `ngram_gpu` achieves suffix-class echo speed **with async intact** → the merge must happen GPU-side. Projected v3: **~84 / ~118 / ~86**.

## Architecture
Run BOTH proposers each step; merge draft tensors on-GPU per request:
`merged[i] = ngram[i] if ngram_valid[i] >= COVER_MIN else mtp[i]` (torch.where on valid counts; MTP rows padded to k_slots with valid=k_mtp). Communicate per-request lengths via the existing `_num_valid_draft_tokens` async D2H path (ngram_gpu already does this under async scheduling — the plumbing exists).

Config shape: scheduler slots `num_speculative_tokens=8`; MTP chain capped at 3 via `VLLM_MTP_DRAFT_CAP` (landed in v2, commit 0219ec7); ngram `prompt_lookup_max=6/min=2`.

## The two integration walls (why this is a dedicated session)
1. **14 `use_ngram_gpu()`/`is_ngram_gpu` gates in `gpu_model_runner.py`** (lines ~550, 834, 1159–1464, 3913, 4364, 4648 at time of writing): buffer allocation (`token_ids_gpu_tensor` ~8MB, fine), incremental tensor sync after batch condense/reorder, accepted-token corrections, optimistic-accept handling, valid-count copies. Each must fire when EITHER method=ngram_gpu OR the overlay is active — and several sit inside flows that the eagle path also touches; each interaction needs review, not a blind `or`.
2. **Drafter KV-cache groups:** EagleProposer/MTP registers its own KV group (`drafter.kv_cache_gid`, runner ~line 2357) during init. With method=mtp primary that's already handled — so **primary=mtp + ngram_gpu overlaid** is the right direction (ngram_gpu needs no KV, only its token buffers), NOT the inversion (primary=ngram_gpu can't feed an unregistered MTP head).

## Build sequence (est. one focused session)
1. Extract ngram_gpu's runner-side state ops into helpers callable when `VLLM_NGRAM_OVERLAY=1` with method=mtp (audit each of the 14 gates; most are mechanical, 1304/1464 corrections need care).
2. In `propose_draft_token_ids` eagle branch: run ngram update+propose first (cheap kernels), then eagle (capped k=3), then the torch merge; set `_num_valid_draft_tokens` for the merged tensor.
3. Gates: syntax → single-stream bench (expect ≥ champion on all three) → 6-way concurrency (0 errors, 0 OOM) → temp-0 exact-answer quality under concurrency → hours of production soak.
4. Rollback = env off (strict no-op, same pattern as v1/v2).

## Session-ops notes
- Engine cold restart ~4–5 min (torch.compile cache invalidation); `TimeoutStartSec=600` drop-in required (deploy/systemd/).
- Gateway (:8000) overflows to DeepSeek during restarts — safe to iterate on live box.
- Bench harness: session scratchpad `specbench/bench.py` (prose/echo/codegen, streams TTFT + decode tok/s); reproduce from this table's workloads if lost.
