# vLLM Load-Test Benchmarks — HNET00 (Qwen3.6-27B on 2× RTX 2080 Ti)

Chaos-tested to failure on 2026-07-17. Config under test (the validated production config):
`--gpu-memory-utilization 0.88`, `--max-num-seqs 2`, `--max-model-len 256000`,
`VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS 262144`, TurboQuant K8V4 KV, MTP K=3.
KV cache = **543,766 tokens (2.12× 256K)**. Harness fires N *unique* concurrent prompts
(`ignore_eos` forces full-length decode), samples GPU peak at 0.5 s.

## Headline: this is a "one big OR two modest" box
The fork is explicitly single-concurrency ("queue, don't parallelize long prefill; not a
multi-tenant stack"). The numbers below confirm it exactly.

## Throughput (end-to-end tok/s, incl. prefill)
| Load | tok/s | Notes |
|---|---|---|
| 1 request @ 7.5K ctx, 800 decode | **45** | single-request decode baseline |
| 2 concurrent @ 7.5K ctx | **63 aggregate** (~31 each) | batching win, 1.4× |
| 2 concurrent @ 14K ctx | 29 aggregate | |
| 2 concurrent @ 30K ctx | 15 aggregate | prefill starting to dominate |
| 2 concurrent @ 42K ctx | 10 aggregate | |
| 2 concurrent @ 60K ctx | ~4 aggregate | contention collapse |
| 1 request @ 200K ctx | **0.3** | prefill-bound; ~4 min to first token |

Prefill ≈ 850 tok/s → a 200K-token prompt is ~4 minutes just to TTFT.

## Single vs. Dual concurrency — "time to complete 2 requests" (gen 400 each)
| Context | Single (2× sequential) | Dual (parallel) | More efficient |
|---|---|---|---|
| 7.5K | 35.2 s | **25.2 s** | Dual (1.40×) |
| 14K | 32.0 s | **27.5 s** | Dual (1.16×) |
| **~42K** | 82.2 s | 80.7 s | **Tied — the crossover** |
| 60K+ | fast per-request | 301 s (collapse) | Single (queue) |

**Rule: dual is more efficient below ~40K context; single/queued above it.** And for
latency (one answer ASAP) single always wins — each request runs 45 tok/s solo vs ~31 shared.

## VRAM ceilings (22.5 GiB/GPU)
| State | VRAM | Free |
|---|---|---|
| idle, fresh boot | 18.9 GiB | 3.6 GiB |
| idle, "worked-in" (allocator cache after heavy ctx) | 21.2 GiB (94%) | 1.3 GiB |
| 2 concurrent @ 15–42K | 21.2 GiB (94%) | stable |
| 1 request @ 200K | **22.0 GiB (97.6%)** | 0.5 GiB — one big req nearly fills the card |
| 2 concurrent long-context (>~80K each) | **OOM** | impossible on 44 GB |

## The two crash cliffs (both fixed)
1. **OOM under 2-concurrent** at `gpu-memory-utilization 0.92` (card 99.9% full, 19 MiB free;
   any spike OOMs both workers → EngineDeadError → AZ sees ConnectionRefused). **Fix: 0.88.**
2. **Workspace-lock at context >131K**: `_continuation_prefill requires 264 MB, workspace
   locked at 256 MB` — the reserve (131072 tok) was half of `max-model-len` (256K).
   **Fix: `WORKSPACE_RESERVE_TOKENS` → 262144** (covers full 256K; single 200K req now works).
- **Do NOT** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — it crashes worker init
  on this SM75 fork (conflicts with FULL_AND_PIECEWISE CUDA graphs).

## Operating envelope (what to actually run)
- **1 big request** up to ~256K ctx — single-concurrency only, and slow at the top.
- **2 concurrent** — only up to ~40K ctx each to stay efficient; safe (no OOM) up to ~60–80K
  each but with collapsing throughput.
- **Never** 2 concurrent long-context (both >~80K) → OOM.
- **>2 concurrent, or anything long** → queue it, or spill to the cloud tier (DeepSeek).

This envelope is why Agent Zero is configured: **local roles = moderate context (fast, dual-safe),
DeepSeek tiers = large context (fast on cloud)**. See ~/Desktop/Qwen-AgentZero-Optimization.md.
