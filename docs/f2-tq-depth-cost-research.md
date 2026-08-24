# F-2 phase 1: TurboQuant decode depth-cost research

Research only. No code changed, no engine run (the GPU-timing half of F-2
happens later in a hardware window). Scope: read our TurboQuant decode
kernels and dispatch tree, compare the scaling behavior against FlashInfer's
FP16-KV paged decode in the same tree, read the fork's own upstream-PR
history (#71, #78) for prior work on exactly this problem, and rank
candidate mechanisms for WHERE the context-depth cost lives, with an
engine-window measurement plan to discriminate between them.

Repo: `weicj/vLLM-2080Ti-Definitive` (this checkout), branch
`frontier-pastnative-20260816`. All line numbers below are against this
branch's HEAD (`48e5eb931`) unless marked otherwise.

## 0. The problem, restated with numbers

The fork's own sweep (`docs/qwen36-kv-throughput-sweep.md`, Qwen3.6 27B,
GPTQ-INT4, MTP3) shows TurboQuant KV is *nearly free* at short context and
*badly degraded* at long context — not a fixed percentage penalty, a
**growing** one:

| depth | FP16 decode | TQK8V4 decode | TQ4NC decode | TQK8V4 penalty | TQ4NC penalty |
|---|---:|---:|---:|---:|---:|
| PP4096/TG128 | 81.7 | 94.6 | 88.3 | (TQ faster here — MTP noise) | (TQ faster here) |
| PP65536/TG512 | 85.5 | 40.2 | 36.9 | **-53%** | **-57%** |

(FP8-checkpoint MTP3 rows show the same shape: 76.0→70.4 (-7%) at 4K vs.
70.8→36.9 (-48%) at 65K.) Our own production measurement on
`turboquant_k3v4_nc` (3-bit K, the more aggressive preset, not in the
committed sweep table) is consistent with this shape: **~76 tok/s at 4K
depth vs. ~28.5 tok/s at 65K depth**. If the cost were a constant per-token
dequant tax, the ratio TQ/FP16 would be roughly flat across depths. It
isn't — something in the TQ decode path adds work that scales *worse* than
FP16's own O(context) attention cost. That something is the target of this
research.

## 1. The decode kernels — what scales with L

Two Triton kernels do TQ's actual dequant/attention math
(`vllm/v1/attention/ops/triton_turboquant_decode.py`):

### 1a. `_tq_decode_stage1` (lines 98–374) — the real fused decode kernel

Grid = `(B, Hq, NUM_KV_SPLITS)` with `NUM_KV_SPLITS` **fixed** (default 32,
`vllm/config/attention.py:34` `tq_max_kv_splits_for_cuda_graph: int = 32`,
overridable via `VLLM_TURBOQUANT_MAX_KV_SPLITS`; must be constant because
"grid dims must be constant for cudagraph",
`turboquant_attn.py:661-667`). Per-program work:

```python
# triton_turboquant_decode.py:154-161
active_len = seq_len - kv_start
split_len = tl.cdiv(active_len, NUM_KV_SPLITS)
split_start = kv_start + split_len * sid
split_end = tl.minimum(split_start + split_len, seq_len)
if split_start >= split_end:
    return
...
for start_n in range(split_start, split_end, BLOCK_KV):   # line 195
```

`BLOCK_KV` defaults to **2** tokens/tile
(`_read_decode_block_kv`, `VLLM_TURBOQUANT_DECODE_BLOCK_KV`, default `"2"`,
lines 35-50). So each of the 32 parallel split-programs walks
`≈ L/(32·2)` tile iterations, each doing an online-softmax update fused with
inline dequant: for MSE keys (k3v4_nc/4bit_nc), 2 gather loads + bit-unpack
+ a **centroid table gather** (`tl.load(Centroids_ptr + mse_idx, ...)`,
lines 253-258) + (if `norm_correction`) an extra `sqrt`/normalize
(lines 261-267) + a norm-scale load/multiply — vs. the FP8-key path
(k8v4), which is a single bitcast load + cast (lines 216-236, no gather, no
extra correction step). Total per-decode-step work is **O(L)**, same
asymptotic class as any exact-attention decode — this is inherent, not a
TQ defect. What differs is the *constant factor per KV element*, and
(§1b) whether this kernel is actually the one that runs.

### 1b. `_tq_full_dequant_kv` (lines 383-518) — the bulk pre-materialize kernel

Grid = `(alloc_len, B·Hk)`, one program per `(position, head)` pair, **no
loop** — this kernel individually is loop-safe/cheap per instance, but the
grid size itself is `alloc_len ≈ cached_len` rounded up to `block_size`.
It dequants **every single cached position**, K and V, unconditionally,
into two full fp16 `[B, Hk, max_seq, D]` output tensors
(`_continuation_prefill`, lines 2103-2208). This is **not** the decode
kernel — it is a bulk "reconstruct everything to fp16" pass, called once
per `_continuation_prefill` invocation.

**The dispatch question that matters: which of these two kernels handles
an MTP verify step (K+1-wide query) at depth L, by default?**

Answer: **1b, every time**, for our shipped MTP config. Trace:

- `TurboQuantMetadataBuilder.build`: `is_prefill=(cam.max_query_len > 1)`
  (line 541). MTP K=3 → query width 4 → `is_prefill=True` for *every*
  verify step, not just the first.
- `reorder_batch_threshold` is only raised above 1 (which would let
  spec-decode-width queries classify as "decode") when
  `supports_spec_as_decode=True`, which is gated on
  `_TQ_CUDAGRAPH_SPEC_DECODE_SAFE`
  (`VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE`, **default `"0"`**,
  line 216-218, 497-499). Off by default → every MTP verify step is
  classified `num_decodes=0`, i.e. "pure prefill batch" (line 1232).
- `forward()` dispatches `is_prefill and num_decodes==0` to
  `_prefill_attention` (line 1236), whose per-request loop hits the
  `q_len != seq_len` "continuation chunk" branch (`cached_len = seq_len -
  q_len`, line 1876) and checks, in order: a Gemma4-shared-cache path (not
  our model), then
  `_SPEC_CONTINUATION_DECODE_FASTPATH and q_len <= 128` (line 1919-1924).
  `_SPEC_CONTINUATION_DECODE_FASTPATH`
  (`VLLM_TURBOQUANT_SPEC_CONTINUATION_DECODE_FASTPATH`) **also defaults off**
  (line 98-100). With both flags at their shipped defaults, every branch
  falls through to the final `else` (line 1982-1997): **`_continuation_prefill`**
  → `_tq_full_dequant_kv` over the *entire* cached prefix, every step
  (lines 2179-2208), then flash_attn/flashinfer/SDPA over the
  materialized+concatenated K/V.

This is not a discovery unique to this research — it's already documented
in-repo, in the source's own comment, written when this exact fastpath was
added:

```python
# turboquant_attn.py:92-97
# Continuation prefill: for small continuation chunks (q_len ≤ threshold),
# use the TQ decode kernel directly instead of full-dequant + flash_attn.
# do_kv_cache_update already stored all tokens to TQ cache, so the decode
# kernel can read them efficiently. This avoids O(cached_len) dequant work
# per continuation, eliminating the O(N²/chunk_size) collapse at long context.
```

The fix this comment describes exists in code
(`_spec_continuation_decode_attention`, lines 1560-1641 — see §3) and is
**disabled by default**. §7 of `docs/k3-tiny-prompt-hang-rca.md` (written
2026-08-24 for an unrelated hang bug) independently traced this same
dispatch tree and reached the same conclusion: *"this fork's default
TurboQuant + MTP config never uses the dedicated decode kernel for
spec-decode verify steps... That's a standing performance property."* This
research corroborates and extends that finding with the depth-scaling
argument the hang RCA didn't need.

## 2. FP16-KV comparison — what FlashInfer's paged decode does at depth

`vllm/v1/attention/backends/flashinfer.py`, `FlashInferMetadataBuilder`:

- `reorder_batch_threshold: int = 1` is an **unconditional class attribute**
  (line 546) — no `supports_spec_as_decode` gate. MTP verify-width queries
  are classified as decode unconditionally.
- `_decode_cudagraph_max_bs = (1 + num_spec_tokens) * max_num_reqs`
  (line 595) — the decode wrapper/cudagraph budget is explicitly **sized
  for the K+1 spec width**, confirming MTP verify steps are meant to run
  through the *decode* wrapper (`BatchDecodeWithPagedKVCacheWrapper`), not
  a prefill/continuation path.
- `decode_wrapper.run(decode_query, kv_cache_permute, ...)` (line
  1746/1762) reads the **paged FP16 KV cache directly** — no separate
  materialize/dequant kernel exists in this path, because the cache's
  native storage format (FP16) already *is* the format the attention
  kernel needs. FlashInfer's own paged-decode kernel is architecturally
  the same shape as our `_tq_decode_stage1` (split-KV tiled, GQA-aware),
  just without an unpack/gather step per element.

**The delta, stated precisely:** FP16 KV never needs a "convert cache
format to something attention math can consume" step, at any query width,
because storage format and compute format are the same. TQ's compressed
format does need that conversion, and our fork has **two** ways to do it —
a fused single-pass way (`_tq_decode_stage1`, cost ≈ FP16's decode cost
plus a per-element dequant constant) and a bulk-materialize way
(`_tq_full_dequant_kv` + full attention over the copy, cost ≈ 2× the O(L)
memory traffic: write the dequant, then read it back for attention). The
fused way is used for plain single-token decode (`_decode_attention`,
line 2613, taken when `not attn_metadata.is_prefill`) and would closely
track FP16's scaling. The bulk way is what MTP verify steps get by
default (§1), and it is **not** what FlashInfer (or our own decode kernel)
does for the FP16 case — this asymmetry, not "TQ dequant is intrinsically
expensive," is the primary depth-cost driver.

## 3. #71 / #78 — do we have their work, and does it fix this?

- **PR #78** (`0xYYP/research/tq-long-context-performance`, merged
  `207a64c9e`, constituent commit `23f44c1bd` "Optimize TurboQuant
  long-context attention") is **already an ancestor of our current HEAD**
  (`git merge-base --is-ancestor 207a64c9e HEAD` → true). Its own commit
  message states the measured result: *PP65536/TG512 improved from
  "1259.39 / 37.28 tok/s to 1578.86 / 42.61 tok/s"* — a ~14% decode-speed
  gain, **not** a fix of the collapse (42.61 tok/s is still far below the
  ~85 tok/s FP16 reference at the same depth).
- **PR #71** (`0xYYP/fix/tqk8v4-kv-quality`, merged `65c727f0c`) is also an
  ancestor of HEAD. Its title and diff shape (essentially the initial
  TQ-quality-hardening squash) target output-quality/format correctness
  (FP8 format selection, norm handling), not decode throughput at depth —
  not directly relevant to this bug.
- **What #78 actually shipped, read from `_continuation_prefill`
  (lines 2002-2450):** the "prefix-combine" mode
  (`_tq_continuation_prefix_combine_enabled`, lines 103-131; env
  `VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE`, default `"auto"`, enabled
  once `seq_len >= 20480` via
  `VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS`). When enabled
  (true for both the 65K depth rung and any 20K+ production request),
  `_continuation_prefill` runs **two** FlashInfer prefill calls — a
  non-causal call against the dequanted cached prefix and a causal call
  against the raw current chunk — merged via `merge_attn_states` on their
  LSE outputs (lines 2243-2375), instead of concatenating into one
  `k_full`/`v_full` buffer and running a single call (lines 2377-2410+).
  **Critically, both branches are downstream of the same
  `_tq_full_dequant_kv` call** (lines 2179-2208) — prefix-combine changes
  what happens to the dequanted K/V *after* materialization (avoids one
  large concat-buffer copy, lets FlashInfer plan two smaller calls), it
  does **not** skip or reduce the O(cached_len) dequant itself. So: **yes,
  we have #78, and no, it is not the fix for this collapse** — it is a
  real, smaller, already-realized optimization (~14% per its own
  benchmark) layered on top of the same O(L)-per-step redundant-dequant
  problem.
- **The actual (unpromoted) fix already in our tree** is the
  `_SPEC_CONTINUATION_DECODE_FASTPATH` path introduced in §1b above (not
  from #78 — it ships in the same commit range but is architecturally
  distinct): `_spec_continuation_decode_attention` (lines 1560-1641).
  For `1 < q_len ≤ 128` (MTP K=3 → q_len=4, well inside the threshold) with
  the flag on, it computes prefix attention via `triton_turboquant_decode_attention`
  reading the **compressed** cache directly (fused dequant+attention, no
  materialize step — architecturally the same shape as FlashInfer's decode
  wrapper) and combines it with a small raw-K/V causal SDPA over just the
  current chunk via LSE merge (`torch.logaddexp`, lines 1635-1641) — same
  merge-by-LSE idea as #78's prefix-combine, but applied *before* dequant
  cost is paid, not after. This is a real fix candidate, present in code,
  off by default, and — per the RCA doc's own read — "currently-inert."

## 4. k3-specific cost (k3v4_nc vs k8v4) — a compounding factor, not the driver

`vllm/model_executor/layers/quantization/turboquant/config.py:20-41`:

| preset | `key_quant_bits` | `key_fp8` | `norm_correction` |
|---|---:|---|---|
| `turboquant_k8v4` | 8 | **True** | False |
| `turboquant_k3v4_nc` | 3 | **False** | **True** |

In both `_tq_decode_stage1` and `_tq_full_dequant_kv`, `KEY_FP8=1` (k8v4)
takes a **single bitcast load + cast** per KV element
(`triton_turboquant_decode.py:216-228` / `432-440`). `KEY_FP8=0` (k3v4_nc)
takes the MSE path: 2 gather loads + bit-unpack + a **data-dependent
centroid-table gather** (`Centroids_ptr + mse_idx`, an indexed load whose
address depends on the quantized code — inherently less coalesced than a
straight load) +, because `norm_correction=True` for k3v4_nc specifically,
an *extra* `sqrt`/normalize step **inside the tile loop**
(`NORM_CORRECTION` branch, lines 261-267/459-463) that k8v4 skips entirely.
This is a real, larger per-element constant factor for k3v4_nc, and it
explains why TQ4NC decode is consistently a few points worse than TQK8V4
at the *same* depth in the sweep table (36.9 vs 33.6 at PP65536/TG512,
GPTQ MTP3). But it is a **flat multiplier per KV element**, applied
uniformly regardless of L — by itself it would produce a constant
percentage penalty at every depth, not the growing one in §0's table. It
compounds §1's redundant-full-redequant problem (each of the O(L)
redundant dequants costs more per element for k3 than k8) rather than
being an independent depth-scaling mechanism.

## 5. Hypothesis ranking

| # | Mechanism | Code evidence | Depth-ladder signature | Engine-window measurement to discriminate |
|---|---|---|---|---|
| **H1 (top)** | MTP verify steps (query width K+1>1) are misclassified as prefill by default (`is_prefill=(max_query_len>1)`, §1b) and fall through to `_continuation_prefill`, which unconditionally re-dequants the **entire** cached K/V via `_tq_full_dequant_kv` on **every** verify step — a materialize-then-attend pattern FP16/FlashInfer never needs (§2) — while the fused fastpath that avoids this (`_spec_continuation_decode_attention`) exists but defaults off | `turboquant_attn.py:92-97` (author's own "O(N²/chunk_size) collapse" comment), `:541` (`is_prefill` classification), `:1919-1997` (fallthrough to `_continuation_prefill`), `:2179-2208` (unconditional full dequant) | Smooth, ~linear-in-L growth of decode-step cost — tok/s should look like a decaying curve (76→28.5), not a plateau or a sharp step (the 20480-token prefix-combine threshold changes post-dequant call shape only, so should barely dent the curve) | `nsys`/torch-profiler wall-clock share of `_tq_full_dequant_kv` vs. the downstream flash_attn/flashinfer call, at L=4K/65K/200K, default config. H1 confirmed if `_tq_full_dequant_kv` time grows ~linearly with L and becomes the dominant term at 65K+. **Decisive A/B**: rerun the same ladder with `VLLM_TURBOQUANT_SPEC_CONTINUATION_DECODE_FASTPATH=1` — if 65K/200K tok/s recovers substantially toward the FP16 MTP numbers, H1 is confirmed and the fix is a one-line env flip (pending requal, §6) |
| **H2** | Fixed decode-kernel tiling (`NUM_KV_SPLITS=32`, `BLOCK_KV=2`, both constant for cudagraph) doesn't scale split-count with L, so very large L serializes more tile iterations per program without adding parallelism | `triton_turboquant_decode.py:35-50, 154-161, 195`; `vllm/config/attention.py:34` | A knee/plateau in `_tq_decode_stage1` time once `L/32` tiles per program exceeds some occupancy threshold — different shape than H1's smooth decay. Only observable on paths that actually reach this kernel (plain single-token decode today; MTP verify steps only if H1's fastpath is enabled) | Time `_tq_decode_stage1` alone (not `_tq_full_dequant_kv`) across the same L ladder, both with plain decode and (if H1's flag is on) MTP verify steps. Sweep `VLLM_TURBOQUANT_MAX_KV_SPLITS` and `VLLM_TURBOQUANT_DECODE_BLOCK_KV` at L=200K to see if a knee moves |
| H3 | k3v4_nc's larger per-element dequant constant (centroid gather + norm-correction, §4) | `config.py:20-41`; kernel `NORM_CORRECTION`/MSE branches | A roughly **constant** multiplicative gap between k3v4_nc and k8v4 curves at every depth — if the gap instead *widens* with L, that would implicate a depth-interacting mechanism beyond a flat per-element constant | Time `_tq_full_dequant_kv` (or stage1) per-token for k8v4 vs k3v4_nc at the *same* L, several depths. Expect a flat ratio; a widening ratio would need new investigation |
| H4 (weak) | `_tq_full_dequant_kv` grid size (`alloc_len × Hk`) scales with L — read as "more kernel launches/dispatch overhead at depth" | `triton_turboquant_decode.py:383-425` grid computation | Grid growth is expected *parallel* work, not per-call launch-count growth (one Python-level launch regardless of L) — weak candidate, likely folds into H1's bandwidth argument rather than being separable | If profiling shows `_tq_full_dequant_kv` is latency- (occupancy-) bound rather than bandwidth-bound at large L, this gains weight; otherwise treat as subsumed by H1 |
| H5 (exonerated) | `merge_attn_states` LSE-merge overhead from prefix-combine (#78) or the fastpath's `logaddexp` merge | `turboquant_attn.py:2357-2363`, `:1635-1641` | Would show as a fixed small cost independent of L (merge operates on already-reduced `(q_len, Hq, D)` tensors, not on the L-sized K/V) | Not worth separate instrumentation; time it once to confirm it's negligible (<1% of step time) and move on |

## 6. Fix directions (ranked to H1)

1. **Flip `VLLM_TURBOQUANT_SPEC_CONTINUATION_DECODE_FASTPATH=1` — zero port,
   engine-window experiment only.** The code already exists
   (`_spec_continuation_decode_attention`, `turboquant_attn.py:1560-1641`)
   and is reachable for our shipped MTP config (`q_len=K+1=4 ≤
   _CONTINUATION_DECODE_THRESHOLD=128`) without needing
   `VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE` (that flag only affects the
   separate `_spec_decode_attention`/`force_spec_decode`/reorder-threshold
   path, not this branch's `elif` condition at line 1919-1924). Before
   trusting it for production: run the same qualification suite the
   MTP-requal effort already defines (`deploy/bench/bench_equivalence.py`,
   `bench_mtp_requal.py`, `bench_toolcalls.sh`, `bench_decode.py` per
   `deploy/docs/QWEN-3.8-MTP-REQUALIFICATION.md`) — this flag is untested at
   scale (the k3-tiny-prompt-hang RCA calls it "currently-inert") and its
   "current chunk" attention is a manual float32 `torch.einsum`
   (`turboquant_attn.py:1619-1633`, not a fused kernel) that could itself
   become a secondary bottleneck at high concurrency — worth a quick check,
   not a blocker (q_len ≤ 8 keeps it tiny per the code's own comment).
   Port-size: **none** (config-only); validation-size: same as any MTP
   requal pass, roughly a half-window.
2. **If (1) reveals a quality gap specific to k3v4_nc** — `_tq_decode_stage1`'s
   MSE/centroid path was benchmarked by #78 against TQK8V4 only (per its
   commit message); the 3-bit path through the fused decode kernel at long
   context hasn't had the same validation. Run the existing evalkit against
   k3v4_nc with the fastpath on, at 128K+ depth, before adopting for that
   preset specifically. No port needed, validation-only.
3. **If (1) doesn't fully close the gap** — generalize prefix-combine so its
   "prefix" leg also reads the compressed cache directly (via
   `triton_turboquant_decode_attention`) instead of always materializing
   first via `_tq_full_dequant_kv`, i.e. merge the fastpath's approach into
   `_continuation_prefill` itself rather than gating it behind
   `_CONTINUATION_DECODE_THRESHOLD=128`. Medium port (refactor one function,
   `turboquant_attn.py:2002-2558` (`_continuation_prefill`), to make the
   dequant call conditional on
   *not* being able to use the compressed-read path) — rough estimate
   1-2 engine-days plus a full requal, only worth it if (1) alone
   underperforms.
4. **Cross-reference, not urgent:** `docs/0.2.x-migration-memo.md` row 9
   already flags that upstream's v0.27.1/0.2.x line has "TQ reworked
   (#71/#78 rows)" — when that migration happens, re-check whether the
   reworked 0.2.x `turboquant_attn.py` (present on
   `origin/vllm-2080ti-definitive-0.2.x`, e.g. commit `80889bef3`, which
   this branch does **not** contain) defaults this fastpath differently or
   restructures the dispatch tree in a way that obsoletes some of the above.
   Out of scope for this doc; noted for the eventual 0.2.x TQ-parity task.

## 7. Engine-window measurement checklist (for the GPU-timing half of F-2)

- Depth ladder: 4K / 65K / 200K, MTP K=3, `turboquant_k3v4_nc`, default
  flags (baseline matching the ~76/~28.5 production numbers).
- Per rung: `nsys profile` or `torch.profiler`, isolate wall-clock for
  `_tq_full_dequant_kv`, `_tq_decode_stage1`, and the downstream
  flash_attn/flashinfer/SDPA call by name.
- Repeat the same ladder with `VLLM_TURBOQUANT_SPEC_CONTINUATION_DECODE_FASTPATH=1`.
- Repeat once more with `turboquant_k8v4` at both flag settings, same
  ladder, to separate H1 (dispatch) from H3 (k3-specific constant factor).
- Per [[Benchmark Rigor]]: warm up, 3+ reps per rung per arm, report median
  and spread.
