# RCA: K=3 MTP + tiny prompt decode deadlock

Status: **analysis only, no fix applied**. Written 2026-08-24 against HEAD
`afd55280d` (which already includes three same-day commits in exactly this
territory — see §6).

## 0. Bug statement (as reported)

With MTP speculative decoding `num_speculative_tokens=3` (K=3), a SHORT
prompt (~4 tokens, e.g. "Say OK.") deadlocks decode forever: request
accepted, 0 tok/s, no errors, `/health` fine. K=2 with the same prompt is
fine. K=3 with 4096-token prompts is fine (72.9 tok/s). Reproduced 5/5 on
hardware.

Config: TP=2, GPTQ-marlin, `kv_cache_dtype=turboquant_k3v4_nc`,
`mamba-cache-mode=align`, chunked prefill `mnbt=3584`, `max_num_seqs=16`,
compilation `FULL_AND_PIECEWISE` `cudagraph_capture_sizes=[4]`,
`max_model_len=524288`, hybrid GDN model (qwen3_5).

Hard evidence:
1. py-spy: both TP workers ACTIVE inside
   `GPUModelRunner._update_states_after_model_execute`,
   `vllm/v1/worker/gpu_model_runner.py:1784` —
   `self.num_accepted_tokens.gpu[:num_reqs].cpu().numpy()`, a device sync
   that never returns.
2. Journal shows these Triton kernels JIT-compiling at the hang moment
   (first inference on the fresh K=3 engine): `_zero_kv_blocks_kernel`,
   `_compute_slot_mapping_kernel`, `eagle_prepare_next_token_padded_kernel`,
   `eagle_step_slot_mapping_metadata_kernel`, `expand_kernel`,
   `eagle_prepare_inputs_padded_kernel`.
3. A hung request showed constant 6.6% GPU KV-cache usage (~637K-token
   pool).
4. `eagle_prepare_*_padded` kernels are per-request O(1) — already
   inspected, cannot loop forever.

## 1. Verdict up front

I could **not** find a literal unbounded/wrapping GPU loop by static
analysis of the kernels named in the evidence, of the cudagraph dispatch
code, or of the scheduler's prefill/spec-decode interleaving. All three of
the hinted mechanisms (tasks #1–#3) are exonerated below with quoted
arithmetic. That is a real result, not a dead end: it means the bug is
almost certainly not "a Triton `for` loop wrapped around to 2^32" — it's
either (a) a genuinely huge-but-finite amount of GPU work being issued
(pathological grid/split count), (b) a CUDA-graph capture/replay pointer
mismatch specific to hybrid Mamba/GDN state under `align` mode (the class
of bug this fork has been actively firefighting *today*, same session), or
(c) a host-side wait (event/stream) that's never satisfied because a
kernel it depends on was never actually launched on one of the two ranks.

The single strongest lead is not in the kernels I was pointed at — it's in
the fork's own git history from a few hours before this report (§6): a
same-day commit whose message is a near word-for-word match for this exact
symptom, in exactly this (Mamba-align × MTP × short/variable draft width)
territory, whose own fix does not actually rule out the configuration it
describes as broken. Start there before spending a hardware window on
`CUDA_LAUNCH_BLOCKING` tracing.

---

## 2. Task 1 — kernel-by-kernel loop-bound audit

All six kernels named in the journal evidence were located and read in
full. None contains a loop whose trip count is computed from an unguarded
subtraction that can go negative and get reinterpreted as a huge unsigned
count. Triton's `for i in range(lo, hi, step)` follows Python `range()`
semantics on signed operands: if `lo >= hi` the loop body executes zero
times — it does **not** wrap to `~2^32` iterations the way a C
`for (unsigned i = lo; i < hi; ...)` would. That was the mechanism
hypothesized in the task brief; it requires an *unsigned* loop variable,
and none of these six kernels declare one.

### 1a. `_zero_kv_blocks_kernel` — `vllm/v1/worker/utils.py:41-77`

No loop at all. One `tl.store` per program instance; `pid` maps
1:1 to `(block_index, seg_index, chunk_index)` and out-of-range
`block_index` returns immediately (line 65-66: `if block_index >= n_blocks:
return`). Grid size (`vllm/v1/worker/utils.py:209`,
`grid = (n_blocks * n_segs * (page_size_el // blk_size),)`) is a function
of the model's fixed page geometry, not of prompt length or K. Exonerated.

### 1b. `_compute_slot_mapping_kernel` — `vllm/v1/worker/block_table.py:326-381`

Two `for i in range(...)` loops, both over **signed** `int64`/derived
bounds:

```python
for i in range(num_tokens, max_num_tokens, BLOCK_SIZE):   # line 345 (padding tail)
...
for i in range(start_idx, end_idx, BLOCK_SIZE):            # line 359 (per-request)
```

`start_idx`/`end_idx` come from `query_start_loc_ptr` (`.to(tl.int64)` at
load, line 354-355) — monotonically non-decreasing by construction, so
`start_idx <= end_idx` always; even if it weren't, a negative range on
signed operands is empty, not wrapped. `num_tokens <= max_num_tokens` by
construction. Exonerated.

### 1c. `eagle_step_slot_mapping_metadata_kernel` — `vllm/v1/spec_decode/utils.py:29-85`

No loop. Pure per-thread arithmetic (`req_idx = tl.program_id(0)`,
early-return for padding lanes at line 56-58). Exonerated.

### 1d. `eagle_prepare_inputs_padded_kernel` — `vllm/v1/spec_decode/utils.py:137-176`

No loop. `num_rejected_tokens` (line 167) can legitimately be 0 (guarded
via `tl.where(num_draft_tokens > 0, num_rejected_tokens, 0)` at line 168)
but is only ever used in a subtraction that's stored, not used as a bound.
Exonerated.

### 1e. `eagle_prepare_next_token_padded_kernel` — `vllm/v1/spec_decode/utils.py:180-239`

No runtime-bounded loop. `token_offs = tl.arange(0, BLOCK_SIZE_TOKENS)`
(line 213) is a compile-time-sized vector (`BLOCK_SIZE_TOKENS` is a
`tl.constexpr`, "power-of-2 >= num_sampled_tokens_per_req" i.e. bounded by
`K+1<=4`), masked by `token_offs < num_sampled_tokens_per_req`. Exonerated.

### 1f. `expand_kernel` — `vllm/v1/sample/rejection_sampler.py:830-850`

```python
offset = tl.arange(0, MAX_NUM_TOKENS)
tl.store(output_ptr + start_idx + offset, src_val, mask=offset < num_tokens)
```

`MAX_NUM_TOKENS` is bound to `MAX_SPEC_LEN = 128`
(`vllm/v1/sample/rejection_sampler.py:34,599`) — a fixed compile-time
unroll, not a function of the request's actual `num_tokens`. Even if
`num_tokens` (`end_idx - start_idx`, line 845) were negative, `offset >= 0`
always fails the mask, so the store is simply a no-op, not a hang.
Exonerated.

**Why these six, specifically, showed up as fresh JIT compiles**: none of
them differ in signature between a "normal" decode step and this one — the
more likely explanation (see §5) is that this is genuinely the *first*
real (non-warmup-shape) invocation of the short-prompt →
first-spec-decode-continuation transition for this engine process, and
these are the first-touched kernels on that transition's control-flow
edge, not evidence that any of them individually loops.

One adjacent kernel worth flagging for the record, *not* in the evidence
list but the only other Triton kernel in the whole spec-decode/rejection
stack with a **runtime-bounded** (not masked-constexpr) loop:

```python
# vllm/v1/sample/rejection_sampler.py:733 (rejection_greedy_sample_kernel)
# and :789 (rejection_random_sample_kernel)
for pos in range(num_draft_tokens):
```

`num_draft_tokens = end_idx - start_idx` where both come from
`cu_num_draft_tokens_ptr` (monotonic cumulative sum, signed int32) — same
"signed subtraction, empty not wrapped" argument as 1b applies, and in
practice `num_draft_tokens <= K = 3` for this config. It is **not** in the
"freshly JIT'd" list, which is consistent with it having already been
compiled during the engine's dummy/profiling warmup (same kernel signature
regardless of prompt content). Flagged only because it's the one kernel in
the whole traced call graph with a genuinely dynamic loop bound; if a
future repro's JIT log *does* include it, revisit this file first.

---

## 3. Task 2 — FULL vs PIECEWISE cudagraph dispatch

**Likely exonerated for this model**, for a reason the task brief didn't
anticipate: this fork unconditionally downgrades **decode** cudagraphs
from FULL to PIECEWISE for any hybrid-Mamba/GDN model doing spec decode,
regardless of what `--compilation-config` asks for, unless a specific
opt-in env var is set. `CUDAGraphMode.FULL_AND_PIECEWISE` decodes as
`(decode_mode=FULL, mixed_mode=PIECEWISE)`
(`vllm/config/compilation.py:63`); the downgrade forces the decode half to
PIECEWISE too:

```python
# vllm/config/compilation.py:1462-1486
# Hybrid Mamba/GDN models update recurrent state from the speculative
# acceptance metadata during decode. Keep speculative decode out of
# full-graph replay for these models; replaying the whole model forward
# can reuse graph-captured recurrent-state update topology across
# changing acceptance patterns. PIECEWISE keeps the stateful attention
# kernels outside the full model graph while preserving compiled fast
# paths.
if (
    cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
    and not allow_mamba_spec_full_cudagraph
    and uniform_decode_query_len > 1
    and kv_cache_config is not None
    and kv_cache_config.has_mamba_layers
):
    ...
    cudagraph_mode = CUDAGraphMode.PIECEWISE
```

`uniform_decode_query_len = 1 + self.num_spec_tokens`
(`vllm/v1/worker/gpu_model_runner.py:839`) = 4 for K=3 → `>1` is true;
`kv_cache_config.has_mamba_layers` is true (qwen3_5 hybrid GDN);
`allow_mamba_spec_full_cudagraph` requires
`VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=1`, which defaults to `"0"`
(`vllm/envs.py:947-949`) and is not part of the reported config. So decode
cudagraphs are PIECEWISE here, not FULL — the "exact-fit capture size 4 vs
K=2's padded-to-4 under FULL replay" scenario in the task brief is not
reachable for this model. (I also checked whether TurboQuant's own
`AttentionCGSupport.UNIFORM_BATCH` claim — line 493 of
`turboquant_attn.py` — could exempt it from this downgrade: it can't, this
particular check is gated on `kv_cache_config.has_mamba_layers`, not on
backend capability.)

**Verify before trusting this exoneration**: the boot log for today's
repro should contain the warning string built at
`vllm/config/compilation.py:1476-1478`
("`CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for
Mamba/GDN KV cache layers ... setting cudagraph_mode=PIECEWISE`"). If that
line is **absent**, this exoneration is wrong (meaning
`VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=1` was set) and task #2's original
FULL-mode hypothesis should be promoted back to the top.

**What PIECEWISE does *not* rule out**, and where I'd point the next
instrumentation pass instead: PIECEWISE still cudagraph-captures whatever
ops aren't explicitly excluded as split points — traditionally that's
"everything except attention." For a hybrid GDN model that includes the
Mamba/GDN recurrent-state-update (chunked gated-delta-net scan) kernels
under `vllm/model_executor/layers/fla/ops/` and
`vllm/model_executor/layers/mamba/ops/`. The fork comment quoted above is
explicitly worried about *FULL* graphs "reusing graph-captured
recurrent-state update topology across changing acceptance patterns" —
but if the GDN scan itself is inside the PIECEWISE-captured region (not
excluded the way attention is), the same class of capture-vs-replay
pointer/topology mismatch could apply there too, just via a narrower
graph. I did not have time in this pass to kernel-audit the FLA/mamba scan
stack the way I did TurboQuant attention and the six named kernels — see
§7 gap list. This is my top recommendation for the next static-analysis
pass if the environmental leads in §6 don't pan out.

---

## 4. Task 3 — can prefill and spec-decode be scheduled in the same step?

**Exonerated.** The scheduler explicitly reserves zero spec/draft-token
lookahead for a request's first-ever scheduling step:

```python
# vllm/v1/core/sched/scheduler.py:693 (context: new-request path)
effective_lookahead_tokens = (
    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
)
```

and the running-request loop that actually assigns spec tokens
(`vllm/v1/core/sched/scheduler.py:487-499`) only fires when
`request.spec_token_ids` is already populated, which only happens *after*
a forward pass has produced them (`update_draft_token_ids`, run between
steps). So for a brand-new request the lifecycle is two cleanly-separated
engine steps, not one blended step:

- **Step N**: pure prefill of all `num_prompt_tokens` tokens
  (`num_computed_tokens: 0 -> 4`), `num_lookahead_tokens=0`, no spec
  tokens scheduled. Model forward produces the first sampled token *and*
  (for MTP, which reuses the model's own last-position hidden state) the K
  draft tokens for the next step.
- **Step N+1**: `request.spec_token_ids` is now populated;
  `num_new_tokens = num_tokens_with_spec - num_computed_tokens = (4 + 1 +
  3) - 4 = 4` → this is the first genuine "continuation" / verify step,
  `num_scheduled_tokens=4=K+1`, `cached_len=4` (exactly the prefill
  length). This is the step that actually exercises the code paths
  discussed in §5, and it is a clean, single-purpose K+1-token batch, not
  a hybrid prefill+draft shape the kernels weren't built for.

I also traced `_mamba_block_aligned_split`
(`vllm/v1/core/sched/scheduler.py:267-313`, the fork's align-mode
prefill-chunk splitter) through this exact case
(`prompt_len=4 < block_size=16`) and confirmed it does not truncate or
zero out the prefill chunk: `prefill_end=max(4,3)=4`,
`chunk_end` stays `num_computed_tokens + num_new_tokens = 4` (neither the
`< last_cache_position` nor `< prefill_end` branch fires because
`chunk_end == prefill_end`), so `num_new_tokens` stays 4 — the whole
4-token prompt is scheduled in one prefill chunk regardless of K.

---

## 5. What actually runs on step N+1 (new finding, not in the task brief)

This required tracing the attention dispatch tree in
`vllm/v1/attention/backends/turboquant_attn.py`, and it surfaces an
undocumented fork switch that changes which of **two structurally
different code paths** handles every single spec-decode attention call —
worth understanding independent of whether it's the root cause.

`TurboQuantMetadataBuilder.__init__` (line 495-499):

```python
self._init_reorder_batch_threshold(
    1, supports_spec_as_decode=_TQ_CUDAGRAPH_SPEC_DECODE_SAFE
)
```

`_TQ_CUDAGRAPH_SPEC_DECODE_SAFE = os.getenv(
"VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE", "0") == "1"` — **defaults
off**, and is not mentioned in the reported config. Upstream's
`_init_reorder_batch_threshold` (`vllm/v1/attention/backend.py:550-574`)
only raises `reorder_batch_threshold` above 1 to accommodate spec-decode
query widths when `supports_spec_as_decode=True`:

```python
if self.reorder_batch_threshold is not None and supports_spec_as_decode:
    ...
    self.reorder_batch_threshold = max(self.reorder_batch_threshold,
                                        max_num_queries_for_spec)
```

With the flag off, `reorder_batch_threshold` stays `1`, so
`split_decodes_and_prefills(..., decode_threshold=1)` classifies **every**
K=3 verify step (query_len=4>1) as *not* a decode. That flows straight
into `TurboQuantMetadata.is_prefill = (cam.max_query_len > 1)` (line 541)
= **True**, for every single decode step of the entire generation, not
just the first. Consequences in the top-level dispatch
(`turboquant_attn.py:1206-1440`):

- The "pure decode fast path" (`if not attn_metadata.is_prefill:` →
  `_decode_attention`, line 1220-1231) is **never** taken for this
  config's default — good, because `_decode_attention` treats
  `query.shape[0]` as one-query-token-per-batch-row and passes
  `attn_metadata.block_table`/`seq_lens` through unexpanded; taking that
  path with a `(K+1)*num_reqs`-row query against `num_reqs`-row
  `seq_lens`/`block_table` would be a real per-token-vs-per-request shape
  mismatch. It's moot only because the flag is off by default.
- Instead, `num_decodes==0` is true (all "prefill"), so every verify step
  runs through `_prefill_attention`'s per-request loop
  (`turboquant_attn.py:1777-1998`), hits the `q_len != seq_len`
  "continuation chunk" branch (`cached_len = seq_len - q_len`, line 1876),
  and — because `_SPEC_CONTINUATION_DECODE_FASTPATH` (env
  `VLLM_TURBOQUANT_SPEC_CONTINUATION_DECODE_FASTPATH`) **also defaults
  off** — falls through every fast-path `elif` to the final `else`
  (line 1982-1997): `_continuation_prefill` (dequant cached K/V via the
  `_tq_full_dequant_kv` Triton kernel, lines 383-518 of
  `triton_turboquant_decode.py`, then flashinfer/flash_attn/SDPA on the
  concatenated K/V).

I traced `_tq_full_dequant_kv` fully: it is pure grid-parallel, one
program instance per `(position, batch*head)`, no loop at all
(`pos = tl.program_id(0)`, line 413) — exonerated the same way as §2's
kernels. I also traced the *other* code path this config is one env flag
away from (`VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE=1` →
`_spec_decode_attention`, line 1442-1499, → `triton_turboquant_decode_attention`
→ `_tq_decode_stage1`, `triton_turboquant_decode.py:98-195`). That kernel
*does* have a real per-tile loop —

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

— but it's guarded by the `split_start >= split_end: return` immediately
above it, `seq_len` is a signed `int32` load, and `NUM_KV_SPLITS` is a
small fixed constexpr (`tq_max_kv_splits_for_cuda_graph`, chosen "fixed...
because grid dims must be constant for cudagraph" — not derived from
`max_model_len`). Exonerated for the same "signed underflow degrades to
empty, not wrapped" reason as everywhere else.

**Net: both candidate attention code paths for the K+1=4 verify step are
individually loop-safe.** Neither is a smoking gun. But this *is* a real,
previously-undocumented fact worth carrying forward: this fork's default
TurboQuant + MTP config never uses the dedicated decode kernel for
spec-decode verify steps — it dequantizes and re-concatenates cached K/V
and runs full attention (flashinfer/flash_attn/SDPA) on every single
decode step, for every request, regardless of K. That's a standing
performance property (not obviously a hang cause) that the person tuning
this config should know about; `VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE=1`
looks like an intentionally-built, currently-inert optimization.

---

## 6. The strongest lead: today's own commit history

`git log` on this exact HEAD shows three commits landing in the hours
before this bug report, all in precisely the "Mamba `align` mode × MTP ×
short/variable draft width" intersection this bug lives in:

| time (local) | commit | what |
|---|---|---|
| 04:18:23 | `b8a6c49e3` | scheduler: read `VLLM_MAMBA_ALIGN_RETAIN_MTP_CACHE_BLOCK` via envs module |
| 04:47:01 | `10da3244f` | kv-cache: tolerate required-below-allocated in Mamba align allocate (fixes an assertion **crash** — `num_required_blocks 17 < len(req_blocks) 18` — triggered by *variable per-step draft width plus rejection rollback* under align mode) |
| 07:29:22 | `b2e84e2ec` | GDN: port vllm-project#51508 — skip stale zero-accept async rows (upstream-diagnosed "silent recurrent-state corruption" under async spec decode) |
| 09:27:17 | **`ec3c6d8b3`** | **M-7: fail fast at boot on `VLLM_MTP_DRAFT_CAP != num_speculative_tokens`** |
| 09:27:59 | `afd55280d` (HEAD) | docs only |

`ec3c6d8b3` is the one to read closely — its commit message is close to a
verbatim match for this bug report:

> `cap(3) == K(3)`: **silent decode deadlock — requests accepted, 0 tok/s
> forever, no errors.**
> ... Only cap unset has ever worked cleanly.

That is this bug's symptom, described hours earlier, in the same repo,
same session. But the guard it shipped
(`vllm/v1/spec_decode/mtp_draft_cap.py:48-81`) does **not** actually
forbid `cap == K` — it explicitly allows it, reasoning that
`min(K, cap) == K` makes it "mathematically a no-op" identical to unset
(`llm_base_proposer.py:80-82`:
`self.num_speculative_tokens = min(self.num_speculative_tokens, _cap)`). I
verified there is no second, un-clamped read of `VLLM_MTP_DRAFT_CAP` that
could make `cap==K` behave differently from unset for the MTP proposer
used here (`grep` turns up exactly three read sites: the guard itself,
the proposer's `min()`-clamp, and `scoped_reemission.py`'s S4 scoped
drafter — which is a separate opt-in feature, `VLLM_S4_SCOPED_DRAFTER=1`,
not part of this config or of `serve-qwen38-mtp-requal-fg.sh`). So per the
code as written, `cap==K` and `cap` unset **should** be behaviorally
identical — which directly contradicts the commit message's claim that
`cap==K` deadlocks while (implicitly) unset does not.

**My reading of this contradiction**: the M-7 "7-experiment bisect"
almost certainly varied `VLLM_MTP_DRAFT_CAP` across experiments without
independently controlling prompt length, and attributed a deadlock to the
cap value when the actual discriminator was the prompt used in that
particular experiment. If the `cap==3` arm happened to use a short prompt
and the `cap unset` arm happened to use a longer one (very plausible in an
ad-hoc bisect run under time pressure), the bisect would reproduce exactly
what we're seeing now and mis-attribute it. That would mean:
`VLLM_MTP_DRAFT_CAP` is a **red herring**, unrelated to root cause, and
M-7's guard — while harmless — does not actually fix, or even mitigate,
this bug.

This matters operationally right now: the closest in-repo launch script
to the reported config, `deploy/bin/serve-qwen38-mtp-requal-fg.sh`,
unconditionally sets `export VLLM_MTP_DRAFT_CAP=3` (line 38) whenever
`MTP_K=3` (its default, line 42). If today's repro used this script or a
derivative, `cap==K==3` was in effect — the one non-unset value the new
M-7 guard permits, and per its own commit message, the one value
empirically tied to the deadlock.

**Cheapest next experiment, before touching any kernel code**: rerun the
exact 4-token-prompt / K=3 repro with `VLLM_MTP_DRAFT_CAP` completely
**unset** (not `=3`) and the tiny prompt. If it still hangs, this
confirms the cap is a red herring and root cause is purely
`(K=3, prompt_len<=~block_size)` as this doc's kernel tracing suggests. If
it stops hanging, the cap genuinely matters through a mechanism this pass
didn't find (worth another read of `llm_base_proposer.py` end-to-end with
that specific question), and M-7's guard needs to flip from "allow
`cap==K`" to "allow only unset."

---

## 7. Ranking against the five discriminators

| Candidate | K=3 bad, K=2 fine | 4-tok bad, 4096-tok fine | both TP workers identical | 5/5 deterministic | verdict |
|---|---|---|---|---|---|
| 6 named kernels (§2) | n/a — none loop | n/a | n/a | n/a | **exonerated** |
| `rejection_*_sample_kernel` dynamic loop (§2, bonus) | plausible (K bounds the loop) | no distinguishing mechanism found | consistent (replicated metadata) | consistent | **unlikely** — same signed-underflow argument applies; not in the fresh-JIT list |
| FULL cudagraph exact-fit dispatch (§3) | would fit (capture_sizes=[4] matches K=3's shape exactly, K=2 pads) | no mechanism tying it to prompt length specifically | consistent | consistent | **downgraded** — FULL isn't used for decode on this model by default; only reopens if `VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=1` was set |
| PIECEWISE capturing the GDN/mamba scan (§3) | fits (uniform_decode_query_len=4 only matches capture_sizes=[4] at K=3; K=2's shape=3 likely never gets captured at all, falls back to safe eager) | untested in this pass — plausible if capture happened against warmup-only mamba state pointers | fits (deterministic capture, replayed identically on both ranks) | fits (capture is deterministic) | **top technical candidate**, not yet confirmed — biggest audit gap (§8) |
| Scheduler combining prefill+spec in one step (§4) | n/a — mechanism doesn't exist | n/a | n/a | n/a | **exonerated** |
| TQ continuation dispatch / `_tq_full_dequant_kv` (§5) | no K-specific branch found | no cached_len-specific branch found (same function for 4 and 4096) | consistent | consistent | **exonerated** (fully grid-parallel, no loop) |
| `_tq_decode_stage1` (only reachable if `VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE=1`) | fits (K sizes the split loop) | no mechanism found | consistent | consistent | **exonerated if reached at all** — early-return guard is sound |
| Mamba-align × MTP short-prefix bookkeeping (§6) | fits — same-day fixes were specifically about K/draft-width-dependent align-mode block accounting | **fits directly** — align mode's block-boundary math is fundamentally different when the whole prompt fits inside one (partial) Mamba block vs. spanning many full blocks | fits (pure metadata arithmetic, replicated per-rank, not a race) | fits | **top circumstantial candidate** — strongest evidence, not yet reduced to a specific offending line |

## 8. What I did not get to (gaps, ranked by priority)

1. **The FLA / Mamba-SSM Triton kernel stack** (`vllm/model_executor/layers/fla/ops/*.py`,
   `vllm/model_executor/layers/mamba/ops/{ssd_chunk_scan,ssd_chunk_state,ssd_state_passing,mamba_ssm}.py`)
   — the actual GDN recurrent-state-update kernels for the hybrid layers.
   This is where §3's "PIECEWISE still captures the mamba scan" concern
   and §6's "align-mode short-prefix bookkeeping" concern physically
   converge, and I did not have time in this pass to read it kernel by
   kernel the way I did TurboQuant attention and the six named kernels.
   **This is the single highest-value next static-analysis target.**
2. `mamba_utils.postprocess_mamba` (called immediately *after* the hang
   point at `gpu_model_runner.py:1787`, so it cannot itself be the cause
   of this specific hang, but is adjacent code in the same align-mode
   bookkeeping path worth a pass regardless).
3. Confirming which of §3's/§5's env-flag branches the *actual* harness
   used today takes — I inferred defaults from `vllm/envs.py` and
   in-repo scripts, but the bug report's "Config:" line doesn't enumerate
   `VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE`,
   `VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH`,
   `VLLM_MAMBA_ALIGN_RETAIN_MTP_CACHE_BLOCK`, or `VLLM_MTP_DRAFT_CAP`
   explicitly. Everything in §3 and §5 should be treated as conditional on
   those actually being at their documented defaults in the failing run.

## 9. Recommended instrumentation for the next hardware window

In priority order (cheapest/most-discriminating first):

1. **Control for the M-7 confound directly** (§6): repeat the 4-token/K=3
   repro with `VLLM_MTP_DRAFT_CAP` unset (grep the actual env file/service
   drop-in used and remove it, don't just rely on script defaults). Zero
   code changes required.
2. **Confirm the cudagraph downgrade actually fired** (§3): grep the boot
   log for `"setting cudagraph_mode=PIECEWISE"` / `"not supported with
   spec-decode for Mamba/GDN"`. Zero code changes required.
3. **`VLLM_TURBOQUANT_DEBUG_MIXED=1`** (`_TQ_DEBUG_MIXED`, already wired
   at `turboquant_attn.py:219` and used at lines 1304-1310/1384-1399/
   1878-1886) — logs `q_len`/`cached_len`/`seq_len`/`kv_dim` per request
   right before the branch decisions traced in §5. This turns "which
   code path actually ran" from inference into direct observation, for
   free.
4. **`CUDA_LAUNCH_BLOCKING=1`** on a repro run, with `py-spy dump` (or
   `cuda-gdb --pid` for the GPU side) taken while hung. Under blocking
   launches, the Python frame that's stuck will be the actual enqueuing
   call for the runaway kernel, not just the first downstream `.cpu()`
   sync — this directly answers "which kernel" instead of requiring more
   static-analysis inference.
5. If (4) points at the GDN/mamba stack, follow up with `nsys profile`
   to see whether the runaway kernel is still resident/executing (genuine
   device-side spin) versus the stream simply never getting the launch
   its CPU-side event-wait is blocked on (a host-side bug, not a GPU
   loop) — these look identical from a `.cpu()`-sync stack trace alone
   and require a GPU-side tool to tell apart.

---

## 10. Proposed minimal fix (NOT applied — diff for review only)

Given §7/§8, I don't have a single confirmed offending line to patch. What
I can respectably propose now is a **fail-fast conversion**: turn this
class of failure from "silent infinite spin" into "loud, fast error,"
mirroring the pattern this fork already uses for the sibling Xid31 issue
(`_TQ_CONTINUATION_BOUNDS_CHECK`, `turboquant_attn.py:2163-2178`). This
does not fix the root cause, but it converts a 5/5-reproducible
un-diagnosable hang into an immediately-actionable stack trace on the very
next repro, which is worth more right now than a guess-and-check kernel
patch.

```diff
--- a/vllm/v1/attention/backends/turboquant_attn.py
+++ b/vllm/v1/attention/backends/turboquant_attn.py
@@ class TurboQuantAttentionImpl(AttentionImpl["TurboQuantMetadata"]):
     def _continuation_prefill(
         self,
         layer: Any,
         query: torch.Tensor,  # (q_len, Hq, D)
         key_chunk: torch.Tensor,  # (q_len, Hk, D)
         val_chunk: torch.Tensor,  # (q_len, Hk, D)
         kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
         block_table: torch.Tensor,  # (1, max_num_blocks)
         cached_len: int,
         seq_len: int,
         Pi: torch.Tensor,
         centroids: torch.Tensor,
         force_sdpa: bool = False,
     ) -> torch.Tensor:
         """Handle continuation chunk by dequanting cached K/V from TQ cache.

         Dequants previously cached K/V, concatenates with the current
         chunk's raw K/V, then runs flash_attn with causal masking.
         """
         q_len, Hq, D = query.shape
         Hk = key_chunk.shape[1]
         device = query.device
+        # [FORK][diagnostic] K3-tiny-prompt-hang RCA (docs/k3-tiny-prompt-hang-rca.md):
+        # fail loud instead of silently spinning if the scheduler/mamba-align
+        # bookkeeping ever hands this function a degenerate shape. Cheap
+        # (host-side int compares only, no extra device sync beyond what
+        # already happens for these Python ints) and always on, unlike
+        # VLLM_TURBOQUANT_CONTINUATION_BOUNDS_CHECK which is opt-in and only
+        # covers the block_table OOB case, not degenerate cached_len/seq_len.
+        if cached_len < 0 or seq_len <= 0 or cached_len >= seq_len or q_len <= 0:
+            raise RuntimeError(
+                f"TQ continuation degenerate shape: layer={getattr(layer, 'layer_name', None)} "
+                f"q_len={q_len} cached_len={cached_len} seq_len={seq_len} "
+                f"Hq={Hq} Hk={Hk} D={D} -- refusing to launch dequant/attention "
+                f"kernels rather than risk a silent hang (see "
+                f"docs/k3-tiny-prompt-hang-rca.md)."
+            )
         prefix_combine_enabled = (
```

And a second, narrower guard right where §5 identified the actual
first-touched path for this bug (`_prefill_attention`'s per-request
continuation branch), so the assertion fires *before* any kernel launch
for this exact scenario rather than one function-call deep:

```diff
--- a/vllm/v1/attention/backends/turboquant_attn.py
+++ b/vllm/v1/attention/backends/turboquant_attn.py
@@ def _prefill_attention(
             else:
                 # Continuation chunk: tokens already stored to TQ cache
                 # by do_kv_cache_update. Use decode kernel directly to
                 # avoid O(cached_len) full-dequant per continuation.
                 # For large continuations, fall back to _continuation_prefill.
                 cached_len = seq_len - q_len
+                if cached_len <= 0:
+                    raise RuntimeError(
+                        f"TQ continuation with cached_len<=0: layer={getattr(layer, 'layer_name', None)} "
+                        f"req={i} q_len={q_len} seq_len={seq_len} -- this should be "
+                        f"impossible (continuation implies a prior cached prefix); "
+                        f"scheduler/mamba-align bookkeeping produced an inconsistent "
+                        f"shape (see docs/k3-tiny-prompt-hang-rca.md, §6/§8)."
+                    )
                 if _TQ_DEBUG_MIXED:
```

Both diffs are pure diagnostics (raise, don't silently clamp/continue) so
they cannot mask the bug or change behavior on the (presumed common) happy
path — they only change a silent infinite spin into an immediate,
informative crash the next time this exact repro is run. I'd land these
*before* the next hardware window regardless of which candidate in §7 the
team decides to chase first, since they cost nothing on the working paths
and turn "0 tok/s forever, no errors" into a stack trace pointing at the
exact request/layer/shape.
