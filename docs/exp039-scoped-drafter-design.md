# EXP-039 (S4): Suffix-scoped drafter for verbatim re-emission

**Branch:** `feat-s4-scoped-drafter` (worktree `~/Desktop/.ftree-s4drafter`, base `frontier-pastnative-20260816`)
**Status:** first A/B ran on hardware (2026-08-16 eve) → **NEGATIVE, root-caused, fixed, re-A/B'd**
(2026-08-17). The gate-open drafts were misaligned by one step (proposing the token *just* sampled),
collapsing copy-span acceptance to ~0. Fixed in `scoped_reemission.py`; unit-test proven, then
**re-A/B'd on hardware (seqs-4 / max-len 65536): quote 365→811.5 tok/s (mean 7.44 accepted/step),
mixed no-harm, generation unchanged** — the fix is confirmed. **Prod-shape re-proof (full
concurrency/context envelope) still pending.** See **§11** for the measured numbers, the proven root
cause, the fix, and the re-test protocol.
**Env switch:** `VLLM_S4_SCOPED_DRAFTER=1` (default `0` = strict no-op).

---

## 1. Motivation (measured, not assumed)

Census over real agent transcripts (file rewrites, diffs, quoted code, tool-call args):

- **44.9%** of generated tokens are ≥16-token **verbatim copies of context**.
- tool_call / code-arg tokens are **54.3%** of generation.

So nearly half of what the model emits, it is *copying* from something already in its prompt
(the file being rewritten, the block being quoted). On those spans a learned drafter is wasted:
the next token is *determined* by the context, and a cheap CPU lookup accepts at ~1.0.

The fork already tried to exploit this with a **blanket suffix-tree overlay**
(`VLLM_SUFFIX_OVERLAY`, `docs/design/layered-speculation-v3.md`). It was just measured at
**−35% single-stream** on generation-shaped work and reverted. Root cause is **not** the plumbing —
it is that the overlay proposes **ungated, always-on**: every step it runs Arctic
`SuffixDecodingCache.speculate()` and, via `_merge_suffix_drafts`, swaps the MTP draft out whenever
the suffix draft is merely `len >= 2` (`VLLM_SUFFIX_OVERLAY_MIN=2`) — i.e. on *any* weak statistical
continuation, not only on provable verbatim copies. On novel/generation text those drafts are
constantly rejected, and each rejection drags the whole spec pipeline (wasted target verification +
the CPU-draft path forces async scheduling off, ≈ −40% on this box — see §7).

Prod baseline to protect: **MTP-K2 dense-draft**, `p0=0.786 p1=0.563`, **67.1 tok/s single**
(`--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`).

**Design goal:** exploit the copy regime **without** touching MTP-K2 on the generation regime.
Only draft from context when we are *provably* inside a verbatim re-emission span; otherwise leave
MTP-K2 exactly as it is.

---

## 2. What we reuse from the reverted overlay (lessons + plumbing)

The overlay was reverted for its *policy*, but two pieces of its *mechanism* are sound and are reused:

1. **`VLLM_MTP_DRAFT_CAP`** (`llm_base_proposer.py:74-82`, landed in overlay v2, commit `0219ec7`).
   Caps the **learned** drafter's chain length independently of the scheduler's draft slots. This is
   the key that lets us size the scheduler / KV-lookahead / cudagraph pipeline for a *long* draft
   (`num_speculative_tokens = K_scoped`, e.g. 16) while MTP still only does its cheap **K_mtp=2** GPU
   chain. Without it, launching `num_speculative_tokens=16` would make MTP autoregress 16 draft steps
   every decode — precisely the dense-draft cost we must avoid.

2. **CPU-list draft merge after the MTP tensor propose** (`_merge_suffix_drafts` shape). We keep the
   same integration point (merge *after* `self.drafter.propose(...)` in the eagle/MTP branch) and the
   same list-path output, but replace the merge *policy* with the FSM gate.

What we **discard**: `_suffix_early` (skip-MTP-if-covered) and `_merge_suffix_drafts` (swap-if-len≥2).
Those are the ungated policy. `VLLM_SUFFIX_OVERLAY` is left intact and independent; S4 is a separate
switch.

---

## 3. Data-structure choice — hashed windowed index over the prompt

We must answer, once per request per decode step, for the request's **prompt** token array `P`
(the context being copied from) and the current generated suffix `s`:

> Do the **last G** tokens of `s` appear as a contiguous G-gram somewhere in `P`, and if so, is the
> continuation **unique** (or near-unique)? If yes, return the next `K_scoped` prompt tokens.

Candidate structures:

| structure | build | query/step | incremental | notes |
|---|---|---|---|---|
| **Suffix automaton (SAM)** | O(n), heavy const | O(1) amortized | yes | pure-Python SAM over ≤256K tokens = thousands of dict-transition nodes; tens of MB, tens of ms build; gives *longest*-match we don't need |
| **Suffix array** | O(n log n) | O(G log n) | no | clean uniqueness (range size) but full SA over 256K each request |
| **Hashed windowed index (chosen)** | O(n log n) in numpy C | O(log n) + O(G) verify | rebuild-free (prompt is fixed) | fixed-window gate = exactly what we need; cheapest adequate |

**Chosen: a fixed-window (length-G) rolling-hash index, stored as a sorted-hash array +
`searchsorted` lookup** ("hashed suffix-array lite"). Justification:

- The gate is a **fixed-window predicate** ("last G tokens match"), *not* a longest-match query, so a
  suffix automaton's variable-length matching is capability we'd pay for and never use.
- Build is a single vectorised numpy pass: `h[s] = Σ_t P[s+t]·base^(G-1-t) (mod 2^64)` computed with
  G≤16 vectorised multiply-adds, then one `argsort`. No Python per-token loop → ~milliseconds even at
  256K (vs a Python dict-build loop which is ~50-100ms at 256K).
- Query is `np.searchsorted(sorted_h, needle_hash, 'left'/'right')` → O(log n), returning the range of
  candidate **end-positions**; hash collisions are resolved by an exact G-token comparison, so the
  hash need not be cryptographic (correctness never depends on it — see §5, losslessness).
- Uniqueness falls straight out of the candidate-range size.
- The prompt is immutable for the life of the request, so the index is built **lazily on the first
  decode step** and never rebuilt (unlike Arctic's cache which also ingests the growing response).

Cost bound (per request, per step): one uint64 add-chain of length G to hash the needle (~16 ops),
one `searchsorted` (~log₂(256K) ≈ 18 comparisons), and G-token verify on the usually-1 candidate.
That is **single-digit microseconds** — four orders of magnitude under the ~15 ms GPU step. Build is
amortised once over the whole generation. At `max_num_seqs=2` this is 2 lookups/step.

**Scope note:** the index is built over prompt tokens only (the target regime — file-rewrites carry
the file in the prompt). Extending the haystack to already-generated tokens (re-emitting earlier
output) is a documented follow-up knob, deliberately out of scope for the first cut.

---

## 4. Gate design (the FSM)

Evaluated **fresh each step** per request — a memoryless FSM with two states:

```
                last-G-gram unique match in P ?
   SEARCHING ───────────── yes ─────────────▶ COPYING   (emit P[end : end+K_scoped])
      ▲                                          │
      └───────────── no unique match ────────────┘
```

- **needle** = `seq[n-G : n]`, the last `G` tokens of the full sequence (`G≈8-16`, knob `VLLM_S4_G`).
- Look the needle's G-gram up in the index → candidate prompt end-positions `E`.
- **Verify** each candidate with an exact G-token compare (kills hash collisions).
- **Uniqueness gate** (`VLLM_S4_MIN_UNIQ`, = max allowed occurrences, default `1` = strictly unique):
  - `len(E_exact) == 0` → **SEARCHING**, gate closed, MTP draft passes through untouched.
  - `1 ≤ len(E_exact) ≤ MIN_UNIQ` → **COPYING**, continuation `= P[e : e+K]` for the match.
  - `len(E_exact) > MIN_UNIQ` → the G-gram is ambiguous; fire only on the **agreed** continuation
    prefix (the longest run of prompt tokens on which *all* candidates agree), capped at `K`. If they
    disagree at the very first token → gate closed.
- `K = min(VLLM_S4_K_SCOPED, num_speculative_tokens, prompt_end − e, agreed_run)`.

`G` is the **confidence** knob (bigger G ⇒ fewer false fires, later engagement); `K_scoped` is the
**reach** knob (how far we run a confirmed copy before re-verifying). The stateless re-evaluation each
step means a copy that ends (e.g. the one renamed identifier in "rewrite this file changing one
function name") is detected within one step: the needle stops matching, the gate closes, MTP resumes.

A per-request **copy cursor** is tracked *for accounting only* (run-length of consecutive COPYING
steps = a detected verbatim span); it is never on the correctness path.

---

## 5. Interaction with MTP-K2, and losslessness

- **Layered, not replacing.** MTP runs every step exactly as in prod (its GPU chain capped to K_mtp
  by `VLLM_MTP_DRAFT_CAP`). We take its per-request draft tensor, convert to a list, and **per request**
  overwrite row `i` with the scoped continuation **iff the gate is open for `i`**. Gate-closed rows are
  byte-for-byte MTP's draft. This is the "fall through to MTP untouched" contract.
- **When both could fire** (gate open *and* MTP produced a draft): the scoped continuation **wins** for
  that request. Rationale: inside a proven verbatim span the prompt continuation is the ground-truth
  predictor and subsumes MTP's 2 tokens; mixing them can only shorten the accepted run. (Alternative
  considered — validate scoped[0] against mtp[0] and concatenate — deferred; it adds a coupling with no
  clear win when the gate already requires an exact G-gram match.)
- **Losslessness is unconditional.** The draft is only a *proposal*; vLLM's standard rejection sampler
  verifies every drafted token against the target distribution. A mis-fired gate cannot corrupt output
  — it can only waste that step's speculative budget. So the gate is tuned purely for *acceptance
  probability*, never for correctness.
- **MTP recurrent state is safe.** MTP's next-step inputs come from the *accepted* (real) tokens, not
  from the draft we substituted, so overriding drafts never desyncs the MTP head.

### K-sizing — the one hard constraint

`num_speculative_tokens` sizes the scheduler's spec slots, the KV **lookahead reservation**
(`num_lookahead_tokens`), the cudagraph `uniform_decode_query_len = 1 + num_spec_tokens`, and the
`SpecDecodingStats` per-position array (whose `observe_draft` asserts `accepted ≤ num_spec_tokens`).
A scoped draft longer than `num_speculative_tokens` would overflow the KV lookahead → **correctness
bug**. Therefore `K_scoped` is hard-clamped to `num_speculative_tokens` in code, and the **required
launch shape** is:

```
--speculative-config '{"method":"mtp","num_speculative_tokens":16}'   # pipeline sized for K_scoped
VLLM_MTP_DRAFT_CAP=2       # MTP GPU chain stays K2 (prod dense-draft cost preserved)
VLLM_S4_SCOPED_DRAFTER=1   # enable the gate
VLLM_S4_K_SCOPED=16 VLLM_S4_G=12 VLLM_S4_MIN_UNIQ=1
```

The list-path draft (variable length per request) then flows through
`_get_draft_token_ids_cpu` **uncapped** by the `(max_num_reqs, num_spec_tokens)` CPU buffer (that
buffer is only used on the padded tensor path), and the scheduler trims each request's drafts only to
what the token budget allows (`scheduler.py:492`), not to `num_spec_tokens`.

#### K-sizing finding #1 — `max-num-batched-tokens` must clear the inflated mamba align block

Qwen3.8 runs the mamba/GDN layers in **`mamba_cache_mode="align"`**, where each in-flight decode
sequence occupies a query window of `1 + num_speculative_tokens` tokens (`mamba_attn.py:98-99,173`:
`max_query_len == 1 + self.num_spec_tokens`, decode-only), and the align-mode batch is padded to the
mamba `chunk_size` boundary. The scheduler's `max_num_batched_tokens` (**mnbt**) must be ≥ that padded
align block or the engine refuses to boot (`config/vllm.py:1996-2002` asserts `block_size ≤ mnbt`, and
the align path further requires the full `(1+K)`-wide decode batch to fit).

Because the block grows with `1 + K`, **raising K raises the minimum mnbt**:

```
align_block(K)  ≈  chunk_pad( max_num_seqs · (1 + K) + prefill_reserve )
required_mnbt(K) ≥ align_block(K)

measured (max_num_seqs=4, this box):
  K=2  (MTP baseline) : align_block ≈ 3488
  K=16 (S4 pipeline)  : align_block ≈ 3856     ← the boot-blocker at K=16
  → ON server booted with mnbt = 3968 (first headroom step above 3856)
```

So the K=2→K=16 jump inflates the required mnbt from ~3488 to ~3856; **3968** is the value that let the
ON server start. (The two calibration points above are empirical for this model/box; the operative rule
is `mnbt ≥ align_block(K)` with `align_block` scaling ~linearly in `1+K`.)

#### K-sizing finding #2 — ~5 GiB profiling underestimate at K=16 (root-caused + fixed 2026-08-17)

At `num_speculative_tokens=16` the KV-cache profiling pass **under-counts peak memory by ≈ 5 GiB**, so a
nominal `gpu-memory-utilization` that profiles fine then OOMs post-profiling. The ON server had to be
brought up with **`--gpu-memory-utilization 0.75`** + `--max-num-seqs 4` (down from 16) to absorb it.

**Root cause (proven by reading allocation sites, not guessed).** `GPUModelRunner.profile_run` runs a
prefill-shaped `_dummy_run` then `_dummy_sampler_run`. The spec branch of `_dummy_sampler_run`
(`gpu_model_runner.py:5984-6006`) exercises the rejection sampler with `draft_token_ids = [[0]]*num_reqs`
— **one** draft token per request (`logits = randn(2·num_reqs, vocab)`), and the prefill dummy computes
logits at only `num_reqs` positions. A *real* decode step with `K=16` instead verifies `(1+K)·num_reqs`
positions: `compute_logits` produces a `((1+K)·num_reqs, vocab)` fp32 tensor
(`gpu_model_runner.py:4221-4222`, `logits_indices` from `_calc_spec_decode_metadata`), and the rejection
sampler (`vllm/v1/sample/rejection_sampler.py`) holds several concurrent full-vocab **fp32** buffers over
the `K·num_reqs` target positions (`raw_target_logits`, its `.clone()`, the top-k/top-p sort scratch, the
`target_probs` softmax). None of that full-width verify peak is materialised during profiling, so it
lands lazily on the first real decode step — after the KV cache has already claimed the rest of VRAM — and
OOMs. The gap scales with `(K−1)·num_reqs` (verify width beyond the profiled K=1 baseline), which is
exactly why it is invisible at K=2, ≈5 GiB at K=16/seqs=16, and ~4× smaller at seqs=4. Ruled out by the
same audit: `_reserve_decode_workspace` (turboquant decode; `B·Hq·S·(D+1)`, no `(1+K)` factor, reserved
eagerly at weight-load), the mamba spec buffers (`state_indices_tensor_d (B,1+K)` / `decode_num_accepted`
— tiny int32, eager), and the drafter (`VLLM_MTP_DRAFT_CAP=2` keeps its chain at K2; buffers are
`max_num_batched_tokens`-sized and eager; `dummy_run` allocates no vocab logits).

**Fix (at source, staged 2026-08-17).** An analytical reserve subtracted from the KV budget *before*
`num_gpu_blocks` is derived — mirroring `_turboquant_prefill_workspace_reserve_bytes`:
`vllm/v1/core/spec_decode_workspace.py` (pure, torch-free formula) +
`kv_cache_utils._spec_decode_verify_workspace_reserve_bytes` (config wrapper) +
the subtract in `get_kv_cache_configs`. Formula:
`reserve = OVERSHOOT_MULT · (K−1)·max_num_seqs · vocab · 4`. No add-back in
`determine_available_memory` is needed (unlike the turboquant arena): the verify buffers are never
allocated during profiling, so there is no double-count. Env-gated (`VLLM_SPEC_RESERVE_VERIFY_WORKSPACE`,
default on when `speculative_config` present and `K>1`; `VLLM_SPEC_VERIFY_OVERSHOOT_MULT`, default 24).
Validated by `tests/v1/core/test_spec_decode_workspace.py` (pure-python, no engine): predicts **5.33 GiB**
at K16/seqs16, **1.33 GiB** at K16/seqs4 (=÷4), **0.36 GiB** at K2/seqs16 — matching all three measured
boot facts.

**Hardware verification (2026-08-17) — the reserve is necessary but NOT sufficient; the
binding OOM is upstream.** Booting `feat-s4-scoped-drafter` at K16/seqs16/util0.82 still OOM'd,
with **zero** reserve log lines. The traceback pins it: `determine_available_memory` ->
`profile_cudagraph_memory` -> `_init_minimal_kv_cache_for_profiling` -> `torch.zeros(868 MiB)` — the
**minimal KV cache for cudagraph-memory profiling** (`min_blocks = max_cudagraph_capture_size = 512`).
That is *inside* `determine_available_memory`, **upstream of `get_kv_cache_configs`**, so the KV-sizing
reserve (and its log) never execute. The env wiring is correct; the reserve is simply unreachable in
this failure mode. Crucially, `gpu_memory_utilization` does **not** cap the profiling-phase peak
(it only sizes the eventual KV budget) — so 0.82 vs 0.75 is not the lever; **`max_num_seqs`** is (it
shrinks the profile peak, the cudagraph decode batch, and the minimal KV), which is why seqs=4 booted.
Fixes staged this round: (1) an **always-on diagnostic** of the reserve decision in
`determine_available_memory` (prints before the OOM-prone step, so every boot is attributable);
(2) `gc.collect(); torch.accelerator.empty_cache()` before `_init_minimal_kv_cache_for_profiling`, which
returns profile_run's fragmented reserved-but-unallocated cache (~0.8 GiB here) to the driver so the
minimal-KV alloc has contiguous room. **Residual (gap):** the model + compiled graphs leave ~19.7 GiB
*live* per rank, so the empty_cache margin is thin and capture may still OOM at seqs=16. The guaranteed
boot path is to cut the profiling peak: cap `cudagraph_capture_sizes` / `max_cudagraph_capture_size` to
the max useful decode batch (`max_num_seqs*(1+K) = 272` -> ~288, vs the current 512 which captures token
counts never reached at decode and inflates the minimal KV to 868 MiB), or lower `max_num_seqs`. That is
a launch-config lever, verified via boot — not a KV-sizing change.

**Honest caveat (per [[Challenge Impossible Claims]]).** The *mechanistically* attributable verify
buffers are only ~1.3 GiB (`OVERSHOOT_MULT≈6`); the arithmetic ceiling of the verify logits is ~2 GiB at
this vocab/batch, so the observed ~5 GiB is **not** all spec-verify logits. The default `OVERSHOOT_MULT=24`
reserves the full observed envelope; the excess over the ~6 mechanistic floor covers the un-instrumented
residual that also scales with the `(1+K)`-wide decode batch — the **GDN/Mamba align-mode decode scan
workspace** and PyTorch caching-allocator **fragmentation**. Attributing that residual precisely needs an
instrumented boot (`torch.cuda.memory._record_memory_history` around the first decode step); once done,
`OVERSHOOT_MULT` can drop toward 6 and the residual be handled at its own source. Until then the default
lets a K=16 run boot at the prod util instead of crashing.

---

## 6. Per-position acceptance accounting (scoped vs MTP, measured separately)

The drafter records, per request, the last draft it emitted **tagged by source** (`scoped`|`mtp`). On
the following step it reconciles against **that step's `sampled_token_ids`**: the draft emitted at step
T−1 is verified by the target model at step T, and its accepted prefix arrives as step T's sampled
tokens (the rejection sampler emits `[accepted drafts…, bonus]`), so `n_accepted = ` the longest prefix
where `draft[i] == sampled_ids[i]`. Scoring against `sampled_token_ids` (not against a `new_len − old_len`
delta over `token_ids_cpu`) is deliberate — at draft-merge time `token_ids_cpu`/`num_tokens_no_spec`
have **not yet** been advanced by `_bookkeeping_sync`, so a length-delta reconcile reads a stale, step-
mismatched sequence (this is the same staleness that caused the §11 root-cause bug). Each observation
lands in per-source, per-position histograms:

```
S4 scoped-drafter: gate_fires=..., copy_spans=..., mean_span_len=...
  scoped: drafts=N_s draft_toks=... accepted=... mean_accept_len=...  per-pos=[...]
  mtp   : drafts=N_m draft_toks=... accepted=... mean_accept_len=...  per-pos=[...]
```

This is self-contained in the drafter (no scheduler/frontend metric changes), and lets the A/B read
the **scoped contribution in isolation** from MTP's, which the aggregate `SpecDecoding metrics` line
cannot. Logged every `VLLM_S4_LOG_EVERY` drafts (default 2000), and flushed on request eviction.

---

## 7. Known headwind — async scheduling off (honest risk, not hidden)

CPU-list drafts are incompatible with async scheduling's on-device draft scatter, so enabling S4
disables async scheduling (mirrors `VLLM_SUFFIX_OVERLAY`; branch added in `config/vllm.py`). On this
box that is ≈ **−40%** single-stream (measured, `layered-speculation-v3.md`). This penalty applies to
the **whole run**, including the generation-shaped spans — we cannot toggle async per-span.

Consequence, stated plainly: on a *mixed* workload (≈45% copy / ≈55% generation) S4 wins only if the
copy-span acceptance gains **outweigh the blanket −40%**. The v3 evidence says pure echo already wins
even async-off (suffix echo 121 tok/s > MTP echo 105), while pure generation loses hard (44-64 vs
84-86). So the sign of the net is **workload-dependent and must be measured** — hence the replay-shaped
bench in §8. The async-preserving fix (GPU-side scatter of the scoped draft, the `ngram_gpu`-style
merge from `layered-speculation-v3.md`) is the documented **next increment** and is where this feature
has to go to be a single-stream win on mixed work. See §10.

---

## 8. A/B protocol — replay-shaped, NOT generation-shaped

The overlay was (correctly) refuted by a **generation-shaped** bench. S4 must be adjudicated on its
**target regime**, so `tools/s4_replay_bench.py` builds requests whose expected output is dominated by
long verbatim context spans:

- **rewrite** — a source file in the prompt + "reproduce it verbatim changing only `foo`→`bar`"; the
  output is ~99% a copy of the prompt with a few edits (the canonical S4 case).
- **quote** — "repeat the following block exactly", pure copy (upper bound / echo analogue).
- **mixed** — half prose continuation, half verbatim quote (realistic agent turn; probes whether the
  gate protects the generation half).
- **generation** (control) — free-form prose, *no* copyable span; must confirm S4 is a **no-op-grade**
  regression here (this is the regime that killed the overlay; S4 must not repeat it).

Each workload is run **drafter-off** (`VLLM_S4_SCOPED_DRAFTER=0`, i.e. prod MTP-K2) then **on**, ≥3
reps, warm-up discarded (per `Benchmark Rigor`). Reported per workload: decode tok/s (median + spread),
mean accepted length, and — from the S4 accounting log — **scoped vs MTP per-position acceptance**. The
bench prints a table and the exact deltas.

Pass criteria (first cut):
- `rewrite`, `quote`: **on ≥ off** on tok/s (target: clear win, since acceptance→~1.0 on the copy).
- `generation`: **on ≥ 0.97 × off** (no meaningful regression — the gate must stay shut).
- `mixed`: report the sign; this is the real adjudication number.

If `generation` regresses > 3%, the gate is leaking → tighten `VLLM_S4_G` / `VLLM_S4_MIN_UNIQ` before
any verdict. If `rewrite` fails to win, flag **NEGATIVE — needs Fable re-adjudication** (the async-off
headwind is beating the copy gain and the GPU-merge increment (§10) is required).

---

## 9. Config surface (all env, default off)

| env | default | meaning |
|---|---|---|
| `VLLM_S4_SCOPED_DRAFTER` | `0` | master switch (0 = strict no-op) |
| `VLLM_S4_G` | `12` | gate window: last-G tokens must match a prompt G-gram |
| `VLLM_S4_K_SCOPED` | `16` | max scoped draft length (clamped to `num_speculative_tokens`) |
| `VLLM_S4_MIN_UNIQ` | `1` | max prompt occurrences of the G-gram for the gate to fire (1 = strictly unique) |
| `VLLM_S4_LOG_EVERY` | `2000` | drafts between accounting log lines (0 = silent) |
| `VLLM_MTP_DRAFT_CAP` | `0` | (reused) cap MTP's learned chain; set `=2` when running S4 with `num_speculative_tokens>2` |

Required companions when enabling: `num_speculative_tokens = VLLM_S4_K_SCOPED`, `VLLM_MTP_DRAFT_CAP=2`.

---

## 10. Gap list / next increments (precise)

1. **Async-off headwind (dominant).** First cut is CPU-list-draft → async scheduling off ≈ −40%.
   Removing it requires scattering the scoped draft **GPU-side** (pad each gate-open row into the
   fixed-width `(num_reqs, num_spec_tokens)` draft tensor with `-1` fill, write via the existing async
   D2H `_num_valid_draft_tokens` path that `ngram_gpu` already uses). That is the `layered-speculation
   -v3.md` merge, unfinished. **This is the increment that turns S4 from "maybe" to "win" on mixed
   work.** Flagged **NEGATIVE — needs Fable re-adjudication** if the §8 `rewrite`/`mixed` A/B does not
   clear the bar without it.
2. **Prompt-only haystack.** Re-emission of *earlier generated* output (not in the prompt) is not
   covered. Extending the index to the generated suffix is a knob (`VLLM_S4_INCLUDE_GEN`), deferred —
   it needs incremental index maintenance as the response grows (append-only into the sorted array, or
   a small secondary dict over recent generation).
3. **Numba build.** Index build is vectorised numpy; a numba kernel (like `ngram_proposer`) would drop
   the once-per-request build further at very large prompts. Not on the per-step hot path, so low
   priority.
4. **Agreement continuation for `MIN_UNIQ>1`** is implemented conservatively (common-prefix across
   candidates). A frequency-weighted variant (Arctic-style `min_token_prob`) is possible but reintroduces
   the statistical-guess failure mode the gate exists to avoid — intentionally not done.
5. **cudagraph capture size.** Prod captures size `[4]`; running `num_speculative_tokens=16` changes
   `uniform_decode_query_len` to 17 and needs a matching `cudagraph_capture_sizes` review before a soak
   (staged-code caveat — verify at first engine bring-up).

---

## 11. A/B result, root cause & fix (2026-08-17)

### 11.1 The measured failure (first A/B, on hardware, PIECEWISE both sides)

`num_speculative_tokens=16`, `VLLM_S4_G=12`, `VLLM_S4_K_SCOPED=16`, `VLLM_S4_MIN_UNIQ=1`.
Per-position numbers are vLLM's **own** `vllm:spec_decode_num_accepted_tokens_per_pos` (Prometheus),
scraped by the bench — i.e. the authoritative rejection-sampler acceptance, not the drafter's internal
accounting.

| workload | OFF tok/s | OFF per_pos[0,1] | ON tok/s | ON per_pos |
|---|---|---|---|---|
| quote | 364.7 | 0.906, 0.818 | **91.0 (−75%)** | **0.165, 0.128, 0, 0, …** |
| mixed | 72.3 | 0.804, 0.656 | 31.3 | 0.220, 0.166, 0, … |
| generation (control) | 69.1 | 0.749, 0.529 | 65.9 (≈passthrough; −5% = async-off headwind) | 0.732, 0.498, 0, … |

The tell: gate-open proposals accepted **~0 even at position 0** (0.165 vs MTP's 0.906 on the *same*
copy workload), and positions **2–15 accepted exactly zero**. That is not a "weak draft" — the drafted
tokens were the **wrong** tokens.

### 11.2 Proven root cause — **Suspect (3): stale token stream** (manifesting as Suspect (1)'s off-by-one)

`ScopedReemissionDrafter.merge` runs inside `gpu_model_runner.propose_draft_token_ids`, which on the
**MTP/EAGLE path executes BEFORE `_bookkeeping_sync`** (call order in `execute_model`:
`propose_draft_token_ids` at ~L4381 → `_bookkeeping_sync` at ~L4440; `_bookkeeping_sync` is what writes
`token_ids_cpu[req, start:end] = sampled_ids` and advances `num_tokens_no_spec`, at ~L3598-3600).
So at merge time the committed sequence `token_ids_cpu[:num_tokens_no_spec]` is **missing the token(s)
sampled this step** — those arrive only via the `sampled_token_ids` argument (exactly what MTP is seeded
from, via `next_token_ids`).

The old gate hashed the **committed** last-G window (`seq[n−g:n]`, `n = num_tokens_no_spec`), so in a
verbatim copy span its needle ended one token *before* this step's sample. It matched the prompt G-gram
ending at `e = (that position)` and drafted `prompt[e : e+K]` — whose **position 0 is the token just
sampled this step** (already emitted). The rejection sampler rejects it immediately, and since rejection
cascades, positions 1–15 never get a chance → the observed `[low, low, 0, 0, …]` shape (the residual
0.165/0.128 at 0–1 is the *gate-closed* steps where MTP-K2 still flows through; the gate-open steps
contribute pure zeros, dragging MTP's 0.906 down to 0.165).

Not Suspect (2) (tensor-layout/slot): a slot/order corruption would zero position 0 uniformly; the
surviving nonzero at 0–1 (= MTP passthrough on gate-closed steps) rules it out. **Proven, not assumed**,
by the unit test in §11.4.

### 11.3 The fix (in `vllm/v1/spec_decode/scoped_reemission.py`)

Splice this step's sampled tokens onto the committed tail **before** hashing the needle and computing the
continuation, so the scoped draft continues from the same point MTP does:

- New `_effective_tail(seq, n, sampled_ids)` returns the true last-G window = `(committed tail) ⧺
  (this step's sampled tokens)`, handling `len(sampled_ids) ≥ g` (needle wholly from sampled tokens).
- `_scoped_draft(st, seq, n, sampled_ids)` now uses `n_eff = n + len(sampled_ids)` for the length/
  `max_model_len` guards and hashes `_effective_tail`; the matched G-gram's continuation `prompt[e:e+K]`
  is therefore the token **after** this step's sample — aligned with MTP. Handles multi-token accept
  steps (`grew > 1`), so it is **not** a hard-coded `+1`.
- `_reconcile` rewritten to score the previous draft against **this step's `sampled_token_ids`** (the
  verification result), replacing the stale `new_len − old_len − 1` delta over `token_ids_cpu`.
- `merge` now normalises `sampled_token_ids` (strips `-1`) on both the tensor and list paths and passes
  the per-request sampled tokens into both reconcile and the gate. Dropped the now-unused
  `last_num_tokens` state.

Fail-safe path (`_scoped_gate_merge` in `gpu_model_runner`) is unchanged: any error → MTP drafts pass
through. The gate remains **lossless** (draft-only; the rejection sampler verifies).

### 11.4 Unit-test evidence — `tests/v1/spec_decode/test_scoped_reemission.py`

Pure-python (numpy only; stubs `vllm.config`/`vllm.logger`, loads the module by path — no torch/CUDA).
It reproduces the exact merge-time timing contract (committed sequence lags by this step's samples).

- `test_verbatim_copy_single_token_step` — **reproduced the bug first**: against the old code the draft
  was `prompt[m−1 : …]` with `got[0] == prompt[m−1]` (the just-sampled token); after the fix it equals
  the true continuation `prompt[m : m+K]`.
- `test_verbatim_copy_multi_token_step` — 3 tokens accepted this step; proves the fix advances by *all*
  sampled tokens (a `+1`-only fix fails this).
- `test_needle_longer_than_g` — `len(sampled) ≥ g`, needle wholly from sampled tokens.
- `test_gate_closed_passes_mtp_through` / `test_no_sampled_token_returns_mtp` — gate stays shut off a
  non-copy tail and on empty (partial-prefill) rows.

Result: **5/5 pass** post-fix; the first two **fail** on the pre-fix code (bug reproduced, then fixed).
Run: `python3 tests/v1/spec_decode/test_scoped_reemission.py`.

### 11.5 Corrected re-A/B protocol

Boot the ON server with the sizing adjustments that let it start (see §5 findings #1/#2): **util 0.75,
max-num-seqs 4, mnbt 3968, max-len 65536**, plus the required spec shape:

```
--speculative-config '{"method":"mtp","num_speculative_tokens":16}'
--gpu-memory-utilization 0.75 --max-num-seqs 4
--max-num-batched-tokens 3968 --max-model-len 65536
# env: VLLM_MTP_DRAFT_CAP=2 VLLM_S4_SCOPED_DRAFTER=1 (ON) / 0 (OFF)
#      VLLM_S4_K_SCOPED=16 VLLM_S4_G=12 VLLM_S4_MIN_UNIQ=1
```

Same bench commands as the first A/B (endpoint `:8001`, model `qwen-local`), OFF then ON, then compare:

```
python tools/s4_replay_bench.py --base-url http://127.0.0.1:8001 --model qwen-local \
    --label off --reps 3 --warmup 1 --out /tmp/s4_off.json      # VLLM_S4_SCOPED_DRAFTER=0
python tools/s4_replay_bench.py --base-url http://127.0.0.1:8001 --model qwen-local \
    --label on  --reps 3 --warmup 1 --out /tmp/s4_on.json       # VLLM_S4_SCOPED_DRAFTER=1
python tools/s4_replay_bench.py --compare /tmp/s4_off.json /tmp/s4_on.json
```

Pass/leak criteria unchanged from §8. (Pre-flight gate: the §11.4 unit test must be green before a
re-run — if `quote` per_pos[0] is still ~0.16 the gate is still misaligned.)

### 11.6 Re-A/B result — measured on hardware (2026-08-17)

Re-ran the protocol above at the boot-feasible shape (**max_num_seqs=4 / max-len 65536**, PIECEWISE both
sides). The fix is confirmed:

| workload | result |
|---|---|
| quote | OFF 365 → ON **811.5 tok/s (+122%)**, mean **7.44** accepted/step; gate-open per_pos[0] recovered from the collapsed 0.165 back toward MTP's ~0.9, extending into positions 2–15 as the copy span is drafted K-deep |
| mixed | ON **101.5 tok/s — no-harm** (copy spans fire; prose stays on MTP) |
| generation (control) | **unchanged** (≈passthrough; ~−5% async-off headwind only) |

This is exactly the recovery §11.3's fix predicted: gate-open acceptance climbs from ~0 back to MTP-grade
at position 0 and extends K-deep on the copy span, confirming the effective-tail splice (`bbc09d3`).

**Still open — prod-shape re-proof.** The numbers above are at `max_num_seqs=4`; a re-proof at the full
production concurrency/context envelope is still pending, as is the GPU-side merge increment (§10.1) that
removes the async-off headwind on `mixed`.

