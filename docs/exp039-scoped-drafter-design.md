# EXP-039 (S4): Suffix-scoped drafter for verbatim re-emission

**Branch:** `feat-s4-scoped-drafter` (worktree `~/Desktop/.ftree-s4drafter`, base `frontier-pastnative-20260816`)
**Status:** first staged implementation — code-complete, env-gated default-off, **no engine runs yet**.
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

---

## 6. Per-position acceptance accounting (scoped vs MTP, measured separately)

The drafter records, per request, the last draft it emitted **tagged by source** (`scoped`|`mtp`) and
the sequence length at emit time. On the following step it reconciles: the sequence grew by
`1 (bonus) + n_accepted_draft`, so `n_accepted_draft = new_len − old_len − 1`, clamped to
`[0, len(draft)]` and confirmed by comparing the draft prefix to the newly-appended real tokens. Each
observation lands in per-source, per-position histograms:

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
```
