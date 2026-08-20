# EXP-039 v3 — GPU-side draft merge for the S4 scoped drafter

**Branch:** `feat-s4-scoped-drafter` (worktree `~/Desktop/.ftree-s4drafter`)
**Status:** staged code, no engine run yet (this document + implementation + tests).
**Env switch:** `VLLM_S4_GPU_MERGE=1` (default `0` = v2 CPU-list `merge`, unchanged).
**Companion of:** `docs/exp039-scoped-drafter-design.md` (the gate/FSM/losslessness design; §7
and §10.1 there flag *this* increment as the dominant open gap).

---

## 0. TL;DR

The S4 gate is correct and wins on copy work (v2: quote 365→811.5 tok/s), but the v2 merge returns a
**Python list**, which is incompatible with async scheduling's on-device draft scatter, so enabling S4
forces `async_scheduling=False` for the whole run — **≈ −40 % single-stream on this box**. That tax is
what keeps S4 from being a single-stream win on *mixed* work and blocks prod eligibility.

v3 keeps the proposer output a **GPU tensor**: the CPU rolling-hash gate is unchanged (microseconds),
but the *merge* is tensorized — gate-open rows are scattered into a resident device buffer and selected
into the MTP draft tensor with one on-device `torch.where`. No draft crosses the PCIe bus as a list, so
async scheduling stays **on**. `config/vllm.py` no longer disables async when `VLLM_S4_GPU_MERGE=1`.

This is the exact fix the GPU-side-drafting research report prescribes (tensor-only drafts + small async
metadata copies), adapted to a **hybrid** shape: CPU gate, GPU merge. See §5 for what was applied vs
overridden, and §7 for the honest async-recovery caveat (removes the *structural* disable; the *magnitude*
of recovery is an A/B question, §8).

---

## 1. Why the v2 list breaks async (the mechanism, from the code)

Async scheduling overlaps step *N*'s GPU work with step *N+1*'s CPU scheduling by scattering the
**previous step's sampled tokens and draft tokens into `input_ids` on-device**, without a D2H/H2D round
trip. Three code facts make a *tensor* draft mandatory for that path:

1. **On-device draft scatter** (`gpu_model_runner._prepare_input_ids`, ~L1802-1840):
   `assert isinstance(self._draft_token_ids, torch.Tensor)`, then
   `self.input_ids.gpu.scatter_(..., src=draft_token_ids.flatten()[prev_draft_token_indices])`.
   A Python list has no device tensor to scatter → the assert fails / the path can't run.
2. **The scatter indexes at full pipeline width.** `start = prev_index * self.num_spec_tokens`
   (~L1751). Each draft row is assumed `num_spec_tokens` wide. So the merged tensor must be
   `[num_reqs, num_spec_tokens]` — *not* the narrower width `VLLM_MTP_DRAFT_CAP` leaves behind (see §4).
3. **Async D2H of drafts is tensor-gated.** `_copy_draft_token_ids_to_cpu` (~L4606):
   `if not torch.is_tensor(draft_token_ids): return`. A list silently skips the copy the
   structured-output / penalties path relies on.

The v2 `merge` returns `list[list[int]]`, tripping all three. So `config/vllm.py` (~L840, ~L890)
disables async whenever `VLLM_S4_SCOPED_DRAFTER=1` on the eagle/MTP path — mirroring `VLLM_SUFFIX_OVERLAY`.
That is the −40 % tax (`docs/design/layered-speculation-v3.md`).

---

## 2. Where drafts are built vs where they are merged

**Build (stays CPU — deliberately).** The gate is a fixed-window rolling-hash lookup over the request's
**prompt** tokens (`_PromptIndex`, `searchsorted` + exact G-token verify). Cost is single-digit
microseconds per request per step, four orders of magnitude under a ~15 ms GPU step (design doc §3). The
prompt (haystack) and the committed tail (needle prefix) are **already on CPU** in
`input_batch.token_ids_cpu`; only *this step's sampled tokens* (the needle suffix, ≤ `K+1` ints/req)
are GPU-resident pre-bookkeeping. Porting this uniqueness/prompt-scoped gate to GPU would be a different
project (essentially `ngram_gpu` plus a uniqueness constraint) and is explicitly **not** what v3 does —
see §5, override (b).

**Merge (must be tensorized).** The output of `merge` is what breaks async, so *that* is what v3 moves
on-device. The gate produces, per gate-open request, up to `K_scoped` continuation tokens (a short CPU
list). v3 scatters those into a **pre-allocated pinned staging buffer**, H2D-copies the tiny
`[num_reqs, num_spec_tokens]` slice to a **resident device buffer**, and selects it into the MTP tensor
with a single on-device `torch.where`.

---

## 3. The tensor path (`ScopedReemissionDrafter.merge_gpu`)

Resident buffers, lazily allocated on the first call for the live `(device, num_spec_tokens)` and grown
to `max_num_seqs` (`_ensure_gpu_buffers`):

| buffer | shape | where | role |
|---|---|---|---|
| `_scoped_pin` | `[R, W]` int32 | pinned CPU | staging for scoped continuations (`-1` = no token) |
| `_scoped_np` | `[R, W]` int32 | numpy view of `_scoped_pin` | zero-alloc per-row scatter (shared storage) |
| `_gate_pin` | `[R]` bool | pinned CPU | gate-open mask |
| `_scoped_gpu` | `[R, W]` int32 | device | H2D destination for the scoped buffer |
| `_gate_gpu` | `[R]` bool | device | H2D destination for the mask |
| `_mtp_wide` | `[R, W]` int32 | device | width-normalized MTP tensor (see §4) |

`W = num_speculative_tokens` (the pipeline width, e.g. 16); `R = max(max_num_seqs, live_batch)`.

Per step (`n` = live requests):

1. **Reset** the active slice of the staging views: `_scoped_np[:n] = -1`, `_gate_np[:n] = False`.
2. **Gate loop** (CPU, the shared `_decide_scoped` — identical to v2 so the A/B gates the same):
   for each request run reconcile + the FSM gate. On a gate-open row, write the continuation into
   `_scoped_np[i, :k]` (left-aligned, rest stays `-1`) and set `_gate_np[i] = True`.
3. **H2D** the tiny active slices non-blocking: `_scoped_gpu[:n] ← _scoped_pin[:n]`,
   `_gate_gpu[:n] ← _gate_pin[:n]`.
4. **Width-normalize** MTP into `_mtp_wide[:n]` (`fill_(-1)` then copy the first `K_mtp` columns), §4.
5. **On-device select:** `merged = where(_gate_gpu[:n, None], _scoped_gpu[:n], _mtp_wide[:n])` →
   `[n, W]` int32 on device. Gate-closed rows are the MTP draft byte-for-byte (width-normalized);
   gate-open rows are the scoped continuation with `-1` padding beyond `k`.

`torch.where` is the masked-select realization of "scatter the gate-open rows into the proposer tensor" —
equivalent to scattering `_scoped_gpu` rows into a clone of `_mtp_wide` at the gate-open indices, but
branch-free and one kernel. The `-1` padding beyond a scoped/MTP draft's real length is correct and
necessary: the async scatter reads only the first `draft_len` (scheduler-scheduled) columns of each row,
and any `-1` that is read is clamped to 0 before the embedding lookup (`_prepare_input_ids` ~L1834) and
rejected by the target — lossless, exactly as the design doc guarantees.

**Data volume across the bus, per step:** D2H of the sampled-token tensor `[n, K+1]` (needle suffix) and
H2D of `[n, W]` int32 + `[n]` bool. At `n=16, W=16` that is ~1 KiB each way — a small **fixed-width**
copy, *not* the large variable-length list transfer the research report flags (§5, applied-3).

---

## 4. Width normalization — the second thing that made S4 async-incompatible

`VLLM_MTP_DRAFT_CAP=2` (required when launching `num_speculative_tokens=16` so MTP keeps its cheap K2
chain) caps the **proposer's** `num_speculative_tokens` (`llm_base_proposer.py:80-82`), so
`EagleProposer.propose` returns `draft_token_ids.view(-1, 2)` — a **`[n, 2]`** tensor, narrower than the
pipeline width 16. The async scatter indexes rows at `prev_index * num_spec_tokens` (= ×16), so the raw
narrow tensor would mis-index. This is *independent* of S4 — it means the capped MTP tensor was **never**
async-safe on its own; it only ever flowed through the v2 CPU-list path, which reshapes everything to
lists. `merge_gpu` fixes it by placing the `K_mtp` real columns into a width-`num_spec_tokens` row,
`-1`-padded. So v3 makes both the scoped draft *and* the capped MTP draft async-safe in one merge.

---

## 5. Calibration against the research report (applied / overridden)

Report: `research-reports/20260817-170947-…-state-of-the-art-in-gpu-side-speculative-decoding-draft-cons.md`.

**Applied:**
1. *"CPU-list drafts force async off; tensor-only drafts keep async on"* (the report's central pitfall
   + the `ngram_proposer_gpu` remedy). v3's whole point: the **merge output is a device tensor**, so the
   on-device scatter and the tensor-gated async D2H both keep working. This is the necessary-and-—
   structurally—sufficient condition to stop disabling async.
2. *`ngram_proposer_gpu` returns `draft_tokens: [batch, k]` on GPU + a `[batch]` valid-count.* v3's
   merged tensor is the same shape contract (`[num_reqs, num_spec_tokens]` int32, `-1` = no token), so it
   drops into the exact downstream the runner already has (`draft_token_ids_cpu` is `[max_num_reqs,
   num_spec_tokens]`; the scatter expects that width).
3. *`copy_num_valid_draft_tokens` does a small **async D2H of per-request metadata** on a side stream +
   event, rather than a big blocking transfer.* v3 mirrors the *shape* of this: the only cross-bus
   traffic is tiny fixed-width `[n, K+1]` / `[n, W]` copies, not a variable-length list. (v3 currently
   issues them inline rather than on a dedicated event-gated stream — see §7 / gap 1 for the overlap
   refinement.)
4. *Proposers integrate at a stable `propose` boundary.* v3 layers **after** `self.drafter.propose(...)`
   at the existing merge point in `propose_draft_token_ids` (same integration point as the v2 merge and
   the suffix overlay), so no proposer-interface surgery.

**Overridden (with reasons):**
- (a) *Report: the GPU n-gram path builds drafts entirely on-device with `unfold`+`argmax`.* v3 keeps the
  **draft/gate computation on CPU**. Reason: the S4 gate is a *uniqueness*-gated lookup over the
  **prompt** with an exact-match verify and a `min_uniq` constraint — a CPU rolling-hash + `searchsorted`
  structure. The report's `unfold`/`argmax` finds the *first* suffix match anywhere in the sequence (no
  uniqueness, no prompt scoping); porting S4's gate to that shape changes the policy, not just the venue.
  The gate is microseconds and off the critical path, so moving it buys nothing; the *merge* was the
  async-killer, and that is what v3 moves. This is the task's explicit prescription ("CPU hash index stays
  CPU — it's microseconds; the MERGE must be tensorized").
- (b) *Report: maintain GPU token state incrementally (`update_ngram_gpu_tensors_incremental`).* v3 does
  **not** allocate a resident `[max_num_reqs, max_model_len]` GPU token tensor. Reason: that tensor is
  built only in the `use_ngram_gpu` branch (`gpu_model_runner.py:553-561`), not the eagle/MTP branch S4
  rides; adding a ~68–268 MiB device tensor purely to host a needle the gate reads from CPU is wasteful
  when the committed tail is already on CPU and only `[n, K+1]` sampled tokens need to come across.

---

## 6. Config change (fail-fast preserved for the v2 path)

`config/vllm.py` now conditions the async-disable on `VLLM_S4_GPU_MERGE`:

- **Explicit `--async-scheduling`** + `VLLM_S4_SCOPED_DRAFTER=1` + `VLLM_S4_GPU_MERGE≠1` → **raise**
  (unchanged fail-fast: the v2 list would silently break the scatter). With `VLLM_S4_GPU_MERGE=1` the
  raise is **skipped** — v3 is async-safe.
- **Auto (`async_scheduling is None`)** + same v2 condition → disable async (unchanged). With
  `VLLM_S4_GPU_MERGE=1` the branch is skipped and async falls through to **enabled**.

So the only combination that still forces async off is the *v2* path (GPU merge off). The runtime
dispatcher (`gpu_model_runner._scoped_gate_merge`) is belt-and-suspenders: GPU-merge + tensor →
`merge_gpu`; async + tensor + GPU-merge-off → strict no-op (returns the MTP tensor unchanged); else v2
`merge`. On any exception it returns the drafts unchanged **as a tensor** when they came in as a tensor,
so an error in the scoped path can never break async.

---

## 7. Async-compat argument — honest scope

**What v3 removes:** the *structural* async incompatibility. After the merge, `_draft_token_ids` is a
`[num_reqs, num_spec_tokens]` device tensor, so (1) the on-device draft scatter runs, (2) the tensor-gated
async D2H runs, (3) the width matches the scatter's `×num_spec_tokens` indexing. Async scheduling can
therefore be left **on** — which the config now does.

**What v3 does *not* claim:** that async fully recovers the entire −40 % automatically. The gate reads
this step's sampled tokens, which are GPU-resident pre-bookkeeping, so there is still a **small D2H**
(`[n, K+1]`) each step before the CPU gate can run, and the scoped H2D must land before the next step's
on-device scatter reads the draft tensor. These are tiny fixed-width copies (§3) — the report's warning is
specifically about *large variable-length list* transfers — but they are not literally zero, and whether
the pipeline hides them fully is an **empirical** question. Per the frontier rule (no impossible-verdicts):
v3 is the *closest increment* that makes async legal for S4; the *magnitude* of recovery is the number the
§8 A/B produces. The dominant, previously-unavoidable cost (async OFF for the whole run) is gone; the
residual is bounded and measurable.

---

## 8. Window plan — A/B (async-on GPU merge vs v2 async-off)

Adjudicate at the **proven S4 shape** (design doc §11.5): `num_speculative_tokens=16`,
`VLLM_MTP_DRAFT_CAP=2`, `VLLM_S4_G=12 VLLM_S4_K_SCOPED=16 VLLM_S4_MIN_UNIQ=1`, boot-feasible sizing
`--gpu-memory-utilization 0.75 --max-num-seqs 4 --max-num-batched-tokens 3968 --max-model-len 65536`.

**Arm A — v2 baseline (proven, async OFF, CPU list):**
```
# env: VLLM_S4_SCOPED_DRAFTER=1 VLLM_S4_GPU_MERGE=0 VLLM_MTP_DRAFT_CAP=2
#      VLLM_S4_K_SCOPED=16 VLLM_S4_G=12 VLLM_S4_MIN_UNIQ=1
--speculative-config '{"method":"mtp","num_speculative_tokens":16}'
--gpu-memory-utilization 0.75 --max-num-seqs 4 --max-num-batched-tokens 3968 --max-model-len 65536
# expect log: "Async scheduling is disabled."   (== the −40% tax; matches v2's 811.5/101.5)
```

**Arm B — v3 target (GPU merge, async AUTO-ON):**
```
# env: VLLM_S4_SCOPED_DRAFTER=1 VLLM_S4_GPU_MERGE=1 VLLM_MTP_DRAFT_CAP=2
#      VLLM_S4_K_SCOPED=16 VLLM_S4_G=12 VLLM_S4_MIN_UNIQ=1
--speculative-config '{"method":"mtp","num_speculative_tokens":16}'
--gpu-memory-utilization 0.75 --max-num-seqs 4 --max-num-batched-tokens 3968 --max-model-len 65536
# expect log: "Async scheduling is enabled."  + "S4 scoped-reemission drafter ENABLED: ... merge=gpu ..."
# (do NOT pass --async-scheduling explicitly; let config auto-enable so a regression is visible)
```

**Arm C — isolation (v3 GPU merge, async forced OFF):** add `--no-async-scheduling` to Arm B. Separates
the merge overhead from the async gain: `B − C` = the async recovery; `C ≈ A` would confirm the GPU merge
adds no per-step cost of its own.

Bench each arm OFF-then-ON is *not* needed here (S4 is ON in every arm); instead run the replay bench once
per arm and compare arms. Reuse `tools/s4_replay_bench.py`:
```
python tools/s4_replay_bench.py --base-url http://127.0.0.1:8001 --model qwen-local \
    --label A_v2_asyncoff --reps 3 --warmup 1 --out /tmp/s4_A.json      # Arm A server
python tools/s4_replay_bench.py --base-url http://127.0.0.1:8001 --model qwen-local \
    --label B_v3_asyncon --reps 3 --warmup 1 --out /tmp/s4_B.json       # Arm B server
python tools/s4_replay_bench.py --base-url http://127.0.0.1:8001 --model qwen-local \
    --label C_v3_asyncoff --reps 3 --warmup 1 --out /tmp/s4_C.json      # Arm C server
python tools/s4_replay_bench.py --compare /tmp/s4_A.json /tmp/s4_B.json
```

**Read (pass criteria):**
- `generation` (control): Arm B ≥ Arm A (async-on should *recover* the ~−5 % headwind, not regress).
- `mixed`: **Arm B > Arm A** is the prod-eligibility number — the async recovery is exactly what should
  flip `mixed` from "no-harm" (v2, 101.5) to a positive single-stream win.
- `quote`/`rewrite`: Arm B ≥ Arm A (copy gains already win async-off; must not regress).
- `B − C` > 0 and `C ≈ A` confirms the recovery is the async toggle, not noise.

If `mixed` (Arm B) does **not** beat Arm A, flag **NEGATIVE — needs Fable re-adjudication**: the residual
per-step D2H/H2D (§7) is eating the async gain, and gap 1 (event-gated overlap) is required before a verdict.

---

## 9. Gap list / next increments (precise)

1. **Overlap the needle D2H with MTP's forward.** Today `merge_gpu` reads the sampled tokens inline
   (`_sampled_rows` → `.tolist()`), a small but synchronous D2H after `self.drafter.propose`. To hide it,
   launch the `[n, K+1]` D2H on a dedicated stream **before** the MTP proposer forward (right where
   `sampled_token_ids` becomes available in the async pre-bookkeeping branch, ~L4396) and gate `merge_gpu`
   on its event — the `copy_num_valid_draft_tokens` pattern. Deferred: needs a small hook in
   `execute_model` and only matters if §8 Arm C shows the inline copy costs measurable throughput.
2. **Structured-output / penalties path `-1` leak.** In the rare async branch where
   `_copy_draft_token_ids_to_cpu` *does* D2H (structured output or penalties), the fixed-width tensor's
   trailing `-1`s reach `scheduler.update_draft_token_ids` un-stripped (the base MTP tensor never has
   `-1`, so the downstream never learned to strip). On-device the `-1` is clamped and rejected (safe), but
   `request.spec_token_ids` could carry `-1` entries. Verify at first boot with structured output + S4;
   if it matters, strip per-row trailing `-1` in `_get_draft_token_ids_cpu` (universally correct for a
   list-of-drafts view).
3. **cudagraph capture at K=16.** `num_speculative_tokens=16` sets `uniform_decode_query_len=17`; the
   async path adds a `disable_cascade_attn` interaction (`config/vllm.py` ~L942). Confirm the capture
   sizes cover `max_num_seqs*(1+K)` at first bring-up (design doc §5 finding #2 / §10.5).
4. **Per-source MTP accounting under GPU merge.** `merge_gpu` sets `last_draft=None` on gate-closed
   (MTP) rows to avoid a diagnostic-only D2H of the MTP tensor, so the drafter's *internal* per-position
   histogram covers scoped rows only in GPU mode. The **authoritative** acceptance (vLLM's
   `vllm:spec_decode_num_accepted_tokens_per_pos` Prometheus metric the bench scrapes) is unaffected.
5. **Prompt-only haystack** (unchanged from v2 §10.2): re-emission of earlier *generated* output is still
   out of scope.

---

## 10. Files touched

- `vllm/v1/spec_decode/scoped_reemission.py` — `merge_gpu`, `_ensure_gpu_buffers`, `_decide_scoped`
  (shared gate), `_sampled_rows` (shared), `gpu_merge` flag + resident buffers. Torch imported lazily so
  the module stays numpy-only importable. **v2 `merge` behavior unchanged.**
- `vllm/v1/worker/gpu_model_runner.py` — `_scoped_gate_merge` dispatches CPU-list / GPU / no-op and is
  tensor-safe on error; the call-site guard no longer force-skips S4 under async.
- `vllm/config/vllm.py` — async-disable (explicit raise + auto) conditioned on `VLLM_S4_GPU_MERGE≠1`.
- `tests/v1/spec_decode/test_scoped_reemission_gpu_merge.py` — real-tensor tests (CUDA or CPU) of the
  scatter/merge, width-normalization, mixed batch, buffer growth, and **v2≡v3 gate equivalence**.

## 11. Config surface (v3 additions)

| env | default | meaning |
|---|---|---|
| `VLLM_S4_GPU_MERGE` | `0` | `1` = tensorized merge (`merge_gpu`), async-safe; `0` = v2 CPU-list `merge` |

All v2 knobs (`VLLM_S4_SCOPED_DRAFTER`, `_G`, `_K_SCOPED`, `_MIN_UNIQ`, `_LOG_EVERY`, `VLLM_MTP_DRAFT_CAP`)
are unchanged (design doc §9). `VLLM_S4_GPU_MERGE=1` requires the padded drafter batch (it needs the MTP
tensor input); with `disable_padded_drafter_batch=True` the drafter is not constructed (unchanged guard).

---

## 10. Window RESULT — 2026-08-19 ~20:54 (NEGATIVE, bug located)

Ran arms Z/A/B/C via `tools/s4_v3_window.sh` at the §8 shape (65536/seqs4/util.75,
num_spec_tokens=16, MTP_DRAFT_CAP=2). **No valid A/B comparison produced** — three of
four arms crashed on first spec-decode traffic:

| arm | config | outcome |
|---|---|---|
| Z | S4 off, plain MTP K=16 | **HTTP 500** on all traffic — the K=16 minimal shape is fragile at HEAD *independent of S4* (confirms backlog #19 "K=16 boot fragility") |
| A | v2 CPU-list, async off | HTTP 500 then engine died — same K=16 fragility |
| B | **v3 GPU-merge, async ON** (the target) | **crash**: `RuntimeError: tensor a (16) must match tensor b (2) at dim 1` |
| C | v3 GPU-merge, async OFF | **clean** — rewrite 123.5 / quote 150.8 / mixed 106.5 / generation 52.75 tok/s (small shape, 3-rep) |

**Root cause of the arm-B crash (async-on path):** `_copy_draft_token_ids_to_cpu`
(`gpu_model_runner.py:4618`) copies the merged draft into `draft_token_ids_cpu`, which
is allocated at width `num_spec_tokens` (=16, L897). The v3 GPU-merge emits a draft of
width `MTP_DRAFT_CAP` (=2), so the copy shape-mismatches (16≠2). The async-off path (C)
never takes this copy, which is why C alone survived. This is precisely the §9 gap-1
width-reconciliation increment: the async path must expand the capped-MTP draft to the
full K=16 pipeline width (pad + valid-count) before the on-device scatter/CPU copy.

**Verdict: NEGATIVE for prod-eligibility.** v3's whole purpose — async ON to recover the
−40% tax — crashes at the merge→async-copy width seam; and the K=16 test shape is itself
unstable (Z/A). C proves the merge math is sound async-OFF, but that's the config v2
already had. **Not shippable. Next: fix the width reconciliation at gpu_model_runner.py:4618
(gap-1), and separately stabilize MTP K=16 at the minimal shape, then re-run the window.**
Each re-run costs a ~30-min engine window (prod → DeepSeek), so batch the two fixes first.

---

## 11. Window r2 RESULT — 2026-08-19 ~21:40 (crash FIXED, verdict DECISIVE NEGATIVE)

After the `_hash_tail` numpy-2.x fix (commit ce6727a), arm B (v3 GPU-merge, async-ON)
**no longer crashes** and produces full bench data. Clean C-vs-B isolation (same merge
code, async toggle only), 3-rep + warmup at the §8 shape:

| workload | C v3 async-OFF | B v3 async-ON | delta |
|---|---|---|---|
| rewrite | 122.5 | 67.6 | **−44.9%** |
| quote | 150.1 | 63.3 | **−57.8%** |
| mixed | 106.3 | 60.9 | **−42.7%** |
| generation | 52.8 | 53.3 | +0.9% (no-op) |

**Smoking gun — per-position acceptance:**
- C (async-off): `[0.896, 0.757, 0.208, 0.208, 0.202, …]`, mean_accept 5.2–6.4 — S4 scoped
  positions 2..15 ARE accepted (the copy speedup is real).
- B (async-on): `[0.955, 0.917, 0.000, 0.000, …]`, mean_accept 2.6–2.9 — positions 2..15 get
  **ZERO acceptance**. Only the 2 capped-MTP tokens land; the S4 scoped continuation never
  reaches/passes verify under async scheduling.

**Verdict: NEGATIVE, decisive.** Turning async ON does not recover the −40% tax — it makes
copy work 42–58% *worse*, because the on-device async draft path drops the scoped positions
(they verify as stale/empty). merge_gpu's math is correct (arm C proves it); the failure is
purely the async scatter/verify timing = the §9 gap-1 (scoped H2D must be event-gated to land
before the next step's `prev_index * num_spec_tokens` scatter, and the inline sampled-token D2H
must overlap MTP's forward). Shipping v3-async-on would REGRESS copy 42–58%.

**What shipped anyway (real wins):** the crash was a genuine bug — numpy≥2.0 `np.uint64(-1)`
raising broke the rolling hash on any -1-padded async sampled token, taking down *both* merge
paths. Fixed at source + 2 regression tests (ce6727a). The 811 copy-burst path stays **v2
async-OFF** (already banked, `serve-PROFILE-copyburst.sh`) — unaffected.

**Next increment (hard, not a tweak):** implement §9 gap-1 (event-gated scoped-H2D-before-scatter
+ overlap the sampled-token D2H). Only then re-window. Arms A (v2 cpu-list) and Z (S4-off+cap under
async) still crash on separate/unsupported paths — out of scope for the C-vs-B verdict.
