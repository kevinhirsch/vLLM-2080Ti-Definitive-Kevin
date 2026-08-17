# EXP-038 Stage-4 — scheduler-level FORK API (`fork_from_handle`)

Status: **STAGED CODE ONLY**, env-gated (`VLLM_TQ_GDN_SNAPSHOT`), default-inert.
Branch: `feat-gdn-snapshot`. Never run against the production `:8001` serve.

This note records the insertion-point analysis, the chosen path and why it is the
least-invasive one, the API shape, the refcount / free semantics, and the failure
modes + honest gap list (frontier rule: no "impossible" verdicts, no rubber-stamps).

---

## 0. What Stages 0-3 already give us (the substrate)

| Stage | Layer | Primitive (proven byte-exact on hardware) |
|------|-------|--------------------------------------------|
| 0 | worker (`GPUModelRunner`) | `snapshot_mamba_state` / `restore_mamba_state` / `release_mamba_snapshot` + park pool; byte-exact D2D park of a live GDN slot, both ranks. |
| 1 | scheduler (`EngineCore`) | `pin_request_kv_blocks` / `verify_pinned_blocks` / `get_request_kv_block_ids(all_running)` / `unpin_kv_blocks`; blocks survive `free()` at `ref_cnt >= 1`. |
| 2 | public API | zero-kernel restore: resubmit identical `token_ids` -> `num_cached_tokens` 2080/2112 -> byte-exact continuation. |
| 3 | public API | fork smoke: one handle -> two co-scheduled children via resubmit; byte-exact 3-way; shared attn block; distinct mamba running slots. |

The load-bearing fact Stage-2/3 established: **a pinned prefix is still hash-cached
in the prefix cache**, so a request whose tokens match the pinned prefix cache-hits
the pinned blocks automatically — `find_longest_cache_hit` returns the pinned **attn**
full blocks (adopted by `block_pool.touch`, i.e. ref-count) *and* the pinned **mamba**
by-hash-cached GDN full block, which align-mode `preprocess_mamba` **copies out into a
fresh per-request running slot for free**. No worker RPC is needed to move GDN state on
a cache hit — align mode does it inline. Stage-4 builds directly on this.

---

## 1. Goal and the honest definition of "the win"

Stage-4 asks for `fork_from_handle(handle_id, n, ...)` that forks **without resubmit**:
children whose prefix is already computed, decoding from `num_computed_tokens` with
their **own** sampling params.

**Premise correction (NEGATIVE — needs Fable re-adjudication).** The task states the
win over resubmit is that "children can differ in sampling/max_tokens — resubmit can't
diverge sampling without breaking the prefix-cache hash sharing." **This is not true for
vLLM V1.** The prefix-cache block hash is computed from token ids plus
`extra_keys = lora + mm + cache_salt + prompt_embeds` only
(`vllm/v1/core/kv_cache_utils.py::generate_block_hash_extra_keys`); **sampling params do
not enter the block hash**. A resubmit with per-child `SamplingParams` therefore already
shares the same pinned prefix cache. So per-child sampling divergence is *not* the thing
that distinguishes fork from resubmit.

The **genuine** wins of an engine-internal fork over an API-level resubmit are:

1. **One atomic engine-side call** (`fork_from_handle(handle, [spec0, spec1, spec2])`)
   instead of N `llm.generate` round-trips the caller must orchestrate.
2. **No re-shipping the prefix `token_ids` over the ZMQ boundary** N times; children are
   born inside `EngineCore` from the pinned handle's own `prompt_token_ids`.
3. **No re-tokenize / no client-side request construction** per child; the fork is a
   pure scheduler-process operation returning plain dicts.

What it does **not** buy (the residual gap, see §5): it does **not** skip the single
internal cache-hit prefill *schedule* step — see §3, option (c).

---

## 2. Insertion-point analysis (V1 request lifecycle)

A request reaches KV allocation through, in order:

```
client -> EngineCoreRequest --(preprocess_add_request)--> Request.from_engine_core_request
      -> Scheduler.add_request (enqueue WAITING)
      -> Scheduler.schedule():
            request.num_computed_tokens == 0:
              new_computed_blocks, num_local = kv_cache_manager.get_computed_blocks(req)   # <-- local prefix cache hit
              (connector) ext = connector.get_num_new_matched_tokens(req, num_local)       # <-- EXTERNAL-KV adoption hook
              num_computed = num_local + ext
            new_blocks = kv_cache_manager.allocate_slots(req, num_new_tokens,
                             new_computed_blocks=..., num_external_computed_tokens=ext)
            connector.update_state_after_alloc(req, blocks, ext)
            self.running.append(req)
```

Three candidate insertion points, evaluated honestly:

**(a) Precomputed-prefix hints on `EngineCoreRequest` + teach `kv_cache_manager` to adopt
specific donor `block_ids` at admission.** Requires new request fields and a new
allocation branch that bypasses the hash lookup to graft caller-named blocks. This is
real surgery on the scheduler/kv-manager *hot path*, and it **duplicates work that
already happens**: the pinned donor blocks are already hash-cached, so
`get_computed_blocks` already returns them. There is nothing to "hint." **Rejected** —
most invasive, highest regression surface, redundant.

**(b) Admit normally, let the existing local prefix cache adopt the donor blocks.**
`get_computed_blocks -> find_longest_cache_hit` already returns the pinned attn full
blocks (touched -> `ref_cnt += 1`) and the pinned mamba GDN full block (copied out to a
fresh running slot by align mode). This is exactly the Stage-2/3 path, already proven
byte-exact. The only new code is *constructing the child `Request`s inside `EngineCore`*
from the pinned handle and driving them — **no scheduler, kv-manager, connector, or
worker changes at all.** **Chosen.**

**(c) The `KVConnector` external-KV path (`get_num_new_matched_tokens` /
`update_state_after_alloc`).** This is the framework's *designed* extension point for
"this request has N externally-computed tokens." But it is built to **load KV from
external storage into freshly-allocated blocks** (async recv, `WAITING_FOR_REMOTE_KVS`).
Our donor blocks are **already resident in the same block pool**; routing through a
connector would mean copying resident blocks onto themselves and standing up an entire
connector lifecycle. **Rejected** — wrong tool for same-pool sharing; heavier than (b).

**Chosen insertion point: (b) — construct child `Request`s inside `EngineCore` and admit
them through the normal scheduler path, letting the *existing* local-prefix-cache
adoption (`find_longest_cache_hit` ref-count touch for attn + align-mode copy-out for
mamba) adopt the pinned donor blocks.** This is "extend the existing prefix-adopt
machinery" as the task prefers: the local prefix cache *is* the adoption mechanism, and
it is already the proven Stage-2/3 path. `fork_from_handle` stays a pure scheduler-side
utility, exactly like the Stage-1 pin set.

### Why not seed `scheduler.running` directly (the true zero-schedule fork)

A fork that decodes from `num_computed_tokens` on the *very next step* with **zero**
prefill schedule would seed the child straight into `scheduler.running` with
`num_computed_tokens = prefix_len` and a hand-populated block table (donor attn blocks
ref-count-touched + a fresh mamba running block with parked GDN state copied in via the
Stage-0 `restore_mamba_state` worker RPC), bypassing the entire WAITING ->
`get_computed_blocks` -> `allocate_slots` prefill path. Reproducing that outside
`add_request` means hand-maintaining: the coordinator's per-group block tables, the
worker `input_batch` registration + `mamba_state_idx`, cascade/common-prefix bookkeeping,
sampling metadata, and output/detokenizer registration. That is precisely the
"process-boundary plumbing too invasive for staged work" that Stage-2's docstring already
flagged for the manual-seed restore. For a **default-inert, staged-only** deliverable it
is the wrong risk trade. We take the achievable increment (b) and document the residual
(§5).

---

## 3. What actually happens on a fork (mechanism trace)

`fork_from_handle(handle_id, child_specs)` on the `EngineCore` side:

1. Look up the pin registry entry -> `prompt_token_ids` (the cached prefix) + `groups`.
   (Stage-4 assumes prefix == the pinned request's `prompt_token_ids`, matching the
   block-aligned driver; the pin is taken so the cacheable full blocks are the prompt's.)
2. For each child spec `i`: build `SamplingParams(**spec)` and a
   `Request(request_id="tqfork-...", prompt_token_ids=list(prompt), sampling_params=...,
   block_hasher=self.request_block_hasher)`; `self.scheduler.add_request(req)`.
   Each child carries its **own** sampling params / `max_tokens`.
3. Drive `self.step()` in a bounded loop until all children finish, accumulating
   per-child `new_token_ids`, capturing `num_cached_tokens` from the first
   `EngineCoreOutput.prefill_stats`, and keeping the widest mid-gen block-table snapshot
   (all children running). This runs on the busy-loop thread inside the utility handler;
   it is **not** re-entrant with `_process_engine_step` (single-threaded busy loop:
   `_process_input_queue` -> utility -> return, *then* `_process_engine_step`). By the
   time the utility returns, the children are finished and freed, so the outer step is a
   no-op and no stray outputs reach the client output socket.
4. Return plain dicts (children outputs + mid-gen block tables + free-block accounting).

On the first schedule step of each child, `get_computed_blocks` cache-hits the pinned
prefix: attn full blocks are `touch`-ed (`ref_cnt` bumped once per child), the mamba GDN
full block is copied out into a fresh running slot (align mode). `num_cached_tokens`
comes back ~= full prefix (2080/2112 in the reference config) — i.e. **~zero prefill
compute**, the measurable win. The children then decode `max_tokens` steps each with
their own sampling. The greedy child, decoding from the identical restored state,
reproduces the reference continuation **byte-exactly**.

---

## 4. API shape

EngineCore utility (env-gated, default-inert), plus client plumbing mirroring the
Stage-1 pin set (`core_client.py` abstract + Inproc + MP; `entrypoints/llm.py`):

```python
llm.fork_from_handle(
    handle_id: str,
    child_specs: list[dict],   # one dict per child; n = len(child_specs)
                               # keys: temperature, top_p, top_k, seed, max_tokens,
                               #       min_tokens, logprobs, ... (SamplingParams kwargs)
    max_steps: int | None = None,   # safety bound on the internal decode loop
) -> dict
```

Return dict:

```python
{
  "handle_id": str,
  "n": int,
  "prefix_len": int,                     # len(pinned prompt_token_ids)
  "children": [
     {"req_id": str, "token_ids": [int], "num_cached_tokens": int,
      "num_output_tokens": int, "finish_reason": str, "spec": {...}},
     ...
  ],
  "midgen_block_tables": {               # widest catch, all children running
     "n_running": int,
     "requests": {req_id: {"groups": {gid: {"spec", "block_ids"}}}},
  },
  "pre_free_blocks": int,                # free-block count BEFORE adding children
  "post_free_blocks": int,               # free-block count AFTER all finish + free
  "steps": int,
}
```

`n` is implied by `len(child_specs)` (cleaner than a separate `n` + shared
`sampling_params`, and it makes per-child divergence first-class). All args cross the
ZMQ utility boundary as plain JSON-ish types; `SamplingParams` is constructed
engine-side. All returns are plain str/int/list/dict — serializable, same discipline as
the Stage-1 methods.

---

## 5. Refcount / free semantics, and the leak invariant

Per child, over its lifetime:

- **attn pinned full blocks**: `ref_cnt` = baseline (the pin's own +1) **+1** while the
  child rides them, back to baseline when the child frees. Shared across all children
  simultaneously (the Stage-3 shared-attn-block observation).
- **mamba pinned GDN full block**: same +1/-1 touch/free; state is *copied out*, never
  mutated in place.
- **mamba running slot**: a **fresh** block per child (distinct across children — the
  Stage-3 "distinct mamba running slots" invariant; a shared running slot would be two
  sequences writing one recurrent state = NEGATIVE), freed on finish.
- **decode blocks**: allocated as the child generates, freed on finish.

**Leak invariant:** after all children finish and free, the block pool's free-block count
returns to the **pre-fork baseline** (the pinned handle's blocks stay pinned — released
only by `unpin_kv_blocks`, not by the fork). The driver asserts
`post_free_blocks == pre_free_blocks`. The pin itself is untouched by the fork;
ownership/release stays with the caller's `unpin_kv_blocks(handle_id)` (EXP-038 Risk #1,
auditable pin/unpin).

---

## 6. Failure modes

| Mode | Cause | Handling |
|------|-------|----------|
| feature disabled | `VLLM_TQ_GDN_SNAPSHOT != 1` | `_tq_snapshot_require_enabled()` raises; default-inert. |
| unknown handle | bad `handle_id` | `KeyError` from the pin registry. |
| no prefix caching | `request_block_hasher is None` | explicit `RuntimeError` (fork needs the prefix cache to adopt donor blocks). |
| child never cache-hits | pin evicted / hash mismatch | `num_cached_tokens ~ 0`; driver flags per-child (byte-exactness may still hold via recompute -> FAIL-inconclusive, not a NEGATIVE). |
| shared mamba running slot | align-mode slot aliasing bug | driver hard-NEGATIVE (Stage-3 rule). |
| decode loop runs away | stop never reached | `max_steps` bound -> `finish_reason="length_or_bound"`; reported. |
| block leak | children left residue | `post_free_blocks != pre_free_blocks` -> driver FAIL. |

## 7. Residual gap vs the ideal (precise, for re-adjudication)

1. **One internal cache-hit prefill schedule step per child is still incurred.** The
   chosen path (b) admits children through the normal WAITING -> schedule path; the
   scheduler always schedules `num_tokens - num_cached_tokens` (~= 1 block, 2112-2080=32
   tokens in the reference config) on the first step. This is a property of the prefix
   cache, **identical to what resubmit incurs** — fork does not make it worse, but it
   also does not eliminate it. Eliminating it requires the direct-`running`-seed path
   (§2, rejected as too invasive for staged work). `num_cached_tokens ~= full prefix`
   is the proxy for "~zero prefill compute"; it is not *literally* zero scheduled tokens.
2. **Premise correction (§1):** per-child sampling divergence is *not* unique to fork;
   resubmit supports it too (block hash excludes sampling params). Flagged
   **NEGATIVE — needs Fable re-adjudication** so the "win" is stated honestly: the wins
   are the atomic engine-side call, no ZMQ re-ship of `token_ids`, and no client-side
   re-tokenize/construction — not sampling divergence.

Both items are **NEGATIVE — needs Fable re-adjudication** and are surfaced by the driver.
