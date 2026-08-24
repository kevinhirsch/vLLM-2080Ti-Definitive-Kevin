# EXP-038 — Fork v2: the non-blocking production fork path (`POST /tq/fork2`)

Fork **v1** (`POST /tq/fork` → `EngineCore.fork_from_handle`) proved the hardware
mechanism — pin a finished prefix's KV blocks, then fork N cheap continuations
that ride the shared prefix — but it is **not** a production shape. Fork **v2**
(`POST /tq/fork2`) keeps the proven cache-hit adoption and fixes the three v1
gaps by moving the fan-out into the **server layer**.

## The v1 gaps this closes

`fork_from_handle` (see docs/exp038-stage4-fork-api.md) runs the children to
completion **inside** an `EngineCore` utility call. That has three consequences:

1. **Blocks the busy loop.** The single-threaded engine loop is occupied for the
   whole fork; no other request is served. `/tq/fork` must cap
   `sum(child.max_tokens)` (`max_total_tokens`, default 2048) to bound the stall.
2. **Bypasses normal output handling → special-token soup.** v1 builds each
   child as a raw `Request(sampling_params=SamplingParams(**spec), …)` directly,
   *skipping* the normal input path
   (`vllm/v1/engine/input_processor.py::process_inputs`) that calls
   `sampling_params.update_from_generation_config(..., eos_token_id)` and
   `update_from_tokenizer(...)`. So the model's **eos / generation-config stop
   tokens are never injected** — children ignore EOS and emit until
   `max_tokens`, trailing special-token soup after the real answer. (Observed in
   the v1 e2e demo.)
3. **Quiescent-only, one blob.** Children are driven and returned in a single
   synchronous payload; no streaming, no per-child abort, no scheduler
   observability.

## The design: admit children as ordinary requests

The pin already guarantees the prefix blocks stay resident (`/tq/pin` `touch`es
them; `ref_cnt ≥ 1` until `/tq/unpin`). The **cache-hit adoption path is exactly
the Stage-2/3 resubmit pattern** that was proven byte-exact: submit a request
whose prompt token ids begin with the pinned prefix, and the local prefix cache
adopts the pinned donor blocks (attn full blocks touched; mamba GDN full block
copied out by align mode) at admission.

So v2 does the fan-out as a thin **server-layer** step:

```
POST /tq/fork2 {handle_id, children:[…]}
  1. get_pin_handle(handle_id)         # pin-check + source the pinned prefix
  2. verify_pinned_blocks(handle_id)   # optional residency assertion (non-fatal)
  3. for each child (in parallel):
        AsyncLLM.generate(
            TokensPrompt(prompt_token_ids=<pinned prefix>, cache_salt=<donor>),
            SamplingParams(**child_spec),   # output_kind=FINAL_ONLY
            request_id)
  4. gather → respond
```

Because each child now travels the **standard `AsyncLLM.generate` path**:

* the scheduler admits it as a **first-class request** — the busy loop is never
  blocked; children co-run with each other and with any other traffic (v2 is
  genuinely non-blocking, no `max_total_tokens` cap);
* the pinned prefix is adopted via the **normal prefix cache** — `num_cached_tokens
  ≈ prefix_len` (~zero prefill compute), same win as v1;
* the input path **injects the model eos / generation-config stop tokens** — so
  children **stop correctly** (gap #2 gone, no per-child stop-token workaround);
* children are individually **abortable / observable** and the response carries
  per-child `finish_reason` + `stop_reason`.

### (a) vs (b): why server-layer fan-out, not engine-side spawn

The naive next step from v1 is a non-blocking **`fork_spawn`** EngineCore utility:
construct each child's `EngineCoreRequest` from the handle's token_ids + cache_salt,
`add_request` it, and **return immediately** (no stepping). That admits children
without blocking — **but the requests injected engine-side have no attached client
stream.** `AsyncLLM` tracks a request's output stream in its `OutputProcessor` /
`RequestState`, created by `AsyncLLM.add_request` (→ the per-request
`RequestOutputCollector` that `generate()` drains). A request that appears inside
`EngineCore` from a utility call was never registered there, so its `EngineCoreOutput`s
have nowhere to go. Closing that gap means **shape (b)**: build an engine-side
output buffer + a polling endpoint (`/tq/fork_poll`), and re-implement detokenization
and stop-string handling on that side — which reintroduces exactly the bypass that
caused gap #2.

**Shape (a)** — the server-layer fan-out implemented here — instead reuses
`AsyncLLM.generate`, which **already** does stream registration, detokenization,
stop handling, sampling, aborts, and metrics. It needs:

* **no new EngineCore fork machinery** — only one tiny **read-only** registry
  accessor, `get_pin_handle` (a sibling of the existing read-only
  `verify_pinned_blocks` / `get_request_kv_block_ids`), so the server can source
  the pinned prefix by `handle_id` alone rather than forcing the caller to
  resupply the prompt;
* nothing else — the pin (`/tq/pin`) already guarantees residency, and the cache
  hit does the block adoption.

(a) is strongly preferred: less code, less surface, and it **inherits every
generate() semantic for free**. (b) is only worth revisiting if a future need
requires children that are *not* expressible as an ordinary `generate()` call
(none today). Per-child **SSE streaming** is the one deferred piece of (a): the
route currently collects each child's final cumulative output
(`output_kind=FINAL_ONLY`) and returns them together; streaming each child back to
its own client stream is a mechanical follow-up (return `N` `StreamingResponse`s
or a multiplexed SSE frame per child) that needs no engine change.

### On `get_pin_handle` and `cache_salt`

`get_pin_handle` returns the serializable fields v2 needs — `prompt_token_ids`,
`cache_salt`, `prefix_len`, `num_computed_tokens` — read straight from the pin
registry `pin_request_kv_blocks` already populated. It performs **no** touch /
free / alloc / step, so it cannot perturb the blocks it describes. Its `KeyError`
on an unknown/released handle **is** the pin-check (`/tq/unpin` pops the registry
entry).

`cache_salt` matters because the prefix-cache block hash mixes it in
(`generate_block_hash_extra_keys`): a child built with a *different* salt than the
pinned donor would hash differently and fail to adopt the pinned blocks. v2
defaults each child's salt to the donor's (from the handle), overridable via the
request body. NB the current `/tq/pin` route never sets a salt (the keepalive
`generate` omits it), so the common path is the unsalted default on both sides —
consistent by construction.

## Route contract

### `POST /tq/fork2`
Body:
```jsonc
{
  "handle_id": "tqpin-…",                 // required; from /tq/pin
  "children": [                            // required, non-empty
    {"temperature": 0.0, "max_tokens": 256},
    {"temperature": 0.7, "max_tokens": 256, "stop": ["</s>"]}
  ],
  "cache_salt": null,                      // optional; default = donor's salt
  "verify_pin": true,                      // optional; residency assertion
  "per_child_timeout_s": null              // optional; wait_for per child
}
```
Child keys are whitelisted by `vllm/entrypoints/openai/tq_fork_specs.py`
(`CHILD_SAMPLING_KEYS`), shared with `/tq/fork`, so a malformed body can't smuggle
arbitrary `SamplingParams` kwargs into the engine.

Response (a `/tq/fork`-compatible superset, so localflow consumes either):
```jsonc
{
  "handle_id": "tqpin-…",
  "n": 2,
  "prefix_len": 2112,
  "pin_resident": true,                    // present unless verify_pin:false
  "children": [
    {"req_id": "tqfork2-…-0", "text": "…", "token_ids": [...],
     "num_output_tokens": 41, "num_cached_tokens": 2100,
     "finish_reason": "stop", "stop_reason": null, "spec": {...}}
  ]
}
```
Per-child failures are **isolated**: a child that raises or times out yields an
entry with `finish_reason` `"error"`/`"timeout"` and an `error` field, rather than
failing the whole batch (`asyncio.gather(return_exceptions=True)`).

The pin is **not** released by `/tq/fork2` — ownership stays with the caller's
`/tq/unpin` (EXP-038 Risk #1), matching `/tq/fork`.

## What changed

**vllm (worktree `.ftree-gdnsnap`, branch `feat-gdn-snapshot`):**

* `vllm/v1/engine/core.py` — new read-only `EngineCore.get_pin_handle` utility.
* `vllm/v1/engine/core_client.py` — `get_pin_handle` mirrors on the abstract
  client, `InprocClient`, `SyncMPClient` (`call_utility`), `AsyncMPClient`
  (`get_pin_handle_async` via `call_utility_async`), and added to
  `_TQ_SNAPSHOT_SINGLE_ENGINE_UTILITIES` (DPLB rejects it like the rest).
* `vllm/v1/engine/async_llm.py`, `vllm/v1/engine/llm_engine.py`,
  `vllm/entrypoints/llm.py`, `vllm/engine/protocol.py` — `get_pin_handle`
  wrappers, mirroring the existing `verify_pinned_blocks` surface exactly.
* `vllm/entrypoints/openai/tq_fork_specs.py` — **new** dependency-free module:
  `CHILD_SAMPLING_KEYS` + `parse_child_specs` (extracted from the router so the
  validation contract is unit-testable offline).
* `vllm/entrypoints/openai/tq_snapshot_router.py` — new `POST /tq/fork2` route;
  `/tq/fork` refactored to reuse `parse_child_specs`; docstring + attach-log
  updated.
* `tests/entrypoints/openai/test_tq_fork_specs.py` — offline unit tests (run in a
  bare checkout; load the helper by path).
* `tests/entrypoints/openai/test_tq_fork2_route.py` — route-level tests with a
  fake engine + fake Request (`importorskip` torch/fastapi).
* `tools/tq_gdn_snapshot_stage6_fork2.py` — throwaway-engine e2e window driver.

**localflow (`/home/kevin/localflow`, branch `master`):**

* `localflow.py` — `Orchestrator.fork_agents` now prefers `/tq/fork2` and falls
  back to `/tq/fork` only on HTTP 404 (older engine), then to N ordinary
  `agent()` calls; new `_tq_fork` helper; no per-child stop-token workaround
  needed (the engine injects eos/stop). Log line shows which route ran.

## Test evidence

* **Offline (ran here, no engine):** `python
  tests/entrypoints/openai/test_tq_fork_specs.py` → **8/8 passed** (whitelist
  rejection, non-dict child, `max_tokens` coercion/default, input-not-mutated,
  error-index).
* **`py_compile`:** all touched vllm files, both test files, the stage-6 tool,
  and `localflow.py` compile clean.
* **Route-level** (`test_tq_fork2_route.py`) and **e2e** (`stage6_fork2.py`)
  require the vllm/torch/fastapi runtime and a throwaway engine respectively —
  see below.

## Window test plan (throwaway engine only — NEVER `:8001`)

The route-level and hardware checks need the real runtime. In a scheduled
maintenance window on the throwaway 2-GPU engine:

1. **Route-level (needs torch+fastapi, no GPU model required):**
   ```
   pytest tests/entrypoints/openai/test_tq_fork2_route.py -q
   ```
   Exercises validation (400/404), happy-path fan-out (each child submitted with
   the full pinned prefix, `num_cached_tokens` surfaced, `pin_resident`), and
   per-child error isolation — all against a fake engine.

2. **Bring up a throwaway server WITH the routes (NOT `:8001`):**
   ```
   VLLM_TQ_GDN_SNAPSHOT=1 vllm serve /path/to/Qwen3.8-27B-hybrid \
       --port 8011 --tensor-parallel-size 2 --mamba-cache-mode align \
       --enable-prefix-caching --kv-cache-dtype turboquant_k8v4 --dtype half \
       --gpu-memory-utilization 0.82
   ```

3. **Drive the e2e demo:**
   ```
   VLLM_TQ_GDN_SNAPSHOT_CONFIRM_THROWAWAY=1 \
   python tools/tq_gdn_snapshot_stage6_fork2.py \
       --engine-url http://127.0.0.1:8011 --children 4 --max-tokens 256
   ```
   **PASS = A ∧ B ∧ C:**
   * **(A) stop-correct** — every child `finish_reason == "stop"` (model eos),
     NOT `"length"`. This is the v1-vs-v2 discriminator: it directly confirms the
     eos/stop injection the server-layer path restores.
   * **(B) cheap prefill** — every child `num_cached_tokens ≈ prefix_len`
     (≥ `prefix_len − one block`).
   * **(C) resident + no leak** — `pin_resident` true during the fork, `/tq/unpin`
     frees a non-zero block count once.

4. **Non-blocking check (optional):** while a large fork2 is running, fire an
   unrelated `/v1/chat/completions` on the same server and confirm it is served
   concurrently (v1 `/tq/fork` would stall it for the whole fork).

## Gaps / follow-ups

* **Per-child SSE streaming** — deferred; the route collects final outputs. No
  engine change needed to add it (return per-child streams from the server).
* **`/tq/pin` is still quiescent-only** — unchanged by v2; `/tq/fork2` inherits
  the throwaway-engine constraint from pin (`req_id=None` → `running[0]`).
* **Not hardware-verified yet** — the offline + route tests pass; (A)/(B)/(C)
  above are the pending window confirmation on real GPUs.
* **Keepalive-shim passthrough** — localflow still talks to the direct engine
  (`:8001`-style direct port) for `/tq/*`; proxying fork2 through the `:8000`
  shim is a separate wiring task.
