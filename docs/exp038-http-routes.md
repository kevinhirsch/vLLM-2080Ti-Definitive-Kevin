# EXP-038 — HTTP routes for the GDN snapshot / fork chain

Staged HTTP surface that exposes the hardware-proven Stage-1/Stage-4 fork chain
(`pin_request_kv_blocks` → `fork_from_handle` → `unpin_kv_blocks`) over the
OpenAI server, so the **localflow** orchestrator can do *"prefill once, fork N"*:
submit one big shared context, then fan out N cheap continuations that ride the
shared KV prefix.

Env-gated by `VLLM_TQ_GDN_SNAPSHOT`, **default-inert**. `attach_router()` is a
no-op unless the gate is set, so wiring it into `build_app` cannot perturb the
production `:8001` serve. **NEVER enable against `:8001`.**

## Why the async client needed new plumbing

The utility surface (`pin_request_kv_blocks`, `get_request_kv_block_ids`,
`verify_pinned_blocks`, `unpin_kv_blocks`, `fork_from_handle`) already existed on
the **sync** path: `EngineCore` (methods) → `SyncMPClient.call_utility` →
`LLMEngine` → `LLM` (`entrypoints/llm.py`). But the OpenAI `api_server` builds
**`AsyncLLM`**, whose engine client is an `AsyncMPClient`. That async client
exposed `call_utility_async` and mirrors for `reset_prefix_cache` etc., but had
**no** pin/fork mirrors. Added (both default-inert, env-gated engine-side):

* `vllm/v1/engine/core_client.py` — `AsyncMPClient.{pin_request_kv_blocks,
  verify_pinned_blocks, get_request_kv_block_ids, unpin_kv_blocks,
  fork_from_handle}_async`, each a one-liner over `call_utility_async` (exactly
  how `reset_prefix_cache_async` travels).
* `vllm/v1/engine/async_llm.py` — `AsyncLLM.{pin_request_kv_blocks,
  verify_pinned_blocks, get_request_kv_block_ids, unpin_kv_blocks,
  fork_from_handle}`, delegating to the async engine client (mirror of the sync
  `LLM` methods).

No new EngineCore code: the routes reuse the Stage-1/4 utilities verbatim.

## Routes

### `POST /tq/pin`
Body: `{prompt | messages | prompt_token_ids, keepalive_tokens?, timeout_s?,
poll_interval_s?}` → `{handle_id, num_computed_tokens, num_pinned_blocks,
prefix_len, fully_prefilled}`.

The pin utility must catch a **live** request — it `touch`-pins the resident KV
blocks of an in-flight request so they survive that request's `free()`. So the
endpoint **submits the prefill itself**: it fires a tiny keepalive generation of
the shared context and, while that request is in flight, polls
`pin_request_kv_blocks(req_id=None)` (→ the single running request) until the
full prefix is computed, then aborts the keepalive. This is the async,
server-side reincarnation of the Stage-2 `_run_and_pin` background-generate +
poll-pin pattern.

Refcount walk (matches Stage-1/2): keepalive holds the prefix blocks
(`ref_cnt ≥ 1`) → pin `touch` (`+1`) → keepalive abort `free` (`−1`) → pinned
blocks resident at `ref_cnt ≥ 1`, out of the free queue, available to a fork.

Mid-prefill catches are released and retried so we never leak a refcount and end
up pinning the whole prefix. On timeout, one final best-effort pin is taken
(may be partial). **Correctness is unaffected by a partial pin**: `fork_from_handle`
rebuilds each child with the *full* `prompt_token_ids` stored at pin time and
re-adopts whatever remains cached; only the compute-saved proxy shrinks.

### `POST /tq/fork`
Body: `{handle_id, children:[{temperature?, max_tokens?, top_p?, seed?, ...}],
max_steps?, max_total_tokens?, detokenize?}` → the raw `fork_from_handle` payload
(`children[i].{token_ids, num_cached_tokens, num_output_tokens, finish_reason}`,
`midgen_block_tables`, `pre/post_free_blocks`, `steps`), with each child
augmented by a decoded `text` field unless `detokenize:false`.

Child sampling keys are whitelisted (`_CHILD_SAMPLING_KEYS`) so a malformed body
can't smuggle arbitrary `SamplingParams` kwargs into the engine.

### `POST /tq/unpin`
Body: `{handle_id}` → `{handle_id, ok, num_freed_blocks}`. Idempotent.

## Known limitations / risks (honest)

1. **Quiescent-engine assumption.** `/tq/pin` uses `req_id=None`, which resolves
   `scheduler.running[0]`. Correct only when the snapshot engine runs one request
   at a time (the intended throwaway-engine model). Under concurrent traffic it
   could pin the wrong request. Targeting a *specific* external request_id was
   deliberately avoided because `AsyncLLM.add_request` may reassign the id before
   it reaches EngineCore; resolving that mapping is the clean follow-up for a
   multi-tenant version. NEGATIVE for any busy serve — hence throwaway-only.

2. **Fork blocks the busy loop.** `fork_from_handle` drives children to
   completion *inside* the EngineCore utility handler (single-threaded; no child
   output leaks to the client socket). For the whole fork, the engine serves no
   other request. `/tq/fork` caps `sum(child.max_tokens)` (`max_total_tokens`,
   default 2048). The non-blocking version (admit children as ordinary requests,
   stream them back) is the real product path and is **not** built here.

3. **Not hardware-verified in the async server.** The Stage-0–4 chain is
   hardware-proven under the *sync* PoC engine driver. These routes drive the
   same utilities through the *async* multiproc client, which is **staged only** —
   no engine run this session. In particular the in-utility step loop of
   `fork_from_handle` has not been exercised on the async `EngineCoreProc` busy
   loop. **NEGATIVE — needs a throwaway-engine smoke run (and possibly Fable
   re-adjudication) before any reliability claim.**

4. **Tokenizer coupling.** `/tq/pin` with `messages` renders via
   `tokenizer.apply_chat_template`; with `prompt` via `tokenizer.encode`. Fork
   children share the pinned prefix verbatim — they cannot diverge in prompt
   text, only in sampling params. That is the intended "same context, N samplers"
   shape (e.g. N verifiers at different temperatures).

## Test (future window — throwaway engine only)

```bash
# 1. Boot a THROWAWAY engine with the gate + routes on (small model, tiny ctx).
#    NEVER :8001. Prefix caching must be ON (default in V1).
VLLM_TQ_GDN_SNAPSHOT=1 \
python -m vllm.entrypoints.openai.api_server \
    --model <small-hybrid-or-any-generate-model> \
    --enable-prefix-caching \
    --port 8099

# 2. Smoke the routes directly.
curl -s localhost:8099/tq/pin -H 'content-type: application/json' \
    -d '{"prompt":"<a few hundred tokens of shared context>"}' | tee /tmp/pin.json
H=$(python -c 'import json;print(json.load(open("/tmp/pin.json"))["handle_id"])')
curl -s localhost:8099/tq/fork -H 'content-type: application/json' \
    -d "{\"handle_id\":\"$H\",\"children\":[
         {\"temperature\":0.0,\"max_tokens\":32},
         {\"temperature\":0.7,\"max_tokens\":32}]}" | python -m json.tool
curl -s localhost:8099/tq/unpin -H 'content-type: application/json' \
    -d "{\"handle_id\":\"$H\"}"

# 3. Drive it end-to-end from localflow (both URLs point at the throwaway engine).
LOCALFLOW_ENGINE_URL=http://127.0.0.1:8099 \
LOCALFLOW_GATEWAY=http://127.0.0.1:8099/v1 \
python ~/localflow/localflow.py ~/localflow/examples/fork_research.py --concurrency 4
```

Expected: `/tq/pin` returns `fully_prefilled:true` with `num_computed_tokens ==
prefix_len`; each fork child returns text + `num_cached_tokens ≈ prefix_len`
(the zero-prefill-compute proxy) + `finish_reason:"stop"`/`"length"`;
`post_free_blocks == pre_free_blocks` (no leak); `/tq/unpin` frees the blocks.
