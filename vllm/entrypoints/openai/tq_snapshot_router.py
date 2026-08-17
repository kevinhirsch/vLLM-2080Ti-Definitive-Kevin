# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXP-038 (VLLM_TQ_GDN_SNAPSHOT) custom HTTP routes: "prefill once, fork N".

Exposes the hardware-proven Stage-1/Stage-4 fork chain (pin a finished prefix's
KV blocks, then fork N cheap continuations that ride that shared prefix) over
three HTTP routes on the OpenAI server:

    POST /tq/pin    {prompt|messages|prompt_token_ids, ...} -> {handle_id, ...}
    POST /tq/fork   {handle_id, children:[{temperature,max_tokens,...}]} -> {...}
    POST /tq/fork2  {handle_id, children:[...]} -> {...}  (non-blocking, v2)
    POST /tq/unpin  {handle_id} -> {ok, num_freed_blocks}

Design notes / honest limitations (full write-ups: docs/exp038-http-routes.md
for pin/fork/unpin, docs/exp038-fork-v2.md for fork2):

* The pin utility must catch a LIVE request (it ``touch``-pins the resident KV
  blocks of an in-flight request so they outlive its ``free()``). So /tq/pin
  SUBMITS the prefill itself: it fires a tiny keepalive generation of the shared
  context and, while that request is in flight, polls ``pin_request_kv_blocks``
  (req_id=None -> the single running request) until the full prefix is computed,
  then aborts the keepalive. This is the async, server-side reincarnation of the
  Stage-2 ``_run_and_pin`` pattern. It assumes a QUIESCENT snapshot engine (one
  request at a time); req_id=None picks ``running[0]`` and would be unreliable
  under concurrent traffic -- which is exactly why this is throwaway-engine-only.

* /tq/fork (v1) drives the children to completion INSIDE the EngineCore utility
  (the single-threaded busy loop is BLOCKED for the whole fork), builds Request
  objects from raw SamplingParams that bypass the model's eos/stop injection, and
  returns one blob. Cap ``sum(child.max_tokens)`` (``max_total_tokens``, default
  2048).

* /tq/fork2 (v2, the production shape) is a thin SERVER-LAYER fan-out: it sources
  the pinned prefix by handle_id (``get_pin_handle``) and admits each child as an
  ORDINARY ``AsyncLLM.generate`` request. The busy loop is never blocked; the
  pinned prefix is adopted via the normal prefix cache (~zero prefill); and the
  model's eos / generation-config stop tokens are injected on the standard input
  path, so children stop correctly instead of emitting special-token soup. Per-
  child SSE streaming is a documented follow-up (this collects final outputs).

Env-gated + default-inert: ``attach_router`` is a no-op unless
VLLM_TQ_GDN_SNAPSHOT=1. NEVER attach against the production :8001 serve.
"""
import asyncio
import time
import uuid
from http import HTTPStatus

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

import vllm.envs as envs
from vllm import TokensPrompt
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.tq_fork_specs import (
    parse_child_specs as _parse_child_specs,
)
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind, SamplingParams

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def _err(status: HTTPStatus, msg: str) -> JSONResponse:
    return JSONResponse(content={"error": msg}, status_code=status.value)


async def _resolve_prompt_token_ids(engine: EngineClient, body: dict) -> list[int]:
    """Turn the request body into prompt token ids. Accepts (in priority order)
    an explicit ``prompt_token_ids`` list, a chat ``messages`` list (rendered via
    the tokenizer's chat template), or a raw ``prompt`` string."""
    if "prompt_token_ids" in body:
        # Presence, not truthiness: an explicitly-provided prompt_token_ids keeps
        # highest priority and an invalid/empty value is rejected consistently,
        # rather than silently falling through to messages/prompt when the key is
        # present but falsy (e.g. [] or null).
        ptids = body["prompt_token_ids"]
        if not isinstance(ptids, list) or not ptids:
            raise ValueError(
                "prompt_token_ids must be a non-empty list[int] (got "
                f"{type(ptids).__name__}); a bare string would be silently "
                "reinterpreted as per-character token ids"
            )
        return [int(t) for t in ptids]
    tokenizer = engine.get_tokenizer()
    messages = body.get("messages")
    if messages is not None:
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        return list(ids)
    prompt = body.get("prompt")
    if not prompt:
        raise ValueError(
            "provide exactly one of: prompt (str), messages (list), "
            "prompt_token_ids (list[int])"
        )
    return list(tokenizer.encode(prompt))


@router.post("/tq/pin")
async def tq_pin(raw_request: Request) -> JSONResponse:
    """Prefill the shared context once and pin its KV prefix mid-flight.

    Body: ``{prompt|messages|prompt_token_ids, keepalive_tokens?, timeout_s?}``.
    Returns ``{handle_id, num_computed_tokens, num_pinned_blocks, prefix_len,
    fully_prefilled}``. Release the handle with /tq/unpin when done.
    """
    engine = engine_client(raw_request)
    try:
        body = await raw_request.json()
    except Exception:  # noqa: BLE001
        return _err(HTTPStatus.BAD_REQUEST, "request body must be valid JSON")
    if not isinstance(body, dict):
        return _err(HTTPStatus.BAD_REQUEST, "request body must be a JSON object")

    try:
        token_ids = await _resolve_prompt_token_ids(engine, body)
    except Exception as e:  # noqa: BLE001 - surface tokenization/validation errors
        return _err(HTTPStatus.BAD_REQUEST, str(e))
    if not token_ids:
        return _err(HTTPStatus.BAD_REQUEST, "prompt resolved to zero tokens")

    prefix_len = len(token_ids)
    # Keepalive: enough decode headroom that the request does not finish in the
    # window between prefill-complete and our abort. ignore_eos guards against an
    # early EOS. The tokens themselves are thrown away.
    try:
        keepalive_tokens = max(2, int(body.get("keepalive_tokens", 16)))
        timeout_s = float(body.get("timeout_s", 60.0))
        poll_s = float(body.get("poll_interval_s", 0.02))
    except (TypeError, ValueError) as e:
        return _err(
            HTTPStatus.BAD_REQUEST,
            f"keepalive_tokens/timeout_s/poll_interval_s must be numeric: {e}",
        )

    request_id = f"tqpin-src-{uuid.uuid4().hex[:12]}"
    sampling = SamplingParams(
        max_tokens=keepalive_tokens,
        min_tokens=keepalive_tokens,
        temperature=0.0,
        ignore_eos=True,
    )

    # Fire the keepalive prefill as a background task that just drains the stream.
    gen = engine.generate(
        TokensPrompt(prompt_token_ids=token_ids), sampling, request_id
    )

    async def _drain() -> None:
        try:
            async for _ in gen:
                pass
        except asyncio.CancelledError:  # expected on abort
            pass
        except Exception:  # noqa: BLE001
            logger.exception("tq_pin: keepalive generation raised")

    drain_task = asyncio.create_task(_drain())

    handle: dict | None = None
    deadline = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline:
            try:
                # req_id=None -> the single in-flight (keepalive) request.
                cand = await engine.pin_request_kv_blocks(None)
            except Exception:  # noqa: BLE001 - not resident yet / transient RPC
                await asyncio.sleep(poll_s)
                continue
            if int(cand.get("num_computed_tokens", 0)) >= prefix_len:
                handle = cand  # caught the whole prefix -> keep this pin
                break
            # Caught mid-prefill: release (exactly undo the touch) and retry so we
            # never leak refcounts and end up pinning the full prefix.
            try:
                await engine.unpin_kv_blocks(cand["handle_id"])
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(poll_s)
        if handle is None:
            # Timed out without catching a full prefill: one final best-effort
            # pin (may be partial). Correctness is unaffected -- fork rebuilds
            # children with the full prompt and re-adopts whatever is cached --
            # only the compute-saved proxy shrinks.
            try:
                handle = await engine.pin_request_kv_blocks(None)
            except Exception:  # noqa: BLE001
                handle = None
    finally:
        # Stop the keepalive producer. Its free() decrements the block refcount
        # once, leaving the pinned (touched) blocks resident (ref_cnt >= 1).
        try:
            await engine.abort(request_id)
        except Exception:  # noqa: BLE001
            pass
        drain_task.cancel()

    if handle is None:
        return _err(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "could not pin a live request (never caught the prefill in flight). "
            "Confirm VLLM_TQ_GDN_SNAPSHOT=1, prefix caching enabled, and that no "
            "other traffic is competing on this engine.",
        )

    num_computed = int(handle.get("num_computed_tokens", 0))
    return JSONResponse(
        content={
            "handle_id": handle["handle_id"],
            "num_computed_tokens": num_computed,
            "num_pinned_blocks": handle.get("num_pinned_blocks"),
            "prefix_len": prefix_len,
            "fully_prefilled": num_computed >= prefix_len,
        }
    )


@router.post("/tq/fork")
async def tq_fork(raw_request: Request) -> JSONResponse:
    """Fork a pinned handle into N children, each with its own sampling params.

    Body: ``{handle_id, children:[{temperature?,max_tokens?,...}], max_steps?,
    max_total_tokens?, detokenize?}``. Returns the raw ``fork_from_handle``
    payload with each child augmented by a decoded ``text`` field (unless
    ``detokenize`` is false).

    WARNING: children run to completion inside the EngineCore busy loop, which is
    blocked for the whole call. Keep ``sum(child.max_tokens)`` small.
    """
    engine = engine_client(raw_request)
    try:
        body = await raw_request.json()
    except Exception:  # noqa: BLE001
        return _err(HTTPStatus.BAD_REQUEST, "request body must be valid JSON")
    if not isinstance(body, dict):
        return _err(HTTPStatus.BAD_REQUEST, "request body must be a JSON object")

    handle_id = body.get("handle_id")
    if not handle_id:
        return _err(HTTPStatus.BAD_REQUEST, "missing handle_id")
    children = body.get("children")
    if not isinstance(children, list) or not children:
        return _err(HTTPStatus.BAD_REQUEST, "children must be a non-empty list")

    try:
        max_total_tokens = int(body.get("max_total_tokens", 2048))
    except (TypeError, ValueError) as e:
        return _err(HTTPStatus.BAD_REQUEST, f"max_total_tokens must be an int: {e}")
    try:
        child_specs = _parse_child_specs(children)
    except ValueError as e:
        return _err(HTTPStatus.BAD_REQUEST, str(e))
    total_budget = sum(s["max_tokens"] for s in child_specs)

    if total_budget > max_total_tokens:
        return _err(
            HTTPStatus.BAD_REQUEST,
            f"sum(child.max_tokens)={total_budget} exceeds max_total_tokens="
            f"{max_total_tokens}. The busy loop is blocked for the whole fork; "
            "lower max_tokens/children or raise max_total_tokens deliberately.",
        )

    try:
        result = await engine.fork_from_handle(
            handle_id, child_specs, body.get("max_steps")
        )
    except KeyError as e:
        return _err(HTTPStatus.NOT_FOUND, f"unknown pin handle: {e}")
    except RuntimeError as e:
        return _err(HTTPStatus.BAD_REQUEST, str(e))
    except Exception as e:  # noqa: BLE001
        logger.exception("tq_fork failed")
        return _err(HTTPStatus.INTERNAL_SERVER_ERROR, f"fork failed: {e}")

    if body.get("detokenize", True):
        tokenizer = engine.get_tokenizer()
        for child in result.get("children", []):
            try:
                child["text"] = tokenizer.decode(child.get("token_ids") or [])
            except Exception:  # noqa: BLE001
                child["text"] = None

    return JSONResponse(content=result)


@router.post("/tq/fork2")
async def tq_fork2(raw_request: Request) -> JSONResponse:
    """Non-blocking production fork (EXP-038 v2): fan a pinned handle out into N
    children submitted as ORDINARY streamed requests.

    Body: ``{handle_id, children:[{temperature?,max_tokens?,...}], cache_salt?,
    verify_pin?, per_child_timeout_s?}``. Returns a /tq/fork-compatible payload:
    ``{handle_id, n, prefix_len, children:[{req_id, text, token_ids,
    num_output_tokens, num_cached_tokens, finish_reason, stop_reason, spec}],
    pin_resident?}``.

    Contrast with /tq/fork (v1): that builds child Requests inside the EngineCore
    utility from raw ``SamplingParams`` (bypassing the model's eos/stop
    injection) and drives them to completion with the busy loop BLOCKED. Here the
    fork lives entirely in the server layer:

      1. pin-check + source the prefix by handle_id: ``get_pin_handle`` reads the
         pin registry /tq/pin populated (KeyError -> released/unknown -> 404) and
         returns the cached ``prompt_token_ids`` + ``cache_salt``.
      2. optional residency assertion (``verify_pin``, default on): confirm the
         pinned blocks are still resident so children cache-hit, not recompute.
      3. N parallel ``AsyncLLM.generate`` calls, each with the pinned prefix as
         ordinary prompt token ids + its own ``SamplingParams``. The scheduler
         admits them as first-class requests (busy loop never blocked), the
         prefix cache adopts the pinned donor blocks (~zero prefill compute), and
         the standard input path injects the model eos / generation-config stop
         tokens -- so children STOP correctly (no special-token soup).

    Per-child SSE streaming is a documented follow-up: this collects each child's
    final cumulative output (``output_kind=FINAL_ONLY``). The pin is NOT released
    here -- ownership stays with the caller's /tq/unpin.
    """
    engine = engine_client(raw_request)
    try:
        body = await raw_request.json()
    except Exception:  # noqa: BLE001
        return _err(HTTPStatus.BAD_REQUEST, "request body must be valid JSON")
    if not isinstance(body, dict):
        return _err(HTTPStatus.BAD_REQUEST, "request body must be a JSON object")

    handle_id = body.get("handle_id")
    if not handle_id:
        return _err(HTTPStatus.BAD_REQUEST, "missing handle_id")
    children = body.get("children")
    if not isinstance(children, list) or not children:
        return _err(HTTPStatus.BAD_REQUEST, "children must be a non-empty list")
    try:
        child_specs = _parse_child_specs(children)
    except ValueError as e:
        return _err(HTTPStatus.BAD_REQUEST, str(e))

    per_child_timeout_s = body.get("per_child_timeout_s")
    if per_child_timeout_s is not None:
        try:
            per_child_timeout_s = float(per_child_timeout_s)
        except (TypeError, ValueError) as e:
            return _err(
                HTTPStatus.BAD_REQUEST,
                f"per_child_timeout_s must be numeric: {e}",
            )

    # (1) pin-check + source the prefix by handle_id alone. Presence in the pin
    # registry IS the pin-check: /tq/unpin pops the entry, so a released handle
    # raises KeyError here (-> 404), exactly like /tq/fork.
    try:
        handle = await engine.get_pin_handle(handle_id)
    except KeyError as e:
        return _err(HTTPStatus.NOT_FOUND, f"unknown pin handle: {e}")
    except Exception as e:  # noqa: BLE001
        logger.exception("tq_fork2: get_pin_handle failed")
        return _err(
            HTTPStatus.INTERNAL_SERVER_ERROR, f"get_pin_handle failed: {e}"
        )

    prompt_token_ids = list(handle.get("prompt_token_ids") or [])
    if not prompt_token_ids:
        return _err(
            HTTPStatus.BAD_REQUEST,
            f"pin handle {handle_id!r} has no prompt_token_ids to fork from",
        )
    prefix_len = len(prompt_token_ids)
    # Default the child cache_salt to the donor's (what the pinned blocks were
    # hashed with) so children hash-match and cache-hit; allow explicit override.
    cache_salt = body.get("cache_salt", handle.get("cache_salt"))

    # (2) Optional residency assertion (diagnostic; non-fatal). If the pinned
    # blocks were evicted, children still produce CORRECT output -- they just
    # re-prefill -- so this only warns and reports, never fails the fork.
    pin_resident: bool | None = None
    if body.get("verify_pin", True):
        try:
            v = await engine.verify_pinned_blocks(handle_id)
            pin_resident = bool(v.get("ok"))
            if not pin_resident:
                logger.warning(
                    "tq_fork2: pinned blocks for %s not fully resident "
                    "(min_ref_cnt=%s); children will re-prefill the prefix",
                    handle_id,
                    v.get("min_ref_cnt"),
                )
        except Exception:  # noqa: BLE001
            logger.exception("tq_fork2: verify_pinned_blocks failed (continuing)")

    # (3) Fan out: each child is an ordinary generate() request riding the pinned
    # prefix. output_kind=FINAL_ONLY -> generate() yields only the final
    # cumulative RequestOutput (no per-token deltas to accumulate).
    async def _run_child(i: int, spec: dict) -> dict:
        req_id = f"tqfork2-{uuid.uuid4().hex[:12]}-{i}"
        sampling = SamplingParams(**spec)
        sampling.output_kind = RequestOutputKind.FINAL_ONLY
        prompt = TokensPrompt(prompt_token_ids=list(prompt_token_ids))
        if cache_salt is not None:
            # TokensPrompt is a TypedDict; cache_salt is a NotRequired field.
            prompt["cache_salt"] = cache_salt
        final = None
        async for out in engine.generate(prompt, sampling, req_id):
            final = out
        if final is None or not final.outputs:
            return {
                "req_id": req_id,
                "text": None,
                "token_ids": [],
                "num_output_tokens": 0,
                "num_cached_tokens": 0,
                "finish_reason": "empty",
                "stop_reason": None,
                "spec": spec,
            }
        co = final.outputs[0]
        tok = list(co.token_ids or [])
        return {
            "req_id": req_id,
            "text": co.text,
            "token_ids": tok,
            "num_output_tokens": len(tok),
            "num_cached_tokens": int(final.num_cached_tokens or 0),
            "finish_reason": co.finish_reason,
            "stop_reason": co.stop_reason,
            "spec": spec,
        }

    async def _guarded(i: int, spec: dict):
        coro = _run_child(i, spec)
        if per_child_timeout_s is not None:
            coro = asyncio.wait_for(coro, per_child_timeout_s)
        return await coro

    results = await asyncio.gather(
        *[_guarded(i, s) for i, s in enumerate(child_specs)],
        return_exceptions=True,
    )

    children_out: list[dict] = []
    for i, (spec, r) in enumerate(zip(child_specs, results)):
        if isinstance(r, (asyncio.TimeoutError, TimeoutError)):
            children_out.append(
                {
                    "req_id": None,
                    "text": None,
                    "token_ids": [],
                    "num_output_tokens": 0,
                    "num_cached_tokens": 0,
                    "finish_reason": "timeout",
                    "stop_reason": None,
                    "spec": spec,
                    "error": "per_child_timeout",
                }
            )
        elif isinstance(r, BaseException):
            logger.error("tq_fork2 child %d failed", i, exc_info=r)
            children_out.append(
                {
                    "req_id": None,
                    "text": None,
                    "token_ids": [],
                    "num_output_tokens": 0,
                    "num_cached_tokens": 0,
                    "finish_reason": "error",
                    "stop_reason": None,
                    "spec": spec,
                    "error": f"{type(r).__name__}: {r}",
                }
            )
        else:
            children_out.append(r)

    resp: dict = {
        "handle_id": handle_id,
        "n": len(children_out),
        "prefix_len": prefix_len,
        "children": children_out,
    }
    if pin_resident is not None:
        resp["pin_resident"] = pin_resident
    return JSONResponse(content=resp)


@router.post("/tq/unpin")
async def tq_unpin(raw_request: Request) -> JSONResponse:
    """Release a pin handle (frees its pinned blocks). Idempotent."""
    engine = engine_client(raw_request)
    try:
        body = await raw_request.json()
    except Exception:  # noqa: BLE001
        return _err(HTTPStatus.BAD_REQUEST, "request body must be valid JSON")
    handle_id = (body or {}).get("handle_id") if isinstance(body, dict) else None
    if not handle_id:
        return _err(HTTPStatus.BAD_REQUEST, "missing handle_id")
    try:
        result = await engine.unpin_kv_blocks(handle_id)
    except Exception as e:  # noqa: BLE001
        logger.exception("tq_unpin failed")
        return _err(HTTPStatus.INTERNAL_SERVER_ERROR, f"unpin failed: {e}")
    return JSONResponse(content=result)


def attach_router(app: FastAPI) -> None:
    """Attach the /tq/* routes -- ONLY when VLLM_TQ_GDN_SNAPSHOT=1.

    Default-inert: a no-op on a normally-configured server, so wiring this into
    build_app cannot affect the production :8001 serve.
    """
    if not envs.VLLM_TQ_GDN_SNAPSHOT:
        return
    app.include_router(router)
    logger.warning(
        "EXP-038 TQ snapshot routes ENABLED (/tq/pin, /tq/fork, /tq/fork2, "
        "/tq/unpin). This is a throwaway-engine feature -- NEVER run it "
        "against :8001."
    )
