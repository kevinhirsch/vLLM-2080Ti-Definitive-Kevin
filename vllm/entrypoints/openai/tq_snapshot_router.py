# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXP-038 (VLLM_TQ_GDN_SNAPSHOT) custom HTTP routes: "prefill once, fork N".

Exposes the hardware-proven Stage-1/Stage-4 fork chain (pin a finished prefix's
KV blocks, then fork N cheap continuations that ride that shared prefix) over
three HTTP routes on the OpenAI server:

    POST /tq/pin    {prompt|messages|prompt_token_ids, ...} -> {handle_id, ...}
    POST /tq/fork   {handle_id, children:[{temperature,max_tokens,...}]} -> {...}
    POST /tq/unpin  {handle_id} -> {ok, num_freed_blocks}

Design notes / honest limitations (full write-up: docs/exp038-http-routes.md):

* The pin utility must catch a LIVE request (it ``touch``-pins the resident KV
  blocks of an in-flight request so they outlive its ``free()``). So /tq/pin
  SUBMITS the prefill itself: it fires a tiny keepalive generation of the shared
  context and, while that request is in flight, polls ``pin_request_kv_blocks``
  (req_id=None -> the single running request) until the full prefix is computed,
  then aborts the keepalive. This is the async, server-side reincarnation of the
  Stage-2 ``_run_and_pin`` pattern. It assumes a QUIESCENT snapshot engine (one
  request at a time); req_id=None picks ``running[0]`` and would be unreliable
  under concurrent traffic -- which is exactly why this is throwaway-engine-only.

* /tq/fork drives the children to completion INSIDE the EngineCore utility (the
  single-threaded busy loop is BLOCKED for the whole fork). Cap
  ``sum(child.max_tokens)`` (``max_total_tokens``, default 2048). The clean
  follow-up is a non-blocking async fork that admits children as normal requests
  and streams them back; that is deliberately out of scope for this staged step.

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
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)

router = APIRouter()

# Sampling kwargs a fork child may override. Everything else is rejected so a
# malformed body can't smuggle unexpected SamplingParams kwargs into the engine.
_CHILD_SAMPLING_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "max_tokens",
        "min_tokens",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "stop_token_ids",
        "ignore_eos",
    }
)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def _err(status: HTTPStatus, msg: str) -> JSONResponse:
    return JSONResponse(content={"error": msg}, status_code=status.value)


async def _resolve_prompt_token_ids(engine: EngineClient, body: dict) -> list[int]:
    """Turn the request body into prompt token ids. Accepts (in priority order)
    an explicit ``prompt_token_ids`` list, a chat ``messages`` list (rendered via
    the tokenizer's chat template), or a raw ``prompt`` string."""
    if body.get("prompt_token_ids"):
        return [int(t) for t in body["prompt_token_ids"]]
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
    keepalive_tokens = max(2, int(body.get("keepalive_tokens", 16)))
    timeout_s = float(body.get("timeout_s", 60.0))
    poll_s = float(body.get("poll_interval_s", 0.02))

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

    max_total_tokens = int(body.get("max_total_tokens", 2048))
    child_specs: list[dict] = []
    total_budget = 0
    for i, child in enumerate(children):
        if not isinstance(child, dict):
            return _err(HTTPStatus.BAD_REQUEST, f"children[{i}] must be an object")
        bad = set(child) - _CHILD_SAMPLING_KEYS
        if bad:
            return _err(
                HTTPStatus.BAD_REQUEST,
                f"children[{i}] has unsupported key(s): {sorted(bad)}; "
                f"allowed: {sorted(_CHILD_SAMPLING_KEYS)}",
            )
        spec = dict(child)
        spec["max_tokens"] = int(spec.get("max_tokens", 64))
        total_budget += spec["max_tokens"]
        child_specs.append(spec)

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
        "EXP-038 TQ snapshot routes ENABLED (/tq/pin, /tq/fork, /tq/unpin). "
        "This is a throwaway-engine feature -- NEVER run it against :8001."
    )
