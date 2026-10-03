# SPDX-License-Identifier: Apache-2.0
"""[FORK][LANE GW2 2026-10-03] POST /v1/fork/prefix_cache_probe -- how many tokens of
this request the engine would serve from its prefix cache RIGHT NOW. Read-only.

Registered only when VLLM_FORK_PREFIX_PROBE=1. The body is either
``{"prompt_token_ids": [...]}`` or the fields of a chat/completions request
(``messages``, ``tools``, ``chat_template_kwargs``, ... or ``prompt``); the latter is
rendered and tokenized by the SAME path as ``/tokenize``, so ``prompt_tokens`` equals the
engine's ``usage.prompt_tokens`` for that request. Response::

    {"enabled": true, "prompt_tokens": N, "cached_tokens": C, "uncached_tokens": N - C,
     "probe_ms": ..., "render_ms": ..., "total_ms": ...}

The gateway uses it to decide local vs remote on the engine's real cache state instead of
its own credit model. Advisory: blocks can be evicted or cached before admission.
"""

import time

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.tokenize.protocol import (
    TokenizeChatRequest,
    TokenizeCompletionRequest,
)
from vllm.logger import init_logger
from vllm.v1.engine.core import FORK_PROBE_METHOD

logger = init_logger(__name__)

router = APIRouter()

_CHAT_FIELDS = (
    "model",
    "messages",
    "tools",
    "chat_template_kwargs",
    "add_generation_prompt",
    "continue_final_message",
    "add_special_tokens",
    "chat_template",
)
_COMPLETION_FIELDS = ("model", "prompt", "add_special_tokens")


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message}})


@router.post("/v1/fork/prefix_cache_probe")
async def prefix_cache_probe(raw_request: Request):
    t0 = time.perf_counter()
    try:
        body = await raw_request.json()
    except Exception:
        return _error(400, "body must be JSON")
    if not isinstance(body, dict):
        return _error(400, "body must be a JSON object")
    ids = body.get("prompt_token_ids")
    render_ms = 0.0
    if ids is None:
        try:
            if "messages" in body:
                req = TokenizeChatRequest(
                    **{k: body[k] for k in _CHAT_FIELDS if k in body}
                )
            elif "prompt" in body:
                req = TokenizeCompletionRequest(
                    **{k: body[k] for k in _COMPLETION_FIELDS if k in body}
                )
            else:
                return _error(400, "need prompt_token_ids, messages or prompt")
        except Exception as e:
            return _error(400, f"invalid request: {e}")
        t_r = time.perf_counter()
        tok = await raw_request.app.state.serving_tokenization.create_tokenize(
            req, raw_request
        )
        if isinstance(tok, ErrorResponse):
            return JSONResponse(content=tok.model_dump(), status_code=tok.error.code)
        ids = tok.tokens
        render_ms = (time.perf_counter() - t_r) * 1000
    elif not isinstance(ids, list):
        return _error(400, "prompt_token_ids must be a list of ints")
    client = raw_request.app.state.engine_client
    try:
        res = await client.engine_core.call_utility_async(
            FORK_PROBE_METHOD, ids, body.get("cache_salt")
        )
    except Exception as e:
        logger.warning("fork prefix probe failed: %s", e)
        return _error(503, f"probe failed: {e}")
    res = dict(res or {})
    if res.get("enabled"):
        res["uncached_tokens"] = max(
            0, int(res.get("prompt_tokens", 0)) - int(res.get("cached_tokens", 0))
        )
    res["render_ms"] = round(render_ms, 2)
    res["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return JSONResponse(content=res)


def attach_router(app: FastAPI):
    app.include_router(router)
