# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXP-038 (VLLM_TQ_GDN_SNAPSHOT) per-child SSE streaming for /tq/fork2.

Implements the ``stream: true`` path of POST /tq/fork2: fan out N children as
ordinary ``engine.generate`` requests over a pinned prefix and stream per-child
SSE events. The router (``tq_snapshot_router.tq_fork2``) performs validation,
pin-check, and residency verification before calling :func:`fork2_sse_response`.

Event contract (authoritative spec: tests/entrypoints/openai/test_tq_fork2_sse.py):

* ``fork_start`` first: ``{"handle_id", "n", "prefix_len"}`` (+ ``pin_resident``
  only when not None).
* Per child i (groups may interleave across children; within a child the order
  is strict): ``child_start`` -> zero+ ``child_delta`` -> exactly one
  ``child_done``.
* Terminal ``fork_done``: ``{"n", "ok_count", "error_count"}`` then close.
"""
import asyncio
import json
import uuid

from fastapi.responses import StreamingResponse

from vllm import TokensPrompt
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind, SamplingParams

logger = init_logger(__name__)


def fork2_sse_response(
    *,
    engine,
    handle_id,
    child_specs,
    prompt_token_ids,
    cache_salt,
    prefix_len,
    pin_resident,
    per_child_timeout_s,
) -> StreamingResponse:
    """Build a ``StreamingResponse`` that streams per-child SSE events.

    Parameters mirror the validated state extracted by the router before this
    function is called. Each child runs as an independent asyncio task so that
    a failure or timeout in one child does not block the others.
    """
    n = len(child_specs)
    queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()

    async def _run_child(i: int) -> None:
        req_id = f"tqfork2-{uuid.uuid4().hex[:12]}-{i}"
        spec = child_specs[i]
        await queue.put(("child_start", {"i": i, "req_id": req_id, "spec": spec}))

        sampling = SamplingParams(**spec)
        sampling.output_kind = RequestOutputKind.DELTA

        prompt = TokensPrompt(prompt_token_ids=list(prompt_token_ids))
        if cache_salt is not None:
            prompt["cache_salt"] = cache_salt

        num_output_tokens = 0
        num_cached_tokens = 0
        finish_reason: str | None = None
        stop_reason: str | None = None
        error_msg: str | None = None

        try:
            # The timeout (when set) wraps each __anext__ individually: it is a
            # per-chunk stall bound, applied in the loop below via wait_for.
            gen = engine.generate(prompt, sampling, req_id)
            while True:
                if per_child_timeout_s is not None:
                    try:
                        chunk = await asyncio.wait_for(gen.__anext__(), per_child_timeout_s)
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        finish_reason = "timeout"
                        error_msg = "per_child_timeout"
                        break
                else:
                    try:
                        chunk = await gen.__anext__()
                    except StopAsyncIteration:
                        break

                co = chunk.outputs[0]
                cached = getattr(chunk, "num_cached_tokens", 0)
                if cached:
                    num_cached_tokens = cached

                if co.finish_reason is None:
                    # Delta chunk: emit if it carries text or token_ids
                    if co.text or co.token_ids:
                        num_output_tokens += len(co.token_ids) if co.token_ids else 0
                        await queue.put((
                            "child_delta",
                            {
                                "i": i,
                                "text": co.text or "",
                                "token_ids": list(co.token_ids) if co.token_ids else [],
                            },
                        ))
                else:
                    # Final chunk: fold into child_done
                    if co.token_ids:
                        num_output_tokens += len(co.token_ids)
                    finish_reason = co.finish_reason
                    stop_reason = co.stop_reason
                    break
        except Exception as e:
            finish_reason = "error"
            error_msg = f"{type(e).__name__}: {e}"
            logger.exception("tq_fork2: child %d generate raised", i)

        done_payload: dict = {
            "i": i,
            "finish_reason": finish_reason,
            "stop_reason": stop_reason,
            "num_output_tokens": num_output_tokens,
            "num_cached_tokens": num_cached_tokens,
        }
        if error_msg is not None:
            done_payload["error"] = error_msg
        await queue.put(("child_done", done_payload))

    async def _event_generator():
        # Emit fork_start first
        start_payload: dict = {
            "handle_id": handle_id,
            "n": n,
            "prefix_len": prefix_len,
        }
        if pin_resident is not None:
            start_payload["pin_resident"] = pin_resident
        yield f"event: fork_start\ndata: {json.dumps(start_payload)}\n\n"

        # Launch all child tasks
        tasks = [asyncio.create_task(_run_child(i)) for i in range(n)]

        # Drain the queue until all child_done events have been seen
        done_count = 0
        ok_count = 0
        error_count = 0
        while done_count < n:
            event_type, payload = await queue.get()
            if event_type == "child_done":
                done_count += 1
                fr = payload.get("finish_reason")
                if fr in ("error", "timeout"):
                    error_count += 1
                else:
                    ok_count += 1
            yield f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"

        # Ensure all tasks are complete (they should be, since they pushed
        # their child_done before returning)
        await asyncio.gather(*tasks, return_exceptions=True)

        # Emit terminal fork_done
        fork_done_payload = {"n": n, "ok_count": ok_count, "error_count": error_count}
        yield f"event: fork_done\ndata: {json.dumps(fork_done_payload)}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
    )
