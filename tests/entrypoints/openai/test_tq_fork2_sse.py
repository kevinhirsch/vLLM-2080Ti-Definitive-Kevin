# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXP-038 fork v2 — per-child SSE streaming tests for POST /tq/fork2.

Contract under test (``"stream": true`` in the request body):

- The route returns a *streaming* response (``text/event-stream``), NOT a
  JSONResponse, framed as SSE: ``event: <type>\\ndata: <json>\\n\\n``.
- Event sequence: one ``fork_start`` {handle_id, n, prefix_len, pin_resident?};
  per child i: ``child_start`` {i, req_id, spec} -> zero or more
  ``child_delta`` {i, text, token_ids} -> exactly one ``child_done``
  {i, finish_reason, num_output_tokens, num_cached_tokens}; one terminal
  ``fork_done`` {n, ok_count, error_count}. Child event groups may interleave
  across children, but within a child the order start < deltas < done holds.
- Streaming children run with ``output_kind == DELTA`` (per-token deltas);
  the non-stream path keeps ``FINAL_ONLY`` (regression-pinned here).
- A child whose generate() raises produces ``child_done`` with
  ``finish_reason == "error"`` (no exception escapes, other children finish).
- Validation errors (unknown handle, bad body) still return plain JSON error
  responses even when ``stream: true`` is requested.

Written by the supervisor as the trusted gate; the implementation is produced
by the local-model coder loop against this file.
"""
import asyncio
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
R = pytest.importorskip("vllm.entrypoints.openai.tq_snapshot_router")

from vllm.sampling_params import RequestOutputKind  # noqa: E402


class _Stub:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeEngine:
    """Streams each child's text in two delta chunks, then a final chunk.

    Captures the SamplingParams handed to generate() so tests can pin
    output_kind per mode.
    """

    def __init__(self, handle=None, resident=True, raise_on=None,
                 deltas_per_child=2):
        self._handle = handle
        self._resident = resident
        self._raise_on = raise_on or set()
        self._deltas = deltas_per_child
        self.calls = []
        self.sampling_by_idx = {}

    async def get_pin_handle(self, handle_id):
        if self._handle is None:
            raise KeyError(handle_id)
        return self._handle

    async def verify_pinned_blocks(self, handle_id):
        return {"ok": self._resident, "min_ref_cnt": 2 if self._resident else 0}

    async def generate(self, prompt, sampling, request_id):
        idx = int(request_id.rsplit("-", 1)[1])
        self.calls.append((request_id, list(prompt["prompt_token_ids"]),
                           prompt.get("cache_salt")))
        self.sampling_by_idx[idx] = sampling
        if idx in self._raise_on:
            raise RuntimeError(f"boom-{idx}")
        await asyncio.sleep(0)  # yield control so children interleave
        if sampling.output_kind == RequestOutputKind.DELTA:
            for d in range(self._deltas):
                co = _Stub(text=f"c{idx}d{d} ", token_ids=[100 * idx + d],
                           finish_reason=None, stop_reason=None)
                yield _Stub(outputs=[co], num_cached_tokens=2100,
                            finished=False)
                await asyncio.sleep(0)
            co = _Stub(text=f"c{idx}fin", token_ids=[900 + idx],
                       finish_reason="stop", stop_reason=None)
            yield _Stub(outputs=[co], num_cached_tokens=2100, finished=True)
        else:
            co = _Stub(text=f"child-{idx}-text",
                       token_ids=[100 + idx, 200 + idx],
                       finish_reason="stop", stop_reason=None)
            yield _Stub(outputs=[co], num_cached_tokens=2100, finished=True)


class FakeState:
    def __init__(self, engine):
        self.engine_client = engine


class FakeApp:
    def __init__(self, engine):
        self.state = FakeState(engine)


class FakeRequest:
    def __init__(self, engine, body):
        self.app = FakeApp(engine)
        self._body = body

    async def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


_HANDLE = {
    "handle_id": "h1",
    "req_id": "src-1",
    "prompt_token_ids": [1, 2, 3, 4, 5],
    "prefix_len": 5,
    "num_computed_tokens": 5,
    "cache_salt": None,
}


async def _drain_sse(resp):
    """Collect (event, payload) tuples from a streaming response."""
    chunks = []
    async for chunk in resp.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    raw = "".join(chunks)
    events = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        etype, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                etype = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
        assert etype is not None, f"SSE block missing event type: {block!r}"
        assert data is not None, f"SSE block missing data: {block!r}"
        events.append((etype, data))
    return events


def _call_stream(engine, body):
    req = FakeRequest(engine, body)

    async def run():
        resp = await R.tq_fork2(req)
        return resp, await _drain_sse(resp)

    return asyncio.run(run())


def _call_json(engine, body):
    req = FakeRequest(engine, body)
    resp = asyncio.run(R.tq_fork2(req))
    return resp.status_code, json.loads(resp.body)


def test_stream_returns_event_stream_not_json():
    resp, events = _call_stream(
        FakeEngine(handle=_HANDLE),
        {"handle_id": "h1", "stream": True,
         "children": [{"temperature": 0.0}]})
    assert "text/event-stream" in resp.media_type
    assert events, "no SSE events emitted"


def test_stream_event_sequence_single_child():
    _, events = _call_stream(
        FakeEngine(handle=_HANDLE),
        {"handle_id": "h1", "stream": True,
         "children": [{"temperature": 0.5, "max_tokens": 8}]})
    types = [e for e, _ in events]
    assert types[0] == "fork_start"
    assert types[-1] == "fork_done"
    # exactly one child: start, 2 deltas, done — in order
    assert types[1:-1] == ["child_start", "child_delta", "child_delta",
                           "child_done"]
    start = dict(events)["fork_start"]
    assert start["handle_id"] == "h1" and start["prefix_len"] == 5
    assert start["n"] == 1
    done = [d for e, d in events if e == "child_done"][0]
    assert done["finish_reason"] == "stop"
    assert done["num_cached_tokens"] == 2100
    deltas = [d for e, d in events if e == "child_delta"]
    assert [d["text"] for d in deltas] == ["c0d0 ", "c0d1 "]
    assert all(d["i"] == 0 for d in deltas)
    fdone = events[-1][1]
    assert fdone["ok_count"] == 1 and fdone["error_count"] == 0


def test_stream_per_child_order_holds_with_three_children():
    _, events = _call_stream(
        FakeEngine(handle=_HANDLE),
        {"handle_id": "h1", "stream": True,
         "children": [{}, {}, {}]})
    for i in range(3):
        seq = [e for e, d in events if d.get("i") == i]
        assert seq[0] == "child_start" and seq[-1] == "child_done"
        assert seq.count("child_done") == 1
        # no event for child i after its child_done
        di = max(k for k, (e, d) in enumerate(events) if d.get("i") == i)
        assert events[di][0] == "child_done"
    assert events[-1][0] == "fork_done"
    assert events[-1][1]["n"] == 3


def test_stream_uses_delta_output_kind_nonstream_stays_final_only():
    eng = FakeEngine(handle=_HANDLE)
    _call_stream(eng, {"handle_id": "h1", "stream": True, "children": [{}]})
    assert eng.sampling_by_idx[0].output_kind == RequestOutputKind.DELTA
    eng2 = FakeEngine(handle=_HANDLE)
    status, _ = _call_json(eng2, {"handle_id": "h1", "children": [{}]})
    assert status == 200
    assert eng2.sampling_by_idx[0].output_kind == RequestOutputKind.FINAL_ONLY


def test_stream_child_error_isolated():
    _, events = _call_stream(
        FakeEngine(handle=_HANDLE, raise_on={0}),
        {"handle_id": "h1", "stream": True, "children": [{}, {}]})
    dones = {d["i"]: d for e, d in events if e == "child_done"}
    assert dones[0]["finish_reason"] == "error"
    assert "boom-0" in dones[0].get("error", "")
    assert dones[1]["finish_reason"] == "stop"
    fdone = events[-1][1]
    assert fdone["ok_count"] == 1 and fdone["error_count"] == 1


def test_stream_per_child_timeout_yields_timeout_done():
    class StallEngine(FakeEngine):
        async def generate(self, prompt, sampling, request_id):
            idx = int(request_id.rsplit("-", 1)[1])
            self.sampling_by_idx[idx] = sampling
            co = _Stub(text="d0 ", token_ids=[1], finish_reason=None,
                       stop_reason=None)
            yield _Stub(outputs=[co], num_cached_tokens=2100, finished=False)
            await asyncio.sleep(30)  # stall far beyond the timeout

    _, events = _call_stream(
        StallEngine(handle=_HANDLE),
        {"handle_id": "h1", "stream": True, "per_child_timeout_s": 0.05,
         "children": [{}]})
    done = [d for e, d in events if e == "child_done"][0]
    assert done["finish_reason"] == "timeout"
    assert done.get("error") == "per_child_timeout"
    fdone = events[-1][1]
    assert fdone["ok_count"] == 0 and fdone["error_count"] == 1


def test_stream_validation_errors_stay_json():
    status, payload = _call_json(
        FakeEngine(handle=None),
        {"handle_id": "gone", "stream": True, "children": [{}]})
    assert status == 404 and "unknown pin handle" in payload["error"]
    resp = asyncio.run(R.tq_fork2(FakeRequest(FakeEngine(handle=_HANDLE),
                                              {"stream": True,
                                               "children": [{}]})))
    assert json.loads(resp.body)["error"]  # missing handle_id -> JSON 400
