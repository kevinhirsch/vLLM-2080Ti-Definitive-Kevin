# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXP-038 fork v2 — route-level tests for POST /tq/fork2.

Exercises the server-layer fan-out logic (pin-check via get_pin_handle, optional
residency verify, N parallel generate() calls, response assembly, per-child error
isolation) with a FAKE engine client + FAKE Request — no GPU, no real model. Skips
automatically where the vllm/torch/fastapi import graph is unavailable (e.g. a
bare source checkout); the hardware e2e demo lives in tools/tq_gdn_snapshot_stage6_fork2.py.
"""
import asyncio
import json

import pytest

# The router imports fastapi + vllm (torch). Skip the whole module if the import
# graph isn't available in this environment.
pytest.importorskip("torch")
pytest.importorskip("fastapi")
R = pytest.importorskip("vllm.entrypoints.openai.tq_snapshot_router")


class _Stub:
    """Minimal stand-in for CompletionOutput/RequestOutput (duck-typed to the
    fields tq_fork2 reads)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeEngine:
    def __init__(self, handle=None, resident=True, raise_on=None):
        # handle=None -> unknown handle (get_pin_handle raises KeyError).
        self._handle = handle
        self._resident = resident
        self._raise_on = raise_on or set()  # child indices whose generate raises
        self.calls = []

    async def get_pin_handle(self, handle_id):
        if self._handle is None:
            raise KeyError(handle_id)
        return self._handle

    async def verify_pinned_blocks(self, handle_id):
        return {"ok": self._resident, "min_ref_cnt": 2 if self._resident else 0}

    async def generate(self, prompt, sampling, request_id):
        # request_id ends in -<i>; use it to drive per-child behavior.
        idx = int(request_id.rsplit("-", 1)[1])
        self.calls.append((request_id, list(prompt["prompt_token_ids"]),
                           prompt.get("cache_salt")))
        if idx in self._raise_on:
            raise RuntimeError(f"boom-{idx}")
        # Yield a single FINAL_ONLY-style cumulative output.
        co = _Stub(text=f"child-{idx}-text", token_ids=[100 + idx, 200 + idx],
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


def _call(engine, body):
    req = FakeRequest(engine, body)
    resp = asyncio.run(R.tq_fork2(req))
    return resp.status_code, json.loads(resp.body)


_HANDLE = {
    "handle_id": "h1",
    "req_id": "src-1",
    "prompt_token_ids": [1, 2, 3, 4, 5],
    "prefix_len": 5,
    "num_computed_tokens": 5,
    "cache_salt": None,
}


def test_missing_handle_id_400():
    status, payload = _call(FakeEngine(handle=_HANDLE), {"children": [{}]})
    assert status == 400 and "handle_id" in payload["error"]


def test_empty_children_400():
    status, payload = _call(FakeEngine(handle=_HANDLE),
                            {"handle_id": "h1", "children": []})
    assert status == 400 and "children" in payload["error"]


def test_unsupported_child_key_400():
    status, payload = _call(
        FakeEngine(handle=_HANDLE),
        {"handle_id": "h1", "children": [{"bogus": 1}]})
    assert status == 400 and "unsupported key" in payload["error"]


def test_unknown_handle_404():
    status, payload = _call(FakeEngine(handle=None),
                            {"handle_id": "gone", "children": [{}]})
    assert status == 404 and "unknown pin handle" in payload["error"]


def test_happy_path_fanout():
    engine = FakeEngine(handle=_HANDLE, resident=True)
    status, payload = _call(
        engine,
        {"handle_id": "h1",
         "children": [{"temperature": 0.0}, {"temperature": 0.7, "max_tokens": 8}]})
    assert status == 200
    assert payload["n"] == 2
    assert payload["prefix_len"] == 5
    assert payload["pin_resident"] is True
    kids = payload["children"]
    assert [k["text"] for k in kids] == ["child-0-text", "child-1-text"]
    assert all(k["num_cached_tokens"] == 2100 for k in kids)
    assert all(k["finish_reason"] == "stop" for k in kids)
    assert all(k["num_output_tokens"] == 2 for k in kids)
    # Each child was submitted with the FULL pinned prefix as ordinary tokens.
    assert all(toks == [1, 2, 3, 4, 5] for _, toks, _ in engine.calls)


def test_per_child_error_isolated():
    engine = FakeEngine(handle=_HANDLE, raise_on={1})
    status, payload = _call(
        engine,
        {"handle_id": "h1", "children": [{}, {}, {}]})
    assert status == 200 and payload["n"] == 3
    kids = payload["children"]
    assert kids[0]["finish_reason"] == "stop"
    assert kids[1]["finish_reason"] == "error" and "boom-1" in kids[1]["error"]
    assert kids[2]["finish_reason"] == "stop"


def test_verify_pin_skipped_when_disabled():
    engine = FakeEngine(handle=_HANDLE, resident=False)
    status, payload = _call(
        engine,
        {"handle_id": "h1", "children": [{}], "verify_pin": False})
    assert status == 200
    assert "pin_resident" not in payload  # verify was not called
