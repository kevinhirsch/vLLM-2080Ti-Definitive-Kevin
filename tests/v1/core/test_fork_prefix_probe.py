# SPDX-License-Identifier: Apache-2.0
"""[FORK][LANE GW2] Read-only prefix-cache probe the gateway routes on.

Drives the REAL Scheduler + KVCacheManager (hybrid full-attention + Mamba "align",
CPU only, helpers shared with CR2's prefix-aware short-first tests): the probe equals the
admission hit, never takes block references or adds a request, is off by default, and
retries a lookup that races a concurrent cache update. The HTTP router is driven with a
fake engine client and tokenizer.
"""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import vllm.v1.engine.core as core_mod
from tests.v1.core.test_short_first_prefix_aware import (
    BLOCK,
    COLD,
    SESSION,
    _build,
    _request,
    _step,
)
from vllm.entrypoints.serve.tokenize.protocol import TokenizeResponse
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher

pytestmark = pytest.mark.cpu_test


def _core(monkeypatch):
    s = _build(monkeypatch, prefix_aware=False)
    t1 = _request("turn1", SESSION)
    s.add_request(t1)
    for _ in range(40):
        _step(s)
        if t1.is_finished():
            break
    assert t1.is_finished()
    return SimpleNamespace(
        scheduler=s, request_block_hasher=get_request_block_hasher(BLOCK, sha256)
    )


def test_probe_equals_admission_hit_and_is_read_only(monkeypatch):
    monkeypatch.setattr(core_mod, "FORK_PREFIX_PROBE", True)
    core = _core(monkeypatch)
    s = core.scheduler
    free = s.kv_cache_manager.block_pool.get_num_free_blocks()
    n_req = len(s.requests)
    t2 = SESSION + [3] * 150
    out = core_mod.fork_prefix_probe(core, t2)
    assert out["enabled"] and out["prompt_tokens"] == len(t2)
    assert out["cached_tokens"] == 6 * BLOCK
    _, admitted, _ = s.kv_cache_manager.get_computed_blocks(_request("turn2", t2))
    assert admitted == out["cached_tokens"]
    assert s.kv_cache_manager.block_pool.get_num_free_blocks() == free
    assert len(s.requests) == n_req  # the throwaway request is never added
    assert core_mod.fork_prefix_probe(core, COLD)["cached_tokens"] == 0
    assert core_mod.fork_prefix_probe(core, [])["cached_tokens"] == 0


def test_default_off(monkeypatch):
    monkeypatch.setattr(core_mod, "FORK_PREFIX_PROBE", False)
    assert core_mod.fork_prefix_probe(SimpleNamespace(), [1, 2, 3]) == {"enabled": False}


def test_concurrent_update_race_is_retried(monkeypatch):
    monkeypatch.setattr(core_mod, "FORK_PREFIX_PROBE", True)
    core = _core(monkeypatch)
    real = core.scheduler.kv_cache_manager.probe_prefix_cache_hit
    calls = {"n": 0}

    def flaky(req, fail=2):
        calls["n"] += 1
        if calls["n"] <= fail:
            raise RuntimeError("dictionary changed size during iteration")
        return real(req)

    monkeypatch.setattr(core.scheduler.kv_cache_manager, "probe_prefix_cache_hit", flaky)
    assert core_mod.fork_prefix_probe(core, SESSION + [3])["cached_tokens"] == 6 * BLOCK
    calls["n"] = -10
    monkeypatch.setattr(
        core.scheduler.kv_cache_manager,
        "probe_prefix_cache_hit",
        lambda req: flaky(req, fail=99),
    )
    with pytest.raises(RuntimeError, match="raced"):
        core_mod.fork_prefix_probe(core, SESSION)


def _app(tokens, result):
    from vllm.entrypoints.serve.fork_probe.api_router import attach_router

    seen = {}

    async def create_tokenize(req, raw):
        seen["req"] = req
        return TokenizeResponse(tokens=tokens, count=len(tokens), max_model_len=1 << 19)

    async def call_utility_async(method, ids, salt):
        seen["call"] = (method, list(ids), salt)
        return result

    app = FastAPI()
    app.state.serving_tokenization = SimpleNamespace(create_tokenize=create_tokenize)
    app.state.engine_client = SimpleNamespace(
        engine_core=SimpleNamespace(call_utility_async=call_utility_async)
    )
    attach_router(app)
    return TestClient(app), seen


def test_router_renders_chat_bodies_like_tokenize_and_ignores_extra_fields():
    c, seen = _app([5] * 10, {"enabled": True, "prompt_tokens": 10, "cached_tokens": 4})
    r = c.post(
        "/v1/fork/prefix_cache_probe",
        json={
            "model": "estate",
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"enable_thinking": True},
            "stream": True,
            "max_tokens": 9,
        },
    )
    assert r.status_code == 200, r.text
    j = r.json()
    assert (j["cached_tokens"], j["uncached_tokens"]) == (4, 6)
    assert seen["call"] == (core_mod.FORK_PROBE_METHOD, [5] * 10, None)
    assert seen["req"].chat_template_kwargs == {"enable_thinking": True}


def test_router_token_ids_and_errors():
    c, seen = _app([], {"enabled": True, "prompt_tokens": 3, "cached_tokens": 0})
    r = c.post("/v1/fork/prefix_cache_probe", json={"prompt_token_ids": [1, 2, 3], "cache_salt": "s"})
    assert r.json()["uncached_tokens"] == 3 and seen["call"][2] == "s"
    assert "req" not in seen  # no render for token ids
    assert c.post("/v1/fork/prefix_cache_probe", json={"x": 1}).status_code == 400
    assert c.post("/v1/fork/prefix_cache_probe", json={"prompt_token_ids": "no"}).status_code == 400
    assert c.post("/v1/fork/prefix_cache_probe", content=b"[1]").status_code == 400
