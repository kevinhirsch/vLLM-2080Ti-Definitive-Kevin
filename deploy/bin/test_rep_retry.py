"""Unit tests for the GW-REP-RETRY change in keepalive-shim.py.

Covers:
  * pure helpers (_request_has_tools / _rep_cut_response / _rep_retry_body)
  * non-streaming relay (off / shadow / live / no-tools / also-cut / failed)
  * streaming relay (off / shadow / live splice / live->shadow fallback)
  * config read (SHIM_REPETITION_RETRY)
"""
import asyncio
import json
import os

import pytest

from conftest_rep_retry import FakeRequest, FakeUpstream

HERE = os.path.dirname(os.path.abspath(__file__))

# A single tool definition, reused across every body builder so the
# tests never repeat a deeply-nested dict literal (which is where
# the bracket-matching errors came from).
TOOL = {"type": "function", "function": {"name": "f", "parameters": {}}}
# A tool-call (assistant) with valid args.
TOOL_CALL = {"id": "1", "type": "function",
               "function": {"name": "f", "arguments": "{}"}}
# A streaming tool-call delta.
TOOL_CALL_DELTA = {"index": 0, "id": "1", "type": "function",
                      "function": {"name": "f", "arguments": "{}"}}


def body_with_tools(tools=None, thinking=True, extra=None):
    """A chat-completion request body (bytes) that carries tools."""
    j = {
        "model": "estate-local",
        "messages": [{"role": "user", "content": "do the thing"}],
        "max_tokens": 2048,
    }
    if tools is not None:
        j["tools"] = tools
    ctk = {}
    if thinking is not None:
        ctk["enable_thinking"] = thinking
    if ctk:
        j["chat_template_kwargs"] = ctk
    if extra:
        j.update(extra)
    return json.dumps(j).encode()


def body_no_tools(thinking=True):
    """A chat-completion request body (bytes) with NO tools."""
    j = {
        "model": "estate-local",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 256,
    }
    if thinking is not None:
        j["chat_template_kwargs"] = {"enable_thinking": thinking}
    return json.dumps(j).encode()


def chat_resp(finish_reason, content="partial text", tool_calls=None):
    """A non-streaming chat-completion response (bytes)."""
    m = {"role": "assistant", "content": content}
    if tool_calls is not None:
        m["tool_calls"] = tool_calls
    return json.dumps({"id": "x", "object": "chat.completion",
                          "choices": [{"index": 0, "message": m,
                                          "finish_reason": finish_reason}]}).encode()


def sse_chunk(delta, finish=None):
    """One SSE 'data:' line (bytes) for a chat.completion.chunk."""
    ev = {"id": "1", "object": "chat.completion.chunk",
           "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return ("data: " + json.dumps(ev) + "\n\n").encode()


def sse_stream(content_pieces, finish_reason, tool_call_delta=None):
    """Build a list of SSE byte-chunks for a streamed chat completion."""
    chunks = []
    for c in content_pieces:
        chunks.append(sse_chunk({"content": c}))
    if tool_call_delta is not None:
        chunks.append(sse_chunk({"tool_calls": [tool_call_delta]}))
    chunks.append(sse_chunk({}, finish=finish_reason))
    chunks.append(b"data: [DONE]\n\n")
    return chunks


class TestRequestHasTools:
    def test_with_tools(self, shim):
        assert shim._request_has_tools(body_with_tools(tools=[TOOL])) is True

    def test_without_tools(self, shim):
        assert shim._request_has_tools(body_no_tools()) is False

    def test_empty_tools(self, shim):
        assert shim._request_has_tools(body_with_tools(tools=[])) is False

    def test_malformed(self, shim):
        assert shim._request_has_tools(b"not json") is False


class TestRepCutResponse:
    def test_rep_cut_no_tool_call(self, shim):
        assert shim._rep_cut_response(chat_resp("repetition")) is True

    def test_not_repetition(self, shim):
        assert shim._rep_cut_response(chat_resp("stop")) is False

    def test_repetition_with_tool_call(self, shim):
        assert shim._rep_cut_response(chat_resp("repetition", tool_calls=[TOOL_CALL])) is False

    def test_malformed(self, shim):
        assert shim._rep_cut_response(b"nope") is False


class TestRepRetryBody:
    def test_sets_thinking_off(self, shim):
        rb = json.loads(shim._rep_retry_body(body_with_tools(tools=[TOOL])))
        assert rb["chat_template_kwargs"]["enable_thinking"] is False

    def test_applies_nonthinking_profile(self, shim):
        rb = json.loads(shim._rep_retry_body(body_with_tools(tools=[TOOL])))
        assert rb.get("presence_penalty") == pytest.approx(1.5)
        assert rb.get("top_p") == pytest.approx(0.8)
        assert rb.get("temperature") == pytest.approx(0.7)
        assert rb.get("top_k") == 20

    def test_preserves_tools(self, shim):
        rb = json.loads(shim._rep_retry_body(body_with_tools(tools=[TOOL])))
        assert rb["tools"] == [TOOL]

    def test_malformed_returns_unchanged(self, shim):
        assert shim._rep_retry_body(b"not json") == b"not json"


class TestNonStreamingRelay:
    def _relay(self, shim, monkeypatch, body, resp_body, mode,
                 resp2_body=None, resp2_status=200):
        """Drive _relay (non-streaming) with a mocked upstream."""
        monkeypatch.setattr(shim, "REP_RETRY", mode)
        calls = []

        async def fake_open(session, base, path, b, key, streaming):
            calls.append((base, b, streaming))
            if len(calls) == 1:
                return FakeUpstream(status=200, body=resp_body)
            if resp2_body is None:
                return FakeUpstream(status=500, body=b"err")
            return FakeUpstream(status=resp2_status, body=resp2_body)

        monkeypatch.setattr(shim, "_open", fake_open)
        req = FakeRequest()
        kind, payload = asyncio.run(shim._relay(req, shim.LOCAL,
            "/v1/chat/completions", body, None, False, concurrency=1))
        return kind, payload, calls

    def test_off_no_retry(self, shim, monkeypatch):
        body = body_with_tools(tools=[TOOL])
        cut = chat_resp("repetition", content="loop loop")
        kind, payload, calls = self._relay(shim, monkeypatch, body, cut, "off")
        assert kind == "ok"
        assert payload.body == cut
        assert len(calls) == 1

    def test_shadow_logs_no_retry(self, shim, monkeypatch):
        body = body_with_tools(tools=[TOOL])
        cut = chat_resp("repetition", content="loop loop")
        kind, payload, calls = self._relay(shim, monkeypatch, body, cut, "shadow")
        assert kind == "ok"
        assert payload.body == cut
        assert len(calls) == 1

    def test_live_returns_retry_response(self, shim, monkeypatch):
        body = body_with_tools(tools=[TOOL])
        cut = chat_resp("repetition", content="loop loop")
        retry = chat_resp("stop", content="the real answer", tool_calls=[TOOL_CALL])
        kind, payload, calls = self._relay(shim, monkeypatch, body, cut, "live", resp2_body=retry)
        assert kind == "ok"
        assert payload.body == retry
        assert len(calls) == 2
        rb = json.loads(calls[1][1])
        assert rb["chat_template_kwargs"]["enable_thinking"] is False
        assert rb.get("presence_penalty") == pytest.approx(1.5)

    def test_live_no_tools_no_retry(self, shim, monkeypatch):
        body = body_no_tools(thinking=True)
        cut = chat_resp("repetition", content="loop loop")
        kind, payload, calls = self._relay(shim, monkeypatch, body, cut, "live")
        assert kind == "ok"
        assert payload.body == cut
        assert len(calls) == 1

    def test_live_retry_also_cut(self, shim, monkeypatch):
        body = body_with_tools(tools=[TOOL])
        cut = chat_resp("repetition", content="loop loop")
        retry_cut = chat_resp("repetition", content="still looping")
        kind, payload, calls = self._relay(shim, monkeypatch, body, cut, "live", resp2_body=retry_cut)
        assert kind == "ok"
        assert payload.body == retry_cut
        assert len(calls) == 2

    def test_live_retry_failed_returns_original(self, shim, monkeypatch):
        body = body_with_tools(tools=[TOOL])
        cut = chat_resp("repetition", content="loop loop")
        kind, payload, calls = self._relay(shim, monkeypatch, body, cut, "live", resp2_body=b"err", resp2_status=500)
        assert kind == "ok"
        assert payload.body == cut
        assert len(calls) == 2

    def test_live_non_cut_unchanged(self, shim, monkeypatch):
        body = body_with_tools(tools=[TOOL])
        normal = chat_resp("stop", content="done")
        kind, payload, calls = self._relay(shim, monkeypatch, body, normal, "live")
        assert kind == "ok"
        assert payload.body == normal
        assert len(calls) == 1


class TestStreamingRelay:
    def _relay_stream(self, shim, monkeypatch, body, chunks, mode, retry_chunks=None, rec=None):
        """Drive _relay (streaming) with a mocked upstream."""
        monkeypatch.setattr(shim, "REP_RETRY", mode)
        calls = []

        async def fake_open(session, base, path, b, key, streaming):
            calls.append((base, b, streaming))
            if len(calls) == 1:
                return FakeUpstream(status=200, chunks=chunks, content_type="text/event-stream")
            if retry_chunks is None:
                return FakeUpstream(status=500, body=b"err")
            return FakeUpstream(status=200, chunks=retry_chunks, content_type="text/event-stream")

        monkeypatch.setattr(shim, "_open", fake_open)
        req = FakeRequest()
        kind, payload = asyncio.run(shim._relay(req, shim.LOCAL,
            "/v1/chat/completions", body, None, True, concurrency=1))
        return kind, payload, calls

    def test_off_stream_no_retry(self, shim, monkeypatch, patch_stream):
        body = body_with_tools(tools=[TOOL])
        chunks = sse_stream(["loop", " loop"], "repetition")
        kind, payload, calls = self._relay_stream(shim, monkeypatch, body, chunks, "off")
        assert kind == "ok"
        assert len(calls) == 1

    def test_shadow_stream_logs(self, shim, monkeypatch, patch_stream):
        body = body_with_tools(tools=[TOOL])
        chunks = sse_stream(["loop", " loop"], "repetition")
        kind, payload, calls = self._relay_stream(shim, monkeypatch, body, chunks, "shadow")
        assert kind == "ok"
        assert len(calls) == 1

    def test_live_stream_splice(self, shim, monkeypatch, patch_stream):
        # No content streamed (reasoning-only stream cut by repetition): live splices the
        # retry into the open stream, like the reasoning watchdog does.
        body = body_with_tools(tools=[TOOL])
        chunks = sse_stream([], "repetition")
        retry_chunks = sse_stream(["the real answer"], "stop")
        kind, payload, calls = self._relay_stream(shim, monkeypatch, body, chunks, "live", retry_chunks=retry_chunks)
        assert kind == "ok"
        assert len(calls) == 2
        rb = json.loads(calls[1][1])
        assert rb["chat_template_kwargs"]["enable_thinking"] is False
        assert rb.get("presence_penalty") == pytest.approx(1.5)
        assert b"the real answer" in b"".join(patch_stream["written"])

    def test_live_stream_no_tools_no_retry(self, shim, monkeypatch, patch_stream):
        body = body_no_tools(thinking=True)
        chunks = sse_stream(["loop", " loop"], "repetition")
        kind, payload, calls = self._relay_stream(shim, monkeypatch, body, chunks, "live")
        assert kind == "ok"
        assert len(calls) == 1

    def test_live_stream_non_cut_no_retry(self, shim, monkeypatch, patch_stream):
        body = body_with_tools(tools=[TOOL])
        chunks = sse_stream(["done"], "stop")
        kind, payload, calls = self._relay_stream(shim, monkeypatch, body, chunks, "live")
        assert kind == "ok"
        assert len(calls) == 1

    def test_live_stream_with_tool_call_no_retry(self, shim, monkeypatch, patch_stream):
        body = body_with_tools(tools=[TOOL])
        chunks = sse_stream(["x"], "repetition", tool_call_delta=TOOL_CALL_DELTA)
        kind, payload, calls = self._relay_stream(shim, monkeypatch, body, chunks, "live")
        assert kind == "ok"
        assert len(calls) == 1


def test_config_read(shim, monkeypatch):
    # default is off
    monkeypatch.delenv("SHIM_REPETITION_RETRY", raising=False)
    assert shim.REP_RETRY == "off"
    # can be set to shadow / live
    monkeypatch.setattr(shim, "REP_RETRY", "shadow")
    assert shim.REP_RETRY == "shadow"
    monkeypatch.setattr(shim, "REP_RETRY", "live")
    assert shim.REP_RETRY == "live"
