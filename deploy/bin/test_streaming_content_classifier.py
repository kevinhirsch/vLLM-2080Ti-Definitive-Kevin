"""gw-streaming-content-classifier (Kevin 2026-09-11 gateway black-box audit): a reproducible
200-with-empty-content (forced thinking + max_tokens too small for the reasoning trace to
finish -- content is empty, reasoning_content is populated) was structurally invisible for
streaming traffic, and the coding runner (pi/OpenCode/OpenHands) streams 100% of its requests.
`_sse_content_shape()` is the fix: a pure, incremental scanner pulled out of _relay's closure
so its logic is testable without mocking the aiohttp streaming machinery around it.
"""
import importlib.util
import json
import os
import sys

import pytest

SHIM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keepalive-shim.py")
_counter = [0]


def load_shim(monkeypatch, tmp_path):
    env = {
        "SHIM_UPSTREAM": "http://127.0.0.1:1", "SHIM_REMOTE_BASE": "", "SHIM_REMOTE_KEY": "",
        "SHIM_ENV_FILE": str(tmp_path / "shim.env"),
        "SHIM_STATS_FILE": str(tmp_path / "gateway-stats.json"),
        "SHIM_LOG_REQUESTS": "0", "SHIM_NO_THINK_IPS": "",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    _counter[0] += 1
    name = "shim_under_test_scc_%d" % _counter[0]
    spec = importlib.util.spec_from_file_location(name, SHIM_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


@pytest.fixture
def shim(monkeypatch, tmp_path):
    return load_shim(monkeypatch, tmp_path)


def _sse(*events):
    """Build one raw SSE byte blob from a list of delta dicts (OpenAI-compatible chunk shape)."""
    lines = []
    for d in events:
        lines.append(b"data: " + json.dumps({"choices": [{"delta": d}]}).encode())
    lines.append(b"data: [DONE]")
    return b"\n\n".join(lines) + b"\n\n"


def test_normal_content_is_detected_as_non_empty(shim):
    data = _sse({"content": "Hello"}, {"content": " world"})
    saw_content, saw_tool = shim._sse_content_shape(data, False, False)
    assert saw_content is True and saw_tool is False


def test_the_reproduced_bug_forced_thinking_empty_content_is_detected_as_empty(shim):
    """The exact failure signature from the gateway black-box audit: content is null/absent
    throughout, only reasoning_content is populated."""
    data = _sse({"reasoning_content": "We need to think about this step by step"},
               {"reasoning_content": " and consider the edge cases"})
    saw_content, saw_tool = shim._sse_content_shape(data, False, False)
    assert saw_content is False and saw_tool is False


def test_whitespace_only_content_still_counts_as_empty(shim):
    """content_empty must mean 'nothing a human would read', matching classify_remote_response's
    `not content.strip()` semantics for the non-streaming path -- not merely 'zero characters'."""
    data = _sse({"content": "   "}, {"content": "\n\t"})
    saw_content, _ = shim._sse_content_shape(data, False, False)
    assert saw_content is False


def test_tool_calls_are_detected_even_with_no_text_content(shim):
    data = _sse({"tool_calls": [{"id": "call_1", "function": {"name": "read_file"}}]})
    saw_content, saw_tool = shim._sse_content_shape(data, False, False)
    assert saw_content is False and saw_tool is True


def test_accumulation_is_incremental_across_multiple_chunks(shim):
    """_relay feeds this function once per physical network chunk, not once for the whole
    stream -- state must accumulate correctly across separate calls."""
    saw_content, saw_tool = False, False
    saw_content, saw_tool = shim._sse_content_shape(_sse({"role": "assistant"}), saw_content, saw_tool)
    assert (saw_content, saw_tool) == (False, False)
    saw_content, saw_tool = shim._sse_content_shape(_sse({"content": "answer"}), saw_content, saw_tool)
    assert (saw_content, saw_tool) == (True, False)
    # a later empty chunk must not un-set what was already seen
    saw_content, saw_tool = shim._sse_content_shape(_sse({"content": ""}), saw_content, saw_tool)
    assert (saw_content, saw_tool) == (True, False)


def test_calling_it_twice_on_the_same_bytes_is_idempotent(shim):
    """The gate re-scans buf after streaming has already partially processed it (mirrors
    _scan_usage's own documented re-entrancy requirement) -- must not double-count or flip."""
    data = _sse({"content": "hi"})
    once = shim._sse_content_shape(data, False, False)
    twice = shim._sse_content_shape(data, *once)
    assert once == twice == (True, False)


def test_malformed_or_partial_json_is_skipped_not_fatal(shim):
    garbage = b"data: {not valid json\n\ndata: [DONE]\n\n"
    saw_content, saw_tool = shim._sse_content_shape(garbage, False, False)
    assert (saw_content, saw_tool) == (False, False)   # no crash, no false positive


def test_empty_or_done_only_stream_is_content_empty(shim):
    data = _sse()
    saw_content, saw_tool = shim._sse_content_shape(data, False, False)
    assert (saw_content, saw_tool) == (False, False)


def test_non_data_lines_and_comments_are_ignored(shim):
    data = b": queued position=1\n\ndata: " + json.dumps(
        {"choices": [{"delta": {"content": "hi"}}]}).encode() + b"\n\n"
    saw_content, _ = shim._sse_content_shape(data, False, False)
    assert saw_content is True
