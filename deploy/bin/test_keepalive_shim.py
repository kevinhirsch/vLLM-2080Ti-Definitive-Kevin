#!/usr/bin/env python3
"""Contract tests for the keepalive-shim gateway routing helpers.

The shim makes every local-vs-remote routing decision for the estate; before
this file it had ZERO tests, and three real routing bugs shipped and were fixed
in one session (2026-08-18): local-pin traffic diverted to paid remote, /tq
prefills aborted by a 30s cap, and empty reasoner replies written as output.
The last two live in request-shaping paths exercised here; each fixed bug has a
named regression test below so it cannot silently come back.

Pure helpers only — no server, no network. The module is import-safe (the
server runs under __main__). Loaded via importlib because the filename is
hyphenated. Thresholds are read from the module, not hardcoded, so the tests
stay valid if the env-driven constants change.

Run:  python3 test_keepalive_shim.py       (unittest, exit 0 = green)
"""
import importlib.util
import json
import os
import pathlib
import unittest

# Deterministic estimation: force the cheap char/CHARS_PER_TOK path (no
# tokenizer round-trip) so token estimates are a pure function of input size.
os.environ.setdefault("SHIM_EXACT_TOKENS", "0")

_HERE = pathlib.Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "keepalive_shim", _HERE / "keepalive-shim.py")
shim = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(shim)


def _body(messages=None, **kw):
    d = {"model": "qwen-local", "messages": messages or [{"role": "user",
                                                          "content": "hi"}]}
    d.update(kw)
    return json.dumps(d).encode()


def _chars_for_tokens(n):
    """A message content string that estimates to ~n tokens."""
    return "x" * int(n * shim.CHARS_PER_TOK)


class EstTokens(unittest.TestCase):
    def test_scales_with_content(self):
        small = shim._est_tokens(_body([{"role": "user", "content": "a" * 350}]))
        big = shim._est_tokens(_body([{"role": "user", "content": "a" * 3500}]))
        self.assertLess(small, big)

    def test_multimodal_text_blocks_counted(self):
        b = _body([{"role": "user", "content": [
            {"type": "text", "text": "a" * 700}, {"type": "image", "x": 1}]}])
        self.assertGreater(shim._est_tokens(b), 100)

    def test_garbage_body_is_zero_not_crash(self):
        self.assertEqual(shim._est_tokens(b"not json"), 0)

    def test_tool_definitions_counted(self):
        # Regression 2026-08-19: tool schemas are part of the prompt the engine
        # prefills, but were omitted from the estimate. A 34-tool Hermes request
        # undercounted by ~20K tokens, so the first-token timeout (scaled off the
        # estimate) fired before prefill finished -> false failover to DeepSeek.
        no_tools = shim._est_tokens(_body([{"role": "user", "content": "hi"}]))
        big_tool = {"type": "function", "function": {
            "name": "bash", "description": "d" * 3500, "parameters": {}}}
        with_tools = shim._est_tokens(
            _body([{"role": "user", "content": "hi"}], tools=[big_tool]))
        self.assertGreater(with_tools, no_tools + 500)


class OverLocalCap(unittest.TestCase):
    def test_small_request_is_under_cap(self):
        self.assertFalse(shim.over_local_cap(_body(max_tokens=100)))

    def test_prompt_plus_maxtokens_over_cap(self):
        # prompt alone near the cap, plus max_tokens, must exceed it
        near = _chars_for_tokens(shim.MAX_LOCAL_TOKENS)
        b = _body([{"role": "user", "content": near}], max_tokens=4000)
        self.assertTrue(shim.over_local_cap(b))

    def test_missing_max_tokens_uses_default(self):
        near = _chars_for_tokens(shim.MAX_LOCAL_TOKENS - shim.DEFAULT_MAX_OUT + 50)
        self.assertTrue(shim.over_local_cap(
            _body([{"role": "user", "content": near}])))


class Tiny(unittest.TestCase):
    def test_micro_call_is_tiny(self):
        self.assertTrue(shim.is_tiny(_body([{"role": "user", "content": "hi"}],
                                           max_tokens=50)))

    def test_big_output_not_tiny(self):
        self.assertFalse(shim.is_tiny(_body(max_tokens=shim.TINY_TOKENS + 1)))

    def test_big_prompt_not_tiny(self):
        b = _body([{"role": "user", "content": _chars_for_tokens(shim.TINY_TOKENS)}],
                  max_tokens=50)
        self.assertFalse(shim.is_tiny(b))


class Background(unittest.TestCase):
    class _Req:
        def __init__(self, xclient=None):
            self.headers = {"X-Client": xclient} if xclient else {}

    def test_cron_client_is_background(self):
        self.assertTrue(shim.is_background(_body(), self._Req("hermes-cron")))

    def test_batch_client_is_background(self):
        self.assertTrue(shim.is_background(_body(), self._Req("nightly-batch")))

    def test_interactive_is_not_background(self):
        self.assertFalse(shim.is_background(_body(), self._Req("pi-session")))

    def test_marker_in_prompt_is_background(self):
        b = _body([{"role": "user", "content": "scheduled cron job: summarize"}])
        self.assertTrue(shim.is_background(b, self._Req()))


class RemapForRemote(unittest.TestCase):
    def test_rewrites_model_and_strips_local_only_params(self):
        b = _body(chat_template_kwargs={"enable_thinking": False},
                  mamba_cache_mode="align", guided_decoding_backend="xgrammar")
        out = json.loads(shim.remap_for_remote(b))
        self.assertEqual(out["model"], shim.REMOTE_MODEL)
        self.assertNotIn("chat_template_kwargs", out)
        self.assertNotIn("mamba_cache_mode", out)
        self.assertNotIn("guided_decoding_backend", out)

    def test_preserves_messages_and_sampling(self):
        b = _body(temperature=0.3, max_tokens=500)
        out = json.loads(shim.remap_for_remote(b))
        self.assertEqual(out["temperature"], 0.3)
        self.assertEqual(out["max_tokens"], 500)
        self.assertTrue(out["messages"])

    def test_garbage_body_passthrough(self):
        self.assertEqual(shim.remap_for_remote(b"not json"), b"not json")


class RemapForRemoteThinking(unittest.TestCase):
    """Regression 2026-08-19: DeepSeek V4 thinking mode 400s ANY conversation
    ending on a tool result whose assistant tool_call lacks reasoning_content
    ('The reasoning_content in the thinking mode must be passed back to the
    API'). Hermes/pi histories are Qwen-generated and never carry it, so EVERY
    failover to DeepSeek died as 'model provider failed'. remap_for_remote now
    forces non-thinking on DeepSeek. Verified against the live API."""

    def _remap(self, base, no_think):
        orig_base, orig_nt = shim.REMOTE_BASE, shim.REMOTE_NO_THINK
        shim.REMOTE_BASE, shim.REMOTE_NO_THINK = base, no_think
        try:
            return json.loads(shim.remap_for_remote(_body()))
        finally:
            shim.REMOTE_BASE, shim.REMOTE_NO_THINK = orig_base, orig_nt

    def test_deepseek_gets_thinking_disabled(self):
        out = self._remap("https://api.deepseek.com", True)
        self.assertEqual(out.get("thinking"), {"type": "disabled"})

    def test_non_deepseek_remote_untouched(self):
        out = self._remap("https://api.openai.com/v1", True)
        self.assertNotIn("thinking", out)

    def test_env_flag_can_disable(self):
        out = self._remap("https://api.deepseek.com", False)
        self.assertNotIn("thinking", out)


class ThinkingBudgetGuard(unittest.TestCase):
    def test_injects_thinking_budget_when_enabled(self):
        if not getattr(shim, "THINK_GUARD", True):
            self.skipTest("thinking guard disabled by env")
        out = json.loads(shim.thinking_budget_guard(_body(max_tokens=400)))
        self.assertIn("thinking_token_budget", out)

    def test_explicit_caller_value_preserved(self):
        b = _body(max_tokens=400, thinking_token_budget=123)
        out = json.loads(shim.thinking_budget_guard(b))
        self.assertEqual(out["thinking_token_budget"], 123)


class RepetitionGuard(unittest.TestCase):
    def test_injects_detection_params_by_default(self):
        if not shim.REP_GUARD:
            self.skipTest("rep guard disabled by env")
        out = json.loads(shim.repetition_guard(_body()))
        self.assertIn("repetition_detection", out)

    def test_explicit_caller_value_preserved(self):
        b = _body(repetition_detection={"min_pattern_size": 99})
        out = json.loads(shim.repetition_guard(b))
        self.assertEqual(out["repetition_detection"]["min_pattern_size"], 99)


class EmptyThinkingResponse(unittest.TestCase):
    """Regression: a reasoner that burns the whole budget on reasoning_content
    and returns empty content (finish_reason=length) must be detected."""

    class _Resp:
        def __init__(self, obj):
            self.body = json.dumps(obj).encode()

    def test_empty_length_response_detected(self):
        r = self._Resp({"choices": [{"finish_reason": "length",
                                     "message": {"content": ""}}]})
        self.assertTrue(shim._is_empty_thinking_response(r))

    def test_normal_content_not_flagged(self):
        r = self._Resp({"choices": [{"finish_reason": "stop",
                                     "message": {"content": "hello"}}]})
        self.assertFalse(shim._is_empty_thinking_response(r))

    def test_tool_call_only_not_flagged(self):
        r = self._Resp({"choices": [{"finish_reason": "length", "message": {
            "content": "", "tool_calls": [{"id": "1"}]}}]})
        self.assertFalse(shim._is_empty_thinking_response(r))


class LooksMeaningful(unittest.TestCase):
    """The streaming first-token gate holds the response until real generated
    output appears, then commits to the client. Regression 2026-08-19: a
    tool-call-only response streams its progress in tool_calls (not content),
    was invisible to the gate, hit the first-token timeout, and false-failed-
    over to DeepSeek -> 'model provider failed'. Tool calls now count."""

    def test_content_token_is_meaningful(self):
        self.assertTrue(shim._looks_meaningful('{"delta":{"content":"Hi"}}'))

    def test_empty_content_not_meaningful(self):
        self.assertFalse(shim._looks_meaningful('{"delta":{"content":""}}'))

    def test_reasoning_token_is_meaningful(self):
        self.assertTrue(shim._looks_meaningful(
            '{"delta":{"reasoning_content":"Let"}}'))

    def test_tool_call_is_meaningful(self):
        self.assertTrue(shim._looks_meaningful(
            '{"delta":{"tool_calls":[{"function":{"name":"bash"}}]}}'))

    def test_function_call_is_meaningful(self):
        self.assertTrue(shim._looks_meaningful(
            '{"delta":{"function_call":{"name":"bash"}}}'))

    def test_finish_reason_is_meaningful(self):
        self.assertTrue(shim._looks_meaningful('{"finish_reason":"stop"}'))

    def test_prelude_noise_not_meaningful(self):
        self.assertFalse(shim._looks_meaningful('{"choices":[{"index":0}]}'))


if __name__ == "__main__":
    unittest.main(verbosity=2)
