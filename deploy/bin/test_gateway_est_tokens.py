#!/usr/bin/env python3
"""L104 (lane A6): the gateway's prompt-size estimate ignored assistant tool_call arguments and history reasoning.

MEASURED 2026-10-03 (GW2, 7,682 local rows): pi undercounts by 2.6K tokens on average and the error grows with size
(exact/estimate 1.055 under 10K tokens, 1.246 over 60K), because _est_tokens counted only message `content` plus the tool
DEFINITIONS. Replayed on 40 flight-recorder bodies against the real chat template and tokenizer (lane A6): current estimate
MAE 2,360 tokens; counting tool_call name+arguments at the same chars/token as the tool definitions: MAE 414, bias +146.

SHIM_EST_TOOLCALLS (default off; hot-reloadable) turns the correction on. It ships OFF because GW2 measured that exact
sizes alone raise routing-threshold flips until the L105 cache-credit fix is live; flip it together with, or after, that.

Run:  python -m pytest -q test_gateway_est_tokens.py
"""
import json
import unittest
from unittest.mock import patch

import test_gateway_cache_model as C

shim = C.shim

ARGS = json.dumps({"path": "/some/where/file.py", "content": "x" * 7000})


def body(**kw):
    msgs = [{"role": "system", "content": "s" * 350},
            {"role": "user", "content": "u" * 350},
            {"role": "assistant", "content": "", "reasoning_content": "r" * 3500,
             "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "write", "arguments": ARGS}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            {"role": "user", "content": "go on"}]
    msgs += kw.pop("extra", [])
    d = {"messages": msgs, "tools": [{"type": "function", "function": {"name": "write", "parameters": {}}}]}
    d.update(kw)
    return json.dumps(d).encode()


class EstimateCountsToolCalls(unittest.TestCase):
    def est(self, b, on, exact=False):
        with patch.object(shim, "EXACT_TOKENS", exact), patch.object(shim, "EST_TOOLCALLS", on):
            return shim._est_tokens(b)

    def test_off_is_byte_for_byte_the_old_estimate(self):
        b = body()
        self.assertEqual(self.est(b, False), self.est(body(), False))
        want = int((350 + 350 + 2 + 5 + len(json.dumps(json.loads(b)["tools"]))) / shim.CHARS_PER_TOK)
        self.assertEqual(self.est(b, False), want)

    def test_on_counts_assistant_tool_call_name_and_arguments(self):
        b = body()
        gain = self.est(b, True) - self.est(b, False)
        want = int((len(ARGS) + len("write")) / shim.CHARS_PER_TOK)
        self.assertAlmostEqual(gain, want, delta=2)          # reasoning of an OLD turn is not counted (see below)

    def test_object_arguments_are_counted_too(self):
        j = json.loads(body())
        j["messages"][2]["tool_calls"][0]["function"]["arguments"] = {"path": "p", "content": "y" * 3500}
        b = json.dumps(j).encode()
        self.assertGreater(self.est(b, True) - self.est(b, False), 900)

    def test_history_reasoning_counts_only_after_the_last_user_turn(self):
        # the Qwen chat templates drop reasoning_content of turns before the last user message; they keep it after
        fresh = body(extra=[{"role": "assistant", "content": "", "reasoning_content": "q" * 3500}])
        old = body()
        d_fresh = self.est(fresh, True) - self.est(fresh, False)
        d_old = self.est(old, True) - self.est(old, False)
        self.assertAlmostEqual(d_fresh - d_old, int(3500 / shim.CHARS_PER_TOK), delta=2)

    def test_exact_path_adds_the_same_correction_on_top_of_the_engine_count(self):
        b = body()
        with patch.object(shim, "_tokenize_exact", lambda text: 5000), patch.object(shim, "_min_decision_threshold", lambda: 1):
            off = self.est(b, False, exact=True)
            on = self.est(b, True, exact=True)
        self.assertEqual(on - off, int((len(ARGS) + len("write")) / shim.CHARS_PER_TOK))

    def test_malformed_shapes_never_raise(self):
        for msgs in ([{"role": "assistant", "tool_calls": "nope"}],
                     [{"role": "assistant", "tool_calls": [None, 3, {"function": None}, {"function": {"arguments": None}}]}],
                     [{"role": "assistant", "reasoning_content": 12, "content": None}]):
            b = json.dumps({"messages": msgs}).encode()
            self.assertGreaterEqual(self.est(b, True), 0)

    def test_the_knob_is_hot_reloadable_and_in_the_schema(self):
        import gateway_config_schema as S
        self.assertIn("SHIM_EST_TOOLCALLS", shim._CFG)
        self.assertEqual(shim._CFG["SHIM_EST_TOOLCALLS"][0], "EST_TOOLCALLS")
        k = S.SCHEMA["SHIM_EST_TOOLCALLS"]
        self.assertEqual((k.type, k.default, k.tunable), ("bool", "0", True))


if __name__ == "__main__":
    unittest.main()
