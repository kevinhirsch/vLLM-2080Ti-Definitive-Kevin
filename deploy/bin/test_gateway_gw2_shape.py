#!/usr/bin/env python3
"""Lane GW2 (2026-10-03): L94 response-shape telemetry + L95 long-empty-generation watchdog.

L94 MEASURED: every local request-log row had finish_reason=null and has_tool_calls recorded presence only,
so stop-vs-length and broken tool calls were unmeasurable. Covered here:
  * _SSEShape reassembles SSE lines split across network chunks and tool-call argument fragments;
  * tool_args_check: JSON object + request tool schema (unknown tool, missing required, wrong type, extras);
  * non-streaming shape kwargs (chat + legacy completions), /gateway/stats summary.
L95 MEASURED: pi's enable_thinking=true bypassed thinking_budget_guard -> 16384-token empty turns. Covered:
  * SHIM_THINK_BUDGET_EXPLICIT=0 keeps the old behaviour byte for byte; >0 bounds explicit thinking;
  * the watchdog in the real _relay against a real local HTTP upstream: off / shadow (logs, no change) /
    retry (upstream aborted, thinking-off answer piped into the same client stream).

Run:  python -m pytest -q test_gateway_gw2_shape.py
"""
import asyncio
import json
import unittest
from unittest.mock import patch

import aiohttp
from aiohttp import web

import test_gateway_cache_model as C

shim = C.shim


def sse(obj):
    return ("data: %s\n\n" % json.dumps(obj, separators=(",", ":"))).encode()     # vLLM emits compact JSON


def delta(**d):
    return {"choices": [{"index": 0, "delta": d, "finish_reason": None}]}


TOOLS = [{"type": "function", "function": {"name": "read", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"},
                                     "mode": {"enum": ["a", "b"]}},
    "required": ["path"], "additionalProperties": False}}},
         {"type": "function", "function": {"name": "noargs", "parameters": {"type": "object", "properties": {}}}}]


class Shape(unittest.TestCase):
    def test_split_lines_and_fragmented_tool_args_are_reassembled(self):
        stream = (sse(delta(role="assistant", content="")) + sse(delta(reasoning="think " * 10))
                  + sse(delta(tool_calls=[{"index": 0, "id": "c1", "type": "function",
                                           "function": {"name": "read", "arguments": '{"pa'}}]))
                  + sse(delta(tool_calls=[{"index": 0, "function": {"arguments": 'th": "/x", "limit": 3}'}}]))
                  + sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
                  + b"data: [DONE]\n\n")
        s = shim._SSEShape()
        for i in range(0, len(stream), 7):          # 7-byte network chunks: every line is split
            s.feed(stream[i:i + 7])
        s.close()
        self.assertEqual(s.finish, "tool_calls")
        self.assertEqual(s.calls(), [{"name": "read", "arguments": '{"path": "/x", "limit": 3}'}])
        self.assertEqual(s.reasoning_chars, 60)
        self.assertEqual(s.reasoning_at_output, 60)
        kw = shim._shape_kw(s, json.dumps({"tools": TOOLS}).encode(), local=True)
        self.assertEqual((kw["finish_reason"], kw["tool_calls_n"], kw["tool_args_valid"]), ("tool_calls", 1, True))

    def test_remote_shape_does_not_claim_finish_reason(self):
        s = shim._SSEShape()
        s.feed(sse({"choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": "stop"}]}))
        self.assertNotIn("finish_reason", shim._shape_kw(s, b"{}", local=False))

    def test_reasoning_content_alias_and_legacy_text(self):
        s = shim._SSEShape()
        s.feed(sse(delta(reasoning_content="abcd")) + sse({"choices": [{"index": 0, "text": "hi"}]}))
        self.assertEqual((s.reasoning_chars, s.reasoning_at_output), (4, 4))


class ToolArgs(unittest.TestCase):
    def setUp(self):
        self.sch = shim._request_tool_schemas(json.dumps({"tools": TOOLS}))

    def check(self, name, args, sch="default"):
        return shim.tool_args_check([{"name": name, "arguments": args}], self.sch if sch == "default" else sch)

    def test_cases(self):
        self.assertEqual(self.check("read", '{"path": "a"}'), (True, None))
        self.assertEqual(self.check("noargs", ""), (True, None))
        self.assertEqual(shim.tool_args_check([], self.sch), (None, None))
        v, e = self.check("read", '{"path": "a"')
        self.assertFalse(v); self.assertTrue(e.startswith("json:"))
        self.assertEqual(self.check("read", '[1]'), (False, "json: arguments not an object"))
        self.assertEqual(self.check("write", '{}')[1], "unknown_tool: write")
        self.assertIn("missing required 'path'", self.check("read", '{}')[1])
        self.assertIn("expected integer", self.check("read", '{"path": "a", "limit": "3"}')[1])
        self.assertIn("expected integer", self.check("read", '{"path": "a", "limit": true}')[1])
        self.assertIn("unexpected property 'x'", self.check("read", '{"path": "a", "x": 1}')[1])
        self.assertIn("not in enum", self.check("read", '{"path": "a", "mode": "c"}')[1])
        # no tools declared: only the JSON-object check applies
        self.assertEqual(self.check("anything", '{"k": 1}', sch=None), (True, None))

    def test_schema_subset_nesting(self):
        sch = {"type": "object", "properties": {"edits": {"type": "array", "minItems": 1, "items": {
            "type": "object", "required": ["old"], "properties": {"old": {"type": "string"}}}},
            "n": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}}
        self.assertIsNone(shim._schema_error({"edits": [{"old": "x"}], "n": None}, sch))
        self.assertIn("$.edits[1]", shim._schema_error({"edits": [{"old": "x"}, {}]}, sch))
        self.assertIn("fewer than minItems", shim._schema_error({"edits": []}, sch))
        self.assertIn("anyOf", shim._schema_error({"n": "s"}, sch))
        self.assertIsNone(shim._schema_error({"x": 1}, {"$ref": "#/defs/x"}))     # unknown keywords: satisfied

    def test_nonstream_kw(self):
        data = json.dumps({"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "content": None, "reasoning": "hmm", "tool_calls": [
                {"id": "1", "type": "function", "function": {"name": "read", "arguments": '{"path": 1}'}}]}}]})
        kw = shim._nonstream_shape_kw(data, json.dumps({"tools": TOOLS}).encode(), local=True)
        self.assertEqual((kw["finish_reason"], kw["content_empty"], kw["has_tool_calls"], kw["tool_args_valid"],
                          kw["reasoning_chars"]), ("tool_calls", True, True, False, 3))
        legacy = json.dumps({"choices": [{"index": 0, "text": "hello", "finish_reason": "length"}]})
        kw = shim._nonstream_shape_kw(legacy, b"{}", local=True)
        self.assertEqual((kw["finish_reason"], kw["content_empty"]), ("length", False))
        self.assertNotIn("finish_reason", shim._nonstream_shape_kw(legacy, b"{}", local=False))
        self.assertEqual(shim._nonstream_shape_kw(b"not json", b"{}", local=True), {})

    def test_stats_summary(self):
        with patch.object(shim, "_SHAPE_STATS", type(shim._SHAPE_STATS)(shim.collections.Counter)):
            shim._shape_stats_note("local", 200, {"finish_reason": "stop", "content_empty": False})
            shim._shape_stats_note("local", 200, {"finish_reason": "length", "content_empty": True,
                                                  "has_tool_calls": False, "reasoning_watchdog": "shadow"})
            shim._shape_stats_note("local", 200, {"finish_reason": "tool_calls", "content_empty": True,
                                                  "has_tool_calls": True, "tool_calls_n": 1, "tool_args_valid": False})
            shim._shape_stats_note("local", 500, {"finish_reason": "stop"})          # errors are not shapes
            shim._shape_stats_note("local", 200, {})                                  # unclassified
            loc = shim._shape_stats_summary()["by_route"]["local"]
        self.assertEqual(loc["responses"], 3)
        self.assertEqual(loc["finish_reason"], {"length": 1, "stop": 1, "tool_calls": 1})
        self.assertEqual((loc["empty_no_tool"], loc["tool_args_invalid"], loc["tool_args_valid_rate"]), (1, 1, 0.0))
        self.assertEqual(loc["reasoning_watchdog"], {"shadow": 1})
        json.dumps(shim._shape_stats_summary())


class EstimateError(unittest.TestCase):
    """CR2: the request log's local `ptok` is the gateway's ESTIMATE; ptok_exact was None on every local row."""

    def note(self, **info):
        rows = []
        base = {"name": "pi", "route": "local", "t0": 1.0, "ptok": 20000, "est_tokens": 20000, "est_computed": 6000}
        with patch.object(shim, "_telemetry_log_enqueue", rows.append), patch.object(shim, "_pm_feedback", lambda i: None):
            shim._telemetry_note_request(dict(base, **info))
        return rows[0]

    def test_local_row_carries_engine_usage(self):
        with patch.object(shim, "_EST_ERR", shim.collections.deque(maxlen=2000)):
            r = self.note(ptok_exact_local=22000, cached_actual=15000, computed_actual=7000)
            self.assertEqual((r["ptok_exact"], r["ptok_exact_src"], r["ptok_est_err"], r["cached_actual"]),
                             (22000, "engine", -2000, 15000))
            r = self.note()                                   # aborted stream: no usage -> honest None
            self.assertEqual((r["ptok_exact"], r["ptok_exact_src"], r["ptok_est_err"]), (None, None, None))
            r = self.note(route="remote", ptok_exact=21000, ptok_exact_local=None)
            self.assertEqual((r["ptok_exact"], r["ptok_exact_src"], r["ptok_est_err"]), (21000, "provider", None))
            summ = shim._est_err_summary()
        self.assertEqual(summ["window"], 1)
        self.assertEqual((summ["prompt_tokens"]["bias_mean"], summ["computed_tokens"]["p50"]), (-2000, -1000))
        self.assertIn("pi", summ["by_client"])
        json.dumps(summ)


class AnchoredCredit(unittest.TestCase):
    """Shadow engine-anchored cache credit: next turn's credit = previous turn's ENGINE prompt_tokens, block-rounded."""

    def setUp(self):
        for p in (patch.object(shim, "_PM_ANCHOR", type(shim._PM_ANCHOR)()),
                  patch.object(shim, "_PM_NODES", type(shim._PM_NODES)()),
                  patch.object(shim, "PREFIX_CREDIT_UNIT", 0), patch.object(shim, "prefix_align_tokens", lambda: 3568)):
            p.start()
            self.addCleanup(p.stop)

    def test_continuation_gets_exact_block_rounded_credit(self):
        t1 = shim._pm_predict(C.convo("A", 10), 20000)["chain"]
        shim._anchor_note(t1, 21000, now=1000.0)
        t2 = shim._pm_predict(C.convo("A", 11), 22000)["chain"]
        self.assertEqual(shim.anchored_credit(t2, 22000, now=1010.0), (5 * 3568, 10.0))
        self.assertEqual(shim.anchored_credit(t2, 10000, now=1010.0)[0], 9999)            # capped below the prompt
        other = shim._pm_predict(C.convo("B", 11), 22000)["chain"]
        self.assertEqual(shim.anchored_credit(other, 22000, now=1010.0), (0, None))       # other conversation
        self.assertEqual(shim.anchored_credit(t2, 22000, now=1000.0 + shim.PREFIX_MODEL_TTL_SECS + 1), (0, None))
        shim._pm_reset("test")
        self.assertEqual(shim.anchored_credit(t2, 22000, now=1010.0), (0, None))         # engine cache gone

    def test_knob_is_hot_reloadable_and_default_off(self):
        self.assertIn("SHIM_CREDIT_ANCHOR", shim._CFG)
        self.assertEqual(shim.CREDIT_ANCHOR, "off")


class ThinkBudgetExplicit(unittest.TestCase):
    def body(self, **kw):
        return json.dumps(dict({"messages": [], "max_tokens": 16384}, **kw)).encode()

    def test_off_is_byte_identical(self):
        with patch.object(shim, "THINK_GUARD", True), patch.object(shim, "THINK_BUDGET_EXPLICIT", 0):
            for b in (self.body(chat_template_kwargs={"enable_thinking": True}),
                      self.body(chat_template_kwargs={"enable_thinking": False}),
                      self.body(thinking_token_budget=100)):
                self.assertEqual(shim.thinking_budget_guard(b), b)
            # the implicit path is unchanged: budget = min(frac * mt, MAX)
            self.assertEqual(json.loads(shim.thinking_budget_guard(self.body()))["thinking_token_budget"],
                             min(int(16384 * shim.THINK_BUDGET_FRAC), shim.THINK_BUDGET_MAX))

    def test_on_bounds_explicit_thinking_only(self):
        with patch.object(shim, "THINK_GUARD", True), patch.object(shim, "THINK_BUDGET_EXPLICIT", 8192):
            out = json.loads(shim.thinking_budget_guard(self.body(chat_template_kwargs={"enable_thinking": True})))
            self.assertEqual(out["thinking_token_budget"], 8192)
            b = self.body(chat_template_kwargs={"enable_thinking": True}, thinking_token_budget=50)
            self.assertEqual(shim.thinking_budget_guard(b), b)
            b = json.dumps({"messages": [], "max_tokens": 4000, "chat_template_kwargs": {"enable_thinking": True}}).encode()
            self.assertEqual(shim.thinking_budget_guard(b), b)       # budget >= max_tokens: leave it

    def test_gate_commits_on_vllm_reasoning_field(self):
        self.assertTrue(shim._looks_meaningful(sse(delta(reasoning="x")).decode()))
        self.assertFalse(shim._looks_meaningful(sse(delta(role="assistant", reasoning="")).decode()))

    def test_knobs_are_hot_reloadable(self):
        for k in ("SHIM_REASONING_WATCHDOG", "SHIM_REASONING_BUDGET_TOKENS", "SHIM_THINK_BUDGET_EXPLICIT",
                  "SHIM_REASONING_CHARS_PER_TOKEN", "SHIM_CHAIN_TELEMETRY", "SHIM_WARM_PRIORITY"):
            self.assertIn(k, shim._CFG)
        self.assertIs(shim._CFG["SHIM_WARM_PRIORITY"][1]("0"), False)
        self.assertIs(shim._CFG["SHIM_WARM_PRIORITY"][1]("1"), True)


class Watchdog(unittest.IsolatedAsyncioTestCase):
    """The real _relay against a real local HTTP upstream that thinks forever unless thinking is off."""

    async def asyncSetUp(self):
        self.calls, self.aborted = [], asyncio.Event()

        async def upstream(request):
            body = await request.json()
            think = (body.get("chat_template_kwargs") or {}).get("enable_thinking") is not False
            self.calls.append(think)
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            try:
                await resp.write(sse(delta(role="assistant", content="")))
                if think:
                    for _ in range(400):                       # 400 x 100 chars = 10000 est tokens @ 4 c/t
                        await resp.write(sse(delta(reasoning="r" * 100)))
                        await asyncio.sleep(0.002)
                    await resp.write(sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}))
                else:
                    await resp.write(sse(delta(content="the answer")))
                    await resp.write(sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                                          "usage": {"prompt_tokens": 10, "completion_tokens": 3}}))
                await resp.write(b"data: [DONE]\n\n")
            except (ConnectionResetError, RuntimeError, asyncio.CancelledError):
                self.aborted.set()
                raise
            return resp

        app = web.Application()
        app.router.add_post("/v1/chat/completions", upstream)
        self.up_runner = web.AppRunner(app)
        await self.up_runner.setup()
        site = web.TCPSite(self.up_runner, "127.0.0.1", 0)
        await site.start()
        self.base = "http://127.0.0.1:%d" % self.up_runner.addresses[0][1]
        self.active = {}

        async def front(request):
            body = await request.read()
            kind, payload = await shim._relay(request, self.base, "/v1/chat/completions", body, None, True)
            return payload

        fapp = web.Application()
        fapp.router.add_post("/v1/chat/completions", front)
        self.front_runner = web.AppRunner(fapp)
        await self.front_runner.setup()
        fsite = web.TCPSite(self.front_runner, "127.0.0.1", 0)
        await fsite.start()
        self.front = "http://127.0.0.1:%d/v1/chat/completions" % self.front_runner.addresses[0][1]
        self.notes = []
        self._p = [patch.object(shim, "LOCAL", self.base),
                   patch.object(shim, "_active_set", lambda req, **kw: self.active.update(kw)),
                   patch.object(shim, "_pm_prefill_done", lambda req: None),
                   patch.object(shim, "_timeout_note", lambda *a, **k: self.notes.append((a[1], k))),
                   patch.object(shim, "REASONING_BUDGET_TOKENS", 2000),
                   patch.object(shim, "REASONING_CHARS_PER_TOKEN", 4.0),
                   patch.object(shim, "_RW_STATS", shim.collections.Counter())]
        for p in self._p:
            p.start()

    async def asyncTearDown(self):
        for p in self._p:
            p.stop()
        await self.front_runner.cleanup()
        await self.up_runner.cleanup()

    async def post(self, **extra):
        body = dict({"model": "estate", "stream": True, "messages": [{"role": "user", "content": "q"}],
                     "chat_template_kwargs": {"enable_thinking": True}}, **extra)
        async with aiohttp.ClientSession() as s:
            async with s.post(self.front, json=body) as r:
                return (await r.read()).decode()

    async def test_off_streams_everything_unchanged(self):
        with patch.object(shim, "REASONING_WATCHDOG", "off"):
            out = await self.post()
        self.assertEqual(self.calls, [True])
        self.assertNotIn("reasoning_watchdog", self.active)
        self.assertEqual(self.active["finish_reason"], "length")       # L94: local finish_reason recorded
        self.assertEqual(self.active["reasoning_chars"], 40000)
        self.assertIsNone(self.active["reasoning_chars_at_output"])
        self.assertTrue(out.rstrip().endswith("[DONE]"))

    async def test_shadow_logs_once_and_changes_nothing(self):
        with patch.object(shim, "REASONING_WATCHDOG", "shadow"):
            out = await self.post()
        self.assertEqual(self.calls, [True])
        self.assertEqual(self.active["reasoning_watchdog"], "shadow")
        self.assertGreaterEqual(self.active["reasoning_watchdog_at_tok"], 2000)
        self.assertEqual([n[0] for n in self.notes], ["gateway:reasoning-budget"])
        self.assertEqual(self.notes[0][1]["outcome"], "shadow-would-retry")
        self.assertEqual(shim._RW_STATS["shadow"], 1)
        self.assertEqual(out.count('"reasoning"'), 400)

    async def test_retry_aborts_and_pipes_the_no_think_answer(self):
        with patch.object(shim, "REASONING_WATCHDOG", "retry"):
            out = await self.post()
        self.assertEqual(self.calls, [True, False])
        await asyncio.wait_for(self.aborted.wait(), 5)                  # first generation really aborted
        self.assertLess(out.count('"reasoning"'), 400)
        self.assertIn("the answer", out)
        self.assertTrue(out.rstrip().endswith("[DONE]"))
        self.assertEqual((self.active["reasoning_watchdog"], self.active["reasoning_watchdog_ok"]), ("retry", True))
        self.assertEqual(self.active["finish_reason"], "stop")
        self.assertEqual(self.active["outtok"], 3)
        self.assertIsNotNone(self.active["reasoning_chars_at_output"])

    async def test_thinking_off_requests_are_never_armed(self):
        with patch.object(shim, "REASONING_WATCHDOG", "retry"):
            await self.post(chat_template_kwargs={"enable_thinking": False})
        self.assertEqual(self.calls, [False])
        self.assertNotIn("reasoning_watchdog", self.active)

    async def test_remote_relay_is_never_armed(self):
        with patch.object(shim, "REASONING_WATCHDOG", "retry"), patch.object(shim, "LOCAL", "http://127.0.0.1:1"):
            await self.post()
        self.assertEqual(self.calls, [True])
        self.assertNotIn("reasoning_watchdog", self.active)
        self.assertEqual(self.active["finish_reason"], "length")        # remote: unchanged _exact_finish path


if __name__ == "__main__":
    unittest.main()
