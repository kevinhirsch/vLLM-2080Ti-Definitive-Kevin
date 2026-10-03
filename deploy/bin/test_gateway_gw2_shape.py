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
import os
import time
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
        self.assertTrue(kw["has_tool_calls"])
        self.assertNotIn("content_empty", kw)          # only false negatives are corrected

    def test_split_content_line_is_not_a_false_empty(self):
        line = sse(delta(content="the only answer"))
        old_saw = shim._sse_content_shape(line[:20], False, False)
        old_saw = shim._sse_content_shape(line[20:], *old_saw)
        self.assertEqual(old_saw, (False, False))       # the legacy scan misses a split line
        s = shim._SSEShape()
        s.feed(line[:20]); s.feed(line[20:])
        self.assertIs(shim._shape_kw(s, b"{}", local=True)["content_empty"], False)

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


class TokenizeMemo(unittest.IsolatedAsyncioTestCase):
    """~7 _est_tokens calls per request were each a synchronous /tokenize round trip on the event loop."""

    async def test_one_round_trip_per_prompt_and_off_the_loop(self):
        calls = []

        class R:
            def __init__(self, n):
                self.n = n

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, *a):
                return json.dumps({"count": self.n}).encode()

        def urlopen(req, timeout=None):
            calls.append(shim.threading.current_thread() is shim.threading.main_thread())
            return R(1234)

        body = json.dumps({"messages": [{"role": "user", "content": "x" * 20000}]}).encode()
        with patch.object(shim, "EXACT_TOKENS", True), patch.object(shim, "_TOK_MEMO", shim.collections.OrderedDict()), \
                patch.object(shim, "_TOK_STATS", shim.collections.Counter()), patch.object(shim, "_tok_fail_until", 0.0), \
                patch.object(shim.urllib.request, "urlopen", urlopen), patch.object(shim, "_min_decision_threshold", lambda: 100):
            await shim._warm_token_estimate(body)
            for _ in range(7):
                self.assertEqual(shim._est_tokens(body), 1234)
            self.assertEqual(calls, [False])                 # exactly one call, on a worker thread
            self.assertEqual((shim._TOK_STATS["calls"], shim._TOK_STATS["calls_on_loop"], shim._TOK_STATS["memo_hits"]),
                             (1, 0, 7))


class CreditProbe(unittest.IsolatedAsyncioTestCase):
    """The gateway's client for the engine's /v1/fork/prefix_cache_probe (real local HTTP server as the engine)."""

    async def asyncSetUp(self):
        self.mode, self.delay, self.seen = "ok", 0.0, []

        async def h(request):
            self.seen.append(await request.json())
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.mode == "absent":
                return web.Response(status=404)
            if self.mode == "disabled":
                return web.json_response({"enabled": False})
            return web.json_response({"enabled": True, "prompt_tokens": 30000, "cached_tokens": 26000})

        app = web.Application()
        app.router.add_post("/v1/fork/prefix_cache_probe", h)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self._p = [patch.object(shim, "LOCAL", "http://127.0.0.1:%d" % self.runner.addresses[0][1]),
                   patch.dict(shim._PROBE, {"disabled_until": 0.0, "why": None, "session": None}),
                   patch.object(shim, "_PROBE_STATS", shim.collections.Counter()),
                   patch.object(shim, "_PROBE_ERR", shim.collections.deque(maxlen=2000)),
                   patch.object(shim, "CREDIT_PROBE_TIMEOUT_S", 0.3), patch.object(shim, "_local_offline", lambda now=None: False)]
        for p in self._p:
            p.start()
        self.active = {}

    async def asyncTearDown(self):
        if shim._PROBE.get("session"):
            await shim._PROBE["session"].close()
        for p in self._p:
            p.stop()
        await self.runner.cleanup()

    def body(self):
        return json.dumps({"model": "estate", "stream": True, "messages": [{"role": "user", "content": "q"}],
                           "tools": TOOLS, "chat_template_kwargs": {"enable_thinking": True}}).encode()

    async def test_ok_sends_only_render_fields(self):
        cached, pt, ms = await shim.credit_probe(self.body())
        self.assertEqual((cached, pt), (26000, 30000))
        self.assertEqual(set(self.seen[0]), {"model", "messages", "tools", "chat_template_kwargs"})

    async def test_absent_and_disabled_back_off_and_timeout_is_bounded(self):
        self.mode = "absent"
        self.assertIsNone(await shim.credit_probe(self.body()))
        self.assertIsNone(await shim.credit_probe(self.body()))
        self.assertEqual(len(self.seen), 1)                   # backed off: no second call
        self.assertEqual(shim._PROBE_STATS["skipped_backoff"], 1)
        shim._PROBE["disabled_until"] = 0.0
        self.mode = "disabled"
        self.assertIsNone(await shim.credit_probe(self.body()))
        self.assertGreater(shim._PROBE["disabled_until"], time.time() + 500)
        shim._PROBE["disabled_until"] = 0.0
        self.mode, self.delay = "ok", 2.0
        t = time.time()
        self.assertIsNone(await shim.credit_probe(self.body()))
        self.assertLess(time.time() - t, 1.0)
        self.assertEqual(shim._PROBE_STATS["timeout"], 1)

    async def sources(self, probe, anchor, chain_anchor=None):
        pm = {"credit": 3568, "computed": 26432, "est": 30000, "chain": [(b"k", 1)]}
        with patch.object(shim, "CREDIT_PROBE", probe), patch.object(shim, "CREDIT_ANCHOR", anchor), \
                patch.object(shim, "_active_set", lambda r, **kw: self.active.update(kw)), \
                patch.object(shim, "anchored_credit", lambda chain, est, now=None: chain_anchor or (0, None)):
            src = await shim._credit_sources(object(), pm, self.body(), 30000)
        return src, pm

    async def test_shadow_logs_and_routes_on_the_model(self):
        src, pm = await self.sources("shadow", "shadow")
        self.assertEqual((src, pm["credit"], pm["computed"]), ("model", 3568, 26432))
        self.assertEqual(self.active["pm_credit_probe"], 26000)
        self.assertEqual(shim._PROBE_ERR[-1], (3568, 26000, None))
        self.assertEqual(shim._credit_probe_summary()["model_minus_probe"]["p50"], 3568 - 26000)

    async def test_live_uses_probe_then_anchor_then_model(self):
        src, pm = await self.sources("live", "live")
        self.assertEqual((src, pm["credit"], pm["computed"]), ("probe", 26000, 4000))
        self.assertEqual(self.active["pm_credit_model"], 3568)
        self.mode = "absent"
        src, pm = await self.sources("live", "live", chain_anchor=(21408, 12.0))
        self.assertEqual((src, pm["credit"], pm["computed"]), ("anchor", 21408, 8592))
        src, pm = await self.sources("live", "live")
        self.assertEqual((src, pm["credit"]), ("model", 3568))

    async def test_small_prompts_and_offline_windows_are_not_probed(self):
        with patch.object(shim, "CREDIT_PROBE", "shadow"), patch.object(shim, "CREDIT_ANCHOR", "off"), \
                patch.object(shim, "_active_set", lambda r, **kw: None):
            await shim._credit_sources(object(), {"credit": 0, "computed": 100, "chain": []}, self.body(), 100)
            with patch.object(shim, "_local_offline", lambda now=None: True):
                await shim._credit_sources(object(), {"credit": 0, "computed": 9000, "chain": []}, self.body(), 9000)
        self.assertEqual(self.seen, [])


class OfflineWindowSurvivesRestart(unittest.IsolatedAsyncioTestCase):
    """GW2 2026-10-03: the 09:11 gateway publish dropped K5's planned offline window (it lived only in memory)."""

    async def asyncSetUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = self._td.name
        self.state = os.path.join(self.tmp, "offline-window.json")
        self._p = [patch.object(shim, "OFFLINE_STATE_FILE", self.state), patch.object(shim, "_admin_ok", lambda r: True),
                   patch.dict(shim._OFFLINE, {"until": 0.0, "lease": None, "reason": None, "by": None, "t0": None,
                                              "ttl_s": None, "refused": 0}),
                   patch.object(shim, "_flow_event", lambda e: None)]
        for p in self._p:
            p.start()
        app = web.Application()
        app.router.add_route("*", "/gateway/offline", shim.gateway_offline)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = "http://127.0.0.1:%d/gateway/offline" % self.runner.addresses[0][1]

    async def asyncTearDown(self):
        await self.runner.cleanup()
        for p in self._p:
            p.stop()

    async def call(self, method, body=None):
        async with aiohttp.ClientSession() as s:
            async with s.request(method, self.url, json=body) as r:
                return r.status, await r.json()

    def restart(self):
        shim._OFFLINE.update(until=0.0, lease=None, reason=None, by=None, t0=None, ttl_s=None, refused=0)
        return shim._offline_restore()

    async def test_window_is_restored_with_its_lease_and_close_persists(self):
        self.assertTrue(self.state.startswith(self.tmp))
        st, d = await self.call("POST", {"ttl_s": 600, "reason": "K5 A/B", "by": "K5"})
        self.assertEqual(st, 200)
        lease = d["lease"]
        self.assertTrue(self.restart())                         # a gateway restart
        st, d = await self.call("GET")
        self.assertTrue(d["offline"])
        self.assertEqual((d["reason"], d["by"]), ("K5 A/B", "K5"))
        st, d = await self.call("POST", {"lease": lease, "ttl_s": 900})     # owner can still extend...
        self.assertEqual(st, 200)
        st, d = await self.call("DELETE", {"lease": lease})                 # ...and close it
        self.assertEqual((st, d["offline"]), (200, False))
        self.assertFalse(self.restart())                        # a closed window stays closed

    async def test_expired_or_corrupt_state_is_not_restored(self):
        json.dump({"until": time.time() - 5, "lease": "x", "reason": "old"}, open(self.state, "w"))
        self.assertFalse(self.restart())
        open(self.state, "w").write("{not json")
        self.assertFalse(self.restart())
        json.dump({"until": time.time() + 99999, "lease": "x"}, open(self.state, "w"))
        self.assertFalse(self.restart())                        # longer than any window may be: not trusted

    async def test_open_is_refused_while_a_publish_drain_is_up_but_extend_is_not(self):
        st, d = await self.call("POST", {"ttl_s": 600, "reason": "A", "by": "A"})
        lease = d["lease"]
        with patch.object(shim, "_draining", lambda now=None: True):
            st, d = await self.call("POST", {"lease": lease, "ttl_s": 900})
            self.assertEqual(st, 200)
            await self.call("DELETE", {"lease": lease})
            st, d = await self.call("POST", {"ttl_s": 600, "reason": "B", "by": "B"})
            self.assertEqual((st, d["code"]), (409, shim.OFFLINE_REFUSED_PUBLISH))


class OfflineDrift(unittest.TestCase):
    """The three tools must agree on the marker and the status field, or the publish-vs-window guard silently fails."""

    def test_marker_and_status_field_agree(self):
        import importlib.util as ilu
        here = os.path.dirname(os.path.abspath(shim.__file__))
        spec = ilu.spec_from_file_location("gw_offline_cli_drift", os.path.join(here, "gateway-offline.py"))
        cli = ilu.module_from_spec(spec)
        spec.loader.exec_module(cli)
        self.assertEqual(cli.OFFLINE_REFUSED_PUBLISH, shim.OFFLINE_REFUSED_PUBLISH)
        self.assertIn("offline", shim._offline_status())
        pub_src = open(os.path.join(here, "gateway_safe_publish.py")).read()
        self.assertIn('st.get("offline")', pub_src)
        self.assertIn('_refuse_if_offline_window(token, "start")', pub_src)
        self.assertIn('_refuse_if_offline_window(token, "restart")', pub_src)
        self.assertIn("_offline_restore()", open(shim.__file__).read())

    def test_cli_waits_out_a_publish_then_opens(self):
        import importlib.util as ilu
        here = os.path.dirname(os.path.abspath(shim.__file__))
        spec = ilu.spec_from_file_location("gw_offline_cli_wait", os.path.join(here, "gateway-offline.py"))
        cli = ilu.module_from_spec(spec)
        spec.loader.exec_module(cli)
        replies = [{"error": '{"code": "publish-in-flight"}', "http": 409}] * 2 + [{"lease": "L1"}]
        calls = []

        def http(path, method="GET", payload=None):
            calls.append((path, method))
            if method == "POST":
                return replies.pop(0)
            return {"local_active": 0, "offline": True}
        with patch.object(cli, "http", http), patch.object(cli.time, "sleep", lambda s: None), \
                patch.object(cli, "save_lease", lambda *a: None):
            st, lease = cli.open_window("K5", "K5", 600, 1)
        self.assertEqual(lease, "L1")
        self.assertEqual(sum(1 for c in calls if c[1] == "POST"), 3)


class OwnUncached(unittest.TestCase):
    def test_probe_then_anchor_then_model(self):
        f = shim.own_uncached_tokens
        self.assertEqual(f({"pm_credit_probe": 26000, "probe_prompt_tokens": 30000, "pm_credit_anchored": 1,
                            "pm_anchor_age_s": 3.0}, 9999, 29000), (4000, "probe"))
        self.assertEqual(f({"pm_credit_anchored": 21408, "pm_anchor_age_s": 3.0}, 9999, 29000), (7592, "anchor"))
        self.assertEqual(f({"pm_credit_anchored": 0, "pm_anchor_age_s": None}, 9999, 29000), (9999, "model"))
        with patch.object(shim, "anchored_credit", lambda chain, est, now=None: (17840, 5.0)):
            self.assertEqual(f({}, 9999, 29000, pm_chain=[(b"k", 1)]), (11160, "anchor"))
        self.assertEqual(f(None, None, None), (0, "model"))
