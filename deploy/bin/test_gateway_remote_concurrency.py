#!/usr/bin/env python3
"""Lane A6 extra: DeepSeek answers 429 'Your current concurrency is 5, which exceeds your concurrency limit of 5 based on your
remaining balance' and the gateway passed that to the client (LIVE 2026-10-03 09:29, 3 in one second). The remote concurrency gate
learns the cap from the refusal, queues the request for a slot and re-sends it; nothing was billed because the provider refused
before generating.

Run:  python -m pytest -q test_gateway_remote_concurrency.py
"""
import asyncio
import json
import unittest
from unittest.mock import patch

import aiohttp
from aiohttp import web

import test_gateway_cache_model as C

shim = C.shim

LIMIT = 5          # the provider's stated limit; a request that SEES current concurrency >= LIMIT is refused (live: '5 exceeds 5')


def msg(current, limit):
    return json.dumps({"error": {"message": f"Too many requests. Your current concurrency is {current}, which exceeds your concurrency "
                                            f"limit of {limit} based on your remaining balance. Please top up your balance to restore "
                                            f"your concurrency. (request_id: x)", "type": "rate_limit_error"}})


class Learn(unittest.TestCase):
    def setUp(self):
        shim._RC.clear()
        shim._RC_STATS.clear()

    def test_live_message_learns_limit_minus_one(self):
        self.assertEqual(shim._rc_learn("p", json.loads(msg(5, 5))["error"]["message"]), 4)

    def test_current_above_limit_uses_the_smaller_number(self):
        self.assertEqual(shim._rc_learn("p", msg(9, 6)), 5)

    def test_cap_never_below_one_and_unnumbered_429_learns_nothing(self):
        self.assertEqual(shim._rc_learn("p", msg(1, 1)), 1)
        self.assertIsNone(shim._rc_learn("q", "Too many requests"))

    def test_learned_cap_expires(self):
        with patch.object(shim, "REMOTE_CONCURRENCY_TTL_S", 0.0):
            shim._rc_learn("p", msg(5, 5))
            self.assertIsNone(shim._rc_cap(shim._rc_state("p")))

    def test_static_cap_and_learned_cap_take_the_tighter(self):
        with patch.object(shim, "REMOTE_CONCURRENCY_CAP", 3):
            self.assertEqual(shim._rc_cap(shim._rc_state("p")), 3)
            shim._rc_learn("p", msg(3, 3))
            self.assertEqual(shim._rc_cap(shim._rc_state("p")), 2)


class Gate(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        shim._RC.clear()
        shim._RC_STATS.clear()
        self.active_now = 0
        self.max_seen = 0
        self.refused = 0
        self.served = 0
        self.always_429 = False
        self.hold_s = 0.08

        async def upstream(request):
            body = await request.json()
            current = self.active_now + 1
            if self.always_429 or current >= LIMIT:
                self.refused += 1
                return web.Response(status=429, text=msg(current, LIMIT), content_type="application/json")
            self.active_now += 1
            self.max_seen = max(self.max_seen, self.active_now)
            try:
                await asyncio.sleep(self.hold_s)
            finally:
                self.active_now -= 1
            self.served += 1
            if body.get("stream"):
                resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await resp.prepare(request)
                await resp.write(b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"ok"}}]}\n\n')
                await resp.write(b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1}}\n\n')
                await resp.write(b"data: [DONE]\n\n")
                return resp
            return web.json_response({"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                                      "usage": {"prompt_tokens": 3, "completion_tokens": 1}})

        app = web.Application()
        app.router.add_post("/v1/chat/completions", upstream)
        self.up = web.AppRunner(app)
        await self.up.setup()
        await web.TCPSite(self.up, "127.0.0.1", 0).start()
        self.base = "http://127.0.0.1:%d" % self.up.addresses[0][1]
        self.active = {}

        async def front(request):
            body = await request.read()
            kind, payload = await shim._relay_gated(request, self.base, "/v1/chat/completions", body, None,
                                                    bool(json.loads(body).get("stream")), provider_name="deepseek-test")
            return payload

        fapp = web.Application()
        fapp.router.add_post("/v1/chat/completions", front)
        self.front_runner = web.AppRunner(fapp)
        await self.front_runner.setup()
        await web.TCPSite(self.front_runner, "127.0.0.1", 0).start()
        self.front = "http://127.0.0.1:%d/v1/chat/completions" % self.front_runner.addresses[0][1]
        self._p = [patch.object(shim, "LOCAL", "http://127.0.0.1:9"),         # the upstream under test is the REMOTE
                   patch.object(shim, "_active_set", lambda req, **kw: self.active.update(kw)),
                   patch.object(shim, "_pm_prefill_done", lambda req: None)]
        for p in self._p:
            p.start()

    async def asyncTearDown(self):
        for p in self._p:
            p.stop()
        await self.front_runner.cleanup()
        await self.up.cleanup()

    async def post(self, stream):
        async with aiohttp.ClientSession() as s:
            async with s.post(self.front, json={"model": "m", "stream": stream, "messages": [{"role": "user", "content": "q"}]}) as r:
                await r.read()
                return r.status

    async def test_burst_over_the_cap_is_queued_not_refused(self):
        statuses = await asyncio.gather(*[self.post(i % 2 == 0) for i in range(14)])
        self.assertEqual(statuses, [200] * 14, statuses)
        self.assertEqual(self.served, 14)
        self.assertGreater(self.refused, 0)                       # the first burst did reach the provider's wall ...
        self.assertEqual(shim._RC_STATS["learned"], self.refused)
        self.assertEqual(shim._rc_cap(shim._rc_state(self.base)), LIMIT - 1)
        self.assertLessEqual(self.max_seen, LIMIT - 1)
        self.assertEqual(shim._RC_STATS["passed_429"], 0)

    async def test_once_learned_the_next_burst_never_touches_the_wall(self):
        await asyncio.gather(*[self.post(False) for _ in range(10)])
        before = self.refused
        self.assertEqual(await asyncio.gather(*[self.post(True) for _ in range(12)]), [200] * 12)
        self.assertEqual(self.refused, before)                    # gated client-side: zero new refusals
        facts = shim.remote_concurrency_facts()
        self.assertEqual(facts["providers"][self.base]["cap"], LIMIT - 1)
        self.assertEqual(facts["providers"][self.base]["inflight"], 0)

    async def test_a_provider_that_always_refuses_is_passed_through_after_the_retry_bound(self):
        self.always_429 = True
        with patch.object(shim, "REMOTE_CONCURRENCY_RETRIES", 2):
            st = await self.post(False)
        self.assertEqual(st, 429)
        self.assertEqual(self.refused, 3)                         # first send + 2 re-sends, then the provider's own 429
        self.assertEqual(shim._RC_STATS["passed_429"], 1)
        self.assertEqual(shim._rc_state(self.base)["inflight"], 0)  # no slot leaked

    async def test_learning_off_is_the_old_pass_through(self):
        self.always_429 = True
        with patch.object(shim, "REMOTE_CONCURRENCY_LEARN", False):
            self.assertEqual(await self.post(False), 429)
        self.assertEqual(self.refused, 1)

    async def test_queue_timeout_answers_429_with_a_reason_and_leaks_nothing(self):
        shim._rc_learn(self.base, msg(2, 2))                      # cap 1
        self.hold_s = 0.6
        with patch.object(shim, "REMOTE_CONCURRENCY_QUEUE_WAIT_S", 0.2):
            a, b = await asyncio.gather(self.post(False), self.post(False))
        self.assertEqual(sorted([a, b]), [200, 429])
        self.assertEqual(shim._rc_state(self.base)["inflight"], 0)
        self.assertEqual(len(shim._rc_state(self.base)["waiters"]), 0)

    async def test_knobs_are_hot_reloadable(self):
        for k in ("SHIM_REMOTE_CONCURRENCY_LEARN", "SHIM_REMOTE_CONCURRENCY_CAP", "SHIM_REMOTE_CONCURRENCY_TTL_S",
                  "SHIM_REMOTE_CONCURRENCY_QUEUE_WAIT_S", "SHIM_REMOTE_CONCURRENCY_RETRIES"):
            self.assertIn(k, shim._CFG)


if __name__ == "__main__":
    unittest.main()
