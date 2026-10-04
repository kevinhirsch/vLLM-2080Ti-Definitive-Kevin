#!/usr/bin/env python3
"""L164 (lane A6): in-memory durations and deadlines use the monotonic clock, so a wall-clock step cannot expire or extend them.

The gateway used time.time() for everything (143 sites, 1 monotonic). Timestamps that are written to disk or shown to people must
stay wall clock, but a deadline that lives only in memory -- the upstream-response wait, the background hold-for-recovery loop, the
local OOM backoff, the /tokenize failure backoff -- is a duration. A forward NTP step turned those into instant timeouts (spurious
failover/remote), a backward step into waits that never ended. Driven by a wall clock that jumps one hour mid-wait.

Run:  python -m pytest -q test_gateway_monotonic_deadlines.py
"""
import asyncio
import time
import types
import unittest
from unittest.mock import patch

import test_gateway_cache_model as C

shim = C.shim


def stepping_clock(step_s):
    """time.time() that jumps forward `step_s` from its second call on (an NTP step while a wait is in progress)."""
    real, n = time.time, [0]

    def clock():
        n[0] += 1
        return real() + (step_s if n[0] > 1 else 0)
    return clock


class WallClockStep(unittest.IsolatedAsyncioTestCase):
    async def test_upstream_wait_survives_a_forward_wall_clock_step(self):
        async def upstream():
            await asyncio.sleep(0.15)
            return "headers"

        req = types.SimpleNamespace()                        # no transport attribute: a test double, caller never gone
        with patch.object(shim.time, "time", stepping_clock(3600)):
            out = await shim._await_unless_gone(req, upstream(), timeout=2.0, poll=0.02)
        self.assertEqual(out, "headers")                     # with wall-clock deadlines this raised TimeoutError at once

    async def test_upstream_wait_still_times_out_on_real_time(self):
        async def stuck():
            await asyncio.sleep(30)

        with patch.object(shim.time, "time", stepping_clock(-3600)):      # a BACKWARD step must not extend the wait
            t0 = time.monotonic()
            with self.assertRaises(asyncio.TimeoutError):
                await shim._await_unless_gone(types.SimpleNamespace(), stuck(), timeout=0.2, poll=0.02)
        self.assertLess(time.monotonic() - t0, 2.0)

    def test_ttft_is_a_monotonic_difference(self):
        src = open(shim.__file__).read()
        i = src.index("async def _relay(")
        body = src[i:i + 60000]
        self.assertIn("t_relay_start = time.monotonic()", body)
        self.assertIn("_t_first[0] = time.monotonic()", body)

    def test_tokenize_failure_backoff_is_a_duration(self):
        src = open(shim.__file__).read()
        i = src.index("def _tokenize_exact")
        body = src[i:i + 2500]
        self.assertIn("time.monotonic() < _tok_fail_until", body)
        self.assertIn("_tok_fail_until = time.monotonic() + 60", body)


class DecisionClocks(unittest.IsolatedAsyncioTestCase):
    """L164 round 2: the remaining in-memory DECISION deadlines (OOM backoff, perf-breaker hold, health TTL, admission wait)."""

    def test_oom_backoff_survives_a_forward_wall_step_and_ends_on_real_time(self):
        with patch.object(shim, "OOM_BACKOFF", 120), patch.object(shim, "_pm_reset", lambda *a, **k: None), \
                patch.object(shim, "BUDGET", 4):
            shim._backoff_until, shim._backoff_mono_until = 0.0, 0.0
            try:
                shim.trigger_backoff("test")
            except Exception:
                pass
            self.assertEqual(shim.effective_budget(), 1)
            with patch.object(shim.time, "time", stepping_clock(3600 * 5)):     # NTP jumps 5 h forward mid-backoff
                self.assertEqual(shim.effective_budget(), 1)                   # wall deadline: back to BUDGET at once (full lanes into a crashed engine)
            with patch.object(shim.time, "monotonic", lambda r=time.monotonic: r() + 121):
                self.assertEqual(shim.effective_budget(), 4)                  # real elapsed time still ends it

    def test_perf_breaker_hold_is_monotonic(self):
        with patch.object(shim, "PERF_BREAKER_ENABLED", True), patch.dict(shim._PERF_STATE, {"until": time.monotonic() + 60}):
            self.assertTrue(shim.perf_breaker_active())
            with patch.object(shim.time, "time", stepping_clock(3600 * 5)):
                self.assertTrue(shim.perf_breaker_active())
            with patch.object(shim.time, "time", stepping_clock(-3600 * 5)):
                self.assertTrue(shim.perf_breaker_active())
        with patch.object(shim, "PERF_BREAKER_ENABLED", True), patch.dict(shim._PERF_STATE, {"until": time.monotonic() - 1}):
            self.assertFalse(shim.perf_breaker_active())

    async def test_health_ttl_is_a_duration(self):
        with patch.dict(shim._health, {"ok": True, "at": time.monotonic()}), patch.object(shim, "HEALTH_TTL", 5.0), \
                patch.object(shim, "LOCAL", "http://127.0.0.1:9"):             # a probe would find nothing: a cache hit is the only True
            with patch.object(shim.time, "time", stepping_clock(3600 * 5)):
                self.assertTrue(await shim.local_healthy())                    # served from the cache, no probe of :8001

    def test_wall_deadline_converts_once_to_the_monotonic_scale(self):
        m0 = time.monotonic()
        d = shim._wall_to_mono(time.time() + 30)
        self.assertAlmostEqual(d - m0, 30, delta=1.0)
        with patch.object(shim.time, "time", stepping_clock(3600)):             # after adoption a wall step cannot move it
            self.assertAlmostEqual(d - time.monotonic(), 30, delta=1.0)

    def test_decision_functions_never_read_the_wall_clock(self):
        """Ratchet: these functions decide something from a duration; time.time() must not appear in them."""
        import ast
        tree = ast.parse(open(shim.__file__).read())
        banned = {"effective_budget", "perf_breaker_active", "_wall_to_mono_never"}
        found = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and n.name in banned:
                found[n.name] = [c.lineno for c in ast.walk(n) if isinstance(c, ast.Call) and ast.unparse(c.func) == "time.time"]
        self.assertEqual({k: v for k, v in found.items() if v}, {})

    def test_admission_wait_and_service_time_use_monotonic(self):
        src = open(shim.__file__).read()
        self.assertIn("deadline = time.monotonic() + admission_wait_seconds(", src)
        self.assertIn("t_admit0 = time.monotonic()", src)
        self.assertIn("if time.monotonic() >= deadline:", src)
        self.assertIn("claimed_at = time.monotonic()", src)
        self.assertIn("_wall_to_mono(flow_t.deadline_at - flow_t.cost_s)", src)


if __name__ == "__main__":
    unittest.main()
