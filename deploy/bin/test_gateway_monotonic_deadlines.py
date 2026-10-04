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

    def test_oom_backoff_is_a_duration(self):
        with patch.object(shim, "_backoff_until", 0.0), patch.object(shim, "OOM_BACKOFF", 120), \
                patch.object(shim, "_pm_reset", lambda *a, **k: None):
            shim.trigger_backoff("test fault")
            self.assertEqual(shim.effective_budget(), 1)
            real = time.time
            with patch.object(shim.time, "time", lambda: real() + 7200):
                self.assertEqual(shim.effective_budget(), 1)             # two-hour wall step: still backing off
            self.assertLessEqual(shim._backoff_until - time.monotonic(), 120)

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


if __name__ == "__main__":
    unittest.main()
