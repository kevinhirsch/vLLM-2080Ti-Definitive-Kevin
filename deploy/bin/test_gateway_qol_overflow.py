#!/usr/bin/env python3
"""Lane CFG: QoL interactive overflow (SHIM_QOL_OVERFLOW=off|shadow|on), Kevin 2026-10-03 "based on quality of life".

Interactive requests overflow only when the PREDICTED local first token is late (> QOL_TTFT_S including time waited)
AND the measured remote TTFT for the prompt's size band is meaningfully sooner. Off = byte-for-byte old routing; shadow =
old routing plus qol_* telemetry; on = the prediction decides for QOL_REASONS and the lane wait. Reuses the isolated
router harness of test_gateway_local_first.py (no network, no live paths).
"""
import collections
import time
import unittest
from unittest.mock import patch

from test_gateway_local_first import RoutingBase, shim


def _remote(samples, ptok=60_000, now=None):
    now = now or time.time()
    return collections.deque([(now - 10, ptok, s) for s in samples], maxlen=4000)


class Rule(unittest.TestCase):
    def test_choice_table(self):
        c = lambda *a: shim.qol_choice(*a, ttft_target_s=15, min_gain_s=5)
        self.assertEqual(c(3, 0, 8), ("local", "fast-local"))               # small / fast locally: never overflows
        self.assertEqual(c(1, 14.5, 8), ("local", "remote-not-faster"))      # about to start: waiting alone never overflows
        self.assertEqual(c(60, 0, 8), ("remote", "remote-faster"))           # stuck behind a monster: overflows at once
        self.assertEqual(c(20, 0, 18), ("local", "remote-not-faster"))       # remote not meaningfully faster
        self.assertEqual(c(60, 0, None), ("legacy", "no-remote-samples"))    # no measured remote: no claim
        self.assertEqual(c(None, 0, 8), ("legacy", "no-local-prediction"))

    def test_remote_ttft_by_band_quantile_and_min_samples(self):
        now = time.time()
        dq = collections.deque(maxlen=4000)
        for s in (1, 2, 3, 4, 10):
            dq.append((now - 5, 50_000, float(s)))
        dq.append((now - 5, 1_000, 0.4))                       # other band
        dq.append((now - 10 ** 6, 50_000, 99.0))                # outside the window
        with patch.object(shim, "_QOL_REMOTE", dq), patch.object(shim, "_QOL_SEEDED", [True]), \
                patch.object(shim, "QOL_REMOTE_QUANTILE", 0.75), patch.object(shim, "QOL_REMOTE_MIN_SAMPLES", 5):
            self.assertEqual(shim.qol_remote_ttft(60_000, now=now), (4.0, 2, 5))   # index int(0.75*5)=3 of [1,2,3,4,10]
            self.assertEqual(shim.qol_remote_ttft(1_000, now=now), (None, 0, 1))
        with patch.object(shim, "_QOL_REMOTE", dq), patch.object(shim, "_QOL_SEEDED", [True]), \
                patch.object(shim, "QOL_REMOTE_MIN_SAMPLES", 6):
            self.assertIsNone(shim.qol_remote_ttft(60_000, now=now)[0])

    def test_decide_prefers_the_engine_anchored_credit(self):
        seen = {}

        def pred(cls, own, **kw):
            seen["own"] = own
            return own / 1000.0, {"queue_s": 0.0, "backlog_s": 0.0, "own_s": own / 1000.0}
        with patch.object(shim, "local_first_predicted_ttft", pred), \
                patch.object(shim, "anchored_credit", lambda chain, est, now=None: (50_000, 3.0)), \
                patch.object(shim, "_QOL_REMOTE", _remote([4.0] * 6)), patch.object(shim, "_QOL_SEEDED", [True]):
            q = shim.qol_decide("kevin", 60_000, 60_000, pm_chain=[("x", 1)])
        self.assertEqual(seen["own"], 10_000)                    # 60K prompt - 50K anchored = 10K uncached, not 60K
        self.assertEqual(q["qol_anchored"], 50_000)
        self.assertEqual(q["qol_would"], "local")                # 10 s local beats remote-not-much-faster
        with patch.object(shim, "local_first_predicted_ttft", lambda *a, **k: (_ for _ in ()).throw(RuntimeError())):
            self.assertEqual(shim.qol_decide("kevin", 1, 1)["qol_would"], "legacy")   # never raises


class QolRouting(RoutingBase):
    def setUp(self):
        super().setUp()
        self.active = {}
        self.stack.enter_context(patch.object(shim, "_active_set", lambda req, **kw: self.active.update(kw)))
        self.stack.enter_context(patch.object(shim, "_QOL_SEEDED", [True]))
        self.stack.enter_context(patch.object(shim, "_QOL_REMOTE", _remote([5.0] * 8)))
        self.stack.enter_context(patch.object(shim, "SLOT_POLL", 0.001))

    def predict(self, seconds):
        return patch.object(shim, "local_first_predicted_ttft",
                            lambda *a, **k: (seconds, {"queue_s": 0.0, "backlog_s": 0.0, "own_s": seconds}))

    async def test_off_is_unchanged_and_logs_nothing(self):
        with patch.object(shim, "QOL_OVERFLOW", "off"), patch.object(shim, "_inflight", 14), self.predict(3.0):
            self.assertEqual(await self.route(), "remote")
        self.assertEqual(self.events, [("remote", "big-prompt")])
        self.assertNotIn("qol_would", self.active)

    async def test_shadow_logs_the_decision_but_routes_as_before(self):
        with patch.object(shim, "QOL_OVERFLOW", "shadow"), patch.object(shim, "_inflight", 14), self.predict(3.0):
            self.assertEqual(await self.route(), "remote")
        self.assertEqual(self.events, [("remote", "big-prompt")])
        self.assertEqual((self.active["qol_at"], self.active["qol_legacy"], self.active["qol_would"]),
                         ("guard:big-prompt", "remote", "local"))

    async def test_on_sends_a_slow_local_request_remote_even_with_free_lanes(self):
        with patch.object(shim, "QOL_OVERFLOW", "on"), self.predict(60.0):
            self.assertEqual(await self.route(), "remote")
        self.assertEqual(self.events, [("remote", "big-prompt")])
        self.assertEqual(self.active["qol_would"], "remote")

    async def test_on_keeps_a_fast_local_request_local_through_a_short_wait(self):
        calls = {"n": 0}

        async def healthy(*a, **k):
            calls["n"] += 1
            if calls["n"] >= 3:
                shim._inflight = 0                       # a lane frees after a few polls
            return True
        with patch.object(shim, "QOL_OVERFLOW", "on"), patch.object(shim, "_inflight", 14), \
                patch.object(shim, "local_healthy", healthy), self.predict(3.0):
            self.assertEqual(await self.route(), "local")
        self.assertEqual(self.events, [("local", "lf-big-prompt")])   # local-first said saturated; QoL kept it local

    async def test_on_breaks_the_wait_when_local_is_predicted_late(self):
        self.ptok = 1_000                                 # no predictive guard fires: only the lane wait is judged
        with patch.object(shim, "QOL_OVERFLOW", "on"), patch.object(shim, "_inflight", 14), self.predict(60.0), \
                patch.object(shim, "_QOL_REMOTE", _remote([2.0] * 8, ptok=1_000)):
            self.assertEqual(await self.route(), "remote")
        self.assertEqual(self.events, [("remote", "qol")])
        self.assertEqual(self.active["qol_at"], "wait")

    async def test_background_is_never_governed(self):
        with patch.object(shim, "QOL_OVERFLOW", "on"), self.predict(60.0), \
                patch.object(shim, "BG_MARKERS", ["scheduled cron job"]):
            await self.route(messages=[{"role": "user", "content": "scheduled cron job: review"}])
        self.assertNotIn("qol_would", self.active)


if __name__ == "__main__":
    unittest.main()
