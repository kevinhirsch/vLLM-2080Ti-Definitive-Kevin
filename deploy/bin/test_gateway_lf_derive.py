#!/usr/bin/env python3
"""Lane LF (2026-10-03): derive the local-first routing tests from the caller's patience and its real output.

MEASURED 10-02 23:21 -> 10-03 05:10: 4,179 'discretionary' remote trips, of which 2,822 were inside declared offline
windows (the predictive guards sat before the offline check), and Halo's max_tokens=16384 ceiling tripped big-out on
every turn although its p90 completion is 846 tokens.  Covered here:
  * offline window is checked BEFORE big-out/big-prompt, so window traffic is labelled local-offline;
  * big-out is decided on the caller's expected output (history), the ceiling being only the hard bound / fallback;
  * big-prompt limit = prefill rate x class patience; the crash-adaptive floor and the flat fallback still win;
  * local_first_decision(cls=...) keeps a request local while queue + backlog + own prefill fit the class budget;
  * the first-token timeout grants a kept request the time its backlog needs;
  * the output history survives a restart (rebuilt from the request log).

Run:  python -m pytest -q test_gateway_lf_derive.py
"""
import json
import os
import unittest
from unittest.mock import patch

import test_gateway_local_first as T

shim = T.shim


class DerivedBase(T.Isolated):
    """Decision-level fixture: prefill 1,000 tok/s, flow deadlines runner=900 background=600 (capped at 300)."""

    def setUp(self):
        super().setUp()
        for name, value in dict(LOCAL_FIRST_DERIVE=True, FLOW_MODE="off", HEAVY_ADMIT_BACKLOG_SECS=15.0,
                                _inflight_computed=0, prefill_tps=lambda: 1000.0,
                                _flow_deadlines=lambda: {"kevin": 0.0, "halo": 0.0, "runner": 900.0, "background": 600.0},
                                ).items():
            self.stack.enter_context(patch.object(shim, name, value))

    def d(self, cls, est_computed=2_000, backlog_s=0.0, reason="big-prompt", background=False, **kw):
        with patch.object(shim, "_inflight_computed", int(backlog_s * 1000)):
            return shim.local_first_decision(reason, background=background, units=1, reservation=60_000,
                                             est_computed=est_computed, cls=cls, **kw)


class Derived(DerivedBase):
    def test_budgets_follow_the_class_and_are_capped_at_the_relay_first_token_wait(self):
        self.assertEqual(shim.local_first_budget_s("kevin"), 300.0)
        self.assertEqual(shim.local_first_budget_s("halo"), 300.0)
        self.assertEqual(shim.local_first_budget_s("runner"), 300.0)       # deadline 900 -> relay cap
        with patch.object(shim, "_flow_deadlines", lambda: {"runner": 120.0, "background": 45.0}):
            self.assertEqual(shim.local_first_budget_s("runner"), 120.0)
            self.assertEqual(shim.local_first_budget_s("background"), 45.0)

    def test_halo_waits_for_a_cold_prefill_behind_a_backlog_that_fits_its_patience(self):
        # 52K cold prompt = 52 s own + 100 s ahead = 152 s <= 300 s -> local (the flat rule sent this remote at 15 s)
        self.assertEqual(self.d("halo", est_computed=52_000, backlog_s=100), (True, "capacity"))
        # 52 + 280 = 332 s > 300 s: it would not get a first token inside the relay wait -> the valve
        self.assertEqual(self.d("halo", est_computed=52_000, backlog_s=280), (False, "saturated:deadline"))

    def test_background_uses_its_own_shorter_deadline(self):
        with patch.object(shim, "_flow_deadlines", lambda: {"background": 60.0}):
            self.assertEqual(self.d("background", est_computed=10_000, backlog_s=30, background=True), (True, "capacity"))
            self.assertEqual(self.d("background", est_computed=10_000, backlog_s=55, background=True),
                             (False, "saturated:deadline"))

    def test_kevin_never_queues_behind_more_than_the_configured_backlog_bound(self):
        self.assertEqual(self.d("kevin", est_computed=2_000, backlog_s=10), (True, "capacity"))
        self.assertEqual(self.d("kevin", est_computed=2_000, backlog_s=40, reason="monster"),
                         (False, "saturated:prefill-backlog"))
        # the same backlog is fine for halo: nobody is waiting at a keyboard
        self.assertEqual(self.d("halo", est_computed=2_000, backlog_s=40, reason="monster"), (True, "capacity"))

    def test_monster_is_no_longer_unconditionally_remote_in_cache_aware_mode(self):
        with patch.object(shim, "USE_COMPUTED_COST", 1):
            self.assertEqual(self.d("runner", est_computed=3_500, backlog_s=40, reason="monster"), (True, "capacity"))

    def test_queue_wait_of_the_class_counts_against_the_budget(self):
        with patch.object(shim, "FLOW_MODE", "enforce"), patch.object(shim, "flow_expected_wait", lambda *a, **k: 280.0):
            self.assertEqual(self.d("halo", est_computed=30_000), (False, "saturated:deadline"))
        with patch.object(shim, "FLOW_MODE", "enforce"), patch.object(shim, "flow_expected_wait", lambda *a, **k: 20.0):
            self.assertEqual(self.d("halo", est_computed=30_000), (True, "capacity"))

    def test_flat_queue_wait_is_the_fallback_only(self):
        now = 1_000_000.0
        for w in (40.0, 50.0):
            shim._note_admission_wait(w, now=now - 5)
        self.assertEqual(self.d("halo", est_computed=1_000, now=now), (True, "capacity"))     # 45 s fits 300 s
        with patch.object(shim, "LOCAL_FIRST_DERIVE", False):
            self.assertEqual(self.d("halo", est_computed=1_000, now=now), (False, "saturated:queue-wait"))
        with patch.object(shim, "prefill_tps", lambda: 0.0):                                  # no measured rate yet
            self.assertEqual(self.d("halo", est_computed=1_000, now=now), (False, "saturated:queue-wait"))

    def test_hard_facts_still_saturate(self):
        with patch.object(shim, "_inflight_reserved_tokens", 490_000):
            self.assertEqual(self.d("halo"), (False, "saturated:tokens"))
        with patch.object(shim, "_health", {"ok": False}):
            self.assertEqual(self.d("halo"), (False, "local-unhealthy"))
        with patch.object(shim, "LOCAL_FIRST", False):
            self.assertEqual(self.d("halo"), (False, "policy-off"))

    def test_without_a_class_the_flat_policy_applies(self):
        with patch.object(shim, "_inflight", 14):
            self.assertEqual(self.d(None), (False, "saturated:lanes"))


class BigPrompt(DerivedBase):
    def test_limit_is_prefill_rate_times_patience_and_never_below_the_configured_value(self):
        with patch.object(shim, "BIG_PROMPT", 24_000):
            self.assertEqual(shim.big_prompt_limit("halo"), 300_000)
            self.assertEqual(shim.big_prompt_limit("runner"), 300_000)
            with patch.object(shim, "_flow_deadlines", lambda: {"runner": 10.0}):
                self.assertEqual(shim.big_prompt_limit("runner"), 24_000)          # 10 s x 1000 < configured
            with patch.object(shim, "_big_prompt_restore", 24_000):                # crash-adaptive floor active
                self.assertEqual(shim.big_prompt_limit("halo"), 24_000)
            with patch.object(shim, "LOCAL_FIRST_DERIVE", False):
                self.assertEqual(shim.big_prompt_limit("halo"), 24_000)
            self.assertEqual(shim.big_prompt_limit(None), 24_000)
            with patch.object(shim, "prefill_tps", lambda: 0.0):
                self.assertEqual(shim.big_prompt_limit("halo"), 24_000)
        with patch.object(shim, "BIG_PROMPT", 0):
            self.assertEqual(shim.big_prompt_limit("halo"), 0)


class ExpectedOutput(T.Isolated):
    def setUp(self):
        super().setUp()
        self.stack.enter_context(patch.object(shim, "_OUT_HIST", {}))
        self.stack.enter_context(patch.object(shim, "EXPECTED_OUTPUT", True))
        self.stack.enter_context(patch.object(shim, "EXPECTED_OUTPUT_MIN_SAMPLES", 20))
        self.stack.enter_context(patch.object(shim, "EXPECTED_OUTPUT_QUANTILE", 0.95))

    def feed(self, client, values):
        for v in values:
            shim.note_output_tokens(client, v)

    def test_ceiling_until_there_is_enough_history(self):
        self.feed("halo-hermes", [300] * 19)
        self.assertEqual(shim.expected_output_tokens("halo-hermes", 16384), (16384, "ceiling"))
        self.feed("halo-hermes", [300])
        self.assertEqual(shim.expected_output_tokens("halo-hermes", 16384), (300, "history-p95"))

    def test_quantile_of_history_bounded_by_the_requested_ceiling(self):
        self.feed("halo-hermes", list(range(100, 2100, 20)))           # 100 values, 100..2080
        tok, src = shim.expected_output_tokens("halo-hermes", 16384)
        self.assertEqual(src, "history-p95")
        self.assertTrue(1900 <= tok <= 2080, tok)
        self.assertEqual(shim.expected_output_tokens("halo-hermes", 500)[0], 500)    # ceiling is the hard bound

    def test_a_client_that_really_writes_long_stays_big_out(self):
        self.feed("writer", [20_000] * 30)
        self.assertEqual(shim.expected_output_tokens("writer", 32_000), (20_000, "history-p95"))

    def test_switch_and_unbounded_requests(self):
        self.feed("c", [10] * 30)
        with patch.object(shim, "EXPECTED_OUTPUT", False):
            self.assertEqual(shim.expected_output_tokens("c", 16384), (16384, "ceiling"))
        self.assertEqual(shim.expected_output_tokens("c", 0), (0, "ceiling"))
        self.assertEqual(shim.expected_output_tokens("unknown", 16384), (16384, "ceiling"))

    def test_history_is_bounded_per_client(self):
        self.feed("c", range(1000))
        self.assertLessEqual(len(shim._OUT_HIST["c"]), shim.EXPECTED_OUTPUT_HISTORY)

    def test_history_is_rebuilt_from_the_request_log_after_a_restart(self):
        now = 1_800_000_000.0
        path = shim._history_day_file(now)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rows = [{"t": now - 600 + i, "client": "halo-hermes", "outtok": 100 + i, "status": 200} for i in range(25)]
        rows += [{"t": now - 30, "client": "halo-hermes", "outtok": 99999, "status": 500},      # failed: not evidence
                 {"t": now + 5, "client": "halo-hermes", "outtok": 77777, "status": 200},       # after process start
                 {"t": now - 90_000, "client": "halo-hermes", "outtok": 55555, "status": 200}]   # older than a day
        with open(path, "w") as fh:
            fh.write("\n".join(json.dumps(r) for r in rows) + "\n")
        try:
            data = shim.output_history_restore_blocking(now, now)
            self.assertEqual(len(data["halo-hermes"]), 25)
            self.assertEqual(shim.output_history_restore(data), 25)
            self.assertEqual(shim.expected_output_tokens("halo-hermes", 16384)[1], "history-p95")
        finally:
            os.unlink(path)


class FirstToken(DerivedBase):
    def test_a_kept_request_behind_a_backlog_gets_the_time_the_backlog_needs(self):
        body = json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode()
        with patch.object(shim, "remote_ok", lambda: True), patch.object(shim, "FIRST_TOKEN_MAX", 60.0), \
                patch.object(shim, "FIRST_TOKEN_BASE", 15.0), patch.object(shim, "_est_tokens", lambda b: 3_000), \
                patch.object(shim, "BIG_PROMPT", 24_000):
            self.assertAlmostEqual(shim.first_token_timeout(body, 1, local=True), 18.0)           # idle engine: as before
            with patch.object(shim, "_inflight_computed", 100_000):                                 # 100 s ahead
                self.assertAlmostEqual(shim.first_token_timeout(body, 1, local=True), 118.0)
                self.assertEqual(shim.first_token_timeout(body, 1, local=False), 18.0)             # remote relay unchanged
            with patch.object(shim, "_inflight_computed", 900_000):
                self.assertEqual(shim.first_token_timeout(body, 1, local=True), 300.0)             # capped at the LF cap
            with patch.object(shim, "LOCAL_FIRST_DERIVE", False), patch.object(shim, "_inflight_computed", 100_000):
                self.assertAlmostEqual(shim.first_token_timeout(body, 1, local=True), 18.0)


class Routing(T.RoutingBase):
    def setUp(self):
        super().setUp()
        for name, value in dict(LOCAL_FIRST_DERIVE=True, EXPECTED_OUTPUT=True, _OUT_HIST={},
                                EXPECTED_OUTPUT_MIN_SAMPLES=20, EXPECTED_OUTPUT_QUANTILE=0.95,
                                USE_COMPUTED_COST=0, perf_breaker_active=lambda: False).items():
            self.stack.enter_context(patch.object(shim, name, value))
        self.ptok = 3_000

    def client(self):
        return shim._friendly_client(T.Request())["name"]

    async def test_window_traffic_is_labelled_local_offline_not_discretionary(self):
        self.ptok = 60_000
        with patch.object(shim, "_local_offline", lambda now=None: True), patch.object(shim, "LOCAL_FIRST", False), \
                patch.dict(shim._OFFLINE, {"until": 10 ** 12, "reason": "test window", "by": "t", "t0": 1.0}):
            self.assertEqual(await self.route(max_tokens=16384), "remote")
        self.assertEqual(self.events, [("remote", "local-offline")])

    async def test_halo_style_ceiling_does_not_trip_big_out_once_its_real_output_is_known(self):
        with patch.object(shim, "LOCAL_FIRST", False):                       # isolate the guard from local-first
            self.assertEqual(await self.route(max_tokens=16384), "remote")    # no history: the ceiling decides, as before
            self.assertEqual(self.events, [("remote", "big-out")])
            for _ in range(30):
                shim.note_output_tokens(self.client(), 800)
            self.assertEqual(await self.route(max_tokens=16384), "local")
            self.assertEqual(self.events, [("local", "-")])
            for _ in range(30):
                shim.note_output_tokens(self.client(), 20_000)               # a caller that really writes long
            self.assertEqual(await self.route(max_tokens=32_000), "remote")
            self.assertEqual(self.events, [("remote", "big-out")])

    async def test_ceiling_stays_a_hard_bound(self):
        for _ in range(30):
            shim.note_output_tokens(self.client(), 50_000)
        with patch.object(shim, "LOCAL_FIRST", False):
            self.assertEqual(await self.route(max_tokens=1000), "local")     # asked for 1000: expected is min(50000, 1000)

    async def test_big_prompt_below_the_derived_limit_is_not_a_big_prompt(self):
        self.ptok = 60_000
        self.assertEqual(await self.route(max_tokens=1000), "local")
        self.assertEqual(self.events, [("local", "-")])
        with patch.object(shim, "LOCAL_FIRST_DERIVE", False), patch.object(shim, "LOCAL_FIRST", False):
            self.assertEqual(await self.route(max_tokens=1000), "remote")     # the flat 24K rule, kept as the fallback
            self.assertEqual(self.events, [("remote", "big-prompt")])


if __name__ == "__main__":
    unittest.main()
