#!/usr/bin/env python3
"""Lane CR (2026-10-03): chain-route telemetry + warm-continuation priority (all default OFF).

MEASURED (trial, post-LF 05:44->06:20): Halo sent ~48% of its prompt tokens remote; 31 of those turns
carried a WARM local chain (pm_credit >= 50% of the prompt: monster 11, prefill 10, tokens 4, perf 3).
Each warm turn served remote costs money AND breaks the local chain for the next turn (cause b). A
warm continuation's uncached remainder (~1-2K tokens) fits the step budget a running cold chunk leaves
(align mode: 3632 budget - one 1856 block), but it queues FIFO behind cold arrivals and local-first
judges it on the whole cold backlog. Covered here:
  * the cost model reports the unshaved matched prefix and rounds credit to SHIM_PREFIX_CREDIT_UNIT;
  * chain-route memory: a continuation finds the route that served its deepest known prefix;
  * warm_continuation() thresholds; the local body gets engine priority only for warm, non-control turns;
  * local_first_decision(warm=True) drops the cold backlog from the predicted TTFT, only when enabled.

Run:  python -m pytest -q test_gateway_cr_chain.py
"""
import json
import unittest
from unittest.mock import patch

import test_gateway_cache_model as C
import test_gateway_lf_derive as D

shim = C.shim
convo = C.convo


class ChainRoute(unittest.TestCase):
    def setUp(self):
        self.p = [patch.object(shim, "_CHAIN_ROUTE", type(shim._CHAIN_ROUTE)()),
                  patch.object(shim, "_PM_NODES", type(shim._PM_NODES)()),
                  patch.object(shim, "_PM_PAIRS", type(shim._PM_PAIRS)(maxlen=500))]
        for x in self.p:
            x.start()
            self.addCleanup(x.stop)

    def test_continuation_finds_the_route_of_its_previous_turn(self):
        t1 = shim._pm_predict(convo("A", 10), 20000)["chain"]
        shim._chain_route_note(t1, "remote", now=1000.0)
        t2 = shim._pm_predict(convo("A", 11), 22000)["chain"]
        route, age, depth = shim._chain_route_lookup(t2, now=1010.0)
        self.assertEqual((route, round(age), depth), ("remote", 10, len(t1) - 1))
        shim._chain_route_note(t2, "local", now=1020.0)
        t3 = shim._pm_predict(convo("A", 12), 24000)["chain"]
        self.assertEqual(shim._chain_route_lookup(t3, now=1030.0)[0], "local")

    def test_unrelated_conversation_and_expiry(self):
        shim._chain_route_note(shim._pm_predict(convo("A", 10), 20000)["chain"], "local", now=1000.0)
        other = shim._pm_predict(convo("B", 11), 22000)["chain"]
        # B shares only the root (tools + system): the system message node is not a finished request's chain end
        self.assertEqual(shim._chain_route_lookup(other, now=1001.0), (None, None, -1))
        mine = shim._pm_predict(convo("A", 11), 22000)["chain"]
        self.assertIsNone(shim._chain_route_lookup(mine, now=1000.0 + shim.PREFIX_MODEL_TTL_SECS + 1)[0])

    def test_only_served_routes_are_remembered(self):
        chain = shim._pm_predict(convo("A", 3), 6000)["chain"]
        shim._chain_route_note(chain, "rejected", now=1.0)
        shim._chain_route_note([], "local", now=1.0)
        self.assertEqual(len(shim._CHAIN_ROUTE), 0)


class CostModel(unittest.TestCase):
    def setUp(self):
        for name, value in dict(_PM_NODES=type(shim._PM_NODES)(), _PM_PAIRS=type(shim._PM_PAIRS)(maxlen=500)).items():
            p = patch.object(shim, name, value); p.start(); self.addCleanup(p.stop)

    def _warm(self, unit):
        with patch.object(shim, "PREFIX_CREDIT_UNIT", unit):
            p = shim._pm_predict(convo("A", 20), 40000)
            shim._pm_commit(p["chain"]); shim._pm_ready(p["chain"])
            return shim._pm_predict(convo("A", 21), 42000)

    def test_matched_is_reported_and_credit_follows_the_unit(self):
        block = self._warm(0)
        self.assertGreater(block["matched"], block["credit"])
        self.assertEqual(block["credit"] % shim.prefix_align_tokens(), 0)
        self.setUp()
        fine = self._warm(64)
        self.assertEqual(fine["credit"] % 64, 0)
        self.assertGreaterEqual(fine["credit"], block["credit"])
        self.assertLess(fine["computed"], block["computed"] + 1)


class WarmPriority(unittest.TestCase):
    def test_thresholds_and_default_off(self):
        warm = {"credit": 30000, "computed": 900}
        self.assertFalse(shim.warm_continuation(warm))                       # default OFF
        with patch.object(shim, "WARM_PRIORITY", True):
            self.assertTrue(shim.warm_continuation(warm))
            self.assertFalse(shim.warm_continuation({"credit": 2000, "computed": 900}))     # not a chain
            self.assertFalse(shim.warm_continuation({"credit": 30000, "computed": 9000}))   # big remainder

    def _body(self, warm, enabled, halo_control=False):
        class R(dict):
            remote = "127.0.0.1"
            headers = {"X-Client": "halo-hermes"}
            path = "/v1/chat/completions"
        r = R()
        if warm:
            r["cr_warm_priority"] = True
        body = convo("A", 2, tools=None)
        with patch.object(shim, "WARM_PRIORITY", enabled), \
                patch.object(shim, "_halo_control_request", lambda *_: halo_control), \
                patch.object(shim, "_alias_for_request", lambda *_: {"kind": "default", "name": "estate"}):
            return json.loads(shim._prepare_local_body(r, body, False))

    def test_local_body_priority_only_for_warm_when_enabled(self):
        self.assertEqual(self._body(True, True).get("priority"), shim.WARM_PRIORITY_VALUE)
        self.assertNotIn("priority", self._body(True, False))
        self.assertNotIn("priority", self._body(False, True))
        self.assertEqual(self._body(True, True, halo_control=True).get("priority"), -100)   # control wins


class WarmLocalFirst(D.DerivedBase):
    """DerivedBase drives its own isolated shim instance (D.shim), so patch that one."""

    def test_warm_continuation_is_not_judged_on_the_cold_backlog(self):
        shim = D.shim
        # 1.5K own + 400 s cold backlog: saturated for a cold request ...
        self.assertEqual(self.d("halo", est_computed=1_500, backlog_s=400, reason="monster"), (False, "saturated:deadline"))
        # ... unchanged with warm=True while the feature is off ...
        self.assertEqual(self.d("halo", est_computed=1_500, backlog_s=400, reason="monster", warm=True),
                         (False, "saturated:deadline"))
        # ... and local when enabled: it is admitted with priority into the budget the cold chunk leaves
        with patch.object(shim, "WARM_PRIORITY", True):
            self.assertEqual(self.d("halo", est_computed=1_500, backlog_s=400, reason="monster", warm=True),
                             (True, "capacity"))
            # a hard memory fact still wins
            with patch.object(shim, "_memory_available", lambda *_: False):
                self.assertEqual(self.d("halo", est_computed=1_500, backlog_s=400, reason="monster", warm=True),
                                 (False, "saturated:tokens"))


if __name__ == "__main__":
    unittest.main()
