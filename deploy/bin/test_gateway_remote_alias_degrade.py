#!/usr/bin/env python3
"""Lane GW3 (2026-10-03): a request pinned to the paid remote (the builtin-remote aliases, Halo's "estate-remote") is
served LOCALLY when the remote cannot be used, instead of being forwarded into a certain 503/429.

Live evidence: DeepSeek out of balance, breaker open, halo-hermes' 28.9K-token estate-remote turns answered 503 and
thrown away.  Kevin's rule "Local First Remote For Spikes": remote is a valve, never a dependency.

Run: python -m pytest -q test_gateway_remote_alias_degrade.py
"""
import json
import time
import unittest
from unittest.mock import AsyncMock, patch

import test_gateway_local_first as lf

shim = lf.shim
REAL_REMOTE_OK = shim.remote_ok
REAL_RELAY = shim._relay
REAL_ACTIVE_SET = shim._active_set
REAL_RECORD_EVENT = shim.record_event


class Upstream:
    """Just enough aiohttp response for a non-streaming _relay."""
    status = 200
    headers = {"Content-Type": "application/json"}

    async def read(self):
        return b'{"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}'

    async def release(self):
        pass


class Base(lf.RoutingBase):
    """RoutingBase with the REAL remote_ok / _spend_allows_overflow wiring replaced by the knobs the production predicate
    reads (REMOTE_ENABLED, the dead window, the balance breaker, the spend ledger)."""

    def setUp(self):
        super().setUp()
        self.active = {}
        self.sent = []
        self.stack.enter_context(patch.object(shim, "remote_ok", REAL_REMOTE_OK))
        self.stack.enter_context(patch.object(shim, "REMOTE_ENABLED", True))
        self.stack.enter_context(patch.object(shim, "_remote_dead_until", 0.0))
        self.stack.enter_context(patch.object(shim, "_REMOTE_BALANCE", {**shim._REMOTE_BALANCE, "exhausted": False}))
        self.cap_room = True
        self.stack.enter_context(patch.object(shim, "_spend_allows_overflow", lambda p, m, model=None: self.cap_room))
        self.stack.enter_context(patch.object(shim, "_active_set",
                                              lambda req, **kw: self.active.setdefault(id(req), {}).update(kw)))
        self.stack.enter_context(patch.object(shim, "_REMOTE_ALIAS_DEGRADED", type(shim._REMOTE_ALIAS_DEGRADED)()))

        async def relay(request, base, path, body, key, streaming, *a, **k):
            self.calls.append("local" if base == shim.LOCAL else "remote")
            self.sent.append((base, json.loads(body)))
            return "ok", "local" if base == shim.LOCAL else "remote"
        self.stack.enter_context(patch.object(shim, "_relay", relay))

    def exhaust(self):
        shim._REMOTE_BALANCE["exhausted"] = True

    async def route(self, **kw):
        self.sent.clear()
        return await super().route(**kw)

    def degraded(self):
        return [v for v in self.active.values() if v.get("degraded")]


class AliasDegrades(Base):
    async def test_balance_exhausted_estate_remote_is_served_locally(self):
        self.exhaust()
        self.assertEqual(await self.route(model="estate-remote"), "local")
        self.assertEqual(self.calls, ["local"])
        self.assertEqual(self.sent[0][1]["model"], shim._local_model_name())       # rewritten, not "estate-remote"
        (d,) = self.degraded()
        self.assertEqual((d["degraded"], d["degraded_reason"]), ("remote-alias-local", "balance-exhausted"))
        self.assertEqual(dict(shim._REMOTE_ALIAS_DEGRADED), {"remote-alias:balance-exhausted": 1})

    async def test_healthy_remote_still_goes_remote_untouched(self):
        self.assertEqual(await self.route(model="estate-remote"), "remote")
        self.assertEqual(self.calls, ["remote"])
        self.assertEqual(self.events, [("remote", "alias")])
        self.assertEqual(self.degraded(), [])

    async def test_breaker_open_degrades(self):
        with patch.object(shim, "_remote_dead_until", time.time() + 600):
            self.assertEqual(await self.route(model="estate-remote"), "local")
        self.assertEqual(self.degraded()[0]["degraded_reason"], "breaker-open")

    async def test_no_remote_configured_degrades(self):
        with patch.object(shim, "REMOTE_ENABLED", False):
            self.assertEqual(await self.route(model="estate-remote"), "local")
        self.assertEqual(self.degraded()[0]["degraded_reason"], "no-remote-configured")

    async def test_cap_without_room_degrades_and_never_forwards(self):
        self.cap_room = False
        self.assertEqual(await self.route(model="estate-remote"), "local")
        self.assertEqual(self.calls, ["local"])
        self.assertEqual(self.degraded()[0]["degraded_reason"], "cap-no-room")

    async def test_pro_alias_degrades_too_and_prices_the_cap_check_with_the_pro_model(self):
        seen = []
        self.stack.enter_context(patch.object(shim, "_spend_allows_overflow",
                                              lambda p, m, model=None: seen.append(model) or False))
        self.assertEqual(await self.route(model="estate-remote-pro"), "local")
        self.assertEqual(seen[-1], shim.REMOTE_PRO_MODEL)

    async def test_reservation_backed_alias_is_not_degraded_by_the_global_cap(self):
        self.cap_room = False
        rid = "a" * 32
        self.assertEqual(await self.route(model=f"estate-remote.{rid}"), "remote")      # its own reservation pays
        self.exhaust()
        self.assertEqual(await self.route(model=f"estate-remote.{rid}"), "local")       # but a dead remote still degrades

    async def test_degraded_request_goes_through_the_normal_local_path(self):
        """Admission applies (the lane wait loop runs); it is pinned to local, so a saturated engine yields a 503 that
        names the cause rather than a paid overflow."""
        self.exhaust()
        waits = []
        self.stack.enter_context(patch.object(shim, "_inflight", 14))
        self.stack.enter_context(patch.object(shim, "effective_budget", lambda: 14))
        self.stack.enter_context(patch.object(shim, "admission_wait_seconds", lambda *a, **k: waits.append(a) or 0.2))
        r = await self.route(model="estate-remote")
        self.assertTrue(waits, "the local admission wait was consulted")
        self.assertEqual(r.status, 503)
        self.assertIn("estate-remote", json.loads(r.body)["error"]["message"])
        self.assertEqual(self.calls, [])

    async def test_local_down_and_remote_unusable_is_a_clear_503_naming_the_cause(self):
        self.exhaust()
        with patch.object(shim, "local_healthy", AsyncMock(return_value=False)):
            r = await self.route(model="estate-remote")
        self.assertEqual(r.status, 503)
        self.assertIn("balance-exhausted", json.loads(r.body)["error"]["message"])
        self.assertEqual(self.calls, [])

    async def test_kill_switch_restores_forward_and_fail(self):
        self.exhaust()
        with patch.object(shim, "REMOTE_ALIAS_DEGRADE", False):
            self.assertEqual(await self.route(model="estate-remote"), "remote")
        self.assertEqual(self.events, [("remote", "alias")])

    async def test_custom_remote_alias_is_not_rewritten(self):
        """Class hunt: a custom endpoint is another provider (maybe vision, maybe free); it is not the DeepSeek valve."""
        self.exhaust()
        with patch.object(shim, "_alias_for_request", lambda _b: {
                "name": "c", "kind": "custom-remote", "endpoint": {"base": "https://example.invalid"}}):
            self.assertEqual(await self.route(model="c"), "remote")
        self.assertEqual(self.degraded(), [])


class FullLocalUnchanged(Base):
    async def test_local_only_still_refuses_remote_aliases(self):
        self.exhaust()
        with patch.object(shim, "LOCAL_ONLY", 1):
            r = await self.route(model="estate-remote")
        self.assertEqual(r.status, 503)
        self.assertEqual(json.loads(r.body)["error"]["type"], "full_local_remote_disabled")
        self.assertEqual(self.events, [("rejected", "full-local-remote-alias")])
        self.assertEqual(self.calls, [])
        self.assertEqual(self.degraded(), [])


class RouteIntent(Base):
    async def test_intent_with_no_cap_room_is_served_locally(self):
        self.cap_room = False
        self.assertEqual(await self.route(headers={"X-Gateway-Route-Intent": "remote"}), "local")
        self.assertEqual(self.degraded()[0]["degraded"], "remote-intent-local")

    async def test_intent_with_room_goes_remote(self):
        self.assertEqual(await self.route(headers={"X-Gateway-Route-Intent": "remote"}), "remote")
        self.assertEqual(self.events, [("remote", "intent")])


class First402(Base):
    async def test_the_request_that_learns_the_balance_is_out_is_rerouted_not_handed_a_402(self):
        responses = []

        async def forward_remote(request, path, body, streaming, endpoint=None, model=None):
            responses.append("remote")
            self.exhaust()                                   # what _note_remote_status does on a 402
            return type("R", (), {"status": 402})()
        self.stack.enter_context(patch.object(shim, "_forward_remote", forward_remote))
        self.assertEqual(await self.route(model="estate-remote"), "local")
        self.assertEqual(responses, ["remote"])              # asked once, then served locally
        self.assertEqual(self.calls, ["local"])

    async def test_a_402_that_did_not_open_the_breaker_is_passed_through(self):
        async def forward_remote(request, path, body, streaming, endpoint=None, model=None):
            return type("R", (), {"status": 402})()
        self.stack.enter_context(patch.object(shim, "_forward_remote", forward_remote))
        r = await self.route(model="estate-remote")
        self.assertEqual(r.status, 402)


class Headers(unittest.TestCase):
    def test_degraded_headers_only_when_degraded(self):
        self.assertEqual(shim._degraded_headers({}), {})
        self.assertEqual(shim._degraded_headers(None), {})
        self.assertEqual(shim._degraded_headers({"degraded": "remote-alias-local", "degraded_reason": "cap-no-room"}),
                         {"X-Gateway-Degraded": "remote-alias-local", "X-Gateway-Degraded-Reason": "cap-no-room"})


class EndToEndReceipt(Base):
    """Through the REAL _relay: the response the caller sees carries the header, and the body the engine got is local."""

    async def test_response_carries_the_degraded_header_and_the_engine_sees_the_local_model(self):
        self.exhaust()
        opened = []

        async def fake_open(session, base, path, body, key, streaming):
            opened.append((base, json.loads(body)))
            return Upstream()
        self.stack.enter_context(patch.object(shim, "_relay", REAL_RELAY))
        self.stack.enter_context(patch.object(shim, "_open", fake_open))
        self.stack.enter_context(patch.object(shim, "_active_set", REAL_ACTIVE_SET))
        self.stack.enter_context(patch.object(shim, "_note_payload_outcome", lambda *a, **k: None))
        req = lf.Request(model="estate-remote")
        shim._ACTIVE[id(req)] = {}
        try:
            r = await shim._route_completions(req)
        finally:
            shim._ACTIVE.pop(id(req), None)
        self.assertEqual(r.status, 200)
        self.assertEqual(r.headers["X-Gateway-Degraded"], "remote-alias-local")
        self.assertEqual(r.headers["X-Gateway-Degraded-Reason"], "balance-exhausted")
        self.assertEqual(r.headers["X-Gateway-Route"], "local")
        self.assertEqual(opened[0][0], shim.LOCAL)
        self.assertEqual(opened[0][1]["model"], shim._local_model_name())


class Telemetry(Base):
    def test_record_event_keeps_route_local_and_reason_remote_alias_degraded(self):
        req = lf.Request(model="estate-remote")
        shim._ACTIVE[id(req)] = {"degraded": "remote-alias-local", "degraded_reason": "balance-exhausted"}
        before = len(shim._events)
        try:
            with patch.object(shim, "record_event", REAL_RECORD_EVENT), \
                    patch.object(shim, "_active_set", REAL_ACTIVE_SET), patch.object(shim, "flow_note_route", lambda *a, **k: None):
                shim.record_event("local", "tokens", req, 1, 0.5, ptok=100, maxtok=10)
            ev = shim._events[0]
            self.assertEqual(len(shim._events), before + 1)
            self.assertEqual((ev["d"], ev["r"]), ("local", "remote-alias-degraded"))
            self.assertEqual((ev["degraded"], ev["degraded_why"], ev["ar"]), ("remote-alias-local", "balance-exhausted", "tokens"))
            self.assertEqual(shim._ACTIVE[id(req)]["reason"], "remote-alias-degraded")
            self.assertEqual(shim._ACTIVE[id(req)]["route"], "local")
        finally:
            shim._ACTIVE.pop(id(req), None)

    def test_a_local_failure_keeps_its_own_reason(self):
        req = lf.Request(model="estate-remote")
        shim._ACTIVE[id(req)] = {"degraded": "remote-alias-local", "degraded_reason": "cap-no-room"}
        try:
            with patch.object(shim, "record_event", REAL_RECORD_EVENT), \
                    patch.object(shim, "_active_set", REAL_ACTIVE_SET), patch.object(shim, "flow_note_route", lambda *a, **k: None):
                shim.record_event("held", "local-failed", req, 1, 0)
            self.assertEqual(shim._events[0]["r"], "local-failed")
        finally:
            shim._ACTIVE.pop(id(req), None)

    def test_ordinary_requests_are_unaffected(self):
        req = lf.Request()
        shim._ACTIVE[id(req)] = {}
        try:
            with patch.object(shim, "record_event", REAL_RECORD_EVENT), \
                    patch.object(shim, "_active_set", REAL_ACTIVE_SET), patch.object(shim, "flow_note_route", lambda *a, **k: None):
                shim.record_event("local", "tokens", req, 1, 0)
            self.assertEqual((shim._events[0]["r"], shim._events[0]["ar"]), ("tokens", None))
        finally:
            shim._ACTIVE.pop(id(req), None)


class PredicateEqualsCapacityMode(unittest.TestCase):
    """The spec: reuse the capacity mode's remote_usable, do not re-derive it.  Pinned equal across the states."""

    def check(self, **state):
        body = b'{"model":"estate-remote","messages":[{"role":"user","content":"x"}],"max_tokens":2000}'
        with patch.object(shim, "REMOTE_ENABLED", state.get("enabled", True)), \
                patch.object(shim, "LOCAL_ONLY", state.get("local_only", 0)), \
                patch.object(shim, "_remote_dead_until", time.time() + 300 if state.get("dead") else 0.0), \
                patch.object(shim, "_REMOTE_BALANCE", {**shim._REMOTE_BALANCE, "exhausted": state.get("bal", False)}), \
                patch.object(shim, "_spend_allows_overflow", lambda p, m, model=None: state.get("room", True)):
            usable = shim.remote_unusable_reason(body, 20000, 2000) is None
            return usable, shim.flow_mode_now()["remote_usable"]

    def test_equal_in_every_state(self):
        for enabled in (True, False):
            for dead in (False, True):
                for bal in (False, True):
                    for room in (True, False):
                        for lo in (0, 1):
                            a, b = self.check(enabled=enabled, dead=dead, bal=bal, room=room, local_only=lo)
                            self.assertEqual(a, b, (enabled, dead, bal, room, lo))


if __name__ == "__main__":
    unittest.main()
