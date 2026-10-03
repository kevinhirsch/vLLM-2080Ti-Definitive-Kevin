#!/usr/bin/env python3
"""Lane SH bug-class fixes in keepalive-shim.py, one named regression test per fix.

Driven through the golden harness (fresh shim per scenario, isolated paths, fake clock, stubbed upstreams), so each
test exercises the real handle_completions -> _route_completions path.

Run:  python -m pytest -q test_gateway_sh_bugfixes.py
"""
import importlib.util
import unittest
from pathlib import Path

_H = Path(__file__).resolve().with_name("test_gateway_golden_routing.py")
_SPEC = importlib.util.spec_from_file_location("golden_harness_sh", _H)
G = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(G)


def _boom_relay(m):
    async def relay(*a, **k):
        raise KeyError("synthetic router bug")
    return relay
_boom_relay._needs_module = True


def _boom_after_prepare(m):
    async def relay(request, *a, **k):
        request["gw_stream_prepared"] = True
        raise RuntimeError("bug after the stream was committed")
    return relay
_boom_after_prepare._needs_module = True


class RequestAlwaysAnswered(unittest.TestCase):
    """Class: a request dropped without the gateway's own response (aiohttp bare 500, invisible to telemetry)."""

    def test_non_object_bodies_get_400_and_a_telemetry_row(self):
        for raw in (b"[1,2,3]", b'"hello"', b"42", b"null"):
            with self.subTest(raw=raw):
                out = G.run_scenario(dict(req=dict(raw=raw)), 0)
                self.assertEqual(out["response"]["status"], 400, out["response"])
                self.assertEqual(out["response"]["error"]["code"], "invalid_body")
                self.assertEqual(len(out["telemetry"]), 1)
                self.assertEqual((out["telemetry"][0]["route"], out["telemetry"][0]["reason"]), ("rejected", "bad-body"))
                self.assertEqual(out["calls"], [])

    def test_bad_messages_shapes_get_400(self):
        for msgs in ("hello", ["hi"], [{"role": "user", "content": "a"}, 7], {"role": "user"}):
            with self.subTest(messages=msgs):
                out = G.run_scenario(dict(req=dict(fields=dict(messages=msgs))), 0)
                self.assertEqual(out["response"]["status"], 400, out["response"])
                self.assertEqual(out["calls"], [])

    def test_valid_body_shapes_are_not_refused(self):
        for fields in (dict(messages=[{"role": "user", "content": {"text": "hi"}}]), dict(tools="x"),
                       dict(prompt="hi", messages=None)):
            with self.subTest(fields=fields):
                out = G.run_scenario(dict(req=dict(fields=fields)), 0)
                self.assertEqual(out["response"]["status"], 200, out["response"])

    def test_router_exception_becomes_counted_json_500_in_telemetry(self):
        out = G.run_scenario(dict(patch=dict(_relay=_boom_relay)), 0)
        self.assertEqual(out["response"]["status"], 500)
        self.assertEqual(out["response"]["error"]["type"], "gateway_internal_error")
        row = out["telemetry"][0]
        self.assertEqual((row["route"], row["reason"], row["status"]), ("error", "gateway-exception:KeyError", 500))
        self.assertEqual(out["counters"]["inflight"], 0)            # the lane was released on the way out

    def test_committed_stream_is_not_answered_twice(self):
        out = G.run_scenario(dict(patch=dict(_relay=_boom_after_prepare)), 0)
        self.assertEqual(out["response"].get("kind"), "raised")    # re-raised: aiohttp closes the stream
        self.assertEqual(out["telemetry"][0]["route"], "error")     # ...but it is still on the record


def _reset_relay(m):
    async def relay(*a, **k):
        raise ConnectionResetError("client went away")
    return relay
_reset_relay._needs_module = True


class ClientDisconnectIsNotACrash(unittest.TestCase):
    def test_connection_reset_propagates_uncounted(self):
        out = G.run_scenario(dict(patch=dict(_relay=_reset_relay)), 0)
        self.assertEqual(out["response"], {"kind": "raised", "exc": "ConnectionResetError", "msg": "client went away"})
        self.assertEqual(out["telemetry"][0]["route"], "local")      # the admission decision stands; no crash counted


class OutcomeAlwaysNamed(unittest.TestCase):
    """Class: early-return refusals logged route '?' with no reason."""

    def test_early_refusals_name_route_and_reason(self):
        cases = {"alias_disabled": "alias_disabled", "local_down_estate_local_503": "local_unavailable",
                 "local_down_cap_exhausted_503": "local_unavailable_cap_exhausted",
                 "token_reservation_over_budget_no_remote": "http-503",
                 "context_too_big_estate_local_error": "context_length_exceeded"}
        for name, reason in cases.items():
            with self.subTest(scenario=name):
                row = G.run_scenario(G.SCENARIOS[name], 0)["telemetry"][0]
                self.assertEqual((row["route"], row["reason"]), ("rejected", reason))

    def test_no_golden_scenario_logs_route_question_mark(self):
        for name, out in G.json.loads(G.GOLDEN.read_text()).items():
            for row in out["telemetry"]:
                with self.subTest(scenario=name):
                    self.assertNotEqual(row["route"], "?")


if __name__ == "__main__":
    unittest.main()
