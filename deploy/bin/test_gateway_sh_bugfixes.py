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


class _Fresh(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory(prefix="gw-sh-fix-")
        self.addCleanup(self._td.cleanup)
        self.tmp = self._td.name
        self.m = G._fresh_shim(self.tmp, 0)


class SwallowedExceptionsAreCounted(_Fresh):
    """Class: a broad `except Exception: pass` hid a failure that fires on every request."""

    def test_unusable_spend_ledger_is_counted_not_silent(self):
        m = self.m

        def boom():
            raise OSError("ledger unreadable")
        m._spend = boom
        self.assertFalse(m._spend_allows_overflow(1000, 100))        # still fails closed, as before
        self.assertFalse(m._spend_allows_overflow(1000, 100))
        self.assertEqual(m._SWALLOWED["_spend_allows_overflow"][0], 2)
        self.assertIn("ledger unreadable", m._SWALLOWED["_spend_allows_overflow"][2])

    def test_unreadable_outcome_alarm_brake_is_counted(self):
        m = self.m
        m.OUTCOME_ALARM_PATH = self.tmp + "/no-such-alarm.json"
        self.assertFalse(m._automatic_remote_budget_allows(1000, 100))
        self.assertEqual(m._SWALLOWED["_automatic_remote_budget_allows"][0], 1)

    def test_endpoint_and_metrics_export_them(self):
        m = self.m
        m._swallowed("unit-site", ValueError("x"))
        m._ROUTER_CRASHES["KeyError"] += 1
        resp = G.asyncio.run(m.gateway_internal_errors(None))
        body = G.json.loads(resp.body)
        self.assertEqual(body["swallowed"]["unit-site"]["n"], 1)
        self.assertEqual(body["router_crashes"], {"KeyError": 1})
        text = G.asyncio.run(m.gateway_metrics(None)).text
        self.assertIn('gateway_swallowed_errors_total{site="unit-site"} 1', text)
        self.assertIn('gateway_router_crashes_total{type="KeyError"} 1', text)

    def test_warning_is_rate_limited_to_powers_of_two(self):
        m = self.m
        with self.assertLogs("gateway-shim", level="WARNING") as cm:
            for _ in range(9):
                m._swallowed("flood", RuntimeError("again"))
        self.assertEqual(len([r for r in cm.output if "flood" in r]), 4)    # 1, 2, 4, 8


class PerClientTablesAreBounded(_Fresh):
    """Class: dict keyed by a caller-chosen header with no eviction."""

    def test_per_client_rollup_caps_distinct_names(self):
        m = self.m
        for i in range(m.CLIENT_KEYS_MAX + 100):
            m._telemetry_note_request({"name": "churn-%d" % i, "route": "local", "t0": m.time.time()})
        self.assertEqual(len(m._PER_CLIENT), m.CLIENT_KEYS_MAX + 1)
        self.assertEqual(m._PER_CLIENT[m.CLIENT_OVERFLOW_KEY]["requests"], 100)
        m._telemetry_note_request({"name": "churn-0", "route": "local", "t0": m.time.time()})
        self.assertEqual(m._PER_CLIENT["churn-0"]["requests"], 2)          # known names keep their own row

    def test_client_key_helper(self):
        m = self.m
        table = {("c%d" % i): 1 for i in range(m.CLIENT_KEYS_MAX)}
        self.assertEqual(m._client_key(table, "c1"), "c1")
        self.assertEqual(m._client_key(table, "new"), m.CLIENT_OVERFLOW_KEY)


class TelemetryLogCountsAreTruthful(_Fresh):
    """Class: a count presented as fact that was not (rows 'written' when the write failed; silent row loss)."""

    def test_failed_write_is_not_counted_as_written(self):
        m = self.m
        blocker = self.tmp + "/not-a-dir"
        open(blocker, "w").close()
        m.TELEMETRY_DIR = blocker + "/telemetry"                         # makedirs/open fail
        r = m._flush_jsonl_blocking([{"a": 1}, {"b": 2}], None, None, 0, False)
        self.assertEqual((r["written"], r["dropped_io"]), (0, 2))
        self.assertEqual(r["bytes"], 0)

    def test_unserialisable_row_is_counted(self):
        m = self.m
        m.TELEMETRY_DIR = self.tmp + "/tel"
        r = m._flush_jsonl_blocking([{"a": 1}, {"bad": object()}], None, None, 0, False)
        self.assertEqual((r["written"], r["dropped_bad"], r["dropped_io"]), (1, 1, 0))

    def test_flusher_error_counts_the_lost_batch(self):
        m = self.m

        async def run():
            m._JSONL_PENDING[:] = [{"a": 1}, {"a": 2}, {"a": 3}]

            def boom(*a, **k):
                raise RuntimeError("executor gone")
            m._flush_jsonl_blocking = boom
            m.TELEMETRY_FLUSH_SECS = 0
            task = G.asyncio.ensure_future(m._jsonl_flusher())
            for _ in range(20):
                await G.asyncio.sleep(0)
                if m._JSONL_STATE["dropped_err"]:
                    break
            task.cancel()
            try:
                await task
            except BaseException:
                pass
        G.asyncio.run(run())
        self.assertEqual(self.m._JSONL_STATE["dropped_err"], 3)
