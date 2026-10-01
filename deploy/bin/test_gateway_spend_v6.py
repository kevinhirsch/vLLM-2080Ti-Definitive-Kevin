#!/usr/bin/env python3
"""R2 v6 regressions (Terra 13:15): no custom-endpoint exemption, a governed v4-pro alias,
compare-and-set REPLACE, and failure-aware settlement.

Run:  python3 -m unittest test_gateway_spend_v6
"""
import atexit
import importlib.util
import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ.setdefault("SHIM_EXACT_TOKENS", "0")
_TMPDIR = tempfile.TemporaryDirectory(prefix="spend-v6-test-")   # managed: removed at exit
atexit.register(_TMPDIR.cleanup)
SPEC = importlib.util.spec_from_file_location("shim_spend_v6_test", os.environ.get(
    "SHIM_TEST_CANDIDATE", str(Path(__file__).with_name("keepalive-shim.py"))))
shim = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shim)

DAY0 = 1790000000.0


def ledger(cap=1.0, enforce=True):
    path = Path(tempfile.mkdtemp(dir=_TMPDIR.name)) / "gateway-spend.json"
    return shim.SpendLedger(str(path), cap=lambda: cap, clock=lambda: DAY0, process_token="p",
                            importer=lambda s, e: (0.0, 0, 0), enforce=lambda: enforce)


class Request:
    path = "/v1/chat/completions"
    method = "POST"

    def __init__(self, model, remote="127.0.0.1", stream=False, **fields):
        self.remote = remote
        self.headers = {"User-Agent": "offline-test"}
        self.body = json.dumps({"model": model, "messages": [{"role": "user", "content": "x"}],
                                "max_tokens": 1000, "stream": stream, **fields}).encode()

    async def read(self):
        return self.body


FREE = {"name": "free-a", "base": "https://free.test/v1", "key": "", "model": "openrouter/free",
        "context_limit": 200000, "max_output": 4096, "enabled": True, "cost": {"policy": "free"}}
PAID = {"name": "paid-a", "base": "https://paid.test/v1", "key": "k", "model": "m-paid",
        "context_limit": 200000, "max_output": 4096, "enabled": True,
        "cost": {"policy": "metered", "provider": "paidco", "cache_hit": 1.0, "cache_miss": 2.0, "output": 3.0}}
BARE = {"name": "bare-a", "base": "https://bare.test/v1", "key": "k", "model": "m-bare",
        "context_limit": 200000, "max_output": 4096, "enabled": True, "cost": None}
PRO_PRICES = json.dumps({"deepseek-v4-pro": {"cache_hit": 0.01, "cache_miss": 0.5, "output": 2.0}})


class PolicyValidation(unittest.TestCase):
    def test_cost_policies(self):
        self.assertIsNone(shim._validate_cost_policy(None))
        self.assertEqual(shim._validate_cost_policy({"policy": "free"}), {"policy": "free"})
        self.assertEqual(shim._validate_cost_policy(PAID["cost"]), PAID["cost"])
        for bad in ({"policy": "cheap"}, {"policy": "metered", "cache_hit": 1, "cache_miss": 1, "output": 1},
                    {"policy": "metered", "provider": "x", "cache_hit": 1}, "free",
                    {"policy": "metered", "provider": "x", "cache_hit": -1, "cache_miss": 1, "output": 1}):
            with self.assertRaises(ValueError):
                shim._validate_cost_policy(bad)
        rec = shim._validate_alias_record("paid-a", {**PAID})
        self.assertEqual(rec["cost"]["provider"], "paidco")
        self.assertIsNone(shim._validate_alias_record("bare-a", {k: v for k, v in BARE.items() if k != "cost"})["cost"])


class Routing(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.led = ledger(cap=1.0)
        self.sent = []

        async def relay(request, base, path, body, key, streaming, *a, **k):
            self.sent.append((base, json.loads(body)))
            if self.usage:
                shim._active_set(request, **self.usage)
            return "ok", shim.web.json_response({"ok": True})
        self.usage = None
        for name, value in dict(_SPEND_LEDGER=self.led, _relay=relay, LOG_REQUESTS=0, FORCE_REMOTE=0,
                                _ALIASES={"free-a": FREE, "paid-a": PAID, "bare-a": BARE},
                                REMOTE_BASE="https://api.deepseek.test", REMOTE_MODEL="deepseek-flash",
                                REMOTE_PRICES_JSON="", REMOTE_COST_IN_PER_MTOK=0.15, REMOTE_COST_OUT_PER_MTOK=0.6,
                                REMOTE_COST_IN_PER_MTOK_PEAK=0.3, REMOTE_COST_OUT_PER_MTOK_PEAK=1.2,
                                _est_tokens=lambda body: 100_000, estimate_units=lambda *a, **k: 1,
                                remote_ok=lambda: True, is_peak=lambda: False,
                                record_event=lambda d, r, request, *a, **k: shim._active_set(request, route=d, reason=r),
                                _note_remote_response=AsyncMock(), _note_payload_outcome=lambda *a, **k: None,
                                _telemetry_note_request=lambda *a, **k: None).items():
            self.stack.enter_context(patch.object(shim, name, value))

    async def test_a_custom_endpoint_with_no_cost_policy_fails_closed(self):
        resp = await shim.handle_completions(Request("bare-a"))
        self.assertEqual(resp.status, 409)
        self.assertEqual(json.loads(resp.body)["error"]["type"], "cost_policy_missing")
        self.assertEqual(self.sent, [])

    async def test_an_explicitly_free_endpoint_is_never_charged(self):
        self.usage = {"remote_cache_hit": 1_000_000, "remote_cache_miss": 1_000_000, "outtok": 1_000_000}
        self.led.settle("pre", 1.0)                             # even with the cap spent
        resp = await shim.handle_completions(Request("free-a"))
        self.assertEqual((resp.status, len(self.sent)), (200, 1))
        snap = self.led.snapshot()
        self.assertEqual((round(snap["spent"], 6), snap["in_flight"]), (1.0, 0))

    async def test_a_metered_custom_endpoint_is_held_capped_and_settled_at_its_own_prices(self):
        self.led.settle("pre", 0.99)                            # $0.01 left; its hold would be $0.203
        resp = await shim.handle_completions(Request("paid-a"))
        self.assertEqual((resp.status, self.sent), (429, []))
        self.led.recover(spent=0.0, replace=True, source="t", expected_revision=self.led.snapshot()["revision"])
        self.usage = {"remote_cache_hit": 1_000, "remote_cache_miss": 2_000, "outtok": 3_000}
        resp = await shim.handle_completions(Request("paid-a"))
        self.assertEqual((resp.status, self.sent[0][0]), (200, "https://paid.test/v1"))
        self.assertAlmostEqual(self.led.snapshot()["spent"], (1_000 * 1 + 2_000 * 2 + 3_000 * 3) / 1e6, places=9)

    async def test_the_pro_alias_is_refused_until_it_has_a_price(self):
        resp = await shim.handle_completions(Request("estate-remote-pro"))
        self.assertEqual(resp.status, 409)
        self.assertIn("deepseek-v4-pro", json.loads(resp.body)["error"]["message"])
        self.assertEqual(self.sent, [])

    async def test_the_pro_alias_uses_the_pro_model_under_the_same_authority(self):
        self.usage = {"remote_cache_hit": 10_000, "remote_cache_miss": 1_000, "outtok": 100}
        with patch.object(shim, "REMOTE_PRICES_JSON", PRO_PRICES):
            resp = await shim.handle_completions(Request("estate-remote-pro"))
        self.assertEqual(resp.status, 200)
        self.assertEqual(self.sent[0][1]["model"], "deepseek-v4-pro")
        self.assertAlmostEqual(self.led.snapshot()["spent"], (10_000 * 0.01 + 1_000 * 0.5 + 100 * 2.0) / 1e6, places=9)
        self.led.settle("pre", 1.0)
        with patch.object(shim, "REMOTE_PRICES_JSON", PRO_PRICES):
            self.assertEqual((await shim.handle_completions(Request("estate-remote-pro"))).status, 429)

    async def test_a_disconnect_after_sending_charges_the_hold(self):
        async def reset(request, *a, **k):
            raise ConnectionResetError("client went away mid-stream")
        with patch.object(shim, "_relay", reset):
            with self.assertRaises(ConnectionResetError):
                await shim.handle_completions(Request("estate-remote", stream=True))
        snap = self.led.snapshot()
        self.assertEqual(snap["in_flight"], 0)
        self.assertAlmostEqual(snap["spent"], shim._spend_hold_estimate(100_000, 1000), places=6)

    async def test_streamed_remote_requests_ask_for_the_usage_trailer(self):
        self.usage = {"remote_cache_hit": 1, "remote_cache_miss": 1, "outtok": 1}
        await shim.handle_completions(Request("estate-remote", stream=True))
        self.assertEqual(self.sent[0][1]["stream_options"], {"include_usage": True})

    def test_pro_and_reservation_alias_resolution(self):
        rid = "b" * 32
        self.assertEqual(shim._resolve_alias("estate-remote-pro")["model"], shim.REMOTE_PRO_MODEL)
        self.assertEqual(shim._resolve_alias(f"estate-remote-pro.{rid}"),
                         {"name": "estate-remote-pro", "kind": "builtin-remote", "reservation": rid,
                          "model": shim.REMOTE_PRO_MODEL})
        self.assertNotIn("model", shim._resolve_alias(f"estate-remote.{rid}"))


class Settlement(unittest.TestCase):
    """_spend_settle on the request record handle_completions' finally passes it."""

    def setUp(self):
        self.led = ledger(cap=100.0)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(shim, "is_peak", lambda: False))
        for n, v in dict(_SPEND_LEDGER=self.led, REMOTE_PRICES_JSON="", REMOTE_MODEL="deepseek-flash",
                         REMOTE_PRICE_CACHE_HIT_PER_MTOK=0.003, REMOTE_PRICE_CACHE_MISS_PER_MTOK=0.15,
                         REMOTE_PRICE_OUTPUT_PER_MTOK=0.6).items():
            self.stack.enter_context(patch.object(shim, n, v))

    def settle(self, **info):
        self.led.hold("k", 0.5)
        base = {"spend_key": "k", "spend_held": True, "route": "remote", "cost_policy": "metered"}
        shim._spend_settle({**base, **info}, None)
        return round(self.led.snapshot()["spent"], 9)

    def test_an_error_response_with_provider_usage_is_charged_that_usage(self):
        got = self.settle(remote_sent=True, http_status=502, remote_cache_hit=10_000, remote_cache_miss=100, outtok=50)
        self.assertEqual(got, round((10_000 * 0.003 + 100 * 0.15 + 50 * 0.6) / 1e6, 9))

    def test_a_truncated_or_lost_stream_without_usage_is_charged_the_hold(self):
        self.assertEqual(self.settle(remote_sent=True, http_status=200, outtok_lb=37), 0.5)

    def test_an_error_without_usage_is_charged_the_hold(self):
        self.assertEqual(self.settle(remote_sent=True, http_status=500), 0.5)

    def test_partial_usage_prices_the_reported_prompt_as_cache_misses(self):
        self.assertEqual(self.settle(remote_sent=True, ptok_exact=1_000, outtok=10),
                         round((1_000 * 0.15 + 10 * 0.6) / 1e6, 9))

    def test_a_held_request_that_never_reached_the_provider_costs_nothing(self):
        self.assertEqual(self.settle(remote_sent=False, http_status=413), 0.0)
        self.assertEqual(self.led.snapshot()["in_flight"], 0)

    def test_a_free_endpoint_releases_its_hold_at_zero(self):
        self.assertEqual(self.settle(remote_sent=True, cost_policy="free", outtok=10**6), 0.0)


class CompareAndSetReplace(unittest.TestCase):
    def test_replace_is_bound_to_the_exact_revision_with_nothing_in_flight(self):
        led = ledger(cap=25.0)
        led.settle("a", 4.0)
        rev = led.snapshot()["revision"]
        self.assertIn("conflict", led.recover(spent=3.3, replace=True, source="bill")[1])          # no revision
        self.assertIn("conflict", led.recover(spent=3.3, replace=True, source="bill", expected_revision=rev - 1)[1])
        led.hold("inflight", 0.2)
        rev2 = led.snapshot()["revision"]
        self.assertGreater(rev2, rev)                                                                # every write
        self.assertIn("in flight", led.recover(spent=3.3, replace=True, source="bill", expected_revision=rev2)[1])
        led.settle("inflight", 0.1)
        self.assertAlmostEqual(led.snapshot()["spent"], 4.1)                                         # untouched
        ok, _ = led.recover(spent=3.3, replace=True, source="bill", expected_revision=led.snapshot()["revision"])
        self.assertTrue(ok)
        self.assertAlmostEqual(led.snapshot()["spent"], 3.3)


class ImportRule(unittest.TestCase):
    def test_v6_rows_count_when_sent_to_a_metered_provider_at_any_status(self):
        start = shim._spend_day_start(DAY0)
        rows = [{"t": start + 1, "route": "remote", "remote_sent": True, "cost_policy": "metered", "status": 502, "cost_est": 0.5, "cost_basis": "actual"},
                {"t": start + 2, "route": "remote", "remote_sent": False, "cost_policy": "metered", "status": 429, "cost_est": 9.0},
                {"t": start + 3, "route": "remote", "remote_sent": True, "cost_policy": "free", "status": 200, "cost_est": 9.0}]
        with tempfile.TemporaryDirectory(dir=_TMPDIR.name) as tdir:
            name = "requests-%s.jsonl" % shim.time.strftime("%Y%m%d", shim.time.gmtime(start + 1))
            Path(tdir, name).write_text("".join(json.dumps(r) + "\n" for r in rows))
            self.assertEqual(shim.telemetry_remote_cost(start, DAY0, tdir), (0.5, 1, 0))


if __name__ == "__main__":
    unittest.main()
