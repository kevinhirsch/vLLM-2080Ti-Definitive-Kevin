#!/usr/bin/env python3
"""R2 regressions for the gateway's remote spend authority (keepalive-shim SpendLedger).

The gateway is the ONE daily remote-spend cap shared by Halo, the card runner and every
`estate-remote` client. Covered here, offline (no server, no network, tmp ledger file):
concurrent reservation, crash/expiry, retry idempotency, cross-client cap, and the routing
seam (an `estate-remote` request is refused before anything is forwarded when there is no
budget, and a reservation-backed one draws from its reservation).

Run:  python3 -m unittest test_gateway_spend_authority
"""
import asyncio
import importlib.util
import json
import os
import tempfile
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ["SHIM_EXACT_TOKENS"] = "0"
_TMP = tempfile.mkdtemp(prefix="spend-test-")
os.environ["SHIM_SPEND_FILE"] = os.path.join(_TMP, "never-used.json")
SPEC = importlib.util.spec_from_file_location("shim_spend_test", os.environ.get(
    "SHIM_TEST_CANDIDATE", str(Path(__file__).with_name("keepalive-shim.py"))))
shim = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shim)

DAY0 = 1790000000.0        # a fixed instant; the ledger never reads the wall clock here


class Clock:
    def __init__(self, t=DAY0):
        self.t = t

    def __call__(self):
        return self.t


def ledger(path, cap=25.0, clock=None, process="p1"):
    return shim.SpendLedger(str(path), cap=lambda: cap, clock=clock or Clock(), process_token=process)


class LedgerBase(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="spend-", dir=_TMP))
        self.path = self.dir / "gateway-spend.json"
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(shim, "SPEND_ENFORCE", True))


class ConcurrentReservation(LedgerBase):
    def test_parallel_reservations_never_exceed_the_cap(self):
        """Many runner slots reserving at the same instant: exactly floor(cap/amount) win."""
        led = ledger(self.path, cap=25.0)
        results = []
        barrier = threading.Barrier(24)

        def go(i):
            barrier.wait()
            results.append(led.reserve("card-runner", f"attempt-{i}", 1.5, 3600)[0])

        threads = [threading.Thread(target=go, args=(i,)) for i in range(24)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sum(results), 16)                  # 16 x 1.50 = 24.00 <= 25 < 25.50
        snap = led.snapshot()
        self.assertAlmostEqual(snap["reserved"], 24.0)
        self.assertAlmostEqual(snap["available"], 1.0)
        # and it is on disk, not just in memory
        self.assertEqual(len(json.loads(self.path.read_text())["reservations"]), 16)


class CrashAndExpiry(LedgerBase):
    def test_orphan_request_holds_are_charged_once_after_a_restart(self):
        clock = Clock()
        led = ledger(self.path, clock=clock, process="before-crash")
        ok, r, _ = led.reserve("card-runner", "a1", 2.0, 3600)
        self.assertTrue(ok)
        self.assertTrue(led.hold("req-1", 0.5, rid=r["id"])[0])
        self.assertTrue(led.hold("req-2", 0.25)[0])
        # process dies here; a new gateway process loads the same file
        led2 = ledger(self.path, clock=clock, process="after-restart")
        snap = led2.snapshot()
        self.assertEqual(snap["in_flight"], 0)
        self.assertAlmostEqual(snap["spent"], 0.75)            # both holds charged in full
        self.assertAlmostEqual(snap["orphans_charged"], 0.75)
        self.assertAlmostEqual(snap["reserved"], 1.5)          # reservation survives: 2.0 - 0.5 used
        # a third process does not charge them again
        self.assertAlmostEqual(ledger(self.path, clock=clock, process="third").snapshot()["spent"], 0.75)

    def test_an_abandoned_reservation_expires_and_releases_its_remainder(self):
        clock = Clock()
        led = ledger(self.path, cap=5.0, clock=clock)
        ok, r, _ = led.reserve("card-runner", "a1", 4.0, 600)
        self.assertTrue(ok)
        self.assertFalse(led.reserve("halo", "run-1", 2.0, 600)[0])
        clock.t += 601
        snap = led.snapshot()
        self.assertEqual(snap["reservations"], [])
        self.assertAlmostEqual(snap["available"], 5.0)
        self.assertTrue(led.reserve("halo", "run-1", 2.0, 600)[0])
        # the expired reservation now refuses requests that still carry its id
        ok, why = led.hold("late", 0.1, rid=r["id"])
        self.assertFalse(ok)
        self.assertIn("expired", why)

    def test_a_stuck_request_hold_is_charged_at_its_ttl(self):
        clock = Clock()
        led = ledger(self.path, clock=clock)
        self.assertTrue(led.hold("stuck", 0.4)[0])
        clock.t += shim.SPEND_REQUEST_HOLD_TTL + 1
        snap = led.snapshot()
        self.assertEqual(snap["in_flight"], 0)
        self.assertAlmostEqual(snap["spent"], 0.4)

    def test_a_corrupt_ledger_refuses_rather_than_resetting_to_zero(self):
        self.path.write_text("{not json")
        led = ledger(self.path)
        self.assertEqual(led.snapshot()["available"], 0.0)
        self.assertFalse(led.reserve("card-runner", "a1", 1.0, 600)[0])
        self.assertFalse(led.hold("r", 0.01)[0])


class RetryIdempotency(LedgerBase):
    def test_reserve_finalize_and_settle_are_idempotent(self):
        led = ledger(self.path, cap=10.0)
        ok1, r1, o1 = led.reserve("card-runner", "attempt-a", 3.0, 3600)
        ok2, r2, o2 = led.reserve("card-runner", "attempt-a", 3.0, 3600)   # client retried
        self.assertTrue(ok1 and ok2)
        self.assertEqual((r1["id"], o1, o2), (r2["id"], "reserved", "replayed"))
        self.assertAlmostEqual(led.snapshot()["reserved"], 3.0)            # held once
        self.assertTrue(led.hold("q1", 0.2, rid=r1["id"])[0])
        self.assertAlmostEqual(led.settle("q1", 0.12), 0.12)
        self.assertEqual(led.settle("q1", 0.12), 0.0)                      # repeated settle
        f1 = led.finalize(r1["id"])
        f2 = led.finalize(owner="card-runner", key="attempt-a")            # retried finalize
        self.assertEqual((f1["status"], f2["status"]), ("finalized", "finalized"))
        self.assertAlmostEqual(f2["used"], 0.12)
        snap = led.snapshot()
        self.assertAlmostEqual(snap["spent"], 0.12)
        self.assertAlmostEqual(snap["reserved"], 0.0)
        # a reservation that was already consumed is not re-issued on replay
        ok3, r3, o3 = led.reserve("card-runner", "attempt-a", 3.0, 3600)
        self.assertEqual((ok3, r3["status"], o3), (True, "finalized", "replayed"))
        self.assertFalse(led.hold("q2", 0.01, rid=r1["id"])[0])


class CrossClientCap(LedgerBase):
    def test_one_cap_across_runner_reservation_halo_requests_and_overflow(self):
        led = ledger(self.path, cap=25.0)
        ok, r, _ = led.reserve("card-runner", "attempt-a", 20.0, 3600)
        self.assertTrue(ok)
        # Halo (plain estate-remote, no reservation) sees only the $5 left
        self.assertFalse(led.hold("halo-big", 6.0)[0])
        self.assertTrue(led.hold("halo-ok", 4.0)[0])
        led.settle("halo-ok", 3.5)
        # a gateway overflow request is charged without a hold (not refused)
        self.assertAlmostEqual(led.settle("overflow-1", 1.4), 1.4)
        self.assertAlmostEqual(led.snapshot()["available"], 0.1)
        # the runner's reservation still covers its own requests although the pool is empty
        self.assertTrue(led.hold("runner-1", 5.0, rid=r["id"])[0])
        led.settle("runner-1", 4.0)
        self.assertFalse(led.hold("halo-2", 0.2)[0])
        # a runner request larger than what is left of its reservation draws on the pool
        self.assertFalse(led.hold("runner-2", 16.2, rid=r["id"])[0])      # 16 left + 0.1 pool
        self.assertTrue(led.hold("runner-3", 16.1, rid=r["id"])[0])

    def test_observe_mode_accounts_but_never_refuses(self):
        led = ledger(self.path, cap=1.0)
        with patch.object(shim, "SPEND_ENFORCE", False):
            self.assertTrue(led.reserve("card-runner", "a", 5.0, 600)[0])
            self.assertTrue(led.hold("x", 3.0)[0])
            led.settle("x", 2.0)
        self.assertAlmostEqual(led.snapshot()["spent"], 2.0)


class Request:
    path = "/v1/chat/completions"
    remote = "127.0.0.1"
    method = "POST"
    headers = {"X-Client": "test", "User-Agent": "offline-test"}

    def __init__(self, model="estate-remote", **fields):
        self.body = json.dumps({"model": model, "messages": [{"role": "user", "content": "x"}],
                                "max_tokens": 1000, **fields}).encode()

    async def read(self):
        return self.body


class RoutingSeam(unittest.IsolatedAsyncioTestCase):
    """`estate-remote` is refused BEFORE forwarding when the one cap has no room, and a
    finished request is priced and released by handle_completions' finally."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="spend-route-", dir=_TMP))
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.led = ledger(self.dir / "gateway-spend.json", cap=1.0)
        self.forward = AsyncMock(return_value=shim.web.json_response({"ok": True}))
        for name, value in dict(_SPEND_LEDGER=self.led, SPEND_ENFORCE=True, LOG_REQUESTS=0,
                                REMOTE_COST_IN_PER_MTOK=0.15, REMOTE_COST_OUT_PER_MTOK=0.6,
                                REMOTE_COST_IN_PER_MTOK_PEAK=0.3, REMOTE_COST_OUT_PER_MTOK_PEAK=1.2,
                                _est_tokens=lambda body: 100_000, estimate_units=lambda *a, **k: 1,
                                remote_ok=lambda: True, is_peak=lambda: False,
                                record_event=self._record, _forward_remote=self.forward,
                                _telemetry_note_request=lambda *a, **k: None).items():
            self.stack.enter_context(patch.object(shim, name, value))

    @staticmethod
    def _record(decision, reason, request, *a, **k):
        shim._active_set(request, route=decision, reason=reason)

    async def test_cap_exhausted_estate_remote_is_refused_before_forwarding(self):
        self.led.settle("earlier", 0.99)                     # $0.01 left
        resp = await shim.handle_completions(Request())
        self.assertEqual(resp.status, 429)
        self.assertEqual(json.loads(resp.body)["error"]["type"], "spend_cap_exhausted")
        self.forward.assert_not_called()
        self.assertEqual(self.led.snapshot()["in_flight"], 0)

    async def test_a_request_is_held_then_settled_at_the_gateway_price(self):
        resp = await shim.handle_completions(Request())
        self.assertEqual(resp.status, 200)
        self.forward.assert_awaited_once()
        snap = self.led.snapshot()
        self.assertEqual(snap["in_flight"], 0)
        # ptok 100k at the (off-peak) input rate; no output count known -> prompt only
        self.assertAlmostEqual(snap["spent"], 0.015, places=6)

    async def test_reservation_model_name_draws_from_its_reservation(self):
        ok, r, _ = self.led.reserve("card-runner", "attempt-a", 0.9, 600)
        self.assertTrue(ok)
        # the pool has $0.10 left; the reservation covers the $0.0312 hold
        resp = await shim.handle_completions(Request(model=f"estate-remote.{r['id']}"))
        self.assertEqual(resp.status, 200)
        self.led.finalize(r["id"])
        resp = await shim.handle_completions(Request(model=f"estate-remote.{r['id']}"))
        self.assertEqual(resp.status, 429)
        self.assertEqual(json.loads(resp.body)["error"]["type"], "spend_reservation_invalid")
        self.assertEqual(self.forward.await_count, 1)

    async def test_reserve_and_finalize_endpoints_round_trip(self):
        class J:
            def __init__(self, payload):
                self.payload = payload

            async def json(self):
                return self.payload
        r1 = json.loads((await shim.gateway_spend_reserve(J({"owner": "card-runner", "key": "k",
                                                              "amount": 0.5, "ttl_s": 600}))).body)
        r2 = json.loads((await shim.gateway_spend_reserve(J({"owner": "card-runner", "key": "k",
                                                              "amount": 0.5, "ttl_s": 600}))).body)
        self.assertEqual(r1["reservation"]["id"], r2["reservation"]["id"])
        self.assertEqual(r1["model"], "estate-remote." + r1["reservation"]["id"])
        refused = await shim.gateway_spend_reserve(J({"owner": "halo", "key": "x", "amount": 0.6, "ttl_s": 600}))
        self.assertEqual(refused.status, 429)
        fin = json.loads((await shim.gateway_spend_finalize(J({"reservation_id": r1["reservation"]["id"]}))).body)
        self.assertEqual(fin["reservation"]["status"], "finalized")
        snap = json.loads((await shim.gateway_spend(None)).body)
        self.assertAlmostEqual(snap["available"], 1.0)

    def test_alias_resolution_carries_only_well_formed_reservation_ids(self):
        rid = "a" * 32
        self.assertEqual(shim._resolve_alias(f"estate-remote.{rid}"),
                         {"name": "estate-remote", "kind": "builtin-remote", "reservation": rid})
        self.assertEqual(shim._resolve_alias("estate-remote")["kind"], "builtin-remote")
        self.assertNotEqual(shim._resolve_alias("estate-remote.not-an-id").get("kind"), "builtin-remote")


if __name__ == "__main__":
    unittest.main()
