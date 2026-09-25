#!/usr/bin/env python3
"""R2 regressions for the gateway's remote spend authority (keepalive-shim SpendLedger).

The gateway is the ONE daily remote-spend cap shared by Halo, the card runner and every
paid remote route. Covered offline (no server, no network, tmp ledger files):
concurrent reservation, crash/expiry, retry idempotency, cross-client cap, durability
failures, corrupt-ledger refusal in enforce AND observe mode with a governed recovery,
mid-day import of already-recorded spend, the enforce transition, credentialed
reserve/finalize and bound reservation use, and the routing seam (every paid route --
alias, route intent, overflow -- is held/refused before anything is forwarded).

Run:  python3 -m unittest test_gateway_spend_authority
"""
import asyncio
import hashlib
import importlib.util
import json
import os
import stat
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

DAY0 = 1790000000.0        # a fixed instant (2026-09-21 06:13 Phoenix); never the wall clock


class Clock:
    def __init__(self, t=DAY0):
        self.t = t

    def __call__(self):
        return self.t


def ledger(path, cap=25.0, clock=None, process="p1", importer=lambda s, e: (0.0, 0),
           enforce=lambda: True):
    return shim.SpendLedger(str(path), cap=lambda: cap, clock=clock or Clock(), process_token=process,
                            importer=importer, enforce=enforce)


class LedgerBase(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="spend-", dir=_TMP))
        self.path = self.dir / "gateway-spend.json"


class ConcurrentReservation(LedgerBase):
    def test_parallel_reservations_never_exceed_the_cap(self):
        led = ledger(self.path, cap=25.0)
        results, barrier = [], threading.Barrier(24)

        def go(i):
            barrier.wait()
            results.append(led.reserve("card-runner", f"attempt-{i}", 1.5, 3600)[0])

        threads = [threading.Thread(target=go, args=(i,)) for i in range(24)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(sum(results), 16)                   # 16 x 1.50 = 24.00 <= 25 < 25.50
        self.assertAlmostEqual(led.snapshot()["available"], 1.0)
        self.assertEqual(len(json.loads(self.path.read_text())["reservations"]), 16)


class CrashAndExpiry(LedgerBase):
    def test_orphan_request_holds_are_charged_once_after_a_restart(self):
        clock = Clock()
        led = ledger(self.path, clock=clock, process="before-crash")
        ok, r, _ = led.reserve("card-runner", "a1", 2.0, 3600)
        self.assertTrue(led.hold("req-1", 0.5, rid=r["id"])[0])
        self.assertTrue(led.hold("req-2", 0.25)[0])
        snap = ledger(self.path, clock=clock, process="after-restart").snapshot()
        self.assertEqual(snap["in_flight"], 0)
        self.assertAlmostEqual(snap["spent"], 0.75)
        self.assertAlmostEqual(snap["orphans_charged"], 0.75)
        self.assertAlmostEqual(snap["reserved"], 1.5)
        self.assertAlmostEqual(ledger(self.path, clock=clock, process="third").snapshot()["spent"], 0.75)

    def test_an_abandoned_reservation_expires_and_releases_its_remainder(self):
        clock = Clock()
        led = ledger(self.path, cap=5.0, clock=clock)
        ok, r, _ = led.reserve("card-runner", "a1", 4.0, 600)
        self.assertFalse(led.reserve("halo", "run-1", 2.0, 600)[0])
        clock.t += 601
        self.assertAlmostEqual(led.snapshot()["available"], 5.0)
        self.assertTrue(led.reserve("halo", "run-1", 2.0, 600)[0])
        ok, why = led.hold("late", 0.1, rid=r["id"])
        self.assertFalse(ok)
        self.assertIn("expired", why)

    def test_a_stuck_request_hold_is_charged_at_its_ttl(self):
        clock = Clock()
        led = ledger(self.path, clock=clock)
        self.assertTrue(led.hold("stuck", 0.4)[0])
        clock.t += shim.SPEND_REQUEST_HOLD_TTL + 1
        snap = led.snapshot()
        self.assertEqual((snap["in_flight"], round(snap["spent"], 6)), (0, 0.4))


class Durability(LedgerBase):
    def test_every_write_is_fsynced_replaced_and_the_directory_fsynced(self):
        led = ledger(self.path)
        synced = []
        real = os.fsync
        with patch.object(shim.os, "fsync", side_effect=lambda fd: (synced.append(fd), real(fd))):
            self.assertTrue(led.reserve("card-runner", "k", 1.0, 600)[0])
        self.assertGreaterEqual(len(synced), 2)                  # the temp file AND the directory
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertEqual([p.name for p in self.dir.iterdir()], ["gateway-spend.json"])  # no temp left

    def test_a_failed_write_refuses_the_reservation_and_changes_nothing(self):
        led = ledger(self.path)
        before = self.path.read_bytes()
        with patch.object(shim.os, "replace", side_effect=OSError("disk full")):
            ok, rec, why = led.reserve("card-runner", "k", 1.0, 600)
        self.assertEqual((ok, rec), (False, None))
        self.assertIn("write failed", why)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(led.snapshot()["reservations"], [])
        self.assertEqual([p.name for p in self.dir.iterdir()], ["gateway-spend.json"])

    def test_enforce_mode_hold_is_refused_when_it_cannot_be_made_durable(self):
        led = ledger(self.path)
        with patch.object(shim.os, "fsync", side_effect=OSError("io error")):
            ok, why = led.hold("r1", 0.2)
        self.assertFalse(ok)
        self.assertIn("write failed", why)
        self.assertEqual(led.snapshot()["in_flight"], 0)

    def test_observe_mode_keeps_counting_but_reports_not_durable_until_a_write_succeeds(self):
        led = ledger(self.path, enforce=lambda: False)
        with patch.object(shim.os, "fsync", side_effect=OSError("io error")):
            self.assertTrue(led.hold("r1", 0.2)[0])
            led.settle("r1", 0.15)
            snap = led.snapshot()
        self.assertEqual((snap["durable"], round(snap["spent"], 6)), (False, 0.15))
        self.assertAlmostEqual(json.loads(self.path.read_text())["spent"], 0.0)   # not yet on disk
        led.settle("r2", 0.01)                                                   # the next write
        self.assertTrue(led.snapshot()["durable"])
        self.assertAlmostEqual(json.loads(self.path.read_text())["spent"], 0.16)


class Corruption(LedgerBase):
    def _corrupt(self, enforce):
        self.path.write_bytes(b"{not json")
        return ledger(self.path, enforce=enforce)

    def test_a_corrupt_ledger_refuses_everything_in_enforce_and_observe_mode(self):
        for enforce in (lambda: True, lambda: False):
            with self.subTest(enforce=enforce()):
                led = self._corrupt(enforce)
                snap = led.snapshot()
                self.assertTrue(snap["corrupt"])
                self.assertFalse(snap["enforce"])
                self.assertEqual(snap["available"], 0.0)
                self.assertFalse(led.reserve("card-runner", "a1", 1.0, 600)[0])
                self.assertFalse(led.hold("r", 0.01)[0])
                self.assertEqual(led.finalize("x" * 32)[1], "spend ledger corrupt: refused until recovered")
                led.settle("r", 0.5)                                 # counted in memory only
                self.assertEqual(self.path.read_bytes(), b"{not json")   # never overwritten
                self.assertEqual(Path(snap["corrupt_artifact"]).read_bytes(), b"{not json")

    def test_governed_recovery_writes_a_fresh_ledger_and_keeps_the_artifact(self):
        led = self._corrupt(lambda: True)
        artifact = led.snapshot()["corrupt_artifact"]
        ok, detail = led.recover(spent=4.2, reason="operator re-count")
        self.assertTrue(ok)
        self.assertEqual(detail["source"], "operator")
        snap = led.snapshot()
        self.assertFalse(snap["corrupt"])
        self.assertAlmostEqual(snap["spent"], 4.2)
        self.assertTrue(snap["enforce"] and snap["day_complete"])
        self.assertEqual(Path(artifact).read_bytes(), b"{not json")
        self.assertTrue(led.reserve("card-runner", "a1", 1.0, 600)[0])

    def test_recovery_can_import_from_the_gateways_own_telemetry(self):
        self.path.write_bytes(b"[]")
        led = ledger(self.path, importer=lambda s, e: (7.5, 90))
        ok, detail = led.recover(reason="rebuild from telemetry")
        self.assertTrue(ok)
        self.assertIn("telemetry (90", detail["source"])
        self.assertAlmostEqual(led.snapshot()["spent"], 7.5)


class DayCompleteness(LedgerBase):
    def test_a_mid_day_ledger_imports_the_days_recorded_spend_before_it_enforces(self):
        # the live 0002 ledger: today's day, no completeness marker, $0.057 seen since 12:20
        self.path.write_text(json.dumps({"version": 1, "day": shim._spend_day(DAY0), "spent": 0.057,
                                         "reservations": {}, "holds": {}, "process": "old"}))
        seen = []
        led = ledger(self.path, importer=lambda s, e: (seen.append((s, e)), (3.2, 40))[1])
        snap = led.snapshot()
        self.assertEqual(seen[0], (shim._spend_day_start(DAY0), DAY0))
        self.assertAlmostEqual(snap["spent"], 3.2)                # max(ledger, telemetry): no double count
        self.assertTrue(snap["day_complete"] and snap["enforce"])
        self.assertEqual(snap["imported"]["requests"], 40)

    def test_without_an_import_enforcement_waits_for_the_next_phoenix_day(self):
        clock = Clock()
        self.path.write_text(json.dumps({"version": 1, "day": shim._spend_day(DAY0), "spent": 0.0,
                                         "reservations": {}, "holds": {}, "process": "old"}))
        calls = []
        led = ledger(self.path, cap=1.0, clock=clock, importer=lambda s, e: calls.append(1))
        snap = led.snapshot()
        self.assertFalse(snap["enforce"])
        self.assertIn("next Phoenix day", snap["enforce_blocked"])
        self.assertTrue(led.hold("big", 5.0)[0])                  # observe until the day is known
        led.snapshot(); led.hold("again", 0.1)
        self.assertEqual(len(calls), 1)                           # telemetry read once, not per request
        clock.t = shim._spend_day_start(DAY0) + 86400 + 60        # 00:01 the next Phoenix day
        snap = led.snapshot()
        self.assertTrue(snap["day_complete"] and snap["enforce"])
        self.assertFalse(led.hold("big-2", 5.0)[0])

    def test_telemetry_import_uses_the_settle_rule(self):
        tdir = self.dir / "telemetry"
        tdir.mkdir()
        start = shim._spend_day_start(DAY0)
        rows = [
            {"t": start + 10, "route": "remote", "alias_kind": "default", "status": 200, "cost_est": 1.0},
            {"t": start + 20, "route": "remote", "alias_kind": "builtin-remote", "status": None, "cost_est": 0.5},
            {"t": start + 30, "route": "remote", "alias_kind": "custom-remote", "status": 200, "cost_est": 9.0},
            {"t": start + 40, "route": "remote", "alias_kind": "default", "status": 502, "cost_est": 9.0},
            {"t": start + 50, "route": "local", "alias_kind": "default", "status": 200, "cost_est": 9.0},
            {"t": start - 10, "route": "remote", "alias_kind": "default", "status": 200, "cost_est": 9.0},
        ]
        for r in rows:
            name = "requests-%s.jsonl" % shim.time.strftime("%Y%m%d", shim.time.gmtime(r["t"]))
            with open(tdir / name, "a") as fh:
                fh.write(json.dumps(r) + "\n")
        self.assertEqual(shim.telemetry_remote_cost(start, DAY0, str(tdir)), (1.5, 2))


class EnforceTransition(LedgerBase):
    def test_turning_enforce_on_expires_newest_observe_reservations_until_the_cap_fits(self):
        mode = {"on": False}
        clock = Clock()
        led = ledger(self.path, cap=25.0, clock=clock, enforce=lambda: mode["on"])
        ids = []
        for i in range(3):
            clock.t += 1
            ids.append(led.reserve("card-runner", f"k{i}", 10.0, 3600)[1]["id"])   # $30 > $25
        mode["on"] = True
        snap = led.snapshot()
        self.assertTrue(snap["enforce"])
        self.assertEqual(sorted(r["id"] for r in snap["reservations"]), sorted(ids[:2]))
        self.assertEqual(led.get(ids[2])["status"], "expired-oversubscribed")
        self.assertAlmostEqual(snap["available"], 5.0)


class RetryIdempotency(LedgerBase):
    def test_reserve_finalize_and_settle_are_idempotent(self):
        led = ledger(self.path, cap=10.0)
        ok1, r1, o1 = led.reserve("card-runner", "attempt-a", 3.0, 3600)
        ok2, r2, o2 = led.reserve("card-runner", "attempt-a", 3.0, 3600)
        self.assertEqual((r1["id"], o1, o2), (r2["id"], "reserved", "replayed"))
        self.assertAlmostEqual(led.snapshot()["reserved"], 3.0)
        self.assertTrue(led.hold("q1", 0.2, rid=r1["id"])[0])
        self.assertEqual(led.hold("q1", 0.2, rid=r1["id"]), (True, "already held"))
        self.assertAlmostEqual(led.settle("q1", 0.12), 0.12)
        self.assertEqual(led.settle("q1", 0.12), 0.0)
        f1, _ = led.finalize(r1["id"], owner="card-runner")
        f2, _ = led.finalize(owner="card-runner", key="attempt-a")
        self.assertEqual((f1["status"], f2["status"], round(f2["used"], 6)), ("finalized", "finalized", 0.12))
        self.assertAlmostEqual(led.snapshot()["spent"], 0.12)
        ok3, r3, o3 = led.reserve("card-runner", "attempt-a", 3.0, 3600)
        self.assertEqual((ok3, r3["status"], o3), (True, "finalized", "replayed"))
        self.assertFalse(led.hold("q2", 0.01, rid=r1["id"])[0])
        self.assertIn("forbidden", led.finalize(r1["id"], owner="halo")[1])


class CrossClientCap(LedgerBase):
    def test_one_cap_across_runner_reservation_halo_requests_and_overflow(self):
        led = ledger(self.path, cap=25.0)
        ok, r, _ = led.reserve("card-runner", "attempt-a", 20.0, 3600, client_ip="10.0.1.12")
        self.assertFalse(led.hold("halo-big", 6.0)[0])
        self.assertTrue(led.hold("halo-ok", 4.0)[0])
        led.settle("halo-ok", 3.5)
        self.assertTrue(led.hold("overflow-1", 1.4)[0])             # overflow is held like anything else
        led.settle("overflow-1", 1.4)
        self.assertAlmostEqual(led.snapshot()["available"], 0.1)
        self.assertTrue(led.hold("runner-1", 5.0, rid=r["id"], client_ip="10.0.1.12")[0])
        led.settle("runner-1", 4.0)
        self.assertFalse(led.hold("halo-2", 0.2)[0])
        self.assertFalse(led.hold("runner-2", 16.2, rid=r["id"], client_ip="10.0.1.12")[0])
        self.assertTrue(led.hold("runner-3", 16.1, rid=r["id"], client_ip="10.0.1.12")[0])

    def test_a_reservation_cannot_be_used_from_another_client(self):
        led = ledger(self.path)
        ok, r, _ = led.reserve("card-runner", "a", 2.0, 600, client_ip="10.0.1.12")
        ok, why = led.hold("stolen", 0.1, rid=r["id"], client_ip="10.0.1.99")
        self.assertFalse(ok)
        self.assertIn("bound to another client", why)

    def test_observe_mode_accounts_but_never_refuses_the_cap(self):
        led = ledger(self.path, cap=1.0, enforce=lambda: False)
        self.assertTrue(led.reserve("card-runner", "a", 5.0, 600)[0])
        self.assertTrue(led.hold("x", 3.0)[0])
        led.settle("x", 2.0)
        self.assertAlmostEqual(led.snapshot()["spent"], 2.0)


# ---- the HTTP seam ------------------------------------------------------------

class Request:
    path = "/v1/chat/completions"
    method = "POST"

    def __init__(self, model="estate-remote", remote="127.0.0.1", headers=None, **fields):
        self.remote = remote
        self.headers = {"User-Agent": "offline-test", **(headers or {})}
        self.body = json.dumps({"model": model, "messages": [{"role": "user", "content": "x"}],
                                "max_tokens": 1000, **fields}).encode()

    async def read(self):
        return self.body


class JsonRequest:
    def __init__(self, payload, token=None, remote="10.0.1.12", admin=None):
        self.payload, self.remote = payload, remote
        self.headers = {}
        if token:
            self.headers["X-Spend-Token"] = token
        if admin:
            self.headers["X-Admin-Token"] = admin

    async def json(self):
        return self.payload


class Seam(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="spend-seam-", dir=_TMP))
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.led = ledger(self.dir / "gateway-spend.json", cap=1.0)
        clients = self.dir / "spend-clients.json"
        clients.write_text(json.dumps({"clients": {
            "card-runner": hashlib.sha256(b"runner-secret").hexdigest(),
            "halo": hashlib.sha256(b"halo-secret").hexdigest()}}))
        os.chmod(clients, 0o600)
        self.clients = clients
        self.relay = AsyncMock(return_value=("ok", shim.web.json_response({"ok": True})))
        for name, value in dict(_SPEND_LEDGER=self.led, SPEND_CLIENTS_FILE=str(clients),
                                _SPEND_CLIENTS={"mtime": None, "map": {}}, SHIM_ADMIN_TOKEN="adm",
                                LOG_REQUESTS=0, FORCE_REMOTE=0,
                                REMOTE_COST_IN_PER_MTOK=0.15, REMOTE_COST_OUT_PER_MTOK=0.6,
                                REMOTE_COST_IN_PER_MTOK_PEAK=0.3, REMOTE_COST_OUT_PER_MTOK_PEAK=1.2,
                                REMOTE_BASE="https://remote.test", _est_tokens=lambda body: 100_000,
                                estimate_units=lambda *a, **k: 1, remote_ok=lambda: True,
                                is_peak=lambda: False, record_event=self._record, _relay=self.relay,
                                _note_remote_response=AsyncMock(), _note_payload_outcome=lambda *a, **k: None,
                                _telemetry_note_request=lambda *a, **k: None).items():
            self.stack.enter_context(patch.object(shim, name, value))

    @staticmethod
    def _record(decision, reason, request, *a, **k):
        shim._active_set(request, route=decision, reason=reason)

    async def _reserve(self, token=b"runner-secret", amount=0.9, remote="10.0.1.12", **extra):
        resp = await shim.gateway_spend_reserve(JsonRequest(
            {"key": "attempt-a", "amount": amount, "ttl_s": 600, **extra},
            token=token.decode() if token else None, remote=remote))
        return resp.status, json.loads(resp.body)

    async def test_cap_exhausted_estate_remote_is_refused_before_forwarding(self):
        self.led.settle("earlier", 0.99)
        resp = await shim.handle_completions(Request())
        self.assertEqual(resp.status, 429)
        self.assertEqual(json.loads(resp.body)["error"]["type"], "spend_cap_exhausted")
        self.relay.assert_not_called()

    async def test_every_paid_route_is_held_overflow_and_intent_included(self):
        self.led.settle("earlier", 0.99)
        with patch.object(shim, "FORCE_REMOTE", 1):                            # forced window
            resp = await shim.handle_completions(Request(model="qwen-local"))
        self.assertEqual(resp.status, 429)
        resp = await shim.handle_completions(Request(model="qwen-local",
                                                     headers={"X-Gateway-Route-Intent": "overflow"}))
        self.assertEqual(resp.status, 429)
        with patch.object(shim, "local_healthy", AsyncMock(return_value=False)):   # local-down overflow
            resp = await shim.handle_completions(Request(model="qwen-local"))
        self.assertEqual(resp.status, 429)
        self.relay.assert_not_called()

    async def test_a_forwarded_request_is_held_then_settled_at_the_gateway_price(self):
        resp = await shim.handle_completions(Request())
        self.assertEqual(resp.status, 200)
        snap = self.led.snapshot()
        self.assertEqual(snap["in_flight"], 0)
        self.assertAlmostEqual(snap["spent"], 0.015, places=6)

    async def test_reserve_requires_the_spend_credential_and_binds_owner_and_ip(self):
        self.assertEqual((await self._reserve(token=None))[0], 401)
        self.assertEqual((await self._reserve(token=b"wrong"))[0], 401)
        self.assertEqual((await self._reserve(owner="halo"))[0], 403)
        status, body = await self._reserve()
        self.assertEqual(status, 200)
        rid = body["reservation"]["id"]
        self.assertEqual(body["reservation"]["owner"], "card-runner")
        self.assertNotIn("bound_ip", body["reservation"])
        ok = await shim.handle_completions(Request(model=f"estate-remote.{rid}", remote="10.0.1.12"))
        self.assertEqual(ok.status, 200)
        stolen = await shim.handle_completions(Request(model=f"estate-remote.{rid}", remote="10.0.1.66"))
        self.assertEqual(stolen.status, 429)
        self.assertEqual(json.loads(stolen.body)["error"]["type"], "spend_reservation_invalid")
        self.assertEqual(self.relay.await_count, 1)

    async def test_finalize_is_owner_only(self):
        _, body = await self._reserve()
        rid = body["reservation"]["id"]
        for token, code in ((None, 401), ("halo-secret", 403), ("runner-secret", 200)):
            resp = await shim.gateway_spend_finalize(JsonRequest({"reservation_id": rid}, token=token))
            self.assertEqual(resp.status, code, token)
        again = await shim.handle_completions(Request(model=f"estate-remote.{rid}", remote="10.0.1.12"))
        self.assertEqual(again.status, 429)

    async def test_a_world_readable_clients_file_authorizes_no_one(self):
        os.chmod(self.clients, 0o644)
        self.assertEqual((await self._reserve())[0], 401)

    async def test_recovery_needs_the_admin_token_and_get_stays_read_only(self):
        resp = await shim.gateway_spend_recover(JsonRequest({"spent": 1.0}))
        self.assertEqual(resp.status, 401)
        resp = await shim.gateway_spend_recover(JsonRequest({"spent": 0.3, "reason": "t"}, admin="adm"))
        self.assertEqual(resp.status, 200)
        snap = json.loads((await shim.gateway_spend(None)).body)
        self.assertAlmostEqual(snap["spent"], 0.3)
        self.assertEqual(snap["gateway_sha256"], shim.GATEWAY_SHA256)

    def test_alias_resolution_carries_only_well_formed_reservation_ids(self):
        rid = "a" * 32
        self.assertEqual(shim._resolve_alias(f"estate-remote.{rid}"),
                         {"name": "estate-remote", "kind": "builtin-remote", "reservation": rid})
        self.assertNotEqual(shim._resolve_alias("estate-remote.not-an-id").get("kind"), "builtin-remote")


if __name__ == "__main__":
    unittest.main()
