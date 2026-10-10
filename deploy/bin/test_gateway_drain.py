"""Accepted gateway calls survive a leased deployment drain."""
import asyncio
import importlib.util
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("shim_drain_test", Path(__file__).with_name("keepalive-shim.py"))
shim = importlib.util.module_from_spec(SPEC)
with patch.dict(os.environ, {"SHIM_ADMIN_TOKEN": "cpu-test-token", "SHIM_ENV_FILE": "/dev/null"}):
    SPEC.loader.exec_module(shim)


class Request:
    path = "/v1/chat/completions"
    remote = "127.0.0.1"
    headers = {}
    method = "POST"

    async def read(self):
        return b'{"model":"estate","messages":[{"role":"user","content":"hi"}]}'


class DrainRequest:
    headers = {}

    def __init__(self, method, payload=None):
        self.method = method
        self.payload = payload or {}

    async def json(self):
        return self.payload


class Drain(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.old = (shim._DRAIN_UNTIL, shim._DRAIN_LEASE, shim._DRAIN_REASON, shim._DRAIN_REC, shim._DRAIN_LEDGER,
                    shim._DRAIN_DURABLE, shim._DRAIN_INVALID, shim.DRAIN_STATE_FILE, shim.PUBLISH_STATE_ROOT)
        shim._DRAIN_UNTIL, shim._DRAIN_LEASE, shim._DRAIN_REASON, shim._DRAIN_REC = 0.0, None, None, None
        shim._DRAIN_DURABLE, shim._DRAIN_INVALID = False, None
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, True)      # ratchet: no unmanaged temp dirs
        shim._DRAIN_LEDGER = self._tmp + "/drains.jsonl"      # never write the production ledger from a test
        shim.DRAIN_STATE_FILE = self._tmp + "/drain-state.json"
        shim.PUBLISH_STATE_ROOT = self._tmp + "/publications"

    async def asyncTearDown(self):
        (shim._DRAIN_UNTIL, shim._DRAIN_LEASE, shim._DRAIN_REASON, shim._DRAIN_REC, shim._DRAIN_LEDGER,
         shim._DRAIN_DURABLE, shim._DRAIN_INVALID, shim.DRAIN_STATE_FILE, shim.PUBLISH_STATE_ROOT) = self.old
        shim._ACTIVE.clear()

    async def test_drain_preserves_an_accepted_call_and_refuses_new_work(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def route(_request):
            entered.set()
            await release.wait()
            return shim.web.json_response({"ok": True})

        with patch.object(shim, "_admin_ok", return_value=True), \
             patch.object(shim, "_route_completions", side_effect=route), \
             patch.object(shim, "_spend_settle"), \
             patch.object(shim, "_telemetry_note_request"):
            first = asyncio.create_task(shim.handle_completions(Request()))
            await asyncio.wait_for(entered.wait(), timeout=1)
            start = await shim.gateway_drain(DrainRequest("POST", {"ttl_s": 30}))
            lease = json.loads(start.body)["lease"]
            self.assertEqual(json.loads(start.body)["active"], 1)
            refused = await shim.handle_completions(Request())
            self.assertEqual(refused.status, 503)
            self.assertEqual(refused.headers["X-Gateway-Drain"], "active")
            self.assertEqual(len(shim._ACTIVE), 1)
            release.set()
            self.assertEqual((await first).status, 200)
            self.assertEqual(json.loads((await shim.gateway_drain(DrainRequest("GET"))).body)["active"], 0)
            self.assertEqual((await shim.gateway_drain(DrainRequest("DELETE", {"lease": lease}))).status, 200)

    async def test_lease_expires_and_bad_release_cannot_clear_another_lease(self):
        with patch.object(shim, "_admin_ok", return_value=True):
            start = await shim.gateway_drain(DrainRequest("POST", {"ttl_s": 30}))
            self.assertEqual(start.status, 200)
            self.assertEqual((await shim.gateway_drain(DrainRequest("DELETE", {"lease": "wrong"}))).status, 409)
            shim._DRAIN_UNTIL = time.time() - 1
            self.assertFalse(shim._draining())
            self.assertEqual((await shim.gateway_drain(DrainRequest("POST", {"ttl_s": 30}))).status, 200)

    def _ledger(self):
        return [json.loads(l) for l in Path(shim._DRAIN_LEDGER).read_text().splitlines()]

    async def test_every_drain_is_one_durable_record_with_reason_holder_and_refusals(self):
        with patch.object(shim, "_admin_ok", return_value=True), patch.object(shim, "_spend_settle"), patch.object(shim, "_telemetry_note_request"):
            start = await shim.gateway_drain(DrainRequest("POST", {"ttl_s": 30, "reason": "engine planned restart: test", "by": "unit-test"}))
            lease = json.loads(start.body)["lease"]
            for _ in range(3):
                r = await shim.handle_completions(Request())
                self.assertEqual(r.status, 503)
            self.assertIn("engine planned restart: test", json.loads(r.body)["error"]["message"])
            self.assertLessEqual(int(r.headers["Retry-After"]), 30)
            status = json.loads((await shim.gateway_drain(DrainRequest("GET"))).body)
            self.assertEqual((status["by"], status["refused"]), ("unit-test", 3))
            await shim.gateway_drain(DrainRequest("DELETE", {"lease": lease}))
        rows = self._ledger()
        self.assertEqual([r["event"] for r in rows], ["open", "close"])
        self.assertEqual((rows[1]["how"], rows[1]["refused"], rows[1]["by"]), ("delete", 3, "unit-test"))
        self.assertEqual(rows[1]["refused_by_client"], {"127.0.0.1": 3})

    async def test_expired_lease_and_gateway_restart_both_close_the_record(self):
        with patch.object(shim, "_admin_ok", return_value=True):
            await shim.gateway_drain(DrainRequest("POST", {"ttl_s": 30, "reason": "x", "by": "t"}))
            shim._DRAIN_UNTIL = time.time() - 1
            await shim.gateway_drain(DrainRequest("GET"))           # lazy reap
            self.assertEqual(self._ledger()[-1]["how"], "expired")
            await shim.gateway_drain(DrainRequest("POST", {"ttl_s": 30, "reason": "y", "by": "t"}))
            shim._DRAIN_REC = None                                  # the process holding the fence "died"
            shim._drain_startup_recover()
            self.assertEqual(self._ledger()[-1]["how"], "gateway-restart")

    async def test_durable_fence_survives_restart_and_ttl_until_owned_release(self):
        with patch.object(shim, "_admin_ok", return_value=True), patch.object(shim, "_spend_settle"), patch.object(shim, "_telemetry_note_request"):
            start = await shim.gateway_drain(DrainRequest("POST", {"ttl_s": 30, "durable": True}))
            lease = json.loads(start.body)["lease"]
            self.assertEqual(Path(shim.DRAIN_STATE_FILE).stat().st_mode & 0o777, 0o600)
            row = json.loads(Path(shim.DRAIN_STATE_FILE).read_text())
            row["until"] = time.time() - 1
            Path(shim.DRAIN_STATE_FILE).write_text(json.dumps(row))
            shim._DRAIN_UNTIL, shim._DRAIN_LEASE, shim._DRAIN_REASON, shim._DRAIN_REC = 0.0, None, None, None
            shim._DRAIN_DURABLE = False
            shim._drain_startup_recover()
            status = json.loads((await shim.gateway_drain(DrainRequest("GET"))).body)
            self.assertTrue(status["draining"])
            self.assertTrue(status["durable"])
            self.assertNotIn("lease", status)
            self.assertEqual(status["lease_sha256"], shim.hashlib.sha256(lease.encode()).hexdigest())
            refused = await shim.handle_completions(Request())
            self.assertEqual(refused.status, 503)
            self.assertEqual((await shim.gateway_drain(DrainRequest("DELETE", {"lease": "wrong"}))).status, 409)
            self.assertEqual((await shim.gateway_drain(DrainRequest("DELETE", {"lease": lease}))).status, 200)
            self.assertFalse(Path(shim.DRAIN_STATE_FILE).exists())
            self.assertFalse(shim._draining())

    async def test_unknown_persisted_state_fences_instead_of_opening_work(self):
        for data in ("{", "null", '{"v":1,"durable":true}', '{"v":999}'):
            with self.subTest(data=data), patch.object(shim, "_admin_ok", return_value=True):
                shim._DRAIN_INVALID = None
                Path(shim.DRAIN_STATE_FILE).write_text(data)
                self.assertTrue(shim._drain_restore())
                self.assertTrue(shim._draining())
                self.assertIsNotNone(shim._DRAIN_INVALID)
                self.assertEqual((await shim.gateway_drain(DrainRequest("POST", {"ttl_s": 30}))).status, 409)

    async def test_write_failure_keeps_durable_admission_fenced(self):
        with patch.object(shim, "_admin_ok", return_value=True), patch.object(shim, "_drain_persist", side_effect=OSError("disk")):
            reply = await shim.gateway_drain(DrainRequest("POST", {"ttl_s": 30, "durable": True}))
            self.assertEqual(reply.status, 503)
            shim._DRAIN_UNTIL = time.time() - 1
            self.assertTrue(shim._draining())

    async def test_durable_fence_can_be_renewed_only_by_same_lease(self):
        with patch.object(shim, "_admin_ok", return_value=True):
            start = await shim.gateway_drain(DrainRequest("POST", {"ttl_s": 30, "durable": True}))
            lease = json.loads(start.body)["lease"]
            self.assertEqual((await shim.gateway_drain(DrainRequest("POST", {"ttl_s": 30, "lease": "wrong"}))).status, 409)
            self.assertEqual((await shim.gateway_drain(DrainRequest("POST", {"ttl_s": 90, "lease": lease}))).status, 200)
            self.assertEqual(json.loads(Path(shim.DRAIN_STATE_FILE).read_text())["lease"], lease)

    def _publication(self, lease, **changes):
        root = Path(shim.PUBLISH_STATE_ROOT)
        directory = root / "publish-test"
        directory.mkdir(parents=True, mode=0o700)
        receipt = directory / "state.json"
        state = {"v": 1, "transaction": directory.name, "source_head": "c" * 40,
                 "status": "release-pending", "lease": lease, **changes}
        receipt.write_text(json.dumps(state))
        receipt.chmod(0o600)
        (root / "active.json").write_text(json.dumps({"receipt": str(receipt), "source_head": "c" * 40}))
        (root / "active.json").chmod(0o600)
        return receipt

    async def test_delete_unlink_then_fsync_failure_survives_restart_from_journal(self):
        with patch.object(shim, "_admin_ok", return_value=True):
            opened = await shim.gateway_drain(DrainRequest("POST", {"ttl_s": 30, "durable": True}))
            lease = json.loads(opened.body)["lease"]
            self._publication(lease)
            with patch.object(shim.os, "fsync", side_effect=OSError("parent fsync")), \
                 patch.object(shim, "_drain_persist", side_effect=OSError("recreate failed")):
                self.assertEqual((await shim.gateway_drain(DrainRequest("DELETE", {"lease": lease}))).status, 503)
            self.assertFalse(Path(shim.DRAIN_STATE_FILE).exists())
            self.assertTrue(shim._draining())
            shim._DRAIN_DURABLE, shim._DRAIN_LEASE, shim._DRAIN_UNTIL = False, None, 0
            self.assertTrue(shim._drain_restore())
            self.assertTrue(shim._draining())
            self.assertEqual(shim._DRAIN_LEASE, lease)
            self.assertEqual(json.loads(Path(shim.DRAIN_STATE_FILE).read_text())["lease"], lease)
            self.assertEqual((await shim.gateway_drain(DrainRequest("DELETE", {"lease": "wrong"}))).status, 409)

    async def test_missing_drain_with_unknown_publication_fails_closed(self):
        self._publication(None)
        self.assertTrue(shim._drain_restore())
        self.assertTrue(shim._draining())
        self.assertIsNotNone(shim._DRAIN_INVALID)

    async def test_qualified_completed_journal_does_not_reopen_a_drain(self):
        self._publication("e" * 32, status="published", qualified="candidate",
                          generation_canary={"ok": True, "runtime_unchanged": True,
                                             "finish_reason": "stop", "completion_tokens": 2})
        self.assertFalse(shim._drain_restore())
        self.assertFalse(shim._draining())


if __name__ == "__main__":
    unittest.main()
