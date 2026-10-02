"""Accepted gateway calls survive a leased deployment drain."""
import asyncio
import importlib.util
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("shim_drain_test", Path(__file__).with_name("keepalive-shim.py"))
shim = importlib.util.module_from_spec(SPEC)
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
        self.old = (shim._DRAIN_UNTIL, shim._DRAIN_LEASE, shim._DRAIN_REASON, shim._DRAIN_REC, shim._DRAIN_LEDGER)
        shim._DRAIN_UNTIL, shim._DRAIN_LEASE, shim._DRAIN_REASON, shim._DRAIN_REC = 0.0, None, None, None
        self._tmp = tempfile.mkdtemp()
        shim._DRAIN_LEDGER = self._tmp + "/drains.jsonl"      # never write the production ledger from a test

    async def asyncTearDown(self):
        shim._DRAIN_UNTIL, shim._DRAIN_LEASE, shim._DRAIN_REASON, shim._DRAIN_REC, shim._DRAIN_LEDGER = self.old
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
        return [json.loads(l) for l in open(shim._DRAIN_LEDGER)]

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


if __name__ == "__main__":
    unittest.main()
