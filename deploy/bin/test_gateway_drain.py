"""Accepted gateway calls survive a leased deployment drain."""
import asyncio
import importlib.util
import json
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
        self.old = (shim._DRAIN_UNTIL, shim._DRAIN_LEASE, shim._DRAIN_REASON)
        shim._DRAIN_UNTIL, shim._DRAIN_LEASE, shim._DRAIN_REASON = 0.0, None, None

    async def asyncTearDown(self):
        shim._DRAIN_UNTIL, shim._DRAIN_LEASE, shim._DRAIN_REASON = self.old
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


if __name__ == "__main__":
    unittest.main()
