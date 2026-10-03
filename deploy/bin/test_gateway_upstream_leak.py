#!/usr/bin/env python3
"""Lane A6: _relay left the upstream response un-released when the first-token gate was abandoned.

LIVE 2026-10-03: 20 x 'ERROR Unclosed connection' (aiohttp Connection.__del__) in the gateway journal, to the local engine
(127.0.0.1:8001) and to the remote provider, always right after a client hung up or a first-token deadline expired. The gate's
three exits did `await session.close()` while `up` (the response whose headers had arrived) was never closed: session.close() shuts
the sockets but leaves the response's pooled Connection un-released, so its garbage collection reports a leak. Driven here
against a real aiohttp upstream that sends headers and one non-meaningful SSE comment, then stalls.

Run:  python -m pytest -q test_gateway_upstream_leak.py
"""
import asyncio
import gc
import time
import types
import unittest
from unittest.mock import patch

from aiohttp import web

import test_gateway_cache_model as C

shim = C.shim


def shim_source():
    from pathlib import Path
    return Path(__file__).with_name("keepalive-shim.py").read_text()


class StalledUpstream(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.leaks = []
        asyncio.get_running_loop().set_exception_handler(lambda loop, ctx: self.leaks.append(ctx.get("message")))

        async def handler(req):
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(req)
            await resp.write(b": keepalive\n\n")                    # headers arrived, nothing meaningful yet
            await asyncio.sleep(30)
            return resp

        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        self.runner = web.AppRunner(app, shutdown_timeout=0.2)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.base = "http://127.0.0.1:%d" % site._server.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def relay(self, gone_after=None, gate_s=30):
        t0 = time.time()

        class Tr:
            @staticmethod
            def is_closing():
                return gone_after is not None and time.time() - t0 > gone_after

        req = types.SimpleNamespace(transport=Tr)
        with patch.object(shim, "first_token_timeout", lambda *a, **k: gate_s):
            out = await shim._relay(req, self.base, "/v1/chat/completions", b'{"stream":true}', "", True)
        del req
        for _ in range(3):
            gc.collect()
            await asyncio.sleep(0.05)
        return out

    async def test_client_hangs_up_in_the_first_token_gate(self):
        kind, res = await self.relay(gone_after=0.3)
        self.assertEqual((kind, res.status), ("ok", 499))
        self.assertEqual(self.leaks, [])

    async def test_first_token_deadline_expires_in_the_gate(self):
        kind, res = await self.relay(gate_s=0.3)
        self.assertEqual(kind, "fail")
        self.assertIn("no first token", res[1])
        self.assertEqual(self.leaks, [])

    async def test_client_hangs_up_mid_stream(self):
        """The committed-stream path (nearly all real traffic): the caller's socket dies on a write while the engine is
        still generating. Upstream streams real SSE chunks; the client-side StreamResponse is a fake that fails on write #3."""
        RealResp = web.StreamResponse                 # the shim's client-side response class is patched below, the stub server's is not

        async def stream_handler(req):
            resp = RealResp(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(req)
            for i in range(200):
                await resp.write(b'data: {"choices":[{"index":0,"delta":{"content":"tok%d"},"finish_reason":null}]}\n\n' % i)
                await asyncio.sleep(0.01)
            return resp

        app = web.Application()
        app.router.add_post("/v1/chat/completions", stream_handler)
        runner = web.AppRunner(app, shutdown_timeout=0.2)
        await runner.setup()
        self.addAsyncCleanup(runner.cleanup)
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        base = "http://127.0.0.1:%d" % site._server.sockets[0].getsockname()[1]
        writes = []

        class FakeResp:
            def __init__(self, status=200, headers=None):
                self.status, self.headers = status, headers or {}

            async def prepare(self, request):
                pass

            async def write(self, data):
                writes.append(data)
                if len(writes) >= 3:
                    raise ConnectionResetError("client hung up")

            async def write_eof(self):
                pass

        req = types.SimpleNamespace(transport=None)
        with patch.object(shim.web, "StreamResponse", FakeResp), patch.object(shim, "_client_gone", lambda r: False):
            kind, res = await shim._relay(req, base, "/v1/chat/completions", b'{"stream":true}', "", True)
        self.assertEqual(kind, "ok")
        del res, req
        for _ in range(3):
            gc.collect()
            await asyncio.sleep(0.05)
        self.assertGreaterEqual(len(writes), 3)
        self.assertEqual(self.leaks, [])

    def test_no_relay_exit_closes_a_session_without_its_response(self):
        """Class ratchet: inside _relay every session close that can follow an opened upstream response goes through
        _drop_upstream (the reasoning-retry session s2 had the same leak)."""
        src = shim_source()
        relay = src[src.index("async def _relay("):src.index("_RW_STATS = ")]
        self.assertNotIn("await s2.close()", relay)
        after_open = relay[relay.index("if up.status >= 500:"):]
        bare = [l.strip() for l in after_open.splitlines() if l.strip() == "await session.close()"]
        # the one bare close left follows up.read() of a streaming 4xx body (read() releases the connection)
        self.assertLessEqual(len(bare), 1, bare)

    async def test_drop_upstream_helper_closes_response_before_session(self):
        order = []

        class Up:
            def close(self):
                order.append("response")

        class Sess:
            async def close(self):
                order.append("session")

        await shim._drop_upstream(Sess(), Up())
        await shim._drop_upstream(Sess(), None)
        self.assertEqual(order, ["response", "session", "session"])


if __name__ == "__main__":
    unittest.main()
