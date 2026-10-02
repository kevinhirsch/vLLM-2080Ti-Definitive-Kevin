"""Client disconnects cannot leak a gateway upstream session."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

spec = importlib.util.spec_from_file_location(
    "keepalive_shim_stream_cleanup", Path(__file__).with_name("keepalive-shim.py"))
shim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shim)


class StreamCleanup(unittest.IsolatedAsyncioTestCase):
    async def test_eof_disconnect_closes_upstream_once(self):
        resp = SimpleNamespace(write_eof=AsyncMock(side_effect=ConnectionResetError()))
        session = SimpleNamespace(close=AsyncMock())
        request = object()
        with patch.object(shim, "_active_set") as active_set:
            await shim._finish_stream_response(resp, request, session)
        session.close.assert_awaited_once()
        active_set.assert_called_once_with(request, client_disconnected=True)

    async def test_unexpected_eof_failure_still_closes_upstream(self):
        resp = SimpleNamespace(write_eof=AsyncMock(side_effect=RuntimeError("broken stream")))
        session = SimpleNamespace(close=AsyncMock())
        with self.assertRaisesRegex(RuntimeError, "broken stream"):
            await shim._finish_stream_response(resp, object(), session)
        session.close.assert_awaited_once()

    async def test_prepare_disconnect_closes_upstream(self):
        resp = SimpleNamespace(prepare=AsyncMock(side_effect=ConnectionResetError()),
                               write=AsyncMock())
        session = SimpleNamespace(close=AsyncMock())
        request = object()
        with patch.object(shim, "_active_set") as active_set:
            sent = await shim._prepare_stream_response(resp, request, b"data", session)
        self.assertFalse(sent)
        session.close.assert_awaited_once()
        resp.write.assert_not_awaited()
        active_set.assert_called_once_with(request, client_disconnected=True)

    async def test_first_write_disconnect_closes_upstream(self):
        resp = SimpleNamespace(prepare=AsyncMock(),
                               write=AsyncMock(side_effect=BrokenPipeError()))
        session = SimpleNamespace(close=AsyncMock())
        with patch.object(shim, "_active_set"):
            sent = await shim._prepare_stream_response(resp, object(), b"data", session)
        self.assertFalse(sent)
        session.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
