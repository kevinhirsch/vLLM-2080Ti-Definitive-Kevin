"""Offline concurrent routing tests, also runnable against an exact-live candidate.

SHIM_TEST_CANDIDATE=/absolute/candidate.py python -m pytest -q this_file.py
"""
import asyncio
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

os.environ["SHIM_EXACT_TOKENS"] = "0"
SPEC = importlib.util.spec_from_file_location("shim_memory_test", os.environ.get(
    "SHIM_TEST_CANDIDATE", str(Path(__file__).with_name("keepalive-shim.py"))))
shim = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shim)


class Request:
    path = "/v1/chat/completions"
    remote = "127.0.0.1"
    method = "POST"
    headers = {"X-Client": "test-batch", "User-Agent": "offline-test"}

    def __init__(self, **fields):
        self.body = json.dumps({"messages": [{"role": "user", "content": "review"}],
                                "max_tokens": 200, **fields}).encode()

    async def read(self):
        return self.body


class ReservationRouting(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        values = dict(_inflight=0, _inflight_tokens=0, _inflight_reserved_tokens=0,
                      _waiting=0, _health={"ok": True}, TOKEN_BUDGET=599,
                      LOCAL_MAX_OUT=200, DEFAULT_MAX_OUT=200, MAX_LOCAL_TOKENS=10000,
                      FORCE_REMOTE=0, BIG_OUTPUT=0, BIG_PROMPT=0, MONSTER_INFLIGHT=0,
                      FOREIGN_LOAD_GUARD=0, TINY_TOKENS=0, LOCAL_WAIT=0, BG_WAIT=0,
                      BG_LOCAL_ONLY=0, FG_RESERVED=0, BG_BIG_LOCAL_WHEN_IDLE=0,
                      LOG_REQUESTS=0, CRASH_ADAPTIVE=0, EMPTY_RETRY=0,
                      FLOW_MODE="off")   # these tests pin the pre-CF admission semantics (CF has its own: test_gateway_flow.py)
        for name, value in values.items():
            self.stack.enter_context(patch.object(shim, name, value))
        mocks = dict(_est_tokens=lambda body: 100, estimate_units=lambda *a, **k: 1,
                     effective_budget=lambda: 10, remote_ok=lambda: True,
                     local_healthy=AsyncMock(return_value=True), is_peak=lambda: False,
                     _active_set=lambda *a, **k: None, record_event=lambda *a, **k: None,
                     _note_payload_outcome=lambda *a, **k: None,
                     _forward_remote=AsyncMock(return_value="remote"))
        for name, value in mocks.items():
            self.stack.enter_context(patch.object(shim, name, value))

    async def test_concurrent_output_reservations_overflow_then_release(self):
        entered, finish = asyncio.Event(), asyncio.Event()

        async def relay(*args, **kwargs):
            entered.set()
            await finish.wait()
            return "ok", "local"

        with patch.object(shim, "_relay", relay):
            first = asyncio.create_task(shim._route_completions(Request()))
            await asyncio.wait_for(entered.wait(), 2)
            self.assertEqual(shim._inflight_reserved_tokens, 300)
            self.assertEqual(await shim._route_completions(Request()), "remote")
            self.assertEqual(shim._inflight_reserved_tokens, 300)
            finish.set()
            self.assertEqual(await first, "local")
        self.assertEqual((shim._inflight, shim._inflight_tokens, shim._inflight_reserved_tokens), (0, 0, 0))

    async def test_tiny_path_obeys_same_memory_cap(self):
        with patch.object(shim, "TINY_TOKENS", 1000):
            await self.test_concurrent_output_reservations_overflow_then_release()

    async def test_idle_engine_cannot_bypass_token_cap(self):
        with patch.object(shim, "TOKEN_BUDGET", 299), patch.object(shim, "_relay", AsyncMock()) as relay:
            self.assertEqual(await shim._route_completions(Request()), "remote")
            relay.assert_not_called()

    async def test_failover_releases_memory_before_remote_work(self):
        async def remote(*args):
            self.assertEqual(shim._inflight_reserved_tokens, 0)
            self.assertEqual(shim._inflight, 0)
            return "remote"

        for tiny in (0, 1000):
            with patch.object(shim, "TINY_TOKENS", tiny), \
                    patch.object(shim, "_relay", AsyncMock(return_value=("error", (500, "fail", False)))), \
                    patch.object(shim, "_forward_remote", remote):
                self.assertEqual(await shim._route_completions(Request()), "remote")
                self.assertEqual(shim._inflight_reserved_tokens, 0)

    async def test_exception_and_cancellation_release_all_reservations(self):
        for tiny in (0, 1000):
            for error in (RuntimeError("relay failed"), asyncio.CancelledError()):
                with patch.object(shim, "TINY_TOKENS", tiny), \
                        patch.object(shim, "_relay", AsyncMock(side_effect=error)):
                    with self.assertRaises(type(error)):
                        await shim._route_completions(Request())
                    self.assertEqual((shim._inflight, shim._inflight_reserved_tokens), (0, 0))

    async def test_cap_applied_to_relay_and_reserved_exactly_once(self):
        async def relay(request, local, path, body, *args, **kwargs):
            data = json.loads(body)
            self.assertEqual(data["max_tokens"], 200)
            self.assertNotIn("max_completion_tokens", data)
            self.assertEqual(shim._inflight_reserved_tokens, 300)
            return "ok", "local"

        # The 2026-09-17 context work added CONTEXT_SAFETY_MARGIN (1024 by default) to the
        # size gate: 100 prompt + 9000 max + 1024 > this fixture's 10000-token local cap, so
        # the 9000 case now routes remote for size before the reservation cap is ever
        # reached. This test is about capping the RESERVATION (9000 -> LOCAL_MAX_OUT 200),
        # not the size gate, so give the size gate room for the margin explicitly.
        room = 100 + 9000 + max(0, shim.CONTEXT_SAFETY_MARGIN) + 1
        with patch.object(shim, "_relay", relay), patch.object(shim, "MAX_LOCAL_TOKENS", room), \
                patch.object(shim, "LOCAL_CONTEXT_LIMIT", room):
            for fields in ({"max_tokens": 9000}, {"max_tokens": 0},
                           {"max_tokens": None}, {"max_tokens": 1, "max_completion_tokens": 9000}):
                self.assertEqual(await shim._route_completions(Request(**fields)), "local")
                self.assertEqual(shim._inflight_reserved_tokens, 0)

    async def test_multiple_sequences_reserve_prompt_and_output_per_sequence(self):
        with patch.object(shim, "_relay", AsyncMock()) as relay:
            self.assertEqual(await shim._route_completions(Request(n=2)), "remote")
            self.assertEqual(await shim._route_completions(Request(n=1, best_of=2)), "remote")
            relay.assert_not_called()

    async def test_idle_big_background_cannot_bypass_sequence_backstop(self):
        with patch.object(shim, "TOKEN_BUDGET", 99999), \
                patch.object(shim, "BG_BIG_LOCAL_WHEN_IDLE", 1), \
                patch.object(shim, "_relay", AsyncMock()) as relay:
            self.assertEqual(await shim._route_completions(Request(n=11)), "remote")
            relay.assert_not_called()

    async def test_invalid_output_returns_error_without_reserving(self):
        response = await shim._route_completions(Request(max_tokens="garbage"))
        self.assertEqual(response.status, 400)
        self.assertEqual(shim._inflight_reserved_tokens, 0)

    async def test_impossible_local_only_request_does_not_wait_forever(self):
        with patch.object(shim, "TOKEN_BUDGET", 299), patch.object(shim, "remote_ok", lambda: False):
            response = await asyncio.wait_for(shim._route_completions(Request()), 1)
            self.assertEqual(response.status, 503)
            self.assertEqual(shim._inflight_reserved_tokens, 0)

    async def test_cancellation_during_flight_recorder_releases_reservation(self):
        with patch.dict(os.environ, {"SHIM_FLIGHTREC_MIN_TOK": "0"}), \
                patch.object(asyncio.get_running_loop(), "run_in_executor", side_effect=asyncio.CancelledError()):
            with self.assertRaises(asyncio.CancelledError):
                await shim._route_completions(Request())
        self.assertEqual((shim._inflight, shim._inflight_reserved_tokens), (0, 0))

    def test_disabled_output_cap_reserves_full_context_when_unbounded(self):
        with patch.object(shim, "LOCAL_MAX_OUT", 0):
            self.assertEqual(shim.local_memory_reservation(Request(max_tokens=0).body), (10000, 1))


if __name__ == "__main__":
    unittest.main()
