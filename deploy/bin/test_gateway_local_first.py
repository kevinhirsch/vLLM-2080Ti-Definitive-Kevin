#!/usr/bin/env python3
"""L1 local-first overflow policy (keepalive-shim, 2026-09-25).

Capacity/latency-PREDICTIVE remote reasons (big-prompt, perf, predicted, big-out, monster)
route remote only when local is measurably saturated (lanes / tokens / queued / queue-wait);
explicit remote intents and safety routes are unchanged; SHIM_LOCAL_FIRST=0 restores the
previous policy exactly.

ISOLATION: the shim is imported with EVERY on-disk path it can write (spend ledger, spend
clients, stats, telemetry, env file, aliases, flight recorder) pointed at a private temp dir,
and SHIM_SPEND_FILE stays pinned to that temp ledger for every test (an earlier lane's test
overwrote the LIVE ledger by importing the shim without it). The environment is restored
after import so other test modules in the same pytest run see exactly what they saw before.

Run:  python -m pytest -q test_gateway_local_first.py
"""
import asyncio
import atexit
import importlib.util
import json
import os
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, patch

_TMPDIR = tempfile.TemporaryDirectory(prefix="gw-local-first-test-")   # managed: removed at exit
atexit.register(_TMPDIR.cleanup)
_TMP = _TMPDIR.name
_ISOLATED_ENV = {
    "SHIM_SPEND_FILE": os.path.join(_TMP, "gateway-spend.json"),
    "SHIM_SPEND_CLIENTS_FILE": os.path.join(_TMP, "gateway-spend-clients.json"),
    "SHIM_STATS_FILE": os.path.join(_TMP, "gateway-stats.json"),
    "SHIM_TELEMETRY_DIR": os.path.join(_TMP, "telemetry"),
    "SHIM_ENV_FILE": os.path.join(_TMP, "shim.env"),
    "SHIM_ALIASES_FILE": os.path.join(_TMP, "gateway-aliases.json"),
    "SHIM_FLIGHTREC_DIR": os.path.join(_TMP, "flightrec"),
    "SHIM_EXACT_TOKENS": "0",
}
with patch.dict(os.environ, _ISOLATED_ENV):
    SPEC = importlib.util.spec_from_file_location("shim_local_first_test", os.environ.get(
        "SHIM_TEST_CANDIDATE", str(Path(__file__).with_name("keepalive-shim.py"))))
    shim = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(shim)

_LIVE_DIR = "/home/kevin/.local/share/vllm-qwen27b"
for _name in ("SPEND_FILE", "STATS_FILE", "TELEMETRY_DIR", "SHIM_ENV_FILE", "ALIASES_FILE"):
    assert not str(getattr(shim, _name)).startswith(_LIVE_DIR), (_name, getattr(shim, _name))


class Request:
    path = "/v1/chat/completions"
    remote = "127.0.0.1"
    method = "POST"

    def __init__(self, headers=None, **fields):
        self.headers = {"X-Client": "test-interactive", "User-Agent": "offline-test", **(headers or {})}
        self.body = json.dumps({"model": "qwen-local",
                                "messages": [{"role": "user", "content": "review"}],
                                "max_tokens": 1000, **fields}).encode()

    async def read(self):
        return self.body


class Isolated(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, _ISOLATED_ENV))
        self.stack.enter_context(patch.object(shim, "_ADMISSION_WAITS", type(shim._ADMISSION_WAITS)(maxlen=512)))
        self.stack.enter_context(patch.object(shim, "_local_first_kept", type(shim._local_first_kept)()))
        self.stack.enter_context(patch.object(shim, "_local_first_remote", type(shim._local_first_remote)()))
        for name, value in dict(LOCAL_FIRST=True,
                                LOCAL_FIRST_REASONS=frozenset({"big-prompt", "perf", "predicted",
                                                               "big-out", "monster"}),
                                LOCAL_FIRST_QUEUE_WAIT_SECS=5.0, LOCAL_FIRST_WAIT_WINDOW_SECS=60.0,
                                LOCAL_FIRST_FIRST_TOKEN_MAX=300.0,
                                LOCAL_FIRST_INTERACTIVE_TTFT_SECS=0.0,
                                _inflight=0, _inflight_tokens=0, _inflight_reserved_tokens=0,
                                _waiting=0, _waiting_by_class={"interactive": 0, "background": 0},
                                _health={"ok": True}, TOKEN_BUDGET=500_000, FG_RESERVED=2,
                                effective_budget=lambda: 14, PREFILL_TPS=850.0,
                                FT_CONCURRENCY_SCALE=1).items():
            self.stack.enter_context(patch.object(shim, name, value))

    def decide(self, reason="big-prompt", background=False, units=1, reservation=60_000,
               est_computed=2_000, now=None):
        return shim.local_first_decision(reason, background=background, units=units,
                                         reservation=reservation, est_computed=est_computed, now=now)


class Decision(Isolated):
    def test_idle_local_keeps_every_predictive_reason(self):
        for reason in ("big-prompt", "perf", "predicted", "big-out", "monster"):
            self.assertEqual(self.decide(reason), (True, "capacity"), reason)

    def test_rollback_flag_restores_old_routing(self):
        with patch.object(shim, "LOCAL_FIRST", False):
            self.assertEqual(self.decide(), (False, "policy-off"))

    def test_reason_outside_the_set_is_untouched(self):
        with patch.object(shim, "LOCAL_FIRST_REASONS", frozenset({"perf"})):
            self.assertEqual(self.decide("big-prompt"), (False, "policy-off"))
            self.assertEqual(self.decide("perf"), (True, "capacity"))
        for explicit in ("alias", "intent", "forced", "local-down", "failover", "size", "tokens"):
            self.assertEqual(self.decide(explicit)[0], False, explicit)

    def test_lanes_full_is_saturated(self):
        with patch.object(shim, "_inflight", 14):
            self.assertEqual(self.decide(), (False, "saturated:lanes"))
        # background may only use budget - FG_RESERVED lanes
        with patch.object(shim, "_inflight", 12):
            self.assertEqual(self.decide(background=True), (False, "saturated:lanes"))
            self.assertEqual(self.decide(background=False), (True, "capacity"))
        # a multi-unit request needs all its units free
        with patch.object(shim, "_inflight", 10):
            self.assertEqual(self.decide(units=5), (False, "saturated:lanes"))

    def test_token_budget_full_is_saturated(self):
        with patch.object(shim, "_inflight_reserved_tokens", 390_000):
            self.assertEqual(self.decide(reservation=60_000), (False, "saturated:tokens"))
            self.assertEqual(self.decide(reservation=40_000), (True, "capacity"))

    def test_existing_queue_for_the_class_is_saturated(self):
        with patch.object(shim, "_waiting_by_class", {"interactive": 0, "background": 2}):
            self.assertEqual(self.decide(background=True), (False, "saturated:queued"))
            self.assertEqual(self.decide(background=False), (True, "capacity"))

    def test_measured_queue_wait_over_threshold_is_saturated(self):
        now = 1_000_000.0
        for w in (6.0, 7.0, 5.0):
            shim._note_admission_wait(w, now=now - 10)
        self.assertEqual(self.decide(now=now), (False, "saturated:queue-wait"))
        # samples older than the window no longer count
        self.assertEqual(self.decide(now=now + 120), (True, "capacity"))
        # threshold 0 disables the signal
        with patch.object(shim, "LOCAL_FIRST_QUEUE_WAIT_SECS", 0.0):
            self.assertEqual(self.decide(now=now), (True, "capacity"))

    def test_zero_waits_dilute_rather_than_trigger(self):
        now = 2_000_000.0
        shim._note_admission_wait(6.0, now=now - 5)
        for _ in range(9):
            shim._note_admission_wait(0.0, now=now - 5)
        self.assertAlmostEqual(shim.recent_admission_wait(now), 0.6)
        self.assertEqual(self.decide(now=now), (True, "capacity"))

    def test_unhealthy_local_never_kept(self):
        with patch.object(shim, "_health", {"ok": False}):
            self.assertEqual(self.decide(), (False, "local-unhealthy"))

    def test_interactive_ttft_ceiling_is_opt_in_and_interactive_only(self):
        # 85,000 computed tokens / 850 tok/s = 100s predicted TTFT
        self.assertEqual(self.decide(est_computed=85_000), (True, "capacity"))   # default off
        with patch.object(shim, "LOCAL_FIRST_INTERACTIVE_TTFT_SECS", 20.0):
            self.assertEqual(self.decide(est_computed=85_000), (False, "interactive-ttft"))
            self.assertEqual(self.decide(est_computed=85_000, background=True), (True, "capacity"))
            self.assertEqual(self.decide(est_computed=2_000), (True, "capacity"))   # prefix-cached turn


class Helpers(Isolated):
    def test_full_remote_requires_a_live_bounded_lease(self):
        with patch.object(shim, "REMOTE_BASE", "https://example.invalid"), \
                patch.object(shim, "REMOTE_KEY", "test-only"), \
                patch.object(shim, "REMOTE_ENABLED", True), patch.object(shim, "FORCE_REMOTE", 1), \
                patch.object(shim, "FORCE_REMOTE_UNTIL_EPOCH", 0.0), \
                patch.object(shim, "_config_owner", lambda: False), \
                patch.object(shim.time, "time", return_value=1000.0):
            self.assertEqual(shim.routing_mode(), "local_first")
            self.assertEqual(shim.current_config()["force_remote"], 0)
            shim.apply_config({"force_remote": 1})
            self.assertEqual(shim.routing_mode(), "full_remote")
            self.assertEqual(shim.FORCE_REMOTE_UNTIL_EPOCH, 4600.0)
            with patch.object(shim.time, "time", return_value=4601.0):
                self.assertEqual(shim.routing_mode(), "local_first")
                self.assertEqual(shim.current_config()["force_remote"], 0)

    def test_reservation_estimate_mirrors_the_local_output_clamp(self):
        with patch.object(shim, "LOCAL_MAX_OUT", 16384):
            self.assertEqual(shim.local_reservation_estimate(50_000, 0), 66_384)
            self.assertEqual(shim.local_reservation_estimate(50_000, 1_000), 51_000)
            self.assertEqual(shim.local_reservation_estimate(50_000, 65_536), 66_384)

    def test_first_token_cap_widens_only_for_big_local_prompts(self):
        big = json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode()
        with patch.object(shim, "FIRST_TOKEN_MAX", 60.0), patch.object(shim, "FIRST_TOKEN_BASE", 15.0), \
                patch.object(shim, "BIG_PROMPT", 24_000), patch.object(shim, "_est_tokens", lambda b: 100_000):
            self.assertAlmostEqual(shim.first_token_timeout(big, 1, local=True), 15.0 + 100_000 / 850.0)
            self.assertEqual(shim.first_token_timeout(big, 1, local=False), 60.0)
            self.assertEqual(shim.first_token_timeout(big, 1), 60.0)            # default = old behaviour
            self.assertEqual(shim.first_token_timeout(big, 4, local=True), 300.0)
            with patch.object(shim, "LOCAL_FIRST", False):
                self.assertEqual(shim.first_token_timeout(big, 1, local=True), 60.0)
        with patch.object(shim, "FIRST_TOKEN_MAX", 60.0), patch.object(shim, "BIG_PROMPT", 24_000), \
                patch.object(shim, "_est_tokens", lambda b: 1_000):
            self.assertEqual(shim.first_token_timeout(big, 1, local=True), shim.first_token_timeout(big, 1))

    def test_knobs_round_trip_through_the_dashboard_config(self):
        cfg = shim.current_config()
        self.assertEqual(cfg["local_first"], 1)
        self.assertEqual(set(cfg["local_first_reasons"].split(",")),
                         {"big-prompt", "perf", "predicted", "big-out", "monster"})
        for field in ("local_first_queue_wait_secs", "local_first_wait_window_secs",
                      "local_first_first_token_max", "local_first_interactive_ttft_secs"):
            self.assertIn(field, cfg)
        with patch.object(shim, "_config_owner", lambda: False):
            changed = shim.apply_config({"local_first": "0", "local_first_reasons": "perf|big-prompt"})
        self.assertEqual(sorted(changed), ["local_first", "local_first_reasons"])
        self.assertIs(shim.LOCAL_FIRST, False)
        self.assertEqual(shim.LOCAL_FIRST_REASONS, frozenset({"perf", "big-prompt"}))
        self.assertEqual(shim._parse_reason_set(shim._fmt_seq(frozenset({"a", "b"}), ",")),
                         frozenset({"a", "b"}))

    def test_stalled_outcome_brakes_automatic_overflow_but_producing_allows_it(self):
        alarm = Path(_TMP) / "outcome-alarm.json"
        with patch.object(shim, "OUTCOME_ALARM_PATH", str(alarm)), \
                patch.object(shim, "STALLED_AUTO_OVERFLOW_CAP_USD", 2.0), \
                patch.object(shim, "_spend", lambda: type("Ledger", (), {
                    "snapshot": lambda self: {"spent": 1.8, "held": 0.1, "reserved": 0}})()), \
                patch.object(shim, "_spend_hold_estimate", lambda *_: 0.2):
            self.assertFalse(shim._automatic_remote_budget_allows(1000, 1000))
            alarm.write_text(json.dumps({"status": "stalled"}))
            self.assertFalse(shim._automatic_remote_budget_allows(1000, 1000))
            alarm.write_text(json.dumps({"status": "producing"}))
            self.assertTrue(shim._automatic_remote_budget_allows(1000, 1000))
            alarm.unlink()


class Routing(Isolated, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp()
        self.events, self.calls, self.ptok = [], [], 60_000

        async def relay(request, base, path, body, key, streaming, *a, **k):
            self.calls.append("local" if base == shim.LOCAL else "remote")
            return "ok", "local"

        async def forward_remote(request, path, body, streaming, endpoint=None):
            self.calls.append("remote")
            return "remote"

        for name, value in dict(
                _relay=relay, _forward_remote=forward_remote,
                record_event=lambda d, r, *a, **k: self.events.append((d, r)),
                _est_tokens=lambda body: self.ptok, estimate_units=lambda *a, **k: 1,
                predict_computed_tokens=lambda c, b, p: 2_000, _prefix_cache_observe=lambda *a: None,
                remote_ok=lambda: True, _spend_allows_overflow=lambda p, m: True,
                _automatic_remote_budget_allows=lambda p, m: True,
                local_healthy=AsyncMock(return_value=True), is_peak=lambda: False,
                _active_set=lambda *a, **k: None, _note_payload_outcome=lambda *a, **k: None,
                _write_flightrec=lambda *a, **k: None, perf_breaker_active=lambda: False,
                predicted_occupancy_seconds=lambda *a, **k: None,
                TOKEN_BUDGET=500_000, LOCAL_MAX_OUT=16384, DEFAULT_MAX_OUT=8192,
                MAX_LOCAL_TOKENS=524_288, LOCAL_CONTEXT_LIMIT=524_288, FORCE_REMOTE=0,
                BIG_OUTPUT=16384, BIG_PROMPT=24_000, MONSTER_INFLIGHT=120_000,
                FOREIGN_LOAD_GUARD=0, TINY_TOKENS=0, LOCAL_WAIT=0, BG_WAIT=0, BG_LOCAL_ONLY=0,
                BG_BIG_LOCAL_WHEN_IDLE=0, LOG_REQUESTS=0, CRASH_ADAPTIVE=0, EMPTY_RETRY=0,
                LOCAL_ONLY=0, INTERACTIVE_NEVER_OVERFLOW=0).items():   # live env value
            self.stack.enter_context(patch.object(shim, name, value))

    async def route(self, **kw):
        self.events.clear(); self.calls.clear()
        return await shim._route_completions(Request(**kw))

    async def test_full_local_refuses_explicit_remote_aliases_without_forwarding(self):
        with patch.object(shim, "LOCAL_ONLY", 1):
            response = await self.route(model="estate-remote")
            self.assertEqual(response.status, 503)
            self.assertEqual(json.loads(response.body)["error"]["type"], "full_local_remote_disabled")
            self.assertEqual(self.calls, [])
            self.assertEqual(self.events, [("rejected", "full-local-remote-alias")])
            with patch.object(shim, "_alias_for_request", lambda _body: {
                    "name": "custom", "kind": "custom-remote", "endpoint": {"base": "https://example.invalid"}}):
                response = await self.route(model="custom")
            self.assertEqual(response.status, 503)
            self.assertEqual(self.calls, [])

    async def test_big_prompt_with_free_capacity_is_served_locally(self):
        self.assertEqual(await self.route(), "local")
        self.assertEqual(self.calls, ["local"])
        self.assertEqual(self.events, [("local", "lf-big-prompt")])
        self.assertEqual(shim._local_first_kept["big-prompt"], 1)
        self.assertEqual(len(shim._ADMISSION_WAITS), 1)            # the queue-wait signal is fed

    async def test_big_prompt_goes_remote_when_local_is_saturated(self):
        with patch.object(shim, "_inflight", 14):
            self.assertEqual(await self.route(), "remote")
        self.assertEqual(self.events, [("remote", "big-prompt")])
        self.assertEqual(shim._local_first_remote["big-prompt:saturated:lanes"], 1)
        with patch.object(shim, "_inflight_reserved_tokens", 480_000):
            self.assertEqual(await self.route(), "remote")
        self.assertEqual(self.events, [("remote", "big-prompt")])

    async def test_rollback_flag_restores_big_prompt_overflow(self):
        with patch.object(shim, "LOCAL_FIRST", False):
            self.assertEqual(await self.route(), "remote")
        self.assertEqual(self.events, [("remote", "big-prompt")])
        self.assertEqual(dict(shim._local_first_remote), {})

    async def test_perf_breaker_with_capacity_stays_local_and_saturated_goes_remote(self):
        self.ptok = 3_000
        with patch.object(shim, "perf_breaker_active", lambda: True):
            self.assertEqual(await self.route(max_tokens=1200), "local")
            self.assertEqual(self.events, [("local", "lf-perf")])
            with patch.object(shim, "_waiting_by_class", {"interactive": 3, "background": 0}):
                self.assertEqual(await self.route(max_tokens=1200), "remote")
            self.assertEqual(self.events, [("remote", "perf")])
            with patch.object(shim, "LOCAL_FIRST", False):
                self.assertEqual(await self.route(max_tokens=1200), "remote")
            self.assertEqual(self.events, [("remote", "perf")])

    async def test_predicted_big_out_and_monster_follow_the_same_rule(self):
        self.ptok = 3_000
        with patch.object(shim, "predicted_occupancy_seconds", lambda *a, **k: 999.0):
            self.assertEqual(await self.route(), "local")
            self.assertEqual(self.events, [("local", "lf-predicted")])
        self.assertEqual(await self.route(max_tokens=65_536), "local")
        self.assertEqual(self.events, [("local", "lf-big-out")])
        with patch.object(shim, "_inflight_tokens", 130_000):
            self.assertEqual(await self.route(), "local")
            self.assertEqual(self.events, [("local", "lf-monster")])
            with patch.object(shim, "LOCAL_FIRST_REASONS", frozenset({"big-prompt", "perf"})):
                self.assertEqual(await self.route(), "remote")
            self.assertEqual(self.events, [("remote", "monster")])

    async def test_explicit_remote_and_safety_routes_are_unchanged(self):
        self.ptok = 3_000
        self.assertEqual(await self.route(model="estate-remote"), "remote")
        self.assertEqual(self.events, [("remote", "alias")])
        self.assertEqual(await self.route(headers={"X-Gateway-Route-Intent": "remote"}), "remote")
        self.assertEqual(self.events, [("remote", "intent")])
        with patch.object(shim, "FORCE_REMOTE", 1), \
                patch.object(shim, "FORCE_REMOTE_UNTIL_EPOCH", time.time() + 60):
            self.assertEqual(await self.route(), "remote")
        self.assertEqual(self.events, [("remote", "forced")])
        with patch.object(shim, "local_healthy", AsyncMock(return_value=False)), \
                patch.object(shim, "_health", {"ok": False}):
            self.assertEqual(await self.route(), "remote")
        self.assertEqual(self.events, [("remote", "local-down")])
        self.assertEqual(dict(shim._local_first_kept), {})

    async def test_stalled_outcome_keeps_automatic_overflow_and_route_intent_local(self):
        with patch.object(shim, "_automatic_remote_budget_allows", lambda *a: False), \
                patch.object(shim, "LOCAL_FIRST", False):
            self.assertEqual(await self.route(), "local")
            self.assertEqual(await self.route(headers={"X-Gateway-Route-Intent": "remote"}), "local")
            self.assertEqual(await self.route(model="estate-remote"), "remote")

    async def test_kept_request_still_overflows_as_cap_if_lanes_fill_during_admission(self):
        # decision saw a free lane; admission (the saturation fallback) finds the lanes full
        with patch.object(shim, "admission_lane_limit", lambda *a, **k: 0):
            with patch.object(shim, "local_saturation", lambda *a, **k: []):
                self.assertEqual(await self.route(), "remote")
        self.assertEqual(self.events, [("remote", "cap")])

    async def test_estate_local_never_overflows_when_admission_fills(self):
        with patch.object(shim, "admission_lane_limit", lambda *a, **k: 0), \
                patch.object(shim, "local_saturation", lambda *a, **k: []):
            response = await self.route(model="estate-local")
        self.assertEqual(response.status, 503)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.events, [("held", "cap")])

    async def test_halo_control_uses_protected_place_when_ordinary_budget_is_full(self):
        with patch.object(shim, "_inflight", 14), \
                patch.object(shim, "TINY_EXTRA_LANES", 1), \
                patch.object(Request, "remote", "10.0.1.95"):
            response = await self.route(model="estate-local")
            self.assertEqual(response.status, 503)
            self.assertEqual(self.calls, [])
            result = await self.route(model="estate-local",
                                      headers={"X-Client": "halo-hermes"})
            self.assertEqual(result, "local")
            self.assertEqual(self.calls, ["local"])
            self.assertEqual(shim._inflight, 14)

    async def test_estate_local_never_failovers_after_a_local_error(self):
        async def failed_local(*args, **kwargs):
            self.calls.append("local")
            return "error", (500, "failed", False)

        self.ptok = 3_000
        with patch.object(shim, "_relay", failed_local):
            response = await self.route(model="estate-local")
        self.assertEqual(response.status, 503)
        self.assertEqual(self.calls, ["local"])
        self.assertEqual(self.events, [("held", "local-failed")])

    async def test_small_prompt_routing_is_byte_for_byte_unchanged(self):
        self.ptok = 3_000
        self.assertEqual(await self.route(), "local")
        self.assertEqual(self.events, [("local", "-")])
        self.assertEqual(dict(shim._local_first_kept), {})


if __name__ == "__main__":
    unittest.main()
