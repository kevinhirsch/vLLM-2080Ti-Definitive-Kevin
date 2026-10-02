#!/usr/bin/env python3
"""Cache-aware prefill cost model (keepalive-shim, LS lane, 2026-10-01).

Halo resends a 40-120K-token prompt on every tool-loop turn and ~95% of it is the previous turn's
prompt, which the engine's prefix cache already holds IF that prompt was prefilled locally. The old
predictor was one-deep per client and learned from requests that went remote; this model is
content-addressed, global, learns only from requests admitted to the local engine, expires on a TTL,
clears on engine restart and unlearns prefixes the engine reports it did not have.

ISOLATION: same as test_gateway_local_first.py -- every on-disk path the shim can write is pointed
at a private temp dir before import.

Run:  python -m pytest -q test_gateway_cache_model.py
"""
import asyncio
import atexit
import importlib.util
import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, patch

_TMPDIR = tempfile.TemporaryDirectory(prefix="gw-cache-model-test-")
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
    SPEC = importlib.util.spec_from_file_location("shim_cache_model_test", os.environ.get(
        "SHIM_TEST_CANDIDATE", str(Path(__file__).with_name("keepalive-shim.py"))))
    shim = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(shim)

TOOLS = [{"type": "function", "function": {"name": f"t{i}", "description": "d" * 200}} for i in range(40)]


def convo(tag, turns, tools=TOOLS, system="You are Halo. " + "s" * 4000, filler=2000, **extra):
    """A conversation of `turns` user/assistant pairs, each message `filler` chars."""
    msgs = [{"role": "system", "content": system}]
    for i in range(turns):
        msgs.append({"role": "user", "content": f"{tag} user {i} " + "u" * filler})
        msgs.append({"role": "assistant", "content": f"{tag} asst {i} " + "a" * filler})
    body = {"model": "estate", "messages": msgs, "stream": True, **extra}
    if tools:
        body["tools"] = tools
    return json.dumps(body).encode()


class Request:
    path = "/v1/chat/completions"
    remote = "127.0.0.1"
    method = "POST"

    def __init__(self, body, headers=None):
        self.headers = {"X-Client": "test-interactive", "User-Agent": "offline-test", **(headers or {})}
        self.body = body

    async def read(self):
        return self.body


class Base(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, _ISOLATED_ENV))
        shim._PM_NODES.clear()
        shim._PM_PAIRS.clear()
        shim._PM_STATS.clear()
        shim._PM_INFLIGHT.clear()
        for name, value in dict(PREFIX_ALIGN_TOKENS=3568, PREFIX_MODEL_TTL_SECS=900.0,
                                PREFIX_HIT_MARGIN_TOKENS=512, PREFIX_MODEL_MAX_NODES=60000,
                                USE_COMPUTED_COST=0, PREFILL_TPS=1000.0).items():
            self.stack.enter_context(patch.object(shim, name, value))

    def predict(self, body, est=None):
        est = est if est is not None else len(body) // 4
        return shim._pm_predict(body, est)

    def serve_local(self, body, est=None):
        p = self.predict(body, est)
        shim._pm_commit(p["chain"])
        shim._pm_ready(p["chain"])             # first token produced -> the prefix is in the engine cache
        return p


class Model(Base):
    def test_cold_request_costs_its_full_size(self):
        p = self.predict(convo("A", 10))
        self.assertEqual(p["credit"], 0)
        self.assertEqual(p["computed"], p["est"])
        self.assertEqual(p["best"], -1)

    def test_next_turn_of_a_locally_served_conversation_is_cheap_and_block_aligned(self):
        shim_est = lambda b: len(b) // 4
        first = convo("A", 20)
        self.serve_local(first, shim_est(first))
        nxt = convo("A", 21)
        p = self.predict(nxt, shim_est(nxt))
        self.assertGreater(p["credit"], 0)
        self.assertEqual(p["credit"] % 3568, 0)                    # whole attention blocks only
        self.assertLess(p["computed"], p["est"] * 0.2)             # ~95% of the prompt is cached
        self.assertEqual(p["best"], len(json.loads(first)["messages"]) - 1)

    def test_interleaved_conversations_do_not_evict_each_other(self):
        """The old one-deep per-client predictor lost A the moment B arrived."""
        a1, b1, c1 = convo("A", 20), convo("B", 20), convo("C", 20)
        for b in (a1, b1, c1):
            self.serve_local(b)
        for tag in ("A", "B", "C"):
            p = self.predict(convo(tag, 21))
            self.assertGreater(p["credit"], 0, tag)

    def test_a_prefix_still_being_prefilled_is_not_yet_a_hit(self):
        p = self.predict(convo("A", 20))
        shim._pm_commit(p["chain"])                                # claimed, no first token yet
        self.assertEqual(self.predict(convo("A", 21))["credit"], 0)
        shim._pm_ready(p["chain"])
        self.assertGreater(self.predict(convo("A", 21))["credit"], 0)

    def test_requests_served_remote_do_not_warm_the_local_cache(self):
        self.predict(convo("A", 20))                               # routed remote: never committed
        self.assertEqual(self.predict(convo("A", 21))["credit"], 0)

    def test_shared_system_and_tools_prefix_is_credited_to_a_new_conversation(self):
        self.serve_local(convo("A", 20))
        p = self.predict(convo("Z", 3))                            # brand-new conversation, same system+tools
        self.assertEqual(p["best"], 0)                             # only the system message is shared
        # tools+system are ~11K chars: well under one 3568-token block of a short prompt -> no credit,
        # but with a bigger shared head the credit is real:
        big_sys = "You are Halo. " + "s" * 60000
        self.serve_local(convo("A", 2, system=big_sys))
        p = self.predict(convo("Z", 2, system=big_sys))
        self.assertGreater(p["credit"], 0)

    def test_different_tools_or_thinking_mode_share_nothing(self):
        self.serve_local(convo("A", 20))
        self.assertEqual(self.predict(convo("A", 21, tools=TOOLS[:-1]))["best"], -1)
        self.assertEqual(self.predict(convo("A", 21, chat_template_kwargs={"enable_thinking": True}))["best"], -1)
        self.assertEqual(self.predict(convo("A", 21, reasoning_effort="low"))["best"], -1)

    def test_edited_history_only_credits_the_common_prefix(self):
        base = json.loads(convo("A", 20))
        self.serve_local(json.dumps(base).encode())
        base["messages"][10]["content"] = "EDITED"
        p = self.predict(json.dumps(base).encode())
        self.assertEqual(p["best"], 9)

    def test_ttl_expiry(self):
        b = convo("A", 20)
        p = self.predict(b)
        shim._pm_commit(p["chain"], now=1000.0)
        shim._pm_ready(p["chain"], now=1000.0)
        self.assertGreater(shim._pm_predict(convo("A", 21), 20000, now=1000.0 + 899)["credit"], 0)
        self.assertEqual(shim._pm_predict(convo("A", 21), 20000, now=1000.0 + 901)["credit"], 0)

    def test_engine_restart_clears_the_model(self):
        self.serve_local(convo("A", 20))
        shim._pm_reset("test")
        self.assertEqual(self.predict(convo("A", 21))["credit"], 0)

    def test_lru_eviction_never_cuts_the_middle_out_of_a_chain(self):
        with patch.object(shim, "PREFIX_MODEL_MAX_NODES", 1000):
            for i in range(60):
                self.serve_local(convo(f"X{i}", 20))                # 41 nodes each -> evictions
            # the freshest conversation still resolves end to end
            last = json.loads(convo("X59", 20))
            self.assertEqual(self.predict(convo("X59", 21))["best"], len(last["messages"]) - 1)
            self.assertLessEqual(len(shim._PM_NODES), 1000)

    def test_overprediction_unlearns_the_nodes_the_engine_did_not_have(self):
        a1 = convo("A", 20)
        self.serve_local(a1)
        nxt = convo("A", 21)
        p = self.predict(nxt, 60000)
        shim._PM_INFLIGHT[1] = p
        info = {"pm_ref": 1, "pm_credit": p["credit"], "cached_actual": 0, "route": "local",
                "est_tokens": 60000, "ptok_exact_local": 60000}
        shim._pm_feedback(info)
        self.assertEqual(shim._PM_STATS["overpredict"], 1)
        self.assertEqual(self.predict(nxt, 60000)["credit"], 0)     # the next turn is no longer trusted
        self.assertNotIn(1, shim._PM_INFLIGHT)

    def test_accurate_and_underpredicted_are_counted_and_summarised(self):
        for credit, cached in ((35680, 35680), (0, 14272), (35680, 32112)):
            shim._PM_INFLIGHT[2] = {"chain": [], "best": -1, "total": 0}
            shim._pm_feedback({"pm_ref": 2, "pm_credit": credit, "cached_actual": cached, "route": "local",
                               "est_tokens": 60000, "ptok_exact_local": 60000})
        s = shim._pm_summary()
        self.assertEqual((s["accurate"], s["underpredict"], s["graded"]), (2, 1, 3))
        self.assertIn("abs_err_p90", s)

    def test_trust_shaves_credit_when_the_engine_keeps_delivering_less(self):
        a1 = convo("A", 20)
        self.serve_local(a1)
        full = self.predict(convo("A", 21), 60000)["credit"]
        for _ in range(10):                      # engine delivered only half of what was predicted
            shim._PM_PAIRS.append((35680, 17840, 60000))
        self.assertAlmostEqual(shim._pm_trust(), 0.5, places=2)
        self.assertLess(self.predict(convo("A", 21), 60000)["credit"], full)
        shim._PM_PAIRS.clear()
        self.assertEqual(shim._pm_trust(), 1.0)  # no history -> neutral

    def test_remote_or_ungraded_requests_are_not_graded(self):
        shim._pm_feedback({"pm_ref": 3, "pm_credit": 100, "cached_actual": 0, "route": "remote"})
        shim._pm_feedback({"pm_ref": 3, "pm_credit": 100, "cached_actual": None, "route": "local"})
        self.assertEqual(shim._pm_summary()["graded"], 0)

    def test_unparsable_body_is_cold_not_a_crash(self):
        p = shim._pm_predict(b"not json", 5000)
        self.assertEqual((p["computed"], p["credit"], p["chain"]), (5000, 0, []))

    def test_legacy_observe_hook_is_inert(self):
        self.assertIsNone(shim._prefix_cache_observe("c", convo("A", 5), 1000))
        self.assertEqual(len(shim._PM_NODES), 0)


class Units(Base):
    def test_units_follow_predicted_computed_only_when_enabled(self):
        b = convo("A", 20)
        with patch.object(shim, "_est_tokens", lambda body: 58_000):
            self.assertEqual(shim.estimate_units(b, budget=14, client="halo", computed=4_000), 5)   # legacy: raw size
            with patch.object(shim, "USE_COMPUTED_COST", 1):
                self.assertEqual(shim.estimate_units(b, budget=14, client="halo", computed=4_000), 1)
                self.assertEqual(shim.estimate_units(b, budget=14, client="halo", computed=58_000), 5)

    def test_backlog_seconds_use_the_measured_rate(self):
        with patch.object(shim, "_inflight_computed", 30_000), patch.object(shim, "PREFILL_TPS", 1000.0):
            self.assertAlmostEqual(shim._prefill_backlog_secs(), 30.0)


class Decision(Base):
    def setUp(self):
        super().setUp()
        for name, value in dict(LOCAL_FIRST=True,
                                LOCAL_FIRST_REASONS=frozenset({"big-prompt", "perf", "predicted", "big-out", "monster"}),
                                LOCAL_FIRST_QUEUE_WAIT_SECS=5.0, LOCAL_FIRST_WAIT_WINDOW_SECS=60.0,
                                LOCAL_FIRST_INTERACTIVE_TTFT_SECS=0.0, _inflight=0, _inflight_computed=0,
                                _inflight_reserved_tokens=0, _waiting_by_class={"interactive": 0, "background": 0},
                                _health={"ok": True}, TOKEN_BUDGET=500_000, FG_RESERVED=2,
                                effective_budget=lambda: 14, HEAVY_PREFILL_SECS=20.0,
                                HEAVY_ADMIT_BACKLOG_SECS=15.0).items():
            self.stack.enter_context(patch.object(shim, name, value))
        shim._ADMISSION_WAITS.clear()

    def decide(self, reason, est_computed, **kw):
        return shim.local_first_decision(reason, background=False, units=1, reservation=60_000,
                                         est_computed=est_computed, **kw)

    def test_legacy_mode_is_unchanged(self):
        self.assertEqual(self.decide("monster", 80_000), (True, "capacity"))
        self.assertEqual(self.decide("big-prompt", 80_000), (True, "capacity"))

    def test_monster_never_stays_local_in_cache_aware_mode(self):
        with patch.object(shim, "USE_COMPUTED_COST", 1):
            self.assertEqual(self.decide("monster", 1_000), (False, "prefill-backlog"))

    def test_heavy_cold_prefill_needs_a_short_backlog(self):
        with patch.object(shim, "USE_COMPUTED_COST", 1):
            self.assertEqual(self.decide("big-prompt", 30_000), (True, "capacity"))          # engine idle
            with patch.object(shim, "_inflight_computed", 16_000):                            # 16s ahead
                self.assertEqual(self.decide("big-prompt", 30_000), (False, "saturated:prefill-backlog"))
                self.assertEqual(self.decide("big-prompt", 8_000), (True, "capacity"))        # light: unaffected
            with patch.object(shim, "_inflight_computed", 14_000):
                self.assertEqual(self.decide("big-prompt", 30_000), (True, "capacity"))


class Routing(Base, unittest.IsolatedAsyncioTestCase):
    """A warm 58K Halo turn is a normal local request; a cold one waits behind a monster."""

    def setUp(self):
        super().setUp()
        self.events, self.calls = [], []

        async def relay(request, base, path, body, key, streaming, *a, **k):
            self.calls.append("local" if base == shim.LOCAL else "remote")
            if base == shim.LOCAL:               # the real relay marks the prefix cached at first token
                shim._pm_ready((shim._PM_INFLIGHT.get(id(request)) or {}).get("chain") or [])
            return "ok", "local"

        async def forward_remote(request, path, body, streaming, endpoint=None):
            self.calls.append("remote")
            return "remote"

        for name, value in dict(
                _relay=relay, _forward_remote=forward_remote,
                record_event=lambda d, r, *a, **k: self.events.append((d, r)),
                _est_tokens=lambda body: 58_000,
                remote_ok=lambda: True, _spend_allows_overflow=lambda p, m: True,
                _automatic_remote_budget_allows=lambda p, m: True,
                local_healthy=AsyncMock(return_value=True), is_peak=lambda: False,
                _active_set=lambda *a, **k: None, _note_payload_outcome=lambda *a, **k: None,
                _write_flightrec=lambda *a, **k: None, perf_breaker_active=lambda: False,
                predicted_occupancy_seconds=lambda *a, **k: None,
                LOCAL_FIRST=True,
                LOCAL_FIRST_REASONS=frozenset({"big-prompt", "perf", "predicted", "big-out", "monster"}),
                LOCAL_FIRST_QUEUE_WAIT_SECS=5.0, LOCAL_FIRST_WAIT_WINDOW_SECS=60.0,
                LOCAL_FIRST_FIRST_TOKEN_MAX=300.0, LOCAL_FIRST_INTERACTIVE_TTFT_SECS=0.0,
                _inflight=0, _inflight_tokens=0, _inflight_reserved_tokens=0, _inflight_computed=0,
                _waiting=0, _waiting_by_class={"interactive": 0, "background": 0},
                _health={"ok": True}, TOKEN_BUDGET=500_000, FG_RESERVED=2, effective_budget=lambda: 14,
                LOCAL_MAX_OUT=16384, DEFAULT_MAX_OUT=8192, MAX_LOCAL_TOKENS=524_288,
                LOCAL_CONTEXT_LIMIT=524_288, FORCE_REMOTE=0, BIG_OUTPUT=16384, BIG_PROMPT=24_000,
                BIG_TOKENS=40_000, TOKENS_PER_UNIT=12_000, MONSTER_INFLIGHT=120_000,
                MONSTER_PREFILL_SECS=30.0, HEAVY_PREFILL_SECS=20.0, HEAVY_ADMIT_BACKLOG_SECS=15.0,
                FOREIGN_LOAD_GUARD=0, TINY_TOKENS=0, LOCAL_WAIT=0, BG_WAIT=0, BG_LOCAL_ONLY=0,
                BG_BIG_LOCAL_WHEN_IDLE=0, LOG_REQUESTS=0, CRASH_ADAPTIVE=0, EMPTY_RETRY=0,
                LOCAL_ONLY=0, INTERACTIVE_NEVER_OVERFLOW=0).items():
            self.stack.enter_context(patch.object(shim, name, value))
        shim._ADMISSION_WAITS.clear()

    async def route(self, body):
        self.events.clear(); self.calls.clear()
        return await shim._route_completions(Request(body))

    async def test_warm_turn_is_an_ordinary_local_request_even_with_lanes_mostly_full(self):
        """9 of 14 lanes held: legacy costing (58K = 5 units) overflows it; computed costing is 1 unit."""
        first, nxt = convo("A", 20), convo("A", 21)
        with patch.object(shim, "USE_COMPUTED_COST", 1):
            await self.route(first)                                  # cold: goes local (idle engine), commits
            self.assertEqual(self.events, [("local", "lf-big-prompt")])
            with patch.object(shim, "_inflight", 9):
                self.assertEqual(await self.route(nxt), "local")
                self.assertEqual(self.events, [("local", "-")])     # no big-prompt reason at all
        shim._PM_NODES.clear()
        with patch.object(shim, "_inflight", 9):                      # legacy mode, same lanes
            self.assertEqual(await self.route(nxt), "remote")
            self.assertEqual(self.events, [("remote", "big-prompt")])

    async def test_cold_heavy_prefill_waits_out_a_monster_but_light_work_does_not_queue_behind_it(self):
        cold = convo("B", 20)
        with patch.object(shim, "USE_COMPUTED_COST", 1), patch.object(shim, "_inflight_computed", 40_000):
            self.assertEqual(await self.route(cold), "remote")        # 40s of prefill already ahead
            self.assertEqual(self.events, [("remote", "big-prompt")])
            light = json.dumps({"model": "estate", "messages": [{"role": "user", "content": "hi"}],
                                "max_tokens": 200}).encode()
            with patch.object(shim, "_est_tokens", lambda b: 2_000):
                self.assertEqual(await self.route(light), "remote")   # arrivals during a monster go remote
                self.assertEqual(self.events, [("remote", "monster")])

    async def test_remote_served_turn_does_not_pretend_to_warm_local(self):
        cold = convo("C", 20)
        with patch.object(shim, "USE_COMPUTED_COST", 1), patch.object(shim, "_inflight_computed", 40_000):
            await self.route(cold)                                    # monster -> remote, not committed
        self.assertEqual(len(shim._PM_NODES), 0)

    async def test_inflight_computed_is_claimed_and_released(self):
        seen = []
        orig = shim._relay

        async def spying_relay(*a, **k):
            seen.append(shim._inflight_computed)
            return await orig(*a, **k)

        with patch.object(shim, "USE_COMPUTED_COST", 1), patch.object(shim, "_relay", spying_relay):
            await self.route(convo("D", 20))
        self.assertEqual(seen, [58_000])
        self.assertEqual(shim._inflight_computed, 0)


if __name__ == "__main__":
    unittest.main()
