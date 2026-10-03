#!/usr/bin/env python3
"""Lane DB2 (2026-10-03): the gateway-side data the dashboard reads.

1. the work class is recorded on EVERY route (remote, alias, vision, forced, local-offline, failover), not only on the
   local admission path, so /gateway/capacity remote_use.by_class is real;
2. /gateway/lanes can be asked for a light form (?active=1&limit=N) with a summary, and still returns the full list when
   asked with no parameters;
3. windowed TTFT / inter-token quantiles are computed from the gateway's own per-request telemetry;
4. the on-disk history summary carries the remote reasons and the work class, and every documented overflow reason is
   still emitted by the shim.

Run:  python -m pytest -q test_gateway_dashboard_feeds.py
"""
import json
import pathlib
import re
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

import test_gateway_local_first as lf

shim = lf.shim
HERE = pathlib.Path(__file__).resolve().parent
REAL_ACTIVE_SET = shim._active_set                       # captured before any fixture patches it
IMG = [{"role": "user", "content": [{"type": "text", "text": "what is this"},
                                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]


class ClassOnEveryRoute(lf.RoutingBase):
    """A work class lands on the live request before any routing decision, so flow_note_route() sees it on each path."""

    def setUp(self):
        super().setUp()
        self.seen = []                                   # (decision, reason, class on the live request)
        self.routes = shim.collections.deque(maxlen=100)
        self.stack.enter_context(patch.object(shim, "_FLOW_ROUTES", self.routes))
        self.stack.enter_context(patch.object(shim, "_active_set", REAL_ACTIVE_SET))

        def record(decision, reason, request, *a, **k):
            self.events.append((decision, reason))
            self.seen.append((decision, reason, (shim._ACTIVE.get(id(request)) or {}).get("flow_class")))
            shim.flow_note_route(decision, reason, request)

        self.stack.enter_context(patch.object(shim, "record_event", record))

    async def route(self, **kw):
        self.events.clear(); self.calls.clear(); self.seen.clear()
        req = lf.Request(**kw)
        shim._ACTIVE[id(req)] = {"t0": time.time()}
        try:
            return await shim._route_completions(req)
        finally:
            shim._ACTIVE.pop(id(req), None)

    def last_class(self):
        return self.routes[-1][3]

    async def test_alias_route_carries_the_class(self):
        self.assertEqual(await self.route(model="estate-remote", headers={"X-Work-Class": "halo"}), "remote")
        self.assertEqual(self.seen[-1], ("remote", "alias", "halo"))
        self.assertEqual(self.last_class(), "halo")

    async def test_forced_route_carries_the_class(self):
        with patch.object(shim, "effective_force_remote", lambda: True):
            await self.route(headers={"X-Client": "card-repair-engine"})
        self.assertEqual(self.seen[-1], ("remote", "forced", "runner"))
        self.assertEqual(self.last_class(), "runner")

    async def test_vision_route_carries_the_class(self):
        with patch.object(shim, "LOCAL_MODALITIES", "text"), patch.object(shim, "REMOTE_VISION", 1):
            await self.route(messages=IMG, headers={"X-Client": "overseer-refine-overflow"})
        self.assertEqual(self.seen[-1], ("remote", "vision", "background"))
        self.assertEqual(self.last_class(), "background")

    async def test_local_offline_route_carries_the_class(self):
        with patch.dict(shim._OFFLINE, dict(until=time.time() + 100, lease="L", reason="bench", by="DB2", t0=time.time(),
                                            ttl_s=100, refused=0)), patch.object(shim, "REMOTE_ENABLED", True):
            await self.route(headers={"X-Client": "halo-hermes"})
        self.assertEqual(self.seen[-1], ("remote", "local-offline", "halo"))
        self.assertEqual(self.last_class(), "halo")

    async def test_failover_route_carries_the_class(self):
        async def failing_relay(request, base, path, body, key, streaming, *a, **k):
            return "err", (500, "engine blew up", False)

        with patch.object(shim, "_relay", failing_relay), patch.object(shim, "_inflight", 0):
            self.ptok = 3_000
            await self.route(headers={"X-Work-Class": "kevin"})
        self.assertEqual(self.seen[-1], ("remote", "failover", "kevin"))
        self.assertEqual(self.last_class(), "kevin")

    async def test_size_overflow_carries_the_class(self):
        with patch.object(shim, "over_local_cap", lambda body: True):
            await self.route(headers={"X-Client": "work-verifier"})
        self.assertEqual(self.seen[-1], ("remote", "size", "runner"))

    async def test_rejected_background_still_has_its_class(self):
        with patch.object(shim, "FORCE_REMOTE", 1), patch.object(shim, "effective_force_remote", lambda: True), \
                patch.object(shim, "BG_LOCAL_ONLY", 1), patch.object(shim, "is_background", lambda body, req: True):
            await self.route(headers={"X-Client": "vault-dreams"})
        self.assertEqual(self.seen[-1][:2], ("rejected-bg", "force-remote"))
        self.assertEqual(self.seen[-1][2], "background")

    async def test_no_remote_request_is_untagged_in_remote_use(self):
        with patch.object(shim, "effective_force_remote", lambda: True):
            for who in ("halo-hermes", "card-repair-engine", "overseer-refine-overflow", "some-new-harness"):
                await self.route(headers={"X-Client": who})
        by_class = shim.flow_remote_use()["windows"]["15m"]["by_class"]
        self.assertNotIn("?", by_class)
        self.assertEqual(by_class, {"halo": 1, "runner": 1, "background": 1, "kevin": 1})


class RemoteUseWindows(unittest.TestCase):
    def test_every_reason_is_itemised_and_coverage_is_reported(self):
        now = time.time()
        reasons = ["r%d" % i for i in range(12)]
        with patch.object(shim, "_FLOW_ROUTES", shim.collections.deque(
                [(now - 5, "remote", r, "kevin", False) for r in reasons], maxlen=100)):
            w = shim.flow_remote_use(now)["windows"]["15m"]
        self.assertEqual(sorted(w["by_reason"]), sorted(reasons))              # no longer cut to the top 8
        self.assertLessEqual(w["covered_s"], 900)
        self.assertGreaterEqual(w["covered_s"], 0)

    def test_person_events_and_clients_carry_the_class(self):
        info = {"name": "halo-hermes", "route": "local", "ttft": 1.0, "t0": time.time() - 3, "flow_class": "halo"}
        before = dict(shim._PER_CLIENT["halo-hermes"]["classes"]) if "halo-hermes" in shim._PER_CLIENT else {}
        shim._telemetry_note_request(info, None)
        c = shim._PER_CLIENT["halo-hermes"]["classes"]
        self.assertEqual(c["halo"], before.get("halo", 0) + 1)


class HistorySummary(unittest.TestCase):
    def test_remote_reasons_and_classes_come_from_the_request_log(self):
        now = time.time()
        rows = [
            {"t": now, "client": "a", "route": "remote", "reason": "big-prompt", "flow_class": "halo", "status": 200},
            {"t": now, "client": "a", "route": "remote", "reason": "big-prompt", "flow_class": "halo", "status": 200},
            {"t": now, "client": "b", "route": "remote", "reason": "local-offline", "flow_class": "background", "status": 200},
            {"t": now, "client": "b", "route": "remote", "reason": "perf", "status": 200},          # older row: no class
            {"t": now, "client": "c", "route": "local", "reason": "-", "flow_class": "kevin", "status": 200},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "requests.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with patch.object(shim, "_history_day_file", lambda _e: str(path)):
                got = shim._history_summary_blocking(now - 60, 1)
        self.assertEqual(got["remote_by_reason"], {"big-prompt": 2, "local-offline": 1, "perf": 1})
        self.assertEqual(got["per_class"]["halo"], {"requests": 2, "remote": 2})
        self.assertEqual(got["per_class"]["kevin"], {"requests": 1, "remote": 0})
        self.assertEqual(got["per_class"]["?"], {"requests": 1, "remote": 1})
        self.assertEqual(sum(got["remote_by_reason"].values()), got["per_route"]["remote"]["requests"])


class OverflowReasons(unittest.TestCase):
    """Every reason the dashboard documents is still emitted by the shim, and every reason the shim emits is documented."""

    @staticmethod
    def emitted():
        src = (HERE / "keepalive-shim.py").read_text()
        route = src[src.index("async def _route_completions("):]
        route = route[:route.index("\n# ---------------- passthrough")]
        literal = set(re.findall(r'record_event\(\s*"(?:remote|rejected|rejected-bg)"\s*,\s*"([a-z-]+)"', route))
        # the final overflow branch picks its reason in an if/elif ladder: `reason = "tokens"` etc. before record_event("remote", reason, ...)
        ladder = set(re.findall(r'^\s+reason = "([a-z-]+)"', route, re.M))
        watchdog = set(re.findall(r'final_reason = "([a-z-]+)" if prior', src))
        return literal | ladder | watchdog

    @staticmethod
    def documented():
        html = (HERE / "gateway_dashboard.html").read_text(encoding="utf-8")
        block = html[html.index("const REASONS={"):html.index("function reasonInfo(")]
        return set(re.findall(r"^  '([a-z-]+)':\{plain:", block, re.M))

    def test_the_shim_emits_every_reason_the_page_documents(self):
        emitted, documented = self.emitted(), self.documented()
        self.assertEqual(sorted(documented - emitted), [], "documented on the page but no longer emitted by the shim")

    def test_the_page_documents_every_reason_the_shim_emits(self):
        self.assertEqual(sorted(self.emitted() - self.documented()), [], "emitted by the shim but not documented on the page")

    def test_there_are_enough_reasons_that_the_scan_is_not_vacuous(self):
        self.assertGreaterEqual(len(self.emitted()), 20)
        self.assertGreaterEqual(len(self.documented()), 20)


class LaneState(unittest.TestCase):
    CASES = [
        ("RUNNING | step 3 | writing the thing", 10, "run"),
        ("RUNNING | step 3 | writing the thing", 5000, "stale"),
        ("DONE | step 9 | merged", 10, "done"),
        ("BLOCKED | needs input | x", 10, "blocked"),
        ("STALE | no heartbeat", 10, "stale"),
        ("2026-10-03 00:25 MST DONE: applied patch", 99999, "done"),
        ("2026-10-03 00:25 working on the fold", 10, "run"),
        ("2026-10-03 00:25 working on the fold", 4000, "stale"),
        ("APPLIED patch 3", 10, "done"),
        ("· FOLD-UNBLOCKED (exact-ref revalidation) all good", 0, "run"),
        ("x" * 100 + " DONE later in a long note", 10, "run"),         # beyond the first 90 characters: ignored
        ("", 10, "empty"),
        (None, 10, "empty"),
    ]

    def test_classification(self):
        for status, age, want in self.CASES:
            self.assertEqual(shim.lane_state(status, age), want, (status, age))


def lane(name, age, status):
    return {"lane": name, "root": "/r", "newest": "STATUS", "age_s": age, "status": status}


class LanesView(unittest.TestCase):
    def data(self):
        lanes = [lane("a", 5, "RUNNING | 1 | going"), lane("b", 400, "DONE | 9 | merged"), lane("c", 4000, "RUNNING | 2 | slow"),
                 lane("d", 80000, "BLOCKED | x | y"), lane("e", 90000, "RUNNING | 1 | old"), lane("f", 700000, "no heartbeat")]
        research = [{"id": "1", "status": "running"}, {"id": "2", "status": "done"}, {"id": "3", "status": "degraded"},
                    {"id": "4", "status": "queued"}]
        return {"ts": 1, "lanes": lanes, "research": research, "active": [], "errors": []}

    def test_no_parameters_returns_everything_and_a_summary(self):
        d = self.data()
        out = shim._lanes_view(d, {})
        self.assertEqual([x["lane"] for x in out["lanes"]], list("abcdef"))
        self.assertEqual(len(out["research"]), 4)
        s = out["summary"]
        self.assertEqual((s["form"], s["lanes_total"], s["lanes_returned"], s["lanes_truncated"]), ("full", 6, 6, False))
        self.assertEqual(s["lanes_by_state"], {"run": 1, "stale": 3, "blocked": 1, "empty": 0, "done": 1})
        self.assertEqual(s["lanes_by_age"], {"le_5m": 1, "le_1h": 1, "le_1d": 2, "le_7d": 1, "older": 1})
        self.assertEqual((s["research_total"], s["research_running"]), (4, 2))
        self.assertIs(out["lanes"][0], d["lanes"][0])            # a view; the cached payload is not copied or changed
        self.assertEqual(len(d["lanes"]), 6)

    def test_active_drops_done_and_long_silent_lanes_and_finished_research(self):
        out = shim._lanes_view(self.data(), {"active": "1"})
        self.assertEqual([x["lane"] for x in out["lanes"]], ["a", "c", "d"])           # b done, e/f silent over a day
        self.assertEqual([j["id"] for j in out["research"]], ["1", "4"])
        s = out["summary"]
        self.assertEqual((s["form"], s["lanes_total"], s["lanes_returned"]), ("light", 6, 3))
        self.assertTrue(s["lanes_truncated"])
        self.assertEqual(s["filter"]["max_age_s"], 86400)

    def test_limit_keeps_the_freshest_and_reports_truncation(self):
        out = shim._lanes_view(self.data(), {"active": "1", "limit": "2"})
        self.assertEqual([x["lane"] for x in out["lanes"]], ["a", "c"])
        self.assertEqual(out["summary"]["filter"]["limit"], 2)
        out = shim._lanes_view(self.data(), {"limit": "4"})                              # limit without active: still the full filter-less list
        self.assertEqual([x["lane"] for x in out["lanes"]], list("abcd"))

    def test_max_age_and_research_limit(self):
        out = shim._lanes_view(self.data(), {"active": "1", "max_age_s": "600"})
        self.assertEqual([x["lane"] for x in out["lanes"]], ["a"])
        out = shim._lanes_view(self.data(), {"research_limit": "1"})
        self.assertEqual([j["id"] for j in out["research"]], ["1"])
        out = shim._lanes_view(self.data(), {"research_limit": "0"})
        self.assertEqual(out["research"], [])

    def test_junk_parameters_fall_back_to_the_defaults(self):
        out = shim._lanes_view(self.data(), {"active": "1", "limit": "lots", "max_age_s": "-5"})
        self.assertEqual([x["lane"] for x in out["lanes"]], ["a", "c", "d"])           # the defaults: 200 lanes, one day
        self.assertEqual(out["summary"]["filter"]["limit"], 200)
        self.assertEqual(out["summary"]["filter"]["max_age_s"], 86400)

    def test_the_light_form_is_much_smaller_on_a_realistic_feed(self):
        lanes = [lane("lane-%d" % i, 30 * i, ("DONE | 9 | merged and verified the whole thing" if i % 4 else "RUNNING | 1 | working " + "x" * 120))
                 for i in range(1200)]
        d = {"ts": 1, "lanes": lanes, "research": [{"id": str(i), "status": "done", "q": "q" * 400} for i in range(200)],
             "active": [], "errors": []}
        full = len(json.dumps(shim._lanes_view(d, {})))
        light = len(json.dumps(shim._lanes_view(d, {"active": "1", "limit": "200"})))
        self.assertLess(light, full / 5)

    def test_route_handler_serves_the_cached_payload_trimmed(self):
        d = self.data()
        d.update(research_counts={}, queue={}, health={"present": False}, inflight=0, waiting=0, budget=1)

        class Q:
            method = "GET"
            path = "/gateway/lanes"
            remote = "127.0.0.1"
            headers = {}

            def __init__(self, query):
                self.query = query

        import asyncio
        with patch.dict(shim._LANES_CACHE, {"t": time.time(), "data": d}):
            full = json.loads(asyncio.run(shim.gateway_lanes(Q({}))).body)
            light = json.loads(asyncio.run(shim.gateway_lanes(Q({"active": "1", "limit": "2"}))).body)
        self.assertEqual(len(full["lanes"]), 6)
        self.assertEqual(len(light["lanes"]), 2)
        self.assertEqual(light["summary"]["lanes_total"], 6)
        self.assertIn("active", light)                          # live requests are still present in the light form
        self.assertEqual(len(d["lanes"]), 6)                    # the cache was not mutated


class WindowedLatency(unittest.TestCase):
    def setUp(self):
        self.ring = shim.collections.deque(maxlen=1000)
        p = patch.object(shim, "_REQ_LAT", self.ring)
        p.start()
        self.addCleanup(p.stop)
        self.now = 10_000.0

    def add(self, age, route="local", ttft=1.0, duration=11.0, waited=0.0, outtok=101):
        shim.lat_note_request(self.now - age, route, ttft, duration, waited, outtok)

    def test_quantile(self):
        self.assertIsNone(shim._quantile([], 0.5))
        self.assertEqual(shim._quantile([3], 0.95), 3)
        self.assertAlmostEqual(shim._quantile([1, 2, 3, 4, 5], 0.5), 3)
        self.assertAlmostEqual(shim._quantile([1, 2, 3, 4], 0.5), 2.5)
        self.assertAlmostEqual(shim._quantile(list(range(101)), 0.95), 95)

    def test_60_second_window_only_counts_recent_requests(self):
        for ttft in (1, 2, 3, 4, 5):
            self.add(10, ttft=ttft)
        self.add(120, ttft=100)                                   # outside 60 s, inside 5 min
        w = shim.req_latency_windows(self.now)
        self.assertEqual(w["60s"]["local"]["n"], 5)
        self.assertEqual(w["60s"]["local"]["ttft_p50"], 3)
        self.assertEqual(w["300s"]["local"]["n"], 6)
        self.assertGreater(w["300s"]["local"]["ttft_p95"], 50)

    def test_inter_token_latency_is_decode_time_over_tokens_minus_one(self):
        # duration 11 s, ttft 1 s, no wait -> 10 s of decode for 101 tokens = 100 gaps = 0.1 s
        self.add(5, ttft=1.0, duration=11.0, waited=0.0, outtok=101)
        w = shim.req_latency_windows(self.now)["60s"]["local"]
        self.assertEqual(w["itl_n"], 1)
        self.assertAlmostEqual(w["itl_p50"], 0.1, places=4)

    def test_admission_wait_is_not_decode_time(self):
        self.add(5, ttft=1.0, duration=21.0, waited=10.0, outtok=101)
        self.assertAlmostEqual(shim.req_latency_windows(self.now)["60s"]["local"]["itl_p50"], 0.1, places=4)

    def test_requests_without_exact_token_counts_have_a_ttft_but_no_inter_token_value(self):
        self.add(5, outtok=None)
        self.add(5, outtok=1)
        w = shim.req_latency_windows(self.now)["60s"]["local"]
        self.assertEqual((w["n"], w["itl_n"], w["itl_p50"]), (2, 0, None))

    def test_non_streaming_and_unrouted_requests_are_skipped(self):
        self.add(5, ttft=None)
        self.add(5, route="rejected-bg")
        shim.lat_note_request(self.now, "local", -1.0, 5.0, 0.0, 10)
        self.assertEqual(len(self.ring), 0)
        self.assertEqual(shim.req_latency_windows(self.now)["60s"]["local"]["n"], 0)

    def test_local_and_remote_are_kept_apart(self):
        self.add(5, route="local", ttft=1.0)
        self.add(5, route="remote", ttft=9.0)
        w = shim.req_latency_windows(self.now)["60s"]
        self.assertEqual((w["local"]["ttft_p50"], w["remote"]["ttft_p50"]), (1.0, 9.0))

    def test_empty_window_is_null_not_zero(self):
        w = shim.req_latency_windows(self.now)["60s"]["local"]
        self.assertEqual((w["n"], w["ttft_p50"], w["ttft_p95"], w["itl_p50"]), (0, None, None, None))

    def test_a_finished_request_reaches_the_ring_through_telemetry(self):
        now = time.time()
        info = {"name": "halo-hermes", "route": "local", "ttft": 2.0, "t0": now - 12.0, "waited": 0.0, "outtok": 51,
                "flow_class": "halo"}
        shim._telemetry_note_request(info, type("R", (), {"status": 200})())
        self.assertEqual(len(self.ring), 1)
        t_end, route, ttft, itl, cls = self.ring[0]
        self.assertEqual((route, ttft, cls), ("local", 2.0, "halo"))
        self.assertAlmostEqual(itl, 0.2, delta=0.02)

    def test_failed_requests_do_not_count(self):
        now = time.time()
        info = {"name": "x", "route": "local", "ttft": 2.0, "t0": now - 12.0, "outtok": 51}
        shim._telemetry_note_request(info, type("R", (), {"status": 500})())
        self.assertEqual(len(self.ring), 0)

    def test_telemetry_endpoint_publishes_the_windows(self):
        self.add(5)

        class Q:
            query = {"fast": "0", "slow": "0"}

        import asyncio
        with patch.object(shim.time, "time", lambda: self.now):
            body = json.loads(asyncio.run(shim.gateway_telemetry(Q())).body)
        wl = body["windowed_latency"]
        self.assertEqual(sorted(wl["windows"]), ["300s", "60s", "900s"])
        self.assertEqual(wl["windows"]["60s"]["local"]["n"], 1)
        self.assertIn("first streamed byte", wl["ttft"])

    def test_samples_carry_the_60_second_value_for_the_charts(self):
        self.add(1, ttft=2.5)
        self.add(1, ttft=3.5)

        async def go():
            with patch.object(shim, "_gpu_stats_async", AsyncMock(return_value=[])), \
                    patch.object(shim, "_host_stats_blocking", lambda: {}), \
                    patch.object(shim, "_scrape_engine_metrics", AsyncMock(return_value={"ok": False})), \
                    patch.object(shim.time, "time", lambda: self.now):
                return await shim._take_sample()

        import asyncio
        g = asyncio.run(go())["gateway"]
        self.assertEqual(g["lat60_n"], 2)
        self.assertEqual(g["ttft60_p50"], 3.0)
        self.assertIsNotNone(g["ttft60_p95"])
        ds = shim._downsample([{"t": 1, "gateway": {"inflight": 0, "budget": 1, "waiting": 0, "backoff_s": 0, "local_healthy": True,
                                                     "ttft60_p50": 2.0, "ttft60_p95": 4.0, "itl60_p50": None, "itl60_p95": None,
                                                     "lat60_n": 3},
                                "host": {}, "gpu": [], "engine": {"ok": False}},
                               {"t": 2, "gateway": {"inflight": 0, "budget": 1, "waiting": 0, "backoff_s": 0, "local_healthy": True,
                                                     "ttft60_p50": 4.0, "ttft60_p95": 6.0, "itl60_p50": None, "itl60_p95": None,
                                                     "lat60_n": 5},
                                "host": {}, "gpu": [], "engine": {"ok": False}}])
        self.assertEqual(ds["gateway"]["ttft60_p50"], 3.0)
        self.assertIsNone(ds["gateway"]["itl60_p50"])


if __name__ == "__main__":
    unittest.main()
