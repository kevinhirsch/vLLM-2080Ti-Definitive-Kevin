#!/usr/bin/env python3
"""Lane GW (2026-10-02): the gateway's capacity model is derived from the LIVE engine (KV pool, token budget, prefill
rate, attention block size) with the configured values only as fallback / explicit override.  Run:
python -m unittest test_gateway_capacity_model"""
import atexit
import importlib.util
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

_TMPDIR = tempfile.TemporaryDirectory(prefix="gw-cap-test-")
atexit.register(_TMPDIR.cleanup)
_TMP = _TMPDIR.name
_ENV = {k: os.path.join(_TMP, v) for k, v in {
    "SHIM_SPEND_FILE": "spend.json", "SHIM_SPEND_CLIENTS_FILE": "spend-clients.json", "SHIM_STATS_FILE": "stats.json",
    "SHIM_TELEMETRY_DIR": "telemetry", "SHIM_ENV_FILE": "shim.env", "SHIM_ALIASES_FILE": "aliases.json",
    "SHIM_FLIGHTREC_DIR": "flightrec", "SHIM_FLOW_EVENTS_FILE": "incidents/capacity-mode.jsonl"}.items()}
_ENV["SHIM_EXACT_TOKENS"] = "0"
with patch.dict(os.environ, _ENV):
    SPEC = importlib.util.spec_from_file_location("shim_cap_test", os.environ.get(
        "SHIM_TEST_CANDIDATE", str(Path(__file__).with_name("keepalive-shim.py"))))
    shim = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(shim)

# The real cache_config_info line of the live engine (2026-10-02 23:21), trimmed to the labels that matter plus the awkward ones.
LIVE_INFO = ('vllm:cache_config_info{_block_size_resolved="True",block_size="3568",cache_dtype="turboquant_k3v4_nc",'
             'kv_cache_dtype_skip_layers="[]",kv_cache_size_tokens="922358",num_gpu_blocks="285",'
             'sliding_window="None"} 1.0\n')


def fam(info=LIVE_INFO, start=1791008353.28):
    txt = info + ("process_start_time_seconds %r\n" % start if start else "")
    return shim._parse_prom_text(txt)


class Base(unittest.TestCase):
    def setUp(self):
        self._p = []
        for name, val in dict(CAPACITY_LIVE=True, CONTEXT_LIVE=True, LOCAL_CONTEXT_LIMIT=524288, MAX_LOCAL_TOKENS=524288, TOKEN_BUDGET=None, POOL_TOKENS=637560, PREFILL_TPS=1100.0,
                              PREFIX_ALIGN_TOKENS=3568, TOKEN_BUDGET_CEIL=680000).items():
            p = patch.object(shim, name, val)
            p.start()
            self._p.append(p)
        self._st = patch.dict(shim._CAPLIVE, {"pool": None, "pool_at": 0.0, "pool_src": None, "block": None, "gen_start": None,
                                              "gen_src": None, "scrapes": 0, "measure": None,
                                              "ctx": None, "ctx_at": 0.0, "ctx_poll_at": 0.0, "ctx_err": None, "ctx_models": None})
        self._st.start()
        shim._CAPLIVE["events"].clear()
        self._eng = patch.dict(shim._ENGINE_METRICS, {"ok": True})
        self._eng.start()
        shim._FLOW_PURE.clear()
        shim._OFFLINE_SPANS.clear()

    def tearDown(self):
        for p in self._p:
            p.stop()
        self._st.stop()
        self._eng.stop()
        shim._FLOW_PURE.clear()
        shim._OFFLINE_SPANS.clear()


class PoolFromEngine(Base):
    def test_pool_read_from_cache_config_info_not_config(self):
        shim.capacity_note_scrape(fam(), 1000.0)
        p = shim.pool_info(1000.0)
        self.assertEqual((p["tokens"], p["source"]), (922358, "live"))
        self.assertTrue(p["configured_stale"])
        self.assertEqual(p["configured_tokens"], 637560)

    def test_blocks_times_size_only_when_the_direct_field_is_absent(self):
        info = 'vllm:cache_config_info{block_size="16",num_gpu_blocks="1000"} 1.0\n'
        shim.capacity_note_scrape(fam(info), 1000.0)
        self.assertEqual(shim.pool_info(1000.0)["tokens"], 16000)

    def test_unreachable_after_seen_keeps_last_known_live_never_the_stale_config(self):
        shim.capacity_note_scrape(fam(), 1000.0)
        shim._ENGINE_METRICS["ok"] = False
        p = shim.pool_info(1060.0)
        self.assertEqual((p["tokens"], p["source"]), (922358, "live-last-known"))

    def test_never_seen_falls_back_to_configured(self):
        p = shim.pool_info(1000.0)
        self.assertEqual((p["tokens"], p["source"]), (637560, "configured"))

    def test_kill_switch_restores_configured(self):
        shim.capacity_note_scrape(fam(), 1000.0)
        with patch.object(shim, "CAPACITY_LIVE", False):
            self.assertEqual(shim.pool_info(1000.0)["tokens"], 637560)
            self.assertEqual(shim.token_budget_info(1000.0)["source"], "configured-pool-derived")
            self.assertEqual(shim.prefix_align_tokens(), 3568)

    def test_restart_with_a_different_pool_is_picked_up_on_the_next_scrape(self):
        shim.capacity_note_scrape(fam(), 1000.0)
        info2 = LIVE_INFO.replace('kv_cache_size_tokens="922358"', 'kv_cache_size_tokens="500000"')
        shim.capacity_note_scrape(fam(info2, start=1791009999.0), 1100.0)
        self.assertEqual(shim.pool_info(1100.0)["tokens"], 500000)
        self.assertEqual(shim.token_budget(), int(500000 * shim.TOKEN_BUDGET_FRAC))
        self.assertTrue(any("restarted" in e["event"] for e in shim._CAPLIVE["events"]))

    def test_foreign_load_guard_uses_the_effective_pool(self):
        shim.capacity_note_scrape(fam(), 1000.0)
        self.assertEqual(int(0.5 * shim.pool_info()["tokens"]), 461179)


class TokenBudget(Base):
    def test_declared_fraction_of_the_live_pool(self):
        shim.capacity_note_scrape(fam(), 1000.0)
        tb = shim.token_budget_info(1000.0)
        self.assertEqual(tb["source"], "live-derived")
        self.assertEqual(tb["tokens"], int(922358 * 500000 / 754068))       # the calibration, not a new number
        self.assertAlmostEqual(tb["tokens"] / 922358, 500000 / 754068, places=4)
        self.assertEqual(shim.token_budget(), tb["tokens"])

    def test_calibration_pool_reproduces_the_calibrated_budget(self):
        info = LIVE_INFO.replace('kv_cache_size_tokens="922358"', 'kv_cache_size_tokens="754068"')
        shim.capacity_note_scrape(fam(info), 1000.0)
        self.assertEqual(shim.token_budget(), 500000)

    def test_ceiling_stops_a_bigger_pool_from_exceeding_the_benchmarked_envelope(self):
        info = LIVE_INFO.replace('kv_cache_size_tokens="922358"', 'kv_cache_size_tokens="3000000"')
        shim.capacity_note_scrape(fam(info), 1000.0)
        tb = shim.token_budget_info()
        self.assertEqual(tb["tokens"], 680000)
        self.assertTrue(tb["ceiling_applied"])

    def test_explicit_override_wins_and_a_stale_one_is_flagged(self):
        shim.capacity_note_scrape(fam(), 1000.0)
        with patch.object(shim, "TOKEN_BUDGET", 500000):
            tb = shim.token_budget_info(1000.0)
            self.assertEqual((tb["tokens"], tb["source"]), (500000, "override"))
            facts = shim.capacity_model_facts(1000.0)
            self.assertEqual(facts["token_budget"]["derived_from_live"], 611588)
            self.assertTrue(any("SHIM_TOKEN_BUDGET=500000" in w for w in facts["warnings"]))
            self.assertEqual(shim._halo_control_token_reserve(), 62500)
        with patch.object(shim, "TOKEN_BUDGET", 0):
            self.assertEqual(shim.token_budget(), 0)             # 0 still means "no cap"
            self.assertTrue(shim._memory_available(10 ** 9))

    def test_budget_fallback_when_engine_never_answered_is_fraction_of_configured_pool(self):
        with patch.object(shim, "POOL_TOKENS", 754068):
            tb = shim.token_budget_info()
            self.assertEqual((tb["tokens"], tb["source"]), (500000, "configured-pool-derived"))

    def test_admission_uses_the_effective_budget(self):
        shim.capacity_note_scrape(fam(), 1000.0)
        with patch.object(shim, "_inflight_reserved_tokens", 500000):
            self.assertTrue(shim._memory_available(100000, halo_control=True))     # 600K <= 611,588: old 500K cap refused this
            self.assertFalse(shim._memory_available(120000, halo_control=True))

    def test_config_parser_and_roundtrip(self):
        self.assertIsNone(shim._parse_budget("auto"))
        self.assertIsNone(shim._parse_budget(""))
        self.assertEqual(shim._parse_budget("500000"), 500000)
        self.assertEqual(shim._parse_budget(0), 0)
        with patch.object(shim, "TOKEN_BUDGET", None):
            self.assertEqual(shim.current_config()["token_budget"], "auto")
        changed = shim.apply_config({"token_budget": "auto"})
        self.assertIn("token_budget", changed)
        self.assertIsNone(shim.TOKEN_BUDGET)


class PrefillRate(Base):
    GEN = 10_000.0

    def feed(self, rate, start, minutes, per_min_secs=20.0):
        """one (t, tokens, secs) row per 2 s scrape for `minutes` minutes at `rate` tok/s of prefill"""
        t = start
        while t < start + minutes * 60:
            shim._FLOW_PURE.append((t, rate * per_min_secs / 30, per_min_secs / 30))
            t += 2.0

    def setup_gen(self):
        shim._CAPLIVE.update(gen_start=self.GEN, gen_src="process_start_time_seconds")

    def test_measured_p75_replaces_configured(self):
        self.setup_gen()
        t0 = self.GEN + 200
        for i, r in enumerate((1200, 1250, 1300, 1280, 1270, 1260, 1290)):
            self.feed(r, t0 + 60 * i, 1)
        now = t0 + 60 * 7 + 1
        pf = shim.prefill_info(now)
        self.assertEqual(pf["source"], "measured")
        self.assertTrue(1260 <= pf["tok_s"] <= 1290, pf["tok_s"])

    def test_transient_dip_does_not_collapse_it(self):
        self.setup_gen()
        t0 = self.GEN + 200
        for i in range(10):
            self.feed(1300 if i not in (6, 7) else 300, t0 + 60 * i, 1)      # a two-minute dip to 300 tok/s
        now = t0 + 600 + 1
        self.assertGreater(shim.prefill_info(now)["tok_s"], 1200)

    def test_too_few_samples_fall_back_to_configured(self):
        self.setup_gen()
        self.feed(1300, self.GEN + 200, 2)
        pf = shim.prefill_info(self.GEN + 330)
        self.assertEqual((pf["tok_s"], pf["source"]), (1100.0, "configured"))
        self.assertEqual(pf["measurement"]["status"], "insufficient")

    def test_previous_engine_generation_and_settle_period_are_excluded(self):
        self.setup_gen()
        self.feed(300, self.GEN - 900, 14)                     # the old engine: slow
        self.feed(300, self.GEN + 10, 2)                      # the new engine's cold settle minutes
        self.feed(1300, self.GEN + 200, 7)
        rep = shim.measured_prefill_pure(self.GEN + 200 + 7 * 60 + 1)
        self.assertEqual(rep["status"], "ok")
        self.assertGreater(rep["tok_s"], 1250)
        self.assertGreater(rep["excluded_pre_generation"], 0)

    def test_settling_engine_reports_settling_and_keeps_configured(self):
        self.setup_gen()
        self.feed(1300, self.GEN + 10, 2)
        rep = shim.measured_prefill_pure(self.GEN + 130)
        self.assertEqual(rep["status"], "settling")
        self.assertIsNone(rep["tok_s"])

    def test_planned_offline_window_samples_are_excluded(self):
        self.setup_gen()
        t0 = self.GEN + 200
        self.feed(1300, t0, 6)
        shim._OFFLINE_SPANS.append((t0 + 360, t0 + 600))
        self.feed(200, t0 + 360, 4)                              # a benchmark shared the engine
        rep = shim.measured_prefill_pure(t0 + 600 + 5)
        self.assertEqual(rep["status"], "ok")
        self.assertGreater(rep["tok_s"], 1250)
        self.assertGreater(rep["excluded_offline"], 0)

    def test_closing_a_window_records_the_span(self):
        shim._flow_event({"event": "offline-close", "t0": 100.0, "t": 200.0})
        self.assertIn((100.0, 200.0), list(shim._OFFLINE_SPANS))

    def test_kill_switch_uses_configured(self):
        self.setup_gen()
        self.feed(1300, self.GEN + 200, 8)
        with patch.object(shim, "CAPACITY_LIVE", False):
            self.assertEqual(shim.prefill_info(self.GEN + 700)["tok_s"], 1100.0)

    def test_consumers_follow_the_effective_rate(self):
        self.setup_gen()
        self.feed(2000, self.GEN + 200, 8)
        shim._CAPLIVE["measure"] = None
        with patch.object(shim, "_inflight_computed", 40000):
            # prefill_tps() caches 5 s keyed on real time; force a fresh read from the synthetic clock
            shim._CAPLIVE["measure"] = {"at": time.time(), "rep": shim.measured_prefill_pure(self.GEN + 700)}
            self.assertAlmostEqual(shim._prefill_backlog_secs(), 40000 / 2000, places=1)


class GenerationAndFacts(Base):
    def test_health_transition_marks_new_generation_when_no_process_start_metric(self):
        shim.capacity_note_scrape(fam(start=None), 1000.0)
        self.assertEqual(shim._CAPLIVE["gen_src"], "first scrape")
        shim._capacity_engine_changed(2000.0, "engine back after being down")
        self.assertEqual((shim._CAPLIVE["gen_start"], shim._CAPLIVE["gen_src"]), (2000.0, "health transition"))

    def test_block_size_is_live(self):
        info = LIVE_INFO.replace('block_size="3568"', 'block_size="1024"')
        shim.capacity_note_scrape(fam(info), 1000.0)
        self.assertEqual(shim.prefix_align_tokens(), 1024)

    def test_facts_are_serialisable_and_name_every_source(self):
        shim.capacity_note_scrape(fam(), time.time())
        f = shim.capacity_model_facts()
        json.dumps(f)
        for k in ("kv_pool_tokens", "token_budget", "prefill_tok_s", "prefix_align_tokens", "engine_generation", "warnings"):
            self.assertIn(k, f)
        for k in ("kv_pool_tokens", "token_budget", "prefill_tok_s"):
            self.assertIn("source", f[k])
            self.assertIn("effective", f[k])

    def test_capacity_endpoint_and_stats_carry_the_model(self):
        shim.capacity_note_scrape(fam(), time.time())
        f = shim.flow_capacity_facts()
        self.assertEqual(f["capacity_model"]["kv_pool_tokens"]["effective"], 922358)
        self.assertIn("prefill_effective_tok_s", f["throughput"])

    def test_scrape_hook_survives_garbage(self):
        shim.capacity_note_scrape({"vllm:cache_config_info": [({"kv_cache_size_tokens": "x"}, 1.0)]}, 1.0)
        shim.capacity_note_scrape({}, 2.0)
        self.assertEqual(shim.pool_info()["source"], "configured")


def models(*lens, extra=None):
    data = [{"id": "m%d" % i, "object": "model", "max_model_len": n} for i, n in enumerate(lens)]
    return {"object": "list", "data": data + (extra or [])}


class ContextWindowFromEngine(Base):
    """Lane FX: SHIM_LOCAL_CONTEXT_LIMIT / SHIM_MAX_LOCAL_TOKENS follow the engine's live /v1/models max_model_len."""
    def test_never_answered_uses_configured(self):
        self.assertEqual(shim.local_context_limit(), 524288)
        i = shim.context_window_info(shim.LOCAL_CONTEXT_LIMIT)
        self.assertEqual((i["source"], i["tokens"]), ("configured", 524288))
        self.assertEqual(i["detail"], "engine has not answered yet")

    def test_live_value_replaces_both_knobs(self):
        shim.capacity_note_models(models(262144, 262144), time.time())
        self.assertEqual((shim.local_context_limit(), shim.max_local_tokens()), (262144, 262144))
        i = shim.context_window_info(shim.MAX_LOCAL_TOKENS)
        self.assertEqual((i["source"], i["configured"], i["live"], i["configured_stale"]), ("live", 524288, 262144, True))

    def test_live_matching_config_is_not_stale(self):
        shim.capacity_note_models(models(524288), time.time())
        self.assertFalse(shim.context_window_info(shim.LOCAL_CONTEXT_LIMIT)["configured_stale"])
        self.assertEqual([w for w in shim.capacity_model_facts()["warnings"] if "max_model_len" in w], [])

    def test_smallest_served_window_wins_and_garbage_entries_are_skipped(self):
        shim.capacity_note_models(models(524288, 131072, "x", 0, None, 12, extra=[{"id": "no-len"}]), time.time())
        self.assertEqual(shim.local_context_limit(), 131072)

    def test_garbage_payloads_keep_the_last_value_and_report(self):
        shim.capacity_note_models(models(300000), time.time())
        for bad in (None, [], {"data": []}, {"data": [{"id": "a", "max_model_len": "n/a"}]}, "x"):
            shim.capacity_note_models(bad, time.time())
        self.assertEqual(shim.local_context_limit(), 300000)
        shim.capacity_note_models({"data": []}, time.time())
        self.assertIn("usable max_model_len", shim._CAPLIVE["ctx_err"])

    def test_unreachable_after_seen_keeps_last_known_live(self):
        shim.capacity_note_models(models(300000), time.time() - 1000)       # answered long ago, silent since
        i = shim.context_window_info(shim.LOCAL_CONTEXT_LIMIT)
        self.assertEqual((i["source"], i["tokens"]), ("live-last-known", 300000))

    def test_kill_switches_restore_configured(self):
        shim.capacity_note_models(models(300000), time.time())
        for name in ("CONTEXT_LIVE", "CAPACITY_LIVE"):
            with patch.object(shim, name, False):
                self.assertEqual((shim.local_context_limit(), shim.max_local_tokens()), (524288, 524288))
                self.assertEqual(shim.context_window_info(shim.LOCAL_CONTEXT_LIMIT)["source"], "configured")

    def test_a_disabled_guard_is_never_replaced(self):
        shim.capacity_note_models(models(300000), time.time())
        with patch.object(shim, "MAX_LOCAL_TOKENS", 0):
            self.assertEqual(shim.max_local_tokens(), 0)

    def test_restart_with_another_window_is_followed_and_forces_a_reread(self):
        shim.capacity_note_models(models(524288), time.time())
        shim._CAPLIVE["ctx_poll_at"] = 1000.0
        shim._capacity_engine_changed(1100.0, "engine restarted")
        self.assertEqual(shim._CAPLIVE["ctx_poll_at"], 0.0)
        shim.capacity_note_models(models(200000), 1101.0)
        self.assertEqual(shim.local_context_limit(), 200000)
        self.assertTrue(any("context window 524288 -> 200000" in e["event"] for e in shim._CAPLIVE["events"]))

    def test_admission_math_follows_the_live_window(self):
        body = json.dumps({"messages": [{"role": "user", "content": "x" * 4000}], "max_tokens": 1000}).encode()
        self.assertFalse(shim.over_local_cap(body))
        shim.capacity_note_models(models(2048), time.time())         # shrunken engine window: the same request no longer fits
        self.assertTrue(shim.over_local_cap(body))
        with patch.object(shim, "LOCAL_MAX_OUT", 0):             # no output clamp: the reservation fills the live window
            self.assertEqual(shim.local_reservation_estimate(500, 0), 2048)

    def test_facts_carry_the_context_window(self):
        shim.capacity_note_models(models(262144), time.time())
        f = shim.flow_capacity_facts()["capacity_model"]["context_window"]
        json.dumps(f)
        self.assertEqual(f["local_context_limit"]["effective"], 262144)
        self.assertEqual(f["max_local_tokens"]["source"], "live")
        self.assertEqual(f["kill_switch"], "SHIM_CONTEXT_LIVE=0")
        self.assertTrue(any("SHIM_LOCAL_CONTEXT_LIMIT=524288 differs" in w for w in shim.capacity_model_facts()["warnings"]))

    def test_config_parser_knows_the_new_knobs(self):
        self.assertIn("SHIM_CONTEXT_LIVE", shim._CFG)
        self.assertIn("SHIM_MODELS_POLL_SECS", shim._CFG)
        self.assertIs(shim._CFG["SHIM_CONTEXT_LIVE"][1]("0"), False)


class ModelsPoll(unittest.IsolatedAsyncioTestCase):
    """The poller itself, against a real local HTTP server standing in for the engine."""
    async def asyncSetUp(self):
        from aiohttp import web
        self.payload, self.status, self.hits = models(262144), 200, 0

        async def h(request):
            self.hits += 1
            return web.json_response(self.payload, status=self.status)
        app = web.Application()
        app.router.add_get("/v1/models", h)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = self.runner.addresses[0][1]
        self._p = [patch.object(shim, "_MODELS_URL", "http://127.0.0.1:%d/v1/models" % port),
                   patch.object(shim, "CAPACITY_LIVE", True), patch.object(shim, "CONTEXT_LIVE", True),
                   patch.object(shim, "MODELS_POLL_S", 30.0),
                   patch.dict(shim._CAPLIVE, {"ctx": None, "ctx_at": 0.0, "ctx_poll_at": 0.0, "ctx_err": None, "ctx_models": None})]
        for p in self._p:
            p.start()

    async def asyncTearDown(self):
        for p in self._p:
            p.stop()
        await self.runner.cleanup()

    async def test_polls_reads_and_is_rate_limited(self):
        await shim._poll_engine_models(1000.0)
        self.assertEqual((shim._CAPLIVE["ctx"], self.hits), (262144, 1))
        self.payload = models(100000)
        await shim._poll_engine_models(1010.0)            # inside MODELS_POLL_S: no request
        self.assertEqual((shim._CAPLIVE["ctx"], self.hits), (262144, 1))
        await shim._poll_engine_models(1031.0)
        self.assertEqual((shim._CAPLIVE["ctx"], self.hits), (100000, 2))

    async def test_http_error_keeps_last_value_and_reports(self):
        await shim._poll_engine_models(1000.0)
        self.status = 503
        await shim._poll_engine_models(1100.0)
        self.assertEqual(shim._CAPLIVE["ctx"], 262144)
        self.assertIn("503", shim._CAPLIVE["ctx_err"])

    async def test_kill_switch_never_calls_the_engine(self):
        with patch.object(shim, "CONTEXT_LIVE", False):
            await shim._poll_engine_models(1000.0)
        self.assertEqual((self.hits, shim._CAPLIVE["ctx"]), (0, None))

    async def test_engine_down_does_not_raise(self):
        with patch.object(shim, "_MODELS_URL", "http://127.0.0.1:1/v1/models"):
            await shim._poll_engine_models(1000.0)
        self.assertIsNone(shim._CAPLIVE["ctx"])
        self.assertTrue(shim._CAPLIVE["ctx_err"])


if __name__ == "__main__":
    unittest.main()
