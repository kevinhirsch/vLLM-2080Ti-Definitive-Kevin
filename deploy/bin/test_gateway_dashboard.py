#!/usr/bin/env python3
"""The gateway dashboard (deploy/bin/gateway_dashboard.html, served by keepalive-shim.py at /gateway/dashboard).

Covers: the route serves the file (mtime-cached, inline fallback), the small server fields the page needs, and the page's
pure data->view functions (DashLib, run under node against recorded endpoint shapes). The visual side is checked in a real
browser (docs/gateway-dashboard-shots/); these tests pin the field mapping and the arithmetic.

Run:  python -m unittest test_gateway_dashboard
"""
import asyncio
import atexit
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PAGE = HERE / "gateway_dashboard.html"

_TMPDIR = tempfile.TemporaryDirectory(prefix="gw-dash-test-")
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
    "SHIM_FLOW_EVENTS_FILE": os.path.join(_TMP, "incidents", "capacity-mode.jsonl"),
    "SHIM_DASHBOARD_FILE": os.path.join(_TMP, "none.html"),
    "SHIM_EXACT_TOKENS": "0",
}
with patch.dict(os.environ, _ISOLATED_ENV):
    SPEC = importlib.util.spec_from_file_location("shim_dash_test", os.environ.get(
        "SHIM_TEST_CANDIDATE", str(HERE / "keepalive-shim.py")))
    shim = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(shim)


class FakeRequest:
    method = "GET"
    path = "/gateway/dashboard"
    remote = "127.0.0.1"
    headers = {}
    query = {}


def run(coro):
    return asyncio.run(coro)


class DashboardRoute(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(dir=_TMP)
        self.file = os.path.join(self.dir, "page.html")
        shim._DASH_CACHE.update(mtime=None, path=None, text=None)
        self._patch = patch.object(shim, "DASHBOARD_FILE", self.file)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_serves_the_file_not_the_inline_copy(self):
        Path(self.file).write_text("<!doctype html><html><body>FROM-FILE</body></html>")
        r = run(shim.gateway_dashboard(FakeRequest()))
        self.assertEqual(r.status, 200)
        self.assertIn("FROM-FILE", r.text)
        self.assertEqual(r.headers["X-Dashboard-Source"], "file")
        self.assertEqual(r.content_type, "text/html")
        self.assertEqual(r.headers["Cache-Control"], "no-store")

    def test_edit_is_picked_up_without_restart(self):
        f = Path(self.file)
        f.write_text("<html>one</html>")
        self.assertIn("one", run(shim.gateway_dashboard(FakeRequest())).text)
        f.write_text("<html>two-two</html>")
        os.utime(self.file, ns=(1, 2_000_000_000_000_000_000))        # force a distinct mtime
        self.assertIn("two-two", run(shim.gateway_dashboard(FakeRequest())).text)

    def test_missing_file_falls_back_to_inline_copy(self):
        r = run(shim.gateway_dashboard(FakeRequest()))
        self.assertEqual(r.status, 200)
        self.assertEqual(r.headers["X-Dashboard-Source"], "inline-fallback")
        self.assertIn("vLLM Gateway", r.text)
        self.assertEqual(r.text, shim.DASHBOARD_HTML)

    def test_file_that_is_not_html_falls_back(self):
        Path(self.file).write_text('{"oops": true}')
        r = run(shim.gateway_dashboard(FakeRequest()))
        self.assertEqual(r.headers["X-Dashboard-Source"], "inline-fallback")

    def test_deleting_the_file_later_falls_back_again(self):
        Path(self.file).write_text("<html>x</html>")
        run(shim.gateway_dashboard(FakeRequest()))
        os.unlink(self.file)
        self.assertEqual(run(shim.gateway_dashboard(FakeRequest())).headers["X-Dashboard-Source"], "inline-fallback")

    def test_the_committed_page_is_what_a_default_shim_serves(self):
        html, source = None, None
        with patch.object(shim, "DASHBOARD_FILE", str(PAGE)):
            shim._DASH_CACHE.update(mtime=None, path=None, text=None)
            html, source = shim.dashboard_html()
        self.assertEqual(source, "file")
        self.assertEqual(html, PAGE.read_text(encoding="utf-8"))


class ServerFields(unittest.TestCase):
    def test_stats_separates_process_uptime_from_counter_age(self):
        body = json.loads(run(shim.gateway_stats(FakeRequest())).body)
        self.assertIn("process_uptime", body)
        self.assertIn("stats_since", body)
        self.assertLessEqual(body["process_uptime"], body["uptime"])      # counters are older than (or as old as) the process
        self.assertGreaterEqual(body["process_uptime"], 0)

    def test_telemetry_tail_trims_and_keeps_back_compat(self):
        self.assertEqual(shim._telem_tail(range(10), "3"), [7, 8, 9])
        self.assertEqual(shim._telem_tail(range(10), "0"), [])
        self.assertEqual(shim._telem_tail(range(10), None), list(range(10)))      # absent: everything, as before
        self.assertEqual(shim._telem_tail(range(10), "junk"), list(range(10)))
        self.assertEqual(shim._telem_tail(range(10), "-5"), list(range(10)))
        self.assertEqual(shim._telem_tail(range(3), "99"), [0, 1, 2])

    def test_telemetry_route_honours_fast_and_slow(self):
        class Q(FakeRequest):
            query = {"fast": "0", "slow": "0"}
        body = json.loads(run(shim.gateway_telemetry(Q())).body)
        self.assertEqual(body["series"], {"fast": [], "slow": []})
        for key in ("latest", "per_client", "errors", "percentiles", "mis_estimates", "engine_scrape"):
            self.assertIn(key, body)


class PageStatic(unittest.TestCase):
    html = PAGE.read_text(encoding="utf-8")

    def test_no_external_resources(self):
        # works offline on the LAN: nothing may be fetched from another origin
        self.assertIsNone(re.search(r'(?:src|href)\s*=\s*["\']?https?://', self.html), "external src/href")
        self.assertIsNone(re.search(r'@import|url\(\s*["\']?https?:', self.html), "external css")
        self.assertNotRegex(self.html, r"cdn\.|googleapis|unpkg|jsdelivr|cdnjs")

    def test_every_element_the_script_looks_up_exists(self):
        ids = set(re.findall(r'\bid\s*=\s*["\']?([A-Za-z0-9_\-]+)', self.html))
        used = set(re.findall(r"""\$\(\s*['"]#([A-Za-z0-9_\-]+)""", self.html))
        used |= set(re.findall(r"""getElementById\(\s*['"]([A-Za-z0-9_\-]+)['"](?!\s*\+)""", self.html))
        missing = sorted(u for u in used if u not in ids)
        self.assertEqual(missing, [], "script looks up ids that the page does not define")

    def test_every_settings_field_is_still_present(self):
        old = {m for m in re.findall(r'\bid=f_([a-z_0-9]+)', self.html)}
        for field in ("remote_base", "remote_model", "remote_key", "force_remote", "local_budget", "local_wait_secs", "token_budget",
                      "oom_backoff_secs", "big_output", "big_prompt", "max_local_tokens", "local_max_out", "big_tokens", "tokens_per_unit",
                      "use_computed_cost", "prefix_hit_margin_tokens", "prefill_admit_secs", "light_prefill_secs", "monster_prefill_secs",
                      "monster_inflight", "first_token_max", "prefill_tps", "tiny_tokens", "tiny_extra_lanes", "fg_reserved", "bg_wait_secs",
                      "bg_markers", "peak_hours_utc", "bg_no_think", "no_think_ips", "log_requests", "interactive_never_overflow"):
            self.assertIn(field, old, "settings field f_%s was dropped" % field)

    def test_token_budget_input_accepts_auto(self):
        m = re.search(r'<input id=f_token_budget[^>]*>', self.html)
        self.assertIsNotNone(m)
        self.assertIn("type=text", m.group(0))
        self.assertIn('placeholder="auto"', m.group(0))

    def test_every_section_the_nav_links_to_exists(self):
        ids = set(re.findall(r'\bid\s*=\s*["\']?([A-Za-z0-9_\-]+)', self.html))
        nav = re.findall(r'<a href="#([a-z]+)">', self.html.split('id="jump"')[1].split("</nav>")[0])
        self.assertGreaterEqual(len(nav), 10)
        self.assertEqual([n for n in nav if n not in ids], [])

    def test_script_parses(self):
        if not shutil.which("node"):
            self.skipTest("node not installed")
        for i, body in enumerate(re.findall(r"<script>(.*?)</script>", self.html, re.S)):
            p = Path(_TMP) / ("script%d.js" % i)
            p.write_text(body)
            r = subprocess.run(["node", "--check", str(p)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)


def _lib_source():
    html = PAGE.read_text(encoding="utf-8")
    return html[html.index("/*LIB-START"):html.index("/*LIB-END*/")]


@unittest.skipUnless(shutil.which("node"), "node not installed")
class DashLibMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = Path(_TMP) / "dashlib.js"
        cls.lib.write_text(_lib_source() + "\nmodule.exports=DashLib;\n")

    def js(self, expr, **data):
        prog = "const L=require(%s);%s;const r=(%s);console.log(JSON.stringify(r===undefined?null:r));" % (
            json.dumps(str(self.lib)), "".join("const %s=%s;" % (k, json.dumps(v)) for k, v in data.items()), expr)
        r = subprocess.run(["node", "-e", prog], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)

    # ---- formatting ----
    def test_formatting_always_carries_units_and_never_prints_nan(self):
        self.assertEqual(self.js("[L.fmtDur(0.4),L.fmtDur(7.26),L.fmtDur(75),L.fmtDur(3725),L.fmtDur(190000),L.fmtDur(null),L.fmtDur(NaN)]"),
                         ["400 ms", "7.3 s", "1 min 15 s", "1 h 2 min", "2 d 5 h", "—", "—"])
        self.assertEqual(self.js("[L.fmtTok(999),L.fmtTok(1234),L.fmtTok(12345),L.fmtTok(1234567),L.fmtTok(undefined)]"),
                         ["999", "1.2K", "12K", "1.23M", "—"])
        self.assertEqual(self.js("[L.fmtPct(54.2),L.fmtFrac(0.735,1),L.fmtUsd(20.961873),L.fmtInt(922358),L.fmtNum('x',1)]"),
                         ["54%", "73.5%", "$20.96", "922,358", "—"])

    def test_html_is_escaped(self):
        self.assertEqual(self.js("L.esc('<b onclick=\"x\">&')"), "&lt;b onclick=&quot;x&quot;&gt;&amp;")

    # ---- routing mode ----
    def test_effective_mode_is_not_the_configured_mode(self):
        cap = {"mode": "remote-only", "why": ["planned local-offline window: bench (by S4, 100s left)"], "planned_offline": True}
        m = self.js("L.effectiveMode(cap,stats,cfg)", cap=cap, stats={"mode": "local_first"}, cfg={"mode": "local_first"})
        self.assertEqual((m["key"], m["label"], m["configured"], m["level"]), ("remote-only", "Remote only", "Local first", "warn"))
        self.assertEqual(m["source"], "capacity")

    def test_effective_mode_falls_back_to_configured_when_capacity_missing(self):
        m = self.js("L.effectiveMode(null,stats,null)", stats={"mode": "full_remote"})
        self.assertEqual(m["source"], "config")
        self.assertIn("configured", m["label"])

    # ---- spend ----
    def test_spend_view_uses_the_ledger_and_counts_holds_against_the_cap(self):
        spend = {"spent": 20.0, "held": 1.0, "reserved": 0.5, "cap": 25.0, "available": 3.5, "tz": "America/Phoenix"}
        v = self.js("L.spendView(null,spend)", spend=spend)
        self.assertAlmostEqual(v["pct"], 86.0)
        self.assertEqual(v["level"], "warn")
        self.assertEqual(v["available"], 3.5)
        self.assertEqual(self.js("L.spendView(null,{spent:24,held:0.5,reserved:0,cap:25}).level"), "bad")
        self.assertEqual(self.js("L.spendView(null,{spent:2,cap:25}).level"), "ok")

    def test_spend_view_falls_back_to_capacity_remote_use(self):
        cap = {"remote_use": {"spend_today": {"spent": 5.0, "held": 0, "reserved": 0, "cap": 25.0}}}
        v = self.js("L.spendView(cap,null)", cap=cap)
        self.assertEqual((v["spent"], v["cap"], v["available"]), (5.0, 25.0, 20.0))

    def test_spend_breakdown_reports_how_much_was_not_metered(self):
        spend = {"attribution": {"accounted_usd": 10.0, "groups": [
            {"client": "a", "reason": "big-out", "basis": "actual", "usd": 4.0, "calls": 100},
            {"client": "a", "reason": "big-out", "basis": "held-fallback", "usd": 5.0, "calls": 10},
            {"client": "b", "reason": "perf", "basis": "orphan-held", "usd": 1.0, "calls": 1}]}}
        b = self.js("L.spendBreakdown(spend)", spend=spend)
        self.assertEqual(b["total"], 10.0)
        self.assertEqual(b["nonActualUsd"], 6.0)
        self.assertAlmostEqual(b["nonActualShare"], 0.6)
        self.assertEqual(b["byClient"][0], {"key": "a", "usd": 9.0, "calls": 110})
        self.assertEqual(b["byBasis"][0]["key"], "held-fallback")

    # ---- routing windows and reasons ----
    CAP = {"remote_use": {"windows": {
        "15m": {"requests": 100, "remote": 80, "remote_share": 0.8, "by_reason": {"big-out": 30, "local-offline": 40},
                "gateway_chosen_remote": 30, "remote_while_local_had_headroom": 2}}}}
    STATS = {"total": 1000, "local": 600, "remote": 300, "held": 10, "rejected_bg": 90,
             "remote_reasons": {"perf": 200, "big-prompt": 100}, "token_budget": 500000}

    def test_remote_windows_keep_requests_and_remote_separate(self):
        w = self.js("L.remoteWindows(cap,stats)", cap=self.CAP, stats=self.STATS)
        self.assertEqual(w[0]["key"], "15m")
        self.assertEqual((w[0]["requests"], w[0]["remote"], w[0]["localOrOther"]), (100, 80, 20))
        self.assertEqual((w[-1]["key"], w[-1]["held"], w[-1]["rejected"]), ("life", 10, 90))

    def test_reason_rows_add_an_other_row_when_the_gateway_itemised_fewer(self):
        r = self.js("L.reasonRows('15m',cap,stats,{big_output:16384})", cap=self.CAP, stats=self.STATS)
        keys = [x["key"] for x in r["rows"]]
        self.assertEqual(keys, ["local-offline", "big-out", "(other)"])
        self.assertEqual(r["rows"][-1]["n"], 10)                                  # 80 remote - 70 itemised
        self.assertAlmostEqual(sum(x["pct"] for x in r["rows"]), 100.0)
        self.assertTrue(r["rows"][0]["explicit"])
        self.assertFalse(r["rows"][1]["explicit"])
        self.assertEqual(r["rows"][1]["setting"], {"field": "big_output", "value": 16384})

    def test_reason_rows_all_time_come_from_the_counters(self):
        r = self.js("L.reasonRows('life',cap,stats,{})", cap=self.CAP, stats=self.STATS)
        self.assertEqual([x["key"] for x in r["rows"]], ["perf", "big-prompt"])
        self.assertEqual(r["total"], 300)
        self.assertAlmostEqual(r["rows"][0]["pct"], 200 / 3.0)

    def test_every_reason_the_gateway_emits_is_documented(self):
        src = (HERE / "keepalive-shim.py").read_text()
        emitted = set(re.findall(r'record_event\(\s*"(?:remote|rejected|rejected-bg)"\s*,\s*"([a-z-]+)"', src))
        documented = set(self.js("Object.keys(L.REASONS)"))
        self.assertEqual(sorted(emitted - documented), [], "reasons emitted by the gateway but missing from the dashboard")

    def test_token_budget_reason_never_shows_nan_when_config_says_auto(self):
        cap = {"capacity_model": {"token_budget": {"effective": 611000}}}
        info = self.js("L.reasonInfo('tokens',{token_budget:'auto'},cap,{token_budget:500000})", cap=cap)
        self.assertEqual(info["setting"], {"field": "token_budget", "value": 611000})
        info = self.js("L.reasonInfo('tokens',{token_budget:'auto'},null,{token_budget:500000})")
        self.assertEqual(info["setting"]["value"], 500000)

    def test_unknown_reason_is_flagged_not_crashed(self):
        info = self.js("L.reasonInfo('brand-new',{},null,null)")
        self.assertIn("undocumented", info["plain"])
        self.assertIsNone(info["setting"])

    def test_still_remote_rows_explain_the_policy_outcome(self):
        rows = self.js("L.stillRemoteRows(s)", s={"local_first": {"still_remote": {
            "big-out:saturated:lanes+prefill-backlog": 54, "big-out:local-unhealthy": 308, "monster:prefill-backlog": 124}}})
        self.assertEqual([r["n"] for r in rows], [308, 124, 54])
        self.assertEqual(rows[0]["reason"], "big-out")
        self.assertIn("unhealthy", rows[0]["plain"])
        self.assertIn("no free lane", rows[2]["plain"])

    # ---- work classes ----
    def test_class_map_matches_like_the_gateway_does(self):
        m = "halo-=halo,estate-entity=halo,pi /=kevin,overseer-=background,bad,x=nonsense"
        self.assertEqual(self.js("L.parseClassMap(m)", m=m), [["halo-", "halo"], ["estate-entity", "halo"], ["pi /", "kevin"], ["overseer-", "background"]])
        self.assertEqual(self.js("L.classOfClient('Halo-Hermes',L.parseClassMap(m))", m=m), "halo")
        self.assertEqual(self.js("L.classOfClient('ubuntuide01 · pi / OpenCode',L.parseClassMap(m))", m=m), "kevin")
        self.assertIsNone(self.js("L.classOfClient('stranger',L.parseClassMap(m))", m=m))

    def test_work_class_rows_merge_queue_demand_and_live_requests(self):
        cap = {"queue": {c: {"share": 10.0, "waiting": 1, "oldest_wait_s": 2.0, "expected_wait_s": 3.0, "wait_p50_s_15m": 0.5, "wait_p95_s_15m": 4.0,
                             "admitted_15m": 7, "held_by": {}, "refused_total": 0, "ceiling_backlog_s": None, "default_deadline_s": None}
                         for c in ("kevin", "halo", "runner", "background")},
               "demand": {c: {"req_per_min_5m": 1.5, "req_per_min_60m": 0.5, "prompt_tok_per_min_5m": 1000,
                              "uncached_prefill_s_per_min_5m": 2.0, "uncached_prefill_s_per_min_60m": 1.0}
                          for c in ("kevin", "halo", "runner", "background")}}
        lanes = {"active": [{"flow_class": "halo", "phase": "local", "route": "local"}, {"flow_class": "halo", "phase": "queued"},
                            {"flow_class": "runner", "phase": "remote", "route": "remote"}, {"phase": "local", "route": "local"}]}
        rows = self.js("L.workClassRows(cap,{},lanes)", cap=cap, lanes=lanes)
        self.assertEqual([r["cls"] for r in rows], ["kevin", "halo", "runner", "background"])
        halo = rows[1]
        self.assertEqual((halo["running"], halo["queuedNow"], halo["rpm5"], halo["pre60"], halo["p95"]), (1, 1, 1.5, 1.0, 4.0))
        self.assertEqual(rows[2]["running"], 0)                                   # a remote request is not occupying the engine
        self.assertEqual(self.js("L.activeByClass(lanes)['?'].running", lanes=lanes), 1)

    # ---- capacity ----
    CM = {"kv_pool_tokens": {"effective": 922358, "source": "live", "live": 922358, "configured": 637560, "configured_stale": True, "age_s": 2},
          "token_budget": {"effective": 611000, "source": "live-derived", "detail": "66.3% of the live KV pool"},
          "prefill_tok_s": {"effective": 1270.0, "source": "measured", "detail": "p75", "configured": 1100.0, "measured_p75": 1270.0,
                            "status": "ok", "valid_minutes": 12, "min_minutes": 5},
          "warnings": ["SHIM_POOL_TOKENS=637560 is stale"]}

    def test_capacity_view_prefers_the_live_model_and_labels_its_source(self):
        v = self.js("L.capacityView(stats,cap,cfg)", stats={"capacity_model": self.CM, "token_budget": 611000, "inflight": 3, "budget": 14},
                    cap={"throughput": {"prefill_effective_tok_s": 1270.0}}, cfg={"pool_tokens": 637560, "prefill_tps": 1100})
        self.assertEqual((v["pool"]["value"], v["pool"]["source"], v["pool"]["stale"]), (922358, "live", True))
        self.assertEqual((v["tokenBudget"]["value"], v["tokenBudget"]["source"]), (611000, "live-derived"))
        self.assertEqual((v["prefill"]["effective"], v["prefill"]["source"]), (1270.0, "measured"))
        self.assertEqual(v["warnings"], self.CM["warnings"])

    def test_capacity_view_labels_configured_values_as_configured_on_an_older_gateway(self):
        v = self.js("L.capacityView({token_budget:500000,inflight:1,budget:14},{},{pool_tokens:637560,prefill_tps:1100})")
        self.assertEqual((v["pool"]["source"], v["pool"]["value"]), ("configured", 637560))
        self.assertEqual((v["tokenBudget"]["source"], v["tokenBudget"]["live"]), ("configured", False))
        self.assertEqual((v["prefill"]["source"], v["prefill"]["effective"]), ("configured", 1100))
        self.assertEqual(v["warnings"], [])

    # ---- measurements from the sample series ----
    SERIES = [{"t": 100 + i, "engine": {"prefix_hit_rate": h, "prompt_tok_s": w, "gen_tok_s": g, "ttft_p50": tt}}
              for i, (h, w, g, tt) in enumerate([(0.9, 1000, 20, None), (0.5, 3000, 40, 2.0), (None, 0, 0, None), (0.1, 0, 10, 4.0)])]

    def test_hit_rate_is_weighted_by_prompt_tokens_and_ignores_silence(self):
        self.assertAlmostEqual(self.js("L.windowHitRate(s,103,60)", s=self.SERIES), (0.9 * 1000 + 0.5 * 3000) / 4000)
        self.assertIsNone(self.js("L.windowHitRate([],103,60)"))

    def test_window_means_use_only_the_window_and_skip_missing_samples(self):
        self.assertAlmostEqual(self.js("L.windowMean(s,103,60,['engine','gen_tok_s'])", s=self.SERIES), 17.5)
        self.assertAlmostEqual(self.js("L.windowMean(s,103,1.5,['engine','gen_tok_s'])", s=self.SERIES), 5.0)   # only the last two samples
        self.assertEqual(self.js("L.windowMedian(s,103,60,['engine','ttft_p50'])", s=self.SERIES), 3.0)
        self.assertIsNone(self.js("L.windowMean(s,103,60,['engine','nope'])", s=self.SERIES))

    def test_bucket_mean_downsamples_and_keeps_gaps(self):
        pts = [[i, float(i)] for i in range(10)]
        out = self.js("L.bucketMean(p,5)", p=pts)
        self.assertEqual(len(out), 5)
        self.assertEqual(out[0][1], 0.5)
        gap = self.js("L.bucketMean([[0,null],[1,null],[2,3],[3,5]],2)")
        self.assertIsNone(gap[0][1])
        self.assertEqual(gap[1][1], 4.0)

    # ---- health verdict ----
    def issues(self, **ctx):
        return self.js("L.healthIssues(ctx)", ctx=ctx)

    def test_healthy_gateway_yields_no_issues_and_an_ok_verdict(self):
        ctx = dict(stats={"local_healthy": True, "gpu": [{"index": 0, "temp_c": 70}]},
                   cap={"mode": "local+remote", "offline_window": {"offline": False}, "pressure": {"demand_over_capacity_5m": 0.4},
                        "remote_use": {"windows": {"15m": {"remote_while_local_had_headroom": 0}}, "spend_today": {"spent": 2, "cap": 25}}},
                   cfg={}, telem={"latest": {"engine": {"ok": True, "kv_cache_pct": 40}}}, spend=None)
        i = self.issues(**ctx)
        self.assertEqual(i, [])
        self.assertEqual(self.js("L.verdict([],true)")["level"], "ok")

    def test_engine_down_is_the_headline(self):
        i = self.issues(stats={"local_healthy": False}, cap={"mode": "remote-only", "why": ["local engine unhealthy"]}, cfg={}, telem=None, spend=None)
        self.assertEqual(i[0]["level"], "bad")
        v = self.js("L.verdict(i,true)", i=i)
        self.assertEqual(v["level"], "bad")
        self.assertIn("DOWN", v["text"])

    def test_offline_window_breaker_spend_and_capacity_warnings_each_surface(self):
        i = self.issues(stats={"local_healthy": True, "perf_breaker": True, "perf_breaker_reason": "gpu-saturation", "backoff": 30, "gpu": [{"index": 1, "temp_c": 90}]},
                        cap={"mode": "remote-only", "why": ["planned"], "offline_window": {"offline": True, "reason": "bench", "by": "S4", "remaining_s": 600},
                             "capacity_model": {"warnings": ["pool stale"]}, "cannot_measure": ["engine /metrics scrape failing"],
                             "pressure": {"demand_over_capacity_5m": 1.4}},
                        cfg={}, telem=None, spend={"spent": 24, "held": 0, "reserved": 0, "cap": 25})
        keys = {x["key"] for x in i}
        self.assertTrue({"mode", "offline", "perf", "oom", "gputemp", "spend", "capmodel", "cannot", "demand"} <= keys, keys)
        self.assertEqual([x for x in i if x["key"] == "gputemp"][0]["level"], "bad")
        self.assertEqual(self.js("L.verdict(i,true).level", i=i), "bad")

    def test_a_stale_feed_is_reported(self):
        i = self.issues(stats={"local_healthy": True}, cap=None, cfg={}, telem=None, spend=None, feedAges=[["stats", 60, 2], ["capacity", 1, 3]])
        self.assertEqual([x["key"] for x in i if x["key"].startswith("stale")], ["stale-stats"])

    def test_no_data_at_all_is_not_reported_as_healthy(self):
        self.assertEqual(self.issues(stats=None, cap=None, cfg=None, telem=None, spend=None), [])      # still connecting
        i = self.issues(stats=None, cap=None, cfg=None, telem=None, spend=None, failing=True)
        self.assertEqual(i[0]["key"], "nodata")
        self.assertEqual(self.js("L.verdict([],false).level"), "unk")

    # ---- charts ----
    def test_line_chart_is_valid_svg_without_nan_and_handles_no_data(self):
        spec = {"title": "t", "t0": 0, "t1": 100, "series": [{"name": "a", "slot": 1, "points": [[0, 1], [50, None], [60, 3], [100, 2]]}]}
        svg = self.js("L.lineChart(spec)", spec=spec)
        self.assertTrue(svg.startswith("<svg") and svg.endswith("</svg>"))
        self.assertNotIn("NaN", svg)
        self.assertNotIn("Infinity", svg)
        self.assertIn("polyline", svg)
        self.assertIn(">now<", svg)
        empty = self.js("L.lineChart(spec)", spec=dict(spec, series=[{"name": "a", "points": [[0, None]]}]))
        self.assertIn("No samples", empty)
        self.assertNotIn("NaN", empty)

    def test_line_chart_gaps_break_the_line(self):
        pts = [[0, 1], [1, 1], [2, 1], [3, 1], [50, 2], [51, 2], [52, 2]]
        svg = self.js("L.lineChart(spec)", spec={"title": "t", "t0": 0, "t1": 60, "series": [{"name": "a", "points": pts}]})
        self.assertEqual(svg.count("<polyline"), 2)


class SafePublishShipsTheDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("gateway_safe_publish_dash_test", HERE / "gateway_safe_publish.py")
        cls.pub = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.pub)

    def test_page_is_installed_next_to_the_live_shim_and_can_be_undone(self):
        pub = self.pub
        with tempfile.TemporaryDirectory() as d:
            runtime = Path(d) / "keepalive-shim.py"
            with patch.object(pub, "DASH_RUNTIME", runtime.with_name("gateway_dashboard.html")):
                first = pub._install_dashboard(b"<html>v1</html>")
                self.assertEqual((first["dashboard"], first["previous"]), ("installed", None))
                self.assertEqual(pub.DASH_RUNTIME.read_bytes(), b"<html>v1</html>")
                self.assertEqual(pub.DASH_RUNTIME.stat().st_mode & 0o777, 0o644)
                self.assertEqual(pub._install_dashboard(b"<html>v1</html>")["dashboard"], "current")
                second = pub._install_dashboard(b"<html>v2</html>")
                pub._restore_dashboard(second)
                self.assertEqual(pub.DASH_RUNTIME.read_bytes(), b"<html>v1</html>")
                pub._restore_dashboard(first)                 # first install undone: the shim falls back to its inline page
                self.assertFalse(pub.DASH_RUNTIME.exists())

    def test_the_shim_finds_the_page_beside_itself_by_default(self):
        self.assertEqual(self.pub.DASH_RUNTIME.name, "gateway_dashboard.html")
        self.assertEqual(self.pub.DASH_RUNTIME.parent, self.pub.RUNTIME.parent)
        src = (HERE / "keepalive-shim.py").read_text()
        self.assertIn('os.path.join(os.path.dirname(os.path.abspath(__file__)), "gateway_dashboard.html")', src)


if __name__ == "__main__":
    unittest.main()
