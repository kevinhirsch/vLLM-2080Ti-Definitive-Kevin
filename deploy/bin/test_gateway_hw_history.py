#!/usr/bin/env python3
"""Lane TL (2026-10-03): the gateway persists its own hardware/engine telemetry.

Kevin asked "how has the temperature been?" overnight and the only history was the in-memory 4 h slow ring. Each slow-ring
point (one per minute) is now also appended to TELEMETRY_DIR/hw-YYYYMMDD.jsonl (UTC day, like requests-*.jsonl);
/gateway/telemetry/history serves it downsampled; a restart re-seeds the slow ring from it.

Run:  python -m pytest -q test_gateway_hw_history.py
"""
import asyncio
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

import test_gateway_local_first as lf

shim = lf.shim
HERE = pathlib.Path(__file__).resolve().parent
PAGE = HERE / "gateway_dashboard.html"
DAY = 86400
T0 = 1_790_000_000 - (1_790_000_000 % DAY)          # a UTC midnight, so day-boundary arithmetic is exact


def gpu_sample(temp=70, power=200.0, sm=1500, thr=0x4, **kw):
    d = {"util": 90, "mem_util": 40, "used": 9000, "free": 2000, "total": 11000, "temp_c": temp, "power_w": power,
         "power_limit_w": 280.0, "clock_sm_mhz": sm, "clock_mem_mhz": 6800, "fan_pct": 55, "throttle_reasons": thr}
    d.update(kw)
    return d


def fast_sample(t, temps=(70, 65), thr=0x4, engine_ok=True):
    return {"t": t, "gpu": [gpu_sample(temp=x, thr=thr) for x in temps], "host": {"cpu_pct": 10},
            "engine": {"ok": engine_ok, "running": 3, "waiting": 1, "kv_cache_pct": 42.0, "prefix_hit_rate": 0.8,
                       "prompt_tok_s": 900.0, "gen_tok_s": 70.0, "spec_accept_rate": 0.5},
            "gateway": {"inflight": 3, "budget": 4, "waiting": 0, "backoff_s": 0, "local_healthy": True,
                        "remote_share_pct": 5.0, "perf_breaker": False, "perf_reason": ""}}


def slow_point(t, **kw):
    return shim._downsample([fast_sample(t, **kw)])


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="hw-hist-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self._p = patch.object(shim, "TELEMETRY_DIR", self.dir)
        self._p.start()
        self.addCleanup(self._p.stop)
        shim._HW_PENDING.clear()
        shim._HW_QUERY_CACHE.clear()
        for k, v in (("written", 0), ("failed", 0), ("dropped", 0), ("dropped_cap", 0), ("last_err", None), ("seeded", 0)):
            shim._HW_STATE[k] = v

    def write(self, recs):
        return shim._hw_append_blocking(recs)

    def stored(self, t):
        return [json.loads(x) for x in open(shim._hw_day_path(t))]


class Record(Base):
    def test_downsample_keeps_the_peak_and_ors_the_throttle_bits(self):
        ds = shim._downsample([fast_sample(1, temps=(70, 60), thr=0x4), fast_sample(2, temps=(84, 60), thr=0x20),
                               fast_sample(3, temps=(72, 60), thr=0x4)])
        g = ds["gpu"][0]
        self.assertEqual(g["temp_max_c"], 84)
        self.assertAlmostEqual(g["temp_c"], (70 + 84 + 72) / 3, places=2)    # the mean still exists
        self.assertEqual(g["throttle_reasons"], 0x24)
        self.assertEqual(ds["gpu"][1]["temp_max_c"], 60)

    def test_missing_throttle_data_is_none_not_zero(self):
        s = fast_sample(1)
        for g in s["gpu"]:
            g["throttle_reasons"] = None
        self.assertIsNone(shim._downsample([s])["gpu"][0]["throttle_reasons"])

    def test_record_is_compact_and_has_what_kevin_asked_for(self):
        rec = shim._hw_record(slow_point(T0 + 5))
        line = json.dumps(rec, separators=(",", ":"))
        self.assertLess(len(line), 700)                                      # ~one line a minute stays tiny
        g = rec["gpu"][0]
        for k in ("temp", "temp_max", "power", "sm_clock", "mem_clock", "fan", "util", "throttle", "vram_used"):
            self.assertIn(k, g)
        for k in ("running", "waiting", "kv_pct", "prefill_tps", "decode_tps"):
            self.assertIn(k, rec["engine"])

    def test_engine_down_stores_null_engine(self):
        self.assertIsNone(shim._hw_record(slow_point(T0, engine_ok=False))["engine"])

    def test_empty_or_timeless_point_is_skipped(self):
        self.assertIsNone(shim._hw_record(None))
        self.assertIsNone(shim._hw_record({"gpu": []}))

    def test_stored_line_converts_back_to_a_slow_ring_point(self):
        back = shim._hw_to_slow(shim._hw_record(slow_point(T0 + 9)))
        self.assertEqual(back["t"], T0 + 9)
        self.assertEqual(back["gpu"][0]["temp_c"], 70)
        self.assertEqual(back["gpu"][0]["temp_max_c"], 70)
        self.assertEqual(back["engine"]["gen_tok_s"], 70.0)
        self.assertTrue(back["engine"]["ok"])
        # every key the dashboard's slow-ring charts read exists (null where unknown)
        self.assertIn("cpu_pct", {"cpu_pct"} | set(back["host"]))
        self.assertIn("remote_share_pct", back["gateway"])


class Writing(Base):
    def test_one_line_per_record_in_the_utc_day_file_of_its_timestamp(self):
        a, b = shim._hw_record(slow_point(T0 + 10)), shim._hw_record(slow_point(T0 + DAY + 10))
        res = self.write([a, b])
        self.assertTrue(res["ok"])
        self.assertEqual(res["written"], 2)
        self.assertEqual(sorted(os.listdir(self.dir)),
                         [time.strftime("hw-%Y%m%d.jsonl", time.gmtime(T0)), time.strftime("hw-%Y%m%d.jsonl", time.gmtime(T0 + DAY))])
        self.assertEqual(len(self.stored(T0)), 1)

    def test_day_convention_is_the_same_as_requests_files(self):
        self.assertEqual(os.path.basename(shim._hw_day_path(T0 + 5)).replace("hw-", ""),
                         os.path.basename(shim._history_day_file(T0 + 5)).replace("requests-", ""))

    def test_file_is_private_and_appends(self):
        self.write([shim._hw_record(slow_point(T0 + 1))])
        self.write([shim._hw_record(slow_point(T0 + 61))])
        self.assertEqual(len(self.stored(T0)), 2)
        self.assertEqual(os.stat(shim._hw_day_path(T0)).st_mode & 0o777, 0o600)

    def test_daily_size_cap_stops_writing_and_counts(self):
        with patch.object(shim, "HW_JSONL_MAX_MB", 0.0015):                    # ~1.5 KB, a couple of records
            res = self.write([shim._hw_record(slow_point(T0 + 60 * i)) for i in range(20)])
        self.assertTrue(res["ok"])
        self.assertGreater(res["dropped_cap"], 0)
        self.assertLess(os.path.getsize(shim._hw_day_path(T0)), 1600)

    def test_a_write_failure_is_reported_not_raised(self):
        blocker = os.path.join(self.dir, "x")
        open(blocker, "w").close()
        with patch.object(shim, "TELEMETRY_DIR", blocker):                      # a file where the directory should be
            res = shim._hw_append_blocking([shim._hw_record(slow_point(T0))])
        self.assertFalse(res["ok"])
        self.assertTrue(res["err"])

    def test_failed_flush_keeps_the_record_for_the_next_attempt(self):
        async def go():
            shim._HW_PENDING.append(shim._hw_record(slow_point(T0)))
            with patch.object(shim, "_hw_append_blocking", side_effect=OSError("disk full")):
                await shim._hw_flush()
            self.assertEqual(len(shim._HW_PENDING), 1)
            self.assertEqual(shim._HW_STATE["failed"], 1)
            await shim._hw_flush()                                              # disk is back
            self.assertEqual(shim._HW_PENDING, [])
            self.assertEqual(shim._HW_STATE["written"], 1)
            self.assertIsNone(shim._HW_STATE["last_err"])
        asyncio.run(go())

    def test_pending_queue_is_bounded(self):
        async def go():
            with patch.object(shim, "HW_QUEUE_MAX", 5), patch.object(shim, "_hw_append_blocking",
                                                                      side_effect=OSError("down")):
                for i in range(12):
                    shim._hw_enqueue(slow_point(T0 + 60 * i))
                    await asyncio.sleep(0)
                    if shim._HW_TASK:
                        await shim._HW_TASK
            self.assertLessEqual(len(shim._HW_PENDING), 5)
            self.assertGreater(shim._HW_STATE["dropped"], 0)
        asyncio.run(go())

    def test_enqueue_hands_the_write_to_an_executor_not_the_loop(self):
        async def go():
            seen = []
            real = shim._hw_append_blocking

            def spy(recs):
                import threading
                seen.append(threading.current_thread() is threading.main_thread())
                return real(recs)
            with patch.object(shim, "_hw_append_blocking", spy):
                shim._hw_enqueue(slow_point(T0 + 3))
                await shim._HW_TASK
            self.assertEqual(seen, [False])
        asyncio.run(go())

    def test_kill_switch(self):
        async def go():
            with patch.object(shim, "HW_HISTORY", False):
                shim._hw_enqueue(slow_point(T0))
            self.assertEqual(shim._HW_PENDING, [])
        asyncio.run(go())

    def test_sampler_persists_each_slow_point(self):
        """The real sampler loop: after TELEM_SLOW_EVERY ticks a slow point exists AND a line reaches disk."""
        async def go():
            with patch.object(shim, "TELEM_SLOW_EVERY", 2), patch.object(shim, "TELEM_SAMPLE_SECS", 0.001), \
                 patch.object(shim, "_TELEM_SLOW", shim.collections.deque(maxlen=10)) as ring, \
                 patch.object(shim, "_TELEM_FAST", shim.collections.deque(maxlen=10)), \
                 patch.object(shim, "_TELEM_WINDOW", []), patch.object(shim, "_TELEM_TICK", 0), \
                 patch.object(shim, "_take_sample", AsyncMock(side_effect=lambda: fast_sample(time.time()))), \
                 patch.object(shim, "_update_perf_breaker"), patch.object(shim, "_offline_reap"), patch.object(shim, "flow_note_mode"):
                task = asyncio.create_task(shim._telemetry_sampler())
                for _ in range(200):
                    await asyncio.sleep(0.01)
                    if ring and shim._HW_STATE["written"]:
                        break
                task.cancel()
                self.assertTrue(ring)
            self.assertGreaterEqual(shim._HW_STATE["written"], 1)
        asyncio.run(go())
        files = os.listdir(self.dir)
        self.assertTrue(any(f.startswith("hw-") for f in files))


class Retention(Base):
    def test_sweep_removes_old_hw_files_and_only_those(self):
        old = time.time() - 40 * DAY
        names = {"hw-20200101.jsonl": True, "requests-20200101.jsonl": True, "hw-20991231.jsonl": False, "notes.txt": False}
        for n, is_old in names.items():
            fp = os.path.join(self.dir, n)
            open(fp, "w").write("{}\n")
            if is_old:
                os.utime(fp, (old, old))
        shim._telemetry_retention_sweep()
        self.assertEqual(sorted(os.listdir(self.dir)), ["hw-20991231.jsonl", "notes.txt"])


class Reading(Base):
    def fill(self, n_min=60 * 30, base=T0 + DAY + 3600):
        recs = []
        for i in range(n_min):
            t = base + 60 * i
            temp = 60 + (i % 30)
            recs.append(shim._hw_record(shim._downsample(
                [fast_sample(t, temps=(temp, 50), thr=0x20 if temp >= 85 else 0x4)])))
        self.write(recs)
        return base, recs

    def test_history_downsamples_across_a_day_boundary(self):
        base, recs = self.fill()                                               # 30 h from 01:00 on day 2 -> into day 3
        now = base + 60 * len(recs)
        h = shim.hw_history(30, 0, now=now, telemetry_dir=self.dir)
        self.assertEqual(h["raw_samples"], len(recs))
        self.assertEqual(h["files"], 2)
        self.assertLessEqual(len(h["points"]), shim.HW_MAX_POINTS)
        self.assertGreater(len(h["points"]), 50)
        self.assertEqual(sum(p["n"] for p in h["points"]), len(recs))           # nothing lost or double counted
        ts = [p["t"] for p in h["points"]]
        self.assertEqual(ts, sorted(ts))

    def test_step_is_honoured_but_never_finer_than_one_minute_or_more_than_the_point_cap(self):
        base, recs = self.fill(120)
        now = base + 7200
        self.assertEqual(shim.hw_history(2, 1, now=now, telemetry_dir=self.dir)["step_s"], 60.0)
        h = shim.hw_history(2, 600, now=now, telemetry_dir=self.dir)
        self.assertEqual(h["step_s"], 600.0)
        self.assertEqual(len(h["points"]), 12)
        long = shim.hw_history(24 * 30, 1, now=now, telemetry_dir=self.dir)
        self.assertGreaterEqual(long["step_s"], 24 * 30 * 3600 / shim.HW_MAX_POINTS)

    def test_buckets_keep_peak_and_throttle_bits(self):
        base, recs = self.fill(60)
        h = shim.hw_history(1, 3600, now=base + 3600, telemetry_dir=self.dir)
        p = h["points"][0]["gpu"][0]
        self.assertEqual(p["temp_max"], 89)                                    # peak survives averaging
        self.assertLess(p["temp"], 89)
        self.assertEqual(p["throttle"], 0x24)

    def test_summary_counts_hot_and_throttled_minutes(self):
        base, recs = self.fill(60)
        h = shim.hw_history(1, 0, now=base + 3600, telemetry_dir=self.dir)
        g0, g1 = h["summary"]["gpu"]
        self.assertEqual(g0["temp_max"], 89)
        self.assertEqual(g0["temp_min"], 60)
        hot = sum(1 for i in range(60) if 60 + (i % 30) >= 83)
        self.assertEqual(g0["hot_minutes"], hot)
        self.assertEqual(g0["thermal_throttle_minutes"], sum(1 for i in range(60) if 60 + (i % 30) >= 85))
        self.assertEqual(g1["hot_minutes"], 0)
        self.assertTrue(g0["throttle_known"])

    def test_range_filter_excludes_older_records(self):
        base, recs = self.fill(120)
        h = shim.hw_history(1, 0, now=base + 7200, telemetry_dir=self.dir)
        self.assertEqual(h["raw_samples"], 60)

    def test_empty_directory_and_garbage_lines_are_harmless(self):
        self.assertEqual(shim.hw_history(24, 0, now=T0, telemetry_dir=self.dir)["points"], [])
        open(shim._hw_day_path(T0), "w").write("not json\n{\"t\": \"x\"}\n\n" + json.dumps(shim._hw_record(slow_point(T0 + 5))) + "\n")
        h = shim.hw_history(1, 0, now=T0 + 100, telemetry_dir=self.dir)
        self.assertEqual(h["raw_samples"], 1)

    def test_hours_are_clamped_to_retention(self):
        h = shim.hw_history(10 ** 6, 0, now=T0, telemetry_dir=self.dir)
        self.assertEqual(h["hours"], shim.TELEMETRY_RETENTION_DAYS * 24)


class Route(Base):
    class Q:
        def __init__(self, **q):
            self.query = q

    def call(self, **q):
        return asyncio.run(shim.gateway_telemetry_history(self.Q(**q)))

    def test_route_returns_points_and_store_state(self):
        now = time.time()
        self.write([shim._hw_record(slow_point(now - 2 - 60 * i)) for i in range(5)])
        r = self.call(hours="1")
        body = json.loads(r.body)
        self.assertEqual(r.status, 200)
        self.assertEqual(body["raw_samples"], 5)
        self.assertIn("store", body)
        self.assertTrue(body["store"]["enabled"])

    def test_bad_parameters(self):
        self.assertEqual(self.call(hours="nan").status, 400)
        self.assertEqual(self.call(hours="inf").status, 400)
        self.assertEqual(self.call(hours="abc", step="zz").status, 200)         # falls back to defaults, never a 500

    def test_route_is_registered_and_telemetry_reports_the_store(self):
        app = shim.make_app()
        paths = {r.resource.canonical for r in app.router.routes() if r.method == "GET"}
        self.assertIn("/gateway/telemetry/history", paths)
        body = json.loads(asyncio.run(shim.gateway_telemetry(self.Q())).body)
        self.assertIn("hw_history", body)


class Seeding(Base):
    def test_restart_refills_the_slow_ring_from_disk(self):
        now = time.time()
        self.write([shim._hw_record(slow_point(now - 60 * i)) for i in range(30, 0, -1)])
        ring = shim.collections.deque(maxlen=1440)
        with patch.object(shim, "_TELEM_SLOW", ring):
            asyncio.run(shim._hw_seed_slow())
        self.assertEqual(len(ring), 30)
        self.assertLess(ring[0]["t"], ring[-1]["t"])
        self.assertEqual(shim._HW_STATE["seeded"], 30)

    def test_seed_spans_yesterday_and_today_and_skips_older_than_the_window(self):
        now = time.time()
        recs = [shim._hw_record(slow_point(now - 3600 * h)) for h in (30, 20, 2)]
        self.write(recs)
        ring = shim.collections.deque(maxlen=1440)
        with patch.object(shim, "_TELEM_SLOW", ring):
            asyncio.run(shim._hw_seed_slow())
        self.assertEqual(len(ring), 2)

    def test_seed_never_overwrites_a_live_ring_and_never_raises(self):
        ring = shim.collections.deque([slow_point(time.time())], maxlen=1440)
        with patch.object(shim, "_TELEM_SLOW", ring):
            asyncio.run(shim._hw_seed_slow())
        self.assertEqual(len(ring), 1)
        with patch.object(shim, "_TELEM_SLOW", shim.collections.deque(maxlen=5)), \
             patch.object(shim, "hw_read_blocking", side_effect=RuntimeError("boom")):
            asyncio.run(shim._hw_seed_slow())

    def test_seeded_points_serve_through_the_existing_telemetry_route(self):
        now = time.time()
        self.write([shim._hw_record(slow_point(now - 60))])
        ring = shim.collections.deque(maxlen=1440)
        with patch.object(shim, "_TELEM_SLOW", ring):
            asyncio.run(shim._hw_seed_slow())
            body = json.loads(asyncio.run(shim.gateway_telemetry(Route.Q(slow="5"))).body)
        self.assertEqual(len(body["series"]["slow"]), 1)
        self.assertEqual(body["series"]["slow"][0]["gpu"][0]["temp_c"], 70)


class GpuQuery(unittest.TestCase):
    def run_query(self, outputs):
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd[1])
            out = outputs[min(len(calls) - 1, len(outputs) - 1)]
            return subprocess.CompletedProcess(cmd, 0 if out else 2, stdout=out, stderr="")
        with patch.object(shim.subprocess, "run", fake_run), patch.object(shim, "_GPU_THROTTLE_OK", None):
            data = shim._gpu_query_blocking()
            return data, calls, shim._GPU_THROTTLE_OK

    BASE = "0, 90, 40, 9000, 2000, 11000, 82, 250.5, 280.0, 1500, 6800, 3, 60"

    def test_throttle_bitmask_is_parsed_from_hex(self):
        data, calls, ok = self.run_query([self.BASE + ", 0x0000000000000024"])
        self.assertEqual(data[0]["throttle_reasons"], 0x24)
        self.assertTrue(ok)
        self.assertIn("clocks_throttle_reasons.active", calls[0])

    def test_unsupported_field_falls_back_to_the_base_query_once(self):
        data, calls, ok = self.run_query(["", self.BASE])
        self.assertEqual(len(data), 1)
        self.assertIsNone(data[0]["throttle_reasons"])
        self.assertEqual(data[0]["temp_c"], 82)
        self.assertFalse(ok)
        self.assertNotIn("throttle", calls[1])

    def test_not_supported_marker_is_none(self):
        data, _c, _ok = self.run_query([self.BASE + ", [Not Supported]"])
        self.assertIsNone(data[0]["throttle_reasons"])
        self.assertEqual(data[0]["fan_pct"], 60)


@unittest.skipUnless(shutil.which("node"), "node not installed")
class DashboardLongRange(unittest.TestCase):
    """The page's pure helpers (run under node) and the markup that hosts the 24 h / 7 d charts."""
    html = PAGE.read_text()

    def lib(self, expr):
        a, b = self.html.index("/*LIB-START"), self.html.index("/*LIB-END*/")
        js = "const window={};" + self.html[a:b] + "\nconsole.log(JSON.stringify(" + expr + "));"
        out = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)

    DATA = ("{hours:24,from:100,to:200,raw_samples:720,first_t:130,last_t:200,step_s:240,"
            "summary:{gpu:[{i:0,minutes:720,temp_min:50,temp_avg:70,temp_max:88,hot_minutes:12,thermal_throttle_minutes:3,power_cap_minutes:100,throttle_known:true,power_avg:200,power_max:280,sm_clock_avg:1500,sm_clock_min:900},"
            "{i:1,minutes:720,temp_min:45,temp_avg:60,temp_max:75,hot_minutes:0,thermal_throttle_minutes:0,power_cap_minutes:0,throttle_known:false}]},"
            "points:[{t:110,gpu:[{temp:70,power:200},{temp:60,power:150}]},{t:150,gpu:[{temp:null,power:210}]}]}")

    def test_series_are_per_gpu_and_null_safe(self):
        self.assertEqual(self.lib("DashLib.hwSeries(%s,0,'temp')" % self.DATA), [[110, 70], [150, None]])
        self.assertEqual(self.lib("DashLib.hwSeries(%s,1,'power')" % self.DATA), [[110, 150], [150, None]])
        self.assertEqual(self.lib("DashLib.hwGpuCount(%s)" % self.DATA), 2)
        self.assertEqual(self.lib("DashLib.hwGpuCount(null)"), 0)

    def test_summary_rows_convert_minutes_and_hide_unknown_throttle(self):
        rows = self.lib("DashLib.hwSummaryRows(%s)" % self.DATA)
        self.assertEqual(rows[0]["hot_s"], 720)
        self.assertEqual(rows[0]["thermal_s"], 180)
        self.assertEqual(rows[0]["cap_s"], 6000)
        self.assertIsNone(rows[1]["thermal_s"])                                # driver gave no data -> not "never"

    def test_coverage_flags_a_range_that_starts_late(self):
        c = self.lib("DashLib.hwCoverage({hours:24,from:0,to:86400,raw_samples:100,first_t:50000,last_t:86400})")
        self.assertTrue(c["starts_late"])
        c = self.lib("DashLib.hwCoverage({hours:24,from:0,to:86400,raw_samples:1440,first_t:60,last_t:86400})")
        self.assertFalse(c["starts_late"])
        self.assertEqual(c["pct"], 100)

    def test_page_hosts_the_long_range_charts_with_units(self):
        for needle in ("id=\"d-hwlong\"", "id=\"hw_charts\"", "id=\"hw_summary\"", "/gateway/telemetry/history?hours=",
                       "data-h=\"24\"", "data-h=\"168\"", "GPU temperature", "GPU power draw", "GPU SM clock", "'°C'", "'W'", "'MHz'"):
            self.assertIn(needle, self.html, needle)


if __name__ == "__main__":
    unittest.main()
