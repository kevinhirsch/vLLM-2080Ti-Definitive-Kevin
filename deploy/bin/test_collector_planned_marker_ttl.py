"""L180: planned-restart.json had no TTL. A marker left behind by a stop that never happened (engine already down, stop
refused, window aborted) turned a LATER real crash into a 'planned-stop' (10-02/10-03 rows; the 09:21 K5 window marker).
The collector must ignore a marker older than the stop it claims, and must record that it did."""
import json, os, subprocess, sys, tempfile, time, unittest
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
COLLECTOR = os.path.join(HERE, "engine-fault-collector.py")


def run(base, *args, **env):
    e = dict(os.environ, FAULT_COLLECTOR_BASE=base, **env)
    return subprocess.run([sys.executable, COLLECTOR, "--no-vault", *args], env=e, capture_output=True, text=True, timeout=120)


def iso(t):
    return datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds")


def last(base):
    return json.loads(open(f"{base}/incidents/ledger.jsonl").read().splitlines()[-1])


class PlannedMarkerTtl(unittest.TestCase):
    def marker(self, base, age_s, **extra):
        m = {"ts": iso(time.time() - age_s), "by": "K5", "reason": "window engine stop", "drain": {}}
        m.update(extra)
        json.dump(m, open(f"{base}/planned-restart.json", "w"))

    def test_fresh_marker_still_makes_a_planned_stop(self):
        with tempfile.TemporaryDirectory() as base:
            self.marker(base, 20)
            r = run(base, "--stop-post", SERVICE_RESULT="success", EXIT_STATUS="0", FAULT_COLLECTOR_MAIN_START=str(time.time() - 3600))
            self.assertEqual(r.returncode, 0, r.stderr)
            row = last(base)
            self.assertEqual(row["kind"], "planned-stop")
            self.assertNotIn("stale_planned_marker", row)

    def test_marker_older_than_ten_minutes_is_ignored_and_the_crash_is_a_fault(self):
        with tempfile.TemporaryDirectory() as base:
            self.marker(base, 11 * 60)
            run(base, "--stop-post", SERVICE_RESULT="exit-code", EXIT_STATUS="1", FAULT_COLLECTOR_MAIN_START=str(time.time() - 3600))
            row = last(base)
            self.assertEqual(row["kind"], "FAULT")
            self.assertEqual(row["stale_planned_marker"]["by"], "K5")
            self.assertEqual(row["stale_planned_marker"]["why"], "older-than-bound")
            # consumed once, and kept for diagnosis under a name that says why it was not honoured
            self.assertFalse(os.path.exists(f"{base}/planned-restart.json"))
            self.assertTrue(os.path.exists(f"{base}/planned-restart.stale.json"))

    def test_marker_written_before_this_engine_boot_is_ignored_even_if_young(self):
        """The 10-boot crash loop: a marker 3 min old, but the engine that just died started AFTER it was written."""
        with tempfile.TemporaryDirectory() as base:
            self.marker(base, 180)
            run(base, "--stop-post", SERVICE_RESULT="exit-code", EXIT_STATUS="1", FAULT_COLLECTOR_MAIN_START=str(time.time() - 60))
            row = last(base)
            self.assertEqual(row["kind"], "FAULT")
            self.assertEqual(row["stale_planned_marker"]["why"], "predates-this-boot")

    def test_unparseable_ts_falls_back_to_file_mtime(self):
        with tempfile.TemporaryDirectory() as base:
            self.marker(base, 0, ts="x")
            run(base, "--stop-post", SERVICE_RESULT="success", EXIT_STATUS="0", FAULT_COLLECTOR_MAIN_START=str(time.time() - 3600))
            self.assertEqual(last(base)["kind"], "planned-stop")
            with tempfile.TemporaryDirectory() as base2:
                self.marker(base2, 0, ts="x")
                old = time.time() - 3600
                os.utime(f"{base2}/planned-restart.json", (old, old))
                run(base2, "--stop-post", SERVICE_RESULT="exit-code", EXIT_STATUS="1", FAULT_COLLECTOR_MAIN_START=str(time.time() - 7200))
                self.assertEqual(last(base2)["kind"], "FAULT")

    def test_bound_is_overridable_and_registered(self):
        with tempfile.TemporaryDirectory() as base:
            self.marker(base, 11 * 60)
            run(base, "--stop-post", SERVICE_RESULT="success", EXIT_STATUS="0", FAULT_COLLECTOR_MAIN_START=str(time.time() - 3600),
                FAULT_COLLECTOR_PLANNED_MARKER_MAX_S="1800")
            self.assertEqual(last(base)["kind"], "planned-stop")
        doc = open(os.path.join(HERE, "..", "docs", "engine-liveness-authority.md")).read()
        self.assertIn("FAULT_COLLECTOR_PLANNED_MARKER_MAX_S", doc)


class ActuatorDropsItsOwnUnconsumedMarker(unittest.TestCase):
    def test_drop_only_when_still_mine(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("engine_actuator_l180", os.path.join(HERE, "engine-actuator.py"))
        ea = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ea)
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/planned-restart.json"
            mine = {"ts": "2026-10-04T01:00:00-07:00", "by": "halo", "reason": "r"}
            json.dump({"ts": "2026-10-04T01:09:00-07:00", "by": "S4"}, open(path, "w"))
            self.assertFalse(ea.drop_planned_marker(mine, path))
            self.assertTrue(os.path.exists(path))
            json.dump(mine, open(path, "w"))
            self.assertTrue(ea.drop_planned_marker(mine, path))
            self.assertFalse(os.path.exists(path))
            self.assertFalse(ea.drop_planned_marker(mine, path))    # already consumed by the collector: nothing to do


if __name__ == "__main__":
    unittest.main()
