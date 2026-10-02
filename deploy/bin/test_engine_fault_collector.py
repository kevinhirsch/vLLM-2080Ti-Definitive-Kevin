"""RS (2026-10-02): the fault collector must record a watchdog wedge kill exactly once, even though ExecStopPost races the restart."""
import json, os, subprocess, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
COLLECTOR = os.path.join(HERE, "engine-fault-collector.py")


def run(base, *args, **env):
    e = dict(os.environ, FAULT_COLLECTOR_BASE=base, **env)
    return subprocess.run([sys.executable, COLLECTOR, "--no-vault", *args], env=e, capture_output=True, text=True, timeout=120)


class CollectorRace(unittest.TestCase):
    def test_pre_kill_records_once_and_stop_post_dedupes(self):
        with tempfile.TemporaryDirectory() as base:
            json.dump({"ts": "x", "by": "watchdog", "wedge": True, "consecutive_failures": 5}, open(f"{base}/wedge-restart.json", "w"))
            r = run(base, "--pre-kill")
            self.assertEqual(r.returncode, 0, r.stderr)
            rows = [json.loads(l) for l in open(f"{base}/incidents/ledger.jsonl")]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["signature"], "generation-wedge")
            self.assertEqual(rows[0]["kind"], "FAULT")
            self.assertEqual(rows[0]["recorded_by"], "watchdog-pre-kill")
            self.assertTrue(os.path.exists(f"{base}/recorded-fault.json"))
            # the later ExecStopPost for the same death must not add a second row
            r = run(base, "--stop-post", SERVICE_RESULT="signal", EXIT_STATUS="KILL")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(len(open(f"{base}/incidents/ledger.jsonl").read().splitlines()), 1)

    def test_stop_post_without_pre_kill_still_records(self):
        with tempfile.TemporaryDirectory() as base:
            r = run(base, "--stop-post", SERVICE_RESULT="exit-code", EXIT_STATUS="1")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(len(open(f"{base}/incidents/ledger.jsonl").read().splitlines()), 1)

    def test_backfill_does_not_consume_live_markers(self):
        with tempfile.TemporaryDirectory() as base:
            json.dump({"by": "watchdog", "wedge": True}, open(f"{base}/wedge-restart.json", "w"))
            run(base, "--at", "2026-10-01 10:00:00", "--reconciled", "--cause", "t")
            self.assertTrue(os.path.exists(f"{base}/wedge-restart.json"))


if __name__ == "__main__":
    unittest.main()
