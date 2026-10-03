"""LV (2026-10-03): a death inside a liveness engine hold is attributed to the holder (the window that owned the engine)."""
import json, os, subprocess, sys, tempfile, time, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
COLLECTOR = os.path.join(HERE, "engine-fault-collector.py")


def run(base, *args, **env):
    e = dict(os.environ, FAULT_COLLECTOR_BASE=base, **env)
    return subprocess.run([sys.executable, COLLECTOR, "--no-vault", *args], env=e, capture_output=True, text=True, timeout=120)


class HoldAttribution(unittest.TestCase):
    def test_death_inside_hold_carries_the_holder(self):
        with tempfile.TemporaryDirectory() as base:
            now = time.time()
            json.dump([{"kind": "engine", "by": "DFT", "reason": "fine-tune window", "lease": "L1", "acquired": now - 60,
                        "until": now + 600}], open(f"{base}/liveness-holds.json", "w"))
            r = run(base, "--stop-post", SERVICE_RESULT="exit-code", EXIT_STATUS="1")
            self.assertEqual(r.returncode, 0, r.stderr)
            row = json.loads(open(f"{base}/incidents/ledger.jsonl").read().splitlines()[-1])
            self.assertEqual(row["during_hold"]["by"], "DFT")
            if row["kind"] == "planned-stop":
                self.assertEqual(row["planned_by"], "DFT")

    def test_expired_hold_is_ignored(self):
        with tempfile.TemporaryDirectory() as base:
            now = time.time()
            json.dump([{"kind": "engine", "by": "DFT", "reason": "old window", "acquired": now - 900, "until": now - 60}],
                      open(f"{base}/liveness-holds.json", "w"))
            run(base, "--stop-post", SERVICE_RESULT="exit-code", EXIT_STATUS="1")
            row = json.loads(open(f"{base}/incidents/ledger.jsonl").read().splitlines()[-1])
            self.assertNotIn("during_hold", row)


if __name__ == "__main__":
    unittest.main()
