"""BS3: every engine death gets a ledger row -- the ExecStopPost race.

Live 2026-10-03 19:55 .. 10-04 06:07: four liveness-authority recovers (`sudo systemctl kill -s KILL vllm-qwen27b`, then restart). The unit's
ExecStopPost collector printed nothing for any of them (normal stops print the ledger row to the journal), so no ledger row was written at stop
time; RS reconcile backfilled them hours-worth later and mislabelled them.

Root cause, reproduced here on a transient --user unit with the same KillMode=control-group: `systemctl kill -s KILL UNIT` (default
--kill-whom=all) kills the whole cgroup and the unit's ExecStopPost never runs; `--kill-whom=main` and `systemctl stop` let it run.
Fix: the actuator records the death BEFORE it kills (collector --pre-kill, reading the liveness-recover marker it just wrote), exactly as the
watchdog's wedge path always did; the collector's dedupe marker makes a later ExecStopPost skip."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from test_engine_liveness_authority import Base, ea

HERE = os.path.dirname(os.path.abspath(__file__))
COLLECTOR = os.path.join(HERE, "engine-fault-collector.py")


def user_systemd():
    try:
        return subprocess.run(["systemctl", "--user", "is-system-running"], capture_output=True, text=True, timeout=10).stdout.strip() in (
            "running", "degraded") and shutil.which("systemd-run") is not None
    except (OSError, subprocess.SubprocessError):
        return False


@unittest.skipUnless(user_systemd(), "no user systemd")
class StopPostRace(unittest.TestCase):
    """The mechanism, on a real (transient, throw-away) user unit. Never touches the engine unit."""

    def run_case(self, label, killer):
        unit = f"bs3-race-{label}-{os.getpid()}"
        with tempfile.TemporaryDirectory() as d:
            out = f"{d}/stop-post.out"
            subprocess.run(["systemd-run", "--user", "-q", f"--unit={unit}", "-p", "Restart=no", "-p", "KillMode=control-group",
                            "-p", f"ExecStopPost=/bin/bash -c 'sleep 1; echo ran >> {out}'",
                            "bash", "-c", "sleep 600 & sleep 600 & wait"], check=True, timeout=30)
            try:
                time.sleep(1)
                subprocess.run(["systemctl", "--user", *killer, unit], check=False, timeout=30)
                deadline = time.time() + 8
                while time.time() < deadline and not os.path.exists(out):
                    time.sleep(0.25)
                time.sleep(1.5)
                return os.path.exists(out)
            finally:
                subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True, timeout=30)
                subprocess.run(["systemctl", "--user", "reset-failed", unit], capture_output=True, timeout=30)

    def test_a_whole_cgroup_kill_suppresses_exec_stop_post(self):
        self.assertFalse(self.run_case("all", ["kill", "-s", "KILL"]),
                         "ExecStopPost ran after `kill -s KILL`: this systemd no longer races it -- then the pre-kill record is redundant but harmless")

    def test_killing_only_the_main_process_or_stopping_lets_it_run(self):
        self.assertTrue(self.run_case("main", ["kill", "--kill-whom=main", "-s", "KILL"]))
        self.assertTrue(self.run_case("stop", ["stop"]))


class RecordedBeforeTheKill(Base):
    def test_every_authority_recover_pre_records_before_the_kill_step(self):
        order = []
        real_sudo = ea._sudo
        with patch.object(ea.subprocess, "run", lambda argv, **k: order.append(("collector", list(argv)[-1])) or subprocess.CompletedProcess(argv, 0, "", "")), \
                patch.object(ea, "_sudo", lambda args: (order.append(("sudo", args[0])), real_sudo(args))[1]):
            self.healthy = False
            f = ea.observe()
            res = ea._act("recover", "medic-output-missing", "medic", "engine:silent", f)
        self.assertTrue(res["acted"])
        names = [o for o in order]
        self.assertEqual(names[0], ("collector", "--pre-kill"))
        self.assertLess(names.index(("collector", "--pre-kill")), names.index(("sudo", "kill")))
        self.assertEqual([s[0] for s in res["steps"]][:2], ["pre-kill", "reset-failed"])
        marker = json.load(open(f"{ea.BASE}/liveness-recover.json"))                 # written before the collector runs
        self.assertEqual(marker["cause"], "medic-output-missing")

    def test_wedge_recover_still_pre_records_once(self):
        with patch.object(ea.subprocess, "run", lambda argv, **k: subprocess.CompletedProcess(argv, 0, "", "")):
            self.healthy = False
            res = ea._act("recover", "wedge", "watchdog", "probe", ea.observe())
        self.assertEqual([s[0] for s in res["steps"]].count("pre-kill"), 1)


def collector(base, *args, **env):
    e = dict(os.environ, FAULT_COLLECTOR_BASE=base, FAULT_COLLECTOR_MAIN_START=str(time.time() - 7200), **env)
    return subprocess.run([sys.executable, COLLECTOR, "--no-vault", *args], env=e, capture_output=True, text=True, timeout=120)


class PreKillRowForAnyCause(unittest.TestCase):
    def test_pre_kill_with_a_liveness_marker_is_a_fault_with_that_cause_and_stop_post_does_not_double_it(self):
        with tempfile.TemporaryDirectory() as base:
            from datetime import datetime
            json.dump({"ts": datetime.now().astimezone().isoformat(timespec="seconds"), "id": "lv-9", "cause": "medic-output-missing",
                       "by": "medic", "evidence": "engine:silent"}, open(f"{base}/liveness-recover.json", "w"))
            collector(base, "--pre-kill")
            rows = [json.loads(l) for l in open(f"{base}/incidents/ledger.jsonl")]
            self.assertEqual(len(rows), 1)
            r = rows[0]
            self.assertEqual(r["kind"], "FAULT")
            self.assertEqual(r["signature"], "medic-output-missing")
            self.assertEqual(r["liveness_recover"]["by"], "medic")
            self.assertEqual(r["recorded_by"], "liveness-pre-kill")
            self.assertNotIn("wedge", r)
            self.assertTrue(os.path.exists(f"{base}/recorded-fault.json"))
            collector(base, "--stop-post", SERVICE_RESULT="signal", EXIT_STATUS="KILL")      # if ExecStopPost does run, it dedupes
            self.assertEqual(len(open(f"{base}/incidents/ledger.jsonl").read().splitlines()), 1)

    def test_pre_kill_without_any_marker_is_still_the_watchdog_wedge_record(self):
        with tempfile.TemporaryDirectory() as base:
            collector(base, "--pre-kill")
            r = json.loads(open(f"{base}/incidents/ledger.jsonl").read().splitlines()[-1])
            self.assertEqual(r["recorded_by"], "watchdog-pre-kill")
            self.assertIn("wedge", r)


if __name__ == "__main__":
    unittest.main()
