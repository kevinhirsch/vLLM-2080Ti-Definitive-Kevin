"""BS3 (found live 2026-10-04): the liveness authority's own recover kills were recorded as 'planned-stop (systemctl)'.

Four medic-initiated `systemctl kill -s KILL` + restart cycles in 24 h (10-03 19:55, 10-04 00:46, 02:40, 06:07). The unit's ExecStopPost
collector printed nothing for any of them (the cgroup SIGKILL races it), so RS reconcile backfilled the deaths -- and read the authority's own
sudo lines ('systemctl kill', 'systemctl restart') as an operator's planned stop. The liveness-recover.json marker the authority wrote was never
consumed and still sat on disk, ready to relabel the next death. Reconcile must read liveness-actions.jsonl; every marker needs the L180 TTL."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
COLLECTOR = os.path.join(HERE, "engine-fault-collector.py")


def run(base, *args, **env):
    e = dict(os.environ, FAULT_COLLECTOR_BASE=base, **env)
    e.setdefault("FAULT_COLLECTOR_MAIN_START", str(time.time() - 7200))
    return subprocess.run([sys.executable, COLLECTOR, "--no-vault", *args], env=e, capture_output=True, text=True, timeout=120)


def iso(t):
    return datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds")


def last(base):
    return json.loads(open(f"{base}/incidents/ledger.jsonl").read().splitlines()[-1])


SUDO_LINES = ("kevin :  PWD=/home/kevin ; USER=root ; COMMAND=/usr/bin/systemctl reset-failed vllm-qwen27b; "
              "kevin :  PWD=/home/kevin ; USER=root ; COMMAND=/usr/bin/systemctl kill -s KILL vllm-qwen27b; "
              "kevin :  PWD=/home/kevin ; USER=root ; COMMAND=/usr/bin/systemctl restart --no-block vllm-qwen27b")


def inproc(base, *args):
    """Run main() in this process: the --cause text names systemctl + the live unit, which live_guard rightly refuses in a child argv."""
    os.environ["FAULT_COLLECTOR_BASE"] = base
    spec = importlib.util.spec_from_file_location("collector_bs3_main", COLLECTOR)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    with patch.object(sys, "argv", ["engine-fault-collector.py", "--no-vault", *args]), \
            patch.object(m, "sh", lambda cmd, timeout=40: ""), patch.object(m, "_ledger_ts", lambda: []):
        m.main()


class AuthorityKillIsAFault(unittest.TestCase):
    def action(self, base, t, aid="lv-1", cause="medic-output-missing", by="medic", action="recover"):
        with open(f"{base}/liveness-actions.jsonl", "a") as fh:
            fh.write(json.dumps({"id": aid, "t": t, "action": action, "cause": cause, "by": by, "evidence": "engine:silent"}) + "\n")

    def test_backfill_with_a_liveness_action_is_a_fault_even_with_systemctl_in_the_sudo_journal(self):
        with tempfile.TemporaryDirectory() as base:
            t = time.time() - 600
            self.action(base, t - 1)
            stamp = datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
            inproc(base, "--at", stamp, "--reconciled", "--cause", SUDO_LINES, "--liveness-action", "lv-1")
            row = last(base)
            self.assertEqual(row["kind"], "FAULT")
            self.assertEqual(row["liveness_recover"]["by"], "medic")
            self.assertEqual(row["liveness_recover"]["cause"], "medic-output-missing")
            self.assertNotEqual(row["signature"], "planned")

    def test_without_the_action_the_old_backfill_rule_is_unchanged(self):
        with tempfile.TemporaryDirectory() as base:
            t = time.time() - 600
            stamp = datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
            inproc(base, "--at", stamp, "--reconciled", "--cause", SUDO_LINES)
            self.assertEqual(last(base)["kind"], "planned-stop")          # an operator's systemctl stop/restart is still planned

    def _module(self, base):
        os.environ["FAULT_COLLECTOR_BASE"] = base
        spec = importlib.util.spec_from_file_location("collector_bs3", COLLECTOR)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_authority_action_near_picks_the_kill_that_precedes_the_death(self):
        with tempfile.TemporaryDirectory() as base:
            m = self._module(base)
            t = 1_800_000_000.0
            self.action(base, t - 2, "lv-a")
            self.action(base, t - 4000, "lv-old")
            self.action(base, t - 3, "lv-start", action="start")
            self.assertEqual(m.authority_action_near(t)["id"], "lv-a")
            self.assertIsNone(m.authority_action_near(t + 600))            # a later death is not this action's
            self.assertIsNone(m.authority_action_near(t - 100000))

    def test_reconcile_passes_the_action_id_to_the_backfill(self):
        with tempfile.TemporaryDirectory() as base:
            m = self._module(base)
            t = time.time() - 900
            self.action(base, t - 2, "lv-77")
            line = f"{iso(t)} HNET00 systemd[1]: vllm-qwen27b.service: Main process exited, code=killed, status=9/KILL"
            calls = []

            def fake_sh(cmd, timeout=40):
                return line if "Main process exited" in cmd else ""

            class R:
                returncode = 0

            with patch.object(m, "sh", fake_sh), patch.object(m, "_ledger_ts", lambda: []), \
                    patch.object(m.subprocess, "run", lambda cmd, **k: calls.append(cmd) or R()):
                m.reconcile(6)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][calls[0].index("--liveness-action") + 1], "lv-77")


class EveryMarkerHasATtl(unittest.TestCase):
    def test_a_stale_liveness_recover_marker_does_not_relabel_a_later_death(self):
        with tempfile.TemporaryDirectory() as base:
            json.dump({"ts": iso(time.time() - 3600), "id": "lv-old", "cause": "medic-output-missing", "by": "medic", "evidence": "x"},
                      open(f"{base}/liveness-recover.json", "w"))
            run(base, "--stop-post", SERVICE_RESULT="exit-code", EXIT_STATUS="1")
            row = last(base)
            self.assertNotIn("liveness_recover", row)
            self.assertEqual(row["stale_markers"][0]["marker"], "liveness-recover")
            self.assertTrue(os.path.exists(f"{base}/liveness-recover.stale.json"))
            self.assertFalse(os.path.exists(f"{base}/liveness-recover.json"))

    def test_a_fresh_liveness_recover_marker_is_still_honoured(self):
        with tempfile.TemporaryDirectory() as base:
            json.dump({"ts": iso(time.time() - 3), "id": "lv-new", "cause": "stuck_boot", "by": "tick", "evidence": "x"},
                      open(f"{base}/liveness-recover.json", "w"))
            run(base, "--stop-post", SERVICE_RESULT="signal", EXIT_STATUS="KILL")
            row = last(base)
            self.assertEqual(row["liveness_recover"]["cause"], "stuck_boot")
            self.assertNotIn("stale_markers", row)

    def test_a_stale_wedge_marker_is_ignored(self):
        with tempfile.TemporaryDirectory() as base:
            json.dump({"ts": iso(time.time() - 3600), "by": "watchdog", "wedge": True}, open(f"{base}/wedge-restart.json", "w"))
            run(base, "--stop-post", SERVICE_RESULT="exit-code", EXIT_STATUS="1")
            row = last(base)
            self.assertNotIn("wedge", row)
            self.assertEqual(row["stale_markers"][0]["marker"], "wedge-restart")

    def test_marker_from_before_this_boot_is_stale_even_when_young(self):
        with tempfile.TemporaryDirectory() as base:
            json.dump({"ts": iso(time.time() - 120), "id": "lv-x", "cause": "dead_core", "by": "tick", "evidence": "x"},
                      open(f"{base}/liveness-recover.json", "w"))
            run(base, "--stop-post", SERVICE_RESULT="exit-code", EXIT_STATUS="1", FAULT_COLLECTOR_MAIN_START=str(time.time() - 30))
            self.assertEqual(last(base)["stale_markers"][0]["why"], "predates-this-boot")


if __name__ == "__main__":
    unittest.main()
