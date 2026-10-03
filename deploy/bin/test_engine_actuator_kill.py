"""L77 (2026-10-03): a killed (TERM/HUP/INT) or crashed planned restart must release the gateway fence / offline window it
opened, never leave the job marked as running, and never leave the estate with the engine stopped and nobody starting it."""
import argparse
import importlib.util
import json
import os
import signal
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("engine_actuator_kill", os.path.join(HERE, "engine-actuator.py"))
ea = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ea)


class FakeGateway:
    def __init__(self, offline_supported=True):
        self.offline_supported = offline_supported
        self.calls = []
        self.drain_lease = None
        self.window_lease = None

    def __call__(self, url, method="GET", payload=None, token=None, timeout=5):
        path = url.replace(ea.GATEWAY, "")
        self.calls.append((path, method, payload))
        if path == "/gateway/offline":
            if not self.offline_supported:
                return {}
            if method == "POST":
                self.window_lease = "W1"
                return {"lease": "W1", "local_active": 2}
            if method == "DELETE":
                assert payload["lease"] == self.window_lease
                self.window_lease = None
                return {"offline": False}
            return {"offline": self.window_lease is not None, "local_active": 2}
        if path == "/gateway/drain":
            if method == "POST":
                self.drain_lease = "D1"
                return {"lease": "D1", "active": 2, "draining": True}
            if method == "DELETE":
                assert payload["lease"] == self.drain_lease
                self.drain_lease = None
                return {"draining": False}
            return {"draining": self.drain_lease is not None, "active": 2 if self.drain_lease else 0}
        return {}


class KilledRestart(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = self.tmp.name
        self.gw = FakeGateway()
        self.events, self.systemctl = [], []
        self.args = argparse.Namespace(by="test", reason="killed mid-restart test", flags=None, clear_diag=False,
                                       drain_s=5, drain_max_s=5, no_drain=False, force=False, foreground=True)
        self.saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}

        def run(cmd, **kw):
            self.systemctl.append(cmd[3:])
            return subprocess.CompletedProcess(cmd, 0, "", "")

        for p in (patch.object(ea, "LOCK", f"{d}/restart.lock"), patch.object(ea, "JOB", f"{d}/job.json"), patch.object(ea, "HOLDS", f"{d}/holds.json"),
                  patch.object(ea, "PLANNED", f"{d}/planned.json"), patch.object(ea, "http", self.gw),
                  patch.object(ea, "admin_token", return_value="t"), patch.object(ea, "engine_healthy", return_value=True),
                  patch.object(ea, "status", return_value={"gateway": {}}), patch.object(ea, "staged_flags", return_value=[]),
                  patch.object(ea, "active_flags", return_value=[]),
                  patch.object(ea, "faults_summary", return_value={"faults": 0}),
                  patch.object(ea, "emit", side_effect=lambda *a, **k: self.events.append(a)),
                  patch.object(ea.subprocess, "run", run),
                  patch.object(ea.time, "sleep", lambda s: None)):
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        for s, h in self.saved.items():
            self.assertEqual(signal.getsignal(s), h, "signal handlers must be given back")

    def job(self):
        return json.load(open(ea.JOB))

    def kill_self(self, sig):
        os.kill(os.getpid(), sig)
        time.sleep(2)                                   # the handler raises long before this returns
        self.fail("signal not delivered")

    def test_term_hup_int_while_waiting_in_the_offline_window_close_the_window(self):
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            with self.subTest(signal=sig.name):
                self.gw.calls.clear()
                self.systemctl.clear()
                with patch.object(ea, "wait_drained", lambda *a, **k: self.kill_self(sig)):
                    rc = ea.do_restart(self.args)
                self.assertEqual(rc, 128 + sig)
                self.assertIsNone(self.gw.window_lease, "offline window left open")
                self.assertIn(("/gateway/offline", "DELETE", {"lease": "W1"}), self.gw.calls)
                self.assertEqual(self.job()["state"], "aborted")
                self.assertEqual(self.systemctl, [], "engine was never stopped; nothing to start")

    def test_term_while_waiting_in_a_drain_fence_releases_the_fence(self):
        self.gw.offline_supported = False                  # older gateway: falls back to the drain fence
        sleeps = iter([None])

        def sleep_then_die(_s):
            next(sleeps)
            self.kill_self(signal.SIGTERM)

        with patch.object(ea.time, "sleep", sleep_then_die):
            self.assertEqual(ea.do_restart(self.args), 143)
        self.assertIsNone(self.gw.drain_lease, "drain fence left up")
        self.assertIn(("/gateway/drain", "DELETE", {"lease": "D1"}), self.gw.calls)
        self.assertEqual(self.job()["state"], "aborted")

    def test_term_while_waiting_for_the_engine_to_go_idle_releases_the_window(self):
        with patch.object(ea, "wait_drained", return_value={"waited_s": 1, "active_at_end": 0, "end_reason": "drained", "extended_s": 0}), \
                patch.object(ea, "wait_engine_idle", lambda *a, **k: self.kill_self(signal.SIGTERM)):
            self.assertEqual(ea.do_restart(self.args), 143)
        self.assertIsNone(self.gw.window_lease)
        self.assertEqual(self.systemctl, [])

    def test_term_while_stopping_the_engine_releases_and_asks_systemd_to_start_it_again(self):
        def run(cmd, **kw):
            self.systemctl.append(cmd[3:])
            if cmd[3] == "stop":
                self.kill_self(signal.SIGTERM)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with patch.object(ea, "wait_drained", return_value={"waited_s": 1, "active_at_end": 0, "end_reason": "drained", "extended_s": 0}), \
                patch.object(ea, "wait_engine_idle", return_value={"running_at_end": 0, "waiting_at_end": 0}), \
                patch.object(ea.subprocess, "run", run):
            self.assertEqual(ea.do_restart(self.args), 143)
        self.assertIsNone(self.gw.window_lease)
        self.assertEqual(self.systemctl[-1], ["start", "--no-block", ea.UNIT])
        self.assertTrue(self.job()["result"]["engine_start_requested"])
        self.assertTrue(any("ABORTED" in e[1] for e in self.events))

    def test_unexpected_exception_after_the_fence_is_up_also_releases_it(self):
        with patch.object(ea, "wait_drained", return_value={"waited_s": 1, "active_at_end": 0, "end_reason": "drained", "extended_s": 0}), \
                patch.object(ea, "wait_engine_idle", side_effect=ValueError("boom")):
            with self.assertRaises(ValueError):
                ea.do_restart(self.args)
        self.assertIsNone(self.gw.window_lease)
        self.assertEqual(self.job()["state"], "aborted")

    def test_normal_restart_releases_once_and_is_unchanged(self):
        with patch.object(ea, "wait_drained", return_value={"waited_s": 1, "active_at_end": 0, "end_reason": "drained", "extended_s": 0}), \
                patch.object(ea, "wait_engine_idle", return_value={"running_at_end": 0, "waiting_at_end": 0}):
            self.assertEqual(ea.do_restart(self.args), 0)
        self.assertEqual([c for c in self.gw.calls if c[1] == "DELETE"], [("/gateway/offline", "DELETE", {"lease": "W1"})])
        self.assertEqual(self.systemctl, [["stop", ea.UNIT], ["start", ea.UNIT]])
        self.assertEqual(self.job()["state"], "done")


if __name__ == "__main__":
    unittest.main()
