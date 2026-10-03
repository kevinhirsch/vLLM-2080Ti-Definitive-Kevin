"""AU 2026-10-03: a planned restart's outcome and the start announcement must be MEASURED health, not a single probe taken
the instant `systemctl start` returns (Type=simple returns at fork whenever the warm-up hook skips itself). Measured
10-02..10-03: 66 of 81 restart-finished events said healthy=False while announce-start said "back and healthy" in the
same second, both before the API was up."""
import argparse, importlib.util, json, os, subprocess, tempfile, unittest
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("engine_actuator_ht", os.path.join(HERE, "engine-actuator.py"))
ea = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ea)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


class WaitHealthy(unittest.TestCase):
    def setUp(self):
        self.c = Clock()
        for p in (patch.object(ea.time, "time", self.c.time), patch.object(ea.time, "sleep", self.c.sleep)):
            p.start(); self.addCleanup(p.stop)

    def test_returns_as_soon_as_health_answers_with_the_measured_wait(self):
        answers = iter([False, False, False, True])
        with patch.object(ea, "engine_healthy", lambda: next(answers)):
            self.assertEqual(ea.wait_engine_healthy(60, poll_s=5), (True, 15.0))

    def test_bounded_when_health_never_answers(self):
        calls = []
        with patch.object(ea, "engine_healthy", lambda: calls.append(1) or False):
            ok, waited = ea.wait_engine_healthy(30, poll_s=5)
        self.assertFalse(ok)
        self.assertGreaterEqual(waited, 30)
        self.assertLessEqual(len(calls), 8)

    def test_zero_budget_still_samples_once(self):
        with patch.object(ea, "engine_healthy", return_value=True):
            self.assertEqual(ea.wait_engine_healthy(0), (True, 0.0))
        with patch.object(ea, "engine_healthy", return_value=False):
            self.assertEqual(ea.wait_engine_healthy(0), (False, 0.0))


class RestartOutcome(unittest.TestCase):
    """The engine comes up 3 polls after `systemctl start` returns: the outcome must say healthy=True after ~15 s."""
    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        d = td.name
        self.events, self.c = [], Clock()
        self.health = iter([True] + [False] * 3 + [True] * 50)   # before the restart: healthy; after start: 3 misses then up

        def run(cmd, **kw):
            return subprocess.CompletedProcess(cmd, 0, "", "")

        for p in (patch.object(ea, "LOCK", f"{d}/l"), patch.object(ea, "JOB", f"{d}/j.json"), patch.object(ea, "PLANNED", f"{d}/p.json"), patch.object(ea, "HOLDS", f"{d}/holds.json"),
                  patch.object(ea, "admin_token", return_value="t"), patch.object(ea, "engine_healthy", lambda: next(self.health)),
                  patch.object(ea, "status", return_value={"gateway": {}}), patch.object(ea, "staged_flags", return_value=[]),
                  patch.object(ea, "active_flags", return_value=[]), patch.object(ea, "faults_summary", return_value={"faults": 0}),
                  patch.object(ea, "emit", side_effect=lambda *a, **k: self.events.append(a)),
                  patch.object(ea.subprocess, "run", run), patch.object(ea.time, "time", self.c.time),
                  patch.object(ea.time, "sleep", self.c.sleep)):
            p.start(); self.addCleanup(p.stop)
        self.args = argparse.Namespace(by="test", reason="health truth test", flags=None, clear_diag=False, drain_s=5,
                                       drain_max_s=5, no_drain=True, force=False, foreground=True, health_wait_s=60)

    def test_outcome_is_measured_health_not_one_instant_probe(self):
        self.assertEqual(ea.do_restart(self.args), 0)
        fin = [e for e in self.events if e[0] == "outcome"][-1]
        self.assertTrue(fin[2]["healthy_after"])
        self.assertEqual(fin[2]["health_wait_s"], 15.0)
        self.assertIn("healthy=True (after 15.0s)", fin[1])
        self.assertEqual(json.load(open(ea.JOB))["state"], "done")

    def test_engine_that_never_comes_up_is_a_failed_restart_after_the_budget(self):
        self.health = iter([True] + [False] * 100)
        self.assertEqual(ea.do_restart(self.args), 1)
        fin = [e for e in self.events if e[0] == "outcome"][-1]
        self.assertFalse(fin[2]["healthy_after"])
        self.assertGreaterEqual(fin[2]["health_wait_s"], 60)
        self.assertEqual(json.load(open(ea.JOB))["state"], "failed")


class AnnounceStart(unittest.TestCase):
    def setUp(self):
        self.events, self.spawned = [], []
        for p in (patch.object(ea, "emit", side_effect=lambda *a, **k: self.events.append(a)),
                  patch.object(ea, "unit_view", return_value={}), patch.object(ea, "active_flags", return_value=[]),
                  patch.object(ea, "faults_summary", return_value={"faults": 0}), patch.object(ea, "sh", return_value=""),
                  patch.object(ea.subprocess, "Popen", side_effect=lambda cmd, **k: self.spawned.append(cmd))):
            p.start(); self.addCleanup(p.stop)

    def test_unhealthy_engine_is_never_announced_healthy_and_a_waiter_is_spawned(self):
        with patch.object(ea, "engine_healthy", return_value=False):
            self.assertEqual(ea.announce_start(argparse.Namespace(wait_healthy=False)), 0)
        self.assertEqual([e[2]["action"] for e in self.events], ["engine-process-started"])
        self.assertFalse(any("back and healthy" in e[1] for e in self.events))
        self.assertEqual(self.spawned[-1][-2:], ["announce-start", "--wait-healthy"])

    def test_healthy_engine_is_announced_as_before(self):
        with patch.object(ea, "engine_healthy", return_value=True):
            ea.announce_start(argparse.Namespace(wait_healthy=False))
        self.assertEqual(self.events[-1][2]["action"], "engine-started")
        self.assertIn("back and healthy", self.events[-1][1])

    def test_waiter_that_times_out_reports_unhealthy_not_healthy(self):
        with patch.object(ea, "wait_engine_healthy", return_value=(False, 420.0)):
            ea.announce_start(argparse.Namespace(wait_healthy=True))
        self.assertEqual([e[2]["action"] for e in self.events], ["engine-start-unhealthy"])

    def test_waiter_that_sees_health_announces_with_the_measured_boot_wait(self):
        with patch.object(ea, "wait_engine_healthy", return_value=(True, 185.0)):
            ea.announce_start(argparse.Namespace(wait_healthy=True))
        self.assertEqual(self.events[-1][2]["action"], "engine-started")
        self.assertEqual(self.events[-1][2]["health_wait_s"], 185.0)


if __name__ == "__main__":
    unittest.main()
