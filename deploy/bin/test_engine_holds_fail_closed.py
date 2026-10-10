"""Hold corruption never grants engine authority; every path uses private files/fake services."""
import argparse
import contextlib
import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("engine_holds_closed", HERE / "engine-actuator.py")
ea = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ea)


class HoldsClosed(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.now, self.pid = 1_800_000_000.0, 100
        self.events, self.commands = [], []
        self.args = argparse.Namespace(by="estate-core", reason="private contract restart", flags=None,
                                       clear_diag=False, force=False, no_drain=False, drain_s=1,
                                       drain_max_s=1, health_wait_s=0, hold=None)
        for name, filename in {"LOCK": "restart.lock", "HOLDS": "holds.json", "HOLDS_LOCK": "holds.lock",
                               "JOB": "job.json", "PLANNED": "planned.json", "WD_STATE": "wd.json",
                               "FQ_RUNNING": "missing-running", "LPAUSE": "PAUSE"}.items():
            self.start_patch(patch.object(ea, name, str(self.root / filename)))
        for name, value in {"admin_token": "private-test-token", "engine_healthy": True,
                            "status": {"gateway": {}}, "staged_flags": [], "active_flags": [],
                            "faults_summary": {"faults": 0}, "runtime_identity": {"engine_pid": 101},
                            "offline_and_wait": ({"active_at_end": 0, "strategy": "offline-window"}, None),
                            "wait_engine_idle": {"idle": True, "running_at_end": 0, "waiting_at_end": 0},
                            "_unit_facts": {}, "_gateway_offline": {}, "_window_procs": [],
                            "_recent_fault_times": [], "gate": {}}.items():
            self.start_patch(patch.object(ea, name, return_value=value))
        self.start_patch(patch.object(ea, "main_pid", side_effect=lambda unit: self.pid))
        self.start_patch(patch.object(ea, "_proc_start", return_value="1000"))
        self.start_patch(patch.object(ea, "emit", side_effect=lambda *a, **k: self.events.append(a)))
        self.start_patch(patch.object(ea.time, "time", return_value=self.now))
        self.start_patch(patch.object(ea.time, "sleep", side_effect=AssertionError("real wait forbidden")))
        self.start_patch(patch.object(ea.subprocess, "run", side_effect=self.fake_run))
        self.start_patch(patch.object(ea.subprocess, "Popen", side_effect=AssertionError("real process forbidden")))
        self.start_patch(patch.object(ea, "http", side_effect=AssertionError("real HTTP forbidden")))
        self.start_patch(patch.object(ea, "_sudo", side_effect=AssertionError("automatic service action forbidden")))

    def start_patch(self, item):
        result = item.start()
        self.addCleanup(item.stop)
        return result

    def fake_run(self, argv, **kwargs):
        self.commands.append(list(argv))
        if argv[:4] == ["sudo", "-n", "systemctl", "stop"]:
            self.pid = 0
        elif argv[:4] == ["sudo", "-n", "systemctl", "start"]:
            self.pid = 101
        else:
            raise AssertionError(f"unqualified subprocess: {argv}")
        return subprocess.CompletedProcess(argv, 0, "", "")

    def row(self, **overrides):
        return {"lease": "window-123", "kind": "engine", "by": "windowctl", "reason": "protected test window",
                "until": self.now + 600, **overrides}

    def save(self, rows):
        Path(ea.HOLDS).write_text(json.dumps(rows))

    def restart(self, **overrides):
        args = argparse.Namespace(**{**vars(self.args), **overrides})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = ea.do_restart(args)
        return rc, json.loads(out.getvalue().splitlines()[-1])

    def assert_no_actions(self):
        self.assertEqual(self.commands, [])
        self.assertFalse(Path(ea.JOB).exists())
        self.assertFalse(Path(ea.PLANNED).exists())

    def test_absent_and_explicit_empty_are_compatible(self):
        self.assertEqual(ea.active_holds(), [])
        self.save([])
        self.assertEqual(ea.active_holds(), [])
        rc, result = self.restart()
        self.assertEqual(rc, 0)
        self.assertTrue(result["healthy_after"])

    def test_invalid_state_refuses_before_token_flags_job_fence_or_services(self):
        cases = ({}, [False], [self.row(lease="")], [self.row(kind="unknown")],
                 [self.row(until=True)], [self.row(until=float("nan"))], [self.row(until=float("inf"))],
                 [self.row(until=10 ** 1000)], [self.row(until=-1)], [self.row(mode="unknown")],
                 [self.row(mode="owned", pid=True)], [self.row(mode="run", pid=0)],
                 [self.row(mode="run", pid=100, pid_start=1000)], [self.row(by=1)],
                 [self.row(), self.row()], [self.row(until=1), {"invalid": "expired rows do not disappear"}])
        for rows in cases:
            with self.subTest(rows=rows):
                self.save(rows)
                before = Path(ea.HOLDS).read_bytes()
                with patch.object(ea, "admin_token", side_effect=AssertionError("token access before refusal")), \
                        patch.object(ea, "write_diag", side_effect=AssertionError("flags staged before refusal")), \
                        patch.object(ea, "offline_and_wait", side_effect=AssertionError("fence before refusal")):
                    rc, result = self.restart(flags="skip-warmup")
                self.assertEqual(rc, 3)
                self.assertIn("hold state", result["refused"])
                self.assertEqual(Path(ea.HOLDS).read_bytes(), before)
                self.assertEqual(self.events, [])
                self.assert_no_actions()

    def test_malformed_json_unreadable_file_and_dangling_path_fail_closed(self):
        Path(ea.HOLDS).write_text("[")
        self.assertEqual(self.restart()[0], 3)
        with patch("builtins.open", side_effect=lambda *a, **k: (_ for _ in ()).throw(PermissionError("denied"))
                   if str(a[0]) == ea.HOLDS else self.original_open(*a, **k)):
            self.assertEqual(self.restart()[0], 3)
        Path(ea.HOLDS).unlink()
        Path(ea.HOLDS).symlink_to(self.root / "absent-target")
        self.assertEqual(self.restart()[0], 3)
        self.assert_no_actions()

    original_open = staticmethod(open)

    def test_foreign_engine_and_quiesce_holds_refuse_without_hold_rewrite(self):
        for kind in ("engine", "quiesce"):
            with self.subTest(kind=kind):
                self.save([self.row(kind=kind)])
                before = Path(ea.HOLDS).read_bytes()
                with patch.object(ea, "admin_token", side_effect=AssertionError("token access before refusal")):
                    rc, result = self.restart()
                self.assertEqual(rc, 3)
                self.assertIn("held by windowctl", result["refused"])
                self.assertEqual(Path(ea.HOLDS).read_bytes(), before)
                self.assert_no_actions()

    def test_expired_foreign_hold_does_not_block(self):
        self.save([self.row(until=self.now - 1)])
        self.assertEqual(self.restart()[0], 0)

    def test_exact_owned_lease_is_admitted_but_cannot_bypass_other_kind(self):
        self.save([self.row(mode="owned", pid=100, pid_start="1000")])
        self.assertEqual(self.restart(hold="wrong-lease")[0], 3)
        self.assert_no_actions()
        self.assertEqual(self.restart(hold="window-123")[0], 0)
        self.commands.clear()
        Path(ea.JOB).unlink()
        Path(ea.PLANNED).unlink()
        self.save([self.row(), self.row(lease="publish-456", kind="quiesce")])
        self.assertEqual(self.restart(hold="window-123")[0], 3)
        self.assert_no_actions()

    def test_process_scoped_expiry_and_plain_acquire_semantics_are_preserved(self):
        self.save([self.row(mode="run", pid=100, pid_start="1000")])
        self.assertEqual(len(ea.active_holds()), 1)
        with patch.object(ea, "_proc_start", return_value="different"):
            self.assertEqual(ea.active_holds(), [])
        with patch.object(ea, "_proc_start", return_value=None):
            self.assertEqual(ea.active_holds(), [])
        self.save([self.row(mode="acquire", pid=100, pid_start="1000")])
        with patch.object(ea, "_proc_start", return_value=None):
            self.assertEqual(len(ea.active_holds()), 1)

    def test_shared_restart_lock_refuses_before_reading_hold_state(self):
        with open(ea.LOCK, "a+") as holder:
            fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(ea, "_holds_read", side_effect=AssertionError("reader after lock refusal")):
                rc, result = self.restart()
        self.assertEqual(rc, 3)
        self.assertIn("another planned restart", result["refused"])
        self.assert_no_actions()

    def test_acquire_renew_and_release_cannot_overwrite_unknown_metadata(self):
        Path(ea.HOLDS).write_text("not JSON")
        for action in (lambda: ea.hold_acquire("engine", "test", "qualified private test", 600),
                       lambda: ea.hold_renew("window-123", 600), lambda: ea.hold_release("window-123")):
            with self.assertRaises(ea.HoldStateError):
                action()
            self.assertEqual(Path(ea.HOLDS).read_text(), "not JSON")
        self.assertEqual(self.events, [])
        self.assert_no_actions()

    def test_automatic_paths_abort_before_any_service_action_on_unknown_holds(self):
        Path(ea.HOLDS).write_text("not JSON")
        for action in (lambda: ea.tick(), lambda: ea.recover("confirmed-wedge", "test", "private proof"),
                       lambda: ea.ensure_engine_up("test")):
            with self.assertRaises(ea.HoldStateError):
                action()
        self.assert_no_actions()


if __name__ == "__main__":
    unittest.main()
