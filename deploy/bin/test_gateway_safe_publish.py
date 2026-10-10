"""CPU-only publication transaction tests; every external authority is mocked."""
import contextlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import types
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("safe_publish_cpu", HERE / "gateway_safe_publish.py")
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)


class World:
    def __init__(self, root, legacy=False):
        self.root = Path(root)
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()
        self.events = []
        self.pid = 111
        self.lease = None
        self.durable = False
        self.active = 0
        self.in_flight = 0
        self.offline = False
        self.measurement = False
        self.engine = {"engine_pid": 10, "pid_start": "123", "root": "/immutable", "model_path": "/model",
                       "flags": {}, "runtime_flags": {}, "argv_sha256": "a" * 64, "fingerprint": "b" * 64}
        self.units = {"estate-self-repair.timer": "active", "estate-self-repair.service": "inactive", "estate-core.service": "active"}
        self.candidate = b"DURABLE_DRAIN_PROTOCOL = 1\n# candidate\n"
        old = b"# legacy\n" if legacy else b"DURABLE_DRAIN_PROTOCOL = 1\n# previous\n"
        (self.runtime / "keepalive-shim.py").write_bytes(old)
        (self.runtime / "keepalive-shim.py").chmod(0o750)
        (self.runtime / "gateway_dashboard.html").write_bytes(b"previous page")
        (self.runtime / "gateway_dashboard.html").chmod(0o640)
        self.sources = {"keepalive-shim.py": self.candidate, "gateway_dashboard.html": b"candidate page",
                        "experiment_controls.py": b"# controls"}
        self.code = {n: (HERE / n).read_bytes() for n in pub.CONTROL_CODE}
        self.api = types.SimpleNamespace(ENGINE_UNIT="engine", ENGINE="http://engine", LOCK=str(self.root / "window.lock"), CONTROL_ENV={})
        self.api.runtime_identity = Mock(side_effect=lambda _: dict(self.engine))
        self.api.main_pid = Mock(side_effect=lambda _: self.pid)
        self.api.engine_inflight = Mock(return_value={"running": 0, "waiting": 0})
        self.api.user_unit_state = Mock(side_effect=lambda u: self.units.get(u, "unknown"))
        self.api.set_user_unit = Mock(side_effect=self.set_unit)
        self.api.actuator_hold = Mock(side_effect=self.hold)
        self.api.restoration_generation = Mock(return_value={"ok": True, "runtime_unchanged": True, "finish_reason": "stop", "completion_tokens": 2})
        self.stop_config = self.root / "dropin.conf"
        self.stop_config.write_bytes(b"[Service]\nTimeoutStopSec=1830\n")
        self.actuator = self.root / "actuator.py"
        self.actuator.write_bytes(b"# mocked control authority\n")
        self.patches = [patch.object(pub, "RUNTIME", self.runtime / "keepalive-shim.py"),
                        patch.object(pub, "DURABLE_DRAIN_STATE", self.runtime / "gateway-drain-state.json"),
                        patch.object(pub, "PUBLISH_STATE_ROOT", self.root / "publications"),
                        patch.object(pub, "DROPIN", self.stop_config), patch.object(pub, "DROPIN_LIVE", self.stop_config),
                        patch.object(pub, "ACTUATOR", self.actuator),
                        patch.object(pub, "_controls", return_value=self.api), patch.object(pub, "_refuse_core_cgroup"),
                        patch.object(pub, "_capture_committed", return_value=("c" * 40, self.sources, self.code)),
                        patch.object(pub, "_http", side_effect=self.http), patch.object(pub, "_run", side_effect=self.run),
                        patch.object(pub, "_proc_stat", side_effect=lambda p: ("S", str(p) + "start")),
                        patch.object(pub, "_arm_recovery", side_effect=lambda *a: self.events.append("armed")),
                        patch.object(pub, "_audit"), patch.object(pub, "_exclusive", side_effect=lambda p: contextlib.nullcontext()),
                        patch.object(pub.subprocess, "check_output", return_value="1830s"),
                        patch.object(pub.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)),
                        patch.object(pub.time, "sleep"), patch.dict(os.environ, {"SHIM_ADMIN_TOKEN": "cpu-test-token"})]

    def __enter__(self):
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *args):
        for p in reversed(self.patches):
            p.stop()

    def hold(self, *args, **kwargs):
        self.events.append("hold:" + args[0])
        return {"lease": "d" * 16} if args[0] in ("acquire", "renew") else {"released": 1}

    def set_unit(self, unit, active):
        self.events.append(("start:" if active else "stop:") + unit)
        self.units[unit] = "active" if active else "inactive"
        return True

    def run(self, *args):
        if args == ("sudo", "-n", "systemctl", "restart", pub.SERVICE):
            self.events.append("restart")
            self.pid += 1
            self.durable = pub._supports_durable(pub.RUNTIME.read_bytes())
        elif args == ("sudo", "-n", "systemctl", "stop", pub.SERVICE):
            self.events.append("gateway-stop")
            self.pid = 0
        else:
            raise AssertionError(args)

    def http(self, path, method="GET", payload=None, token=""):
        if path == "/health":
            return {"http_status": 200}
        if path == "/gateway/offline":
            return {"offline": self.offline}
        if path == "/gateway/measurement":
            return {"active": self.measurement}
        if path == "/gateway/modules":
            return {"modules": {n: {"loaded": True, "sha256": pub._sha((self.runtime / n).read_bytes())}
                                for n in pub._module_names() if (self.runtime / n).exists()}}
        if path == "/gateway/spend":
            return {"enforce": True, "durable": True, "cap": 25.0, "in_flight": self.in_flight, "attribution_gap_usd": 0,
                    "gateway_sha256": pub._sha(pub.RUNTIME.read_bytes())}
        assert path == "/gateway/drain", path
        if method == "POST":
            self.events.append("fence")
            self.lease = "e" * 32
            self.durable = bool(payload.get("durable")) and pub._supports_durable(pub.RUNTIME.read_bytes())
            return {"lease": self.lease, "active": self.active, "until": time.time() + 1800}
        if method == "DELETE":
            assert payload["lease"] == self.lease
            self.events.append("unfence")
            self.lease = None
            if self.durable:
                pub.DURABLE_DRAIN_STATE.unlink()
            return {"draining": False}
        return {"draining": self.lease is not None, "active": self.active, "until": time.time() + 1800,
                "durable": self.durable, "lease_sha256": pub._sha(self.lease.encode()) if self.lease else None}

    def state(self):
        return json.loads(next((self.root / "publications").glob("publish-*/state.json")).read_text())


class Publish(unittest.TestCase):
    def world(self, legacy=False):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return World(tmp.name, legacy)

    def test_success_is_qualified_before_producer_and_fence_release(self):
        with self.world() as w:
            result = pub._publish(10, 10)
            self.assertEqual(result["status"], "published")
            self.assertEqual(result["qualified"], "candidate")
            self.assertTrue(result["generation_canary"]["ok"])
            self.assertEqual(w.units["estate-self-repair.service"], "inactive")
            self.assertLess(w.events.index("armed"), w.events.index("stop:estate-core.service"))
            self.assertLess(w.events.index("stop:estate-self-repair.timer"), w.events.index("stop:estate-core.service"))
            self.assertLess(w.events.index("restart"), w.events.index("start:estate-core.service"))
            self.assertLess(w.events.index("start:estate-core.service"), w.events.index("start:estate-self-repair.timer"))
            self.assertEqual(w.state()["gateway_generation"]["pid"], 112)
            self.assertFalse(pub.DURABLE_DRAIN_STATE.exists())

    def test_unknown_units_refuse_before_any_maintenance(self):
        with self.world() as w:
            w.units["estate-self-repair.timer"] = "unknown"
            with self.assertRaisesRegex(RuntimeError, "activity"):
                pub._publish(10, 10)
            self.assertEqual(w.events, [])

    def test_noop_still_refuses_offline_or_measurement_window(self):
        for key in ("offline", "measurement"):
            with self.subTest(key=key), self.world() as w:
                w.sources["keepalive-shim.py"] = pub.RUNTIME.read_bytes()
                setattr(w, key, True)
                with self.assertRaises(RuntimeError):
                    pub._publish(10, 10)
                self.assertEqual(w.events, [])

    def test_missing_counts_refuse_before_any_maintenance(self):
        with self.world() as w:
            w.active = None
            with self.assertRaisesRegex(RuntimeError, "authority"):
                pub._publish(10, 10)
            self.assertEqual(w.events, [])

    def test_expected_head_mismatch_refuses(self):
        with self.world() as w:
            with self.assertRaisesRegex(RuntimeError, "requested committed"):
                pub._publish(10, 10, expected_head="f" * 40)
            self.assertEqual(w.events, [])

    def test_hold_failure_restores_original_unit_activity_without_restart(self):
        with self.world() as w:
            w.api.actuator_hold.side_effect = lambda *a, **k: {"refused": "busy"}
            with self.assertRaises(RuntimeError):
                pub._publish(10, 10)
            self.assertNotIn("restart", w.events)
            self.assertEqual(w.units["estate-core.service"], "active")

    def test_generation_failure_keeps_admission_and_producers_fenced(self):
        with self.world() as w, patch.object(pub.time, "monotonic", side_effect=itertools.count(100)):
            w.api.restoration_generation.return_value = {"ok": False}
            # Drain has its own tests below. Bound qualification immediately.
            with patch.object(pub, "_wait_empty"):
                with self.assertRaises(RuntimeError):
                    pub._publish(10, 10)
            self.assertIsNotNone(w.lease)
            self.assertTrue(pub.DURABLE_DRAIN_STATE.exists())
            self.assertEqual(w.units["estate-core.service"], "inactive")
            self.assertEqual(w.state()["status"], "recovery-required")

    def test_legacy_failed_qualification_restores_exact_files_then_stops_gateway(self):
        with self.world(legacy=True) as w, patch.object(pub.time, "monotonic", side_effect=itertools.count(100)):
            with patch.object(pub, "_wait_empty"), patch.object(pub, "_qualify", side_effect=RuntimeError("readback failed")):
                with self.assertRaises(RuntimeError):
                    pub._publish(10, 10)
            self.assertEqual(pub.RUNTIME.read_bytes(), b"# legacy\n")
            self.assertEqual(pub.RUNTIME.stat().st_mode & 0o7777, 0o750)
            self.assertEqual((w.runtime / "gateway_dashboard.html").read_bytes(), b"previous page")
            self.assertEqual((w.runtime / "gateway_dashboard.html").stat().st_mode & 0o7777, 0o640)
            self.assertFalse((w.runtime / "experiment_controls.py").exists())
            self.assertIn("gateway-stop", w.events)
            self.assertNotIn("unfence", w.events)
            self.assertEqual(w.state()["status"], "manual-recovery-required")

    def test_engine_generation_change_does_not_trigger_gateway_restart_or_unfence(self):
        with self.world() as w:
            before = dict(w.engine)
            def identity(_):
                row = dict(before)
                if "stop:estate-core.service" in w.events:
                    row["engine_pid"] = 20
                return row
            w.api.runtime_identity.side_effect = identity
            with self.assertRaisesRegex(RuntimeError, "generation"):
                pub._publish(10, 10)
            self.assertNotIn("restart", w.events)
            self.assertEqual(w.units["estate-core.service"], "inactive")

    def test_cleanup_error_after_qualification_is_not_a_rollback(self):
        with self.world() as w, patch.object(pub.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)):
            result = pub._publish(10, 10)
            self.assertEqual(result["status"], "published")
            self.assertEqual(w.events.count("restart"), 1)

    def test_all_backup_hashes_checked_before_first_rollback_write(self):
        with self.world() as w:
            directory = w.root / "tx"
            directory.mkdir()
            records = pub._snapshot_files(directory, w.sources)
            (directory / "files" / "gateway_dashboard.html.previous").write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "backup hash"):
                pub._apply_files(directory / "state.json", {"files": records}, restored=True)
            self.assertEqual(pub.RUNTIME.read_bytes(), b"DURABLE_DRAIN_PROTOCOL = 1\n# previous\n")

    def test_core_cgroup_is_refused(self):
        with patch.object(Path, "read_text", return_value="0::/user.slice/app.slice/estate-core.service\n"):
            with self.assertRaisesRegex(RuntimeError, "own control-group"):
                pub._refuse_core_cgroup()

    def test_required_module_cannot_be_absent(self):
        with tempfile.TemporaryDirectory() as d, patch.object(pub, "SOURCE", Path(d) / "shim.py"):
            with self.assertRaisesRegex(RuntimeError, "required gateway"):
                pub._module_sources()

    def test_drain_unknown_counts_or_wrong_owner_cannot_be_empty(self):
        for field, value in (("active", None), ("active", False), ("lease_sha256", "f" * 64)):
            with self.subTest(field=field), self.world() as w:
                w.lease = "e" * 32
                w.durable = True
                state = {"hold": "d" * 16, "engine": dict(w.engine), "lease": w.lease}
                original = w.http
                def http(path, **kw):
                    row = original(path, **kw)
                    if path == "/gateway/drain":
                        row[field] = value
                    return row
                with patch.object(pub, "_http", side_effect=http):
                    with self.assertRaises(RuntimeError):
                        pub._wait_empty("test", 10, state)

    def test_expired_durable_fence_still_allows_verified_drain(self):
        with self.world() as w:
            w.lease, w.durable = "e" * 32, True
            state = {"hold": "d" * 16, "engine": dict(w.engine), "lease": w.lease}
            original = w.http
            def http(path, **kw):
                row = original(path, **kw)
                if path == "/gateway/drain":
                    row["until"] = 1
                return row
            with patch.object(pub, "_http", side_effect=http):
                pub._wait_empty("test", 10, state)

    def test_engine_requests_block_dead_owner_recovery_before_rollback(self):
        with self.world() as w:
            pub._publish(10, 10)
            path = next((w.root / "publications").glob("publish-*/state.json"))
            state = json.loads(path.read_text())
            state.update(status="recovery-required", qualified=None, restart_intent=True)
            pub._save_state(path, state)
            w.lease, w.durable = "e" * 32, True
            pub._persist_bootstrap_fence({"lease": w.lease, "active": 0, "until": time.time() + 1800})
            w.api.engine_inflight.return_value = {"running": 1, "waiting": 0}
            for unit in w.units:
                w.units[unit] = "inactive"
            old_events = len(w.events)
            with patch.object(pub, "_owner_dead", return_value=True), patch.object(pub.time, "monotonic", side_effect=[0, 0, 200]):
                with self.assertRaises(TimeoutError):
                    pub.recover(path)
            self.assertNotIn("restart", w.events[old_events:])
            self.assertNotIn("unfence", w.events[old_events:])
            self.assertEqual(w.units["estate-core.service"], "inactive")

    def test_submit_returns_durable_subject_then_qualified_receipt(self):
        with self.world() as w:
            submitted = pub.submit(10, 10, expected_head="c" * 40)
            self.assertEqual(submitted["status"], "submitted")
            self.assertEqual(submitted["source_head"], "c" * 40)
            self.assertTrue(Path(submitted["receipt"]).parent.joinpath("submission.json").is_file())
            self.assertIsNone(submitted["control_bundle"])
            self.assertEqual(w.events, [])
            pub._publish(10, 10, state_dir=Path(submitted["receipt"]).parent, expected_head="c" * 40)
            completed = pub.submit(10, 10, expected_head="c" * 40)
            self.assertEqual(completed["status"], "published")
            self.assertEqual(completed["receipt"], submitted["receipt"])
            self.assertEqual(completed["engine_fingerprint"], w.engine["fingerprint"])
            self.assertTrue(completed["generation_canary"]["runtime_unchanged"])

    def test_same_subject_pending_reuses_receipt_and_different_head_refuses(self):
        with self.world() as w:
            first = pub.submit(10, 10)
            original = w.api.user_unit_state.side_effect
            w.api.user_unit_state.side_effect = lambda u: "active" if u.startswith("gateway-publish-") else original(u)
            again = pub.submit(10, 10)
            self.assertEqual(first["receipt"], again["receipt"])
            with patch.object(pub, "_capture_committed", return_value=("f" * 40, w.sources, w.code)):
                with self.assertRaisesRegex(RuntimeError, "another committed"):
                    pub.submit(10, 10)

    def test_failed_submitted_worker_does_not_report_published(self):
        with self.world() as w:
            first = pub.submit(10, 10)
            result = pub.submit(10, 10)
            self.assertEqual(result["status"], "recovery-required")
            self.assertEqual(result["receipt"], first["receipt"])

    def test_receipt_cannot_certify_changed_engine_generation(self):
        with self.world() as w:
            first = pub.submit(10, 10)
            pub._publish(10, 10, state_dir=Path(first["receipt"]).parent)
            w.engine["engine_pid"] += 1
            fresh = pub.submit(10, 10)
            self.assertEqual(fresh["status"], "submitted")
            self.assertNotEqual(fresh["receipt"], first["receipt"])

    def test_generation_canary_malformed_success_cannot_release(self):
        with self.world() as w, patch.object(pub.time, "monotonic", side_effect=itertools.count(100)), patch.object(pub, "_wait_empty"):
            w.api.restoration_generation.return_value = {"ok": True}
            with self.assertRaises(RuntimeError):
                pub._publish(10, 10)
            self.assertNotIn("unfence", w.events)

    def test_recovery_timer_uses_snapshot_code_and_recorded_actuator(self):
        with self.world() as w:
            pub._publish(10, 10)
            path = next((w.root / "publications").glob("publish-*/state.json"))
            state = json.loads(path.read_text())
            # Call the actual builder with a mocked systemd-run boundary.
            with patch.object(pub, "_arm_recovery", wraps=SPEC_ARM_RECOVERY), patch.object(pub.subprocess, "run") as run:
                run.return_value = subprocess.CompletedProcess([], 0)
                pub._arm_recovery(path, state)
                argv = run.call_args.args[0]
            self.assertIn(str(path.parent / "code" / "gateway_safe_publish.py"), argv)
            self.assertIn("--setenv=ENGINE_ACTUATOR=" + str(w.actuator), argv)
            self.assertEqual(argv[-2:], ["--recover", str(path)])

    def test_release_pending_reestablishes_closed_drain_and_hold_without_rollback(self):
        with self.world() as w:
            closed = False
            failed = False
            real_save = pub._save_state
            def hold(*args, **kwargs):
                nonlocal closed
                if args[0] == "renew" and closed:
                    return {"refused": "no such live hold (released, expired, or its holder died)"}
                if args[0] == "release":
                    closed = True
                if args[0] == "acquire":
                    closed = False
                return w.hold(*args, **kwargs)
            def save(path, state):
                nonlocal failed
                if state.get("hold_closed") and state.get("status") != "release-pending" and not failed:
                    failed = True
                    raise OSError("kill boundary after hold release")
                real_save(path, state)
            w.api.actuator_hold.side_effect = hold
            with patch.object(pub, "_save_state", side_effect=save):
                with self.assertRaises(OSError):
                    pub._publish(10, 10)
            self.assertEqual(w.state()["status"], "release-pending")
            self.assertIsNotNone(w.lease)  # failed cleanup immediately fences the current process again
            self.assertTrue(pub.DURABLE_DRAIN_STATE.exists())
            self.assertEqual(w.units["estate-core.service"], "inactive")
            self.assertTrue(closed)
            path = next((w.root / "publications").glob("publish-*/state.json"))
            with patch.object(pub, "_owner_dead", return_value=True):
                result = pub.recover(path)
            self.assertEqual(result["status"], "published")
            self.assertEqual(pub.RUNTIME.read_bytes(), w.candidate)
            self.assertEqual(w.events.count("restart"), 1)
            self.assertEqual(w.events.count("hold:acquire"), 2)
            self.assertEqual(w.api.restoration_generation.call_count, 2)
            self.assertFalse(pub.DURABLE_DRAIN_STATE.exists())
            self.assertEqual(w.units["estate-core.service"], "active")

    def test_known_activating_submission_is_pending(self):
        with self.world() as w:
            first = pub.submit(10, 10)
            with patch.object(pub.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "LoadState=loaded\nActiveState=activating\n")):
                again = pub.submit(10, 10)
            self.assertEqual(again["status"], "submitted")
            self.assertEqual(again["receipt"], first["receipt"])

    def test_receipt_does_not_certify_replaced_gateway_process(self):
        with self.world() as w:
            first = pub.submit(10, 10)
            pub._publish(10, 10, state_dir=Path(first["receipt"]).parent)
            w.pid += 1
            fresh = pub.submit(10, 10)
            self.assertEqual(fresh["status"], "submitted")
            self.assertNotEqual(first["receipt"], fresh["receipt"])

    def test_sleeping_running_process_state_is_the_same_generation(self):
        with self.world() as w:
            seen = {}
            def stat(pid):
                seen[pid] = seen.get(pid, 0) + 1
                return ("R" if seen[pid] > 1 else "S", str(pid) + "start")
            with patch.object(pub, "_proc_stat", side_effect=stat):
                completed = pub._publish(10, 10)
                receipt = pub.submit(10, 10)
            self.assertEqual(completed["status"], "published")
            self.assertEqual(receipt["receipt"], completed["receipt"])
            self.assertEqual(receipt["status"], "published")

    def test_engine_change_when_core_resumes_retains_fence_and_repauses_producers(self):
        with self.world() as w:
            original = w.set_unit
            def unit(name, active):
                if name == "estate-core.service" and active:
                    w.engine["engine_pid"] += 1
                return original(name, active)
            w.api.set_user_unit.side_effect = unit
            with self.assertRaisesRegex(RuntimeError, "generation"):
                pub._publish(10, 10)
            self.assertEqual(w.state()["status"], "release-pending")
            self.assertEqual(w.units["estate-core.service"], "inactive")
            self.assertTrue(pub.DURABLE_DRAIN_STATE.exists())
            self.assertNotIn("unfence", w.events)

    def test_failed_refence_after_partial_delete_stops_gateway_without_reverting_candidate(self):
        with self.world() as w:
            original = w.http
            deleted = False
            def http(path, method="GET", payload=None, token=""):
                nonlocal deleted
                if path == "/gateway/drain" and method == "POST" and deleted:
                    raise OSError("gateway unavailable during cleanup recovery")
                row = original(path, method, payload, token)
                if path == "/gateway/drain" and method == "DELETE":
                    deleted = True
                    raise OSError("reply lost after deletion")
                return row
            with patch.object(pub, "_http", side_effect=http):
                with self.assertRaises(OSError):
                    pub._publish(10, 10)
            self.assertEqual(w.pid, 0)
            self.assertEqual(w.state()["status"], "manual-recovery-required")
            self.assertEqual(pub.RUNTIME.read_bytes(), w.candidate)
            self.assertEqual(w.units["estate-core.service"], "inactive")

    def test_unknown_legacy_drain_grant_stops_gateway_and_keeps_core_paused(self):
        with self.world(legacy=True) as w:
            original = w.http
            def http(path, method="GET", payload=None, token=""):
                row = original(path, method, payload, token)
                if path == "/gateway/drain" and method == "POST":
                    raise OSError("lost drain grant reply")
                return row
            with patch.object(pub, "_http", side_effect=http):
                with self.assertRaises(OSError):
                    pub._publish(10, 10)
            self.assertEqual(w.pid, 0)
            self.assertEqual(w.state()["status"], "manual-recovery-required")
            self.assertEqual(pub.RUNTIME.read_bytes(), b"# legacy\n")
            self.assertEqual(w.units["estate-core.service"], "inactive")
            self.assertNotIn("unfence", w.events)


SPEC_ARM_RECOVERY = pub._arm_recovery


if __name__ == "__main__":
    unittest.main()
