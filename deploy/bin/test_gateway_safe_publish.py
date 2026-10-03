"""The publisher refuses to restart while accepted calls remain visible."""
import importlib.util
import shutil
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("gateway_safe_publish_test",
                                               Path(__file__).with_name("gateway_safe_publish.py"))
pub = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pub)


class SafePublish(unittest.TestCase):
    def test_release_pause_is_bounded_owned_and_removed(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(pub, "HALO_START_PAUSE", Path(directory) / "pause.json"):
            token = pub._begin_halo_quiesce(1800, 1500)
            row = json.loads(pub.HALO_START_PAUSE.read_text())
            self.assertEqual(row["token"], token)
            self.assertLessEqual(row["expires_at_epoch"] - row["issued_at_epoch"], 3600)
            with self.assertRaisesRegex(RuntimeError, "another release"):
                pub._begin_halo_quiesce(1800, 1500)
            pub._end_halo_quiesce(token)
            self.assertFalse(pub.HALO_START_PAUSE.exists())

    def test_quiet_check_counts_active_runs_and_refuses_unreadable_authority(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(pub, "HALO_INCIDENTS", Path(directory)):
            path = Path(directory) / "one.json"
            lease = {"expires_at_epoch": time.time() + 600}
            path.write_text(json.dumps({"id": "one", "status": "running", "lease": lease,
                                        "halo_run": {"status": "running", "run_id": "run-1"}}))
            self.assertEqual(pub._halo_active_runs(), ["one"])
            with patch.object(pub.time, "monotonic", side_effect=[0, 2]):
                with self.assertRaisesRegex(TimeoutError, "one"):
                    pub._wait_halo_quiet(1)
            path.write_text(json.dumps({"id": "one", "status": "recovered", "lease": lease,
                                        "halo_run": {"status": "running", "run_id": "run-1"}}))
            self.assertEqual(pub._halo_active_runs(), [])
            path.write_text("{")
            with self.assertRaisesRegex(RuntimeError, "unreadable Halo incident"):
                pub._halo_active_runs()

    def test_release_refuses_if_installed_supervisor_cannot_read_lease(self):
        with patch.object(pub.subprocess, "check_output", return_value="null\n"):
            with self.assertRaisesRegex(RuntimeError, "exact release lease"):
                pub._assert_halo_quiesce_live("expected")

    def test_stop_grace_parser_matches_live_systemd_units(self):
        self.assertEqual(pub._unit_seconds("15s"), 15)
        self.assertEqual(pub._unit_seconds("30min 30s"), 1830)
        self.assertEqual(pub._unit_seconds("1830000000us"), 1830)

    def test_drain_cannot_be_treated_as_empty_after_lease_expiry(self):
        with patch.object(pub, "_http", side_effect=[
            {"draining": False, "active": 0, "until": 0}, {"in_flight": 0}]):
            with self.assertRaisesRegex(RuntimeError, "expired"):
                pub._wait_empty("token", 10)

    def test_a_visible_accepted_call_times_out_without_restarting(self):
        with patch.object(pub, "_http", side_effect=lambda path, **kw:
                          {"draining": True, "active": 1, "until": pub.time.time() + 100}
                          if path == "/gateway/drain" else {"in_flight": 0}), \
             patch.object(pub.time, "sleep", return_value=None), \
             patch.object(pub.time, "monotonic", side_effect=[0, 0, 2]):
            with self.assertRaises(TimeoutError):
                pub._wait_empty("token", 1)

    def test_failed_release_cannot_rollback_through_an_unknown_drain_owner(self):
        with patch.object(pub, "_http", return_value={"draining": True, "active": 1}), \
             patch.object(pub, "_wait_empty") as wait:
            with self.assertRaisesRegex(RuntimeError, "unknown drain owner"):
                pub._fence_failed_release("token", 10)
            wait.assert_not_called()

    def test_failed_release_fences_new_arrivals_and_waits_for_accepted_calls(self):
        # RL/live_guard 2026-10-03: this test used to write the LIVE gateway-publish-drain.json + audit log
        tmp = Path(tempfile.mkdtemp(prefix="sp-fence-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        with patch.object(pub, "_http", side_effect=[
                {"draining": False, "active": 1}, {"lease": "new-release"}]) as http, \
             patch.object(pub, "DRAIN_STATE", tmp / "drain.json"), patch.object(pub, "AUDIT_LOG", tmp / "audit.log"), \
             patch.object(pub, "_wait_empty") as wait:
            pub._fence_failed_release("token", 10)
            http.assert_any_call("/gateway/drain", "POST", {
                "ttl_s": 1800, "reason": "governed gateway rollback", "by": pub._BY}, "token")
            wait.assert_called_once_with("token", 10)


def _dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


class FakeGateway:
    """Just enough of /gateway/drain to prove a fence is opened and then closed."""

    def __init__(self):
        self.lease = None
        self.events = []
        self.offline = False          # GW2: a planned offline window the publisher must not restart inside

    def __call__(self, path, method="GET", payload=None, token=""):
        if path == "/health":
            return {"http_status": 200}
        if path == "/gateway/offline":
            return {"offline": self.offline, "reason": "K5 A/B" if self.offline else None,
                    "by": "K5" if self.offline else None, "remaining_s": 900 if self.offline else 0}
        if path == "/gateway/spend":
            return {"enforce": True, "durable": True, "cap": 25.0, "in_flight": 0}
        assert path == "/gateway/drain", path
        if method == "POST":
            self.lease = "lease-1"
            self.events.append("open")
            return {"lease": self.lease, "draining": True, "active": 1, "until": time.time() + 1800}
        if method == "DELETE":
            if payload.get("lease") != self.lease:
                raise RuntimeError("409 drain lease mismatch")
            self.lease = None
            self.events.append("delete")
            return {"draining": False}
        return {"draining": self.lease is not None, "active": 1 if self.lease else 0,
                "until": time.time() + 1800 if self.lease else None}


class TerminationReleasesEverything(unittest.TestCase):
    """L77 (2026-10-03): `timeout 900` SIGTERMed a publish mid-drain and its Halo start pause stood for an hour."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = Path(self.tmp.name)
        (d / "incidents").mkdir()
        (d / "keepalive-shim.py").write_text("new gateway\n")
        (d / "dash.html").write_text("page\n")
        (d / "live-shim.py").write_text("old gateway\n")
        self.paths = {"HALO_START_PAUSE": d / "pause.json", "DRAIN_STATE": d / "drain.json",
                      "AUDIT_LOG": d / "audit.log", "HALO_INCIDENTS": d / "incidents",
                      "SOURCE": d / "keepalive-shim.py", "DASH_SOURCE": d / "dash.html",
                      "DASH_RUNTIME": d / "live-dash.html", "RUNTIME": d / "live-shim.py",
                      "HALO_SUPERVISOR_LOCK": d / "supervisor.lock", "DROPIN": d / "dropin.conf",
                      "DROPIN_LIVE": d / "dropin-live.conf", "ACTUATOR": d / "no-actuator.py"}
        self.gw = FakeGateway()

        def check_output(args, **kw):
            if args[:2] == ["git", "show"]:
                return (d / ("keepalive-shim.py" if args[2].endswith("keepalive-shim.py") else "dash.html")).read_bytes()
            return "30min 30s\n"            # systemctl show -p TimeoutStopUSec

        self.patches = [patch.object(pub, k, v) for k, v in self.paths.items()]
        self.patches += [patch.object(pub, "_http", self.gw), patch.object(pub.subprocess, "check_output", check_output),
                         patch.object(pub, "_run"), patch.object(pub, "_assert_halo_quiesce_live"),
                         patch.dict(os.environ, {"SHIM_ADMIN_TOKEN": "t"})]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def assert_nothing_held(self):
        self.assertFalse(pub.HALO_START_PAUSE.exists(), "Halo start pause left behind")
        self.assertFalse(pub.DRAIN_STATE.exists(), "drain record left behind")
        self.assertIsNone(self.gw.lease, "drain fence left up")
        self.assertEqual(sorted(p.name for p in Path(self.tmp.name).glob("*.tmp.*")), [], "temp state left behind")
        self.assertEqual(list(Path(self.tmp.name).glob("live-shim.py.bak-*")), [], "unused backup left behind")

    def test_signals_mid_drain_release_pause_fence_and_temp_state(self):
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            with self.subTest(signal=sig.name):
                self.gw.events.clear()

                def drain_then_killed(token, timeout_s):
                    self.assertTrue(pub.HALO_START_PAUSE.exists())       # held while draining
                    self.assertEqual(self.gw.lease, "lease-1")
                    os.kill(os.getpid(), sig)
                    time.sleep(2)                                         # the handler raises well before this returns
                    self.fail("signal was not delivered")

                before = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
                with patch.object(pub, "_wait_empty", drain_then_killed), \
                        patch.object(pub, "_wait_halo_quiet"):
                    with self.assertRaises(pub.PublishTerminated) as cm:
                        pub.publish(10, 10)
                self.assertEqual(cm.exception.signum, sig)
                self.assertEqual(self.gw.events, ["open", "delete"])
                self.assert_nothing_held()
                self.assertEqual({s: signal.getsignal(s) for s in before}, before)   # handlers given back
                self.assertTrue(pub.AUDIT_LOG.read_text().count("terminated-before-install") >= 1)

    def test_signal_while_waiting_for_halo_releases_the_pause(self):
        def killed(_timeout):
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(2)
        with patch.object(pub, "_wait_halo_quiet", killed):
            with self.assertRaises(pub.PublishTerminated):
                pub.publish(10, 10)
        self.assert_nothing_held()

    def test_ordinary_failure_also_releases_everything(self):
        with patch.object(pub, "_wait_halo_quiet"), \
                patch.object(pub, "_wait_empty", side_effect=TimeoutError("accepted calls did not drain")):
            with self.assertRaises(TimeoutError):
                pub.publish(10, 10)
        self.assertEqual(self.gw.events, ["open", "delete"])
        self.assert_nothing_held()

    def test_refuses_to_start_inside_a_planned_offline_window(self):
        """GW2 2026-10-03: a 09:11 publish restarted the gateway inside K5's engine window and dropped the window."""
        self.gw.offline = True
        with patch.object(pub, "_wait_halo_quiet"), patch.object(pub, "_wait_empty"):
            with self.assertRaisesRegex(RuntimeError, "planned offline window open .* at start"):
                pub.publish(10, 10)
        self.assertEqual(self.gw.events, [])                  # never even opened the fence
        self.assertEqual(pub.RUNTIME.read_text(), "old gateway\n")
        self.assert_nothing_held()

    def test_aborts_before_restart_when_a_window_opens_mid_drain(self):
        def drained(token, timeout_s):
            self.gw.offline = True                             # K5 opened its window while the publisher waited
        with patch.object(pub, "_wait_halo_quiet"), patch.object(pub, "_wait_empty", drained), \
                patch.object(pub, "_run") as run:
            with self.assertRaisesRegex(RuntimeError, "at restart"):
                pub.publish(10, 10)
        self.assertNotIn(("sudo", "-n", "systemctl", "restart", pub.SERVICE), [c.args for c in run.call_args_list])
        self.assertEqual(pub.RUNTIME.read_text(), "old gateway\n")
        self.assertEqual(self.gw.events, ["open", "delete"])
        self.assert_nothing_held()

    def test_signal_after_install_does_not_start_a_rollback_but_releases_leases(self):
        def restart_then_killed(*args):
            if args[:3] == ("sudo", "-n", "systemctl") and args[3] == "restart":
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(2)
        with patch.object(pub, "_wait_halo_quiet"), patch.object(pub, "_wait_empty"), \
                patch.object(pub, "_run", restart_then_killed), \
                patch.object(pub, "_fence_failed_release") as fence:
            with self.assertRaises(pub.PublishTerminated):
                pub.publish(10, 10)
        fence.assert_not_called()                      # no long fenced rollback inside a dying process
        self.assertEqual(pub.RUNTIME.read_text(), "new gateway\n")
        self.assertIn("terminated-after-install", pub.AUDIT_LOG.read_text())
        self.assertFalse(pub.HALO_START_PAUSE.exists())
        self.assertIsNone(self.gw.lease)
        self.assertFalse(pub.DRAIN_STATE.exists())

    def test_a_real_sigterm_to_a_real_process_leaves_no_lease(self):
        """End to end: the same shape as the incident (child process, outer SIGTERM while draining)."""
        d = Path(self.tmp.name)
        driver = textwrap.dedent(f"""
            import importlib.util, json, os, sys, time
            from pathlib import Path
            from unittest.mock import patch
            spec = importlib.util.spec_from_file_location("pub", {str(Path(pub.__file__))!r})
            pub = importlib.util.module_from_spec(spec); spec.loader.exec_module(pub)
            d = Path({self.tmp.name!r})
            lease = {{"v": None}}
            def http(path, method="GET", payload=None, token=""):
                if path == "/health": return {{"http_status": 200}}
                if path == "/gateway/offline": return {{"offline": False}}
                if path == "/gateway/spend": return {{"enforce": True, "durable": True, "cap": 25.0}}
                if method == "POST": lease["v"] = "L"; return {{"lease": "L"}}
                if method == "DELETE":
                    (d / "fence-closed").write_text("1"); lease["v"] = None; return {{}}
                return {{"draining": lease["v"] is not None, "active": 1}}
            def check_output(args, **kw):
                if args[:2] == ["git", "show"]:
                    return (d / ("keepalive-shim.py" if args[2].endswith("keepalive-shim.py") else "dash.html")).read_bytes()
                return "30min 30s\\n"
            def wait_empty(token, timeout_s):
                (d / "draining").write_text("1")
                time.sleep(60)
            for k, v in {{"HALO_START_PAUSE": d/"pause.json", "DRAIN_STATE": d/"drain.json", "AUDIT_LOG": d/"audit.log",
                         "HALO_INCIDENTS": d/"incidents", "SOURCE": d/"keepalive-shim.py", "DASH_SOURCE": d/"dash.html",
                         "DASH_RUNTIME": d/"live-dash.html", "RUNTIME": d/"live-shim.py", "DROPIN": d/"x", "DROPIN_LIVE": d/"y",
                         "ACTUATOR": d/"no-actuator.py"}}.items():
                setattr(pub, k, v)
            pub._http, pub._run, pub._wait_empty = http, lambda *a: None, wait_empty
            pub._assert_halo_quiesce_live = lambda t: None
            pub.subprocess.check_output = check_output
            os.environ["SHIM_ADMIN_TOKEN"] = "t"
            try:
                pub.publish(10, 10)
            except pub.PublishTerminated:
                sys.exit(143)
        """)
        proc = subprocess.Popen([sys.executable, "-c", driver], stderr=subprocess.PIPE)
        try:
            deadline = time.time() + 20
            while not (d / "draining").exists():
                self.assertIsNone(proc.poll(), proc.stderr.read().decode() if proc.poll() is not None else "")
                self.assertLess(time.time(), deadline)
                time.sleep(0.05)
            self.assertTrue((d / "pause.json").exists())                  # the lease the incident left behind
            proc.send_signal(signal.SIGTERM)
            self.assertEqual(proc.wait(timeout=20), 143)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.stderr.close()
        self.assertFalse((d / "pause.json").exists())
        self.assertFalse((d / "drain.json").exists())
        self.assertTrue((d / "fence-closed").exists())

    def test_signal_between_pause_write_and_token_return_is_still_released_by_pid(self):
        pub._begin_halo_quiesce(10, 10)                 # the caller never received the token
        pub._end_halo_quiesce(None)
        self.assertFalse(pub.HALO_START_PAUSE.exists())
        pub.HALO_START_PAUSE.write_text(json.dumps({"owner": "gateway-safe-publish", "token": "x", "pid": _dead_pid(),
                                                    "expires_at_epoch": time.time() + 99}))
        pub._end_halo_quiesce(None)                     # someone else's lease is never removed by the pid fallback
        self.assertTrue(pub.HALO_START_PAUSE.exists())

    def test_atomic_write_removes_its_temp_file_when_interrupted(self):
        target = Path(self.tmp.name) / "x.json"
        with patch.object(pub.os, "replace", side_effect=pub.PublishTerminated(15)):
            with self.assertRaises(pub.PublishTerminated):
                pub._atomic_write(target, b"data")
        self.assertEqual(list(Path(self.tmp.name).glob("x.json*")), [])


class StaleLeaseRecovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = Path(self.tmp.name)
        for k, v in {"HALO_START_PAUSE": d / "pause.json", "DRAIN_STATE": d / "drain.json",
                     "AUDIT_LOG": d / "audit.log"}.items():
            p = patch.object(pub, k, v)
            p.start()
            self.addCleanup(p.stop)

    def write_lease(self, **over):
        row = {"owner": "gateway-safe-publish", "token": "deadbeef" * 4, "issued_at_epoch": time.time(),
               "expires_at_epoch": time.time() + 3000}
        row.update(over)
        pub.HALO_START_PAUSE.write_text(json.dumps(row))

    def test_dead_owner_pid_lets_the_next_publish_take_over_with_an_audit_line(self):
        dead = _dead_pid()
        self.write_lease(pid=dead)
        token = pub._begin_halo_quiesce(1800, 1500)
        row = json.loads(pub.HALO_START_PAUSE.read_text())
        self.assertEqual((row["token"], row["pid"]), (token, os.getpid()))
        audit = json.loads(pub.AUDIT_LOG.read_text().splitlines()[-1])
        self.assertEqual((audit["event"], audit["dead_pid"]), ("halo-start-pause-takeover", dead))

    def test_live_owner_pid_is_still_refused(self):
        self.write_lease(pid=os.getpid(), pid_start=pub._own_stamp()["pid_start"])
        with self.assertRaisesRegex(RuntimeError, "another release owns"):
            pub._begin_halo_quiesce(1800, 1500)

    def test_reused_pid_is_not_mistaken_for_a_live_owner(self):
        self.write_lease(pid=os.getpid(), pid_start="1")           # same pid number, different process start time
        pub._begin_halo_quiesce(1800, 1500)
        self.assertIn("takeover", pub.AUDIT_LOG.read_text())

    def test_zombie_owner_counts_as_dead(self):
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        deadline = time.time() + 5
        while pub._proc_stat(child.pid)[0] != "Z" and time.time() < deadline:
            time.sleep(0.02)
        try:
            self.assertTrue(pub._owner_dead({"pid": child.pid}))
        finally:
            child.wait()

    def test_legacy_lease_without_a_pid_and_other_owners_are_never_taken_over(self):
        self.write_lease()                                           # written by the pre-L77 publisher
        with self.assertRaisesRegex(RuntimeError, "another release owns"):
            pub._begin_halo_quiesce(1800, 1500)
        self.write_lease(owner="someone-else", pid=_dead_pid())
        with self.assertRaisesRegex(RuntimeError, "another release owns"):
            pub._begin_halo_quiesce(1800, 1500)

    def test_orphan_drain_fence_of_a_dead_publisher_is_closed_by_recorded_lease(self):
        pub.DRAIN_STATE.write_text(json.dumps({"lease": "orphan", "pid": _dead_pid(), "by": "old publisher"}))
        with patch.object(pub, "_http", return_value={"draining": False}) as http:
            self.assertTrue(pub._reap_orphan_drain("t"))
        http.assert_called_once_with("/gateway/drain", "DELETE", {"lease": "orphan"}, "t")
        self.assertFalse(pub.DRAIN_STATE.exists())
        self.assertIn("orphan-drain-released", pub.AUDIT_LOG.read_text())

    def test_a_live_publishers_or_unrecorded_fence_is_never_reaped(self):
        pub.DRAIN_STATE.write_text(json.dumps({"lease": "mine", "pid": os.getpid(), **{"pid_start": pub._own_stamp()["pid_start"]}}))
        with patch.object(pub, "_http") as http:
            self.assertFalse(pub._reap_orphan_drain("t"))
            http.assert_not_called()
        pub.DRAIN_STATE.unlink()
        self.assertFalse(pub._reap_orphan_drain("t"))                # no record: the fence belongs to someone else

    def test_orphan_record_for_a_replaced_fence_is_refused_and_kept_out_of_the_way(self):
        pub.DRAIN_STATE.write_text(json.dumps({"lease": "stale", "pid": _dead_pid()}))
        with patch.object(pub, "_http", side_effect=RuntimeError("409 drain lease mismatch")):
            self.assertFalse(pub._reap_orphan_drain("t"))
        self.assertIn("orphan-drain-release-refused", pub.AUDIT_LOG.read_text())


if __name__ == "__main__":
    unittest.main()
