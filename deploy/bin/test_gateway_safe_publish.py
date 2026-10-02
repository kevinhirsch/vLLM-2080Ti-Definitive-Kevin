"""The publisher refuses to restart while accepted calls remain visible."""
import importlib.util
import json
from pathlib import Path
import tempfile
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
            path.write_text(json.dumps({"id": "one", "status": "running",
                                        "halo_run": {"status": "running", "run_id": "run-1"}}))
            self.assertEqual(pub._halo_active_runs(), ["one"])
            with patch.object(pub.time, "monotonic", side_effect=[0, 2]):
                with self.assertRaisesRegex(TimeoutError, "one"):
                    pub._wait_halo_quiet(1)
            path.write_text(json.dumps({"id": "one", "status": "recovered",
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
        with patch.object(pub, "_http", side_effect=[
                {"draining": False, "active": 1}, {"lease": "new-release"}]) as http, \
             patch.object(pub, "_wait_empty") as wait:
            pub._fence_failed_release("token", 10)
            http.assert_any_call("/gateway/drain", "POST", {
                "ttl_s": 1800, "reason": "governed gateway rollback", "by": pub._BY}, "token")
            wait.assert_called_once_with("token", 10)


if __name__ == "__main__":
    unittest.main()
