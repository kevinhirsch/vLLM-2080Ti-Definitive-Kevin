"""The publisher refuses to restart while accepted calls remain visible."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("gateway_safe_publish_test",
                                               Path(__file__).with_name("gateway_safe_publish.py"))
pub = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pub)


class SafePublish(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
