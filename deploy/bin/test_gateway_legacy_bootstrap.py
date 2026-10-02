"""The first drain rollout must never restart through an accepted call."""
import unittest
from unittest.mock import patch
import sys
from pathlib import Path
import tempfile
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gateway_legacy_bootstrap as bootstrap


class LegacyBootstrap(unittest.TestCase):
    def _transaction(self, *, drained=True):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            source = base / "source.py"; source.write_bytes(b"new gateway")
            runtime = base / "runtime.py"; runtime.write_bytes(b"old gateway")
            token = base / "token"; token.write_text("admin")
            dropin = base / "grace.conf"; dropin.write_text("TimeoutStopSec=1830\n")
            commands = []
            def run(*args):
                return ("0.0.0.0:8000" if args[0] == "ss" else "30min 30s")
            def drain(*_args):
                commands.append("drained")
                if not drained:
                    raise TimeoutError("accepted call is still active")
            def systemctl(args, **_kwargs):
                if args[0] == "systemctl" and args[1] == "restart":
                    self.assertIn("drained", commands)
                    commands.append("restart")
            with patch.object(bootstrap, "SOURCE", source), \
                 patch.object(bootstrap, "RUNTIME", runtime), \
                 patch.object(bootstrap, "TOKEN_FILE", token), \
                 patch.object(bootstrap, "DROPIN", dropin), \
                 patch.object(bootstrap, "DROPIN_LIVE", base / "installed"), \
                 patch.object(bootstrap.os, "geteuid", return_value=0), \
                 patch.object(bootstrap.os, "chown"), \
                 patch.object(bootstrap.subprocess, "check_output", return_value=b"new gateway"), \
                 patch.object(bootstrap.subprocess, "run", side_effect=systemctl), \
                 patch.object(bootstrap, "_run", side_effect=run), \
                 patch.object(bootstrap, "_healthy", return_value=True), \
                 patch.object(bootstrap, "_http", side_effect=urllib.error.HTTPError(
                     "http://127.0.0.1:8000/gateway/drain", 404, "Not Found", {}, None)), \
                 patch.object(bootstrap, "_wait_drained", side_effect=drain), \
                 patch.object(bootstrap, "_wait_healthy"), \
                 patch.object(bootstrap, "_iptables", side_effect=lambda *args: commands.append(args)):
                if drained:
                    result = bootstrap.publish(timeout_s=1)
                    self.assertEqual(result["status"], "published")
                    self.assertEqual(runtime.read_bytes(), b"new gateway")
                    self.assertEqual(commands.count("restart"), 1)
                else:
                    with self.assertRaises(TimeoutError):
                        bootstrap.publish(timeout_s=1)
                    self.assertEqual(runtime.read_bytes(), b"old gateway")
                    self.assertNotIn("restart", commands)
            self.assertEqual(sum(1 for c in commands if isinstance(c, tuple) and c[0] == "-I"), 2)
            self.assertEqual(sum(1 for c in commands if isinstance(c, tuple) and c[0] == "-D"), 2)

    def test_publish_waits_for_drain_before_restarting(self):
        self._transaction(drained=True)

    def test_timeout_preserves_old_image_and_removes_kernel_fence(self):
        self._transaction(drained=False)

    def test_kernel_fence_expires_and_preserves_existing_connections(self):
        for chain in ("INPUT", "OUTPUT"):
            rule = bootstrap._rule(chain, "2026-10-01T11:00:00")
            self.assertIn("--ctstate", rule)
            self.assertIn("NEW", rule)
            self.assertIn("--datestop", rule)
            self.assertIn("2026-10-01T11:00:00", rule)
            self.assertNotIn("ESTABLISHED", rule)
        self.assertEqual(bootstrap._rule("INPUT", "x")[:3], ["!", "-i", "lo"])
        self.assertEqual(bootstrap._rule("OUTPUT", "x")[:4],
                         ["-m", "owner", "!", "--uid-owner"])

    def test_accepted_call_or_socket_never_means_drained(self):
        cases = [({"active": [{}], "inflight": 1, "waiting": 0}, 0),
                 ({"active": [], "inflight": 0, "waiting": 0}, 1),
                 ({"active": [], "inflight": 0, "waiting": 0}, 0)]
        for lanes, sockets in cases:
            with self.subTest(lanes=lanes, sockets=sockets), \
                 patch.object(bootstrap, "_http", side_effect=lambda path, _token:
                              lanes if path == "/gateway/lanes" else {"in_flight": 0}), \
                 patch.object(bootstrap, "_established", return_value=sockets), \
                 patch.object(bootstrap.time, "sleep", return_value=None), \
                 patch.object(bootstrap.time, "monotonic", side_effect=[0, 0, 2]):
                with self.assertRaises(TimeoutError):
                    bootstrap._wait_drained("token", 1)

    def test_three_independent_empty_samples_are_required(self):
        with patch.object(bootstrap, "_http", side_effect=lambda path, _token:
                          {"active": [], "inflight": 0, "waiting": 0}
                          if path == "/gateway/lanes" else {"in_flight": 0}), \
             patch.object(bootstrap, "_established", return_value=0), \
             patch.object(bootstrap.time, "sleep", return_value=None), \
             patch.object(bootstrap.time, "monotonic", side_effect=[0, 0, 0, 0]):
            self.assertEqual(bootstrap._wait_drained("token", 1)["active"], 0)


if __name__ == "__main__":
    unittest.main()
