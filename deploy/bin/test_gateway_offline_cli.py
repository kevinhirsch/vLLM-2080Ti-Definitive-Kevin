"""gateway-offline.py persists its lease so a lost wrapper cannot strand a planned-offline window (S3 2026-10-02)."""
import importlib.util
import json
import os
import signal
import stat
import subprocess
import sys
import shutil
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "gateway-offline.py")


def load(lease_file):
    os.environ["OFFLINE_LEASE_FILE"] = lease_file
    spec = importlib.util.spec_from_file_location("gateway_offline_cli_test", PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class LeaseFileTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)
        self.lf = os.path.join(self.d, "sub", "offline-lease.json")
        self.m = load(self.lf)

    def test_save_is_mode_600_and_roundtrips(self):
        self.m.save_lease("abc", "S3", "bench", 600)
        self.assertEqual(stat.S_IMODE(os.stat(self.lf).st_mode), 0o600)
        cur = self.m.load_lease()
        self.assertEqual((cur["lease"], cur["by"], cur["reason"]), ("abc", "S3", "bench"))
        self.assertGreater(cur["until"], time.time() + 590)

    def test_drop_only_matching_lease(self):
        self.m.save_lease("abc", "S3", "bench", 60)
        self.m.drop_lease("other")
        self.assertIsNotNone(self.m.load_lease())
        self.m.drop_lease("abc")
        self.assertIsNone(self.m.load_lease())

    def test_close_without_lease_uses_recorded_lease(self):
        self.m.save_lease("abc", "S3", "bench", 60)
        calls = []
        self.m.http = lambda path, method="GET", payload=None: calls.append((path, method, payload)) or {"offline": False}
        sys.argv = ["gateway-offline.py", "close"]
        self.assertEqual(self.m.main(), 0)
        self.assertEqual(calls, [("/gateway/offline", "DELETE", {"lease": "abc"})])
        self.assertIsNone(self.m.load_lease())

    def test_close_stale_lease_clears_the_record(self):
        self.m.save_lease("abc", "S3", "bench", 60)
        self.m.http = lambda *a, **k: {"error": "offline lease mismatch", "http": 409}
        sys.argv = ["gateway-offline.py", "close"]
        self.m.main()
        self.assertIsNone(self.m.load_lease())

    def test_close_with_nothing_recorded_fails_loudly(self):
        sys.argv = ["gateway-offline.py", "close"]
        self.assertEqual(self.m.main(), 2)

    def test_run_records_then_clears_and_closes_on_term(self):
        calls = []

        def fake_http(path, method="GET", payload=None):
            calls.append((method, payload))
            if method == "POST":
                return {"lease": "L1", "local_active": 0}
            return {"local_active": 0}

        self.m.http = fake_http
        seen = {}

        def fake_call(cmd):
            seen["recorded"] = self.m.load_lease()
            os.kill(os.getpid(), signal.SIGTERM)          # the wrapper is TERM'd mid-command
            time.sleep(1)
            return 0

        self.m.subprocess.call = fake_call
        sys.argv = ["gateway-offline.py", "run", "--by", "S3", "--reason", "t", "--", "true"]
        with self.assertRaises(SystemExit) as cm:
            self.m.main()
        self.assertEqual(cm.exception.code, 143)
        self.assertEqual(seen["recorded"]["lease"], "L1")
        self.assertIn(("DELETE", {"lease": "L1"}), calls)     # closed despite TERM
        self.assertIsNone(self.m.load_lease())

    def test_term_while_opening_the_window_closes_it(self):
        """L77: the signal handlers are live before the window opens, and the lease is recorded at once."""
        calls = []

        def fake_http(path, method="GET", payload=None):
            calls.append((method, payload))
            if method == "POST":
                return {"lease": "L2"}
            if method == "GET":
                os.kill(os.getpid(), signal.SIGTERM)       # TERM'd while waiting for accepted local work
                time.sleep(1)
            return {}

        self.m.http = fake_http
        sys.argv = ["gateway-offline.py", "open", "--by", "S3", "--wait-s", "30"]
        with self.assertRaises(SystemExit) as cm:
            self.m.main()
        self.assertEqual(cm.exception.code, 143)
        self.assertIn(("DELETE", {"lease": "L2"}), calls)
        self.assertIsNone(self.m.load_lease())

    def test_handlers_are_given_back_after_main(self):
        before = signal.getsignal(signal.SIGTERM)
        self.m.http = lambda *a, **k: {"offline": False}
        sys.argv = ["gateway-offline.py", "status"]
        self.m.main()
        sys.argv = ["gateway-offline.py", "run", "--", "true"]
        self.m.http = lambda path, method="GET", payload=None: {"lease": "L3"} if method == "POST" else {"local_active": 0}
        self.m.main()
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)

    def test_orphaned_run_window_of_a_dead_wrapper_is_closed_by_the_next_run(self):
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        self.m.save_lease("OLD", "S3", "bench", 600, mode="run")
        rec = self.m.load_lease()
        rec["pid"] = dead.pid
        with open(self.lf, "w") as fh:
            json.dump(rec, fh)
        calls = []

        def fake_http(path, method="GET", payload=None):
            calls.append((method, payload))
            return {"lease": "NEW", "local_active": 0} if method == "POST" else {"local_active": 0}

        self.m.http = fake_http
        sys.argv = ["gateway-offline.py", "run", "--", "true"]
        self.m.main()
        self.assertEqual(calls[0], ("DELETE", {"lease": "OLD"}))
        self.assertIsNone(self.m.load_lease())

    def test_open_records_and_a_live_run_record_are_never_presumed_dead(self):
        self.m.save_lease("A", "S3", "bench", 60, mode="open")   # the CLI that wrote it exits by design
        self.assertFalse(self.m.owner_dead(self.m.load_lease()))
        self.m.save_lease("B", "S3", "bench", 60, mode="run")    # this very process is alive
        self.assertFalse(self.m.owner_dead(self.m.load_lease()))


if __name__ == "__main__":
    unittest.main()
