"""A second gateway process cannot spend against a stale copy of the ledger."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "shim_single_spend_owner_test", Path(__file__).with_name("keepalive-shim.py"))
shim = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shim)


class SpendOwner(unittest.TestCase):
    def test_separate_process_holding_ledger_refuses_second_gateway(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = str(Path(directory) / "spend.json")
            script = (
                "import fcntl,os,sys; "
                "fd=os.open(sys.argv[1],os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600); "
                "fcntl.flock(fd,fcntl.LOCK_EX); print('ready',flush=True); sys.stdin.read(1)"
            )
            owner = subprocess.Popen(
                [sys.executable, "-c", script, ledger + ".owner.lock"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(owner.stdout.readline().strip(), "ready")
                with self.assertRaisesRegex(RuntimeError, "already owns"):
                    shim._claim_spend_authority(ledger)
            finally:
                owner.stdin.write("x")
                owner.stdin.flush()
                owner.wait(timeout=5)
                owner.stdin.close()
                owner.stdout.close()
            try:
                shim._claim_spend_authority(ledger)
                self.assertIsNotNone(shim._RUNTIME_SPEND_LOCK)
            finally:
                if shim._RUNTIME_SPEND_LOCK is not None:
                    os.close(shim._RUNTIME_SPEND_LOCK)
                    shim._RUNTIME_SPEND_LOCK = None


if __name__ == "__main__":
    unittest.main()
