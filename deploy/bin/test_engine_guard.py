"""EB-122 (2026-10-21): the suite-wide engine guard in conftest.py is in force.

No deploy/bin test may reach the production engine on 127.0.0.1:8001; a test that forgets to stub the engine must fail
fast, the same with the 27B up or down, instead of passing only while it is up (or hanging while it is held).
"""
import os
import socket
import time
import unittest
from urllib.parse import urlsplit


@unittest.skipIf(os.environ.get("DEPLOY_BIN_TESTS_REAL_ENGINE") == "1", "opted out of the engine guard")
class EngineGuard(unittest.TestCase):
    def test_the_shim_upstream_is_not_the_production_engine(self):
        up = urlsplit(os.environ["SHIM_UPSTREAM"])
        self.assertNotEqual((up.hostname, up.port), ("127.0.0.1", 8001))
        self.assertLessEqual(float(os.environ["SHIM_LOCAL_ONLY_HOLD_SECS"]), 5)

    def test_the_guarded_upstream_refuses_at_once(self):
        up = urlsplit(os.environ["SHIM_UPSTREAM"])
        t0 = time.monotonic()
        with self.assertRaises(OSError):
            socket.create_connection((up.hostname, up.port), timeout=2).close()
        self.assertLess(time.monotonic() - t0, 2)


if __name__ == "__main__":
    unittest.main()
