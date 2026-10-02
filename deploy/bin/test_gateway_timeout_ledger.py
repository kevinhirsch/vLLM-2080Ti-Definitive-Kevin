"""RS: every gateway-layer timeout is one durable record (layer, deadline, wait, size, outcome)."""
import importlib.util, json, tempfile, unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("shim_timeout_test", Path(__file__).with_name("keepalive-shim.py"))
shim = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shim)


class Req:
    headers = {"X-Client": "halo-hermes"}
    remote = "10.0.1.95"


class TimeoutLedger(unittest.TestCase):
    def test_note_writes_layer_deadline_wait_and_size(self):
        old = shim._TIMEOUT_LEDGER
        shim._TIMEOUT_LEDGER = tempfile.mkdtemp() + "/timeouts.jsonl"
        r = Req()
        shim._ACTIVE[id(r)] = {"name": "halo", "t0": shim.time.time() - 40, "bg": False, "stream": True, "route": "local"}
        try:
            shim._timeout_note(r, "gateway:first-token", 30, shim.LOCAL, b'{"messages":[{"role":"user","content":"x"}]}', outcome="failover")
            row = json.loads(open(shim._TIMEOUT_LEDGER).read())
        finally:
            shim._TIMEOUT_LEDGER = old
            shim._ACTIVE.pop(id(r), None)
        self.assertEqual((row["layer"], row["deadline_s"], row["base"], row["outcome"], row["client"]),
                         ("gateway:first-token", 30.0, "local", "failover", "halo"))
        self.assertGreaterEqual(row["waited_s"], 39)

    def test_bookkeeping_failure_never_raises(self):
        old = shim._TIMEOUT_LEDGER
        shim._TIMEOUT_LEDGER = "/proc/definitely/not/writable/x.jsonl"
        try:
            shim._timeout_note(Req(), "gateway:stream-idle", 5, shim.LOCAL, b"{}")
        finally:
            shim._TIMEOUT_LEDGER = old


class RemoteBreakerPersistence(unittest.TestCase):
    def test_402_breaker_survives_a_gateway_restart(self):
        import time
        old = (shim._REMOTE_DEAD_FILE, shim._REMOTE_DEAD_PERSIST, shim._remote_dead_until)
        shim._REMOTE_DEAD_FILE = tempfile.mkdtemp() + "/dead.json"
        shim._REMOTE_DEAD_PERSIST = True
        try:
            shim._remote_dead_until = 0.0
            shim._note_remote_status("https://api.remote.example", 402)
            self.assertGreater(shim._remote_dead_until, time.time())
            shim._remote_dead_until = 0.0            # "restart": memory gone
            shim._remote_dead_load()
            self.assertGreater(shim._remote_dead_until, time.time())
            shim._note_remote_status("https://api.remote.example", 200)   # remote recovered: cleared and persisted
            shim._remote_dead_until = 99.0
            shim._remote_dead_load()
            self.assertEqual(shim._remote_dead_until, 99.0)               # stale/cleared file does not re-arm
        finally:
            shim._REMOTE_DEAD_FILE, shim._REMOTE_DEAD_PERSIST, shim._remote_dead_until = old


if __name__ == "__main__":
    unittest.main()
