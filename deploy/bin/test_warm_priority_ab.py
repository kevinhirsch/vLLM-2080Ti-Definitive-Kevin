#!/usr/bin/env python3
"""L77 class (lane A6): a live-config A/B driver must restore the config when it is killed, not only when it returns.

warm_priority_ab.py flips SHIM_WARM_PRIORITY on the LIVE gateway for an hour and promises 'finally: OFF'. A SIGTERM (an outer
`timeout`, a window abort, systemd) does not run `finally` in Python, so the live gateway stayed in the ON arm: the same
leak L77 found in gateway_safe_publish. Driven here against a stub gateway + engine.

Run:  python -m pytest -q test_warm_priority_ab.py
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

SCRIPT = Path(__file__).resolve().with_name("warm_priority_ab.py")


class Stub:
    def __init__(self):
        self.state = {"warm_priority": 0, "planned_offline": False}
        self.posts = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj, ctype="application/json"):
                raw = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == "/gateway/config":
                    self._send({"warm_priority": outer.state["warm_priority"]})
                elif self.path == "/gateway/capacity":
                    self._send({"planned_offline": outer.state["planned_offline"]})
                elif self.path == "/metrics":
                    self._send(b"vllm:num_preemptions_total 0\n", "text/plain")
                else:
                    self.send_error(404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                outer.posts.append(body)
                outer.state["warm_priority"] = body.get("warm_priority", outer.state["warm_priority"])
                self._send({"ok": True})

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        self.url = "http://127.0.0.1:%d" % self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


class KilledMidArm(unittest.TestCase):
    def run_and_kill(self, sig):
        stub = Stub()
        self.addCleanup(stub.close)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tok = Path(tmp.name) / "admin.token"
        tok.write_text("t\n")
        env = dict(os.environ, WARM_AB_GATEWAY=stub.url, WARM_AB_ENGINE=stub.url, WARM_AB_TOKEN_FILE=str(tok))
        p = subprocess.Popen([sys.executable, str(SCRIPT), "run", "--out", str(Path(tmp.name) / "ab.jsonl"), "--arm-min", "10",
                              "--arms", "4", "--poll-s", "0.2"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            deadline = time.time() + 20
            while time.time() < deadline and stub.state["warm_priority"] != 0:     # arm 0 is OFF; wait for arm 1 (ON)
                time.sleep(0.05)
            # arm 0 is OFF and lasts 10 min: shorten by driving the script straight into an ON arm is not possible, so
            # accept either arm: the invariant is what the gateway holds AFTER the kill
            time.sleep(0.5)
            stub.state["warm_priority"] = 1                  # simulate the ON arm being live when the kill lands
            p.send_signal(sig)
            rc = p.wait(timeout=20)
        finally:
            if p.poll() is None:
                p.kill()
        return stub, rc, Path(tmp.name) / "ab.jsonl"

    def test_sigterm_restores_warm_priority_off(self):
        stub, rc, out = self.run_and_kill(signal.SIGTERM)
        self.assertEqual(stub.state["warm_priority"], 0, "live gateway left in the ON arm after SIGTERM")
        self.assertEqual(rc, 128 + signal.SIGTERM)
        self.assertIn('"restored"', out.read_text())

    def test_sighup_restores_warm_priority_off(self):
        stub, rc, out = self.run_and_kill(signal.SIGHUP)
        self.assertEqual(stub.state["warm_priority"], 0)


if __name__ == "__main__":
    unittest.main()
