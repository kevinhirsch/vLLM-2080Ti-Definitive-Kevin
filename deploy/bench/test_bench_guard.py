"""FX2: bench_guard refuses engine-direct benches unless a gateway window is open (or BENCH_DIRECT_OK=1)."""
import http.server, json, os, subprocess, sys, threading, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_guard as g


def gw(offline, by="S4"):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            b = json.dumps({"offline": offline, "by": by}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    s = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_port}"


class Guard(unittest.TestCase):
    def run_cli(self, url, **env):
        e = {k: v for k, v in os.environ.items() if k != "BENCH_DIRECT_OK"}; e.update(env)
        return subprocess.run([sys.executable, os.path.join(HERE, "bench_guard.py"), "--gateway", url], env=e, capture_output=True, text=True, timeout=30)

    def test_refused_with_exit_3_and_the_exact_command_when_no_window(self):
        s, url = gw(False)
        try:
            r = self.run_cli(url)
        finally:
            s.shutdown(); s.server_close()
        self.assertEqual(r.returncode, 3)
        self.assertIn("REFUSED", r.stderr); self.assertIn("gateway-offline.py", r.stderr); self.assertIn("BENCH_DIRECT_OK=1", r.stderr)

    def test_allowed_inside_a_window(self):
        s, url = gw(True)
        try:
            r = self.run_cli(url)
        finally:
            s.shutdown(); s.server_close()
        self.assertEqual(r.returncode, 0); self.assertIn("window open", r.stderr)

    def test_explicit_override_is_allowed_and_noted(self):
        s, url = gw(False)
        try:
            r = self.run_cli(url, BENCH_DIRECT_OK="1")
        finally:
            s.shutdown(); s.server_close()
        self.assertEqual(r.returncode, 0); self.assertIn("BENCH_DIRECT_OK", r.stderr)

    def test_unreachable_gateway_is_allowed(self):
        r = self.run_cli("http://127.0.0.1:9")
        self.assertEqual(r.returncode, 0); self.assertIn("unreachable", r.stderr)

    def test_require_raises_systemexit_3(self):
        s, url = gw(False)
        try:
            os.environ.pop("BENCH_DIRECT_OK", None)
            with self.assertRaises(SystemExit) as c:
                g.require(url)
            self.assertEqual(c.exception.code, 3)
        finally:
            s.shutdown(); s.server_close()


if __name__ == "__main__":
    unittest.main()
