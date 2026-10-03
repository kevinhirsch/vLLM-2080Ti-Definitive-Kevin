"""FX2: bench_lib warns (never blocks) when a bench hits the engine directly with no gateway offline window open."""
import http.server, json, os, sys, threading, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_lib


def gateway(offline):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            b = json.dumps({"offline": offline}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    s = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_port}"


class Warn(unittest.TestCase):
    def setUp(self):
        bench_lib._WINDOW_WARNED = False
        os.environ.pop("BENCH_DIRECT_OK", None)

    def test_warns_once_when_engine_direct_and_no_window(self):
        s, gw = gateway(False)
        try:
            self.assertTrue(bench_lib.warn_if_no_bench_window("http://127.0.0.1:8001/v1", gw))
            self.assertFalse(bench_lib.warn_if_no_bench_window("http://127.0.0.1:8001/v1", gw))   # once
        finally:
            s.shutdown(); s.server_close()

    def test_quiet_inside_a_window(self):
        s, gw = gateway(True)
        try:
            self.assertFalse(bench_lib.warn_if_no_bench_window("http://127.0.0.1:8001/v1", gw))
        finally:
            s.shutdown(); s.server_close()

    def test_quiet_for_the_gateway_url_opt_out_and_unreachable_gateway(self):
        self.assertFalse(bench_lib.warn_if_no_bench_window("http://127.0.0.1:8000/v1", "http://127.0.0.1:9"))
        bench_lib._WINDOW_WARNED = False
        os.environ["BENCH_DIRECT_OK"] = "1"
        self.assertFalse(bench_lib.warn_if_no_bench_window("http://127.0.0.1:8001/v1", "http://127.0.0.1:9"))
        del os.environ["BENCH_DIRECT_OK"]; bench_lib._WINDOW_WARNED = False
        self.assertFalse(bench_lib.warn_if_no_bench_window("http://127.0.0.1:8001/v1", "http://127.0.0.1:9"))


if __name__ == "__main__":
    unittest.main()
