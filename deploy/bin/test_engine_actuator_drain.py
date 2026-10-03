"""FX2 (2026-10-03): the planned-restart drain must also wait on the ENGINE's own num_requests_running/waiting
(127.0.0.1:8001/metrics), because the gateway fence only counts gateway-accepted requests and direct :8001 callers
would otherwise be cut mid-request (the 23:18 restart)."""
import http.server, importlib.util, os, threading, time, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("engine_actuator", os.path.join(HERE, "engine-actuator.py"))
ea = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ea)


class FakeEngine:
    """Serves /metrics from a mutable state so a test can make the engine busy, then idle."""
    def __init__(self):
        self.running, self.waiting, self.broken = 0, 0, False
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if outer.broken:
                    self.send_response(500); self.end_headers(); return
                body = (f'# HELP vllm:num_requests_running x\nvllm:num_requests_running{{engine="0",model_name="m"}} {float(outer.running)}\n'
                        f'vllm:num_requests_waiting{{engine="0",model_name="m"}} {float(outer.waiting)}\n'
                        f'vllm:num_requests_waiting_by_reason{{engine="0",model_name="m",reason="deferred"}} 9.0\n').encode()
                self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        self.srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown(); self.srv.server_close()


class EngineDrain(unittest.TestCase):
    def setUp(self):
        self.e = FakeEngine()

    def tearDown(self):
        self.e.close()

    def test_parses_running_and_waiting_not_by_reason(self):
        self.e.running, self.e.waiting = 2, 3
        self.assertEqual(ea.engine_inflight(self.e.url), {"running": 2, "waiting": 3})

    def test_unreadable_metrics_is_none_and_does_not_block(self):
        self.e.broken = True
        t = time.time()
        f = ea.wait_engine_idle(30, self.e.url, poll_s=0.05)
        self.assertLess(time.time() - t, 3)
        self.assertIsNone(f["idle"])
        self.assertIn("error", f)

    def test_idle_engine_returns_quickly(self):
        f = ea.wait_engine_idle(30, self.e.url, poll_s=0.05)
        self.assertTrue(f["idle"])
        self.assertEqual((f["running_at_start"], f["running_at_end"]), (0, 0))

    def test_waits_for_direct_caller_to_finish(self):
        self.e.running = 1
        threading.Timer(0.4, lambda: setattr(self.e, "running", 0)).start()
        t = time.time()
        f = ea.wait_engine_idle(30, self.e.url, poll_s=0.05)
        self.assertTrue(f["idle"])
        self.assertGreaterEqual(time.time() - t, 0.35)          # it did wait for the in-flight request
        self.assertEqual(f["running_at_start"], 1)
        self.assertEqual(f["running_at_end"], 0)

    def test_waiting_requests_count_too(self):
        self.e.waiting = 1
        f = ea.wait_engine_idle(0.3, self.e.url, poll_s=0.05)
        self.assertFalse(f["idle"])
        self.assertEqual(f["waiting_at_end"], 1)

    def test_bounded_by_budget_when_never_idle(self):
        self.e.running = 4
        t = time.time()
        f = ea.wait_engine_idle(0.5, self.e.url, poll_s=0.1)
        self.assertFalse(f["idle"])
        self.assertLess(time.time() - t, 2.0)
        self.assertEqual(f["running_at_end"], 4)

    def test_zero_budget_still_takes_one_sample(self):
        self.e.running = 1
        f = ea.wait_engine_idle(0, self.e.url, poll_s=0.05)
        self.assertFalse(f["idle"])
        self.assertEqual(f["running_at_start"], 1)

    def test_single_zero_between_back_to_back_requests_is_not_idle(self):
        # running flickers 1 -> 0 -> 1: one zero sample must not release the stop
        seq = iter([{"running": 1, "waiting": 0}, {"running": 0, "waiting": 0}, {"running": 1, "waiting": 0}] + [{"running": 0, "waiting": 0}] * 5)
        orig = ea.engine_inflight
        ea.engine_inflight = lambda engine=None, timeout=3: next(seq)
        try:
            f = ea.wait_engine_idle(10, poll_s=0.01)
        finally:
            ea.engine_inflight = orig
        self.assertTrue(f["idle"])
        self.assertEqual(f["waited_s"], 0)
        self.assertEqual(f["running_at_start"], 1)


if __name__ == "__main__":
    unittest.main()
