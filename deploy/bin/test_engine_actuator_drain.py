"""FX2 (2026-10-03): the planned-restart drain must also wait on the ENGINE's own num_requests_running/waiting
(127.0.0.1:8001/metrics), because the gateway fence only counts gateway-accepted requests and direct :8001 callers
would otherwise be cut mid-request (the 23:18 restart)."""
import http.server, importlib.util, json, os, threading, time, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("engine_actuator", os.path.join(HERE, "engine-actuator.py"))
ea = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ea)


class FakeEngine:
    """Serves /metrics from a mutable state so a test can make the engine busy, then idle."""
    def __init__(self):
        self.running, self.waiting, self.broken = 0, 0, False
        self.gen, self.prompt = None, None
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if outer.broken:
                    self.send_response(500); self.end_headers(); return
                body = (f'# HELP vllm:num_requests_running x\nvllm:num_requests_running{{engine="0",model_name="m"}} {float(outer.running)}\n'
                        f'vllm:num_requests_waiting{{engine="0",model_name="m"}} {float(outer.waiting)}\n'
                        f'vllm:num_requests_waiting_by_reason{{engine="0",model_name="m",reason="deferred"}} 9.0\n'
                        + (f'vllm:generation_tokens_total{{engine="0",model_name="m"}} {float(outer.gen)}\n' if outer.gen is not None else "")
                        + (f'vllm:prompt_tokens_total{{engine="0",model_name="m"}} {float(outer.prompt)}\n' if outer.prompt is not None else "")).encode()
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




class Clock:
    """Fake time: sleep() advances the clock, so extension tests run instantly."""
    def __init__(self): self.t = 0.0
    def now(self): return self.t
    def sleep(self, d): self.t += d


class Extend(unittest.TestCase):
    """L63: past the base deadline keep waiting while tokens advance (to the hard cap); stop early on a stall."""
    def run_wait(self, active_fn, progress_fn, deadline=120, cap=600, **kw):
        c = Clock()
        r = ea.wait_drained(lambda: active_fn(c.t), deadline, hard_cap_s=cap, get_progress=lambda: progress_fn(c.t),
                            sleep=c.sleep, clock=c.now, **kw)
        return r, c

    def test_finishes_inside_the_base_deadline_without_extending(self):
        r, c = self.run_wait(lambda t: 0 if t >= 30 else 2, lambda t: int(t))
        self.assertEqual(r["end_reason"], "idle"); self.assertEqual(r["extended_s"], 0); self.assertTrue(r["idle"])

    def test_keeps_waiting_past_the_deadline_while_tokens_advance_then_finishes(self):
        # the 23:18 case: one long stream still generating at 120 s, done at 300 s
        r, c = self.run_wait(lambda t: 0 if t >= 300 else 1, lambda t: int(t * 91))
        self.assertEqual(r["end_reason"], "idle"); self.assertTrue(r["idle"])
        self.assertGreaterEqual(r["waited_s"], 300); self.assertGreater(r["extended_s"], 150)

    def test_hard_cap_ends_the_wait(self):
        r, c = self.run_wait(lambda t: 1, lambda t: int(t * 91))
        self.assertEqual(r["end_reason"], "cap"); self.assertFalse(r["idle"])
        self.assertLessEqual(r["waited_s"], 610); self.assertGreaterEqual(r["waited_s"], 600)

    def test_a_stall_after_the_deadline_stops_early(self):
        # tokens stop advancing at t=200: wedge, not work -> stop after EXT_STALL_POLLS polls, long before the cap
        r, c = self.run_wait(lambda t: 1, lambda t: int(min(t, 200) * 91))
        self.assertEqual(r["end_reason"], "stalled"); self.assertFalse(r["idle"])
        self.assertLess(r["waited_s"], 260)

    def test_a_stall_from_the_start_of_the_extension_stops_within_a_few_polls(self):
        r, c = self.run_wait(lambda t: 1, lambda t: 12345)
        self.assertEqual(r["end_reason"], "stalled"); self.assertLessEqual(r["extended_s"], 30)

    def test_no_extension_without_a_cap_or_progress_source(self):
        c = Clock()
        r = ea.wait_drained(lambda: 1, 120, hard_cap_s=None, get_progress=lambda: int(c.t), sleep=c.sleep, clock=c.now)
        self.assertEqual(r["end_reason"], "deadline"); self.assertLessEqual(r["waited_s"], 122)
        c = Clock()
        r = ea.wait_drained(lambda: 1, 120, hard_cap_s=600, get_progress=None, sleep=c.sleep, clock=c.now)
        self.assertEqual(r["end_reason"], "deadline")

    def test_cap_not_above_the_deadline_means_no_extension(self):
        r, c = self.run_wait(lambda t: 1, lambda t: int(t), deadline=120, cap=120)
        self.assertEqual(r["end_reason"], "deadline")

    def test_unreadable_after_the_deadline_stops(self):
        r, c = self.run_wait(lambda t: None if t >= 130 else 1, lambda t: int(t))
        self.assertEqual(r["end_reason"], "unreadable")

    def test_engine_progress_sums_generation_and_prompt_counters(self):
        e = FakeEngine()
        try:
            self.assertIsNone(ea.engine_progress(e.url))      # no token counters served
            e.gen = 1000
            self.assertEqual(ea.engine_progress(e.url), 1000)
            e.prompt = 250
            self.assertEqual(ea.engine_progress(e.url), 1250)
            e.broken = True
            self.assertIsNone(ea.engine_progress(e.url))
        finally:
            e.close()


class OfflineExtension(unittest.TestCase):
    """offline_and_wait against a stand-in gateway whose local_active falls only at t_done, with a stand-in engine whose
    token counter advances (or not)."""
    def serve(self, state):
        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def _send(self, o):
                b = json.dumps(o).encode(); self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
            def do_GET(self):
                if self.path.startswith("/gateway/offline"):
                    self._send({"offline": state["open"], "local_active": state["active"]()})
                else:
                    self._send({})
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0)); state["open"] = True
                self._send({"lease": "L", "local_active": state["active"]()})
            do_DELETE = do_POST
        s = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=s.serve_forever, daemon=True).start()
        return s

    def run_offline(self, active_fn, progress_fn, deadline=120, cap=600):
        c = Clock()
        state = {"open": False, "active": lambda: active_fn(c.t)}
        srv = self.serve(state)
        old = (ea.GATEWAY, ea.engine_progress, ea.time.sleep, ea.time.time)
        ea.GATEWAY = f"http://127.0.0.1:{srv.server_port}"
        ea.engine_progress = lambda *a, **k: progress_fn(c.t)
        ea.time.sleep, ea.time.time = c.sleep, (lambda: c.t)
        try:
            return ea.offline_and_wait(deadline, "t", "tok", "FX2", hard_cap_s=cap)
        finally:
            ea.GATEWAY, ea.engine_progress, ea.time.sleep, ea.time.time = old
            srv.shutdown(); srv.server_close()

    def test_extends_past_the_drain_budget_and_records_why(self):
        facts, lease = self.run_offline(lambda t: 0 if t >= 250 else 1, lambda t: int(t * 91))
        self.assertEqual(facts["strategy"], "offline-window"); self.assertEqual(facts["end_reason"], "idle")
        self.assertEqual(facts["active_at_end"], 0); self.assertGreater(facts["extended_s"], 100); self.assertEqual(facts["hard_cap_s"], 600)

    def test_stalled_generation_is_cut_early_with_the_reason(self):
        facts, lease = self.run_offline(lambda t: 1, lambda t: 777)
        self.assertEqual(facts["end_reason"], "stalled"); self.assertEqual(facts["active_at_end"], 1)
        self.assertLess(facts["waited_s"], 160)

    def test_without_a_cap_it_is_the_old_hard_deadline(self):
        facts, lease = self.run_offline(lambda t: 1, lambda t: int(t * 91), cap=None)
        self.assertEqual(facts["end_reason"], "deadline"); self.assertLessEqual(facts["waited_s"], 122); self.assertEqual(facts["extended_s"], 0)


if __name__ == "__main__":
    unittest.main()
