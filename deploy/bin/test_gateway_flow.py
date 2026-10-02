#!/usr/bin/env python3
"""CF capacity-aware flow (keepalive-shim, 2026-10-02): work classes, priority queuing, ceilings, deadline-aware
admission, prefix affinity/hold, capacity facts and the mode event.

Same isolation as test_gateway_local_first.py: every on-disk path the shim can write is redirected into a private
temp dir before import. Run:  python -m unittest test_gateway_flow
"""
import asyncio
import atexit
import importlib.util
import json
import os
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, patch

_TMPDIR = tempfile.TemporaryDirectory(prefix="gw-flow-test-")
atexit.register(_TMPDIR.cleanup)
_TMP = _TMPDIR.name
_ISOLATED_ENV = {
    "SHIM_SPEND_FILE": os.path.join(_TMP, "gateway-spend.json"),
    "SHIM_SPEND_CLIENTS_FILE": os.path.join(_TMP, "gateway-spend-clients.json"),
    "SHIM_STATS_FILE": os.path.join(_TMP, "gateway-stats.json"),
    "SHIM_TELEMETRY_DIR": os.path.join(_TMP, "telemetry"),
    "SHIM_ENV_FILE": os.path.join(_TMP, "shim.env"),
    "SHIM_ALIASES_FILE": os.path.join(_TMP, "gateway-aliases.json"),
    "SHIM_FLIGHTREC_DIR": os.path.join(_TMP, "flightrec"),
    "SHIM_FLOW_EVENTS_FILE": os.path.join(_TMP, "incidents", "capacity-mode.jsonl"),
    "SHIM_EXACT_TOKENS": "0",
}
with patch.dict(os.environ, _ISOLATED_ENV):
    SPEC = importlib.util.spec_from_file_location("shim_flow_test", os.environ.get(
        "SHIM_TEST_CANDIDATE", str(Path(__file__).with_name("keepalive-shim.py"))))
    shim = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(shim)

_LIVE_DIR = "/home/kevin/.local/share/vllm-qwen27b"
for _name in ("SPEND_FILE", "STATS_FILE", "TELEMETRY_DIR", "SHIM_ENV_FILE", "ALIASES_FILE", "_FLOW_EVENTS_FILE"):
    assert not str(getattr(shim, _name)).startswith(_LIVE_DIR), _name


class Request:
    path = "/v1/chat/completions"
    remote = "127.0.0.1"
    method = "POST"

    def __init__(self, xclient="test-interactive", headers=None, content="review", **fields):
        if "transport" in fields:
            self.transport = fields.pop("transport")
        self.headers = {"X-Client": xclient, "User-Agent": "offline-test", **(headers or {})}
        self.body = json.dumps({"model": "qwen-local", "messages": [{"role": "user", "content": content}],
                                "max_tokens": 1000, **fields}).encode()

    async def read(self):
        return self.body


_DEFAULT_MAP = shim.FLOW_CLASS_MAP            # captured at import, before any test mutates it


def chain(*parts):
    """A fake prefix chain [(key, cumulative_chars)] and its total."""
    out, cum = [], 0
    for name, n in parts:
        cum += n
        out.append((name.encode().ljust(8, b"_"), cum))
    return out, cum


def pm_for(parts, est, computed):
    c, total = chain(*parts)
    return {"chain": c, "total": total, "est": est, "computed": computed, "credit": est - computed}


class Base(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, _ISOLATED_ENV))
        flow = dict(waiters=[], vtime={c: 0.0 for c in shim.FLOW_CLASSES}, vclock=0.0, last_prefix=None, run=0,
                    last_admit={c: 0.0 for c in shim.FLOW_CLASSES}, prefilling={}, version=0, head=(-1, 0.0, None), seq=0)
        self.stack.enter_context(patch.dict(shim._FLOW, flow, clear=True))
        for name in ("_FLOW_STATS", "_FLOW_DEMAND", "_FLOW_WAITS", "_FLOW_METER", "_FLOW_SERVICE", "_FLOW_PURE"):
            self.stack.enter_context(patch.object(shim, name, type(getattr(shim, name))(
                *([] if isinstance(getattr(shim, name), shim.collections.Counter) else [[]]),
                **({"maxlen": getattr(shim, name).maxlen} if hasattr(getattr(shim, name), "maxlen") else {}))))
        for name, value in dict(FLOW_MODE="enforce", FLOW_SHARES="kevin=1000,halo=60,runner=25,background=10",
                                FLOW_CEIL="kevin=0,halo=1.5,runner=1,background=0.5", FLOW_BACKLOG_S=40.0,
                                FLOW_DEADLINES="kevin=0,halo=0,runner=900,background=600",
                                FLOW_AFFINITY_MAX=6, FLOW_STARVE_S=300.0, PREFILL_TPS=500.0,
                                FLOW_CLASS_MAP=_DEFAULT_MAP, FLOW_MARKERS="local-lane-runner batch job=runner", FLOW_MODEL_MAP="", FLOW_PREFIX_HOLD_MAX_S=60.0, FLOW_URGENT_SLACK_S=30.0,
                                _inflight=0, _inflight_computed=0, _health={"ok": True},
                                effective_budget=lambda: 14, LIGHT_PREFILL_SECS=5.0,
                                PREFIX_HIT_MARGIN_TOKENS=0).items():
            self.stack.enter_context(patch.object(shim, name, value))

    def ticket(self, cls="background", prefix_parts=(("sys", 1000), ("a", 100)), est=20000, computed=20000,
               fits=lambda: True, deadline_s=None, rid=None):
        pm = pm_for(prefix_parts, est, computed)
        t = shim.FlowTicket(cls, pm, est, 1, fits, None if deadline_s is None else time.time() + deadline_s,
                            deadline_s is not None, "x", rid or id(object()), max(0.5, computed / 500.0))
        shim.flow_enqueue(t)
        return t


class Classes(Base):
    def test_class_of(self):
        f = lambda x, **kw: shim.flow_class_of(Request(x, **kw), b"{}", kw.get("bg", False), False)
        self.assertEqual(f("halo-hermes"), "halo")
        self.assertEqual(f("halo-incident-supervisor"), "halo")
        self.assertEqual(f("card-repair-engine"), "runner")
        self.assertEqual(f("work-verifier"), "runner")
        self.assertEqual(f("overseer-refine-overflow"), "background")
        self.assertEqual(f("vault-dreams"), "background")
        self.assertEqual(f("acceptance-review-workflow-bg"), "background")
        self.assertEqual(f("estate-control"), "kevin")
        self.assertEqual(f("some-new-harness"), "kevin")                      # unknown = interactive
        self.assertEqual(shim.flow_class_of(Request("some-new-harness"), b"{}", True, False), "background")
        self.assertEqual(shim.flow_class_of(Request("anything", headers={"X-Work-Class": "runner"}), b"{}", False, False), "runner")
        self.assertEqual(shim.flow_class_of(Request("anything", headers={"X-Work-Class": "bogus"}), b"{}", False, False), "kevin")
        self.assertEqual(shim.flow_class_of(Request("anything"), b"{}", True, True), "halo")   # Halo control turn
        # a model alias can carry the class for harnesses that cannot send a per-request header (checked before the client map)
        with patch.object(shim, "FLOW_MODEL_MAP", "halo-chat=kevin"):
            chat = Request("halo-hermes", model="halo-chat")
            self.assertEqual(shim.flow_class_of(chat, chat.body, False, False), "kevin")
            other = Request("halo-hermes", model="estate")
            self.assertEqual(shim.flow_class_of(other, other.body, False, False), "halo")
        # the card runner's pi sessions share Kevin's host and provider: only the opening line tells them apart
        runner = Request("ubuntuide01 pi / x", content="local-lane-runner batch job. The full task card follows")
        self.assertEqual(shim.flow_class_of(runner, runner.body, False, False), "runner")
        kevin = Request("ubuntuide01 pi / x", content="please fix my zsh config")
        self.assertEqual(shim.flow_class_of(kevin, kevin.body, False, False), "kevin")

    def test_config_round_trip_and_validation(self):
        cfg = shim.current_config()
        for k in ("flow_mode", "flow_shares", "flow_ceil", "flow_deadlines", "flow_backlog_s", "flow_class_map"):
            self.assertIn(k, cfg)
        with patch.object(shim, "_config_owner", lambda: False):
            ch = shim.apply_config({"flow_shares": "halo=90,background=5", "flow_mode": "shadow"})
            self.assertEqual(sorted(ch), ["flow_mode", "flow_shares"])
            self.assertEqual(shim.FLOW_SHARES, "halo=90,background=5")
            self.assertEqual(shim._flow_shares()["halo"], 90.0)
            self.assertEqual(shim._flow_shares()["runner"], 25.0)              # unspecified keeps the default
            before = shim.FLOW_SHARES
            for bad in ("nonsense=3", "halo=-1", "halo=0", "halo=abc"):
                self.assertEqual(shim.apply_config({"flow_shares": bad}), [], bad)
            self.assertEqual(shim.FLOW_SHARES, before)
            self.assertEqual(shim.apply_config({"flow_mode": "yolo"}), [])
            self.assertEqual(shim.apply_config({"flow_class_map": "halo-=halo,x=background"}), ["flow_class_map"])
            self.assertEqual(shim.apply_config({"flow_class_map": "halo-=nope"}), [])


class Ordering(Base):
    def admit(self, t):
        shim.flow_on_admit(t)
        shim.flow_dequeue(t)

    def test_strict_order_when_all_wait(self):
        ts = {c: self.ticket(c, prefix_parts=((c, 1000),)) for c in reversed(shim.FLOW_CLASSES)}
        order = []
        while shim._FLOW["waiters"]:
            h = shim.flow_pick(time.time() + len(order))
            order.append(h.cls)
            self.admit(h)
        # kevin's share is large enough that it always goes first; the rest follow their shares
        self.assertEqual(order[0], "kevin")
        self.assertEqual(sorted(order), sorted(shim.FLOW_CLASSES))

    def test_shares_split_service_in_proportion_when_backlogged(self):
        served = {c: 0 for c in ("halo", "runner", "background")}
        for c in served:
            for i in range(300):
                self.ticket(c, prefix_parts=((c + str(i), 1000),), est=1000, computed=1000)
        for n in range(150):
            h = shim.flow_pick(time.time() + n)
            served[h.cls] += 1
            self.admit(h)
        tot = sum(served.values())
        # 60:25:10 of 95 -> 63% / 26% / 11%
        self.assertAlmostEqual(served["halo"] / tot, 60 / 95, delta=0.06)
        self.assertAlmostEqual(served["runner"] / tot, 25 / 95, delta=0.06)
        self.assertAlmostEqual(served["background"] / tot, 10 / 95, delta=0.06)
        self.assertGreater(served["background"], 0)                              # never starved by fair queuing

    def test_a_returning_class_does_not_bank_credit(self):
        for _ in range(20):                                                       # halo alone runs its tag up
            self.ticket("halo", prefix_parts=(("h", 1000),))
            self.admit(shim.flow_pick(time.time()))
        self.ticket("halo", prefix_parts=(("h", 1000),))
        shim._FLOW["vtime"]["background"] = 0.0                                   # idle for ages: tag far behind
        self.ticket("background", prefix_parts=(("b", 1000),))
        self.assertAlmostEqual(shim._FLOW["vtime"]["background"], shim._FLOW["vtime"]["halo"])   # levelled, not 20 turns ahead

    def test_ineligible_head_does_not_block_the_next(self):
        blocked = self.ticket("kevin", fits=lambda: False)
        ok = self.ticket("background")
        self.assertIs(shim.flow_pick(time.time()), ok)
        self.assertEqual(blocked.held, "fits")

    def test_edf_when_slack_is_short(self):
        a = self.ticket("runner", prefix_parts=(("a", 1000),), deadline_s=500)
        b = self.ticket("runner", prefix_parts=(("b", 1000),), deadline_s=10)
        self.assertIs(shim.flow_pick(time.time()), b)

    def test_affinity_prefers_the_same_prefix_then_gives_way(self):
        shim._FLOW["last_prefix"] = chain(("shared", 1000))[0][0][0].hex()[:12]
        a = self.ticket("halo", prefix_parts=(("other", 1000),))
        b = self.ticket("halo", prefix_parts=(("shared", 1000), ("x", 10)))
        self.assertEqual(b.prefix, shim._FLOW["last_prefix"])
        self.assertIs(shim.flow_pick(time.time()), b)                            # newer, but warm
        with patch.object(shim, "FLOW_AFFINITY_MAX", 0):                         # affinity budget used up -> FIFO
            shim._FLOW["head"] = (-1, 0.0, None)
            self.assertIs(shim.flow_pick(time.time() + 1), a)

    def test_off_and_shadow_never_gate(self):
        t = self.ticket("background")
        other = self.ticket("kevin", prefix_parts=(("k", 10),))
        with patch.object(shim, "FLOW_MODE", "off"):
            self.assertTrue(shim.flow_turn(t))
        with patch.object(shim, "FLOW_MODE", "shadow"):
            self.assertTrue(shim.flow_turn(t))
            self.assertEqual(shim._FLOW_STATS["shadow_would_hold"], 1)
        self.assertFalse(shim.flow_turn(t))                                      # enforce: kevin is the head
        self.assertTrue(shim.flow_turn(other))


class Ceilings(Base):
    def test_background_waits_for_a_short_engine_queue_but_kevin_and_halo_do_not(self):
        kev = self.ticket("kevin", prefix_parts=(("k", 1000),))
        halo = self.ticket("halo", prefix_parts=(("h", 1000),))
        bg = self.ticket("background", prefix_parts=(("b", 1000),))
        run = self.ticket("runner", prefix_parts=(("r", 1000),))
        # 30 s of uncached prefill already in the engine (15,000 tokens at 500 tok/s): over bg (20 s) and below runner (40 s)
        with patch.object(shim, "_inflight", 3), patch.object(shim, "_inflight_computed", 15000):
            self.assertAlmostEqual(shim.flow_backlog_s(), 30.0)
            for t in (kev, halo, run):
                self.assertTrue(shim._flow_eligible(t, time.time(), 30.0)[0], t.cls)
            self.assertEqual(shim._flow_eligible(bg, time.time(), 30.0), (False, "ceiling"))
        # an idle engine admits anything, whatever the ceiling
        self.assertTrue(shim._flow_eligible(bg, time.time(), 0.0)[0])
        with patch.object(shim, "_inflight", 0):
            self.assertTrue(shim._flow_eligible(bg, time.time(), 99.0)[0])

    def test_starved_class_gets_through_once(self):
        bg = self.ticket("background", prefix_parts=(("b", 1000),))
        bg.t_enq = time.time() - 400
        with patch.object(shim, "_inflight", 3):
            self.assertTrue(shim._flow_eligible(bg, time.time(), 30.0)[0])


class PrefixHold(Base):
    def test_waits_for_a_prefill_that_will_warm_its_prefix_when_that_is_cheaper(self):
        shim._FLOW["prefilling"][111] = {"cum": {k for k, _ in chain(("sys", 30000), ("a", 100))[0]},
                                         "t0": time.time() - 5, "prefix": "x", "est_s": 20.0}
        t = self.ticket("halo", prefix_parts=(("sys", 30000), ("a", 100), ("b", 50)), est=60000, computed=60000)
        # shares ~99% of 60k tokens = ~59k tokens = ~118 s of recompute at 500 tok/s, vs ~15 s left on the other prefill
        self.assertIsNotNone(shim._flow_prefix_hold(t, time.time()))
        self.assertEqual(shim._flow_eligible(t, time.time(), 0.0), (False, "prefix-hold"))
        shim.flow_prefill_done(111)
        self.assertTrue(shim._flow_eligible(t, time.time(), 0.0)[0])

    def test_does_not_wait_when_the_shared_part_is_small_or_the_wait_is_long(self):
        shim._FLOW["prefilling"][111] = {"cum": {k for k, _ in chain(("sys", 100))[0]},
                                         "t0": time.time() - 1, "prefix": "x", "est_s": 200.0}
        t = self.ticket("halo", prefix_parts=(("sys", 100), ("b", 90000)), est=60000, computed=60000)
        self.assertIsNone(shim._flow_prefix_hold(t, time.time()))                # 0.1% shared
        shim._FLOW["prefilling"][111]["cum"] = {k for k, _ in chain(("sys", 100), ("b", 90000))[0]}
        self.assertIsNone(shim._flow_prefix_hold(t, time.time()))                # 200 s to wait > 120 s saved

    def test_a_hold_is_bounded(self):
        shim._FLOW["prefilling"][111] = {"cum": {k for k, _ in chain(("sys", 30000))[0]},
                                         "t0": time.time() - 61, "prefix": "x", "est_s": 500.0}
        t = self.ticket("halo", prefix_parts=(("sys", 30000), ("b", 50)), est=60000, computed=60000)
        self.assertIsNone(shim._flow_prefix_hold(t, time.time()))


class Deadlines(Base):
    def test_refusable_class_that_cannot_start_in_time_is_refused_with_retry_after(self):
        for _ in range(4):
            self.ticket("kevin", prefix_parts=(("k", 1000),), est=25000, computed=25000)   # ~50 s each ahead
        t = self.ticket("background", prefix_parts=(("b", 1000),), est=5000, computed=5000, deadline_s=60)
        with patch.object(shim, "_inflight", 3), patch.object(shim, "_inflight_computed", 15000):
            r = shim.flow_admission_check(t, remote_can_take=False)
        self.assertIsNotNone(r)
        self.assertGreater(r["expected_wait_s"], 60)
        self.assertGreaterEqual(r["retry_after"], 5)
        resp = shim._flow_refusal_response(r)
        self.assertEqual(resp.status, 429)
        self.assertEqual(resp.headers["X-Gateway-Refused"], "flow-deadline")
        self.assertEqual(int(resp.headers["Retry-After"]), r["retry_after"])

    def test_not_refused_when_it_can_start_or_remote_can_take_it_or_class_is_not_refusable(self):
        t = self.ticket("background", deadline_s=600)
        self.assertIsNone(shim.flow_admission_check(t, False))                   # idle engine -> start now
        self.assertEqual(t.expected_wait_s, 0.0)
        for _ in range(6):
            self.ticket("kevin", prefix_parts=(("k", 1000),), est=25000, computed=25000)
        late = self.ticket("background", prefix_parts=(("late", 1000),), deadline_s=20)
        with patch.object(shim, "_inflight", 3), patch.object(shim, "_inflight_computed", 15000):
            valve = shim.flow_admission_check(late, remote_can_take=True)         # local+remote: the spike valve takes it now
            self.assertTrue(valve and valve["overflow"])
            halo = self.ticket("halo", prefix_parts=(("h", 1000),), deadline_s=1)
            self.assertIsNone(shim.flow_admission_check(halo, False))            # Halo is told, never refused
            self.assertGreater(halo.expected_wait_s, 0)
            with patch.object(shim, "FLOW_MODE", "shadow"):
                self.assertIsNone(shim.flow_admission_check(late, False))
                self.assertEqual(shim._FLOW_STATS["shadow_would_refuse_background"], 1)
            with patch.object(shim, "FLOW_MODE", "off"):
                self.assertIsNone(shim.flow_admission_check(late, False))

    def test_declared_deadline_is_total_patience_so_service_time_counts(self):
        for _ in range(8):
            shim._FLOW_SERVICE.append((time.time(), "background", 30.0))           # this class typically takes 30 s end to end
        undeclared = self.ticket("background", prefix_parts=(("u", 100),), est=1000, computed=1000, deadline_s=40)
        self.assertIsNone(shim.flow_admission_check(undeclared, False))            # a 'start within 40 s' default: starts at once
        declared = self.ticket("background", prefix_parts=(("d", 100),), est=1000, computed=1000, deadline_s=20)
        declared.declared = True
        self.assertIsNotNone(shim.flow_admission_check(declared, False))           # 'I will wait 20 s for the answer' < 30 s service

    def test_caller_declared_deadline_overrides_the_class_default(self):
        req = Request("overseer-x", headers={"X-Gateway-Deadline-S": "42"})
        t = shim.flow_make_ticket(req, req.body, "background", pm_for((("a", 10),), 1000, 1000), 1000, 1, lambda: True)
        self.assertTrue(t.declared)
        self.assertAlmostEqual(t.deadline_at - time.time(), 42, delta=1)
        req = Request("overseer-x")
        t = shim.flow_make_ticket(req, req.body, "background", pm_for((("a", 10),), 1000, 1000), 1000, 1, lambda: True)
        self.assertFalse(t.declared)
        self.assertAlmostEqual(t.deadline_at - time.time(), 600, delta=1)
        t = shim.flow_make_ticket(req, req.body, "kevin", pm_for((("a", 10),), 1000, 1000), 1000, 1, lambda: True)
        self.assertIsNone(t.deadline_at)


class FailOpen(Base):
    def test_a_broken_flow_hook_serves_with_legacy_admission(self):
        t = self.ticket("background")
        with patch.object(shim, "flow_pick", side_effect=RuntimeError("boom")):
            self.assertTrue(shim.flow_turn(t))                                      # admit, as before CF
        self.assertEqual(shim._FLOW_STATS["failopen_flow_turn"], 1)
        with patch.object(shim, "flow_expected_wait", side_effect=KeyError("x")):
            self.assertIsNone(shim.flow_admission_check(t, False))                  # no refusal
        shim.flow_note_arrival("not-a-class-but-harmless", 1, 1)                    # bookkeeping never raises into a request


class Facts(Base):
    def test_mode_is_derived_from_the_router_flags(self):
        base = dict(REMOTE_ENABLED=True, LOCAL_ONLY=0, _remote_dead_until=0.0, effective_force_remote=lambda: False,
                    _spend_allows_overflow=lambda p, m: True)
        with ExitStack() as st:
            for k, v in base.items():
                st.enter_context(patch.object(shim, k, v))
            self.assertEqual(shim.flow_mode_now()["mode"], "local+remote")
            with patch.object(shim, "LOCAL_ONLY", 1):
                m = shim.flow_mode_now()
                self.assertEqual(m["mode"], "local-only")
                self.assertTrue(any("full-local" in w for w in m["why"]))
            with patch.object(shim, "_remote_dead_until", time.time() + 100):
                self.assertEqual(shim.flow_mode_now()["mode"], "local-only")
            with patch.object(shim, "REMOTE_ENABLED", False):
                self.assertEqual(shim.flow_mode_now()["mode"], "local-only")
            with patch.object(shim, "_spend_allows_overflow", lambda p, m: False):
                self.assertEqual(shim.flow_mode_now()["mode"], "local-only")
            with patch.object(shim, "_health", {"ok": False}):
                self.assertEqual(shim.flow_mode_now()["mode"], "remote-only")
                with patch.object(shim, "REMOTE_ENABLED", False):
                    self.assertEqual(shim.flow_mode_now()["mode"], "none")
            with patch.object(shim, "effective_force_remote", lambda: True):
                self.assertEqual(shim.flow_mode_now()["mode"], "remote-only")

    def test_mode_change_is_an_event_and_the_first_observation_is_not(self):
        path = Path(os.environ["SHIM_FLOW_EVENTS_FILE"]) if False else Path(shim._FLOW_EVENTS_FILE)
        if path.exists():
            path.unlink()
        modes = iter([{"mode": "local+remote", "why": [], "local_up": True, "remote_usable": True},
                      {"mode": "local+remote", "why": [], "local_up": True, "remote_usable": True},
                      {"mode": "remote-only", "why": ["local engine unhealthy"], "local_up": False, "remote_usable": True}])
        st = {"mode": None, "since": None, "basis": None, "events": shim.collections.deque(maxlen=200)}
        with patch.object(shim, "_FLOW_MODE_STATE", st), patch.object(shim, "flow_mode_now", lambda: next(modes)):
            shim.flow_note_mode(100.0)
            self.assertFalse(path.exists())
            shim.flow_note_mode(101.0)
            self.assertFalse(path.exists())
            shim.flow_note_mode(102.0)
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["from"], rows[0]["to"]), ("local+remote", "remote-only"))
        self.assertEqual(rows[0]["why"], ["local engine unhealthy"])

    def test_prefill_rate_is_measured_from_busy_samples(self):
        self.assertEqual(shim.flow_prefill_tps(), 500.0)                         # nothing measured: configured
        for i in range(8):
            shim._FLOW_METER.append((time.time(), 900.0 + i, 60.0, 4, 3, 0.4))
        for i in range(8):
            shim._FLOW_METER.append((time.time(), 0.0, 0.0, 0, 0, None))        # idle samples do not drag the rate
        self.assertAlmostEqual(shim.flow_prefill_tps(), 904.0, delta=4)
        self.assertEqual(shim.flow_decode_tps(), 60.0)

    def test_pure_prefill_speed_excludes_queue_and_needs_enough_prefill(self):
        self.assertIsNone(shim.flow_prefill_pure_tps())
        mk = lambda toks, secs: {"vllm:request_prefill_kv_computed_tokens_sum": [({}, toks)], "vllm:request_prefill_time_seconds_sum": [({}, secs)]}
        shim.flow_meter_update(mk(1000.0, 5.0), mk(0.0, 0.0), 10.0, 1, 0, None, None)
        self.assertIsNone(shim.flow_prefill_pure_tps())                           # 5 prefill-seconds: too little to say
        shim.flow_meter_update(mk(21000.0, 25.0), mk(1000.0, 5.0), 10.0, 1, 0, None, None)
        self.assertAlmostEqual(shim.flow_prefill_pure_tps(), 21000.0 / 25.0)      # 840 tok/s per request, whatever the queue was
        f = shim.flow_capacity_facts()
        self.assertEqual(f["throughput"]["prefill_pure_tok_s"], 840.0)
        self.assertIn("EXCLUDED", f["throughput"]["prefill_pure_basis"])

    def test_meter_update_reads_the_local_compute_counter(self):
        mk = lambda v: {"vllm:prompt_tokens_by_source_total": [({"source": "local_compute"}, v), ({"source": "local_cache_hit"}, 9e9)]}
        shim.flow_meter_update(mk(30_000.0), mk(0.0), 10.0, 4, 2, 0.3, 50.0)
        self.assertAlmostEqual(shim._FLOW_METER[-1][1], 3000.0)

    def test_capacity_facts_shape(self):
        for c, n in (("halo", 6), ("background", 3)):
            for _ in range(n):
                shim.flow_note_arrival(c, 40000, 10000)
        self.ticket("kevin", prefix_parts=(("k", 10),))
        f = shim.flow_capacity_facts(cls="halo", ptok=40000)
        for k in ("mode", "flow_mode", "throughput", "pressure", "demand", "queue", "affinity", "config", "cannot_measure", "estimate"):
            self.assertIn(k, f)
        self.assertEqual(f["queue"]["kevin"]["waiting"], 1)
        self.assertGreater(f["demand"]["halo"]["uncached_prefill_s_per_min_5m"], 0)
        self.assertEqual(f["throughput"]["prefill_uncached_tok_s"], 500.0)
        self.assertIn("configured", f["throughput"]["prefill_basis"])
        json.dumps(f)                                                             # serialisable


class FakeAdmin:
    def __init__(self, method, body=None):
        self.method, self._body, self.headers, self.remote = method, body, {"X-Client": "t"}, "127.0.0.1"

    async def json(self):
        if self._body is None:
            raise ValueError
        return self._body


class OfflineWindow(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, _ISOLATED_ENV))
        self.stack.enter_context(patch.dict(shim._OFFLINE, dict(until=0.0, lease=None, reason=None, by=None, t0=None,
                                                                ttl_s=None, refused=0)))
        st = {"mode": None, "since": None, "basis": None, "events": shim.collections.deque(maxlen=200)}
        self.stack.enter_context(patch.object(shim, "_FLOW_MODE_STATE", st))
        for name, value in dict(REMOTE_ENABLED=True, LOCAL_ONLY=0, _remote_dead_until=0.0, SHIM_ADMIN_TOKEN="",
                                effective_force_remote=lambda: False, _spend_allows_overflow=lambda p, m: True,
                                _health={"ok": True}).items():
            self.stack.enter_context(patch.object(shim, name, value))
        p = Path(shim._FLOW_EVENTS_FILE)
        if p.exists():
            p.unlink()

    def rows(self):
        p = Path(shim._FLOW_EVENTS_FILE)
        return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []

    async def test_open_status_extend_conflict_close(self):
        shim.flow_note_mode()                                                       # the sampler's baseline observation
        r = await shim.gateway_offline(FakeAdmin("POST", {"ttl_s": 120, "reason": "bench arm D1", "by": "EF2"}))
        d = json.loads(r.body)
        self.assertEqual(r.status, 200)
        self.assertTrue(d["offline"] and d["lease"])
        self.assertEqual(d["mode"], "remote-only")
        self.assertTrue(shim._local_offline())
        r2 = await shim.gateway_offline(FakeAdmin("POST", {"ttl_s": 120, "reason": "other", "by": "UP"}))
        self.assertEqual(r2.status, 409)                                            # one window at a time
        r3 = await shim.gateway_offline(FakeAdmin("POST", {"ttl_s": 300, "lease": d["lease"]}))
        self.assertEqual(r3.status, 200)
        self.assertGreater(json.loads(r3.body)["remaining_s"], 200)
        bad = await shim.gateway_offline(FakeAdmin("DELETE", {"lease": "nope"}))
        self.assertEqual(bad.status, 409)
        ok = await shim.gateway_offline(FakeAdmin("DELETE", {"lease": d["lease"]}))
        self.assertFalse(json.loads(ok.body)["offline"])
        self.assertFalse(shim._local_offline())
        evs = [r_["event"] for r_ in self.rows() if "event" in r_]
        self.assertEqual(evs, ["offline-open", "offline-close"])
        modes = [(r_["from"], r_["to"]) for r_ in self.rows() if "to" in r_]
        self.assertEqual(modes, [("local+remote", "remote-only"), ("remote-only", "local+remote")])
        self.assertEqual(shim.flow_mode_now()["mode"], "local+remote")              # and back to local-first by itself

    async def test_bad_ttl_and_auth(self):
        self.assertEqual((await shim.gateway_offline(FakeAdmin("POST", {"ttl_s": 5}))).status, 400)
        self.assertEqual((await shim.gateway_offline(FakeAdmin("POST", {"ttl_s": 99999}))).status, 400)
        with patch.object(shim, "SHIM_ADMIN_TOKEN", "secret"):
            self.assertEqual((await shim.gateway_offline(FakeAdmin("POST", {"ttl_s": 60}))).status, 401)
            self.assertEqual((await shim.gateway_offline(FakeAdmin("GET"))).status, 200)    # status is open

    async def test_lease_expiry_closes_the_window_and_writes_the_row(self):
        await shim.gateway_offline(FakeAdmin("POST", {"ttl_s": 60, "reason": "r", "by": "x"}))
        shim._OFFLINE["until"] = time.time() - 1
        shim._offline_reap()
        self.assertFalse(shim._local_offline())
        self.assertEqual([r["event"] for r in self.rows() if "event" in r], ["offline-open", "offline-close"])
        self.assertTrue(any(r.get("how") == "expired" for r in self.rows()))

    def test_mode_without_remote_is_none_during_a_window(self):
        shim._OFFLINE.update(until=time.time() + 100, reason="r", by="x")
        with patch.object(shim, "REMOTE_ENABLED", False):
            m = shim.flow_mode_now()
        self.assertEqual(m["mode"], "none")
        self.assertTrue(m["planned_offline"])
        self.assertTrue(any("planned local-offline" in w for w in m["why"]))

    def test_remote_use_separates_forced_from_gateway_chosen_and_flags_headroom(self):
        with patch.object(shim, "_FLOW_ROUTES", shim.collections.deque(maxlen=100)):
            now = time.time()
            for dec, reason, cls, head in (("local", "-", "halo", True), ("remote", "local-offline", "halo", False),
                                           ("remote", "big-prompt", "kevin", True), ("remote", "bg-yield", "background", False),
                                           ("local", "-", "runner", True)):
                shim._FLOW_ROUTES.append((now, dec, reason, cls, head))
            u = shim.flow_remote_use(now)["windows"]["15m"]
        self.assertEqual((u["requests"], u["remote"], u["gateway_chosen_remote"]), (5, 3, 2))
        self.assertEqual(u["remote_while_local_had_headroom"], 1)
        self.assertAlmostEqual(u["remote_share"], 0.6)



class Transport:
    def __init__(self):
        self.closing = False

    def is_closing(self):
        return self.closing


class ClientGone(unittest.IsolatedAsyncioTestCase):
    """aiohttp does not cancel a handler when the caller hangs up: a queued request of a caller that already timed out
    used to be admitted and prefilled for nobody."""

    async def test_await_unless_gone_returns_the_result_or_cancels_the_upstream_call(self):
        tr = Transport()
        req = Request(transport=tr)

        async def quick():
            return 7
        self.assertEqual(await shim._await_unless_gone(req, quick(), poll=0.01), 7)
        cancelled = []

        async def slow():
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.append(1)
                raise
        asyncio.get_running_loop().call_later(0.05, lambda: setattr(tr, "closing", True))
        with self.assertRaises(shim._ClientGone):
            await shim._await_unless_gone(req, slow(), poll=0.01)
        self.assertEqual(cancelled, [1])                                           # the upstream call was aborted
        with self.assertRaises(asyncio.TimeoutError):
            await shim._await_unless_gone(Request(transport=Transport()), slow(), timeout=0.05, poll=0.01)

    def test_a_double_without_a_connection_is_never_gone(self):
        self.assertFalse(shim._client_gone(Request()))
        tr = Transport()
        self.assertFalse(shim._client_gone(Request(transport=tr)))
        tr.closing = True
        self.assertTrue(shim._client_gone(Request(transport=tr)))


class Routing(unittest.IsolatedAsyncioTestCase):
    """End to end through _route_completions: with one engine place, waiters are admitted by class, not by luck."""

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, _ISOLATED_ENV))
        flow = dict(waiters=[], vtime={c: 0.0 for c in shim.FLOW_CLASSES}, vclock=0.0, last_prefix=None, run=0,
                    last_admit={c: 0.0 for c in shim.FLOW_CLASSES}, prefilling={}, version=0, head=(-1, 0.0, None), seq=0)
        self.stack.enter_context(patch.dict(shim._FLOW, flow, clear=True))
        self.admitted, self.gates, self.events = [], {}, []

        async def relay(request, base, path, body, key, streaming, *a, **k):
            name = request.headers["X-Client"]
            self.admitted.append(name)
            if name in self.gates:
                await self.gates[name].wait()
            return "ok", name

        for name, value in dict(
                _relay=relay, record_event=lambda d, r, *a, **k: self.events.append((d, r)),
                _est_tokens=lambda body: 3000, estimate_units=lambda *a, **k: 1,
                predict_computed_tokens=lambda c, b, p: 3000, _prefix_cache_observe=lambda *a: None,
                remote_ok=lambda: False, _spend_allows_overflow=lambda p, m: False,
                _automatic_remote_budget_allows=lambda p, m: False,
                local_healthy=AsyncMock(return_value=True), is_peak=lambda: False,
                _active_set=lambda *a, **k: None, _note_payload_outcome=lambda *a, **k: None,
                _write_flightrec=lambda *a, **k: None, perf_breaker_active=lambda: False,
                predicted_occupancy_seconds=lambda *a, **k: None, effective_budget=lambda: 1,
                _health={"ok": True}, _inflight=0, _inflight_tokens=0, _inflight_reserved_tokens=0, _inflight_computed=0,
                _waiting=0, _waiting_by_class={"interactive": 0, "background": 0},
                TOKEN_BUDGET=500_000, LOCAL_MAX_OUT=16384, DEFAULT_MAX_OUT=8192, MAX_LOCAL_TOKENS=524_288,
                LOCAL_CONTEXT_LIMIT=524_288, FORCE_REMOTE=0, BIG_OUTPUT=16384, BIG_PROMPT=24_000, MONSTER_INFLIGHT=120_000,
                FOREIGN_LOAD_GUARD=0, TINY_TOKENS=0, LOCAL_WAIT=0, BG_WAIT=0, BG_LOCAL_ONLY=0, BG_BIG_LOCAL_WHEN_IDLE=0,
                LOG_REQUESTS=0, CRASH_ADAPTIVE=0, EMPTY_RETRY=0, LOCAL_ONLY=0, INTERACTIVE_NEVER_OVERFLOW=1,
                FG_RESERVED=0, TINY_EXTRA_LANES=0, SLOT_POLL=0.005, FLOW_MODE="enforce",
                FLOW_DEADLINES="kevin=0,halo=0,runner=900,background=600", PREFILL_TPS=500.0,
                FLOW_CLASS_MAP=_DEFAULT_MAP, FLOW_SHARES="kevin=1000,halo=60,runner=25,background=10",
                FLOW_CEIL="kevin=0,halo=1.5,runner=1,background=0.5", FLOW_BACKLOG_S=40.0).items():
            self.stack.enter_context(patch.object(shim, name, value))

    async def go(self, xclient, **kw):
        return await shim._route_completions(Request(xclient, **kw))

    async def test_waiters_are_admitted_in_class_order_not_arrival_order(self):
        self.gates["holder"] = asyncio.Event()
        holder = asyncio.create_task(self.go("holder"))
        await asyncio.sleep(0.05)                                                  # the only engine place is taken
        order = ["overseer-refine-overflow", "card-repair-engine", "halo-hermes", "ubuntuide01 pi / x"]
        tasks = []
        for i, name in enumerate(order):                                           # lowest priority arrives first
            tasks.append(asyncio.create_task(self.go(name, content="q%d" % i)))
            await asyncio.sleep(0.02)
        self.gates["holder"].set()
        await asyncio.wait_for(asyncio.gather(holder, *tasks), 10)
        self.assertEqual(self.admitted, ["holder", "ubuntuide01 pi / x", "halo-hermes", "card-repair-engine",
                                         "overseer-refine-overflow"])

    async def test_flow_off_is_the_previous_behaviour(self):
        with patch.object(shim, "FLOW_MODE", "off"):
            self.assertEqual(await self.go("overseer-refine-overflow"), "overseer-refine-overflow")
            self.assertEqual(shim._FLOW["waiters"], [])

    async def test_background_that_cannot_start_in_time_is_refused_before_any_prefill(self):
        self.gates["holder"] = asyncio.Event()
        holder = asyncio.create_task(self.go("holder"))
        await asyncio.sleep(0.05)
        waiters = [asyncio.create_task(self.go("halo-hermes", content="h%d" % i)) for i in range(3)]
        await asyncio.sleep(0.05)
        with patch.object(shim, "_inflight_computed", 60000):                      # ~2 min of prefill already in the engine
            resp = await self.go("overseer-refine-overflow", headers={"X-Gateway-Deadline-S": "20"})
        self.assertEqual(resp.status, 429)
        self.assertEqual(self.events[-1], ("rejected-bg", "flow-deadline"))
        self.assertNotIn("overseer-refine-overflow", self.admitted)
        self.gates["holder"].set()
        await asyncio.wait_for(asyncio.gather(holder, *waiters), 10)
        self.assertEqual(shim._FLOW["waiters"], [])                                # nothing left queued

    async def test_planned_offline_window_sends_new_work_to_the_valve_and_not_to_the_engine(self):
        remote_calls = []

        async def fwd(request, path, body, streaming, endpoint=None, model=None):
            remote_calls.append(request.headers["X-Client"])
            return "remote"

        with patch.dict(shim._OFFLINE, dict(until=time.time() + 100, lease="L", reason="bench", by="EF2", t0=time.time(),
                                            ttl_s=100, refused=0)), \
                patch.object(shim, "remote_ok", lambda: True), patch.object(shim, "_spend_allows_overflow", lambda p, m: True), \
                patch.object(shim, "_automatic_remote_budget_allows", lambda p, m: True), \
                patch.object(shim, "_forward_remote", fwd), patch.object(shim, "REMOTE_ENABLED", True):
            for who in ("halo-hermes", "card-repair-engine", "overseer-refine-overflow", "ubuntuide01 pi / x"):
                self.assertEqual(await self.go(who), "remote", who)
            self.assertEqual(self.admitted, [])                                      # nothing touched the engine
            self.assertEqual(self.events[-1], ("remote", "local-offline"))
            # a caller that must stay local (pinned / estate-local) is told to retry, not spent on
            r = await self.go("local-pin-coder")
            self.assertEqual(r.status, 503)
            self.assertIn("Retry-After", r.headers)
            self.assertEqual((await shim._route_completions(Request("x", model="estate-local"))).status, 503)

    async def test_planned_offline_window_without_a_remote_holds_callers_with_retry_after(self):
        with patch.dict(shim._OFFLINE, dict(until=time.time() + 100, lease="L", reason="bench", by="EF2", t0=time.time(),
                                            ttl_s=100, refused=0)):
            r = await self.go("halo-hermes")
        self.assertEqual(r.status, 503)
        self.assertEqual(r.headers["X-Gateway-Offline"], "active")
        self.assertEqual(self.admitted, [])

    async def test_a_waiter_is_sent_to_the_valve_when_a_window_opens_under_it(self):
        async def fwd(request, path, body, streaming, endpoint=None, model=None):
            return "remote"

        self.gates["holder"] = asyncio.Event()
        with patch.object(shim, "remote_ok", lambda: True), patch.object(shim, "_spend_allows_overflow", lambda p, m: True), \
                patch.object(shim, "_automatic_remote_budget_allows", lambda p, m: True), patch.object(shim, "_forward_remote", fwd), \
                patch.object(shim, "REMOTE_ENABLED", True), patch.object(shim, "LOCAL_WAIT", 30), patch.object(shim, "BG_WAIT", 30):
            holder = asyncio.create_task(self.go("holder"))
            await asyncio.sleep(0.05)
            waiter = asyncio.create_task(self.go("halo-hermes", content="w"))
            await asyncio.sleep(0.05)
            self.assertFalse(waiter.done())
            with patch.dict(shim._OFFLINE, dict(until=time.time() + 100, lease="L", reason="bench", by="EF2", t0=time.time(),
                                                ttl_s=100, refused=0)):
                self.assertEqual(await asyncio.wait_for(waiter, 5), "remote")
            self.assertEqual(self.events[-1], ("remote", "local-offline"))
            self.gates["holder"].set()
            await holder

    async def test_valve_takes_robot_work_that_cannot_start_in_time_when_remote_can(self):
        async def fwd(request, path, body, streaming, endpoint=None, model=None):
            return "remote"

        self.gates["holder"] = asyncio.Event()
        holder = asyncio.create_task(self.go("holder"))
        await asyncio.sleep(0.05)
        waiters = [asyncio.create_task(self.go("halo-hermes", content="h%d" % i)) for i in range(3)]
        await asyncio.sleep(0.05)
        with patch.object(shim, "remote_ok", lambda: True), patch.object(shim, "_spend_allows_overflow", lambda p, m: True), \
                patch.object(shim, "_automatic_remote_budget_allows", lambda p, m: True), patch.object(shim, "_forward_remote", fwd), \
                patch.object(shim, "REMOTE_ENABLED", True), patch.object(shim, "_inflight_computed", 60000):
            r = await self.go("overseer-refine-overflow", headers={"X-Gateway-Deadline-S": "20"})
        self.assertEqual(r, "remote")
        self.assertEqual(self.events[-1], ("remote", "flow-deadline"))
        self.gates["holder"].set()
        await asyncio.wait_for(asyncio.gather(holder, *waiters), 10)

    async def test_a_caller_who_hung_up_while_queued_is_dropped_before_any_prefill(self):
        self.gates["holder"] = asyncio.Event()
        holder = asyncio.create_task(self.go("holder"))
        await asyncio.sleep(0.05)
        tr = Transport()
        waiter = asyncio.create_task(shim._route_completions(Request("halo-hermes", content="w", transport=tr)))
        await asyncio.sleep(0.05)
        self.assertEqual(len(shim._FLOW["waiters"]), 1)
        tr.closing = True                                                          # the caller timed out and left
        resp = await asyncio.wait_for(waiter, 5)
        self.assertEqual(resp.status, 499)
        self.assertEqual(shim._FLOW["waiters"], [])
        self.assertNotIn("halo-hermes", self.admitted)                             # nothing was sent to the engine
        self.assertEqual(self.events[-1], ("gone", "queued"))
        self.gates["holder"].set()
        await holder

    async def test_queued_work_goes_to_the_valve_when_local_dies_under_it(self):
        async def fwd(request, path, body, streaming, endpoint=None, model=None):
            return "remote"

        self.gates["holder"] = asyncio.Event()
        with patch.object(shim, "remote_ok", lambda: True), patch.object(shim, "_spend_allows_overflow", lambda p, m: True), \
                patch.object(shim, "_automatic_remote_budget_allows", lambda p, m: True), patch.object(shim, "_forward_remote", fwd), \
                patch.object(shim, "REMOTE_ENABLED", True), patch.object(shim, "LOCAL_WAIT", 30), patch.object(shim, "BG_WAIT", 30):
            holder = asyncio.create_task(self.go("holder"))
            await asyncio.sleep(0.05)
            waiter = asyncio.create_task(self.go("halo-hermes", content="w"))
            await asyncio.sleep(0.05)
            self.assertFalse(waiter.done())
            with patch.object(shim, "_health", {"ok": False}):                     # the engine died; the waiter was queued
                self.assertEqual(await asyncio.wait_for(waiter, 5), "remote")
            self.assertEqual(self.events[-1], ("remote", "local-down"))
            self.gates["holder"].set()
            await holder

    async def test_robot_work_waits_locally_until_its_bound_then_the_valve_takes_it(self):
        async def fwd(request, path, body, streaming, endpoint=None, model=None):
            return "remote"

        self.gates["holder"] = asyncio.Event()
        with patch.object(shim, "remote_ok", lambda: True), patch.object(shim, "_spend_allows_overflow", lambda p, m: True), \
                patch.object(shim, "_automatic_remote_budget_allows", lambda p, m: True), patch.object(shim, "_forward_remote", fwd), \
                patch.object(shim, "REMOTE_ENABLED", True), patch.object(shim, "BG_WAIT_LOCAL", 0.4), patch.object(shim, "BG_WAIT", 0.0), \
                patch.object(shim, "LOCAL_WAIT", 0):
            holder = asyncio.create_task(self.go("holder"))
            await asyncio.sleep(0.05)
            t0 = time.time()
            r = await asyncio.wait_for(self.go("overseer-refine-overflow", headers={"X-Gateway-Deadline-S": "600"}), 5)
            self.assertEqual(r, "remote")
            self.assertGreaterEqual(time.time() - t0, 0.35)                        # it waited for local first (not BG_WAIT=0)
            self.assertLess(time.time() - t0, 2.0)
            self.gates["holder"].set()
            await holder

if __name__ == "__main__":
    unittest.main()