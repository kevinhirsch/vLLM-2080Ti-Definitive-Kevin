#!/usr/bin/env python3
"""Lane K4 (2026-10-03): micro fast lane v2 -- a learned tiny lane for short-output call types, and the gateway-answered
liveness probe.  Measured basis: 7 days of telemetry, 40.8k local completions; ~10% finish in <=64 output tokens yet
queue in the full lanes because max_tokens is unset/large.  Run: python -m pytest -q test_gateway_micro_lane.py"""
import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch

import test_gateway_local_first as lf

shim = lf.shim
PROBE = "Reply with the single word: pong"
PONG_CTX = PROBE + "  ## Live estate state (measured, not remembered)\n- lanes 3/4"


def pbody(text, **kw):
    return json.dumps({"model": "estate", "messages": [{"role": "system", "content": "s"},
                                                      {"role": "user", "content": text}], **kw}).encode()


class Learner(unittest.TestCase):
    def setUp(self):
        shim._MICRO_HIST.clear()

    def feed(self, outs, client="c", text="Adversarially fact-check this claim 12"):
        for o in outs:
            shim.micro_observe(client, text, o)

    def test_needs_min_samples_then_qualifies(self):
        self.feed([5] * (shim.MICRO_MIN_SAMPLES - 1))
        self.assertFalse(shim.micro_predict("c", "Adversarially fact-check this claim 12", 500))
        self.feed([5])
        self.assertTrue(shim.micro_predict("c", "Adversarially fact-check this claim 12", 500))

    def test_digits_fold_so_ids_do_not_split_a_call_type(self):
        self.feed([7] * shim.MICRO_MIN_SAMPLES, text="Audit decision 1001 now")
        self.assertTrue(shim.micro_predict("c", "Audit decision 99999 now", 100))

    def test_one_long_answer_in_the_window_disqualifies(self):
        self.feed([5] * 10 + [shim.MICRO_MAX_OUT + 1])
        self.assertFalse(shim.micro_predict("c", "Adversarially fact-check this claim 12", 500))
        self.feed([5] * shim.MICRO_HISTORY)                       # ages the long one out of the window
        self.assertTrue(shim.micro_predict("c", "Adversarially fact-check this claim 12", 500))

    def test_big_prompt_client_and_empty_text_never_qualify(self):
        self.feed([5] * 10)
        self.assertFalse(shim.micro_predict("c", "Adversarially fact-check this claim 12", shim.MICRO_MAX_PROMPT + 1))
        self.assertFalse(shim.micro_predict("other", "Adversarially fact-check this claim 12", 100))
        self.assertFalse(shim.micro_predict("c", "", 100))

    def test_kill_switch(self):
        self.feed([5] * 10)
        with patch.object(shim, "MICRO_LEARN", False):
            self.assertFalse(shim.micro_predict("c", "Adversarially fact-check this claim 12", 100))

    def test_signature_table_is_bounded(self):
        with patch.object(shim, "MICRO_MAX_SIGS", 50):
            for i in range(200):
                shim.micro_observe("c", "unique-%s-head xxxxxxxxxxxxxxxxxxxxxxxx" % ("a" * (i % 20) + chr(97 + i % 26) * (i // 26 + 1)), 3)
            self.assertLessEqual(len(shim._MICRO_HIST), 50)

    def test_restore_from_request_log(self):
        with tempfile.TemporaryDirectory() as d, patch.object(shim, "TELEMETRY_DIR", d):
            now = time.time()
            rows = [{"t": now - 500 + i, "client": "c", "route": "local", "status": 200, "outtok": 4,
                     "preview": "Classify this card 7"} for i in range(8)]
            rows += [{"t": now - 400, "client": "c", "route": "remote", "status": 200, "outtok": 4, "preview": "remote one"},
                     {"t": now - 400, "client": "c", "route": "local", "status": 503, "outtok": 4, "preview": "failed one"}]
            with open(shim._history_day_file(now), "w") as fh:
                fh.write("\n".join(json.dumps(r) for r in rows) + "\n")
            got = shim.micro_history_restore_blocking(now, now)
            self.assertEqual(len(got), 8)                          # remote and failed rows are not evidence
            shim._MICRO_HIST.clear()
            shim.micro_history_restore(got)
            self.assertTrue(shim.micro_predict("c", "Classify this card 99", 100))


class ProbeParse(unittest.TestCase):
    def test_plain_and_context_tailed_probes_match(self):
        self.assertEqual(shim.probe_word(pbody(PROBE)), "pong")
        self.assertEqual(shim.probe_word(pbody(PONG_CTX)), "pong")
        self.assertEqual(shim.probe_word(pbody(PROBE + "  [plugin hook output truncated]")), "pong")
        self.assertEqual(shim.probe_word(pbody(PROBE + "  NOW (HNET00 system clock, 2026)")), "pong")
        self.assertEqual(shim.probe_word(pbody([{"type": "text", "text": PROBE}])), "pong")

    def test_real_requests_never_match(self):
        for t in (PROBE + ", then explain how the gateway routes", "Please " + PROBE, "Reply with the single word: pong. Then list the lanes",
                  "Reply with one word: yes or no, is 7 prime and why?", "hi", ""):
            self.assertIsNone(shim.probe_word(pbody(t)), t)
        self.assertIsNone(shim.probe_word(b"not json"))
        self.assertIsNone(shim.probe_word(json.dumps({"messages": [{"role": "user", "content": PROBE},
                                                                    {"role": "assistant", "content": "pong"},
                                                                    {"role": "user", "content": "now write the report"}]}).encode()))


class ProbeDecision(unittest.TestCase):
    def setUp(self):
        shim._PROBE_SEEN = 0

    def d(self, **kw):
        a = dict(word="pong", alias_kind="builtin-local", local_ok_age_s=5.0, health_ok=True, offline_or_forced=False)
        a.update(kw)
        return shim.probe_synth_decision(**a)

    def test_fresh_engine_serves_and_every_nth_goes_real(self):
        with patch.object(shim, "PROBE_REAL_EVERY", 4):
            out = [self.d()[0] for _ in range(8)]
        self.assertEqual(out, [True, True, True, False, True, True, True, False])

    def test_never_synthesises_without_proof(self):
        self.assertEqual(self.d(local_ok_age_s=shim.PROBE_FRESH_S + 1), (False, "stale"))
        self.assertEqual(self.d(health_ok=False), (False, "unhealthy"))
        self.assertEqual(self.d(offline_or_forced=True), (False, "window"))
        self.assertEqual(self.d(alias_kind="builtin-remote"), (False, "alias-builtin-remote"))
        self.assertEqual(self.d(alias_kind="custom-remote")[0], False)
        with patch.object(shim, "PROBE_SYNTH", False):
            self.assertFalse(self.d()[0])

    def test_response_shapes(self):
        r = shim.probe_synth_response(pbody(PROBE), "pong", False)
        j = json.loads(r.body)
        self.assertEqual(j["choices"][0]["message"]["content"], "pong")
        self.assertEqual(r.headers["X-Shim-Synth"], "probe")
        s = shim.probe_synth_response(pbody(PROBE), "pong", True)
        self.assertEqual(s.content_type, "text/event-stream")
        text = s.body.decode()
        self.assertTrue(text.rstrip().endswith("data: [DONE]"))
        chunks = [json.loads(x[6:]) for x in text.split("\n\n") if x.startswith("data: {")]
        self.assertEqual("".join(c["choices"][0]["delta"].get("content", "") for c in chunks), "pong")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")


class Routing(lf.RoutingBase):
    def setUp(self):
        super().setUp()
        shim._MICRO_HIST.clear()
        shim._PROBE_SEEN = 0
        self.ptok = 1_000
        for name, value in dict(_health={"ok": True, "at": time.time() + 1e6}, TINY_EXTRA_LANES=2, BUDGET=4,
                                _inflight=0, _LAST_LOCAL_OK=time.time(), MICRO_LEARN=True, PROBE_SYNTH=True,
                                PROBE_REAL_EVERY=0, _memory_available=lambda *a, **k: True).items():
            self.stack.enter_context(patch.object(shim, name, value))
        self.client = shim._friendly_client(lf.Request())["name"]

    def teach(self, text="review", outs=None):
        for o in outs or [3] * shim.MICRO_MIN_SAMPLES:
            shim.micro_observe(self.client, text, o)

    async def test_learned_signature_uses_the_tiny_lane_and_is_labelled(self):
        self.teach()
        self.assertEqual(await self.route(), "local")
        self.assertEqual(self.events, [("local", "tiny-learned")])
        self.assertEqual(self.calls, ["local"])

    async def test_unlearned_signature_is_not_tiny(self):
        self.assertEqual(await self.route(), "local")
        self.assertNotIn(("local", "tiny-learned"), self.events)

    async def test_learned_tiny_never_takes_the_tiny_fast_paid_overflow_when_the_tiny_lanes_are_full(self):
        # Full lanes: a learned request falls through to the NORMAL admission loop (here LOCAL_WAIT=0 so it times out
        # at once); it must not be shipped to the paid provider by the tiny-fast shortcut.
        self.teach()
        with patch.object(shim, "_inflight", 99):
            await self.route()
        self.assertNotIn(("remote", "tiny-fast"), self.events)
        self.assertNotIn(("local", "tiny-learned"), self.events)

    async def test_static_tiny_still_overflows_when_full(self):
        with patch.object(shim, "TINY_TOKENS", 5000), patch.object(shim, "_inflight", 99):
            self.assertEqual(await self.route(max_tokens=100), "remote")
        self.assertEqual(self.events, [("remote", "tiny-fast")])

    async def test_probe_is_answered_by_the_gateway_when_the_engine_is_fresh(self):
        r = await self.route(model="estate-local", messages=[{"role": "user", "content": PONG_CTX}])
        self.assertEqual(r.headers["X-Shim-Synth"], "probe")
        self.assertEqual(self.calls, [])
        self.assertEqual(self.events, [])                          # no slot, no engine, no remote

    async def test_probe_goes_through_for_real_when_the_engine_is_stale_or_alias_is_remote(self):
        with patch.object(shim, "_LAST_LOCAL_OK", time.time() - 3600):
            self.assertEqual(await self.route(messages=[{"role": "user", "content": PROBE}]), "local")
        self.assertEqual(self.calls, ["local"])
        r = await self.route(model="estate-remote", messages=[{"role": "user", "content": PROBE}])
        self.assertEqual(r, "remote")
        self.assertEqual(self.calls, ["remote"])

    async def test_probe_is_not_synthesised_inside_a_planned_offline_window(self):
        with patch.object(shim, "_local_offline", lambda now=None: True):
            r = await self.route(messages=[{"role": "user", "content": PROBE}])
        self.assertNotEqual(getattr(r, "headers", {}).get("X-Shim-Synth") if not isinstance(r, str) else None, "probe")


class Replay(unittest.TestCase):
    def test_replay_counts_admissions_misses_and_freed_slot_seconds(self):
        import micro_lane_replay as mr
        shim._MICRO_HIST.clear()
        n, t0 = shim.MICRO_MIN_SAMPLES, 1000.0
        rows = [{"t": t0 + i, "client": "c", "route": "local", "status": 200, "outtok": 5, "preview": "Triage item %d" % i,
                 "ptok": 800, "duration": 10.0, "admission_wait": 4.0} for i in range(n + 3)]
        rows.append({"t": t0 + 99, "client": "c", "route": "local", "status": 200, "outtok": 900,
                     "preview": "Triage item 77", "ptok": 800, "duration": 50.0, "admission_wait": 0.0})   # a wrong call
        rows.append({"t": t0 + 100, "client": "c", "route": "local", "status": 200, "outtok": 5, "tiny": True,
                     "preview": "static tiny", "ptok": 10, "duration": 1.0})
        res = mr.replay(rows, shim)
        self.assertEqual(res["admitted"], 4)                       # 3 qualifying + the wrong one; static-tiny never counted
        self.assertEqual(res["wrong_long"], 1)
        self.assertEqual(res["freed_slot_s"], 3 * 10.0 + 50.0)
        self.assertEqual(res["freed_wait_s"], 12.0)
        self.assertEqual(res["already_tiny"], 1)


if __name__ == "__main__":
    unittest.main()
