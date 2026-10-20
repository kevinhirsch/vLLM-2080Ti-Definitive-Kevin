#!/usr/bin/env python3
"""GW-STRICT: a strict provider (DeepSeek) must receive a valid tool-call history.

Measured 2026-10-19: the same OpenAI-format request is accepted by vLLM and 400'd by DeepSeek
("Messages with role 'tool' must be a response to a preceding message with 'tool_calls'") when a
`tool` message does not directly follow an assistant `tool_calls` carrying its id. OpenHands'
crash recovery produced exactly that and killed two agent lanes. The routing layer creates the
difference between backends, so the routing layer repairs it: strict_tool_history() on the
remote path only.

Pure helpers, no network. Loaded via importlib because the filename is hyphenated.
Run:  python3 -m pytest test_strict_tool_history.py
"""
import asyncio
import importlib.util
import json
import os
import pathlib
import types
import unittest

os.environ.setdefault("SHIM_EXACT_TOKENS", "0")

_HERE = pathlib.Path(__file__).resolve().parent


_MOD = None


def _load_candidate():
    # Loaded on first use, not at import: on a gateway WITHOUT this fix a module-level load
    # aborts the whole suite at collection, and the control run then covers nothing. A failed
    # load fails the tests one by one; a clean module fails every construction test.
    # Cached in _MOD: module_from_spec does not register the module, so re-loading on every
    # call would re-execute the gateway (and silently discard setUp's patches).
    global _MOD
    import pytest
    if _MOD is not None:
        return _MOD
    try:
        spec = importlib.util.spec_from_file_location("keepalive_shim_strict", _HERE / "keepalive-shim.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MOD = mod
        return mod
    except Exception as e:          # noqa: BLE001 -- surfaced per test, never swallowed
        pytest.fail(f"gateway module failed to load: {type(e).__name__}: {e}")


class _LazyShim:
    def __getattr__(self, name):
        return getattr(load(), name)


shim = _LazyShim()


def load():
    global ORPHAN, MISSING, shim
    m = _load_candidate()
    # Rebind: shim = _LazyShim() above ran before this def existed, so it captured the pytest
    # module instead; and pytest_configure runs before this body finishes, so no hook can bind.
    shim = m
    ORPHAN = m.STRICT_ORPHAN_TOOL_PREFIX
    MISSING = m.STRICT_MISSING_RESULT
    return m


ORPHAN = MISSING = None


def _call(cid, name="read", args="{}"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


def _assistant(*cids, content=None):
    m = {"role": "assistant", "content": content}
    if cids:
        m["tool_calls"] = [_call(c) for c in cids]
    return m


def _tool(cid, content="ok"):
    return {"role": "tool", "tool_call_id": cid, "content": content}


def _user(text="go"):
    return {"role": "user", "content": text}


def _roles(messages):
    return [((m.get("role"), m.get("tool_call_id")) if isinstance(m, dict) else m) for m in messages]


def _assert_strict(test, messages):
    """What DeepSeek requires: every tool message answers the nearest preceding assistant
    (only tool messages between), and every tool call of that assistant is answered."""
    pending = []
    answered = set()
    for m in messages:
        role = m.get("role")
        if role == "tool":
            cid = m.get("tool_call_id")
            cid = None if cid is None else str(cid)
            test.assertTrue(pending, f"tool message with no assistant turn: {m}")
            test.assertIn(cid, pending, f"orphan tool result: {m}")
            answered.add(cid)
        elif role == "assistant" and load()._call_ids(m):
            # A turn is closed only where a provider reads it as closed: an assistant message
            # that carries tool_calls and is not itself a tool result.
            test.assertEqual(set(pending) - answered, set(),
                             f"assistant turn left calls unanswered: {set(pending) - answered}")
            pending = load()._call_ids(m)
            answered = set()
    test.assertEqual(set(pending) - answered, set(), "last assistant turn left calls unanswered")


class ValidHistoryIdentity(unittest.TestCase):
    def test_valid_history_is_the_same_objects_and_zero_changes(self):
        h = [_user(), _assistant("A"), _tool("A"), _assistant("B", content="done"), _tool("B", "rb")]
        out, changes = load().strict_tool_history(h)
        self.assertEqual(changes, 0)
        self.assertIs(out, h)
        for got, want in zip(out, h):
            self.assertIs(got, want)

    def test_valid_history_with_messages_after_results_is_unchanged(self):
        h = [_user(), _assistant("A"), _tool("A"), _assistant(content="done"), _user("/next")]
        out, changes = load().strict_tool_history(h)
        self.assertEqual(changes, 0)
        self.assertIs(out, h)


class RepairRuleOrphanTool(unittest.TestCase):
    def test_tool_with_unknown_id_becomes_user_and_keeps_content(self):
        h = [_user(), _assistant("A"), _tool("A"), _tool("ZZZ", "file body")]
        out, changes = load().strict_tool_history(h)
        self.assertEqual(changes, 1)
        self.assertEqual(out[-1]["role"], "user")
        self.assertEqual(out[-1]["content"], ORPHAN + "file body")
        self.assertNotIn("tool_call_id", out[-1])
        self.assertEqual(len(out), len(h))
        _assert_strict(self, out)

    def test_tool_before_any_assistant_is_orphaned_too(self):
        h = [_tool("A", "early"), _user(), _assistant("A"), _tool("A")]
        self.assertEqual(_roles(h)[-2:], [("assistant", None), ("tool", "A")])
        out, changes = load().strict_tool_history(h)
        self.assertEqual(changes, 1)
        self.assertEqual((out[0]["role"], out[0]["content"]), ("user", ORPHAN + "early"))
        _assert_strict(self, out)

    def test_tool_matching_an_earlier_turn_is_not_claimed_by_the_next(self):
        # The turn's call A is answered; the trailing tool A is a NEW tool message with nothing
        # pending (the assistant before it answers with content, not tool_calls).
        h = [_user(), _assistant("A"), _tool("A"), _assistant(content="ok"), _tool("A")]
        out, changes = load().strict_tool_history(h)
        self.assertEqual(changes, 1)
        self.assertEqual(out[-1]["role"], "user")
        _assert_strict(self, out)

    def test_non_string_content_is_serialised_not_dropped(self):
        h = [_user(), _assistant("A"), _tool("B", [{"type": "text", "text": "x"}])]
        out, changes = load().strict_tool_history(h)
        self.assertEqual(changes, 2)                # B orphaned, then A gets its missing result
        self.assertEqual(len(out), len(h) + 1)
        self.assertEqual(out[2]["role"], "user")
        self.assertIn('"text": "x"', out[2]["content"])
        self.assertTrue(out[2]["content"].startswith(ORPHAN))
        self.assertEqual(out[3], {"role": "tool", "tool_call_id": "A", "content": MISSING})

    def test_extra_keys_on_the_tool_message_survive(self):
        m = _tool("B", "r")
        m["name"] = "read"
        h = [_user(), _assistant("A"), _tool("A"), m]
        out, _ = load().strict_tool_history(h)
        self.assertEqual(out[-1]["name"], "read")
        self.assertNotIn("tool_call_id", out[-1])
        self.assertEqual(out[-1]["content"], ORPHAN + "r")


class RepairRuleMissingResult(unittest.TestCase):
    def test_call_without_result_gets_one_before_the_next_message(self):
        h = [_user(), _assistant("A"), _user("never mind")]
        out, changes = load().strict_tool_history(h)
        self.assertEqual(changes, 1)
        self.assertEqual(_roles(out), [("user", None), ("assistant", None), ("tool", "A"), ("user", None)])
        self.assertEqual((out[2]["content"], out[2]["role"], out[2]["tool_call_id"]), (MISSING, "tool", "A"))
        _assert_strict(self, out)

    def test_call_without_result_at_the_end_of_the_history(self):
        h = [_user(), _assistant("A")]
        out, changes = load().strict_tool_history(h)
        self.assertEqual(changes, 1)
        self.assertEqual(out[-1], {"role": "tool", "tool_call_id": "A", "content": MISSING})

    def test_inserted_result_follows_the_partial_results_of_the_same_turn(self):
        h = [_user(), _assistant("A", "B", "C"), _tool("A"), _tool("C"), _user("q")]
        out, changes = load().strict_tool_history(h)
        self.assertEqual(changes, 1)
        self.assertEqual(_roles(out)[2:6],
                         [("tool", "A"), ("tool", "C"), ("tool", "B"), ("user", None)])
        _assert_strict(self, out)

    def test_duplicate_result_for_one_call_is_not_an_orphan(self):
        h = [_user(), _assistant("A"), _tool("A", "1"), _tool("A", "2")]
        out, changes = load().strict_tool_history(h)
        self.assertEqual(changes, 0)
        self.assertIs(out, h)


class OpenHandsCrashHistory(unittest.TestCase):
    """The measured shape: tool, assistant(call A), tool(A), tool(B, orphan), user. The leading
    `tool` belongs to a turn the crash recovery dropped, so it is orphaned too; B answers a call
    that is not in this history and becomes a user note. Both keep their content, in order."""

    def setUp(self):
        self.h = [_tool("pre", "old result"), _assistant("A"), _tool("A", "ra"),
                  _tool("B", "rb"), _user("continue")]

    def test_repaired_history_is_strict_and_keeps_every_result(self):
        out, changes = load().strict_tool_history(self.h)
        self.assertEqual(changes, 2)
        _assert_strict(self, out)
        self.assertEqual(_roles(out), [("user", None), ("assistant", None), ("tool", "A"),
                                       ("user", None), ("user", None)])
        self.assertEqual(out[0]["content"], ORPHAN + "old result")
        self.assertEqual(out[3]["content"], ORPHAN + "rb")
        self.assertEqual(out[4], self.h[4])

    def test_second_run_is_a_no_op(self):
        once, _ = load().strict_tool_history(self.h)
        twice, changes = load().strict_tool_history(once)
        self.assertEqual(changes, 0)
        self.assertIs(twice, once)

    def test_input_is_not_mutated(self):
        snapshot = json.dumps(self.h, sort_keys=True)
        load().strict_tool_history(self.h)
        self.assertEqual(json.dumps(self.h, sort_keys=True), snapshot)


class ParallelToolCalls(unittest.TestCase):
    def test_valid_parallel_results_unanswered_out_of_order(self):
        h = [_user(), _assistant("A", "B", "C"), _tool("C", "rc"), _tool("A", "ra"), _tool("B", "rb")]
        out, changes = load().strict_tool_history(h)
        self.assertEqual(changes, 0)
        self.assertIs(out, h)

    def test_missing_one_parallel_result_inserts_only_that_one(self):
        h = [_user(), _assistant("A", "B", "C"), _tool("A", "ra"), _tool("C", "rc"), _assistant(content="x")]
        out, changes = load().strict_tool_history(h)
        self.assertEqual(changes, 1)
        self.assertEqual(_roles(out), [("user", None), ("assistant", None), ("tool", "A"),
                                       ("tool", "C"), ("tool", "B"), ("assistant", None)])
        _assert_strict(self, out)

    def test_orphan_beside_valid_parallel_results(self):
        h = [_user(), _assistant("A", "B"), _tool("A", "ra"), _tool("Q", "rq"), _tool("B", "rb")]
        out, changes = load().strict_tool_history(h)
        self.assertEqual(changes, 1)
        self.assertEqual(out[3]["role"], "user")
        self.assertEqual(out[3]["content"], ORPHAN + "rq")
        self.assertEqual(out[4], h[4])
        self.assertEqual(len(out), len(h))
        _assert_strict(self, out)


class NoTools(unittest.TestCase):
    def test_plain_conversation_is_unchanged(self):
        h = [{"role": "system", "content": "s"}, _user("a"), _assistant(content="b"), _user("c")]
        out, changes = load().strict_tool_history(h)
        self.assertEqual(changes, 0)
        self.assertIs(out, h)

    def test_empty_history(self):
        self.assertEqual(load().strict_tool_history([]), ([], 0))

    def test_assistant_without_tool_calls_then_tool_is_orphaned(self):
        h = [_user(), _assistant(content="no call"), _tool("A", "r")]  # no tool_calls anywhere
        out, changes = load().strict_tool_history(h)
        self.assertEqual(changes, 1)
        self.assertEqual(out[-1]["role"], "user")


class NeverRaises(unittest.TestCase):
    def test_non_list_passthrough(self):
        for v in (None, "text", {"role": "user"}, 7):
            self.assertEqual(load().strict_tool_history(v), (v, 0))

    def test_malformed_message_passes_through_untouched(self):
        bad = [None, 3, "x", [], {"role": 7}, {"no": "role"}]
        h = [_user()] + bad + [_assistant("A"), _tool("A")]
        out, changes = load().strict_tool_history(h)
        self.assertEqual(changes, 0)
        self.assertIs(out, h)

    def test_malformed_tool_calls_and_ids(self):
        h = [_user(),
             {"role": "assistant", "content": None, "tool_calls": "nope"},
             _tool("A"),                             # nothing pending: orphaned (rule 1), not a crash
             {"role": "tool"},                       # no tool_call_id at all: orphaned too
             {"role": "assistant", "content": None, "tool_calls": [1, None, {"no": "id"}, {"id": 5}]},
             _tool(5, "five"),                       # int id matches the str()'d call id
             ]
        out, changes = load().strict_tool_history(h)
        self.assertEqual(_roles(out), [("user", None), ("assistant", None), ("user", None), ("user", None),
                                      ("assistant", None), ("tool", 5)])
        self.assertEqual(changes, 2)
        _assert_strict(self, out)

    def test_broken_entries_do_not_stop_the_repair_of_the_rest(self):
        h = [_user(), 42, {"role": "assistant", "tool_calls": [{"id": "A"}]}, _user("next"),
             _assistant("B"), _tool("B")]
        out, changes = load().strict_tool_history(h)
        self.assertEqual(changes, 1)                # only A is missing its result
        self.assertEqual(_roles(out)[1:], [42, ("assistant", None), ("tool", "A"), ("user", None),
                                          ("assistant", None), ("tool", "B")])


class RemoteBody(unittest.TestCase):
    def _body_of(self, messages):
        return json.dumps({"model": "qwen-local", "messages": messages}).encode()

    def test_valid_body_is_byte_identical(self):
        b = self._body_of([_user(), _assistant("A"), _tool("A")])
        out, changes = load()._strict_remote_body(b)
        self.assertEqual(changes, 0)
        self.assertIs(out, b)

    def test_orphan_body_is_repaired_on_the_wire(self):
        b = self._body_of([_user(), _assistant("A"), _tool("A"), _tool("B", "rb")])
        out, changes = load()._strict_remote_body(b)
        self.assertEqual(changes, 1)
        msgs = json.loads(out)["messages"]
        self.assertEqual(msgs[-1]["content"], ORPHAN + "rb")
        _assert_strict(self, msgs)

    def test_garbage_and_message_less_bodies_pass_through(self):
        for b in (b"not json", b"[]", b'{"model":"x"}', b'{"messages":"nope"}'):
            self.assertIs(load()._strict_remote_body(b)[0], b)


class _Req:
    """Just what _forward_remote/_relay touch on an aiohttp request."""

    def __init__(self, body, client="openhands-lane-7"):
        self._body = body
        self.headers = {"X-Client": client, "Content-Type": "application/json"}
        self.remote = "10.0.1.50"
        self.path = "/v1/chat/completions"
        self.method = "POST"
        self.content_type = "application/json"

    async def read(self):
        return self._body


class RemotePathAppliesLocalDoesNot(unittest.TestCase):
    """Drive the real _forward_remote() and _relay(): only the HTTP POST (aiohttp.ClientSession)
    and the spend authority are replaced; routing, remap, the strict step, telemetry and the
    local path are the shipped code."""

    def setUp(self):
        self.sent = []

        class _Resp:
            status = 200
            headers = {"Content-Type": "application/json"}   # _relay reads upstream headers

            async def text(self):
                return '{"ok": 1}'

            async def read(self):
                return b'{"ok": 1}'

        session = types.SimpleNamespace()

        def _fake_post(url, data=None, headers=None, timeout=None):
            self.sent.append({"url": url, "body": data, "headers": dict(headers or {})})
            fut = asyncio.get_event_loop().create_future()
            fut.set_result(_Resp())
            return fut

        session.post = _fake_post
        session.close = lambda: asyncio.sleep(0)
        self._session = session

        class _CM:
            def __init__(self_):
                self_.session = session

            async def __aenter__(self_):
                return session

            async def __aexit__(self_, *a):
                return False

            async def close(self_):
                return None

            post = session.post

        # _relay uses aiohttp.ClientSession() as a plain object, not a context manager,
        # so the fake has to look like the session itself.
        fake = _CM()
        fake.post = session.post
        fake.close = session.close
        self._saved = (load().aiohttp.ClientSession, load()._spend_hold_for, load().remote_ok, load().REMOTE_BASE)
        load().aiohttp.ClientSession = lambda *a, **k: fake
        load()._spend_hold_for = lambda *a, **k: None
        load().remote_ok = lambda *a, **k: True
        if not load().REMOTE_BASE:
            load().REMOTE_BASE = "https://api.deepseek.com"
        self._active_before = dict(load()._ACTIVE)

    def tearDown(self):
        load().aiohttp.ClientSession, load()._spend_hold_for, load().remote_ok, load().REMOTE_BASE = self._saved
        load()._ACTIVE.clear()
        load()._ACTIVE.update(self._active_before)

    def _broken_body(self):
        return json.dumps({"model": "qwen-local", "messages":
                           [_tool("pre", "old"), _assistant("A"), _tool("A", "ra"),
                            _tool("B", "rb"), _user("go")]}).encode()

    def test_remote_body_carries_the_repaired_history(self):
        req = _Req(self._broken_body())
        load()._ACTIVE[id(req)] = {}
        with self.assertLogs("gateway-shim", level="INFO") as logs:
            resp = asyncio.run(load()._forward_remote(req, "/v1/chat/completions", self._broken_body(), False))
        self.assertEqual(resp.status, 200)
        self.assertEqual(len(self.sent), 1)
        sent = json.loads(self.sent[0]["body"])
        _assert_strict(self, sent["messages"])
        self.assertEqual(sent["model"], load().REMOTE_MODEL)          # remap still ran first
        self.assertEqual(load()._ACTIVE[id(req)].get("tool_history_changes"), 2)
        self.assertTrue(any("strict tool history" in m and "openhands-lane-7" in m for m in logs.output),
                        logs.output)
        # the strict-history line, wherever it falls (in the repo the shim first logs loading remote-pricing.json)
        self.assertIn("client=openhands-lane-7", next(m for m in logs.output if "strict tool history" in m))
        self.assertIn("changes=2", next(m for m in logs.output if "strict tool history" in m))

    def test_remote_valid_history_is_sent_unchanged_and_not_logged(self):
        body = json.dumps({"model": "qwen-local", "messages": [_user(), _assistant("A"), _tool("A")]}).encode()
        req = _Req(body)
        load()._ACTIVE[id(req)] = {}
        with self.assertLogs("gateway-shim", level="INFO") as logs:
            asyncio.run(load()._forward_remote(req, "/v1/chat/completions", body, False))
        self.assertNotIn("strict tool history", "\n".join(logs.output))
        self.assertNotIn("tool_history_changes", load()._ACTIVE[id(req)])

    def test_local_path_does_not_call_the_repair(self):
        calls = []
        orig = load().strict_tool_history

        def _spy(messages):
            calls.append(messages)
            return orig(messages)

        load().strict_tool_history = _spy
        try:
            asyncio.run(load()._relay(_Req(self._broken_body()), "http://127.0.0.1:8000", "/v1/chat/completions",
                                    self._broken_body(), "", False))
        finally:
            load().strict_tool_history = orig
        self.assertEqual(calls, [])
        sent = json.loads(self.sent[0]["body"])                      # what the local engine receives
        self.assertEqual(sent["messages"][0]["role"], "tool")        # untouched, as vLLM expects
