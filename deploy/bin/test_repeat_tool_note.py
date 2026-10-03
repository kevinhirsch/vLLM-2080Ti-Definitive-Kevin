"""[LANE RA / L143] cross-turn identical tool-call loop note.

Measured root (runner pi transcripts, 2026-10-03): Qwen re-issued one identical tool call with an identical
result 21x mid-task and 7-54x after finishing work; nothing in the request told it the result was unchanged.
"""
import importlib.util
import json
import pathlib
import unittest
from unittest.mock import patch

_HERE = pathlib.Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("keepalive_shim_ra", _HERE / "keepalive-shim.py")
shim = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(shim)


def _exchange(n, cmd="grep -rn prompt tools/estate_telemetry.py | head -20", result="5: prompt size\n35: bands"):
    cid = "call_%d" % n
    return [
        {"role": "assistant", "content": "Let me look at the telemetry row shape.",
         "tool_calls": [{"id": cid, "type": "function",
                         "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}]},
        {"role": "tool", "tool_call_id": cid, "content": result},
    ]


def _body(exchanges):
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "do the task"}]
    for e in exchanges:
        msgs += e
    return json.dumps({"model": "estate", "messages": msgs, "stream": True}).encode()


class RepeatToolNote(unittest.TestCase):
    def test_three_identical_exchanges_get_one_note(self):
        out, n = shim.repeated_tool_call_note(_body([_exchange(i) for i in range(3)]))
        self.assertEqual(n, 3)
        msgs = json.loads(out)["messages"]
        self.assertIn(shim.REPEAT_NOTE_MARKER, msgs[-1]["content"])
        self.assertIn("3 times in a row", msgs[-1]["content"])
        self.assertTrue(msgs[-1]["content"].startswith("5: prompt size"))   # result itself untouched
        # nothing else changed
        self.assertEqual(sum(shim.REPEAT_NOTE_MARKER in str(m.get("content")) for m in msgs), 1)

    def test_counts_the_whole_identical_run(self):
        ex = [_exchange(0, cmd="ls")] + [_exchange(i) for i in range(1, 22)]
        _, n = shim.repeated_tool_call_note(_body(ex))
        self.assertEqual(n, 21)

    def test_two_is_not_a_loop(self):
        body = _body([_exchange(i) for i in range(2)])
        self.assertEqual(shim.repeated_tool_call_note(body), (body, 0))

    def test_changed_result_is_progress_not_a_loop(self):
        # re-running tests after an edit: same call, different output -> never noted
        ex = [_exchange(0, result="2 failed"), _exchange(1, result="1 failed"), _exchange(2, result="ok")]
        body = _body(ex)
        self.assertEqual(shim.repeated_tool_call_note(body), (body, 0))

    def test_different_calls_are_exploration_not_a_loop(self):
        ex = [_exchange(i, cmd="sed -n %d,%dp f.py" % (i, i + 9)) for i in range(5)]
        body = _body(ex)
        self.assertEqual(shim.repeated_tool_call_note(body), (body, 0))

    def test_argument_key_order_does_not_hide_a_loop(self):
        ex = []
        for i, args in enumerate(['{"path":"a","limit":5}', '{"limit":5,"path":"a"}', '{"path": "a", "limit": 5}']):
            cid = "c%d" % i
            ex.append([{"role": "assistant", "content": "", "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": "read", "arguments": args}}]},
                {"role": "tool", "tool_call_id": cid, "content": "x"}])
        _, n = shim.repeated_tool_call_note(_body(ex))
        self.assertEqual(n, 3)

    def test_idempotent_on_second_pass(self):
        out, n = shim.repeated_tool_call_note(_body([_exchange(i) for i in range(4)]))
        self.assertEqual(n, 4)
        self.assertEqual(shim.repeated_tool_call_note(out), (out, 0))

    def test_only_when_last_message_is_a_tool_result(self):
        ex = [_exchange(i) for i in range(4)]
        d = json.loads(_body(ex))
        d["messages"].append({"role": "user", "content": "keep going"})
        body = json.dumps(d).encode()
        self.assertEqual(shim.repeated_tool_call_note(body), (body, 0))

    def test_list_content_gets_a_text_part(self):
        ex = [_exchange(i) for i in range(3)]
        d = json.loads(_body(ex))
        for m in d["messages"]:
            if m["role"] == "tool":
                m["content"] = [{"type": "text", "text": "same"}]
        out, n = shim.repeated_tool_call_note(json.dumps(d).encode())
        self.assertEqual(n, 3)
        last = json.loads(out)["messages"][-1]["content"]
        self.assertEqual(last[0], {"type": "text", "text": "same"})
        self.assertIn(shim.REPEAT_NOTE_MARKER, last[-1]["text"])

    def test_parallel_calls_compared_as_a_set(self):
        def ex(i):
            a, b = "a%d" % i, "b%d" % i
            return [{"role": "assistant", "content": "", "tool_calls": [
                {"id": a, "type": "function", "function": {"name": "read", "arguments": '{"path":"x"}'}},
                {"id": b, "type": "function", "function": {"name": "read", "arguments": '{"path":"y"}'}}]},
                {"role": "tool", "tool_call_id": a, "content": "X"},
                {"role": "tool", "tool_call_id": b, "content": "Y"}]
        _, n = shim.repeated_tool_call_note(_body([ex(i) for i in range(3)]))
        self.assertEqual(n, 3)

    def test_kill_switch(self):
        body = _body([_exchange(i) for i in range(5)])
        with patch.object(shim, "REPEAT_NOTE", False):
            self.assertEqual(shim.repeated_tool_call_note(body), (body, 0))

    def test_malformed_body_is_passed_through(self):
        for body in (b"not json", b"{}", json.dumps({"messages": "x"}).encode(),
                     json.dumps({"messages": [{"role": "tool"}]}).encode()):
            self.assertEqual(shim.repeated_tool_call_note(body), (body, 0))

    def test_wired_into_the_one_local_prep_chain(self):
        src = (_HERE / "keepalive-shim.py").read_text()
        self.assertEqual(src.count("repeated_tool_call_note(prepared)"), 1)

    def test_prep_chain_applies_note_and_records_it(self):
        class Req(dict):
            remote = "10.0.1.12"
            headers = {}
        req = Req()
        seen = {}
        with patch.object(shim, "_active_set", lambda r, **kw: seen.update(kw)):
            out = shim._prepare_local_body(req, _body([_exchange(i) for i in range(3)]), False)
        self.assertIn(shim.REPEAT_NOTE_MARKER, json.loads(out)["messages"][-1]["content"])
        self.assertEqual(seen.get("repeat_tool_note"), 3)

    def test_schema_registers_the_knobs(self):
        spec = importlib.util.spec_from_file_location("gcs_ra", _HERE / "gateway_config_schema.py")
        gcs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gcs)
        self.assertIn("SHIM_REPEAT_NOTE", gcs.SCHEMA)
        self.assertIn("SHIM_REPEAT_NOTE_MIN", gcs.SCHEMA)


if __name__ == "__main__":
    unittest.main()
