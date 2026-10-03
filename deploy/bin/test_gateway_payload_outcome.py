#!/usr/bin/env python3
"""Lane A6: _note_payload_outcome counted every non-JSON error body as a 'swallowed exception'.

Live /gateway/internal-errors (2026-10-03 16:20): _note_payload_outcome n=20, last JSONDecodeError('Expecting value: line 1
column 1 (char 0)'). The helper json.loads() every buffered non-streaming body to read usage; a plain-text/empty error body
(no usage to read) is normal, not a failure, and it made the SH swallowed-exception counter -- the thing that is supposed to
surface every-request failures -- cry wolf. Usage exists only in a JSON object body of a successful answer.

Run:  python -m pytest -q test_gateway_payload_outcome.py
"""
import json
import types
import unittest

import test_gateway_cache_model as C

shim = C.shim


class Payload:
    def __init__(self, status, body):
        self.status, self.body = status, body


class NotePayloadOutcome(unittest.TestCase):
    def setUp(self):
        shim._SWALLOWED.pop("_note_payload_outcome", None)
        self.req = types.SimpleNamespace()
        shim._ACTIVE[id(self.req)] = {"route": "local"}
        self.addCleanup(shim._ACTIVE.pop, id(self.req), None)

    def note(self, status, body):
        shim._note_payload_outcome(self.req, Payload(status, body), False)
        return shim._ACTIVE[id(self.req)]

    def test_success_usage_is_recorded(self):
        body = json.dumps({"usage": {"completion_tokens": 7, "prompt_tokens": 100,
                                     "prompt_tokens_details": {"cached_tokens": 40}}}).encode()
        a = self.note(200, body)
        self.assertEqual((a["outtok"], a["computed_actual"], a["cached_actual"], a["ptok_exact_local"], a["http_status"]),
                         (7, 60, 40, 100, 200))
        self.assertNotIn("_note_payload_outcome", shim._SWALLOWED)

    def test_plain_text_and_empty_error_bodies_are_not_swallowed_failures(self):
        for status, body in ((502, b"Bad Gateway"), (503, b""), (500, b"<html>oops</html>"), (429, b"slow down\n")):
            with self.subTest(status=status, body=body):
                a = self.note(status, body)
                self.assertEqual(a["http_status"], status)       # the status is still recorded
                self.assertNotIn("outtok", a)
        self.assertNotIn("_note_payload_outcome", shim._SWALLOWED)

    def test_a_200_with_a_corrupt_json_body_is_still_counted(self):
        self.note(200, b'{"usage": {"completion_tokens": ')       # a real parse failure on an answer that should carry usage
        self.assertEqual(shim._SWALLOWED["_note_payload_outcome"][0], 1)


if __name__ == "__main__":
    unittest.main()
