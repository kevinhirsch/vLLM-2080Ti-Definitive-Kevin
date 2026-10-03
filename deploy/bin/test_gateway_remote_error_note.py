#!/usr/bin/env python3
"""Lane A6: _note_remote_response classified provider ERROR answers as 'remote returned EMPTY content'.

LIVE 2026-10-03 (journal): every provider 400 ('Prompt must contain the word json') and 429 ('concurrency limit of 6 based on your
remaining balance') was followed by 'WARNING remote returned EMPTY content (finish_reason=None, completion_tokens=None)' -- 12 times.
Each one also (a) stamped content_empty=True on the request-log row, so the empty-answer rate the dashboards and the empty-turn lane
read counted refusals as empty generations, and (b) wrote the error body into the 40-file flight-recorder ring (flightrec/*_remote_*.resp.json),
which is the ring the Xid-31 repro payloads live in: a 429 storm evicts them. Only a successful (< 400) answer has content to classify.

Run:  python -m pytest -q test_gateway_remote_error_note.py
"""
import asyncio
import json
import os
import tempfile
import types
import unittest
from unittest.mock import patch

import test_gateway_cache_model as C

shim = C.shim


class P:
    def __init__(self, status, body):
        self.status, self.body = status, body


REQ = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
ERR429 = json.dumps({"error": {"message": "Too many requests. Your current concurrency is 6", "type": "rate_limit_error"}}).encode()
EMPTY200 = json.dumps({"choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 5, "completion_tokens": 0}}).encode()
OK200 = json.dumps({"choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2}}).encode()


class RemoteErrorNote(unittest.TestCase):
    def run_note(self, status, body):
        sets, warns, request = [], [], types.SimpleNamespace()
        with tempfile.TemporaryDirectory() as d, patch.object(shim, "_flightrec_dir", lambda: d), \
                patch.object(shim, "_active_set", lambda r, **kw: sets.append(kw)), \
                patch.object(shim.log, "warning", lambda fmt, *a: warns.append(fmt % a if a else fmt)):
            asyncio.run(shim._note_remote_response(request, P(status, body), REQ))
            files = sorted(os.listdir(d))
        return sets, files, "\n".join(warns)

    def test_provider_errors_are_not_empty_answers(self):
        for status in (400, 401, 402, 429, 500, 503):
            with self.subTest(status=status):
                sets, files, log = self.run_note(status, ERR429)
                self.assertNotIn("EMPTY", log)
                self.assertEqual(files, [])                             # nothing into the flight-recorder ring
                self.assertFalse(any("content_empty" in s for s in sets), sets)

    def test_a_real_empty_200_is_still_flagged_and_kept(self):
        sets, files, log = self.run_note(200, EMPTY200)
        self.assertIn("remote returned EMPTY content", log)
        self.assertTrue(any(s.get("content_empty") for s in sets))
        self.assertTrue(any(f.endswith(".resp.json") for f in files), files)

    def test_a_normal_200_is_classified_not_empty(self):
        sets, files, log = self.run_note(200, OK200)
        self.assertNotIn("EMPTY", log)
        self.assertTrue(any(s.get("content_empty") is False for s in sets), sets)


if __name__ == "__main__":
    unittest.main()
