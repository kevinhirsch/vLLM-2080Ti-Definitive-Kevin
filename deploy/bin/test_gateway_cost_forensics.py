#!/usr/bin/env python3
"""Cost attribution and duplicate-request telemetry regressions."""
import importlib.util
import json
import pathlib
import tempfile
import time
import unittest


HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("shim_cost_forensics", HERE / "keepalive-shim.py")
shim = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shim)


class Request:
    def __init__(self, headers=None):
        self.headers = headers or {}


class Identity(unittest.TestCase):
    def test_raw_caller_identity_and_prompt_are_not_persisted(self):
        body = b'{"messages":[{"role":"user","content":"private prompt"}]}'
        got = shim._request_identity(Request({"Idempotency-Key": "secret-job-42"}), body)
        self.assertEqual(got["request_id_source"], "idempotency-key")
        self.assertEqual(len(got["request_id"]), 24)
        self.assertNotIn("secret-job-42", repr(got))
        self.assertNotIn("private prompt", repr(got))
        self.assertEqual(len(got["request_body_sha256"]), 64)

    def test_body_identity_exists_without_a_caller_request_id(self):
        got = shim._request_identity(Request(), b"same body")
        self.assertIsNone(got["request_id"])
        self.assertEqual(len(got["request_body_sha256"]), 64)


class HistoryCostSummary(unittest.TestCase):
    def test_cache_cost_and_exact_repeats_are_attributed_per_client(self):
        now = time.time()
        rows = [
            {"t": now, "client": "halo", "route": "remote", "remote_sent": True,
             "ptok": 100, "outtok": 5, "remote_cache_hit": 90, "remote_cache_miss": 10,
             "cost_est": 0.01, "request_body_sha256": "a" * 64, "status": 200},
            {"t": now, "client": "halo", "route": "remote", "remote_sent": True,
             "ptok": 100, "outtok": 7, "remote_cache_hit": 95, "remote_cache_miss": 5,
             "cost_est": 0.02, "request_body_sha256": "a" * 64, "status": 200},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "requests.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            old = shim._history_day_file
            shim._history_day_file = lambda _epoch: str(path)
            try:
                got = shim._history_summary_blocking(now - 60, 1)
            finally:
                shim._history_day_file = old
        halo = got["per_client"]["halo"]
        self.assertEqual(halo["remote_cache_hit_tokens"], 185)
        self.assertEqual(halo["remote_cache_miss_tokens"], 15)
        self.assertEqual(halo["remote_cost_usd"], 0.03)
        self.assertEqual(halo["exact_body_repeats"], 1)


if __name__ == "__main__":
    unittest.main()
