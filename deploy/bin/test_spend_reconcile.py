#!/usr/bin/env python3
"""spend_reconcile.py against the provider's bill (R2 v5).

Fixture NUMBERS are Kevin's DeepSeek bill for 2026-09-25 00:00-13:00 Phoenix (model
deepseek-flash): 5,754 requests; 262,519,512 input cache-hit tokens @ $0.003/Mtok;
9,879,145 cache-miss @ $0.15/Mtok; 1,733,123 output @ $0.60/Mtok; $3.3093 billed. The CSV
here is synthetic (no user id, no key): only the columns the tool reads.

Run:  python3 -m unittest test_spend_reconcile
"""
import csv
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("spend_reconcile", Path(__file__).with_name("spend_reconcile.py"))
sr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sr)

START, END = "2026-09-25T00:00:00-07:00", "2026-09-25T13:00:00-07:00"
BILL = {"request_count": ("", 5_754), "input_cache_hit_tokens": ("0.000000003", 262_519_512),
        "input_cache_miss_tokens": ("0.00000015", 9_879_145), "output_tokens": ("0.0000006", 1_733_123)}


def write_bill(path, hours=13):
    """Spread the day's totals over `hours` hourly rows, as the provider export does."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["start_time_iso", "end_time_iso", "model", "type", "price", "amount"])
        for kind, (price, total) in BILL.items():
            for h in range(hours):
                amount = total // hours + (total % hours if h == hours - 1 else 0)
                w.writerow([f"2026-09-25T{h:02d}:00:00-07:00", f"2026-09-25T{h + 1:02d}:00:00-07:00",
                            "deepseek-flash", kind, price, amount])


def telemetry(tdir, rows):
    tdir.mkdir(parents=True, exist_ok=True)
    with open(tdir / "requests-20260925.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def usage_rows(n=5_754, hit=262_519_512, miss=9_879_145, out=1_733_123, t0=None):
    t0 = t0 or sr._ts(START)
    step = (sr._ts(END) - t0) / n
    return [{"t": t0 + i * step, "route": "remote", "alias_kind": "default", "status": 200,
             "cost_basis": "actual",
             "remote_cache_hit": hit // n + (1 if i < hit % n else 0),
             "remote_cache_miss": miss // n + (1 if i < miss % n else 0),
             "outtok": out // n + (1 if i < out % n else 0)} for i in range(n)]


class Reconcile(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="reconcile-"))
        self.bill = self.dir / "amount.csv"
        write_bill(self.bill)
        self.rows = sr.read_billing([str(self.bill)])

    def test_billing_totals_are_the_bill(self):
        b = sr.billing_totals(self.rows, sr._ts(START), sr._ts(END))
        self.assertEqual((b["requests"], b["cache_hit"], b["cache_miss"], b["output"]),
                         (5_754, 262_519_512, 9_879_145, 1_733_123))
        self.assertAlmostEqual(b["usd"], 3.309304, places=5)
        self.assertEqual(sr.bill_prices(self.rows), (3e-09, 1.5e-07, 6e-07))

    def test_a_gateway_log_with_provider_usage_replays_to_the_bill_within_2_percent(self):
        telemetry(self.dir / "t", usage_rows())
        tel = sr.telemetry_rows(str(self.dir / "t"), sr._ts(START), sr._ts(END))
        r = sr.reconcile(self.rows, tel, sr._ts(START), sr._ts(END))
        self.assertTrue(r["comparable"] and r["ok"])
        self.assertLessEqual(abs(r["relative_error"]), 0.02)
        self.assertAlmostEqual(r["telemetry"]["usd_with_usage"], 3.309304, delta=0.066)

    def test_a_pre_v5_log_is_not_comparable_and_the_bill_is_the_best_figure(self):
        rows = [{"t": sr._ts(START) + i, "route": "remote", "alias_kind": "default", "status": 200,
                 "ptok": 44_000, "outtok": 290, "cost_est": 0.0068} for i in range(100)]
        telemetry(self.dir / "t", rows)
        tel = sr.telemetry_rows(str(self.dir / "t"), sr._ts(START), sr._ts(END))
        r = sr.reconcile(self.rows, tel, sr._ts(START), sr._ts(END))
        self.assertFalse(r["comparable"] or r["ok"])
        self.assertIsNone(r["relative_error"])
        best = sr.best_figure(r["billing"], sr.bill_prices(self.rows), [])
        self.assertAlmostEqual(best["best_usd"], 3.309304, places=5)

    def test_the_unbilled_gap_is_priced_at_the_bills_effective_rates(self):
        gap = [{"t": sr._ts(END) + 10, "route": "remote", "alias_kind": "default", "status": 200,
                "ptok": 1_000_000, "outtok": 1_000}]
        best = sr.best_figure(sr.billing_totals(self.rows, sr._ts(START), sr._ts(END)),
                              sr.bill_prices(self.rows), gap)
        per_prompt = (262_519_512 * 3e-9 + 9_879_145 * 1.5e-7) / (262_519_512 + 9_879_145)
        self.assertAlmostEqual(best["gap_usd"], 1_000_000 * per_prompt + 1_000 * 6e-7, places=6)
        self.assertAlmostEqual(best["effective_prompt_usd_per_mtok"], per_prompt * 1e6, places=4)

    def test_post_replace_sends_an_operator_replace_with_the_admin_token_from_a_file(self):
        token = self.dir / "admin.token"
        token.write_text("adm-secret\n")
        seen = {}

        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def urlopen(req, timeout=0):
            seen.update(url=req.full_url, headers=dict(req.header_items()), body=json.loads(req.data))
            return Resp(b'{"ok": true}')
        with patch.object(sr.urllib.request, "urlopen", urlopen):
            sr.post_replace("http://127.0.0.1:8000", str(token), 3.31, "bill")
        self.assertEqual(seen["url"], "http://127.0.0.1:8000/gateway/spend/recover")
        self.assertEqual(seen["headers"]["X-admin-token"], "adm-secret")
        self.assertEqual(seen["body"], {"spent": 3.31, "replace": True, "source": "bill",
                                        "reason": "provider bill reconciliation"})


if __name__ == "__main__":
    unittest.main()
