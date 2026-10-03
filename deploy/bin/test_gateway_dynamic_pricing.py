#!/usr/bin/env python3
"""Lane SL (2026-10-02): the gateway charges ACTUAL usage at DYNAMIC prices.

Kevin's DeepSeek export for 2026-10-02 billed $8.15; the gateway's ledger said $20.98. Root causes:
  1. failed calls (402 Insufficient Balance, 4xx/5xx) were charged their whole upfront hold;
  2. answered calls whose usage trailer never arrived were charged the whole hold, not their use;
  3. peak pricing ignored the provider's rule that Chinese public holidays are off-peak
     (2026-10-02 sits inside the Oct 1-7 National Day holiday), and prices were constants.

Run:  python3 -m unittest test_gateway_dynamic_pricing
"""
import atexit
import copy
import csv
import datetime
import importlib.util
import json
import os
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ.setdefault("SHIM_EXACT_TOKENS", "0")
_TMPDIR = tempfile.TemporaryDirectory(prefix="dyn-pricing-test-")
atexit.register(_TMPDIR.cleanup)
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("shim_dyn_pricing_test", os.environ.get(
    "SHIM_TEST_CANDIDATE", str(HERE / "keepalive-shim.py")))
shim = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shim)

spec2 = importlib.util.spec_from_file_location("spend_reconcile_dyn", HERE / "spend_reconcile.py")
recon = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(recon)
spec3 = importlib.util.spec_from_file_location("pricing_refresh_dyn", HERE / "remote_pricing_refresh.py")
refresh = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(refresh)

DAY0 = 1790000000.0

# Kevin's DeepSeek usage export for 2026-10-02, deepseek-flash, key "Pi Prod" -- NUMBERS ONLY
# (hour start epoch, requests, cache-hit tokens, cache-miss tokens, output tokens, provider cost USD).
# Hours 02:00-11:00 MST are absent: the account was out of credit and every call got a 402.
EXPORT_2026_10_02 = [
    (1790924400, 1000, 36708221, 1975537, 410015, 0.652464213),
    (1790928000, 243, 5415424, 620745, 115307, 0.178542222),
    (1790967600, 4, 58112, 35441, 1328, 0.006287286),
    (1790971200, 400, 13249532, 1039073, 100056, 0.255643146),
    (1790974800, 4, 19840, 8530, 866, 0.00185862),
    (1790978400, 2, 2048, 3602, 173, 0.000650244),
    (1790982000, 3, 23680, 42702, 584, 0.00682674),
    (1790985600, 140, 4973312, 676691, 89883, 0.170353386),
    (1790989200, 790, 33212032, 3471680, 536256, 0.942141696),
    (1790992800, 1483, 43238396, 4689870, 871341, 1.356000288),
    (1790996400, 919, 29580672, 2765742, 400258, 0.743758116),
    (1791000000, 1690, 55048036, 5214394, 925128, 1.502380008),
    (1791003600, 1899, 57228268, 5228173, 854060, 1.468346754),
    (1791007200, 1010, 36591104, 3002314, 507949, 0.864889812),
]
EXPORT_TOTAL = 8.150142531


def utc(text):
    return datetime.datetime.fromisoformat(text).replace(tzinfo=datetime.timezone.utc).timestamp()


def ledger(cap=100.0):
    path = Path(tempfile.mkdtemp(dir=_TMPDIR.name)) / "gateway-spend.json"
    return shim.SpendLedger(str(path), cap=lambda: cap, clock=lambda: DAY0, process_token="p",
                            importer=lambda s, e: (0.0, 0, 0), enforce=lambda: True)


class TableTestCase(unittest.TestCase):
    """Runs with the committed table; tests that swap the file restore it."""

    def setUp(self):
        shim._pricing_table(force=True)
        self.addCleanup(lambda: (setattr(shim, "PRICING_FILE", str(HERE / "remote-pricing.json")),
                                 shim._PRICING_CACHE.update(table=None, mtime=None, checked=0.0),
                                 shim._pricing_table(force=True)))

    def use_table(self, table):
        path = Path(tempfile.mkdtemp(dir=_TMPDIR.name)) / "remote-pricing.json"
        path.write_text(json.dumps(table))
        shim.PRICING_FILE = str(path)
        shim._PRICING_CACHE.update(table=None, mtime=None, checked=0.0)
        shim._pricing_table(force=True)
        return path


class PricingTable(TableTestCase):
    def test_the_embedded_default_is_identical_to_the_committed_file(self):
        self.assertEqual(json.loads((HERE / "remote-pricing.json").read_text()), shim._PRICING_DEFAULT)

    def test_provenance_travels_with_the_prices(self):
        t = json.loads((HERE / "remote-pricing.json").read_text())
        self.assertEqual(t["source_url"], "https://api-docs.deepseek.com/quick_start/pricing")
        datetime.date.fromisoformat(t["fetched"])
        self.assertTrue(t["peak"]["holiday_source"].startswith("https://"))

    def test_price_is_a_function_of_model_token_type_and_time(self):
        off, peak = utc("2026-11-07T07:00"), utc("2026-11-06T07:00")      # Saturday / Friday, no holiday
        self.assertEqual([shim.remote_price("deepseek-flash", t, off) for t in ("cache_hit", "cache_miss", "output")],
                         [0.003, 0.15, 0.6])
        self.assertEqual([shim.remote_price("deepseek-flash", t, peak) for t in ("cache_hit", "cache_miss", "output")],
                         [0.006, 0.3, 1.2])
        self.assertEqual(shim.remote_price("deepseek-v4-pro", "output", off), 1.98)
        self.assertEqual(shim.remote_price("deepseek-v4-pro", "output", peak), 3.96)

    def test_legacy_names_are_billed_at_flash_and_unknown_models_have_no_price(self):
        for alias in ("deepseek-v4-flash", "DeepSeek-V4-Flash", "deepseek-v4-flash-vision-exp"):
            self.assertEqual(shim.remote_price(alias, "cache_miss", utc("2026-11-07T07:00")), 0.15)
        self.assertIsNone(shim.remote_price("some-other-model", "output"))
        with self.assertRaises(ValueError):
            shim.remote_price("deepseek-flash", "reasoning")

    def test_peak_windows_are_utc_weekdays_and_half_open(self):
        fri = "2026-11-06"      # an ordinary Friday
        cases = {"00:59": False, "01:00": True, "03:59": True, "04:00": False, "05:59": False,
                 "06:00": True, "09:59": True, "10:00": False, "23:00": False}
        for hhmm, expected in cases.items():
            self.assertEqual(shim.is_peak(utc(f"{fri}T{hhmm}")), expected, hhmm)
        self.assertTrue(shim.is_peak(utc("2026-11-02T07:00")))            # Monday
        self.assertFalse(shim.is_peak(utc("2026-11-07T07:00")))           # Saturday
        self.assertFalse(shim.is_peak(utc("2026-11-08T07:00")))           # Sunday

    def test_chinese_public_holidays_are_off_peak_all_day(self):
        # Friday 2026-10-02 07:00 UTC is peak by the clock; it is National Day, so it is not.
        self.assertFalse(shim.is_peak(utc("2026-10-02T07:00")))
        self.assertFalse(shim.is_peak(utc("2026-10-05T02:00")))           # Monday inside Oct 1-7
        self.assertTrue(shim.is_peak(utc("2026-10-08T02:00")))            # Thursday after it
        self.assertFalse(shim.is_peak(utc("2026-09-25T02:00")))           # Mid-Autumn Friday
        self.assertTrue(shim.is_peak(utc("2026-09-24T02:00")))

    def test_past_the_holiday_calendar_a_weekday_is_priced_as_peak_and_the_status_says_stale(self):
        self.assertTrue(shim.is_peak(utc("2027-02-17T02:00")))           # unknown holiday -> the dearer side
        status = shim.pricing_status(utc("2027-01-05T00:00"))
        self.assertTrue(status["stale"])
        self.assertEqual(status["stale_reason"], "holiday calendar ended")
        self.assertFalse(shim.pricing_status(utc("2026-10-03T00:00"))["stale"])

    def test_every_hour_of_kevins_export_matches_the_providers_cost(self):
        """The table, applied per hour to the billed tokens, reproduces the provider's cost."""
        for start, _n, hit, miss, out, cost in EXPORT_2026_10_02:
            mid = start + 1800
            got = shim._remote_cost_actual("deepseek-flash", hit, miss, out, when=mid)
            self.assertAlmostEqual(got, cost, places=6, msg=time.strftime("%H", time.gmtime(start)))
        total = sum(shim._remote_cost_actual("deepseek-flash", h, m, o, when=s + 1800)
                    for s, _n, h, m, o, _c in EXPORT_2026_10_02)
        self.assertAlmostEqual(total, EXPORT_TOTAL, places=5)

    def test_without_the_holiday_calendar_the_first_hours_would_be_billed_double(self):
        """Proves the holiday rule is load-bearing: 00:00-01:59 MST is 07:00-08:59 UTC on a Friday."""
        t = json.loads((HERE / "remote-pricing.json").read_text())
        t["peak"]["off_peak_dates"] = []
        self.use_table(t)
        first = EXPORT_2026_10_02[0]
        got = shim._remote_cost_actual("deepseek-flash", first[2], first[3], first[4], when=first[0] + 1800)
        self.assertAlmostEqual(got, first[5] * 2, places=6)

    def test_the_file_is_hot_reloaded_and_a_bad_file_keeps_the_last_good_one(self):
        t = json.loads((HERE / "remote-pricing.json").read_text())
        path = self.use_table(t)
        self.assertEqual(shim.remote_price("deepseek-flash", "output", utc("2026-11-07T07:00")), 0.6)
        t2 = copy.deepcopy(t)
        t2["models"]["deepseek-flash"]["off_peak"]["output"] = 0.7
        path.write_text(json.dumps(t2))
        os.utime(path, ns=(time.time_ns() + 10**9, time.time_ns() + 10**9))
        shim._pricing_table(force=True)
        self.assertEqual(shim.remote_price("deepseek-flash", "output", utc("2026-11-07T07:00")), 0.7)
        path.write_text("{ not json")
        os.utime(path, ns=(time.time_ns() + 2 * 10**9, time.time_ns() + 2 * 10**9))
        shim._pricing_table(force=True)
        self.assertEqual(shim.remote_price("deepseek-flash", "output", utc("2026-11-07T07:00")), 0.7)
        self.assertIsNotNone(shim.pricing_status()["error"])

    def test_a_missing_file_falls_back_to_the_embedded_table(self):
        shim.PRICING_FILE = str(Path(_TMPDIR.name) / "does-not-exist.json")
        shim._PRICING_CACHE.update(table=None, mtime=None, checked=0.0)
        shim._pricing_table(force=True)
        self.assertEqual(shim.pricing_status()["source"], "embedded")
        self.assertEqual(shim.remote_price("deepseek-flash", "cache_hit", utc("2026-11-07T07:00")), 0.003)

    def test_an_invalid_table_is_rejected(self):
        t = json.loads((HERE / "remote-pricing.json").read_text())
        for mutate in (lambda x: x.update(schema=2),
                       lambda x: x["models"]["deepseek-flash"]["peak"].update(output=-1),
                       lambda x: x["peak"].update(windows=[[5, 3]]),
                       lambda x: x["peak"].update(weekdays=[9]),
                       lambda x: x.update(models={})):
            bad = copy.deepcopy(t)
            mutate(bad)
            with self.assertRaises(ValueError):
                shim._pricing_validate(bad)

    def test_an_operator_override_row_beats_the_table(self):
        row = json.dumps({"deepseek-flash": {"cache_hit": 1, "cache_miss": 2, "output": 3}})
        with patch.object(shim, "REMOTE_PRICES_JSON", row):
            self.assertEqual(shim._remote_prices("deepseek-flash", peak=False), (1.0, 2.0, 3.0))
            self.assertEqual(shim._remote_prices("deepseek-flash", peak=True), (2.0, 4.0, 6.0))
        self.assertEqual(shim._remote_prices("deepseek-flash", peak=False), (0.003, 0.15, 0.6))


class Settlement(unittest.TestCase):
    """_spend_settle on the request record handle_completions' finally passes it."""

    SAT = utc("2026-11-07T07:00")        # off-peak (weekend)
    FRI = utc("2026-11-06T07:00")        # peak

    def setUp(self):
        shim._pricing_table(force=True)
        self.led = ledger()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for n, v in dict(_SPEND_LEDGER=self.led, REMOTE_PRICES_JSON="", REMOTE_MODEL="deepseek-flash",
                         _CACHE_RATIO={"hit": 0.0, "miss": 0.0, "n": 0}).items():
            self.stack.enter_context(patch.object(shim, n, v))

    def settle(self, hold=0.5, **info):
        self.led.hold("k", hold)
        base = {"spend_key": "k", "spend_held": True, "route": "remote", "cost_policy": "metered",
                "remote_sent": True, "remote_model": "deepseek-flash", "remote_sent_at": self.SAT}
        shim._spend_settle({**base, **info}, None)
        snap = self.led.snapshot()
        self.assertEqual(snap["in_flight"], 0, "the hold must always be released")
        return round(snap["spent"], 9)

    def test_usage_is_settled_per_token_type(self):
        got = self.settle(remote_cache_hit=2_000_000, remote_cache_miss=100_000, outtok=50_000, remote_outcome="ok")
        self.assertAlmostEqual(got, 2 * 0.003 + 0.1 * 0.15 + 0.05 * 0.6, places=9)

    def test_cache_hit_tokens_cost_one_fiftieth_of_cache_miss_tokens(self):
        hit = self.settle(remote_cache_hit=1_000_000, remote_cache_miss=0, outtok=0, remote_outcome="ok")
        self.assertAlmostEqual(hit, 0.003, places=9)
        self.led2 = ledger()
        with patch.object(shim, "_SPEND_LEDGER", self.led2):
            self.led2.hold("k", 0.5)
            shim._spend_settle({"spend_key": "k", "spend_held": True, "route": "remote", "cost_policy": "metered",
                                "remote_sent": True, "remote_sent_at": self.SAT, "remote_cache_hit": 0,
                                "remote_cache_miss": 1_000_000, "outtok": 0, "remote_outcome": "ok"}, None)
        self.assertAlmostEqual(self.led2.snapshot()["spent"], 0.15, places=9)

    def test_peak_costs_exactly_double_off_peak(self):
        use = dict(remote_cache_hit=500_000, remote_cache_miss=40_000, outtok=10_000, remote_outcome="ok")
        off = self.settle(**use)
        self.led = ledger()
        with patch.object(shim, "_SPEND_LEDGER", self.led):
            peak = self.settle(remote_sent_at=self.FRI, **use)
        self.assertAlmostEqual(peak, 2 * off, places=9)

    def test_the_price_is_read_at_the_time_the_call_was_sent(self):
        """A call sent at 09:59:59 UTC is peak even if it is settled after 10:00."""
        sent = utc("2026-11-06T09:59:59")
        self.assertAlmostEqual(self.settle(remote_sent_at=sent, remote_cache_hit=0, remote_cache_miss=1_000_000,
                                           outtok=0, remote_outcome="ok"), 0.3, places=9)

    def test_failed_calls_charge_zero_and_release_the_hold(self):
        for status in (402, 400, 401, 404, 429, 500, 502, 503):
            self.led = ledger()
            with patch.object(shim, "_SPEND_LEDGER", self.led):
                got = self.settle(remote_outcome="failed", remote_status=status, ptok=300_000, outtok_lb=0)
            self.assertEqual(got, 0.0, status)

    def test_a_relay_that_died_before_any_completion_charges_zero(self):
        self.assertEqual(self.settle(remote_outcome="failed", remote_status=0), 0.0)     # connect error / timeout

    def test_a_held_request_that_was_never_sent_charges_zero(self):
        self.assertEqual(self.settle(remote_sent=False), 0.0)

    def test_provider_usage_wins_over_a_failure_flag(self):
        got = self.settle(remote_outcome="failed", remote_status=502, remote_cache_hit=10_000, remote_cache_miss=100, outtok=50)
        self.assertAlmostEqual(got, (10_000 * 0.003 + 100 * 0.15 + 50 * 0.6) / 1e6, places=9)

    def test_prompt_tokens_alone_are_priced_as_cache_misses(self):
        got = self.settle(ptok_exact=1_000, outtok=10, remote_outcome="ok")
        self.assertAlmostEqual(got, (1_000 * 0.15 + 10 * 0.6) / 1e6, places=9)

    def test_an_answer_with_no_usage_is_an_estimate_bounded_by_its_hold_not_the_hold(self):
        got = self.settle(hold=0.5, remote_outcome="ok", ptok=80_000, outtok_lb=40)
        self.assertAlmostEqual(got, (80_000 * 0.15 + 40 * 0.6) / 1e6, places=9)      # all-miss until calibrated
        self.assertLess(got, 0.5)
        self.led = ledger()
        with patch.object(shim, "_SPEND_LEDGER", self.led):
            self.assertEqual(self.settle(hold=0.001, remote_outcome="ok", ptok=80_000, outtok_lb=40), 0.001)

    def test_the_estimate_uses_the_recently_observed_cache_hit_share(self):
        for _ in range(60):
            shim._cache_ratio_note(90_000, 10_000)                                   # 90% hits, 60 calls
        self.assertAlmostEqual(shim._cache_hit_share(), 0.9, places=6)
        got = self.settle(remote_outcome="ok", ptok=100_000, outtok_lb=0)
        self.assertAlmostEqual(got, (90_000 * 0.003 + 10_000 * 0.15) / 1e6, places=9)

    def test_too_few_observations_assume_nothing(self):
        for _ in range(5):
            shim._cache_ratio_note(90_000, 10_000)
        self.assertEqual(shim._cache_hit_share(), 0.0)

    def test_only_when_the_outcome_is_unknown_is_the_whole_hold_charged(self):
        self.assertEqual(self.settle(hold=0.5, ptok=80_000, outtok_lb=40), 0.5)       # handler torn down mid-call

    def test_a_free_endpoint_is_never_charged(self):
        self.assertEqual(self.settle(cost_policy="free", outtok=10**6), 0.0)


class HoldThenSettle(unittest.TestCase):
    """The upfront upper-bound hold keeps in-flight calls inside the cap; settlement trues it up."""

    def setUp(self):
        shim._pricing_table(force=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(shim, "REMOTE_PRICES_JSON", ""))

    def test_the_hold_is_an_upper_bound_priced_from_the_table_at_the_dearer_period(self):
        # 100k prompt tokens (all cache-miss, peak) + 8k output (peak)
        self.assertAlmostEqual(shim._spend_hold_estimate(100_000, 8_000, "deepseek-flash"),
                               100_000 * 0.3 / 1e6 + 8_000 * 1.2 / 1e6, places=6)
        self.assertAlmostEqual(shim._spend_hold_estimate(100_000, 8_000, "deepseek-flash"), 0.0396, places=6)
        # the dearer model's hold is dearer: 100k * 1.32 + 8k * 3.96 per Mtok
        self.assertAlmostEqual(shim._spend_hold_estimate(100_000, 8_000, "deepseek-v4-pro"),
                               100_000 * 1.32 / 1e6 + 8_000 * 3.96 / 1e6, places=6)

    def test_a_hold_covers_any_usage_the_call_can_report(self):
        hold = shim._spend_hold_estimate(50_000, 4_000, "deepseek-flash")
        worst = shim._remote_cost_actual("deepseek-flash", 0, 50_000, 4_000, peak=True)
        self.assertGreaterEqual(hold, worst - 1e-9)

    def test_concurrent_calls_can_never_overshoot_the_cap_and_settling_frees_headroom(self):
        cap = 1.0
        led = ledger(cap=cap)
        amount = shim._spend_hold_estimate(200_000, 8_000, "deepseek-flash")           # ~$0.0696
        keys, refused = [], 0
        for i in range(40):                                                            # 40 concurrent calls
            ok, _why = led.hold(f"c{i}", amount)
            if ok:
                keys.append(f"c{i}")
            else:
                refused += 1
            snap = led.snapshot()
            self.assertLessEqual(snap["spent"] + snap["held"] + snap["reserved"], cap + 1e-6)
        self.assertGreater(refused, 0, "the cap must bind while calls are in flight")
        self.assertEqual(len(keys), int(cap // amount))
        # each call finishes far cheaper than its hold (mostly cache hits): headroom comes back
        actual = shim._remote_cost_actual("deepseek-flash", 190_000, 10_000, 600, peak=False)
        for k in keys:
            led.settle(k, actual)
            snap = led.snapshot()
            self.assertLessEqual(snap["spent"] + snap["held"] + snap["reserved"], cap + 1e-6)
        snap = led.snapshot()
        self.assertEqual(snap["in_flight"], 0)
        self.assertAlmostEqual(snap["spent"], actual * len(keys), places=5)
        ok, _ = led.hold("again", amount)
        self.assertTrue(ok, "released headroom must be spendable again")

    def test_failures_return_their_whole_hold_to_the_day(self):
        led = ledger(cap=0.2)
        with patch.object(shim, "_SPEND_LEDGER", led):
            for i in range(10):
                ok, _ = led.hold(f"f{i}", 0.05)
                self.assertTrue(ok or i >= 4)
                if ok:
                    shim._spend_settle({"spend_key": f"f{i}", "spend_held": True, "route": "remote",
                                        "cost_policy": "metered", "remote_sent": True, "remote_outcome": "failed",
                                        "remote_status": 402}, None)
        snap = led.snapshot()
        self.assertEqual((snap["spent"], snap["in_flight"]), (0.0, 0))


class Replay(unittest.TestCase):
    """Today's export, replayed through the gateway's own hold -> settle path."""

    def test_replaying_kevins_export_through_settlement_costs_about_8_15(self):
        shim._pricing_table(force=True)
        led = ledger(cap=1000.0)
        per_hour = 12                       # 12 synthetic calls per billed hour, tokens split evenly
        with ExitStack() as st:
            for n, v in dict(_SPEND_LEDGER=led, REMOTE_PRICES_JSON="", REMOTE_MODEL="deepseek-flash").items():
                st.enter_context(patch.object(shim, n, v))
            for start, _n, hit, miss, out, _cost in EXPORT_2026_10_02:
                for i in range(per_hour):
                    key = f"{start}-{i}"
                    led.hold(key, shim._spend_hold_estimate(100_000, 8_000, "deepseek-flash"))
                    share = lambda total: total // per_hour + (1 if i < total % per_hour else 0)  # noqa: E731
                    shim._spend_settle({"spend_key": key, "spend_held": True, "route": "remote",
                                        "cost_policy": "metered", "remote_sent": True, "remote_outcome": "ok",
                                        "remote_model": "deepseek-flash", "remote_sent_at": start + 60 * i,
                                        "remote_cache_hit": share(hit), "remote_cache_miss": share(miss),
                                        "outtok": share(out)}, None)
            # the 402 outage: 02:00-11:00 MST, ~140 failed calls the old ledger charged ~$3.40 for
            for i in range(140):
                key = f"402-{i}"
                led.hold(key, shim._spend_hold_estimate(100_000, 8_000, "deepseek-flash"))
                shim._spend_settle({"spend_key": key, "spend_held": True, "route": "remote",
                                    "cost_policy": "metered", "remote_sent": True, "remote_outcome": "failed",
                                    "remote_status": 402, "remote_sent_at": 1790935200 + i * 60}, None)
        snap = led.snapshot()
        self.assertEqual(snap["in_flight"], 0)
        self.assertAlmostEqual(snap["spent"], EXPORT_TOTAL, delta=EXPORT_TOTAL * 0.005)

    def test_the_old_ledger_would_have_charged_the_failures_their_whole_holds(self):
        hold = shim._spend_hold_estimate(100_000, 8_000, "deepseek-flash")
        self.assertGreater(140 * hold, 3.0)          # ~$3.40 of the 2.6x over-count


class StreamUsageTrailer(unittest.TestCase):
    def test_a_line_split_across_chunks_is_seen_whole(self):
        line = b'data: {"usage":{"completion_tokens":7,"prompt_cache_hit_tokens":90,"prompt_cache_miss_tokens":10}}\n\n'
        tail, seen = b"", b""
        for chunk in (line[:20], line[20:61], line[61:]):
            complete, tail = shim._sse_split_tail(tail, chunk)
            seen += complete
        self.assertEqual(seen, line)
        self.assertEqual(tail, b"")

    def test_an_unterminated_final_line_stays_in_the_tail(self):
        complete, tail = shim._sse_split_tail(b"", b'data: {"a":1}\n\ndata: {"usage"')
        self.assertEqual(complete, b'data: {"a":1}\n\n')
        self.assertEqual(tail, b'data: {"usage"')

    def test_the_tail_is_bounded(self):
        _, tail = shim._sse_split_tail(b"", b"x" * 5000, cap=1000)
        self.assertEqual(len(tail), 1000)


class Seam(unittest.IsolatedAsyncioTestCase):
    """handle_completions end to end with a scripted provider."""

    def setUp(self):
        shim._pricing_table(force=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.led = ledger(cap=5.0)
        self.script = None

        async def relay(request, base, path, body, key, streaming, *a, **k):
            return self.script(request)
        for name, value in dict(_SPEND_LEDGER=self.led, _relay=relay, LOG_REQUESTS=0, FORCE_REMOTE=1,
                                FORCE_REMOTE_UNTIL_EPOCH=time.time() + 600,
                                REMOTE_BASE="https://api.deepseek.test", REMOTE_MODEL="deepseek-flash",
                                REMOTE_PRICES_JSON="", _est_tokens=lambda body: 60_000,
                                _CACHE_RATIO={"hit": 0.0, "miss": 0.0, "n": 0},
                                estimate_units=lambda *a, **k: 1, remote_ok=lambda: True,
                                record_event=lambda d, r, request, *a, **k: shim._active_set(request, route=d, reason=r),
                                _note_remote_response=AsyncMock(), _note_payload_outcome=lambda *a, **k: None,
                                _telemetry_note_request=lambda *a, **k: None).items():
            self.stack.enter_context(patch.object(shim, name, value))

    class Req:
        path = "/v1/chat/completions"
        method = "POST"
        remote = "127.0.0.1"

        def __init__(self, stream=False):
            self.headers = {"User-Agent": "dyn-test"}
            self.body = json.dumps({"model": "estate-remote", "messages": [{"role": "user", "content": "x"}],
                                    "max_tokens": 1000, "stream": stream}).encode()

        async def read(self):
            return self.body

    async def test_a_402_insufficient_balance_costs_nothing(self):
        self.script = lambda request: ("ok", shim.web.json_response(
            {"error": {"message": "Insufficient Balance"}}, status=402))
        resp = await shim.handle_completions(self.Req())
        snap = self.led.snapshot()
        self.assertEqual((resp.status, snap["spent"], snap["in_flight"]), (402, 0.0, 0))

    async def test_a_relay_failure_costs_nothing(self):
        self.script = lambda request: ("fail", (503, "upstream down", False))
        resp = await shim.handle_completions(self.Req(stream=True))
        snap = self.led.snapshot()
        self.assertEqual((resp.status, snap["spent"], snap["in_flight"]), (502, 0.0, 0))

    async def test_a_good_call_is_settled_at_its_reported_usage(self):
        def ok(request):
            shim._active_set(request, remote_cache_hit=55_000, remote_cache_miss=5_000, outtok=300)
            return "ok", shim.web.json_response({"ok": True})
        self.script = ok
        t0 = time.time()
        await shim.handle_completions(self.Req())
        expected = shim._remote_cost_actual("deepseek-flash", 55_000, 5_000, 300, when=t0)
        snap = self.led.snapshot()
        self.assertAlmostEqual(snap["spent"], expected, places=6)
        self.assertEqual(snap["in_flight"], 0)

    async def test_the_hold_is_visible_while_the_call_is_in_flight(self):
        seen = {}

        def ok(request):
            seen["held"] = self.led.snapshot()["in_flight"]
            shim._active_set(request, remote_cache_hit=1, remote_cache_miss=1, outtok=1)
            return "ok", shim.web.json_response({"ok": True})
        self.script = ok
        await shim.handle_completions(self.Req())
        self.assertEqual(seen["held"], 1)

    async def test_a_hung_up_client_is_billed_an_estimate_not_the_hold(self):
        t0 = time.time()
        self.script = lambda request: ("ok", shim.web.json_response({"gone": True}))
        await shim.handle_completions(self.Req(stream=True))
        snap = self.led.snapshot()
        hold = shim._spend_hold_estimate(60_000, 1000, "deepseek-flash")
        self.assertGreater(snap["spent"], 0.0)
        self.assertLess(snap["spent"], hold / 2)
        self.assertAlmostEqual(snap["spent"], shim._remote_cost_actual("deepseek-flash", 0, 60_000, 0, when=t0), places=6)

    async def test_streamed_requests_still_ask_for_the_usage_trailer(self):
        sent = []

        async def relay(request, base, path, body, key, streaming, *a, **k):
            sent.append(json.loads(body))
            shim._active_set(request, remote_cache_hit=1, remote_cache_miss=1, outtok=1)
            return "ok", shim.web.json_response({"ok": True})
        with patch.object(shim, "_relay", relay):
            await shim.handle_completions(self.Req(stream=True))
        self.assertEqual(sent[0]["stream_options"], {"include_usage": True})


class TelemetryRows(unittest.TestCase):
    def test_failed_rows_are_not_counted_as_remote_spend_by_the_importer(self):
        d = Path(tempfile.mkdtemp(dir=_TMPDIR.name))
        t = 1790000100.0
        day = time.strftime("%Y%m%d", time.gmtime(t))
        rows = [
            {"t": t, "route": "remote", "remote_sent": True, "status": 200, "cost_est": 0.25, "cost_basis": "actual"},
            {"t": t, "route": "remote", "remote_sent": True, "status": 402, "remote_outcome": "failed",
             "cost_est": 0.0, "cost_basis": "failed"},
        ]
        (d / f"requests-{day}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        self.assertEqual(shim.telemetry_remote_cost(t - 10, t + 10, str(d)), (0.25, 1, 0))


class ReconciliationFact(unittest.TestCase):
    def rows(self):
        out = []
        for start, n, hit, miss, out_t, cost in EXPORT_2026_10_02[:2]:
            for ty, amt, price in (("input_cache_hit_tokens", hit, 0.000000003),
                                   ("input_cache_miss_tokens", miss, 0.00000015), ("output_tokens", out_t, 0.0000006)):
                out.append({"start_time_iso": datetime.datetime.fromtimestamp(start).astimezone().isoformat(),
                            "type": ty, "price": str(price), "amount": str(amt)})
        return out

    def test_hourly_drift_names_the_hours_and_the_failed_calls_that_were_charged(self):
        start = EXPORT_2026_10_02[0][0]
        tel = [{"t": start + 10, "route": "remote", "remote_sent": True, "status": 200, "charged_usd": 2.0},
               {"t": start + 3700, "route": "remote", "remote_sent": True, "status": 200, "charged_usd": 0.18},
               {"t": start + 3800, "route": "remote", "remote_sent": True, "status": 402,
                "remote_outcome": "failed", "charged_usd": 0.9},
               {"t": start + 7300, "route": "remote", "remote_sent": True, "status": 402, "charged_usd": 1.5}]
        d = recon.hourly_drift(self.rows(), None, tel, start, start + 3 * 3600)
        h = d["hours"]
        self.assertEqual([x["flag"] for x in h], ["drift", "drift", "drift"])
        self.assertAlmostEqual(h[0]["provider_usd"], 0.652464, places=5)
        self.assertEqual(h[1]["failed_calls"], 1)
        self.assertEqual(h[2]["provider_usd"], 0.0)                 # an hour the provider billed nothing
        self.assertAlmostEqual(d["failed_calls_charged_usd"], 2.4, places=6)
        fact = recon.drift_fact(d, ("a", "b"))
        self.assertEqual(fact["verdict"], "drift")
        self.assertIn("cap is unchanged", fact["note"])

    def test_an_honest_ledger_has_no_drift(self):
        start = EXPORT_2026_10_02[0][0]
        tel = [{"t": start + 10, "route": "remote", "remote_sent": True, "status": 200, "charged_usd": 0.652464},
               {"t": start + 3610, "route": "remote", "remote_sent": True, "status": 200, "charged_usd": 0.1785}]
        d = recon.hourly_drift(self.rows(), None, tel, start, start + 2 * 3600)
        self.assertEqual(d["hours_drifted"], 0)
        self.assertEqual(recon.drift_fact(d, ("a", "b"))["verdict"], "ok")

    def test_the_fact_file_is_written_atomically_and_summarised_by_the_gateway(self):
        d = recon.hourly_drift(self.rows(), None, [], EXPORT_2026_10_02[0][0], EXPORT_2026_10_02[0][0] + 7200)
        path = Path(tempfile.mkdtemp(dir=_TMPDIR.name)) / "spend-reconciliation.json"
        recon.write_fact(str(path), recon.drift_fact(d, ("a", "b"), now=time.time()))
        with patch.object(shim, "RECONCILIATION_FACT_FILE", str(path)):
            got = shim._reconciliation_fact()
        self.assertEqual(got["verdict"], "drift")
        self.assertNotIn("hours", got)
        with patch.object(shim, "RECONCILIATION_FACT_FILE", str(path) + ".none"):
            self.assertIsNone(shim._reconciliation_fact())

    def test_replayed_failed_rows_are_not_counted_by_the_reconciler(self):
        rows = [{"route": "remote", "remote_sent": True, "remote_outcome": "failed", "cost_policy": "metered"},
                {"route": "remote", "remote_sent": True, "remote_outcome": "ok", "cost_policy": "metered"}]
        self.assertEqual([recon._counted(r, "deepseek") for r in rows], [False, True])


PAGE = """<html><body><table>
<tr><td>MODEL</td><td>deepseek-flash(1)</td><td>deepseek-v4-pro</td></tr>
<tr><td>PRICING(2)</td><td>1M INPUT TOKENS<br>(CACHE HIT)</td><td>OFF-PEAK</td><td>$0.003</td><td>$0.022</td></tr>
<tr><td>PEAK</td><td>$0.006</td><td>$0.044</td></tr>
<tr><td>1M INPUT TOKENS<br>(CACHE MISS)</td><td>OFF-PEAK</td><td>$0.15</td><td>$0.66</td></tr>
<tr><td>PEAK</td><td>$0.3</td><td>$1.32</td></tr>
<tr><td>1M OUTPUT TOKENS</td><td>OFF-PEAK</td><td>$0.6</td><td>$1.98</td></tr>
<tr><td>PEAK</td><td>$1.2</td><td>$3.96</td></tr></table>
<p>(1) Use deepseek-flash as the model name. The legacy names deepseek-v4-flash and deepseek-v4-flash-vision-exp are still accepted, but billed at the Flash price.</p>
<p>(2) Off-peak rates are half of the peak rates. Peak hours are 01:00 - 04:00 and 06:00 - 10:00 UTC, Monday through Friday, excluding Chinese public holidays. All other hours are off-peak.</p>
</body></html>"""


class RefreshTool(unittest.TestCase):
    def table(self):
        p = Path(tempfile.mkdtemp(dir=_TMPDIR.name)) / "remote-pricing.json"
        p.write_text((HERE / "remote-pricing.json").read_text())
        return p

    def test_the_committed_table_matches_the_provider_page(self):
        r = refresh.refresh(str(self.table()), PAGE, refresh.DEFAULT_URL, datetime.date(2026, 10, 2))
        self.assertEqual((r["drift"], r["changes"]), (False, []))

    def test_a_price_change_is_reported_as_drift_and_only_applied_on_request(self):
        path = self.table()
        page = PAGE.replace("<td>$0.15</td><td>$0.66</td>", "<td>$0.2</td><td>$0.66</td>").replace(
            "<td>$0.3</td><td>$1.32</td>", "<td>$0.4</td><td>$1.32</td>")
        r = refresh.refresh(str(path), page, refresh.DEFAULT_URL, datetime.date(2026, 10, 9))
        self.assertTrue(r["drift"])
        self.assertIn("deepseek-flash.off_peak.cache_miss: 0.15 -> 0.2 $/Mtok", r["changes"])
        self.assertFalse(r["applied"])
        self.assertEqual(json.loads(path.read_text())["models"]["deepseek-flash"]["off_peak"]["cache_miss"], 0.15)
        r = refresh.refresh(str(path), page, refresh.DEFAULT_URL, datetime.date(2026, 10, 9), apply=True)
        self.assertTrue(r["applied"])
        new = json.loads(path.read_text())
        self.assertEqual(new["models"]["deepseek-flash"]["off_peak"]["cache_miss"], 0.2)
        self.assertEqual(new["fetched"], "2026-10-09")
        self.assertTrue(Path(r["backup"]).exists())
        shim._pricing_validate(new)                                  # the gateway accepts what the tool wrote

    def test_a_clean_apply_restamps_the_verification_date(self):
        path = self.table()
        r = refresh.refresh(str(path), PAGE, refresh.DEFAULT_URL, datetime.date(2026, 10, 20), apply=True)
        self.assertEqual((r["drift"], r["applied"]), (False, True))
        self.assertEqual(json.loads(path.read_text())["fetched"], "2026-10-20")

    def test_a_changed_peak_window_is_drift(self):
        page = PAGE.replace("06:00 - 10:00", "06:00 - 11:00")
        r = refresh.refresh(str(self.table()), page, refresh.DEFAULT_URL, datetime.date(2026, 10, 2))
        self.assertIn("peak windows [[1, 4], [6, 10]] -> [[1, 4], [6, 11]]", r["changes"])

    def test_a_page_it_does_not_understand_is_never_applied(self):
        for broken in ("<html>maintenance</html>", PAGE.replace("Monday through Friday", "every day"),
                       PAGE.replace("<td>$1.2</td>", "<td>$1.5</td>")):
            with self.assertRaises(refresh.PageError):
                refresh.refresh(str(self.table()), broken, refresh.DEFAULT_URL, datetime.date(2026, 10, 2), apply=True)

    def test_calendar_warnings(self):
        t = json.loads((HERE / "remote-pricing.json").read_text())
        self.assertEqual(refresh.calendar_warnings(t, datetime.date(2026, 10, 2)), [])
        self.assertIn("ends", refresh.calendar_warnings(t, datetime.date(2026, 12, 1))[0])
        self.assertIn("ended", refresh.calendar_warnings(t, datetime.date(2027, 1, 5))[0])


if __name__ == "__main__":
    unittest.main()
