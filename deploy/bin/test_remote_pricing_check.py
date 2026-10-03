"""FX2: the weekly pricing check is REPORT-ONLY -- a drift becomes a fact + hand-off + vault note and the table is not rewritten."""
import copy, datetime, json, os, shutil, tempfile, unittest

import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("remote_pricing_check", os.path.join(HERE, "remote_pricing_check.py"))
chk = importlib.util.module_from_spec(spec); spec.loader.exec_module(chk)
REAL = chk._load_refresh()
TABLE = os.path.join(HERE, "remote-pricing.json")
TABLE_URL = json.load(open(TABLE))["source_url"]


def parsed_from(table, mutate=None):
    p = {"models": {n: {"off_peak": dict(m["off_peak"]), "peak": dict(m["peak"])} for n, m in table["models"].items()},
         "windows": table["peak"]["windows"], "weekdays": table["peak"]["weekdays"], "holidays_off_peak": True,
         "aliases": {a: n for n, m in table["models"].items() for a in m.get("aliases", [])}}
    if mutate:
        mutate(p)
    return p


class Check(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.file = os.path.join(self.d, "remote-pricing.json")
        shutil.copy(TABLE, self.file)
        self.table = json.load(open(self.file))
        self.note = os.path.join(self.d, "note.md")
        self.today = datetime.date.fromisoformat(self.table["fetched"]) + datetime.timedelta(days=7)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def go(self, mutate=None, **kw):
        parsed = parsed_from(self.table, mutate)
        chk._load_refresh = lambda: type("R", (), {"refresh": staticmethod(REAL.refresh), "PageError": REAL.PageError,
                                         "DEFAULT_URL": TABLE_URL, "fetch": staticmethod(lambda u: "<html/>")})
        orig = REAL.parse_page
        REAL.parse_page = lambda markup: parsed
        try:
            return chk.run(self.file, today=self.today, base=self.d, note=self.note, notify=False, **kw)
        finally:
            REAL.parse_page = orig

    def test_drift_is_reported_and_never_applied_even_with_restamp(self):
        before = open(self.file).read()
        name = next(iter(self.table["models"]))
        def bump(p): p["models"][name]["peak"]["output"] *= 1.5; p["models"][name]["off_peak"]["output"] *= 1.5
        r = self.go(bump, restamp=True)
        self.assertTrue(r["drift"]); self.assertTrue(r["changes"]); self.assertFalse(r["applied"])
        self.assertEqual(open(self.file).read(), before)                 # table untouched
        fact = json.load(open(f"{self.d}/facts/remote_pricing_table.json"))
        self.assertTrue(fact["drift"]); self.assertTrue(fact["report_only"])

    def test_clean_run_with_restamp_changes_only_the_fetched_date(self):
        r = self.go(None, restamp=True)
        self.assertFalse(r["drift"]); self.assertTrue(r["restamped"])
        after = json.load(open(self.file))
        self.assertEqual(after["fetched"], self.today.isoformat())
        a, b = copy.deepcopy(after), copy.deepcopy(self.table)
        a.pop("fetched"); b.pop("fetched")
        self.assertEqual(a, b)

    def test_clean_run_without_restamp_changes_nothing(self):
        before = open(self.file).read()
        r = self.go(None)
        self.assertFalse(r["drift"]); self.assertEqual(open(self.file).read(), before)

    def test_unparseable_page_is_an_error_fact_and_the_table_is_untouched(self):
        before = open(self.file).read()
        chk._load_refresh = lambda: type("R", (), {"refresh": staticmethod(lambda *a, **k: (_ for _ in ()).throw(REAL.PageError("shape"))),
                                         "PageError": REAL.PageError, "DEFAULT_URL": TABLE_URL, "fetch": staticmethod(lambda u: "x")})
        r = chk.run(self.file, today=self.today, base=self.d, note=self.note, notify=False, restamp=True)
        self.assertIn("PageError", r["error"]); self.assertEqual(open(self.file).read(), before)

    def test_note_records_problem_then_clean_run_clears_attention_keeps_history(self):
        chk.update_note(self.note, {"changes": ["x.peak.output: 1 -> 2 $/Mtok"], "warnings": []}, "2026-10-10")
        t = open(self.note).read()
        self.assertIn("ATTENTION", t); self.assertIn("- 2026-10-10: x.peak.output", t)
        chk.update_note(self.note, {"changes": [], "warnings": []}, "2026-10-17")
        t = open(self.note).read()
        self.assertIn("**Latest check (2026-10-17):** clean", t); self.assertIn("- 2026-10-10: x.peak.output", t)

    def test_fingerprint_is_stable_for_the_same_problem(self):
        a = {"changes": ["c"], "warnings": []}
        self.assertEqual(chk.fingerprint(a), chk.fingerprint(dict(a, checked_at=1)))
        self.assertNotEqual(chk.fingerprint(a), chk.fingerprint({"changes": ["d"], "warnings": []}))


if __name__ == "__main__":
    unittest.main()
