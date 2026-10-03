#!/usr/bin/env python3
"""Lane CFG: the gateway's typed config schema (gateway_config_schema.py) and its wiring in keepalive-shim.py.

The schema is only useful if it is COMPLETE and TRUE, so the first tests pin it to the code:
  * every env key keepalive-shim.py reads is in the schema (add the knob there when this fails), and every schema
    key is still read (no stale entries);
  * every literal default in the schema equals the code's default, every type matches the live global's type,
    the tunable flag matches the shim's _CFG table, and every gname names a real module global.
Then the behaviours, including two production regressions:
  * a dashboard-persisted "False" used to come back ON after a restart (readers test `not in ("0","false","")`
    without lowercasing) -- booleans are now canonicalised before the import-time reads;
  * an unparseable number in shim.env used to crash the gateway at import -- it is now ignored with a warning.
"""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault("SHIM_EXACT_TOKENS", "0")
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import gateway_config_schema as S  # noqa: E402

_SHIM = _HERE / "keepalive-shim.py"
_SPEC = importlib.util.spec_from_file_location("keepalive_shim_cfg", _SHIM)
shim = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(shim)

FAKE_SECRET = "sk-cfgtest-0123456789abcdef"


def _fresh_shim(env_overrides, attrs):
    """Import the shim in a clean subprocess with env_overrides; return {attr: repr(value)}."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("SHIM_")}
    with tempfile.TemporaryDirectory() as td:
        env.update(SHIM_EXACT_TOKENS="0", SHIM_ENV_FILE=os.path.join(td, "shim.env"),
                   SHIM_STATS_FILE=os.path.join(td, "stats.json"), SHIM_TELEMETRY_DIR=os.path.join(td, "tel"))
        env.update(env_overrides)
        code = ("import importlib.util,json,sys\n"
                "s=importlib.util.spec_from_file_location('k',%r);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
                "print('RESULT'+json.dumps({a: repr(getattr(m,a)) for a in %r}))\n") % (str(_SHIM), list(attrs))
        p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        raise AssertionError("shim import failed (rc=%d): %s" % (p.returncode, p.stderr[-2000:]))
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")][-1]
    return json.loads(line[6:]), p.stderr


class SchemaMatchesCode(unittest.TestCase):
    def setUp(self):
        self.hits = S.scan_code(str(_SHIM))

    def test_every_key_the_code_reads_is_in_the_schema(self):
        missing = sorted(k for k in self.hits if k not in S.SCHEMA)
        self.assertEqual(missing, [], "keepalive-shim.py reads env keys the config schema does not list. Add a _k(...) "
                         "line to gateway_config_schema.py for each (type, the code's default, unit, description): %s" % missing)

    def test_no_stale_schema_entries(self):
        stale = sorted(k for k in S.SCHEMA if k not in self.hits)
        self.assertEqual(stale, [], "schema lists keys the gateway no longer reads: %s" % stale)

    def test_defaults_match_the_code(self):
        bad = []
        for k, reads in self.hits.items():
            key = S.SCHEMA.get(k)
            if key is None or key.computed_default:
                continue
            for line, d in reads:
                if d in ("<required>", "<presence>"):
                    continue
                if d == "<expr>":
                    bad.append("%s L%d: code default is an expression; mark computed_default=True" % (k, line))
                elif d != key.default:
                    bad.append("%s L%d: code default %r != schema default %r" % (k, line, d, key.default))
        self.assertEqual(bad, [])

    def test_schema_defaults_are_valid_values(self):
        bad = []
        for k, key in S.SCHEMA.items():
            if key.default in (None, "") or key.computed_default:
                continue
            ok, problem, _ = S.check_value(key, key.default)
            if not ok or problem:
                bad.append((k, key.default, problem))
        self.assertEqual(bad, [])

    def test_tunable_flag_matches_cfg_table(self):
        tunable = {k for k, v in S.SCHEMA.items() if v.tunable}
        self.assertEqual(sorted(set(shim._CFG) - tunable), [], "in _CFG but not marked tunable")
        self.assertEqual(sorted(tunable - set(shim._CFG)), [], "marked tunable but not in _CFG")

    def test_gnames_are_real_globals_of_the_declared_type(self):
        g = vars(shim)
        problems = []
        for k, key in S.SCHEMA.items():
            if not key.gname:
                continue
            if key.gname not in g:
                problems.append("%s: no global %s" % (k, key.gname))
                continue
            x = g[key.gname]
            want = {"int": (int,), "float": (int, float), "bool": (bool, int), "budget": (int, type(None))}.get(key.type)
            if want and (not isinstance(x, want) or (key.type in ("int", "float") and isinstance(x, bool))):
                problems.append("%s: %s is %s, schema says %s" % (k, key.gname, type(x).__name__, key.type))
        self.assertEqual(problems, [])

    def test_cli_scan_is_clean(self):
        self.assertEqual(S._main(["scan", str(_SHIM)]), 0)


class ValueChecks(unittest.TestCase):
    def test_numbers(self):
        k = S.SCHEMA["SHIM_LOCAL_BUDGET"]
        self.assertEqual(S.check_value(k, "14")[:2], (True, None))
        self.assertFalse(S.check_value(k, "abc")[0])
        self.assertFalse(S.check_value(k, "")[0])
        self.assertFalse(S.check_value(k, "1.5")[0])
        ok, problem, healed = S.check_value(k, "6.0")
        self.assertTrue(ok)
        self.assertEqual(healed, "6")
        ok, problem, _ = S.check_value(k, "0")
        self.assertTrue(ok)
        self.assertIn("below minimum", problem)
        self.assertFalse(S.check_value(S.SCHEMA["SHIM_LOCAL_WAIT_SECS"], "nan")[0])

    def test_token_budget_accepts_auto(self):
        k = S.SCHEMA["SHIM_TOKEN_BUDGET"]
        for v in ("auto", "", "none", "live", "500000", "0"):
            self.assertTrue(S.check_value(k, v)[0], v)
        self.assertFalse(S.check_value(k, "lots")[0])

    def test_bools(self):
        k = S.SCHEMA["SHIM_BG_LOCAL_ONLY"]
        self.assertEqual(S.check_value(k, "False")[2], "0")
        self.assertEqual(S.check_value(k, "True")[2], "1")
        self.assertEqual(S.check_value(k, "off")[2], "0")
        self.assertIsNone(S.check_value(k, "1")[2])
        self.assertIn("not a boolean", S.check_value(k, "maybe")[1])

    def test_enum_url_hours_map_json(self):
        self.assertIn("not one of", S.check_value(S.SCHEMA["SHIM_FLOW_MODE"], "enforcing")[1])
        self.assertIsNone(S.check_value(S.SCHEMA["SHIM_FLOW_MODE"], "Shadow")[1])
        self.assertIn("URL", S.check_value(S.SCHEMA["SHIM_UPSTREAM"], "127.0.0.1:8001")[1])
        self.assertIsNone(S.check_value(S.SCHEMA["SHIM_PEAK_HOURS_UTC"], "1-4,6-10")[1])
        self.assertIsNotNone(S.check_value(S.SCHEMA["SHIM_PEAK_HOURS_UTC"], "1-4;6-10")[1])
        self.assertIsNotNone(S.check_value(S.SCHEMA["SHIM_FLOW_SHARES"], "kevin=1,halo")[1])
        self.assertIsNotNone(S.check_value(S.SCHEMA["SHIM_REMOTE_PRICES_JSON"], "{bad")[1])


class Linter(unittest.TestCase):
    def _lint(self, text):
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
            f.write(text)
        try:
            entries, err = S.parse_env_file(f.name)
            self.assertIsNone(err)
            return S.lint_entries(entries)
        finally:
            os.unlink(f.name)

    def test_duplicates_unknown_invalid_malformed(self):
        f = self._lint("# c\nSHIM_BG_XCLIENTS=cron\nSHIM_LOCAL_BUDGET=abc\nSHIM_BG_XCLIENTS=cron,batch\n"
                       "SHIM_TYPO_KNOB=1\ngarbage line\nSHIM_FORCE_REMOTE=0\nSHIM_FORCE_REMOTE=0\n")
        codes = {(x["code"], x["key"]) for x in f}
        self.assertIn(("duplicate", "SHIM_BG_XCLIENTS"), codes)
        self.assertIn(("duplicate", "SHIM_FORCE_REMOTE"), codes)
        self.assertIn(("invalid", "SHIM_LOCAL_BUDGET"), codes)
        self.assertIn(("unknown", "SHIM_TYPO_KNOB"), codes)
        self.assertIn(("malformed", "garbage line"), codes)
        dup = [x for x in f if x["key"] == "SHIM_BG_XCLIENTS"][0]
        self.assertIn("LAST (line 4)", dup["msg"])
        self.assertIn("values differ", dup["msg"])

    def test_secret_values_never_appear(self):
        f = self._lint("SHIM_REMOTE_KEY=%s\nSHIM_REMOTE_KEY=%sX\n" % (FAKE_SECRET, FAKE_SECRET))
        self.assertTrue(any(x["code"] == "duplicate" for x in f))
        self.assertNotIn(FAKE_SECRET, json.dumps(f))

    def test_quotes_and_export(self):
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
            f.write('SHIM_FLOW_MODE="shadow"\nexport SHIM_LOCAL_BUDGET=3\n')
        try:
            d = S.last_wins(S.parse_env_file(f.name)[0])
        finally:
            os.unlink(f.name)
        self.assertEqual(d["SHIM_FLOW_MODE"][1], "shadow")
        self.assertEqual(d["SHIM_LOCAL_BUDGET"][1], "3")

    def test_cli_exit_code(self):
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
            f.write("SHIM_LOCAL_BUDGET=abc\n")
        try:
            self.assertEqual(S._main(["lint", f.name, "--json"]), 1)
        finally:
            os.unlink(f.name)


class Sanitizer(unittest.TestCase):
    def test_sanitize_is_fail_safe_and_records_raw(self):
        env = {"SHIM_LOCAL_BUDGET": "abc", "SHIM_LOCAL_WAIT_SECS": "6.0", "SHIM_TINY_TOKENS": "1500.0",
               "SHIM_BG_LOCAL_ONLY": "False", "SHIM_FLOW_MODE": "bogus", "SHIM_UNHEARD_OF": "1",
               "SHIM_REMOTE_KEY": FAKE_SECRET}
        rep = S.sanitize_environ(env, env_file=None)
        self.assertNotIn("SHIM_LOCAL_BUDGET", env)            # dropped -> code default applies
        self.assertEqual(env["SHIM_TINY_TOKENS"], "1500")      # integral float healed for an int key
        self.assertEqual(env["SHIM_LOCAL_WAIT_SECS"], "6.0")   # float key untouched
        self.assertEqual(env["SHIM_BG_LOCAL_ONLY"], "0")       # canonical boolean
        self.assertEqual(env["SHIM_FLOW_MODE"], "bogus")       # warn only
        self.assertEqual(S.STARTUP["raw"]["SHIM_BG_LOCAL_ONLY"], "False")
        self.assertIn("SHIM_LOCAL_BUDGET", rep["dropped"])
        self.assertTrue(any(f["key"] == "SHIM_UNHEARD_OF" for f in rep["findings"]))
        self.assertNotIn(FAKE_SECRET, json.dumps(rep["findings"]))

    def test_sanitize_never_raises(self):
        class Boom(dict):
            def __contains__(self, k):
                raise RuntimeError("boom")
        rep = S.sanitize_environ(Boom(), env_file=None)
        self.assertIn("boom", rep["error"])


class ShimRegressions(unittest.TestCase):
    def test_dashboard_persisted_False_stays_off_after_restart(self):
        vals, _ = _fresh_shim({"SHIM_BG_LOCAL_ONLY": "False", "SHIM_THINK_GUARD": "False", "SHIM_EMPTY_RETRY": "False",
                               "SHIM_REP_GUARD": "False", "SHIM_BG_NO_THINK": "False", "SHIM_NONTHINK_PROFILE": "False",
                               "SHIM_HW_HISTORY": "false", "SHIM_EXACT_TOKENS": "False"},
                              ["BG_LOCAL_ONLY", "THINK_GUARD", "EMPTY_RETRY", "REP_GUARD", "BG_NO_THINK",
                               "NONTHINK_PROFILE", "HW_HISTORY", "EXACT_TOKENS"])
        self.assertEqual(set(vals.values()), {"False"}, vals)

    def test_invalid_number_no_longer_crashes_the_gateway(self):
        vals, err = _fresh_shim({"SHIM_LOCAL_BUDGET": "fourteen", "SHIM_OOM_BACKOFF_SECS": "120s",
                                 "SHIM_LOCAL_WAIT_SECS": "6.0", "SHIM_TINY_TOKENS": "1500.0"},
                                ["BUDGET", "OOM_BACKOFF", "LOCAL_WAIT", "TINY_TOKENS"])
        self.assertEqual(vals, {"BUDGET": "2", "OOM_BACKOFF": "120", "LOCAL_WAIT": "6.0", "TINY_TOKENS": "1500"})
        self.assertIn("CONFIG INVALID SHIM_LOCAL_BUDGET", err)

    def test_writer_persists_booleans_as_1_0(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "shim.env")
            with open(p, "w") as f:
                f.write("SHIM_BG_LOCAL_ONLY=True\n")
            with patch.object(shim, "SHIM_ENV_FILE", p), patch.object(shim, "BG_LOCAL_ONLY", False), \
                    patch.object(shim, "THINK_GUARD", True):
                shim._persist_config()
            d = S.last_wins(S.parse_env_file(p)[0])
        self.assertEqual(d["SHIM_BG_LOCAL_ONLY"][1], "0")
        self.assertEqual(d["SHIM_THINK_GUARD"][1], "1")

    def test_config_owner_compares_what_systemd_loaded(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "shim.env")
            with open(p, "w") as f:
                f.write("SHIM_BG_LOCAL_ONLY=True\nSHIM_THINK_GUARD=True\nSHIM_LOCAL_BUDGET=14\n")
            vals, _ = _fresh_shim({"SHIM_ENV_FILE": p, "SHIM_BG_LOCAL_ONLY": "True", "SHIM_THINK_GUARD": "True",
                                   "SHIM_LOCAL_BUDGET": "14"}, ["BUDGET"])
            code = ("import importlib.util,os\n"
                    "s=importlib.util.spec_from_file_location('k',%r);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
                    "print('OWNER', m._config_owner(), os.environ['SHIM_BG_LOCAL_ONLY'])\n") % str(_SHIM)
            env = {k: v for k, v in os.environ.items() if not k.startswith("SHIM_")}
            env.update(SHIM_EXACT_TOKENS="0", SHIM_ENV_FILE=p, SHIM_STATS_FILE=os.path.join(td, "s.json"),
                       SHIM_TELEMETRY_DIR=os.path.join(td, "t"), SHIM_BG_LOCAL_ONLY="True", SHIM_THINK_GUARD="True",
                       SHIM_LOCAL_BUDGET="14")
            out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120).stdout
        self.assertIn("OWNER True 1", out)


class EffectiveView(unittest.TestCase):
    def test_effective_masks_secrets_and_reports_sources(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "shim.env")
            with open(p, "w") as f:
                f.write("SHIM_REMOTE_KEY=%s\nSHIM_LOCAL_BUDGET=14\nSHIM_LOCAL_BUDGET=12\nSHIM_GHOST=1\n" % FAKE_SECRET)
            env = {"SHIM_REMOTE_KEY": FAKE_SECRET, "SHIM_LOCAL_BUDGET": "12"}
            g = {"REMOTE_KEY": FAKE_SECRET, "BUDGET": 13, "LOCAL_WAIT": 8.0, "TOKEN_BUDGET": None}
            out = S.effective(g, env, p)
        blob = json.dumps(out)
        self.assertNotIn(FAKE_SECRET, blob)
        rows = {r["key"]: r for r in out["keys"]}
        self.assertTrue(rows["SHIM_REMOTE_KEY"]["value"].startswith("<set"))
        self.assertTrue(rows["SHIM_LOCAL_BUDGET"]["source"].startswith("live-edit"))
        self.assertEqual(rows["SHIM_LOCAL_BUDGET"]["persisted"], "12")
        self.assertTrue(rows["SHIM_LOCAL_BUDGET"]["differs_from_default"])
        self.assertEqual(rows["SHIM_LOCAL_WAIT_SECS"]["source"], "default")
        self.assertFalse(rows["SHIM_LOCAL_WAIT_SECS"]["differs_from_default"])
        self.assertEqual(rows["SHIM_TOKEN_BUDGET"]["value"], "auto")
        self.assertEqual(out["summary"]["unknown_in_file"], ["SHIM_GHOST"])
        self.assertEqual(out["summary"]["duplicates_in_file"], ["SHIM_LOCAL_BUDGET"])

    def test_route_is_registered_and_handler_serves(self):
        import asyncio

        class Req:
            query = {"changed": "1"}
        resp = asyncio.run(shim.gateway_config_effective(Req()))
        self.assertEqual(resp.status, 200)
        body = json.loads(resp.body)
        self.assertEqual(body["schema_version"], S.SCHEMA_VERSION)
        self.assertIn("summary", body)
        src = _SHIM.read_text()
        self.assertIn('add_get("/gateway/config/effective", gateway_config_effective)', src)


if __name__ == "__main__":
    unittest.main()


class PublisherShipsTheSchema(unittest.TestCase):
    """gateway_safe_publish installs gateway_config_schema.py beside the live shim and undoes it on rollback."""

    def test_install_current_and_restore(self):
        from unittest.mock import patch
        import gateway_safe_publish as pub
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td)
            (d / "src").mkdir()
            (d / "live").mkdir()
            (d / "src" / "keepalive-shim.py").write_text("x")
            (d / "src" / "gateway_config_schema.py").write_text("schema v2\n")
            (d / "src" / "dash.html").write_text("page")
            with patch.object(pub, "SOURCE", d / "src" / "keepalive-shim.py"), \
                    patch.object(pub, "RUNTIME", d / "live" / "keepalive-shim.py"), \
                    patch.object(pub, "DASH_SOURCE", d / "src" / "dash.html"), \
                    patch.object(pub, "DASH_RUNTIME", d / "live" / "dash.html"):
                live = d / "live" / "gateway_config_schema.py"
                st = pub._install_dashboard()
                self.assertEqual(st["config_schema"], "installed")
                self.assertEqual(live.read_text(), "schema v2\n")
                self.assertEqual(pub._install_dashboard()["config_schema"], "current")
                pub._restore_dashboard(st)                         # first install rolled back -> removed
                self.assertFalse(live.exists())
                live.write_text("schema v1\n")
                st = pub._install_dashboard()
                pub._restore_dashboard(st)
                self.assertEqual(live.read_text(), "schema v1\n")
            # a source tree without the module ships nothing and reports it
            (d / "src" / "gateway_config_schema.py").unlink()
            with patch.object(pub, "SOURCE", d / "src" / "keepalive-shim.py"), \
                    patch.object(pub, "RUNTIME", d / "live" / "keepalive-shim.py"), \
                    patch.object(pub, "DASH_SOURCE", d / "src" / "dash.html"), \
                    patch.object(pub, "DASH_RUNTIME", d / "live" / "dash.html"):
                self.assertEqual(pub._install_dashboard()["config_schema"], "absent")
