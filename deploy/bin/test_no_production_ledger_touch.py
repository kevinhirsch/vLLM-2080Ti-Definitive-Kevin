#!/usr/bin/env python3
"""RATCHET (R2 v6): no shim test or tool module may touch the production spend ledger.

2026-09-25 12:47:57: a test run imported keepalive-shim.py without SHIM_SPEND_FILE and
rewrote the LIVE gateway-spend.json for ~7 minutes (the running gateway wrote its own state
back at 12:55:22). 0005 made an imported shim use a private ledger; this ratchet makes the
property executable for EVERY module in this directory, present and future:

  * every other test_*.py here is imported AND run, and every import-safe tool module here
    (spend_reconcile.py, model-router.py) is imported, in a subprocess with SHIM_SPEND_FILE unset, under a sys.addaudithook
    that records any open / os.rename (covers os.replace) / os.chmod / os.remove / os.mkdir /
    os.utime / os.truncate / shutil event whose path is the production ledger, its clients
    file, or anything beside them that starts with those paths (temp, corrupt copies). fsync
    takes a descriptor, so it can only follow an `open` of that path -- which is caught.
  * the subprocess gets its own TMPDIR; after it exits that directory must be empty, i.e.
    every temp directory an imported shim or a test created was managed and cleaned up.

The production paths are read from the shim's own source (its default SHIM_SPEND_FILE /
SHIM_SPEND_CLIENTS_FILE), so the ratchet follows the code, not a copy.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIM = HERE / "keepalive-shim.py"
SELF = Path(__file__).name

PROBE = r'''
import json, os, sys, unittest, importlib.util
prod = json.loads(os.environ["RATCHET_PROD_PATHS"])
hits = []
EVENTS = {"open", "os.rename", "os.chmod", "os.remove", "os.mkdir", "os.utime", "os.truncate",
          "os.link", "os.symlink", "shutil.copyfile", "shutil.copymode", "shutil.move", "shutil.rmtree"}
def hook(event, args):
    if event not in EVENTS:
        return
    for a in args:
        if isinstance(a, (str, bytes)) or hasattr(a, "__fspath__"):
            try:
                p = os.fsdecode(a)
            except Exception:
                continue
            if any(p.startswith(x) for x in prod):
                hits.append([event, p])
sys.addaudithook(hook)
sys.path.insert(0, os.environ["RATCHET_DIR"])
mods = json.loads(os.environ["RATCHET_MODULES"])
tools = json.loads(os.environ["RATCHET_TOOLS"])
for t in tools:   # tool modules: import only (their main() needs real inputs)
    spec = importlib.util.spec_from_file_location("ratchet_tool_" + t.replace("-", "_"), os.path.join(os.environ["RATCHET_DIR"], t + ".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
suite = unittest.defaultTestLoader.loadTestsFromNames(mods)
result = unittest.TextTestRunner(stream=open(os.devnull, "w"), verbosity=0).run(suite)
print("RATCHET " + json.dumps({"hits": hits, "ran": result.testsRun,
                               "failures": len(result.failures) + len(result.errors)}))
'''


def production_paths():
    src = SHIM.read_text()
    paths = []
    for var in ("SHIM_SPEND_FILE", "SHIM_SPEND_CLIENTS_FILE"):
        m = re.search(var + r'",\s*\n?\s*"([^"]+)"', src)
        assert m, f"cannot find the default for {var} in keepalive-shim.py"
        paths.append(m.group(1))
    return paths


def modules_in_scope():
    tests = sorted(p.stem for p in HERE.glob("test_*.py") if p.name != SELF)
    # Every import-safe (__main__-guarded) tool module beside the shim is imported too.
    tools = sorted(p.stem for p in HERE.glob("*.py")
                   if not p.name.startswith("test_") and p.name != SHIM.name
                   and 'if __name__ == "__main__"' in p.read_text(errors="replace"))
    return tests, tools


class ProductionLedgerRatchet(unittest.TestCase):
    def test_no_module_touches_the_production_ledger_and_temp_dirs_are_managed(self):
        tests, tools = modules_in_scope()
        self.assertIn("test_gateway_spend_authority", tests)
        self.assertIn("spend_reconcile", tools)
        with tempfile.TemporaryDirectory(prefix="ratchet-tmp-") as tmp:
            env = {k: v for k, v in os.environ.items() if not k.startswith("SHIM_SPEND")}
            env.update(TMPDIR=tmp, RATCHET_DIR=str(HERE), RATCHET_MODULES=json.dumps(tests),
                       RATCHET_TOOLS=json.dumps(tools), RATCHET_PROD_PATHS=json.dumps(production_paths()))
            out = subprocess.run([sys.executable, "-c", PROBE], cwd=str(HERE), env=env,
                                 capture_output=True, text=True, timeout=600)
            line = next((l for l in out.stdout.splitlines() if l.startswith("RATCHET ")), None)
            self.assertIsNotNone(line, out.stderr[-2000:])
            got = json.loads(line[len("RATCHET "):])
            self.assertEqual(got["hits"], [], "a shim test/tool touched the production ledger")
            self.assertGreater(got["ran"], 100)
            self.assertEqual(got["failures"], 0, out.stderr[-3000:])
            self.assertEqual(sorted(os.listdir(tmp)), [], "unmanaged temp dirs were left behind")

    def test_the_ratchet_itself_catches_a_touch(self):
        """Self-test on a stand-in path (never the real ledger): write+replace+chmod are reported."""
        with tempfile.TemporaryDirectory(prefix="ratchet-self-") as tmp:
            fake = os.path.join(tmp, "gateway-spend.json")
            probe = PROBE.split("sys.addaudithook(hook)")[0] + (
                "sys.addaudithook(hook)\n"
                "open(prod[0] + '.tmp', 'w').close()\n"
                "os.replace(prod[0] + '.tmp', prod[0])\n"
                "os.chmod(prod[0], 0o600)\n"
                "print('RATCHET ' + json.dumps({'hits': hits}))\n")
            env = dict(os.environ, RATCHET_PROD_PATHS=json.dumps([fake]))
            out = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, timeout=60)
            line = next(l for l in out.stdout.splitlines() if l.startswith("RATCHET "))
            events = [h[0] for h in json.loads(line[8:])["hits"]]
            self.assertEqual(set(events), {"open", "os.rename", "os.chmod"})


if __name__ == "__main__":
    unittest.main()
