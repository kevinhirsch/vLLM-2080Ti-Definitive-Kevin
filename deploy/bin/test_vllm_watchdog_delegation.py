"""LV (2026-10-03): vllm-watchdog.sh detects; the liveness authority (engine-actuator.py) decides and acts.

Runs the real script against fakes on PATH (curl, journalctl, sudo) and a fake actuator that records its argv."""
import json
import os
import stat
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "vllm-watchdog.sh")

FAKE_CURL = r'''#!/usr/bin/env python3
import os, sys
a = sys.argv[1:]
url = [x for x in a if x.startswith("http")][-1]
gen = os.environ.get("FAKE_GEN", "000")
if url.endswith("/metrics"):
    print("vllm:prompt_tokens_total 100\nvllm:generation_tokens_total 50")
elif "-w" in a and url.endswith("/v1/models"):
    print("200 0.01", end="")
elif "-w" in a:
    print(f"{gen} 20.0", end="")
elif url.endswith("/v1/models"):
    print('{"data":[{"id":"qwen-local"}]}')
'''

FAKE_ACTUATOR = r'''import json, os, sys
with open(os.environ["FAKE_ACT_LOG"], "a") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\n")
cmd = sys.argv[1]
if cmd == "tick":
    print(os.environ.get("FAKE_TICK", json.dumps({"state": "UP", "probe": True, "reason": "healthy"})))
elif cmd == "recover":
    mode = os.environ.get("FAKE_RECOVER", "acted")
    if mode == "crash":
        sys.exit(1)
    if mode == "acted":
        print(json.dumps({"acted": True, "id": "lv-1", "action": "recover", "steps": []}))
    else:
        print(json.dumps({"acted": False, "refused": "2 automatic actions in the last hour (max 2)"}))
        sys.exit(3)
'''


class WatchdogDelegation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = self.tmp.name
        self.bin = os.path.join(d, "bin")
        os.makedirs(self.bin)
        self._exe("curl", FAKE_CURL)
        self._exe("journalctl", "#!/bin/sh\nexit 0\n")
        self._exe("sudo", "#!/bin/sh\necho \"$@\" >> \"$FAKE_SUDO_LOG\"\nexit 0\n")
        self.act = os.path.join(d, "engine-actuator.py")
        open(self.act, "w").write(FAKE_ACTUATOR)
        self.state = os.path.join(d, "watchdog-state.json")
        self.log = os.path.join(d, "watchdog.log")
        self.actlog = os.path.join(d, "act.log")
        self.sudolog = os.path.join(d, "sudo.log")
        json.dump({"consecutive_failures": 4, "last_restart_epoch": 0, "restart_timestamps": [], "last_gen": 50, "skipped_ticks": 0},
                  open(self.state, "w"))

    def _exe(self, name, body):
        p = os.path.join(self.bin, name)
        open(p, "w").write(body)
        os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)

    def run_wd(self, **env):
        e = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}", WATCHDOG_STATE_FILE=self.state, WATCHDOG_ACTION_LOG=self.log,
                 WATCHDOG_ACTUATOR=self.act, FAKE_ACT_LOG=self.actlog, FAKE_SUDO_LOG=self.sudolog, WATCHDOG_SKIP_MAX_TICKS="0")
        e.update(env)
        subprocess.run(["bash", SCRIPT], env=e, check=True, timeout=60)
        calls = [json.loads(l) for l in open(self.actlog)] if os.path.exists(self.actlog) else []
        sudo = open(self.sudolog).read() if os.path.exists(self.sudolog) else ""
        return calls, sudo, open(self.log).read(), json.load(open(self.state))

    def test_confirmed_wedge_is_handed_to_the_authority(self):
        calls, sudo, log, st = self.run_wd()
        self.assertEqual(calls[0][0], "tick")
        self.assertEqual(calls[1][:5], ["recover", "--cause", "wedge", "--by", "watchdog"])
        self.assertEqual(sudo, "")                      # the watchdog itself no longer kills or restarts
        self.assertIn("liveness authority ACTED", log)
        self.assertEqual(st["consecutive_failures"], 0)

    def test_refusal_is_logged_and_nothing_else_happens(self):
        calls, sudo, log, st = self.run_wd(FAKE_RECOVER="refused")
        self.assertIn("did not act: 2 automatic actions", log)
        self.assertEqual(sudo, "")
        self.assertEqual(st["consecutive_failures"], 5)

    def test_held_engine_is_not_probed(self):
        calls, sudo, log, st = self.run_wd(FAKE_TICK=json.dumps({"state": "HELD", "probe": False, "reason": "engine held by DFT"}))
        self.assertEqual([c[0] for c in calls], ["tick"])
        self.assertIn("DEFER liveness=HELD", log)
        self.assertNotIn("PROBE", log)
        self.assertEqual(st["consecutive_failures"], 0)

    def test_crashed_authority_falls_back_so_a_wedge_always_has_an_exit(self):
        calls, sudo, log, st = self.run_wd(FAKE_RECOVER="crash")
        self.assertIn("legacy fallback", log)
        self.assertIn("kill -s KILL", sudo)
        self.assertIn("restart --no-block", sudo)

    def test_dry_run_passes_through(self):
        calls, sudo, log, st = self.run_wd()
        os.remove(self.actlog)
        json.dump({"consecutive_failures": 4, "last_restart_epoch": 0, "restart_timestamps": [], "last_gen": 50}, open(self.state, "w"))
        e = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}", WATCHDOG_STATE_FILE=self.state, WATCHDOG_ACTION_LOG=self.log,
                 WATCHDOG_ACTUATOR=self.act, FAKE_ACT_LOG=self.actlog, FAKE_SUDO_LOG=self.sudolog, WATCHDOG_SKIP_MAX_TICKS="0",
                 FAKE_RECOVER="refused")
        subprocess.run(["bash", SCRIPT, "--dry-run"], env=e, check=True, timeout=60)
        calls = [json.loads(l) for l in open(self.actlog)]
        self.assertIn("--no-act", calls[0])
        self.assertIn("--dry-run", calls[1])


if __name__ == "__main__":
    unittest.main()
