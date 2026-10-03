#!/usr/bin/env python3
"""L111 + L153 (lane A6): legacy engine-window shell scripts never stop the watchdog timer, and never leave it stopped.

The window_guard.sh contract, driven with a fake `systemctl`/`sudo` on PATH and a fake actuator:
  * actuator with `hold`: the script is re-executed under `hold run --kind engine` and the timer is never touched;
  * no `hold` subcommand: the timer is paused and restored to its EXACT prior state on every exit path
    (success, failure, SIGTERM); a timer that was inactive before is NOT started;
  * ratchet: no other script under deploy/ or tools/ stops the engine watchdog timer (the repo's two legacy window scripts
    used to; one of them restored it on the success path only).

Run:  python -m pytest -q test_window_guard.py
"""
import os
import re
import signal
import subprocess
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
GUARD = HERE / "window_guard.sh"

FAKE_SYSTEMCTL = textwrap.dedent("""\
    #!/bin/bash
    # fake systemctl: state of the timer in $FAKE/timer.state, every call logged to $FAKE/systemctl.log
    echo "$*" >> "$FAKE/systemctl.log"
    case "$1" in
      show) cat "$FAKE/wants" 2>/dev/null ;;
      is-active) cat "$FAKE/timer.state"; [ "$(cat "$FAKE/timer.state")" = active ] ;;
      stop) echo inactive > "$FAKE/timer.state" ;;
      start) echo active > "$FAKE/timer.state" ;;
    esac
    """)
FAKE_SUDO = '#!/bin/bash\n[ "$1" = "-n" ] && shift\nexec "$@"\n'
# fake actuator: `hold status` ok; `hold run ... -- CMD` runs CMD with ENGINE_HOLD_LEASE set and logs the call
FAKE_ACTUATOR = textwrap.dedent("""\
    import os, subprocess, sys
    a = sys.argv[1:]
    open(os.environ["FAKE"] + "/actuator.log", "a").write(" ".join(a) + "\\n")
    if a[:2] == ["hold", "status"]:
        print("{}")
    elif a[:2] == ["hold", "run"]:
        cmd = a[a.index("--") + 1:]
        sys.exit(subprocess.call(cmd, env=dict(os.environ, ENGINE_HOLD_LEASE="lease1")))
    else:
        sys.exit(2)
    """)
OLD_ACTUATOR = "import sys\nsys.exit(2)\n"        # argparse 'invalid choice' for every unknown subcommand


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        (self.d / "bin").mkdir()
        for n, body in (("systemctl", FAKE_SYSTEMCTL), ("sudo", FAKE_SUDO)):
            (self.d / "bin" / n).write_text(body)
            (self.d / "bin" / n).chmod(0o755)
        (self.d / "timer.state").write_text("active\n")
        self.env = dict(os.environ, FAKE=str(self.d), PATH=f"{self.d / 'bin'}:{os.environ['PATH']}",
                        WINDOW_GUARD_ACTUATOR=str(self.d / "act.py"))
        self.env.pop("ENGINE_HOLD_LEASE", None)
        self.env.pop("WINDOW_GUARD_FALLBACK_PASS", None)

    def tearDown(self):
        self.tmp.cleanup()

    def script(self, body):
        p = self.d / "window.sh"
        p.write_text("#!/usr/bin/env bash\nset -u\n. %s\nwindow_guard_enter T 'a test window' 600 -- \"$@\"\n%s\n" % (GUARD, body))
        p.chmod(0o755)
        return p

    def log(self, name):
        f = self.d / name
        return f.read_text() if f.exists() else ""

    def timer(self):
        return (self.d / "timer.state").read_text().strip()


class WithHold(Base):
    def setUp(self):
        super().setUp()
        (self.d / "act.py").write_text(FAKE_ACTUATOR)

    def test_script_runs_under_hold_run_and_never_touches_the_timer(self):
        s = self.script('echo "lease=${ENGINE_HOLD_LEASE:-none} args=$*" > "$FAKE/ran"')
        r = subprocess.run([str(s), "one", "two"], env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.log("ran").strip(), "lease=lease1 args=one two")
        self.assertIn("hold run --kind engine --by T --reason a test window --ttl 600 -- bash", self.log("actuator.log"))
        self.assertNotRegex(self.log("systemctl.log"), r"(?m)^(stop|start) ")      # no stop, no start (only the read-only 'show')
        self.assertEqual(self.timer(), "active")

    def test_a_stale_installed_watchdog_unit_still_gets_the_timer_paused_inside_the_hold(self):
        # the installed unit has Wants=vllm-qwen27b.service: the timer would restart the stopped engine under the hold
        (self.d / "wants").write_text("vllm-qwen27b.service\n")
        s = self.script('echo "during=$(cat "$FAKE/timer.state") lease=${ENGINE_HOLD_LEASE:-none}" > "$FAKE/ran"')
        r = subprocess.run([str(s)], env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.log("ran").strip(), "during=inactive lease=lease1")
        self.assertEqual(self.timer(), "active")                    # restored exactly
        self.assertIn("hold run --kind engine", self.log("actuator.log"))

    def test_exit_code_passes_through(self):
        s = self.script("exit 7")
        r = subprocess.run([str(s)], env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 7)

    def test_refused_hold_means_the_window_script_never_runs(self):
        (self.d / "act.py").write_text(FAKE_ACTUATOR.replace('sys.exit(subprocess.call(cmd,', 'sys.exit(75) if True else (subprocess.call(cmd,') + "\n")
        s = self.script('touch "$FAKE/ran"')
        r = subprocess.run([str(s)], env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 75)
        self.assertFalse((self.d / "ran").exists())


class Fallback(Base):
    def setUp(self):
        super().setUp()
        (self.d / "act.py").write_text(OLD_ACTUATOR)

    def test_timer_is_paused_during_and_restored_after(self):
        s = self.script('echo "during=$(cat "$FAKE/timer.state")" > "$FAKE/ran"')
        r = subprocess.run([str(s)], env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.log("ran").strip(), "during=inactive")
        self.assertEqual(self.timer(), "active")

    def test_timer_is_restored_when_the_script_fails_hard(self):
        s = self.script("false; exit 3")
        r = subprocess.run([str(s)], env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 3)
        self.assertEqual(self.timer(), "active")

    def test_a_timer_that_was_stopped_stays_stopped(self):
        (self.d / "timer.state").write_text("inactive\n")
        s = self.script("true")
        subprocess.run([str(s)], env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(self.timer(), "inactive")
        self.assertNotIn("start", self.log("systemctl.log"))

    def test_sigterm_restores_the_timer(self):
        s = self.script('touch "$FAKE/up"; sleep 30')
        p = subprocess.Popen([str(s)], env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        for _ in range(100):
            if (self.d / "up").exists():
                break
            time.sleep(0.1)
        self.assertEqual(self.timer(), "inactive")                 # paused while the window runs
        os.kill(p.pid, signal.SIGTERM)
        p.wait(timeout=20)
        self.assertEqual(self.timer(), "active")
        subprocess.run(["pkill", "-f", f"{self.d}/window.sh"], check=False)


class NoOtherScriptStopsTheWatchdogTimer(unittest.TestCase):
    """Ratchet (class): the timer is the liveness authority's clock. Only window_guard.sh's labelled fallback and windowctl's
    exact snapshot/restore (pause_timers) may stop it; install/estate-watchdog/vllm-watchdog only name it."""
    ALLOWED = {"deploy/bin/window_guard.sh", "deploy/bin/windowctl.py", "deploy/bin/test_window_guard.py",
               "deploy/bin/test_windowctl.py"}
    STOP = re.compile(r"systemctl[^\n#]*\bstop\b[^\n#]*watchdog\.timer")

    def test_no_legacy_script_stops_the_timer(self):
        hits = []
        for root in ("deploy", "tools"):
            for p in (REPO / root).rglob("*"):
                if not p.is_file() or p.suffix not in (".sh", ".py", ".service", ".conf", ".yaml") or "__pycache__" in p.parts:
                    continue
                rel = str(p.relative_to(REPO))
                if rel in self.ALLOWED:
                    continue
                for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                    if self.STOP.search(line):
                        hits.append(f"{rel}:{n}: {line.strip()[:100]}")
        self.assertEqual(hits, [], "use window_guard.sh (liveness hold) instead of stopping the watchdog timer")

    def test_the_two_legacy_window_scripts_use_the_guard(self):
        for rel in ("tools/s4_v3_window.sh", "tools/update-from-remote.sh"):
            text = (REPO / rel).read_text()
            self.assertIn("window_guard_enter", text, rel)
            self.assertNotIn("watchdog.timer", re.sub(r"(?m)^\s*#.*$", "", text), rel)


if __name__ == "__main__":
    unittest.main()
