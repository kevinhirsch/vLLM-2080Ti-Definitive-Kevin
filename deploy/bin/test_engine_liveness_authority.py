"""LV (2026-10-03): the single engine-liveness authority in engine-actuator.py.

Holds instead of stopped timers, one gate (rate limit + backoff + breaker) over every automatic action, the declared state machine
(every non-terminal state has an owner, a deadline and exits), and actions verified by their effect on /health."""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("engine_actuator_lv", os.path.join(HERE, "engine-actuator.py"))
ea = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ea)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = self.tmp.name
        self.now = 1_800_000_000.0
        self.healthy = True
        self.unit = {"ActiveState": "active", "SubState": "running", "MainPID": "100", "NRestarts": "0",
                     "ExecMainStartTimestamp": self.now - 3600, "ActiveEnterTimestamp": self.now - 3600,
                     "InactiveEnterTimestamp": None, "Result": "success"}
        self.offline = {"offline": False}
        self.events, self.sudo = [], []
        self.ledger = []
        self.windows = []
        self.code = None
        self.needs = []
        paths = {"BASE": d, "HOLDS": f"{d}/holds.json", "HOLDS_LOCK": f"{d}/holds.lock", "LSTATE": f"{d}/state.json",
                 "LACTIONS": f"{d}/actions.jsonl", "LPAUSE": f"{d}/PAUSE", "WD_STATE": f"{d}/wd.json",
                 "FQ_RUNNING": f"{d}/fq/RUNNING", "JOB": f"{d}/job.json", "LOCK": f"{d}/restart.lock", "PLANNED": f"{d}/planned.json"}
        for k, v in paths.items():
            p = patch.object(ea, k, v)
            p.start()
            self.addCleanup(p.stop)
        for name, fn in {"engine_healthy": lambda *a, **k: self.healthy, "_unit_facts": lambda: dict(self.unit),
                         "_gateway_offline": lambda: dict(self.offline),
                         "emit": lambda *a, **k: self.events.append((a, k)),
                         "_sudo": lambda args: (self.sudo.append(args), 0)[1],
                         "ledger_rows": lambda limit=None, since=None: [r for r in self.ledger if not since or r["ts"] >= since],
                         "now_iso": lambda: "2027-01-15T00:00:00+00:00",
                         "_window_procs": lambda: list(self.windows),
                         "health_code": lambda *a, **k: self.code,
                         "_raise_need": lambda st, r, f: (self.needs.append(st), "need-1")[1]}.items():
            p = patch.object(ea, name, fn)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(ea.time, "time", lambda: self.now)
        p.start()
        self.addCleanup(p.stop)
        p = patch.object(ea.time, "sleep", lambda s: None)
        p.start()
        self.addCleanup(p.stop)

    def state(self):
        return json.load(open(ea.LSTATE))

    def actions(self):
        return ea._actions()


class DeclaredMachine(Base):
    def test_every_non_terminal_state_declares_owner_deadline_exits(self):
        for name, d in ea.STATES.items():
            self.assertIn(d["kind"], ("resting", "active"), name)
            if d["kind"] == "active":
                self.assertTrue(d["owner"] and d["owner"] != "-", name)
                self.assertTrue(d["exits"], name)
                if name != "CRASH_LOOP":      # handed to Halo: its exit is Halo's action, bounded by the hand-off deadline
                    self.assertIsNotNone(d["deadline_s"], name)

    def test_classify_only_returns_declared_states(self):
        src = open(os.path.join(HERE, "engine-actuator.py")).read()
        import re
        body = src[src.index("def classify("):src.index("def _actions_cached(")]
        for st in set(re.findall(r'return "([A-Z_]+)"', body)):
            self.assertIn(st, ea.STATES)

    def test_published_state_carries_its_declaration(self):
        ea.tick(act=False)
        s = self.state()
        self.assertEqual(s["state"], "UP")
        self.assertEqual(s["declared"], ea.STATES["UP"])
        self.assertTrue(s["probe"])


class Holds(Base):
    def test_hold_defers_everything_and_expires_by_itself(self):
        got = ea.hold_acquire("engine", "DFT", "MTP drafter fine-tune window", 600)
        self.assertIn("lease", got)
        self.healthy = False
        self.unit.update(ActiveState="inactive", InactiveEnterTimestamp=self.now - 3000)
        out = ea.tick()
        self.assertEqual(out["state"], "HELD")
        self.assertFalse(out["probe"])
        self.assertEqual(self.sudo, [])
        r = ea.recover("wedge", "watchdog", "5 probes")
        self.assertFalse(r["acted"])
        self.assertIn("deferred", r["refused"])
        # the window dies without releasing: the TTL is the exit, then the abandoned DOWN engine is started
        self.now += 601
        out = ea.tick()
        self.assertEqual(out["state"], "RECOVERING")
        self.assertIn(["start", "--no-block", ea.UNIT], self.sudo)

    def test_second_hold_of_same_kind_refused_other_kind_allowed(self):
        self.assertIn("lease", ea.hold_acquire("engine", "A", "window A owns engine", 600))
        r = ea.hold_acquire("engine", "B", "window B owns engine", 600)
        self.assertIn("refused", r)
        self.assertEqual(r["holder"]["by"], "A")
        self.assertIn("lease", ea.hold_acquire("quiesce", "gateway-safe-publish", "gateway publish drain", 600))

    def test_bounds(self):
        self.assertIn("refused", ea.hold_acquire("engine", "A", "too long a hold", ea.HOLD_MAX_TTL_S + 1))
        self.assertIn("refused", ea.hold_acquire("engine", "A", "short", 600))
        self.assertIn("refused", ea.hold_acquire("bogus", "A", "unknown kind of hold", 600))

    def test_run_mode_hold_dies_with_its_wrapper(self):
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        got = ea.hold_acquire("engine", "W", "window run by a wrapper", 3600, mode="run", pid=dead.pid)
        self.assertIn("lease", got)
        rows = json.load(open(ea.HOLDS))
        rows[0]["pid_start"] = "1"          # the pid is gone (or reused): the hold is void
        json.dump(rows, open(ea.HOLDS, "w"))
        self.assertEqual(ea.active_holds(), [])

    def test_release_by_lease_and_by_owner(self):
        a = ea.hold_acquire("engine", "kevin-desktop", "manual stop from desktop", 600)
        self.assertEqual(ea.hold_release(by="kevin-desktop")["released"], 1)
        self.assertEqual(ea.active_holds(), [])
        b = ea.hold_acquire("engine", "X", "window X owns engine", 600)
        self.assertEqual(ea.hold_release(lease="nope")["released"], 0)
        self.assertEqual(ea.hold_release(lease=b["lease"])["released"], 1)
        self.assertNotEqual(a["lease"], b["lease"])

    def test_renew(self):
        h = ea.hold_acquire("engine", "X", "window X owns engine", 600)
        self.now += 500
        self.assertIn("until", ea.hold_renew(h["lease"], 600))
        self.now += 500
        self.assertEqual(len(ea.active_holds()), 1)
        self.now += 200
        self.assertIn("refused", ea.hold_renew(h["lease"], 600))

    def test_hold_run_releases_and_ensures_engine_up(self):
        self.healthy = False
        self.unit.update(ActiveState="inactive", InactiveEnterTimestamp=self.now - 10)
        rc = ea.hold_run("engine", "W", "window that stops the engine", 600, [sys.executable, "-c", "import sys; sys.exit(4)"])
        self.assertEqual(rc, 4)
        self.assertEqual(ea.active_holds(), [])
        self.assertIn(["start", "--no-block", ea.UNIT], self.sudo)   # immediately, not after DOWN_GRACE_S

    def test_offline_window_and_legacy_frontier_window_are_holds(self):
        self.offline = {"offline": True, "by": "DFT", "reason": "fine-tune", "remaining_s": 900}
        self.healthy = False
        self.unit.update(ActiveState="inactive", InactiveEnterTimestamp=self.now - 3000)
        self.assertEqual(ea.tick()["state"], "OFFLINE_WINDOW")
        self.offline = {"offline": False}
        os.makedirs(os.path.dirname(ea.FQ_RUNNING))
        open(ea.FQ_RUNNING, "w").close()
        os.utime(ea.FQ_RUNNING, (self.now - 60, self.now - 60))
        self.assertEqual(ea.tick()["state"], "HELD")
        os.utime(ea.FQ_RUNNING, (self.now - ea.FQ_RUNNING_MAX_S - 1,) * 2)   # the runner's own 3 h bound
        self.assertEqual(self.sudo, [])                              # nothing while either window owned the engine
        self.assertEqual(ea.tick()["state"], "RECOVERING")           # stale flag: the abandoned DOWN engine gets its exit
        self.assertEqual(self.sudo[-1], ["start", "--no-block", ea.UNIT])


class LegacyWindows(Base):
    def test_window_process_is_a_bounded_implicit_hold(self):
        self.windows = [(904804, 600, "bash /home/kevin/Desktop/wt-k3/tools/k3/window.sh /home/kevin/projects/lanes/k3")]
        self.healthy = False
        self.unit.update(ActiveState="inactive", InactiveEnterTimestamp=self.now - 3000)
        self.assertEqual(ea.tick()["state"], "HELD")
        self.assertEqual(self.sudo, [])
        self.windows = [(904804, ea.HOLD_MAX_TTL_S + 1, "bash x/window.sh")]   # past the bound: no longer a hold
        self.assertEqual(ea.tick()["state"], "RECOVERING")

    def test_pattern(self):
        for ok in ("bash /home/kevin/projects/lanes/dft/window.sh", "/usr/bin/bash /x/92-crash_state_repro_window.sh",
                   "bash tools/s4_v3_window.sh", "bash /x/bench_driver.sh arg"):
            self.assertTrue(ea.WINDOW_PROC_RE.match(ok), ok)
        for no in ("tail -f /x/window.sh", "grep window.sh", "bash -c 'echo window.sh'", "python3 windowctl.py"):
            self.assertFalse(ea.WINDOW_PROC_RE.match(no), no)


class Exits(Base):
    def test_down_unowned_started_after_grace_only(self):
        self.healthy = False
        self.unit.update(ActiveState="inactive", InactiveEnterTimestamp=self.now - 30)
        self.assertEqual(ea.tick()["state"], "DOWN")
        self.assertEqual(self.sudo, [])
        self.now += ea.DOWN_GRACE_S
        out = ea.tick()
        self.assertEqual(out["state"], "RECOVERING")
        self.assertEqual(self.sudo[-1], ["start", "--no-block", ea.UNIT])

    def test_stuck_boot_recovered(self):
        self.healthy = False
        self.unit.update(ActiveState="activating", ExecMainStartTimestamp=self.now - ea.BOOT_DEADLINE_S + 10)
        self.assertEqual(ea.tick()["state"], "BOOTING")
        self.now += 20
        out = ea.tick()
        self.assertEqual(out["state"], "RECOVERING")
        self.assertIn(["kill", "-s", "KILL", ea.UNIT], self.sudo)
        self.assertIn(["restart", "--no-block", ea.UNIT], self.sudo)

    def test_stuck_boot_recover_leaves_the_collector_a_marker(self):
        self.healthy = False
        self.unit.update(ActiveState="activating", ExecMainStartTimestamp=self.now - ea.BOOT_DEADLINE_S - 5)
        ea.tick()
        m = json.load(open(os.path.join(ea.BASE, "liveness-recover.json")))
        self.assertEqual(m["cause"], "stuck_boot")
        self.assertFalse(os.path.exists(os.path.join(ea.BASE, "wedge-restart.json")))

    def test_dead_core_behind_live_api_recovered_in_two_ticks(self):
        """EF2 class gap: /health 503 = EngineDeadError while /v1/models stays 200 -- the wedge path needed ~5 min."""
        ea.tick()
        self.healthy, self.code = False, 503
        self.now += 60
        self.assertEqual(ea.tick()["state"], "SUSPECT")
        self.assertEqual(self.sudo, [])
        self.now += 60
        out = ea.tick()
        self.assertEqual(out["state"], "RECOVERING")
        self.assertIn(["kill", "-s", "KILL", ea.UNIT], self.sudo)
        self.assertEqual(json.load(open(os.path.join(ea.BASE, "liveness-recover.json")))["cause"], "dead_core")

    def test_a_single_503_or_a_timeout_is_not_a_dead_core(self):
        ea.tick()
        self.healthy, self.code = False, 503
        self.now += 60
        ea.tick()
        self.healthy, self.code = True, 200          # recovered on its own
        self.now += 60
        self.assertEqual(ea.tick()["state"], "UP")
        self.healthy, self.code = False, None        # no answer at all: the slower UNRESPONSIVE path, not DEAD_CORE
        self.now += 60
        self.assertEqual(ea.tick()["state"], "SUSPECT")
        self.now += 60
        self.assertEqual(ea.tick()["state"], "SUSPECT")
        self.assertEqual(self.sudo, [])

    def test_unresponsive_after_being_up_this_boot(self):
        ea.tick()                               # healthy: last_healthy recorded
        self.healthy = False
        self.now += 60
        self.assertEqual(ea.tick()["state"], "SUSPECT")
        self.now += ea.UNRESPONSIVE_S
        self.assertEqual(ea.tick()["state"], "RECOVERING")

    def test_action_verified_by_effect(self):
        self.healthy = False
        self.unit.update(ActiveState="inactive", InactiveEnterTimestamp=self.now - 1000)
        ea.tick()
        self.assertEqual(self.actions()[-1]["outcome"], "pending")
        self.now += 120
        self.healthy = True
        self.unit.update(ActiveState="active", ExecMainStartTimestamp=self.now - 100)
        self.assertEqual(ea.tick()["state"], "UP")
        self.assertEqual(self.actions()[-1]["outcome"], "ok")

    def test_paused_never_acts(self):
        open(ea.LPAUSE, "w").close()
        self.healthy = False
        self.unit.update(ActiveState="inactive", InactiveEnterTimestamp=self.now - 3000)
        self.assertEqual(ea.tick()["state"], "PAUSED")
        self.assertFalse(ea.recover("wedge", "watchdog", "x")["acted"])
        self.assertEqual(self.sudo, [])

    def test_crash_loop_handed_to_halo_not_acted_on(self):
        self.healthy = False
        self.unit.update(ActiveState="activating", ExecMainStartTimestamp=self.now - 10)
        for i in range(ea.CRASH_LOOP_FAULTS):
            self.ledger.append({"kind": "FAULT", "signature": "boot-failed",
                                "ts": ea.datetime.fromtimestamp(self.now - 60 * (i + 1)).isoformat(timespec="seconds")})
        out = ea.tick()
        self.assertEqual(out["state"], "CRASH_LOOP")
        self.assertEqual(self.sudo, [])
        self.assertTrue(any(k.get("handoff") for _a, k in self.events))
        # hand-offs have no consumer yet: a need (-> Kevin) is the observable second exit, filed once per entry
        self.assertEqual(self.needs, ["CRASH_LOOP"])
        self.assertEqual(self.state()["need"], "need-1")
        ea.tick()
        self.assertEqual(self.needs, ["CRASH_LOOP"])

    def test_faults_before_last_healthy_do_not_make_a_crash_loop(self):
        for i in range(ea.CRASH_LOOP_FAULTS):
            self.ledger.append({"kind": "FAULT", "ts": ea.datetime.fromtimestamp(self.now - 600 - i).isoformat(timespec="seconds")})
        ea.tick()                                # healthy now: the window's failed arms are history
        self.healthy = False
        self.unit.update(ActiveState="activating", ExecMainStartTimestamp=self.now + 5)
        self.now += 30
        self.assertEqual(ea.tick()["state"], "BOOTING")

    def test_abandoned_planned_job_reconciled(self):
        json.dump({"state": "draining", "by": "S4"}, open(ea.JOB, "w"))
        ea.tick()
        self.assertEqual(json.load(open(ea.JOB))["state"], "abandoned")

    def test_planned_lock_means_planned(self):
        import fcntl
        fh = open(ea.LOCK, "w")
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            child = subprocess.run([sys.executable, "-c",
                                    "import fcntl,sys; f=open(sys.argv[1],'a+');\n"
                                    "try:\n fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB); print('free')\n"
                                    "except OSError: print('held')", ea.LOCK], capture_output=True, text=True)
            self.assertEqual(child.stdout.strip(), "held")
            self.healthy = False
            self.unit.update(ActiveState="inactive", InactiveEnterTimestamp=self.now - 3000)
            # flock is per open file description: our own probe sees the lock as held by another description
            self.assertTrue(ea._lock_held())
            self.assertEqual(ea.tick()["state"], "PLANNED")
            self.assertEqual(self.sudo, [])
        finally:
            fh.close()


class Gate(Base):
    def _fail_once(self):
        self.healthy = False
        self.unit.update(ActiveState="inactive", InactiveEnterTimestamp=self.now - 3000)
        out = ea.tick()
        self.now += ea.VERIFY_S
        return out

    def test_rate_limit_and_backoff_and_breaker(self):
        out = self._fail_once()
        self.assertTrue(out["tick_result"]["acted"])
        g = ea.gate(self.now)
        # verification happens on the next tick; then a failed action doubles the gap
        ea.tick()
        acts = self.actions()
        self.assertEqual(acts[0]["outcome"], "failed")
        g = ea.gate(self.now)
        self.assertEqual(g["consecutive_failed"], 1)
        self.assertEqual(g["gap_s"], ea.AUTO_MIN_GAP_S * 2)

    def test_breaker_opens_after_consecutive_failures_and_half_opens(self):
        rows = []
        for i in range(ea.BREAKER_FAILS):
            t = self.now - 7200 * (ea.BREAKER_FAILS - i)
            rows.append({"id": f"a{i}", "t": t, "action": "start", "cause": "x"})
            rows.append({"outcome_of": f"a{i}", "outcome": "failed", "t": t + ea.VERIFY_S})
        with open(ea.LACTIONS, "w") as fh:
            fh.write("\n".join(json.dumps(r) for r in rows) + "\n")
        # last failure was 7200 s ago; backoff = min(max, base*2^3)
        g = ea.gate(self.now)
        self.assertEqual(g["consecutive_failed"], ea.BREAKER_FAILS)
        self.assertEqual(g["breaker"], "half-open")
        self.assertTrue(g["allowed"])
        self.now = rows[-2]["t"] + 60
        g = ea.gate(self.now)
        self.assertEqual(g["breaker"], "open")
        self.assertFalse(g["allowed"])
        self.healthy = False
        self.unit.update(ActiveState="inactive", InactiveEnterTimestamp=self.now - 3000)
        self.assertEqual(ea.tick()["state"], "BREAKER_OPEN")
        self.assertTrue(any(k.get("handoff") for _a, k in self.events))
        _ = ea.main  # reset path: a reset row ends the failure run
        ea._actions_append({"id": "reset-1", "t": self.now, "reset": True, "by": "kevin", "reason": "fixed the env"})
        self.assertEqual(ea.gate(self.now)["consecutive_failed"], 0)

    def test_hourly_cap_spans_all_causes(self):
        for i in range(ea.AUTO_MAX_PER_HOUR):
            ea._actions_append({"id": f"a{i}", "t": self.now - 3000 + i * 1300, "action": "start", "cause": f"c{i}"})
            ea._actions_append({"outcome_of": f"a{i}", "outcome": "ok", "t": self.now - 2900 + i * 1300})
        r = ea.recover("wedge", "watchdog", "5 probes failed")
        self.assertFalse(r["acted"])
        self.assertIn("last hour", r["refused"])

    def test_wedge_recover_kills_records_and_restarts_without_blocking(self):
        calls = []
        real_run = subprocess.run

        def fake_run(cmd, *a, **k):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        with patch.object(ea.subprocess, "run", fake_run):
            r = ea.recover("wedge", "watchdog", "consecutive_failures=5")
        self.assertTrue(r["acted"], r)
        self.assertTrue(any("--pre-kill" in c for c in calls))
        self.assertEqual(self.sudo, [["reset-failed", ea.UNIT], ["kill", "-s", "KILL", ea.UNIT], ["restart", "--no-block", ea.UNIT]])
        self.assertTrue(json.load(open(os.path.join(ea.BASE, "wedge-restart.json")))["wedge"])
        # a second detector call while the first is being verified is refused (one action at a time)
        self.assertFalse(ea.recover("wedge", "watchdog", "again")["acted"])
        del real_run

    def test_dry_run_consumes_nothing(self):
        r = ea.recover("wedge", "watchdog", "x", dry_run=True)
        self.assertTrue(r["dry_run"])
        self.assertEqual(self.actions(), [])
        self.assertEqual(self.sudo, [])


class PlannedRestartHonoursHolds(Base):
    def args(self, **kw):
        d = dict(by="halo", reason="arm restore after a test", flags=None, clear_diag=False, no_drain=True, force=False,
                 drain_s=1, drain_max_s=1, health_wait_s=0, hold=None)
        d.update(kw)
        return argparse.Namespace(**d)

    def test_refused_while_someone_else_holds(self):
        ea.hold_acquire("engine", "DFT", "fine-tune window owns engine", 600)
        out = []
        with patch("builtins.print", lambda s: out.append(s)):
            self.assertEqual(ea.do_restart(self.args()), 3)
        self.assertIn("held by DFT", json.loads(out[-1])["refused"])

    def test_quiesce_hold_refuses_planned_restart(self):
        ea.hold_acquire("quiesce", "gateway-safe-publish", "gateway publish drain", 600)
        with patch("builtins.print", lambda s: None):
            self.assertEqual(ea.do_restart(self.args()), 3)

    def test_holder_may_restart_its_own_engine(self):
        h = ea.hold_acquire("engine", "windowctl", "window owns engine for arms", 600)
        with patch("builtins.print", lambda s: None), patch.object(ea, "status", lambda: {"gateway": {}}), \
                patch.object(ea, "active_flags", lambda: []), patch.object(ea, "staged_flags", lambda: []), \
                patch.object(ea.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "", "")):
            rc = ea.do_restart(self.args(hold=h["lease"]))
        self.assertEqual(rc, 0)


class LeaseInheritance(Base):
    """Coordinator 2026-10-03: lane window scripts call `restart` with no --hold; wrapped in `hold run` they must keep working."""

    def test_hold_run_exports_the_lease_to_its_child(self):
        out = os.path.join(self.tmp.name, "env.json")
        rc = ea.hold_run("engine", "W", "window wrapped in hold run", 600,
                         [sys.executable, "-c", "import json,os,sys; json.dump({k: os.environ.get(k) for k in "
                          "('ENGINE_HOLD_LEASE','ENGINE_HOLD_LEASE_KIND')}, open(sys.argv[1],'w'))", out], ensure_up=False)
        self.assertEqual(rc, 0)
        env = json.load(open(out))
        self.assertTrue(env["ENGINE_HOLD_LEASE"])
        self.assertEqual(env["ENGINE_HOLD_LEASE_KIND"], "engine")
        self.assertEqual(ea.active_holds(), [])          # released after the child exits

    def test_restart_reads_the_lease_from_the_environment(self):
        seen = {}
        with patch.dict(os.environ, {ea.HOLD_ENV: "LEASE-FROM-ENV"}), \
                patch.object(sys, "argv", ["engine-actuator.py", "restart", "--reason", "arm restore inside a window"]), \
                patch.object(ea, "spawn_detached", lambda a: seen.setdefault("hold", a.hold) and {"scheduled": True}), \
                patch("builtins.print", lambda *a, **k: None):
            ea.main()
        self.assertEqual(seen["hold"], "LEASE-FROM-ENV")
        with patch.dict(os.environ, {ea.HOLD_ENV: "LEASE-FROM-ENV"}), \
                patch.object(sys, "argv", ["engine-actuator.py", "restart", "--reason", "arm restore inside a window", "--hold", "EXPLICIT"]), \
                patch.object(ea, "spawn_detached", lambda a: seen.__setitem__("hold2", a.hold) or {"scheduled": True}), \
                patch("builtins.print", lambda *a, **k: None):
            ea.main()
        self.assertEqual(seen["hold2"], "EXPLICIT")

    def test_child_restart_with_inherited_lease_is_admitted(self):
        h = ea.hold_acquire("engine", "W", "window wrapped in hold run", 600)
        args = PlannedRestartHonoursHolds.args(self, hold=h["lease"])
        with patch("builtins.print", lambda s: None), patch.object(ea, "status", lambda: {"gateway": {}}), \
                patch.object(ea, "active_flags", lambda: []), patch.object(ea, "staged_flags", lambda: []), \
                patch.object(ea.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "", "")):
            self.assertEqual(ea.do_restart(args), 0)


class ImplicitHoldsOneRule(Base):
    """Implicit holds (gateway offline window, frontier RUNNING, legacy window process) defer AUTOMATIC actions and never
    refuse a PLANNED restart; explicit holds do both."""

    def test_offline_window_defers_automatic_but_admits_planned(self):
        self.offline = {"offline": True, "by": "K3", "reason": "microbench", "remaining_s": 900}
        out = ea.tick()
        self.assertEqual(out["state"], "OFFLINE_WINDOW")
        self.assertEqual(out["implicit_holds"][0]["implicit"], "gateway-offline-window")
        self.assertFalse(ea.recover("wedge", "watchdog", "x")["acted"])
        self.assertEqual(ea.active_holds(), [])          # active_holds() = explicit only, by definition
        args = PlannedRestartHonoursHolds.args(self)
        with patch("builtins.print", lambda s: None), patch.object(ea, "status", lambda: {"gateway": {}}), \
                patch.object(ea, "active_flags", lambda: []), patch.object(ea, "staged_flags", lambda: []), \
                patch.object(ea.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "", "")):
            self.assertEqual(ea.do_restart(args), 0)

    def test_order_matches_classify(self):
        self.offline = {"offline": True, "by": "K3", "reason": "r", "remaining_s": 10}
        self.windows = [(1, 5, "bash x/window.sh")]
        f = ea.observe()
        imp = ea.implicit_holds(f)
        self.assertEqual([h["implicit"] for h in imp], ["gateway-offline-window", "window-process"])
        self.assertEqual(ea.classify(f, {})[0], "OFFLINE_WINDOW")


class RestartRidesAnOpenOfflineWindow(unittest.TestCase):
    """A planned restart inside someone's open offline window used to fall back to the drain FENCE (refuses all admission)."""

    def test_rides_existing_window(self):
        calls = []

        def http(url, method="GET", payload=None, token=None, timeout=5):
            calls.append((url.replace(ea.GATEWAY, ""), method))
            return {"offline": True, "by": "K3", "reason": "bench", "remaining_s": 900, "local_active": 0}
        with patch.object(ea, "http", http), patch.object(ea, "engine_progress", lambda *a, **k: None):
            facts, lease = ea.offline_and_wait(5, "arm", "tok", "W", hard_cap_s=5)
        self.assertEqual(facts["strategy"], "existing-offline-window")
        self.assertEqual(facts["window_by"], "K3")
        self.assertIsNone(lease)
        self.assertEqual({m for _p, m in calls}, {"GET"})        # no POST (new window), no DELETE (not ours), no drain
        self.assertTrue(all(p == "/gateway/offline" for p, _m in calls))


class Cli(Base):
    def test_status_shows_liveness_and_holds(self):
        ea.tick()
        ea.hold_acquire("engine", "W", "window W owns engine", 600)
        with patch.object(ea, "unit_view", lambda: {}), patch.object(ea, "gateway_view", lambda: {}), \
                patch.object(ea, "active_flags", lambda: []), patch.object(ea, "staged_flags", lambda: []):
            st = ea.status()
        self.assertEqual(st["liveness"]["state"], "UP")
        self.assertEqual(st["holds"][0]["by"], "W")

    def test_spawn_detached_forwards_hold(self):
        seen = {}

        class P:
            pid = 1

        def fake_popen(cmd, **k):
            seen["cmd"] = cmd
            return P()
        with patch.object(ea.subprocess, "Popen", fake_popen):
            ea.spawn_detached(argparse.Namespace(reason="x" * 10, by="w", drain_s=1, drain_max_s=1, flags=None, clear_diag=False,
                                                 no_drain=False, force=False, health_wait_s=None, hold="L1"))
        self.assertEqual(seen["cmd"][-2:], ["--hold", "L1"])


if __name__ == "__main__":
    unittest.main()
