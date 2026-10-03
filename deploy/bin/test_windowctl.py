"""RL (L147): one window framework -- declarative spec, exact snapshot/restore, gates, units, dead-man."""
import json
import os
import subprocess
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gpuguard  # noqa: E402
import unitrun  # noqa: E402
import windowctl as wc  # noqa: E402

K5_SPEC = os.path.join(HERE, "..", "windows", "k5-gdn-fused-ab.yaml")


class World:
    """Everything outside the process, faked: engine, gateway, journal, timers, GPUs, units."""

    def __init__(self, tmp, monkeypatch):
        self.tmp = tmp
        self.health = 200
        self.offline = False
        self.lease_closed = []
        self.xids = ["1 kernel: NVRM: Xid (PCI:0000:01:00): 31, pid=1, old"]
        self.timers = {"vllm-qwen27b-watchdog.timer": "active"}
        self.spend = 5.0
        self.pools = []                   # kv_pool_since results (popped per call), default 960000
        self.foreign = []
        self.units = []
        self.actuator_calls = []
        self.root = tmp / "prod"
        (self.root / ".deps" / "tq_gqa_build").mkdir(parents=True)
        (self.root / "vllm").mkdir()
        (self.root / ".deps" / "tq_gqa_build" / "tq_gqa_sm75.so").write_bytes(b"prod-tq")
        (self.root / "vllm" / "_C.abi3.so").write_bytes(b"prod-C")
        self.override = tmp / "v02.override.env"
        self.override.write_bytes(b"export V02_STACK=1\nexport VLLM_SERVE_EXTRA_ARGS='--foo 1'\n")
        self.busy = tmp / "busy.json"
        mp = monkeypatch
        mp.setattr(wc, "OVERRIDE", str(self.override))
        mp.setattr(wc, "PLANNED", str(tmp / "planned.json"))
        mp.setattr(wc, "LOCK", str(tmp / "window.lock"))
        mp.setattr(wc, "MARKER", str(tmp / "window-active.json"))
        mp.setattr(gpuguard, "BUSY_FILE", str(self.busy))
        mp.setattr(wc, "engine_health", lambda timeout=3: self.health)
        mp.setattr(wc, "wait_health", lambda budget_s, **k: (self.health == 200, 1.0))
        mp.setattr(wc, "running_root", lambda: str(self.root))
        mp.setattr(wc, "xid_lines", lambda: list(self.xids))
        mp.setattr(wc, "timer_state", lambda t: self.timers.get(t, "inactive"))
        mp.setattr(wc, "set_timer", lambda t, on: self.timers.__setitem__(t, "active" if on else "inactive") or True)
        mp.setattr(wc, "spend_usd", lambda: self.spend)
        mp.setattr(wc, "http_json", self.http)
        mp.setattr(wc, "gateway_offline_mod", lambda: self)
        mp.setattr(wc.Window, "arm_deadman", lambda w: None)
        mp.setattr(gpuguard, "kv_pool_since", lambda *a, **k: self.pools.pop(0) if self.pools else 960000)
        mp.setattr(gpuguard, "wait_no_foreign", lambda *a, **k: (not self.foreign, list(self.foreign)))
        mp.setattr(unitrun, "run", self.unit_run)
        mp.setattr(unitrun, "stop_lane", lambda lane, prefix=None, exclude=(): [])
        self.holds = []
        self.hold_supported = False
        mp.setattr(wc, "actuator_hold", self.hold)
        mp.setattr(unitrun, "owner_of_pid", lambda pid, at=None: {"unit": "k2-bench.service", "lane": "k2"} if pid == 777 else None)

    def hold(self, *args, timeout=60):
        self.holds.append(args)
        if not self.hold_supported:
            return None
        return {"lease": "HOLD1", "kind": "engine", "until": 1} if args[0] == "acquire" else {"released": True}

    # gateway-offline.py module surface
    def open_window(self, reason, by, ttl, wait_s, mode="open"):
        self.offline = True
        return {"local_active": 0}, "LEASE1"

    def drop_lease(self, lease=None):
        pass

    def http(self, url, method="GET", payload=None, timeout=8):
        if url.endswith("/gateway/offline"):
            if method == "DELETE":
                self.lease_closed.append(payload.get("lease"))
                self.offline = False
                return {"offline": False}
            if method == "POST":
                return {"lease": payload.get("lease")}
            return {"offline": self.offline, "by": "K5" if self.offline else None}
        return {}

    def unit_run(self, lane, job, cmd, timeout_s=None, env=None, cwd=None, out=None, wait=True, **kw):
        self.units.append({"lane": lane, "job": job, "cmd": cmd, "env": env})
        if any("engine-actuator.py" in c for c in cmd):
            self.actuator_calls.append({"cmd": cmd, "override": self.override.read_bytes()})
            self.health = 200
            return {"unit": unitrun.unit_name(lane, job), "rc": 0, "result": "success"}
        if cmd[:3] == ["sudo", "-n", "systemctl"]:
            if cmd[3] == "stop":
                self.health = None
            return {"unit": unitrun.unit_name(lane, job), "rc": 0}
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})}, cwd=cwd, timeout=60)
        return {"unit": unitrun.unit_name(lane, job), "rc": r.returncode, "result": "success" if r.returncode == 0 else "exit-code"}


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def spec(tmp, **over):
    s = {"lane": "t5", "window": "t5-ab", "reason": "test window for the framework", "by": "K5",
         "results": str(tmp / "results"), "max_s": 600, "ttl_s": 600, "wait_s": 0, "gpu_gate_wait_s": 0,
         "vars": {"OUT": str(tmp / "out")},
         "steps": [
             {"name": "mk", "run": "mkdir -p {{OUT}} && echo '{\"pass\": true}' > {{OUT}}/t.json && echo WINDOW=$WINDOW_ID",
              "capture": {"OK": {"json": "{{OUT}}/t.json", "key": "pass", "default": "False"}}},
             {"name": "stop", "engine": "stop"},
             {"name": "arm", "when": "{{OK}} == True",
              "boot": {"release": str(tmp / "rel"), "env": {"VLLM_K5_GDN_FUSED": "1"}, "extra_args_append": "--profiler-config {}"}},
             {"name": "measure", "run": ["bash", "-c", "echo measured > {{RESULTS}}/m.txt"], "outputs": ["{{RESULTS}}/m.txt"]},
         ]}
    s.update(over)
    return s


# ---------------------------------------------------------------- spec

def test_validate_rejects_unknown_variables_dupes_and_ambiguous_steps(tmp_path):
    bad = spec(tmp_path, steps=[{"name": "a", "run": "echo {{NOPE}}"}, {"name": "a", "run": "x"},
                                {"name": "b", "run": "x", "engine": "stop"}, {"name": "c", "boot": {}}])
    with pytest.raises(wc.SpecError) as e:
        wc.validate(bad)
    msg = str(e.value)
    assert "{{NOPE}}" in msg and "duplicate name a" in msg and "exactly one of" in msg and "boot needs exactly one" in msg


def test_captured_variables_are_usable_only_after_their_step(tmp_path):
    ok = spec(tmp_path)
    assert wc.validate(ok)
    early = spec(tmp_path, steps=[{"name": "x", "run": "echo {{OK}}"}, {"name": "y", "run": "true", "capture": {"OK": {"cmd": "echo 1"}}}])
    with pytest.raises(wc.SpecError, match="EARLIER capture"):
        wc.validate(early)


def test_render_truthy_and_capture(tmp_path):
    assert wc.render({"a": ["x {{V}}"]}, {"V": 3}) == {"a": ["x 3"]}
    with pytest.raises(wc.SpecError):
        wc.render("{{Q}}", {})
    assert wc.truthy("True == True") and not wc.truthy("False == True") and wc.truthy("1 != 2") and wc.truthy("yes")
    (tmp_path / "j.json").write_text('{"a": {"b": [5, 6]}}')
    (tmp_path / "log").write_text("x=1\nx=2\n")
    caps = wc.do_capture({"J": {"json": str(tmp_path / "j.json"), "key": "a.b.1"},
                          "R": {"regex": r"x=(\d)"}, "C": {"cmd": "echo hi"}, "D": {"json": "/nope", "default": "dflt"}},
                         str(tmp_path / "log"))
    assert caps == {"J": "6", "R": "2", "C": "hi", "D": "dflt"}


def test_bash_variables_are_left_alone(tmp_path):
    assert wc.render("for s in 1 4; do echo $s ${s}; done {{A}}", {"A": "z"}) == "for s in 1 4; do echo $s ${s}; done z"


def test_override_value_is_what_bash_would_see():
    assert wc.override_value(b"export VLLM_SERVE_EXTRA_ARGS='--a 1 --b {\"x\":1}'\n", "VLLM_SERVE_EXTRA_ARGS") == '--a 1 --b {"x":1}'
    assert wc.override_value(b"", "VLLM_SERVE_EXTRA_ARGS") == ""


def test_conflict_scan_never_matches_itself(tmp_path):
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "trial_guard.sh"])
    try:
        time.sleep(0.2)
        hits = wc.find_conflicts(["trial_guard.sh"])
        assert any(h["pid"] == p.pid for h in hits)
        mine = wc.find_conflicts(["pytest", "python"])          # our own command line contains both
        assert os.getpid() not in [h["pid"] for h in mine]
    finally:
        p.kill()
        p.wait()


def test_the_ported_k5_spec_validates():
    s = wc.load_spec(K5_SPEC)
    assert wc.validate(s)
    names = [x["name"] for x in s["steps"]]
    assert names[:3] == ["clean", "stop-engine", "gdn-test"] and "k5-boot" in names
    k5 = next(x for x in s["steps"] if x["name"] == "k5-boot")
    assert k5["boot"]["release"]["sha"].startswith("6235bb4956")     # the lane arm is a release, not the tree
    assert "trial_guard.sh" in s["conflicts"] and "vllm-qwen27b-watchdog.timer" in wc.pause_timers_of(s)
    tq = next(x for x in s["steps"] if x["name"] == "tq-test")
    assert s["vars"]["K5"] == "/home/kevin/Desktop/wt-k5"
    assert "VLLM_TQ_GQA_BUILD_DIR={{K5}}/.deps/tq_gqa_build" in tq["run"] and "wt-integrate/.deps" not in tq["run"]


# ---------------------------------------------------------------- a whole window

def test_window_happy_path_runs_steps_in_units_and_restores_exactly(world, tmp_path):
    rel = tmp_path / "rel"
    rel.mkdir()
    before = world.override.read_bytes()
    s = wc.Window(spec(tmp_path)).run()
    assert s["status"] == "ok", s.get("why")
    r = s["restore"]
    assert r["ok"], r["problems"]
    assert world.override.read_bytes() == before and r["override_verbatim"]
    arm = [c for c in world.actuator_calls if b"arm arm" in c["override"]][0]
    ov = arm["override"].decode()
    assert ov.startswith(before.decode())                                  # window-start override + the arm's diff
    assert f"export V02_ROOT={rel}" in ov and "export VLLM_K5_GDN_FUSED=1" in ov
    assert "export VLLM_SERVE_EXTRA_ARGS='--foo 1 --profiler-config {}'" in ov
    assert "--no-drain" in arm["cmd"] and "--foreground" in arm["cmd"]
    assert len(world.actuator_calls) == 2                                   # arm boot + restore boot
    assert world.lease_closed == ["LEASE1"] and not world.offline
    jobs = [u["job"] for u in world.units]
    assert jobs[0] == "t5-ab-mk" and "t5-ab-stop" in jobs and "t5-ab-boot-arm" in jobs and "t5-ab-restore-1" in jobs
    assert world.units[0]["env"]["WINDOW_ID"] == "t5-ab"
    assert s["vars"]["OK"] == "True"
    assert s["boots"][0]["kv_pool"] == 960000
    assert not os.path.exists(wc.MARKER) and not os.path.exists(gpuguard.BUSY_FILE)
    assert json.load(open(tmp_path / "results" / "summary.json"))["status"] == "ok"
    assert json.load(open(wc.PLANNED))["by"] == "K5"                       # the stop is recorded as planned


def test_refused_on_spend_conflict_or_existing_offline_window(world, tmp_path):
    world.spend = 21.0
    s = wc.Window(spec(tmp_path)).run()
    assert s["status"] == "refused" and any("spend" in p for p in s["why"])
    assert not world.units and world.override.read_bytes().startswith(b"export V02_STACK=1")
    world.spend = 1.0
    world.offline = True
    s = wc.Window(spec(tmp_path, results=str(tmp_path / "r2"))).run()
    assert s["status"] == "refused" and any("offline window" in p for p in s["why"])


def test_spend_crossing_the_pause_threshold_mid_window_pauses_and_restores(world, tmp_path, monkeypatch):
    (tmp_path / "rel").mkdir()
    calls = {"n": 0}

    def spend():
        calls["n"] += 1
        return 5.0 if calls["n"] <= 2 else 20.4
    monkeypatch.setattr(wc, "spend_usd", spend)
    s = wc.Window(spec(tmp_path)).run()
    assert s["status"] == "paused-spend" and s["restore"]["ok"]
    assert world.lease_closed == ["LEASE1"]


def test_new_xid_aborts_and_is_attributed_to_its_unit(world, tmp_path, monkeypatch):
    (tmp_path / "rel").mkdir()
    orig = world.unit_run

    def run_and_xid(*a, **k):
        res = orig(*a, **k)
        world.xids.append("2 kernel: NVRM: Xid (PCI:0000:01:00): 31, pid=777, name=python")
        return res
    monkeypatch.setattr(unitrun, "run", run_and_xid)
    s = wc.Window(spec(tmp_path)).run()
    assert s["status"] == "aborted-xid"
    assert s["xids"][0]["pid"] == 777 and s["xids"][0]["owner"]["unit"] == "k2-bench.service"
    assert s["restore"]["override_verbatim"]


def test_foreign_gpu_app_blocks_the_boot_with_its_owner(world, tmp_path):
    (tmp_path / "rel").mkdir()
    world.foreign = [{"pid": 33, "gpu": 0, "used_mib": 202, "unit": "lp-smoke.service", "comm": "python", "cmd": "g8_smoke.py"}]
    s = wc.Window(spec(tmp_path)).run()
    assert s["status"] == "gpu-foreign" and "lp-smoke.service" in s["why"]
    assert not any(b"arm arm" in c["override"] for c in world.actuator_calls)   # never booted into a shrunk pool
    assert s["restore"]["override_verbatim"]


def test_restore_below_the_expected_kv_pool_is_a_failed_restore(world, tmp_path):
    (tmp_path / "rel").mkdir()
    world.pools = [960000, 960000, 899704, 899704]   # snapshot, arm boot, restore x2 shrunk
    s = wc.Window(spec(tmp_path, expected_kv_pool=960000)).run()
    r = s["restore"]
    assert not r["ok"] and any("KV pool after restore" in p for p in r["problems"])
    assert "restore_boot_2" in r                                       # it retried once before failing


def test_prod_jit_so_rebuilt_during_the_window_is_put_back(world, tmp_path, monkeypatch):
    (tmp_path / "rel").mkdir()
    orig = world.unit_run

    def boot_rebuilds(*a, **k):
        res = orig(*a, **k)
        if "boot-arm" in a[1]:
            (world.root / ".deps" / "tq_gqa_build" / "tq_gqa_sm75.so").write_bytes(b"REBUILT by a lane boot")
        return res
    monkeypatch.setattr(unitrun, "run", boot_rebuilds)
    s = wc.Window(spec(tmp_path)).run()
    assert (world.root / ".deps" / "tq_gqa_build" / "tq_gqa_sm75.so").read_bytes() == b"prod-tq"
    assert s["restore"]["so_changed_after_restore"] == []


def test_paused_timer_is_restarted_and_a_stopped_timer_stays_stopped(world, tmp_path):
    (tmp_path / "rel").mkdir()
    world.timers["other.timer"] = "inactive"
    s = wc.Window(spec(tmp_path, pause_timers=["vllm-qwen27b-watchdog.timer"], watch_timers=["other.timer"])).run()
    assert world.timers == {"vllm-qwen27b-watchdog.timer": "active", "other.timer": "inactive"}
    assert s["restore"]["timers"]["vllm-qwen27b-watchdog.timer"] == {"was": "active", "now": "active"}
    assert any("other.timer was already inactive" in n for n in s["notes"])


def test_a_tree_with_shared_jit_dirs_is_refused(world, tmp_path):
    lane = tmp_path / "lane"
    (lane / ".deps").mkdir(parents=True)
    (lane / ".deps" / "tq_gqa_build").symlink_to(world.root / ".deps" / "tq_gqa_build")
    sp = spec(tmp_path, steps=[{"name": "arm", "boot": {"tree": str(lane)}}])
    s = wc.Window(sp).run()
    assert s["status"] == "refused" and "symlink" in s["why"] and "release.py build" in s["why"]
    assert s["restore"]["override_verbatim"]


def test_signal_mid_step_still_restores(world, tmp_path, monkeypatch):
    (tmp_path / "rel").mkdir()
    orig = world.unit_run

    def killed(*a, **k):
        if a[1] == "t5-ab-measure":
            raise wc.Terminated(15)
        return orig(*a, **k)
    monkeypatch.setattr(unitrun, "run", killed)
    s = wc.Window(spec(tmp_path)).run()
    assert s["status"] == "interrupted" and s["restore"]["ok"] and world.lease_closed == ["LEASE1"]


def test_a_second_window_is_refused_while_one_holds_the_lock(world, tmp_path):
    import fcntl
    fh = open(wc.LOCK, "a+")
    fcntl.flock(fh, fcntl.LOCK_EX)
    try:
        s = wc.Window(spec(tmp_path)).run()
        assert s["status"] == "refused" and "lock" in s["why"]
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)


def test_deadman_restores_when_the_window_process_is_gone(world, tmp_path, monkeypatch):
    res = tmp_path / "results"
    res.mkdir()
    snap = {"override_b64": b"export A=1\n".hex(), "override_exists": True, "timers": {}, "prod_root": str(world.root),
            "so": wc.so_hashes(str(world.root)), "kv_pool": 960000, "xid_count": 1, "jit_backup": {}}
    (res / "snapshot.json").write_text(json.dumps(snap))
    p = subprocess.Popen(["true"])
    p.wait()
    state = {"window": "t5-ab", "lane": "t5", "results": str(res), "pid": p.pid, "pid_start": "1", "lease": "L9",
             "boots": 1, "engine_stopped": False, "deadline": time.time() + 999, "restored": False}
    (res / "state.json").write_text(json.dumps(state))
    world.override.write_bytes(b"export A=arm\n")
    assert wc.deadman(str(res / "state.json")) == 0
    assert world.override.read_bytes() == b"export A=1\n"
    assert world.lease_closed == ["L9"]
    out = json.load(open(res / "summary.json"))
    assert out["restored_by"] == "dead-man" and out["status"] == "killed"
    assert json.load(open(res / "state.json"))["restored"] is True


def test_deadman_leaves_a_live_window_alone(world, tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    state = {"window": "w", "lane": "t5", "results": str(res), "pid": os.getpid(),
             "pid_start": unitrun.proc_start(os.getpid()), "deadline": time.time() + 999, "restored": False}
    (res / "state.json").write_text(json.dumps(state))
    world.override.write_bytes(b"export A=arm\n")
    assert wc.deadman(str(res / "state.json")) == 0
    assert world.override.read_bytes() == b"export A=arm\n"


class FakeRelease:
    def __init__(self, root):
        self.ROOT, self.REPO, self.DEFAULT_FROM = str(root), "/nonexistent", "/nonexistent"
        self.activated = []

    def release_dir(self, rid):
        return os.path.join(self.ROOT, rid)

    def list_releases(self):
        return [{"id": "r1"}]

    def manifest_of(self, rid):
        return {"sha": "a" * 40, "label": None}

    def activate(self, rid, **kw):
        self.activated.append((rid, kw))
        return {"to": rid, "restart": kw.get("restart")}


def test_snapshot_files_are_restored_verbatim_when_not_promoted(world, tmp_path):
    (tmp_path / "rel").mkdir()
    f = tmp_path / "serve.sh"
    f.write_text("OLD\n")
    sp = spec(tmp_path, snapshot_files=[str(f)])
    sp["steps"].insert(0, {"name": "deploy", "run": f"echo NEW > {f}"})
    sp["steps"].append({"name": "gate", "run": "exit 1"})          # the identity gate fails -> no promotion
    s = wc.Window(sp).run()
    assert s["status"] == "failed-step" and "promoted" not in s
    assert f.read_text() == "OLD\n" and s["restore"]["files"][str(f)] == "verbatim"


def test_promote_after_success_makes_the_new_state_the_restore_target(world, tmp_path, monkeypatch):
    relroot = tmp_path / "releases"
    (relroot / "r1" / "vllm").mkdir(parents=True)
    (relroot / "r1" / "vllm" / "_C.abi3.so").write_bytes(b"release-C")
    fake = FakeRelease(relroot)
    monkeypatch.setattr(wc, "release_mod", lambda: fake)
    f = tmp_path / "serve.sh"
    f.write_text("OLD\n")
    booted = {"root": str(world.root)}
    monkeypatch.setattr(wc, "running_root", lambda: booted["root"])
    orig = world.unit_run

    def run(*a, **k):
        res = orig(*a, **k)
        if "restore" in a[1]:
            booted["root"] = os.path.realpath(relroot / "r1")         # the restore boot lands on `current`
        return res
    monkeypatch.setattr(unitrun, "run", run)
    sp = spec(tmp_path, snapshot_files=[str(f)], promote={"release": "r1", "files": [str(f)]},
              steps=[{"name": "deploy", "run": f"echo NEW > {f}"}, {"name": "rel", "boot": {"release": "r1"}},
                     {"name": "use", "run": "echo {{RELEASE_rel}}"}])
    s = wc.Window(sp).run()
    assert s["status"] == "ok" and s["promoted"]["release"]["to"] == "r1"
    assert fake.activated[0][0] == "r1" and fake.activated[0][1]["restart"] is False
    r = s["restore"]
    assert r["ok"], r["problems"]
    assert f.read_text() == "NEW\n"                                  # promoted file kept
    assert r["running_root"] == os.path.realpath(relroot / "r1")
    assert world.override.read_bytes().startswith(b"export V02_STACK=1")   # override is still the window-start one


def test_release_step_variable_is_known_to_validate(tmp_path):
    sp = spec(tmp_path, steps=[{"name": "rel-boot", "boot": {"release": "x"}}, {"name": "u", "run": "echo {{RELEASE_rel_boot}}"}])
    assert wc.validate(sp)


def test_the_migration_spec_validates_and_promotes_only_at_the_end():
    s = wc.load_spec(os.path.join(HERE, "..", "windows", "rl-release-migration.yaml"))
    assert wc.validate(s)
    names = [x["name"] for x in s["steps"]]
    assert names[-1] == "identity-gate" and names.index("base-boot") < names.index("rel-boot")
    assert set(s["promote"]["files"]) == set(s["snapshot_files"])
    assert s["promote"]["release"] == {"tree_head": "/home/kevin/Desktop/wt-integrate"}


def test_engine_liveness_hold_is_taken_named_on_every_restart_and_released(world, tmp_path):
    (tmp_path / "rel").mkdir()
    world.hold_supported = True
    s = wc.Window(spec(tmp_path)).run()
    assert s["status"] == "ok" and s["restore"]["ok"]
    acq = world.holds[0]
    assert acq[0] == "acquire" and "--owner-pid" in acq and acq[acq.index("--owner-pid") + 1] == str(os.getpid())
    assert all(c["cmd"][c["cmd"].index("--hold") + 1] == "HOLD1" for c in world.actuator_calls)
    assert world.holds[-1] == ("release", "--lease", "HOLD1")


def test_without_lv_deployed_the_window_runs_and_says_so(world, tmp_path):
    (tmp_path / "rel").mkdir()
    s = wc.Window(spec(tmp_path)).run()
    assert s["status"] == "ok" and any("no `hold` yet" in n for n in s["notes"])
    assert all("--hold" not in c["cmd"] for c in world.actuator_calls)


def test_engine_stop_auto_pauses_the_watchdog_timer_and_restores_it(world, tmp_path, monkeypatch):
    """A watchdog tick STARTS a stopped engine (its service Wants= the engine): any `engine: stop` pauses its timer."""
    (tmp_path / "rel").mkdir()
    seen = {}
    orig = world.unit_run

    def run(*a, **k):
        if a[1] == "t5-ab-stop":
            seen["timer_during_stop"] = world.timers["vllm-qwen27b-watchdog.timer"]
        return orig(*a, **k)
    monkeypatch.setattr(unitrun, "run", run)
    s = wc.Window(spec(tmp_path)).run()
    assert seen["timer_during_stop"] == "inactive"
    assert world.timers["vllm-qwen27b-watchdog.timer"] == "active" and s["restore"]["ok"]
    assert wc.pause_timers_of({"steps": [{"name": "s", "engine": "stop"}]}) == ["vllm-qwen27b-watchdog.timer"]
    assert wc.pause_timers_of({"steps": [{"name": "s", "run": "x"}]}) == []
    assert wc.pause_timers_of({"keep_watchdog_timer": True, "steps": [{"name": "s", "engine": "stop"}]}) == []


def test_deadman_restarts_a_paused_watchdog_timer(world, tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    snap = {"override_b64": world.override.read_bytes().hex(), "override_exists": True,
            "timers": {"vllm-qwen27b-watchdog.timer": "active"}, "prod_root": str(world.root),
            "so": wc.so_hashes(str(world.root)), "kv_pool": 960000, "xid_count": 1, "jit_backup": {}}
    (res / "snapshot.json").write_text(json.dumps(snap))
    world.timers["vllm-qwen27b-watchdog.timer"] = "inactive"           # the window paused it, then was SIGKILLed
    state = {"window": "w", "lane": "t5", "results": str(res), "pid": 2 ** 22 + 7, "pid_start": "x", "boots": 0,
             "engine_stopped": True, "deadline": time.time() + 999, "restored": False}
    (res / "state.json").write_text(json.dumps(state))
    wc.deadman(str(res / "state.json"))
    assert world.timers["vllm-qwen27b-watchdog.timer"] == "active"


def test_lease_is_renewed_so_windows_can_outlive_the_gateway_ttl_cap(world, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(wc, "http_json", lambda url, method="GET", payload=None, timeout=8: calls.append((method, payload)) or {"lease": "LEASE1"})
    w = wc.Window(spec(tmp_path, ttl_s=1800))
    w.renew_s = 0.05
    w.state["lease"] = "LEASE1"
    import threading
    th = threading.Thread(target=w._renew_loop, daemon=True)
    th.start()
    time.sleep(0.3)
    w._lease_stop.set()
    th.join(1)
    posts = [p for m, p in calls if m == "POST"]
    assert len(posts) >= 2 and all(p["lease"] == "LEASE1" and p["ttl_s"] == 1800 for p in posts)


def test_script_steps_run_from_a_read_only_snapshot(world, tmp_path):
    (tmp_path / "rel").mkdir()
    lane_script = tmp_path / "lane_win.sh"
    lane_script.write_text("echo from-script $1 > {}/script.out\n".format(tmp_path))
    sp = spec(tmp_path, steps=[{"name": "sc", "script": [str(lane_script), "argA"]}])
    s = wc.Window(sp).run()
    assert s["status"] == "ok"
    assert (tmp_path / "script.out").read_text().strip() == "from-script argA"
    snap = tmp_path / "results" / "steps" / "sc.snap.sh"
    assert snap.exists() and not os.access(snap, os.W_OK)
    assert world.units[0]["cmd"] == ["bash", str(snap), "argA"]


def test_submit_runs_the_window_from_a_code_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(unitrun, "STATE_DIR", str(tmp_path / "units"))
    seen = {}
    monkeypatch.setattr(wc.subprocess, "run", lambda argv, **k: seen.setdefault("argv", argv) and type("R", (), {"returncode": 0})())
    p = tmp_path / "s.json"
    p.write_text(json.dumps(spec(tmp_path)))
    assert wc.main(["submit", str(p)]) == 0
    argv = seen["argv"]
    i = argv.index("--")
    code, spec_path = argv[i + 2], argv[i + 4]
    assert code.startswith(str(tmp_path / "units" / "windowctl-snap")) and code.endswith("windowctl.py")
    assert os.path.dirname(spec_path) == os.path.dirname(code) and not os.access(code, os.W_OK)
    for f in ("unitrun.py", "gpuguard.py", "release.py", "gateway-offline.py"):
        assert os.path.exists(os.path.join(os.path.dirname(code), f))
    assert "--unit=win-t5-t5-ab" in argv


def test_run_commands_naming_a_script_run_its_read_only_snapshot(world, tmp_path):
    (tmp_path / "rel").mkdir()
    sh = tmp_path / "lane" / "w.sh"
    sh.parent.mkdir()
    sh.write_text(f"echo ran $1 > {tmp_path}/w.out\n")
    sp = spec(tmp_path, steps=[{"name": "a", "run": f"cd /tmp && bash {sh} X && echo after"}])
    s = wc.Window(sp).run()
    assert s["status"] == "ok" and (tmp_path / "w.out").read_text().strip() == "ran X"
    cmd = world.units[0]["cmd"][2]
    assert str(sh) not in cmd and ".snap.sh" in cmd
    snap = cmd.split("bash ")[1].split()[0]
    assert not os.access(snap, os.W_OK) and open(snap).read() == sh.read_text()
    assert s["steps"][0]["script_snapshots"] == {str(sh): snap}


def test_a_gateway_restart_that_forgets_the_window_is_healed_by_the_renewal(world, tmp_path, monkeypatch):
    """09:13:22 2026-10-03: the shim restarted mid-window and dropped the in-memory offline window; the next renewal
    POST then opens a NEW window. The framework must adopt the new lease so close/restore still closes it."""
    saved = []
    world.save_lease = lambda lease, by, reason, ttl, mode="open": saved.append(lease)
    monkeypatch.setattr(wc, "http_json", lambda url, method="GET", payload=None, timeout=8: {"lease": "LEASE2"})
    w = wc.Window(spec(tmp_path))
    os.makedirs(w.path("steps"), exist_ok=True)
    w.state["lease"] = "LEASE1"
    w.renew_once("LEASE1", 600)
    assert w.state["lease"] == "LEASE2" and saved == ["LEASE2"]
    assert any("re-opened" in n for n in w.summary["notes"])
    monkeypatch.setattr(wc, "http_json", lambda url, method="GET", payload=None, timeout=8: {"lease": "LEASE2"})
    w.renew_once("LEASE2", 600)
    assert sum("re-opened" in n for n in w.summary["notes"]) == 1


def test_the_cr2_spec_validates_and_boots_a_release():
    s = wc.load_spec(os.path.join(HERE, "..", "windows", "cr2-short-first-ab.yaml"))
    assert wc.validate(s)
    boot = next(x for x in s["steps"] if "boot" in x)
    assert boot["boot"]["release"]["sha"] == "326846fdc2" and boot["boot"]["env"] == {"VLLM_SCHED_SHORT_FIRST_PREFIX_AWARE": "1"}
    assert [x["name"] for x in s["steps"]][:2] == ["clean", "base-probe"] and s["steps"][-1]["name"] == "gate"
