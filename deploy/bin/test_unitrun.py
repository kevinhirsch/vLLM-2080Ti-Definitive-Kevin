"""RL (L106): lane jobs run in their own systemd user units; kill/attribution by unit + cgroup, never pkill -f."""
import json
import os
import shutil
import subprocess
import sys
import time
import uuid

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import unitrun  # noqa: E402


def _user_systemd():
    if not shutil.which("systemd-run"):
        return False
    r = subprocess.run(["systemctl", "--user", "is-system-running"], capture_output=True, text=True)
    return r.stdout.strip() in ("running", "degraded")


needs_systemd = pytest.mark.skipif(not _user_systemd(), reason="needs a user systemd manager")


@pytest.fixture
def state(tmp_path, monkeypatch):
    d = tmp_path / "units"
    monkeypatch.setattr(unitrun, "STATE_DIR", str(d))
    return d


def test_unit_name_is_sanitized_and_stable():
    assert unitrun.unit_name("k5", "bench") == "k5-bench"
    assert unitrun.unit_name("K 5/x", "a b;rm -rf") == "K-5-x-a-b-rm--rf"
    assert unitrun.unit_name("", "") == "lane-job"
    assert len(unitrun.unit_name("x" * 300, "y")) <= 200


def test_unit_of_cgroup_prefers_the_job_not_the_user_manager():
    assert unitrun.unit_of_cgroup("0::/user.slice/user-1000.slice/user@1000.service/app.slice/k5-bench.service") == "k5-bench.service"
    assert unitrun.unit_of_cgroup("0::/system.slice/vllm-qwen27b.service") == "vllm-qwen27b.service"
    assert unitrun.unit_of_cgroup("0::/user.slice/user-1000.slice/user@1000.service/app.slice/app-x-12.scope") == "app-x-12.scope"
    assert unitrun.unit_of_cgroup("0::/user.slice/user-1000.slice/session-3.scope") == "session-3.scope"
    assert unitrun.unit_of_cgroup("") is None


def test_build_cmd_has_collect_timeout_status_hook_env_and_output(state, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    argv = unitrun.build_cmd("k5-bench", ["python", "x.py"], timeout_s=600, env={"A": "1"}, cwd="/tmp", out="/tmp/o.log")
    s = " ".join(argv)
    assert "--unit=k5-bench" in argv and "--collect" in argv and "--wait" in argv
    assert "RuntimeMaxSec=600" in s and "KillMode=control-group" in s
    assert "ExecStopPost=/bin/sh -c 'echo $SERVICE_RESULT $EXIT_CODE $EXIT_STATUS > " in s
    assert "--setenv=A=1" in argv and "--setenv=PATH=/usr/bin:/bin" in argv
    assert "StandardOutput=append:/tmp/o.log" in s and "--working-directory=/tmp" in argv
    assert argv[-3:] == ["--", "python", "x.py"]


def test_build_cmd_refuses_a_state_path_that_would_break_the_hook(monkeypatch):
    monkeypatch.setattr(unitrun, "STATE_DIR", "/tmp/has space")
    with pytest.raises(ValueError):
        unitrun.build_cmd("u", ["true"])


def test_owner_of_pid_from_registry_when_the_pid_is_gone(state, monkeypatch, tmp_path):
    monkeypatch.setattr(unitrun, "PROC", str(tmp_path / "noproc"))
    os.makedirs(state)
    rec = {"unit": "k2-bench.service", "lane": "k2", "job": "bench", "started": 1000.0, "finished": 1100.0,
           "pids": {"4242": {"comm": "python", "first": 1001.0, "last": 1090.0}}}
    with open(state / "history.jsonl", "w") as fh:
        fh.write(json.dumps(rec) + "\n")
    own = unitrun.owner_of_pid(4242, at=1050.0)
    assert own["unit"] == "k2-bench.service" and own["lane"] == "k2" and own["how"] == "registry"
    assert unitrun.owner_of_pid(4242, at=99999.0) is None       # pid reuse much later is not blamed on k2
    assert unitrun.owner_of_pid(1, at=1050.0) is None


def test_owner_of_pid_live_cgroup(state, monkeypatch, tmp_path):
    proc = tmp_path / "proc"
    (proc / "77").mkdir(parents=True)
    (proc / "77" / "cgroup").write_text("0::/user.slice/user-1000.slice/user@1000.service/app.slice/k9-gdn.service\n")
    (proc / "77" / "comm").write_text("python\n")
    monkeypatch.setattr(unitrun, "PROC", str(proc))
    own = unitrun.owner_of_pid(77)
    assert own == {"pid": 77, "unit": "k9-gdn.service", "lane": None, "how": "live-cgroup", "comm": "python"}


def test_sampler_records_every_pid_in_the_cgroup_subtree(state, monkeypatch, tmp_path):
    cg = tmp_path / "cg"
    (cg / "a.service" / "child").mkdir(parents=True)
    (cg / "a.service" / "cgroup.procs").write_text("11\n12\n")
    (cg / "a.service" / "child" / "cgroup.procs").write_text("13\n")
    monkeypatch.setattr(unitrun, "CGROUP_ROOT", str(cg))
    monkeypatch.setattr(unitrun, "show", lambda unit, *p: {"ControlGroup": "/a.service"})
    os.makedirs(state)
    rec = {"unit": "a.service", "pids": {}}
    s = unitrun.Sampler("a", rec)
    s.tick()
    assert sorted(rec["pids"]) == ["11", "12", "13"]
    saved = json.load(open(state / "a.json"))
    assert sorted(saved["pids"]) == ["11", "12", "13"]


def test_find_no_pkill_patterns_in_new_framework_files():
    """The class this lane removes: kill by unit, never by a command-line pattern."""
    for f in ("unitrun.py", "windowctl.py", "release.py", "gpuguard.py"):
        src = open(os.path.join(HERE, f)).read()
        assert '"pkill"' not in src and "'pkill" not in src and '"pgrep", "-f"' not in src and "pkill -f " not in src.replace("`pkill -f <pattern>`", "").replace("`pkill -f`", ""), f


@needs_systemd
def test_live_run_exit_code_output_pids_and_registry(state, tmp_path):
    job = "t" + uuid.uuid4().hex[:8]
    out = tmp_path / "o.log"
    res = unitrun.run("rltest", job, ["bash", "-c", "sleep 1 & sleep 1.5; echo hello; exit 4"], timeout_s=30, out=str(out))
    assert res["rc"] == 4 and res["result"] == "exit-code"
    assert "hello" in out.read_text()
    assert res["pids"], "the sampler saw the job's pids"
    hist = [json.loads(x) for x in open(state / "history.jsonl")]
    assert hist[-1]["unit"] == f"rltest-{job}.service" and hist[-1]["rc"] == 4
    own = unitrun.owner_of_pid(res["pids"][0], at=time.time())
    assert own and own["unit"] == f"rltest-{job}.service"


@needs_systemd
def test_live_timeout_is_reported_as_timeout(state):
    res = unitrun.run("rltest", "to" + uuid.uuid4().hex[:6], ["sleep", "30"], timeout_s=1)
    assert res["timed_out"] and res["result"] == "timeout"


@needs_systemd
def test_live_second_start_of_a_running_unit_is_refused_and_stop_kills_it(state):
    job = "dup" + uuid.uuid4().hex[:6]
    first = unitrun.run("rltest", job, ["sleep", "60"], wait=False)
    try:
        assert first["result"] == "started"
        for _ in range(50):
            if unitrun.is_active(f"rltest-{job}"):
                break
            time.sleep(0.1)
        again = unitrun.run("rltest", job, ["sleep", "1"])
        assert again["result"] == "refused"
        assert f"rltest-{job}.service" in unitrun.stop_lane("rltest", prefix=job)
        assert not unitrun.is_active(f"rltest-{job}")
    finally:
        unitrun.stop(f"rltest-{job}")
        unitrun.stop(f"rltest-{job}--sampler")


@needs_systemd
def test_cli_run_uses_the_callers_cwd_for_relative_paths(state, tmp_path, monkeypatch):
    (tmp_path / "rel.txt").write_text("found\n")
    monkeypatch.chdir(tmp_path)
    job = "cwd" + uuid.uuid4().hex[:6]
    rc = unitrun.main(["run", "--lane", "rltest", "--job", job, "--out", "o.log", "--", "cat", "rel.txt"])
    assert rc == 0 and (tmp_path / "o.log").read_text().strip() == "found"
