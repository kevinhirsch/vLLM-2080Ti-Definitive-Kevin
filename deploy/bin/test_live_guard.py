"""RL: live_guard blocks writes/execs under the live dirs from test code (10:23:12 K6 test hit the live actuator)."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_guard  # noqa: E402

PROBE = r'''
import json, os, subprocess, sys, shutil
sys.path.insert(0, sys.argv[1])
import live_guard
live = sys.argv[2]
live_guard.install([live])
res = {}
def t(name, fn):
    try:
        fn(); res[name] = "allowed"
    except PermissionError as e:
        res[name] = "blocked" if "live_guard" in str(e) else "perm:" + str(e)[:60]
    except Exception as e:
        res[name] = "other:" + type(e).__name__
t("read", lambda: open(os.path.join(live, "existing.json")).read())
t("write", lambda: open(os.path.join(live, "restart-job.json"), "w").write("x"))
t("append", lambda: open(os.path.join(live, "existing.json"), "a"))
t("replace", lambda: os.replace(os.path.join(sys.argv[3], "a"), os.path.join(live, "b")))
t("remove", lambda: os.remove(os.path.join(live, "existing.json")))
t("mkdir", lambda: os.mkdir(os.path.join(live, "d")))
t("copy", lambda: shutil.copy(os.path.join(sys.argv[3], "a"), os.path.join(live, "c")))
t("exec_live", lambda: subprocess.run(["python3", os.path.join(live, "engine-actuator.py"), "restart"]))
t("exec_live_shell", lambda: subprocess.run(f"python3 {live}/engine-actuator.py restart --by K6", shell=True))
t("sudo", lambda: subprocess.run(["sudo", "-n", "true"]))
t("systemctl_engine", lambda: subprocess.run(["systemctl", "restart", "vllm-qwen27b"]))
t("systemctl_user_ok", lambda: subprocess.run(["systemctl", "--user", "is-active", "rltest-nothing"], capture_output=True))
t("tmp_write_ok", lambda: open(os.path.join(sys.argv[3], "ok"), "w").write("x"))
t("os_system", lambda: os.system(f"cat {live}/x > /dev/null"))
print("RES " + json.dumps(res))
'''


def test_live_guard_blocks_writes_and_execs_but_not_reads(tmp_path):
    live, tmp = tmp_path / "vllm-qwen27b", tmp_path / "scratch"
    live.mkdir()
    tmp.mkdir()
    (live / "existing.json").write_text("{}")
    (tmp / "a").write_text("a")
    r = subprocess.run([sys.executable, "-c", PROBE, HERE, str(live), str(tmp)], capture_output=True, text=True, timeout=60)
    line = next(l for l in r.stdout.splitlines() if l.startswith("RES "))
    res = json.loads(line[4:])
    assert res.pop("read") == "allowed" and res.pop("tmp_write_ok") == "allowed" and res.pop("systemctl_user_ok") == "allowed"
    assert res == {k: "blocked" for k in res}, res
    assert (live / "existing.json").read_text() == "{}" and not (live / "restart-job.json").exists()


def test_exec_violation_rules(tmp_path):
    d = [str(tmp_path)]
    assert live_guard.exec_violation(["python3", f"{tmp_path}/engine-actuator.py"], d)
    assert live_guard.exec_violation(["bash", "-c", "sudo -n systemctl stop vllm-qwen27b"], d) == "runs sudo"
    assert live_guard.exec_violation(["systemctl", "start", "vllm-qwen27b-watchdog.timer"], d)
    assert live_guard.exec_violation(["systemctl", "--user", "stop", "rl-x"], d) is None
    assert live_guard.exec_violation(["python3", "/tmp/x.py", "--out", "/tmp/y"], d) is None


def test_this_suite_runs_under_the_guard():
    assert live_guard._state["installed"], "deploy/bin/conftest.py must load live_guard"
    assert os.path.realpath(os.path.expanduser("~/.local/share/vllm-qwen27b")) in live_guard._state["dirs"]
