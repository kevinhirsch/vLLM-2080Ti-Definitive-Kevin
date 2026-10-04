"""RL: the GPU cap is enforced for the whole run -- allocator cap in-process + a unit watcher that kills over cap."""
import json
import os
import subprocess
import sys
import uuid

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gpucap  # noqa: E402
import unitrun  # noqa: E402

FAKE_TORCH = '''
import types, json, os
calls = []
class _Props:
    total_memory = 11 * (1 << 30)
def _lazy_init():
    calls.append("init")
def set_per_process_memory_fraction(f, d):
    calls.append(("frac", round(f, 6), d))
cuda = types.SimpleNamespace(_lazy_init=_lazy_init, device_count=lambda: 1,
                             get_device_properties=lambda i: _Props(), set_per_process_memory_fraction=set_per_process_memory_fraction)
'''


def _py(tmp_path, code, cap="600"):
    (tmp_path / "torch").mkdir(exist_ok=True)
    (tmp_path / "torch" / "__init__.py").write_text(FAKE_TORCH)
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{gpucap.SITE}:{tmp_path}"}
    if cap is not None:
        env["GPU_CAP_MIB"] = cap
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60)


def test_sitecustomize_caps_the_allocator_at_first_cuda_init_not_at_import(tmp_path):
    r = _py(tmp_path, "import torch; a = list(torch.calls); torch.cuda._lazy_init(); torch.cuda._lazy_init(); print(a, torch.calls)")
    assert r.returncode == 0, r.stderr
    before, after = r.stdout.strip().split("] [")
    assert before == "["                                              # nothing at import
    want = round(600 * (1 << 20) / (11 * (1 << 30)), 6)
    assert after.count("frac") == 1 and str(want) in after            # once, at the first init


def test_no_cap_env_no_hook(tmp_path):
    r = _py(tmp_path, "import torch; torch.cuda._lazy_init(); print(torch.calls)", cap=None)
    assert r.returncode == 0 and "frac" not in r.stdout


def test_env_for_puts_the_site_first_and_pins_the_gpu(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/lane")
    monkeypatch.setenv("WINDOW_ID", "w1")
    e = gpucap.env_for(600, 1)
    assert e["PYTHONPATH"] == f"{gpucap.SITE}:/lane" and e["GPU_CAP_MIB"] == "600" and e["CUDA_VISIBLE_DEVICES"] == "1"
    assert e["WINDOW_ID"] == "w1"


def test_watcher_kills_the_unit_over_cap_plus_margin(monkeypatch):
    stopped = []
    monkeypatch.setattr(unitrun, "show", lambda unit, *p: {"ControlGroup": "/x.service"})
    monkeypatch.setattr(unitrun, "cgroup_pids", lambda cg: {10, 11})
    monkeypatch.setattr(unitrun, "stop", lambda u: stopped.append(u) or True)
    monkeypatch.setattr(unitrun, "is_active", lambda u: False)
    monkeypatch.setattr(gpucap.time, "sleep", lambda s: None)
    apps = [{"pid": 10, "gpu": 1, "used_mib": 500}, {"pid": 11, "gpu": 1, "used_mib": 300}, {"pid": 99, "gpu": 1, "used_mib": 20000}]
    monkeypatch.setattr(gpucap.gpuguard, "compute_apps", lambda: apps)
    w = gpucap.Watcher("k7-kbench", cap_mib=600, margin_mib=300, gpu=1, log=lambda m: None)
    assert w.tick() == 800 and not stopped                           # 800 <= 900: fine (engine pid 99 not counted)
    apps[0]["used_mib"] = 700
    assert w.tick() == 1000 and stopped == ["k7-kbench"] and w.killed["used_mib"] == 1000
    assert w.peak == 1000


def test_run_reports_gate_refusal(monkeypatch, tmp_path):
    sh = tmp_path / "gpuok.sh"
    sh.write_text("echo 'GUARD: GPU busy: K5 window'; exit 1\n")
    monkeypatch.setattr(gpucap, "GPUOK", str(sh))
    res = gpucap.run("t", "j", ["true"], gpu=1, cap_mib=100)
    assert res["rc"] == 3 and "GPU busy" in res["why"]


def _user_systemd():
    r = subprocess.run(["systemctl", "--user", "is-system-running"], capture_output=True, text=True)
    return r.stdout.strip() in ("running", "degraded")


@pytest.mark.skipif(not _user_systemd(), reason="needs a user systemd manager")
def test_live_run_over_cap_is_killed_by_unit(monkeypatch, tmp_path):
    """A fake nvidia-smi reports the job's own pid at 2000 MiB: the watcher must stop its unit."""
    monkeypatch.setattr(unitrun, "STATE_DIR", str(tmp_path / "units"))
    smi = tmp_path / "nvidia-smi"
    smi.write_text("#!/bin/bash\ncase \"$*\" in *query-gpu*) echo '1, GPU-b';; *query-compute-apps*) "
                   f"for p in $(cat {tmp_path}/pids 2>/dev/null); do echo \"$p, python, 2000, GPU-b\"; done;; esac\n")
    smi.chmod(0o755)
    monkeypatch.setattr(gpucap.gpuguard, "NVIDIA_SMI", str(smi))
    job = "cap" + uuid.uuid4().hex[:6]
    res = gpucap.run("rltest", job, ["bash", "-c", f"echo $$ > {tmp_path}/pids; sleep 30"], gpu=1, cap_mib=500,
                     margin_mib=100, gate=False, poll_s=0.5, timeout_s=60)
    assert res["result"] == "gpu-cap-killed" and res["rc"] == 137 and res["gpu_cap"]["used_mib"] == 2000
    assert res["duration_s"] < 25
