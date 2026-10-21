#!/usr/bin/env python3
"""L110/L133/L138 (lane A6): every engine boot waits (bounded) for foreign GPU processes, not only the boots a window owns.

An engine boot profiles FREE GPU memory; a foreign CUDA process (or the previous engine's workers still dying) shrinks the KV pool
for the whole generation or fails the boot ('No available memory for the cache blocks' -> the 01:19-01:24 crash loop). The gate must
(a) pass at once on free GPUs, (b) wait for apps that leave, (c) give up after the bound and boot anyway, naming the apps, (d) never
hang on a wedged nvidia-smi, (e) be switchable off, and (f) be wired into both v02 serve scripts. Fake nvidia-smi throughout.

Run:  python -m pytest -q test_gpu_boot_gate.py
"""
import os
import re
import subprocess
import textwrap
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.join(HERE, "gpu-boot-gate.sh")


def run_gate(tmp_path, smi_body, env_extra=None, timeout=30):
    smi = tmp_path / "nvidia-smi"
    smi.write_text("#!/bin/bash\n" + smi_body)
    smi.chmod(0o755)
    env = {"PATH": "/usr/bin:/bin", "GPU_GATE_SMI": str(smi), "GPU_GATE_POLL_S": "0.2", **(env_extra or {})}
    t0 = time.time()
    r = subprocess.run(["bash", "-c", f". {GATE}; gpu_boot_gate; echo rc=$?"], capture_output=True, text=True, env=env, timeout=timeout)
    return r, time.time() - t0


def test_free_gpus_pass_immediately(tmp_path):
    r, dt = run_gate(tmp_path, "exit 0\n", {"V02_GPU_WAIT_S": "30"})
    assert "rc=0" in r.stdout and r.stderr == "" and dt < 3


def test_waits_for_apps_that_leave(tmp_path):
    flag = tmp_path / "gone"
    body = textwrap.dedent(f"""\
        if [ -f {flag} ]; then exit 0; fi
        if [ -f {tmp_path}/seen ]; then touch {flag}; fi
        touch {tmp_path}/seen
        echo "4242, python, 1024"
        """)
    r, dt = run_gate(tmp_path, body, {"V02_GPU_WAIT_S": "30"})
    assert "rc=0" in r.stdout and "GPUs free after" in r.stderr and "WARNING" not in r.stderr
    assert dt < 10


def test_gives_up_after_the_bound_names_the_apps_and_boots_anyway(tmp_path):
    r, dt = run_gate(tmp_path, 'echo "31337, lane_bench.py, 20456"\n', {"V02_GPU_WAIT_S": "1"})
    assert "rc=0" in r.stdout                           # never turns a recovery into an outage
    assert "WARNING" in r.stderr and "31337" in r.stderr and "lane_bench.py" in r.stderr
    assert 1 <= dt < 10


def test_a_wedged_nvidia_smi_cannot_hang_the_boot(tmp_path):
    # `timeout 10` per query, bounded by V02_GPU_WAIT_S overall: a smi that sleeps forever costs <= wait + one query
    r, dt = run_gate(tmp_path, "sleep 300\n", {"V02_GPU_WAIT_S": "1", "GPU_GATE_QUERY_TIMEOUT_S": "1"}, timeout=60)
    assert "rc=0" in r.stdout
    assert dt < 8


def test_off_switch_and_missing_smi(tmp_path):
    r, dt = run_gate(tmp_path, 'echo "1, x, 1"\n', {"V02_GPU_WAIT_S": "0"})
    assert "rc=0" in r.stdout and r.stderr == "" and dt < 2
    r, dt = run_gate(tmp_path, 'echo "1, x, 1"\n', {"V02_GPU_WAIT_S": "5", "GPU_GATE_SMI": "/nonexistent/nvidia-smi"})
    assert "rc=0" in r.stdout and dt < 2


@pytest.mark.parametrize("script", ["serve-hauhaucs-v02.sh", "serve-profile-v02.sh"])
def test_both_v02_serve_scripts_run_the_gate_before_launching(script):
    text = open(os.path.join(HERE, script)).read()
    code = re.sub(r"(?m)^\s*#.*$", "", text)
    assert "gpu_boot_gate" in code and "gpu-boot-gate.sh" in code
    gate_at = code.index("gpu_boot_gate")
    launch_at = code.index("ARGS=(") if "ARGS=(" in code else code.index('eval "exec')
    assert gate_at < launch_at


def test_install_ships_the_gate_next_to_the_serve_scripts():
    # install.sh copies every non-test file of deploy/bin to the runtime dir, so the sourced path resolves there
    assert os.path.isfile(GATE) and not os.path.basename(GATE).startswith("test_")
