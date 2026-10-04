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
import sys
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
    env = {"PATH": "/usr/bin:/bin", "GPU_GATE_SMI": str(smi), "GPU_GATE_POLL_S": "0.2", "V02_GPU_ROGUE_WAIT_S": "0",
           "GPU_LEASES_FILE": str(tmp_path / "gpu-leases.json"), **(env_extra or {})}
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


# ---- L138: leases ------------------------------------------------------------------------------------------------------
def rich_smi(pid, mib):
    """A fake nvidia-smi that answers both queries gpuguard.py makes (gpu uuid map, compute apps with uuid)."""
    return ('case "$*" in\n'
            "  *query-gpu*) echo '0, GPU-aaa'; echo '1, GPU-bbb';;\n"
            f'  *query-compute-apps*) if [ "$1" = "--query-compute-apps=pid,process_name,used_memory,gpu_uuid" ]; then echo "{pid}, python, {mib}, GPU-bbb"; else echo "{pid}, python, {mib}"; fi;;\n'
            "esac\n")


def test_an_unleased_app_gets_the_extra_wait_and_is_named_in_the_audit(tmp_path):
    victim = subprocess.Popen(["sleep", "60"])
    try:
        r, dt = run_gate(tmp_path, rich_smi(victim.pid, 900), {"V02_GPU_WAIT_S": "1", "V02_GPU_ROGUE_WAIT_S": "3",
                                                              "GPUGUARD_PY": os.path.join(HERE, "gpuguard.py")})
    finally:
        victim.kill()
        victim.wait()
    assert "rc=0" in r.stdout                                     # still boots anyway
    assert dt >= 3.5, dt                                          # 1 s base wait + 3 s for the unleased app
    assert "unleased" in r.stderr and f"pid {victim.pid}" in r.stderr


def test_an_app_under_a_sufficient_lease_gets_no_extra_wait(tmp_path):
    import json
    victim = subprocess.Popen(["sleep", "60"])
    try:
        sys.path.insert(0, HERE)
        import gpuguard
        gpuguard.lease_acquire("lp/bench", 1, 1200, pid=victim.pid, ttl_s=600, path=str(tmp_path / "gpu-leases.json"))
        r, dt = run_gate(tmp_path, rich_smi(victim.pid, 900), {"V02_GPU_WAIT_S": "1", "V02_GPU_ROGUE_WAIT_S": "30",
                                                              "GPUGUARD_PY": os.path.join(HERE, "gpuguard.py")})
    finally:
        victim.kill()
        victim.wait()
    assert "rc=0" in r.stdout and dt < 6, dt
    assert "leased-ok" in r.stderr and "lp/bench" in r.stderr


def test_an_app_over_its_lease_is_treated_as_unleased(tmp_path):
    victim = subprocess.Popen(["sleep", "60"])
    try:
        sys.path.insert(0, HERE)
        import gpuguard
        gpuguard.lease_acquire("lp/bench", 1, 100, pid=victim.pid, ttl_s=600, path=str(tmp_path / "gpu-leases.json"))
        r, dt = run_gate(tmp_path, rich_smi(victim.pid, 5000), {"V02_GPU_WAIT_S": "1", "V02_GPU_ROGUE_WAIT_S": "3",
                                                               "GPUGUARD_PY": os.path.join(HERE, "gpuguard.py")})
    finally:
        victim.kill()
        victim.wait()
    assert dt >= 3.5 and "leased-over" in r.stderr, (dt, r.stderr)


def test_rogue_wait_is_off_with_zero_and_default_is_bounded(tmp_path):
    text = open(GATE).read()
    assert "V02_GPU_ROGUE_WAIT_S:-120" in text
    r, dt = run_gate(tmp_path, rich_smi(1, 900), {"V02_GPU_WAIT_S": "1", "V02_GPU_ROGUE_WAIT_S": "0"})
    assert "rc=0" in r.stdout and dt < 5
