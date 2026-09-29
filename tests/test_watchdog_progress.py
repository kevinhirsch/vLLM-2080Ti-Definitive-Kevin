"""The vLLM watchdog must not kill an engine that is doing real work."""

import os
import pathlib
import subprocess


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "deploy/bin/vllm-watchdog.sh"


def run_watchdog(tmp_path, scenario, runs=1):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    curl = bin_dir / "curl"
    curl.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  *'/metrics'*)\n"
        "    n=$(cat \"$COUNT_FILE\" 2>/dev/null || echo 0); n=$((n+1)); echo \"$n\" > \"$COUNT_FILE\"\n"
        "    [ \"$SCENARIO\" = unreadable ] && exit 22\n"
        "    [ \"$SCENARIO\" = progressing ] && g=$n || g=1\n"
        "    printf 'vllm:prompt_tokens_total{engine=\"0\"} 10\\nvllm:generation_tokens_total{engine=\"0\"} %s\\n' \"$g\"\n"
        "    ;;\n"
        "  *'/v1/models'*) printf '200 0.001' ;;\n"
        "  *'/v1/chat/completions'*) exit 28 ;;\n"
        "esac\n"
    )
    curl.chmod(0o755)
    sudo = bin_dir / "sudo"
    sudo.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$SUDO_LOG\"\n")
    sudo.chmod(0o755)
    sleep = bin_dir / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n")
    sleep.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        WATCHDOG_MODEL="estate",
        WATCHDOG_STATE_FILE=str(tmp_path / "state.json"),
        WATCHDOG_ACTION_LOG=str(tmp_path / "actions.log"),
        WATCHDOG_CONSEC_FAIL_THRESHOLD="2",
        WATCHDOG_COOLDOWN_SEC="0",
        SCENARIO=scenario,
        COUNT_FILE=str(tmp_path / "count"),
        SUDO_LOG=str(tmp_path / "sudo.log"),
    )
    for _ in range(runs):
        subprocess.run(["bash", str(SCRIPT)], env=env, check=True, timeout=10)
    return (tmp_path / "actions.log").read_text(), (tmp_path / "sudo.log")


def test_busy_engine_cannot_be_killed(tmp_path):
    log, sudo = run_watchdog(tmp_path, "progressing", runs=3)
    assert "engine progressing" in log
    assert not sudo.exists()


def test_unreadable_metrics_cannot_authorize_kill(tmp_path):
    log, sudo = run_watchdog(tmp_path, "unreadable", runs=3)
    assert "metrics unreadable" in log
    assert not sudo.exists()


def test_flat_metrics_and_failed_probe_can_recover_wedge(tmp_path):
    log, sudo = run_watchdog(tmp_path, "flat", runs=2)
    assert "RESTARTING" in log
    assert "systemctl kill -s KILL vllm-qwen27b.service" in sudo.read_text()
