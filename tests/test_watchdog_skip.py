"""FX2/L64: the watchdog skips its generation POST when the engine's generation counter advanced since the last tick,
but never more than WATCHDOG_SKIP_MAX_TICKS in a row, and never when the counter is flat, reset or unreadable."""
import json
import os
import pathlib
import subprocess

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "deploy/bin/vllm-watchdog.sh"


class Harness:
    def __init__(self, tmp_path, skip_max="9"):
        self.p = tmp_path
        b = tmp_path / "bin"; b.mkdir()
        (b / "curl").write_text(
            "#!/bin/sh\n"
            "case \"$*\" in\n"
            "  *'/metrics'*)\n"
            "    [ -f \"$UNREADABLE\" ] && exit 22\n"
            "    printf 'vllm:prompt_tokens_total{engine=\"0\"} 10\\nvllm:generation_tokens_total{engine=\"0\"} %s\\n' \"$(cat \"$GEN_FILE\")\"\n"
            "    ;;\n"
            "  *'/v1/models'*) printf '200 0.001' ;;\n"
            "  *'/v1/chat/completions'*) echo x >> \"$POST_LOG\"; printf '200 0.05' ;;\n"
            "esac\n")
        (b / "curl").chmod(0o755)
        for name, body in (("sudo", "#!/bin/sh\nexit 0\n"), ("journalctl", "#!/bin/sh\nexit 0\n"), ("sleep", "#!/bin/sh\nexit 0\n")):
            (b / name).write_text(body); (b / name).chmod(0o755)
        self.env = dict(os.environ, PATH=f"{b}:{os.environ['PATH']}", WATCHDOG_MODEL="estate",
                        WATCHDOG_STATE_FILE=str(tmp_path / "state.json"), WATCHDOG_ACTION_LOG=str(tmp_path / "actions.log"),
                        WATCHDOG_SKIP_MAX_TICKS=skip_max, GEN_FILE=str(tmp_path / "gen"), POST_LOG=str(tmp_path / "posts"),
                        UNREADABLE=str(tmp_path / "unreadable"))

    def tick(self, gen):
        (self.p / "gen").write_text(str(gen))
        subprocess.run(["bash", str(SCRIPT)], env=self.env, check=True, timeout=10)

    def posts(self):
        f = self.p / "posts"
        return len(f.read_text().split()) if f.exists() else 0

    def log(self):
        return (self.p / "actions.log").read_text()


def test_first_tick_probes_then_advancing_generation_skips_the_post(tmp_path):
    h = Harness(tmp_path)
    h.tick(1000)                 # no previous reading -> real probe
    assert h.posts() == 1
    h.tick(1500); h.tick(2200)   # generated tokens both times -> no POST
    assert h.posts() == 1
    assert h.log().count("SKIP-PROBE") == 2


def test_flat_counter_still_probes(tmp_path):
    h = Harness(tmp_path)
    h.tick(1000); h.tick(1000); h.tick(1000)
    assert h.posts() == 3 and "SKIP-PROBE" not in h.log()


def test_counter_reset_after_an_engine_restart_probes(tmp_path):
    h = Harness(tmp_path)
    h.tick(900000); h.tick(5)
    assert h.posts() == 2


def test_never_more_than_skip_max_in_a_row(tmp_path):
    h = Harness(tmp_path, skip_max="3")
    g = 100
    for _ in range(10):
        g += 50; h.tick(g)
    # tick1 probes; then 3 skips, 1 probe, 3 skips, 1 probe, 1 skip -> 3 probes
    assert h.posts() == 3
    assert h.log().count("SKIP-PROBE") == 7


def test_skip_max_zero_is_the_old_behaviour(tmp_path):
    h = Harness(tmp_path, skip_max="0")
    for g in (10, 20, 30, 40):
        h.tick(g)
    assert h.posts() == 4 and "SKIP-PROBE" not in h.log()


def test_unreadable_metrics_never_skip(tmp_path):
    h = Harness(tmp_path)
    h.tick(10)
    (tmp_path / "unreadable").write_text("1")
    h.tick(99)
    assert h.posts() == 2


def test_a_skip_resets_the_failure_streak(tmp_path):
    h = Harness(tmp_path)
    h.tick(10)
    st = json.loads((tmp_path / "state.json").read_text()); st["consecutive_failures"] = 3
    (tmp_path / "state.json").write_text(json.dumps(st))
    h.tick(500)
    assert json.loads((tmp_path / "state.json").read_text())["consecutive_failures"] == 0
