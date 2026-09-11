"""gw-interactive-never-overflows (Kevin 2026-09-11): "the uncensored local model IS the point
of the estate." Measured: Kevin's own interactive pi session was served locally only 7% of the
time -- 448 of 482 requests overflowed to paid DeepSeek in 2 days ($25.41).

`admission_wait_seconds()` / `admission_lane_limit()` are the deadline/lane-limit computation,
pulled out of `_route_completions` (an aiohttp handler entangled with the network) into pure
functions so the class policy is directly testable without a running server.
"""
import importlib.util
import os
import sys

import pytest

SHIM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keepalive-shim.py")
_counter = [0]


def load_shim(monkeypatch, tmp_path, env=None):
    base_env = {
        "SHIM_UPSTREAM": "http://127.0.0.1:1", "SHIM_REMOTE_BASE": "", "SHIM_REMOTE_KEY": "",
        "SHIM_ENV_FILE": str(tmp_path / "shim.env"),
        "SHIM_STATS_FILE": str(tmp_path / "gateway-stats.json"),
        "SHIM_LOG_REQUESTS": "0", "SHIM_NO_THINK_IPS": "",
    }
    base_env.update(env or {})
    for k, v in base_env.items():
        monkeypatch.setenv(k, v)
    _counter[0] += 1
    name = "shim_under_test_niov_%d" % _counter[0]
    spec = importlib.util.spec_from_file_location(name, SHIM_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


@pytest.fixture
def shim(monkeypatch, tmp_path):
    return load_shim(monkeypatch, tmp_path)


def test_interactive_never_overflows_by_default_even_with_remote_configured(shim):
    """The regression itself: interactive traffic must not time out into an overflow just
    because remote happens to be configured and local is busy."""
    wait = shim.admission_wait_seconds(background=False, remote_available=True, peak=False,
                                       local_wait=30.0, bg_wait=5.0,
                                       interactive_never_overflow=True)
    assert wait == shim.NO_OVERFLOW_SECONDS


def test_interactive_still_overflows_when_the_policy_is_explicitly_off(shim):
    """Reversibility: SHIM_INTERACTIVE_NEVER_OVERFLOW=0 restores the old LOCAL_WAIT behaviour."""
    wait = shim.admission_wait_seconds(background=False, remote_available=True, peak=False,
                                       local_wait=30.0, bg_wait=5.0,
                                       interactive_never_overflow=False)
    assert wait == 30.0


def test_background_is_unaffected_by_the_interactive_policy(shim):
    """Background must keep its short timer regardless of the interactive flag -- this policy
    is scoped to interactive traffic only, not a global 'never overflow' switch."""
    for flag in (True, False):
        wait = shim.admission_wait_seconds(background=True, remote_available=True, peak=False,
                                           local_wait=30.0, bg_wait=5.0,
                                           interactive_never_overflow=flag)
        assert wait == 5.0, flag


def test_background_waits_longer_during_peak_hours(shim):
    """Unchanged pre-existing behaviour: peak DeepSeek pricing biases background toward local
    queueing (LOCAL_WAIT) instead of its normal short BG_WAIT."""
    wait = shim.admission_wait_seconds(background=True, remote_available=True, peak=True,
                                       local_wait=30.0, bg_wait=5.0,
                                       interactive_never_overflow=True)
    assert wait == 30.0


@pytest.mark.parametrize("background", [True, False])
@pytest.mark.parametrize("interactive_flag", [True, False])
def test_no_remote_configured_means_nobody_overflows_regardless_of_class_or_policy(
        shim, background, interactive_flag):
    """Pre-existing invariant this change must not disturb: with no remote at all, every
    class waits forever -- there is nowhere to overflow to."""
    wait = shim.admission_wait_seconds(background=background, remote_available=False, peak=False,
                                       local_wait=30.0, bg_wait=5.0,
                                       interactive_never_overflow=interactive_flag)
    assert wait == shim.NO_OVERFLOW_SECONDS


def test_a_genuinely_down_local_engine_still_overflows_interactive_immediately(shim):
    """The policy only changes what happens while WAITING for a busy-but-healthy engine. A
    dead engine is caught by the separate local_healthy() check earlier in _route_completions,
    entirely before admission_wait_seconds() is ever called -- confirm that check still exists
    and precedes the admission loop in source order."""
    src = open(shim.__file__ if hasattr(shim, "__file__") else SHIM_PATH).read()
    down_check = src.index("if not await local_healthy():")
    wait_call = src.index("deadline = time.time() + admission_wait_seconds(")
    assert down_check < wait_call, "the local-down overflow must still be checked BEFORE admission waits"


def test_lane_limit_reserves_fg_lanes_only_from_background_not_interactive(shim):
    assert shim.admission_lane_limit(background=True, budget=14, fg_reserved=2) == 12
    assert shim.admission_lane_limit(background=False, budget=14, fg_reserved=2) == 14


def test_lane_limit_never_goes_below_one_even_if_fg_reserved_exceeds_budget(shim):
    assert shim.admission_lane_limit(background=True, budget=1, fg_reserved=2) == 1


def test_hot_reload_can_turn_the_policy_off_live(shim):
    assert shim.INTERACTIVE_NEVER_OVERFLOW == 1
    shim.apply_config({"interactive_never_overflow": "0"})
    assert shim.INTERACTIVE_NEVER_OVERFLOW == 0
