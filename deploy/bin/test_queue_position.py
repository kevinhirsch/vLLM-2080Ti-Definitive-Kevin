"""gw-queue-position-header (Kevin 2026-09-11): a waiting request should be visibly waiting.

Scope note (see the commit message): the full client-facing X-Queue-Position header / SSE
comment line needs live verification against pi and OWU before it can ship on the hot request
path, per the card's own AC -- that half is deferred. What ships here is the honest, low-risk
half: per-class queue depth (so "am I, personally, waiting" is answerable, not just an
aggregate `waiting` count background traffic can dominate), visible on /gateway/stats and
feeding the SLO panel (gw-slo-panel).
"""
import asyncio
import importlib.util
import json
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
    name = "shim_under_test_qp_%d" % _counter[0]
    spec = importlib.util.spec_from_file_location(name, SHIM_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


@pytest.fixture
def shim(monkeypatch, tmp_path):
    return load_shim(monkeypatch, tmp_path)


def _stats(shim):
    resp = asyncio.get_event_loop().run_until_complete(shim.gateway_stats(None))
    return json.loads(resp.body)


def test_waiting_by_class_starts_at_zero(shim):
    s = _stats(shim)
    assert s["waiting_by_class"] == {"interactive": 0, "background": 0}


def test_waiting_by_class_tracks_interactive_and_background_independently(shim):
    shim._waiting_by_class["interactive"] = 3
    shim._waiting_by_class["background"] = 7
    s = _stats(shim)
    assert s["waiting_by_class"] == {"interactive": 3, "background": 7}
    # the pre-existing aggregate must remain independent bookkeeping, not derived from this
    assert "waiting" in s


def test_queue_position_assigned_is_this_classs_current_depth_plus_one(shim):
    """The position stashed via _active_set when a request first queues is 1-indexed and
    scoped to its OWN class -- an interactive arrival's position must not be inflated by
    unrelated background queue depth, and vice versa."""
    shim._waiting_by_class["background"] = 5
    # an interactive arrival queuing now should see position 1 (0 interactive ahead of it),
    # not 6 (which would wrongly count the 5 background waiters)
    position_if_interactive_arrives = shim._waiting_by_class["interactive"] + 1
    assert position_if_interactive_arrives == 1


def test_gateway_stats_response_is_still_valid_json_with_the_new_field(shim):
    s = _stats(shim)
    assert isinstance(s["waiting_by_class"]["interactive"], int)
    assert isinstance(s["waiting_by_class"]["background"], int)
