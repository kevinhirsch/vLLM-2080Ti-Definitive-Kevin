"""_config_owner() (2026-09-10 incident, never pinned by a test until now).

A shim smoke-tested on a spare port with none of the production env inherited its globals as
library DEFAULTS; its first dashboard POST called _persist_config(), which rewrites the WHOLE
shim.env from those globals -- silently replacing tuned production values (lane budget 14 -> 2)
and emptying the remote base+key. The fix: a process whose env agrees with fewer than 60% of
the file's own tunables is not the owner and must not write it. This was fixed the same night
it happened but shipped with no test -- exactly the kind of gap gw-admission-proportional-units
would otherwise reopen every time a new SHIM_* knob (like SHIM_TOKENS_PER_UNIT) is added,
since apply_config()/_persist_config() is the SAME shared code path for every field.
"""
import importlib.util
import os
import sys

import pytest

SHIM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keepalive-shim.py")
_counter = [0]


def load_shim(monkeypatch, tmp_path, env=None):
    base_env = {
        "SHIM_UPSTREAM": "http://127.0.0.1:1",
        "SHIM_REMOTE_BASE": "", "SHIM_REMOTE_KEY": "",
        "SHIM_ENV_FILE": str(tmp_path / "shim.env"),
        "SHIM_STATS_FILE": str(tmp_path / "gateway-stats.json"),
        "SHIM_LOG_REQUESTS": "0", "SHIM_NO_THINK_IPS": "",
    }
    base_env.update(env or {})
    for k, v in base_env.items():
        monkeypatch.setenv(k, v)
    _counter[0] += 1
    name = "shim_under_test_owner_%d" % _counter[0]
    spec = importlib.util.spec_from_file_location(name, SHIM_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def _write_env_file(path, values):
    with open(path, "w") as f:
        for k, v in values.items():
            f.write(f"{k}={v}\n")


def test_no_env_file_yet_means_we_are_creating_it(monkeypatch, tmp_path):
    shim = load_shim(monkeypatch, tmp_path)
    assert not os.path.exists(shim.SHIM_ENV_FILE)
    assert shim._config_owner() is True


def test_a_process_that_matches_the_files_own_tunables_may_write(monkeypatch, tmp_path):
    """The exact PRODUCTION scenario: the real service process, whose env DID load these
    values, must still be able to persist a live dashboard change."""
    shim = load_shim(monkeypatch, tmp_path, env={
        "SHIM_LOCAL_BUDGET": "14", "SHIM_FG_RESERVED": "2", "SHIM_TOKENS_PER_UNIT": "12000",
    })
    _write_env_file(shim.SHIM_ENV_FILE, {
        "SHIM_LOCAL_BUDGET": "14", "SHIM_FG_RESERVED": "2", "SHIM_TOKENS_PER_UNIT": "12000",
    })
    assert shim._config_owner() is True


def test_a_foreign_process_with_default_globals_is_refused(monkeypatch, tmp_path):
    """The incident, reproduced: a foreign instance whose env never loaded the tuned file
    (its globals are library defaults) must be refused, so it cannot clobber production."""
    shim = load_shim(monkeypatch, tmp_path)   # defaults only -- did NOT load the file below
    _write_env_file(shim.SHIM_ENV_FILE, {
        "SHIM_LOCAL_BUDGET": "14", "SHIM_FG_RESERVED": "2",
        "SHIM_TOKENS_PER_UNIT": "12000", "SHIM_BIG_TOKENS": "40000",
    })
    assert shim._config_owner() is False


def test_the_new_tokens_per_unit_knob_rides_the_same_guard(monkeypatch, tmp_path):
    """gw-admission-proportional-units adds SHIM_TOKENS_PER_UNIT to _CFG; confirm it is
    actually counted by the ownership check rather than silently exempted."""
    shim = load_shim(monkeypatch, tmp_path)
    assert "SHIM_TOKENS_PER_UNIT" in shim._CFG
    _write_env_file(shim.SHIM_ENV_FILE, {"SHIM_TOKENS_PER_UNIT": "99999"})
    # a lone foreign-valued tunable is < 60% agreement (0/1) -> refused
    assert shim._config_owner() is False


def test_persist_config_actually_refuses_to_write_when_not_owner(monkeypatch, tmp_path):
    """End-to-end: apply_config() must not touch the file on disk when _config_owner() is
    False, even though the change still applies to this process's own in-memory globals."""
    shim = load_shim(monkeypatch, tmp_path)
    _write_env_file(shim.SHIM_ENV_FILE, {
        "SHIM_LOCAL_BUDGET": "14", "SHIM_TOKENS_PER_UNIT": "12000", "SHIM_BIG_TOKENS": "40000",
    })
    before = open(shim.SHIM_ENV_FILE).read()
    shim.apply_config({"tokens_per_unit": "1"})
    assert shim.TOKENS_PER_UNIT == 1, "in-memory change still applies for this process"
    assert open(shim.SHIM_ENV_FILE).read() == before, "but disk must be untouched -- not the owner"
