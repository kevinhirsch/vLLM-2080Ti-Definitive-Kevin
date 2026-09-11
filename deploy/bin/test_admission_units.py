"""gw-admission-proportional-units (Kevin 2026-09-11): "response time... unbearably slow".

`estimate_units()` used to charge ANY request >= BIG_TOKENS the WHOLE current budget. Measured
effect: one of Kevin's own 111K-token pi turns held 14/14 lanes, so every other caller --
including his own next turn -- queued up to LOCAL_WAIT and then overflowed to paid remote (his
pi session was served locally only 7% of the time; see the Qwen3.8 Capacity Tuning Backlog
2026-09-11 entry for the full telemetry). This pins the fix: units are now proportional to
estimated size, floored at 1, and capped so a single request can never claim more than
budget - FG_RESERVED -- at least FG_RESERVED lanes stay free for the next caller, of any
class, no matter how big this one is.

Reuses the exact `load_shim` fixture pattern from test_no_think_ips.py: a fresh module import
per test, config redirected to tmp_path, never touching the real shim.env.
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
        "SHIM_REMOTE_BASE": "",
        "SHIM_REMOTE_KEY": "",
        "SHIM_ENV_FILE": str(tmp_path / "shim.env"),
        "SHIM_STATS_FILE": str(tmp_path / "gateway-stats.json"),
        "SHIM_LOG_REQUESTS": "0",
        "SHIM_NO_THINK_IPS": "",
    }
    base_env.update(env or {})
    for k, v in base_env.items():
        monkeypatch.setenv(k, v)
    _counter[0] += 1
    name = "shim_under_test_admission_%d" % _counter[0]
    spec = importlib.util.spec_from_file_location(name, SHIM_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


@pytest.fixture
def shim(monkeypatch, tmp_path):
    # Production-shaped defaults so the AC's "500 / 8K / 40K / 111K / 500K" sizes land where
    # they actually do in prod: budget=14, big-threshold=40K, 2 lanes always reserved.
    return load_shim(monkeypatch, tmp_path, env={
        "SHIM_LOCAL_BUDGET": "14", "SHIM_BIG_TOKENS": "40000",
        "SHIM_FG_RESERVED": "2", "SHIM_TOKENS_PER_UNIT": "12000",
    })


def _body_for_tokens(tokens, chars_per_tok, **extra):
    """A request body whose ESTIMATED token count (via the shim's own char/CHARS_PER_TOK
    formula, int-truncated exactly like _est_tokens does) is as close to `tokens` as an
    integer char count allows. Driving off the shim's real constant, not a guessed ratio, is
    what caught two arithmetic bugs in this file's first draft: CHARS_PER_TOK is 3.5, not 4,
    and int(chars/3.5) truncates -- a naive chars=tokens*3.5 lands one token short/over at
    exact boundaries."""
    import json
    chars = int(tokens * chars_per_tok) + 1   # +1 to clear the int-truncation edge, then verify
    body = json.dumps({"messages": [{"role": "user", "content": "x" * chars}], **extra}).encode()
    return body


@pytest.fixture
def cpt(shim):
    return shim.CHARS_PER_TOK


@pytest.mark.parametrize("tokens", [500, 8_000, 40_000, 111_000, 500_000])
def test_units_never_exceed_reserved_cap_for_any_size(shim, cpt, tokens):
    body = _body_for_tokens(tokens, cpt)
    units = shim.estimate_units(body, budget=14)
    assert 1 <= units <= 14 - 2, (tokens, units)


def test_under_big_tokens_always_costs_exactly_one(shim, cpt):
    assert shim.estimate_units(_body_for_tokens(1000, cpt), budget=14) == 1
    just_under = _body_for_tokens(shim.BIG_TOKENS - 1, cpt)
    assert shim._est_tokens(just_under) < shim.BIG_TOKENS   # confirm the fixture landed correctly
    assert shim.estimate_units(just_under, budget=14) == 1


def test_at_big_tokens_costs_more_than_one_but_not_the_whole_budget(shim, cpt):
    """The regression itself: a 111K request used to cost 14 (the whole budget). Now it must
    cost more than the floor (it IS big) but strictly less than the full budget."""
    units = shim.estimate_units(_body_for_tokens(111_000, cpt), budget=14)
    assert 1 < units < 14, units


def test_a_500k_request_still_only_costs_the_reserved_cap_not_more(shim, cpt):
    """Old behaviour: any request >= BIG_TOKENS cost effective_budget() (14). New: even an
    enormous request is capped at budget - FG_RESERVED (12), never higher."""
    assert shim.estimate_units(_body_for_tokens(500_000, cpt), budget=14) == 12


def test_reserved_cap_holds_at_every_budget_level(shim, cpt):
    """Under backoff (effective_budget()==1) the reserved cap floors at 1, not 0 -- a request
    is never inadmissible."""
    huge = _body_for_tokens(500_000, cpt)
    for budget in (1, 2, 3, 14):
        units = shim.estimate_units(huge, budget=budget)
        assert 1 <= units <= max(1, budget - 2), (budget, units)


def test_default_budget_is_effective_budget_when_not_supplied(shim, cpt):
    shim.trigger_backoff("test")   # effective_budget() -> 1 during backoff
    assert shim.estimate_units(_body_for_tokens(200_000, cpt)) == 1


def test_desired_units_is_uncapped_unlike_estimate_units(shim, cpt):
    """_desired_units is the raw size-implied cost with NO ceiling -- the bg-idle-bypass reads
    this directly to decide whether a request is big enough to deserve MORE than the normal
    reserved cap when the engine is otherwise idle."""
    huge = _body_for_tokens(500_000, cpt)
    desired = shim._desired_units(huge)
    capped = shim.estimate_units(huge, budget=14)
    assert desired > capped
    tok = shim._est_tokens(huge)
    assert desired == max(1, -(-tok // shim.TOKENS_PER_UNIT))   # ceil division against the SAME
                                                                  # estimate the code itself used


def test_allow_full_budget_escape_hatch_only_raises_the_ceiling_to_budget_not_beyond(shim, cpt):
    huge = _body_for_tokens(500_000, cpt)
    normal = shim.estimate_units(huge, budget=14, allow_full_budget=False)
    bypassed = shim.estimate_units(huge, budget=14, allow_full_budget=True)
    assert normal == 12          # budget - FG_RESERVED
    assert bypassed == 14        # full budget, still not more than budget itself
    assert bypassed > normal


def test_a_moderate_request_is_identical_with_or_without_the_bypass_flag(shim):
    """The escape hatch only matters for requests that actually exceed the normal cap; it must
    not change behaviour for ordinary traffic."""
    import json
    small = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
    assert shim.estimate_units(small, allow_full_budget=False) == \
        shim.estimate_units(small, allow_full_budget=True) == 1


def test_hot_reload_of_tokens_per_unit_changes_the_ratio(shim, cpt):
    body = _body_for_tokens(120_000, cpt)
    before = shim.estimate_units(body, budget=14)
    shim.apply_config({"tokens_per_unit": "60000"})
    after = shim.estimate_units(body, budget=14)
    assert after < before, "a larger tokens-per-unit ratio must cost FEWER units for the same size"


def test_stale_docstring_no_longer_claims_a_fixed_big_equals_two(shim):
    """The module docstring used to say 'big (est. prompt >= SHIM_BIG_TOKENS) = 2' while the
    code actually charged the whole budget -- neither statement is true anymore, and neither
    should reappear undetected."""
    src = open(SHIM_PATH).read()
    head = src[:2000]
    assert "big" in head.lower() and "= 2" not in head.split("Placement")[1][:400]
