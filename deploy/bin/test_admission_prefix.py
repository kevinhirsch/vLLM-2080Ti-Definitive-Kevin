"""gw-admission-computed-token-cost (Kevin 2026-09-11): charge admission for tokens the engine
will actually COMPUTE, not raw prompt length. Prefix caching means pi's repeated-context turns
(each turn resends the whole history plus one new message) mostly hit cache: a 111K-token turn
that is 98% cached is ~2K tokens of real prefill work, yet was charged as if it cost 111K.

predict_computed_tokens() tracks a one-deep per-client prefix model: if this request's messages
start with everything the client sent last time (see _is_prefix_of -- a real conversation grows
by 2+ messages per turn, not exactly 1, since the client echoes the assistant's own reply back),
predict only the delta since last turn, plus PREFIX_HIT_MARGIN_TOKENS of slop. Any miss (new
client, first turn, edited/branched history) falls back to full cost -- the explicit AC
requirement, never silently under-costed.

USE_COMPUTED_COST is a kill switch, default OFF: the prediction is measured and logged (see
test_ttft_decomposition-adjacent est_computed/computed_actual telemetry) from day one, but does
not change live admission cost until the switch is flipped, which needs the accuracy gate (200
live pi turns, not reproducible in a unit suite) checked first.
"""
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
    name = "shim_under_test_prefix_%d" % _counter[0]
    spec = importlib.util.spec_from_file_location(name, SHIM_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


@pytest.fixture
def shim(monkeypatch, tmp_path):
    return load_shim(monkeypatch, tmp_path, env={
        "SHIM_LOCAL_BUDGET": "14", "SHIM_BIG_TOKENS": "40000",
        "SHIM_FG_RESERVED": "2", "SHIM_TOKENS_PER_UNIT": "12000",
        "SHIM_PREFIX_HIT_MARGIN_TOKENS": "512",
    })


def _msg(role, tokens, cpt, tag=""):
    chars = int(tokens * cpt) + 1
    return {"role": role, "content": tag + ("x" * chars)}


def _body(messages, cpt):
    return json.dumps({"messages": messages}).encode()


@pytest.fixture
def cpt(shim):
    return shim.CHARS_PER_TOK


# ---- the AC's own scenario -------------------------------------------------------------------

def test_repeated_prefix_100k_with_2k_suffix_first_call_is_full_cost(shim, cpt):
    turn1 = [_msg("user", 100_000, cpt, "prefix-")]
    body1 = _body(turn1, cpt)
    est1 = shim._est_tokens(body1)
    predicted1 = shim.predict_computed_tokens("pi", body1, est1)
    assert predicted1 == est1   # no prior state for this client -> full cost, not a guess


def test_repeated_prefix_100k_with_2k_suffix_second_call_is_cheap(shim, cpt):
    """The literal AC: <= 4K predicted computed tokens on the second call."""
    turn1 = [_msg("user", 100_000, cpt, "prefix-")]
    body1 = _body(turn1, cpt)
    est1 = shim._est_tokens(body1)
    shim._prefix_cache_observe("pi", body1, est1)

    turn2 = turn1 + [_msg("assistant", 50, cpt, "ack-"), _msg("user", 2_000, cpt, "suffix-")]
    body2 = _body(turn2, cpt)
    est2 = shim._est_tokens(body2)
    predicted2 = shim.predict_computed_tokens("pi", body2, est2)
    assert predicted2 <= 4_000, predicted2
    assert predicted2 < est2   # actually cheaper than full cost, not a no-op fallback


# ---- hit / miss / eviction / isolation (per the card's own TDD list) -------------------------

def test_first_turn_ever_from_a_client_is_a_miss(shim, cpt):
    body = _body([_msg("user", 5_000, cpt)], cpt)
    est = shim._est_tokens(body)
    assert shim.predict_computed_tokens("new-client", body, est) == est


def test_single_message_request_is_always_a_miss_nothing_could_be_a_carried_prefix(shim, cpt):
    """len(messages) < 2 short-circuits before even consulting _prefix_seen -- a lone message
    can never be "this client's history plus one new message"."""
    shim._prefix_cache_observe("solo", _body([_msg("user", 50_000, cpt)], cpt), 50_000)
    body = _body([_msg("user", 3_000, cpt)], cpt)
    est = shim._est_tokens(body)
    assert shim.predict_computed_tokens("solo", body, est) == est


def test_edited_history_is_a_miss_not_a_false_hit(shim, cpt):
    """The client resent a DIFFERENT first message (edited/branched history) -- the stored
    prefix must not match, so this is full cost, not an under-estimate."""
    turn1 = [_msg("user", 50_000, cpt, "original-")]
    shim._prefix_cache_observe("pi", _body(turn1, cpt), shim._est_tokens(_body(turn1, cpt)))
    turn2_edited = [_msg("user", 50_000, cpt, "EDITED-"), _msg("assistant", 50, cpt),
                    _msg("user", 1_000, cpt)]
    body2 = _body(turn2_edited, cpt)
    est2 = shim._est_tokens(body2)
    assert shim.predict_computed_tokens("pi", body2, est2) == est2


def test_shrunk_history_is_a_miss(shim, cpt):
    """A NEW request shorter than what's on record can't have the record as its prefix."""
    turn1 = [_msg("user", 10_000, cpt), _msg("assistant", 50, cpt), _msg("user", 2_000, cpt)]
    shim._prefix_cache_observe("pi", _body(turn1, cpt), shim._est_tokens(_body(turn1, cpt)))
    shorter = _body([_msg("user", 10_000, cpt)], cpt)   # same first message, but that's ALL
    est = shim._est_tokens(shorter)
    assert shim.predict_computed_tokens("pi", shorter, est) == est


def test_eviction_signalled_client_falls_back_to_full_cost(shim, cpt):
    """Once a client's tracked entry is evicted (bounded cache, FIFO), it behaves exactly like a
    client that was never seen -- full cost, no crash, no stale prediction."""
    turn1 = [_msg("user", 20_000, cpt), _msg("assistant", 50, cpt), _msg("user", 1_000, cpt)]
    shim._prefix_cache_observe("pi", _body(turn1, cpt), shim._est_tokens(_body(turn1, cpt)))
    del shim._prefix_seen["pi"]   # simulate eviction directly rather than filling 200 slots
    turn2 = turn1 + [_msg("assistant", 50, cpt), _msg("user", 500, cpt)]
    body2 = _body(turn2, cpt)
    est2 = shim._est_tokens(body2)
    assert shim.predict_computed_tokens("pi", body2, est2) == est2


def test_prefix_cache_max_clients_bound_is_enforced_fifo(shim, cpt):
    for i in range(shim.PREFIX_CACHE_MAX_CLIENTS):
        shim._prefix_cache_observe("client-%d" % i, _body([_msg("user", 100, cpt),
                                   _msg("assistant", 10, cpt)], cpt), 100)
    assert len(shim._prefix_seen) == shim.PREFIX_CACHE_MAX_CLIENTS
    assert "client-0" in shim._prefix_seen
    shim._prefix_cache_observe("one-more", _body([_msg("user", 100, cpt),
                               _msg("assistant", 10, cpt)], cpt), 100)
    assert len(shim._prefix_seen) == shim.PREFIX_CACHE_MAX_CLIENTS   # never grows past the bound
    assert "client-0" not in shim._prefix_seen                       # oldest was evicted
    assert "one-more" in shim._prefix_seen


def test_client_isolation_one_clients_history_never_predicts_for_another(shim, cpt):
    shared_prefix = [_msg("user", 30_000, cpt, "shared-"), _msg("assistant", 50, cpt)]
    shim._prefix_cache_observe("client-a", _body(shared_prefix, cpt),
                                shim._est_tokens(_body(shared_prefix, cpt)))
    # client-b sends the SAME prefix content -- but has no history of its own on record
    turn = shared_prefix + [_msg("user", 500, cpt)]
    body = _body(turn, cpt)
    est = shim._est_tokens(body)
    assert shim.predict_computed_tokens("client-b", body, est) == est   # miss: no record for B


# ---- safety: never below 1 unit; extreme sizes don't misbehave --------------------------------

def test_prediction_never_returns_less_than_one(shim, cpt):
    turn1 = [_msg("user", 100_000, cpt), _msg("assistant", 10, cpt)]
    shim._prefix_cache_observe("pi", _body(turn1, cpt), shim._est_tokens(_body(turn1, cpt)))
    # a "hit" whose new suffix is tiny (even 0 new tokens) must still floor at 1 (the margin
    # already guarantees this in practice, but the floor itself must hold independent of margin)
    identical_len_turn = turn1 + [_msg("user", 1, cpt)]
    body = _body(identical_len_turn, cpt)
    est = shim._est_tokens(body)
    assert shim.predict_computed_tokens("pi", body, est) >= 1


@pytest.mark.parametrize("tokens", [500_000, 2_000_000])
def test_extreme_prompt_sizes_produce_sane_bounded_units_not_a_crash(shim, cpt, tokens):
    body = _body([_msg("user", tokens, cpt)], cpt)
    units = shim.estimate_units(body, budget=14, client="pi")
    assert isinstance(units, int)
    assert 1 <= units <= 14 - 2


# ---- USE_COMPUTED_COST kill switch --------------------------------------------------------------

def test_kill_switch_off_by_default_predicted_cost_does_not_affect_admission(shim, cpt):
    assert shim.USE_COMPUTED_COST == 0
    turn1 = [_msg("user", 100_000, cpt), _msg("assistant", 10, cpt)]
    shim._prefix_cache_observe("pi", _body(turn1, cpt), shim._est_tokens(_body(turn1, cpt)))
    turn2 = turn1 + [_msg("user", 1_000, cpt)]   # would be a cheap HIT if the switch were on
    body2 = _body(turn2, cpt)
    off_units = shim._desired_units(body2, "pi")
    on_units = shim._desired_units(body2, None)   # no client at all == same raw-size-only path
    assert off_units == on_units   # switch off -> client-aware and client-blind paths agree


def test_kill_switch_on_uses_the_predicted_cheap_cost(shim, cpt):
    shim.apply_config({"use_computed_cost": "1"})
    assert shim.USE_COMPUTED_COST == 1
    turn1 = [_msg("user", 100_000, cpt), _msg("assistant", 10, cpt)]
    shim._prefix_cache_observe("pi", _body(turn1, cpt), shim._est_tokens(_body(turn1, cpt)))
    turn2 = turn1 + [_msg("user", 2_000, cpt)]
    body2 = _body(turn2, cpt)
    raw_units = shim._desired_units(body2, None)     # no client -> can't predict -> full cost
    predicted_units = shim._desired_units(body2, "pi")   # same client, hit -> cheap
    assert predicted_units < raw_units
    assert predicted_units == 1   # a mostly-cached 100K turn should no longer be "big" at all


def test_hot_reload_can_flip_the_switch_live(shim):
    assert shim.USE_COMPUTED_COST == 0
    shim.apply_config({"use_computed_cost": "1"})
    assert shim.USE_COMPUTED_COST == 1


# ---- mis-estimate logging (AC safety requirement) ----------------------------------------------

def test_mis_estimate_over_2x_is_logged_with_client_name(shim):
    shim._telemetry_note_request({"name": "pi", "t0": __import__("time").time(),
                                   "est_computed": 1000, "computed_actual": 2500})
    assert len(shim._MISESTIMATE_FEED) == 1
    entry = shim._MISESTIMATE_FEED[0]
    assert entry["client"] == "pi" and entry["est_computed"] == 1000 and entry["computed_actual"] == 2500


def test_mis_estimate_within_2x_is_not_logged(shim):
    shim._telemetry_note_request({"name": "pi", "t0": __import__("time").time(),
                                   "est_computed": 1000, "computed_actual": 1900})
    assert len(shim._MISESTIMATE_FEED) == 0


def test_mis_estimate_check_tolerates_missing_fields(shim):
    """Most requests never touch computed-cost fields at all (e.g. small requests, or before
    this card existed) -- must not log a false mis-estimate just because one side is None."""
    shim._telemetry_note_request({"name": "pi", "t0": __import__("time").time()})
    assert len(shim._MISESTIMATE_FEED) == 0
