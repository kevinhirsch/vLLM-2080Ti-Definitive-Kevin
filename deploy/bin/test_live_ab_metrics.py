"""EC: live-traffic A/B metrics (hit rate from the cumulative counters, TTFT, uncached share, the keep rule)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_ab_metrics as lam  # noqa: E402


def write(tmp, name, rows):
    with open(tmp / name, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_prefix_hit_rate_uses_counter_deltas_and_survives_an_engine_restart(tmp_path):
    rows = [{"t": 100 + 60 * i, "queries": 1000 * i, "hits": 500 * i} for i in range(5)]
    rows += [{"t": 400 + 60 * i, "queries": 100 * i, "hits": 90 * i} for i in range(1, 4)]   # counters reset at t=400
    write(tmp_path, "prefixcache.jsonl", rows)
    s = lam.prefix_stats(100, 700, str(tmp_path))
    assert s["queries"] == 4000 + 200 and abs(s["hit_rate"] - (2000 + 180) / 4200) < 1e-9
    assert lam.prefix_stats(0, 50, str(tmp_path))["hit_rate"] is None


def test_request_stats_local_only_ttft_percentiles_uncached_share_and_failures():
    rows = [{"route": "local", "status": 200, "ttft": float(i), "ptok": 1000, "est_computed": 250} for i in range(1, 11)]
    rows += [{"route": "local", "status": 500}, {"route": "remote", "status": 200, "ttft": 99.0}]
    rows += [{"route": "local", "status": 200, "ttft": None, "ptok": 1000, "ptok_exact": 2000, "computed_actual": 500}]
    s = lam.request_stats(rows)
    assert s["local_n"] == 12 and s["ok_n"] == 11 and s["ttft_n"] == 10
    assert s["ttft_p50"] == 5.5 and abs(s["ttft_p90"] - 9.1) < 1e-9
    assert abs(s["uncached_share"] - (2500 + 500) / (10000 + 2000)) < 1e-9
    assert s["local_fail_n"] == 1 and abs(s["local_fail_rate"] - 1 / 12) < 1e-9


def test_keep_rule_hit_rate_ttft_faults_and_enough_data():
    base = {"ok_n": 200, "queries": 10, "hit_rate": 0.60, "ttft_p90": 10.0}
    good = {"ok_n": 100, "queries": 10, "hit_rate": 0.60, "ttft_p90": 10.9}
    assert lam.decide(base, good)[0]
    assert not lam.decide(base, {**good, "hit_rate": 0.59})[0]            # lower than baseline - 0.5 pt
    assert lam.decide(base, {**good, "hit_rate": 0.596})[0]               # equal within tolerance
    assert not lam.decide(base, {**good, "ttft_p90": 11.1})[0]            # > +10 %
    assert not lam.decide(base, good, faults=1)[0]
    assert not lam.decide(base, {**good, "ok_n": 10})[0]                  # not enough data: roll back
