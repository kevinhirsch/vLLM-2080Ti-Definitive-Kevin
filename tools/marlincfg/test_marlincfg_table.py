"""CPU tests for the LP-MARLINCFG table builder (no GPU, no torch needed)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import marlincfg_bench as mb  # noqa: E402

A4 = (64, 256, 256, 4, 1, 0)   # upstream choice
B3 = (64, 128, 128, 3, 2, 0)
C2 = (128, 64, 128, 2, 1, 0)


def pts(spec, prod=1.0):
    """spec: {M: {cand: (ms, ok)}}"""
    return {m: {"prod_ms": prod, "cands": {c: {"ms": t, "ok": ok} for c, (t, ok) in cs.items()}}
            for m, cs in spec.items()}


def test_candidates_cover_all_configs():
    cs = mb.candidates()
    assert len(cs) == len(set(cs)) == 30
    assert A4 in cs and B3 in cs
    assert all(c[4] == 1 for c in cs if c[2] == 256)   # 256-thread large-M never 2 blocks/SM


def test_no_gain_no_rule():
    p = {"o_proj": {"n": 5120, "k": 3072, "pts": pts({64: {A4: (1.0, True), B3: (0.99, True)}})}}
    table, rules = mb.build_table(p, 0.03)
    assert table == "" and rules == []


def test_failing_numerics_never_chosen():
    p = {"o_proj": {"n": 5120, "k": 3072, "pts": pts({64: {B3: (0.5, False), C2: (0.95, True)}})}}
    table, rules = mb.build_table(p, 0.03)
    assert [r["cand"] for r in rules] == [C2]


def test_runs_merge_and_midpoint_ranges():
    spec = {32: {B3: (0.9, True)}, 64: {B3: (0.9, True)}, 128: {A4: (1.0, True)},
            256: {C2: (0.8, True)}, 512: {C2: (0.8, True)}}
    p = {"down": {"n": 5120, "k": 8704, "pts": pts(spec)}}
    table, rules = mb.build_table(p, 0.03)
    assert [(r["lo"], r["hi"], r["cand"]) for r in rules] == [(32, 96, B3), (193, mb.M_MAX_RULE, C2)]
    parsed = mb.parse_table(table)
    assert mb.match(parsed, 96, 5120, 8704)["cand"] == B3
    assert mb.match(parsed, 128, 5120, 8704) is None          # upstream kept at the losing point
    assert mb.match(parsed, 193, 5120, 8704)["cand"] == C2
    assert mb.match(parsed, 3632, 5120, 8704)["cand"] == C2
    assert mb.match(parsed, 3632, 5120, 3072) is None          # other shape untouched


def test_w8x_shapes_capped_below_2048():
    spec = {1024: {B3: (0.9, True)}, 2047: {B3: (0.9, True)}}
    p = {"gate_up": {"n": 17408, "k": 5120, "pts": pts(spec), "cap_m": True}}
    _, rules = mb.build_table(p, 0.03)
    assert rules[0]["hi"] == 2047


def test_longest_run_preferred_over_local_best():
    spec = {64: {B3: (0.90, True), C2: (0.89, True)}, 128: {B3: (0.90, True)}, 256: {B3: (0.9, True)}}
    p = {"qkv": {"n": 7168, "k": 5120, "pts": pts(spec)}}
    _, rules = mb.build_table(p, 0.03)
    assert len(rules) == 1 and rules[0]["cand"] == B3


def test_parse_rejects_malformed():
    with pytest.raises(ValueError):
        mb.parse_table("5120,3072,10,5:64,128,128,3,1,0")      # empty range
    with pytest.raises(ValueError):
        mb.parse_table("5120,3072,1,5:64,128,128,5,1,0")       # MB out of 2..4
    assert mb.parse_table("") == []
    assert len(mb.parse_table(mb.rule_str(5120, 3072, 17, 99, B3) + ";")) == 1


def test_weighted_skips_w8x_at_large_m():
    p = {"gate_up": {"n": 17408, "k": 5120, "pts": {3632: {"prod_ms": 9.0, "t": 1.0}}},
         "down": {"n": 5120, "k": 8704, "pts": {3632: {"prod_ms": 2.0, "t": 1.0}}}}
    w = mb.weighted(p, "t")
    assert w[3632]["prod_ms"] == 64 * 2.0 and w[3632]["speedup"] == 2.0
