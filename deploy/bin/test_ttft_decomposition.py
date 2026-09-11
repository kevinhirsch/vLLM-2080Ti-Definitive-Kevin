"""gw-ttft-decomposition-telemetry (Kevin 2026-09-11): make the parts of response time
separately measurable. The card as written asks for four independently-measured fields
(admission_wait, engine_queue, prefill_time, decode_time); the shim can only deliver three --
engine_queue and prefill_time both happen inside the engine's own scheduler between the
upstream POST and the first streamed byte, with no shim-visible boundary between them (see
decompose_timing()'s docstring for the full RCA: the engine tracks them separately internally,
but only as process-wide Prometheus histograms -- not attributable to one request under
concurrency -- and the real per-request values are never surfaced on the OpenAI-compatible
response without an engine-side patch, which needs a restart). This suite pins the honest
three-way split that IS shim-only: admission_wait, queue_plus_prefill (today's `ttft` field,
kept under its existing name), and decode_time (new, derived).
"""
import importlib.util
import os
import sys

import pytest

SHIM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keepalive-shim.py")
_counter = [0]


def load_shim(monkeypatch, tmp_path):
    env = {
        "SHIM_UPSTREAM": "http://127.0.0.1:1", "SHIM_REMOTE_BASE": "", "SHIM_REMOTE_KEY": "",
        "SHIM_ENV_FILE": str(tmp_path / "shim.env"),
        "SHIM_STATS_FILE": str(tmp_path / "gateway-stats.json"),
        "SHIM_LOG_REQUESTS": "0", "SHIM_NO_THINK_IPS": "",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    _counter[0] += 1
    name = "shim_under_test_ttft_%d" % _counter[0]
    spec = importlib.util.spec_from_file_location(name, SHIM_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


@pytest.fixture
def shim(monkeypatch, tmp_path):
    return load_shim(monkeypatch, tmp_path)


def test_streaming_request_splits_into_three_phases_summing_within_5pct(shim):
    """admission_wait=1.0, queue_plus_prefill(ttft)=2.0, so decode_time should be duration-3.0.
    duration=10.0 -> decode_time=7.0, and all three must sum back to within 5% of duration."""
    r = shim.decompose_timing(waited=1.0, ttft=2.0, duration=10.0)
    assert r == {"admission_wait": 1.0, "queue_plus_prefill": 2.0, "decode_time": 7.0}
    total = r["admission_wait"] + r["queue_plus_prefill"] + r["decode_time"]
    assert abs(total - 10.0) / 10.0 <= 0.05


def test_non_streaming_response_has_no_ttft_so_decode_time_is_null_not_zero(shim):
    """The regression this test guards: _relay()'s ttft insertion point is streaming-only (see
    its own TELEMETRY comment) -- a non-streaming response must never be reported as a 0-second
    decode, which would silently lie that generation was instant."""
    r = shim.decompose_timing(waited=0.5, ttft=None, duration=12.0)
    assert r["admission_wait"] == 0.5
    assert r["queue_plus_prefill"] is None
    assert r["decode_time"] is None


def test_missing_duration_also_nulls_out_the_derived_fields(shim):
    r = shim.decompose_timing(waited=0.5, ttft=2.0, duration=None)
    assert r == {"admission_wait": 0.5, "queue_plus_prefill": None, "decode_time": None}


def test_zero_wait_request_that_went_straight_through(shim):
    """The common case: an interactive request that hit a free lane immediately."""
    r = shim.decompose_timing(waited=0.0, ttft=0.3, duration=4.3)
    assert r["admission_wait"] == 0.0
    assert r["decode_time"] == 4.0


def test_none_waited_is_treated_as_zero_not_a_crash(shim):
    """info.get('waited') can be None (never queued) -- decompose_timing must not choke on it,
    matching the existing `waited = info.get('waited') or 0.0` guard at the call site."""
    r = shim.decompose_timing(waited=None, ttft=1.0, duration=5.0)
    assert r["admission_wait"] == 0.0


def test_measurement_noise_never_produces_a_negative_decode_time(shim):
    """admission_wait + queue_plus_prefill measured (very slightly) larger than duration, from
    clock reads taken at different points in an async pipeline -- must clamp to 0, not go
    negative, which would be a nonsensical 'anti-decode'."""
    r = shim.decompose_timing(waited=5.0, ttft=5.05, duration=10.0)
    assert r["decode_time"] == 0.0


def test_latency_class_pctls_splits_interactive_vs_background(shim):
    rows = [
        (False, 0.0, 0.5, 10.0), (False, 0.0, 0.6, 12.0),   # interactive
        (True, 3.0, 0.5, 20.0),                              # background
    ]
    out = shim._latency_class_pctls(rows)
    assert out["interactive"]["decode_time"]["n"] == 2
    assert out["background"]["decode_time"]["n"] == 1
    assert out["background"]["admission_wait"]["p50"] == 3.0


def test_latency_class_pctls_ignores_none_values_not_counting_them_as_zero(shim):
    """A non-streaming request contributes admission_wait but must not drag decode_time's
    distribution toward 0 -- it should be excluded from that metric's n entirely."""
    rows = [(False, 1.0, None, None), (False, 2.0, 3.0, 8.0)]
    out = shim._latency_class_pctls(rows)
    assert out["interactive"]["admission_wait"]["n"] == 2
    assert out["interactive"]["decode_time"]["n"] == 1
    assert out["interactive"]["decode_time"]["p50"] == 8.0


def test_latency_class_pctls_empty_class_reports_none_not_an_error(shim):
    out = shim._latency_class_pctls([(False, 1.0, 1.0, 1.0)])
    assert out["background"]["admission_wait"] == {"p50": None, "p90": None, "n": 0}


def test_latency_class_pctls_p90_is_the_90th_percentile_not_p95(shim):
    """The card's AC asks for p50/p90 specifically, distinct from the pre-existing p50/p95
    duration/ttft aggregate elsewhere in this file -- confirm _raw_pctl is actually called with
    0.90, not accidentally reusing 0.95."""
    rows = [(False, float(i), None, None) for i in range(1, 11)]   # 1.0 .. 10.0
    out = shim._latency_class_pctls(rows)
    assert out["interactive"]["admission_wait"]["p90"] == 9.0
