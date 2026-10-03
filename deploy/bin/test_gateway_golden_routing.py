#!/usr/bin/env python3
"""Lane SH characterization ("golden") tests for the gateway router and its admission paths.

WHY: keepalive-shim.py is being decomposed into modules. Every move must be behaviour-preserving, so this file pins
what `handle_completions` -> `_route_completions` does TODAY for a matrix of ~70 recorded inputs: the response
(status, error type/code, gateway headers), every upstream call (local relay vs remote forward, the exact body bytes
sent as a sha256, stream flag, concurrency, alias endpoint/model), every routing event (_events ring: decision,
reason, units, waited, class, cost estimate), the telemetry JSONL row (all fields except wall-clock ones), the
admission counters after the request (must return to zero), and how long the request queued (on a FAKE clock).

HOW: each scenario imports a FRESH copy of the shim (no state leaks between scenarios) with every on-disk path
pointed at a private temp dir and every SHIM_*/GATEWAY_* variable cleared, so the scenario is a pure function of
its own inputs. Only the edges are stubbed: the upstream calls (_relay/_forward_remote), engine health, the spend
authority answers, and the clock (shim.time / shim.asyncio.sleep -> a fake clock that advances on sleep, so queue
waits are exact and instant). Everything in between -- routing guards, local-first, admission lanes, flow tickets,
cache-model prediction, body preparation, record_event, the telemetry row -- is the real code.

The golden file is golden/routing_golden.json. To re-record after an INTENDED behaviour change (a bug fix with its
own test), run `SH_GOLDEN_REGEN=1 python -m pytest -q test_gateway_golden_routing.py` and review the diff: every
changed line must be explained by that fix.

Run:  python -m pytest -q test_gateway_golden_routing.py
"""
import asyncio
import hashlib
import importlib.util
import json
import os
import re
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp import web
from multidict import CIMultiDict

HERE = Path(__file__).resolve().parent
SHIM_PATH = Path(os.environ.get("SHIM_TEST_CANDIDATE", str(HERE / "keepalive-shim.py")))
GOLDEN = HERE / "golden" / "routing_golden.json"
T0 = 1_790_000_000.0          # fake epoch (2026-09-21); the shim never sees the real clock in these tests
MAX_SLEEPS = 20_000           # a request still queued after this many polls is recorded as "spins" (waits forever)

# Telemetry-row fields that are wall-clock or per-process random: excluded from the golden.
_ROW_VOLATILE = {"t", "duration", "decode_time", "e2e", "ttft"}


class FakeClock:
    def __init__(self):
        self.now = T0
        self.sleeps = 0
        self.hooks = {}            # sleep number -> callable(module) run when that poll happens

    def time(self):
        return self.now

    def monotonic(self):
        return self.now - T0 + 1000.0


def _time_proxy(clock, real_time):
    ns = types.SimpleNamespace(**{k: getattr(real_time, k) for k in dir(real_time) if not k.startswith("__")})
    ns.time = clock.time
    ns.monotonic = clock.monotonic
    return ns


class _Spin(Exception):
    pass


def _asyncio_proxy(clock, module_ref):
    real = asyncio
    ns = types.SimpleNamespace(**{k: getattr(real, k) for k in dir(real) if not k.startswith("__")})

    async def sleep(delay, result=None):
        clock.sleeps += 1
        clock.now += max(0.0, float(delay))
        hook = clock.hooks.get(clock.sleeps)
        if hook:
            hook(module_ref[0])
        if clock.sleeps >= MAX_SLEEPS:
            raise _Spin()
        await real.sleep(0)
        return result
    ns.sleep = sleep
    return ns


class Request(dict):
    """The parts of aiohttp.web.Request the router reads."""
    method = "POST"

    def __init__(self, fields=None, headers=None, path="/v1/chat/completions", remote="127.0.0.1", raw=None):
        super().__init__()
        self.path = path
        self.remote = remote
        self.headers = CIMultiDict({"X-Client": "golden-interactive", "User-Agent": "golden-test", **(headers or {})})
        if raw is not None:
            self.body = raw
        else:
            d = {"model": "qwen-local", "messages": [{"role": "user", "content": "review this function please"}],
                 "max_tokens": 4000}
            d.update(fields or {})
            self.body = json.dumps(d).encode()
        self.gone_after_sleeps = None
        self._clock = None

    async def read(self):
        return self.body

    @property
    def transport(self):
        if self.gone_after_sleeps is None:
            return "absent"
        if self._clock.sleeps >= self.gone_after_sleeps:
            return None
        return types.SimpleNamespace(is_closing=lambda: False)


def _isolated_env(tmp):
    keep = {k: v for k, v in os.environ.items() if not k.startswith(("SHIM_", "GATEWAY_"))}
    keep.update({
        "SHIM_SPEND_FILE": f"{tmp}/gateway-spend.json",
        "SHIM_SPEND_CLIENTS_FILE": f"{tmp}/gateway-spend-clients.json",
        "SHIM_STATS_FILE": f"{tmp}/gateway-stats.json",
        "SHIM_TELEMETRY_DIR": f"{tmp}/telemetry",
        "SHIM_ENV_FILE": f"{tmp}/shim.env",
        "SHIM_ALIASES_FILE": f"{tmp}/gateway-aliases.json",
        "SHIM_FLIGHTREC_DIR": f"{tmp}/flightrec",
        "SHIM_REMOTE_DEAD_FILE": f"{tmp}/remote-dead-until.json",
        "GATEWAY_DRAIN_LEDGER": f"{tmp}/drains.jsonl",
        "GATEWAY_TIMEOUT_LEDGER": f"{tmp}/timeouts.jsonl",
        "SHIM_EXACT_TOKENS": "0",
    })
    for k, v in list(keep.items()):
        if k.startswith("SHIM_") and k.endswith(("_FILE", "_DIR", "_LEDGER")) and not v.startswith(tmp):
            keep.pop(k)
    return keep


def _fresh_shim(tmp, n):
    with patch.dict(os.environ, _isolated_env(tmp), clear=True):
        spec = importlib.util.spec_from_file_location(f"shim_golden_{n}", str(SHIM_PATH))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    mod._CLIENT_NAMES_FILE = f"{tmp}/clients.map"
    return mod


def _sha(b):
    return hashlib.sha256(b if isinstance(b, (bytes, bytearray)) else json.dumps(b).encode()).hexdigest()[:16]


_SYNTH_ID = re.compile(rb"chatcmpl-synth-[0-9a-f]+")


def _resp_summary(resp):
    if isinstance(resp, web.Response) and isinstance(resp.body, (bytes, bytearray)):
        resp.body = _SYNTH_ID.sub(b"chatcmpl-synth-X", bytes(resp.body))   # per-call random id
    if isinstance(resp, web.StreamResponse) and not isinstance(resp, web.Response):
        return {"kind": "stream", "status": resp.status}
    if not isinstance(resp, web.Response):
        return {"kind": type(resp).__name__}
    out = {"status": resp.status}
    hdrs = {k: v for k, v in resp.headers.items() if k.lower().startswith("x-gateway") or k == "Retry-After"}
    if hdrs:
        out["headers"] = hdrs
    try:
        j = json.loads(resp.body)
    except Exception:
        j = None
    if isinstance(j, dict):
        err = j.get("error")
        if isinstance(err, dict):
            out["error"] = {k: err.get(k) for k in ("type", "code") if err.get(k) is not None}
        elif err is not None:
            out["error"] = str(err)[:80]
        else:
            out["body"] = j if len(json.dumps(j)) < 300 else _sha(json.dumps(j, sort_keys=True).encode())
    elif resp.body is not None:
        out["body_sha"] = _sha(resp.body if isinstance(resp.body, bytes) else str(resp.body).encode())
    return out


LOCAL_OK = {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 3}}
LOCAL_EMPTY = {"choices": [{"message": {"role": "assistant", "content": "", "reasoning_content": "hmm" * 50},
                            "finish_reason": "length"}], "usage": {"prompt_tokens": 50, "completion_tokens": 400}}


def run_scenario(sc, n):
    with tempfile.TemporaryDirectory(prefix="gw-golden-") as tmp:      # managed: the ledger ratchet checks
        return _run_scenario(sc, n, tmp)


def _run_scenario(sc, n, tmp):
    m = _fresh_shim(tmp, n)
    ref = [m]
    clock = FakeClock()
    m.time = _time_proxy(clock, m.time)
    m.asyncio = _asyncio_proxy(clock, ref)
    st = dict(healthy=True, remote_ok=True, spend=True, auto_budget=True)
    st.update(sc.get("edges", {}))
    m._health.update(ok=st["healthy"], at=clock.now)
    calls = []
    relay_plan = list(sc.get("relay", ["ok"]))
    forward_plan = list(sc.get("forward", ["ok"]))

    async def relay(request, base, path, body, key, streaming, concurrency=1, provider_name=None):
        calls.append({"to": "local" if base == m.LOCAL else str(base), "path": path, "body": _sha(body),
                      "stream": bool(streaming), "conc": concurrency, "key": bool(key)})
        what = relay_plan.pop(0) if relay_plan else "ok"
        if what == "ok":
            m._active_set(request, outtok=3, ptok_exact_local=50, http_status=200)
            return "ok", web.json_response(LOCAL_OK)
        if what == "empty":
            return "ok", web.json_response(LOCAL_EMPTY)
        if what == "oom":
            return "err", (503, "CUDA out of memory", True)
        return "err", (500, "engine error", False)

    async def forward_remote(request, path, body, streaming, endpoint=None, model=None, **kw):
        calls.append({"to": "remote", "path": path, "body": _sha(body), "stream": bool(streaming),
                      "endpoint": (endpoint or {}).get("base") if isinstance(endpoint, dict) else endpoint,
                      "model": model, **({"kw": sorted(kw)} if kw else {})})
        what = forward_plan.pop(0) if forward_plan else "ok"
        if what == "refused":
            return web.json_response({"error": {"type": "spend_refused"}}, status=429,
                                     headers={"X-Gateway-Spend-Refused": "1"})
        return web.json_response({"remote": True})

    async def local_healthy():
        return m._health["ok"]

    flightrec = []
    m._relay = relay
    m._forward_remote = forward_remote
    m.local_healthy = local_healthy
    m.remote_ok = lambda: st["remote_ok"]
    m._spend_allows_overflow = lambda p, mt: st["spend"]
    m._automatic_remote_budget_allows = lambda p, mt: st["auto_budget"]
    m.is_peak = lambda: False
    m._write_flightrec = lambda fr, fn, body: flightrec.append(_sha(body))
    for name, value in sc.get("patch", {}).items():
        if callable(value) and getattr(value, "_needs_module", False):
            value = value(m)
        setattr(m, name, value)
    for k, fn in sc.get("setup", {}).items():
        fn(m)
    for at, fn in sc.get("hooks", {}).items():
        clock.hooks[int(at)] = fn

    base_inflight = (m._inflight, m._inflight_tokens, m._inflight_reserved_tokens, m._inflight_computed)
    req = Request(**sc.get("req", {}))
    req._clock = clock
    if sc.get("gone_after") is not None:
        req.gone_after_sleeps = sc["gone_after"]
    m._JSONL_PENDING.clear()
    m._events.clear()
    outcome = {}
    try:
        resp = asyncio.run(m.handle_completions(req))
        outcome["response"] = _resp_summary(resp)
    except _Spin:
        outcome["response"] = {"kind": "spins", "note": f"still queued after {MAX_SLEEPS} polls"}
    except Exception as e:                                   # pinned too: today's unhandled paths
        outcome["response"] = {"kind": "raised", "exc": type(e).__name__, "msg": str(e)[:120]}
    outcome["calls"] = calls
    outcome["events"] = [{k: e.get(k) for k in ("d", "r", "units", "waited", "cls", "ptok", "maxtok", "stream",
                                                  "alias", "alias_kind", "cost_est")} for e in list(m._events)][::-1]
    rows = list(m._JSONL_PENDING)
    outcome["telemetry"] = [{k: v for k, v in sorted(r.items()) if k not in _ROW_VOLATILE} for r in rows]
    outcome["counters"] = {"inflight": m._inflight, "inflight_tokens": m._inflight_tokens,
                           "inflight_reserved": m._inflight_reserved_tokens, "inflight_computed": m._inflight_computed,
                           "waiting": m._waiting, "waiting_by_class": dict(m._waiting_by_class),
                           "active_left": len(m._ACTIVE), "pm_inflight_left": len(m._PM_INFLIGHT),
                           "baseline_inflight": list(base_inflight)}
    outcome["queued_s"] = round(clock.now - T0, 3)
    outcome["polls"] = clock.sleeps
    if flightrec:
        outcome["flightrec"] = flightrec
    return outcome


# ---------------------------------------------------------------------------------------------------------------
# Scenario matrix. Names are stable keys in the golden file.
# ---------------------------------------------------------------------------------------------------------------
def _msgs(chars, role="user"):
    return [{"role": role, "content": "x" * int(chars)}]


def _sat(m):                                    # every lane busy
    m._inflight = m.effective_budget() + 4


def _bg_boundary(m):                            # background lanes exactly full, interactive lanes still free
    m._inflight = max(1, m.effective_budget() - m.FG_RESERVED)


def _fg_boundary(m):                            # one lane short of the interactive limit
    m._inflight = m.admission_lane_limit(False, m.effective_budget(), m.FG_RESERVED,
                                         tiny_extra_lanes=m.TINY_EXTRA_LANES) - 1


def _free(m):
    m._inflight = 0


def _alias(name, kind, **kw):
    def setup(m):
        m._ALIASES[name] = {"name": name, "kind": kind, "enabled": True, **kw}
    return setup


def _offline(m):
    m._OFFLINE.update(until=T0 + 600, reason="golden-bench", since=T0, by="golden")


def _health_down(m):
    m._health.update(ok=False)


def _health_up(m):
    m._health.update(ok=True)


def _fresh_engine(m):
    m._LAST_LOCAL_OK = T0 - 5


def _big_tokens_inflight(m):
    m._inflight_tokens = 10 ** 7


BIG = 200_000                                     # chars: ~57K tokens at the default 3.5 chars/token
HUGE = 2_400_000                                  # chars: beyond local + remote context
IMG = [{"role": "user", "content": [{"type": "text", "text": "what is this"},
                                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
PROBE = [{"role": "user", "content": "Reply with the single word: pong"}]

SCENARIOS = {
    # --- refusals before any routing --------------------------------------------------------------------
    "prompt_exceeds_all_providers": dict(req=dict(fields=dict(messages=_msgs(HUGE)))),
    "garbage_body": dict(req=dict(raw=b"not json at all")),
    "json_array_body": dict(req=dict(raw=b"[1,2,3]")),
    # --- probe synth ----------------------------------------------------------------------------------
    "probe_fresh_engine_synth": dict(req=dict(fields=dict(model="estate-local", messages=PROBE, max_tokens=8)),
                                     setup=dict(a=_fresh_engine)),
    # pinned as-is: kind "native"/"default" (model qwen-local / estate) is never synthesized (see lane SH report)
    "probe_native_model_not_synth": dict(req=dict(fields=dict(messages=PROBE, max_tokens=8)), setup=dict(a=_fresh_engine)),
    "probe_estate_default_not_synth": dict(req=dict(fields=dict(model="estate", messages=PROBE, max_tokens=8)),
                                           setup=dict(a=_fresh_engine)),
    "probe_stale_engine_real": dict(req=dict(fields=dict(model="estate-local", messages=PROBE, max_tokens=8))),
    "probe_fresh_stream_synth": dict(req=dict(fields=dict(model="estate-local", messages=PROBE, max_tokens=8, stream=True)),
                                     setup=dict(a=_fresh_engine)),
    # --- aliases --------------------------------------------------------------------------------------
    "alias_estate_remote": dict(req=dict(fields=dict(model="estate-remote"))),
    "alias_estate_local": dict(req=dict(fields=dict(model="estate-local"))),
    "alias_custom_remote": dict(req=dict(fields=dict(model="golden-custom")),
                                setup=dict(a=_alias("golden-custom", "custom-remote",
                                                    endpoint={"base": "https://custom.invalid/v1", "model": "m1"}))),
    "alias_disabled": dict(req=dict(fields=dict(model="golden-off")), setup=dict(a=_alias("golden-off", "custom-remote", enabled=False))),
    "local_only_refuses_remote_alias": dict(req=dict(fields=dict(model="estate-remote")), patch=dict(LOCAL_ONLY=1)),
    # --- text-only guard ------------------------------------------------------------------------------
    "image_no_vision_400": dict(req=dict(fields=dict(messages=IMG))),
    "image_vision_remote": dict(req=dict(fields=dict(messages=IMG)), patch=dict(REMOTE_VISION=1)),
    "image_vision_cap_exhausted": dict(req=dict(fields=dict(messages=IMG)), patch=dict(REMOTE_VISION=1),
                                       edges=dict(spend=False)),
    "image_vision_pinned_400": dict(req=dict(fields=dict(messages=IMG), headers={"X-Client": "coder-local-pin"}),
                                    patch=dict(REMOTE_VISION=1)),
    "image_vision_no_remote_400": dict(req=dict(fields=dict(messages=IMG)), patch=dict(REMOTE_VISION=1),
                                       edges=dict(remote_ok=False)),
    # --- forced window / intent -----------------------------------------------------------------------
    "force_remote_interactive": dict(patch=dict(FORCE_REMOTE=1, FORCE_REMOTE_UNTIL_EPOCH=T0 + 600)),
    "force_remote_lease_expired_local": dict(patch=dict(FORCE_REMOTE=1, FORCE_REMOTE_UNTIL_EPOCH=T0 - 1)),
    "force_remote_background_rejected": dict(req=dict(headers={"X-Client": "cron-job"}),
                                             patch=dict(FORCE_REMOTE=1, FORCE_REMOTE_UNTIL_EPOCH=T0 + 600)),
    "force_remote_estate_local_stays_local": dict(req=dict(fields=dict(model="estate-local")),
                                                  patch=dict(FORCE_REMOTE=1, FORCE_REMOTE_UNTIL_EPOCH=T0 + 600)),
    "route_intent_remote": dict(req=dict(headers={"X-Gateway-Route-Intent": "remote"})),
    "route_intent_remote_local_pin": dict(req=dict(headers={"X-Gateway-Route-Intent": "remote",
                                                            "X-Client": "x-local-pin"})),
    "route_intent_no_auto_budget": dict(req=dict(headers={"X-Gateway-Route-Intent": "overflow"}),
                                        edges=dict(auto_budget=False)),
    # --- size / offline / predictive overflows --------------------------------------------------------
    "size_cap_remote": dict(patch=dict(MAX_LOCAL_TOKENS=1000, LOCAL_CONTEXT_LIMIT=1000),
                            req=dict(fields=dict(messages=_msgs(20_000), max_tokens=4000))),
    "offline_window_remote": dict(setup=dict(a=_offline)),
    "offline_window_no_remote_503": dict(setup=dict(a=_offline), edges=dict(remote_ok=False)),
    "offline_window_local_pin_503": dict(setup=dict(a=_offline), req=dict(headers={"X-Client": "a-local-pin"})),
    "offline_window_brake_bypassed": dict(setup=dict(a=_offline), edges=dict(auto_budget=False)),
    "big_out_idle_kept_local": dict(req=dict(fields=dict(max_tokens=65536))),
    "big_out_saturated_kevin_waits": dict(req=dict(fields=dict(max_tokens=65536)), setup=dict(a=_sat),
                                          hooks={300: _free}),
    "big_out_lf_off_remote": dict(req=dict(fields=dict(max_tokens=65536)), patch=dict(LOCAL_FIRST=False)),
    "big_out_saturated_bg_remote": dict(req=dict(fields=dict(max_tokens=65536), headers={"X-Client": "cron-job"}),
                                        setup=dict(a=_sat)),
    "big_prompt_idle_kept_local": dict(req=dict(fields=dict(messages=_msgs(BIG)))),
    "big_prompt_saturated_kevin_waits": dict(req=dict(fields=dict(messages=_msgs(BIG))), setup=dict(a=_sat),
                                             hooks={300: _free}),
    "big_prompt_saturated_bg_remote": dict(req=dict(fields=dict(messages=_msgs(BIG)), headers={"X-Client": "cron-job"}),
                                           setup=dict(a=_sat)),
    "big_prompt_local_first_off_remote": dict(req=dict(fields=dict(messages=_msgs(BIG))), patch=dict(LOCAL_FIRST=False)),
    "big_prompt_local_pin_local": dict(req=dict(fields=dict(messages=_msgs(BIG)), headers={"X-Client": "a-local-pin"}),
                                       patch=dict(LOCAL_FIRST=False)),
    "big_prompt_no_overflow_budget": dict(req=dict(fields=dict(messages=_msgs(BIG))), patch=dict(LOCAL_FIRST=False),
                                          edges=dict(spend=False)),
    "monster_inflight_saturated_kevin_waits": dict(setup=dict(a=_big_tokens_inflight, b=_sat), hooks={300: _free}),
    "monster_inflight_lf_off_remote": dict(setup=dict(a=_big_tokens_inflight), patch=dict(LOCAL_FIRST=False)),
    "perf_breaker_lf_off_remote": dict(patch=dict(perf_breaker_active=lambda: True, LOCAL_FIRST=False)),
    "perf_breaker_saturated_bg_remote": dict(setup=dict(b=_sat), patch=dict(perf_breaker_active=lambda: True),
                                             req=dict(headers={"X-Client": "cron-job"})),
    "perf_breaker_idle_local": dict(patch=dict(perf_breaker_active=lambda: True)),
    "predicted_occupancy_lf_off_remote": dict(patch=dict(predicted_occupancy_seconds=lambda *a, **k: 10_000.0,
                                                         LOCAL_FIRST=False)),
    "predicted_occupancy_idle_kept_local": dict(patch=dict(predicted_occupancy_seconds=lambda *a, **k: 10_000.0)),
    # --- local down -----------------------------------------------------------------------------------
    "local_down_interactive_remote": dict(edges=dict(healthy=False)),
    "local_down_estate_local_503": dict(edges=dict(healthy=False), req=dict(fields=dict(model="estate-local"))),
    "local_down_cap_exhausted_503": dict(edges=dict(healthy=False, spend=False)),
    "local_down_no_remote_spins": dict(edges=dict(healthy=False, remote_ok=False)),
    "local_down_no_remote_recovers": dict(edges=dict(healthy=False, remote_ok=False), hooks={80: _health_up}),
    "local_down_bg_held_recovers": dict(edges=dict(healthy=False), req=dict(headers={"X-Client": "cron-job"}),
                                        hooks={40: _health_up}),
    "local_down_bg_held_rejected": dict(edges=dict(healthy=False), req=dict(headers={"X-Client": "cron-job"})),
    "local_down_brake_bypassed_remote": dict(edges=dict(healthy=False, auto_budget=False)),
    # --- context / reservation ------------------------------------------------------------------------
    "size_cap_via_context_limit_remote": dict(patch=dict(LOCAL_CONTEXT_LIMIT=2000, MAX_LOCAL_TOKENS=10 ** 6,
                                                         local_context_limit=lambda: 2000),
                                              req=dict(fields=dict(messages=_msgs(30_000), max_tokens=500))),
    "context_too_big_local_remote": dict(patch=dict(local_context_limit=lambda: 2000, max_local_tokens=lambda: 10 ** 6),
                                         req=dict(fields=dict(messages=_msgs(30_000), max_tokens=500))),
    "context_too_big_estate_local_error": dict(patch=dict(local_context_limit=lambda: 2000,
                                                          max_local_tokens=lambda: 10 ** 6),
                                               req=dict(fields=dict(model="estate-local", messages=_msgs(30_000),
                                                                    max_tokens=500))),
    "token_reservation_over_budget_remote": dict(patch=dict(token_budget=lambda: 100)),
    "token_reservation_over_budget_no_remote": dict(patch=dict(token_budget=lambda: 100), edges=dict(remote_ok=False)),
    # --- tiny lane ------------------------------------------------------------------------------------
    "tiny_idle_local": dict(req=dict(fields=dict(max_tokens=50))),
    "tiny_stream_local": dict(req=dict(fields=dict(max_tokens=50, stream=True))),
    "tiny_full_remote_tiny_fast": dict(req=dict(fields=dict(max_tokens=50)), setup=dict(a=_sat)),
    "tiny_full_no_remote_waits_then_admits": dict(req=dict(fields=dict(max_tokens=50)), setup=dict(a=_sat),
                                                  edges=dict(remote_ok=False), hooks={30: _free}),
    "tiny_local_fails_failover": dict(req=dict(fields=dict(max_tokens=50)), relay=["fail"]),
    "tiny_local_oom_failover": dict(req=dict(fields=dict(max_tokens=50)), relay=["oom"]),
    "tiny_local_fails_estate_local_503": dict(req=dict(fields=dict(max_tokens=50, model="estate-local")),
                                              relay=["fail"]),
    "tiny_local_fails_no_remote_503": dict(req=dict(fields=dict(max_tokens=50)), relay=["fail"],
                                           edges=dict(remote_ok=False)),
    # --- normal admission -----------------------------------------------------------------------------
    "normal_idle_local": dict(),
    "normal_idle_stream_local": dict(req=dict(fields=dict(stream=True))),
    "normal_background_idle_local": dict(req=dict(headers={"X-Client": "cron-job"})),
    "normal_bg_marker_idle_local": dict(req=dict(fields=dict(messages=[{"role": "user",
                                                                        "content": "scheduled cron job: tidy"}]))),
    "normal_full_interactive_waits_then_admits": dict(setup=dict(a=_sat), hooks={120: _free}),
    "normal_full_interactive_never_overflow_off": dict(setup=dict(a=_sat), patch=dict(INTERACTIVE_NEVER_OVERFLOW=0)),
    "normal_full_background_overflows": dict(setup=dict(a=_sat), req=dict(headers={"X-Client": "cron-job"})),
    "normal_full_background_no_remote_waits": dict(setup=dict(a=_sat), req=dict(headers={"X-Client": "cron-job"}),
                                                   edges=dict(remote_ok=False), hooks={200: _free}),
    "normal_full_estate_local_waits_then_admits": dict(setup=dict(a=_sat), req=dict(fields=dict(model="estate-local")),
                                                       hooks={60: _free}),
    "normal_queued_client_gone_499": dict(setup=dict(a=_sat), gone_after=25),
    "normal_queued_local_dies_valve": dict(setup=dict(a=_sat), hooks={20: _health_down}),
    "normal_local_fails_failover": dict(relay=["fail"]),
    "normal_local_oom_failover": dict(relay=["oom"]),
    "normal_local_fails_estate_local_503": dict(relay=["fail"], req=dict(fields=dict(model="estate-local"))),
    "normal_local_fails_no_remote_503": dict(relay=["fail"], edges=dict(remote_ok=False)),
    "normal_empty_thinking_retry": dict(relay=["empty", "ok"]),
    "normal_empty_thinking_retry_still_empty": dict(relay=["empty", "empty"]),
    "normal_empty_retry_stream_not_retried": dict(relay=["empty"], req=dict(fields=dict(stream=True))),
    "overflow_spend_refused_reenters_local": dict(req=dict(fields=dict(messages=_msgs(BIG))),
                                                  patch=dict(LOCAL_FIRST=False), forward=["refused"]),
    "failover_spend_refused_503": dict(relay=["fail"], forward=["refused"]),
    "flightrec_big_local_body": dict(req=dict(fields=dict(messages=_msgs(60_000)))),
    "halo_control_turn_local": dict(req=dict(fields=dict(model="estate-local"), headers={"X-Client": "halo-hermes"},
                                             remote="10.0.1.95"), setup=dict(a=_sat), hooks={50: _free}),
    "kevin_librechat_brake_bypass": dict(req=dict(fields=dict(messages=_msgs(BIG)), headers={"X-Client": "librechat"}),
                                         patch=dict(LOCAL_FIRST=False), edges=dict(auto_budget=False)),
    "unlabelled_brake_applies_local": dict(req=dict(fields=dict(messages=_msgs(BIG))),
                                           patch=dict(LOCAL_FIRST=False), edges=dict(auto_budget=False)),
    "flow_mode_off_idle_local": dict(patch=dict(FLOW_MODE="off")),
    "flow_mode_shadow_full_bg": dict(patch=dict(FLOW_MODE="shadow"), setup=dict(a=_sat),
                                     req=dict(headers={"X-Client": "cron-job"})),
    # --- lane-limit boundaries ------------------------------------------------------------------------
    "bg_at_reserved_boundary_overflows": dict(setup=dict(a=_bg_boundary), req=dict(headers={"X-Client": "cron-job"})),
    "fg_at_bg_boundary_admitted": dict(setup=dict(a=_bg_boundary)),
    "bg_boundary_budget8_overflows": dict(patch=dict(effective_budget=lambda: 8), setup=dict(a=_bg_boundary),
                                          req=dict(headers={"X-Client": "cron-job"})),
    "bg_below_boundary_budget8_admitted": dict(patch=dict(effective_budget=lambda: 8),
                                               setup=dict(a=lambda m: setattr(m, "_inflight", 5)),
                                               req=dict(headers={"X-Client": "cron-job"})),
    "fg_boundary_budget8_admitted": dict(patch=dict(effective_budget=lambda: 8), setup=dict(a=_fg_boundary)),
    "fg_one_below_limit_admitted": dict(setup=dict(a=_fg_boundary)),
    "tiny_at_fg_limit_uses_extra_lane": dict(setup=dict(a=_fg_boundary), req=dict(fields=dict(max_tokens=50))),
    "halo_at_full_takes_last_place": dict(req=dict(fields=dict(model="estate-local"), headers={"X-Client": "halo-hermes"},
                                                   remote="10.0.1.95"),
                                          setup=dict(a=lambda m: setattr(m, "_inflight", m.effective_budget() + m.TINY_EXTRA_LANES - 1))),
    # --- local-pin never pays for congestion ----------------------------------------------------------
    **{f"local_pin_{k}_stays_local": dict(req=dict(fields=f, headers={"X-Client": "coder-local-pin"}),
                                          patch=dict(LOCAL_FIRST=False, **p), setup=su)
       for k, f, p, su in (
           ("big_out", dict(max_tokens=65536), {}, {}),
           ("size", dict(messages=_msgs(20_000)), dict(MAX_LOCAL_TOKENS=1000, LOCAL_CONTEXT_LIMIT=1000), {}),
           ("monster", {}, {}, dict(a=_big_tokens_inflight)),
           ("perf", {}, dict(perf_breaker_active=lambda: True), {}),
           ("predicted", {}, dict(predicted_occupancy_seconds=lambda *a, **k: 10_000.0), {}),
           ("tokens", {}, dict(token_budget=lambda: 100), {}),
       )},
    "idempotency_key_identity": dict(req=dict(headers={"Idempotency-Key": "golden-key-1"})),
    "completions_endpoint": dict(req=dict(path="/v1/completions", fields=dict(prompt="hello", messages=None))),
}


def record_all():
    return {name: run_scenario(sc, i) for i, (name, sc) in enumerate(SCENARIOS.items())}


class GoldenRouting(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        if os.environ.get("SH_GOLDEN_REGEN") == "1":
            GOLDEN.parent.mkdir(exist_ok=True)
            GOLDEN.write_text(json.dumps(record_all(), indent=1, sort_keys=True) + "\n")
        cls.golden = json.loads(GOLDEN.read_text())

    def test_matrix_is_complete(self):
        self.assertEqual(sorted(self.golden), sorted(SCENARIOS), "re-record: scenarios added/removed")

    def test_every_scenario_matches_golden(self):
        for i, (name, sc) in enumerate(SCENARIOS.items()):
            with self.subTest(scenario=name):
                got = json.loads(json.dumps(run_scenario(sc, 1000 + i), sort_keys=True))
                self.assertEqual(got, self.golden.get(name), name)

    def test_admission_counters_always_return_to_zero(self):
        """Invariant pinned on top of the golden: no path leaks a lane, tokens, a queue slot or a registry entry."""
        for name, out in self.golden.items():
            if out["response"].get("kind") == "spins":
                continue
            c = out["counters"]
            with self.subTest(scenario=name):
                self.assertEqual(c["waiting"], 0)
                self.assertEqual(sum(c["waiting_by_class"].values()), 0)
                self.assertEqual(c["active_left"], 0)
                self.assertEqual(c["pm_inflight_left"], 0)

    # Pinned bugs: today these paths give the caller no HTTP response of the gateway's own (an unhandled exception
    # becomes aiohttp's bare 500 and the request never reaches telemetry). Each fix removes its name from this set.
    KNOWN_NO_RESPONSE = {"json_array_body"}

    def test_every_request_gets_a_response(self):
        """A "spins" scenario is a deliberate unbounded wait (no remote to overflow to, interactive never overflows on
        a timeout); the caller hanging up ends it (normal_queued_client_gone_499)."""
        for name, out in self.golden.items():
            if name in self.KNOWN_NO_RESPONSE or out["response"].get("kind") == "spins":
                continue
            with self.subTest(scenario=name):
                self.assertIn("status", out["response"], out["response"])


if __name__ == "__main__":
    unittest.main()
