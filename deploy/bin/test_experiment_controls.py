"""Offline regressions for experiment identity, leases and completed drains."""
import asyncio
import importlib.util
import json
import pathlib
import sys
import io
import subprocess

import pytest

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import experiment_controls as controls
import windowctl as wc


def test_identity_binds_process_model_flags_without_exposing_credentials(tmp_path):
    p = tmp_path / "9"
    p.mkdir()
    fields = ["S"] + ["0"] * 19
    fields[19] = "123"
    (p / "stat").write_text("9 (engine) " + " ".join(fields))
    (p / "cmdline").write_bytes(b"python\0vllm\0serve\0/models/q\0--max-model-len\0" + b"524288\0--long-prefill-token-threshold\x0032768\0--api-key\0private\0")
    (p / "environ").write_bytes(b"VLLM_GDN_SR=1\0VLLM_SCHED_SHORT_FIRST_PREFIX_AWARE=1\0VLLM_SCHED_PREFILL_SHARE_EXACT=0\0SHIM_ADMIN_TOKEN=private\0")
    (p / "cwd").symlink_to(tmp_path)
    identity = controls.runtime_identity(proc=str(tmp_path), pid=9)
    assert identity["model_path"] == "/models/q"
    assert identity["flags"]["--max-model-len"] == "524288"
    assert identity["flags"]["--long-prefill-token-threshold"] == "32768"
    assert identity["runtime_flags"] == {"VLLM_GDN_SR": "1", "VLLM_SCHED_SHORT_FIRST_PREFIX_AWARE": "1", "VLLM_SCHED_PREFILL_SHARE_EXACT": "0"}
    assert "private" not in json.dumps(identity)
    original = identity["fingerprint"]
    (p / "environ").write_bytes(b"VLLM_GDN_SR=0\0")
    assert controls.runtime_identity(proc=str(tmp_path), pid=9)["fingerprint"] != original
    (p / "stat").unlink()
    assert controls.runtime_identity(proc=str(tmp_path), pid=9) is None


@pytest.mark.parametrize("gateway,engine", [({}, {}), ({"local_active": None}, {"running": 0, "waiting": 0}),
    ({"local_active": 0}, None), ({"local_active": 1}, {"running": 0, "waiting": 0}),
    ({"local_active": 0}, {"running": 1, "waiting": 0})])
def test_unknown_or_busy_counts_do_not_prove_a_drain(gateway, engine):
    assert not controls.drain_complete(gateway, engine)


def test_drain_metrics_include_all_engine_labels(monkeypatch):
    metrics = (b'vllm:num_requests_running{engine="0"} 0\n'
               b'vllm:num_requests_running{engine="1"} 1\n'
               b'vllm:num_requests_waiting{engine="0"} 0\n'
               b'vllm:num_requests_waiting{engine="1"} 2\n')
    monkeypatch.setattr(controls.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(metrics))
    assert controls.engine_inflight("http://test") == {"running": 1, "waiting": 2}


@pytest.mark.parametrize("load,active,want", [("loaded", "active", "active"), ("loaded", "inactive", "inactive"),
                                             ("not-found", "inactive", "unknown"), ("loaded", "failed", "unknown")])
def test_user_service_inventory_uses_user_scope_and_refuses_unknown_state(monkeypatch, load, active, want):
    commands = []
    def run(argv, **kwargs):
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 0, f"ActiveState={active}\nLoadState={load}\n", "")
    monkeypatch.setattr(wc.subprocess, "run", run)
    assert wc.user_unit_state("estate-core.service") == want
    assert commands[0][:4] == ["systemctl", "--user", "show", "estate-core.service"]


def test_phase_revoked_before_drain_and_requires_two_zero_samples(monkeypatch, tmp_path):
    events, samples = [], iter([{"running": 1, "waiting": 0}, {"running": 0, "waiting": 0}, {"running": 0, "waiting": 0}])
    token = tmp_path / "token"
    token.write_text("private")
    state = {"window": "test", "lease": "offline", "measurement": {}, "measurement_token_file": str(token)}
    state["measurement"] = {"phase": "test"}
    def http(url, method="GET", payload=None):
        events.append(method)
        return {"active": False} if method == "DELETE" else {"offline": True, "local_active": 0}
    monkeypatch.setattr(wc, "http_json", http)
    monkeypatch.setattr(wc, "engine_inflight", lambda url: next(samples))
    wc.revoke_and_drain(state, 5, sleep=lambda s: None)
    assert events == ["DELETE", "GET", "GET", "GET"]
    assert not token.exists() and state["measurement"] is None


def test_unreadable_metrics_prevent_restart(monkeypatch):
    monkeypatch.setattr(wc, "http_json", lambda *a, **k: {"offline": True, "local_active": 0})
    monkeypatch.setattr(wc, "engine_inflight", lambda url: None)
    monkeypatch.setattr(wc, "main_pid", lambda unit: 9)
    ticks = iter([0, 1, 3])
    with pytest.raises(wc.WindowAbort, match="unreadable"):
        wc.revoke_and_drain({"lease": "offline"}, 1, clock=lambda: next(ticks), sleep=lambda s: None)


@pytest.fixture
def shim(monkeypatch):
    spec = importlib.util.spec_from_file_location("measurement_test_shim", HERE / "keepalive-shim.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    identity = {"engine_pid": 9, "pid_start": "123", "root": "/root/test", "model_path": "/models/q",
                "flags": {}, "runtime_flags": {}, "fingerprint": "known"}
    monkeypatch.setattr(module, "measurement_runtime_identity", lambda *a, **k: identity.copy())
    monkeypatch.setattr(module, "SHIM_ADMIN_TOKEN", "admin")
    monkeypatch.setattr(module, "_draining", lambda: False)
    monkeypatch.setattr(module, "_offline_status", lambda: {"local_active": 0})
    module._OFFLINE.update(until=module.time.time() + 300, lease="offline")
    module._health["ok"] = True
    return module, identity


class Request(dict):
    def __init__(self, method, body=None, headers=None):
        super().__init__()
        self.method, self.body, self.headers = method, body, headers or {}
    async def json(self):
        return self.body


def grant(module, identity, **over):
    body = {"window": "w", "phase": "a", "ttl_s": 60, "offline_lease": "offline", "runtime_identity": identity}
    body.update(over)
    response = asyncio.run(module.gateway_measurement(Request("POST", body, {"X-Admin-Token": "admin"})))
    return response, json.loads(response.text)


def test_measurement_requires_configured_admin_and_exact_identity(shim):
    module, identity = shim
    response = asyncio.run(module.gateway_measurement(Request("GET")))
    assert response.status == 401
    response, _ = grant(module, {**identity, "fingerprint": "wrong"})
    assert response.status == 409 and not module._MEASUREMENT
    module.SHIM_ADMIN_TOKEN = ""
    response = asyncio.run(module.gateway_measurement(Request("POST", {})))
    assert response.status == 401


def test_only_token_bearing_estate_local_can_use_phase_and_revocation_is_immediate(shim):
    module, identity = shim
    response, receipt = grant(module, identity)
    assert response.status == 200
    request = Request("POST", headers={"X-Window-Token": receipt["token"], "X-Client": "benchmark"})
    request["measurement_body"] = {"model": "estate-local"}
    assert not module._request_local_offline(request)
    assert module._request_local_offline(Request("POST", headers={"X-Client": "benchmark"}))
    request["measurement_body"]["model"] = "estate"
    assert module._request_local_offline(request)
    request["measurement_body"]["model"] = "estate-local"
    response = asyncio.run(module.gateway_measurement(Request("DELETE", {"window": "w"}, {"X-Admin-Token": "admin"})))
    assert response.status == 200 and module._request_local_offline(request)
    assert "token" not in module._measurement_receipt()


def test_phase_renewal_preserves_token_and_never_moves_the_deadline(shim, monkeypatch):
    module, identity = shim
    now = [1000.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    module._OFFLINE["until"] = 5000
    _, original = grant(module, identity, ttl_s=60, max_s=120)
    now[0] += 30
    response, renewed = grant(module, identity, ttl_s=180, renew=True)
    assert response.status == 200 and "token" not in renewed
    assert renewed["until"] == renewed["deadline"] == 1120
    assert module._MEASUREMENT["token"] == original["token"]
    now[0] = 1120
    response, _ = grant(module, identity, renew=True)
    assert response.status == 409


@pytest.mark.parametrize("fault", [None, "empty", "identity", "length", "zero-tokens"])
def test_restoration_requires_completed_generation_from_the_same_runtime(monkeypatch, fault):
    identity = {"engine_pid": 9, "pid_start": "123", "root": "/root", "model_path": "/models/q",
                "flags": {}, "runtime_flags": {}, "argv_sha256": "argv"}
    observed = iter([identity, {**identity, "pid_start": "new"} if fault == "identity" else identity])
    monkeypatch.setattr(wc, "runtime_identity", lambda unit: next(observed))
    captured = []
    def urlopen(request, timeout):
        if isinstance(request, str):
            return io.StringIO(json.dumps({"data": [{"id": "served-model"}]}))
        captured.append(json.loads(request.data))
        return io.StringIO(json.dumps({"choices": [{"message": {"content": "" if fault == "empty" else "CONTROL_OK"},
                    "finish_reason": "length" if fault == "length" else "stop"}],
                    "usage": {"completion_tokens": 0 if fault == "zero-tokens" else 3}}))
    monkeypatch.setattr(wc.urllib.request, "urlopen", urlopen)
    report = wc.restoration_generation(identity)
    assert report["ok"] is (fault is None)
    assert captured[0]["model"] == "served-model" and captured[0]["max_tokens"] == 32


def test_phase_expires_and_process_change_rejects_old_token(shim, monkeypatch):
    module, identity = shim
    _, receipt = grant(module, identity)
    request = Request("POST", headers={"X-Window-Token": receipt["token"]})
    request["measurement_body"] = {"model": "estate-local"}
    module._MEASUREMENT["until"] = module.time.time() - 1
    assert module._request_local_offline(request)
    module._MEASUREMENT["until"] = module.time.time() + 60
    monkeypatch.setattr(module, "measurement_runtime_identity", lambda **k: {**identity, "pid_start": "new"})
    assert module._request_local_offline(request)
