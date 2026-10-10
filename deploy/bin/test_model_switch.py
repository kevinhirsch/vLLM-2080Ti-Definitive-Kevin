"""Model switches use private checkpoints/runtime stubs, never the live runtime."""
import fcntl
import io
import json
from pathlib import Path
import subprocess

import pytest

import model_switch as switch


@pytest.fixture
def private_runtime(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    models = tmp_path / "models with spaces"
    for name in list(switch.CHECKPOINTS.values())[:3]:
        model = models / name
        model.mkdir(parents=True)
        method = "gptq" if name == switch.CHECKPOINTS["default"] else "compressed-tensors"
        (model / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "quantization_config": {"quant_method": method}}))
    (runtime / switch.LAUNCHER).write_text("#!/bin/bash\n")
    (runtime / switch.GPTQ_LAUNCHER).write_text(f'#!/bin/bash\nARGS=(--model "{models / switch.CHECKPOINTS["default"]}")\n')
    (runtime / "engine-actuator.py").write_text("# private stub\n")
    (runtime / "active-serve").write_text(switch.LAUNCHER + "\n")
    return runtime, models


@pytest.mark.parametrize("target", ["default", "hauhaucs", "dft1"])
def test_installed_alias_and_absolute_path_use_same_validated_launcher(private_runtime, target):
    runtime, models = private_runtime
    model = models / switch.CHECKPOINTS[target]
    alias = switch.plan(target, runtime, models)
    path = switch.plan(str(model), runtime, models)
    assert alias["selection"] == path["selection"]
    if target == "default":
        assert alias["launcher"] == switch.GPTQ_LAUNCHER
        assert alias["selection"]["model_config"] is None
        return
    assert alias["launcher"] == switch.LAUNCHER
    cfg = json.loads(alias["selection"]["model_config"])
    assert cfg["model"] == str(model)
    assert set(switch.ALIASES).issubset(cfg["served_model_name"])
    assert model.name.lower() in cfg["served_model_name"]


def test_missing_legacy_alias_and_cache_dir_are_rejected(private_runtime):
    runtime, models = private_runtime
    with pytest.raises(switch.SwitchError, match="not installed"):
        switch.plan("abliterated", runtime, models)
    cache = models / (switch.CHECKPOINTS["dft1"] + "-u2cache")
    cache.mkdir()
    with pytest.raises(switch.SwitchError, match="unsupported checkpoint"):
        switch.plan(str(cache), runtime, models)


def test_mismatched_incumbent_is_rejected_before_staging(private_runtime):
    runtime, models = private_runtime
    (runtime / switch.GPTQ_LAUNCHER).write_text('#!/bin/bash\nARGS=(--model /wrong/checkpoint)\n')
    with pytest.raises(switch.SwitchError, match="expected"):
        switch.plan("default", runtime, models)


def test_dry_run_does_not_write_or_call_actuator(private_runtime, monkeypatch, capsys):
    runtime, models = private_runtime
    before = {p.name: p.read_bytes() for p in runtime.iterdir()}
    monkeypatch.setattr(switch.subprocess, "run", lambda *a, **k: pytest.fail("dry-run called actuator"))
    assert switch.main(["--dry-run", "--runtime-dir", str(runtime), "--models-dir", str(models), "dft1"]) == 0
    assert json.loads(capsys.readouterr().out)["dry_run"] is True
    assert {p.name: p.read_bytes() for p in runtime.iterdir()} == before


def test_noop_for_actual_root_does_not_write_or_acquire_hold(private_runtime, monkeypatch, capsys):
    runtime, models = private_runtime
    before = {p.name: p.read_bytes() for p in runtime.iterdir()}
    monkeypatch.setattr(switch, "already_serving", lambda path: path == str(models / switch.CHECKPOINTS["hauhaucs"]))
    monkeypatch.setattr(switch.subprocess, "run", lambda *a, **k: pytest.fail("no-op called actuator"))
    assert switch.main(["--runtime-dir", str(runtime), "--models-dir", str(models), "hauhaucs"]) == 0
    assert "ALREADY SERVING" in capsys.readouterr().out
    assert {p.name: p.read_bytes() for p in runtime.iterdir()} == before


class Response(io.BytesIO):
    status = 200


def test_current_checkpoint_uses_root_not_compatibility_alias(monkeypatch, tmp_path):
    wanted = tmp_path / "dft1"
    wrong = tmp_path / "base"
    payload = {"data": [{"id": wanted.name, "root": str(wrong)}]}
    monkeypatch.setattr(switch.urllib.request, "urlopen", lambda *a, **k: Response(json.dumps(payload).encode()))
    assert not switch.already_serving(str(wanted))
    payload["data"][0]["root"] = str(wanted)
    assert switch.already_serving(str(wanted))
    payload["data"][0].pop("root")
    assert not switch.already_serving(str(wanted))


@pytest.mark.parametrize("failure_rc, expected_restarts", [(1, 2), (3, 1)])
def test_failed_switch_restores_missing_json_and_old_pointer(private_runtime, monkeypatch, failure_rc, expected_restarts):
    runtime, models = private_runtime
    (runtime / "active-serve").write_text("serve-hauhaucs.sh\n")
    original = switch.snapshot(runtime)
    calls = []
    monkeypatch.setattr(switch, "preflight", lambda path: None)
    def restart(path, reason="model-switch"):
        calls.append((reason, switch.snapshot(path)))
        return failure_rc if len(calls) == 1 else 0
    monkeypatch.setattr(switch, "restart", restart)
    with pytest.raises(switch.SwitchError, match="engine restart failed"):
        switch.execute_switch(runtime, switch.plan("dft1", runtime, models))
    assert switch.snapshot(runtime) == original
    assert len(calls) == expected_restarts
    assert json.loads(calls[0][1]["model_config"])["model"].endswith("-dft1")
    if failure_rc == 1:
        assert calls[1] == ("model-switch rollback", original)


def test_success_requires_actual_checkpoint_and_preserves_backup(private_runtime, monkeypatch):
    runtime, models = private_runtime
    original = switch.snapshot(runtime)
    monkeypatch.setattr(switch, "preflight", lambda path: None)
    monkeypatch.setattr(switch, "restart", lambda *a, **k: 0)
    monkeypatch.setattr(switch, "already_serving", lambda path: True)
    assert switch.execute_switch(runtime, switch.plan("dft1", runtime, models)) == 0
    assert json.loads((runtime / switch.PREVIOUS).read_text()) == original
    assert switch.plan("rollback", runtime, models)["selection"] == original


def test_healthy_wrong_checkpoint_rolls_back(private_runtime, monkeypatch):
    runtime, models = private_runtime
    original = switch.snapshot(runtime)
    monkeypatch.setattr(switch, "preflight", lambda path: None)
    monkeypatch.setattr(switch, "already_serving", lambda path: False)
    calls = []
    monkeypatch.setattr(switch, "restart", lambda path, reason="model-switch": calls.append(reason) or 0)
    with pytest.raises(switch.SwitchError, match="did not load"):
        switch.execute_switch(runtime, switch.plan("dft1", runtime, models))
    assert switch.snapshot(runtime) == original
    assert calls == ["model-switch", "model-switch rollback"]


@pytest.mark.parametrize("foreign", [False, True])
def test_preflight_checks_owned_hold_and_foreign_quiesce(private_runtime, monkeypatch, foreign):
    runtime, _ = private_runtime
    monkeypatch.setenv("ENGINE_HOLD_LEASE", "mine")
    monkeypatch.setenv("ENGINE_HOLD_LEASE_KIND", "engine")
    holds = [{"lease": "mine", "kind": "engine"}]
    if foreign:
        holds.append({"lease": "other", "kind": "quiesce", "by": "publish"})
    monkeypatch.setattr(switch.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps({"holds": holds})))
    if foreign:
        with pytest.raises(switch.SwitchError, match="publish"):
            switch.preflight(runtime)
    else:
        switch.preflight(runtime)


def test_running_restart_refused_before_staging(private_runtime, monkeypatch):
    runtime, models = private_runtime
    monkeypatch.setenv("ENGINE_HOLD_LEASE", "mine")
    monkeypatch.setenv("ENGINE_HOLD_LEASE_KIND", "engine")
    monkeypatch.setattr(switch.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, '{"holds":[{"lease":"mine","kind":"engine"}]}'))
    original = switch.snapshot(runtime)
    with (runtime / "restart.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(switch.SwitchError, match="planned restart"):
            switch.execute_switch(runtime, switch.plan("dft1", runtime, models))
    assert switch.snapshot(runtime) == original


def test_hold_wrapper_and_restart_cli_keep_drain_enabled(private_runtime, monkeypatch):
    runtime, models = private_runtime
    calls = []
    monkeypatch.setattr(switch, "already_serving", lambda path: False)
    monkeypatch.setattr(switch.subprocess, "run", lambda command, **kw: calls.append(command) or subprocess.CompletedProcess(command, 0))
    assert switch.main(["--runtime-dir", str(runtime), "--models-dir", str(models), "dft1"]) == 0
    assert calls[0][2:5] == ["hold", "run", "--kind"]
    assert "--held" in calls[0] and "--no-drain" not in calls[0]
    switch.restart(runtime)
    assert calls[1][2:] == ["restart", "--foreground", "--reason", "model-switch", "--by", "dashboard"]
