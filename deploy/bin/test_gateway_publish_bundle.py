"""Private Git/runtime transactions; no live paths, control calls or CUDA."""
from pathlib import Path
import stat
import subprocess

import pytest

import gateway_publish_bundle as bundle


@pytest.fixture
def private_bundle(tmp_path):
    source = tmp_path / "repo" / "deploy" / "bin"
    source.mkdir(parents=True)
    repo = source.parents[1]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for name in bundle.CONTROL_FILES:
        path = source / name
        path.write_text(f"#!/bin/sh\n# committed {name}\n")
        path.chmod(0o755)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Offline test", "-c", "user.email=offline@example.invalid",
                    "commit", "-qm", "private fixture"], check=True)
    runtime = tmp_path / "private-runtime"
    runtime.mkdir()
    (runtime / "switch-model.sh").write_bytes(b"previous launcher\x00exact\n")
    (runtime / "switch-model.sh").chmod(0o640)
    (runtime / "serve-hauhaucs-v02.sh").write_text("previous executable\n")
    (runtime / "serve-hauhaucs-v02.sh").chmod(0o751)
    # The incumbent legacy GPTQ route and engine-selection pointer are unrelated.
    (runtime / "serve-tqk8v4-fg.sh").write_text("legacy GPTQ\n")
    (runtime / "active-serve").write_text("incumbent\n")
    return source, runtime, tmp_path / "private-backup"


def view(directory):
    return {p.name: (p.read_bytes(), stat.S_IMODE(p.stat().st_mode))
            for p in directory.iterdir() if p.is_file()}


def test_exact_bytes_modes_absence_and_unrelated_files_roundtrip(private_bundle):
    source, runtime, backup = private_bundle
    before = view(runtime)
    state = bundle.snapshot(source, runtime, backup)
    assert bundle.verify(state, restored=True)
    assert not bundle.verify(state)
    bundle.install(state)
    assert bundle.verify(state)
    assert view(runtime)["serve-tqk8v4-fg.sh"] == before["serve-tqk8v4-fg.sh"]
    assert view(runtime)["active-serve"] == before["active-serve"]
    bundle.restore(state)
    assert bundle.verify(state, restored=True)
    assert view(runtime) == before
    bundle.restore(state)  # idempotent recovery, including first-install absence
    assert view(runtime) == before


@pytest.mark.parametrize("change", ["bytes", "mode", "missing", "symlink"])
def test_uncommitted_or_nonregular_sources_refused_before_backups(private_bundle, change):
    source, runtime, backup = private_bundle
    path = source / "model_switch.py"
    if change == "bytes":
        path.write_text("dirty source\n")
    elif change == "mode":
        path.chmod(0o644)
    else:
        path.unlink()
        if change == "symlink":
            path.symlink_to(source / "switch-model.sh")
    before = view(runtime)
    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        bundle.snapshot(source, runtime, backup)
    assert not backup.exists()
    assert view(runtime) == before


def test_interrupted_second_install_restores_all_prior_files(private_bundle, monkeypatch):
    source, runtime, backup = private_bundle
    before = view(runtime)
    state = bundle.snapshot(source, runtime, backup)
    actual = bundle._write
    calls = []
    def interrupted(path, data, mode):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("simulated interruption before atomic replacement")
        actual(path, data, mode)
    monkeypatch.setattr(bundle, "_write", interrupted)
    with pytest.raises(OSError):
        bundle.install(state)
    monkeypatch.setattr(bundle, "_write", actual)
    bundle.restore(state)
    assert view(runtime) == before


def test_corrupted_backup_refuses_install_and_recovery(private_bundle):
    source, runtime, backup = private_bundle
    state = bundle.snapshot(source, runtime, backup)
    before = view(runtime)
    (backup / "switch-model.sh.previous").write_text("corrupted backup\n")
    assert not bundle.verify(state, restored=True)
    for operation in (bundle.install, bundle.restore):
        with pytest.raises(RuntimeError, match="hash mismatch"):
            operation(state)
    assert view(runtime) == before


def test_foreign_write_preserved_on_install_and_recovery(private_bundle):
    source, runtime, backup = private_bundle
    state = bundle.snapshot(source, runtime, backup)
    (runtime / "switch-model.sh").write_text("foreign writer\n")
    before = view(runtime)
    for operation in (bundle.install, bundle.restore):
        with pytest.raises(RuntimeError, match="changed"):
            operation(state)
    assert view(runtime) == before


def test_symlink_destination_never_followed(private_bundle):
    source, runtime, backup = private_bundle
    victim = runtime / "active-serve"
    (runtime / "model_switch.py").symlink_to(victim)
    with pytest.raises(RuntimeError, match="nonregular"):
        bundle.snapshot(source, runtime, backup)
    assert victim.read_text() == "incumbent\n"


def test_mutated_manifest_cannot_delete_unrelated_file(private_bundle):
    source, runtime, backup = private_bundle
    state = bundle.snapshot(source, runtime, backup)
    state["files"][0]["name"] = "active-serve"
    with pytest.raises(RuntimeError, match="manifest changed"):
        bundle.restore(state)
    assert (runtime / "active-serve").read_text() == "incumbent\n"


def test_commit_moving_during_snapshot_refused_before_backups(private_bundle, monkeypatch):
    source, runtime, backup = private_bundle
    actual = bundle.subprocess.check_output
    heads = []
    def moving_head(argv, **kwargs):
        result = actual(argv, **kwargs)
        if argv[-2:] == ["rev-parse", "HEAD"]:
            heads.append(result)
            if len(heads) == 2:
                return "0" * 40 + "\n"
        return result
    monkeypatch.setattr(bundle.subprocess, "check_output", moving_head)
    with pytest.raises(RuntimeError, match="commit changed"):
        bundle.snapshot(source, runtime, backup)
    assert not backup.exists()
