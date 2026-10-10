"""Exact, opt-in control-file installation for the governed gateway publisher.

This is file transaction machinery, not a serving or admission authority. The
publisher owns the actuator hold, durable drain, recovery and qualification.
The installed GPTQ launcher, engine selection and window controls are untouched.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import subprocess
import tempfile


CONTROL_FILES = ("model_switch.py", "switch-model.sh", "serve-hauhaucs-v02.sh")


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _regular(path):
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"control bundle requires a regular file: {path}")


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write(path, data, mode):
    """Atomic regular-file replacement, including the specified permission bits."""
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError(f"refusing nonregular bundle destination: {path}")
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fchmod(handle.fileno(), mode)
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _sync_dir(path.parent)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _committed(source_dir, name):
    source = source_dir / name
    _regular(source)
    repo = Path(subprocess.check_output(["git", "-C", str(source_dir), "rev-parse", "--show-toplevel"], text=True).strip()).resolve()
    relative = source.relative_to(repo).as_posix()
    blob = subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:{relative}"])
    row = subprocess.check_output(["git", "-C", str(repo), "ls-tree", "HEAD", "--", relative], text=True).split()
    if not row or row[0] not in ("100644", "100755") or source.read_bytes() != blob:
        raise RuntimeError(f"control bundle source differs from committed regular file: {source}")
    mode = 0o755 if row[0] == "100755" else 0o644
    if stat.S_IMODE(source.stat().st_mode) != mode:
        raise RuntimeError(f"control bundle source mode differs from commit: {source}")
    return repo, blob, mode


def snapshot(source_dir: Path, runtime_dir: Path, backup_dir: Path):
    """Capture committed candidate and exact prior bytes/mode/absence before writes.

    Returned state can be saved in the publisher's existing recovery journal.
    Backup files contain only the three named control scripts, never tokens or
    environment files. A first-install file is removed on rollback.
    """
    source_dir, runtime_dir, backup_dir = map(lambda p: Path(p).resolve(), (source_dir, runtime_dir, backup_dir))
    if not runtime_dir.is_dir() or backup_dir.exists():
        raise RuntimeError("bundle needs an existing runtime directory and fresh backup directory")
    candidates = []
    repo = None
    for name in CONTROL_FILES:
        current_repo, candidate, mode = _committed(source_dir, name)
        if repo is not None and current_repo != repo:
            raise RuntimeError("control bundle source roots differ")
        repo = current_repo
        target = runtime_dir / name
        previous = None
        previous_mode = None
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise RuntimeError(f"refusing nonregular previous control file: {target}")
        if target.exists():
            previous = target.read_bytes()
            previous_mode = stat.S_IMODE(target.stat().st_mode)
        candidates.append((name, candidate, mode, previous, previous_mode))
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    backup_dir.mkdir(mode=0o700, parents=False)
    rows = []
    for name, candidate, mode, previous, previous_mode in candidates:
        _write(backup_dir / f"{name}.candidate", candidate, 0o600)
        if previous is not None:
            _write(backup_dir / f"{name}.previous", previous, 0o600)
        rows.append({"name": name, "candidate_sha256": _digest(candidate), "candidate_mode": mode,
                     "previous_sha256": _digest(previous) if previous is not None else None,
                     "previous_mode": previous_mode})
    _sync_dir(backup_dir.parent)
    return {"v": 1, "kind": "model-switch", "source_head": head,
            "runtime_dir": str(runtime_dir), "backup_dir": str(backup_dir), "files": rows}


def _state(state):
    if not isinstance(state, dict) or state.get("v") != 1 or state.get("kind") != "model-switch":
        raise RuntimeError("invalid control bundle transaction")
    rows = state.get("files")
    if not isinstance(rows, list) or [r.get("name") for r in rows if isinstance(r, dict)] != list(CONTROL_FILES):
        raise RuntimeError("control bundle file manifest changed")
    runtime, backup = Path(state["runtime_dir"]), Path(state["backup_dir"])
    if not runtime.is_absolute() or not backup.is_absolute() or runtime.is_symlink() or backup.is_symlink():
        raise RuntimeError("invalid control bundle roots")
    for row in rows:
        for which in ("candidate", "previous"):
            digest = row.get(f"{which}_sha256")
            mode = row.get(f"{which}_mode")
            if digest is None and which == "previous" and mode is None:
                continue
            if not isinstance(mode, int) or not 0 <= mode <= 0o7777 or not isinstance(digest, str):
                raise RuntimeError("invalid control bundle metadata")
            path = backup / f"{row['name']}.{which}"
            _regular(path)
            if _digest(path.read_bytes()) != digest:
                raise RuntimeError(f"control bundle backup hash mismatch: {path}")
    return runtime, backup, rows


def _matches(path, digest, mode):
    if digest is None:
        return not path.exists() and not path.is_symlink()
    return (not path.is_symlink() and path.is_file()
            and _digest(path.read_bytes()) == digest
            and stat.S_IMODE(path.stat().st_mode) == mode)


def install(state):
    runtime, backup, rows = _state(state)
    # Validate all prior files first. A foreign writer must not be overwritten.
    if not all(_matches(runtime / r["name"], r["previous_sha256"], r["previous_mode"]) for r in rows):
        raise RuntimeError("control bundle destination changed after snapshot")
    for row in rows:
        _write(runtime / row["name"], (backup / f"{row['name']}.candidate").read_bytes(), row["candidate_mode"])
    if not verify(state):
        raise RuntimeError("installed control bundle does not match committed candidate")


def restore(state):
    runtime, backup, rows = _state(state)
    # A partially installed transaction contains only previous or candidate
    # atomic files. Preserve foreign bytes instead of silently clobbering them.
    for row in rows:
        target = runtime / row["name"]
        if not any(_matches(target, row[f"{which}_sha256"], row[f"{which}_mode"]) for which in ("previous", "candidate")):
            raise RuntimeError(f"control bundle destination changed during recovery: {target}")
    for row in rows:
        target = runtime / row["name"]
        if row["previous_sha256"] is None:
            target.unlink(missing_ok=True)
            _sync_dir(runtime)
        else:
            _write(target, (backup / f"{row['name']}.previous").read_bytes(), row["previous_mode"])
    if not verify(state, restored=True):
        raise RuntimeError("restored control bundle does not match exact previous state")


def verify(state, restored=False):
    try:
        runtime, _, rows = _state(state)
        which = "previous" if restored else "candidate"
        return all(_matches(runtime / r["name"], r[f"{which}_sha256"], r[f"{which}_mode"]) for r in rows)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return False
