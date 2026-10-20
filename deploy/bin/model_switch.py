#!/usr/bin/env python3
"""Model selection data + a guarded, reversible switch through the engine actuator.

Install beside switch-model.sh and serve-hauhaucs-v02.sh in the runtime directory.
--dry-run validates and prints a plan without writes, holds, or service calls.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import tempfile
import urllib.request

HERE = Path(__file__).resolve().parent
LAUNCHER = "serve-hauhaucs-v02.sh"
GPTQ_LAUNCHER = "serve-tqk8v4-fg.sh"
MODEL_CONFIG = "v02.model.json"
PREVIOUS = "model-switch.previous.json"
CHECKPOINTS = {
    "default": "Qwen3.8-27B-GPTQ-Int4",
    "hauhaucs": "Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven",
    "dft1": "Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven-dft1",
    "abliterated": "Qwen3.8-27B-Uncensored-W4A16-AutoRound",
}
# Existing clients pin these names; keep them across checkpoint switches.
ALIASES = ["estate-hauhaucs", "qwen3.8-27b-hauhaucs-aggressive", "estate-abliterated",
           "qwen3.8-27b-abliterated", "qwen-local", "qwen3.8-27b-uncensored", "qwen3.6:27b",
           "qwen3.8-27b-gptq-int4", "qwen27b-int4-tqk8v4-two250K-mtp3-text-only-cu128"]


class SwitchError(RuntimeError):
    pass


def validate_model(path: Path, models: Path) -> Path:
    resolved = path.resolve()
    allowed = {(models / name).resolve() for name in CHECKPOINTS.values()}
    if resolved not in allowed:
        raise SwitchError(f"unsupported checkpoint: {path}; use an installed model shown by list")
    if not (resolved / "config.json").is_file():
        raise SwitchError(f"checkpoint is not installed: {resolved}")
    config = json.loads((resolved / "config.json").read_text())
    method = (config.get("quantization_config") or {}).get("quant_method", "")
    if method not in ("gptq", "compressed-tensors") or config.get("model_type") != "qwen3_5":
        raise SwitchError(f"unsupported checkpoint format: {resolved} ({method!r})")
    return resolved


def read_optional(path: Path) -> str | None:
    return path.read_text() if path.exists() else None


def snapshot(runtime: Path) -> dict:
    return {"active_serve": read_optional(runtime / "active-serve"),
            "model_config": read_optional(runtime / MODEL_CONFIG)}


def launcher_model(launcher: Path) -> Path:
    """Read a literal --model token without evaluating any shell code."""
    lexer = shlex.shlex(launcher.read_text(), posix=True, punctuation_chars="()")
    lexer.whitespace_split = True
    tokens = list(lexer)
    values = [tokens[i + 1] for i, token in enumerate(tokens[:-1]) if token == "--model"]
    if len(values) != 1 or not Path(values[0]).is_absolute() or "$" in values[0] or "`" in values[0]:
        raise SwitchError(f"launcher must have one literal absolute --model path: {launcher}")
    return Path(values[0]).resolve()


def plan(target: str, runtime: Path, models: Path) -> dict:
    if target == "rollback":
        previous = json.loads((runtime / PREVIOUS).read_text())
        if set(previous) != {"active_serve", "model_config"} or any(v is not None and not isinstance(v, str) for v in previous.values()):
            raise SwitchError("invalid previous model selection")
        active = (previous["active_serve"] or "serve-tqk8v4-fg.sh").strip()
        if active not in (LAUNCHER, "serve-tqk8v4-fg.sh", "serve-abliterated.sh", "serve-hauhaucs.sh"):
            raise SwitchError(f"unsupported previous launcher: {active}")
        if not (runtime / active).is_file():
            raise SwitchError(f"previous launcher is missing: {runtime / active}")
        if active == LAUNCHER:
            cfg = json.loads(previous["model_config"] or "{}")
            model = validate_model(Path(cfg.get("model") or models / CHECKPOINTS["hauhaucs"]), models)
            return {"target": target, "model": str(model), "launcher": active, "selection": previous}
        model = validate_model(launcher_model(runtime / active), models)
        return {"target": target, "model": str(model), "launcher": active, "selection": previous}
    alias = {"3.8": "default", "hauhaucs-dft1": "dft1"}.get(target, target)
    model = models / CHECKPOINTS[alias] if alias in CHECKPOINTS else Path(target)
    model = validate_model(model, models)
    if model == (models / CHECKPOINTS["default"]).resolve():
        # Keep the existing validated GPTQ profile (v02's U2 packing is CT-only).
        # Do not install the older repository incumbent over the runtime launcher.
        installed = launcher_model(runtime / GPTQ_LAUNCHER)
        if installed != model:
            raise SwitchError(f"installed GPTQ launcher serves {installed}, expected {model}")
        return {"target": target, "model": str(model), "launcher": GPTQ_LAUNCHER,
                "selection": {"active_serve": GPTQ_LAUNCHER + "\n", "model_config": read_optional(runtime / MODEL_CONFIG)}}
    if not (runtime / LAUNCHER).is_file():
        raise SwitchError(f"launcher is missing: {runtime / LAUNCHER}")
    cfg = {"model": str(model), "served_model_name": list(dict.fromkeys(ALIASES + [model.name.lower()]))}
    return {"target": target, "model": str(model), "launcher": LAUNCHER,
            "selection": {"active_serve": LAUNCHER + "\n", "model_config": json.dumps(cfg, indent=2) + "\n"}}


def atomic_write(path: Path, content: str | None):
    if content is None:
        path.unlink(missing_ok=True)
        return
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def apply_selection(runtime: Path, selection: dict):
    atomic_write(runtime / MODEL_CONFIG, selection["model_config"])
    atomic_write(runtime / "active-serve", selection["active_serve"])


def actuator_cmd(runtime: Path, *args: str) -> list[str]:
    actuator = runtime / "engine-actuator.py"
    if not actuator.is_file():
        raise SwitchError(f"actuator is missing: {actuator}")
    return [sys.executable, str(actuator), *args]


def preflight(runtime: Path):
    lease = os.environ.get("ENGINE_HOLD_LEASE")
    if not lease or os.environ.get("ENGINE_HOLD_LEASE_KIND") != "engine":
        raise SwitchError("model switch requires an actuator engine hold")
    result = subprocess.run(actuator_cmd(runtime, "hold", "status"), capture_output=True, text=True, check=True)
    holds = json.loads(result.stdout).get("holds", [])
    if not any(h.get("lease") == lease and h.get("kind") == "engine" for h in holds):
        raise SwitchError("model switch engine hold expired or was released")
    foreign = [h for h in holds if h.get("lease") != lease]
    if foreign:
        raise SwitchError(f"engine is held by {foreign[0].get('by')} ({foreign[0].get('kind')})")
    # A restart begun before we acquired our hold must finish before selection changes.
    with (runtime / "restart.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SwitchError("another planned restart is already running") from exc


def restart(runtime: Path, reason="model-switch") -> int:
    return subprocess.run(actuator_cmd(runtime, "restart", "--foreground", "--reason", reason,
                                      "--by", "dashboard")).returncode


def already_serving(model: str) -> bool:
    """Aliases follow a switch; only /v1/models.root identifies the checkpoint."""
    engine = os.environ.get("VLLM_ENGINE_URL", "http://127.0.0.1:8001").rstrip("/")
    try:
        with urllib.request.urlopen(engine + "/v1/models", timeout=5) as response:
            rows = json.load(response).get("data", [])
        roots = [Path(row["root"]).resolve() for row in rows
                 if isinstance(row.get("root"), str) and Path(row["root"]).is_absolute()]
        if not roots or any(root != Path(model).resolve() for root in roots):
            return False
        with urllib.request.urlopen(engine + "/health", timeout=5) as response:
            return response.status == 200
    except (OSError, ValueError, TypeError, KeyError):
        return False


def execute_switch(runtime: Path, desired: dict) -> int:
    # Serialize selection/backup/rollback, including independently started CLI switches.
    with (runtime / "model-switch.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SwitchError("another model switch is already running") from exc
        preflight(runtime)
        previous = snapshot(runtime)
        changed = False
        try:
            changed = True  # even an incomplete two-file write must be restored
            apply_selection(runtime, desired["selection"])
            rc = restart(runtime)
            if rc:
                raise SwitchError(f"engine restart failed (exit {rc})")
            if desired.get("model") and not already_serving(desired["model"]):
                raise SwitchError("engine is healthy but did not load the selected checkpoint")
            atomic_write(runtime / PREVIOUS, json.dumps(previous, indent=2) + "\n")
            print(f"SWITCH COMPLETE: {desired.get('model') or desired['target']}", flush=True)
            return 0
        except BaseException as exc:
            if changed:
                apply_selection(runtime, previous)
                print(f"Switch failed; restored previous selection: {exc}", file=sys.stderr, flush=True)
                # Exit 3 means the actuator refused before touching the engine.
                if locals().get("rc") != 3:
                    rolled_back = restart(runtime, "model-switch rollback")
                    if rolled_back:
                        print(f"Rollback restart failed (exit {rolled_back}); previous selection remains staged", file=sys.stderr)
            raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", default="list")
    parser.add_argument("--dry-run", "--validate", action="store_true")
    parser.add_argument("--runtime-dir", type=Path, default=Path(os.environ.get("VLLM_RUNTIME_DIR", HERE)))
    parser.add_argument("--models-dir", type=Path, default=Path(os.environ.get("VLLM_MODELS_DIR", "/home/kevin/Desktop/models")))
    parser.add_argument("--held", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    runtime, models = args.runtime_dir.resolve(), args.models_dir.resolve()
    try:
        if args.target in ("status", "list"):
            current = snapshot(runtime)
            print(json.dumps({"active_serve": (current["active_serve"] or "serve-tqk8v4-fg.sh").strip(),
                              "selection": json.loads(current["model_config"] or "{}"),
                              "available": {k: str(models / v) for k, v in CHECKPOINTS.items() if (models / v / "config.json").is_file()}}, indent=2))
            return 0
        desired = plan(args.target, runtime, models)
        if args.dry_run:
            print(json.dumps(dict(desired, dry_run=True, restart=["restart", "--foreground", "--reason", "model-switch", "--by", "dashboard"]), indent=2))
            return 0
        if desired.get("model") and already_serving(desired["model"]):
            print(f"ALREADY SERVING: {desired['model']} (selection and runtime unchanged)", flush=True)
            return 0
        # A test must never stage real runtime files, even when run in a child process.
        live = Path.home() / ".local/share/vllm-qwen27b"
        if os.environ.get("PYTEST_CURRENT_TEST") and runtime == live.resolve():
            raise SwitchError("refusing to switch the live runtime from a test")
        if not args.held:
            command = actuator_cmd(runtime, "hold", "run", "--kind", "engine", "--by", "gateway-model-switch",
                                   "--reason", "dashboard model switch", "--ttl", "3600", "--no-ensure-up", "--",
                                   sys.executable, str(Path(__file__).resolve()), "--held", "--runtime-dir", str(runtime),
                                   "--models-dir", str(models), args.target)
            return subprocess.run(command).returncode
        def terminated(signum, _frame):
            raise SwitchError(f"model switch interrupted by signal {signum}")
        for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(signum, terminated)
        return execute_switch(runtime, desired)
    except (SwitchError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
