#!/usr/bin/env python3
"""Publish committed gateway bytes through externally owned Core maintenance.

Snapshot exact files and unit activity before taking the existing engine hold
and durable all-traffic drain. Qualification requires unchanged engine identity,
loaded startup-module hashes, and a successful direct generation. Independent
recovery owns interrupted transactions; retired Halo pause authority is untouched.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = HERE / "keepalive-shim.py"
RUNTIME = Path("/home/kevin/.local/share/vllm-qwen27b/keepalive-shim.py")
# The dashboard page ships beside the shim (the shim reads it per request from its own directory and falls back to its
# inline copy when absent), so a page-only change needs no restart.
DASH_SOURCE = HERE / "gateway_dashboard.html"
DASH_RUNTIME = RUNTIME.with_name("gateway_dashboard.html")
TOKEN_FILE = Path("/home/kevin/.local/share/vllm-qwen27b/admin.token")
DROPIN = REPO / "deploy/systemd/vllm-keepalive-shim.service.d/zz-graceful-stop.conf"
DROPIN_LIVE = Path("/etc/systemd/system/vllm-keepalive-shim.service.d/zz-graceful-stop.conf")
BASE = "http://127.0.0.1:8000"
SERVICE = "vllm-keepalive-shim.service"
HALO_INCIDENTS = Path("/home/kevin/.local/share/estate-overseer/halo-incidents")
HALO_START_PAUSE = Path("/home/kevin/.local/share/estate-overseer/HALO_INCIDENT_START_PAUSE.json")
HALO_SUPERVISOR_LOCK = Path("/tmp/halo-incident-supervisor.lock")
ESTATE_RUNTIME = Path("/home/kevin/.local/share/estate-overseer")
# L77: the gateway keeps the drain lease id only in memory, so a publisher that dies without its `finally` strands the fence
# until the TTL. The lease id is recorded here (with the owner pid) so the next publish can close an orphan at once.
DRAIN_STATE = Path("/home/kevin/.local/share/vllm-qwen27b/gateway-publish-drain.json")
AUDIT_LOG = Path("/home/kevin/.local/share/vllm-qwen27b/gateway-publish-audit.log")
HALO_LEASE_OWNER = "gateway-safe-publish"
# LV 2026-10-03: the engine-liveness authority's quiesce hold. While it is held no PLANNED engine restart can start (its offline
# window / drain would collide with this drain); automatic wedge recovery still runs (a wedged engine would never let the drain end).
ACTUATOR = Path(os.environ.get("ENGINE_ACTUATOR", "/home/kevin/.local/share/vllm-qwen27b/engine-actuator.py"))
_TERMINATION_SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)


_BY = (os.environ.get("PUBLISH_BY") or "gateway_safe_publish")[:60] + f" pid={os.getpid()}"


class PublishTerminated(BaseException):
    """A termination signal arrived. BaseException so `except Exception` rollback paths do not swallow it, while every
    `finally` (lease release) still runs."""

    def __init__(self, signum: int):
        super().__init__(f"terminated by signal {signum}")
        self.signum = signum


@contextlib.contextmanager
def _terminate_as_exception():
    """SIGTERM/SIGHUP/SIGINT raise PublishTerminated in the main thread so try/finally can release what is held."""
    def handler(signum, _frame):
        raise PublishTerminated(signum)
    previous = {}
    try:
        for sig in _TERMINATION_SIGNALS:
            previous[sig] = signal.signal(sig, handler)
    except ValueError:      # not the main thread: signals cannot be hooked here
        pass
    try:
        yield
    finally:
        for sig, old in previous.items():
            signal.signal(sig, old)


@contextlib.contextmanager
def _shielded():
    """Cleanup must not be interrupted halfway by a second signal."""
    previous = {}
    try:
        for sig in _TERMINATION_SIGNALS:
            previous[sig] = signal.signal(sig, signal.SIG_IGN)
    except ValueError:
        pass
    try:
        yield
    finally:
        for sig, old in previous.items():
            signal.signal(sig, old)


def _audit(event: str, **fields) -> None:
    """One line to stderr and (best effort) to the audit log. Never raises."""
    line = json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, "by": _BY, **fields},
                      sort_keys=True, default=str)
    try:
        print(line, file=sys.stderr, flush=True)
    except Exception:
        pass
    try:
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_LOG.open("a") as out:
            out.write(line + "\n")
    except Exception:
        pass


def _proc_stat(pid: int) -> tuple[str, str] | None:
    """(state, starttime) from /proc/<pid>/stat, or None when the process does not exist."""
    if not Path("/proc/self/stat").exists():
        raise OSError("/proc is not available; process liveness is unknowable")
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text()
    except FileNotFoundError:
        return None
    rest = raw.rsplit(")", 1)[1].split()
    return rest[0], rest[19]


def _own_stamp() -> dict:
    stat = _proc_stat(os.getpid())
    return {"pid": os.getpid(), "pid_start": stat[1] if stat else None}


def _owner_dead(row: dict) -> bool:
    """True only when the lease records a pid that is provably gone (missing, zombie, or the pid was reused).
    A lease without a pid (older publisher) or with an unreadable /proc is never presumed dead."""
    pid = row.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        stat = _proc_stat(pid)
    except (OSError, ValueError, IndexError):
        return False
    if stat is None or stat[0] in {"Z", "X"}:
        return True
    recorded = row.get("pid_start")
    return bool(recorded) and str(recorded) != stat[1]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _http(path: str, method: str = "GET", payload: dict | None = None, token: str = "") -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=body, method=method,
                                 headers={"Content-Type": "application/json", "X-Admin-Token": token})
    with urllib.request.urlopen(req, timeout=5) as response:
        raw = response.read()
        if response.headers.get("Content-Type", "").startswith("application/json"):
            return json.loads(raw)
        return {"http_status": response.status, "body": raw.decode("utf-8", "replace")[:80]}


def _unit_seconds(text: str) -> float:
    scales = {"us": 0.000001, "ms": 0.001, "s": 1, "min": 60, "h": 3600}
    parts = re.findall(r"(\d+(?:\.\d+)?)\s*(us|ms|min|s|h)\b", text)
    if not parts:
        raise ValueError(f"unreadable systemd duration: {text!r}")
    return sum(float(n) * scales[unit] for n, unit in parts)


def _run(*args: str) -> None:
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL)


def _atomic_write(path: Path, data: bytes, mode: int = 0o755) -> None:
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.unlink(missing_ok=True)         # a stale tmp from a killed process whose pid was reused
    try:
        with tmp.open("xb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        tmp.chmod(mode)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)     # never leave temp state behind, including when a signal lands mid-write
        raise
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _install_dashboard(source_bytes: bytes | None = None) -> dict:
    """Ship gateway_dashboard.html next to the live shim. Atomic; returns what changed so a failed publish can undo it."""
    new = DASH_SOURCE.read_bytes() if source_bytes is None else source_bytes
    old = DASH_RUNTIME.read_bytes() if DASH_RUNTIME.is_file() else None
    if old == new:
        return {"dashboard": "current", "previous": old, "changed": False}
    _atomic_write(DASH_RUNTIME, new, 0o644)
    return {"dashboard": "installed", "previous": old, "changed": True}


def _restore_dashboard(state: dict) -> None:
    if not state.get("changed"):
        return
    if state.get("previous") is None:
        DASH_RUNTIME.unlink(missing_ok=True)       # first install: the shim falls back to its inline page
    else:
        _atomic_write(DASH_RUNTIME, state["previous"], 0o644)


# Lane CFG: the gateway's typed config schema module ships beside the shim like the dashboard (the shim loads it from
# its own directory at startup and runs unvalidated without it). Paths derive from SOURCE/RUNTIME at call time so a
# test that redirects those never touches the production directory.
CONFIG_SCHEMA_NAME = "gateway_config_schema.py"
# Lane SH (2026-10-03): the manifest of Python modules that ship beside the shim. The shim loads each one from its
# own directory at STARTUP (spec_from_file_location; the live dir is never put on sys.path), so -- unlike the
# dashboard, which is read per request -- a changed module only takes effect through the drained restart: a
# module-only change is a full publish, never the no-restart "current" path. (filename, result key). A module absent
# from the source tree is "absent" (not an error); a first install is removed again on rollback.
GATEWAY_MODULES = (
    ("gateway_config_schema.py", "config_schema"),
    ("experiment_controls.py", "experiment_controls"),
)
REQUIRED_GATEWAY_MODULES = frozenset({"experiment_controls.py"})


def _module_names() -> list[str]:
    return [name for name, _ in GATEWAY_MODULES]


def _module_sources() -> dict:
    """{filename: bytes} of every manifest module present in the source tree (beside SOURCE, read at call time)."""
    out = {}
    for name in _module_names():
        src = SOURCE.with_name(name)
        if src.is_file():
            out[name] = src.read_bytes()
        elif name in REQUIRED_GATEWAY_MODULES:
            raise RuntimeError(f"required gateway module is missing: {name}")
    return out


def _modules_current(sources: dict) -> bool:
    """True when every shipped module is byte-identical at its live path (RUNTIME's directory)."""
    for name, data in sources.items():
        dst = RUNTIME.with_name(name)
        if not dst.is_file() or dst.read_bytes() != data:
            return False
    return True


def _modules_loaded_match(token: str, sources: dict) -> bool:
    """Lane SH: the restarted gateway reports (GET /gateway/modules) the sha256 of every module it executed; each
    shipped module must be loaded with exactly the published bytes. A gateway without the endpoint can only be one
    that predates the manifest, i.e. one shipping no code modules: then only the optional schema may be unverified."""
    try:
        got = (_http("/gateway/modules", token=token).get("modules") or {})
    except Exception:
        got = None
    for name, data in sources.items():
        rec = (got or {}).get(name)
        if got is None and name == CONFIG_SCHEMA_NAME:
            continue
        if not rec or not rec.get("loaded") or rec.get("sha256") != _sha(data):
            return False
    return True


def _install_gateway_files(dash: bytes) -> dict:
    """Every file that ships beside the shim (dashboard + the GATEWAY_MODULES manifest), for _publish only. Lane CFG
    2026-10-03: the schema install used to ride inside _install_dashboard(), so a test that redirected only
    DASH_RUNTIME wrote the LIVE gateway_config_schema.py. _install_dashboard() is dashboard-only again; only the
    publish path ships both."""
    return {**_install_dashboard(dash), **_install_modules()}


def _restore_gateway_files(state: dict) -> None:
    _restore_modules(state)
    _restore_dashboard(state)


def _install_modules() -> dict:
    out = {"modules": {}}
    for name, key in GATEWAY_MODULES:
        src, dst = SOURCE.with_name(name), RUNTIME.with_name(name)
        if not src.is_file():
            out[key] = "absent"
            out["modules"][name] = {"status": "absent", "changed": False}
            continue
        new = src.read_bytes()
        old = dst.read_bytes() if dst.is_file() else None
        if old == new:
            out[key] = "current"
            out["modules"][name] = {"status": "current", "changed": False}
            continue
        _atomic_write(dst, new, 0o644)
        out[key] = "installed"
        out["modules"][name] = {"status": "installed", "changed": True, "previous": old}
    return out


def _restore_modules(state: dict) -> None:
    for name, rec in (state.get("modules") or {}).items():
        if not rec.get("changed"):
            continue
        dst = RUNTIME.with_name(name)
        if rec.get("previous") is None:
            dst.unlink(missing_ok=True)
        else:
            _atomic_write(dst, rec["previous"], 0o644)


def _install_config_schema() -> dict:
    """Back-compat single-module entry point (the schema is now one row of GATEWAY_MODULES)."""
    st = _install_modules()
    rec = st["modules"].get(CONFIG_SCHEMA_NAME, {})
    out = {"config_schema": st.get("config_schema", "absent"), "schema_changed": bool(rec.get("changed"))}
    if rec.get("changed"):
        out["schema_previous"] = rec.get("previous")
    return out


def _restore_config_schema(state: dict) -> None:
    if state.get("schema_changed"):
        _restore_modules({"modules": {CONFIG_SCHEMA_NAME: {"changed": True, "previous": state.get("schema_previous")}}})


# LV 2026-10-03: "is a Halo run live?" is the incident SUPERVISOR's question, answered by its lease-backed occupancy, not by this
# file's own reading of status labels. The 08:01 GW2 publish aborted because two incidents finished their runs DURING the drain and
# became repair-requested with run status "stopping": the supervisor counts "stopping" as TERMINAL (the run has ended; the next round
# waits for capacity, which the start pause holds), this file counted it as ACTIVE. Same names, same sets, same lease rule as
# tools/halo_incident_supervisor.py (_active_execution_incidents); test_gateway_safe_publish_halo_predicate.py fails on any drift.
HALO_TERMINAL_RUN_STATUSES = frozenset({
    "completed", "complete", "failed", "interrupted", "stopping", "stopped",
    "cancelled", "canceled", "expired", "done", "error",
})
HALO_OPEN_STATES = ("assigned", "running", "investigate", "escalated", "repair-requested", "repair-running")
HALO_ACTIVE_EXECUTION_STATES = ("running", "repair-requested", "repair-running")


def _halo_lease_live(row: dict, now: float) -> bool:
    lease = row.get("lease")
    if not isinstance(lease, dict):
        return False
    try:
        if now >= float(lease.get("expires_at_epoch", 0)):
            return False
        used, cap = int(lease.get("tool_calls_used", 0)), int(lease.get("tool_calls_max", 0))
    except (TypeError, ValueError):
        return True                      # unreadable budget: treat as live (fail toward waiting, bounded by halo_wait_s)
    return not (cap and used >= cap)


def halo_run_live(row: dict, now: float | None = None) -> bool:
    """A Halo run that can still make a gateway turn: the supervisor's slot occupancy, plus any open incident whose run is in a
    non-terminal state under a live lease (an investigation run is a live run too)."""
    now = time.time() if now is None else now
    if not _halo_lease_live(row, now):
        return False
    status = row.get("status")
    run_status = str((row.get("halo_run") or {}).get("status") or "")
    if status in HALO_ACTIVE_EXECUTION_STATES:
        return not (status == "repair-requested" and run_status in HALO_TERMINAL_RUN_STATUSES)
    return status in HALO_OPEN_STATES and bool(run_status) and run_status not in HALO_TERMINAL_RUN_STATUSES


def _refuse_if_offline_window(token: str, phase: str) -> None:
    """GW2 2026-10-03: never restart the gateway inside a planned offline window. At 09:06 a K5 engine window opened while
    a publish waited for Halo; the restart dropped the window (it lived only in gateway memory) while the engine was
    stopped. The gateway now persists windows and refuses to OPEN one while the drain fence is up; this closes the gap
    before the fence (start) and between fence and restart (restart). An unreadable status refuses too."""
    st = _http("/gateway/offline", token=token)
    if st.get("http_status") not in (None, 200) or "offline" not in st:
        raise RuntimeError(f"planned-offline status unreadable before {phase}: {str(st)[:200]}")
    if st.get("offline") is not False:
        raise RuntimeError(f"planned offline window open ({st.get('reason')!r} by {st.get('by')!r}, "
                           f"{st.get('remaining_s')} s left): publish refused at {phase}")
    try:
        measurement = _http("/gateway/measurement", token=token)
    except urllib.error.HTTPError as error:
        # Bootstrap only: the actual legacy shim has no measurement endpoint.
        if error.code == 404 and b'"/gateway/measurement"' not in RUNTIME.read_bytes():
            return
        raise RuntimeError("measurement authority is unreadable") from error
    if measurement.get("active") is not False:
        raise RuntimeError(f"measurement window active or unknown at {phase}")


def _halo_active_runs() -> list[str]:
    """Do not restart the gateway between turns of an active Halo repair."""
    if not HALO_INCIDENTS.is_dir():
        raise RuntimeError("Halo incident authority is unreadable")
    active = []
    now = time.time()
    for path in HALO_INCIDENTS.glob("*.json"):
        try:
            row = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"unreadable Halo incident: {path.name}") from exc
        if halo_run_live(row, now):
            active.append(str(row.get("id") or path.stem))
    return active


def _begin_halo_quiesce(halo_wait_s: float, drain_timeout_s: float) -> str:
    """Lease admission only; the independent Halo lease enforcer keeps running."""
    try:
        prior = json.loads(HALO_START_PAUSE.read_text())
    except FileNotFoundError:
        prior = {}
    except (OSError, ValueError) as exc:
        raise RuntimeError("existing Halo start pause is unreadable") from exc
    if float(prior.get("expires_at_epoch") or 0) > time.time():
        # L77: our own lease left behind by a dead publisher is taken over, not waited out (the 2026-10-03 incident
        # blocked Halo for an hour). Only a lease this tool wrote, with a recorded pid that is provably gone.
        if prior.get("owner") == HALO_LEASE_OWNER and _owner_dead(prior):
            _audit("halo-start-pause-takeover", dead_pid=prior.get("pid"), prior_token=str(prior.get("token"))[:8],
                   prior_expires_at_epoch=prior.get("expires_at_epoch"))
        else:
            raise RuntimeError("another release owns the Halo start pause")
    token = secrets.token_hex(16)
    now = time.time()
    lease = {"owner": HALO_LEASE_OWNER, "token": token, **_own_stamp(),
             "issued_at_epoch": now,
             "expires_at_epoch": now + min(3550, halo_wait_s + drain_timeout_s + 180)}
    _atomic_write(HALO_START_PAUSE, (json.dumps(lease, sort_keys=True) + "\n").encode(), 0o644)
    return token


def _mark_halo_quiesce_phase(token: str, phase: str) -> None:
    """Record the release phase on the start-pause lease (quiet-wait -> draining -> swapping). The supervisor reads the lease; a
    phase lets it hold even its exempt (restoration/control) starts during the short drain+swap, and only then."""
    try:
        row = json.loads(HALO_START_PAUSE.read_text())
    except (OSError, ValueError):
        return
    if row.get("owner") != HALO_LEASE_OWNER or row.get("token") != token:
        return
    row["phase"], row["phase_at_epoch"] = phase, time.time()
    _atomic_write(HALO_START_PAUSE, (json.dumps(row, sort_keys=True) + "\n").encode(), 0o644)


def _liveness_hold(ttl_s: float) -> str | None:
    """Best effort: take the liveness authority's quiesce hold, owned by this process (void the moment it dies)."""
    if not ACTUATOR.is_file():
        _audit("liveness-hold-skipped", reason="engine-actuator.py not installed")
        return None
    try:
        out = subprocess.run(["/usr/bin/python3", str(ACTUATOR), "hold", "acquire", "--kind", "quiesce", "--by", HALO_LEASE_OWNER,
                              "--reason", f"governed gateway publish by {_BY}", "--ttl", str(int(max(60, min(ttl_s, 28800)))),
                              "--owner-pid", str(os.getpid())], capture_output=True, text=True, timeout=30)
        row = json.loads(out.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        _audit("liveness-hold-failed", error=repr(exc)[:200])
        return None
    if not row.get("lease"):
        _audit("liveness-hold-refused", detail=json.dumps(row)[:300])
        return None
    return row["lease"]


def _liveness_release(lease: str | None) -> None:
    if not lease:
        return
    try:
        subprocess.run(["/usr/bin/python3", str(ACTUATOR), "hold", "release", "--lease", lease], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        _audit("liveness-release-failed", error=repr(exc)[:200])     # owned by this pid: void when this process exits anyway


def _end_halo_quiesce(token: str | None) -> None:
    """Release the pause only if it is still ours. token=None (a signal landed before the token reached the caller)
    matches on this process's pid instead."""
    try:
        row = json.loads(HALO_START_PAUSE.read_text())
    except FileNotFoundError:
        return
    ours = (row.get("token") == token) if token is not None else (row.get("pid") == os.getpid())
    if row.get("owner") != HALO_LEASE_OWNER or not ours:
        if token is None:
            return      # never held one
        raise RuntimeError("Halo start pause ownership changed during gateway publication")
    HALO_START_PAUSE.unlink()
    fd = os.open(HALO_START_PAUSE.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _assert_halo_quiesce_live(token: str) -> None:
    """The installed supervisor must actually honor this release lease."""
    program = ("import json,sys; "
               f"sys.path.insert(0,{str(ESTATE_RUNTIME)!r}); "
               "from tools.halo_incident_supervisor import deployment_quiesce; "
               "print(json.dumps(deployment_quiesce()))")
    try:
        output = subprocess.check_output(["/usr/bin/python3", "-c", program],
                                         timeout=10, text=True)
        row = json.loads(output.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        raise RuntimeError("installed Halo supervisor does not honor release quiescence") from exc
    if not isinstance(row, dict) or row.get("token") != token:
        raise RuntimeError("installed Halo supervisor did not read the exact release lease")


def _wait_halo_quiet(timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        active = _halo_active_runs()
        if not active:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Halo runs did not reach terminal receipts: {active[:6]}")
        time.sleep(3)


def _wait_empty(token: str, timeout_s: float, state=None) -> None:
    deadline = time.monotonic() + timeout_s
    zero_polls = 0
    while time.monotonic() < deadline:
        if state is not None:
            _renew_publish_hold(state)
            _assert_engine(state["engine"])
        drain = _http("/gateway/drain", token=token)
        spend = _http("/gateway/spend", token=token)
        if (drain.get("draining") is not True or drain.get("invalid_state")
                or (drain.get("durable") is not True and float(drain.get("until") or 0) - time.time() < 30)):
            raise RuntimeError("drain lease expired before accepted calls completed")
        if state is not None and drain.get("durable") is True and drain.get("lease_sha256") != _sha(state["lease"].encode()):
            raise RuntimeError("durable drain has another lease owner")
        if not _known_count(drain.get("active")) or not _known_count(spend.get("in_flight")):
            raise RuntimeError("accepted-call counts are unknown")
        engine = _controls().engine_inflight(_controls().ENGINE) if state is not None else {"running": 0, "waiting": 0}
        if not isinstance(engine, dict) or not all(_known_count(engine.get(k)) for k in ("running", "waiting")):
            raise RuntimeError("engine drain counts are unknown")
        zero_polls = zero_polls + 1 if drain["active"] == spend["in_flight"] == engine["running"] == engine["waiting"] == 0 else 0
        if zero_polls >= 2:
            return
        time.sleep(1)
    raise TimeoutError("accepted calls did not drain within the deployment bound")


def _record_drain(lease: str) -> None:
    """Persist the lease id + owner pid so a publisher that dies without its `finally` can be cleaned up after."""
    try:
        _atomic_write(DRAIN_STATE, (json.dumps({"lease": lease, **_own_stamp(), "by": _BY, "ts": time.time()},
                                               sort_keys=True) + "\n").encode(), 0o600)
    except OSError as exc:
        _audit("drain-record-failed", error=repr(exc))


def _clear_drain_record(lease: str | None = None) -> None:
    try:
        row = json.loads(DRAIN_STATE.read_text())
    except (OSError, ValueError):
        return
    if lease is None or row.get("lease") == lease:
        DRAIN_STATE.unlink(missing_ok=True)


def _release_drain(token: str, lease: str | None) -> None:
    """Close a drain lease this process opened. Never raises: the lease also expires on its own."""
    if not lease:
        return
    try:
        if _http("/gateway/drain", token=token).get("draining"):
            _http("/gateway/drain", "DELETE", {"lease": lease}, token)
    except Exception as exc:
        _audit("drain-release-failed", error=repr(exc))     # lease itself expires; no permanent admission stop
    else:
        _clear_drain_record(lease)


def _reap_orphan_drain(token: str) -> bool:
    """A drain fence is up. If it is the recorded fence of a publisher that is provably dead, close it and report True."""
    try:
        row = json.loads(DRAIN_STATE.read_text())
    except (OSError, ValueError):
        return False
    if not row.get("lease") or not _owner_dead(row):
        return False
    try:
        _http("/gateway/drain", "DELETE", {"lease": row["lease"]}, token)
    except Exception as exc:       # 409 = the lease is not that fence any more (it expired or was replaced)
        _audit("orphan-drain-release-refused", dead_pid=row.get("pid"), error=repr(exc))
        return False
    _audit("orphan-drain-released", dead_pid=row.get("pid"), prior_by=row.get("by"))
    _clear_drain_record(row["lease"])
    return True


def _fence_failed_release(token: str, timeout_s: float) -> str:
    """A failed new process may already have accepted work; fence it before rollback.

    An unreadable process is ambiguous, including when systemd says it failed.
    Leave it in place for diagnosis instead of issuing another destructive restart.
    Returns the fence lease so the caller releases it; if the wait itself fails the fence is released here.
    """
    current = _http("/gateway/drain", token=token)
    if current.get("draining"):
        raise RuntimeError("new gateway already has an unknown drain owner")
    opened = _http("/gateway/drain", "POST",
                   {"ttl_s": 1800, "reason": "governed gateway rollback", "by": _BY}, token)
    lease = opened.get("lease")
    if not lease:
        raise RuntimeError("new gateway did not grant a rollback drain lease")
    _record_drain(lease)
    try:
        _wait_empty(token, timeout_s)
    except BaseException:
        _release_drain(token, lease)
        raise
    return lease


PUBLISH_STATE_ROOT = RUNTIME.parent / "gateway-publications"
DURABLE_DRAIN_STATE = RUNTIME.parent / "gateway-drain-state.json"
MAINTENANCE_UNITS = ("estate-self-repair.timer", "estate-self-repair.service", "estate-core.service")
CONTROL_CODE = ("gateway_safe_publish.py", "windowctl.py", "experiment_controls.py", "gpuguard.py", "unitrun.py",
                "gateway_publish_bundle.py")
_CONTROL_API = None


def _controls():
    """Use the existing window controller's typed unit, hold and canary mechanics."""
    global _CONTROL_API
    if _CONTROL_API is None:
        sys.path.insert(0, str(HERE))
        spec = importlib.util.spec_from_file_location("gateway_publish_window_controls", HERE / "windowctl.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CONTROL_API = module
    _CONTROL_API.ACTUATOR = str(ACTUATOR)
    return _CONTROL_API


def _bundle():
    spec = importlib.util.spec_from_file_location("gateway_publish_bundle", HERE / "gateway_publish_bundle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _refuse_core_cgroup():
    try:
        groups = Path("/proc/self/cgroup").read_text()
    except OSError as error:
        raise RuntimeError("publication cgroup is unreadable") from error
    if any("estate-core.service" in row.split("/") for row in groups.splitlines()):
        raise RuntimeError("Core cannot pause its own control-group; use externally owned --submit maintenance")


def _known_count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _assert_engine(expected):
    if not expected or _controls().runtime_identity(_controls().ENGINE_UNIT) != expected:
        raise RuntimeError("engine process generation or exact serving configuration changed")


def _hold_call(*args):
    row = _controls().actuator_hold(*args, timeout=30)
    if not isinstance(row, dict) or row.get("error") or row.get("refused"):
        raise RuntimeError("engine actuator hold authority unavailable or refused")
    return row


def _acquire_publish_hold(ttl_s):
    # Plain acquire-mode survives writer death; recovery renews the bounded
    # lease. An owned hold would become void exactly when recovery is needed.
    row = _hold_call("acquire", "--kind", "engine", "--by", "gateway-safe-publish",
                     "--reason", "governed gateway publication and independent recovery", "--ttl", str(int(ttl_s)))
    if not row.get("lease"):
        raise RuntimeError("engine actuator did not grant a publication hold")
    return row["lease"]


def _renew_publish_hold(state):
    if state.get("actuator_sha256") and (not ACTUATOR.is_file() or _sha(ACTUATOR.read_bytes()) != state["actuator_sha256"]):
        raise RuntimeError("publication actuator code changed during maintenance")
    if state.get("hold"):
        row = _hold_call("renew", "--lease", state["hold"], "--ttl", "3600")
        if row.get("lease") != state["hold"]:
            raise RuntimeError("publication engine hold expired or changed owner")


def _save_state(path, state):
    _atomic_write(path, (json.dumps(state, sort_keys=True) + "\n").encode(), 0o600)


@contextlib.contextmanager
def _exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another serving maintenance transaction holds " + path.name) from error
        yield


def _capture_committed():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    sources = {"keepalive-shim.py": SOURCE.read_bytes(), "gateway_dashboard.html": DASH_SOURCE.read_bytes(), **_module_sources()}
    code = {name: (HERE / name).read_bytes() for name in CONTROL_CODE}
    for name, data in {**sources, **code}.items():
        if data != subprocess.check_output(["git", "show", f"{head}:deploy/bin/{name}"], cwd=REPO):
            raise RuntimeError(f"{name} source differs from committed HEAD")
    if DROPIN.read_bytes() != subprocess.check_output(["git", "show", f"{head}:deploy/systemd/vllm-keepalive-shim.service.d/zz-graceful-stop.conf"], cwd=REPO):
        raise RuntimeError("graceful-stop prerequisite source differs from committed HEAD")
    if head != subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip():
        raise RuntimeError("publication HEAD changed during source capture")
    return head, sources, code


def _snapshot_files(directory, sources):
    records = {}
    backup = directory / "files"
    backup.mkdir(mode=0o700)
    for name, candidate in sources.items():
        path = RUNTIME.with_name(name)
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise RuntimeError(f"nonregular runtime file: {name}")
        old = path.read_bytes() if path.exists() else None
        mode = path.stat().st_mode & 0o7777 if old is not None else None
        if old is not None:
            _atomic_write(backup / (name + ".previous"), old, 0o600)
        _atomic_write(backup / (name + ".candidate"), candidate, 0o600)
        records[name] = {"previous_sha256": _sha(old) if old is not None else None, "previous_mode": mode,
                         "candidate_sha256": _sha(candidate), "candidate_mode": 0o755 if name == "keepalive-shim.py" else 0o644}
    if records["keepalive-shim.py"]["previous_sha256"] is None:
        raise RuntimeError("live gateway file is missing")
    return records


def _file_matches(path, digest, mode):
    if digest is None:
        return not path.exists() and not path.is_symlink()
    return (not path.is_symlink() and path.is_file() and _sha(path.read_bytes()) == digest
            and path.stat().st_mode & 0o7777 == mode)


def _apply_files(state_path, state, restored=False):
    which = "previous" if restored else "candidate"
    # Refuse a foreign writer before touching any destination.
    for name, row in state["files"].items():
        if not any(_file_matches(RUNTIME.with_name(name), row[k + "_sha256"], row[k + "_mode"])
                   for k in ("previous", "candidate")):
            raise RuntimeError(f"runtime file changed outside publication: {name}")
    captured = {}
    for name, row in state["files"].items():
        if row[which + "_sha256"] is not None:
            data = (state_path.parent / "files" / (name + "." + which)).read_bytes()
            if _sha(data) != row[which + "_sha256"]:
                raise RuntimeError(f"publication backup hash mismatch: {name}")
            captured[name] = data
    for name, row in state["files"].items():
        path = RUNTIME.with_name(name)
        if row[which + "_sha256"] is None:
            path.unlink(missing_ok=True)
            continue
        _atomic_write(path, captured[name], row[which + "_mode"])
    if not _files_verified(state, restored):
        raise RuntimeError("publication file readback mismatch")


def _files_verified(state, restored=False):
    which = "previous" if restored else "candidate"
    return all(_file_matches(RUNTIME.with_name(name), row[which + "_sha256"], row[which + "_mode"])
               for name, row in state["files"].items())


def _persist_bootstrap_fence(opened):
    lease, until = opened.get("lease"), opened.get("until")
    if not isinstance(lease, str) or not re.fullmatch(r"[0-9a-f]{32}", lease) or not _known_count(opened.get("active")):
        raise RuntimeError("gateway did not grant an identifiable drain")
    if not isinstance(until, (int, float)) or isinstance(until, bool) or until - time.time() < 30:
        raise RuntimeError("gateway granted an expired drain")
    if DURABLE_DRAIN_STATE.exists():
        info = DURABLE_DRAIN_STATE.stat()
        if DURABLE_DRAIN_STATE.is_symlink() or not DURABLE_DRAIN_STATE.is_file() or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise RuntimeError("untrusted durable drain file")
        prior = json.loads(DURABLE_DRAIN_STATE.read_text())
        if prior.get("lease") != lease or prior.get("durable") is not True:
            raise RuntimeError("unknown durable drain owner")
        return
    row = {"v": 1, "durable": True, "lease": lease, "until": until, "reason": "governed gateway publish",
           "record": {"t0": time.time(), "reason": "governed gateway publish", "by": _BY, "ttl_s": 1800,
                      "active_at_open": opened["active"], "refused": 0, "refused_by_client": {}}}
    _atomic_write(DURABLE_DRAIN_STATE, (json.dumps(row, sort_keys=True) + "\n").encode(), 0o600)


def _arm_recovery(state_path, state):
    unit = state["recovery_unit"]
    # Snapshot executable code: recovery must not follow later checkout edits.
    program = state_path.parent / "code" / "gateway_safe_publish.py"
    env = {**_controls().CONTROL_ENV, "ENGINE_ACTUATOR": str(ACTUATOR)}
    argv = ["systemd-run", "--user", f"--unit={unit}", "--collect", "--quiet", "--expand-environment=no",
            "--on-active=30", "--on-unit-active=30", "--timer-property=AccuracySec=5s",
            *[f"--setenv={k}={v}" for k, v in env.items()], "/usr/bin/python3", str(program), "--recover", str(state_path)]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError("independent publication recovery timer could not be armed")


def _restore_units(state):
    # Core resumes first, repair service next, timer last. Inactive units remain
    # inactive; a new process PID is expected for an originally active service.
    for unit in reversed(state["pause_units"]):
        original = state["unit_states"][unit]
        current = _controls().user_unit_state(unit)
        if current == "unknown" or (current != original and not _controls().set_user_unit(unit, original == "active")):
            raise RuntimeError(f"user unit restoration failed: {unit}")
        if _controls().user_unit_state(unit) != original:
            raise RuntimeError(f"user unit restoration readback failed: {unit}")


def _pause_units(state):
    for unit in state["pause_units"]:
        current = _controls().user_unit_state(unit)
        if current not in ("active", "inactive"):
            raise RuntimeError(f"maintenance unit activity is unknown: {unit}")
        if current == "active" and (not _controls().set_user_unit(unit, False)
                                    or _controls().user_unit_state(unit) != "inactive"):
            raise RuntimeError(f"maintenance stop/readback failed: {unit}")


def _recover_qualified_fence(token, state_path, state):
    """Partial release is retried under new qualification, never rollback."""
    _assert_engine(state["engine"])
    restored = state.get("qualified") == "previous"
    if not _files_verified(state, restored) or (state.get("bundle") and not _bundle().verify(state["bundle"], restored=restored)):
        raise RuntimeError("qualified publication bytes changed before recovery")
    _refuse_if_offline_window(token, "recovery")
    drain = _http("/gateway/drain", token=token)
    if drain.get("draining") is True:
        if (drain.get("durable") is not True or drain.get("invalid_state")
                or drain.get("lease_sha256") != _sha(state["lease"].encode())):
            raise RuntimeError("qualified recovery drain has an unknown owner")
        opened = _http("/gateway/drain", "POST", {"ttl_s": 1800, "durable": True, "lease": state["lease"]}, token)
    elif drain.get("draining") is False and not DURABLE_DRAIN_STATE.exists():
        opened = _http("/gateway/drain", "POST", {"ttl_s": 1800, "durable": True,
                       "reason": "qualified publication recovery", "by": _BY}, token)
        state["lease"] = opened.get("lease")
        _save_state(state_path, state)
    else:
        raise RuntimeError("qualified recovery drain state is unknown")
    _persist_bootstrap_fence(opened)
    _pause_units(state)
    _assert_engine(state["engine"])
    if not ACTUATOR.is_file() or _sha(ACTUATOR.read_bytes()) != state["actuator_sha256"]:
        raise RuntimeError("publication actuator code changed during recovery")
    row = _controls().actuator_hold("renew", "--lease", state["hold"], "--ttl", "3600", timeout=30)
    if isinstance(row, dict) and row.get("refused") == "no such live hold (released, expired, or its holder died)":
        state["hold"] = _acquire_publish_hold(3600)  # existing authority refuses any foreign holder
        _save_state(state_path, state)
    elif not isinstance(row, dict) or row.get("error") or row.get("refused") or row.get("lease") != state["hold"]:
        raise RuntimeError("qualified publication hold recovery failed")
    state.update(drain_closed=False, hold_closed=False, units_restored=False)
    _save_state(state_path, state)
    _wait_empty(token, 120, state)


def _retain_qualified_fence(token, state_path, state):
    """Failed cleanup cannot leave the current process admitting new work."""
    try:
        _pause_units(state)
    except Exception as error:
        state["producer_pause_error"] = type(error).__name__
    try:
        drain = _http("/gateway/drain", token=token)
        if (drain.get("draining") is True and drain.get("durable") is True and not drain.get("invalid_state")
                and drain.get("lease_sha256") == _sha(state["lease"].encode())):
            opened = _http("/gateway/drain", "POST", {"ttl_s": 1800, "durable": True, "lease": state["lease"]}, token)
        elif drain.get("draining") is False and not DURABLE_DRAIN_STATE.exists():
            opened = _http("/gateway/drain", "POST", {"ttl_s": 1800, "durable": True,
                           "reason": "qualified publication cleanup incomplete", "by": _BY}, token)
            state["lease"] = opened.get("lease")
            _save_state(state_path, state)
        else:
            raise RuntimeError("cleanup admission fence is unknown")
        _persist_bootstrap_fence(opened)
        state["drain_closed"] = False
        _save_state(state_path, state)
    except Exception as error:
        state["fence_retention_error"] = type(error).__name__
        state["deadline"] = time.time() + 1900  # allow the verified accepted-call stop grace
        with contextlib.suppress(OSError):
            _save_state(state_path, state)
        _run("sudo", "-n", "systemctl", "stop", SERVICE)
        if _controls().main_pid(SERVICE) != 0:
            raise RuntimeError("failed qualification cleanup could not stop gateway admission")
        state["status"] = "manual-recovery-required"
        _save_state(state_path, state)


def _qualify(token, state, *, restored=False, require_new=False):
    _renew_publish_hold(state)
    _assert_engine(state["engine"])
    pid = _controls().main_pid(SERVICE)
    proc = _proc_stat(pid) if isinstance(pid, int) and pid > 0 else None
    if not proc or proc[0] in {"Z", "X"} or (require_new and pid == state["gateway_pid"]):
        raise RuntimeError("gateway restart did not produce a new SYSTEM-unit PID")
    if _http("/health", token=token).get("http_status") != 200:
        raise RuntimeError("gateway health unreadable")
    spend = _http("/gateway/spend", token=token)
    which = "previous" if restored else "candidate"
    if not (spend.get("enforce") is True and spend.get("durable") is True and spend.get("cap") == 25.0
            and spend.get("attribution_gap_usd") == 0
            and spend.get("gateway_sha256") == state["files"]["keepalive-shim.py"][which + "_sha256"]):
        raise RuntimeError("gateway spend/source readback failed")
    drain = _http("/gateway/drain", token=token)
    if (drain.get("draining") is not True or drain.get("durable") is not True or drain.get("invalid_state")
            or drain.get("lease_sha256") != _sha(state["lease"].encode())):
        raise RuntimeError("restarted gateway did not retain the durable all-traffic fence")
    persisted = json.loads(DURABLE_DRAIN_STATE.read_text())
    if persisted.get("lease") != state["lease"] or persisted.get("durable") is not True:
        raise RuntimeError("durable publication fence readback failed")
    sources = {name: RUNTIME.with_name(name).read_bytes() for name in _module_names()
               if state["files"].get(name, {}).get(which + "_sha256") is not None}
    if not _modules_loaded_match(token, sources) or not _files_verified(state, restored):
        raise RuntimeError("published file or startup-module readback failed")
    if state.get("bundle") and not _bundle().verify(state["bundle"], restored=restored):
        raise RuntimeError("control bundle readback failed")
    canary = _controls().restoration_generation(state["engine"], timeout=60)
    _assert_engine(state["engine"])
    after = _proc_stat(pid)
    if _controls().main_pid(SERVICE) != pid or not after or after[0] in {"Z", "X"} or after[1] != proc[1]:
        raise RuntimeError("gateway generation changed during qualification")
    if (not isinstance(canary, dict) or canary.get("ok") is not True or canary.get("runtime_unchanged") is not True
            or canary.get("finish_reason") != "stop" or not _known_count(canary.get("completion_tokens"))
            or canary["completion_tokens"] <= 0):
        raise RuntimeError("engine did not complete the direct successful generation canary")
    state["gateway_generation"] = {"pid": pid, "pid_start": proc[1]}
    return canary


def _finish_release(token, state_path, state, restored=False):
    # Qualification has completed while producers and all traffic remain held.
    state["qualified"] = "previous" if restored else "candidate"
    _save_state(state_path, state)
    _restore_units(state)
    _assert_engine(state["engine"])
    state["units_restored"] = True
    _save_state(state_path, state)
    if state.get("lease"):
        current = _http("/gateway/drain", token=token)
        if current.get("draining") is True:
            result = _http("/gateway/drain", "DELETE", {"lease": state["lease"]}, token)
            if result.get("draining") is not False:
                raise RuntimeError("qualified gateway did not close the owned drain")
        elif current.get("draining") is not False or DURABLE_DRAIN_STATE.exists():
            raise RuntimeError("qualified drain release state is unknown")
        # A bootstrap old process does not delete the new durable state itself.
        if DURABLE_DRAIN_STATE.exists():
            row = json.loads(DURABLE_DRAIN_STATE.read_text())
            if row.get("lease") != state["lease"]:
                raise RuntimeError("durable drain ownership changed during release")
            DURABLE_DRAIN_STATE.unlink()
            fd = os.open(DURABLE_DRAIN_STATE.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        state["drain_closed"] = True
        _save_state(state_path, state)
    if state.get("hold"):
        _assert_engine(state["engine"])
        if _hold_call("release", "--lease", state["hold"]).get("released") != 1:
            raise RuntimeError("publication engine hold release failed")
        state["hold_closed"] = True
        _save_state(state_path, state)
    state["status"] = "rolled-back" if restored else "published"
    state["finished"] = time.time()
    _save_state(state_path, state)
    try:
        cleanup = subprocess.run(["systemctl", "--user", "stop", state["recovery_unit"] + ".timer"],
                                 capture_output=True, timeout=30)
        if cleanup.returncode:
            _audit("recovery-timer-cleanup-pending", receipt=str(state_path))
    except (OSError, subprocess.SubprocessError):
        _audit("recovery-timer-cleanup-pending", receipt=str(state_path))


def _supports_durable(data):
    import ast
    try:
        return any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DURABLE_DRAIN_PROTOCOL" for t in n.targets)
                   and isinstance(n.value, ast.Constant) and n.value.value == 1 for n in ast.parse(data).body)
    except (SyntaxError, ValueError):
        return False


def _rollback(token, state_path, state):
    _renew_publish_hold(state)
    _assert_engine(state["engine"])
    if state.get("drain_requested") and not state.get("lease"):
        raise RuntimeError("drain grant is unknown; automatic release cannot invent its owner")
    _pause_units(state)
    if state.get("restart_intent"):
        # Never restore files underneath uncounted accepted work.
        _wait_empty(token, 120, state)
        _apply_files(state_path, state, restored=True)
        if state.get("bundle"):
            _bundle().restore(state["bundle"])
        previous = (state_path.parent / "files" / "keepalive-shim.py.previous").read_bytes()
        if not _supports_durable(previous):
            state["deadline"] = time.time() + 1900
            _save_state(state_path, state)
            _run("sudo", "-n", "systemctl", "stop", SERVICE)
            if _controls().main_pid(SERVICE) != 0:
                raise RuntimeError("legacy rollback gateway stop did not clear the SYSTEM-unit PID")
            state["status"] = "manual-recovery-required"
            _save_state(state_path, state)
            raise RuntimeError("legacy rollback has no durable all-traffic fence; exact files restored, gateway kept stopped")
        state["gateway_pid"] = _controls().main_pid(SERVICE)
        _save_state(state_path, state)
        _run("sudo", "-n", "systemctl", "restart", SERVICE)
        state["generation_canary"] = _qualify(token, state, restored=True, require_new=True)
    else:
        # Original process still has the owned in-memory fence. Verify original
        # bytes and exact engine generation before restoring its producer state.
        _apply_files(state_path, state, restored=True)
        if state.get("bundle"):
            _bundle().restore(state["bundle"])
        _assert_engine(state["engine"])
        state["generation_canary"] = _controls().restoration_generation(state["engine"], timeout=60)
        if (state["generation_canary"].get("ok") is not True or state["generation_canary"].get("runtime_unchanged") is not True
                or state["generation_canary"].get("finish_reason") != "stop"
                or not _known_count(state["generation_canary"].get("completion_tokens"))
                or state["generation_canary"]["completion_tokens"] <= 0):
            raise RuntimeError("original engine generation canary failed during abort")
        _assert_engine(state["engine"])
    _finish_release(token, state_path, state, restored=True)


def _legacy_fallback(state_path, state):
    """A bootstrap fence cannot rely on a legacy process's expiring memory.

    A manual SYSTEM-unit stop honors its verified accepted-call grace. It does
    not restart or modify the engine and prevents TTL from reopening generation.
    """
    if state.get("qualified") == "candidate":
        return  # already qualified bytes must never be reverted by release cleanup
    if not state.get("lease") and not state.get("drain_requested"):
        return
    previous = (state_path.parent / "files" / "keepalive-shim.py.previous").read_bytes()
    if _supports_durable(previous):
        return
    state["deadline"] = time.time() + 1900
    with contextlib.suppress(OSError):
        _save_state(state_path, state)
    try:
        _pause_units(state)
        _renew_publish_hold(state)
    except Exception as error:
        state["fallback_authority_error"] = type(error).__name__
        with contextlib.suppress(OSError):
            _save_state(state_path, state)
    _run("sudo", "-n", "systemctl", "stop", SERVICE)
    if _controls().main_pid(SERVICE) != 0:
        raise RuntimeError("legacy recovery fallback did not stop the gateway SYSTEM unit")
    _apply_files(state_path, state, restored=True)
    if state.get("bundle"):
        _bundle().restore(state["bundle"])
    state["status"] = "manual-recovery-required"
    _save_state(state_path, state)


def publish(timeout_s: float = 1500, halo_wait_s: float = 1800, *, control_bundle=None, state_dir=None, expected_head=None) -> dict:
    """External maintenance transaction; interrupted work remains owned by recovery."""
    with _terminate_as_exception(), _exclusive(Path("/tmp/vllm-gateway-publish.lock")), _exclusive(_controls().LOCK):
        return _publish(timeout_s, halo_wait_s, control_bundle=control_bundle, state_dir=state_dir, expected_head=expected_head)


def _publish(timeout_s: float, halo_wait_s: float, *, control_bundle=None, state_dir=None, expected_head=None) -> dict:
    if not 1 <= timeout_s <= 1700 or not 1 <= halo_wait_s <= 1800:
        raise ValueError("publication waits outside documented bounds")
    if control_bundle not in (None, "model-switch"):
        raise ValueError("unknown opt-in control bundle")
    _refuse_core_cgroup()
    head, sources, code = _capture_committed()
    if expected_head is not None and head != expected_head:
        raise RuntimeError("publisher HEAD differs from the requested committed HEAD")
    token = os.environ.get("SHIM_ADMIN_TOKEN") or TOKEN_FILE.read_text().strip()
    if not token:
        raise RuntimeError("configured gateway admin token is required")
    # These gates also run for a no-op/page-only publication. It must not be a
    # way around active windows or unknown serving authority.
    _refuse_if_offline_window(token, "start")
    drain = _http("/gateway/drain", token=token)
    spend = _http("/gateway/spend", token=token)
    if (drain.get("draining") is not False or not _known_count(drain.get("active"))
            or not _known_count(spend.get("in_flight")) or spend.get("enforce") is not True
            or spend.get("durable") is not True or spend.get("cap") != 25.0
            or spend.get("attribution_gap_usd") != 0):
        raise RuntimeError("gateway drain/spend authority is unknown or already owned")
    if DURABLE_DRAIN_STATE.exists():
        raise RuntimeError("an unresolved durable gateway drain requires recovery")
    # Host prerequisite, not a publication mutation: exact rollback must not
    # leave a newly installed privileged drop-in behind.
    grace = _unit_seconds(subprocess.check_output(["systemctl", "show", "-p", "TimeoutStopUSec", "--value", SERVICE], text=True).strip())
    if not DROPIN_LIVE.is_file() or DROPIN_LIVE.read_bytes() != DROPIN.read_bytes() or grace < 1830:
        raise RuntimeError("committed graceful-stop host prerequisite is not installed")
    api = _controls()
    engine = api.runtime_identity(api.ENGINE_UNIT)
    if not engine:
        raise RuntimeError("engine runtime identity is unreadable")
    gateway_pid = api.main_pid(SERVICE)
    gateway_stat = _proc_stat(gateway_pid) if isinstance(gateway_pid, int) and gateway_pid > 0 else None
    if not gateway_stat or gateway_stat[0] in {"Z", "X"}:
        raise RuntimeError("gateway SYSTEM-unit generation is unreadable")
    unit_states = {unit: api.user_unit_state(unit) for unit in MAINTENANCE_UNITS}
    if any(value not in ("active", "inactive") for value in unit_states.values()):
        raise RuntimeError("maintenance unit activity is unknown")
    directory = Path(state_dir) if state_dir else PUBLISH_STATE_ROOT / ("publish-" + secrets.token_hex(12))
    if directory.resolve().parent != PUBLISH_STATE_ROOT.resolve():
        raise RuntimeError("publication state must be under its private transaction root")
    directory.mkdir(mode=0o700, parents=True, exist_ok=state_dir is not None)
    state_path = directory / "state.json"
    state = {"v": 1, "transaction": directory.name, "status": "preparing", "source_head": head,
             **_own_stamp(), "deadline": time.time() + timeout_s + 300, "engine": engine,
             "gateway_pid": gateway_pid, "gateway_start": gateway_stat[1], "pause_units": list(MAINTENANCE_UNITS),
             "unit_states": unit_states, "hold": None, "lease": None, "installed": False, "restart_intent": False,
             "recovery_unit": "gateway-publish-" + secrets.token_hex(12) + "-recovery",
             "control_bundle": control_bundle, "files": _snapshot_files(directory, sources),
             "actuator_sha256": _sha(ACTUATOR.read_bytes())}
    (directory / "code").mkdir(mode=0o700)
    for name, data in code.items():
        _atomic_write(directory / "code" / name, data, 0o600)
    state["code_hashes"] = {name: _sha(data) for name, data in code.items()}
    if control_bundle:
        state["bundle"] = _bundle().snapshot(HERE, RUNTIME.parent, directory / "control-files")
    _save_state(state_path, state)  # exact snapshots precede every maintenance action
    PUBLISH_STATE_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    _atomic_write(PUBLISH_STATE_ROOT / "active.json", (json.dumps({"receipt": str(state_path), "source_head": head}) + "\n").encode(), 0o600)
    _arm_recovery(state_path, state)
    try:
        state["hold"] = _acquire_publish_hold(max(3600, timeout_s + 900))
        _save_state(state_path, state)
        _refuse_if_offline_window(token, "fence")
        state["drain_requested"] = True
        _save_state(state_path, state)
        opened = _http("/gateway/drain", "POST", {"ttl_s": 1800, "durable": True,
                       "reason": "governed gateway publish", "by": _BY}, token)
        state["lease"] = opened.get("lease")
        _save_state(state_path, state)  # grant survives a kill before bootstrap persistence
        _persist_bootstrap_fence(opened)
        _pause_units(state)
        _assert_engine(engine)
        _wait_empty(token, timeout_s, state)
        _refuse_if_offline_window(token, "restart")
        _renew_publish_hold(state)
        _assert_engine(engine)
        state["installed"] = True  # partial atomic install is also recoverable
        state["status"] = "installing"
        _save_state(state_path, state)
        _apply_files(state_path, state)
        if state.get("bundle"):
            _bundle().install(state["bundle"])
        restart = any(row["previous_sha256"] != row["candidate_sha256"]
                      for name, row in state["files"].items() if name != "gateway_dashboard.html")
        # A legacy process must restart once to gain durable admission, even if
        # its source happens to compare equal to an incomplete deployment.
        restart = restart or _http("/gateway/drain", token=token).get("durable") is not True
        if restart:
            state["restart_intent"] = True
            _save_state(state_path, state)
            _run("sudo", "-n", "systemctl", "restart", SERVICE)
        state["status"] = "qualifying"
        _save_state(state_path, state)
        deadline = time.monotonic() + 90
        while True:
            try:
                state["generation_canary"] = _qualify(token, state, require_new=restart)
                break
            except (OSError, RuntimeError, urllib.error.URLError):
                # An engine change is irreversible for this publisher: don't
                # wait or overwrite it with a gateway rollback.
                _assert_engine(engine)
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)
        _finish_release(token, state_path, state)
        return _receipt(state_path, state)
    except BaseException as error:
        state["error"] = type(error).__name__ + ": " + str(error)[:300]
        # A qualified candidate is not rolled back because timer cleanup or
        # producer restoration failed. Recovery finishes that same release.
        if state.get("qualified") == "candidate":
            state["status"] = "release-pending"
            with _shielded():
                with contextlib.suppress(OSError):
                    _save_state(state_path, state)
                try:
                    _retain_qualified_fence(token, state_path, state)
                except BaseException as retention_error:
                    state["retention_error"] = type(retention_error).__name__
                    with contextlib.suppress(OSError):
                        _save_state(state_path, state)
        else:
            state["status"] = "recovery-required"
            _save_state(state_path, state)
            with _shielded():
                try:
                    _rollback(token, state_path, state)
                except BaseException as recovery_error:
                    state["recovery_error"] = type(recovery_error).__name__ + ": " + str(recovery_error)[:300]
                    if state.get("status") != "manual-recovery-required":
                        state["status"] = "recovery-required"
                    _save_state(state_path, state)
                    try:
                        _legacy_fallback(state_path, state)
                    except BaseException as stop_error:
                        state["fallback_error"] = type(stop_error).__name__
                        _save_state(state_path, state)
        _audit("publication-failed", receipt=str(state_path), status=state["status"], error=state["error"])
        raise


def _receipt(path, state):
    status = "published" if state.get("status") == "published" else (
        "recovery-required" if state.get("status") in ("rolled-back", "recovery-required", "manual-recovery-required") else "submitted")
    return {"status": status,
            "transaction": state["transaction"], "receipt": str(path), "source_head": state["source_head"],
            "sha256": state["files"]["keepalive-shim.py"]["candidate_sha256"],
            "modules": {name: state["files"][name]["candidate_sha256"] for name in _module_names() if name in state["files"]},
            "control_bundle": state.get("control_bundle"), "qualified": state.get("qualified"),
            "engine_fingerprint": state["engine"]["fingerprint"], "generation_canary": state.get("generation_canary"),
            "gateway_generation": state.get("gateway_generation")}


def _read_recovery(path):
    path = Path(path)
    if path.is_symlink() or path.parent.is_symlink():
        raise RuntimeError("untrusted publication recovery journal")
    path = path.resolve()
    if (path.name != "state.json" or not path.is_file()
            or path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077):
        raise RuntimeError("untrusted publication recovery journal")
    state = json.loads(path.read_text())
    if (state.get("v") != 1 or state.get("transaction") != path.parent.name
            or path.parent.parent != PUBLISH_STATE_ROOT.resolve()
            or state.get("pause_units") != list(MAINTENANCE_UNITS)
            or set(state.get("unit_states", {})) != set(MAINTENANCE_UNITS)
            or any(v not in ("active", "inactive") for v in state["unit_states"].values())):
        raise RuntimeError("invalid publication recovery journal")
    allowed = {"keepalive-shim.py", "gateway_dashboard.html", *_module_names()}
    if (not isinstance(state.get("files"), dict)
            or not {"keepalive-shim.py", "gateway_dashboard.html", *REQUIRED_GATEWAY_MODULES}.issubset(state["files"])
            or not set(state["files"]).issubset(allowed)):
        raise RuntimeError("invalid publication recovery file manifest")
    for row in state["files"].values():
        for which in ("previous", "candidate"):
            digest, mode = row.get(which + "_sha256"), row.get(which + "_mode")
            if which == "previous" and digest is None and mode is None:
                continue
            if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                    or isinstance(mode, bool) or not isinstance(mode, int) or not 0 <= mode <= 0o7777):
                raise RuntimeError("invalid publication recovery file metadata")
    for name, digest in state.get("code_hashes", {}).items():
        if name not in CONTROL_CODE or _sha((path.parent / "code" / name).read_bytes()) != digest:
            raise RuntimeError("publication recovery code changed")
    if set(state.get("code_hashes", {})) != set(CONTROL_CODE):
        raise RuntimeError("publication recovery code manifest incomplete")
    return path, state


def recover(path):
    path, state = _read_recovery(path)
    if state.get("status") in ("published", "rolled-back"):
        return _receipt(path, state)
    if not _owner_dead(state) and time.time() < state["deadline"]:
        return {"status": "pending", "receipt": str(path)}
    if not _owner_dead(state):
        # Stop only the exact publisher PID/start generation, never the Core
        # control group or a pattern matching unrelated probes.
        os.kill(state["pid"], signal.SIGTERM)
        for _ in range(10):
            if _owner_dead(state):
                break
            time.sleep(1)
        if not _owner_dead(state):
            raise RuntimeError("expired publisher still owns its recovery transaction")
    _refuse_core_cgroup()
    with _exclusive(Path("/tmp/vllm-gateway-publish.lock")), _exclusive(_controls().LOCK), _shielded():
        token = ""
        try:
            if state.get("status") == "manual-recovery-required":
                _renew_publish_hold(state)
                return {"status": "recovery-required", "receipt": str(path), "why": "gateway remains stopped for manual recovery"}
            token = os.environ.get("SHIM_ADMIN_TOKEN") or TOKEN_FILE.read_text().strip()
            if state.get("qualified") in ("candidate", "previous"):
                _recover_qualified_fence(token, path, state)
                restored = state["qualified"] == "previous"
                state["generation_canary"] = _qualify(token, state, restored=restored)
                _finish_release(token, path, state, restored=restored)
            else:
                _renew_publish_hold(state)
                _rollback(token, path, state)
        except Exception as error:
            state["recovery_error"] = type(error).__name__
            with contextlib.suppress(OSError):
                _save_state(path, state)
            if state.get("qualified") == "candidate":
                _retain_qualified_fence(token, path, state)
            else:
                _legacy_fallback(path, state)
            raise
    return _receipt(path, state)


def _submission_unit_state(unit):
    known = _controls().user_unit_state(unit)
    if known != "unknown":
        return known
    # The maintenance snapshot deliberately rejects transitions. Submission
    # has a different contract: a known loaded activating worker is pending.
    try:
        result = subprocess.run(["systemctl", "--user", "show", unit, "--property=LoadState,ActiveState"],
                                capture_output=True, text=True, timeout=15)
        values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        if result.returncode == 0 and values.get("LoadState") == "loaded" and values.get("ActiveState") == "activating":
            return "activating"
    except (OSError, subprocess.SubprocessError, AttributeError):
        pass
    return "unknown"


def submit(timeout_s=1500, halo_wait_s=1800, *, control_bundle=None, expected_head=None):
    head, sources, _ = _capture_committed()
    if expected_head is not None and expected_head != head:
        raise RuntimeError("publisher HEAD differs from the Core's expected committed HEAD")
    active = PUBLISH_STATE_ROOT / "active.json"
    if active.exists():
        pointer = json.loads(active.read_text())
        if not Path(pointer["receipt"]).exists():
            request = json.loads((Path(pointer["receipt"]).parent / "submission.json").read_text())
            if request["source_head"] != head or request.get("control_bundle") != control_bundle:
                raise RuntimeError("another committed publication subject is submitted")
            if request.get("status") == "submission-failed" or _submission_unit_state(request["unit"] + ".service") not in ("active", "activating"):
                return {"status": "recovery-required", "receipt": pointer["receipt"], "source_head": head}
            return {"status": "submitted", **request, "sha256": _sha(sources["keepalive-shim.py"]),
                    "modules": {n: _sha(sources[n]) for n in _module_names() if n in sources}}
        path, state = _read_recovery(pointer["receipt"])
        if state["source_head"] != head or state.get("control_bundle") != control_bundle:
            if state.get("status") not in ("published", "rolled-back"):
                raise RuntimeError("another committed publication subject is active")
        elif state.get("status") == "published":
            token = os.environ.get("SHIM_ADMIN_TOKEN") or TOKEN_FILE.read_text().strip()
            if (not _files_verified(state) or not _modules_loaded_match(token, {n: sources[n] for n in _module_names() if n in sources})
                    or (state.get("bundle") and not _bundle().verify(state["bundle"]))):
                raise RuntimeError("completed publication receipt no longer matches runtime")
            if _controls().runtime_identity(_controls().ENGINE_UNIT) == state["engine"]:
                gateway = state.get("gateway_generation", {})
                proc = _proc_stat(gateway.get("pid"))
                if (_controls().main_pid(SERVICE) != gateway.get("pid")
                        or not proc or proc[0] in {"Z", "X"} or proc[1] != gateway.get("pid_start")):
                    # A loaded shim replacement must be qualified again too.
                    gateway = None
                _refuse_if_offline_window(token, "receipt")
                health, spend = _http("/health", token=token), _http("/gateway/spend", token=token)
                if (health.get("http_status") != 200 or spend.get("gateway_sha256") != _sha(sources["keepalive-shim.py"])
                        or spend.get("enforce") is not True or spend.get("durable") is not True or spend.get("cap") != 25.0
                        or spend.get("attribution_gap_usd") != 0):
                    raise RuntimeError("completed publication authority readback failed")
                if gateway is not None:
                    return _receipt(path, state)
            # A completed receipt belongs to its process generation. Requalify
            # a later authorized engine generation through a fresh transaction.
        elif state.get("status") in ("recovery-required", "manual-recovery-required", "rolled-back"):
            return {"status": "recovery-required", "receipt": str(path), "source_head": head}
        else:
            return _receipt(path, state)
    PUBLISH_STATE_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory = PUBLISH_STATE_ROOT / ("publish-" + secrets.token_hex(12))
    directory.mkdir(mode=0o700)
    unit = "gateway-publish-" + secrets.token_hex(12)
    # Submission metadata is separate from the recovery journal created by the
    # worker. It exists before launch, so the receipt survives its caller dying.
    request = {"v": 1, "source_head": head, "transaction": directory.name, "unit": unit,
               "receipt": str(directory / "state.json"), "control_bundle": control_bundle}
    _atomic_write(directory / "submission.json", (json.dumps(request, sort_keys=True) + "\n").encode(), 0o600)
    _atomic_write(active, (json.dumps(request, sort_keys=True) + "\n").encode(), 0o600)
    env = {**_controls().CONTROL_ENV, "ENGINE_ACTUATOR": str(ACTUATOR)}
    argv = ["systemd-run", "--user", f"--unit={unit}", "--collect", "--quiet", "--expand-environment=no",
            "--property=KillMode=control-group", "--property=TimeoutStopSec=1800",
            *[f"--setenv={k}={v}" for k, v in env.items()], "/usr/bin/python3", str(Path(__file__).resolve()),
            "--apply", "--state-dir", str(directory), "--expected-head", head,
            "--timeout-s", str(timeout_s), "--halo-wait-s", str(halo_wait_s)]
    if control_bundle:
        argv.extend(["--control-bundle", control_bundle])
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    if result.returncode:
        request["status"] = "submission-failed"
        _atomic_write(directory / "submission.json", (json.dumps(request, sort_keys=True) + "\n").encode(), 0o600)
        raise RuntimeError("independent publication unit could not be submitted")
    return {"status": "submitted", **request, "sha256": _sha(sources["keepalive-shim.py"]),
            "modules": {n: _sha(sources[n]) for n in _module_names() if n in sources}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--submit", action="store_true", help="dispatch independent maintenance; return a durable receipt")
    mode.add_argument("--recover", metavar="STATE", help="independent recovery timer entry point")
    parser.add_argument("--timeout-s", type=float, default=1500)
    parser.add_argument("--halo-wait-s", type=float, default=1800, help="legacy CLI compatibility; no retired Halo authority is changed")
    parser.add_argument("--control-bundle", choices=("model-switch",))
    parser.add_argument("--state-dir", help="private transaction directory for a submitted worker")
    parser.add_argument("--expected-head", help="exact committed source HEAD requested by Core")
    args = parser.parse_args()
    if not (args.apply or args.submit or args.recover):
        print(json.dumps({"status": "preview", "source_sha256": _sha(SOURCE.read_bytes()),
                          "live_sha256": _sha(RUNTIME.read_bytes()),
                          "dashboard_source_sha256": _sha(DASH_SOURCE.read_bytes()),
                          "dashboard_live_sha256": _sha(DASH_RUNTIME.read_bytes()) if DASH_RUNTIME.is_file() else None}, sort_keys=True))
        return
    try:
        if args.submit:
            with _exclusive(PUBLISH_STATE_ROOT / "submit.lock"):
                result = submit(args.timeout_s, args.halo_wait_s, control_bundle=args.control_bundle, expected_head=args.expected_head)
        elif args.recover:
            result = recover(args.recover)
        else:
            result = publish(args.timeout_s, args.halo_wait_s, control_bundle=args.control_bundle,
                             state_dir=args.state_dir, expected_head=args.expected_head)
        print(json.dumps(result, sort_keys=True))
        if result.get("status") == "recovery-required":
            sys.exit(1)
    except (Exception, PublishTerminated) as error:
        print(json.dumps({"status": "recovery-required", "error": type(error).__name__,
                          "why": str(error)[:300]}, sort_keys=True))
        sys.exit(128 + error.signum if isinstance(error, PublishTerminated) else 1)


if __name__ == "__main__":
    main()
