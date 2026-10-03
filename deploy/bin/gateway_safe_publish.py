#!/usr/bin/env python3
"""Publish a committed gateway after Halo runs and accepted calls have drained.

The gateway's /gateway/drain admission fence and /gateway/lanes.active share
one event loop. A request either registered before the fence and is counted,
or sees the fence and is refused with Retry-After before it can consume local
or paid work. A bounded wait aborts and releases the fence; it never kills an
accepted call to make a release finish on schedule.

Halo's separate bounded start lease pauses new incident runs while its fast
lease enforcer keeps existing repairs supervised. The publisher verifies that
the installed supervisor honors the lease before draining gateway calls.

L77 (2026-10-03): a publisher killed by SIGTERM/SIGHUP/SIGINT (e.g. an outer `timeout`) used to leave its Halo start pause
standing until expiry, which blocked Halo incident runs for an hour. Termination signals are now converted to an exception
so every `finally` runs (Halo pause released by token, drain fences closed, temp state removed). Both leases carry the
owner pid so a SIGKILLed publisher's leftovers are taken over by the next publish (audit-logged) instead of refused.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
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
    state = _install_config_schema()
    if old == new:
        return {"dashboard": "current", "previous": old, "changed": False, **state}
    _atomic_write(DASH_RUNTIME, new, 0o644)
    return {"dashboard": "installed", "previous": old, "changed": True, **state}


def _restore_dashboard(state: dict) -> None:
    _restore_config_schema(state)
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


def _install_config_schema() -> dict:
    src, dst = SOURCE.with_name(CONFIG_SCHEMA_NAME), RUNTIME.with_name(CONFIG_SCHEMA_NAME)
    if not src.is_file():
        return {"config_schema": "absent", "schema_changed": False}
    new = src.read_bytes()
    old = dst.read_bytes() if dst.is_file() else None
    if old == new:
        return {"config_schema": "current", "schema_changed": False}
    _atomic_write(dst, new, 0o644)
    return {"config_schema": "installed", "schema_previous": old, "schema_changed": True}


def _restore_config_schema(state: dict) -> None:
    if not state.get("schema_changed"):
        return
    dst = RUNTIME.with_name(CONFIG_SCHEMA_NAME)
    if state.get("schema_previous") is None:
        dst.unlink(missing_ok=True)
    else:
        _atomic_write(dst, state["schema_previous"], 0o644)


def _halo_active_runs() -> list[str]:
    """Do not restart the gateway between turns of an active Halo repair."""
    if not HALO_INCIDENTS.is_dir():
        raise RuntimeError("Halo incident authority is unreadable")
    active = []
    for path in HALO_INCIDENTS.glob("*.json"):
        try:
            row = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"unreadable Halo incident: {path.name}") from exc
        run = row.get("halo_run") or {}
        if (row.get("status") in {"assigned", "running", "repair-requested"}
                and run.get("status") in {"pending", "started", "queued", "running", "unknown", "stopping"}):
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


def _wait_empty(token: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        drain = _http("/gateway/drain", token=token)
        spend = _http("/gateway/spend", token=token)
        if not drain.get("draining") or float(drain.get("until") or 0) - time.time() < 30:
            raise RuntimeError("drain lease expired before accepted calls completed")
        if int(drain.get("active") or 0) == 0 and int(spend.get("in_flight") or 0) == 0:
            return
        time.sleep(3)
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


def publish(timeout_s: float = 1500, halo_wait_s: float = 1800) -> dict:
    """Publish with termination-safe leases: SIGTERM/SIGHUP/SIGINT become PublishTerminated and the finally releases."""
    with _terminate_as_exception():
        return _publish(timeout_s, halo_wait_s)


def _publish(timeout_s: float, halo_wait_s: float) -> dict:
    if not 1 <= timeout_s <= 1700:
        raise ValueError("timeout_s must be 1..1700")
    if not 1 <= halo_wait_s <= 1800:
        raise ValueError("halo_wait_s must be 1..1800")
    source = SOURCE.read_bytes()
    committed = subprocess.check_output(["git", "show", "HEAD:deploy/bin/keepalive-shim.py"], cwd=REPO)
    if source != committed:
        raise RuntimeError("gateway source differs from committed HEAD")
    dash = DASH_SOURCE.read_bytes()
    dash_committed = subprocess.check_output(["git", "show", "HEAD:deploy/bin/gateway_dashboard.html"], cwd=REPO)
    if dash != dash_committed:
        raise RuntimeError("dashboard source differs from committed HEAD")
    schema_src = SOURCE.with_name(CONFIG_SCHEMA_NAME)
    if schema_src.is_file() and schema_src.read_bytes() != subprocess.check_output(
            ["git", "show", "HEAD:deploy/bin/" + CONFIG_SCHEMA_NAME], cwd=REPO):
        raise RuntimeError("config schema source differs from committed HEAD")
    if not RUNTIME.is_file():
        raise RuntimeError("live gateway file is missing")
    previous = RUNTIME.read_bytes()
    if previous == source:
        # Same gateway code: the page alone may still be new. It is read per request, so no drain or restart is needed.
        state = _install_dashboard(dash)
        return {"status": "current", "sha256": _sha(source), "dashboard": state["dashboard"], "dashboard_sha256": _sha(dash),
                "config_schema": state.get("config_schema")}
    token = os.environ.get("SHIM_ADMIN_TOKEN") or TOKEN_FILE.read_text().strip()
    if _http("/health", token=token).get("http_status") != 200:
        raise RuntimeError("gateway health unreadable")
    spend = _http("/gateway/spend", token=token)
    if not (spend.get("enforce") and spend.get("durable") and float(spend.get("cap") or 0) == 25.0):
        raise RuntimeError("hard $25 spend authority is not healthy")
    drain_before = _http("/gateway/drain", token=token)
    if drain_before.get("draining") and not _reap_orphan_drain(token):
        raise RuntimeError("another publisher already holds the drain lease")

    halo_pause = None
    lease = None
    rollback_lease = None
    backup = None
    installed = False
    dash_state = {"changed": False}
    try:
        halo_pause = _begin_halo_quiesce(halo_wait_s, timeout_s)
        _assert_halo_quiesce_live(halo_pause)
        _wait_halo_quiet(halo_wait_s)
        # Unit readiness is installed without stopping the current process.
        _run("sudo", "-n", "install", "-D", "-m", "0644", str(DROPIN), str(DROPIN_LIVE))
        _run("sudo", "-n", "systemctl", "daemon-reload")
        stop_time = _unit_seconds(subprocess.check_output(["systemctl", "show", "-p", "TimeoutStopUSec",
                                                           "--value", SERVICE], text=True).strip())
        if stop_time < 1830:
            raise RuntimeError("systemd stop timeout is shorter than accepted-call grace")
        backup = RUNTIME.with_name(f"{RUNTIME.name}.bak-{int(time.time())}-{os.getpid()}")
        _atomic_write(backup, previous)
        opened = _http("/gateway/drain", "POST", {"ttl_s": 1800, "reason": "governed gateway publish", "by": _BY}, token)
        lease = opened["lease"]
        _record_drain(lease)
        _wait_empty(token, timeout_s)
        # The minute scheduler and ten-second enforcer share this lock. Hold
        # it only across the final active-run check and quick restart, never
        # during the potentially long drain.
        with HALO_SUPERVISOR_LOCK.open("a+") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            active = _halo_active_runs()
            if active:
                raise RuntimeError(f"Halo became active during gateway drain: {active[:6]}")
            dash_state = _install_dashboard(dash)      # before the restart: the new shim finds its page at once
            _atomic_write(RUNTIME, source)
            installed = True
            _run("sudo", "-n", "systemctl", "restart", SERVICE)
        for _ in range(30):
            try:
                health = _http("/health", token=token)
                new_spend = _http("/gateway/spend", token=token)
                if (health.get("http_status") == 200 and new_spend.get("gateway_sha256") == _sha(source)
                        and new_spend.get("enforce") and new_spend.get("durable")
                        and new_spend.get("attribution_gap_usd", 0) == 0):
                    return {"status": "published", "sha256": _sha(source), "dashboard": dash_state["dashboard"] if "dashboard" in dash_state else "current",
                            "dashboard_sha256": _sha(dash), "backup": str(backup), "spent": new_spend.get("spent"),
                            "config_schema": dash_state.get("config_schema")}
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError("new gateway failed health/spend/readback checks")
    except PublishTerminated as term:
        # Killed (outer timeout, operator, logout). Do not start a long fenced rollback inside a dying process: release
        # every lease in the finally below and say loudly what state the gateway is in.
        if installed:
            _audit("terminated-after-install", signal=term.signum, backup=str(backup),
                   note="new gateway file installed and restarted but NOT verified; roll back from the backup if it is unhealthy")
        else:
            with _shielded():
                _restore_dashboard(dash_state)
            _audit("terminated-before-install", signal=term.signum)
        raise
    except Exception as publish_error:
        if installed:
            try:
                rollback_lease = _fence_failed_release(token, timeout_s)
            except Exception as fence_error:
                raise RuntimeError(
                    "gateway publish failed; automatic rollback refused because the new process "
                    f"could not be safely drained: {fence_error}"
                ) from publish_error
            with HALO_SUPERVISOR_LOCK.open("a+") as guard:
                fcntl.flock(guard, fcntl.LOCK_EX)
                active = _halo_active_runs()
                if active:
                    raise RuntimeError(
                        f"gateway rollback refused while Halo runs are active: {active[:6]}") from publish_error
                _atomic_write(RUNTIME, previous)
                _restore_dashboard(dash_state)
                _run("sudo", "-n", "systemctl", "restart", SERVICE)
        else:
            _restore_dashboard(dash_state)
        raise
    finally:
        with _shielded():
            release_errors = []
            for held in (lease, rollback_lease):
                _release_drain(token, held)
            if backup is not None and not installed:
                with contextlib.suppress(OSError):
                    backup.unlink(missing_ok=True)       # a publish that never swapped the file has no use for its backup
            try:
                _end_halo_quiesce(halo_pause)           # halo_pause None: a signal landed before the token returned; match by pid
            except Exception as exc:
                release_errors.append(exc)
                _audit("halo-pause-release-failed", error=repr(exc))
        if release_errors and sys.exc_info()[0] is None:
            raise release_errors[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=1500)
    parser.add_argument("--halo-wait-s", type=float, default=1800)
    args = parser.parse_args()
    if not args.apply:
        print(json.dumps({"status": "preview", "source_sha256": _sha(SOURCE.read_bytes()),
                          "live_sha256": _sha(RUNTIME.read_bytes()),
                          "dashboard_source_sha256": _sha(DASH_SOURCE.read_bytes()),
                          "dashboard_live_sha256": _sha(DASH_RUNTIME.read_bytes()) if DASH_RUNTIME.is_file() else None},
                         sort_keys=True))
        return
    lock_path = Path("/tmp/vllm-gateway-publish.lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            print(json.dumps(publish(args.timeout_s, args.halo_wait_s), sort_keys=True))
        except PublishTerminated as term:
            # Leases are already released (publish()'s finally). Exit the conventional 128+signal.
            print(json.dumps({"status": "terminated", "signal": term.signum, "leases_released": True}, sort_keys=True))
            sys.exit(128 + term.signum)


if __name__ == "__main__":
    main()
