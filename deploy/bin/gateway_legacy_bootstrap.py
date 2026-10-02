#!/usr/bin/env python3
"""One-time bridge from the pre-drain gateway to the governed drain publisher.

Run as root. A finite-lived kernel rule rejects NEW connections while already
accepted TCP connections finish. Only when both the gateway's active registry
and kernel established sockets are empty is the old process restarted. Future
releases use gateway_safe_publish.py and its in-process admission fence.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

from gateway_safe_publish import (BASE, DROPIN, DROPIN_LIVE, REPO, RUNTIME,
                                  SERVICE, SOURCE, TOKEN_FILE, _atomic_write,
                                  _unit_seconds)

LOCK = Path("/tmp/vllm-gateway-publish.lock")
FENCE_MAX_S = 900
WAIT_MAX_S = 600


def _run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def _http(path: str, token: str = "") -> dict:
    req = urllib.request.Request(BASE + path,
                                 headers={"X-Admin-Token": token, "Connection": "close"})
    with urllib.request.urlopen(req, timeout=5) as response:
        return json.load(response)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rule(chain: str, expiry: str) -> list[str]:
    common = ["-p", "tcp", "--dport", "8000", "-m", "conntrack",
              "--ctstate", "NEW", "-m", "time", "--datestop", expiry,
              "-j", "REJECT", "--reject-with", "tcp-reset"]
    if chain == "INPUT":
        return ["!", "-i", "lo", *common]
    return ["-m", "owner", "!", "--uid-owner", "0", *common]


def _iptables(*args: str) -> None:
    subprocess.run(["iptables", "-w", *args], check=True, stdout=subprocess.DEVNULL)


def _established() -> int:
    return len(_run("ss", "-Htn", "state", "established", "(", "sport", "=", ":8000", ")")
               .splitlines())


def _wait_drained(token: str, timeout_s: float) -> dict:
    until = time.monotonic() + timeout_s
    consecutive = 0
    last = {}
    while time.monotonic() < until:
        lanes = _http("/gateway/lanes", token)
        spend = _http("/gateway/spend", token)
        last = {"active": len(lanes.get("active") or []),
                "inflight": int(lanes.get("inflight") or 0),
                "waiting": int(lanes.get("waiting") or 0),
                "paid_in_flight": int(spend.get("in_flight") or 0),
                "established": _established()}
        if all(value == 0 for value in last.values()):
            consecutive += 1
            if consecutive >= 3:
                return last
        else:
            consecutive = 0
        time.sleep(2)
    raise TimeoutError(f"legacy gateway did not drain; preserved accepted calls: {last}")


def _healthy(token: str, digest: str) -> bool:
    try:
        spend = _http("/gateway/spend", token)
        with urllib.request.urlopen(BASE + "/health", timeout=5) as response:
            health = response.status == 200
        return bool(health and spend.get("gateway_sha256") == digest
                    and spend.get("enforce") and spend.get("durable")
                    and float(spend.get("cap") or 0) == 25.0
                    and float(spend.get("attribution_gap_usd") or 0) == 0)
    except (OSError, ValueError, urllib.error.URLError):
        return False


def _wait_healthy(token: str, digest: str, timeout_s: float = 45) -> None:
    until = time.monotonic() + timeout_s
    while time.monotonic() < until:
        if _healthy(token, digest):
            return
        time.sleep(1)
    raise RuntimeError(f"gateway failed health/spend/readback for {digest}")


def publish(timeout_s: float = WAIT_MAX_S) -> dict:
    if os.geteuid() != 0:
        raise PermissionError("run the one-time bootstrap as root")
    if not 1 <= timeout_s <= WAIT_MAX_S:
        raise ValueError("timeout must be within the finite kernel fence")
    source = SOURCE.read_bytes()
    committed = subprocess.check_output(
        ["git", "-c", f"safe.directory={REPO}", "show", "HEAD:deploy/bin/keepalive-shim.py"],
        cwd=REPO)
    if source != committed:
        raise RuntimeError("candidate gateway differs from committed HEAD")
    old = RUNTIME.read_bytes()
    if old == source:
        return {"status": "current", "sha256": _sha(source)}
    token = TOKEN_FILE.read_text().strip()
    if not _healthy(token, _sha(old)):
        raise RuntimeError("old gateway spend authority is not healthy")
    try:
        _http("/gateway/drain", token)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    else:
        raise RuntimeError("live gateway already has a drain endpoint; use gateway_safe_publish.py")
    listeners = _run("ss", "-Hltpn", "sport", "=", ":8000")
    if "[::]" in listeners:
        raise RuntimeError("IPv6 listener is outside this IPv4 bootstrap fence")
    subprocess.run(["install", "-D", "-m", "0644", str(DROPIN), str(DROPIN_LIVE)], check=True)
    subprocess.run(["systemctl", "daemon-reload"], check=True)
    stop = _unit_seconds(_run("systemctl", "show", "-p", "TimeoutStopUSec", "--value", SERVICE))
    if stop < 1830:
        raise RuntimeError("gateway stop grace is too short")
    expiry = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=FENCE_MAX_S))
    stamp = expiry.strftime("%Y-%m-%dT%H:%M:%S")
    installed = []
    replaced = False
    old_stat = RUNTIME.stat()
    backup = RUNTIME.with_name(f"{RUNTIME.name}.bootstrap-{int(time.time())}")
    _atomic_write(backup, old)
    os.chown(backup, old_stat.st_uid, old_stat.st_gid)
    try:
        for chain in ("INPUT", "OUTPUT"):
            rule = _rule(chain, stamp)
            _iptables("-I", chain, "1", *rule)
            installed.append((chain, rule))
        _wait_drained(token, timeout_s)
        _atomic_write(RUNTIME, source)
        replaced = True
        os.chown(RUNTIME, old_stat.st_uid, old_stat.st_gid)
        subprocess.run(["systemctl", "restart", SERVICE], check=True)
        _wait_healthy(token, _sha(source))
        return {"status": "published", "sha256": _sha(source),
                "backup": str(backup), "fenced_new_connections": True}
    except Exception:
        if replaced:
            # External callers are still fenced, and only root health requests
            # reached the candidate. Roll back before reopening admissions.
            _atomic_write(RUNTIME, old)
            os.chown(RUNTIME, old_stat.st_uid, old_stat.st_gid)
            subprocess.run(["systemctl", "restart", SERVICE], check=True)
            _wait_healthy(token, _sha(old))
        raise
    finally:
        cleanup_errors = []
        for chain, rule in reversed(installed):
            try:
                _iptables("-D", chain, *rule)
            except Exception as exc:
                cleanup_errors.append(f"{chain}: {exc}")
        if cleanup_errors:
            raise RuntimeError("kernel admission fence cleanup failed; finite expiry remains: "
                               + "; ".join(cleanup_errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=WAIT_MAX_S)
    args = parser.parse_args()
    if not args.apply:
        print(json.dumps({"status": "preview", "source_sha256": _sha(SOURCE.read_bytes()),
                          "live_sha256": _sha(RUNTIME.read_bytes())}, sort_keys=True))
        return
    with LOCK.open("a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(publish(args.timeout_s), sort_keys=True))


if __name__ == "__main__":
    main()
