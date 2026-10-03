#!/usr/bin/env python3
"""Planned local-offline window for the gateway (lane CF, 2026-10-02).

Benchmarks, engine upgrades and restarts must not be outages: while a window is open the gateway routes new work
to the remote valve (inside the daily spend cap), keeps pinned-local callers waiting with Retry-After, lets
accepted local work finish, and returns to local-first by itself when the window closes or its lease expires.
Use this INSTEAD of a /gateway/drain lease; the drain fence stays for gateway code swaps only.

  gateway-offline.py status
  gateway-offline.py open  --reason "EF2 arm D1" --by EF2 [--ttl 1800] [--wait-s 120]   # prints the lease
  gateway-offline.py close [--lease LEASE]      # no --lease: closes the lease THIS host recorded in offline-lease.json
  gateway-offline.py run   --reason ... --by ... [--ttl 1800] [--wait-s 120] -- CMD ARGS   # open, run, always close
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

GATEWAY = os.environ.get("GATEWAY", "http://127.0.0.1:8000")
TOKEN_FILE = os.path.expanduser("~/.local/share/vllm-qwen27b/admin.token")
# S3 2026-10-02: the lease id lives only in the shim's memory, so a wrapper killed before its `finally` used to strand the window
# until the TTL (the 21:18 S3 incident). open/run now persist {lease, by, reason, until, pid} (mode 600) and close/run read it back.
LEASE_FILE = os.environ.get("OFFLINE_LEASE_FILE", os.path.expanduser("~/.local/share/vllm-qwen27b/offline-lease.json"))


def proc_start(pid):
    """starttime field of /proc/<pid>/stat (None when the process is gone or /proc is unreadable)."""
    try:
        with open(f"/proc/{int(pid)}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[19]
    except (OSError, ValueError, IndexError):
        return None


def owner_dead(rec):
    """L77: a `run` wrapper's record is provably orphaned when its pid is gone (or reused by another process).
    `open` records are never presumed dead: the short-lived CLI that wrote them exits by design."""
    if not rec or rec.get("mode") != "run" or not isinstance(rec.get("pid"), int):
        return False
    if not os.path.exists("/proc/self/stat"):
        return False
    now = proc_start(rec["pid"])
    return now is None or (bool(rec.get("pid_start")) and rec["pid_start"] != now)


def save_lease(lease, by, reason, ttl, mode="open"):
    os.makedirs(os.path.dirname(LEASE_FILE), exist_ok=True)
    tmp = LEASE_FILE + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump({"lease": lease, "by": by, "reason": reason, "until": time.time() + ttl, "pid": os.getpid(),
                   "pid_start": proc_start(os.getpid()), "mode": mode}, fh)
    os.replace(tmp, LEASE_FILE)


def load_lease():
    try:
        with open(LEASE_FILE) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def drop_lease(lease=None):
    cur = load_lease()
    if cur and (lease is None or cur.get("lease") == lease):
        try:
            os.unlink(LEASE_FILE)
        except OSError:
            pass


def http(path, method="GET", payload=None):
    tok = ""
    try:
        tok = open(TOKEN_FILE).read().strip()
    except OSError:
        pass
    req = urllib.request.Request(GATEWAY + path, method=method, headers={"X-Admin-Token": tok, "Content-Type": "application/json"},
                                 data=None if payload is None else json.dumps(payload).encode())
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()[:300], "http": e.code}


def open_window(reason, by, ttl, wait_s, mode="open"):
    d = http("/gateway/offline", "POST", {"ttl_s": ttl, "reason": reason, "by": by})
    if not d.get("lease"):
        return d, None
    # L77: persist at once (not after the wait) and close the window if this process is interrupted while waiting, so a
    # TERM/INT during the local-work wait cannot strand it until the TTL.
    try:
        save_lease(d["lease"], by, reason, ttl, mode)
        t0 = time.time()
        while time.time() - t0 < wait_s:                  # accepted local work finishes; nothing new is admitted
            if (http("/gateway/offline").get("local_active") or 0) == 0:
                break
            time.sleep(2)
        return http("/gateway/offline"), d["lease"]
    except BaseException:
        try:
            http("/gateway/offline", "DELETE", {"lease": d["lease"]})
        finally:
            drop_lease(d["lease"])
        raise


def _exit_on_signal(code):
    def handler(*_):
        sys.exit(code)
    return handler


def reap_orphan_run_window():
    """A previous `run` wrapper that died without its finally (SIGKILL/OOM) left a recorded window: close it now."""
    rec = load_lease()
    if owner_dead(rec):
        r = http("/gateway/offline", "DELETE", {"lease": rec.get("lease")})
        if not r.get("error") or r.get("http") == 409:
            drop_lease(rec.get("lease"))
        print(json.dumps({"audit": "orphan-offline-window-released", "dead_pid": rec.get("pid"), "by": rec.get("by"),
                          "reason": rec.get("reason"), "result": r}), file=sys.stderr)


def main():
    saved = {sg: signal.getsignal(sg) for sg in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    try:
        return _main()
    finally:
        for sg, h in saved.items():     # handlers are per-process; give them back (matters for in-process callers/tests)
            if h is not None:
                signal.signal(sg, h)


def _main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["status", "open", "close", "run"])
    ap.add_argument("--reason", default="planned local work")
    ap.add_argument("--by", default=os.environ.get("USER", "?"))
    ap.add_argument("--ttl", type=int, default=1800)
    ap.add_argument("--wait-s", type=int, default=120)
    ap.add_argument("--lease")
    ap.add_argument("rest", nargs="*")
    # `run ... -- CMD ARGS` (L54): argparse matches the `nargs="*"` positional as EMPTY as soon as the
    # command word is consumed (before the interleaved --reason/--by options), so the trailing
    # "-- CMD" was then rejected as "unrecognized arguments" on every Python we run (3.11 venv and
    # 3.14 system alike). Split at the first bare "--" ourselves; CMD's own flags (`-- ls -l`) then
    # never reach argparse either.
    argv = sys.argv[1:]
    tail = []
    if "--" in argv:
        i = argv.index("--")
        argv, tail = argv[:i], argv[i + 1:]
    a = ap.parse_args(argv)
    a.rest = list(a.rest) + tail
    if a.cmd == "status":
        print(json.dumps(http("/gateway/offline"), indent=1)); return 0
    if a.cmd == "close":
        lease = a.lease or (load_lease() or {}).get("lease")
        if not lease:
            print(json.dumps({"error": "no --lease given and no recorded lease in " + LEASE_FILE})); return 2
        r = http("/gateway/offline", "DELETE", {"lease": lease})
        if not r.get("error") or r.get("http") == 409:          # closed, or already gone/expired/replaced: the record is stale
            drop_lease(lease)
        print(json.dumps(r)); return 0
    for sg, code in ((signal.SIGTERM, 143), (signal.SIGHUP, 129), (signal.SIGINT, 130)):
        signal.signal(sg, _exit_on_signal(code))               # a TERM'd wrapper must still close its window, even while opening
    reap_orphan_run_window()
    st, lease = open_window(a.reason, a.by, a.ttl, a.wait_s, "run" if a.cmd == "run" else "open")
    if not lease:
        print(json.dumps({"opened": False, **st}), file=sys.stderr); return 2
    if a.cmd == "open":
        print(json.dumps({"opened": True, "lease": lease, **st})); return 0
    cmd = list(a.rest)
    if not cmd:
        print(json.dumps({"error": "run needs a command after --"}), file=sys.stderr)
        http("/gateway/offline", "DELETE", {"lease": lease}); drop_lease(lease); return 2
    try:
        return subprocess.call(cmd)
    finally:
        http("/gateway/offline", "DELETE", {"lease": lease})
        drop_lease(lease)


if __name__ == "__main__":
    sys.exit(main())
