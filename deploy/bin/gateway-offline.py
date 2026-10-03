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
import subprocess
import sys
import time
import urllib.request

GATEWAY = os.environ.get("GATEWAY", "http://127.0.0.1:8000")
TOKEN_FILE = os.path.expanduser("~/.local/share/vllm-qwen27b/admin.token")
# S3 2026-10-02: the lease id lives only in the shim's memory, so a wrapper killed before its `finally` used to strand the window
# until the TTL (the 21:18 S3 incident). open/run now persist {lease, by, reason, until, pid} (mode 600) and close/run read it back.
LEASE_FILE = os.environ.get("OFFLINE_LEASE_FILE", os.path.expanduser("~/.local/share/vllm-qwen27b/offline-lease.json"))


def save_lease(lease, by, reason, ttl):
    os.makedirs(os.path.dirname(LEASE_FILE), exist_ok=True)
    tmp = LEASE_FILE + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump({"lease": lease, "by": by, "reason": reason, "until": time.time() + ttl, "pid": os.getpid()}, fh)
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


def open_window(reason, by, ttl, wait_s):
    d = http("/gateway/offline", "POST", {"ttl_s": ttl, "reason": reason, "by": by})
    if not d.get("lease"):
        return d, None
    t0 = time.time()
    while time.time() - t0 < wait_s:                      # accepted local work finishes; nothing new is admitted
        if (http("/gateway/offline").get("local_active") or 0) == 0:
            break
        time.sleep(2)
    return http("/gateway/offline"), d["lease"]


def main():
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
    st, lease = open_window(a.reason, a.by, a.ttl, a.wait_s)
    if not lease:
        print(json.dumps({"opened": False, **st}), file=sys.stderr); return 2
    save_lease(lease, a.by, a.reason, a.ttl)
    if a.cmd == "open":
        print(json.dumps({"opened": True, "lease": lease, **st})); return 0
    cmd = list(a.rest)
    if not cmd:
        print(json.dumps({"error": "run needs a command after --"}), file=sys.stderr)
        http("/gateway/offline", "DELETE", {"lease": lease}); drop_lease(lease); return 2
    import signal
    for sg in (signal.SIGTERM, signal.SIGHUP):                    # a TERM'd wrapper must still close its window
        signal.signal(sg, lambda *_: sys.exit(143))
    try:
        return subprocess.call(cmd)
    finally:
        http("/gateway/offline", "DELETE", {"lease": lease})
        drop_lease(lease)


if __name__ == "__main__":
    sys.exit(main())
