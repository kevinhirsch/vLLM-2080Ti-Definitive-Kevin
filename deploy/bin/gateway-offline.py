#!/usr/bin/env python3
"""Planned local-offline window for the gateway (lane CF, 2026-10-02).

Benchmarks, engine upgrades and restarts must not be outages: while a window is open the gateway routes new work
to the remote valve (inside the daily spend cap), keeps pinned-local callers waiting with Retry-After, lets
accepted local work finish, and returns to local-first by itself when the window closes or its lease expires.
Use this INSTEAD of a /gateway/drain lease; the drain fence stays for gateway code swaps only.

  gateway-offline.py status
  gateway-offline.py open  --reason "EF2 arm D1" --by EF2 [--ttl 1800] [--wait-s 120]   # prints the lease
  gateway-offline.py close --lease LEASE
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
    a = ap.parse_args()
    if a.cmd == "status":
        print(json.dumps(http("/gateway/offline"), indent=1)); return 0
    if a.cmd == "close":
        print(json.dumps(http("/gateway/offline", "DELETE", {"lease": a.lease}))); return 0
    st, lease = open_window(a.reason, a.by, a.ttl, a.wait_s)
    if not lease:
        print(json.dumps({"opened": False, **st}), file=sys.stderr); return 2
    if a.cmd == "open":
        print(json.dumps({"opened": True, "lease": lease, **st})); return 0
    cmd = [x for x in a.rest if x != "--"]
    try:
        return subprocess.call(cmd)
    finally:
        http("/gateway/offline", "DELETE", {"lease": lease})


if __name__ == "__main__":
    sys.exit(main())
