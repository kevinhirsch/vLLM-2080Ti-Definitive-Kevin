#!/usr/bin/env python3
"""gpuguard.py -- GPU facts and the cooperative "GPU busy" signal for engine windows (lane RL, 2026-10-03).

Why: an engine boot profiles FREE GPU memory, so any foreign CUDA process alive at boot (a lane bench beside the engine)
shrinks the KV pool for the whole life of that engine generation (2026-10-03: 966K -> 922K -> 900K tokens over three
boots). So:
  * every engine boot is gated on "no non-engine compute app on either GPU" (foreign_apps(); each one is named with its
    owning systemd unit via unitrun.owner_of_pid());
  * the KV pool of every boot is read from the engine journal (kv_pool_since());
  * while a window owns the GPUs it publishes BUSY (~/.local/share/vllm-qwen27b/gpu-busy.json). Lane benches check it
    before every GPU launch (~/projects/lanes/windows/gpuok.sh, or `gpuguard.py check`) and stay off the cards.

CLI:
  gpuguard.py apps                     # compute apps on both GPUs, engine vs foreign, with owner unit
  gpuguard.py check [--window ID]      # exit 0 = GPUs free for a bench, 1 = busy (prints why). --window = I am that window
  gpuguard.py busy                     # print the busy record (or {"busy": false})
  gpuguard.py set --by X --reason R [--ttl S] [--window ID] [--phase P]   # manual busy (pid-scoped to the caller's parent)
  gpuguard.py clear [--window ID]
  gpuguard.py kv-pool [--since EPOCH]  # last 'GPU KV cache size' of the engine journal
  gpuguard.py lease acquire --owner O --gpu N --mib M [--unit U] [--ttl S]   # L138: declare a non-engine CUDA process BEFORE it starts
  gpuguard.py lease release --id ID | list
  gpuguard.py audit [--brief]          # every foreign compute app vs the lease registry; exit 1 when any is unleased or over its lease

L138 -- the GPU lease. A non-engine CUDA process on the production cards is allowed only under a LEASE: owner, GPU, a MiB budget
and a TTL, recorded in gpu-leases.json (flock-serialised). gpucap.py takes one for every lane job it runs. audit() classifies every
foreign compute app as leased-ok | leased-over (holds more than its budget + slack) | unleased; the engine boot gate (gpu-boot-gate.sh)
prints that classification and waits longer for the apps that are NOT under a lease than for the ones that are. Leases that are expired,
or whose unit/pid is gone, are void.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unitrun  # noqa: E402

HOME = os.path.expanduser("~")
BASE = os.environ.get("GPUGUARD_BASE", f"{HOME}/.local/share/vllm-qwen27b")
BUSY_FILE = os.environ.get("GPU_BUSY_FILE", f"{BASE}/gpu-busy.json")
LEASES_FILE = os.environ.get("GPU_LEASES_FILE", f"{BASE}/gpu-leases.json")
LEASE_UNIT_GRACE_S = float(os.environ.get("GPU_LEASE_UNIT_GRACE_S", "120"))
LEASE_SLACK_MIB = int(os.environ.get("GPU_LEASE_SLACK_MIB", "400"))      # CUDA context + allocator slack beyond the declared budget
ENGINE_UNIT = os.environ.get("ENGINE_UNIT", "vllm-qwen27b.service")
ENGINE_CGROUP = os.environ.get("ENGINE_CGROUP", f"/system.slice/{ENGINE_UNIT}")
NVIDIA_SMI = os.environ.get("NVIDIA_SMI", "nvidia-smi")
JOURNALCTL = os.environ.get("JOURNALCTL", "journalctl")
KV_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")


def _sh(argv, timeout=20):
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def gpu_index_by_uuid() -> dict:
    out = {}
    for line in _sh([NVIDIA_SMI, "--query-gpu=index,uuid", "--format=csv,noheader"]).splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) == 2:
            out[f[1]] = int(f[0]) if f[0].isdigit() else f[0]
    return out


def compute_apps() -> list[dict]:
    """[{pid, name, used_mib, gpu}] from nvidia-smi --query-compute-apps (empty list when nvidia-smi is unreadable)."""
    idx = gpu_index_by_uuid()
    apps = []
    txt = _sh([NVIDIA_SMI, "--query-compute-apps=pid,process_name,used_memory,gpu_uuid", "--format=csv,noheader,nounits"])
    for line in txt.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 4 or not f[0].isdigit():
            continue
        try:
            used = int(float(f[2]))
        except ValueError:
            used = None
        apps.append({"pid": int(f[0]), "name": f[1], "used_mib": used, "gpu": idx.get(f[3], f[3])})
    return apps


def proc_cmdline(pid: int, limit: int = 160) -> str | None:
    try:
        with open(f"{unitrun.PROC}/{int(pid)}/cmdline", "rb") as fh:
            return fh.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()[:limit]
    except (OSError, ValueError):
        return None


def engine_pids() -> set[int]:
    return unitrun.cgroup_pids(ENGINE_CGROUP)


def foreign_apps(apps: list[dict] | None = None, eng: set[int] | None = None) -> list[dict]:
    """Compute apps that are NOT in the engine unit's cgroup, each with its owner (unit/lane) when known."""
    apps = compute_apps() if apps is None else apps
    eng = engine_pids() if eng is None else eng
    out = []
    for a in apps:
        if a["pid"] in eng:
            continue
        cg = unitrun.proc_cgroup(a["pid"])
        if cg and unitrun.unit_of_cgroup(cg) == ENGINE_UNIT:
            continue
        own = unitrun.owner_of_pid(a["pid"], time.time()) or {}
        out.append({**a, "unit": own.get("unit"), "lane": own.get("lane"), "how": own.get("how"),
                    "comm": own.get("comm") or unitrun.proc_comm(a["pid"]), "cmd": proc_cmdline(a["pid"])})
    return out


def describe(apps: list[dict]) -> str:
    return "; ".join(f"gpu{a.get('gpu')} pid {a['pid']} {a.get('comm') or a.get('name')} {a.get('used_mib')} MiB "
                     f"unit={a.get('unit') or '?'} cmd={a.get('cmd') or '?'}" for a in apps)


def wait_no_foreign(max_wait_s: float = 0, poll_s: float = 5.0, clock=time.time, sleep=time.sleep) -> tuple[bool, list]:
    """True when no foreign compute app is on either GPU; waits up to max_wait_s for them to go away."""
    t0 = clock()
    while True:
        apps = foreign_apps()
        if not apps:
            return True, []
        if clock() - t0 >= max_wait_s:
            return False, apps
        sleep(poll_s)


# ---------------------------------------------------------------- busy signal

def _alive(pid, pid_start) -> bool:
    if not isinstance(pid, int):
        return False
    now = unitrun.proc_start(pid)
    if now is None:
        return False
    return not pid_start or pid_start == now


def busy_state(path: str | None = None, now: float | None = None) -> dict:
    """The live busy record, or {"busy": False, ...} when absent, expired or its owner process is gone."""
    path = path or BUSY_FILE
    now = time.time() if now is None else now
    try:
        with open(path) as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        return {"busy": False}
    if not rec.get("busy"):
        return {"busy": False}
    if rec.get("until") and now > float(rec["until"]):
        return {"busy": False, "stale": "expired", "was": rec}
    if not _alive(rec.get("pid"), rec.get("pid_start")):
        return {"busy": False, "stale": "owner-dead", "was": rec}
    return rec


def set_busy(by: str, reason: str, ttl_s: float, window: str | None = None, phase: str = "window",
             pid: int | None = None, path: str | None = None) -> dict:
    path = path or BUSY_FILE
    pid = os.getpid() if pid is None else pid
    rec = {"busy": True, "by": by, "reason": reason, "window": window, "phase": phase, "since": round(time.time(), 1),
           "until": round(time.time() + ttl_s, 1), "pid": pid, "pid_start": unitrun.proc_start(pid),
           "how_to_check": "~/projects/lanes/windows/gpuok.sh (exit 1 = stay off the GPUs)"}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(tmp, path)
    return rec


def clear_busy(window: str | None = None, path: str | None = None) -> bool:
    path = path or BUSY_FILE
    try:
        with open(path) as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        return False
    if window and rec.get("window") not in (None, window):
        return False            # someone else's window: never clear it
    try:
        os.unlink(path)
    except OSError:
        return False
    return True


def check(window: str | None = None) -> tuple[bool, str]:
    st = busy_state()
    if st.get("busy"):
        if window and st.get("window") == window:
            return True, f"busy by my own window {window}"
        return False, f"GPU busy: {st.get('by')} {st.get('phase')} ({st.get('reason')}) until {time.strftime('%H:%M:%S', time.localtime(st.get('until') or 0))}"
    return True, "free"


# ---------------------------------------------------------------- L138: GPU leases

def _leases_locked(path, fn):
    import fcntl
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".lock", "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            with open(path) as fh:
                rows = json.load(fh)
            rows = rows if isinstance(rows, list) else []
        except (OSError, ValueError):
            rows = []
        rows, out = fn(rows)
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(rows, fh, indent=1)
        os.replace(tmp, path)
        return out


def _lease_alive(row, now) -> bool:
    if now > float(row.get("until") or 0):
        return False
    unit = row.get("unit")
    if unit and unit != "-" and now - float(row.get("since") or now) > LEASE_UNIT_GRACE_S:
        # a lane job's unit may not exist yet (the lease precedes the launch): after the grace, a unit that is not running voids its lease
        if not unitrun.is_active(unit):
            return False
    pid = row.get("pid")
    if isinstance(pid, int) and not _alive(pid, row.get("pid_start")):
        return False
    return True


def lease_acquire(owner: str, gpu, mib: int, unit: str | None = None, ttl_s: float = 3600, pid: int | None = None,
                  path: str | None = None) -> dict:
    """Declare a non-engine CUDA process: owner, GPU index (or 'all'), MiB budget, the systemd unit it will run in (gpucap/unitrun)
    and/or a pid, and a TTL. Returns the lease row (its `id` releases it)."""
    path = path or LEASES_FILE
    now = time.time()
    row = {"id": f"gl-{int(now)}-{os.getpid()}-{abs(hash((owner, unit, mib))) % 10000}", "owner": owner, "gpu": gpu, "mib": int(mib),
           "unit": unit or "-", "pid": pid, "pid_start": unitrun.proc_start(pid) if isinstance(pid, int) else None,
           "since": round(now, 1), "until": round(now + ttl_s, 1)}

    def fn(rows):
        rows = [r for r in rows if _lease_alive(r, now)]
        rows.append(row)
        return rows, row
    return _leases_locked(path, fn)


def lease_release(lease_id: str, path: str | None = None) -> bool:
    path = path or LEASES_FILE

    def fn(rows):
        keep = [r for r in rows if r.get("id") != lease_id]
        return keep, len(keep) != len(rows)
    return _leases_locked(path, fn)


def leases(path: str | None = None, now: float | None = None) -> list[dict]:
    path = path or LEASES_FILE
    now = time.time() if now is None else now
    try:
        with open(path) as fh:
            rows = json.load(fh)
    except (OSError, ValueError):
        return []
    out = []
    for r in rows if isinstance(rows, list) else []:
        if _lease_alive(r, now):
            out.append(r)
    return out


def _lease_for(app: dict, rows: list[dict]):
    for r in rows:
        if r.get("gpu") not in ("all", None, app.get("gpu")) and str(r.get("gpu")) != str(app.get("gpu")):
            continue
        if app.get("unit") and r.get("unit") == app["unit"]:
            return r
        if isinstance(r.get("pid"), int) and r["pid"] == app["pid"]:
            return r
    return None


def audit(apps: list[dict] | None = None, rows: list[dict] | None = None, eng: set[int] | None = None) -> dict:
    """Every foreign compute app against the lease registry. The budget of one lease is shared by all its apps on that GPU."""
    foreign = foreign_apps(apps, eng)
    rows = leases() if rows is None else rows
    used_by_lease: dict = {}
    for a in foreign:
        lz = _lease_for(a, rows)
        if lz:
            used_by_lease[(lz["id"], a.get("gpu"))] = used_by_lease.get((lz["id"], a.get("gpu")), 0) + int(a.get("used_mib") or 0)
    out = []
    for a in foreign:
        lz = _lease_for(a, rows)
        if lz is None:
            st = "unleased"
        elif used_by_lease[(lz["id"], a.get("gpu"))] > int(lz["mib"]) + LEASE_SLACK_MIB:
            st = "leased-over"
        else:
            st = "leased-ok"
        out.append({**a, "state": st, "lease": (lz or {}).get("id"), "lease_owner": (lz or {}).get("owner"),
                    "lease_mib": (lz or {}).get("mib")})
    bad = [a for a in out if a["state"] != "leased-ok"]
    return {"apps": out, "ok": not bad, "unleased": [a for a in out if a["state"] == "unleased"],
            "over": [a for a in out if a["state"] == "leased-over"]}


# ---------------------------------------------------------------- KV pool

def kv_pools(text: str) -> list[int]:
    return [int(m.replace(",", "")) for m in KV_RE.findall(text or "")]


def kv_pool_since(since_epoch: float | None = None, lines: int = 20000) -> int | None:
    """The last 'GPU KV cache size: N tokens' the engine logged (since `since_epoch` when given)."""
    argv = [JOURNALCTL, "-u", ENGINE_UNIT, "--no-pager", "-o", "cat"]
    argv += ["--since", f"@{int(since_epoch)}"] if since_epoch else ["-n", str(lines)]
    vals = kv_pools(_sh(argv, timeout=60))
    return vals[-1] if vals else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("apps")
    p = sp.add_parser("check")
    p.add_argument("--window")
    sp.add_parser("busy")
    p = sp.add_parser("set")
    p.add_argument("--by", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--ttl", type=float, default=1800)
    p.add_argument("--window")
    p.add_argument("--phase", default="manual")
    p = sp.add_parser("clear")
    p.add_argument("--window")
    p = sp.add_parser("kv-pool")
    p.add_argument("--since", type=float)
    p = sp.add_parser("audit")
    p.add_argument("--brief", action="store_true")
    p = sp.add_parser("lease")
    ls = p.add_subparsers(dest="lcmd", required=True)
    q = ls.add_parser("acquire")
    q.add_argument("--owner", required=True)
    q.add_argument("--gpu", default="all")
    q.add_argument("--mib", type=int, required=True)
    q.add_argument("--unit")
    q.add_argument("--pid", type=int)
    q.add_argument("--ttl", type=float, default=3600)
    q = ls.add_parser("release")
    q.add_argument("--id", required=True)
    ls.add_parser("list")
    a = ap.parse_args(argv)
    if a.cmd == "apps":
        apps = compute_apps()
        eng = engine_pids()
        print(json.dumps({"engine": [x for x in apps if x["pid"] in eng], "foreign": foreign_apps(apps, eng)}, indent=1))
        return 0
    if a.cmd == "check":
        ok, why = check(a.window)
        print(why)
        return 0 if ok else 1
    if a.cmd == "busy":
        print(json.dumps(busy_state(), indent=1))
        return 0
    if a.cmd == "set":
        # a CLI call exits at once: scope the record to the calling shell so it goes stale when that shell ends
        print(json.dumps(set_busy(a.by, a.reason, a.ttl, a.window, a.phase, pid=os.getppid())))
        return 0
    if a.cmd == "clear":
        return 0 if clear_busy(a.window) else 1
    if a.cmd == "kv-pool":
        print(json.dumps({"kv_pool_tokens": kv_pool_since(a.since)}))
        return 0
    if a.cmd == "audit":
        res = audit()
        if a.brief:
            for x in res["apps"]:
                print(f"{x['state']:11s} gpu{x.get('gpu')} pid {x['pid']} {x.get('comm') or x.get('name')} {x.get('used_mib')} MiB "
                      f"unit={x.get('unit') or '?'} lease={x.get('lease') or '-'}({x.get('lease_owner') or '-'}, {x.get('lease_mib') or '-'} MiB)")
        else:
            print(json.dumps(res, indent=1))
        return 0 if res["ok"] else 1
    if a.cmd == "lease":
        if a.lcmd == "acquire":
            gpu = a.gpu if a.gpu == "all" else int(a.gpu)
            print(json.dumps(lease_acquire(a.owner, gpu, a.mib, a.unit, a.ttl, a.pid)))
            return 0
        if a.lcmd == "release":
            return 0 if lease_release(a.id) else 1
        print(json.dumps(leases(), indent=1))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
