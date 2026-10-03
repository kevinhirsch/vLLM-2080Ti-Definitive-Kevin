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
    return 2


if __name__ == "__main__":
    sys.exit(main())
