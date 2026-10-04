#!/usr/bin/env python3
"""unitrun.py -- run lane jobs and window commands in their own systemd user units (lane RL, 2026-10-03, lead L106).

Every lane job runs as `systemd-run --user --unit=<lane>-<job> --collect`, so it is killed and attributed by its unit /
cgroup instead of by `pkill -f <pattern>` (which matches the killer's own command line, other lanes' look-alike jobs and
nothing at all once the pattern drifts). What this adds on top of systemd-run:

  * a stable, sanitized unit name per (lane, job); a second start of a still-running unit is refused, not doubled;
  * the exit status is recorded by the unit's own ExecStopPost (`<state>/<unit>.status`), so it survives the caller
    being killed and the unit being garbage-collected (--collect);
  * a pid registry: the pids in the unit's cgroup are sampled while it runs (`<state>/<unit>.json` + history.jsonl),
    so a kernel Xid that names a pid which is already gone can still be attributed to the unit that owned it
    (engine-fault-collector.py uses owner_of_pid());
  * kill = `systemctl --user stop/kill <unit>` (the whole cgroup, never a pattern).

CLI:
  unitrun.py run --lane k5 --job bench [--timeout 600] [--out FILE] [--cwd DIR] [--env K=V ...] [--no-wait] -- CMD ARGS
  unitrun.py stop UNIT | --lane LANE          # stop one unit / every running unit of a lane
  unitrun.py kill UNIT [--signal KILL]
  unitrun.py list [--lane LANE]
  unitrun.py owner PID [--at EPOCH]           # which unit owns (or owned) a pid
  unitrun.py sample UNIT                       # internal: pid sampler for --no-wait units (runs in <unit>--sampler)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

HOME = os.path.expanduser("~")
STATE_DIR = os.environ.get("UNITRUN_STATE_DIR", f"{HOME}/.local/share/lane-units")
SYSTEMD_RUN = os.environ.get("UNITRUN_SYSTEMD_RUN", "systemd-run")
SYSTEMCTL = os.environ.get("UNITRUN_SYSTEMCTL", "systemctl")
CGROUP_ROOT = os.environ.get("UNITRUN_CGROUP_ROOT", "/sys/fs/cgroup")
PROC = os.environ.get("UNITRUN_PROC", "/proc")
# environment a lane command gets by default (the user manager's own env is minimal and differs from a login shell)
PASS_ENV = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TERM", "TZ")
HISTORY_MAX_BYTES = 8 << 20

_NAME_OK = re.compile(r"[^A-Za-z0-9_.:-]+")


def unit_name(lane: str, job: str) -> str:
    """`<lane>-<job>`, sanitized to what systemd accepts; '.service' is implied (systemd-run adds it)."""
    lane = _NAME_OK.sub("-", str(lane).strip()).strip("-.") or "lane"
    job = _NAME_OK.sub("-", str(job).strip()).strip("-.") or "job"
    name = f"{lane}-{job}"
    if len(name) > 200:
        name = name[:200]
    return name


def _svc(unit: str) -> str:
    return unit if unit.endswith((".service", ".scope", ".timer")) else unit + ".service"


def _ctl(*args, timeout=30):
    return subprocess.run([SYSTEMCTL, "--user", *args], capture_output=True, text=True, timeout=timeout)


def show(unit: str, *props) -> dict:
    r = _ctl("show", _svc(unit), *[f"--property={p}" for p in props])
    out = {}
    for line in (r.stdout or "").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def is_active(unit: str) -> bool:
    st = show(unit, "ActiveState").get("ActiveState", "")
    return st in ("active", "activating", "deactivating", "reloading")


def unit_of_cgroup(text: str) -> str | None:
    """The innermost *.service / *.scope of a /proc/<pid>/cgroup line (cgroup v2 '0::/path')."""
    for line in (text or "").splitlines():
        parts = line.split(":", 2)
        path = parts[2] if len(parts) == 3 else line
        units = [seg for seg in path.split("/") if seg.endswith((".service", ".scope"))]
        # user@1000.service is the user MANAGER, not a job; prefer the innermost unit below it
        units = [u for u in units if not re.fullmatch(r"user@\d+\.service", u)] or units
        if units:
            return units[-1]
    return None


def proc_cgroup(pid: int) -> str | None:
    try:
        with open(f"{PROC}/{int(pid)}/cgroup") as fh:
            return fh.read()
    except (OSError, ValueError):
        return None


def proc_comm(pid: int) -> str | None:
    try:
        with open(f"{PROC}/{int(pid)}/comm") as fh:
            return fh.read().strip()
    except (OSError, ValueError):
        return None


def proc_start(pid: int) -> str | None:
    try:
        with open(f"{PROC}/{int(pid)}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[19]
    except (OSError, ValueError, IndexError):
        return None


def cgroup_pids(cgroup: str) -> set[int]:
    """Every pid in a cgroup subtree (cgroup.procs of the group and all its children)."""
    base = CGROUP_ROOT + cgroup if cgroup.startswith("/") else os.path.join(CGROUP_ROOT, cgroup)
    pids: set[int] = set()
    for root, _dirs, files in os.walk(base):
        if "cgroup.procs" in files:
            try:
                with open(os.path.join(root, "cgroup.procs")) as fh:
                    pids.update(int(x) for x in fh.read().split() if x.strip().isdigit())
            except OSError:
                pass
    return pids


# ---------------------------------------------------------------- registry

def _state_path(unit: str, ext: str) -> str:
    return os.path.join(STATE_DIR, f"{unit.removesuffix('.service')}.{ext}")


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def load_record(unit: str) -> dict | None:
    try:
        with open(_state_path(unit, "json")) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def read_status(unit: str) -> dict | None:
    """{"result": "success|exit-code|timeout|signal|...", "code": "exited|killed", "status": "N|TERM"} from ExecStopPost."""
    try:
        with open(_state_path(unit, "status")) as fh:
            parts = fh.read().split()
    except OSError:
        return None
    if not parts:
        return None
    return {"result": parts[0], "code": parts[1] if len(parts) > 1 else "", "status": parts[2] if len(parts) > 2 else ""}


def _append_history(rec: dict) -> None:
    path = os.path.join(STATE_DIR, "history.jsonl")
    try:
        if os.path.exists(path) and os.path.getsize(path) > HISTORY_MAX_BYTES:
            os.replace(path, path + ".1")
        with open(path, "a") as fh:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
    except OSError:
        pass


class Sampler:
    """Polls the unit's cgroup and records every pid seen (pid, comm, first/last seen)."""

    def __init__(self, unit: str, rec: dict, every_s: float = 2.0):
        self.unit, self.rec, self.every_s = unit, rec, every_s
        self.cgroup = None
        self.stop = threading.Event()
        self.rec.setdefault("pids", {})

    def tick(self) -> None:
        if not self.cgroup:
            cg = show(self.unit, "ControlGroup").get("ControlGroup") or ""
            if cg:
                self.cgroup = cg
                self.rec["cgroup"] = cg
        if not self.cgroup:
            return
        now = round(time.time(), 1)
        changed = False
        for pid in cgroup_pids(self.cgroup):
            k = str(pid)
            row = self.rec["pids"].get(k)
            if row is None:
                self.rec["pids"][k] = {"comm": proc_comm(pid), "first": now, "last": now, "start": proc_start(pid)}
                changed = True
            else:
                row["last"] = now
        if changed:
            _write_json(_state_path(self.unit, "json"), self.rec)

    def loop(self) -> None:
        while not self.stop.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - attribution is best effort, never fail the job
                pass
            self.stop.wait(self.every_s if self.cgroup else 0.2)


# ---------------------------------------------------------------- run / stop

def build_cmd(unit: str, cmd: list[str], *, timeout_s=None, env=None, cwd=None, out=None, wait=True,
              extra_props=None, pass_env=PASS_ENV) -> list[str]:
    status_file = _state_path(unit, "status")
    if re.search(r"[\s'\"\\$]", status_file):
        raise ValueError(f"unit state path must not contain whitespace/quotes: {status_file}")
    # --expand-environment=no: systemd otherwise rewrites $VAR / $$ in the COMMAND LINE before bash sees it
    # (`bash -c 'echo $$'` ran as `echo $`): every lane's shell snippet would be silently altered
    argv = [SYSTEMD_RUN, "--user", f"--unit={unit}", "--collect", "--service-type=exec", "--expand-environment=no",
            "-p", "KillMode=control-group", "-p", "TimeoutStopSec=20",
            "-p", f"ExecStopPost=/bin/sh -c 'echo $SERVICE_RESULT $EXIT_CODE $EXIT_STATUS > {status_file}'"]
    if wait:
        argv.append("--wait")
    if timeout_s:
        argv += ["-p", f"RuntimeMaxSec={int(timeout_s)}"]
    if cwd:
        argv.append(f"--working-directory={cwd}")
    if out:
        argv += ["-p", f"StandardOutput=append:{out}", "-p", f"StandardError=append:{out}"]
    merged = {k: os.environ[k] for k in pass_env if k in os.environ}
    merged.update({k: str(v) for k, v in (env or {}).items()})
    for k, v in merged.items():
        argv.append(f"--setenv={k}={v}")
    for p in extra_props or ():
        argv += ["-p", p]
    argv.append("--")
    argv += list(cmd)
    return argv


def _parse_result(stderr: str) -> str | None:
    m = re.search(r"Finished with result:\s*(\S+)", stderr or "")
    return m.group(1) if m else None


def run(lane: str, job: str, cmd: list[str], *, timeout_s=None, env=None, cwd=None, out=None, wait=True,
        sample_s: float = 2.0, extra_props=None, meta=None) -> dict:
    """Start `cmd` in unit <lane>-<job>. wait=True blocks and returns {unit, rc, result, duration_s, pids...}."""
    unit = unit_name(lane, job)
    if is_active(unit):
        return {"unit": unit, "rc": None, "result": "refused", "error": f"unit {unit} is already running"}
    _ctl("reset-failed", _svc(unit))            # a stale failed (uncollected) instance would block the name
    os.makedirs(STATE_DIR, exist_ok=True)
    try:
        os.unlink(_state_path(unit, "status"))
    except OSError:
        pass
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    rec = {"unit": unit + ".service", "lane": lane, "job": job, "cmd": list(cmd), "cwd": cwd, "out": out,
           "timeout_s": timeout_s, "started": round(time.time(), 1), "finished": None, "caller_pid": os.getpid(),
           "pids": {}, **(meta or {})}
    _write_json(_state_path(unit, "json"), rec)
    argv = build_cmd(unit, cmd, timeout_s=timeout_s, env=env, cwd=cwd, out=out, wait=wait, extra_props=extra_props)
    t0 = time.time()
    if not wait:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            rec.update(finished=round(time.time(), 1), result="start-failed", error=(r.stderr or "")[-400:])
            _write_json(_state_path(unit, "json"), rec)
            return {"unit": unit, "rc": None, "result": "start-failed", "error": (r.stderr or "")[-400:]}
        # the pid sampler is itself a unit (attributable, and it outlives this caller)
        subprocess.run([SYSTEMD_RUN, "--user", f"--unit={unit}--sampler", "--collect", "--quiet", "--expand-environment=no",
                        f"--setenv=UNITRUN_STATE_DIR={STATE_DIR}", sys.executable, os.path.abspath(__file__), "sample", unit, "--every", str(sample_s)],
                       capture_output=True, text=True, timeout=60)
        return {"unit": unit, "rc": None, "result": "started", "started": rec["started"]}
    sampler = Sampler(unit, rec, sample_s)
    th = threading.Thread(target=sampler.loop, daemon=True)
    th.start()
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        _, err = proc.communicate()
    except BaseException:
        # the caller is being torn down: the job must not outlive it unattended
        with _suppress():
            stop(unit)
        with _suppress():
            proc.wait(timeout=30)
        raise
    finally:
        sampler.stop.set()
        th.join(timeout=5)
    dur = round(time.time() - t0, 1)
    st = read_status(unit) or {}
    result = st.get("result") or _parse_result(err) or ("success" if proc.returncode == 0 else "exit-code")
    rc = proc.returncode
    if st.get("code") == "exited" and st.get("status", "").isdigit():
        rc = int(st["status"])
    rec.update(finished=round(time.time(), 1), rc=rc, result=result, duration_s=dur)
    _write_json(_state_path(unit, "json"), rec)
    _append_history({k: rec[k] for k in ("unit", "lane", "job", "started", "finished", "rc", "result", "pids", "cgroup")
                     if k in rec})
    return {"unit": unit, "rc": rc, "result": result, "duration_s": dur, "timed_out": result == "timeout",
            "pids": sorted(int(p) for p in rec["pids"]), "out": out}


class _suppress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True


def stop(unit: str) -> bool:
    r = _ctl("stop", _svc(unit), timeout=60)
    return r.returncode == 0


def kill(unit: str, signal_name: str = "TERM") -> bool:
    r = _ctl("kill", f"--signal={signal_name}", _svc(unit))
    return r.returncode == 0


def list_units(lane: str | None = None) -> list[dict]:
    pat = f"{unit_name(lane, 'x')[:-2]}-*" if lane else "*"
    r = _ctl("list-units", "--type=service", "--all", "--no-legend", "--plain", "--output=json", pat)
    try:
        rows = json.loads(r.stdout or "[]")
    except ValueError:
        rows = []
        for line in (r.stdout or "").splitlines():
            f = line.split(None, 4)
            if len(f) >= 4:
                rows.append({"unit": f[0], "load": f[1], "active": f[2], "sub": f[3]})
    return [{"unit": x.get("unit"), "active": x.get("active"), "sub": x.get("sub")} for x in rows]


def stop_lane(lane: str, prefix: str | None = None, exclude=()) -> list[str]:
    """Stop every RUNNING unit of a lane (optionally only those starting with <lane>-<prefix>), except `exclude`."""
    want = unit_name(lane, prefix) if prefix else unit_name(lane, "x")[:-2] + "-"
    stopped = []
    for row in list_units(lane):
        u = row.get("unit") or ""
        if u in exclude or u.removesuffix(".service") in exclude:
            continue
        if u.startswith(want) and row.get("active") in ("active", "activating", "deactivating"):
            if stop(u):
                stopped.append(u)
    return stopped


# ---------------------------------------------------------------- attribution

def _records():
    """Every pid record we have: current per-unit files + the history log (newest last)."""
    out = []
    try:
        with open(os.path.join(STATE_DIR, "history.jsonl")) as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    try:
        for f in os.listdir(STATE_DIR):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(STATE_DIR, f)) as fh:
                        out.append(json.load(fh))
                except (OSError, ValueError):
                    pass
    except OSError:
        pass
    return out


def owner_of_pid(pid: int, at: float | None = None, slack_s: float = 120.0) -> dict | None:
    """Which unit owns (or owned, around time `at`) a pid. Live /proc first, then the sampled registry."""
    cg = proc_cgroup(pid)
    if cg:
        u = unit_of_cgroup(cg)
        if u:
            lane = None
            rec = load_record(u)
            if rec:
                lane = rec.get("lane")
            return {"pid": int(pid), "unit": u, "lane": lane, "how": "live-cgroup", "comm": proc_comm(pid)}
    best = None
    for rec in _records():
        row = (rec.get("pids") or {}).get(str(int(pid)))
        if not row:
            continue
        if at is not None:
            lo = float(row.get("first") or rec.get("started") or 0) - slack_s
            hi = float(rec.get("finished") or row.get("last") or time.time()) + slack_s
            if not lo <= at <= hi:
                continue
        cand = {"pid": int(pid), "unit": rec.get("unit"), "lane": rec.get("lane"), "job": rec.get("job"),
                "how": "registry", "comm": row.get("comm"), "seen": [row.get("first"), row.get("last")]}
        if best is None or (row.get("last") or 0) >= (best["seen"][1] or 0):
            best = cand
    return best


# ---------------------------------------------------------------- CLI

def _cmd_sample(unit: str, every: float) -> int:
    rec = load_record(unit) or {"unit": unit, "pids": {}, "started": round(time.time(), 1)}
    s = Sampler(unit, rec, every)
    seen_active = False
    deadline = time.time() + 30
    while True:
        act = is_active(unit)
        seen_active = seen_active or act
        if not act and (seen_active or time.time() > deadline):
            break
        with _suppress():
            s.tick()
        time.sleep(every)
    st = read_status(unit) or {}
    rec.update(finished=round(time.time(), 1), result=st.get("result"),
               rc=int(st["status"]) if st.get("code") == "exited" and str(st.get("status", "")).isdigit() else None)
    _write_json(_state_path(unit, "json"), rec)
    _append_history({k: rec[k] for k in ("unit", "lane", "job", "started", "finished", "rc", "result", "pids", "cgroup")
                     if k in rec})
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    tail = []
    if "--" in argv:
        i = argv.index("--")
        argv, tail = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("run")
    p.add_argument("--lane", required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--timeout", type=int)
    p.add_argument("--out")
    p.add_argument("--cwd")
    p.add_argument("--env", action="append", default=[])
    p.add_argument("--no-wait", action="store_true")
    p = sp.add_parser("stop")
    p.add_argument("unit", nargs="?")
    p.add_argument("--lane")
    p = sp.add_parser("kill")
    p.add_argument("unit")
    p.add_argument("--signal", default="TERM")
    p = sp.add_parser("list")
    p.add_argument("--lane")
    p = sp.add_parser("owner")
    p.add_argument("pid", type=int)
    p.add_argument("--at", type=float)
    p = sp.add_parser("sample")
    p.add_argument("unit")
    p.add_argument("--every", type=float, default=2.0)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        if not tail:
            print(json.dumps({"error": "run needs a command after --"}))
            return 2
        env = dict(kv.split("=", 1) for kv in a.env if "=" in kv)
        # default cwd = the caller's: a user unit otherwise starts in $HOME and relative paths silently resolve there
        res = run(a.lane, a.job, tail, timeout_s=a.timeout, env=env, cwd=a.cwd or os.getcwd(),
                  out=os.path.abspath(a.out) if a.out else None, wait=not a.no_wait)
        print(json.dumps(res))
        if res.get("result") in ("refused", "start-failed"):
            return 2
        return int(res["rc"]) if isinstance(res.get("rc"), int) else 0
    if a.cmd == "stop":
        if a.lane:
            print(json.dumps({"stopped": stop_lane(a.lane)}))
            return 0
        if not a.unit:
            print(json.dumps({"error": "stop needs UNIT or --lane"}))
            return 2
        return 0 if stop(a.unit) else 1
    if a.cmd == "kill":
        return 0 if kill(a.unit, a.signal) else 1
    if a.cmd == "list":
        print(json.dumps(list_units(a.lane), indent=1))
        return 0
    if a.cmd == "owner":
        print(json.dumps(owner_of_pid(a.pid, a.at)))
        return 0
    if a.cmd == "sample":
        return _cmd_sample(a.unit, a.every)
    return 2


if __name__ == "__main__":
    sys.exit(main())
