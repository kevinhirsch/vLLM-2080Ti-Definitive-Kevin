#!/usr/bin/env python3
"""engine-actuator.py -- the typed, allow-listed actions Halo (or Kevin, or a script) may take on the vLLM engine.

Lane EF2 (2026-10-02). Kevin: "full creative control" over local serving; deterministic decision thresholds are an
anti-pattern. So this file holds MECHANICS and FACTS only. It decides nothing: it never restarts on its own, never
counts faults to trigger anything. The deciding is Halo's (via tools/engine_mcp.py -> halo-mcp).

Actions (every one writes an event to the estate event log, source "engine", so Halo's catch-up sees it):
  status                      facts: unit state, health, in-flight, staged/active diagnostic flags, recent faults
  faults [--limit N]          the fault ledger tail + signature counts (facts)
  flags                       the diagnostic flags that exist (the allow-list) and which are staged
  stage-diag --flags a,b      stage diagnostic flags for the NEXT start (any start, planned or unplanned). --clear removes all.
  restart --reason R [...]    planned restart: stage flags (optional) -> DRAIN the gateway fence -> wait for accepted
                              requests to finish (bounded) -> stop -> release the fence -> start. Detached by default.
  announce-start              (ExecStartPost) emit a "engine is back" observation with the flags that are active
  restart-status              state of the last/running planned restart

The allow-list is DIAG below: a fixed map name -> env vars. Nothing else can be set through this tool. Every change is
reversible (stage-diag --clear) and recorded. Flags only take effect at the next engine start.
"""
import argparse, fcntl, json, os, re, subprocess, sys, time, urllib.request
from datetime import datetime

HOME = os.path.expanduser("~")
BASE = f"{HOME}/.local/share/vllm-qwen27b"
INC = f"{BASE}/incidents"
LEDGER = f"{INC}/ledger.jsonl"
DIAG_ENV = f"{BASE}/diag.env"            # EnvironmentFile=- in zz-diag-flags.conf; written ONLY by this tool
PLANNED = f"{BASE}/planned-restart.json"  # marker read by engine-fault-collector.py (so a planned stop is not a FAULT)
JOB = f"{BASE}/restart-job.json"
LOCK = f"{BASE}/restart.lock"
UNIT = "vllm-qwen27b"
GATEWAY = "http://127.0.0.1:8000"
ENGINE = "http://127.0.0.1:8001"
TOKEN = f"{BASE}/admin.token"
ESTATE = f"{HOME}/.local/share/estate-control"

# name -> (env vars, what it does / cost). The ONLY knobs that exist. Each is a fork/CUDA/torch diagnostic switch.
DIAG = {
    "cuda_launch_blocking": ({"CUDA_LAUNCH_BLOCKING": "1"},
        "synchronous kernel launches: a CUDA fault is raised at the kernel that caused it (pinpoints Xid31/illegal address). Big slowdown (~2-4x decode)."),
    "ef_fence_off": ({"VLLM_EF_FENCE": "0"},
        "disable the EF attribution fences in mixed prefill+decode steps (fences localise the first CUDA fault; they cost little)."),
    "sched_guard_off": ({"VLLM_SCHED_INVARIANT_GUARD": "0"},
        "disable the scheduler progress-invariant guard (preempt+recompute); restores upstream behaviour so the raw scheduler fault reproduces."),
    "torch_cpp_stack": ({"TORCH_SHOW_CPP_STACKTRACES": "1"},
        "C++ stack traces on torch errors. Free until an error occurs."),
    "enforce_eager": ({"VLLM_SERVE_EXTRA_ARGS": "--enforce-eager"},
        "no CUDA graphs: rules graph capture/replay in or out of a fault. Large decode slowdown."),
    "no_async_scheduling": ({"VLLM_SERVE_EXTRA_ARGS": "--no-async-scheduling"},
        "synchronous scheduling (already the live default on hauhaucs; here for completeness)."),
}


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sh(cmd, timeout=40):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout
    except Exception as e:  # noqa: BLE001
        return f"<{e!r}>"


def http(url, method="GET", payload=None, token=None, timeout=5):
    body = json.dumps(payload).encode() if payload is not None else None
    h = {"Content-Type": "application/json"}
    if token:
        h["X-Admin-Token"] = token
    req = urllib.request.Request(url, data=body, method=method, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        try:
            return json.loads(raw)
        except ValueError:
            return {"http_status": r.status}


def admin_token():
    try:
        return open(TOKEN).read().strip()
    except OSError:
        return ""


# ------------------------------------------------------------------ event log (best-effort; never fails an action)
def emit(kind, summary, facts=None, handoff=False, action="engine-event", fingerprint=None):
    """Append to the estate event log (source=engine). handoff=True goes through tools/handoff.emit so the repeat
    of the identical situation is idempotent and it appears in Halo's inbox (list_handoffs)."""
    try:
        if ESTATE not in sys.path:
            sys.path.insert(0, ESTATE)
        subject = {"type": "component", "id": UNIT}
        if handoff:
            from tools import handoff as _h
            r = _h.emit("engine", UNIT, action, summary, facts or {}, fingerprint=fingerprint or f"{action}-{time.time()}",
                        subject_type="component", hold=False)
            return r.get("seq")
        from tools import event_log
        return event_log.append("engine", kind, subject, summary, facts=facts or {})
    except Exception as e:  # noqa: BLE001
        print(f"[engine-actuator] event emit failed: {e!r}", file=sys.stderr)
        return None


# ------------------------------------------------------------------ facts
def read_diag_env():
    out = {}
    try:
        for l in open(DIAG_ENV):
            l = l.strip()
            if l and not l.startswith("#") and "=" in l:
                k, v = l.split("=", 1)
                out[k] = v
    except OSError:
        pass
    return out


def staged_flags():
    """Names of DIAG flags whose env vars are all present in diag.env."""
    env = read_diag_env()
    names = []
    for name, (vars_, _d) in DIAG.items():
        if all(env.get(k) == v or (k == "VLLM_SERVE_EXTRA_ARGS" and v in env.get(k, "").split()) for k, v in vars_.items()):
            names.append(name)
    return names


def active_flags():
    """Flags the RUNNING engine process was started with (reads /proc/<pid>/environ of the unit's main pid)."""
    pid = sh(["systemctl", "show", UNIT, "-p", "MainPID", "--value"]).strip()
    env = {}
    try:
        for kv in open(f"/proc/{pid}/environ", "rb").read().split(b"\0"):
            if b"=" in kv:
                k, v = kv.decode("utf-8", "replace").split("=", 1)
                env[k] = v
    except Exception:  # noqa: BLE001
        return None
    names = []
    for name, (vars_, _d) in DIAG.items():
        if all(env.get(k) == v or (k == "VLLM_SERVE_EXTRA_ARGS" and v in env.get(k, "").split()) for k, v in vars_.items()):
            names.append(name)
    return names


def ledger_rows(limit=None, since=None):
    rows = []
    try:
        for l in open(LEDGER):
            try:
                rows.append(json.loads(l))
            except ValueError:
                pass
    except OSError:
        pass
    if since:
        rows = [r for r in rows if r.get("ts", "") >= since]
    seen, uniq = set(), []          # a backfill/replay can re-add a row; one fault is one (ts, signature)
    for r in rows:
        k = (r.get("ts"), r.get("signature"), r.get("kind"))
        if k not in seen:
            seen.add(k); uniq.append(r)
    rows = uniq
    return rows[-limit:] if limit else rows


def faults_summary(hours=24):
    since = datetime.fromtimestamp(time.time() - hours * 3600).strftime("%Y-%m-%dT%H:%M:%S")
    rows = [r for r in ledger_rows(since=since) if r.get("kind") == "FAULT"]
    sigs = {}
    for r in rows:
        k = r.get("signature", "?") + (":" + r["detail"] if r.get("detail") else "")
        sigs[k] = sigs.get(k, 0) + 1
    return {"window_h": hours, "faults": len(rows), "by_signature": sigs, "last": rows[-1] if rows else None}


def gateway_view():
    out = {}
    try:
        out["drain"] = http(f"{GATEWAY}/gateway/drain")
    except Exception as e:  # noqa: BLE001
        out["drain_error"] = repr(e)
    try:
        s = http(f"{GATEWAY}/gateway/stats")
        out["stats"] = {k: s.get(k) for k in ("inflight", "waiting", "local_healthy", "prefill_backlog_secs", "inflight_computed", "mode", "local_only")}
    except Exception as e:  # noqa: BLE001
        out["stats_error"] = repr(e)
    return out


def engine_healthy():
    try:
        urllib.request.urlopen(f"{ENGINE}/health", timeout=3).read()
        return True
    except Exception:  # noqa: BLE001
        return False


def unit_view():
    kv = {}
    for l in sh(["systemctl", "show", UNIT, "-p", "ActiveState,SubState,NRestarts,ActiveEnterTimestamp,MainPID,Result"]).splitlines():
        if "=" in l:
            k, v = l.split("=", 1)
            kv[k] = v
    return kv


def status():
    job = None
    try:
        job = json.load(open(JOB))
    except Exception:  # noqa: BLE001
        pass
    return {"as_of": now_iso(), "unit": unit_view(), "engine_healthy": engine_healthy(), "gateway": gateway_view(),
            "diag": {"staged_for_next_start": staged_flags(), "active_in_running_engine": active_flags()},
            "faults_24h": faults_summary(24), "recent_ledger": ledger_rows(limit=5), "restart_job": job}


# ------------------------------------------------------------------ actions
def write_diag(names, by, reason):
    for n in names:
        if n not in DIAG:
            raise SystemExit(f"unknown diagnostic flag {n!r}; allowed: {sorted(DIAG)}")
    env = {}
    for n in names:
        for k, v in DIAG[n][0].items():
            if k == "VLLM_SERVE_EXTRA_ARGS":
                env[k] = (env.get(k, "") + " " + v).strip()
            else:
                env[k] = v
    prev = staged_flags()
    lines = [f"# written by engine-actuator.py {now_iso()} by={by} reason={reason!r}; flags={names}"]
    lines += [f"{k}={v}" for k, v in env.items()]
    tmp = DIAG_ENV + ".tmp"
    open(tmp, "w").write("\n".join(lines) + "\n")
    os.replace(tmp, DIAG_ENV)
    seq = emit("action", f"engine diagnostic flags staged for next start: {names or 'none (cleared)'} (was {prev or 'none'}) by {by}: {reason}",
               {"action": "stage-diag", "flags": names, "previous": prev, "by": by, "reason": reason,
                "env": env, "takes_effect": "next engine start (planned or unplanned)"})
    return {"staged": names, "previous": prev, "event_seq": seq}


def drain_and_wait(deadline_s, reason, token):
    """Raise the gateway admission fence and wait for accepted requests to finish. Returns facts."""
    facts = {"fence": False, "waited_s": 0, "active_at_start": None, "active_at_end": None}
    try:
        cur = http(f"{GATEWAY}/gateway/drain", token=token)
        if cur.get("draining"):
            facts["note"] = "another drain lease is already held (a gateway publish?); not taking it"
            facts["active_at_start"] = cur.get("active")
            return facts, None
        opened = http(f"{GATEWAY}/gateway/drain", "POST", {"ttl_s": int(deadline_s) + 1200, "reason": f"engine planned restart: {reason}"[:120]}, token)
        lease = opened.get("lease")
        facts["fence"] = bool(lease)
        facts["active_at_start"] = opened.get("active")
    except Exception as e:  # noqa: BLE001
        facts["error"] = repr(e)
        return facts, None
    t0 = time.time()
    active = facts["active_at_start"]
    while time.time() - t0 < deadline_s:
        try:
            d = http(f"{GATEWAY}/gateway/drain", token=token)
            active = int(d.get("active") or 0)
            if active == 0:
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    facts["waited_s"] = round(time.time() - t0)
    facts["active_at_end"] = active
    return facts, lease


def release_lease(lease, token):
    if not lease:
        return
    try:
        if http(f"{GATEWAY}/gateway/drain", token=token).get("draining"):
            http(f"{GATEWAY}/gateway/drain", "DELETE", {"lease": lease}, token)
    except Exception:  # noqa: BLE001
        pass  # the lease expires on its own


def write_job(**kw):
    try:
        old = json.load(open(JOB))
    except Exception:  # noqa: BLE001
        old = {}
    old.update(kw)
    tmp = JOB + ".tmp"
    open(tmp, "w").write(json.dumps(old, indent=1))
    os.replace(tmp, JOB)


def do_restart(a):
    """The synchronous body of a planned restart. Runs detached when called through the MCP."""
    lk = open(LOCK, "w")
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(json.dumps({"refused": "another planned restart is already running"}))
        return 3
    by, reason = a.by, a.reason
    token = admin_token()
    write_job(state="starting", by=by, reason=reason, started=now_iso(), finished=None, drain=None, result=None)
    if a.flags is not None or a.clear_diag:
        names = [] if a.clear_diag else [x for x in (a.flags or "").split(",") if x]
        write_diag(names, by, reason)
    healthy = engine_healthy()
    before = status()
    inflight_before = (before["gateway"].get("drain") or {}).get("active")
    emit("action", f"planned engine restart requested by {by}: {reason}",
         {"action": "restart-requested", "by": by, "reason": reason, "engine_healthy": healthy, "gateway_active": inflight_before,
          "diag_staged": staged_flags(), "faults_24h": faults_summary(24)["faults"]})
    drain_facts, lease = ({"skipped": "engine unhealthy; nothing to drain"}, None)
    if healthy and not a.no_drain:
        write_job(state="draining")
        drain_facts, lease = drain_and_wait(a.drain_s, reason, token)
    write_job(state="stopping", drain=drain_facts)
    json.dump({"ts": now_iso(), "by": by, "reason": reason, "drain": drain_facts, "active_at_stop": drain_facts.get("active_at_end")},
              open(PLANNED, "w"))
    t_stop = time.time()
    r = subprocess.run(["sudo", "-n", "systemctl", "stop", UNIT], capture_output=True, text=True, timeout=200)
    stop_s = round(time.time() - t_stop, 1)
    # EF2 measured 2026-10-02: releasing the fence here let live traffic hit the cold engine during warm-up
    # (warm-up 504 s instead of ~160 s). Keep the fence until `start` returns (it returns after the warm-up hook).
    write_job(state="starting-engine", stop_s=stop_s, stop_rc=r.returncode)
    try:
        r2 = subprocess.run(["sudo", "-n", "systemctl", "start", UNIT], capture_output=True, text=True, timeout=900)
    finally:
        release_lease(lease, token)
    ok = engine_healthy()
    res = {"restart_rc": r2.returncode, "stop_s": stop_s, "healthy_after": ok, "drain": drain_facts,
           "diag_active": active_flags()}
    write_job(state="done" if ok else "failed", finished=now_iso(), result=res)
    emit("outcome", f"planned engine restart by {by} finished: healthy={ok}, stop took {stop_s}s, "
         f"{drain_facts.get('active_at_end')} request(s) still active when it stopped (was {drain_facts.get('active_at_start')})",
         {"action": "restart-finished", "by": by, "reason": reason, **res})
    print(json.dumps(res))
    return 0 if ok else 1


def spawn_detached(a):
    cmd = [sys.executable, os.path.abspath(__file__), "restart", "--reason", a.reason, "--by", a.by, "--drain-s", str(a.drain_s), "--foreground"]
    if a.flags is not None:
        cmd += ["--flags", a.flags]
    if a.clear_diag:
        cmd += ["--clear-diag"]
    if a.no_drain:
        cmd += ["--no-drain"]
    # refuse early (and visibly) if one is running
    try:
        j = json.load(open(JOB))
        if j.get("state") in ("starting", "draining", "stopping", "starting-engine"):
            lk = open(LOCK, "w")
            try:
                fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(lk, fcntl.LOCK_UN)
            except OSError:
                return {"refused": "a planned restart is already running", "job": j}
    except Exception:  # noqa: BLE001
        pass
    log = open(f"{BASE}/restart-job.log", "a")
    p = subprocess.Popen(cmd, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
    return {"scheduled": True, "pid": p.pid, "poll": "engine_status (restart_job.state: draining -> stopping -> starting-engine -> done)",
            "note": "restart drains the gateway first (bounded by drain_s), then stops, then starts; expect ~3-5 min until healthy + warm"}


def announce_start(_a):
    j = None
    try:
        j = json.load(open(JOB))
    except Exception:  # noqa: BLE001
        pass
    pool = None
    m = re.findall(r"GPU KV cache size: ([\d,]+) tokens", sh(["journalctl", "-u", UNIT, "--no-pager", "-n", "4000", "-o", "cat"], 60))
    if m:
        pool = int(m[-1].replace(",", ""))
    flags = active_flags()
    emit("observation", f"engine is back and healthy (flags active: {flags or 'none'}; KV pool {pool}); "
         f"faults in last 24h: {faults_summary(24)['faults']}",
         {"action": "engine-started", "diag_active": flags, "kv_pool_tokens": pool, "unit": unit_view(),
          "faults_24h": faults_summary(24), "planned_restart_job": (j or {}).get("state")})
    return 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("status")
    p = sp.add_parser("faults"); p.add_argument("--limit", type=int, default=10)
    sp.add_parser("flags")
    p = sp.add_parser("stage-diag"); p.add_argument("--flags", default=""); p.add_argument("--clear", action="store_true")
    p.add_argument("--by", default="cli"); p.add_argument("--reason", default="")
    p = sp.add_parser("restart")
    p.add_argument("--reason", required=True); p.add_argument("--by", default="cli")
    p.add_argument("--flags", default=None, help="comma list of DIAG names to stage before restarting")
    p.add_argument("--clear-diag", action="store_true"); p.add_argument("--drain-s", type=int, default=120)
    p.add_argument("--no-drain", action="store_true"); p.add_argument("--foreground", action="store_true")
    sp.add_parser("announce-start"); sp.add_parser("restart-status")
    a = ap.parse_args()
    if a.cmd == "status":
        print(json.dumps(status(), default=str))
    elif a.cmd == "faults":
        print(json.dumps({"summary": {h: faults_summary(h) for h in (1, 24, 168)}, "tail": ledger_rows(limit=a.limit)}, default=str))
    elif a.cmd == "flags":
        print(json.dumps({"allowed": {k: {"env": v[0], "about": v[1]} for k, v in DIAG.items()}, "staged": staged_flags(),
                          "active_in_running_engine": active_flags()}))
    elif a.cmd == "stage-diag":
        names = [] if a.clear else [x for x in a.flags.split(",") if x]
        print(json.dumps(write_diag(names, a.by, a.reason or ("clear" if a.clear else "stage"))))
    elif a.cmd == "restart":
        a.reason = a.reason.strip()
        if len(a.reason) < 8:
            print(json.dumps({"refused": "reason is required (what you saw and why a restart is the move)"})); return 2
        if a.foreground:
            return do_restart(a)
        print(json.dumps(spawn_detached(a)))
    elif a.cmd == "announce-start":
        return announce_start(a)
    elif a.cmd == "restart-status":
        print(open(JOB).read() if os.path.exists(JOB) else "{}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
