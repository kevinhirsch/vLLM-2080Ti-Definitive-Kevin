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
                              requests AND the engine's own num_requests_running/waiting (:8001/metrics, catches direct callers) to finish (bounded) -> stop -> release the fence -> start. Detached by default.
  announce-start              (ExecStartPost) emit a "engine is back" observation with the flags that are active
  restart-status              state of the last/running planned restart

LV (2026-10-03) -- this file is also the ONE engine-liveness authority (see the block above main() and
deploy/docs/engine-liveness-authority.md):
  tick [--no-act]             observe -> verify own actions -> classify into the declared state machine -> publish
                              liveness-state.json -> take the declared exit (start an unowned DOWN engine, recover a STUCK_BOOT /
                              UNRESPONSIVE one) under the one lock, rate limit, backoff and breaker. vllm-watchdog.sh runs it each minute.
  recover --cause C --by B    a detector (the watchdog's confirmed wedge) asks for a kill+restart; refused/deferred when not allowed
  hold acquire|release|renew|run|status   TTL-bounded holds. kind engine = a window owns the engine (no automatic action, no
                              probes, planned restarts only by the holder via --hold); kind quiesce = a release (gateway publish) is in flight
  liveness                    print the published state;   reset-breaker --by --reason   close an open breaker

The allow-list is DIAG below: a fixed map name -> env vars. Nothing else can be set through this tool. Every change is
reversible (stage-diag --clear) and recorded. Flags only take effect at the next engine start.
"""
import argparse, contextlib, fcntl, json, os, re, signal, subprocess, sys, time, urllib.request
from datetime import datetime

_TERMINATION_SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)


class Terminated(BaseException):
    """L77 (2026-10-03): a termination signal arrived. BaseException so no `except Exception` swallows it, while every
    `finally` still runs and the gateway fence / offline window this process opened is released (it used to be stranded
    until its TTL when the actuator was killed mid-drain)."""

    def __init__(self, signum):
        super().__init__(f"terminated by signal {signum}")
        self.signum = signum


@contextlib.contextmanager
def terminate_as_exception():
    def handler(signum, _frame):
        raise Terminated(signum)
    previous = {}
    try:
        for sg in _TERMINATION_SIGNALS:
            previous[sg] = signal.signal(sg, handler)
    except ValueError:      # not the main thread
        pass
    try:
        yield
    finally:
        for sg, old in previous.items():
            signal.signal(sg, old)


@contextlib.contextmanager
def shielded():
    """Cleanup is not interrupted halfway by a second signal."""
    previous = {}
    try:
        for sg in _TERMINATION_SIGNALS:
            previous[sg] = signal.signal(sg, signal.SIG_IGN)
    except ValueError:
        pass
    try:
        yield
    finally:
        for sg, old in previous.items():
            signal.signal(sg, old)

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


#: AU 2026-10-03: how long a planned restart / start announcement waits for /health before calling the boot failed.
#: Matches warmup-after-start.sh CAP_SECS. The unit is Type=simple, so `systemctl start` returns as soon as the process
#: forks whenever the warm-up hook skips itself (a frontier-queue window is running) -- measured 10-02..10-03: 66 of 81
#: restart-finished events said healthy=False and the same second announce-start said "engine is back and healthy",
#: both read before the API was up. Waiting on /health turns both into measured facts.
HEALTH_WAIT_S = int(os.environ.get("ENGINE_ACTUATOR_HEALTH_WAIT_S", "420"))


def wait_engine_healthy(budget_s, poll_s=5.0):
    """Poll /health until it answers or budget_s passes. Returns (healthy, waited_s). Always samples at least once."""
    t0 = time.time()
    while True:
        if engine_healthy():
            return True, round(time.time() - t0, 1)
        if time.time() - t0 >= budget_s:
            return False, round(time.time() - t0, 1)
        time.sleep(poll_s)


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
    try:   # LV: the liveness authority's published state + holds (what engine_status shows Halo)
        live = json.load(open(LSTATE))
        live = {k: live.get(k) for k in ("state", "since", "reason", "as_of", "gate", "last_action", "declared", "implicit_holds")}
    except Exception:  # noqa: BLE001
        live = None
    return {"as_of": now_iso(), "unit": unit_view(), "engine_healthy": engine_healthy(), "gateway": gateway_view(),
            "diag": {"staged_for_next_start": staged_flags(), "active_in_running_engine": active_flags()},
            "faults_24h": faults_summary(24), "recent_ledger": ledger_rows(limit=5), "restart_job": job,
            "liveness": live, "holds": active_holds()}


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


def drain_and_wait(deadline_s, reason, token, by=None):
    """Raise the gateway admission fence and wait for accepted requests to finish. Returns facts."""
    facts = {"fence": False, "waited_s": 0, "active_at_start": None, "active_at_end": None}
    try:
        cur = http(f"{GATEWAY}/gateway/drain", token=token)
        if cur.get("draining"):
            facts["note"] = "another drain lease is already held (a gateway publish?); not taking it"
            facts["active_at_start"] = cur.get("active")
            return facts, None
        opened = http(f"{GATEWAY}/gateway/drain", "POST", {"ttl_s": int(deadline_s) + 1200, "reason": f"engine planned restart: {reason}"[:120], "by": by or "engine-actuator"}, token)
        lease = opened.get("lease")
        facts["fence"] = bool(lease)
        facts["active_at_start"] = opened.get("active")
    except Exception as e:  # noqa: BLE001
        facts["error"] = repr(e)
        return facts, None
    t0 = time.time()
    active = facts["active_at_start"]
    try:
        while time.time() - t0 < deadline_s:
            try:
                d = http(f"{GATEWAY}/gateway/drain", token=token)
                active = int(d.get("active") or 0)
                if active == 0:
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)
    except BaseException:  # noqa: BLE001  L77: the caller never saw this lease; a signal here must not strand the fence
        with shielded():
            release_lease(lease, token)
        raise
    facts["waited_s"] = round(time.time() - t0)
    facts["active_at_end"] = active
    return facts, lease


def engine_progress(engine=None, timeout=3):
    """Token progress counter from the engine's own /metrics: generation_tokens_total + prompt_tokens_total summed over
    label sets (a prefill-only step moves the second one). None when unreadable."""
    try:
        txt = urllib.request.urlopen(f"{engine or ENGINE}/metrics", timeout=timeout).read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return None
    tot, seen = 0.0, False
    for name in ("vllm:generation_tokens_total", "vllm:prompt_tokens_total"):
        for m in re.finditer(r"^" + re.escape(name) + r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", txt, re.M):
            tot += float(m.group(1)); seen = True
    return int(tot) if seen else None


EXT_POLL_S = 5.0        # polling interval once past the base deadline
EXT_STALL_POLLS = 3     # consecutive polls with no token progress = a wedge, not work (15 s with EXT_POLL_S)


def wait_drained(get_active, deadline_s, *, hard_cap_s=None, get_progress=None, poll_s=2.0, ext_poll_s=None, stall_polls=None,
                 settle=1, unreadable="continue", sleep=None, clock=None):
    """Wait for get_active() (int, or None if unreadable) to reach 0 on `settle` consecutive polls.
    Base budget deadline_s (always >= 1 sample). If still active at the deadline AND hard_cap_s > deadline_s AND get_progress
    is given, keep waiting while the progress counter advances, until hard_cap_s; stop early when it does not advance for
    `stall_polls` consecutive polls (a wedge, not work). Returns facts incl. end_reason in
    idle | deadline | cap | stalled | unreadable, and extended_s (time spent past the deadline)."""
    sleep, clock = sleep or time.sleep, clock or time.time        # resolved at call time so tests can patch time
    ext_poll_s = EXT_POLL_S if ext_poll_s is None else ext_poll_s
    stall_polls = EXT_STALL_POLLS if stall_polls is None else stall_polls
    t0 = clock()
    out = {"active_at_start": None, "active_at_end": None, "end_reason": None, "idle": None, "extended_s": 0, "waited_s": 0}
    zeros = stall = 0
    last_p = None
    ext_start = None
    can_extend = bool(get_progress) and bool(hard_cap_s) and hard_cap_s > deadline_s
    while True:
        elapsed = clock() - t0
        past = elapsed >= deadline_s
        a = get_active()
        if a is None:
            if unreadable == "stop" or past:
                out["end_reason"] = "unreadable"
                break
        else:
            if out["active_at_start"] is None:
                out["active_at_start"] = a
            out["active_at_end"] = a
            zeros = zeros + 1 if a == 0 else 0
            if zeros >= settle:
                out["idle"], out["end_reason"] = True, "idle"
                break
            if past:
                if not can_extend:
                    out["idle"], out["end_reason"] = False, "deadline"
                    break
                if elapsed >= hard_cap_s:
                    out["idle"], out["end_reason"] = False, "cap"
                    break
                p = get_progress()
                if ext_start is None:
                    ext_start, last_p, stall = elapsed, p, 0
                else:
                    stall = 0 if (p is not None and last_p is not None and p > last_p) else stall + 1
                    last_p = p if p is not None else last_p
                    if stall >= stall_polls:
                        out["idle"], out["end_reason"] = False, "stalled"
                        break
        if past:
            step = max(0.001, min(ext_poll_s, (hard_cap_s or deadline_s) - elapsed))
        else:
            step = max(0.001, min(poll_s, deadline_s - elapsed))
        sleep(step)
    end = clock() - t0
    out["waited_s"] = round(end)
    out["extended_s"] = round(end - deadline_s) if end > deadline_s else 0
    return out


def offline_and_wait(deadline_s, reason, token, by=None, hard_cap_s=None):
    """CF (2026-10-02): open a PLANNED LOCAL-OFFLINE window instead of a drain fence. The gateway routes new work to the
    remote valve (inside the daily cap) rather than refusing it, lets accepted local work finish, and the estate keeps
    flowing during the restart. Returns (facts, lease) or None when the gateway has no such endpoint (older gateway,
    or the window cannot be taken): the caller then falls back to the drain fence."""
    facts = {"fence": False, "strategy": "offline-window", "waited_s": 0, "active_at_start": None, "active_at_end": None}
    lease = None
    try:
        cur = http(f"{GATEWAY}/gateway/offline", token=token)
        if "offline" not in cur:
            return None
        if cur.get("offline"):
            # LV 2026-10-03: a window is ALREADY open (typically the caller's own `gateway-offline.py run` around a window that
            # restarts through here). New work already goes remote, so ride it: never fall back to the drain fence, which
            # refuses ALL admission and is for gateway code swaps only. We hold no lease of it, so we release nothing.
            facts.update(strategy="existing-offline-window", fence=True, window_by=cur.get("by"), window_reason=cur.get("reason"),
                         window_remaining_s=cur.get("remaining_s"), active_at_start=cur.get("local_active"))
        else:
            opened = http(f"{GATEWAY}/gateway/offline", "POST", {"ttl_s": min(3600, int(deadline_s) + 1200),
                          "reason": f"engine planned restart: {reason}"[:120], "by": by or "engine-actuator"}, token)
            lease = opened.get("lease")
            if not lease:
                return None
            facts["fence"] = True
            facts["active_at_start"] = opened.get("local_active")
    except Exception:  # noqa: BLE001
        return None
    # FX2/L63: inside an offline window NEW work already goes remote, so the local count can only fall; a longer wait costs
    # restart latency only. Past deadline_s keep waiting while the engine is still making token progress, up to hard_cap_s.
    def _active():
        try:
            return int(http(f"{GATEWAY}/gateway/offline", token=token).get("local_active") or 0)
        except Exception:  # noqa: BLE001
            return None
    try:
        w = wait_drained(_active, deadline_s, hard_cap_s=hard_cap_s, get_progress=engine_progress)
    except BaseException:  # noqa: BLE001  L77: the caller never saw this lease; release the window rather than strand it to TTL
        if lease:
            with shielded():
                release_lease(("offline", lease), token)
        raise
    facts.update(waited_s=w["waited_s"], active_at_end=w["active_at_end"] if w["active_at_end"] is not None else facts["active_at_start"],
                 end_reason=w["end_reason"], extended_s=w["extended_s"], hard_cap_s=hard_cap_s)
    return facts, (("offline", lease) if lease else None)


def engine_inflight(engine=None, timeout=3):
    """Requests the ENGINE itself holds, read from its own /metrics: {"running": n, "waiting": n}, or None if unreadable.
    The gateway fence/offline window only counts requests the gateway accepted; callers that go straight to :8001
    (bypassing routing, the spend ledger and the drain) are visible only here. FX2 2026-10-03: the 23:18 planned restart
    stopped the engine under one such caller."""
    try:
        txt = urllib.request.urlopen(f"{engine or ENGINE}/metrics", timeout=timeout).read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return None
    out = {}
    for key, name in (("running", "vllm:num_requests_running"), ("waiting", "vllm:num_requests_waiting")):
        m = re.search(r"^" + re.escape(name) + r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", txt, re.M)
        if not m:
            return None
        out[key] = int(float(m.group(1)))
    return out


def wait_engine_idle(budget_s, engine=None, poll_s=2.0, settle=2, hard_cap_s=None, **kw):
    """Wait until the engine's own running+waiting count is 0 on `settle` consecutive polls (one zero can be the gap between
    two back-to-back direct requests), for at most budget_s seconds (always at least one sample); with hard_cap_s > budget_s
    keep waiting past the budget while the engine's token counters advance (L63). Never raises, never blocks on an
    unreadable /metrics (the engine is then unhealthy and there is nothing to wait for). Returns facts."""
    last = {}

    def _active():
        cur = engine_inflight(engine)
        if cur is None:
            return None
        if "running_at_start" not in last:
            last["running_at_start"], last["waiting_at_start"] = cur["running"], cur["waiting"]
        last["running_at_end"], last["waiting_at_end"] = cur["running"], cur["waiting"]
        return cur["running"] + cur["waiting"]
    w = wait_drained(_active, budget_s, hard_cap_s=hard_cap_s, get_progress=lambda: engine_progress(engine), poll_s=poll_s,
                     settle=settle, unreadable="stop", **kw)
    f = {"checked": True, "idle": w["idle"], "waited_s": w["waited_s"], "running_at_start": last.get("running_at_start"),
         "waiting_at_start": last.get("waiting_at_start"), "running_at_end": last.get("running_at_end"),
         "waiting_at_end": last.get("waiting_at_end"), "end_reason": w["end_reason"], "extended_s": w["extended_s"]}
    if w["end_reason"] == "unreadable":
        f["error"] = "engine /metrics unreadable"
        f["idle"] = None
    return f


def release_lease(lease, token):
    if not lease:
        return
    if isinstance(lease, tuple) and lease[0] == "offline":
        try:
            http(f"{GATEWAY}/gateway/offline", "DELETE", {"lease": lease[1]}, token)
        except Exception:  # noqa: BLE001
            pass  # the lease expires on its own
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
    # LV: a planned restart is refused while someone else holds the engine (a window) or a release is in flight (quiesce:
    # a gateway publish is draining). The holder itself passes --hold <lease>.
    mine = getattr(a, "hold", None)
    blocking = [h for h in active_holds() if h.get("lease") != mine]
    if blocking:
        h = blocking[0]
        refused = {"refused": f"{h.get('kind')} hold held by {h.get('by')} until {datetime.fromtimestamp(h['until']).isoformat(timespec='seconds')}: "
                              f"{h.get('reason')}", "hint": "the holder passes --hold <lease>; otherwise wait for release/expiry"}
        emit("observation", f"planned engine restart by {by} refused: {refused['refused']}"[:300], {"action": "restart-refused-hold", "by": by, "reason": reason, **refused})
        print(json.dumps(refused))
        return 3
    token = admin_token()
    # RS: a restart whose only purpose is to apply flags the RUNNING engine already has is a pure no-op that still costs a full
    # outage (drain + ~4 min boot, every request refused). Refuse it unless --force; Halo can still restart for any other reason.
    if (a.flags is not None or a.clear_diag) and not getattr(a, "force", False) and engine_healthy():
        want = sorted([] if a.clear_diag else [x for x in (a.flags or "").split(",") if x])
        have = active_flags()
        if want and have is not None and sorted(have) == want:  # non-empty only: '--flags ""' / --clear-diag may accompany a non-DIAG change
            skipped = {"refused": "no-op restart: the running engine already has exactly these diag flags", "flags": want,
                       "hint": "pass --force to restart anyway (e.g. to clear engine state)"}
            emit("observation", f"planned engine restart by {by} skipped: flags {want or 'none'} already active", {"action": "restart-skipped-noop", "by": by, "reason": reason, **skipped})
            print(json.dumps(skipped))
            return 0
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
    lease, stop_issued, started = None, False, False
    try:
        with terminate_as_exception():
            drain_facts, lease = ({"skipped": "engine unhealthy; nothing to drain"}, None)
            if healthy and not a.no_drain:
                write_job(state="draining")
                t_drain = time.time()
                hard_cap = float(max(a.drain_s, getattr(a, "drain_max_s", 0) or 0))
                got = None if os.environ.get("ENGINE_ACTUATOR_FORCE_DRAIN") else offline_and_wait(a.drain_s, reason, token, by, hard_cap_s=hard_cap)
                drain_facts, lease = got if got is not None else drain_and_wait(a.drain_s, reason, token, by)
                # FX2: the fence counts only gateway-accepted requests. Also wait on the engine's own in-flight count inside the SAME
                # budget (what is left of it; at least one sample), so a direct :8001 caller is not cut mid-request. In an offline
                # window (new work already goes remote) both waits may run past --drain-s while tokens still advance, to the hard cap.
                offline = drain_facts.get("strategy") in ("offline-window", "existing-offline-window")
                used = time.time() - t_drain
                eng = wait_engine_idle(max(0.0, a.drain_s - used), hard_cap_s=(hard_cap - used) if offline and hard_cap > a.drain_s else None)
                drain_facts["engine"] = eng
                drain_facts["total_waited_s"] = round(time.time() - t_drain)
                drain_facts["engine_active_at_end"] = None if eng.get("running_at_end") is None else eng["running_at_end"] + (eng["waiting_at_end"] or 0)
            write_job(state="stopping", drain=drain_facts)
            json.dump({"ts": now_iso(), "by": by, "reason": reason, "drain": drain_facts, "active_at_stop": drain_facts.get("active_at_end")},
                      open(PLANNED, "w"))
            t_stop = time.time()
            stop_issued = True
            r = subprocess.run(["sudo", "-n", "systemctl", "stop", UNIT], capture_output=True, text=True, timeout=200)
            stop_s = round(time.time() - t_stop, 1)
            # EF2 measured 2026-10-02: releasing the fence here let live traffic hit the cold engine during warm-up
            # (warm-up 504 s instead of ~160 s). Keep the fence until `start` returns (it returns after the warm-up hook).
            write_job(state="starting-engine", stop_s=stop_s, stop_rc=r.returncode)
            try:
                r2 = subprocess.run(["sudo", "-n", "systemctl", "start", UNIT], capture_output=True, text=True, timeout=900)
                started = True
            finally:
                with shielded():
                    release_lease(lease, token)
                lease = None
            ok, health_wait_s = wait_engine_healthy(getattr(a, "health_wait_s", HEALTH_WAIT_S))
            res = {"restart_rc": r2.returncode, "stop_s": stop_s, "healthy_after": ok, "health_wait_s": health_wait_s,
                   "drain": drain_facts, "diag_active": active_flags()}
            write_job(state="done" if ok else "failed", finished=now_iso(), result=res)
            emit("outcome", f"planned engine restart by {by} finished: healthy={ok} (after {health_wait_s}s), stop took {stop_s}s, "
                 f"{drain_facts.get('active_at_end')} gateway request(s) and {drain_facts.get('engine_active_at_end')} engine request(s) "
                 f"still active when it stopped (gateway was {drain_facts.get('active_at_start')}); waited {drain_facts.get('total_waited_s')}s, "
                 f"drain ended: {drain_facts.get('end_reason')}/{(drain_facts.get('engine') or {}).get('end_reason')}"
                 f"{' (extended ' + str(drain_facts.get('extended_s')) + 's past --drain-s)' if drain_facts.get('extended_s') else ''}",
                 {"action": "restart-finished", "by": by, "reason": reason, **res})
            print(json.dumps(res))
            return 0 if ok else 1
    except BaseException as exc:  # noqa: BLE001  Terminated (signal) or any unexpected error: never strand the fence/window
        sig = getattr(exc, "signum", None)
        with shielded():
            if stop_issued and not started:
                # The engine may be down because of this restart: ask systemd to bring it back without blocking, so a killed
                # actuator does not leave the estate with no engine (the watchdog is the backstop, not the plan).
                try:
                    subprocess.run(["sudo", "-n", "systemctl", "start", "--no-block", UNIT], capture_output=True, text=True, timeout=30)
                except Exception:  # noqa: BLE001
                    pass
            release_lease(lease, token)
            lease = None
            try:
                write_job(state="aborted", finished=now_iso(), result={"aborted": repr(exc)[:200], "signal": sig,
                                                                       "engine_start_requested": bool(stop_issued and not started)})
                emit("outcome", f"planned engine restart by {by} ABORTED ({'signal ' + str(sig) if sig else repr(exc)[:120]}); "
                     f"gateway fence/window released" + ("; engine start requested" if stop_issued and not started else ""),
                     {"action": "restart-aborted", "by": by, "reason": reason, "signal": sig, "stop_issued": stop_issued})
            except Exception:  # noqa: BLE001
                pass
        if sig:
            return 128 + sig
        raise
    finally:
        if lease is not None:
            with shielded():
                release_lease(lease, token)


def spawn_detached(a):
    cmd = [sys.executable, os.path.abspath(__file__), "restart", "--reason", a.reason, "--by", a.by, "--drain-s", str(a.drain_s), "--drain-max-s", str(getattr(a, "drain_max_s", 600)), "--foreground"]
    if a.flags is not None:
        cmd += ["--flags", a.flags]
    if a.clear_diag:
        cmd += ["--clear-diag"]
    if a.no_drain:
        cmd += ["--no-drain"]
    if getattr(a, "force", False):
        cmd += ["--force"]
    if getattr(a, "health_wait_s", None) is not None:
        cmd += ["--health-wait-s", str(a.health_wait_s)]
    if getattr(a, "hold", None):
        cmd += ["--hold", a.hold]
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
    """ExecStartPost hook. Announces "back and healthy" only once /health answers (AU 2026-10-03).

    When the engine is not healthy yet (the warm-up hook skipped itself, or gave up), this records that the process
    started and hands the wait to a detached waiter, so the unit's start is never held longer than before."""
    if not getattr(_a, "wait_healthy", False) and not engine_healthy():
        emit("observation", "engine process started; waiting for /health before announcing it healthy",
             {"action": "engine-process-started", "unit": unit_view(), "health_wait_s": HEALTH_WAIT_S})
        try:
            subprocess.Popen([sys.executable, os.path.abspath(__file__), "announce-start", "--wait-healthy"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, start_new_session=True)
        except Exception as e:  # noqa: BLE001
            emit("observation", f"engine start announcement could not wait for health: {e!r}"[:200],
                 {"action": "engine-start-unconfirmed", "error": repr(e)[:200]})
        return 0
    boot_wait_s = 0.0
    if getattr(_a, "wait_healthy", False):
        ok, boot_wait_s = wait_engine_healthy(HEALTH_WAIT_S)
        if not ok:
            emit("observation", f"engine process started but /health did not answer within {boot_wait_s}s",
                 {"action": "engine-start-unhealthy", "waited_s": boot_wait_s, "unit": unit_view(),
                  "faults_24h": faults_summary(24)})
            return 0
    j = None
    try:
        j = json.load(open(JOB))
    except Exception:  # noqa: BLE001
        pass
    pool = None
    m = re.findall(r"GPU KV cache size: ([\d,]+) tokens", sh(["journalctl", "-u", UNIT, "--no-pager", "-n", "4000", "-o", "cat"], 60))
    if m:
        pool = int(m[-1].replace(",", ""))
    try:  # RS: heal any engine death the ExecStopPost collector lost (idempotent, no new cron)
        subprocess.Popen([sys.executable, f"{BASE}/engine-fault-collector.py", "--reconcile", "--hours", "3"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except Exception:  # noqa: BLE001
        pass
    flags = active_flags()
    emit("observation", f"engine is back and healthy (flags active: {flags or 'none'}; KV pool {pool}); "
         f"faults in last 24h: {faults_summary(24)['faults']}",
         {"action": "engine-started", "diag_active": flags, "kv_pool_tokens": pool, "unit": unit_view(), "health_wait_s": boot_wait_s,
          "faults_24h": faults_summary(24), "planned_restart_job": (j or {}).get("state")})
    return 0


# ================================================================== LV (2026-10-03): the ONE engine-liveness authority
# Before this, five owners (vllm-watchdog.sh, estate-watchdog.sh, this actuator, the fault collector, the gateway's local-down
# flag) each had their own probe, cooldown and kill/restart path; windows stopped the watchdog TIMER to keep it off their engine
# (DFT left it stopped 4.5 h), and nothing at all restarted an engine that was stopped and abandoned, or one whose API never came
# up. Now: every automatic stop/start/kill goes through recover()/tick() here, under ONE lock (LOCK, shared with planned restarts),
# ONE rate limit with backoff, ONE circuit breaker, and ONE published state (LSTATE) that everything else reads.
# Windows take a TTL-bounded HOLD instead of stopping the timer: a hold expires by itself, so a dead window can never leave the
# engine unwatched. Design: deploy/docs/engine-liveness-authority.md.
#
# Doctrine split (EF2 + Never Stuck By Construction): the AUTOMATIC actions here only restore the declared default ("the engine is
# up unless a holder says otherwise") -- they are the declared exits of stuck states. Discretionary restarts (flags, configs,
# arms) stay with Halo via `restart`; a crash loop or an open breaker is handed to Halo, never "fixed" by guessing.
HOLDS = f"{BASE}/liveness-holds.json"
HOLDS_LOCK = f"{BASE}/liveness-holds.lock"
LSTATE = f"{BASE}/liveness-state.json"
LACTIONS = f"{BASE}/liveness-actions.jsonl"
LPAUSE = f"{BASE}/LIVENESS_PAUSE"                 # kill switch: observe + publish state only, never act
WD_STATE = f"{BASE}/watchdog-state.json"          # vllm-watchdog.sh probe counters (read-only here)
FQ_RUNNING = f"{BASE}/frontier-queue/RUNNING"     # legacy frontier-queue window flag (runner trusts it for 3 h)
FQ_RUNNING_MAX_S = 10800
#: legacy windows that neither hold nor stop the timer (lane window.sh / *_window.sh / *_driver.sh, the frontier runner's own guard
#: pattern): treated as an implicit engine hold for at most HOLD_MAX_TTL_S of their runtime, until every window takes a real hold
WINDOW_PROC_RE = re.compile(r"^(?:\S*/)?bash\s+\S*(?:_window|_driver|/window)\.sh(?:\s|$)")
HOLD_ENV = "ENGINE_HOLD_LEASE"                    # `hold run` exports its lease to the child; `restart` reads it as --hold
HOLD_KINDS = ("engine", "quiesce")                # engine: the holder owns the engine. quiesce: a release (gateway publish) is in flight

def _env_i(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default

HOLD_MAX_TTL_S = _env_i("LIVENESS_HOLD_MAX_TTL_S", 8 * 3600)   # DFT's 16000 s fine-tune window fits; nothing holds forever
BOOT_DEADLINE_S = _env_i("LIVENESS_BOOT_DEADLINE_S", 900)       # cold compile+capture ~4-5 min; window boot budgets use 900
UNRESPONSIVE_S = _env_i("LIVENESS_UNRESPONSIVE_S", 180)         # API was up this boot, then stopped answering for this long
DOWN_GRACE_S = _env_i("LIVENESS_DOWN_GRACE_S", 180)            # systemd relaunches crashes in 15 s; leave stop->start pairs room
AUTO_MAX_PER_HOUR = _env_i("LIVENESS_AUTO_MAX_PER_HOUR", 2)     # the watchdog's long-standing rail, now for ALL automatic actions
AUTO_MIN_GAP_S = _env_i("LIVENESS_AUTO_MIN_GAP_S", 600)         # base gap; doubles per consecutive failed action (backoff)
BACKOFF_MAX_S = _env_i("LIVENESS_BACKOFF_MAX_S", 7200)
BREAKER_FAILS = _env_i("LIVENESS_BREAKER_FAILS", 3)            # consecutive automatic actions that did not bring /health back
VERIFY_S = BOOT_DEADLINE_S + 60                                 # an action is judged failed when /health is still down after this
CRASH_LOOP_FAULTS = _env_i("LIVENESS_CRASH_LOOP_FAULTS", 5)     # FAULT ledger rows ...
CRASH_LOOP_WINDOW_S = _env_i("LIVENESS_CRASH_LOOP_WINDOW_S", 1800)  # ... inside this window, engine still not healthy

#: The declared machine (Never Stuck By Construction): every non-terminal state names its owner, its deadline and its exits.
#: Published verbatim in LSTATE so the estate's generic progress invariant can hold each state to its own declaration.
STATES = {
    "UP":             {"kind": "resting", "owner": "-", "deadline_s": None, "exits": ["SUSPECT", "UNRESPONSIVE", "DOWN", "PLANNED", "HELD"]},
    "SUSPECT":        {"kind": "active", "owner": "vllm-watchdog.sh probes", "deadline_s": UNRESPONSIVE_S,
                       "exits": ["UP", "RECOVERING (wedge confirmed -> recover)", "UNRESPONSIVE"]},
    "BOOTING":        {"kind": "active", "owner": "systemd + warm-up hook", "deadline_s": BOOT_DEADLINE_S,
                       "exits": ["UP", "STUCK_BOOT", "DOWN (process died; systemd Restart=always relaunches)"]},
    "STOPPING":       {"kind": "active", "owner": "systemd (TimeoutStopSec, then SIGKILL)", "deadline_s": 120, "exits": ["DOWN", "BOOTING"]},
    "DOWN":           {"kind": "active", "owner": "liveness authority (start)", "deadline_s": DOWN_GRACE_S, "exits": ["BOOTING", "BREAKER_OPEN"]},
    "STUCK_BOOT":     {"kind": "active", "owner": "liveness authority (recover)", "deadline_s": 0, "exits": ["RECOVERING", "BREAKER_OPEN"]},
    "UNRESPONSIVE":   {"kind": "active", "owner": "liveness authority (recover)", "deadline_s": 0, "exits": ["RECOVERING", "BREAKER_OPEN"]},
    "RECOVERING":     {"kind": "active", "owner": "liveness authority (verifies its own action)", "deadline_s": VERIFY_S,
                       "exits": ["UP", "action failed -> backoff / BREAKER_OPEN"]},
    "PLANNED":        {"kind": "active", "owner": "engine-actuator planned restart (restart.lock holder)", "deadline_s": 2400,
                       "exits": ["UP", "BOOTING", "lock holder died -> job reconciled 'abandoned'"]},
    "HELD":           {"kind": "active", "owner": "the hold's holder", "deadline_s": HOLD_MAX_TTL_S,
                       "exits": ["release", "TTL expiry", "holder pid died (run mode) -> re-evaluated"]},
    "OFFLINE_WINDOW": {"kind": "active", "owner": "the gateway offline-window lease holder", "deadline_s": 3600,
                       "exits": ["window closed or lease expired -> re-evaluated"]},
    "CRASH_LOOP":     {"kind": "active", "owner": "Halo (hand-off engine-crash-loop)", "deadline_s": None,
                       "exits": ["UP", "Halo planned restart / rollback", "BREAKER_OPEN"]},
    "BREAKER_OPEN":   {"kind": "active", "owner": "liveness authority (half-open retry) + Halo hand-off", "deadline_s": BACKOFF_MAX_S,
                       "exits": ["half-open single attempt at next_try", "UP", "reset-breaker"]},
    "PAUSED":         {"kind": "active", "owner": "Kevin (kill switch LIVENESS_PAUSE)", "deadline_s": 86400,
                       "exits": ["rm LIVENESS_PAUSE"]},
}
NO_PROBE_STATES = ("HELD", "OFFLINE_WINDOW", "PLANNED", "PAUSED")   # vllm-watchdog.sh does not probe (or count) in these


# ------------------------------------------------------------------ holds
def _proc_start(pid):
    try:
        with open(f"/proc/{int(pid)}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[19]
    except (OSError, ValueError, IndexError, TypeError):
        return None


def _hold_owner_dead(h):
    """A `hold run` record is void the moment its wrapper is gone (or its pid reused). Plain `acquire` holds are never presumed
    dead: the CLI that wrote them exits by design; their TTL is their exit."""
    if h.get("mode") not in ("run", "owned") or not isinstance(h.get("pid"), int):
        return False
    cur = _proc_start(h["pid"])
    return cur is None or (bool(h.get("pid_start")) and h["pid_start"] != cur)


def _hold_valid(h, now):
    return float(h.get("until") or 0) > now and not _hold_owner_dead(h)


@contextlib.contextmanager
def _holds_locked():
    fh = open(HOLDS_LOCK, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def _holds_read():
    try:
        rows = json.load(open(HOLDS))
        return rows if isinstance(rows, list) else []
    except (OSError, ValueError):
        return []


def _holds_write(rows):
    tmp = HOLDS + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rows, fh, indent=1)
    os.replace(tmp, HOLDS)


def active_holds(now=None, kind=None):
    now = time.time() if now is None else now
    return [h for h in _holds_read() if _hold_valid(h, now) and (kind is None or h.get("kind") == kind)]


def hold_acquire(kind, by, reason, ttl_s, mode="acquire", pid=None):
    if kind not in HOLD_KINDS:
        return {"refused": f"kind must be one of {HOLD_KINDS}"}
    reason = (reason or "").strip()
    if len(reason) < 8:
        return {"refused": "reason is required (what the hold protects)"}
    if not 60 <= int(ttl_s) <= HOLD_MAX_TTL_S:
        return {"refused": f"ttl_s must be 60..{HOLD_MAX_TTL_S}"}
    now = time.time()
    with _holds_locked():
        rows = _holds_read()
        live = [h for h in rows if _hold_valid(h, now)]
        expired = [h for h in rows if not _hold_valid(h, now)]
        clash = [h for h in live if h.get("kind") == kind]
        if clash:
            return {"refused": f"a {kind} hold is already held", "holder": {k: clash[0].get(k) for k in ("by", "reason", "until", "lease")}}
        pid = os.getpid() if pid is None else pid
        h = {"lease": os.urandom(8).hex(), "kind": kind, "by": by, "reason": reason[:200], "acquired": now, "until": now + int(ttl_s),
             "ttl_s": int(ttl_s), "mode": mode, "pid": pid, "pid_start": _proc_start(pid)}
        _holds_write(live + [h])
    for e in expired:
        emit("observation", f"liveness hold by {e.get('by')} ended without release ({'holder died' if _hold_owner_dead(e) else 'TTL expired'})",
             {"action": "liveness-hold-expired", **{k: e.get(k) for k in ("kind", "by", "reason", "lease", "until")}})
    emit("action", f"liveness {kind} hold taken by {by} for {int(ttl_s)}s: {reason}"[:300],
         {"action": "liveness-hold-acquired", **{k: h[k] for k in ("kind", "by", "reason", "lease", "until", "mode")}})
    return {"lease": h["lease"], "kind": kind, "until": h["until"]}


def hold_release(lease=None, by=None):
    now = time.time()
    with _holds_locked():
        rows = _holds_read()
        gone = [h for h in rows if (lease and h.get("lease") == lease) or (not lease and by and h.get("by") == by)]
        _holds_write([h for h in rows if h not in gone and _hold_valid(h, now)])
    for h in gone:
        emit("action", f"liveness {h.get('kind')} hold released by {h.get('by')} after {round(now - float(h.get('acquired') or now))}s",
             {"action": "liveness-hold-released", **{k: h.get(k) for k in ("kind", "by", "reason", "lease")}})
    return {"released": len(gone)}


def hold_renew(lease, ttl_s):
    if not 60 <= int(ttl_s) <= HOLD_MAX_TTL_S:
        return {"refused": f"ttl_s must be 60..{HOLD_MAX_TTL_S}"}
    now = time.time()
    with _holds_locked():
        rows = _holds_read()
        for h in rows:
            if h.get("lease") == lease and _hold_valid(h, now):
                h["until"] = now + int(ttl_s)
                h["renewed"] = int(h.get("renewed") or 0) + 1
                _holds_write([r for r in rows if _hold_valid(r, now)])
                emit("action", f"liveness hold by {h.get('by')} renewed for {int(ttl_s)}s", {"action": "liveness-hold-renewed", "lease": lease, "until": h["until"]})
                return {"lease": lease, "until": h["until"]}
    return {"refused": "no such live hold (released, expired, or its holder died)"}


def hold_run(kind, by, reason, ttl_s, cmd, ensure_up=True):
    """Take a hold, run cmd, ALWAYS release, then make sure the engine is coming back (the old per-window 'ALWAYS a healthy engine
    at exit' trap, once, here). The hold is void the moment this wrapper dies, so a SIGKILLed window strands nothing."""
    got = hold_acquire(kind, by, reason, ttl_s, mode="run")
    if "lease" not in got:
        print(json.dumps(got), file=sys.stderr)
        return 75
    rc = 1
    child = None
    try:
        with terminate_as_exception():
            # the child (a window script) inherits the lease: its own `engine-actuator.py restart` calls pick it up from
            # ENGINE_HOLD_LEASE without knowing about holds, so wrapping a legacy window in `hold run` never breaks it
            env = dict(os.environ, **{HOLD_ENV: got["lease"], HOLD_ENV + "_KIND": kind})
            child = subprocess.Popen(cmd, env=env)
            rc = child.wait()
    except Terminated as t:
        rc = 128 + t.signum
        if child and child.poll() is None:
            with shielded():
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
    finally:
        with shielded():
            hold_release(got["lease"])
            if ensure_up and kind == "engine":
                print(json.dumps({"ensure_up": ensure_engine_up(by=f"{by} (hold run exit)")}), file=sys.stderr)
    return rc


# ------------------------------------------------------------------ observation
def _unit_facts():
    kv = {}
    for l in sh(["systemctl", "show", UNIT, "--timestamp=unix", "-p",
                 "ActiveState,SubState,MainPID,NRestarts,ExecMainStartTimestamp,ActiveEnterTimestamp,InactiveEnterTimestamp,Result"]).splitlines():
        if "=" in l:
            k, v = l.split("=", 1)
            kv[k] = v
    for k in ("ExecMainStartTimestamp", "ActiveEnterTimestamp", "InactiveEnterTimestamp"):
        v = kv.get(k, "")
        try:
            kv[k] = float(v[1:]) if v.startswith("@") else None
        except ValueError:
            kv[k] = None
    return kv


def _lock_held():
    try:
        fh = open(LOCK, "a+")
    except OSError:
        return False
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fh, fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        fh.close()


def _window_procs():
    """[(pid, elapsed_s, argv)] of running legacy window scripts (ps; empty when unreadable)."""
    out = []
    for l in sh(["ps", "-eo", "pid=,etimes=,args="], timeout=10).splitlines():
        parts = l.strip().split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit() and WINDOW_PROC_RE.match(parts[2]):
            out.append((int(parts[0]), int(parts[1]), parts[2][:160]))
    return out


def _gateway_offline():
    try:
        d = http(f"{GATEWAY}/gateway/offline", timeout=3)
        return {"offline": bool(d.get("offline")), "by": d.get("by"), "reason": d.get("reason"), "remaining_s": d.get("remaining_s")}
    except Exception:  # noqa: BLE001  gateway down: no window can be open on it
        return {"offline": False, "unreadable": True}


def _actions():
    rows, outcomes = [], {}
    try:
        for l in open(LACTIONS):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if r.get("outcome_of"):
                outcomes[r["outcome_of"]] = r
            else:
                rows.append(r)
    except OSError:
        pass
    for r in rows:
        if r.get("reset"):
            r["outcome"] = "reset"
            continue
        o = outcomes.get(r.get("id"))
        r["outcome"] = o.get("outcome") if o else "pending"
    return rows


def _actions_append(row):
    with open(LACTIONS, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def gate(now, actions=None):
    """The ONE rate limit + backoff + breaker over every automatic action. Returns facts incl. allowed and why."""
    allrows = [a for a in (_actions() if actions is None else actions) if not a.get("dry_run")]
    acts = [a for a in allrows if not a.get("reset")]
    hour = [a for a in acts if now - float(a.get("t") or 0) < 3600]
    fails = 0
    for a in reversed(allrows):
        if a.get("outcome") == "failed":
            fails += 1
        elif a.get("outcome") in ("ok", "reset"):
            break
        # pending: neither breaks nor extends the run
    last = acts[-1] if acts else None
    gap = min(BACKOFF_MAX_S, AUTO_MIN_GAP_S * (2 ** fails)) if fails else AUTO_MIN_GAP_S
    since_last = now - float(last["t"]) if last else None
    out = {"actions_last_hour": len(hour), "max_per_hour": AUTO_MAX_PER_HOUR, "consecutive_failed": fails, "gap_s": gap,
           "since_last_s": None if since_last is None else round(since_last), "breaker": "closed", "allowed": True, "why": ""}
    if fails >= BREAKER_FAILS:
        out["breaker"] = "open"
        out["next_try"] = float(last["t"]) + gap
        if since_last is not None and since_last >= gap:
            out["breaker"] = "half-open"          # one attempt; its outcome closes or re-opens with a longer gap
        else:
            out.update(allowed=False, why=f"breaker open after {fails} failed automatic actions; next try in {round(gap - since_last)}s")
            return out
    if last and last.get("outcome") == "pending" and since_last is not None and since_last < VERIFY_S:
        out.update(allowed=False, why=f"previous automatic action {last.get('id')} still being verified ({round(since_last)}s of {VERIFY_S}s)")
    elif len(hour) >= AUTO_MAX_PER_HOUR:
        out.update(allowed=False, why=f"{len(hour)} automatic actions in the last hour (max {AUTO_MAX_PER_HOUR})")
    elif since_last is not None and since_last < gap:
        out.update(allowed=False, why=f"min gap {gap}s since the last automatic action not reached ({round(since_last)}s)")
    return out


def _recent_fault_times(now):
    """Epoch times of FAULT ledger rows inside the crash-loop window (classify keeps only those after the last healthy moment)."""
    since = datetime.fromtimestamp(now - CRASH_LOOP_WINDOW_S).isoformat(timespec="seconds")
    out = []
    for r in ledger_rows(since=since):
        if r.get("kind") != "FAULT":
            continue
        try:
            out.append(datetime.fromisoformat(r["ts"]).timestamp())
        except (KeyError, ValueError, TypeError):
            pass
    return out


def observe(now=None):
    now = time.time() if now is None else now
    try:
        wd = json.load(open(WD_STATE))
    except (OSError, ValueError):
        wd = {}
    try:
        fq = os.path.getmtime(FQ_RUNNING)
        fq_age = now - fq if now - fq < FQ_RUNNING_MAX_S else None
    except OSError:
        fq_age = None
    try:
        job = json.load(open(JOB))
    except (OSError, ValueError):
        job = None
    return {"now": now, "paused": os.path.exists(LPAUSE), "unit": _unit_facts(), "healthy": engine_healthy(),
            "planned_lock": _lock_held(), "job": job, "holds": active_holds(now), "offline": _gateway_offline(),
            "frontier_window_age_s": fq_age, "window_procs": [w for w in _window_procs() if w[1] < HOLD_MAX_TTL_S],
            "watchdog_failures": int(wd.get("consecutive_failures") or 0),
            "fault_times": _recent_fault_times(now), "gate": gate(now)}


def implicit_holds(f):
    """Holds nobody took explicitly but that DO own the engine, in the order classify() honours them. ONE rule for all of them
    (documented in deploy/docs/engine-liveness-authority.md section 4):
      * they defer every AUTOMATIC action (tick start/recover, watchdog probes and its recover requests), exactly like an
        explicit engine hold;
      * they never refuse a PLANNED restart: their owner is the one calling `restart` (a `gateway-offline.py run` window or a
        legacy window script restarting its own engine), and a planned restart inside an open offline window rides that window
        instead of opening a fence. Only an explicit hold, which names its holder, can refuse a planned restart.
    Each is bounded: the gateway's own lease TTL, FQ_RUNNING_MAX_S, HOLD_MAX_TTL_S of process runtime."""
    out = []
    off = f.get("offline") or {}
    if off.get("offline"):
        out.append({"kind": "engine", "implicit": "gateway-offline-window", "by": off.get("by"), "reason": off.get("reason"),
                    "remaining_s": off.get("remaining_s"), "state": "OFFLINE_WINDOW"})
    if f.get("frontier_window_age_s") is not None:
        out.append({"kind": "engine", "implicit": "frontier-queue-running", "by": "frontier-queue",
                    "reason": f"legacy frontier-queue window running ({round(f['frontier_window_age_s'])}s; runner trusts it {FQ_RUNNING_MAX_S}s)",
                    "remaining_s": round(FQ_RUNNING_MAX_S - f["frontier_window_age_s"]), "state": "HELD"})
    for w in f.get("window_procs") or []:
        out.append({"kind": "engine", "implicit": "window-process", "by": f"pid {w[0]}",
                    "reason": f"legacy window process running without a hold (pid {w[0]}, {w[1]}s, bounded {HOLD_MAX_TTL_S}s): {w[2]}",
                    "remaining_s": HOLD_MAX_TTL_S - w[1], "state": "HELD"})
    return out


def classify(f, prev):
    """Pure: facts + previous published state -> (state, reason, facts-to-carry). Ordered: who owns the engine first, then health."""
    now, u = f["now"], f.get("unit") or {}
    prev = prev or {}
    boot_t = u.get("ExecMainStartTimestamp")
    carry = {"boot_started": boot_t, "last_healthy": prev.get("last_healthy")}
    if f["healthy"]:
        carry["last_healthy"] = now
    if f.get("paused"):
        return "PAUSED", "kill switch LIVENESS_PAUSE present: observing only", carry
    if f.get("planned_lock"):
        j = f.get("job") or {}
        return "PLANNED", f"planned restart in progress (by {j.get('by')}: {str(j.get('reason'))[:80]}; state {j.get('state')})", carry
    eh = [h for h in f.get("holds") or [] if h.get("kind") == "engine"]
    if eh:
        h = eh[0]
        return "HELD", f"engine held by {h.get('by')} until {datetime.fromtimestamp(h['until']).strftime('%H:%M:%S')}: {h.get('reason')}", carry
    imp = implicit_holds(f)
    if imp:
        h = imp[0]
        if h["implicit"] == "gateway-offline-window":
            return "OFFLINE_WINDOW", f"gateway planned-offline window by {h['by']} ({h['remaining_s']}s left): {h['reason']}", carry
        return h["state"], h["reason"], carry
    pend = [a for a in _actions_cached(f) if a.get("outcome") == "pending" and now - float(a.get("t") or 0) < VERIFY_S]
    if f["healthy"]:
        if f.get("watchdog_failures"):
            return "SUSPECT", f"API up but {f['watchdog_failures']} consecutive generation probe failure(s)", carry
        return "UP", "healthy", carry
    g = f.get("gate") or {}
    # faults since the engine was last healthy only: a window's own failed arms, followed by a healthy restore, do not count
    faults = [t for t in f.get("fault_times") or [] if t > float(carry.get("last_healthy") or 0)]
    f["faults_window"] = len(faults)
    if len(faults) >= CRASH_LOOP_FAULTS:
        return "CRASH_LOOP", f"{len(faults)} engine faults in the last {CRASH_LOOP_WINDOW_S}s since it was last healthy, and not healthy now", carry
    if g.get("breaker") == "open" and not g.get("allowed"):
        return "BREAKER_OPEN", g.get("why"), carry
    if pend:
        return "RECOVERING", f"automatic action {pend[-1].get('id')} ({pend[-1].get('action')}, cause {pend[-1].get('cause')}) awaiting /health", carry
    st = u.get("ActiveState", "")
    if u.get("SubState") == "auto-restart":
        return "BOOTING", "systemd is relaunching the engine after it exited (Restart=always)", carry
    if st == "deactivating":
        return "STOPPING", "unit stopping", carry
    if st in ("inactive", "failed", ""):
        since = u.get("InactiveEnterTimestamp") or prev.get("down_since") or now
        carry["down_since"] = since
        return "DOWN", f"unit {st or 'unknown'} for {round(now - since)}s (result {u.get('Result')})", carry
    # process exists (active, or activating while the warm-up hook runs) but /health does not answer
    age = now - boot_t if boot_t else 0
    if carry["last_healthy"] and boot_t and carry["last_healthy"] >= boot_t:
        down_for = now - carry["last_healthy"]
        if down_for >= UNRESPONSIVE_S:
            return "UNRESPONSIVE", f"API answered this boot but not for {round(down_for)}s (process alive)", carry
        return "SUSPECT", f"API not answering for {round(down_for)}s after being up this boot", carry
    if age >= BOOT_DEADLINE_S:
        return "STUCK_BOOT", f"process up {round(age)}s and /health never answered (deadline {BOOT_DEADLINE_S}s)", carry
    return "BOOTING", f"boot {round(age)}s old", carry


def _actions_cached(f):
    if "_actions" not in f:
        f["_actions"] = _actions()
    return f["_actions"]


def _verify_actions(f):
    """Judge pending automatic actions by their effect: /health back on a boot that started after the action = ok;
    still down after VERIFY_S = failed. This is what moves the breaker."""
    now = f["now"]
    for a in _actions_cached(f):
        if a.get("outcome") != "pending" or a.get("dry_run"):
            continue
        t = float(a.get("t") or 0)
        boot = (f.get("unit") or {}).get("ExecMainStartTimestamp") or 0
        verdict = None
        if f["healthy"] and boot >= t - 5:
            verdict = "ok"
        elif now - t >= VERIFY_S:
            verdict = "failed"
        if verdict:
            _actions_append({"outcome_of": a["id"], "outcome": verdict, "t": now, "after_s": round(now - t)})
            a["outcome"] = verdict
            emit("outcome", f"automatic engine {a.get('action')} ({a.get('cause')}) {verdict}: "
                 f"{'healthy' if verdict == 'ok' else 'still not healthy'} {round(now - t)}s later",
                 {"action": "liveness-action-verified", "id": a["id"], "outcome": verdict, "cause": a.get("cause")})


NEED_WHAT = {
    "CRASH_LOOP": "The local vLLM engine is crash-looping; automatic recovery cannot fix it. It needs a different configuration or a rollback.",
    "BREAKER_OPEN": "The local vLLM engine's automatic recovery failed repeatedly; the liveness breaker is open.",
}


def _raise_need(state, reason, f):
    """AU 2026-10-03: hand-offs have no consumer yet (spec 02), so a CRASH_LOOP / BREAKER_OPEN also files a NEED (kind decision ->
    Kevin, Discord DM + thread), deduped by signature while it stays open. Serving continues on the remote valve meanwhile (the
    gateway routes around an unhealthy local engine by itself). Best effort: never fails the tick."""
    try:
        if ESTATE not in sys.path:
            sys.path.insert(0, ESTATE)
        from tools import halo_needs
        r = halo_needs.raise_need(kind="decision", what=NEED_WHAT[state], why=reason[:600], raised_by="probe",
                                  evidence=[{"source": "engine-actuator liveness", "state": state, "reason": reason[:300],
                                             "gate": f.get("gate"), "faults_24h": faults_summary(24).get("by_signature")}],
                                  proposed_resolution="engine_restart with a known-good profile/flags (or roll back active-serve), "
                                                      "then `engine-actuator.py reset-breaker` if the breaker is open")
        return r.get("id") if isinstance(r, dict) else None
    except Exception as e:  # noqa: BLE001
        print(f"[engine-actuator] need filing failed: {e!r}", file=sys.stderr)
        return None


def _publish_state(state, reason, carry, f, extra=None):
    try:
        prev = json.load(open(LSTATE))
    except (OSError, ValueError):
        prev = {}
    since = prev.get("since") if prev.get("state") == state else f["now"]
    out = {"as_of": f["now"], "as_of_iso": now_iso(), "state": state, "since": since, "reason": reason,
           "declared": STATES[state], "probe": state not in NO_PROBE_STATES, **carry,
           "unit": {k: (f.get("unit") or {}).get(k) for k in ("ActiveState", "SubState", "MainPID", "NRestarts", "ExecMainStartTimestamp")},
           "healthy": f["healthy"], "holds": [{k: h.get(k) for k in ("kind", "by", "reason", "until", "mode")} for h in f.get("holds") or []],
           "implicit_holds": implicit_holds(f),
           "gate": f.get("gate"), "faults_window": f.get("faults_window"), "last_action": (_actions_cached(f) or [None])[-1]}
    if extra:
        out.update(extra)
    tmp = LSTATE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    os.replace(tmp, LSTATE)
    if prev.get("state") != state:
        emit("observation", f"engine liveness {prev.get('state') or '?'} -> {state}: {reason}"[:300],
             {"action": "liveness-state", "from": prev.get("state"), "to": state, "reason": reason,
              "owner": STATES[state]["owner"], "deadline_s": STATES[state]["deadline_s"]})
        if state in ("CRASH_LOOP", "BREAKER_OPEN"):
            emit("observation", f"engine {state}: {reason}. Automatic recovery is not fixing it; a planned restart with a different "
                 f"configuration, or a rollback, is the move.", {"state": state, "reason": reason, "gate": f.get("gate"),
                 "faults_24h": faults_summary(24)}, handoff=True, action=f"engine-{state.lower().replace('_', '-')}",
                 fingerprint=f"engine-{state}-{int(f['now'] // 3600)}")
            need = _raise_need(state, reason, f)
            if need:
                out["need"] = need
                with open(tmp, "w") as fh:
                    json.dump(out, fh, indent=1, default=str)
                os.replace(tmp, LSTATE)
    return out


def _act(action, cause, by, evidence, f, dry_run=False):
    """Run ONE automatic action under the shared LOCK. Never blocks on the engine boot (--no-block): the verdict is taken by a
    later tick from /health, so a 60 s watchdog oneshot can never be killed mid-recovery."""
    g = gate(f["now"], _actions_cached(f))
    if not g["allowed"]:
        return {"acted": False, "refused": g["why"], "gate": g}
    lk = open(LOCK, "a+")
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lk.close()
        return {"acted": False, "refused": "a planned restart or another automatic action holds restart.lock"}
    try:
        aid = f"lv-{int(f['now'])}-{os.getpid()}"
        row = {"id": aid, "t": f["now"], "action": action, "cause": cause, "by": by, "evidence": (evidence or "")[:300],
               "dry_run": bool(dry_run), "half_open": g.get("breaker") == "half-open"}
        if dry_run:
            return {"acted": False, "dry_run": True, "would": row}
        _actions_append(row)
        steps = []
        if action == "recover":
            if cause == "wedge":
                try:   # the collector reads this marker: a FAULT for Halo, not a planned stop
                    with open(f"{BASE}/wedge-restart.json", "w") as fh:
                        json.dump({"ts": now_iso(), "by": by, "wedge": True, "evidence": evidence}, fh)
                except OSError:
                    pass
                # RS: record the death BEFORE the kill (journal intact; ExecStopPost then dedupes)
                steps.append(("pre-kill", subprocess.run([sys.executable, f"{BASE}/engine-fault-collector.py", "--pre-kill"],
                                                         capture_output=True, text=True, timeout=60).returncode))
            steps.append(("reset-failed", _sudo(["reset-failed", UNIT])))
            # a confirmed wedge / stuck boot has never honoured SIGTERM (2026-09-02, 09-05): kill the control group up front
            steps.append(("kill", _sudo(["kill", "-s", "KILL", UNIT])))
            time.sleep(3)
            steps.append(("restart", _sudo(["restart", "--no-block", UNIT])))
        elif action == "start":
            steps.append(("reset-failed", _sudo(["reset-failed", UNIT])))
            steps.append(("start", _sudo(["start", "--no-block", UNIT])))
        row["steps"] = steps
        emit("action", f"liveness authority: automatic engine {action} (cause {cause}) by {by}: {evidence}"[:300],
             {"action": f"liveness-{action}", "id": aid, "cause": cause, "by": by, "evidence": evidence, "steps": steps, "gate": g})
        return {"acted": True, "id": aid, "action": action, "cause": cause, "steps": steps, "gate": g}
    finally:
        fcntl.flock(lk, fcntl.LOCK_UN)
        lk.close()


def _sudo(args):
    try:
        return subprocess.run(["sudo", "-n", "systemctl", *args], capture_output=True, text=True, timeout=60).returncode
    except Exception:  # noqa: BLE001
        return -1


def tick(by="liveness-tick", act=True):
    """One authority cycle: observe -> verify earlier actions -> classify -> publish -> take the declared exit, if it is ours."""
    f = observe()
    _verify_actions(f)
    f["gate"] = gate(f["now"], _actions_cached(f))
    try:
        prev = json.load(open(LSTATE))
    except (OSError, ValueError):
        prev = {}
    state, reason, carry = classify(f, prev)
    result = None
    if act and not f.get("paused"):
        if state == "DOWN" and f["now"] - float(carry.get("down_since") or f["now"]) >= DOWN_GRACE_S:
            result = _act("start", "down-unowned", by, reason, f)
        elif state in ("STUCK_BOOT", "UNRESPONSIVE"):
            result = _act("recover", state.lower(), by, reason, f)
        if result and result.get("acted"):
            f.pop("_actions", None)
            state, reason = "RECOVERING", f"automatic {result['action']} issued ({result['id']}) for: {reason}"
    if not f.get("planned_lock"):     # a job left non-terminal by a dead actuator: reconcile it (it can never finish on its own)
        j = f.get("job") or {}
        if j.get("state") in ("starting", "draining", "stopping", "starting-engine"):
            write_job(state="abandoned", finished=now_iso(), result={"abandoned": "restart.lock not held: the actuator that ran this job is gone"})
            emit("observation", f"planned restart job by {j.get('by')} abandoned (its actuator died in state {j.get('state')})",
                 {"action": "restart-abandoned", "job": j})
    out = _publish_state(state, reason, carry, f, {"tick_result": result} if result else None)
    return out


def ensure_engine_up(by):
    """Called when a holder lets go: if nothing else owns the engine and it is not running, start it now (not after DOWN_GRACE_S)."""
    f = observe()
    state, reason, _c = classify(f, {})
    if state == "DOWN":
        return _act("start", "hold-released", by, reason, f)
    return {"acted": False, "state": state, "reason": reason}


def recover(cause, by, evidence, dry_run=False):
    """The ONLY entry for an automatic kill/restart requested by a detector (vllm-watchdog.sh's confirmed generation wedge).
    The detector decides THAT the engine is wedged; the authority decides WHETHER acting is allowed now (holds, windows, planned
    restarts, rate limit, breaker) and does it the one way."""
    f = observe()
    _verify_actions(f)
    state, reason, _c = classify(f, {})
    if f.get("paused"):
        return {"acted": False, "refused": "LIVENESS_PAUSE kill switch present", "state": state}
    if state in ("PLANNED", "HELD", "OFFLINE_WINDOW"):
        out = {"acted": False, "refused": f"deferred: {reason}", "state": state}
        emit("observation", f"automatic engine recover ({cause}) by {by} deferred: {reason}"[:300],
             {"action": "liveness-recover-deferred", "cause": cause, "by": by, "evidence": evidence, "state": state})
        return out
    return _act("recover", cause, by, evidence, f, dry_run=dry_run)


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
    p.add_argument("--drain-max-s", type=int, default=600, help="hard cap (offline strategy only): keep waiting past --drain-s while token progress continues")
    p.add_argument("--no-drain", action="store_true"); p.add_argument("--foreground", action="store_true")
    p.add_argument("--force", action="store_true", help="restart even when the requested diag flags are already active")
    p.add_argument("--health-wait-s", type=int, default=HEALTH_WAIT_S, help="after start, wait this long for /health before calling the restart failed")
    p.add_argument("--hold", default=os.environ.get(HOLD_ENV) or None,
                   help=f"LV: the lease of the engine hold this caller owns (a window restarting its own engine); default ${HOLD_ENV}")
    p = sp.add_parser("announce-start"); p.add_argument("--wait-healthy", action="store_true", help=argparse.SUPPRESS)
    sp.add_parser("restart-status")
    # LV: the liveness authority
    p = sp.add_parser("tick", help="one authority cycle (vllm-watchdog.sh runs it every minute)"); p.add_argument("--by", default="liveness-tick")
    p.add_argument("--no-act", action="store_true", help="observe + publish only")
    p = sp.add_parser("recover", help="a detector asks for an automatic kill+restart")
    p.add_argument("--cause", required=True); p.add_argument("--by", required=True); p.add_argument("--evidence", default="")
    p.add_argument("--dry-run", action="store_true")
    sp.add_parser("liveness", help="print the published liveness state")
    p = sp.add_parser("reset-breaker"); p.add_argument("--by", required=True); p.add_argument("--reason", required=True)
    p = sp.add_parser("hold", help="TTL-bounded holds: engine (a window owns the engine) | quiesce (a release is in flight)")
    hs = p.add_subparsers(dest="hcmd", required=True)
    q = hs.add_parser("acquire"); q.add_argument("--kind", default="engine", choices=HOLD_KINDS); q.add_argument("--by", required=True)
    q.add_argument("--reason", required=True); q.add_argument("--ttl", type=int, required=True)
    q.add_argument("--owner-pid", type=int, default=None, help="the long-lived process this hold belongs to: void when it dies")
    q = hs.add_parser("release"); q.add_argument("--lease"); q.add_argument("--by")
    q = hs.add_parser("renew"); q.add_argument("--lease", required=True); q.add_argument("--ttl", type=int, required=True)
    q = hs.add_parser("run"); q.add_argument("--kind", default="engine", choices=HOLD_KINDS); q.add_argument("--by", required=True)
    q.add_argument("--reason", required=True); q.add_argument("--ttl", type=int, required=True)
    q.add_argument("--no-ensure-up", action="store_true", help="do not start the engine when the command ends")
    q.add_argument("argv", nargs=argparse.REMAINDER)
    hs.add_parser("status")
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
    elif a.cmd == "tick":
        print(json.dumps(tick(by=a.by, act=not a.no_act), default=str))
    elif a.cmd == "recover":
        r = recover(a.cause, a.by, a.evidence, dry_run=a.dry_run)
        print(json.dumps(r, default=str))
        return 0 if r.get("acted") or r.get("dry_run") else 3
    elif a.cmd == "liveness":
        print(open(LSTATE).read() if os.path.exists(LSTATE) else "{}")
    elif a.cmd == "reset-breaker":
        _actions_append({"id": f"reset-{int(time.time())}", "t": time.time(), "reset": True, "by": a.by, "reason": a.reason})
        emit("action", f"liveness breaker reset by {a.by}: {a.reason}"[:300], {"action": "liveness-breaker-reset", "by": a.by, "reason": a.reason})
        print(json.dumps({"reset": True}))
    elif a.cmd == "hold":
        if a.hcmd == "acquire":
            r = hold_acquire(a.kind, a.by, a.reason, a.ttl, **({"mode": "owned", "pid": a.owner_pid} if a.owner_pid else {}))
        elif a.hcmd == "release":
            if not (a.lease or a.by):
                print(json.dumps({"refused": "pass --lease or --by"})); return 2
            r = hold_release(a.lease, a.by)
        elif a.hcmd == "renew":
            r = hold_renew(a.lease, a.ttl)
        elif a.hcmd == "run":
            argv = a.argv[1:] if a.argv[:1] == ["--"] else a.argv
            if not argv:
                print(json.dumps({"refused": "nothing to run (hold run ... -- CMD ARGS)"})); return 2
            return hold_run(a.kind, a.by, a.reason, a.ttl, argv, ensure_up=not a.no_ensure_up)
        else:
            f = observe()
            r = {"holds": active_holds(), "implicit_holds": implicit_holds(f), "max_ttl_s": HOLD_MAX_TTL_S,
                 "rule": "explicit holds defer automatic actions AND refuse others' planned restarts; implicit holds defer "
                         "automatic actions only (their owner is the restart caller)"}
        print(json.dumps(r, default=str))
        return 3 if r.get("refused") else 0
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
