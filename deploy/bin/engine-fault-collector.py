#!/usr/bin/env python3
"""engine-fault-collector.py -- classify + archive + ledger EVERY unplanned vllm-qwen27b death.

Why (2026-10-02): of the 3 engine faults on 2026-10-01 (14:40 Xid31, 17:03 scheduler ValueError, 20:45 Xid31)
only the Xid ones were archived (the watchdog keys on kernel Xid lines), the 17:03 one left no incident
folder; nothing classified the faults, counted them, linked them to the requests in flight, or told the
estate. systemd restarts the unit (Restart=always) so service came back -- but nobody *learned* anything.

Wired as ExecStopPost on vllm-qwen27b.service (runs on every stop incl. exit-0 crashes) and runnable by hand:
  engine-fault-collector.py --stop-post            # from systemd
  engine-fault-collector.py --at "2026-10-01 17:03:30"   # backfill / replay a past moment
Deterministic, no LLM. Never fails the unit (exit 0). Kill switch: touch ~/.local/share/vllm-qwen27b/NO_FAULT_COLLECTOR
Writes: incidents/fault-<ts>/ , incidents/ledger.jsonl , and (append) the vault note Memory/Engine Fault Ledger.md
"""
import argparse, glob, json, os, re, shutil, subprocess, sys, time
from datetime import datetime, timezone

HOME = os.path.expanduser("~")
BASE = os.environ.get("FAULT_COLLECTOR_BASE") or f"{HOME}/.local/share/vllm-qwen27b"
INC = f"{BASE}/incidents"
LEDGER = f"{INC}/ledger.jsonl"
VAULT_NOTE = f"{HOME}/Obsidian/Memory/Engine Fault Ledger.md"
UNIT = "vllm-qwen27b"
NOISE = re.compile(r"tokenize|/metrics|/health|/v1/models|qwen3xml|api/show|SpecDecoding|Avg prompt")


def sh(cmd, timeout=40):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout
    except Exception as e:  # noqa: BLE001
        return f"<{e!r}>"


def classify(journal: str, kernel: str):
    sig, detail = "unknown-exit", ""
    if "repeats may not contain negative values" in journal:
        sig = "sched-negative-num-scheduled-tokens"
    elif re.search(r"illegal memory access|cudaErrorIllegalAddress", journal) or re.search(r"Xid.*\b31\b", kernel):
        sig = "cuda-illegal-address"
        m = re.search(r"EF-FENCE first observed CUDA fault at ([^:]+):", journal)
        if m:
            detail = "fence=" + m.group(1)
        else:
            head = journal.split("AcceleratorError", 1)[0]  # frames of the FIRST fault report, not shutdown cleanup
            frames = re.findall(r'File "[^"]*/(vllm/[^"]+)", line (\d+), in (\w+)', head)
            if frames:
                detail = "last-frame=%s:%s:%s" % frames[-1]
    elif re.search(r"Xid.*\b13\b", kernel):
        sig = "xid13-sm-exception"
    elif re.search(r"out of memory|OutOfMemory|CUDA OOM", journal, re.I):
        sig = "oom"
    elif "EngineDeadError" in journal or "WorkerProc hit an exception" in journal:
        sig = "engine-dead-other"
    return sig, detail


def scheduler_dump_shape(journal: str):
    """Summarise the last 'Dumping scheduler output' line: rows, prefill rows, spec rows."""
    lines = [l for l in journal.splitlines() if "Dumping scheduler output" in l]
    if not lines:
        return {}
    l = lines[-1]
    out = {}
    m = re.search(r"num_scheduled_tokens=\{([^}]*)\}", l)
    if m:
        vals = [int(x.split(":")[-1]) for x in m.group(1).split(",") if ":" in x]
        out["rows"] = len(vals)
        out["scheduled"] = vals
        out["prefill_rows"] = sum(1 for v in vals if v > 8)
        out["negative_rows"] = sum(1 for v in vals if v < 0)
    sp = re.search(r"scheduled_spec_decode_tokens=\{(.*?)\}, scheduled_encoder_inputs", l)
    out["spec_rows"] = len(re.findall(r"chatcmpl-[0-9a-f\-]+:", sp.group(1))) if sp else 0
    out["mixed_prefill_and_spec"] = out.get("prefill_rows", 0) > 0 and out["spec_rows"] > 0
    return out


def inflight(ts: float):
    """Gateway telemetry rows whose [start,end] straddles ts (requests alive at the fault)."""
    rows = []
    for fn in sorted(glob.glob(f"{BASE}/telemetry/requests-*.jsonl"))[-2:]:
        try:
            for l in open(fn):
                try:
                    d = json.loads(l)
                except Exception:  # noqa: BLE001
                    continue
                end = d.get("t", 0)
                start = end - (d.get("duration") or 0)
                if start <= ts <= end + 2 and "local" in str(d.get("route")):
                    rows.append({"client": d.get("client"), "ptok": d.get("ptok"), "bg": d.get("bg"),
                                 "age_s": round(ts - start, 1), "ttft": d.get("ttft")})
        except Exception:  # noqa: BLE001
            pass
    return rows


def uptime_before(ts: float):
    s = sh(f"systemctl show {UNIT} -p ExecMainStartTimestampMonotonic,ActiveEnterTimestamp --value 2>/dev/null")
    m = re.search(r"ActiveEnterTimestamp=(.+)", sh(f"systemctl show {UNIT} -p ActiveEnterTimestamp"))
    try:
        t = datetime.strptime(m.group(1).strip().split(" MST")[0].split(" ", 1)[1], "%Y-%m-%d %H:%M:%S").timestamp()
        return round(ts - t)
    except Exception:  # noqa: BLE001
        return None


def _recently_recorded(window_s=240):
    try:
        d = json.load(open(f"{BASE}/recorded-fault.json"))
        fresh = time.time() - float(d["t"]) < window_s
        os.rename(f"{BASE}/recorded-fault.json", f"{BASE}/recorded-fault.last.json")
        return fresh
    except Exception:  # noqa: BLE001
        return False


def nearest_xid_dir(ts: float, before_s=900, after_s=60):
    """RS: the kernel-Xid archive (incidents/xid-*) written by the Xid archiver nearest BEFORE this death, so one record links both."""
    best = None
    for d in glob.glob(f"{INC}/xid-*"):
        try:
            t = datetime.strptime(os.path.basename(d)[4:], "%Y%m%d-%H%M%S").timestamp()
        except Exception:  # noqa: BLE001
            continue
        if -after_s <= ts - t <= before_s and (best is None or t > best[0]):
            best = (t, d)
    return best[1] if best else None


def emit_event(row, a):
    """RS: every recorded death is also one estate event_log record (source=engine), so Halo's catch-up and the incident
    timeline see it even when it is not a hand-off (planned stops, backfills)."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("engine_actuator", f"{BASE}/engine-actuator.py")
        ea = importlib.util.module_from_spec(spec); spec.loader.exec_module(ea)
        if row["kind"] == "FAULT" and not (a.reconciled or a.at):
            return  # the live FAULT path emits its hand-off (handoff_to_halo) itself
        kind = "observation"
        summ = (f"engine death recorded ({row['kind']} {row['signature']}{' ' + row['detail'] if row.get('detail') else ''})"
                f"{' [backfilled]' if row.get('reconciled') else ''} at {row['ts']}"
                f"{'; cause: ' + row['cause'] if row.get('cause') else ''}")
        ea.emit(kind, summ, {"action": "engine-death", **{k: row.get(k) for k in
                ("kind", "signature", "detail", "uptime_s", "xid_incident", "incident_dir", "wedge", "planned_by",
                 "planned_reason", "in_flight", "reconciled")}})
    except Exception as e:  # noqa: BLE001
        print("emit_event error", repr(e), file=sys.stderr)


def _ledger_ts():
    out = []
    try:
        for l in open(LEDGER):
            try:
                out.append(datetime.fromisoformat(json.loads(l)["ts"]).timestamp())
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    return out


def reconcile(hours: float):
    """RS: any unit death in the journal with no ledger row within +-240 s is backfilled. A death is the systemd line
    'Main process exited' (covers SIGKILL, crash, exit-0). Cause is attributed from what the journal itself shows: a watchdog
    wedge decision (watchdog.log) or the sudo 'systemctl kill|stop|restart' line. Idempotent; run from engine-actuator
    announce-start (every engine start) so a lost record is healed within one restart, with no new cron."""
    since = datetime.fromtimestamp(time.time() - hours * 3600).strftime("%Y-%m-%d %H:%M:%S")
    jr = sh(f'journalctl -u {UNIT} -o short-iso --no-pager --since "{since}" | grep -E "Main process exited|Stopping vLLM"', timeout=60)
    have = _ledger_ts()
    wd = ""
    try:
        wd = open(f"{BASE}/watchdog.log", errors="replace").read()[-400000:]
    except Exception:  # noqa: BLE001
        pass
    wedge_times = []
    for x in wd.splitlines():
        if "DECISION wedge CONFIRMED" in x and "RESTARTING" in x and "[DRY-RUN]" not in x:
            try:  # watchdog.log stamps are UTC ("...Z")
                wedge_times.append(datetime.strptime(x[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
            except Exception:  # noqa: BLE001
                pass
    done = []
    for l in jr.splitlines():
        m = re.match(r"(\S+) \S+ systemd\[1\]: .*Main process exited, code=(\w+), status=(\S+)", l)
        if not m:
            continue
        t = datetime.fromisoformat(m.group(1)).timestamp()
        if any(abs(t - h) <= 240 for h in have):
            continue
        stamp = datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
        near = sh(f'journalctl --since "{datetime.fromtimestamp(t-20).strftime("%Y-%m-%d %H:%M:%S")}" '
                  f'--until "{datetime.fromtimestamp(t+5).strftime("%Y-%m-%d %H:%M:%S")}" --no-pager -o cat 2>/dev/null | grep -E "COMMAND=.*(systemctl|vllm)" | head -3')
        wedge = any(abs(wt - t) <= 90 for wt in wedge_times)
        cause = ("watchdog confirmed generation wedge -> systemctl kill -s KILL" if wedge else
                 ("; ".join(x.strip()[:140] for x in near.splitlines()) or f"code={m.group(2)} status={m.group(3)}; actor not in journal"))
        cmd = [sys.executable, os.path.abspath(__file__), "--at", stamp, "--reconciled", "--no-vault", "--cause", cause]
        if wedge:
            cmd.append("--wedge")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        done.append({"at": stamp, "cause": cause, "rc": r.returncode})
        have.append(t)
    print(json.dumps({"reconciled": done}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop-post", action="store_true")
    ap.add_argument("--at", help="fault time 'YYYY-mm-dd HH:MM:SS' (local) for backfill")
    ap.add_argument("--window", type=int, default=240, help="journal seconds before the fault to read")
    ap.add_argument("--no-vault", action="store_true")
    ap.add_argument("--drill", action="store_true", help="with --at: replay a past fault AND emit its hand-off (tagged drill)")
    ap.add_argument("--pre-kill", action="store_true",
                    help="RS: called by the watchdog BEFORE it kills a wedged engine. Records the death now (ledger + incident + event) "
                         "so it cannot be lost to the kill->restart race; the later --stop-post then skips (dedupe marker)")
    ap.add_argument("--reconcile", action="store_true",
                    help="RS: find engine deaths in the journal that have no ledger row and backfill them")
    ap.add_argument("--hours", type=float, default=6.0, help="with --reconcile: how far back to look")
    ap.add_argument("--wedge", action="store_true", help="with --at: the death was a watchdog-confirmed generation wedge")
    ap.add_argument("--cause", default="", help="with --at: free-text cause recorded on the row")
    ap.add_argument("--reconciled", action="store_true", help="internal: row is a backfill (emit an event, not a hand-off)")
    a = ap.parse_args()
    if os.path.exists(f"{BASE}/NO_FAULT_COLLECTOR"):
        return
    if a.reconcile:
        return reconcile(a.hours)
    if a.stop_post and _recently_recorded():
        # RS: the watchdog already recorded this death via --pre-kill; ExecStopPost must not double-count it.
        for src in ("wedge-restart.json", "planned-restart.json"):
            try:
                os.rename(f"{BASE}/{src}", f"{BASE}/{src[:-5]}.last.json")
            except Exception:  # noqa: BLE001
                pass
        return
    now = datetime.strptime(a.at, "%Y-%m-%d %H:%M:%S").timestamp() if a.at else time.time()
    since = datetime.fromtimestamp(now - a.window).strftime("%Y-%m-%d %H:%M:%S")
    until = datetime.fromtimestamp(now + 45).strftime("%Y-%m-%d %H:%M:%S")
    journal = sh(f'journalctl -u {UNIT} --no-pager --since "{since}" --until "{until}"')
    kernel = sh(f'journalctl -k --no-pager -o short-iso --since "{since}" --until "{until}" | grep -i "xid\\|NVRM"')
    result = os.environ.get("SERVICE_RESULT", "") or "backfill"
    sig, detail = classify(journal, kernel)
    planned = (sig == "unknown-exit" and not a.at and result in ("success", "") and "Traceback" not in journal)
    # EF2: a stop that SOMEONE REQUESTED (systemctl stop/restart logs "Stopping <unit description>") is planned even when it
    # ended in a SIGKILL after the stop timeout -- 03:45 and 03:55 on 2026-10-02 were EF's own restarts mislabelled FAULT.
    marker = None
    try:
        if a.at:
            raise FileNotFoundError  # RS: a backfill must never consume the LIVE markers (it ate the 12:05 wedge marker)
        marker = json.load(open(f"{BASE}/planned-restart.json"))
        os.rename(f"{BASE}/planned-restart.json", f"{BASE}/planned-restart.last.json")
    except Exception:  # noqa: BLE001
        pass
    requested_stop = bool(re.search(r"systemd\[1\]: Stopping vLLM", journal)) and sig in ("unknown-exit",) and not a.at
    killed_after_timeout = result == "timeout" and (os.environ.get("EXIT_STATUS") in ("KILL", "9"))
    wedge = None
    try:
        if a.at:
            raise FileNotFoundError
        wedge = json.load(open(f"{BASE}/wedge-restart.json"))
        os.rename(f"{BASE}/wedge-restart.json", f"{BASE}/wedge-restart.last.json")
    except Exception:  # noqa: BLE001
        pass
    if a.wedge and a.at:
        wedge = {"by": "watchdog", "wedge": True, "backfilled": True}
    if a.pre_kill and not wedge:
        wedge = {"by": "watchdog", "wedge": True}
    if wedge:
        sig, detail, planned = "generation-wedge", "watchdog-confirmed (models healthy, generation probes timed out)", False
    elif marker or (requested_stop and "Traceback" not in journal):
        planned = True
    if a.reconciled and sig == "unknown-exit" and not wedge and "COMMAND=" in a.cause and re.search(r"systemctl (stop|restart)", a.cause):
        planned = True  # RS backfill: the journal shows someone ran systemctl stop/restart
    kind = "planned-stop" if planned else "FAULT"
    ts_s = datetime.fromtimestamp(now).strftime("%Y%m%d-%H%M%S")
    shape = scheduler_dump_shape(journal)
    live = inflight(now)
    inc = f"{INC}/fault-{ts_s}"
    row = {"ts": datetime.fromtimestamp(now).isoformat(timespec="seconds"), "kind": kind, "signature": sig,
           "detail": detail, "service_result": result, "exit_status": os.environ.get("EXIT_STATUS"),
           "uptime_s": uptime_before(now) if not a.at else None, "dump": shape,
           "in_flight": {"n": len(live), "ptok": [r["ptok"] for r in live], "clients": sorted({str(r["client"]) for r in live})},
           "incident_dir": inc if kind == "FAULT" else None}
    xid = nearest_xid_dir(now)
    if xid:
        row["xid_incident"] = xid
    if a.cause:
        row["cause"] = a.cause
    if a.reconciled:
        row["reconciled"] = True
    if a.pre_kill:
        row["recorded_by"] = "watchdog-pre-kill"
    if wedge:
        row["wedge"] = {k: wedge.get(k) for k in ("by", "consecutive_failures", "ts")}
    if planned:
        row["planned_by"] = (marker or {}).get("by") or "systemctl"
        row["planned_reason"] = (marker or {}).get("reason")
        row["killed_after_stop_timeout"] = killed_after_timeout  # in-flight work was cut at the stop timeout (drain didn't finish)
        row["drain"] = (marker or {}).get("drain")
    os.makedirs(INC, exist_ok=True)
    if kind == "FAULT":
        os.makedirs(f"{inc}/flightrec", exist_ok=True)
        keep = [l for l in journal.splitlines() if not NOISE.search(l)]
        open(f"{inc}/engine-journal.txt", "w").write("\n".join(keep[-2500:]))
        open(f"{inc}/kernel.txt", "w").write(kernel)
        open(f"{inc}/in-flight.json", "w").write(json.dumps(live, indent=1))
        open(f"{inc}/META.json", "w").write(json.dumps(row, indent=1))
        for p in sorted(glob.glob(f"{BASE}/flightrec/*.json"))[-30:]:
            try:
                shutil.copy2(p, f"{inc}/flightrec/")
            except Exception:  # noqa: BLE001
                pass
    if not a.drill:
        with open(LEDGER, "a") as f:
            f.write(json.dumps(row) + "\n")
        if a.pre_kill:
            try:
                json.dump({"t": time.time(), "ts": row["ts"], "sig": sig}, open(f"{BASE}/recorded-fault.json", "w"))
            except Exception:  # noqa: BLE001
                pass
        emit_event(row, a)
    if kind == "FAULT" and not a.no_vault:
        try:
            if not os.path.exists(VAULT_NOTE):
                return
            mix = "MIXED prefill+spec" if shape.get("mixed_prefill_and_spec") else "not-mixed"
            line = (f"| {row['ts']} | {sig} {detail} | {mix} | up {row['uptime_s']}s | in-flight {len(live)} "
                    f"(ptok {row['in_flight']['ptok']}) | `{inc}` |\n")
            with open(VAULT_NOTE, "a") as f:
                f.write(line)
        except Exception:  # noqa: BLE001
            pass
    if kind == "FAULT" and (not a.at or a.drill) and not a.reconciled:
        handoff_to_halo(row, sig, detail, shape, live, inc, drill=a.drill)
    print(json.dumps(row))


def handoff_to_halo(row, sig, detail, shape, live, inc, drill=False):
    """Fault -> Halo hand-off (EF2). FACTS only; deciding what to do is Halo's (no thresholds here).
    Goes through the estate's hand-off queue (idempotent per distinct fault) so it appears in list_handoffs and the
    gap-free event log; Halo's catch-up sees it at the start of its next cycle."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("engine_actuator", f"{BASE}/engine-actuator.py")
        ea = importlib.util.module_from_spec(spec); spec.loader.exec_module(ea)
        recent = ea.faults_summary(24)
        facts = {"action": "engine-fault", "signature": sig, "detail": detail, "service_result": row.get("service_result"),
                 "exit_status": row.get("exit_status"), "uptime_s": row.get("uptime_s"),
                 "batch_shape": shape, "requests_in_flight": row["in_flight"], "incident_dir": inc,
                 "faults_in_last_24h": recent["faults"], "by_signature_24h": recent["by_signature"],
                 "diag_flags_staged": ea.staged_flags(),
                 "engine_state": "systemd restarts the unit automatically (Restart=always, 15 s); warm-up follows",
                 "actuators_you_have": ["engine_status", "engine_faults", "engine_flags", "engine_stage_diag", "engine_restart"],
                 "evidence": [f"{inc}/engine-journal.txt", f"{inc}/META.json", f"{inc}/in-flight.json", "vault: Memory/Engine Fault Ledger.md"]}
        if drill:
            facts["drill"] = True
        summ = ((" [DRILL: replay of a past fault] " if drill else "") + f"vLLM engine died ({sig}{' ' + detail if detail else ''}) after {row.get('uptime_s')}s up; "
                f"batch rows={shape.get('rows')} prefill_rows={shape.get('prefill_rows')} spec_rows={shape.get('spec_rows')}; "
                f"{row['in_flight']['n']} request(s) in flight (ptok {row['in_flight']['ptok']}); "
                f"{recent['faults']} fault(s) in the last 24 h")
        r = ea.emit("handoff", summ, facts, handoff=True, action="engine-fault", fingerprint=f"{'drill-' if drill else ''}fault-{row['ts']}-{sig}")
        row["handoff_seq"] = r
    except Exception as e:  # noqa: BLE001
        print("handoff error", repr(e), file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print("collector error", repr(e), file=sys.stderr)
    sys.exit(0)
