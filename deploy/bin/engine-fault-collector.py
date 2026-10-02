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
from datetime import datetime

HOME = os.path.expanduser("~")
BASE = f"{HOME}/.local/share/vllm-qwen27b"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop-post", action="store_true")
    ap.add_argument("--at", help="fault time 'YYYY-mm-dd HH:MM:SS' (local) for backfill")
    ap.add_argument("--window", type=int, default=240, help="journal seconds before the fault to read")
    ap.add_argument("--no-vault", action="store_true")
    a = ap.parse_args()
    if os.path.exists(f"{BASE}/NO_FAULT_COLLECTOR"):
        return
    now = datetime.strptime(a.at, "%Y-%m-%d %H:%M:%S").timestamp() if a.at else time.time()
    since = datetime.fromtimestamp(now - a.window).strftime("%Y-%m-%d %H:%M:%S")
    until = datetime.fromtimestamp(now + 45).strftime("%Y-%m-%d %H:%M:%S")
    journal = sh(f'journalctl -u {UNIT} --no-pager --since "{since}" --until "{until}"')
    kernel = sh(f'journalctl -k --no-pager -o short-iso --since "{since}" --until "{until}" | grep -i "xid\\|NVRM"')
    result = os.environ.get("SERVICE_RESULT", "") or "backfill"
    sig, detail = classify(journal, kernel)
    planned = (sig == "unknown-exit" and not a.at and result in ("success", "") and "Traceback" not in journal)
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
    with open(LEDGER, "a") as f:
        f.write(json.dumps(row) + "\n")
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
    print(json.dumps(row))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print("collector error", repr(e), file=sys.stderr)
    sys.exit(0)
