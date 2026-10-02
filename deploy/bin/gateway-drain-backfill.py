#!/usr/bin/env python3
"""Reconstruct gateway drain records that pre-date the live drain ledger (lane RS, 2026-10-02).

Source: the gateway's own access log in the journal (POST /gateway/drain = open, DELETE = close, gateway restart = close,
/v1/chat/completions 503 inside the window = refused). Reason/holder are INFERRED (engine planned-stop ledger row within 20 min
after the open => that row's reason); rows are tagged backfilled=true. Idempotent: a drain whose open time is within 5 s of an
existing ledger open is skipped.   usage: gateway-drain-backfill.py [--since "2026-10-02 00:00"] [--dry]
"""
import argparse, json, os, re, subprocess
from datetime import datetime

H = os.path.expanduser("~/.local/share/vllm-qwen27b/incidents")
ap = argparse.ArgumentParser(); ap.add_argument("--since", default="2026-10-02 00:00"); ap.add_argument("--dry", action="store_true")
a = ap.parse_args()
out = subprocess.run(["journalctl", "-u", "vllm-keepalive-shim", "--since", a.since, "--no-pager", "-o", "short-iso"], capture_output=True, text=True).stdout
ts_re = re.compile(r"^(\S+) \S+ ")
T = lambda l: datetime.fromisoformat(ts_re.match(l).group(1)).timestamp()
def rows(path):
    try: return [json.loads(l) for l in open(path) if l.strip()]
    except OSError: return []
have = [r["t"] for r in rows(f"{H}/drains.jsonl") if r.get("event") == "open"]
planned = [(datetime.fromisoformat(r["ts"]).timestamp(), r) for r in rows(f"{H}/ledger.jsonl") if r.get("kind") == "planned-stop"]
drains, cur = [], None
for l in out.splitlines():
    if not ts_re.match(l): continue
    if '"POST /gateway/drain' in l and '" 200 ' in l:
        ua = re.search(r'"-" "([^"]*)"', l)
        cur = {"t": T(l), "refused": 0, "ua": ua.group(1) if ua else "?"}
    elif cur and (('"DELETE /gateway/drain' in l and '" 200 ' in l) or "systemd[1]: Started vllm-keepalive-shim" in l):
        cur.update(close=T(l), how="delete" if "DELETE" in l else "gateway-restart"); drains.append(cur); cur = None
    elif cur and "/v1/chat/completions" in l and '" 503 ' in l:
        cur["refused"] += 1
if cur: drains.append(dict(cur, close=None, how="still-open"))
new = []
prev = None
for d in drains:
    if any(abs(d["t"] - h) < 5 for h in have): continue
    reason, by = "unknown (pre-ledger; inferred from access log)", d["ua"]
    near = [r for t, r in planned if 0 <= t - d["t"] <= 1200]
    if near:
        r = near[0]; reason = "engine planned restart: %s" % (r.get("planned_reason") or "")[:90]; by = r.get("planned_by") or by
    elif prev and abs(d["t"] - prev["close"]) <= 10 and prev["reason"].startswith("engine planned restart"):
        reason = "benchmark/quiesce fence re-raised right after: " + prev["reason"][:110] + " (inferred)"
    elif d["how"] == "gateway-restart":
        reason, by = "governed gateway publish (inferred: drain ended by a gateway restart)", d["ua"]
    new.append({"event": "open", "t": round(d["t"], 3), "reason": reason, "by": by, "backfilled": True})
    d["reason"] = reason; prev = d if d.get("close") else None
    if d.get("close"):
        new.append({"event": "close", "how": d["how"], "t": round(d["close"], 3), "t0": round(d["t"], 3), "duration_s": round(d["close"] - d["t"], 1),
                    "reason": reason, "by": by, "refused": d["refused"], "backfilled": True})
new.sort(key=lambda r: r["t"])
if a.dry:
    print(json.dumps(new, indent=1)[:3000]); print(len(new), "rows (dry)")
else:
    with open(f"{H}/drains.jsonl", "a") as fh:
        for r in new: fh.write(json.dumps(r, sort_keys=True) + "\n")
    print("appended", len(new), "rows")
