#!/usr/bin/env python3
"""remote_pricing_check.py -- the weekly, REPORT-ONLY cron entry point for the remote pricing table.

Lane FX2 (2026-10-03). Wraps remote_pricing_refresh.py. It never changes a price: a price/window/alias change on the
provider's page becomes (1) a fact file for Halo, (2) an idempotent estate hand-off (Halo's inbox), (3) a vault note
(`Memory/Remote Pricing Drift.md`, sole author = this job). Applying is a human/Halo decision:
`remote_pricing_refresh.py --apply`. The ONE thing it writes to the table is, with --restamp and only when the page
matches the table exactly (zero changes), the `fetched` date -- "verified against the page on"; the gateway flags the
table stale after 30 days, so without it a correct table would be reported stale every month. No price is touched.

    remote_pricing_check.py --file ~/.local/share/vllm-qwen27b/remote-pricing.json --restamp

Exit: 0 clean, 3 drift (or calendar warning), 2 the page/table could not be read. Kill switch:
touch ~/.local/share/vllm-qwen27b/REMOTE_PRICING_CHECK_OFF  (or remove the cron line).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")
BASE = os.environ.get("REMOTE_PRICING_BASE") or f"{HOME}/.local/share/vllm-qwen27b"
ESTATE = f"{HOME}/.local/share/estate-control"
VAULT_NOTE = os.environ.get("REMOTE_PRICING_NOTE") or f"{HOME}/Obsidian/Memory/Remote Pricing Drift.md"


def _load_refresh():
    spec = importlib.util.spec_from_file_location("remote_pricing_refresh", os.path.join(HERE, "remote_pricing_refresh.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_json_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def fingerprint(result):
    basis = json.dumps({"changes": result.get("changes") or [], "warnings": result.get("warnings") or [],
                        "error": result.get("error")}, sort_keys=True)
    return "remote-pricing-" + hashlib.sha1(basis.encode()).hexdigest()[:12]


def update_note(path, result, today):
    """Sole-authored vault note: latest state on top, one dated history line per distinct problem. Direct write is
    allowed for a note only this job authors; it is never a hot/shared note."""
    problem = bool(result.get("changes") or result.get("warnings") or result.get("error"))
    history = []
    try:
        for line in open(path, encoding="utf-8").read().splitlines():
            if line.startswith("- 20") and line not in history:
                history.append(line)
    except OSError:
        pass
    if problem:
        what = ("ERROR: " + result["error"]) if result.get("error") else "; ".join((result.get("changes") or []) + (result.get("warnings") or []))
        history.append(f"- {today}: {what[:600]}")
    state = "ATTENTION" if problem else "clean"
    body = (f"---\nname: Remote Pricing Drift\ndescription: weekly report-only check of the gateway's remote pricing table against DeepSeek's official page; "
            f"state {state} as of {today}\nmetadata:\n  type: reference\nmanaged_by: remote-pricing-check\n---\n"
            f"# Remote Pricing Drift\n\nWritten by `deploy/bin/remote_pricing_check.py` (weekly cron on HNET00, see [[Automation Registry]]). "
            f"It never changes a price. To accept a drift run `remote_pricing_refresh.py --apply` on HNET00 (backup kept; the gateway hot-reloads).\n\n"
            f"**Latest check ({today}):** {state}"
            + (f" -- {result.get('error') or '; '.join((result.get('changes') or []) + (result.get('warnings') or []))}" if problem else "")
            + "\n\n## History (problems only)\n" + ("\n".join(history) if history else "- none yet") + "\n")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    open(tmp, "w", encoding="utf-8").write(body)
    os.replace(tmp, path)


def emit_handoff(result):
    """Idempotent hand-off into Halo's inbox (same mechanism engine-actuator.py uses). Best effort; never fails the job."""
    try:
        if ESTATE not in sys.path:
            sys.path.insert(0, ESTATE)
        from tools import handoff as _h
        what = result.get("error") or "; ".join((result.get("changes") or []) + (result.get("warnings") or []))
        r = _h.emit("pricing", "remote-pricing-table", "pricing-drift",
                    f"remote pricing table needs attention: {what[:300]}", result, fingerprint=fingerprint(result),
                    subject_type="component", hold=False)
        return (r or {}).get("seq")
    except Exception as e:  # noqa: BLE001
        print(f"[remote-pricing-check] handoff failed: {e!r}", file=sys.stderr)
        return None


def run(file, url=None, html=None, restamp=False, today=None, notify=True, base=None, note=None):
    rp = _load_refresh()
    base = base or BASE
    note = note or VAULT_NOTE
    today = today or datetime.date.today()
    try:
        markup = open(html, encoding="utf-8").read() if html else rp.fetch(url or rp.DEFAULT_URL)
        result = rp.refresh(file, markup, url or rp.DEFAULT_URL, today, apply=False)       # check-only, always
        if restamp and not result["drift"]:
            result = rp.refresh(file, markup, url or rp.DEFAULT_URL, today, apply=True)    # zero changes: date stamp only
            result["restamped"] = bool(result.get("applied"))
    except (rp.PageError, OSError, ValueError) as exc:
        result = {"fact": "remote_pricing_table", "checked_at": time.time(), "error": f"{type(exc).__name__}: {exc}",
                  "drift": False, "changes": [], "warnings": [], "applied": False}
    problem = bool(result.get("changes") or result.get("warnings") or result.get("error"))
    result["report_only"] = True
    write_json_atomic(f"{base}/facts/remote_pricing_table.json", result)
    with open(f"{base}/remote-pricing-check.log", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "drift": result["drift"],
                             "changes": result.get("changes"), "warnings": result.get("warnings"), "error": result.get("error"),
                             "restamped": result.get("restamped", False)}) + "\n")
    if problem and notify:
        result["handoff_seq"] = emit_handoff(result)
    if notify and (problem or os.path.exists(note)):       # a clean run refreshes the note only if one exists (clears ATTENTION)
        update_note(note, result, today.isoformat())
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--file", default=f"{BASE}/remote-pricing.json")
    ap.add_argument("--url"); ap.add_argument("--html")
    ap.add_argument("--restamp", action="store_true", help="when the page matches the table exactly, re-stamp `fetched` (date only)")
    ap.add_argument("--no-notify", action="store_true")
    a = ap.parse_args(argv)
    if os.path.exists(f"{BASE}/REMOTE_PRICING_CHECK_OFF"):
        print(json.dumps({"skipped": "REMOTE_PRICING_CHECK_OFF present"}))
        return 0
    res = run(os.path.expanduser(a.file), a.url, a.html, a.restamp, notify=not a.no_notify)
    print(json.dumps({k: res.get(k) for k in ("drift", "changes", "warnings", "error", "restamped", "handoff_seq")}))
    return 2 if res.get("error") else (3 if res.get("changes") or res.get("warnings") else 0)


if __name__ == "__main__":
    sys.exit(main())
