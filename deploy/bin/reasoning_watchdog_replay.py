#!/usr/bin/env python3
"""Size the L95 reasoning watchdog (SHIM_REASONING_WATCHDOG) from the gateway's own request log. Read-only.

For each candidate budget (tokens of reasoning streamed with no content and no tool call yet):
  * true positives  = local streamed rows that ended EMPTY (no content, no tool call) and whose reasoning
                      reached the budget -- the turns the watchdog exists to rescue; saved_s = decode time past
                      the budget that retry would not have spent.
  * false positives = rows that DID produce output, but only after >= budget reasoning tokens (retry would have
                      thrown that reasoning away). Exact from `reasoning_chars_at_output` (logged since GW2);
                      for older rows only an UPPER bound exists (outtok >= budget, labelled fp_upper).
  * chars/token is calibrated from rows whose whole output was reasoning (empty, outtok exact).

  python3 reasoning_watchdog_replay.py [--days 7] [--budgets 4096,8192,12288] [--client pi] [--json]
"""
import argparse
import glob
import json
import os
import time

TDIR = os.environ.get("SHIM_TELEMETRY_DIR", os.path.expanduser("~/.local/share/vllm-qwen27b/telemetry"))


def rows(days, client=None, tdir=TDIR):
    cutoff = time.time() - days * 86400
    for f in sorted(glob.glob(os.path.join(tdir, "requests-*.jsonl"))):
        with open(f, errors="replace") as fh:
            for ln in fh:
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                if (r.get("t") or 0) < cutoff or r.get("route") != "local" or not r.get("stream"):
                    continue
                if (r.get("status") or 200) >= 400 or r.get("content_empty") is None:
                    continue
                if client and client not in str(r.get("client")):
                    continue
                yield r


def replay(rs, budgets, cpt=None):
    rs = list(rs)
    empty = [r for r in rs if r.get("content_empty") and not r.get("has_tool_calls")]
    cal = [(r["reasoning_chars"], r["outtok"]) for r in empty if r.get("reasoning_chars") and r.get("outtok")]
    measured_cpt = round(sum(c for c, _ in cal) / max(1, sum(o for _, o in cal)), 3) if cal else None
    cpt = cpt or measured_cpt or 4.0
    out = {"rows": len(rs), "empty_no_tool": len(empty), "chars_per_token": cpt,
           "chars_per_token_measured": measured_cpt, "calibration_rows": len(cal), "budgets": {}}
    eids = {id(r) for r in empty}
    for b in budgets:
        tp = [r for r in empty if (r.get("outtok") or 0) >= b]
        exact_fp = exact_n = fp_upper = 0
        for r in rs:
            if id(r) in eids:
                continue
            if r.get("reasoning_chars_at_output") is not None:
                exact_n += 1
                exact_fp += (r["reasoning_chars_at_output"] / cpt) >= b
            elif (r.get("outtok") or 0) >= b:
                fp_upper += 1
        saved = sum((r.get("duration") or 0) * (1 - b / r["outtok"]) for r in tp)
        shadow = sum(1 for r in rs if r.get("reasoning_watchdog") and (r.get("reasoning_watchdog_at_tok") or 0) >= b)
        out["budgets"][b] = {"tp": len(tp), "tp_saved_s": round(saved), "fp_exact": exact_fp,
                             "fp_exact_of_rows": exact_n, "fp_upper_legacy": fp_upper, "shadow_triggers": shadow}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=7)
    ap.add_argument("--budgets", default="2048,4096,6144,8192,12288,16000")
    ap.add_argument("--client")
    ap.add_argument("--cpt", type=float)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    res = replay(rows(a.days, a.client), [int(x) for x in a.budgets.split(",")], a.cpt)
    if a.json:
        print(json.dumps(res, indent=1))
        return
    print("rows=%d empty_no_tool=%d chars/token=%s (measured %s from %d rows)" % (
        res["rows"], res["empty_no_tool"], res["chars_per_token"], res["chars_per_token_measured"],
        res["calibration_rows"]))
    print("%8s %4s %10s %9s %10s %9s %7s" % ("budget", "tp", "tp_saved_s", "fp_exact", "exact_rows", "fp_upper", "shadow"))
    for b, v in res["budgets"].items():
        print("%8d %4d %10d %9d %10d %9d %7d" % (b, v["tp"], v["tp_saved_s"], v["fp_exact"], v["fp_exact_of_rows"],
                                               v["fp_upper_legacy"], v["shadow_triggers"]))


if __name__ == "__main__":
    main()
