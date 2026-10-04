#!/usr/bin/env python3
"""live_ab_metrics.py -- live-traffic A/B metrics for a windowctl `live` step (lane EC, 2026-10-03).

A lane window used to measure with synthetic probes while the gateway was OFFLINE (no estate traffic on the engine). A
cache/scheduler change is judged by the estate's own traffic, so the candidate must serve it. This module reads what the
gateway and the engine already record, for any interval:

  * prefix-cache hit rate: delta(hits)/delta(queries) of telemetry/prefixcache.jsonl (cumulative engine counters, one row
    per minute; a counter reset = an engine restart, handled);
  * TTFT p50/p90 and the uncached-prefill share of route=local rows of telemetry/requests-*.jsonl;
  * the local failure rate (route=local, status >= 500).

CLI:
  live_ab_metrics.py summary --start T --end T                 # one interval (epoch seconds)
  live_ab_metrics.py compare --base-start T --base-end T --cand-start T --cand-end T [--warmup-s 480] [--out F]
      exit 0 = KEEP (hit rate up or equal, TTFT p90 not worse by > 10 %, zero faults, enough data), 1 = ROLL BACK.

Decision rule (Kevin 2026-10-03): keep iff hit_rate_cand >= hit_rate_base - 0.005 (0.5 point = "equal"), ttft_p90_cand <=
1.10 x ttft_p90_base, faults == 0, and the candidate has >= --min-n successful local rows. The candidate is measured AFTER
--warmup-s (a restart empties the prefix cache; the baseline is a warm engine); the whole-window numbers are reported too.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

TELEMETRY = os.environ.get("LIVE_AB_TELEMETRY", os.path.expanduser("~/.local/share/vllm-qwen27b/telemetry"))
_T_RE = re.compile(r'"t":\s*([0-9.]+)')


def pct(vals, q):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def read_rows(pattern, start, end, telemetry=None):
    """Rows (dicts) with start <= t < end from the telemetry files matching `pattern` (cheap pre-filter on the t field)."""
    telemetry = telemetry or TELEMETRY
    out = []
    for path in sorted(glob.glob(os.path.join(telemetry, pattern))):
        try:
            if os.path.getmtime(path) < start - 120:
                continue
        except OSError:
            continue
        try:
            with open(path, "r", errors="replace") as fh:
                for line in fh:
                    m = _T_RE.search(line[:80])
                    if not m:
                        continue
                    t = float(m.group(1))
                    if t < start or t >= end:
                        continue
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
        except OSError:
            continue
    return out


def request_stats(rows):
    local = [r for r in rows if r.get("route") == "local"]
    ok = [r for r in local if r.get("status") == 200]
    ttft = [float(r["ttft"]) for r in ok if r.get("ttft") is not None]
    ptok = comp = 0
    for r in ok:
        p = r.get("ptok_exact") or r.get("ptok")
        c = r.get("computed_actual")
        if c is None:
            c = r.get("est_computed")
        if p and c is not None:
            ptok += p
            comp += min(c, p)
    fails = [r for r in local if isinstance(r.get("status"), int) and r["status"] >= 500]
    return {"local_n": len(local), "ok_n": len(ok), "ttft_n": len(ttft), "ttft_p50": pct(ttft, 0.5), "ttft_p90": pct(ttft, 0.9),
            "prompt_tokens": ptok, "uncached_tokens": comp, "uncached_share": (comp / ptok) if ptok else None,
            "local_fail_n": len(fails), "local_fail_rate": (len(fails) / len(local)) if local else None}


def prefix_stats(start, end, telemetry=None):
    """Hit rate over [start, end) from the cumulative counters: sum of per-interval deltas (a decrease = engine restart:
    that row's delta is dropped, the counters restart from it)."""
    rows = [r for r in read_rows("prefixcache.jsonl", start - 90, end + 90, telemetry) if r.get("queries") is not None]
    rows.sort(key=lambda r: r["t"])
    q = h = 0
    prev = None
    n = 0
    for r in rows:
        if prev is not None and start <= r["t"] < end + 1e-9:
            dq, dh = r["queries"] - prev["queries"], r["hits"] - prev["hits"]
            if dq >= 0 and dh >= 0:
                q += dq
                h += dh
                n += 1
        prev = r
    return {"queries": q, "hits": h, "hit_rate": (h / q) if q else None, "samples": n}


def interval(start, end, telemetry=None):
    s = request_stats(read_rows("requests-*.jsonl", start, end, telemetry))
    s.update(prefix_stats(start, end, telemetry))
    s.update(start=start, end=end, minutes=round((end - start) / 60, 1))
    return s


def decide(base, cand, faults=0, min_n=60, hit_tol=0.005, ttft_slack=0.10):
    g = {}
    g["data"] = (cand["ok_n"] >= min_n and cand["queries"] > 0 and base["queries"] > 0 and base["ttft_p90"] is not None
                 and cand["ttft_p90"] is not None)
    g["hit_rate_not_lower"] = bool(g["data"] and cand["hit_rate"] >= base["hit_rate"] - hit_tol)
    g["ttft_p90_within_10pct"] = bool(g["data"] and cand["ttft_p90"] <= base["ttft_p90"] * (1 + ttft_slack))
    g["zero_faults"] = faults == 0
    return all(g.values()), g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("summary")
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--end", type=float, required=True)
    p = sp.add_parser("compare")
    for k in ("base-start", "base-end", "cand-start", "cand-end"):
        p.add_argument("--" + k, type=float, required=True)
    p.add_argument("--warmup-s", type=float, default=480)
    p.add_argument("--faults", type=int, default=0)
    p.add_argument("--min-n", type=int, default=60)
    p.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "summary":
        print(json.dumps(interval(a.start, a.end), indent=1))
        return 0
    base = interval(a.base_start, a.base_end)
    full = interval(a.cand_start, a.cand_end)
    steady = interval(min(a.cand_start + a.warmup_s, a.cand_end), a.cand_end)
    keep, gates = decide(base, steady, a.faults, a.min_n)
    out = {"keep": keep, "gates": gates, "baseline": base, "candidate_steady": steady, "candidate_full": full,
           "warmup_s": a.warmup_s, "faults": a.faults,
           "rule": "hit_rate_cand_steady >= base - 0.005, ttft_p90_cand_steady <= 1.10 x base, 0 faults, >= min_n ok rows"}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)

    def f(x, nd=3):
        return "n/a" if x is None else f"{x:.{nd}f}"
    for name, s in (("baseline", base), ("cand steady", steady), ("cand full", full)):
        print(f"{name:12s} min={s['minutes']:5.1f} ok_n={s['ok_n']:5d} hit_rate={f(s['hit_rate'])} ttft_p50={f(s['ttft_p50'], 2)} "
              f"ttft_p90={f(s['ttft_p90'], 2)} uncached_share={f(s['uncached_share'])} fail_rate={f(s['local_fail_rate'])}")
    print("VERDICT", "KEEP" if keep else "ROLL BACK", json.dumps(gates))
    return 0 if keep else 1


if __name__ == "__main__":
    sys.exit(main())
