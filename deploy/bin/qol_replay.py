#!/usr/bin/env python3
"""Replay the QoL interactive-overflow rule over the gateway's request log (lane CFG, 2026-10-03). Read-only.

For every INTERACTIVE request in the window it reconstructs what the live predictor would have seen at arrival:
  own     its uncached prefill (est_computed; ptok - engine-anchored credit when logged) at the prefill rate measured
          from recent local rows (p75 of uncached tokens / first-token seconds over 20K+ token prompts, last 30 min)
  backlog uncached prefill of local requests (any class) still in progress at that moment, at the same rate
          (halo capped at its flow ceiling, like local_first_predicted_ttft)
  queue   lane term: median recent local service time / budget when every interactive lane was busy
and the remote TTFT for its prompt-size band (QOL_REMOTE_QUANTILE of streamed remote TTFT, last 6 h, >= 5 samples).
Then applies keepalive-shim.qol_choice() and compares with what happened.

Reports: predicted vs actual local first-token accuracy, how many interactive requests QoL would overflow vs today,
interactive TTFT p50/p90 before/after (switched requests use MODELED values, marked), remote spend before/after.
Governed = interactive, not estate-local / local-pin, and either served local or sent remote for a reason QoL governs
(perf, big-prompt, monster, predicted, or the lane wait: cap/tokens/prefill). Windows, forced remote, aliases, size,
failover and big-out are reported but not changed.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import heapq
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TELEMETRY = Path(os.environ.get("SHIM_TELEMETRY_DIR", "/home/kevin/.local/share/vllm-qwen27b/telemetry"))
GOVERNED_REMOTE = {"perf", "big-prompt", "monster", "predicted", "cap", "tokens", "prefill"}
LOCAL_OK = lambda reason: reason in ("-", "", None) or str(reason).startswith(("lf-", "tiny"))
FIELDS = ("t", "duration", "route", "reason", "bg", "stream", "ttft", "waited", "est_computed", "ptok", "ptok_exact",
          "pm_credit_anchored", "alias", "xclient", "flow_class", "charged_usd", "cost_est", "status")


def _shim():
    os.environ.setdefault("SHIM_EXACT_TOKENS", "0")
    spec = importlib.util.spec_from_file_location("keepalive_shim_qol_replay", HERE / "keepalive-shim.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load(days, now=None):
    now = now or time.time()
    cutoff = now - days * 86400
    rows = []
    for f in sorted(TELEMETRY.glob("requests-*.jsonl")):
        try:
            day = time.mktime(time.strptime(f.stem.split("-")[1], "%Y%m%d"))
        except ValueError:
            continue
        if day < cutoff - 2 * 86400:
            continue
        with f.open() as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if (r.get("t") or 0) < cutoff - 3600 or r.get("duration") is None:
                    continue
                rows.append({k: r.get(k) for k in FIELDS})
    rows.sort(key=lambda r: r["t"] - (r["duration"] or 0))
    return [r for r in rows], cutoff


def pct(v, q):
    if not v:
        return None
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def replay(rows, cutoff, shim, *, ttft_target=15.0, min_gain=5.0, rate_default=1100.0, budget=14,
           remote_q=0.75, remote_window=21600, remote_min=5):
    band = shim.qol_band
    # rolling measured inputs, keyed by finish time
    rate_t, rate_v = [], []                    # local prefill rate samples
    remote = collections.defaultdict(lambda: ([], []))   # band -> (t list, ttft list)
    service_t, service_v = [], []
    for r in sorted(rows, key=lambda r: r["t"]):
        est = r["est_computed"] or r["ptok"] or 0
        if r["route"] == "local" and r["ttft"] and est >= 20000 and (r["waited"] or 0) < 0.5:
            rate_t.append(r["t"]); rate_v.append(est / max(0.05, r["ttft"]))
        if r["route"] == "local" and r["duration"]:
            service_t.append(r["t"]); service_v.append(r["duration"])
        if r["route"] == "remote" and r["ttft"] and (r["status"] or 200) < 400:
            tl, vl = remote[band(r["ptok_exact"] or r["ptok"] or 0)]
            tl.append(r["t"]); vl.append(r["ttft"])

    def window(ts, vs, a, span):
        i, j = bisect.bisect_left(ts, a - span), bisect.bisect_left(ts, a)
        return vs[i:j]

    def rate_at(a):
        w = window(rate_t, rate_v, a, 1800)
        return pct(w, 0.75) if len(w) >= 5 else rate_default

    def remote_at(a, ptok):
        tl, vl = remote[band(ptok)]
        w = window(tl, vl, a, remote_window)
        return pct(w, remote_q) if len(w) >= remote_min else None

    # sweep: local prefill intervals and lane occupancy, in arrival order
    starts = sorted(((r["t"] - r["duration"]) + (r["waited"] or 0), i) for i, r in enumerate(rows) if r["route"] == "local")
    prefill_heap, lane_heap = [], []
    si = 0
    out = []
    for r in rows:
        a = r["t"] - r["duration"]
        while si < len(starts) and starts[si][0] <= a:
            p0, j = starts[si]
            rj = rows[j]
            estj = rj["est_computed"] or rj["ptok"] or 0
            rr = rate_at(p0)
            p1 = p0 + (rj["ttft"] if rj["ttft"] else estj / rr)
            heapq.heappush(prefill_heap, (p1, p0, estj))
            heapq.heappush(lane_heap, rj["t"])
            si += 1
        while prefill_heap and prefill_heap[0][0] <= a:
            heapq.heappop(prefill_heap)
        while lane_heap and lane_heap[0] <= a:
            heapq.heappop(lane_heap)
        if r["bg"] or r["t"] < cutoff:
            continue
        if r["alias"] == "estate-local" or "local-pin" in (r["xclient"] or ""):
            continue
        rate = rate_at(a)
        backlog = sum(est * max(0.0, (p1 - a)) / max(1e-3, p1 - p0) for p1, p0, est in prefill_heap) / rate
        if (r["flow_class"] or "kevin") == "halo":
            backlog = min(backlog, 60.0)
        sv = window(service_t, service_v, a, 1800)
        queue = (statistics.median(sv) / budget) if (len(lane_heap) >= budget and sv) else 0.0
        own_tok = r["est_computed"] or r["ptok"] or 0
        if r["pm_credit_anchored"]:
            own_tok = max(0, (r["ptok"] or 0) - r["pm_credit_anchored"])
        pred = queue + backlog + own_tok / rate
        ptok = r["ptok_exact"] or r["ptok"] or 0
        rem = remote_at(a, ptok)
        would, why = shim.qol_choice(pred, 0.0, rem, ttft_target_s=ttft_target, min_gain_s=min_gain)
        reason = r["reason"]
        governed = (r["route"] == "local" and LOCAL_OK(reason)) or (r["route"] == "remote" and reason in GOVERNED_REMOTE)
        actual_ttft = None
        if r["ttft"] is not None:
            actual_ttft = (r["waited"] or 0) + r["ttft"]
        cost = r["charged_usd"] if r["charged_usd"] is not None else (r["cost_est"] or 0.0)
        out.append(dict(a=a, route=r["route"], reason=reason, governed=governed, pred=pred, remote=rem, would=would,
                        why=why, actual_ttft=actual_ttft, ptok=ptok, band=band(ptok), cost=cost if r["route"] == "remote" else 0.0,
                        own_s=own_tok / rate, backlog_s=backlog, queue_s=queue, anchored=bool(r["pm_credit_anchored"])))
    return out


def summarize(res, ttft_target=15.0):
    gov = [x for x in res if x["governed"]]
    loc = [x for x in gov if x["route"] == "local" and x["actual_ttft"] is not None]
    err = [x["pred"] - x["actual_ttft"] for x in loc]
    aerr = sorted(abs(e) for e in err)
    tp = sum(1 for x in loc if x["pred"] > ttft_target and x["actual_ttft"] > ttft_target)
    fp = sum(1 for x in loc if x["pred"] > ttft_target and x["actual_ttft"] <= ttft_target)
    fn = sum(1 for x in loc if x["pred"] <= ttft_target and x["actual_ttft"] > ttft_target)
    by_band = {}
    for b in sorted({x["band"] for x in loc}):
        e = [x["pred"] - x["actual_ttft"] for x in loc if x["band"] == b]
        by_band[b] = dict(n=len(e), median_err_s=round(statistics.median(e), 1), p90_abs_err_s=round(pct([abs(v) for v in e], 0.9), 1))
    accuracy = dict(n=len(loc), median_signed_err_s=round(statistics.median(err), 2) if err else None,
                    mae_s=round(sum(aerr) / len(aerr), 2) if aerr else None, p50_abs_err_s=pct(aerr, 0.5), p90_abs_err_s=pct(aerr, 0.9),
                    within_5s=round(sum(1 for e in aerr if e <= 5) / len(aerr), 3) if aerr else None,
                    late_threshold=dict(true_pos=tp, false_pos=fp, false_neg=fn,
                                        precision=round(tp / (tp + fp), 3) if tp + fp else None,
                                        recall=round(tp / (tp + fn), 3) if tp + fn else None),
                    by_band=by_band)
    bias = statistics.median(err) if err else 0.0
    today_over = [x for x in gov if x["route"] == "remote"]
    qol_over = [x for x in gov if (x["would"] == "remote") or (x["would"] == "legacy" and x["route"] == "remote")]
    for x in res:
        x["switch"] = None
        if x["governed"] and x["route"] == "local" and x["would"] == "remote":
            x["switch"] = "to_remote"
        elif x["governed"] and x["route"] == "remote" and x["would"] == "local":
            x["switch"] = "to_local"
    to_remote = [x for x in gov if x["switch"] == "to_remote"]
    to_local = [x for x in gov if x["switch"] == "to_local"]
    # TTFT before/after over governed rows with a measured TTFT; switched rows use modeled values
    before, after, modeled = [], [], 0
    for x in gov:
        if x["actual_ttft"] is None:
            continue
        before.append(x["actual_ttft"])
        if x["switch"] == "to_remote":
            after.append(x["remote"]); modeled += 1
        elif x["switch"] == "to_local":
            after.append(max(0.0, x["pred"] - bias)); modeled += 1
        else:
            after.append(x["actual_ttft"])
    # spend: per-band remote $/token from rows that were remote and charged
    per_tok = collections.defaultdict(list)
    for x in res:
        if x["route"] == "remote" and x["cost"] and x["ptok"]:
            per_tok[x["band"]].append(x["cost"] / x["ptok"])
    add = sum(statistics.median(per_tok[x["band"]]) * x["ptok"] for x in to_remote if per_tok.get(x["band"]))
    saved = sum(x["cost"] for x in to_local)
    all_int = [x for x in res]
    spend_all = sum(x["cost"] for x in all_int)
    spend_gov = sum(x["cost"] for x in today_over)
    days = collections.Counter(time.strftime("%m-%d", time.localtime(x["a"])) for x in today_over)
    days_q = collections.Counter(time.strftime("%m-%d", time.localtime(x["a"])) for x in qol_over)
    reasons_today = collections.Counter(x["reason"] for x in today_over)
    whys = collections.Counter(x["why"] for x in gov)
    not_gov = collections.Counter((x["route"], x["reason"]) for x in res if not x["governed"])
    return dict(
        interactive_requests=len(res), governed=len(gov), not_governed={"%s:%s" % k: v for k, v in not_gov.most_common(12)},
        accuracy=accuracy,
        overflow=dict(today=len(today_over), qol=len(qol_over), today_by_reason=dict(reasons_today),
                      local_to_remote=len(to_remote), remote_to_local=len(to_local), qol_why=dict(whys),
                      per_day_today=dict(sorted(days.items())), per_day_qol=dict(sorted(days_q.items()))),
        ttft=dict(n=len(before), modeled=modeled,
                  before_p50=pct(before, 0.5), before_p90=pct(before, 0.9), after_p50=pct(after, 0.5), after_p90=pct(after, 0.9)),
        spend=dict(interactive_remote_usd=round(spend_all, 2), governed_remote_usd=round(spend_gov, 2),
                   qol_saved_usd=round(saved, 2), qol_added_usd_est=round(add, 2),
                   governed_after_usd=round(spend_gov - saved + add, 2),
                   remote_share_interactive_before=round(sum(1 for x in res if x["route"] == "remote") / max(1, len(res)), 3),
                   remote_share_interactive_after=round((sum(1 for x in res if x["route"] == "remote") - len(to_local) + len(to_remote))
                                                         / max(1, len(res)), 3)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=float, default=7)
    ap.add_argument("--ttft", type=float, default=15.0)
    ap.add_argument("--gain", type=float, default=5.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    shim = _shim()
    rows, cutoff = load(a.days)
    res = replay(rows, cutoff, shim, ttft_target=a.ttft, min_gain=a.gain)
    s = summarize(res, a.ttft)
    s["window"] = dict(days=a.days, from_=time.strftime("%Y-%m-%d %H:%M", time.localtime(cutoff)), rows_loaded=len(rows),
                       ttft_target_s=a.ttft, min_gain_s=a.gain)
    print(json.dumps(s, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
