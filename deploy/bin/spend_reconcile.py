#!/usr/bin/env python3
"""spend_reconcile.py -- reconcile the gateway's remote spend with the provider's own bill.

R2 v5 (2026-09-25). Kevin's $25/day cap is REAL dollars. The provider bill (DeepSeek
"amount" export: per hour, per token type, with price) is the ground truth; the gateway's
request telemetry (requests-YYYYMMDD.jsonl) is what the spend authority replays.

    spend_reconcile.py --billing amount-2026-09-25_2026-09-25.csv \
        --telemetry ~/.local/share/vllm-qwen27b/telemetry \
        --start 2026-09-25T00:00:00-07:00 --end 2026-09-25T13:00:00-07:00 [--now ...]
        [--post http://127.0.0.1:8000 --admin-token-file ~/.local/share/vllm-qwen27b/admin.token]

It reports, for the same window:
  * billing: requests, cache-hit / cache-miss / output tokens, dollars (amount x price);
  * telemetry replay: the gateway's cache-aware cost for rows that carry provider usage
    (cost_basis actual/usage), and how many rows only carry the old no-cache estimate;
  * the verdict: the replay must match the bill within --tolerance (default 2%) whenever
    every row in the window carries provider usage;
  * the BEST FIGURE for today so far: the bill for the billed window, plus the telemetry after
    the bill's end priced at the bill's own effective $/token (for rows without cache fields).
With --post it hands that best figure to POST /gateway/spend/recover as an operator REPLACE,
compare-and-set bound to the exact ledger revision it read, with nothing in flight and the
telemetry flushed past the last settle; a conflict recomputes (v6). The admin token is read
from a file, never from argv or the environment dump.
The billing CSV is only read; nothing from it (user id, key prefix) is written anywhere.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.request

TYPES = ("input_cache_hit_tokens", "input_cache_miss_tokens", "output_tokens")


def _ts(value: str) -> float:
    return datetime.datetime.fromisoformat(value).timestamp()


def billing_totals(rows, start: float, end: float) -> dict:
    """Sum an amount export (dict rows) over [start, end). Hour rows are counted when they
    START inside the window."""
    tot = {t: 0.0 for t in TYPES}
    cost, requests = 0.0, 0.0
    for r in rows:
        if not (start <= _ts(r["start_time_iso"]) < end):
            continue
        amount = float(r.get("amount") or 0)
        kind = r.get("type")
        if kind == "request_count":
            requests += amount
        elif kind in tot:
            tot[kind] += amount
            cost += amount * float(r.get("price") or 0)
    return {"requests": int(requests), "cache_hit": int(tot["input_cache_hit_tokens"]),
            "cache_miss": int(tot["input_cache_miss_tokens"]), "output": int(tot["output_tokens"]),
            "usd": round(cost, 6)}


def read_billing(paths) -> list:
    rows = []
    for p in paths:
        with open(p, encoding="utf-8-sig", newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def _counted(r: dict, provider: str) -> bool:
    """The spend authority's rule for one telemetry row (v6): rows since v6 carry
    remote_sent / cost_policy / remote_provider -- a paid row is one that reached a metered
    provider, at any HTTP status; older rows keep the old rule (status < 400, and only the
    default provider, i.e. not a custom alias)."""
    if r.get("route") != "remote":
        return False
    if "remote_sent" in r:
        return bool(r.get("remote_sent")) and r.get("cost_policy") != "free" and \
            (r.get("remote_provider") or provider) == provider
    status = r.get("status")
    return r.get("alias_kind") != "custom-remote" and (status is None or int(status) < 400)


def telemetry_rows(tdir: str, start: float, end: float, provider: str = "deepseek") -> list:
    out = []
    for path in sorted(glob.glob(os.path.join(tdir, "requests-*.jsonl"))):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if start <= float(r.get("t") or 0) < end and _counted(r, provider):
                    out.append(r)
    return out


def replay(rows, prices) -> dict:
    """Price telemetry rows. `prices` = $/token (cache_hit, cache_miss, output)."""
    hit_p, miss_p, out_p = prices
    usd = 0.0
    n_usage = n_est = 0
    tok = {"cache_hit": 0, "cache_miss": 0, "output": 0, "est_prompt": 0, "est_output": 0}
    for r in rows:
        out = r.get("outtok") if r.get("outtok") is not None else (r.get("outtok_lb") or 0)
        if r.get("remote_cache_hit") is not None and r.get("remote_cache_miss") is not None:
            usd += r["remote_cache_hit"] * hit_p + r["remote_cache_miss"] * miss_p + out * out_p
            tok["cache_hit"] += r["remote_cache_hit"]
            tok["cache_miss"] += r["remote_cache_miss"]
            tok["output"] += out
            n_usage += 1
        else:
            n_est += 1
            tok["est_prompt"] += int(r.get("ptok_exact") or r.get("ptok") or 0)
            tok["est_output"] += int(out or 0)
    return {"requests": len(rows), "with_usage": n_usage, "estimate_only": n_est,
            "usd_with_usage": round(usd, 6), "tokens": tok}


def best_figure(bill: dict, bill_prices: tuple, gap_rows: list) -> dict:
    """Bill for the billed window + the unbilled gap priced at the bill's effective rates."""
    hit_p, miss_p, out_p = bill_prices
    prompt = bill["cache_hit"] + bill["cache_miss"]
    per_prompt = ((bill["cache_hit"] * hit_p + bill["cache_miss"] * miss_p) / prompt) if prompt else miss_p
    gap = replay(gap_rows, bill_prices)
    gap_usd = gap["usd_with_usage"] + gap["tokens"]["est_prompt"] * per_prompt + gap["tokens"]["est_output"] * out_p
    return {"billed_usd": bill["usd"], "gap_usd": round(gap_usd, 6), "gap_requests": gap["requests"],
            "effective_prompt_usd_per_mtok": round(per_prompt * 1e6, 6),
            "best_usd": round(bill["usd"] + gap_usd, 6)}


def bill_prices(rows) -> tuple:
    """$/token per type as billed (the latest price seen for each type)."""
    p = {}
    for r in rows:
        if r.get("type") in TYPES and r.get("price"):
            p[r["type"]] = float(r["price"])
    return (p.get("input_cache_hit_tokens", 0.0), p.get("input_cache_miss_tokens", 0.0),
            p.get("output_tokens", 0.0))


def reconcile(billing_rows, tel_rows, start, end, tolerance=0.02) -> dict:
    prices = bill_prices(billing_rows)
    bill = billing_totals(billing_rows, start, end)
    window_rows = [r for r in tel_rows if start <= float(r.get("t") or 0) < end]
    rep = replay(window_rows, prices)
    comparable = rep["estimate_only"] == 0 and rep["requests"] > 0
    delta = ((rep["usd_with_usage"] - bill["usd"]) / bill["usd"]) if (comparable and bill["usd"]) else None
    # Rows from before the gateway logged provider usage: price their (estimated) tokens at the
    # bill's own effective rates. Informational only -- the estimate undercounts prompt tokens.
    calib = best_figure({**bill, "usd": 0.0}, prices, window_rows)["gap_usd"] if rep["estimate_only"] else None
    return {"window": [start, end], "billing": bill, "prices_usd_per_mtok": [x * 1e6 for x in prices],
            "telemetry": rep, "comparable": comparable,
            "relative_error": None if delta is None else round(delta, 5),
            "calibrated_estimate_usd": calib,
            "calibrated_relative_error": (round((calib - bill["usd"]) / bill["usd"], 5)
                                          if calib is not None and bill["usd"] else None),
            "ok": bool(comparable and delta is not None and abs(delta) <= tolerance)}


def _http(url: str, payload=None, token=None, timeout=15):
    """(status, json) for a GET (payload None) or POST to the local gateway."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Admin-Token"] = token
    req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                 headers=headers, method="GET" if payload is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 -- the local gateway
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def post_replace_cas(base: str, token_file: str, compute, source: str, *, telemetry_max_t=None,
                     tries: int = 6, wait: float = 5.0, http=_http, sleep=time.sleep) -> dict:
    """Compare-and-set REPLACE of today's spend (v6, Terra 13:15).

    Each try: read the ledger (revision, in-flight, last settle); require nothing in flight and
    the telemetry to already contain the last settled request; compute the figure; POST it
    bound to that exact revision. A conflict (any settle/hold/write since the read) refuses on
    the gateway side and this recomputes. It never posts an unconditional replace; after
    `tries` conflicts it gives up and reports why, leaving the live total untouched."""
    with open(token_file) as fh:
        token = fh.read().strip()
    base = base.rstrip("/")
    last = None
    for attempt in range(1, tries + 1):
        status, snap = http(base + "/gateway/spend")
        if status != 200:
            last = f"GET /gateway/spend -> {status}"
        elif snap.get("in_flight"):
            last = f"{snap['in_flight']} paid request(s) in flight"
        elif telemetry_max_t is not None and snap.get("last_settle_at") and \
                (telemetry_max_t() or 0) < float(snap["last_settle_at"]) - 2:
            last = "telemetry has not flushed the last settled request yet"
        else:
            usd = compute()
            status, body = http(base + "/gateway/spend/recover",
                                {"spent": usd, "replace": True, "source": source,
                                 "reason": "provider bill reconciliation",
                                 "expected_revision": int(snap.get("revision") or 0)}, token=token)
            if status == 200 and body.get("ok"):
                return {"ok": True, "attempts": attempt, "spent": usd,
                        "revision": snap.get("revision"), "detail": body.get("detail")}
            last = f"conflict: {body.get('detail') or body.get('error') or status}"
        sleep(wait)
    return {"ok": False, "attempts": tries, "reason": last}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--billing", nargs="+", required=True)
    ap.add_argument("--telemetry", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True, help="end of the billed window")
    ap.add_argument("--now", help="end of today's figure (default: now)")
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--provider", default="deepseek", help="the billed provider's identity in the telemetry")
    ap.add_argument("--post", help="gateway base URL: replace today's spend with the best figure")
    ap.add_argument("--admin-token-file", default=os.path.expanduser("~/.local/share/vllm-qwen27b/admin.token"))
    a = ap.parse_args(argv)
    start, end = _ts(a.start), _ts(a.end)
    now = _ts(a.now) if a.now else time.time()
    rows = read_billing(a.billing)
    tel = telemetry_rows(a.telemetry, start, max(end, now), a.provider)
    result = reconcile(rows, tel, start, end, a.tolerance)
    result["best"] = best_figure(result["billing"], bill_prices(rows),
                                 [r for r in tel if end <= float(r.get("t") or 0) < now])
    if a.post:
        def compute():
            t_now = time.time()
            gap = [r for r in telemetry_rows(a.telemetry, end, t_now, a.provider)]
            return best_figure(result["billing"], bill_prices(rows), gap)["best_usd"]

        def telemetry_max_t():
            return max((float(r.get("t") or 0) for r in telemetry_rows(a.telemetry, start, time.time(), a.provider)),
                       default=0.0)
        result["posted"] = post_replace_cas(a.post, a.admin_token_file, compute,
                                            f"provider bill {a.start}..{a.end} + gateway telemetry after it",
                                            telemetry_max_t=telemetry_max_t)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
