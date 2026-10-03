#!/usr/bin/env python3
"""remote_pricing_refresh.py -- keep the gateway's dated pricing table honest.

Lane SL (2026-10-02). The gateway prices a remote token as a function of (model, token type,
time) from remote-pricing.json (see keepalive-shim.py, "dated pricing table"). This tool
re-reads the provider's OFFICIAL pricing page, compares it with the table on disk and reports
the drift as a fact. It changes nothing unless you pass --apply:

    remote_pricing_refresh.py                       # fetch, diff, print; exit 3 on drift
    remote_pricing_refresh.py --fact-out f.json     # also write the drift fact for Halo
    remote_pricing_refresh.py --apply               # back the file up, rewrite it atomically
    remote_pricing_refresh.py --html page.html      # offline: parse a saved copy of the page

Exit codes: 0 table matches the page, 3 drift found, 2 the page could not be parsed (the table is
NEVER rewritten from a page it did not fully understand). The running gateway re-stats the file,
so an applied refresh needs no restart and no gateway publish.

What the page gives: per-model off-peak/peak prices for cache-hit input, cache-miss input and
output; the UTC peak windows and the weekday rule; the legacy model-name aliases. What it does
not give: the Chinese public-holiday calendar (peak is suspended on those days). That stays in
the table's `off_peak_dates`, and this tool warns when `holidays_through` is within 60 days.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import html as htmllib
import json
import os
import re
import sys
import time
import urllib.request

DEFAULT_URL = "https://api-docs.deepseek.com/quick_start/pricing"
DEFAULT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remote-pricing.json")
KINDS = (("(CACHE HIT)", "cache_hit"), ("(CACHE MISS)", "cache_miss"), ("OUTPUT TOKENS", "output"))


class PageError(ValueError):
    pass


def page_text(markup: str) -> str:
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", markup, flags=re.S)
    t = re.sub(r"<(/tr|/p|/h\d|br|/li)[^>]*>", "\n", t)
    t = re.sub(r"<(/td|/th)[^>]*>", " | ", t)
    t = htmllib.unescape(re.sub(r"<[^>]+>", "", t))
    return re.sub(r"\n\s*\n+", "\n", t)


def parse_page(markup: str) -> dict:
    """The provider's pricing page -> {models:{name:{off_peak,peak}}, windows, weekdays,
    holidays_off_peak, aliases:{legacy: model}}. Raises PageError when anything is missing."""
    text = page_text(markup)
    names = None
    for line in text.splitlines():
        if line.strip().upper().startswith("MODEL |"):
            cells = [re.sub(r"\(\d+\)", "", c).strip() for c in line.split("|")[1:]]
            names = [c for c in cells if c]
            break
    if not names:
        raise PageError("model header row not found")
    models = {n: {"off_peak": {}, "peak": {}} for n in names}
    kind = None
    for line in text.splitlines():
        for marker, k in KINDS:
            if marker in line.upper():
                kind = k
        m = re.search(r"(?:^|\|)\s*(OFF-PEAK|PEAK)\s*\|(.*)$", line.strip(), re.I)
        if not (m and kind):
            continue
        prices = [float(x) for x in re.findall(r"\$\s*([0-9]*\.?[0-9]+)", m.group(2))]
        if len(prices) < len(names):
            continue
        period = "off_peak" if m.group(1).upper() == "OFF-PEAK" else "peak"
        for n, v in zip(names, prices):
            models[n][period][kind] = v
    for n, row in models.items():
        for period in ("off_peak", "peak"):
            if set(row[period]) != {"cache_hit", "cache_miss", "output"}:
                raise PageError(f"{n}: {period} prices incomplete ({sorted(row[period])})")
    sent = re.search(r"Peak hours are(.{0,200}?)UTC,?\s*(Monday through Friday)?", text, re.S)
    if not sent:
        raise PageError("peak-hours sentence not found")
    windows = [[int(a), int(c)] for a, _b, c, _d in
               re.findall(r"(\d\d):(\d\d)\s*-\s*(\d\d):(\d\d)", sent.group(1))]
    if not windows:
        raise PageError("peak windows not found")
    if not sent.group(2):
        raise PageError("weekday rule not found (expected 'Monday through Friday')")
    holidays = bool(re.search(r"excluding Chinese public holidays", text, re.I))
    aliases = {}
    use = re.search(r"Use\s+(\S+?)\s+as the model name", text)
    legacy = re.search(r"legacy names?\s+(.+?)\s+(?:are|is)\s+still accepted", text, re.S)
    if use and legacy:
        for name in re.split(r",|\band\b", legacy.group(1)):
            name = name.strip()
            if name:
                aliases[name] = use.group(1)
    for ratio_model, row in models.items():          # the page's own claim: off-peak = half of peak
        for t in ("cache_hit", "cache_miss", "output"):
            if row["peak"][t] and abs(row["off_peak"][t] * 2 - row["peak"][t]) > 1e-9:
                raise PageError(f"{ratio_model}.{t}: off-peak is not half of peak -- the page changed shape")
    return {"models": models, "windows": windows, "weekdays": [0, 1, 2, 3, 4],
            "holidays_off_peak": holidays, "aliases": aliases}


def build_table(current: dict, parsed: dict, today: str, url: str) -> dict:
    """The current table updated with what the page says (holiday calendar carried over)."""
    new = copy.deepcopy(current)
    new["fetched"], new["source_url"] = today, url
    new["peak"]["windows"], new["peak"]["weekdays"] = parsed["windows"], parsed["weekdays"]
    for name, row in parsed["models"].items():
        old = new["models"].get(name, {"aliases": []})
        new["models"][name] = {"aliases": sorted(set(old.get("aliases", []))
                                                 | {a for a, m in parsed["aliases"].items() if m == name}),
                               "off_peak": row["off_peak"], "peak": row["peak"]}
    return new


def diff_tables(current: dict, new: dict) -> list:
    out = []
    if current["peak"]["windows"] != new["peak"]["windows"]:
        out.append(f"peak windows {current['peak']['windows']} -> {new['peak']['windows']}")
    if current["peak"]["weekdays"] != new["peak"]["weekdays"]:
        out.append(f"peak weekdays {current['peak']['weekdays']} -> {new['peak']['weekdays']}")
    for name in sorted(set(current["models"]) | set(new["models"])):
        a, b = current["models"].get(name), new["models"].get(name)
        if a is None:
            out.append(f"model {name}: NEW on the page")
        elif b is None:
            out.append(f"model {name}: in the table but not on the page")
        else:
            for period in ("off_peak", "peak"):
                for t in ("cache_hit", "cache_miss", "output"):
                    if a[period][t] != b[period][t]:
                        out.append(f"{name}.{period}.{t}: {a[period][t]} -> {b[period][t]} $/Mtok")
            if sorted(a.get("aliases", [])) != sorted(b.get("aliases", [])):
                out.append(f"{name}.aliases {sorted(a.get('aliases', []))} -> {sorted(b.get('aliases', []))}")
    return out


def calendar_warnings(table: dict, today: datetime.date) -> list:
    through = table["peak"].get("holidays_through")
    if not through:
        return ["table has no holidays_through: the holiday calendar's coverage is unknown"]
    left = (datetime.date.fromisoformat(through) - today).days
    if left < 0:
        return [f"the holiday calendar ended {through}: later holidays are priced as peak days (dearer)"]
    if left < 60:
        return [f"the holiday calendar ends {through} ({left} days): add next year's off_peak_dates"]
    return []


def fetch(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "gateway-pricing-refresh/1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 -- the provider's public docs page
        return resp.read().decode("utf-8", "replace")


def write_atomic(path: str, table: dict) -> str:
    backup = f"{path}.bak-{int(time.time())}"
    if os.path.exists(path):
        with open(path, "rb") as src, open(backup, "wb") as dst:
            dst.write(src.read())
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(table, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)
    return backup


def refresh(path: str, markup: str, url: str, today: datetime.date, apply: bool = False) -> dict:
    with open(path, encoding="utf-8") as fh:
        current = json.load(fh)
    parsed = parse_page(markup)
    new = build_table(current, parsed, today.isoformat(), url)
    changes = diff_tables(current, new)
    warnings = calendar_warnings(new, today)
    if not parsed["holidays_off_peak"]:
        warnings.append("the page no longer says peak excludes Chinese public holidays: review off_peak_dates")
    result = {"fact": "remote_pricing_table", "checked_at": time.time(), "source_url": url,
              "table_fetched": current.get("fetched"), "drift": bool(changes), "changes": changes,
              "warnings": warnings, "applied": False}
    if apply and (changes or current.get("fetched") != new["fetched"]):
        # a clean re-check still re-stamps `fetched`: it is the "verified against the page on" date
        # the gateway's staleness check reads
        result["backup"] = write_atomic(path, new)
        result["applied"] = True
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--file", default=DEFAULT_FILE, help="the pricing table to check (default: beside this tool)")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--html", help="parse this saved copy of the page instead of fetching")
    ap.add_argument("--apply", action="store_true", help="rewrite the table (backup kept) when the page differs")
    ap.add_argument("--fact-out", help="write the result as a JSON fact here")
    a = ap.parse_args(argv)
    try:
        markup = open(a.html, encoding="utf-8").read() if a.html else fetch(a.url)
        result = refresh(a.file, markup, a.url, datetime.date.today(), a.apply)
    except (PageError, OSError, ValueError) as exc:
        print(json.dumps({"fact": "remote_pricing_table", "error": f"{type(exc).__name__}: {exc}",
                          "applied": False}), file=sys.stderr)
        return 2
    if a.fact_out:
        tmp = f"{a.fact_out}.tmp-{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, a.fact_out)
    print(json.dumps(result, indent=2))
    return 3 if result["drift"] and not result["applied"] else 0


if __name__ == "__main__":
    sys.exit(main())
