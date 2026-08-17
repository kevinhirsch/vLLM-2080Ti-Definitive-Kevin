#!/usr/bin/env python3
"""Reference decode matrix — same reference points as deploy/docs/vLLM-Benchmarks.md
so MTP-on / MTP-off / 3.6-vs-3.8 rows are directly comparable:

    1x7.5K   single-stream daily-driver point   (3.6 champion: 83.7-105.7 tok/s MTP3)
    2x7.5K   two concurrent moderate            (3.6: ~63 aggregate)
    2x30K    the concurrency OOM-cliff regression point
    1x60K    the old MTP garble band
    1x200K   long-context envelope              (3.6: ~0.3 tok/s decode, ~4 min TTFT)

Prints table rows ready to paste into RESULTS-TEMPLATE.md.
"""
import argparse
import concurrent.futures as cf
import sys

from bench_lib import (DEFAULT_BASE, DEFAULT_MODEL, acceptance_rate,
                       build_context_prompt, chat, garble_score, nrestarts,
                       scrape_spec_metrics)

MATRIX = [  # (label, ctx_tokens, concurrency, max_tokens)
    ("1x7.5K", 7500, 1, 800),
    ("2x7.5K", 7500, 2, 800),
    ("2x30K", 30000, 2, 500),
    ("1x60K", 60000, 1, 500),
    ("1x200K", 200000, 1, 300),
]


def one(base, model, ctx, max_tokens, timeout, salt):
    # unique salt FIRST so concurrent requests share no prefix — otherwise
    # prefix caching turns "2x30K" into one prefill + one cache hit and the
    # row stops matching the documented reference workload
    msgs = [{"role": "user", "content":
             f"[run {salt}] " + build_context_prompt(ctx, f"DM-{ctx}-{salt}") +
             "\n\nSummarize the operational policy in detail."}]
    return chat(base, model, msgs, max_tokens=max_tokens, temperature=0.7,
                timeout=timeout,
                extra={"chat_template_kwargs": {"enable_thinking": False}})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--only", help="comma list of labels to run (default all)")
    ap.add_argument("--min-single-tps", type=float, default=None,
                    help="fail if the 1x7.5K per-stream decode rate is below "
                    "this (G4 ship bar: 70 for the MTP-on run; leave unset "
                    "for the intentionally slower MTP-off baseline)")
    args = ap.parse_args()

    labels = set(args.only.split(",")) if args.only else None
    if labels is not None:
        known = {m[0] for m in MATRIX}
        unknown = labels - known
        if unknown or not labels:
            ap.error(f"unknown --only label(s): {sorted(unknown)}; "
                     f"choose from {sorted(known)}")
    print(f"| point | TTFT s | decode tok/s (per stream) | aggregate tok/s | "
          f"acceptance | garble | NRestarts Δ |")
    print("|---|---|---|---|---|---|---|")
    rc = 0
    for label, ctx, conc, mt in MATRIX:
        if labels and label not in labels:
            continue
        timeout = max(300.0, ctx / 850 * 1.6 + 300.0)
        before, r0 = scrape_spec_metrics(args.base_url), nrestarts()
        try:
            with cf.ThreadPoolExecutor(max_workers=conc) as ex:
                futs = [ex.submit(one, args.base_url, args.model, ctx, mt,
                                  timeout, f"{label}-{i}")
                        for i in range(conc)]
                results = [f.result() for f in futs]
        except Exception as exc:
            print(f"| {label} | — | — | — | — | ERROR {type(exc).__name__} | "
                  f"{(nrestarts() or 0) - (r0 or 0) if r0 is not None else '?'} |")
            rc = 1
            continue
        after, r1 = scrape_spec_metrics(args.base_url), nrestarts()
        per = [r["decode_tps"] for r in results]
        agg_tokens = sum(r["completion_tokens"] for r in results)
        wall = max(r["total_s"] for r in results)
        ttft = max((r["ttft_s"] or 0) for r in results)
        garbled = any(garble_score(r["content"])["garbled"] for r in results)
        acc = acceptance_rate(before, after)
        delta = (r1 - r0) if (r0 is not None and r1 is not None) else None
        print(f"| {label} | {ttft:.1f} | "
              f"{'/'.join(f'{p:.1f}' for p in per)} | "
              f"{agg_tokens / wall:.1f} | "
              f"{'—' if acc is None else f'{acc:.2f}'} | "
              f"{'YES' if garbled else 'no'} | {delta} |", flush=True)
        # fail closed: an unavailable NRestarts delta cannot prove the engine
        # stayed up, so it fails the row just like a real restart would
        if garbled or delta != 0:
            rc = 1
        if (args.min_single_tps is not None and label == "1x7.5K"
                and max(per) < args.min_single_tps):
            print(f"SHIP BAR MISSED: 1x7.5K {max(per):.1f} tok/s "
                  f"< {args.min_single_tps}", flush=True)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
