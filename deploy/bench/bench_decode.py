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


def one(base, model, ctx, max_tokens, timeout):
    msgs = [{"role": "user", "content":
             build_context_prompt(ctx, f"DM-{ctx}") +
             "\n\nSummarize the operational policy in detail."}]
    return chat(base, model, msgs, max_tokens=max_tokens, temperature=0.7,
                timeout=timeout,
                extra={"chat_template_kwargs": {"enable_thinking": False}})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--only", help="comma list of labels to run (default all)")
    args = ap.parse_args()

    labels = set(args.only.split(",")) if args.only else None
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
                                  timeout) for _ in range(conc)]
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
        if garbled or (delta not in (0, None)):
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
