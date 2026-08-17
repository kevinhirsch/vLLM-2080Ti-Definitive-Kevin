#!/usr/bin/env python3
"""MTP re-qualification ladder — the test that decides whether speculative
decoding comes back on after the 2026-08-14 incident.

Background: MTP3 + prefix caching + cudagraphs garbled output / crashed the
engine at ~57-64K context, so the 2026-08-15 campaign shipped with MTP OFF
(commit 746d8b1) at a ~40-60%% single-stream decode cost. Upstream v0.1.15
landed two fixes squarely on that failure surface:
  - 40129ea  preserve hybrid Mamba prefix-cache correctness with MTP
  - c256ad2  preserve GDN state slot zero during decode
plus this fork's own negative-draft-id clamp (746d8b1, gpu_model_runner.py).

This script walks a context ladder ACROSS and BEYOND the old failure band and
fails loudly at the first garble, crash, or acceptance collapse. Run it against
the MTP-enabled serve variant (serve-qwen38-mtp-requal-fg.sh):

    python3 bench_mtp_requal.py --stages 8000,32000,57000,64000,96000,128000,200000

Each stage:
  1. builds a synthetic prompt of ~N tokens with a recall needle at 25%% depth
  2. runs a generation probe (needle recall + continuation prose)
  3. checks: needle recalled, no garble signature, finish_reason sane
  4. scrapes spec-decode acceptance for the stage (delta around the probe)
  5. checks NRestarts did not move (engine did not crash-restart mid-stage)

Exit code 0 = all stages pass (MTP may ship). Nonzero = first failing stage is
reported; keep MTP off and file the stage's JSON with the incident notes.
"""
import argparse
import json
import sys
import time

from bench_lib import (DEFAULT_BASE, DEFAULT_MODEL, acceptance_rate,
                       build_context_prompt, chat, garble_score, nrestarts,
                       scrape_spec_metrics)

# 3.6 reference: MTP3 acceptance 67.9% @T0.4 (docs/mtp-task-sensitivity.md).
# Below this floor speculation is hurting more than helping at long context —
# treat as a soft failure worth recording even if output is clean.
ACCEPTANCE_FLOOR = 0.45


def run_stage(base: str, model: str, ctx_tokens: int, *, temperature: float,
              max_tokens: int, timeout: float) -> dict:
    needle = f"K7-{ctx_tokens}-QUINCE"
    doc = build_context_prompt(ctx_tokens, needle)
    messages = [
        {"role": "system", "content": "You are a precise operations analyst."},
        {"role": "user", "content": (
            doc + "\n\nFirst, state the exact AUDIT MARKER value verbatim. "
            "Then summarize the operational policy above in 3 bullet points.")},
    ]
    before = scrape_spec_metrics(base)
    r0 = nrestarts()
    result = chat(base, model, messages, max_tokens=max_tokens,
                  temperature=temperature, timeout=timeout,
                  extra={"chat_template_kwargs": {"enable_thinking": False}})
    after = scrape_spec_metrics(base)
    r1 = nrestarts()

    text = result["content"]
    g = garble_score(text)
    acc = acceptance_rate(before, after)
    stage = {
        "ctx_tokens": ctx_tokens,
        "prompt_tokens": result["prompt_tokens"],
        "ttft_s": None if result["ttft_s"] is None else round(result["ttft_s"], 2),
        "decode_tps": round(result["decode_tps"], 1),
        "completion_tokens": result["completion_tokens"],
        "finish_reason": result["finish_reason"],
        "needle_recalled": needle in text,
        "garble": g,
        "acceptance": None if acc is None else round(acc, 3),
        # raw deltas kept so foreign traffic during a stage is visible in the
        # results JSON (drafts >> this probe's expected count = contaminated
        # sample; re-run the stage with the engine isolated)
        "draft_tokens_delta": None if (before is None or after is None)
        else after["draft_tokens"] - before["draft_tokens"],
        "nrestarts_delta": None if (r0 is None or r1 is None) else r1 - r0,
    }
    # FAIL CLOSED: missing telemetry is missing evidence, not health.
    #  - nrestarts None: cannot prove the engine stayed up
    #  - acceptance None: MTP is not drafting or /metrics is down — either way
    #    a re-qualification of MTP cannot be scored
    # CONTAMINATION: /metrics counters are engine-global. A single probe can be
    # drafted at most once per emitted token, so a drafts delta exceeding this
    # probe's completion tokens proves concurrent traffic reached the engine
    # during the stage — the acceptance figure then scores a mixture, not this
    # probe, and cannot qualify MTP.
    drafts_delta = (None if (before is None or after is None)
                    else after["drafts"] - before["drafts"])
    stage["drafts_delta"] = drafts_delta
    contaminated = (
        drafts_delta is not None
        and result["completion_tokens"] > 0
        and drafts_delta > result["completion_tokens"]
    )
    stage["contaminated"] = contaminated
    hard_fail = (
        not stage["needle_recalled"]
        or g["garbled"]
        or stage["nrestarts_delta"] != 0
        or acc is None
        or contaminated
        or result["finish_reason"] not in ("stop", "length")
    )
    soft_fail = acc is not None and acc < ACCEPTANCE_FLOOR
    stage["verdict"] = ("FAIL" if hard_fail else
                       "SOFT-FAIL(acceptance)" if soft_fail else "PASS")
    if hard_fail and stage["nrestarts_delta"] is None:
        stage["fail_reason"] = "NRestarts unavailable (fail-closed)"
    elif hard_fail and acc is None:
        stage["fail_reason"] = "no acceptance telemetry (fail-closed)"
    elif hard_fail and contaminated:
        stage["fail_reason"] = (
            "spec counters contaminated by concurrent traffic "
            f"(drafts_delta={drafts_delta} > completion_tokens="
            f"{result['completion_tokens']}) — isolate the engine and re-run"
        )
    return stage


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--stages",
                    default="8000,32000,57000,64000,96000,128000,200000",
                    help="comma-separated context sizes in tokens")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="probe temp (0.7 = official 3.8 instruct rec)")
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--repeat", type=int, default=2,
                    help="probes per stage (garble is intermittent; 2 minimum)")
    ap.add_argument("--out", default="mtp-requal-results.json")
    args = ap.parse_args()

    stages = [int(s) for s in args.stages.split(",") if s.strip()]
    if not stages:
        ap.error("--stages must name at least one context size")
    if args.repeat < 2:
        ap.error("--repeat must be >= 2 (garble is intermittent; "
                 "a single probe per stage proves nothing)")
    results, failed = [], False
    for ctx in stages:
        # generous ceiling: prefill at worst ~850 tok/s + decode + slack
        timeout = max(300.0, ctx / 850 * 1.5 + 240.0)
        for attempt in range(1, args.repeat + 1):
            print(f"[stage {ctx:>6}] probe {attempt}/{args.repeat} "
                  f"(timeout {int(timeout)}s)...", flush=True)
            t0 = time.time()
            try:
                stage = run_stage(args.base_url, args.model, ctx,
                                  temperature=args.temperature,
                                  max_tokens=args.max_tokens, timeout=timeout)
            except Exception as exc:  # engine died / connection dropped
                stage = {"ctx_tokens": ctx, "verdict": "FAIL",
                         "error": f"{type(exc).__name__}: {exc}",
                         "nrestarts_now": nrestarts()}
            stage["attempt"] = attempt
            stage["wall_s"] = round(time.time() - t0, 1)
            results.append(stage)
            print(f"           -> {stage['verdict']} "
                  f"decode={stage.get('decode_tps', '?')} tok/s "
                  f"acceptance={stage.get('acceptance')}", flush=True)
            if stage["verdict"] == "FAIL":
                failed = True
                break
        if failed:
            print(f"[stage {ctx}] HARD FAIL — stopping ladder. MTP stays OFF.")
            break

    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"\nresults -> {args.out}")
    if failed:
        return 1
    soft = [r for r in results if r["verdict"].startswith("SOFT")]
    if soft:
        # exit 2 (distinct from hard-fail 1): output was clean but acceptance
        # sat below the floor — the ladder does NOT approve MTP on this run.
        # Ship only after a deliberate decision recorded in RESULTS-TEMPLATE.md.
        print(f"SOFT FAIL: {len(soft)} stage(s) below acceptance floor "
              f"{ACCEPTANCE_FLOOR} — clean output, but speculation is not "
              "paying for itself. NOT auto-approving; exit 2.")
        return 2
    print("ladder complete: no garble, no crash, acceptance above floor. "
          "MTP is re-qualifiable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
