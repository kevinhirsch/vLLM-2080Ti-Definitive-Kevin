#!/usr/bin/env python3
"""
EXP-026 -- loop_probe rescue: profile A vs. profile B A/B test.

CONTEXT (see BASELINE-v0.md): baseline-v0-20260816 scored loop_probe at
4/10 using whatever the engine's default sampling params happen to be.
All 6 failures were degeneration/self-stop issues on long "list exactly
100 distinct X" generations (duplicate items, mirrored-repetition blocks,
or a clean self-issued stop token short of 100), never truncation by
max_tokens and never stray commentary. This script re-runs all 10
loop_probe items under two candidate sampling profiles to see whether
either one measurably reduces that degeneration:

    Profile A: temperature=0.6, top_p=0.95, top_k=20              ("gencfg")
    Profile B: temperature=0.7, top_p=0.80, top_k=20,
               presence_penalty=1.5           ("official non-thinking" + pp)

max_tokens is bumped to the [2500, 3000] range per item (see
clamp_max_tokens below) so a genuine budget cutoff can't masquerade as a
degeneration failure -- baseline-v0 already showed max_tokens wasn't the
binding constraint for any loop_probe failure, but we hold that fixed
here too so any residual failure is unambiguously a generation-quality
issue, not a token-budget issue.

HARD RULE -- DO NOT RUN THIS AGAINST THE LIVE ENGINE WHILE IT IS OWNED BY
ANOTHER TEST. This file talks to config.BASE_URL (:8001) for real,
20 chat completions (10 items x 2 profiles) at up to 3000 completion
tokens each. Only invoke it for real once the engine is confirmed free.
`--dry-run` builds every request payload and prints it without opening a
socket -- safe to run any time (no GPU inference, doesn't touch :8001) --
useful to sanity check profiles/payload construction before a real run.

Usage (once the engine is free):
    python3 exp026_loop_ab.py --tag exp026-loop-ab-20260817
    python3 exp026_loop_ab.py --tag exp026-loop-ab-20260817 --profiles A
    python3 exp026_loop_ab.py --dry-run   # safe any time, no network calls
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config          # noqa: E402
import scoring         # noqa: E402
import run_eval        # noqa: E402  -- reuses post_chat/wait_for_recovery crash-tolerance, no side effects at import
from scoring import _LOOP_LINE_RE  # noqa: E402 -- same parsing regex the real scorer uses, kept in sync

ITEMS_DIR = ROOT / "items" / "loop_probe"
RESULTS_DIR = ROOT / "results"

# ---------------------------------------------------------------------------
# Profiles under test
# ---------------------------------------------------------------------------

PROFILES = {
    "A": {
        "label": "A_gencfg",
        "description": "gencfg default sampling",
        "params": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    },
    "B": {
        "label": "B_official_nonthinking_pp",
        "description": "Qwen3 official non-thinking-mode sampling + presence_penalty",
        "params": {"temperature": 0.7, "top_p": 0.80, "top_k": 20, "presence_penalty": 1.5},
    },
}

MIN_MAX_TOKENS = 2500
MAX_MAX_TOKENS = 3000


def clamp_max_tokens(item, override=None):
    """Every loop_probe item's own spec already sets a max_tokens (2000-3000
    across the 10 baseline items); clamp it into [2500, 3000] rather than
    trusting the item spec verbatim, so a 100-item list has comfortable
    budget under both profiles and no item is left at the 2000 baseline
    outlier that's tighter than the rest of the block."""
    if override is not None:
        return override
    base = item.get("request", {}).get("max_tokens", MIN_MAX_TOKENS)
    return max(MIN_MAX_TOKENS, min(MAX_MAX_TOKENS, base))


def load_loop_probe_items():
    files = sorted(ITEMS_DIR.glob("item_*.json"))
    return [json.loads(f.read_text()) for f in files]


def build_payload(item, profile, max_tokens):
    payload = {
        "model": config.MODEL_NAME,
        "chat_template_kwargs": dict(config.DEFAULT_CHAT_TEMPLATE_KWARGS),
        "messages": item["request"]["messages"],
        "max_tokens": max_tokens,
    }
    payload.update(profile["params"])
    return payload


# ---------------------------------------------------------------------------
# Degeneration diagnostics (richer than the pass/fail scorer alone -- pass/
# fail truth still comes solely from scoring.score_loop_probe, the one
# place that logic is supposed to live; this is extra reporting detail for
# the A/B comparison).
# ---------------------------------------------------------------------------


def analyze_degeneration(content, finish_reason, n_expected):
    text = content or ""
    matches = _LOOP_LINE_RE.findall(text)
    numbers = [int(m[0]) for m in matches]
    items_text = [m[1].strip().lower() for m in matches]

    order_ok = numbers == list(range(1, n_expected + 1))

    seen_at = {}
    duplicate_pairs = []
    for i, t in enumerate(items_text, start=1):
        if t in seen_at:
            duplicate_pairs.append({"item": t, "first_line": seen_at[t], "repeat_line": i})
        else:
            seen_at[t] = i
    duplicate_count = len(duplicate_pairs)
    distinct_ok = len(items_text) == n_expected and duplicate_count == 0

    non_blank_lines = [ln for ln in text.strip("\n").split("\n") if ln.strip() != ""]
    extra_line_count = len(non_blank_lines) - len(numbers)

    return {
        "finish_reason": finish_reason,
        "n_lines_parsed": len(numbers),
        "n_expected": n_expected,
        "order_ok": order_ok,
        "duplicate_count": duplicate_count,
        # capped so a badly-degenerated response can't blow up the report file
        "duplicate_pairs": duplicate_pairs[:20],
        "distinct_ok": distinct_ok,
        "extra_line_count": extra_line_count,
        "hit_max_tokens": finish_reason == "length",
        "self_stopped_short": finish_reason == "stop" and len(numbers) < n_expected,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_item_under_profile(item, profile_key, profile, max_tokens, out_dir):
    payload = build_payload(item, profile, max_tokens)
    result = run_eval.post_chat(payload, timeout_s=240)

    raw_record = {
        "id": item["id"],
        "category": "loop_probe",
        "profile": profile_key,
        "profile_label": profile["label"],
        "profile_params": profile["params"],
        "max_tokens": max_tokens,
        "request": payload,
        "result": result,
    }

    if result["status"] == "engine-crash":
        recovered = run_eval.wait_for_recovery()
        raw_record["recovered"] = recovered
        score = {"passed": False, "reason": "engine-crash"}
        degeneration = {"finish_reason": None, "note": "engine-crash, no content to analyze"}
    elif result["status"] == "http_error":
        score = {"passed": False, "reason": f"http {result['status_code']}"}
        degeneration = {"finish_reason": None, "note": f"http_error {result['status_code']}"}
    else:
        msg = result["response"]["choices"][0]["message"]
        finish_reason = result["response"]["choices"][0].get("finish_reason")
        content = msg.get("content")
        n_expected = item["check"].get("count", 100)
        score = scoring.score_loop_probe(item, content, finish_reason)
        degeneration = analyze_degeneration(content, finish_reason, n_expected)

    raw_record["score"] = score
    raw_record["degeneration"] = degeneration

    out_path = out_dir / profile_key / f"{item['id']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(raw_record, indent=2, ensure_ascii=False))

    return score, degeneration


def summarize_profile(rows):
    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    dup_counts = [r["degeneration"].get("duplicate_count", 0) for r in rows if "duplicate_count" in r["degeneration"]]
    return {
        "pass": passed,
        "total": total,
        "order_ok": sum(1 for r in rows if r["degeneration"].get("order_ok")),
        "distinct_ok": sum(1 for r in rows if r["degeneration"].get("distinct_ok")),
        "self_stopped_short": sum(1 for r in rows if r["degeneration"].get("self_stopped_short")),
        "hit_max_tokens": sum(1 for r in rows if r["degeneration"].get("hit_max_tokens")),
        "total_duplicates": sum(dup_counts),
        "avg_duplicates": round(sum(dup_counts) / len(dup_counts), 2) if dup_counts else 0,
        "avg_n_lines_parsed": round(
            sum(r["degeneration"].get("n_lines_parsed", 0) for r in rows) / total, 1
        ) if total else 0,
    }


def print_report(tag, by_profile, rows_by_profile):
    print()
    print(f"=== EXP-026 loop_probe A/B report ({tag}) ===")
    print(f"{'profile':<28}{'pass':>6}{'/':>1}{'total':<8}{'order_ok':>10}{'distinct_ok':>13}"
          f"{'self_stop_short':>17}{'hit_maxtok':>12}{'tot_dupes':>11}{'avg_dupes':>11}{'avg_lines':>11}")
    for key in rows_by_profile.keys():
        profile = PROFILES[key]
        s = by_profile[key]
        print(
            f"{profile['label']:<28}{s['pass']:>6}{'/':>1}{s['total']:<8}{s['order_ok']:>10}"
            f"{s['distinct_ok']:>13}{s['self_stopped_short']:>17}{s['hit_max_tokens']:>12}"
            f"{s['total_duplicates']:>11}{s['avg_duplicates']:>11}{s['avg_n_lines_parsed']:>11}"
        )
    print()
    for key in rows_by_profile.keys():
        print(f"--- {PROFILES[key]['label']} per-item ---")
        for r in rows_by_profile[key]:
            d = r["degeneration"]
            print(
                f"  {r['id']:<16} passed={r['passed']!s:<5} "
                f"finish_reason={d.get('finish_reason')!r:<10} "
                f"n_lines={d.get('n_lines_parsed', '-')!s:<5} "
                f"dupes={d.get('duplicate_count', '-')!s:<5} "
                f"self_stopped_short={d.get('self_stopped_short', '-')}"
            )
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=None, help="results subdir under results/; default exp026-loop-ab-<UTC timestamp>")
    ap.add_argument("--profiles", default="A,B", help="comma list of profile keys to run (default: A,B)")
    ap.add_argument("--max-tokens", type=int, default=None,
                     help="override max_tokens for every item/profile (default: per-item, clamped into [2500,3000])")
    ap.add_argument("--dry-run", action="store_true",
                     help="build and print request payloads without contacting the engine (no network calls)")
    args = ap.parse_args()

    profile_keys = [p.strip() for p in args.profiles.split(",") if p.strip()]
    for p in profile_keys:
        if p not in PROFILES:
            print(f"unknown profile {p!r}; choices are {list(PROFILES)}", file=sys.stderr)
            sys.exit(1)

    items = load_loop_probe_items()
    if not items:
        print(f"No loop_probe items found under {ITEMS_DIR}", file=sys.stderr)
        sys.exit(1)

    tag = args.tag or f"exp026-loop-ab-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = RESULTS_DIR / tag

    if args.dry_run:
        print(f"[dry-run] {len(items)} items x {len(profile_keys)} profile(s), no network calls")
        for profile_key in profile_keys:
            profile = PROFILES[profile_key]
            for item in items:
                max_tokens = clamp_max_tokens(item, args.max_tokens)
                payload = build_payload(item, profile, max_tokens)
                print(f"--- {profile_key} / {item['id']} (max_tokens={max_tokens}) ---")
                print(json.dumps(payload, indent=2)[:800])
        print("[dry-run] OK -- no requests sent")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Engine health check: {config.BASE_URL}{config.HEALTH_PATH}")
    if not run_eval.wait_for_recovery(timeout_s=10):
        print(f"WARNING: engine did not respond healthy within 10s at {config.BASE_URL}{config.HEALTH_PATH}. "
              f"Continuing anyway -- crash-tolerance below will retry/wait on any failed request.", file=sys.stderr)

    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    rows_by_profile = {}

    for profile_key in profile_keys:
        profile = PROFILES[profile_key]
        rows = []
        for item in items:
            max_tokens = clamp_max_tokens(item, args.max_tokens)
            item_t0 = time.time()
            score, degeneration = run_item_under_profile(item, profile_key, profile, max_tokens, out_dir)
            wall = time.time() - item_t0
            passed = bool(score.get("passed"))
            rows.append({"id": item["id"], "passed": passed, "degeneration": degeneration, "wall_s": round(wall, 2)})
            print(f"[{profile_key}/{item['id']}] passed={passed} finish_reason={degeneration.get('finish_reason')!r} "
                  f"dupes={degeneration.get('duplicate_count', '-')!s} ({wall:.1f}s)")
        rows_by_profile[profile_key] = rows

    total_wall = time.time() - t0
    by_profile = {k: summarize_profile(v) for k, v in rows_by_profile.items()}

    summary = {
        "tag": tag,
        "exp": "EXP-026",
        "engine_base_url": config.BASE_URL,
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "wall_s": round(total_wall, 2),
        "profiles": {k: PROFILES[k] for k in profile_keys},
        "max_tokens_override": args.max_tokens,
        "results": rows_by_profile,
        "by_profile": by_profile,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print_report(tag, by_profile, rows_by_profile)
    print(f"Raw results + summary.json: {out_dir}")


if __name__ == "__main__":
    main()
