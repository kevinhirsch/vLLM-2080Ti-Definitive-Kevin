#!/usr/bin/env python3
"""
Offline re-scorer: reloads results/<tag>/<item_id>.json raw files plus the
matching items/<category>/<item_id>.json spec, and re-runs the *current*
scoring.py logic against the saved raw data -- no network calls. This is a
real recomputation (re-executes code_exec sandboxes, re-validates JSON/
regex/tool-call checks from the saved response), not a replay of whatever
booleans were cached at run time, so it also lets you re-score an old run
against a scoring.py bugfix without re-hitting the live engine.

Usage:
    python3 score_only.py --tag smoke
    python3 score_only.py --tag smoke --out results/smoke/summary.rescored.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import scoring  # noqa: E402

ITEMS_DIR = ROOT / "items"
RESULTS_DIR = ROOT / "results"


_ITEM_SPEC_CACHE = {}


def _index_items_dir():
    """items/<category>/item_NNN.json are named by position, not by the
    item's own "id" field (e.g. "tool_call_001") -- index by id once and
    reuse for every raw file instead of re-globbing per lookup."""
    if _ITEM_SPEC_CACHE:
        return _ITEM_SPEC_CACHE
    for cat_dir in sorted(ITEMS_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        for f in cat_dir.glob("item_*.json"):
            spec = json.loads(f.read_text())
            _ITEM_SPEC_CACHE[spec["id"]] = spec
    return _ITEM_SPEC_CACHE


def load_item_spec(item_id, category_hint=None):
    return _index_items_dir().get(item_id)


def rescore_single_shot(item, raw):
    result = raw.get("result", {})
    if result.get("status") == "engine-crash":
        return {"passed": False, "reason": "engine-crash"}
    if result.get("status") == "http_error":
        return {"passed": False, "reason": f"http {result.get('status_code')}"}
    if result.get("status") != "ok":
        return {"passed": False, "reason": f"unknown raw status {result.get('status')!r}"}

    resp = result["response"]
    choice = resp["choices"][0]
    msg = choice["message"]
    content = msg.get("content")
    finish_reason = choice.get("finish_reason")
    cat = item["category"]

    if cat == "tool_call":
        return scoring.score_tool_call(item, msg)
    if cat == "code_exec":
        return scoring.score_code_exec(item, content)
    if cat == "json_schema":
        return scoring.score_json_schema(item, content)
    if cat == "instruction":
        return scoring.score_instruction(item, content)
    if cat == "loop_probe":
        return scoring.score_loop_probe(item, content, finish_reason)
    if cat == "long_ctx":
        return scoring.score_long_ctx(item, content)
    return {"passed": False, "reason": f"no single-shot rescorer for category {cat}"}


def rescore_one(raw_path):
    raw = json.loads(raw_path.read_text())
    item_id = raw.get("id") or raw_path.stem
    category = raw.get("category")
    item = load_item_spec(item_id, category)
    if item is None:
        return {"id": item_id, "category": category, "passed": False, "reason": "item spec not found"}

    if category == "agentic_chain":
        score = scoring.score_agentic_chain(item, raw)
    elif "runner_exception" in raw:
        score = {"passed": False, "reason": raw.get("score", {}).get("reason", "runner exception")}
    else:
        score = rescore_single_shot(item, raw)

    return {"id": item_id, "category": category, "passed": bool(score.get("passed")), "score": score}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results_dir = RESULTS_DIR / args.tag
    raw_files = sorted(
        f for f in results_dir.glob("*.json")
        if f.name != "summary.json" and not f.name.startswith("summary.")
    )
    if not raw_files:
        print(f"No raw result files found under {results_dir}", file=sys.stderr)
        sys.exit(1)

    rescored = [rescore_one(f) for f in raw_files]

    by_cat = {}
    for r in rescored:
        c = by_cat.setdefault(r["category"], {"pass": 0, "total": 0})
        c["total"] += 1
        c["pass"] += int(r["passed"])

    summary = {
        "tag": args.tag,
        "rescored_from": str(results_dir),
        "items": [{"id": r["id"], "category": r["category"], "passed": r["passed"]} for r in rescored],
        "by_category": by_cat,
        "total_pass": sum(v["pass"] for v in by_cat.values()),
        "total_items": sum(v["total"] for v in by_cat.values()),
    }

    out_path = Path(args.out) if args.out else (results_dir / "summary.rescored.json")
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"{summary['total_pass']}/{summary['total_items']} passed (rescored)")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
