#!/usr/bin/env python3
"""
Crash-tolerant live runner against the :8001 vLLM engine.

Loads items/<category>/*.json, builds an OpenAI-compatible chat request per
item, POSTs it to config.BASE_URL, scores the response immediately via
scoring.py, and saves EVERY raw response (success, HTTP error, or crash) to
results/<tag>/<item_id>.json so score_only.py can independently re-derive
pass/fail later without hitting the network again.

Never talk to :8000 (the capacity-gateway shim) -- config.BASE_URL is the
single source of truth for the endpoint and is hardcoded to :8001.

Usage:
    python3 run_eval.py --tag smoke --categories tool_call,instruction
    python3 run_eval.py --tag full --profile profiles/greedy.json
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

import config
import corpus
import scoring

ROOT = Path(__file__).resolve().parent
ITEMS_DIR = ROOT / "items"
RESULTS_DIR = ROOT / "results"

CATEGORY_ORDER = [
    "tool_call", "code_exec", "json_schema", "instruction",
    "loop_probe", "long_ctx", "agentic_chain",
]


def load_items(categories=None, limit_per_category=None):
    items = []
    for cat in CATEGORY_ORDER:
        if categories and cat not in categories:
            continue
        cat_dir = ITEMS_DIR / cat
        if not cat_dir.exists():
            continue
        files = sorted(cat_dir.glob("item_*.json"))
        if limit_per_category:
            files = files[:limit_per_category]
        for f in files:
            items.append(json.loads(f.read_text()))
    return items


def wait_for_recovery(timeout_s=None):
    """Poll HEALTH_PATH until the engine responds or we give up."""
    timeout_s = timeout_s or config.CRASH_POLL_TIMEOUT_S
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(config.BASE_URL + config.HEALTH_PATH, timeout=5)
            if r.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(config.CRASH_POLL_INTERVAL_S)
    return False


def capture_journal(item_id, start_iso, out_dir):
    """Best-effort journalctl slice since start_iso, grepped for the GOLD
    patterns. Writes results/<tag>/journal_<item_id>.log if anything
    matches; returns True if a gold pattern was found."""
    try:
        proc = subprocess.run(
            ["journalctl", "-u", config.JOURNAL_UNIT, "-S", start_iso, "--no-pager"],
            capture_output=True, text=True, timeout=30,
        )
        text = proc.stdout
    except Exception as e:
        return False, f"journalctl capture failed: {e}"

    import re
    hit = any(re.search(p, text, re.IGNORECASE) for p in config.JOURNAL_GOLD_PATTERNS)
    if hit:
        (out_dir / f"journal_{item_id}.log").write_text(text)
        # best-effort kernel ring buffer too
        try:
            kproc = subprocess.run(
                ["journalctl", "-k", "-S", start_iso, "--no-pager"],
                capture_output=True, text=True, timeout=30,
            )
            (out_dir / f"journal_{item_id}.kernel.log").write_text(kproc.stdout)
        except Exception:
            pass
    return hit, None


def post_chat(payload, timeout_s=180):
    url = config.BASE_URL + "/v1/chat/completions"
    t0 = time.time()
    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
        wall = time.time() - t0
        if r.status_code != 200:
            return {"status": "http_error", "status_code": r.status_code, "body": r.text[:4000], "wall_s": wall}
        return {"status": "ok", "response": r.json(), "wall_s": wall}
    except requests.exceptions.RequestException as e:
        wall = time.time() - t0
        return {"status": "engine-crash", "error": str(e), "wall_s": wall}


def build_base_payload(request_spec):
    payload = {
        "model": config.MODEL_NAME,
        "chat_template_kwargs": dict(config.DEFAULT_CHAT_TEMPLATE_KWARGS),
    }
    payload.update(request_spec)
    return payload


def approx_tokens(s):
    return len(s) / config.CHARS_PER_TOKEN


def run_single_shot_item(item, out_dir):
    req = dict(item["request"])
    large_prompt = False

    if item["category"] == "long_ctx":
        corpus_text, _meta = corpus.build_corpus(item["corpus_tokens"])
        question = req.pop("question")
        content = corpus_text + "\n\n" + question
        req["messages"] = [{"role": "user", "content": content}]
        large_prompt = approx_tokens(content) >= config.LARGE_PROMPT_TOKEN_THRESHOLD
    else:
        large_prompt = any(
            approx_tokens(m.get("content", "")) >= config.LARGE_PROMPT_TOKEN_THRESHOLD
            for m in req.get("messages", [])
        )

    payload = build_base_payload(req)
    start_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    result = post_chat(payload)

    raw_record = {"id": item["id"], "category": item["category"], "request": payload, "result": result}

    gold = False
    if large_prompt:
        gold, jerr = capture_journal(item["id"], start_iso, out_dir)
        raw_record["journal_captured"] = gold
        if jerr:
            raw_record["journal_error"] = jerr

    if result["status"] == "engine-crash":
        recovered = wait_for_recovery()
        raw_record["recovered"] = recovered
        score = {"passed": False, "reason": "engine-crash"}
    elif result["status"] == "http_error":
        score = {"passed": False, "reason": f"http {result['status_code']}"}
    else:
        msg = result["response"]["choices"][0]["message"]
        finish_reason = result["response"]["choices"][0].get("finish_reason")
        content = msg.get("content")
        cat = item["category"]
        if cat == "tool_call":
            score = scoring.score_tool_call(item, msg)
        elif cat == "code_exec":
            score = scoring.score_code_exec(item, content)
        elif cat == "json_schema":
            score = scoring.score_json_schema(item, content)
        elif cat == "instruction":
            score = scoring.score_instruction(item, content)
        elif cat == "loop_probe":
            score = scoring.score_loop_probe(item, content, finish_reason)
        elif cat == "long_ctx":
            score = scoring.score_long_ctx(item, content)
        else:
            score = {"passed": False, "reason": f"no single-shot scorer for category {cat}"}

    raw_record["score"] = score
    (out_dir / f"{item['id']}.json").write_text(json.dumps(raw_record, indent=2, ensure_ascii=False))
    return score


def run_agentic_chain_item(item, out_dir):
    messages = [{"role": "user", "content": item["user"]}]
    steps_raw = []
    prev_inject_result = None
    aborted = False

    for i, spec in enumerate(item["steps"]):
        if aborted:
            break
        req = {
            "messages": list(messages),
            "max_tokens": spec.get("max_tokens", 300),
        }
        if not spec.get("final"):
            req["tools"] = item["tools"]
            req["tool_choice"] = "auto"
        payload = build_base_payload(req)
        result = post_chat(payload)

        if result["status"] != "ok":
            steps_raw.append({"status": result["status"], "detail": result})
            if result["status"] == "engine-crash":
                wait_for_recovery()
            aborted = True
            continue

        msg = result["response"]["choices"][0]["message"]
        steps_raw.append({"status": "ok", "response": result["response"]})
        messages.append(msg)

        if spec.get("final"):
            break

        calls = scoring.parse_tool_calls(msg)
        if not calls or len(calls) != 1:
            # can't sensibly continue the chain without exactly one call
            aborted = True
            continue
        tool_call_id = msg["tool_calls"][0].get("id")
        inject_result = spec.get("inject_result", {})
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(inject_result),
        })
        prev_inject_result = inject_result

    raw_record = {"id": item["id"], "category": "agentic_chain", "steps": steps_raw}
    score = scoring.score_agentic_chain(item, raw_record)
    raw_record["score"] = score
    (out_dir / f"{item['id']}.json").write_text(json.dumps(raw_record, indent=2, ensure_ascii=False))
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--categories", default=None, help="comma-separated category filter")
    ap.add_argument("--limit-per-category", type=int, default=None)
    args = ap.parse_args()

    categories = set(args.categories.split(",")) if args.categories else None
    items = load_items(categories, args.limit_per_category)
    if not items:
        print("No items matched -- nothing to run.", file=sys.stderr)
        sys.exit(1)

    out_dir = RESULTS_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"tag": args.tag, "started": datetime.now(timezone.utc).isoformat(), "items": []}
    t0 = time.time()

    for item in items:
        item_t0 = time.time()
        try:
            if item["category"] == "agentic_chain":
                score = run_agentic_chain_item(item, out_dir)
            else:
                score = run_single_shot_item(item, out_dir)
        except Exception as e:
            score = {"passed": False, "reason": f"runner exception: {e!r}"}
            (out_dir / f"{item['id']}.json").write_text(json.dumps(
                {"id": item["id"], "category": item["category"], "runner_exception": repr(e), "score": score},
                indent=2,
            ))
        wall = time.time() - item_t0
        passed = bool(score.get("passed"))
        summary["items"].append({
            "id": item["id"], "category": item["category"], "passed": passed, "wall_s": round(wall, 2),
        })
        print(f"[{item['id']}] passed={passed} ({wall:.1f}s)")

    total_wall = time.time() - t0
    by_cat = {}
    for r in summary["items"]:
        c = by_cat.setdefault(r["category"], {"pass": 0, "total": 0})
        c["total"] += 1
        c["pass"] += int(r["passed"])
    summary["by_category"] = by_cat
    summary["total_pass"] = sum(v["pass"] for v in by_cat.values())
    summary["total_items"] = sum(v["total"] for v in by_cat.values())
    summary["wall_s"] = round(total_wall, 2)
    summary["finished"] = datetime.now(timezone.utc).isoformat()

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{summary['total_pass']}/{summary['total_items']} passed, {total_wall:.1f}s total")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
