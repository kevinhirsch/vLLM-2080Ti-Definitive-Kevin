#!/usr/bin/env python3
"""
Crash-tolerant live runner against the :8001 vLLM engine.

Loads items/<category>/*.json, builds an OpenAI-compatible chat request per
item, POSTs it to config.BASE_URL, scores the response immediately via
scoring.py, and saves EVERY raw response (success, HTTP error, or crash) to
results/<tag>/<item_id>.json so score_only.py can independently re-derive
pass/fail later without hitting the network again.

Generation goes directly to :8001, using config.BASE_URL. The runner reads
the gateway offline fence before generation; it never routes inference via
the gateway.

Usage:
    python3 run_eval.py --tag smoke --categories tool_call,instruction
    python3 run_eval.py --tag full --profile profiles/greedy.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

import config
import corpus
import scoring
import measurement

ROOT = Path(__file__).resolve().parent
ITEMS_DIR = ROOT / "items"
RESULTS_DIR = ROOT / "results"

CATEGORY_ORDER = [
    "tool_call", "code_exec", "json_schema", "instruction",
    "loop_probe", "long_ctx", "agentic_chain",
]


def load_items(categories=None, limit_per_category=None):
    if categories and set(categories) - set(CATEGORY_ORDER):
        raise ValueError(f"Unknown categories: {sorted(set(categories) - set(CATEGORY_ORDER))}")
    seen = set()
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
            item = json.loads(f.read_text())
            if item["id"] in seen:
                raise ValueError(f"Duplicate fixture ID {item['id']!r}; results would overwrite")
            seen.add(item["id"])
            if item["category"] != cat:
                raise ValueError(f"Fixture category disagrees with directory: {f}")
            if not item["id"].replace("_", "").replace("-", "").isalnum():
                raise ValueError(f"Unsafe fixture ID {item['id']!r}")
            items.append(item)
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
    return measurement.post_chat(config.BASE_URL, payload, timeout_s)


def build_base_payload(request_spec, profile=None):
    return measurement.merge_request({
        "model": config.MODEL_NAME,
        "chat_template_kwargs": dict(config.DEFAULT_CHAT_TEMPLATE_KWARGS),
    }, request_spec, profile)


def fixture_manifest(items, profile=None):
    """Offline validation/preparation: no inference or health requests."""
    rows = []
    for item in items:
        req = item.get("request", {})
        if item["category"] == "agentic_chain":
            if not item.get("steps") or not item.get("tools"):
                raise ValueError(f"Invalid chain fixture {item['id']}")
        elif not req or "check" not in item:
            raise ValueError(f"Invalid fixture {item['id']}")
        rows.append({"id": item["id"], "category": item["category"],
                     "fixture_sha256": measurement.canonical_hash(item),
                     "sampling": {k: v for k, v in build_base_payload(req, profile).items()
                                  if k in measurement.SAMPLING_KEYS}})
    counts = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    return {"total_items": len(rows), "by_category": counts, "fixtures": rows,
            "manifest_sha256": measurement.canonical_hash(rows), "profile": profile or {}}


def approx_tokens(s):
    return len(s) / config.CHARS_PER_TOKEN


def run_single_shot_item(item, out_dir, profile=None):
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

    payload = build_base_payload(req, profile)
    start_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    result = post_chat(payload, timeout_s=item.get("timeout_s", 180))

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
    elif result["status"] != "ok" or measurement.generation_error(result.get("response")):
        score = {"passed": False, "reason": result.get("error") or measurement.generation_error(result.get("response")) or result["status"]}
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


def run_agentic_chain_item(item, out_dir, profile=None):
    messages = [{"role": "user", "content": item["user"]}]
    steps_raw = []
    prev_inject_result = None
    aborted = False

    for i, spec in enumerate(item["steps"]):
        if aborted:
            break
        req = {
            **item.get("request", {}),
            **spec.get("request", {}),
            "messages": list(messages),
            "max_tokens": spec.get("max_tokens", 300),
        }
        if not spec.get("final"):
            req["tools"] = item["tools"]
            req["tool_choice"] = "auto"
        payload = build_base_payload(req, profile)
        result = post_chat(payload, timeout_s=spec.get("timeout_s", item.get("timeout_s", 180)))
        error = measurement.generation_error(result.get("response")) if result["status"] == "ok" else None

        if result["status"] != "ok" or error:
            steps_raw.append({"status": "invalid_generation" if error else result["status"], "request": payload, "detail": result})
            if result["status"] == "engine-crash":
                wait_for_recovery()
            aborted = True
            continue

        msg = result["response"]["choices"][0]["message"]
        steps_raw.append({"status": "ok", "request": payload, "response": result["response"],
                          "timeline": result.get("timeline"), "wall_s": result.get("wall_s")})
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
    ap.add_argument("--tag")
    ap.add_argument("--native-manifest", default=os.environ.get("WINDOW_NATIVE_MANIFEST"), help="explicit legacy controller snapshot; immutable roots use RELEASE.json")
    ap.add_argument("--engine-identity-file", default=os.environ.get("WINDOW_ENGINE_IDENTITY_FILE"))
    ap.add_argument("--profile", help="JSON sampling profile applied to every item and agentic step")
    ap.add_argument("--validate-fixtures", action="store_true", help="offline fixture validation; no engine calls")
    ap.add_argument("--prepare", type=Path, help="write offline baseline fixture/settings manifest")
    ap.add_argument("--engine-pid", type=int, default=int(os.environ.get("WINDOW_ENGINE_PID", "0")) or None, help="required live engine PID for process/checkpoint provenance")
    ap.add_argument("--categories", default=None, help="comma-separated category filter")
    ap.add_argument("--limit-per-category", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                     help="clear pre-existing results under --tag instead of failing")
    args = ap.parse_args()

    profile = measurement.load_profile(args.profile)
    categories = set(args.categories.split(",")) if args.categories else None
    items = load_items(categories, args.limit_per_category)
    if not items:
        print("No items matched -- nothing to run.", file=sys.stderr)
        sys.exit(1)

    manifest = fixture_manifest(items, profile)
    if args.validate_fixtures or args.prepare:
        if args.prepare:
            args.prepare.parent.mkdir(parents=True, exist_ok=True)
            args.prepare.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({k: manifest[k] for k in ("total_items", "by_category", "manifest_sha256")}))
        return 0
    if not args.tag or Path(args.tag).name != args.tag:
        ap.error("Use a simple --tag for live runs")
    if not args.engine_pid:
        ap.error("Live runs require --engine-pid; offline --validate-fixtures does not")
    provenance = measurement.runtime_identity(args.engine_pid)
    provenance["model_roots"] = measurement.verify_models_root(config.BASE_URL, provenance)
    measurement.require_offline_window(drained=True)
    provenance["native_attestation"] = measurement.verify_loaded_native(provenance, args.engine_identity_file, args.native_manifest)
    provenance["measurement_files"] = [measurement.file_identity(ROOT / name) for name in
        ("run_eval.py", "measurement.py", "scoring.py", "predicates.py", "config.py", "corpus.py")]
    out_dir = RESULTS_DIR / args.tag
    if out_dir.exists() and any(out_dir.glob("*.json")):
        if args.force:
            shutil.rmtree(out_dir)
        else:
            print(
                f"Results dir {out_dir} already has result files -- "
                "pick a fresh --tag or pass --force to clear it.",
                file=sys.stderr,
            )
            sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"tag": args.tag, "started": datetime.now(timezone.utc).isoformat(), "items": [], "fixture_manifest": manifest, "provenance": provenance}
    t0 = time.time()

    for item in items:
        item_t0 = time.time()
        try:
            measurement.require_offline_window(drained=True)
            measurement.verify_loaded_native(provenance, args.engine_identity_file, args.native_manifest)
            if item["category"] == "agentic_chain":
                score = run_agentic_chain_item(item, out_dir, profile)
            else:
                score = run_single_shot_item(item, out_dir, profile)
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
    try:
        summary["runtime_unchanged"] = measurement.runtime_identity(args.engine_pid)["fingerprint"] == provenance["fingerprint"]
    except (OSError, ValueError):
        summary["runtime_unchanged"] = False
    if not summary["runtime_unchanged"]:
        summary["valid_run"] = False
        summary["invalid_reason"] = "engine/source/config identity changed during run"
    else:
        summary["valid_run"] = True

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{summary['total_pass']}/{summary['total_items']} passed, {total_wall:.1f}s total")
    print(f"Results: {out_dir}")
    return 0 if summary["valid_run"] and summary["total_pass"] == summary["total_items"] else 1


if __name__ == "__main__":
    sys.exit(main())
