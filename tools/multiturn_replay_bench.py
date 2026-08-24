#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""multiturn_replay_bench.py -- F-1/M-3 acceptance-criteria instrument.

Client-side-only replay bench for the two traffic shapes analyzed in
``docs/f1-partial-prefix-hits-research.md`` Section 6 ("Bench protocol to
prove the win -- multi-turn replay TTFT per turn"):

  (a) diverged-siblings -- M independent conversations that share a large
      common preamble (repo/system context) and then branch into distinct
      tasks from turn 1. This is the shape that exposes vLLM PR #53479's
      "sparse Mamba states" bug: today a sibling that shares anything less
      than the exact deepest chunk-end position the preamble's own prefill
      happened to stop at gets ~zero reuse.
  (b) single-session-growth -- one real conversation, growing turn over
      turn (each turn resubmits the FULL prior transcript verbatim plus a
      fresh user message), context ramping from ~8K to 100K+ tokens. This
      is the shape that exposes the EAGLE/MTP one-block resume back-off
      that ``VLLM_MAMBA_ALIGN_RETAIN_MTP_CACHE_BLOCK`` already addresses.

This script ONLY sends HTTP requests to ``/v1/chat/completions`` (streaming)
on an already-running server. It never touches engine config, env vars, or
the server process -- point it at whatever is already up and it measures
what that server does today. It does not itself decide "pass/fail"; it
produces the per-turn table and summary numbers Section 6 asks for so a
baseline JSON and a post-change JSON can be diffed by hand (or with a
follow-up ``--compare``-style script, matching the pattern already used by
``tools/s4_replay_bench.py``).

Cached-token accounting requires the server to be started with
``--enable-prompt-tokens-details`` (both live serve scripts,
``deploy/bin/serve-qwen38-mtp-requal-fg.sh`` and
``deploy/bin/serve-qwen-8001.sh``, already pass this flag). Without it, the
server never populates ``usage.prompt_tokens_details.cached_tokens`` and
every turn reports ``cached_tokens=0`` -- the tool still runs (TTFT-only
mode), it just can't attribute a TTFT delta to caching.

AUTHOR-TIME NOTE: do not point this at the live :8001 server while a
benchmark window owns the engine. This file is author + py_compile only
until a window is scheduled.

Usage
-----
  # single-session growth only, 24 turns, 3 reps, against the live server
  python tools/multiturn_replay_bench.py --arm growth \\
      --base-url http://127.0.0.1:8001 --turns 24 --reps 3 \\
      --out /tmp/growth_baseline.json

  # diverged siblings only, 6 parallel conversations off a 30K preamble
  python tools/multiturn_replay_bench.py --arm siblings \\
      --base-url http://127.0.0.1:8001 --siblings 6 --reps 3 \\
      --out /tmp/siblings_baseline.json

  # both arms in one run (default), quick smoke-sized (fast, low fidelity)
  python tools/multiturn_replay_bench.py --base-url http://127.0.0.1:8001 \\
      --turns 6 --siblings 2 --reps 1 --out /tmp/replay_smoke.json

  # full default run, both arms, 3 reps (this is the real acceptance run --
  # budget it a proper window; see runtime note in --help)
  python tools/multiturn_replay_bench.py --base-url http://127.0.0.1:8001 \\
      --out /tmp/replay_baseline.json

Retention/port A/B (doc Section 6, arms 1-4): run this script unmodified
against each server config in turn, one JSON per config, same flags:

  1. current code, baseline                        -> replay_1_baseline.json
  2. + VLLM_MAMBA_ALIGN_RETAIN_MTP_CACHE_BLOCK=1     -> replay_2_retain_mtp.json
  3. + ported #53479 (both hunks, atomically)        -> replay_3_pr53479.json
  4. (3) + feat-retention-interval set dense          -> replay_4_dense_retention.json

Diff ``growth.summary`` and ``siblings.summary`` across the JSON files: a
real win shows up as a mean-TTFT drop for growth turns 2..N and/or a
``total_cached_fraction`` increase for siblings, not as noise inside one
run's own rep spread (median-over-3-reps is meant to filter that out).

Only the Python stdlib is used (urllib, concurrent.futures) so this runs in
any venv without extra dependencies.
"""

import argparse
import concurrent.futures
import json
import random
import statistics
import sys
import time
import urllib.request
from typing import Any

# --------------------------------------------------------------------------
# Deterministic, synthetic-but-realistic "code review" text generator.
#
# NOT an LLM, not a corpus -- a small deterministic grammar over vocab lists
# so the exact same --seed always produces the exact same session content,
# which is what makes rep-to-rep and baseline-vs-change comparisons mean
# anything. random.Random(seed) (NOT Python's built-in hash()/PYTHONHASHSEED,
# which is randomized per-process by default) is the only source of
# randomness anywhere in this file.
# --------------------------------------------------------------------------

_CHARS_PER_TOKEN = 3.8  # rough English-prose heuristic, NOT real tokenization.
# All growth/preamble token targets below are therefore approximate by
# construction. This is fine: the per-turn table's ctx_len column is always
# the SERVER's own ground-truth usage.prompt_tokens, never this estimate --
# the estimate only steers how much text we generate and how big max_tokens
# requests are, so the growth curve lands in the right neighborhood without
# this script needing a tokenizer/model-dir dependency.

_FILES = [
    "scheduler.py", "single_type_kv_cache_manager.py", "mamba_utils.py",
    "cache_engine.py", "block_pool.py", "kv_cache_manager.py",
    "sequence.py", "worker.py", "model_runner.py", "attention.py",
    "sampler.py", "logits_processor.py", "tokenizer_group.py",
    "async_llm_engine.py", "output_processor.py", "request.py",
]

_SYMBOLS = [
    "_mamba_block_aligned_split", "reachable_block_mask", "allocate_new_blocks",
    "postprocess_mamba", "last_cache_position", "chunk_end", "prefill_end",
    "num_cached_tokens", "retain_mamba_align_mtp_cache_block", "block_size",
    "compute_hash", "free_block_queue", "req_to_block_hashes", "kv_cache_groups",
    "num_computed_tokens", "cached_block_hash_to_block", "eviction_policy",
]

_REVIEW_OPENERS = [
    "In `{file}`, the `{sym}` path looks correct on the happy path, but",
    "Looking at `{sym}` more closely,",
    "Small nit on `{file}`:",
    "I traced through `{sym}` and I think",
    "Re: the diff touching `{file}` --",
    "One concern with this change to `{sym}`:",
    "This mostly matches what I'd expect from `{file}`, except",
    "Can you double check `{sym}` against the case where",
]

_REVIEW_BODIES = [
    "the boundary condition at the block edge isn't obviously covered by "
    "the existing tests, and a one-off here would silently regress "
    "prefix-cache hit rate rather than crash, which makes it easy to miss "
    "in review.",
    "there's an implicit assumption that `{sym2}` is monotonic across "
    "scheduler steps; if a request gets preempted and resumed, does that "
    "invariant still hold, or does `{file2}` need a guard for that path too?",
    "the retry/backoff interacts with the cache-hit accounting in a way "
    "that isn't obvious from the diff alone -- worth a comment explaining "
    "why the off-by-one is intentional here, since the next person to "
    "touch this will assume it's a bug.",
    "we should add a regression test that pins the exact token count at "
    "the chunk boundary, the same way `{file2}` does for the aligned-prompt "
    "case, so a future refactor of `{sym2}` can't silently reintroduce this.",
    "I'd rather this stayed a narrow, function-scoped change than grow "
    "into a general utility right now -- we don't have a second caller yet "
    "and the abstraction would just be guessing at what `{sym2}` needs.",
    "performance-wise this should be a wash (same number of hash lookups), "
    "but it's worth confirming with a quick before/after on the replay "
    "bench rather than reasoning about it from the diff.",
    "this changes observable behavior for anyone relying on `{sym2}` "
    "staying stable across a resumed request, so it probably needs a "
    "changelog line even though the code diff itself is small.",
]

_REVIEW_CLOSERS = [
    "Overall looks reasonable, just flagging for discussion before we land it.",
    "Not a blocker, but let's not merge without at least a short test.",
    "LGTM once the comment above is addressed.",
    "Happy to pair on this if it's easier to talk through than to write up.",
    "Curious what `{sym}` does under concurrent access here -- did you test that path?",
]


def _fake_diff_hunk(rng: random.Random) -> str:
    file = rng.choice(_FILES)
    sym = rng.choice(_SYMBOLS)
    return (
        f"```diff\n--- a/vllm/v1/core/sched/{file}\n+++ b/vllm/v1/core/sched/{file}\n"
        f"@@ def _mamba_block_aligned_split(...):\n"
        f"-    if self.use_eagle:\n"
        f"-        last_cache_position -= block_size\n"
        f"+    if self.use_eagle and not {sym}:\n"
        f"+        last_cache_position = max(last_cache_position - block_size, 0)\n"
        f"```"
    )


def _review_paragraph(rng: random.Random) -> str:
    file = rng.choice(_FILES)
    file2 = rng.choice(_FILES)
    sym = rng.choice(_SYMBOLS)
    sym2 = rng.choice(_SYMBOLS)
    opener = rng.choice(_REVIEW_OPENERS).format(file=file, sym=sym)
    body = rng.choice(_REVIEW_BODIES).format(file2=file2, sym2=sym2)
    parts = [opener + " " + body]
    if rng.random() < 0.35:
        parts.append(_fake_diff_hunk(rng))
    if rng.random() < 0.6:
        parts.append(rng.choice(_REVIEW_CLOSERS).format(sym=sym))
    return "\n\n".join(parts)


def _synthetic_text(rng: random.Random, target_tokens: int) -> str:
    """Deterministic code-review-ish text of approximately target_tokens."""
    target_tokens = max(1, int(target_tokens))
    target_chars = max(1, int(target_tokens * _CHARS_PER_TOKEN))
    out = []
    total = 0
    while total < target_chars:
        p = _review_paragraph(rng)
        out.append(p)
        total += len(p) + 2
    text = "\n\n".join(out)
    if len(text) > target_chars:
        text = text[:target_chars]
    return text


def _sub_seed(seed: int, tag: str, rep: int) -> int:
    """Stable deterministic sub-seed, independent of PYTHONHASHSEED.

    Python's built-in hash() randomizes str hashing per-process by default,
    which would silently break the "same --seed -> same session content"
    contract (b) requires. This is a plain FNV-1a mix -- not cryptographic,
    just stable.
    """
    h = 2166136261
    for ch in f"{seed}:{tag}:{rep}":
        h = (h ^ ord(ch)) * 16777619 & 0xFFFFFFFF
    return h


# --------------------------------------------------------------------------
# HTTP: streaming /v1/chat/completions, TTFT + usage capture.
# --------------------------------------------------------------------------


def _stream_chat(
    base_url: str,
    model: str,
    messages: list,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    """POST /v1/chat/completions with stream=True; return timing + usage.

    TTFT is measured as the wall-clock time of the first SSE chunk that
    carries actual delta content -- not a role-only preamble chunk, and not
    the trailing usage-only chunk -- i.e. "first (content) chunk time",
    matching tools/s4_replay_bench.py's convention so numbers from both
    tools are comparable. stream_options.include_usage=True is required to
    get the final usage chunk at all; without it prompt_tokens/cached_tokens
    would be unavailable in streaming mode.
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.monotonic()
    t_first = None
    usage: dict[str, Any] | None = None
    text_parts: list[str] = []
    error = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        if t_first is None:
                            t_first = time.monotonic()
                        text_parts.append(content)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    t_end = time.monotonic()
    ttft_s = (t_first - t0) if t_first is not None else None
    prompt_tokens = None
    completion_tokens = None
    cached_tokens = 0
    if usage:
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        details = usage.get("prompt_tokens_details") or {}
        cached_tokens = details.get("cached_tokens") or 0
    return {
        "error": error,
        "ttft_s": ttft_s,
        "elapsed_s": t_end - t0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "text": "".join(text_parts),
    }


def _turn_record(turn: int, result: dict[str, Any], **extra) -> dict[str, Any]:
    ctx_len = result["prompt_tokens"] or 0
    cached = result["cached_tokens"] or 0
    ttft = result["ttft_s"]
    uncached = max(ctx_len - cached, 0)
    row = {
        "turn": turn,
        "ctx_len": ctx_len,
        "ttft_s": ttft,
        "cached_tokens": cached,
        "cached_frac": (cached / ctx_len) if ctx_len else 0.0,
        "prefill_tok_s_effective": (uncached / ttft) if ttft else None,
        "completion_tokens": result["completion_tokens"] or 0,
    }
    row.update(extra)
    return row


# --------------------------------------------------------------------------
# Arm (a): single-session-growth.
# --------------------------------------------------------------------------


def _target_ctx(t: int, turns: int, start_ctx: int, end_ctx: int) -> float:
    """Target context size (tokens) once turn t's assistant reply lands.

    t=0 means "right after the seed exchange" (== start_ctx). t=turns hits
    end_ctx exactly. Linear ramp in between.
    """
    if turns <= 0:
        return float(end_ctx)
    frac = t / turns
    return start_ctx + (end_ctx - start_ctx) * frac


def _run_growth_session(
    base_url: str,
    model: str,
    turns: int,
    start_ctx: int,
    end_ctx: int,
    user_tail_range: tuple[int, int],
    assistant_cap: int,
    timeout: float,
    rng: random.Random,
    salt: str,
) -> list[dict[str, Any]]:
    """One single-session-growth conversation.

    A seed exchange (recorded as turn 0, excluded from summary means)
    bootstraps the context up near start_ctx, then `turns` real turns, each
    a genuinely fresh ~user_tail_range-token user message appended to the
    FULL prior transcript (system + every previous user/assistant message,
    verbatim -- this is what makes it a same-lineage RESUME, never a
    rewrite of history, which is the thing the EAGLE/MTP back-off tax
    actually applies to). Assistant reply length per turn is chosen
    adaptively off the server's own last-measured prompt_tokens, so the
    session's real context size tracks the start_ctx -> end_ctx ramp
    without this script needing a tokenizer.
    """
    system_prompt = (
        f"SESSION SALT: {salt}\n"
        "You are a senior engineer in an ongoing code-review conversation "
        "spanning a multi-file change. Be concise and specific, and "
        "reference exact symbol/file names when you comment."
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    records: list[dict[str, Any]] = []

    # --- turn 0: seed exchange, bootstraps ctx up near start_ctx ---
    seed_text = _synthetic_text(rng, start_ctx)
    messages.append({"role": "user", "content": "SESSION CONTEXT:\n\n" + seed_text})
    result = _stream_chat(base_url, model, messages, 64, timeout)
    if result["error"]:
        print(f"[growth] seed turn error: {result['error']}", file=sys.stderr)
        return records
    records.append(_turn_record(0, result))
    messages.append({"role": "assistant", "content": result["text"] or "Understood."})
    prev_actual = (result["prompt_tokens"] or 0) + (result["completion_tokens"] or 0)

    # --- turns 1..N: real, fresh ~user_tail_range-token turns ---
    for t in range(1, turns + 1):
        tail_tokens = rng.randint(*user_tail_range)
        messages.append({"role": "user", "content": _synthetic_text(rng, tail_tokens)})

        target_next = _target_ctx(t, turns, start_ctx, end_ctx)
        projected_this_turn_ctx = prev_actual + tail_tokens
        assistant_max_tokens = int(
            max(64, min(assistant_cap, target_next - projected_this_turn_ctx))
        )
        if t == turns:
            # No further turn depends on this reply's length; don't pay for
            # generation the growth curve no longer needs.
            assistant_max_tokens = min(assistant_max_tokens, 512)

        result = _stream_chat(base_url, model, messages, assistant_max_tokens, timeout)
        if result["error"]:
            print(f"[growth] turn {t} error: {result['error']}", file=sys.stderr)
            break

        records.append(_turn_record(t, result))
        messages.append({"role": "assistant", "content": result["text"] or ""})
        prev_actual = (result["prompt_tokens"] or 0) + (result["completion_tokens"] or 0)

    return records


# --------------------------------------------------------------------------
# Arm (b): diverged-siblings.
# --------------------------------------------------------------------------


def _run_one_sibling(
    base_url: str,
    model: str,
    sibling_idx: int,
    preamble_messages: list[dict[str, str]],
    task_seed: int,
    sibling_turns: int,
    user_tail_range: tuple[int, int],
    assistant_cap: int,
    timeout: float,
) -> list[dict[str, Any]]:
    """One sibling: preamble (shared, passed in) + a distinct branch task.

    Runs in its own thread (see _run_siblings_batch) so siblings are
    genuinely in flight together, matching the real "N subagents fan out
    from a common context" traffic this arm targets -- not sequential
    replay of the same requests.
    """
    rng = random.Random(task_seed)
    task_text = _synthetic_text(rng, rng.randint(*user_tail_range))
    task_text = f"YOUR TASK (sibling {sibling_idx}, distinct from the others):\n\n" + task_text
    messages = list(preamble_messages) + [{"role": "user", "content": task_text}]
    records: list[dict[str, Any]] = []

    for t in range(1, sibling_turns + 1):
        if t > 1:
            tail_tokens = rng.randint(*user_tail_range)
            messages.append(
                {"role": "user", "content": _synthetic_text(rng, tail_tokens)}
            )
        result = _stream_chat(base_url, model, messages, assistant_cap, timeout)
        if result["error"]:
            print(
                f"[siblings] sibling {sibling_idx} turn {t} error: {result['error']}",
                file=sys.stderr,
            )
            break
        records.append(_turn_record(t, result, sibling=sibling_idx))
        messages.append({"role": "assistant", "content": result["text"] or ""})

    return records


def _run_siblings_batch(
    base_url: str,
    model: str,
    siblings: int,
    preamble_tokens: int,
    sibling_turns: int,
    user_tail_range: tuple[int, int],
    assistant_cap: int,
    timeout: float,
    rng: random.Random,
    salt: str,
    workers: int,
) -> list[dict[str, Any]]:
    """M siblings sharing one common preamble, dispatched concurrently."""
    system_prompt = (
        f"BATCH SALT: {salt}\n"
        "You are a senior engineer. Several teammates are independently "
        "reviewing different parts of the same change described below; "
        "answer only the specific task you personally are given."
    )
    preamble_text = _synthetic_text(rng, preamble_tokens)
    preamble_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "SHARED CONTEXT:\n\n" + preamble_text},
        {"role": "assistant", "content": "Understood, I have the shared context."},
    ]

    # Draw all per-sibling task seeds up front, sequentially, in the main
    # thread -- random.Random is not thread-safe, and this keeps content
    # fully deterministic regardless of thread scheduling order.
    task_seeds = [rng.randint(0, 2**31 - 1) for _ in range(siblings)]

    results: list[list[dict[str, Any]]] = [[] for _ in range(siblings)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {
            ex.submit(
                _run_one_sibling,
                base_url,
                model,
                i,
                preamble_messages,
                task_seeds[i],
                sibling_turns,
                user_tail_range,
                assistant_cap,
                timeout,
            ): i
            for i in range(siblings)
        }
        for fut in concurrent.futures.as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"[siblings] sibling {i} failed: {exc}", file=sys.stderr)
                results[i] = []

    return [rec for recs in results for rec in recs]


# --------------------------------------------------------------------------
# Aggregation: N reps -> median per (turn) / (sibling, turn).
# --------------------------------------------------------------------------

_VALUE_FIELDS = (
    "ctx_len",
    "ttft_s",
    "cached_tokens",
    "cached_frac",
    "prefill_tok_s_effective",
    "completion_tokens",
)


def _median_by_key(
    rep_lists: list[list[dict[str, Any]]],
    key_fields: tuple[str, ...],
    value_fields: tuple[str, ...] = _VALUE_FIELDS,
) -> list[dict[str, Any]]:
    """Group records from N reps by key_fields, median each value field."""
    buckets: dict[tuple, list[dict[str, Any]]] = {}
    for rep in rep_lists:
        for rec in rep:
            key = tuple(rec[k] for k in key_fields)
            buckets.setdefault(key, []).append(rec)
    out = []
    for key in sorted(buckets):
        recs = buckets[key]
        row: dict[str, Any] = dict(zip(key_fields, key))
        for vf in value_fields:
            vals = [r[vf] for r in recs if r.get(vf) is not None]
            row[vf] = statistics.median(vals) if vals else None
        row["n_reps"] = len(recs)
        out.append(row)
    # Integer-valued fields come back as float medians (statistics.median of
    # an even-length list averages the two middle values) -- round for
    # sane display/serialization.
    for row in out:
        for k in ("ctx_len", "cached_tokens", "completion_tokens"):
            if row.get(k) is not None:
                row[k] = int(round(row[k]))
    return out


def _growth_summary(median_rows: list[dict[str, Any]], turns: int) -> dict[str, Any]:
    real_turns = [r for r in median_rows if r["turn"] >= 1]
    tail_turns = [r for r in real_turns if r["turn"] >= 2]
    ttfts = [r["ttft_s"] for r in tail_turns if r["ttft_s"] is not None]
    total_ctx = sum(r["ctx_len"] for r in real_turns)
    total_cached = sum(r["cached_tokens"] for r in real_turns)
    return {
        "mean_ttft_s_turns_2_to_n": statistics.mean(ttfts) if ttfts else None,
        "n_turns_in_mean": len(ttfts),
        "total_cached_fraction": (total_cached / total_ctx) if total_ctx else None,
        "turn1_ctx_len": next((r["ctx_len"] for r in real_turns if r["turn"] == 1), None),
        "final_ctx_len": real_turns[-1]["ctx_len"] if real_turns else None,
        "turns_completed": len(real_turns),
        "turns_requested": turns,
    }


def _siblings_summary(median_rows: list[dict[str, Any]], siblings: int) -> dict[str, Any]:
    ttfts = [r["ttft_s"] for r in median_rows if r["ttft_s"] is not None]
    total_ctx = sum(r["ctx_len"] for r in median_rows)
    total_cached = sum(r["cached_tokens"] for r in median_rows)
    n_first_turn = len({r["sibling"] for r in median_rows if r["turn"] == 1})
    return {
        "mean_ttft_s": statistics.mean(ttfts) if ttfts else None,
        "n_records_in_mean": len(ttfts),
        "total_cached_fraction": (total_cached / total_ctx) if total_ctx else None,
        "siblings_completed": n_first_turn,
        "siblings_requested": siblings,
    }


# --------------------------------------------------------------------------
# Printing.
# --------------------------------------------------------------------------


def _fmt(x, spec: str) -> str:
    return format(x, spec) if x is not None else "n/a"


def _print_growth_table(median_rows: list[dict[str, Any]]) -> None:
    print("\n=== growth arm: per-turn (median over reps) ===")
    print(
        f"{'turn':>5s} {'ctx_len':>9s} {'ttft_s':>8s} {'cached':>9s} "
        f"{'cached%':>8s} {'prefill_tok/s_eff':>18s}"
    )
    for r in median_rows:
        tag = " (seed)" if r["turn"] == 0 else ""
        cf = f"{r['cached_frac'] * 100:.1f}%" if r.get("cached_frac") is not None else "n/a"
        print(
            f"{r['turn']:>5d} {r['ctx_len']:>9d} {_fmt(r['ttft_s'], '8.3f')} "
            f"{r['cached_tokens']:>9d} {cf:>8s} "
            f"{_fmt(r['prefill_tok_s_effective'], '18.1f')}{tag}"
        )


def _print_siblings_table(median_rows: list[dict[str, Any]]) -> None:
    print("\n=== siblings arm: per-(sibling,turn) (median over reps) ===")
    print(
        f"{'sib':>4s} {'turn':>5s} {'ctx_len':>9s} {'ttft_s':>8s} {'cached':>9s} "
        f"{'cached%':>8s} {'prefill_tok/s_eff':>18s}"
    )
    for r in median_rows:
        cf = f"{r['cached_frac'] * 100:.1f}%" if r.get("cached_frac") is not None else "n/a"
        print(
            f"{r['sibling']:>4d} {r['turn']:>5d} {r['ctx_len']:>9d} "
            f"{_fmt(r['ttft_s'], '8.3f')} {r['cached_tokens']:>9d} {cf:>8s} "
            f"{_fmt(r['prefill_tok_s_effective'], '18.1f')}"
        )


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:8001")
    ap.add_argument(
        "--model",
        default="qwen-local",
        help="served-model-name (matches --served-model-name on the live serve scripts)",
    )
    ap.add_argument("--turns", type=int, default=24, help="growth arm: real turns after the seed exchange")
    ap.add_argument("--siblings", type=int, default=6, help="siblings arm: number of parallel diverged conversations")
    ap.add_argument(
        "--sibling-turns",
        type=int,
        default=1,
        help="siblings arm: turns per sibling after divergence (default 1: single-shot branch, matching doc Section 6)",
    )
    ap.add_argument("--arm", choices=["growth", "siblings", "both"], default="both")
    ap.add_argument("--start-ctx-tokens", type=int, default=8000, help="growth arm: starting context size (~tokens)")
    ap.add_argument("--end-ctx-tokens", type=int, default=110000, help="growth arm: ending context size (~tokens)")
    ap.add_argument("--preamble-tokens", type=int, default=30000, help="siblings arm: shared preamble size (~tokens)")
    ap.add_argument("--user-tail-min", type=int, default=1000, help="fresh per-turn user message: min tokens (~)")
    ap.add_argument("--user-tail-max", type=int, default=2000, help="fresh per-turn user message: max tokens (~)")
    ap.add_argument(
        "--assistant-max-tokens-cap",
        type=int,
        default=3500,
        help="hard cap on assistant reply length per turn (both arms); "
        "growth arm targets this only as needed to hit --end-ctx-tokens",
    )
    ap.add_argument(
        "--sibling-concurrency",
        type=int,
        default=0,
        help="max siblings in flight at once (0 = all of --siblings at once, i.e. fully parallel)",
    )
    ap.add_argument("--reps", type=int, default=3, help="independent reps per arm; reported as the median")
    ap.add_argument("--seed", type=int, default=1337, help="deterministic seed for all synthetic text")
    ap.add_argument("--timeout", type=float, default=1800.0, help="per-request socket timeout, seconds")
    ap.add_argument("--out", help="write full results JSON here (raw reps + medians + summaries)")
    return ap


def main() -> int:
    ap = build_arg_parser()
    args = ap.parse_args()
    if args.turns < 1:
        ap.error("--turns must be >= 1")
    if args.siblings < 1:
        ap.error("--siblings must be >= 1")
    if args.reps < 1:
        ap.error("--reps must be >= 1")
    if args.user_tail_min > args.user_tail_max:
        ap.error("--user-tail-min must be <= --user-tail-max")

    user_tail_range = (args.user_tail_min, args.user_tail_max)
    workers = args.sibling_concurrency or args.siblings

    out: dict[str, Any] = {
        "base_url": args.base_url,
        "model": args.model,
        "reps": args.reps,
        "seed": args.seed,
        "arm": args.arm,
    }

    if args.arm in ("growth", "both"):
        rep_lists = []
        for r in range(args.reps):
            salt = f"{args.seed}-growth-rep{r}"
            rng = random.Random(_sub_seed(args.seed, "growth", r))
            recs = _run_growth_session(
                args.base_url,
                args.model,
                args.turns,
                args.start_ctx_tokens,
                args.end_ctx_tokens,
                user_tail_range,
                args.assistant_max_tokens_cap,
                args.timeout,
                rng,
                salt,
            )
            rep_lists.append(recs)
            print(f"[growth] rep {r} done: {len(recs)} turns recorded", file=sys.stderr)
        median_rows = _median_by_key(rep_lists, ("turn",))
        _print_growth_table(median_rows)
        summary = _growth_summary(median_rows, args.turns)
        if summary["mean_ttft_s_turns_2_to_n"] is not None and summary["total_cached_fraction"] is not None:
            print(
                f"\n-- growth summary -- mean TTFT (turns 2..{args.turns}): "
                f"{summary['mean_ttft_s_turns_2_to_n']:.3f}s over {summary['n_turns_in_mean']} turns   "
                f"total cached fraction: {summary['total_cached_fraction'] * 100:.1f}%   "
                f"ctx {summary['turn1_ctx_len']} -> {summary['final_ctx_len']} tokens "
                f"({summary['turns_completed']}/{summary['turns_requested']} turns completed)"
            )
        else:
            print("\n-- growth summary -- insufficient data (all turns failed?)")
        out["growth"] = {
            "turns": args.turns,
            "start_ctx_tokens": args.start_ctx_tokens,
            "end_ctx_tokens": args.end_ctx_tokens,
            "reps_raw": rep_lists,
            "median_by_turn": median_rows,
            "summary": summary,
        }

    if args.arm in ("siblings", "both"):
        rep_lists = []
        for r in range(args.reps):
            salt = f"{args.seed}-siblings-rep{r}"
            rng = random.Random(_sub_seed(args.seed, "siblings", r))
            recs = _run_siblings_batch(
                args.base_url,
                args.model,
                args.siblings,
                args.preamble_tokens,
                args.sibling_turns,
                user_tail_range,
                args.assistant_max_tokens_cap,
                args.timeout,
                rng,
                salt,
                workers,
            )
            rep_lists.append(recs)
            print(f"[siblings] rep {r} done: {len(recs)} records", file=sys.stderr)
        median_rows = _median_by_key(rep_lists, ("sibling", "turn"))
        _print_siblings_table(median_rows)
        summary = _siblings_summary(median_rows, args.siblings)
        if summary["mean_ttft_s"] is not None and summary["total_cached_fraction"] is not None:
            print(
                f"\n-- siblings summary -- mean TTFT: {summary['mean_ttft_s']:.3f}s "
                f"over {summary['n_records_in_mean']} records   "
                f"total cached fraction: {summary['total_cached_fraction'] * 100:.1f}%   "
                f"({summary['siblings_completed']}/{summary['siblings_requested']} siblings completed)"
            )
        else:
            print("\n-- siblings summary -- insufficient data (all siblings failed?)")
        out["siblings"] = {
            "siblings": args.siblings,
            "sibling_turns": args.sibling_turns,
            "preamble_tokens": args.preamble_tokens,
            "reps_raw": rep_lists,
            "median_by_sibling_turn": median_rows,
            "summary": summary,
        }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
