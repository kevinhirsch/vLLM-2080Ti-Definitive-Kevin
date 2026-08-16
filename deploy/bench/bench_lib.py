#!/usr/bin/env python3
"""Shared helpers for the Qwen 3.8 MTP re-qualification benches.

Everything talks to the ENGINE directly (:8001 by default), bypassing the
gateway so SHIM_LOCAL_BUDGET / guards don't shape the measurements. stdlib +
requests only (same dependency bar as tools/tool_call_smoke.py).
"""
import json
import re
import subprocess
import time
from typing import Any

import requests

DEFAULT_BASE = "http://127.0.0.1:8001/v1"
DEFAULT_MODEL = "qwen-local"

# Degenerate-output signatures, borrowed from tools/tool_call_smoke.py and the
# 2026-08-14 garble incident logs (mixed-script noise, tag loops).
REPEATED_CHARS = re.compile(r"(.)\1{24,}", re.DOTALL)
REPEATED_TOOL_TAGS = re.compile(r"(<tool_call>\s*){3,}", re.DOTALL)
REPEATED_PHRASE = re.compile(r"(\b.{6,48}?\b)(?:\s*\1){5,}", re.DOTALL)


def chat(base_url: str, model: str, messages: list[dict[str, Any]], *,
         max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.8,
         top_k: int = 20, seed: int | None = None, stream: bool = True,
         extra: dict[str, Any] | None = None,
         timeout: float = 900.0) -> dict[str, Any]:
    """One chat completion. Returns dict with content, ttft_s, decode_tps,
    completion_tokens, prompt_tokens, finish_reason.

    Streaming so TTFT and decode rate are measured separately -- long-context
    prefill on this box (~850-1600 tok/s) must not pollute the decode number.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stream": stream,
    }
    if seed is not None:
        payload["seed"] = seed
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if extra:
        payload.update(extra)

    start = time.perf_counter()
    if not stream:
        r = requests.post(f"{base_url.rstrip('/')}/chat/completions",
                          json=payload, timeout=timeout)
        r.raise_for_status()
        d = r.json()
        total = time.perf_counter() - start
        ch = (d.get("choices") or [{}])[0]
        usage = d.get("usage") or {}
        ct = usage.get("completion_tokens") or 0
        return {
            "content": (ch.get("message") or {}).get("content") or "",
            "reasoning": (ch.get("message") or {}).get("reasoning_content") or "",
            "finish_reason": ch.get("finish_reason"),
            "ttft_s": None,
            "total_s": total,
            "decode_tps": (ct / total) if total > 0 else 0.0,
            "completion_tokens": ct,
            "prompt_tokens": usage.get("prompt_tokens") or 0,
        }

    r = requests.post(f"{base_url.rstrip('/')}/chat/completions",
                      json=payload, timeout=timeout, stream=True)
    r.raise_for_status()
    first_tok_t = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish = None
    usage: dict[str, Any] = {}
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            d = json.loads(data)
        except json.JSONDecodeError:
            continue
        if d.get("usage"):
            usage = d["usage"]
        for ch in d.get("choices") or []:
            delta = ch.get("delta") or {}
            tok = delta.get("content") or delta.get("reasoning_content")
            if tok:
                if first_tok_t is None:
                    first_tok_t = time.perf_counter()
                (content_parts if delta.get("content")
                 else reasoning_parts).append(tok)
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    end = time.perf_counter()
    ct = usage.get("completion_tokens") or 0
    decode_window = (end - first_tok_t) if first_tok_t else 0.0
    return {
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "finish_reason": finish,
        "ttft_s": (first_tok_t - start) if first_tok_t else None,
        "total_s": end - start,
        "decode_tps": (ct / decode_window) if decode_window > 0 else 0.0,
        "completion_tokens": ct,
        "prompt_tokens": usage.get("prompt_tokens") or 0,
    }


def scrape_spec_metrics(base_url: str) -> dict[str, float]:
    """Cumulative spec-decode counters from /metrics. Take a delta around a probe
    to get per-probe acceptance: accepted_delta / draft_delta."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    out = {"drafts": 0.0, "draft_tokens": 0.0, "accepted_tokens": 0.0}
    try:
        text = requests.get(f"{root}/metrics", timeout=10).text
    except requests.RequestException:
        return out
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for key, metric in (
            ("drafts", "vllm:spec_decode_num_drafts_total"),
            ("draft_tokens", "vllm:spec_decode_num_draft_tokens_total"),
            ("accepted_tokens", "vllm:spec_decode_num_accepted_tokens_total"),
        ):
            if line.startswith(metric):
                try:
                    out[key] += float(line.rsplit(None, 1)[-1])
                except ValueError:
                    pass
    return out


def acceptance_rate(before: dict[str, float], after: dict[str, float]) -> float | None:
    drafted = after["draft_tokens"] - before["draft_tokens"]
    if drafted <= 0:
        return None
    return (after["accepted_tokens"] - before["accepted_tokens"]) / drafted


# ---------------------------------------------------------------------------
# Long-context prompt construction + garble detection
# ---------------------------------------------------------------------------

_FILLER_PARAGRAPH = (
    "Section {i}. The service mesh forwards each request through the ingress "
    "tier, applies the retry budget configured for the route, and records the "
    "outcome in the regional ledger. Operators review the ledger during the "
    "weekly capacity meeting and adjust the shard weights when the p99 latency "
    "drifts above the objective. No configuration change ships without a "
    "canary window and a rollback owner. "
)


def build_context_prompt(target_tokens: int, needle: str) -> str:
    """Deterministic synthetic document of ~target_tokens (est. 4 chars/token)
    with a recall needle buried at ~25% depth."""
    target_chars = target_tokens * 4
    parts: list[str] = []
    total = 0
    i = 0
    needle_at = target_chars // 4
    needle_placed = False
    while total < target_chars:
        if not needle_placed and total >= needle_at:
            marker = f"\nAUDIT MARKER: {needle}\n"
            parts.append(marker)
            total += len(marker)
            needle_placed = True
            continue
        p = _FILLER_PARAGRAPH.format(i=i)
        parts.append(p)
        total += len(p)
        i += 1
    return "".join(parts)


def garble_score(text: str) -> dict[str, Any]:
    """Heuristics for the 2026-08-14 failure signature. Any flag true = FAIL."""
    non_ascii = sum(1 for c in text if ord(c) > 0x2FFF)
    ratio = (non_ascii / len(text)) if text else 0.0
    flags = {
        "repeated_chars": bool(REPEATED_CHARS.search(text)),
        "tool_tag_loop": bool(REPEATED_TOOL_TAGS.search(text)),
        "phrase_loop": bool(REPEATED_PHRASE.search(text)),
        "cjk_noise": ratio > 0.10,  # English probe answering >10% CJK/symbols
    }
    return {"flags": flags, "non_ascii_ratio": round(ratio, 4),
            "garbled": any(flags.values())}


def nrestarts(service: str = "vllm-qwen27b") -> int | None:
    try:
        out = subprocess.run(
            ["systemctl", "show", service, "-p", "NRestarts", "--value"],
            capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip())
    except Exception:
        return None
