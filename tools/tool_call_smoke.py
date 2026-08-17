#!/usr/bin/env python3
import argparse
import json
import re
import sys
import time
from typing import Any

import requests


# Garble signatures are single-sourced in deploy/bench/bench_lib.py (review:
# the two copies had already begun to diverge — bench_lib gained
# REPEATED_PHRASE). Path-safe import since both files run as standalone
# scripts from different cwds.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "deploy" / "bench"))
from bench_lib import REPEATED_CHARS, REPEATED_TOOL_TAGS  # noqa: E402


def request_chat(
    base_url: str,
    model: str,
    message: str,
    *,
    tool_choice: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "tool_choice": tool_choice,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    }
    start = time.perf_counter()
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=payload,
        timeout=(30, timeout),
    )
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    return response.json(), elapsed


def extract_choice(data: dict[str, Any]) -> tuple[str, list[dict[str, Any]], str]:
    choices = data.get("choices") or []
    if not choices:
        raise AssertionError("response has no choices")
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    finish_reason = choice.get("finish_reason") or ""
    return content, tool_calls, finish_reason


def assert_not_degenerate(label: str, content: str) -> None:
    if REPEATED_CHARS.search(content):
        raise AssertionError(f"{label}: repeated character pattern detected")
    if REPEATED_TOOL_TAGS.search(content):
        raise AssertionError(f"{label}: repeated <tool_call> tags detected")


def summarize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, str]]:
    summary = []
    for item in tool_calls:
        function = item.get("function") or {}
        summary.append(
            {
                "name": str(function.get("name") or ""),
                "arguments": str(function.get("arguments") or ""),
            }
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--require-auto-weather-tool",
        action="store_true",
        help="Also require tool_choice=auto to emit a weather tool call.",
    )
    parser.add_argument("--json-out")
    args = parser.parse_args()

    cases = [
        ("auto_greeting", "hello", "auto", None),
        (
            "auto_weather",
            "what is the weather in tokyo? Use the get_weather tool if needed.",
            "auto",
            True if args.require_auto_weather_tool else None,
        ),
        (
            "required_weather",
            "what is the weather in tokyo?",
            "required",
            True,
        ),
    ]
    results = []
    for label, message, tool_choice, expect_tool in cases:
        data, elapsed = request_chat(
            args.base_url,
            args.model,
            message,
            tool_choice=tool_choice,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
        )
        content, tool_calls, finish_reason = extract_choice(data)
        assert_not_degenerate(label, content)
        if expect_tool is True and not tool_calls:
            raise AssertionError(f"{label}: expected at least one tool_call")
        if expect_tool is False and tool_calls:
            raise AssertionError(f"{label}: did not expect a tool_call")
        results.append(
            {
                "label": label,
                "tool_choice": tool_choice,
                "elapsed_s": elapsed,
                "finish_reason": finish_reason,
                "content_preview": content[:200],
                "tool_calls": summarize_tool_calls(tool_calls),
                "usage": data.get("usage"),
            }
        )

    output = {"ok": True, "results": results}
    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"tool_call_smoke_failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
