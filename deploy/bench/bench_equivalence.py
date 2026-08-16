#!/usr/bin/env python3
"""Spec-decode losslessness check: greedy output with MTP ON must equal greedy
output with MTP OFF (speculative decoding is verify-and-reject; a divergence
means the verify path is broken, which is exactly what garbled 2026-08-14).

Two engines can't run side-by-side on this box, so this is record/replay:

    # 1. against the CURRENT MTP-off engine (the trusted reference):
    python3 bench_equivalence.py record --out ref-mtp-off.json
    # 2. switch to the requal serve variant, then:
    python3 bench_equivalence.py check --ref ref-mtp-off.json

Prompts cover the failure surface: short prose, code emission, a 60K-context
recall (the old garble band), and a tool-flavored prompt. Greedy (temperature
0) is deliberate: Qwen discourages it for QUALITY, but for equivalence it is
the correct instrument — sampling would mask real divergence. Tiny numeric
drift can legitimately flip a late token (fp nondeterminism, cudagraph vs
eager kernels), so the bar is a >=PREFIX_MIN-token identical prefix, not
byte equality to the end.
"""
import argparse
import json
import sys

from bench_lib import DEFAULT_BASE, DEFAULT_MODEL, build_context_prompt, chat

PREFIX_MIN = 200  # identical leading tokens required (whitespace-normalized chars used as proxy)

PROBES = [
    ("prose", "Explain in one paragraph why a watchdog must clear a systemd "
              "start-limit latch before restarting a unit."),
    ("code", "Write a Python function `parse_kv(line: str) -> dict` that parses "
             "'k1=v1;k2=v2' pairs, ignoring empty segments. Code only."),
    ("ctx60k", None),  # built at runtime: 60K-token doc + recall question
    ("toolish", "You have a tool `get_weather(city)`. The user asks: what's the "
                "weather in Tempe? Respond with the tool call you would make."),
]


def build_messages(name: str) -> list:
    if name == "ctx60k":
        doc = build_context_prompt(60000, "EQ-60K-SALTBUSH")
        return [{"role": "user", "content": doc +
                 "\n\nState the exact AUDIT MARKER value, then summarize the "
                 "policy in two sentences."}]
    text = dict(PROBES)[name]
    return [{"role": "user", "content": text}]


def run_probes(base: str, model: str) -> dict:
    out = {}
    for name, _ in PROBES:
        msgs = build_messages(name)
        r = chat(base, model, msgs, max_tokens=400, temperature=0.0, top_p=1.0,
                 top_k=-1, seed=42, timeout=600.0,
                 extra={"chat_template_kwargs": {"enable_thinking": False}})
        out[name] = r["content"]
        print(f"[{name}] {r['completion_tokens']} tok, "
              f"{r['decode_tps']:.1f} tok/s", flush=True)
    return out


def norm(s: str) -> str:
    return " ".join(s.split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["record", "check"])
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="ref-mtp-off.json")
    ap.add_argument("--ref", default="ref-mtp-off.json")
    args = ap.parse_args()

    outputs = run_probes(args.base_url, args.model)
    if args.mode == "record":
        with open(args.out, "w") as f:
            json.dump(outputs, f, indent=2)
        print(f"reference recorded -> {args.out}")
        return 0

    with open(args.ref) as f:
        ref = json.load(f)
    failures = []
    for name, text in outputs.items():
        a, b = norm(ref.get(name, "")), norm(text)
        common = 0
        for ca, cb in zip(a, b):
            if ca != cb:
                break
            common += 1
        ok = common >= min(PREFIX_MIN, len(a), len(b)) and bool(a) and bool(b)
        print(f"[{name}] common prefix {common} chars "
              f"(ref {len(a)}, now {len(b)}) -> {'OK' if ok else 'DIVERGED'}")
        if not ok:
            failures.append(name)
    if failures:
        print(f"\nEQUIVALENCE FAIL: {', '.join(failures)} — the MTP verify "
              "path is altering output. Do NOT ship MTP; attach both JSONs "
              "to the incident notes.")
        return 1
    print("\nequivalence OK: MTP output matches the MTP-off reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
