#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXP-039 (S4) replay-shaped micro-bench for the scoped re-emission drafter.

Adjudicates the scoped drafter on its *target regime* -- verbatim re-emission --
NOT the generation-shaped bench that (correctly) refuted the blanket overlay.

It builds requests whose expected output is dominated by long verbatim copies of
context (file rewrites, quoted blocks), plus a pure-generation control that must
show *no* regression. For each workload it measures decode tok/s (median + spread
over reps, warm-up discarded) and, by scraping the server's Prometheus /metrics,
the per-position speculative acceptance vector.

This script does NOT launch or configure the engine. Run it twice -- once against
a server started WITHOUT the drafter (prod MTP-K2, the "off" baseline) and once
WITH it -- then compare:

  # baseline server:  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
  python tools/s4_replay_bench.py --base-url http://127.0.0.1:8000 \
      --model qwen3.6:27b --label off --out /tmp/s4_off.json

  # scoped server:    --speculative-config '{"method":"mtp","num_speculative_tokens":16}'
  #   env: VLLM_S4_SCOPED_DRAFTER=1 VLLM_MTP_DRAFT_CAP=2 VLLM_S4_K_SCOPED=16 VLLM_S4_G=12
  python tools/s4_replay_bench.py --base-url http://127.0.0.1:8000 \
      --model qwen3.6:27b --label on  --out /tmp/s4_on.json

  python tools/s4_replay_bench.py --compare /tmp/s4_off.json /tmp/s4_on.json

Only the Python stdlib is used (urllib) so it runs in any venv.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.request

# --------------------------------------------------------------------------
# Sample payload used to force verbatim re-emission. A self-contained source
# blob that is large enough to dominate the output when the model copies it.
# --------------------------------------------------------------------------

_SOURCE_FILE = '''\
import math
from dataclasses import dataclass


@dataclass
class Vector3:
    x: float
    y: float
    z: float

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> "Vector3":
        n = self.length()
        if n == 0.0:
            return Vector3(0.0, 0.0, 0.0)
        return Vector3(self.x / n, self.y / n, self.z / n)

    def dot(self, other: "Vector3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3") -> "Vector3":
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )


def reflect(incident: Vector3, normal: Vector3) -> Vector3:
    d = incident.dot(normal)
    return Vector3(
        incident.x - 2.0 * d * normal.x,
        incident.y - 2.0 * d * normal.y,
        incident.z - 2.0 * d * normal.z,
    )


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
'''

_QUOTE_BLOCK = _SOURCE_FILE  # reuse the same blob for the pure-copy workload


def _workloads():
    """Return {name: (messages, max_tokens)}."""
    return {
        # Canonical S4 case: reproduce a file verbatim with a tiny edit. Output
        # is ~99% a copy of the prompt (the file), except the renamed symbol.
        "rewrite": (
            [
                {
                    "role": "user",
                    "content": (
                        "Below is a Python file between <<<FILE>>> markers. "
                        "Reproduce it EXACTLY, changing only every occurrence of "
                        "the class name `Vector3` to `Vec3` (including in type "
                        "hints). Output ONLY the resulting file, no prose, no "
                        "code fences.\n\n<<<FILE>>>\n" + _SOURCE_FILE + "<<<FILE>>>\n"
                    ),
                }
            ],
            900,
        ),
        # Upper bound / echo analogue: a pure verbatim copy, no edits.
        "quote": (
            [
                {
                    "role": "user",
                    "content": (
                        "Repeat the following text EXACTLY, character for "
                        "character, with no commentary and no code fences:\n\n"
                        + _QUOTE_BLOCK
                    ),
                }
            ],
            900,
        ),
        # Realistic agent turn: a little novel prose, then a verbatim quote. The
        # gate must stay shut for the prose and open for the quote.
        "mixed": (
            [
                {
                    "role": "user",
                    "content": (
                        "Write exactly two sentences explaining what a normalized "
                        "vector is. Then, on a new line, write 'Reference "
                        "implementation:' followed by an exact verbatim copy of "
                        "the code below (no code fences):\n\n" + _SOURCE_FILE
                    ),
                }
            ],
            700,
        ),
        # Control: free-form generation with NO copyable span. This is the regime
        # that killed the overlay; S4 must be a no-op-grade regression here.
        "generation": (
            [
                {
                    "role": "user",
                    "content": (
                        "Write a detailed, original ~400-word explanation of how "
                        "ray-triangle intersection works in a path tracer. Do not "
                        "include any code."
                    ),
                }
            ],
            600,
        ),
    }


# --------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# --------------------------------------------------------------------------


def _post_stream(base_url: str, model: str, messages, max_tokens: int, timeout: float):
    """Stream a chat completion; return (n_completion_tokens, decode_seconds).

    Decode time excludes time-to-first-token so tok/s reflects the decode
    (speculative) phase, not prefill.
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
    n_chunks = 0
    usage_completion = None
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
            usage = obj.get("usage")
            if usage and usage.get("completion_tokens") is not None:
                usage_completion = usage["completion_tokens"]
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    if t_first is None:
                        t_first = time.monotonic()
                    n_chunks += 1
    t_end = time.monotonic()
    decode_s = t_end - (t_first if t_first is not None else t0)
    if usage_completion is None:
        # Fail fast: an SSE content chunk is NOT one token (a chunk can carry
        # several, especially under speculative decode), so falling back to
        # n_chunks would silently mismeasure tok/s. The request sets
        # stream_options.include_usage=True; if the server omits usage, error out
        # rather than report a wrong throughput number.
        raise RuntimeError(
            "stream omitted usage.completion_tokens (include_usage not honored); "
            "refusing to use SSE chunk count as a token proxy"
        )
    return usage_completion, max(decode_s, 1e-9)


def _scrape_spec_metrics(base_url: str):
    """Return cumulative spec-decode counters from Prometheus /metrics, or None."""
    url = base_url.rstrip("/") + "/metrics"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            text = resp.read().decode("utf-8", "replace")
    except Exception:
        return None
    out = {"drafts": 0.0, "draft_tokens": 0.0, "accepted": 0.0, "per_pos": {}}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        try:
            name_labels, value = line.rsplit(" ", 1)
            v = float(value)
        except ValueError:
            continue
        if name_labels.startswith("vllm:spec_decode_num_drafts"):
            out["drafts"] += v
        elif name_labels.startswith("vllm:spec_decode_num_draft_tokens"):
            out["draft_tokens"] += v
        elif name_labels.startswith("vllm:spec_decode_num_accepted_tokens_per_pos"):
            pos = _extract_label(name_labels, "position")
            if pos is not None:
                out["per_pos"][pos] = out["per_pos"].get(pos, 0.0) + v
        elif name_labels.startswith("vllm:spec_decode_num_accepted_tokens"):
            out["accepted"] += v
    return out


def _extract_label(name_labels: str, key: str):
    marker = key + '="'
    i = name_labels.find(marker)
    if i < 0:
        return None
    i += len(marker)
    j = name_labels.find('"', i)
    if j < 0:
        return None
    try:
        return int(name_labels[i:j])
    except ValueError:
        return name_labels[i:j]


def _delta_metrics(before, after):
    if before is None or after is None:
        return None
    d = {
        "drafts": after["drafts"] - before["drafts"],
        "draft_tokens": after["draft_tokens"] - before["draft_tokens"],
        "accepted": after["accepted"] - before["accepted"],
        "per_pos": {},
    }
    for pos in sorted(set(before["per_pos"]) | set(after["per_pos"])):
        d["per_pos"][pos] = after["per_pos"].get(pos, 0.0) - before["per_pos"].get(
            pos, 0.0
        )
    d["mean_accept_len"] = (
        1 + d["accepted"] / d["drafts"] if d["drafts"] else float("nan")
    )
    d["accept_rate"] = (
        d["accepted"] / d["draft_tokens"] if d["draft_tokens"] else float("nan")
    )
    return d


# --------------------------------------------------------------------------
# Bench driver
# --------------------------------------------------------------------------


def run_bench(args):
    workloads = _workloads()
    if args.only:
        workloads = {k: v for k, v in workloads.items() if k in set(args.only)}
    results = {}
    for name, (messages, max_tokens) in workloads.items():
        toks_per_s = []
        metrics_delta = None
        # warm-up (discarded)
        for _ in range(args.warmup):
            try:
                _post_stream(
                    args.base_url, args.model, messages, max_tokens, args.timeout
                )
            except Exception as e:
                print(f"[{name}] warmup error: {e}", file=sys.stderr)
        # timed reps
        for r in range(args.reps):
            m_before = _scrape_spec_metrics(args.base_url)
            try:
                n_tokens, decode_s = _post_stream(
                    args.base_url, args.model, messages, max_tokens, args.timeout
                )
            except Exception as e:
                print(f"[{name}] rep {r} error: {e}", file=sys.stderr)
                continue
            m_after = _scrape_spec_metrics(args.base_url)
            tps = n_tokens / decode_s
            toks_per_s.append(tps)
            # Record on every successful rep so a failure on the LAST rep does not
            # blank out metrics captured by earlier reps (keeps the last good one).
            metrics_delta = _delta_metrics(m_before, m_after)
            print(
                f"[{args.label}] {name:11s} rep{r}: {tps:7.2f} tok/s "
                f"({n_tokens} toks / {decode_s:.3f}s)"
            )
        if toks_per_s:
            results[name] = {
                "tok_s_median": statistics.median(toks_per_s),
                "tok_s_min": min(toks_per_s),
                "tok_s_max": max(toks_per_s),
                "reps": toks_per_s,
                "spec_metrics_last_rep": metrics_delta,
            }
    out = {
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "reps": args.reps,
        "warmup": args.warmup,
        "results": results,
    }
    print("\n=== summary (%s) ===" % args.label)
    for name, r in results.items():
        acc = ""
        md = r.get("spec_metrics_last_rep")
        if md and md.get("drafts"):
            pos = ", ".join(
                f"{(md['per_pos'][p] / md['drafts']):.3f}"
                for p in sorted(md["per_pos"])
            )
            acc = f"  mean_accept_len={md['mean_accept_len']:.2f} per_pos=[{pos}]"
        print(
            f"  {name:11s} {r['tok_s_median']:7.2f} tok/s "
            f"[{r['tok_s_min']:.1f}-{r['tok_s_max']:.1f}]{acc}"
        )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out}")
    return out


def compare(path_off, path_on):
    with open(path_off) as f:
        off = json.load(f)
    with open(path_on) as f:
        on = json.load(f)
    print(f"\n=== A/B: off={off['label']} ({path_off})  on={on['label']} ({path_on}) ===")
    print(f"{'workload':12s} {'off tok/s':>10s} {'on tok/s':>10s} {'delta':>9s}  verdict")
    names = list(off["results"].keys())
    for name in names:
        o = off["results"].get(name)
        n = on["results"].get(name)
        if not o or not n:
            continue
        om, nm = o["tok_s_median"], n["tok_s_median"]
        delta = (nm / om - 1.0) * 100 if om else float("nan")
        if name == "generation":
            verdict = "OK (no-op)" if delta >= -3.0 else "REGRESSION (gate leaking)"
        elif name in ("rewrite", "quote"):
            verdict = "WIN" if delta > 0 else "NEGATIVE - needs Fable re-adjudication"
        else:
            verdict = "win" if delta > 0 else "loss (async-off headwind)"
        print(f"{name:12s} {om:10.2f} {nm:10.2f} {delta:+8.1f}%  {verdict}")
    print(
        "\nNote: if rewrite/quote are not clear wins, the async-off headwind "
        "(design doc S7) is beating the copy gain -> the GPU-side merge increment "
        "(S10.1) is required. Flag NEGATIVE - needs Fable re-adjudication."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen3.6:27b")
    ap.add_argument("--label", default="run", help="off | on | any tag")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--only", nargs="*", help="subset of workloads to run")
    ap.add_argument("--out", help="write results JSON here")
    ap.add_argument(
        "--compare",
        nargs=2,
        metavar=("OFF_JSON", "ON_JSON"),
        help="compare two result files and print the A/B table (no requests sent)",
    )
    args = ap.parse_args()
    if args.compare:
        compare(args.compare[0], args.compare[1])
        return
    run_bench(args)


if __name__ == "__main__":
    main()
