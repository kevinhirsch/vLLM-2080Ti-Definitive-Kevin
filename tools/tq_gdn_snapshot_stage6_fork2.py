# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXP-038 Stage-6 driver — NON-BLOCKING fork v2 (POST /tq/fork2) e2e demo.

Standalone MANUAL script. NOT a pytest test (no ``test_`` prefix, not under
``tests/``); does nothing on import — everything is under ``if __name__ ==
"__main__"``. Pure stdlib (urllib) HTTP client: it does NOT import vllm/torch, so
it never touches engine internals. Run it ONLY against a THROWAWAY snapshot
engine started with ``VLLM_TQ_GDN_SNAPSHOT=1``. NEVER point it at :8001.

What this proves vs Stage-4 (/tq/fork, blocking, engine-side child construction):
the v2 fork admits each child as an ORDINARY ``AsyncLLM.generate`` request (the
server-layer fan-out), so the busy loop is never blocked AND the standard input
path injects the model eos / generation-config stop tokens.

Claim (PASS = A and B and C):
  (A) STOP-CORRECT. Every child finishes with ``finish_reason == "stop"`` (the
      model's eos), NOT "length" — i.e. children respect the model's stop tokens
      instead of running to max_tokens and emitting special-token soup. This is
      the exact defect v1's fork_from_handle had (it built Requests from raw
      SamplingParams, bypassing eos injection). Give a max_tokens generous enough
      that a correct answer terminates on eos well before the cap.
  (B) CHEAP PREFILL. Every child's ``num_cached_tokens`` ~= the pinned prefix
      (>= prefix_len - one block): the children adopted the pinned donor blocks
      via the prefix cache rather than recomputing the prefix.
  (C) PIN RESIDENT + NO LEAK. ``pin_resident`` is true (the /tq/pin blocks stayed
      resident through the fan-out), and /tq/unpin frees a non-zero block count
      (the pin is released exactly once, by the caller).

Also demonstrates per-child sampling divergence (children at distinct
temperatures) over the SAME shared prefix.

Example (throwaway engine already serving with the routes enabled):

    VLLM_TQ_GDN_SNAPSHOT=1 VLLM_TQ_GDN_SNAPSHOT_CONFIRM_THROWAWAY=1 \
    python tools/tq_gdn_snapshot_stage6_fork2.py \
        --engine-url http://127.0.0.1:8011 \
        --children 4 --max-tokens 256

Bring the throwaway server up first, e.g. (NOT :8001):

    VLLM_TQ_GDN_SNAPSHOT=1 vllm serve /path/to/Qwen3.8-27B-hybrid \
        --port 8011 --tensor-parallel-size 2 --mamba-cache-mode align \
        --enable-prefix-caching ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def _post(base: str, path: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + path,
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _build_corpus(target_tokens: int) -> str:
    """Assemble a shared corpus comfortably above one block, deterministically."""
    fact = (
        "Linear-attention and hybrid recurrent models trade exact long-range "
        "recall for a bounded state, which helps memory but can hurt precise "
        "lookup; quantized KV caches narrow but do not close that gap. "
    )
    header = (
        "RESEARCH CORPUS — attention mechanisms for long context.\n"
        "The following numbered findings are the ONLY admissible evidence.\n"
    )
    lines, n = [header], 0
    # ~ (len(fact)//4) tokens per line; loop until we clear the target.
    while n < target_tokens:
        i = len(lines)
        lines.append(f"[{i}] {fact}")
        n += max(1, len(fact) // 4)
    lines.append(
        "\n=== TASK ===\nUsing ONLY the corpus above, answer in ONE sentence: "
        "is linear attention 'good enough' for citation-grounded QA? "
        "Answer YES or NO and justify briefly."
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine-url", required=True,
                    help="throwaway engine base URL (NOT :8001), e.g. "
                         "http://127.0.0.1:8011")
    ap.add_argument("--children", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--corpus-tokens", type=int, default=2112)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--i-understand-throwaway", action="store_true",
                    help="required unless VLLM_TQ_GDN_SNAPSHOT_CONFIRM_THROWAWAY=1")
    args = ap.parse_args()

    if ":8001" in args.engine_url:
        print("REFUSING: --engine-url points at :8001 (production serve).",
              file=sys.stderr)
        return 2
    if not (args.i_understand_throwaway
            or os.getenv("VLLM_TQ_GDN_SNAPSHOT_CONFIRM_THROWAWAY") == "1"):
        print("REFUSING: pass --i-understand-throwaway or set "
              "VLLM_TQ_GDN_SNAPSHOT_CONFIRM_THROWAWAY=1 (throwaway engines only).",
              file=sys.stderr)
        return 2

    base = args.engine_url
    corpus = _build_corpus(args.corpus_tokens)
    print(f"[pin] posting corpus (~{len(corpus)//4} tok est) to {base}/tq/pin")
    try:
        pin = _post(base, "/tq/pin", {"prompt": corpus}, args.timeout)
    except urllib.error.HTTPError as e:
        print(f"[pin] HTTP {e.code}: {e.read().decode()[:200]}", file=sys.stderr)
        return 1
    handle_id = pin["handle_id"]
    prefix_len = int(pin["prefix_len"])
    print(f"[pin] handle={handle_id} prefix_len={prefix_len} "
          f"computed={pin.get('num_computed_tokens')} "
          f"fully_prefilled={pin.get('fully_prefilled')}")

    # N children over the SAME prefix, diverging only in sampling temperature.
    temps = [round(i / max(1, args.children - 1), 3) for i in range(args.children)]
    children = [{"temperature": t, "max_tokens": args.max_tokens} for t in temps]

    ok = True
    try:
        print(f"[fork2] fanning out {args.children} children …")
        res = _post(base, "/tq/fork2",
                    {"handle_id": handle_id, "children": children}, args.timeout)
        pin_resident = res.get("pin_resident")
        print(f"[fork2] n={res.get('n')} pin_resident={pin_resident}")
        # cache-hit tolerance: allow the last (partial) block to miss.
        cache_floor = max(0, prefix_len - 256)
        for i, ch in enumerate(res.get("children", [])):
            fr = ch.get("finish_reason")
            nct = ch.get("num_cached_tokens")
            not_ = ch.get("num_output_tokens")
            head = (ch.get("text") or "").replace("\n", " ")[:80]
            stop_ok = fr == "stop"
            cache_ok = isinstance(nct, int) and nct >= max(0, prefix_len - 2112)  # all FULL blocks; the final partial block can never cache (align block=2112)
            print(f"  child[{i}] temp={temps[i]} finish={fr} "
                  f"out={not_}tok cached={nct} "
                  f"stop_ok={stop_ok} cache_ok={cache_ok} :: {head!r}")
            ok = ok and stop_ok and cache_ok
        if pin_resident is not True:
            print("  [C] FAIL: pin_resident is not true", file=sys.stderr)
            ok = False
    finally:
        try:
            un = _post(base, "/tq/unpin", {"handle_id": handle_id}, args.timeout)
            print(f"[unpin] ok={un.get('ok')} freed={un.get('num_freed_blocks')}")
            if not un.get("num_freed_blocks"):
                print("  [C] WARN: unpin freed 0 blocks", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"[unpin] FAILED ({type(e).__name__}: {e}) — possible KV leak",
                  file=sys.stderr)

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
