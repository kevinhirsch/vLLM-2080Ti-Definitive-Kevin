#!/usr/bin/env python3
"""K5: base vs k5 summary from a window dir (~/projects/lanes/k5/win): quick.py decode, per-step anatomy by stream count."""
import json
import os
import sys

W = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/projects/lanes/k5/win")


def q(lab):
    try:
        return json.loads(open(f"{W}/quick_{lab}.json").read().strip().splitlines()[-1])
    except Exception:
        return {}


def steps(lab):
    out = []
    try:
        for line in open(f"{W}/steps_{lab}.jsonl"):
            line = line.strip()
            if line.startswith("{"):
                out.append(json.loads(line))
    except Exception:
        pass
    return out


qb, qk = q("base"), q("k5")
print("natural tok/s base", qb.get("natural_tok_s"), "k5", qk.get("natural_tok_s"))
print("the-lane tok/s base", qb.get("the_lane_tok_s"), "k5", qk.get("the_lane_tok_s"))
print("accept base", qb.get("natural_accept"), "k5", qk.get("natural_accept"))
sb, sk = steps("base"), steps("k5")
for i, (b, k) in enumerate(zip(sb, sk)):
    print(f"[profile {i}] period {b['period_ms']} -> {k['period_ms']} ms; busy {b['busy_ms']} -> {k['busy_ms']}; idle {b['idle_pct']}% -> {k['idle_pct']}%; "
          f"kernels {b['kernels_per_step']} -> {k['kernels_per_step']}; glue {b['class_ms'].get('glue')} -> {k['class_ms'].get('glue')}; "
          f"gdn {b['class_ms'].get('gdn')} -> {k['class_ms'].get('gdn')}; attn {b['class_ms'].get('attn')} -> {k['class_ms'].get('attn')}")
