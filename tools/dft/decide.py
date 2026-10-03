#!/usr/bin/env python3
"""decide.py BASE_LIVE.json TUNED_LIVE.json -> prints verdict JSON; exit 0 = KEEP, 1 = restore base.
KEEP only if acceptance AND speed improved BEYOND THE SPREAD (non-overlapping ranges over the 3 reps) and natural-text decode did not regress."""
import json, sys, statistics as S
b, t = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
def acc(r): return [x["accept_len"] for x in r["estate_accept"]]
v = {}
v["estate_accept_len_base"], v["estate_accept_len_tuned"] = acc(b), acc(t)
v["estate_wall_base"], v["estate_wall_tuned"] = b["estate_wall_s"], t["estate_wall_s"]
v["natural_tok_s_base"], v["natural_tok_s_tuned"] = b["natural_tok_s"], t["natural_tok_s"]
v["natural_accept_len_base"], v["natural_accept_len_tuned"] = b["natural_accept"]["accept_len"], t["natural_accept"]["accept_len"]
v["accept_up"] = min(acc(t)) > max(acc(b))
v["wall_down"] = max(t["estate_wall_s"]) < min(b["estate_wall_s"])
v["natural_ok"] = S.mean(t["natural_tok_s"]) >= min(b["natural_tok_s"])
v["keep"] = bool(v["accept_up"] and v["wall_down"] and v["natural_ok"])
print(json.dumps(v, indent=1)); sys.exit(0 if v["keep"] else 1)
