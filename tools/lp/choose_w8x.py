#!/usr/bin/env python3
"""Print True if W8X qualified in the window bench (expansion exact on the real repacked layout everywhere AND >=1.10x vs
Marlin W4A16 on gate_up / gdn_qkvz at M>=2048), else False. Usage: choose_w8x.py RESULTS_DIR"""
import glob, json, sys
ok, g = True, []
for f in glob.glob(sys.argv[1] + "/w4a8_bench_gpu*.json"):
    for r in json.load(open(f))["rows"]:
        ok = ok and r.get("w8x_exact_frac", 0) >= 0.9999
        if r["shape"] in ("gate_up", "gdn_qkvz") and r["M"] >= 2048:
            g.append(r["w4a16"]["min"] / r["w8x"]["min"])
print(bool(ok and g and min(g) >= 1.10))
