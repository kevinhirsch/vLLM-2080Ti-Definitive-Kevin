#!/usr/bin/env python3
"""Build model-kvscales.safetensors from QSA K/V amax dumps (VLLM_QWEN4_QSA_KV_AMAX_DUMP).

usage: kvscales_from_amax.py <dump-prefix> <model-dir> [--margin 1.0]

Takes the max over TP ranks per layer (each rank stores its own KV heads; the scale
is per layer) and writes the fp8-contract scale s = margin * max_abs / 448 under
model.language_model.layers.<i>.self_attn.{k,v}_scale, which the int8 QSA cache
reads (finalize_kv_cache_scales: s_int8 = s * 448 / 127).
"""
import glob
import json
import math
import os
import re
import sys

import torch
from safetensors.torch import save_file

prefix, model_dir = sys.argv[1], sys.argv[2]
margin = float(sys.argv[sys.argv.index("--margin") + 1]) if "--margin" in sys.argv else 1.0
merged: dict[int, list[float]] = {}
files = sorted(glob.glob(f"{prefix}.rank*.json"))
if not files:
    sys.exit(f"no dumps match {prefix}.rank*.json")
for path in files:
    for name, (k_amax, v_amax) in json.load(open(path)).items():
        m = re.search(r"layers\.(\d+)\.", name)
        if m is None:
            continue
        i = int(m.group(1))
        if not all(math.isfinite(a) for a in (k_amax, v_amax)):  # max() would drop a NaN silently
            sys.exit(f"{path}: layer {i} amax K={k_amax} V={v_amax} (the recorder saw non-finite K/V)")
        cur = merged.setdefault(i, [0.0, 0.0])
        cur[0], cur[1] = max(cur[0], k_amax), max(cur[1], v_amax)
tensors = {}
for i, (k_amax, v_amax) in sorted(merged.items()):
    for side, amax in (("k", k_amax), ("v", v_amax)):
        if not amax > 0.0:
            sys.exit(f"layer {i}: no {side} amax recorded")
        tensors[f"model.language_model.layers.{i}.self_attn.{side}_scale"] = torch.tensor(
            [margin * amax / 448.0], dtype=torch.float32
        )
out = os.path.join(model_dir, "model-kvscales.safetensors")
save_file(tensors, out)
print(f"{len(merged)} layers from {len(files)} rank file(s) -> {out}")
for i, (k_amax, v_amax) in sorted(merged.items()):
    print(f"  layer {i:2d}: |K| <= {k_amax:8.3f}  |V| <= {v_amax:8.3f}")
