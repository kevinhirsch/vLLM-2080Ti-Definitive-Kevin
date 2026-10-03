#!/usr/bin/env python
"""Lane U2: pre-build the int4 lm_head + MTP block cache so the engine boot only reads it (no GPU, no engine needed).
Writes ONLY into <model_dir>-u2cache (never touches the original weights).  Idempotent; keyed on the source files' size+mtime.
Usage: CUDA_VISIBLE_DEVICES= python tools/u2/prebuild_headquant.py [--model DIR] [--only head|mtp]
"""
import argparse, importlib.util, os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["VLLM_U2_QUANT_DEVICE"] = os.environ.get("VLLM_U2_QUANT_DEVICE", "cpu")
import torch
from safetensors import safe_open
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("u2hq", os.path.join(HERE, "../../vllm/model_executor/layers/quantization/u2_headquant.py"))
hq = importlib.util.module_from_spec(spec); spec.loader.exec_module(hq)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven")
ap.add_argument("--only", choices=["head", "mtp", "embed"])
a = ap.parse_args()
class L:  # tiny logger shim
    info = staticmethod(lambda f, *x: print("[prebuild]", f % x, flush=True)); warning = info
jobs = []
if a.only in (None, "head"):
    jobs.append(("model.safetensors", "lm_head.weight"))
if a.only in (None, "embed"):
    jobs.append(("model.safetensors", "model.language_model.embed_tokens.weight"))
if a.only in (None, "mtp"):
    m = safe_open(f"{a.model}/model-mtp.safetensors", "pt")
    jobs += [("model-mtp.safetensors", k) for k in m.keys() if hq._MTP_LINEAR.match(k)]
for fn, name in jobs:
    t0 = time.time()
    w = safe_open(f"{a.model}/{fn}", "pt").get_tensor(name)
    path = hq._cache_path(a.model, name, hq._src_stamp(a.model))
    pre = os.path.exists(path)
    t = hq.get_quantized(a.model, name, w, L)
    print(f"[prebuild] {name} {tuple(w.shape)} -> {os.path.basename(path)} {'(cached)' if pre else ''} {time.time()-t0:.0f}s", flush=True)
print("cache dir:", hq.cache_dir_for(a.model))
