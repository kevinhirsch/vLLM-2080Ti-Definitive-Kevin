#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""EXP-021: mixed-precision (per-module dynamic bits) self-quant of Qwen3.8-27B.

WHY: official FP8 (measured 2026-08-23) closed at 50 tok/s vs our proven int4
self-quant's 68-77 tok/s -- FP8 fails the speed bar on this SM75 hardware (no
native fp8 tensor cores; runs through a slow fallback path). Pure int4 is the
speed champion but leaves quality on the table on the modules most sensitive to
precision loss. This job raises precision ONLY on those modules while keeping
the int4 backbone -- same recipe, same speed profile, better quality.

RECIPE (per-module dynamic bits via gptqmodel `QuantizeConfig.dynamic`):
  - backbone (everything not listed below):
        int4, group_size=128, sym=True, desc_act=False
        -- this is the EXACT proven recipe from quantize_qwen38.py, the script
        that produced the currently-serving /home/kevin/Desktop/models/
        Qwen3.8-27B-GPTQ-Int4 (19GB, 148 min quantize() wall time, 2026-08-14
        run; log at /home/kevin/Desktop/qwythos-quant/gptq38.log).
  - full-attention layer self_attn projections (q/k/v/o_proj), 16 of 64 layers
    (full_attention_interval=4 -> layers 3,7,11,...,63 per config.json
    `layer_types`, cross-checked at runtime against the live config):
        int8, same group_size/sym/desc_act
  - GDN (linear-attention) in_proj_a / in_proj_b, ALL visual-tower blocks,
    mtp.*, lm_head:
        SKIPPED -- kept at source BF16

SENSITIVITY PROVENANCE: mirrors the ignore-list shape of
lued/Qwen3.8-27B-INT8-W8A16-MTP (that checkpoint kept GDN in_proj_a/in_proj_b,
all visual blocks, lm_head, and mtp.* in high precision). Same modules flagged
sensitive there, applied here as int4-vs-skip instead of int8-vs-skip since our
backbone target is int4, not int8.

NOTE: gptqmodel's own Qwen3_5QModel.module_tree (site-packages/gptqmodel/
models/definitions/qwen3_5.py) already marks in_proj_a/in_proj_b as
structurally non-quantized (`:!`) independent of anything in this script -- the
explicit exclude below is belt-and-suspenders / self-documenting, not strictly
required to get that behavior. Likewise `model.visual.*` and `lm_head` are
never walked by the quantizer at all unless `quantize_config.lm_head=True`
(we don't set it), so those excludes are also redundant-but-harmless.

CALIBRATION: identical mixed corpus + mix ratios to quantize_qwen38.py --
50% c4 (generic English, keeps general ability from degrading), 30% real
captured workload (gateway flight recorder: agentic turns, tool schemas, JSON
tool results -- the distribution this box actually serves), 20% local source
code (coding-agent traffic). N=512 samples, SEQ<=1024 tokens, batch_size=1,
offload_to_disk=True (real disk, never /tmp -- tmpfs is RAM and a 50+GB
offload would OOM the box).

USAGE:
  ~/quant-env/bin/python exp021_mixed48_quant.py --dry-run   # config-only,
                                                               # no weights, no GPU
  ~/quant-env/bin/python exp021_mixed48_quant.py              # the real
                                                               # 2.5-4h run

GPUs MUST be free first. This script does NOT stop prod itself -- see the
sibling runner exp021_mixed48_run.sh, which aborts if any VLLM:: process is
running. Prod is stopped by the overnight launcher, which invokes that runner
only after prod is confirmed down.
"""
import argparse
import glob
import json
import os
import random
import sys
import time

# ---------------------------------------------------------------------------
# Env pins -- identical to the proven quantize_qwen38.py recipe (2026-08-14).
# Must be set before any torch/gptqmodel import.
# ---------------------------------------------------------------------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.update(CC="/usr/bin/gcc-13", CXX="/usr/bin/g++-13", CUDAHOSTCXX="/usr/bin/g++-13",
                   NVCC_CCBIN="/usr/bin/g++-13", CUDA_HOME="/usr/local/cuda-12.8",
                   TORCH_CUDA_ARCH_LIST="7.5")
os.environ["PATH"] = "/home/kevin/quant-env/bin:/usr/local/cuda-12.8/bin:" + os.environ.get("PATH", "")

SRC = os.environ.get("EXP021_SRC", "/home/kevin/Desktop/models/Qwen3.8-27B")
OUT = os.environ.get("EXP021_OUT", "/home/kevin/Desktop/models/Qwen3.8-27B-GPTQ-mixed48")
OFFLOAD = os.environ.get("EXP021_OFFLOAD", "/home/kevin/Desktop/qwythos-quant/offload_mixed48")
FLIGHTREC = "/home/kevin/.local/share/vllm-qwen27b/flightrec"
CODE_ROOTS = ["/home/kevin/.local/share/vllm-qwen27b",
              "/home/kevin/Desktop/vLLM-2080Ti-Definitive/vllm/v1/spec_decode",
              "/home/kevin/Desktop/vLLM-2080Ti-Definitive/vllm/compilation"]

N = int(os.environ.get("EXP021_N", "512"))
SEQ = int(os.environ.get("EXP021_SEQ", "1024"))  # 22GB cards OOM on long-seq activations
MIX = {"c4": 0.50, "workload": 0.30, "code": 0.20}

# Disk headroom. The launch plan's own stated floor is 25GB free (a soft trip-wire
# checked before this plan is written at all -- see the authoring session's `df -h`).
# This script's OWN preflight uses a more realistic estimate for what the run needs
# to actually complete without ENOSPC:
#   final output ~19-25GB (pure-int4 baseline measured at 19GB; mixed48 adds a few GB
#   for the int8/BF16 modules kept above int4) + ~15-30GB transient offload_to_disk
#   working set (15GB measured leftover from the pure-int4 run) + margin.
# Override with EXP021_MIN_FREE_GB if you've sized it differently.
MIN_FREE_GB = float(os.environ.get("EXP021_MIN_FREE_GB", "60"))
DISK_CHECK_PATH = "/home/kevin/Desktop"  # same filesystem as SRC/OUT/OFFLOAD


def resolve_full_attention_layers(text_cfg, num_layers):
    """Return sorted full-attention layer indices, preferring the explicit
    `layer_types` list from config.json (ground truth) with
    `full_attention_interval` as a fallback for configs that omit it."""
    layer_types = getattr(text_cfg, "layer_types", None)
    if layer_types:
        return [i for i, t in enumerate(layer_types) if t == "full_attention"]
    interval = getattr(text_cfg, "full_attention_interval", 4)
    return [i for i in range(num_layers) if (i + 1) % interval == 0]


def build_dynamic_config(text_cfg, num_layers):
    """Build the gptqmodel `QuantizeConfig.dynamic` per-module override map.

    Key format (gptqmodel 7.3.2, quantization/config.py:dynamic_get): dict of
    {PCRE_pattern: overrides}. A `-:pattern` key means exclude (module kept at
    source precision / not quantized at all); a plain or `+:pattern` key means
    include with the `overrides` dict applied (e.g. {"bits": 8}). Patterns are
    matched with `.match()` (prefix-anchored, PCRE) against the module's dotted
    path with NO trailing `.weight`, e.g.
    "model.language_model.layers.3.self_attn.q_proj". Exclude patterns are
    always evaluated first regardless of dict insertion order -- gptqmodel's
    own QuantizeConfig.__post_init__ reorders them (config.py ~line 2528).
    """
    full_attn = resolve_full_attention_layers(text_cfg, num_layers)
    expected = num_layers // 4
    if len(full_attn) != expected:
        print(f"[WARN] expected {expected} full-attention layers (interval=4), "
              f"got {len(full_attn)}: {full_attn}")
    idx_alt = "|".join(str(i) for i in full_attn)

    dynamic = {
        # ---- SKIP: kept at source BF16 ----
        r"-:^model\.visual\.": {},
        r"-:^mtp\.": {},
        r"-:^lm_head(\.|$)": {},
        r"-:^model\.language_model\.layers\.\d+\.linear_attn\.in_proj_[ab](\.|$)": {},
        # ---- INT8: full-attention layer self_attn projections ----
        rf"^model\.language_model\.layers\.({idx_alt})\.self_attn\.(q_proj|k_proj|v_proj|o_proj)(\.|$)": {"bits": 8},
        # everything else falls through to the top-level QuantizeConfig default: int4.
    }
    return dynamic, full_attn


def load_config_only(src):
    """Config-only load: parses config.json, touches no weights, touches no GPU.
    Safe to run on a box with prod holding both GPUs."""
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(src, trust_remote_code=True)


def die(msg):
    print(f"[preflight] FAIL: {msg}")
    sys.exit(1)


def disk_free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def check_gpus_idle():
    import subprocess
    try:
        used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True).stdout.split()
        busy = [u for u in used if u.isdigit() and int(u) > 2000]
        if busy:
            die(f"GPUs still busy ({used} MiB used) -- something is holding VRAM; abort")
    except FileNotFoundError:
        pass


def print_dry_run_bitmap(dynamic, text_cfg, num_layers, full_attn):
    from gptqmodel.quantization.config import dynamic_get

    print(f"[dry-run] model_type={getattr(text_cfg, 'model_type', '?')} "
          f"num_hidden_layers={num_layers} "
          f"full_attention_interval={getattr(text_cfg, 'full_attention_interval', '?')}")
    print(f"[dry-run] full-attention layers ({len(full_attn)}/{num_layers}): {full_attn}")
    print(f"[dry-run] dynamic map has {len(dynamic)} pattern(s):")
    for pat, ov in dynamic.items():
        print(f"    {pat!r}: {ov}")
    print()
    print("[dry-run] resolved per-module bit map, layers 0-7:")
    layer_types = getattr(text_cfg, "layer_types", None) or [
        ("full_attention" if i in full_attn else "linear_attention") for i in range(num_layers)
    ]
    for i in range(min(8, num_layers)):
        ltype = layer_types[i]
        if ltype == "full_attention":
            mods = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
        else:
            mods = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
                     "linear_attn.in_proj_b", "linear_attn.in_proj_a", "linear_attn.out_proj"]
        mods += ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
        print(f"  layer {i:2d} [{ltype}]")
        for m in mods:
            full_name = f"model.language_model.layers.{i}.{m}"
            skip = dynamic_get(dynamic, full_name) is False
            bits = "SKIP(bf16)" if skip else f"{dynamic_get(dynamic, full_name, 'bits', default=4)}bit"
            print(f"      {full_name:<62s} -> {bits}")
    print()
    print("[dry-run] top-level module smoke test:")
    for name in ("lm_head",
                 "model.visual.blocks.0.attn.qkv",
                 "model.visual.merger.linear_fc1",
                 "mtp.layers.0.self_attn.q_proj",
                 "mtp.fc",
                 "model.language_model.layers.3.linear_attn.in_proj_a",
                 "model.language_model.layers.3.linear_attn.in_proj_qkv"):
        skip = dynamic_get(dynamic, name) is False
        bits = "SKIP(bf16)" if skip else f"{dynamic_get(dynamic, name, 'bits', default=4)}bit"
        print(f"      {name:<62s} -> {bits}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                     help="Print the resolved per-module bit map for layers 0-7 and exit. "
                          "Config-only load -- no weights, no GPU.")
    args = ap.parse_args()

    if not os.path.isdir(SRC):
        die(f"source model not found: {SRC}")

    cfg = load_config_only(SRC)
    text_cfg = getattr(cfg, "text_config", cfg)
    num_layers = getattr(text_cfg, "num_hidden_layers", None)
    if num_layers is None:
        die(f"could not read num_hidden_layers from {SRC}/config.json")

    dynamic, full_attn = build_dynamic_config(text_cfg, num_layers)

    if args.dry_run:
        print_dry_run_bitmap(dynamic, text_cfg, num_layers, full_attn)
        free_gb = disk_free_gb(DISK_CHECK_PATH)
        tag = "ok" if free_gb >= MIN_FREE_GB else "FAIL (would abort a real run)"
        print(f"\n[dry-run] disk free at {DISK_CHECK_PATH}: {free_gb:.1f}GB "
              f"(need >= {MIN_FREE_GB:.0f}GB for a real run) -> {tag}")
        print("[dry-run] OK -- no weights loaded, no GPU touched. Exiting.")
        return

    # ---------------- real run below this line ----------------
    free_gb = disk_free_gb(DISK_CHECK_PATH)
    if free_gb < MIN_FREE_GB:
        die(f"only {free_gb:.1f}GB free at {DISK_CHECK_PATH}; need >= {MIN_FREE_GB:.0f}GB "
            f"(override with EXP021_MIN_FREE_GB if you've sized it differently)")
    os.makedirs(OFFLOAD, exist_ok=True)
    check_gpus_idle()
    print(f"[preflight] ok -- src={SRC} out={OUT} offload={OFFLOAD} free={free_gb:.0f}GB")

    from transformers import AutoTokenizer
    from gptqmodel import GPTQModel, QuantizeConfig
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
    rng = random.Random(1234)

    def enc(text):
        e = tok(text, truncation=True, max_length=SEQ)
        if len(e["input_ids"]) < 256:
            return None
        return {"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]}

    # ---------- 1. real captured workload (flight recorder) ----------
    want_w = int(N * MIX["workload"])
    workload = []
    files = sorted(glob.glob(f"{FLIGHTREC}/*.json"))
    rng.shuffle(files)
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        parts = []
        for m in (d.get("messages") or []):
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                parts.append(f"<|{m.get('role','user')}|>\n{c}")
            elif isinstance(c, list):
                parts.append("\n".join(b.get("text", "") for b in c if isinstance(b, dict)))
        if d.get("tools"):
            parts.append("TOOLS:\n" + json.dumps(d["tools"])[:4000])
        blob = "\n\n".join(parts)
        # slice long conversations into multiple windows so one capture != one sample
        for i in range(0, max(1, len(blob) - 2000), SEQ * 3):
            s = enc(blob[i:i + SEQ * 4])
            if s: workload.append(s)
            if len(workload) >= want_w: break
        if len(workload) >= want_w: break
    print(f"[calib] workload: {len(workload)}/{want_w} samples from {len(files)} captures")

    # ---------- 2. local source code ----------
    want_c = int(N * MIX["code"])
    code = []
    srcs = []
    for root in CODE_ROOTS:
        srcs += glob.glob(f"{root}/**/*.py", recursive=True) + glob.glob(f"{root}/*.sh")
    rng.shuffle(srcs)
    for f in srcs:
        try:
            t = open(f, errors="ignore").read()
        except Exception:
            continue
        if len(t) < 800: continue
        for i in range(0, max(1, len(t) - 800), SEQ * 3):
            s = enc(t[i:i + SEQ * 4])
            if s: code.append(s)
            if len(code) >= want_c: break
        if len(code) >= want_c: break
    print(f"[calib] code: {len(code)}/{want_c} samples from {len(srcs)} files")

    # ---------- 3. generic c4 (fills the remainder) ----------
    want_g = N - len(workload) - len(code)
    generic = []
    for ex in load_dataset("allenai/c4", "en", split="train", streaming=True):
        t = ex.get("text", "")
        if len(t) < 400: continue
        s = enc(t)
        if s: generic.append(s)
        if len(generic) >= want_g: break
    print(f"[calib] c4: {len(generic)}/{want_g} samples")

    calib = workload + code + generic
    rng.shuffle(calib)
    print(f"[calib] TOTAL {len(calib)} samples "
          f"({len(generic)} c4 / {len(workload)} workload / {len(code)} code), max {SEQ} tok")

    qc = QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=False,
                         dynamic=dynamic,
                         offload_to_disk=True, offload_to_disk_path=OFFLOAD)

    print("[mixed48] loading + quantizing -- multi-hour; progress prints per layer...", flush=True)
    model = GPTQModel.load(SRC, qc)
    t0 = time.time()
    model.quantize(calib, batch_size=1)
    print(f"[mixed48] quantized in {(time.time()-t0)/60:.0f} min; saving -> {OUT}", flush=True)
    model.save(OUT)
    tok.save_pretrained(OUT)
    print(f"[mixed48] DONE -> {OUT}")
    print("NEXT: bench vs current champion (Qwen3.8-27B-GPTQ-Int4) before swapping prod.")


if __name__ == "__main__":
    main()
