#!/usr/bin/env python
"""Lane K7: W4A4 (s4 IMMA, CUTLASS sm75 + fused per-token/per-channel epilogue) vs production W4A16 Marlin, on the
per-rank TP=2 prefill shapes of Qwen3.8-27B.  Marlin runs the REAL checkpoint slices (U2 _ctlayer path); W4A4 runs random
int4 codes of the same shape (integer GEMM time is data-independent).  Arms are interleaved per rep, so contention from
the live engine hits all equally; we report min and median.
  w4a16 : Marlin (fp16 act, fp16 accumulate on sm_75)                        -- production
  gemm  : k7 w4a4_gemm alone (best config from a per-shape scan)
  aq128 : k7 act_quant_had, block-Hadamard 128 + per-token int4 (reads fp16 x, writes packed int4)
  aq512 : same with 512-blocks
  w8a8  : vLLM cutlass_scaled_mm int8 (sm75 c2x) -- the existing W8A8 integer path, for reference
Safe next to a live engine: refuses unless --min-free-mib free; caps the allocator at --cap-mib.
Usage: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/kevin/Desktop/wt-integrate/tools/u2 python tools/k7/w4a4_bench.py"""
import argparse, json, os, statistics, subprocess, sys
ap = argparse.ArgumentParser()
ap.add_argument("--Ms", default="512,1024,2048,3632")
ap.add_argument("--iters", type=int, default=25); ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--shapes", default="gate,down,gdn_qkvz,gdn_out,attn_qkv,attn_o")
ap.add_argument("--min-free-mib", type=int, default=420); ap.add_argument("--cap-mib", type=int, default=330)
ap.add_argument("--no-w8a8", action="store_true")
ap.add_argument("--json", default="/home/kevin/projects/lanes/k7/w4a4_bench.json")
a = ap.parse_args()
gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
free = int(subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).decode())
if free < a.min_free_mib:
    sys.exit(f"refusing: GPU{gpu} {free} MiB free < {a.min_free_mib}")
import torch
torch.cuda.set_per_process_memory_fraction(a.cap_mib / (torch.cuda.get_device_properties(0).total_memory / 2**20))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k7ext
from safetensors import safe_open
from _ctlayer import make_marlin_layer
from vllm import _custom_ops as ops

E = k7ext.ext()
MD = "/home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven/model.safetensors"
P = "model.language_model.layers"
f = safe_open(MD, "pt"); g = f.get_tensor


def col(names, rows):
    out = {"weight_packed": [], "weight_scale": [], "weight_zero_point": []}
    for n, r in zip(names, rows):
        out["weight_packed"].append(g(n + ".weight_packed")[:r]); out["weight_scale"].append(g(n + ".weight_scale")[:r])
        out["weight_zero_point"].append(g(n + ".weight_zero_point")[: r // 8])
    t = {k: torch.cat(v, 0) for k, v in out.items()}
    return t, t["weight_packed"].shape[0], t["weight_packed"].shape[1] * 8


def row(n, k):
    t = {"weight_packed": g(n + ".weight_packed")[:, : k // 8], "weight_scale": g(n + ".weight_scale")[:, : k // 128],
         "weight_zero_point": g(n + ".weight_zero_point")[:, : k // 128]}
    return {k2: v.contiguous() for k2, v in t.items()}, t["weight_packed"].shape[0], k


# (builder, layers using it per forward, multiplier): "gate" = half of the merged gate_up (memory cap); x2 = gate_up
SHAPES = {
    "gate": (lambda: col([f"{P}.3.mlp.gate_proj"], [8704]), 64, 2),
    "down": (lambda: row(f"{P}.3.mlp.down_proj", 8704), 64, 1),
    "gdn_qkvz": (lambda: col([f"{P}.0.linear_attn.in_proj_qkv", f"{P}.0.linear_attn.in_proj_z"], [5120, 3072]), 48, 1),
    "gdn_out": (lambda: row(f"{P}.0.linear_attn.out_proj", 3072), 48, 1),
    "attn_qkv": (lambda: col([f"{P}.3.self_attn.q_proj", f"{P}.3.self_attn.k_proj", f"{P}.3.self_attn.v_proj"], [6144, 512, 512]), 16, 1),
    "attn_o": (lambda: row(f"{P}.3.self_attn.o_proj", 3072), 16, 1),
}


def timeit(fn):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); fn(); e.record(); e.synchronize(); return s.elapsed_time(e)


rows = []
Ms = [int(m) for m in a.Ms.split(",")]
for name in a.shapes.split(","):
    mk, nl, mult = SHAPES[name]
    t, N, K = mk()
    t = {k: v.contiguous() for k, v in t.items()}; t["weight_scale"] = t["weight_scale"].half()
    l16, s16 = make_marlin_layer(t, N, K, device="cuda")
    B4 = torch.randint(-128, 128, (N, K // 2), dtype=torch.int8, device="cuda")
    sb = torch.rand(N, device="cuda") * 1e-2
    B8 = None
    if not a.no_w8a8:
        B8 = torch.randint(-127, 128, (N, K), dtype=torch.int8, device="cuda").t()  # [K,N] column-major view
    # config scan at the largest M
    Mx = max(Ms)
    x = torch.randn(Mx, K, device="cuda", dtype=torch.float16)
    A4, sa = E.act_quant_had(x, 128, 7.0)
    best, cfg_t = None, {}
    for cfg in range(7):
        try:
            for _ in range(2): E.w4a4_gemm(A4, B4, sa, sb, cfg)
            cfg_t[cfg] = min(timeit(lambda: E.w4a4_gemm(A4, B4, sa, sb, cfg)) for _ in range(8))
        except RuntimeError as ex:
            cfg_t[cfg] = None
    best = min((c for c in cfg_t if cfg_t[c]), key=lambda c: cfg_t[c])
    print(f"{name:9s} N={N} K={K}  cfg scan @M={Mx} (ms): " + " ".join(f"c{c}={v:.3f}" if v else f"c{c}=x" for c, v in cfg_t.items())
          + f"  -> cfg{best}", flush=True)
    del x, A4, sa
    for M in Ms:
        x = torch.randn(M, K, device="cuda", dtype=torch.float16); x[:, :8] *= 20
        A4, sa = E.act_quant_had(x, 128, 7.0)
        _, SI, SM = E.act_quant_h128g(x, 7.0, 11)
        fns = {"w4a16": lambda: s16.apply_weights(l16, x, None),
               "gemm": lambda: E.w4a4_gemm(A4, B4, sa, sb, best),
               "aq128": lambda: E.act_quant_had(x, 128, 7.0),
               "aq512": lambda: E.act_quant_had(x, 512, 7.0),
               "aqv2": lambda: E.act_quant_h128(x, 7.0),
               "gemmg": lambda: E.w4a4g_gemm(A4, B4, SI, SM, sb, 11, 2),
               "aqg": lambda: E.act_quant_h128g(x, 7.0, 11)}
        if B8 is not None:
            a8 = torch.randint(-127, 128, (M, K), dtype=torch.int8, device="cuda"); sa8 = torch.rand(M, 1, device="cuda")
            sb8 = torch.rand(1, N, device="cuda")
            fns["w8a8"] = lambda: ops.cutlass_scaled_mm(a8, B8, sa8, sb8, torch.float16)
        for fn in fns.values():
            for _ in range(a.warmup): fn()
        ts = {k: [] for k in fns}
        for _ in range(a.iters):
            for k, fn in fns.items():
                ts[k].append(timeit(fn))
        fl = 2.0 * M * N * K
        r = {"shape": name, "M": M, "N": N, "K": K, "layers": nl, "mult": mult, "cfg": best}
        for k in ts:
            r[k] = {"min": min(ts[k]), "med": statistics.median(ts[k])}
            r[k]["tops_min"] = fl / r[k]["min"] / 1e9
        r["w4a4_total_min"] = r["gemm"]["min"] + min(r["aq128"]["min"], r["aqv2"]["min"])
        r["w4a4g_total_min"] = r["gemmg"]["min"] + r["aqg"]["min"]
        r["speedup_g_with_actq"] = r["w4a16"]["min"] / r["w4a4g_total_min"]
        r["speedup_gemm_only"] = r["w4a16"]["min"] / r["gemm"]["min"]
        r["speedup_with_actq"] = r["w4a16"]["min"] / r["w4a4_total_min"]
        rows.append(r)
        extra = f"  w8a8 {r['w8a8']['min']:.3f} ({r['w8a8']['tops_min']:.0f} TOPS)" if "w8a8" in r else ""
        print(f"  M={M:5d} w4a16 {r['w4a16']['min']:7.3f} ms ({r['w4a16']['tops_min']:5.1f} TF) | w4a4 gemm {r['gemm']['min']:7.3f} "
              f"({r['gemm']['tops_min']:5.1f} TOPS) + aq128 {r['aq128']['min']:.3f} v2 {r['aqv2']['min']:.3f} (aq512 {r['aq512']['min']:.3f}){extra} | "
              f"speedup gemm-only {r['speedup_gemm_only']:.2f}x, with act-quant {r['speedup_with_actq']:.2f}x | GROUP: gemm {r['gemmg']['min']:.3f} ({r['gemmg']['tops_min']:.0f} TOPS) + aqg {r['aqg']['min']:.3f} -> {r['speedup_g_with_actq']:.2f}x", flush=True)
        del x, A4, sa, fns, SI, SM
        if B8 is not None:
            del a8
    del l16, B4, B8; torch.cuda.empty_cache()

tot = {"w4a16": 0.0, "w4a4": 0.0, "w4a4_gemm_only": 0.0, "w4a4g": 0.0}
for r in rows:
    if r["M"] == max(Ms):
        k = r["layers"] * r["mult"]
        tot["w4a16"] += r["w4a16"]["min"] * k; tot["w4a4"] += r["w4a4_total_min"] * k; tot["w4a4_gemm_only"] += r["gemm"]["min"] * k; tot["w4a4g"] += r["w4a4g_total_min"] * k
print(f"PER-CHUNK (M={max(Ms)}) Marlin-linear time, all layers, one rank (ms): {json.dumps({k: round(v, 1) for k, v in tot.items()})}"
      f"  ratio w4a16/w4a4 {tot['w4a16'] / max(tot['w4a4'], 1e-9):.2f}x")
clk = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu", "--format=csv,noheader"]).decode().strip()
json.dump({"rows": rows, "chunk_ms": tot, "gpu_state_end": clk, "contended_by_live_engine": True}, open(a.json, "w"), indent=1)
print("gpu state at end:", clk)
