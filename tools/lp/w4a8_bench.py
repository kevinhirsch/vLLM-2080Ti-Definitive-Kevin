#!/usr/bin/env python
"""Lane LP: W4A8-INT8 Marlin on the SHIPPED asymmetric (zero-point) int4 weights vs production W4A16 Marlin.
REAL checkpoint tensors (layer 3 mlp/attn, layer 0 GDN), per-rank TP=2 slices, real vLLM CompressedTensorsWNA16 ->
MarlinLinearKernel objects (wt-lp branch lp/turing-int8 relaxes the uint4b8-only assert; s8 x u4 kernels are compiled for sm_75).
Arms (interleaved per rep so contention from a live engine hits both equally):
  w4a16 : production kernel (fp16 act, fp16 accumulate inside Marlin on sm_75)
  w4a8  : per_token_quant_int8 + s8xu4(zp) Marlin (IMMA int8, int32 accumulate, int16 group scales)
  quant : per_token_quant_int8 alone (included in w4a8)
Correctness: w4a8 kernel output vs a CPU emulation of the same arithmetic (int8 act, int scales) on 32 rows -> rel err must be ~fp16 rounding.
Safe next to a live engine: refuses unless >= --min-free-mib free on the chosen GPU; caps its own allocator at --cap-mib.
Usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/kevin/Desktop/wt-lp:/home/kevin/Desktop/wt-integrate/tools/u2 python tools/lp/w4a8_bench.py
"""
import os as _os, subprocess as _sp, sys as _sys
if not _os.environ.get("GPU_CAP_MIB") and _os.environ.get("LP_IN_WINDOW") != "1" and not _os.environ.get("WINDOW_ID"):
    _sys.exit("RL rule: launch via ~/projects/lanes/windows/gpuok.sh --run --lane LP --job <name> --gpu <N> --cap <MiB> -- <cmd> (or inside a window)")
_g = _sp.run(["/home/kevin/projects/lanes/windows/gpuok.sh", _os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0], "800"], capture_output=True, text=True)
if _g.returncode != 0 and _os.environ.get("LP_IN_WINDOW") != "1" and not _os.environ.get("WINDOW_ID"):
    _sys.exit("gpuok.sh refused: " + _g.stdout.strip() + " " + _g.stderr.strip())

import argparse, json, os, statistics, subprocess, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--Ms", default="16,64,256,512,1024,2048,3632")
ap.add_argument("--iters", type=int, default=30); ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--shapes", default="gate_up,down,gdn_qkvz,gdn_out,attn_qkv,attn_o")
ap.add_argument("--min-free-mib", type=int, default=900); ap.add_argument("--cap-mib", type=int, default=520)
ap.add_argument("--sustain", type=float, default=0, help="seconds per arm of back-to-back gate_up M=3632 to read clocks/power under each format")
ap.add_argument("--json", default="/home/kevin/projects/lanes/lp/w4a8_bench.json")
a = ap.parse_args()
gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
free = int(subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).decode().strip())
if free < a.min_free_mib:
    sys.exit(f"refusing: GPU{gpu} has {free} MiB free < {a.min_free_mib}")
import torch
from safetensors import safe_open
torch.cuda.set_per_process_memory_fraction(a.cap_mib / (torch.cuda.get_device_properties(0).total_memory / 2**20))
from _ctlayer import make_marlin_layer
from vllm import _custom_ops as ops
G8 = None
LPE = int(os.environ.get("LP_EMAX", "3"))
if os.environ.get("LP_G8", "1") == "1":
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import lp_g8 as G8
        G8.build("g"); G8.build("e", LPE)
        print("LP_A8G extension loaded", flush=True)
    except Exception as ex:  # keep the stock arms running
        print("LP_A8G extension unavailable:", repr(ex)[:300], flush=True); G8 = None

MD = "/home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven/model.safetensors"
P = "model.language_model.layers"
f = safe_open(MD, "pt")
g = lambda n: f.get_tensor(n)


def col(names, rows):  # column-parallel: rank-0 rows of each, concatenated (merged layer)
    out = {"weight_packed": [], "weight_scale": [], "weight_zero_point": []}
    for n, r in zip(names, rows):
        out["weight_packed"].append(g(n + ".weight_packed")[:r]); out["weight_scale"].append(g(n + ".weight_scale")[:r])
        out["weight_zero_point"].append(g(n + ".weight_zero_point")[: r // 8])
    t = {k: torch.cat(v, 0) for k, v in out.items()}
    return t, t["weight_packed"].shape[0], t["weight_packed"].shape[1] * 8


def row(n, k):  # row-parallel: rank-0 input slice
    t = {"weight_packed": g(n + ".weight_packed")[:, : k // 8], "weight_scale": g(n + ".weight_scale")[:, : k // 128],
         "weight_zero_point": g(n + ".weight_zero_point")[:, : k // 128]}
    return {k2: v.contiguous() for k2, v in t.items()}, t["weight_packed"].shape[0], k


SHAPES = {
    "gate_up": (lambda: col([f"{P}.3.mlp.gate_proj", f"{P}.3.mlp.up_proj"], [8704, 8704]), 64),
    "down": (lambda: row(f"{P}.3.mlp.down_proj", 8704), 64),
    "gdn_qkvz": (lambda: col([f"{P}.0.linear_attn.in_proj_qkv", f"{P}.0.linear_attn.in_proj_z"], [5120, 3072]), 48),
    "gdn_out": (lambda: row(f"{P}.0.linear_attn.out_proj", 3072), 48),
    "attn_qkv": (lambda: col([f"{P}.3.self_attn.q_proj", f"{P}.3.self_attn.k_proj", f"{P}.3.self_attn.v_proj"], [6144, 512, 512]), 16),
    "attn_o": (lambda: row(f"{P}.3.self_attn.o_proj", 3072), 16),
}


def emulate(t, x, N, K):
    """CPU fp32 emulation of s8xu4(zp) Marlin: per-token int8 act, int16 group scales (per-layer max -> 4096)."""
    sh = torch.arange(8, dtype=torch.int32) * 4
    q = ((t["weight_packed"].unsqueeze(-1) >> sh) & 15).reshape(N, K // 128, 128)
    zp = ((t["weight_zero_point"].unsqueeze(1) >> sh.view(1, 8, 1)) & 15).reshape(N, -1)
    s = t["weight_scale"].half().float(); smax = s.max(); s_int = torch.round(s / smax * 4096)
    w = ((q - zp.unsqueeze(-1)).float() * (s_int * smax / 4096).unsqueeze(-1)).reshape(N, K)
    w16 = ((q - zp.unsqueeze(-1)).float() * s.unsqueeze(-1)).reshape(N, K)
    xf = x.float(); amax = xf.abs().amax(-1, keepdim=True); sc = amax / 127
    xq = torch.clamp(torch.round(xf / sc), -127, 127) * sc
    return xq @ w.T, xf @ w16.T


def emulate_g(t, x, N, K):
    """CPU emulation of the LP_A8G kernel: per-(row,128) int8 act, fp16 weight group scales."""
    sh = torch.arange(8, dtype=torch.int32) * 4
    q = ((t["weight_packed"].unsqueeze(-1) >> sh) & 15).reshape(N, K // 128, 128)
    zp = ((t["weight_zero_point"].unsqueeze(1) >> sh.view(1, 8, 1)) & 15).reshape(N, -1)
    w = ((q - zp.unsqueeze(-1)).float() * t["weight_scale"].half().float().unsqueeze(-1)).reshape(N, K)
    xg = x.float().view(x.shape[0], -1, 128); sc = xg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127
    return (torch.clamp(torch.round(xg / sc), -127, 127) * sc).view(x.shape[0], K) @ w.T


def timeit(fn):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); fn(); e.record(); e.synchronize(); return s.elapsed_time(e)


rows, torch_ = [], torch
torch.manual_seed(0)
Ms = [int(m) for m in a.Ms.split(",")]
for name in a.shapes.split(","):
    mk, nl = SHAPES[name]
    t, N, K = mk()
    t = {k: v.contiguous() for k, v in t.items()}
    t["weight_scale"] = t["weight_scale"].half()
    l16, s16 = make_marlin_layer({k: v for k, v in t.items()}, N, K, device="cuda")
    from vllm.model_executor.layers.quantization.utils import lp_w8x as W8
    st8x = W8.W8XState(l16.weight_packed.data, l16.weight_scale.data, l16.weight_zero_point.data, K, N)
    # exactness of the expansion on the REAL CUDA-repacked layout vs a CPU computation from the CT tensors
    _w8 = W8.expand(st8x).cpu().float()
    _sh = torch.arange(8, dtype=torch.int32) * 4
    _q = ((t["weight_packed"].unsqueeze(-1) >> _sh) & 15).reshape(N, K // 128, 128).float()
    _z = ((t["weight_zero_point"].unsqueeze(1) >> _sh.view(1, 8, 1)) & 15).reshape(N, -1).float()
    _s = t["weight_scale"].half().float(); _sch = _s.amax(1, keepdim=True) * 15.0 / 127.0
    _v = ((_q - _z.unsqueeze(-1)) * (_s / _sch).unsqueeze(-1)).reshape(N, K)
    _v = torch.clamp(torch.where(_v >= 0, torch.floor(_v + 0.5), torch.ceil(_v - 0.5)), -127, 127)
    w8x_exact_frac = float((_w8 == _v).float().mean()); w8x_maxdiff = float((_w8 - _v).abs().max())
    print(f"  W8X expand vs CT-reference: exact {w8x_exact_frac:.6f} maxdiff {w8x_maxdiff}", flush=True)
    del _w8, _q, _z, _v
    os.environ["VLLM_MARLIN_INPUT_DTYPE"] = "int8"
    try:
        l8, s8 = make_marlin_layer({k: v for k, v in t.items()}, N, K, device="cuda")
    finally:
        os.environ.pop("VLLM_MARLIN_INPUT_DTYPE", None)
    assert s8.kernel.config.act_type == torch.int8, "int8 act not engaged"
    g8 = G8.G8Linear({k: v for k, v in t.items()}, N, K) if G8 is not None else None
    e2 = G8.G8Linear({k: v for k, v in t.items()}, N, K, mode="e", emax=LPE) if G8 is not None else None
    # correctness on 32 rows with outlier channels like post-norm activations
    xc = torch.randn(32, K) ; xc[:, :8] *= 20
    y8 = s8.apply_weights(l8, xc.half().cuda(), None).float().cpu(); y16 = s16.apply_weights(l16, xc.half().cuda(), None).float().cpu()
    e8, e16 = emulate(t, xc.half(), N, K)
    corr = {"rel_w4a8_vs_emul": ((y8 - e8).norm() / e8.norm()).item(), "rel_w4a16_vs_fp32": ((y16 - e16).norm() / e16.norm()).item(),
            "rel_w4a8_vs_w4a16_kernel": ((y8 - y16).norm() / y16.norm()).item()}
    if g8 is not None:
        yg = g8.forward(xc.half().cuda(), G8.quant_act_g128_triton).float().cpu(); eg = emulate_g(t, xc.half(), N, K)
        corr["rel_w4a8g_vs_emul"] = ((yg - eg).norm() / eg.norm()).item()
        corr["rel_w4a8g_vs_fp32"] = ((yg - e16).norm() / e16.norm()).item()
        corr["rel_w4a8_vs_fp32"] = ((y8 - e16).norm() / e16.norm()).item()
        from w4a8_bench_emul import emulate_e
        ye = e2.forward(xc.half().cuda()).float().cpu(); ee = emulate_e(t, xc.half(), N, K, LPE, e2.level)
        corr["rel_w4a8e2_vs_emul"] = ((ye - ee).norm() / ee.norm()).item(); corr["rel_w4a8e2_vs_fp32"] = ((ye - e16).norm() / e16.norm()).item()
        for Mt in (1, 7, 65, 300):  # odd M: tail rows + M-split paths
            xt = torch.randn(Mt, K) ; xt[:, :8] *= 20
            yt = g8.forward(xt.half().cuda(), G8.quant_act_g128_triton).float().cpu(); et = emulate_g(t, xt.half(), N, K)
            corr[f"rel_w4a8g_vs_emul_M{Mt}"] = ((yt - et).norm() / et.norm()).item()
    print(f"{name:9s} N={N} K={K} correctness {json.dumps({k: round(v, 5) for k, v in corr.items()})}", flush=True)
    for M in Ms:
        x = torch.randn(M, K, device="cuda", dtype=torch.float16); x[:, :8] *= 20
        fns = {"w4a16": lambda: s16.apply_weights(l16, x, None), "w4a8": lambda: s8.apply_weights(l8, x, None),
               "quant": lambda: ops.scaled_int8_quant(x, None, None, symmetric=True),
               "w8x": lambda: W8.gemm(x, st8x), "w8x_expand": lambda: W8.expand(st8x, W8.scratch(x.device, N * K).view(N, K))}
        if g8 is not None:
            fns["w4a8g"] = lambda: g8.forward(x, G8.quant_act_g128_triton)
            fns["quant_g"] = lambda: G8.quant_act_g128_triton(x)
            fns["w4a8e2"] = lambda: e2.forward(x)
        for fn in fns.values():
            for _ in range(a.warmup): fn()
        ts = {k: [] for k in fns}
        for _ in range(a.iters):
            for k, fn in fns.items():
                ts[k].append(timeit(fn))
        fl = 2.0 * M * N * K
        r = {"shape": name, "M": M, "N": N, "K": K, "layers": nl, **{f"corr_{k}": v for k, v in corr.items()},
             "w8x_exact_frac": w8x_exact_frac, "w8x_maxdiff": w8x_maxdiff}
        for k in ts:
            r[k] = {"med": statistics.median(ts[k]), "min": min(ts[k]), "p25": sorted(ts[k])[len(ts[k]) // 4]}
            r[k]["tflops_min"] = fl / r[k]["min"] / 1e9
        r["speedup_min"] = r["w4a16"]["min"] / r["w4a8"]["min"]; r["speedup_med"] = r["w4a16"]["med"] / r["w4a8"]["med"]
        r["speedup_w8x_min"] = r["w4a16"]["min"] / r["w8x"]["min"]
        yx = W8.gemm(x[:64], st8x).float(); y1 = s16.apply_weights(l16, x[:64], None).float()
        r["w8x_rel_vs_w4a16"] = ((yx - y1).norm() / y1.norm()).item()
        if "w4a8g" in r: r["speedup_g_min"] = r["w4a16"]["min"] / r["w4a8g"]["min"]; r["speedup_e2_min"] = r["w4a16"]["min"] / r["w4a8e2"]["min"]
        rows.append(r)
        print(f"  M={M:5d} w4a16 {r['w4a16']['min']:7.3f}/{r['w4a16']['med']:7.3f} ms ({r['w4a16']['tflops_min']:5.1f} TF)  "
              f"w4a8 {r['w4a8']['min']:7.3f}/{r['w4a8']['med']:7.3f} ms ({r['w4a8']['tflops_min']:5.1f} TOPS)  quant {r['quant']['min']:.3f}  "
              f"speedup min {r['speedup_min']:.2f}x med {r['speedup_med']:.2f}x | W8X {r['w8x']['min']:.3f} x{r['speedup_w8x_min']:.2f} (expand {r['w8x_expand']['min']:.3f}, rel {r['w8x_rel_vs_w4a16']:.4f})" +
              (f" | w4a8g {r['w4a8g']['min']:7.3f} ms ({r['w4a8g']['tflops_min']:5.1f}) x{r['speedup_g_min']:.2f} quant_g {r['quant_g']['min']:.3f} | e2 x{r['speedup_e2_min']:.2f}" if "w4a8g" in r else ""), flush=True)
        del x
    del l16, l8, g8, e2; torch.cuda.empty_cache()
tot = {}
for r in rows:
    if r["M"] == max(Ms):
        for k in ("w4a16", "w4a8", "w4a8g", "w4a8e2", "w8x"):
            if k in r: tot[k] = tot.get(k, 0) + r[k]["min"] * r["layers"]
print("PER-CHUNK linear time (min, ms, all layers, M=%d): %s  ratio %.2fx" % (max(Ms), {k: round(v, 1) for k, v in tot.items()}, tot["w4a16"] / tot["w4a8"]))
sustain = {}
if a.sustain > 0:
    import threading
    t, N, K = SHAPES["gate_up"][0](); t = {k: v.contiguous() for k, v in t.items()}; t["weight_scale"] = t["weight_scale"].half()
    l16, s16 = make_marlin_layer(dict(t), N, K, device="cuda")
    os.environ["VLLM_MARLIN_INPUT_DTYPE"] = "int8"; l8, s8 = make_marlin_layer(dict(t), N, K, device="cuda"); os.environ.pop("VLLM_MARLIN_INPUT_DTYPE")
    x = torch.randn(3632, K, device="cuda", dtype=torch.float16)
    for arm, fn in (("w4a16", lambda: s16.apply_weights(l16, x, None)), ("w4a8", lambda: s8.apply_weights(l8, x, None))):
        samples, stop = [], threading.Event()
        def samp():
            while not stop.is_set():
                samples.append(subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=clocks.sm,power.draw,temperature.gpu", "--format=csv,noheader,nounits"]).decode().strip())
                time.sleep(1)
        th = threading.Thread(target=samp); th.start(); n = 0; t0 = time.time()
        while time.time() - t0 < a.sustain:
            for _ in range(20): fn()
            torch.cuda.synchronize(); n += 20
        el = time.time() - t0; stop.set(); th.join()
        v = [[float(z) for z in sm.split(",")] for sm in samples[2:]] or [[0, 0, 0]]
        sustain[arm] = {"ms_per_gemm": el / n * 1e3, "sm_mhz_mean": sum(r[0] for r in v) / len(v), "power_w_mean": sum(r[1] for r in v) / len(v),
                        "temp_c_end": v[-1][2], "samples": len(v)}
        print(f"SUSTAIN {arm}: {sustain[arm]}", flush=True)
        time.sleep(5)
clk = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu", "--format=csv,noheader"]).decode().strip()
json.dump({"rows": rows, "chunk_ms": tot, "gpu_state_end": clk, "sustain": sustain, "contended_by_live_engine": True}, open(a.json, "w"), indent=1)
print("gpu state at end:", clk)
