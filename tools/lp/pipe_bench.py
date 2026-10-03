#!/usr/bin/env python
"""Lane LP LP_PIPE bench: does register-staged prefetch (Turing has no cp.async) speed up Marlin?
Real checkpoint per-rank shapes. Arms:
  stock16 : vLLM's production Marlin W4A16 (CompressedTensorsWNA16 -> MarlinLinearKernel)
  r16     : the same template compiled standalone WITHOUT LP_PIPE (must be bit-identical to stock16)
  p16     : the same template WITH LP_PIPE (must be bit-identical: only load timing changes)
  stock8  : stock W4A8 (per-token int8);  pe0 : LP_PIPE + LP_A8E with EMAX=0 (== stock8 numerics);  pe3 : LP_PIPE + LP_A8E EMAX 3
Usage (window): LP_IN_WINDOW=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=wt-lp:wt-integrate/tools/u2 python tools/lp/pipe_bench.py --json out.json"""
import os as _os, subprocess as _sp, sys as _sys
if not _os.environ.get("GPU_CAP_MIB") and _os.environ.get("LP_IN_WINDOW") != "1" and not _os.environ.get("WINDOW_ID"):
    _sys.exit("RL rule: launch via ~/projects/lanes/windows/gpuok.sh --run --lane LP --job <name> --gpu <N> --cap <MiB> -- <cmd> (or inside a window)")
_g = _sp.run(["/home/kevin/projects/lanes/windows/gpuok.sh", _os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0], "2500"], capture_output=True, text=True)
if _g.returncode != 0 and _os.environ.get("LP_IN_WINDOW") != "1" and not _os.environ.get("WINDOW_ID"):
    _sys.exit("gpuok.sh refused: " + _g.stdout.strip())
import argparse, json, os, statistics, sys, torch
ap = argparse.ArgumentParser(); ap.add_argument("--Ms", default="16,48,64,256,1024,3632"); ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--shapes", default="gate_up,down,gdn_qkvz,gdn_out"); ap.add_argument("--json", default="/home/kevin/projects/lanes/lp/pipe_bench.json")
a = ap.parse_args()
torch.cuda.set_per_process_memory_fraction(2500 / 22528)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lp_shapes as S
sys.path.insert(0, "/home/kevin/Desktop/wt-integrate/tools/u2")
from _ctlayer import make_marlin_layer
from vllm.model_executor.layers.quantization.utils import lp_w4a8g as L


def tmin(fn, iters):
    for _ in range(3): fn()
    ts = []
    for _ in range(iters):
        s_, e_ = torch.cuda.Event(True), torch.cuda.Event(True); s_.record(); fn(); e_.record(); e_.synchronize(); ts.append(s_.elapsed_time(e_))
    return min(ts), statistics.median(ts)


p16, r16, pe0, pe3 = L.load_ext("p16", 0), L.load_ext("r16", 0), L.load_ext("pe", 0), L.load_ext("pe", 3)
Ms = [int(m) for m in a.Ms.split(",")]
rows = []
for name in a.shapes.split(","):
    mk, nl = S.SHAPES[name]
    t, N, K = mk(); t = {k: v.contiguous() for k, v in t.items()}; t["weight_scale"] = t["weight_scale"].half()
    l16, s16 = make_marlin_layer(dict(t), N, K, device="cuda")
    os.environ["VLLM_MARLIN_INPUT_DTYPE"] = "int8"
    try:
        l8, s8 = make_marlin_layer(dict(t), N, K, device="cuda")
    finally:
        os.environ.pop("VLLM_MARLIN_INPUT_DTYPE", None)
    ws = torch.zeros(1024, dtype=torch.int32, device="cuda")
    q16, sc16, z16 = l16.weight_packed.data, l16.weight_scale.data, l16.weight_zero_point.data
    q8, sc8, z8 = l8.weight_packed.data, l8.weight_scale.data, l8.weight_zero_point.data   # int16 levels (4096)
    # pe3 needs 512-level int scales from the real fp16 ones (same layout as the int8 path)
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales
    sp = marlin_permute_scales(t["weight_scale"].cuda().t().contiguous(), size_k=K, size_n=N, group_size=128, is_a_8bit=True).contiguous()
    sc8e3, wg3 = L.process_scales_e(sp, 512)
    gsc = float(getattr(l8, "input_global_scale").item()) if getattr(l8, "input_global_scale", None) is not None else 1.0
    for M in Ms:
        x = torch.randn(M, K, device="cuda", dtype=torch.float16); x[:, :8] *= 20
        y_st = s16.apply_weights(l16, x, None)
        y_r = r16.gemm16(x, q16, sc16, z16, ws, N, True); y_p = p16.gemm16(x, q16, sc16, z16, ws, N, True)
        y_s8 = s8.apply_weights(l8, x, None)
        q0, e0, r0 = L.quant_e(x, gsc, 0); y_pe0 = pe0.gemm(q0, r0, e0, q8, sc8, z8, ws, N, True)
        q3, e3, r3 = L.quant_e(x, wg3, 3); y_pe3 = pe3.gemm(q3, r3, e3, q8, sc8e3, z8, ws, N, True)
        rel = lambda u, v: ((u.float() - v.float()).norm() / v.float().norm()).item()
        r = {"shape": name, "M": M, "N": N, "K": K, "layers": nl,
             "r16_eq_stock": bool(torch.equal(y_r, y_st)), "p16_eq_stock": bool(torch.equal(y_p, y_st)),
             "p16_rel_stock": rel(y_p, y_st), "pe0_rel_stock8": rel(y_pe0, y_s8), "pe3_rel_w4a16": rel(y_pe3, y_st), "stock8_rel_w4a16": rel(y_s8, y_st)}
        arms = {"stock16": lambda: s16.apply_weights(l16, x, None), "r16": lambda: r16.gemm16(x, q16, sc16, z16, ws, N, True),
                "p16": lambda: p16.gemm16(x, q16, sc16, z16, ws, N, True), "stock8": lambda: s8.apply_weights(l8, x, None),
                "pe0": lambda: (lambda qq, ee, rr: pe0.gemm(qq, rr, ee, q8, sc8, z8, ws, N, True))(*L.quant_e(x, gsc, 0)),
                "pe3_gemm_only": lambda: pe3.gemm(q3, r3, e3, q8, sc8e3, z8, ws, N, True), "quant_e": lambda: L.quant_e(x, wg3, 3)}
        tt = {k: [] for k in arms}
        for rep in range(3):
            for k, fn in arms.items(): tt[k].append(tmin(fn, a.iters)[0])
        for k in arms: r[k] = min(tt[k])
        r["p16_speedup"] = r["stock16"] / r["p16"]; r["pe3_speedup_vs16"] = r["stock16"] / (r["pe3_gemm_only"] + r["quant_e"])
        rows.append(r)
        print(f"{name:8s} M={M:5d} stock16 {r['stock16']:.3f} r16 {r['r16']:.3f} p16 {r['p16']:.3f} (x{r['p16_speedup']:.2f}, eq={r['p16_eq_stock']}/{r['r16_eq_stock']}) | "
              f"stock8 {r['stock8']:.3f} pe3 {r['pe3_gemm_only']:.3f}+q{r['quant_e']:.3f} (x{r['pe3_speedup_vs16']:.2f} vs 16, rel {r['pe3_rel_w4a16']:.4f}) pe0~stock8 rel {r['pe0_rel_stock8']:.1e}", flush=True)
        del x
    del l16, l8; torch.cuda.empty_cache()
json.dump(rows, open(a.json, "w"), indent=1)
