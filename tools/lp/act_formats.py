#!/usr/bin/env python
"""Lane LP: "FP8-class on Turing" precision gap, measured on REAL linear inputs of the shipped model.
Runs the U2 CPU fp32 reference over the first --layers layers on --windows of the U2 estate windows, captures every Marlin linear's
input, and quantizes it in each 8-bit format, reporting SQNR (dB) and the GEMM-output relative error against the layer's real weights:
  int8_tok    : per-token symmetric int8 (what W4A8 Marlin does; IMMA int8 on sm_75)
  e4m3_tok    : per-token scaled FP8 E4M3 (what Ada/Hopper FP8 GEMMs do with dynamic per-token scales)
  e5m2_tok    : per-token scaled FP8 E5M2
  int8_g128   : per-(row,128-group) float scale (the proposed s8 Marlin epilogue; aligns with the weight group)
  int8_g32    : per-(row,32) float scale
  mx_int8_b32 : block-scaled int8, one power-of-two scale per 32 contiguous K elements (MX-style; on sm_75 = IMMA per k-block +
                integer shift-rescale of the int32 partial: the kernel cost of a per-group activation scale)
  mx_e4m3_b32 : MXFP8 E4M3 (OCP MX spec: E8M0 scale per 32)
Usage: CUDA_VISIBLE_DEVICES= python tools/lp/act_formats.py --layers 8 --windows 3"""
import argparse, json, os, sys, time, types
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
sys.path.insert(0, "/home/kevin/Desktop/wt-integrate/tools/u2")
import ref_dump as R
ap = argparse.ArgumentParser(); ap.add_argument("--layers", type=int, default=8); ap.add_argument("--windows", type=int, default=3)
ap.add_argument("--rows", type=int, default=256); ap.add_argument("--out", default="/home/kevin/projects/lanes/lp/act_formats.json")
a = ap.parse_args()
ref = torch.load("/home/kevin/projects/lanes/u2-quant/ref.pt"); ids = ref["ids"][: a.windows]
B, T = ids.shape
E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2
MAX = {E4: 448.0, E5: 57344.0}


def q_int8_tok(x):
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 127
    return torch.clamp(torch.round(x / s), -127, 127) * s


def q_fp8_tok(x, dt):
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / MAX[dt]
    return (x / s).to(dt).float() * s


def q_mx(x, kind, blk=32):
    xs = x.view(x.shape[0], -1, blk)
    amax = xs.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    if kind == "int8":
        e = torch.ceil(torch.log2(amax / 127)); s = torch.pow(2.0, e)
        q = torch.clamp(torch.round(xs / s), -127, 127) * s
    else:  # MXFP8 E4M3: shared exponent so that amax maps under 448 (OCP: floor(log2 amax) - emax(8))
        e = torch.floor(torch.log2(amax)) - 8; s = torch.pow(2.0, e)
        q = (xs / s).clamp(-448, 448).to(E4).float() * s
    return q.view_as(x)


def q_int8_g(x, g=128):
    xg = x.view(x.shape[0], -1, g); sc = xg.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 127
    return (torch.clamp(torch.round(xg / sc), -127, 127) * sc).view_as(x)


FMTS = {"int8_tok": q_int8_tok, "int8_g128": q_int8_g, "int8_g32": lambda x: q_int8_g(x, 32), "e4m3_tok": lambda x: q_fp8_tok(x, E4), "e5m2_tok": lambda x: q_fp8_tok(x, E5),
        "mx_int8_b32": lambda x: q_mx(x, "int8"), "mx_e4m3_b32": lambda x: q_mx(x, "e4m3")}
QW = {}
stats = {}
_lin = torch.nn.functional.linear


def lin(x, w, b=None):
    n = QW.get(id(w))
    if n is not None:
        xs = x.reshape(-1, x.shape[-1]); xs = xs[:: max(1, xs.shape[0] // a.rows)][: a.rows].half().float()
        y0 = xs @ w.T
        st = stats.setdefault(n, {})
        for f, fn in FMTS.items():
            xq = fn(xs)
            st[f + "_sqnr_db"] = (10 * torch.log10(xs.pow(2).sum() / (xq - xs).pow(2).sum().clamp(min=1e-30))).item()
            st[f + "_out_rel"] = ((xq @ w.T - y0).norm() / y0.norm()).item()
    return _lin(x, w, b)


R.F = types.SimpleNamespace(**{k: getattr(torch.nn.functional, k) for k in dir(torch.nn.functional) if not k.startswith("__")}); R.F.linear = lin
x = R.load_embed()[ids.reshape(-1)].float(); T0 = time.time()
for i in range(a.layers):
    W = R.load_layer_weights(i)
    for n, w in W.items():
        if isinstance(w, torch.Tensor) and w.dim() == 2 and n not in ("in_proj_a", "in_proj_b") and w.shape[1] >= 1024:
            QW[id(w)] = f"L{i}.{n}"
    with torch.no_grad():
        x = R.decoder_layer(x, i, W, B, T)
    QW.clear(); del W
    print(f"[{time.time()-T0:5.0f}s] layer {i}", flush=True)
summ = {}
for f in FMTS:
    v = [s[f + "_sqnr_db"] for s in stats.values()]; o = [s[f + "_out_rel"] for s in stats.values()]
    dn = [s[f + "_sqnr_db"] for k, s in stats.items() if k.endswith("down_proj")]
    summ[f] = {"sqnr_db_median": sorted(v)[len(v) // 2], "sqnr_db_min": min(v), "down_proj_sqnr_db_median": sorted(dn)[len(dn) // 2],
               "gemm_out_rel_median": sorted(o)[len(o) // 2], "gemm_out_rel_max": max(o)}
json.dump({"summary": summ, "per_linear": stats, "meta": vars(a)}, open(a.out, "w"), indent=1)
for f, s in summ.items(): print(f"{f:12s} " + " ".join(f"{k}={v:.4g}" for k, v in s.items()))
