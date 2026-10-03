#!/usr/bin/env python
"""Lane K7: GPTQ re-quantization of the abliterated HauhauCS W4A16 model into ROTATED symmetric per-channel int4 weights
for the Turing W4A4 kernels (k7 w4a4_gemm per-token and w4a4g group-scaled both take per-output-channel weight scales).

Source = shipped int4 dequant (no bf16 abliterated weights exist locally); original files are only read.
Per Marlin linear: W_rot = W @ blockdiag(H128)^T; Hessian H = X_rot^T X_rot from the BASELINE (W4A16, fp32) forward of the
calibration windows (ref.pt windows 6..11 by default, disjoint from the gate's eval windows 0..5); GPTQ (act-order,
1% damp, symmetric per-channel scales from an MSE-clip search) -> codes int4 + fp32 scale.
Output: <out>/layer_XX.safetensors with "<name>.codes" (int8 packed 2/byte, cutlass int4b_t order) and "<name>.scale",
plus meta.json.  Resumable (skips layers whose file exists; carries the calib residual stream in <out>/stream.pt).
Usage: CUDA_VISIBLE_DEVICES= REF_THREADS=6 python tools/k7/gptq_requant.py [--device cpu|cuda] [--calib 6:12]"""
import argparse, json, os, sys, time, types
import torch
sys.path.insert(0, "/home/kevin/Desktop/wt-integrate/tools/u2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_dump as R
import rotquant as RQ
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--ref", default="/home/kevin/projects/lanes/u2-quant/ref.pt")
ap.add_argument("--out", default="/home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A4rot128-gptq-k7")
ap.add_argument("--calib", default="6:12")
ap.add_argument("--hb", type=int, default=128)
ap.add_argument("--layers", type=int, default=R.N_LAYERS)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
T0 = time.time(); log = lambda *s: print(f"[{time.time()-T0:6.0f}s]", *s, flush=True)
dev = torch.device(a.device)
ref = torch.load(a.ref)
c0, c1 = map(int, a.calib.split(":"))
ids = ref["ids"][c0:c1]
B, T = ids.shape
json.dump({"source": R.MODEL_DIR, "calib_windows": [c0, c1], "calib_kinds": [m["kind"] for m in ref["manifest"][c0:c1]],
           "hb": a.hb, "scheme": "rotated blockdiag-Hadamard(hb) on input dim; symmetric int4 per-output-channel; GPTQ act-order damp 0.01",
           "pack": "int8 [N,K/2], low nibble = even k (cutlass int4b_t)", "by": "Lane K7 2026-10-03"},
          open(os.path.join(a.out, "meta.json"), "w"), indent=1)

CAP = {}
_lin = torch.nn.functional.linear
NAMES = {}


def lin(x, w, b=None):
    n = NAMES.get(id(w))
    if n is not None:
        xs = RQ.block_had(x.reshape(-1, x.shape[-1]).float(), a.hb)
        H = xs.T @ xs
        CAP[n] = CAP[n] + H if n in CAP else H
    return _lin(x, w, b)


R.F = types.SimpleNamespace(**{k: getattr(torch.nn.functional, k) for k in dir(torch.nn.functional) if not k.startswith("__")})
R.F.linear = lin

sp = os.path.join(a.out, "stream.pt")
start = 0
if os.path.exists(sp):
    st = torch.load(sp); x, start = st["x"], st["next_layer"]
    log(f"resume at layer {start}")
else:
    x = R.load_embed()[ids.reshape(-1)].float()
for i in range(start, a.layers):
    tl = time.time()
    W = R.load_layer_weights(i)
    lnames = ["gate_proj", "up_proj", "down_proj"] + (["q_proj", "k_proj", "v_proj", "o_proj"] if R.LAYER_TYPES[i] == "full_attention"
                                                    else ["in_proj_qkv", "in_proj_z", "out_proj"])
    NAMES.clear(); CAP.clear()
    for n in lnames:
        NAMES[id(W[n])] = n
    with torch.no_grad():
        xn = R.decoder_layer(x, i, W, B, T)
    tf = time.time() - tl
    out, stats = {}, {}
    for n in lnames:
        w = W[n]
        wr = RQ.rotate_weight(w, a.hb)
        H = CAP[n].to(dev)
        c, s = RQ.gptq_sym(wr.to(dev), H, 4, 0)
        c, s = c.cpu(), s.cpu()
        Xr_eval = None
        d = RQ.dequant(c, s, 0)
        cr, sr = RQ.sym_quant(wr, 4, 0)
        dr = RQ.dequant(cr, sr, 0)
        # output-error proxy on the calib Hessian: tr(E H E^T) relative to tr(W H W^T)
        Hc = CAP[n]
        def oerr(dw):
            E = dw - wr
            return ((E @ Hc) * E).sum().item() / ((wr @ Hc) * wr).sum().item()
        stats[n] = {"out_sqnr_gptq_db": -10 * torch.log10(torch.tensor(oerr(d))).item(),
                    "out_sqnr_rtn_db": -10 * torch.log10(torch.tensor(oerr(dr))).item()}
        out[f"{n}.codes"] = RQ.pack_s4(c) if hasattr(RQ, "pack_s4") else None
        out[f"{n}.scale"] = s.float().contiguous()
    save_file(out, os.path.join(a.out, f"layer_{i:02d}.safetensors"))
    x = xn
    torch.save({"x": x, "next_layer": i + 1}, sp)
    log(f"layer {i:2d} fwd {tf:5.1f}s total {time.time()-tl:6.1f}s  " +
        " ".join(f"{n}:{v['out_sqnr_rtn_db']:.1f}->{v['out_sqnr_gptq_db']:.1f}dB" for n, v in stats.items()))
    with open(os.path.join(a.out, "stats.jsonl"), "a") as fh:
        fh.write(json.dumps({"layer": i, **stats}) + "\n")
log("done")
