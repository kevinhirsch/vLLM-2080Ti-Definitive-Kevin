#!/usr/bin/env python
"""Lane K9: capture REAL linear-layer inputs of the shipped model (U2 CPU fp32 reference) for the Marlin
accumulate-precision gate. Saves per-linear fp32 inputs (subsampled rows) + the checkpoint prefix of the weight.
Usage: CUDA_VISIBLE_DEVICES= nice python capture_acts.py --layers 6 --windows 2"""
import argparse, os, sys, time, types
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
sys.path.insert(0, "/home/kevin/Desktop/wt-integrate/tools/u2")
import ref_dump as R
ap = argparse.ArgumentParser(); ap.add_argument("--layers", type=int, default=6); ap.add_argument("--windows", type=int, default=2)
ap.add_argument("--rows", type=int, default=640); ap.add_argument("--cap", default=""); ap.add_argument("--out", default="/home/kevin/projects/lanes/k9/acts/acts.pt")
a = ap.parse_args()
ref = torch.load("/home/kevin/projects/lanes/u2-quant/ref.pt"); ids = ref["ids"][: a.windows]
B, T = ids.shape
QW, caps = {}, {}
_lin = torch.nn.functional.linear
def lin(x, w, b=None):
    n = QW.get(id(w))
    y = _lin(x, w, b)
    if n is not None:
        xs = x.reshape(-1, x.shape[-1]); step = max(1, xs.shape[0] // a.rows)
        caps[n] = {"x": xs[::step][: a.rows].clone(), "y_absmax": y.abs().max().item(), "x_absmax": xs.abs().max().item()}
    return y
R.F = types.SimpleNamespace(**{k: getattr(torch.nn.functional, k) for k in dir(torch.nn.functional) if not k.startswith("__")}); R.F.linear = lin
PREF = {"gate_proj": "mlp.gate_proj", "up_proj": "mlp.up_proj", "down_proj": "mlp.down_proj", "q_proj": "self_attn.q_proj",
        "k_proj": "self_attn.k_proj", "v_proj": "self_attn.v_proj", "o_proj": "self_attn.o_proj",
        "in_proj_qkv": "linear_attn.in_proj_qkv", "in_proj_z": "linear_attn.in_proj_z", "out_proj": "linear_attn.out_proj"}
CAPL = {int(v) for v in a.cap.split(',') if v}
x = R.load_embed()[ids.reshape(-1)].float(); T0 = time.time()
resid_max = []
for i in range(a.layers):
    W = R.load_layer_weights(i)
    for n, w in W.items():
        if n in PREF and (not a.cap or i in CAPL): QW[id(w)] = f"{R.PFX}.layers.{i}.{PREF[n]}"
    with torch.no_grad():
        x = R.decoder_layer(x, i, W, B, T)
    resid_max.append(x.abs().max().item())
    QW.clear(); del W
    print(f"[{time.time()-T0:5.0f}s] layer {i} resid absmax {resid_max[-1]:.1f}", flush=True)
torch.save({"caps": caps, "resid_absmax": resid_max, "meta": vars(a)}, a.out)
for n, c in caps.items(): print(n, tuple(c["x"].shape), f"x_absmax={c['x_absmax']:.2f} y_absmax={c['y_absmax']:.2f}")
