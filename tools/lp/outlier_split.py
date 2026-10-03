#!/usr/bin/env python
"""Lane LP: are down_proj activation outliers STATIC channels? If yes, W4A8 down_proj = int8 Marlin on (x minus k outlier channels)
+ a tiny fp16 GEMM over the k channels (LLM.int8-style split; no new GEMM kernel needed).
Calibrate the channel set on windows [0, ncal), evaluate GEMM output rel err on the other windows. Compares to per-token int8 and per-(row,128) int8."""
import json, os, sys, time, types
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
sys.path.insert(0, "/home/kevin/Desktop/wt-integrate/tools/u2"); import ref_dump as R
LAYERS = int(sys.argv[1]) if len(sys.argv) > 1 else 8; NW = 4; NCAL = 2
ref = torch.load("/home/kevin/projects/lanes/u2-quant/ref.pt"); ids = ref["ids"][:NW]; B, T = ids.shape
cap = {}; QW = {}
_lin = torch.nn.functional.linear
def lin(x, w, b=None):
    n = QW.get(id(w))
    if n is not None: cap[n] = (x.reshape(B, T, -1).half().float().clone(), w)
    return _lin(x, w, b)
R.F = types.SimpleNamespace(**{k: getattr(torch.nn.functional, k) for k in dir(torch.nn.functional) if not k.startswith("__")}); R.F.linear = lin
def q8(x):
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 127; return torch.clamp(torch.round(x / s), -127, 127) * s
def q8g(x, g=128):
    xg = x.view(x.shape[0], -1, g); s = xg.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 127
    return (torch.clamp(torch.round(xg / s), -127, 127) * s).view_as(x)
x = R.load_embed()[ids.reshape(-1)].float(); res = {}
for i in range(LAYERS):
    W = R.load_layer_weights(i); QW[id(W["down_proj"])] = f"L{i}.down_proj"; QW[id(W.get("out_proj", W.get("o_proj")))] = f"L{i}.out"
    with torch.no_grad(): x = R.decoder_layer(x, i, W, B, T)
    for n, (a, w) in cap.items():
        cal = a[:NCAL, 16:].reshape(-1, a.shape[-1]); ev = a[NCAL:, 16:].reshape(-1, a.shape[-1])
        y0 = ev @ w.T; rel = lambda xq: ((xq @ w.T - y0).norm() / y0.norm()).item()
        r = {"tok": rel(q8(ev)), "g128": rel(q8g(ev))}
        score = cal.abs().amax(0)  # calibration: per-channel absmax
        # how much of the eval tokens' absmax lands on the calibrated top-k channels
        for k in (4, 8, 16, 32, 64):
            idx = score.topk(k).indices; m = torch.ones(ev.shape[-1], dtype=torch.bool); m[idx] = False
            xin = ev * m; xq = q8(xin) + ev * (~m)
            r[f"ol{k}"] = rel(xq)
        top1 = ev.abs().argmax(-1); r["eval_absmax_in_cal_top16"] = float(torch.isin(top1, score.topk(16).indices).float().mean())
        res[n] = r
        print(n, {k: round(v, 4) for k, v in r.items()}, flush=True)
    cap.clear(); QW.clear(); del W
json.dump(res, open("/home/kevin/projects/lanes/lp/outlier_split.json", "w"), indent=1)
