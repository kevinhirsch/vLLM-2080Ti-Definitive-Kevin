#!/usr/bin/env python
"""Lane LP tiny GPU correctness smoke (<~300 MB incl. context; safe next to the engine if >=1.2 GiB free):
stock W4A8 (s8 x u4 zp, per-token int8, int16 scales) and LP_A8G (per-(row,128) int8, fp16 scales) vs their CPU emulations,
on REAL checkpoint rows (layer 3 down_proj K=8704 slice, gate_proj K=5120) cut to N=256, M in {1,7,16,65,300}."""
import os as _os, subprocess as _sp, sys as _sys
_g = _sp.run(["/home/kevin/projects/lanes/windows/gpuok.sh", _os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0], "800"], capture_output=True, text=True)
if _g.returncode != 0:
    _sys.exit("gpuok.sh refused: " + _g.stdout.strip() + " " + _g.stderr.strip())

import os, subprocess, sys, json
free = int(subprocess.check_output(["nvidia-smi", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "1"), "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).decode())
if free < 1000: sys.exit(f"refusing: {free} MiB free")
import torch
torch.cuda.set_per_process_memory_fraction(200 / 22528)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/home/kevin/Desktop/wt-integrate/tools/u2")
import lp_g8 as G8
from w4a8_bench_emul import emulate, emulate_g, emulate_e
from safetensors import safe_open
f = safe_open("/home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven/model.safetensors", "pt")
P = "model.language_model.layers.3.mlp."
res = {}
for name, K in (("down_proj", 8704), ("gate_proj", 5120)):
    t = {"weight_packed": f.get_tensor(P + name + ".weight_packed")[:256, : K // 8].contiguous(),
         "weight_scale": f.get_tensor(P + name + ".weight_scale")[:256, : K // 128].half().contiguous(),
         "weight_zero_point": f.get_tensor(P + name + ".weight_zero_point")[:32, : K // 128].contiguous()}
    g8 = G8.G8Linear(t, 256, K)
    e8s = {em: G8.G8Linear(t, 256, K, mode="e", emax=em) for em in (1, 2, 3)}
    layer8, scheme8 = g8.stock
    for M in (1, 7, 16, 65, 300):
        x = torch.randn(M, K) * 0.3; x[:, ::997] *= 30
        xh = x.half()
        yg = g8.forward(xh.cuda(), G8.quant_act_g128_triton).float().cpu()
        yg_t = g8.forward(xh.cuda(), G8.quant_act_g128).float().cpu()
        y8 = scheme8.apply_weights(layer8, xh.cuda(), None).float().cpu()
        e8, e16 = emulate(t, xh, 256, K); eg = emulate_g(t, xh, 256, K)
        r = {"g8_vs_emul": ((yg - eg).norm() / eg.norm()).item(), "g8torchq_vs_emul": ((yg_t - eg).norm() / eg.norm()).item(),
             "stock8_vs_emul": ((y8 - e8).norm() / e8.norm()).item(),
             "g8_vs_fp16act": ((yg - e16).norm() / e16.norm()).item(), "stock8_vs_fp16act": ((y8 - e16).norm() / e16.norm()).item(),
             "finite": bool(torch.isfinite(yg).all())}
        for em, le in e8s.items():
            ye = le.forward(xh.cuda()).float().cpu(); ee = emulate_e(t, xh, 256, K, em, le.level)
            r[f"e{em}_vs_emul"] = ((ye - ee).norm() / ee.norm()).item(); r[f"e{em}_vs_fp16act"] = ((ye - e16).norm() / e16.norm()).item()
        res[f"{name}_M{M}"] = r
        print(name, M, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
json.dump(res, open("/home/kevin/projects/lanes/lp/g8_smoke.json", "w"), indent=1)
ok = all(r["g8_vs_emul"] < 5e-3 and r["stock8_vs_emul"] < 5e-3 and r["finite"] and all(r[f"e{em}_vs_emul"] < 5e-3 for em in (1, 2, 3)) for r in res.values())
print("SMOKE", "PASS" if ok else "FAIL", "peak MiB", torch.cuda.max_memory_allocated() / 2**20)
