"""Lane LP contended mini speed probe (<100 MB tensors): real gate_proj/down_proj rows, N=2048 slice, M=16..3632,
W4A16 Marlin vs stock W4A8 vs LP_A8G (incl. Triton quant), interleaved, min of reps. GPU shared with production -> ratios only."""
import os as _os, subprocess as _sp, sys as _sys
_g = _sp.run(["/home/kevin/projects/lanes/windows/gpuok.sh", _os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0], "800"], capture_output=True, text=True)
if _g.returncode != 0:
    _sys.exit("gpuok.sh refused: " + _g.stdout.strip() + " " + _g.stderr.strip())

import os, sys, subprocess, statistics, torch
free = int(subprocess.check_output(["nvidia-smi", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0"), "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).decode())
if free < 1000: sys.exit(f"refusing: {free} MiB free")
torch.cuda.set_per_process_memory_fraction(300 / 22528)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/home/kevin/Desktop/wt-integrate/tools/u2")
import lp_g8 as G8
from _ctlayer import make_marlin_layer
from safetensors import safe_open
f = safe_open("/home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven/model.safetensors", "pt")
P = "model.language_model.layers.3.mlp."; N = 2048
def ev(fn, reps=25):
    for _ in range(3): fn()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record(); fn(); e.record(); e.synchronize(); ts.append(s.elapsed_time(e))
    return min(ts), statistics.median(ts)
for name, K in (("gate_proj", 5120), ("down_proj", 8704)):
    t = {"weight_packed": f.get_tensor(P + name + ".weight_packed")[:N, : K // 8].contiguous(),
         "weight_scale": f.get_tensor(P + name + ".weight_scale")[:N, : K // 128].half().contiguous(),
         "weight_zero_point": f.get_tensor(P + name + ".weight_zero_point")[:N // 8, : K // 128].contiguous()}
    l16, s16 = make_marlin_layer(dict(t), N, K, device="cuda"); g8 = G8.G8Linear(t, N, K); l8, s8 = g8.stock
    e2 = G8.G8Linear(t, N, K, mode="e", emax=3)
    for M in (16, 64, 256, 1024, 2048, 3632):
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        arms = {"w4a16": lambda: s16.apply_weights(l16, x, None), "w4a8": lambda: s8.apply_weights(l8, x, None),
                "w4a8g": lambda: g8.forward(x, G8.quant_act_g128_triton), "q_g": lambda: G8.quant_act_g128_triton(x), "e2": lambda: e2.forward(x),
                "q_e": lambda: G8.L.quant_e(x, e2.wglob, 3)}
        r = {k: [] for k in arms}
        for rep in range(4):
            for k, fn in arms.items(): r[k].append(ev(fn)[0])
        m = {k: min(v) for k, v in r.items()}
        fl = 2 * M * N * K
        print(f"{name} K={K} M={M:5d} w4a16 {m['w4a16']:.3f} ms ({fl/m['w4a16']/1e9:5.1f} TF) | w4a8 {m['w4a8']:.3f} x{m['w4a16']/m['w4a8']:.2f} | "
              f"w4a8g {m['w4a8g']:.3f} x{m['w4a16']/m['w4a8g']:.2f} (quant {m['q_g']:.3f}) | e2 {m['e2']:.3f} x{m['w4a16']/m['e2']:.2f} (quant {m['q_e']:.3f})", flush=True)
        del x
