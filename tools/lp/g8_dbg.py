
import os as _os, subprocess as _sp, sys as _sys
_g = _sp.run(["/home/kevin/projects/lanes/windows/gpuok.sh", _os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0], "800"], capture_output=True, text=True)
if _g.returncode != 0:
    _sys.exit("gpuok.sh refused: " + _g.stdout.strip() + " " + _g.stderr.strip())
import os, sys, torch
torch.cuda.set_per_process_memory_fraction(200 / 22528)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/home/kevin/Desktop/wt-integrate/tools/u2")
import lp_g8 as G8
from w4a8_bench_emul import emulate_g
from safetensors import safe_open
f = safe_open("/home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven/model.safetensors", "pt")
P = "model.language_model.layers.3.mlp."; name, K = "gate_proj", 5120; N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
t = {"weight_packed": f.get_tensor(P + name + ".weight_packed")[:N, : K // 8].contiguous(),
     "weight_scale": f.get_tensor(P + name + ".weight_scale")[:N, : K // 128].half().contiguous(),
     "weight_zero_point": f.get_tensor(P + name + ".weight_zero_point")[:N // 8, : K // 128].contiguous()}
g8 = G8.G8Linear(t, N, K)
for M in (64, 65, 80, 128, 129, 192, 256, 300):
    x = (torch.randn(M, K) * 0.3).half()
    yg = g8.forward(x.cuda(), G8.quant_act_g128).float().cpu(); eg = emulate_g(t, x, N, K)
    re = ((yg - eg).norm(dim=1) / eg.norm(dim=1))
    bad = (re > 1e-2).nonzero().flatten().tolist()
    # does a bad row match emulation with a DIFFERENT row's act scales / group shift?
    print(M, "bad rows", len(bad), bad[:6], "...", bad[-3:] if bad else "", flush=True)
