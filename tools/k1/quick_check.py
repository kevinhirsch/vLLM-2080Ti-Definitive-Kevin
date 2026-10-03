"""K1 quick numerical check (small shapes, <= ~120 MB): all variants vs fp32, split-KV, segmented pieces."""
import json
import sys

import torch

from k1_common import K1, gpu_guard, ref_attention


def rel(a, b):
    return ((a.float() - b).norm() / b.norm()).item()


def main():
    torch.cuda.set_device(0)
    gpu_guard(0)
    dev, sc = "cuda", 256**-0.5
    worst = 0.0
    for Tq, Tkv, causal, qm in ((300, 1000, True, 1.0), (512, 4096, True, 4.0), (777, 3001, False, 2.0), (64, 9000, True, 8.0)):
        g = torch.Generator(device=dev).manual_seed(Tq + Tkv)
        q = torch.randn(Tq, 6, 256, device=dev, generator=g, dtype=torch.half).mul_(qm)
        k = torch.randn(Tkv, 1, 256, device=dev, generator=g, dtype=torch.half)
        v = torch.randn(Tkv, 1, 256, device=dev, generator=g, dtype=torch.half)
        ro, rl = ref_attention(q, k, v, sc, causal)
        r = {}
        for vv in range(8):
            o, l = K1.fa75_prefill(q, k, v, scale=sc, causal=causal, return_lse=True, variant=vv, nsplit=1)
            r[vv] = (round(rel(o, ro), 6), round((l - rl).abs().max().item(), 6))
            worst = max(worst, r[vv][0])
        for S in (2, 3):
            o = K1.fa75_prefill(q, k, v, scale=sc, causal=causal, nsplit=S)
            r[f"S{S}"] = round(rel(o, ro), 6)
            worst = max(worst, r[f"S{S}"])
        print(json.dumps(dict(Tq=Tq, Tkv=Tkv, causal=causal, r=r)), flush=True)
    ok = worst < 1e-3
    print("QUICK", "PASS" if ok else "FAIL", "worst", worst)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
