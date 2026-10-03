"""K1 timing ablations of the v7 kernel (outputs are intentionally wrong): which phase is on the critical path."""
import statistics
import sys

import torch

from k1_common import K1, gpu_guard, smi_free
from bench_fa75_prefill import causal_flops, timeit

ARMS = {"v7": 7, "no_gload": 8, "no_qk": 16, "no_pv": 32, "no_exp": 64, "no_sync": 128, "no_qk_pv": 48,
        "no_gload_qk": 24, "no_gload_pv": 40, "no_gload_sync": 136}


def main():
    Tq, Tkv = (int(x) for x in (sys.argv[1:3] if len(sys.argv) > 2 else (3632, 32768)))
    torch.cuda.set_device(0)
    gpu_guard(0)
    print("smi:", smi_free())
    q = torch.randn(Tq, 12, 256, device="cuda", dtype=torch.half)
    k = torch.randn(Tkv, 2, 256, device="cuda", dtype=torch.half)
    o = torch.empty_like(q)
    fl = causal_flops(Tq, Tkv, 12)
    fns = {a: (lambda v=v: K1.fa75_prefill(q, k, k, scale=0.0625, causal=True, out=o, variant=v)) for a, v in ARMS.items()}
    for f in fns.values():
        f()
    torch.cuda.synchronize()
    t = {a: [] for a in fns}
    for r in range(7):
        for a in (list(fns) if r % 2 == 0 else list(reversed(fns))):
            t[a].append(timeit(fns[a], 5))
    base = statistics.median(t["v7"])
    for a, ts in t.items():
        m = statistics.median(ts)
        print(f"{a:14s} {m:8.3f} ms  {fl / m / 1e9:6.1f} TF-equiv  {m / base:5.2f}x of v7")
    print("smi:", smi_free())


if __name__ == "__main__":
    main()
