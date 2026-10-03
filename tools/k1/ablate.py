"""K1 timing ablations of the v7 kernel (outputs are intentionally wrong): which phase is on the critical path."""
import statistics
import sys

import torch

from k1_common import K1, gpu_guard, preflight, smi_free
from bench_fa75_prefill import causal_flops, timeit

ARMS = {"v7": 7, "v3": 3, "qk16_c64": 8, "no_gload": 64, "no_qk": 128, "no_pv": 256, "no_exp": 512, "no_sync": 1024,
        "no_qk_pv": 384, "no_gload_qk": 192, "no_gload_pv": 320, "no_gload_sync": 1088}


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
        preflight()
        for a in (list(fns) if r % 2 == 0 else list(reversed(fns))):
            t[a].append(timeit(fns[a], 5))
    base = statistics.median(t["v7"])
    for a, ts in t.items():
        m = statistics.median(ts)
        print(f"{a:14s} {m:8.3f} ms  {fl / m / 1e9:6.1f} TF-equiv  {m / base:5.2f}x of v7")
    print("smi:", smi_free())




def gqa_proxy(Tq=3632, Tkv=16384):
    """Is K/V traffic the limit? Same Hq=12 and FLOPs, K/V shared by 6 q heads (Hk=2) vs not shared (Hk=12: 6x the
    distinct K/V bytes). If the kernel were K/V-traffic bound, Hk=12 would be far slower."""
    torch.cuda.set_device(0)
    gpu_guard(0)
    q = torch.randn(Tq, 12, 256, device="cuda", dtype=torch.half)
    o = torch.empty_like(q)
    fl = causal_flops(Tq, Tkv, 12)
    res = {}
    for hk in (2, 12):
        k = torch.randn(Tkv, hk, 256, device="cuda", dtype=torch.half)
        f = lambda: K1.fa75_prefill(q, k, k, scale=0.0625, causal=True, out=o, variant=7)
        f(); torch.cuda.synchronize()
        res[hk] = statistics.median(timeit(f, 5) for _ in range(7))
        del k
        torch.cuda.empty_cache()
    for hk, m in res.items():
        print(f"Hk={hk:2d}: {m:8.3f} ms {fl / m / 1e9:6.1f} TF")


if __name__ == "__main__":
    if "--gqa-proxy" in sys.argv:
        gqa_proxy()
    else:
        main()
