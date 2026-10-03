"""Lane K1 microbench: K1 sm_75 flash-prefill vs the production FlashInfer ragged prefill (fa2, hd256, fp16).

Interleaved A/B rounds with CUDA events, median per arm. On a card shared with the live engine the absolute
TFLOPS are contended (the engine takes most SM time); the ratios are the measurement. Memory: V aliases K (the
timing does not depend on the values), one output buffer is shared, K1_CAP_MB bounds this process.

usage: bench_fa75_prefill.py [--quick] [--out file.json] [--variants 0,1,2,3] [--bn 16]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from k1_common import K1, flashinfer_run, gpu_guard, preflight, smi_free


def causal_flops(Tq, Tkv, Hq, D=256, causal=True):
    if not causal:
        return 4.0 * Tq * Tkv * D * Hq
    offs = Tkv - Tq
    vis = Tq * offs + Tq * (Tq + 1) / 2
    return 4.0 * vis * D * Hq


def timeit(fn, reps):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--variants", default="0,1,2,3")
    ap.add_argument("--bn", default="16")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--hq", type=int, default=12)
    ap.add_argument("--hk", type=int, default=2)
    ap.add_argument("--budget-mb", type=int, default=150)
    ap.add_argument("--shapes", default=None, help="Tq:Tkv,... overrides the default grid")
    args = ap.parse_args()
    torch.cuda.set_device(0)
    gpu_guard(0)
    variants = [int(x) for x in args.variants.split(",")]
    bns = [int(x) for x in args.bn.split(",")]
    dev = "cuda"
    scale = 256**-0.5
    if args.shapes:
        grid = [tuple(int(y) for y in x.split(":")) for x in args.shapes.split(",")]
    elif args.quick:
        grid = [(3632, 3632), (3632, 32768), (512, 32768)]
    else:
        grid = []
        for tq in (512, 1024, 2048, 3632, 8192):
            for tkv in (tq, 16384, 32768, 65536, 131072):
                if tkv >= tq and (tkv, tq) not in [(g[1], g[0]) for g in grid]:
                    grid.append((tq, tkv))
    print("smi:", smi_free(), flush=True)
    rows = []
    for Tq, Tkv in grid:
        preflight()  # re-check the shared gate between shapes
        Hq, Hk = args.hq, args.hk
        mem = (2 * Tq * Hq + Tkv * Hk) * 512
        while mem > args.budget_mb * 2**20 and Hk > 1:
            Hq //= 2
            Hk //= 2
            mem = (2 * Tq * Hq + Tkv * Hk) * 512
        if mem > args.budget_mb * 2**20:
            print(json.dumps(dict(Tq=Tq, Tkv=Tkv, skipped="memory budget")), flush=True)
            continue
        g = torch.Generator(device=dev).manual_seed(1)
        q = torch.randn(Tq, Hq, 256, device=dev, generator=g, dtype=torch.half).mul_(2)
        k = torch.randn(Tkv, Hk, 256, device=dev, generator=g, dtype=torch.half)
        v = k  # alias: timing is value independent
        out = torch.empty_like(q)
        flops = causal_flops(Tq, Tkv, Hq)
        w, _ = flashinfer_run(q, k, v, scale, True)
        arms = {"flashinfer": lambda: w.run(q, k, v, out=out)}
        for vv in variants:
            for bn in bns:
                arms[f"k1_v{vv}_bn{bn}"] = (lambda vv=vv, bn=bn: K1.fa75_prefill(q, k, v, scale=scale, causal=True,
                                                                                out=out, variant=vv, bn=bn))
        # warm-up (JIT, plan, clocks) and rep count targeting ~30 ms per arm-round
        for fn in arms.values():
            fn()
        torch.cuda.synchronize()
        t0 = timeit(arms["flashinfer"], 1)
        reps = max(1, min(50, int(30.0 / max(t0, 1e-3))))
        times = {a: [] for a in arms}
        for r in range(args.rounds):
            order = list(arms) if r % 2 == 0 else list(reversed(arms))
            for a in order:
                times[a].append(timeit(arms[a], reps))
        res = dict(Tq=Tq, Tkv=Tkv, Hq=Hq, Hk=Hk, reps=reps, gflop=flops / 1e9)
        fi_med = statistics.median(times["flashinfer"])
        for a, ts in times.items():
            med = statistics.median(ts)
            res[a] = dict(ms=round(med, 4), tflops=round(flops / med / 1e9, 2), spread=round((max(ts) - min(ts)) / med, 3),
                          speedup=round(fi_med / med, 3))
        rows.append(res)
        print(json.dumps(res), flush=True)
        del q, k, v, out, arms
        torch.cuda.empty_cache()
    print("smi:", smi_free(), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(when=time.strftime("%Y-%m-%d %H:%M:%S"), rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
