"""K1: interleaved timing A/B of two builds of the kernel (.so paths), same process, small shapes (~70 MB).
usage: ab_so.py OLD.so NEW.so [variant] [Tq:Tkv,...]"""
import importlib.util
import statistics
import sys

import torch

from k1_common import gpu_guard, preflight
from bench_fa75_prefill import causal_flops, timeit


def load(path, tag):
    spec = importlib.util.spec_from_file_location("k1fa_sm75", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    variant = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    shapes = [tuple(int(y) for y in x.split(":")) for x in (sys.argv[4] if len(sys.argv) > 4 else "3632:16384,512:16384,3632:3632").split(",")]
    torch.cuda.set_device(0)
    gpu_guard(0)
    A, B = load(a_path, "A"), load(b_path, "B")
    for Tq, Tkv in shapes:
        preflight()
        q = torch.randn(Tq, 12, 256, device="cuda", dtype=torch.half)
        k = torch.randn(Tkv, 2, 256, device="cuda", dtype=torch.half)
        o = torch.empty_like(q)
        cq = torch.tensor([0, Tq], dtype=torch.int32, device="cuda")
        ck = torch.tensor([0, Tkv], dtype=torch.int32, device="cuda")
        run = {n: (lambda m=m: m.fwd(q, k, k, o, None, cq, ck, Tq, 0.0625, True, 16, variant, None, None, 0, 1, 0, None, None))
               for n, m in (("old", A), ("new", B))}
        for f in run.values():
            f()
        torch.cuda.synchronize()
        t = {n: [] for n in run}
        for r in range(9):
            for n in (("old", "new") if r % 2 == 0 else ("new", "old")):
                t[n].append(timeit(run[n], 5))
        fl = causal_flops(Tq, Tkv, 12)
        mo, mn = statistics.median(t["old"]), statistics.median(t["new"])
        print(f"Tq{Tq} Tkv{Tkv} v{variant}: old {mo:.3f} ms {fl/mo/1e9:.1f} TF | new {mn:.3f} ms {fl/mn/1e9:.1f} TF | new/old speed {mo/mn:.3f}", flush=True)
        del q, k, o
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
