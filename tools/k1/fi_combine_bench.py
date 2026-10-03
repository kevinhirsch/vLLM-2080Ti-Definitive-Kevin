"""K1: is TurboQuant's FlashInfer prefix-combine (non-causal prefix call + causal chunk call + merge) faster than the
single causal call it replaces? Same FLOPs; per-rank shapes (Hq 12, Hk 2), V aliases K."""
import statistics

import torch
from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

from k1_common import K1, gpu_guard, preflight, smi_free
from bench_fa75_prefill import causal_flops, timeit
from vllm_merge import merge


def main():
    torch.cuda.set_device(0)
    gpu_guard(0)
    print("smi:", smi_free())
    ws = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
    w1 = BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="fa2")
    wp = BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="fa2")
    wc = BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="fa2")
    Tq, Hq, Hk = 3632, 12, 2
    for Tkv in (24576, 65536, 131072):
        preflight()
        cached = Tkv - Tq
        q = torch.randn(Tq, Hq, 256, device="cuda", dtype=torch.half)
        k = torch.randn(Tkv, Hk, 256, device="cuda", dtype=torch.half)
        o = torch.empty_like(q)
        po, so = torch.empty_like(q), torch.empty_like(q)
        pl = torch.empty(Tq, Hq, device="cuda")
        sl = torch.empty(Tq, Hq, device="cuda")
        it = lambda *a: torch.tensor(list(a), dtype=torch.int32)
        kw = dict(q_data_type=torch.float16, kv_data_type=torch.float16, sm_scale=0.0625)
        w1.plan(it(0, Tq), it(0, Tkv), Hq, Hk, 256, causal=True, **kw)
        wp.plan(it(0, Tq), it(0, cached), Hq, Hk, 256, causal=False, **kw)
        wc.plan(it(0, Tq), it(0, Tq), Hq, Hk, 256, causal=True, **kw)

        def single():
            w1.run(q, k, k, out=o)

        def combine():
            wp.run(q, k[:cached], k[:cached], out=po, lse=pl, return_lse=True)
            wc.run(q, k[cached:], k[cached:], out=so, lse=sl, return_lse=True)
            merge(o, po, pl, so, sl)

        def k1():
            K1.fa75_prefill(q, k, k, scale=0.0625, causal=True, out=o)

        arms = dict(fi_single=single, fi_combine=combine, k1_single=k1)
        for f in arms.values():
            f()
        torch.cuda.synchronize()
        t = {a: [] for a in arms}
        for r in range(5):
            for a in arms:
                t[a].append(timeit(arms[a], 3))
        fl = causal_flops(Tq, Tkv, Hq)
        print(f"Tkv {Tkv}: " + " | ".join(f"{a} {statistics.median(v):.2f} ms {fl / statistics.median(v) / 1e9:.1f} TF"
                                         for a, v in t.items()), flush=True)
        del q, k, o, po, so, pl, sl
        torch.cuda.empty_cache()
    print("smi:", smi_free())


if __name__ == "__main__":
    main()
