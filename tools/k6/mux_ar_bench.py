#!/usr/bin/env python3
"""K6 lead M, 2-GPU part: cost and safety of a SECOND TP communicator for a decode lane that runs on a
high-priority stream concurrently with TP2 prefill (K3 constraint: one all-reduce in flight per communicator).

Launch (both GPUs idle, engine stopped):
  CUDA_DEVICE_ORDER=PCI_BUS_ID python -m torch.distributed.run --nproc-per-node 2 mux_ar_bench.py --out res.json

Per rank: 64 layers of real Marlin W4 GEMMs at TP2 per-rank shapes (~6.3 GiB) + 2 all-reduces per layer.
  prefill lane : M=1856 rows, all-reduce 1856x5120 fp16 (19 MB) on the existing NCCL comm A (eager, lo-pri)
  prefill-tail : M=256 rows, all-reduce via existing CustomAllreduce A (2.6 MB) (eager, lo-pri)
  decode lane  : M rows (CUDA graph, hi-pri) with all-reduce on (a) CustomAllreduce A [today], (b) a second
                 CustomAllreduce B, (c) a second NCCL comm B
Reports: creation cost (GPU MiB, seconds) of B; decode step alone (A vs B: per-op overhead of the 2nd instance);
concurrent decode step p50/p90 + prefill slowdown for {prefill NCCL A || decode CA-B}, {prefill-tail CA-A || decode CA-B},
{prefill NCCL A || decode NCCL-B}; correctness of a constant all-reduce inside every lane (detects barrier corruption);
a 120 s watchdog reports DEADLOCK instead of hanging. The unsafe arrangement (both lanes on one communicator) is NOT run.
"""
import argparse, contextlib, json, os, statistics as st, sys, threading, time
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (init_distributed_environment, ensure_model_parallel_initialized,
                                             get_tp_group)
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mux_bench import build, SHAPES  # noqa: E402
from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new  # noqa: E402

def mib(dev):
    f, t = torch.cuda.mem_get_info(dev); return (t - f) / 2**20

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--decode-m", type=int, default=20); ap.add_argument("--n-decode", type=int, default=40)
    ap.add_argument("--out", default=None); a = ap.parse_args()
    rank = int(os.environ["RANK"]); dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    port = int(os.environ.get("MASTER_PORT", "29511")) + 7
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world_size=2, rank=rank, distributed_init_method=f"tcp://127.0.0.1:{port}",
                                     local_rank=rank)
        ensure_model_parallel_initialized(2, 1)
    tp = get_tp_group(); cpu = tp.cpu_group
    dc = tp.device_communicator; caA = dc.ca_comm; ncA = dc.pynccl_comm
    torch.distributed.all_reduce(torch.zeros(1, device=dev), group=tp.device_group); torch.cuda.synchronize()
    res = dict(rank=rank, caA=caA is not None and not caA.disabled, ncA=ncA is not None)
    deadline = {"t": time.time() + 900}
    def wd():
        while True:
            time.sleep(2)
            if time.time() > deadline["t"]:
                print(json.dumps(dict(rank=rank, DEADLOCK=True, phase=res.get("phase"))), flush=True); os._exit(3)
    threading.Thread(target=wd, daemon=True).start()
    # ---- cost of the second communicators
    torch.cuda.synchronize(); m0 = mib(dev); t0 = time.time()
    caB = CustomAllreduce(group=cpu, device=dev); torch.cuda.synchronize(); m1 = mib(dev); t1 = time.time()
    ncB = PyNcclCommunicator(group=cpu, device=dev)
    x = torch.ones(1024, device=dev, dtype=torch.half); ncB.all_reduce(x); torch.cuda.synchronize()
    m2 = mib(dev); t2 = time.time()
    res["second_custom_ar"] = dict(mib=round(m1 - m0, 1), s=round(t1 - t0, 2), disabled=caB.disabled)
    res["second_nccl_comm"] = dict(mib=round(m2 - m1, 1), s=round(t2 - t1, 2))
    # ---- weights and lanes
    layers, ws, qt = build(a.layers, str(dev))
    lo_p, hi_p = torch.cuda.Stream.priority_range()
    s_lo = torch.cuda.Stream(priority=0); s_hi = torch.cuda.Stream(priority=hi_p)
    res["priority_range"] = [lo_p, hi_p]
    def lane(M, ar):
        acts = {K: torch.randn(M, K, dtype=torch.half, device=dev) * 0.01 for K, _ in SHAPES}
        outs = {N: torch.empty(M, N, dtype=torch.half, device=dev) for _, N in SHAPES}
        wsl = ws if M > 512 else marlin_make_workspace_new(dev, 4)  # each lane owns its Marlin workspace (locks)
        const = torch.full((M, 64), float(rank + 1), dtype=torch.half, device=dev)
        chk = {}
        def run():
            for L in layers:
                for j, (K, N, qw, s) in enumerate(L):
                    ops.marlin_gemm(acts[K], outs[N], qw, None, s, None, None, None, wsl, qt, M, N, K,
                                    use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
                    if N == 5120:  # row-parallel outputs (down, o-proj) are all-reduced
                        ar(outs[N])
            chk["out"] = ar(const)
        return run, chk
    def ar_caA(t): return caA.custom_all_reduce(t)
    def ar_caB(t): return caB.custom_all_reduce(t)
    def ar_ncA(t): o = torch.empty_like(t); ncA.all_reduce(t, o); return o
    def ar_ncB(t): o = torch.empty_like(t); ncB.all_reduce(t, o); return o
    def capture(fn, ca):
        g = torch.cuda.CUDAGraph()
        ctx = ca.capture() if ca is not None else contextlib.nullcontext()
        with ctx:
            with torch.cuda.stream(s_hi):
                fn()  # warm-up (CA returns empty_like while capturing-mode is on but stream not capturing)
            torch.cuda.synchronize(); torch.distributed.barrier(group=cpu)
            with torch.cuda.graph(g, stream=s_hi):
                fn()
        torch.cuda.synchronize(); torch.distributed.barrier(group=cpu)
        return g
    def ok(chk):  # constant all-reduce must be exactly 1+2=3
        return bool(torch.all(chk["out"] == 3).item()) if chk.get("out") is not None else None
    def time_graph(g, n, stream):
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n)]
        with torch.cuda.stream(stream):
            for x, y in ev: x.record(); g.replay(); y.record()
        return ev
    def time_eager(fn, n, stream):
        torch.cuda.synchronize(); torch.distributed.barrier(group=cpu)
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            e0.record()
            for _ in range(n): fn()
            e1.record()
        return e0, e1
    M = a.decode_m
    dec = {}
    for name, ar, ca in (("caA", ar_caA, caA), ("caB", ar_caB, caB), ("ncB", ar_ncB, None)):
        res["phase"] = f"capture decode {name}"; deadline["t"] = time.time() + 120
        fn, chk = lane(M, ar); g = capture(fn, ca)
        ev = time_graph(g, 30, s_hi); torch.cuda.synchronize()
        dec[name] = dict(g=g, chk=chk, alone_ms=st.median([x.elapsed_time(y) for x, y in ev][5:]), alone_ok=ok(chk))
    res["decode_alone_ms"] = {k: round(v["alone_ms"], 2) for k, v in dec.items()}
    res["decode_alone_ok"] = {k: v["alone_ok"] for k, v in dec.items()}
    pf, pchk = lane(1856, ar_ncA); tail, tchk = lane(256, ar_caA)
    res["phase"] = "prefill alone"; deadline["t"] = time.time() + 120
    e0, e1 = time_eager(pf, 2, s_lo); torch.cuda.synchronize(); pf_ms = e0.elapsed_time(e1) / 2
    e0, e1 = time_eager(tail, 3, s_lo); torch.cuda.synchronize(); tail_ms = e0.elapsed_time(e1) / 3
    res["prefill_alone_ms"] = round(pf_ms, 1); res["prefill_tail256_alone_ms"] = round(tail_ms, 1)
    res["prefill_ok"] = ok(pchk); res["tail_ok"] = ok(tchk)
    res["concurrent"] = {}
    for label, pfn, pms, pc, dname in (("prefill_ncA||decode_caB", pf, pf_ms, pchk, "caB"),
                                        ("tail_caA||decode_caB", tail, tail_ms, tchk, "caB"),
                                        ("prefill_ncA||decode_ncB", pf, pf_ms, pchk, "ncB")):
        res["phase"] = label; deadline["t"] = time.time() + 180
        d = dec[dname]; n_pf = max(2, int(a.n_decode * d["alone_ms"] * 2.5 / pms) + 1)
        torch.cuda.synchronize(); torch.distributed.barrier(group=cpu)
        p0 = torch.cuda.Event(enable_timing=True); p1 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(s_lo):
            p0.record()
            for _ in range(n_pf): pfn()
            p1.record()
        time.sleep(0.002)
        ev = time_graph(d["g"], a.n_decode, s_hi)
        torch.cuda.synchronize()
        dts = [x.elapsed_time(y) for x, y in ev]; tot = p0.elapsed_time(p1)
        span = ev[0][0].elapsed_time(ev[-1][1])
        res["concurrent"][label] = dict(decode_ms_p50=round(st.median(dts), 2), decode_ms_p90=round(sorted(dts)[int(.9 * len(dts))], 2),
                                        decode_slowdown=round(st.median(dts) / d["alone_ms"], 2),
                                        prefill_passes=n_pf, prefill_total_ms=round(tot, 1),
                                        prefill_loss_during_overlap=round(max(0.0, tot - n_pf * pms) / min(span, tot), 3),
                                        decode_ok=ok(d["chk"]), prefill_ok=ok(pc))
    res.pop("phase", None)
    print(json.dumps(res), flush=True)
    if a.out and rank == 0: json.dump(res, open(a.out, "w"), indent=1)
    torch.distributed.barrier(group=cpu); os._exit(0)

if __name__ == "__main__":
    main()
