#!/usr/bin/env python3
"""Lane K3 (L78) TP=2 all-reduce microbench + GEMM/AR overlap emulation.

Launch (spawns its own 2 workers, one per GPU, same wiring as production:
vllm tp group -> CustomAllreduce (IPC, NVLink) + PyNccl):

  cd /home/kevin/Desktop/wt-k3
  PYTHONPATH=/home/kevin/Desktop/wt-k3 /home/kevin/Desktop/wt-integrate/.venv/bin/python \
      tools/k3/bench_allreduce.py [--mode ar|overlap|check|all] [--tokens 16,256,...]

Needs both GPUs and, for tokens=3632, ~250 MiB (ar) / ~600 MiB (overlap) of
VRAM per GPU on top of the CUDA context. Do NOT run next to the live engine
unless the engine is stopped (production uses ~21.4 of 22.5 GiB).

Modes
  check    tiny correctness only (<= 64 KiB messages, ~350 MiB incl. contexts)
  ar       NCCL vs custom-1stage vs custom-2stage, eager/unregistered
           (the path chunked prefill takes), per message size
  overlap  RowParallel emulation: GEMM [n,8704]x[8704,5120] fp16 + AR,
           serial vs K-chunk overlapped on a priority side stream
"""
import argparse
import json
import os
import subprocess
import sys
import time

HIDDEN = 5120
K_PER_RANK = 17408 // 2  # MLP down_proj contraction per rank


def worker(rank, args):
    import torch

    os.environ["VLLM_CUSTOM_ALLREDUCE_MAX_SIZE_MB"] = str(args.cap_mb)
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state as ps

    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    with set_current_vllm_config(VllmConfig()):
        ps.init_distributed_environment(
            world_size=2,
            rank=rank,
            distributed_init_method=f"file://{args.rdv}",
            local_rank=rank,
        )
        ps.initialize_model_parallel(2, 1)
    grp = ps.get_tp_group()
    dc = grp.device_communicator
    ca, nccl = dc.ca_comm, dc.pynccl_comm
    assert ca is not None and not ca.disabled, "custom allreduce disabled"
    out_rows = []

    def sync_all():
        torch.cuda.synchronize()
        torch.distributed.barrier(group=grp.cpu_group)

    def timeit(fn, reps, inner):
        for _ in range(3):
            fn()
        sync_all()
        ts = []
        for _ in range(reps):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            sync_all()
            s.record()
            for _ in range(inner):
                fn()
            e.record()
            e.synchronize()
            ts.append(s.elapsed_time(e) / inner)
        ts.sort()
        return ts[len(ts) // 2], ts[0]

    tokens = [int(t) for t in args.tokens.split(",")]

    if args.mode in ("check", "ar", "all"):
        for n in tokens:
            x = (torch.randn(n, HIDDEN, device=dev, dtype=torch.float16) * 4).contiguous()
            nbytes = x.numel() * 2
            ref = nccl.all_reduce(x.clone())
            res = {"n": n, "MB": round(nbytes / 2**20, 3)}
            variants = {}
            if nbytes < args.cap_mb * 2**20:
                for algo in ("1stage", "2stage"):
                    os.environ["VLLM_CUSTOM_ALLREDUCE_ALGO"] = algo
                    o = ca.all_reduce(x, registered=False)
                    torch.cuda.synchronize()
                    mism = int((o != ref).sum().item())
                    md = float((o.float() - ref.float()).abs().max().item())
                    res[f"{algo}_mismatch"] = mism
                    res[f"{algo}_maxdiff"] = md
                    variants[algo] = lambda a=algo: (
                        os.environ.__setitem__("VLLM_CUSTOM_ALLREDUCE_ALGO", a),
                        ca.all_reduce(x, registered=False),
                    )
                os.environ.pop("VLLM_CUSTOM_ALLREDUCE_ALGO", None)
            variants["nccl"] = lambda: nccl.all_reduce(x)
            if args.mode != "check":
                inner = 20 if nbytes < 8 * 2**20 else 8
                for name, fn in variants.items():
                    med, mn = timeit(fn, args.reps, inner)
                    res[f"{name}_ms"] = round(med, 4)
                    res[f"{name}_min_ms"] = round(mn, 4)
                    res[f"{name}_GBps"] = round(nbytes / med / 1e6, 1)
            out_rows.append(res)

    if args.mode in ("overlap", "all"):
        import vllm.distributed.k3_ar_overlap as ov

        W = (torch.randn(K_PER_RANK, HIDDEN, device=dev, dtype=torch.float16) * 0.02)
        for n in tokens:
            if n < 256:
                continue
            x = torch.randn(n, K_PER_RANK, device=dev, dtype=torch.float16)
            comm = ov._comm_stream(dev)
            main = torch.cuda.current_stream()

            def serial():
                y = x @ W
                return grp.all_reduce(y)

            def overlapped(k):
                out = torch.empty(n, HIDDEN, device=dev, dtype=torch.float16)
                step = ((n + k - 1) // k + 7) // 8 * 8
                a = 0
                while a < n:
                    b = min(n, a + step)
                    part = x[a:b] @ W
                    ev = torch.cuda.Event()
                    ev.record(main)
                    comm.wait_event(ev)
                    with torch.cuda.stream(comm):
                        part.record_stream(comm)
                        ov.all_reduce_into(part, out[a:b], comm)
                    a = b
                main.wait_stream(comm)
                return out

            ref = serial()
            res = {"n": n, "mode": "overlap"}
            res["gemm_only_ms"] = round(timeit(lambda: x @ W, args.reps, 6)[0], 4)
            res["serial_ms"] = round(timeit(serial, args.reps, 6)[0], 4)
            for k in (2, 3, 4):
                o = overlapped(k)
                torch.cuda.synchronize()
                res[f"k{k}_maxdiff"] = float((o.float() - ref.float()).abs().max().item())
                res[f"k{k}_ms"] = round(timeit(lambda k=k: overlapped(k), args.reps, 6)[0], 4)
            out_rows.append(res)

    # gather rows from both ranks (rank 1 skew is informative)
    with open(f"{args.out}.rank{rank}.json", "w") as f:
        json.dump(out_rows, f, indent=1)
    sync_all()
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="check")
    ap.add_argument("--tokens", default="1,8,64")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--cap-mb", type=int, default=48)
    ap.add_argument("--out", default="/tmp/k3_bench")
    ap.add_argument("--rdv", default=f"/tmp/k3_rdv_{os.getpid()}")
    ap.add_argument("--worker", type=int, default=-1)
    args = ap.parse_args()
    if args.worker >= 0:
        worker(args.worker, args)
        return
    ps = [
        subprocess.Popen([sys.executable, __file__, "--worker", str(r)] + sys.argv[1:])
        for r in (0, 1)
    ]
    rc = [p.wait() for p in ps]
    for r in (0, 1):
        try:
            rows = json.load(open(f"{args.out}.rank{r}.json"))
        except Exception as e:  # noqa
            print("rank", r, "no result", e)
            continue
        print(f"--- rank {r}")
        for row in rows:
            print(json.dumps(row))
    sys.exit(max(rc))


if __name__ == "__main__":
    main()
