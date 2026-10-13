"""Lane K2 (L83) microbench: MTP verifier prefix attention, today's production path vs IMMA (run INSIDE a window).

Arms, each captured in ONE CUDA graph (production runs FULL cudagraph, so launch overhead is excluded the same way):
  gqa        per request: tq_gqa_decode_attention(q_per_seq=4, 128 splits)  <- production today (raw-current path)
  imma_req   per request: tq_imma, num_seqs=1                                 <- backend per-request IMMA branch
  imma_batch one tq_imma call for all B requests                              <- backend _imma_spec_prefix_batched
Grid: ctx 4K/16K/28K/64K x B 1/4/8, 4 verifier rows (1 + MTP3), per-rank TP=2 shapes, block 3568.
Timing: 3 reps x (20 warm + 100 timed graph replays), CUDA events; reports per-call median of reps and [min,max].
step_saving_ms_est = 16 x (gqa - best imma): Qwen3.8-27B has 16 full-attention layers (64 layers, interval 4; the rest
are GDN), each runs this prefix call once per verifier step on each rank (the MTP head's own attention is not counted).

usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$REL $REL/.venv/bin/python $REL/tools/k2/k2_microbench.py [out.jsonl]
  K2_BENCH_CTX=4096,16384,28000,65536  K2_BENCH_B=1,4,8  K2_BENCH_REPS=3  (env overrides)"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k2_cases import HQ, Case  # noqa: E402

ATTN_LAYERS = int(os.environ.get("K2_ATTN_LAYERS", "16"))


def graph_time(fn, reps, warm=20, iters=100):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    out = []
    for _ in range(reps):
        for _ in range(warm):
            g.replay()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        out.append(e0.elapsed_time(e1) / iters)
    out.sort()
    return {"ms": round(out[len(out) // 2], 4), "min": round(out[0], 4), "max": round(out[-1], 4)}


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else None
    ctxs = [int(x) for x in os.environ.get("K2_BENCH_CTX", "4096,16384,28000,65536").split(",")]
    Bs = [int(x) for x in os.environ.get("K2_BENCH_B", "1,4,8").split(",")]
    reps = int(os.environ.get("K2_BENCH_REPS", "3"))
    fh = open(out_path, "a") if out_path else None
    print(json.dumps({"gpu": torch.cuda.get_device_name(0), "clock_note": "check nvidia-smi clocks; engine must be down",
                      "fusedq": os.environ.get("VLLM_TQ_IMMA_FUSEDQ", "1"), "splits_env": os.environ.get("VLLM_TQ_IMMA_SPLITS")}),
          flush=True)
    for ctx in ctxs:
        for B in Bs:
            c = Case(B, ctx, regime="diffuse", seed=1, jitter=0)
            ql = c.q_len
            outs = [torch.empty(ql, HQ, 256, dtype=torch.float16, device=c.dev) for _ in range(B)]
            lses = [torch.empty(ql, HQ, dtype=torch.float32, device=c.dev) for _ in range(B)]

            def gqa():
                for i in range(B):
                    c.gqa_prefix(i, outs[i], lses[i])

            def imma_req():
                for i in range(B):
                    c.imma_prefix(i, outs[i], lses[i])

            def imma_batch():
                c.imma_batched()

            r = {"ctx": ctx, "B": B, "rows": B * ql, "kres": os.environ.get("VLLM_TQ_IMMA_KRES", "1")}
            for name, fn in (("gqa", gqa), ("imma_req", imma_req), ("imma_batch", imma_batch)):
                r[name] = graph_time(fn, reps)
            if os.environ.get("K2_BENCH_KRES_AB", "1") == "1":
                # speed cost of the K residual plane: same arms captured with VLLM_TQ_IMMA_KRES=0 (args baked at capture)
                prev = os.environ.get("VLLM_TQ_IMMA_KRES", "1")
                os.environ["VLLM_TQ_IMMA_KRES"] = "0"
                r["imma_req_nokres"] = graph_time(imma_req, reps)
                r["imma_batch_nokres"] = graph_time(imma_batch, reps)
                os.environ["VLLM_TQ_IMMA_KRES"] = prev
            g = r["gqa"]["ms"]
            r["x_req"] = round(g / r["imma_req"]["ms"], 2)
            r["x_batch"] = round(g / r["imma_batch"]["ms"], 2)
            best = min(r["imma_req"]["ms"], r["imma_batch"]["ms"])
            r["step_saving_ms_est"] = round(ATTN_LAYERS * (g - best), 3)
            print(json.dumps(r), flush=True)
            if fh:
                fh.write(json.dumps(r) + "\n")
                fh.flush()
            c = outs = lses = None  # free the cache before the next case
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
