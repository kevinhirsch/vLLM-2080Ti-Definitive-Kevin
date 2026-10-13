"""Lane E04 microbench (run INSIDE a window, engine down): tq_gqa stage1+stage2, shipped layout vs C3 (3 CTAs/SM).

Arms, each captured in ONE CUDA graph (production replays FULL graphs, so launch overhead is excluded the same way):
  base   decode     (28192 B smem, 2 CTAs/SM)
  c3     decode_c3  (21600 B smem, 3 CTAs/SM)
Shapes: MTP verifier prefix (S=1, QL=4: one call per request per attention layer) and plain decode (S=B, QL=1),
production block_size 3568, 128 splits, per-rank TP=2 heads.  Arms are interleaved per rep (base, c3, base, c3, ...)
so clock drift hits both.  3 reps x (20 warm + 100 timed replays); per-call median of reps and [min, max].
step_saving_ms_est = 16 x (base - c3) for the verifier shape (16 full-attention layers in Qwen3.8-27B).

usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$REL $REL/.venv/bin/python $REL/tools/e04/e04_microbench.py [out.jsonl]
  E04_BENCH_CTX=4096,16384,28000,65536,131072  E04_BENCH_REPS=3  (env overrides)
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e04_cases import Case  # noqa: E402

ATTN_LAYERS = int(os.environ.get("E04_ATTN_LAYERS", "16"))


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def time_graph(g, warm=20, iters=100):
    for _ in range(warm):
        g.replay()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def stat(xs):
    xs = sorted(xs)
    return {"ms": round(xs[len(xs) // 2], 4), "min": round(xs[0], 4), "max": round(xs[-1], 4)}


def main():
    from vllm.v1.attention.ops import tq_gqa_cuda as C

    ext = C._load()
    out_path = sys.argv[1] if len(sys.argv) > 1 else None
    ctxs = [int(x) for x in os.environ.get("E04_BENCH_CTX", "4096,16384,28000,65536,131072").split(",")]
    reps = int(os.environ.get("E04_BENCH_REPS", "3"))
    fh = open(out_path, "a") if out_path else None

    def emit(rec):
        line = json.dumps(rec)
        print(line, flush=True)
        if fh:
            fh.write(line + "\n")
            fh.flush()

    emit({"gpu": torch.cuda.get_device_name(0), "note": "engine must be down; check clocks", "reps": reps})
    shapes = [("verifier", 1, 4), ("decode_b4", 4, 1), ("verifier_b4_loop", 4, 4)]
    for ctx in ctxs:
        for name, S, QL in shapes:
            if name == "verifier_b4_loop":
                # production per-request continuation: B=4 requests -> 4 separate S=1,QL=4 calls
                cases = [Case(ctx, 1, 4, seed=7 + i, ragged=False) for i in range(4)]
            else:
                cases = [Case(ctx, S, QL, seed=7, ragged=False)]
            bufs = {}
            graphs = {}
            for arm in ("base", "c3"):
                bufs[arm] = [c.run(ext, arm) for c in cases]

                def fn(arm=arm):
                    for c, (o, l, m) in zip(cases, bufs[arm]):
                        c.run(ext, arm, mid=m, out=o, lse=l)

                graphs[arm] = capture(fn)
            t = {"base": [], "c3": []}
            for _ in range(reps):
                for arm in ("base", "c3"):
                    t[arm].append(time_graph(graphs[arm]))
            sb, sc = stat(t["base"]), stat(t["c3"])
            same = all(torch.equal(a[0].view(torch.int16), b[0].view(torch.int16))
                       for a, b in zip(bufs["base"], bufs["c3"]))
            rec = {"ctx": ctx, "shape": name, "S": S, "QL": QL, "base": sb, "c3": sc,
                   "speedup": round(sb["ms"] / sc["ms"], 3), "bit_identical": same,
                   "ranges_overlap": not (sc["max"] < sb["min"] or sb["max"] < sc["min"])}
            if name == "verifier":
                rec["step_saving_ms_est"] = round(ATTN_LAYERS * (sb["ms"] - sc["ms"]), 3)
            emit(rec)
            del graphs, bufs, cases
            torch.cuda.empty_cache()
    if fh:
        fh.close()


if __name__ == "__main__":
    main()
