"""Lane K2 (L83) GPU numerics gate for the IMMA TurboQuant verifier-prefix kernel (run INSIDE a GPU window).

Checks, on a real k3v4_nc cache written by the production store kernel (per-rank TP=2 shapes, block 3568, 4 rows):
  1. prefix out/lse of tq_imma (per-request and batched) vs the exact fp32 reference over the dequantized codes,
     next to tq_gqa (today's production path) on the same data;
  2. the full verifier output (prefix merged with the raw current chunk by the production merge kernel) for
     imma vs gqa vs reference -> proves the LSE convention matches the merge;
  3. empty prefix rows (cached_len 0, zero page) -> out 0, lse -inf, no NaN;
  4. CUDA-graph capture of the batched call, replayed after the row lengths change, equals eager;
  5. VLLM_TQ_IMMA_FUSEDQ=0 (cuBLAS rotation + quant kernel) equals the fused in-kernel rotation within int8 noise.
Prints one JSON line per check and a final {"verdict": PASS|FAIL}; exit code 0 only on PASS.

usage (window, one GPU):  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$REL $REL/.venv/bin/python $REL/tools/k2/k2_numerics.py
  env: VLLM_TQ_IMMA_BUILD_DIR / VLLM_TQ_GQA_BUILD_DIR -> the release's prebuilt .deps dirs (else JIT builds)."""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k2_cases import HQ, SCALE, Case  # noqa: E402

from vllm.v1.attention.ops.triton_turboquant_decode import triton_turboquant_spec_continuation_merge  # noqa: E402

# gates: K2's 2026-10-03 microbench measured imma rel mean 2-9e-4, max <= 6e-3, |dLSE| <= 2.3e-3 (int8 LUT ratio
# error 0.26% is the floor). Gate with margin, and never more than 3x worse than tq_gqa on the same data.
REL_MEAN, REL_MAX, LSE_MAX = 2e-3, 2e-2, 2e-2
GQA_FACTOR, ABS_MAX_FLOOR = 3.0, 5e-3   # rel_max must also be <= max(GQA_FACTOR * tq_gqa rel_max, ABS_MAX_FLOOR)
results, ok = [], True


def err(o, ref, lse=None, rlse=None):
    o = o.float()
    e = (o - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-12)
    d = {"rel_mean": float(f"{e.mean().item():.3g}"), "rel_max": float(f"{e.max().item():.3g}"),
         "nan": bool(torch.isnan(o).any().item())}
    if lse is not None:
        d["lse_max"] = float(f"{(lse - rlse).abs().max().item():.3g}")
    return d


def gate(name, d, ref_d=None):
    global ok
    good = (not d["nan"] and d["rel_mean"] <= REL_MEAN and d["rel_max"] <= REL_MAX
            and d.get("lse_max", 0.0) <= LSE_MAX)
    if ref_d is not None:
        good = good and d["rel_max"] <= max(GQA_FACTOR * ref_d["rel_max"], ABS_MAX_FLOOR)
    ok = ok and good
    return good


def emit(**kw):
    results.append(kw)
    print(json.dumps(kw), flush=True)


def check_case(B, ctx, regime):
    c = Case(B, ctx, regime=regime, seed=B + ctx)
    ref_o, ref_l = c.reference(with_current=False)
    gq = [c.gqa_prefix(i) for i in range(B)]
    go, gl = torch.cat([x[0] for x in gq]), torch.cat([x[1] for x in gq])
    im = [c.imma_prefix(i) for i in range(B)]
    io, il = torch.cat([x[0] for x in im]), torch.cat([x[1] for x in im])
    bo, bl = c.imma_batched()
    torch.cuda.synchronize()
    dg, di, db = err(go, ref_o, gl, ref_l), err(io, ref_o, il, ref_l), err(bo, ref_o, bl, ref_l)
    emit(check="prefix", B=B, ctx=ctx, regime=regime, gqa=dg, imma_req=di, imma_batch=db,
         ok=gate("imma_req", di, dg) & gate("imma_batch", db, dg),
         batch_vs_req_maxabs=float(f"{(bo.float() - io.float()).abs().max().item():.3g}"))
    # full verifier output through the production merge kernel
    ref_full, _ = c.reference(with_current=True)
    ql = c.q_len
    fulls = {}
    for name, (o, l) in (("gqa", (go, gl)), ("imma", (bo, bl))):
        parts = []
        for i in range(B):
            sl = slice(i * ql, (i + 1) * ql)
            parts.append(triton_turboquant_spec_continuation_merge(
                c.query[sl], c.key_cur[sl], c.val_cur[sl], o[sl].clone(), l[sl].clone(), SCALE))
        fulls[name] = err(torch.cat(parts), ref_full)
    emit(check="verifier_full", B=B, ctx=ctx, regime=regime, **fulls, ok=gate("full", fulls["imma"], fulls["gqa"]))
    return c


def check_empty_and_graph(c: Case):
    # empty prefix rows: request 0 gets length 0 + zero page (what the backend does at capture / fresh requests)
    ql = c.q_len
    keep_bt, keep_pl = c.block_table.clone(), c.prefix_lens.clone()
    c.block_table[0].zero_()
    c.prefix_lens[0] = 0
    o, l = c.imma_batched()
    torch.cuda.synchronize()
    e0 = {"out0_maxabs": float(o[:ql].float().abs().max().item()), "lse0_all_neg_inf": bool(torch.isneginf(l[:ql]).all().item()),
          "nan": bool(torch.isnan(o).any().item())}
    good = e0["out0_maxabs"] == 0.0 and e0["lse0_all_neg_inf"] and not e0["nan"]
    global ok
    ok = ok and good
    emit(check="empty_prefix_rows", **e0, ok=good)
    c.block_table.copy_(keep_bt)
    c.prefix_lens.copy_(keep_pl)

    # CUDA graph: capture with shorter lengths, replay after restoring the real ones (static buffers, device lengths)
    from vllm.v1.attention.ops import tq_imma_cuda
    rl = c.prefix_lens.reshape(-1, 1).expand(c.B, ql).contiguous().reshape(-1)
    out = torch.empty_like(c.query)
    lse = torch.empty(c.query.shape[0], HQ, dtype=torch.float32, device=c.dev)
    real = rl.clone()
    rl.copy_((real // 3).clamp_min(1))

    def call():
        tq_imma_cuda.tq_imma_decode_attention(c.query, c.kv, c.block_table, rl, c.cent, SCALE, True, q_per_seq=ql,
                                              num_seqs=c.B, PiT=c.H, output_buf=out, lse_buf=lse)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call()                       # warm (LUT cached, extension loaded) outside capture
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    rl.copy_(real)
    g.replay()
    torch.cuda.synchronize()
    eo, el = c.imma_batched()
    torch.cuda.synchronize()
    d = {"maxabs_out": float((out.float() - eo.float()).abs().max().item()),
         "maxabs_lse": float((lse - el).abs().max().item())}
    good = d["maxabs_out"] <= 1e-3 and d["maxabs_lse"] <= 1e-4
    ok = ok and good
    emit(check="cudagraph_replay_new_lengths", **d, ok=good)

    # fused in-kernel rotation vs cuBLAS rotation + quant kernel
    os.environ["VLLM_TQ_IMMA_FUSEDQ"] = "0"
    uo, ul = c.imma_batched()
    os.environ["VLLM_TQ_IMMA_FUSEDQ"] = "1"
    torch.cuda.synchronize()
    d = {"maxrel_out": float(((uo.float() - eo.float()).norm(dim=-1) / eo.float().norm(dim=-1).clamp_min(1e-9)).max().item()),
         "maxabs_lse": float((ul - el).abs().max().item())}
    good = d["maxrel_out"] <= 1e-2 and d["maxabs_lse"] <= 1e-2
    ok = ok and good
    emit(check="fusedq_vs_unfused", **d, ok=good)


def main():
    global REL_MEAN, REL_MAX, LSE_MAX, GQA_FACTOR, ABS_MAX_FLOOR
    import argparse
    ap = argparse.ArgumentParser(description="K2 numerics gate (defaults = the strict gate)")
    ap.add_argument("--rel-mean", type=float, default=REL_MEAN)
    ap.add_argument("--rel-max", type=float, default=REL_MAX, help="absolute cap on rel_max")
    ap.add_argument("--lse-max", type=float, default=LSE_MAX)
    ap.add_argument("--gqa-factor", type=float, default=GQA_FACTOR)
    ap.add_argument("--abs-max-floor", type=float, default=ABS_MAX_FLOOR,
                    help="rel_max <= max(gqa_factor * tq_gqa rel_max, this)")
    a = ap.parse_args()
    REL_MEAN, REL_MAX, LSE_MAX, GQA_FACTOR, ABS_MAX_FLOOR = a.rel_mean, a.rel_max, a.lse_max, a.gqa_factor, a.abs_max_floor
    torch.manual_seed(0)
    emit(check="gates", rel_mean=REL_MEAN, rel_max=REL_MAX, lse_max=LSE_MAX, gqa_factor=GQA_FACTOR,
         abs_max_floor=ABS_MAX_FLOOR)
    emit(check="env", gpu=torch.cuda.get_device_name(0), cap=list(torch.cuda.get_device_capability(0)),
         fusedq=os.environ.get("VLLM_TQ_IMMA_FUSEDQ", "1"), qsplit=os.environ.get("VLLM_TQ_IMMA_QSPLIT", "1"))
    last = None
    for B, ctx, regime in ((1, 1024, "diffuse"), (1, 4096, "mixed"), (4, 16384, "mixed"), (1, 28000, "needle"),
                           (4, 28000, "mixed"), (8, 28000, "diffuse"), (2, 65536, "mixed")):
        last = check_case(B, ctx, regime)
        if (B, ctx) == (4, 28000):
            check_empty_and_graph(last)
        del last
        torch.cuda.empty_cache()
    print(json.dumps({"verdict": "PASS" if ok else "FAIL", "n_checks": len(results)}), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
