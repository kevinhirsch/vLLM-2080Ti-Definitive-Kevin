#!/usr/bin/env python3
"""K6 microbench: can a decode step run CONCURRENTLY with a prefill chunk on one
Turing card (sm_75) if decode sits on a high-priority CUDA stream?

Uses the real Marlin W4A16 kernel at TP2 per-rank shapes of Qwen3.8-27B
(64 layers x ~98 MB of int4 weights per rank = the decode working set, so decode
is bandwidth-realistic, not L2-resident).  Prints JSON with:
  decode_alone_ms, prefill_alone_tok_s,
  concurrent (hi-pri decode): decode step ms p50/p90 + prefill tok/s during overlap,
  concurrent (equal priority) control,
  time-sliced reference (what the vLLM step loop does today).
Needs ~7.5 GiB free on the chosen GPU and an otherwise idle GPU (engine stopped).
"""
import argparse, json, time, statistics as st
import torch
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize

# per-rank (TP2) projections of one layer: (K, N); ~98 MB int4 / layer / rank
SHAPES = [(5120, 17408), (8704, 5120), (5120, 8192), (4096, 5120)]

def build(n_layers, dev):
    qt = scalar_types.uint4b8
    base = []
    for K, N in SHAPES:
        w = torch.randn(K, N, dtype=torch.half, device=dev) * 0.02
        _, qw, s = marlin_quantize(w, qt, 128)
        base.append((K, N, qw, s)); del w
    layers = []
    for _ in range(n_layers):
        layers.append([(K, N, qw.clone(), s.clone()) for (K, N, qw, s) in base])
    ws = marlin_make_workspace_new(torch.device(dev), 4)
    return layers, ws, qt

def make_pass(layers, ws, qt, M, dev):
    acts = {K: torch.randn(M, K, dtype=torch.half, device=dev) for K, _ in SHAPES}
    outs = {N: torch.empty(M, N, dtype=torch.half, device=dev) for _, N in SHAPES}
    def run():
        for L in layers:
            for K, N, qw, s in L:
                ops.marlin_gemm(acts[K], outs[N], qw, None, s, None, None, None, ws, qt, M, N, K,
                                use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
    return run

def graph_of(fn, stream):
    with torch.cuda.stream(stream):
        for _ in range(2): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        fn()
    torch.cuda.synchronize()
    return g

def timed_replays(g, stream, n):
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n)]
    with torch.cuda.stream(stream):
        for a, b in ev:
            a.record(); g.replay(); b.record()
    return ev

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--layers', type=int, default=64)
    ap.add_argument('--prefill-m', type=int, default=1856)
    ap.add_argument('--decode-m', type=int, nargs='+', default=[4, 20, 48])
    ap.add_argument('--n-decode', type=int, default=60)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    dev = f'cuda:{a.gpu}'; torch.cuda.set_device(a.gpu)
    lo_p, hi_p = torch.cuda.Stream.priority_range()
    s_lo = torch.cuda.Stream(priority=0); s_hi = torch.cuda.Stream(priority=hi_p); s_eq = torch.cuda.Stream(priority=0)
    layers, ws, qt = build(a.layers, dev)
    pf_layers = layers  # prefill touches the same 64 layers
    res = dict(gpu=a.gpu, name=torch.cuda.get_device_name(a.gpu), priority_range=[lo_p, hi_p], layers=a.layers,
               weight_gib=sum(qw.numel()*qw.element_size()+s.numel()*s.element_size() for L in layers for (_,_,qw,s) in L)/2**30,
               prefill_m=a.prefill_m, rows={})
    pf = make_pass(pf_layers, ws, qt, a.prefill_m, dev)
    g_pf = graph_of(pf, s_lo)
    # prefill alone
    t0 = time.perf_counter()
    with torch.cuda.stream(s_lo):
        for _ in range(5): g_pf.replay()
    torch.cuda.synchronize(); pf_alone = (time.perf_counter() - t0) / 5
    res['prefill_alone_ms'] = pf_alone * 1e3
    res['prefill_alone_tok_s_rank_gemm_only'] = a.prefill_m / pf_alone
    for M in a.decode_m:
        dws = marlin_make_workspace_new(torch.device(dev), 4)
        dfn = make_pass(layers, dws, qt, M, dev)
        g_d_hi = graph_of(dfn, s_hi); g_d_eq = graph_of(dfn, s_eq)
        row = {}
        ev = timed_replays(g_d_hi, s_hi, 30); torch.cuda.synchronize()
        alone = [x.elapsed_time(y) for x, y in ev][5:]
        row['decode_alone_ms_p50'] = st.median(alone)
        for label, g_d, s_d in (('hi_prio', g_d_hi, s_hi), ('equal_prio', g_d_eq, s_eq)):
            # enough prefill work queued to outlast the decode steps
            n_pf = max(3, int(a.n_decode * row['decode_alone_ms_p50'] * 2.5 / (pf_alone * 1e3)) + 2)
            torch.cuda.synchronize()
            pe0 = torch.cuda.Event(enable_timing=True); pe1 = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(s_lo):
                pe0.record()
                for _ in range(n_pf): g_pf.replay()
                pe1.record()
            time.sleep(0.002)  # let the prefill start occupying the SMs
            ev = timed_replays(g_d, s_d, a.n_decode)
            de = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(s_d): de.record()
            torch.cuda.synchronize()
            dts = [x.elapsed_time(y) for x, y in ev]
            dec_span = ev[0][0].elapsed_time(de)
            pf_total = pe0.elapsed_time(pe1)
            # prefill progress: total time vs alone; attribute extra time to the overlap
            pf_slow = (pf_total - n_pf * pf_alone * 1e3)
            overlap = min(dec_span, pf_total)
            row[label] = dict(decode_step_ms_p50=st.median(dts), decode_step_ms_p90=sorted(dts)[int(.9*len(dts))],
                              decode_slowdown=st.median(dts) / row['decode_alone_ms_p50'],
                              prefill_passes=n_pf, prefill_total_ms=pf_total, prefill_extra_ms=pf_slow,
                              prefill_rate_during_overlap=max(0.0, 1 - pf_slow / overlap) if overlap > 0 else None,
                              decode_span_ms=dec_span)
        # time-sliced reference (today's step loop): a decode row waits for a whole chunk
        row['time_sliced_step_ms'] = pf_alone * 1e3 + row['decode_alone_ms_p50'] * 0.15  # decode rows ride in the chunk (~free)
        res['rows'][M] = row
        del g_d_hi, g_d_eq
    print(json.dumps(res, indent=1))
    if a.out: json.dump(res, open(a.out, 'w'), indent=1)

if __name__ == '__main__':
    main()
