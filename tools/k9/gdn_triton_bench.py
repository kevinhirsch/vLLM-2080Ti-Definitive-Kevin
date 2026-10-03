#!/usr/bin/env python
"""Lane K9 (L102): is the existing chunked FLA Triton GDN (vllm third_party, gdn_prefill_backend=triton) faster than
FlashQLA legacy on sm_75? Same per-rank shapes; output agreement with zero initial state."""
import os, statistics, sys, json
sys.path.insert(0, "/home/kevin/Desktop/wt-integrate/.deps/FlashQLA-SM70-SM75")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/home/kevin/projects/lanes/k9/flashqla_ext")
os.environ.setdefault("TRITON_CACHE_DIR", "/home/kevin/projects/lanes/k9/triton_cache")
import torch
from flash_qla.ops.gated_delta_rule.legacy.sm_legacy import chunk_gated_delta_rule_fwd_legacy_varlen as fql
from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule as fla
dev = torch.device("cuda"); Hk, Hv, D = 8, 24, 128
def t(fn, n=7):
    for _ in range(2): fn()
    ts = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record(); fn(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1))
    return statistics.median(ts), min(ts), max(ts)
for T, nseq in [(3632, 1), (3632, 4)]:
    q = torch.nn.functional.normalize(torch.randn(1, T, Hk, D, device=dev), dim=-1).half()
    k = torch.nn.functional.normalize(torch.randn(1, T, Hk, D, device=dev), dim=-1).half()
    v = torch.randn(1, T, Hv, D, device=dev).half()
    g = (-torch.rand(1, T, Hv, device=dev) * 0.1).float(); beta = torch.rand(1, T, Hv, device=dev).half()
    cu = torch.linspace(0, T, nseq + 1, device=dev).round().int(); cul = cu.long()
    qr, kr = q.repeat_interleave(Hv // Hk, 2), k.repeat_interleave(Hv // Hk, 2)
    st = torch.zeros(nseq, Hv, D, D, device=dev)
    o1, _ = fql(q, k, v, g, beta.float(), cu, initial_state=st.clone())
    o2, _ = fla(qr, kr, v, g, beta, scale=D ** -0.5, initial_state=st.clone(), output_final_state=True, cu_seqlens=cul)
    rel = ((o2.float() - o1.float()).norm() / o1.float().norm()).item()
    a = t(lambda: fql(q, k, v, g, beta.float(), cu, initial_state=st.clone()))
    b = t(lambda: fla(qr, kr, v, g, beta, scale=D ** -0.5, initial_state=st.clone(), output_final_state=True, cu_seqlens=cul))
    print(json.dumps({"T": T, "nseq": nseq, "flashqla_ms": [round(x, 3) for x in a], "fla_triton_ms": [round(x, 3) for x in b],
                      "triton_speedup": round(a[0] / b[0], 2), "out_rel_diff": rel}), flush=True)
