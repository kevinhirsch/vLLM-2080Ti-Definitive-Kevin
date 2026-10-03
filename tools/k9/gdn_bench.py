#!/usr/bin/env python
"""Lane K9 (L102 scoping): FlashQLA legacy GDN prefill forward (the live gdn_prefill_backend=flashqla_legacy) at TP2
per-rank shapes (Hq=Hk=8, Hv=24, D=128), one 3,632-token chunk and packed varlen splits; time per layer call."""
import os, statistics, sys, json
sys.path.insert(0, "/home/kevin/Desktop/wt-integrate/.deps/FlashQLA-SM70-SM75")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/home/kevin/projects/lanes/k9/flashqla_ext")
import torch
from flash_qla.ops.gated_delta_rule.legacy.sm_legacy import chunk_gated_delta_rule_fwd_legacy_varlen as fwd
dev = torch.device("cuda"); Hk, Hv, D = 8, 24, 128
res = []
for T, nseq in [(3632, 1), (3632, 4), (3632, 12), (1024, 1)]:
    q = torch.nn.functional.normalize(torch.randn(1, T, Hk, D, device=dev), dim=-1).half()
    k = torch.nn.functional.normalize(torch.randn(1, T, Hk, D, device=dev), dim=-1).half()
    v = torch.randn(1, T, Hv, D, device=dev).half()
    g = (-torch.rand(1, T, Hv, device=dev) * 0.1).float(); beta = torch.rand(1, T, Hv, device=dev).float()
    cu = torch.linspace(0, T, nseq + 1, device=dev).round().int()
    st = torch.zeros(nseq, Hv, D, D, device=dev)
    for _ in range(2): fwd(q, k, v, g, beta, cu, initial_state=st.clone())
    ts = []
    for _ in range(7):
        s0 = st.clone(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); fwd(q, k, v, g, beta, cu, initial_state=s0); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1))
    m = statistics.median(ts); flop = T * Hv * 4 * D * D * 2
    r = {"T": T, "nseq": nseq, "med_ms": round(m, 3), "min_ms": round(min(ts), 3), "max_ms": round(max(ts), 3),
         "us_per_token_step": round(1000 * m / (T / nseq), 3), "gflops": round(flop / m / 1e6, 1), "x48_layers_ms": round(48 * m, 1)}
    res.append(r); print(json.dumps(r), flush=True)
json.dump(res, open("/home/kevin/projects/lanes/k9/gdn_bench.json", "w"), indent=1)
