#!/usr/bin/env python
"""Lane K9 (L102): K9 chunked GDN kernel vs (a) fp64 token recurrence (small shapes, 2 sequences, ragged lengths,
non-zero initial state) and (b) FlashQLA legacy (live backend) at per-rank shapes; then interleaved timing.
Footprint < 200 MiB. Usage: python test_gdn_chunk.py [--bench]"""
import argparse, importlib.util, json, os, statistics, sys
import torch
sys.path.insert(0, os.path.dirname(__file__)); import gdn_chunk_ref as REF
sys.path.insert(0, "/home/kevin/Desktop/wt-integrate/.deps/FlashQLA-SM70-SM75")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/home/kevin/projects/lanes/k9/flashqla_ext")
from flash_qla.ops.gated_delta_rule.legacy.sm_legacy import chunk_gated_delta_rule_fwd_legacy_varlen as fql
s = importlib.util.spec_from_file_location("k9g", "/home/kevin/Desktop/wt-k9/vllm/model_executor/layers/mamba/gdn/k9_gdn_chunk/__init__.py")
K9 = importlib.util.module_from_spec(s); s.loader.exec_module(K9)
ap = argparse.ArgumentParser(); ap.add_argument("--bench", action="store_true"); ap.add_argument("--reps", type=int, default=7)
a = ap.parse_args()
dev = torch.device("cuda"); D = 128; scale = D ** -0.5
def mk(lens, Hk, Hv, gmax=0.2, seed=0):
    g_ = torch.Generator().manual_seed(seed); T = sum(lens)
    q = torch.nn.functional.normalize(torch.randn(1, T, Hk, D, generator=g_), dim=-1).half()
    k = torch.nn.functional.normalize(torch.randn(1, T, Hk, D, generator=g_), dim=-1).half()
    v = torch.randn(1, T, Hv, D, generator=g_).half()
    g = -torch.rand(1, T, Hv, generator=g_) * gmax; beta = torch.rand(1, T, Hv, generator=g_)
    st = torch.randn(len(lens), Hv, D, D, generator=g_) * 0.05
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32)
    return q, k, v, g, beta, st, cu
def rel(x, y): return ((x.double() - y.double()).norm() / y.double().norm()).item()
# (a) vs fp64 recurrence
lens, Hk, Hv = [130, 77], 2, 6
q, k, v, g, beta, st, cu = mk(lens, Hk, Hv)
ref_o, ref_s = [], []
for i in range(len(lens)):
    sl = slice(int(cu[i]), int(cu[i + 1]))
    o_, s_ = REF.naive(q[0, sl].float(), k[0, sl].float(), v[0, sl].float(), g[0, sl], beta[0, sl], scale, st[i])
    ref_o.append(o_); ref_s.append(s_)
ref_o = torch.cat(ref_o); ref_s = torch.stack(ref_s)
for f16 in (False, True):
    o, s2 = K9.gdn_chunk_fwd_varlen(q.to(dev), k.to(dev), v.to(dev), g.to(dev), beta.to(dev), cu.to(dev), scale, st.to(dev), f16qk=f16)
    print(json.dumps({"check": "vs_fp64_recurrence", "f16qk": f16, "out_rel": rel(o[0].cpu(), ref_o), "state_rel": rel(s2.cpu(), ref_s),
                      "nonfinite": int((~torch.isfinite(o)).sum())}), flush=True)
o, s2 = fql(q.to(dev), k.to(dev), v.to(dev), g.to(dev), beta.to(dev), cu.to(dev), initial_state=st.to(dev).clone())
print(json.dumps({"check": "flashqla_vs_fp64", "out_rel": rel(o[0].cpu(), ref_o), "state_rel": rel(s2.cpu(), ref_s)}), flush=True)
# (b) per-rank shapes vs FlashQLA, timing
if a.bench:
    for lens in ([3632], [908] * 4, [302] * 12, [1024]):
        q, k, v, g, beta, st, cu = (x.to(dev) for x in mk(lens, 8, 24, gmax=0.1, seed=1))
        oA, sA = fql(q, k, v, g, beta, cu, initial_state=st.clone())
        row = {"lens": f"{len(lens)}x{lens[0]}"}
        fns = {"flashqla": lambda: fql(q, k, v, g, beta, cu, initial_state=st.clone())}
        for f16 in (False, True):
            oB, sB = K9.gdn_chunk_fwd_varlen(q, k, v, g, beta, cu, scale, st, f16qk=f16)
            row[f"k9_f16qk{int(f16)}_out_rel_vs_flashqla"] = rel(oB, oA); row[f"k9_f16qk{int(f16)}_state_rel"] = rel(sB, sA)
            fns[f"k9_f16qk{int(f16)}"] = (lambda f16=f16: K9.gdn_chunk_fwd_varlen(q, k, v, g, beta, cu, scale, st, f16qk=f16))
        for fn in fns.values():
            for _ in range(2): fn()
        t = {n: [] for n in fns}
        for r in range(a.reps):
            for n_ in (list(fns) if r % 2 == 0 else list(fns)[::-1]):
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record(); fns[n_](); e1.record(); e1.synchronize()
                t[n_].append(e0.elapsed_time(e1))
        for n_ in fns:
            row[n_ + "_ms"] = [round(statistics.median(t[n_]), 3), round(min(t[n_]), 3), round(max(t[n_]), 3)]
        row["speedup_f16qk0"] = round(row["flashqla_ms"][0] / row["k9_f16qk0_ms"][0], 2)
        row["speedup_f16qk1"] = round(row["flashqla_ms"][0] / row["k9_f16qk1_ms"][0], 2)
        print(json.dumps(row), flush=True)
