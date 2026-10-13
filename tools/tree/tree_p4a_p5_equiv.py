#!/usr/bin/env python3
"""E16 P4a + P5 GPU equivalence (engine stopped; ~3 GB of GPU memory on one card).

P4a, the K5 tree-leaf kernel (gdn_mtp_tree), at production per-TP-rank shapes (TP=2: H=8, HV=24, K=V=128,
fp32 temporal state, fp16 activations, fp16 dt_bias / norm weight, D=3 so W=4 slots per request):
  * engine order (P9): one eager padding-only warmup launch, then the FIRST real call inside torch.cuda.graph
    capture with capture-time metadata (as in the engine), then the real
    metadata is written into the same persistent buffers and the graph is replayed;
  * trees: production M=7 (one rank-1 leaf per depth), M=8 (two leaves at depths 1 and 2), M=5 (one leaf at 2);
  * real requests 1..4 inside a 4-request graph (graph-padded requests: cu filled, slots -1, accepted 1),
    random accepted counts, slot rows drawn from a shared pool (slot 0 included);
  checks against the P1 CPU reference tree_reference.gdn_tree_ref (+ RMSNormGated * silu(z)) and against the
  chain kernel (gdn_mtp, the production K5 path) run on the same requests' first W rows:
    - chain slots BIT-IDENTICAL to the chain kernel's, chain output rows bit-identical;
    - no other slot written (leaves store no state); graph-padded requests untouched;
    - leaf output rows vs a chain-kernel run over the leaf's path [x0, c1..c_{j-1}, leaf] (bitwise reported,
      tolerance 1e-3 fp16) and vs the reference;
    - leaf stash vs reference (k | v | decay | beta).
P5, the commit (TreeCommit, one launch), on 48 GDN layers (state from the real K5 tree verify of each layer) + 16
TurboQuant caches (k3v4_nc packed slots, LBNHC [blocks, Hk=2, 16, slot] views), FIRST call inside capture:
    - rescued requests: slot idx[b, j] == a from-scratch chain run (gdn_mtp) over the accepted path, max abs <= 1e-5;
    - slot idx[b, j-1] and every other slot unchanged; non-rescued requests untouched (bitwise);
    - conv column (Kw-2)+j of slot idx[b, 0] == the conv stash row (bitwise); other columns unchanged;
    - TQ packed row of slot_mapping[b*M+j] == row of slot_mapping[b*M+R] (bitwise), nothing else changed.
Prints one JSON line per case and a final {"test": "tree_p4a_p5_equiv", "verdict": "PASS"|"FAIL", ...} line.
Exit 0 only on PASS.  Run with PYTHONPATH=<this worktree or its release root>.
Needs ~5 GB on one GPU.  `--cpu-dry-run` checks this script's own logic on the CPU (no GPU): the CUDA kernels are
replaced by the torch emulations in tests/v1/spec_decode/test_tree_p4a_p5.py, graph capture by record/replay,
and the P5 cases use 3 GDN + 2 TQ layers.  It proves nothing about the kernels.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

import torch
import torch.nn.functional as F

LANE_BUILD = "/home/kevin/projects/lanes/coexec/build/tree-p4a-p5"
os.environ.setdefault("VLLM_K5_GDN_BUILD_DIR", f"{LANE_BUILD}/k5")
os.environ.setdefault("VLLM_TREE_COMMIT_BUILD_DIR", f"{LANE_BUILD}/commit")
os.environ.setdefault("VLLM_K5_GDN_FUSED", "1")
os.environ.setdefault("VLLM_K5_GDN_VARIANT", "2")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")  # same flags as the CPU pre-build (no rebuild in the window)

from vllm.model_executor.layers.mamba.gdn import gdn_sm75_cuda as k5  # noqa: E402
from vllm.v1.spec_decode import tree_reference as R  # noqa: E402
from vllm.v1.spec_decode.tree_commit import GDNCommitLayer, TreeCommit  # noqa: E402
from vllm.v1.spec_decode.tree_spec import TreeSpec, TreeStash  # noqa: E402

DRY = "--cpu-dry-run" in sys.argv
DEV = torch.device("cpu" if DRY else "cuda")
TP = 2
NK, NV, DK, DV, KW = 16, 48, 128, 128, 4
H, HV = NK // TP, NV // TP  # 8, 24
QKV = (2 * NK * DK + NV * DV) // TP  # 5120 (= conv dim per rank)
D = 3
W = D + 1
CONV_LEN = KW - 1 + D  # 6
PAD = -1
EPS = 1e-6
SCALE = DK ** -0.5
NCAP = 4  # graph batch (B_tree)


def tq_slot_bytes() -> int:
    try:
        from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
        return int(TurboQuantConfig.from_cache_dtype("turboquant_k3v4_nc", 256).slot_size_aligned)
    except Exception:
        return 612


def maxabs(a, b):
    return (a.float() - b.float()).abs().max().item() if a.numel() else 0.0


class Inputs:
    """Persistent per-layer verify inputs + metadata buffers (what the engine's graph reads)."""

    def __init__(self, spec: TreeSpec, nslots: int, g: torch.Generator):
        M = spec.M
        self.spec, self.M = spec, M
        r = lambda *s, sc=1.0: torch.randn(*s, generator=g, device=DEV) * sc
        self.mixed = (r(NCAP * M, QKV, sc=0.7)).half()
        self.a = r(NCAP * M, HV).half()
        self.b = r(NCAP * M, HV).half()
        self.z = r(NCAP * M, HV, DV).half()
        self.A_log = r(HV, sc=0.5).float()
        self.dt_bias = r(HV, sc=0.3).half()
        self.norm_w = (1 + 0.1 * r(DV)).half()
        self.state = (r(nslots, HV, DV, DK, sc=0.05)).float()
        self.sidx = torch.full((NCAP, W), PAD, dtype=torch.int32, device=DEV)
        self.cu = torch.zeros(NCAP + 1, dtype=torch.int32, device=DEV)
        self.nacc = torch.ones(NCAP, dtype=torch.int32, device=DEV)
        self.out = torch.zeros(NCAP * M, HV, DV, dtype=torch.float16, device=DEV)

    def load(self, rows, acc, n_real):
        self.sidx.fill_(PAD)
        self.nacc.fill_(1)
        for i in range(n_real):
            self.sidx[i] = torch.tensor(rows[i], dtype=torch.int32)
            self.nacc[i] = int(acc[i])
        cu = [min(i, n_real) * self.M for i in range(NCAP + 1)]  # padded requests: zero rows
        self.cu.copy_(torch.tensor(cu, dtype=torch.int32))

    def tree_call(self, stash_view):
        k5.gdn_mtp_tree(self.mixed, self.a, self.b, self.A_log, self.dt_bias, self.sidx, self.cu, self.nacc,
                        self.state, self.z, self.norm_w, self.out, SCALE, EPS, PAD, False,
                        self.spec.leaf_rows, self.spec.leaf_depth, stash_view)


def chain_run(inp: Inputs, state0, req_rows, sidx_row, acc, path):
    """Production chain kernel over the given rows of one request, on a copy of state0."""
    st = state0.clone()
    rows = torch.tensor(path, device=DEV) + req_rows
    T = len(path)
    out = torch.zeros(T, HV, DV, dtype=torch.float16, device=DEV)
    k5.gdn_mtp(inp.mixed[rows].contiguous(), inp.a[rows].contiguous(), inp.b[rows].contiguous(), inp.A_log,
               inp.dt_bias, torch.tensor([sidx_row], dtype=torch.int32, device=DEV),
               torch.tensor([0, T], dtype=torch.int32, device=DEV),
               torch.tensor([acc], dtype=torch.int32, device=DEV), st, inp.z[rows].contiguous(), inp.norm_w, out,
               SCALE, EPS, PAD, False, 2)
    return st, out


def reference(inp: Inputs, state0, rows, acc, n):
    spec, M = inp.spec, inp.M
    mx = inp.mixed[: n * M].float().cpu().reshape(n, M, -1)
    q = mx[..., : H * DK].reshape(n, M, H, DK)
    k = mx[..., H * DK: 2 * H * DK].reshape(n, M, H, DK)
    v = mx[..., 2 * H * DK:].reshape(n, M, HV, DV)
    x = inp.a[: n * M].float().cpu() + inp.dt_bias.float().cpu()
    g = (-torch.exp(inp.A_log.cpu()) * torch.where(x <= 20, torch.log1p(torch.exp(x)), x)).reshape(n, M, HV)
    beta = torch.sigmoid(inp.b[: n * M].float().cpu()).reshape(n, M, HV)
    h0 = torch.stack([state0[rows[i][acc[i] - 1]].cpu() for i in range(n)])
    ref = R.gdn_tree_ref(spec, h0, q, k, v, g, beta, SCALE)
    o = ref["o"]
    zz = inp.z[: n * M].float().cpu().reshape(n, M, HV, DV)
    y = (o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS)) * inp.norm_w.float().cpu() * F.silu(zz)
    return ref, y


def case_p4a(name, spec: TreeSpec, n_real: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    nslots = NCAP * W + 6
    inp = Inputs(spec, nslots, g)
    stash = TreeStash.allocate(spec, ["layer"], HV, DK, DV, QKV, torch.float16, DEV)  # before capture
    view = stash.gdn_view("layer")
    perm = torch.randperm(nslots, generator=torch.Generator().manual_seed(seed)).tolist()
    perm.remove(0)
    perm.insert(0, 0)
    rows = [perm[i * W:(i + 1) * W] for i in range(NCAP)]
    # Engine order (P9 runner init: layer.tree_warmup): one eager padding-only launch before any capture
    # (extension load + lazy CUDA module load happen here, never inside capture); writes nothing.
    inp.load([], [], 0)
    pre_warm = inp.state.clone()
    inp.tree_call(view)
    torch.cuda.synchronize()
    if not torch.equal(inp.state, pre_warm):
        return dict(case=name, ok=False, error="padding-only warmup wrote state")
    # capture-time metadata: all requests real, all accepted (what build_for_cudagraph_capture produces)
    inp.load(rows, [W] * NCAP, NCAP)
    state_cap = inp.state.clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        inp.tree_call(view)  # FIRST real call of the tree kernel = inside capture (after the warmup)
    torch.cuda.synchronize()
    if not torch.equal(inp.state, state_cap):
        return dict(case=name, ok=False, error="capture mutated state")
    acc = torch.randint(1, W + 1, (NCAP,), generator=torch.Generator().manual_seed(seed + 1)).tolist()
    inp.load(rows, acc, n_real)
    state0 = inp.state.clone()
    inp.out.fill_(float("nan"))
    stash.gdn_leaf.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    M = spec.M
    st = inp.state
    res = dict(case=name, M=M, L=spec.L, n_real=n_real, acc=acc[:n_real])
    errs = []
    # chain: bit-identical to the production chain kernel, per request
    chain_bitwise, leaf_bitwise, leaf_vs_path, out_vs_ref, stash_err, chain_ref = True, True, 0.0, 0.0, 0.0, 0.0
    ref, y = reference(inp, state0, rows, acc, n_real)
    written = set()
    for i in range(n_real):
        written |= set(rows[i])
        st_c, out_c = chain_run(inp, state0, i * M, rows[i], acc[i], list(range(W)))
        for d in range(W):
            if not torch.equal(st[rows[i][d]], st_c[rows[i][d]]):
                chain_bitwise = False
            chain_ref = max(chain_ref, maxabs(st[rows[i][d]].cpu(), ref["chain_states"][i, d]))
        if not torch.equal(inp.out[i * M: i * M + W], out_c):
            chain_bitwise = False
        for l, Rr in enumerate(spec.leaf_rows):
            j = spec.leaf_depth[l]
            _, out_p = chain_run(inp, state0, i * M, rows[i], acc[i], list(range(j)) + [Rr])
            leaf_bitwise &= torch.equal(inp.out[i * M + Rr], out_p[-1])
            leaf_vs_path = max(leaf_vs_path, maxabs(inp.out[i * M + Rr], out_p[-1]))
        out_vs_ref = max(out_vs_ref, maxabs(inp.out[i * M:(i + 1) * M].cpu(), y[i]))
        stash_err = max(stash_err, maxabs(view[i, : spec.L].cpu(), ref["leaf_stash"][i]))
    others = [s for s in range(nslots) if s not in written]
    untouched = torch.equal(st[others], state0[others])
    pad_rows_ok = not torch.isnan(inp.out[: n_real * M]).any().item()
    pad_untouched = bool(torch.isnan(inp.out[n_real * M:]).all().item()) if n_real < NCAP else True
    smag = state0.abs().max().item()
    ok = (chain_bitwise and untouched and pad_rows_ok and pad_untouched and leaf_vs_path <= 1e-3
          and out_vs_ref <= 2e-2 and stash_err <= 1e-5 and chain_ref <= 1e-3 * smag + 1e-5)
    if not chain_bitwise:
        errs.append("chain slots/outputs differ from the chain kernel")
    res.update(ok=ok, chain_bitwise=chain_bitwise, other_slots_untouched=untouched, pad_ok=pad_untouched,
               leaf_out_bitwise_vs_path_run=leaf_bitwise, leaf_out_maxabs_vs_path_run=leaf_vs_path,
               out_maxabs_vs_ref=out_vs_ref, stash_maxabs_vs_ref=stash_err, chain_state_maxabs_vs_ref=chain_ref,
               errors=errs)
    del graph
    return res


def case_p5(name, spec: TreeSpec, n_real: int, rescue: list[int], seed: int, NG=48, NT=16):
    torch.manual_seed(seed)
    g = torch.Generator(device=DEV).manual_seed(seed)
    M = spec.M
    nslots = NCAP * W + 4
    names = [f"model.layers.{i}.linear_attn" for i in range(NG)]
    stash = TreeStash.allocate(spec, names, HV, DK, DV, QKV, torch.float16, DEV)
    perm = torch.randperm(nslots, generator=torch.Generator().manual_seed(seed)).tolist()
    rows = [perm[i * W:(i + 1) * W] for i in range(NCAP)]
    acc = torch.randint(1, W + 1, (NCAP,), generator=torch.Generator().manual_seed(seed + 1)).tolist()
    idx_buf = torch.full((64, W), PAD, dtype=torch.int32, device=DEV)  # the builder's persistent buffer
    for i in range(n_real):
        idx_buf[i] = torch.tensor(rows[i], dtype=torch.int32)
    layers, inps, state0s = [], [], []
    for li, nm in enumerate(names):
        inp = Inputs(spec, nslots, g)
        inp.load(rows, acc, n_real)
        state0s.append(inp.state.clone())
        inp.tree_call(stash.gdn_view(nm))  # the real verify writes chain slots + this layer's stash
        conv = torch.randn(nslots, CONV_LEN, QKV, generator=g, device=DEV).half()  # SD layout
        layers.append(GDNCommitLayer(nm, inp.state, conv.transpose(-1, -2), idx_buf))
        inps.append(inp)
    stash.conv_leaf_x.normal_(generator=g)
    sb = tq_slot_bytes()
    nblocks = 8
    tq = [torch.randint(0, 256, (nblocks, 16, 2, sb), generator=g, device=DEV, dtype=torch.uint8).transpose(1, 2)
          for _ in range(NT)]
    tc = TreeCommit(spec, stash, layers[::-1], tq, DEV)  # tables built here, before capture
    tc.warmup()  # engine order (P9 runner init): eager load + one fully masked launch, before capture
    rescue_buf = torch.full((NCAP,), -1, dtype=torch.int32, device=DEV)
    sm_buf = torch.full((NCAP * M,), -1, dtype=torch.int64, device=DEV)
    torch.cuda.synchronize()
    pre_ssm = [x.ssm_state.clone() for x in layers]
    pre_conv = [x.conv_state.clone() for x in layers]
    pre_tq = [kv.clone() for kv in tq]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tc.commit(rescue_buf, sm_buf, n_real)  # FIRST real call of the commit = inside capture (after warmup)
    torch.cuda.synchronize()
    sm = torch.randperm(nblocks * 16, generator=torch.Generator().manual_seed(seed + 2))[: NCAP * M]
    sm_buf.copy_(sm.to(DEV))
    rescue_buf.copy_(torch.tensor(rescue + [-1] * (NCAP - len(rescue)), dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()
    worst_gate, bit_other, conv_ok, tq_ok = 0.0, True, True, True
    for x, p_ssm, p_conv, inp, s0 in zip(layers, pre_ssm, pre_conv, inps, state0s):
        gi = stash.layer_index[x.name]
        expect_slots = p_ssm.clone()
        expect_conv = p_conv.clone()
        for i in range(n_real):
            l = rescue[i]
            if l < 0:
                continue
            j = spec.leaf_depth[l]
            st_c, _ = chain_run(inp, s0, i * M, rows[i], acc[i], list(range(j)) + [spec.leaf_rows[l]])
            dst = rows[i][j]
            worst_gate = max(worst_gate, maxabs(x.ssm_state[dst], st_c[dst]))
            expect_slots[dst] = x.ssm_state[dst]  # checked against the chain run above
            expect_conv[rows[i][0], :, CONV_LEN - W + j] = stash.conv_leaf_x[gi, i, l]
        bit_other &= torch.equal(x.ssm_state, expect_slots)
        conv_ok &= torch.equal(x.conv_state, expect_conv)
    for kv, kv0 in zip(tq, pre_tq):
        exp = kv0.clone()
        for i in range(n_real):
            l = rescue[i]
            if l < 0:
                continue
            j, Rr = spec.leaf_depth[l], spec.leaf_rows[l]
            d, s = int(sm[i * M + j]), int(sm[i * M + Rr])
            exp[d // 16, :, d % 16] = kv0[s // 16, :, s % 16]
        tq_ok &= torch.equal(kv, exp)
    ok = worst_gate <= 1e-5 and bit_other and conv_ok and tq_ok
    del graph
    return dict(case=name, ok=ok, M=M, n_real=n_real, rescue=rescue, acc=acc[:n_real], layers=NG, tq_layers=NT,
                slot_j_maxabs_vs_chain_run=worst_gate, other_slots_bitwise=bit_other, conv_column_ok=conv_ok,
                tq_rows_ok=tq_ok)


def _install_dry_run():
    """CPU stand-ins: emulated kernels, a record/replay 'graph'.  Only for checking this script."""
    global case_p5
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(root, "tests", "v1", "spec_decode"))
    import test_tree_p4a_p5 as T

    def tree(mixed, a, b, A_log, dt_bias, sidx, cu, nacc, state, z, nw, out, scale, eps, null, sig, lr, ld, stash):
        T.emulate_k5(mixed, a, b, A_log, dt_bias, sidx, cu, nacc, state, z, nw, out, scale, eps, null, sig, lr, ld,
                     stash)

    def chain(mixed, a, b, A_log, dt_bias, sidx, cu, nacc, state, z, nw, out, scale, eps, null, sig, variant=None):
        T.emulate_k5(mixed, a, b, A_log, dt_bias, sidx, cu, nacc, state, z, nw, out, scale, eps, null, sig)

    k5.gdn_mtp_tree, k5.gdn_mtp = tree, chain
    TreeCommit.commit = lambda self, rl, sm, n: T.emulate_commit(self, rl[:n], sm)

    class FakeGraph:
        cur = None

        def __init__(self):
            self.calls = []

        def replay(self):
            for f in self.calls:
                f()

    class Capture:
        def __init__(self, gr):
            self.gr = gr

        def __enter__(self):
            FakeGraph.cur = self.gr

        def __exit__(self, *a):
            FakeGraph.cur = None

    def recording(fn):
        def w(*a, **k):
            if FakeGraph.cur is not None:
                FakeGraph.cur.calls.append(lambda: fn(*a, **k))  # captured: recorded, not run
                return None
            return fn(*a, **k)
        return w

    k5.gdn_mtp_tree = recording(k5.gdn_mtp_tree)
    TreeCommit.commit = recording(TreeCommit.commit)
    torch.cuda.CUDAGraph = FakeGraph
    torch.cuda.graph = Capture
    torch.cuda.synchronize = lambda *a: None
    torch.cuda.empty_cache = lambda: None
    torch.cuda.is_available = lambda: True
    small = case_p5
    case_p5 = lambda *a, **k: small(*a, NG=3, NT=2, **k)


def main() -> int:
    if DRY:
        _install_dry_run()
    if not torch.cuda.is_available():
        print(json.dumps(dict(test="tree_p4a_p5_equiv", verdict="FAIL", error="no CUDA device")))
        return 1
    m7 = TreeSpec.chain_plus_leaves(3)
    m8 = TreeSpec.chain_plus_leaves(3, leaf_depths=(1, 2), width=3)
    m5 = TreeSpec.chain_plus_leaves(3, leaf_depths=(2,))
    cases = [
        ("p4a-m7-c1-of-4-graph", lambda: case_p4a("p4a-m7-c1-of-4-graph", m7, 1, 11)),
        ("p4a-m7-c3-of-4-graph", lambda: case_p4a("p4a-m7-c3-of-4-graph", m7, 3, 12)),
        ("p4a-m7-c4-of-4-graph", lambda: case_p4a("p4a-m7-c4-of-4-graph", m7, 4, 13)),
        ("p4a-m8-c2-of-4-graph", lambda: case_p4a("p4a-m8-c2-of-4-graph", m8, 2, 14)),
        ("p4a-m5-c4-of-4-graph", lambda: case_p4a("p4a-m5-c4-of-4-graph", m5, 4, 15)),
        ("p5-m7-c4-mixed-rescue", lambda: case_p5("p5-m7-c4-mixed-rescue", m7, 4, [0, -1, 2, 1], 21)),
        ("p5-m7-c1-rescue-d3", lambda: case_p5("p5-m7-c1-rescue-d3", m7, 1, [2], 22)),
        ("p5-m8-c3-rescue", lambda: case_p5("p5-m8-c3-rescue", m8, 3, [3, 1, -1], 23)),
    ]
    results = []
    for name, fn in cases:
        try:
            r = fn()
        except Exception as e:  # a crash is a FAIL with its cause, not a hang
            r = dict(case=name, ok=False, error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
        print(json.dumps(r), flush=True)
        results.append(r)
        torch.cuda.empty_cache()
    ok = all(r.get("ok") for r in results)
    print(json.dumps(dict(test="tree_p4a_p5_equiv", verdict="PASS" if ok else "FAIL", cases=len(results),
                          failed=[r["case"] for r in results if not r.get("ok")])), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
