#!/usr/bin/env python3
"""E16 P3 + P4c GPU equivalence (engine stopped; < 1 GB on one GPU).

P3, the TurboQuant continuation merge with the tree ancestor mask (Triton), production per-rank shapes
(Hq=12, Hk=2, head dim 256, fp16 q/k/v, fp32 prefix state), trees M=7 / M=8 / M=5:
  * tree output vs the dense fp32 reference tree_reference.tree_attention_ref (atol/rtol 2e-3);
  * every row r == the chain kernel run over the gathered path to r (last row), atol 1e-3;
  * a causal anc_mask is bit-identical to anc_mask=None (the chain path).
P4c, causal_conv1d_update_tree (Triton), conv dim 5120 fp16, SD layout ([slot, state_len, dim] storage),
B = 1..4 requests, engine order: eager padding-only warmup launch for every B, then the FIRST real call
inside torch.cuda.graph capture with capture-time metadata, then real metadata + replay:
  * outputs vs tree_reference.conv_tree_ref (fp16 tol), leaf stash bitwise = leaf pre-conv x;
  * chain rows and the stored window bit-identical to the production chain kernel (causal_conv1d_update);
  * slots of other requests untouched.
Prints one JSON line per case and a final {"test": "tree_p3_p4c_equiv", "verdict": "PASS"|"FAIL", ...} line;
exit 0 only on PASS.  `--cpu-dry-run` runs the same logic on the CPU (TRITON_INTERPRET=1, small shapes, no graph).
"""
from __future__ import annotations

import json
import os
import sys
import traceback

DRY = "--cpu-dry-run" in sys.argv
if DRY:
    os.environ.setdefault("TRITON_INTERPRET", "1")

import torch  # noqa: E402

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (  # noqa: E402
    causal_conv1d_update,
    causal_conv1d_update_tree,
)
from vllm.v1.attention.ops.triton_turboquant_decode import (  # noqa: E402
    triton_turboquant_spec_continuation_merge,
)
from vllm.v1.spec_decode import tree_reference as R  # noqa: E402
from vllm.v1.spec_decode.tree_spec import TreeSpec  # noqa: E402

DEV = torch.device("cpu" if DRY else "cuda")
HQ, HK, DH = (4, 2, 32) if DRY else (12, 2, 256)
C = 64 if DRY else 5120
KW, D = 4, 3
PAD = -1
SPECS = {
    "m7": TreeSpec.chain_plus_leaves(3),
    "m8": TreeSpec.chain_plus_leaves(3, leaf_depths=(1, 2), width=3),
    "m5": TreeSpec.chain_plus_leaves(3, leaf_depths=(2,)),
}


def prefix_state(q, pk, pv, scale):
    G = q.shape[1] // pk.shape[1]
    k = pk.float().repeat_interleave(G, 1)
    v = pv.float().repeat_interleave(G, 1)
    s = torch.einsum("mhd,nhd->hmn", q.float(), k) * scale
    return (torch.einsum("hmn,nhd->mhd", torch.softmax(s, -1), v),
            torch.logsumexp(s, -1).t().contiguous())


def case_p3(name, spec, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    M = spec.M
    scale = DH ** -0.5
    anc = torch.tensor(spec.anc_mask, dtype=torch.int32, device=DEV)
    rnd = lambda *s: torch.randn(*s, generator=g).to(DEV)  # noqa: E731
    q, k, v = rnd(M, HQ, DH).half(), rnd(M, HK, DH).half(), rnd(M, HK, DH).half()
    P = 37
    pk, pv = rnd(P, HK, DH).half(), rnd(P, HK, DH).half()
    po, pl = prefix_state(q, pk, pv, scale)
    out = triton_turboquant_spec_continuation_merge(q, k, v, po.clone(), pl, scale, anc_mask=anc)
    ref = R.tree_attention_ref(spec, q[None].float().cpu(), k[None].float().cpu(), v[None].float().cpu(),
                               [pk.float().cpu()], [pv.float().cpu()], scale)[0]
    err_ref = (out.float().cpu() - ref).abs().max().item()
    err_path = 0.0
    for r in range(M):
        path = list(spec.path_rows(r))
        po_p, pl_p = prefix_state(q[path], pk, pv, scale)
        o = triton_turboquant_spec_continuation_merge(q[path], k[path], v[path], po_p.clone(), pl_p, scale)
        err_path = max(err_path, (out[r].float() - o[-1].float()).abs().max().item())
    n = spec.D + 1
    causal = torch.tensor([(1 << (r + 1)) - 1 for r in range(n)], dtype=torch.int32, device=DEV)
    po4, pl4 = prefix_state(q[:n], pk, pv, scale)
    off = triton_turboquant_spec_continuation_merge(q[:n], k[:n], v[:n], po4.clone(), pl4, scale)
    on = triton_turboquant_spec_continuation_merge(q[:n], k[:n], v[:n], po4.clone(), pl4, scale, anc_mask=causal)
    ok = err_ref <= 2e-3 and err_path <= 1e-3 and torch.equal(off, on)
    return dict(case=name, ok=ok, M=M, maxabs_vs_ref=err_ref, maxabs_vs_path_chain=err_path,
                causal_mask_bitwise=bool(torch.equal(off, on)))


def case_p4c(name, spec, B, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    M, L = spec.M, spec.L
    state_len = KW - 1 + D
    rnd = lambda *s: torch.randn(*s, generator=g).to(DEV)  # noqa: E731
    weight = rnd(C, KW).half()
    bias = rnd(C).half()
    nslots = 1 + 4 * (D + 1) + 3
    store0 = rnd(nslots, state_len, C).half()  # SD storage
    tree = spec.device_tensors(DEV)
    stash = torch.zeros(4, L, C, dtype=torch.float16, device=DEV)  # B_tree = 4
    # persistent buffers the graph reads (engine style)
    x_buf = torch.zeros(B * M, C, dtype=torch.float16, device=DEV)
    idx_buf = torch.full((B, D + 1), PAD, dtype=torch.int32, device=DEV)
    na_buf = torch.ones(B, dtype=torch.int32, device=DEV)
    store = store0.clone()
    view = store.transpose(1, 2)

    def call():
        return causal_conv1d_update_tree(x_buf, view, weight, bias, "silu", idx_buf[:, 0], na_buf, tree,
                                         stash[:B], B, null_block_id=PAD)

    # engine order (P9 layer.tree_warmup): eager padding-only launch, nothing written
    call()
    if not DRY:
        torch.cuda.synchronize()
    if not torch.equal(store, store0):
        return dict(case=name, ok=False, error="padding-only warmup wrote conv state")
    # capture-time metadata: all slots real, accepted D+1 (build_for_cudagraph_capture)
    perm = torch.randperm(nslots - 1, generator=g)[: B * (D + 1)] + 1
    idx = perm.view(B, D + 1).int().to(DEV)
    if DRY:
        graph = None
    else:
        idx_buf.copy_(idx)
        na_buf.fill_(D + 1)
        snap = store.clone()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()  # FIRST real call inside capture
        torch.cuda.synchronize()
        if not torch.equal(store, snap):
            return dict(case=name, ok=False, error="capture mutated conv state")
    # real metadata
    x = rnd(B, M, C).half()
    na = torch.randint(1, D + 2, (B,), generator=g).int().to(DEV)
    x_buf.copy_(x.reshape(B * M, C))
    idx_buf.copy_(idx)
    na_buf.copy_(na)
    store.copy_(store0)
    stash.fill_(float("nan"))
    if graph is None:
        out = call()
    else:
        graph.replay()
        torch.cuda.synchronize()
        out = x_buf
    out = out.view(B, M, C).float().cpu()
    hist = torch.stack([store0[int(idx[b, 0]), int(na[b]) - 1: int(na[b]) - 1 + KW - 1] for b in range(B)])
    ref = R.conv_tree_ref(spec, hist.float().cpu(), x.float().cpu(), weight.float().cpu(), bias.float().cpu())
    err = (out - ref["out"]).abs().max().item()
    stash_ok = torch.equal(stash[:B].cpu(), x[:, list(spec.leaf_rows)].cpu())
    # chain kernel on the chain rows (production path), same inputs
    s2 = store0.clone()
    oc = causal_conv1d_update(x[:, : D + 1].transpose(1, 2).contiguous(), s2.transpose(1, 2), weight, bias,
                              "silu", conv_state_indices=idx[:, 0], num_accepted_tokens=na,
                              null_block_id=PAD)
    chain_bitwise = torch.equal(out[:, : D + 1], oc.transpose(1, 2).float().cpu())
    window_bitwise = torch.equal(store, s2)
    ok = err <= 3e-2 and stash_ok and chain_bitwise and window_bitwise
    return dict(case=name, ok=ok, M=M, B=B, maxabs_vs_ref=err, stash_bitwise=stash_ok,
                chain_rows_bitwise=chain_bitwise, window_bitwise_vs_chain=window_bitwise)


def main():
    cases = [(f"p3-{n}", lambda n=n, i=i: case_p3(f"p3-{n}", SPECS[n], 10 + i)) for i, n in enumerate(SPECS)]
    for n in SPECS:
        for B in (1, 2, 3, 4):
            cases.append((f"p4c-{n}-b{B}", lambda n=n, B=B: case_p4c(f"p4c-{n}-b{B}", SPECS[n], B, 100 + B)))
    failed = []
    for name, fn in cases:
        try:
            res = fn()
        except Exception as e:  # noqa: BLE001
            res = dict(case=name, ok=False, error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
        print(json.dumps(res), flush=True)
        if not res.get("ok"):
            failed.append(name)
    verdict = "PASS" if not failed else "FAIL"
    print(json.dumps({"test": "tree_p3_p4c_equiv", "verdict": verdict, "cases": len(cases), "failed": failed,
                      "dry_run": DRY}), flush=True)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
