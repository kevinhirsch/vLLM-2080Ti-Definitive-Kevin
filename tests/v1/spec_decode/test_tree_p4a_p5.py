# SPDX-License-Identifier: Apache-2.0
"""CPU tests for E16 P4a (K5 GDN tree-leaf kernel) and P5 (tree commit).

No GPU.  Three kinds of test:

* torch emulations of the two CUDA kernels, written CTA-by-CTA from the .cu
  (same row/leaf schedule, single-pass algebra, fp16 rounding points, stash
  layout, slot/column/packed-row address math from the layer tables), checked
  against the P1 reference ``tree_reference`` (gdn_tree_ref, gdn_commit_ref,
  conv_commit_ref, gather_chain_view, gdn_sequence_ref) and end to end
  (verify -> walk-chosen rescue -> commit -> the next step reads exactly the
  state of a from-scratch chain run over the accepted path);
* contracts: tree off leaves every chain path unchanged (metadata, kernel
  arguments, chain SASS), nothing is created lazily inside graph-captured
  code, the builder fills the persistent slot buffer the commit reads;
* nvcc: both .cu files compile for sm_75, and the chain kernel instantiations
  of gdn_mtp_sm75.cu are SASS-identical to the K5 base commit's.
"""

from __future__ import annotations

import ast
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.v1.spec_decode import tree_reference as R
from vllm.v1.spec_decode.tree_spec import TreeSpec, TreeStash, TreeStepOutputs

ROOT = Path(__file__).resolve().parents[3]
K5_CU = ROOT / "vllm/model_executor/layers/mamba/gdn/gdn_sm75_ext/gdn_mtp_sm75.cu"
COMMIT_CU = ROOT / "vllm/v1/spec_decode/tree_commit_ext/tree_commit_sm75.cu"
K5_BASE_COMMIT = "b3f2a4ffb4"  # fixed K5 (k5fix-prod-20261013), the chain kernel this branch must not change

KD = VD = 128
STASH_W = KD + VD + 2


def f16r(x: torch.Tensor) -> torch.Tensor:
    return x.half().float()


# =====================================================================================================================
# Emulation of gdn_mtp_sm75_v2_kernel<S, SIGMOID, TREE> (one CTA = (request n, v-head hv))
# =====================================================================================================================
def emulate_k5(
    mixed, a, b, A_log, dt_bias, sidx, cu, nacc, state, z, norm_w, out, scale, eps, null_id,
    sigmoid_gate=False, leaf_rows=(), leaf_depth=(), stash=None,
):
    """Mutates state / out / stash exactly where the kernel would.  Returns the per-row pre-norm outputs (so) for
    inspection: dict[(n, hv)] -> [T, V]."""
    tree = len(leaf_rows) > 0
    N, W = sidx.shape
    HV = state.shape[1]
    H = (mixed.shape[1] - HV * VD) // (2 * KD)
    hpk = HV // H
    num_tokens = mixed.shape[0]
    num_slots = state.shape[0]
    pre = {}
    for n in range(N):
        bos, eos = int(cu[n]), int(cu[n + 1])
        for hv in range(HV):
            if bos < 0 or eos < bos or eos > num_tokens:
                s0, e0 = min(max(bos, 0), num_tokens), min(max(eos, 0), num_tokens)
                out[s0:e0, hv] = 0
                continue
            T = eos - bos
            if T <= 0:
                continue
            acc = int(nacc[n])
            idx = [int(x) for x in sidx[n]]
            # invalid_request
            nleaf = len(leaf_rows)
            bad = (T != W + nleaf) if tree else (T > W)
            bad = bad or T > 8 or acc < 1 or acc > W
            src = -1
            if not bad:
                first, src = idx[0], idx[acc - 1]
                bad = not (0 <= first < num_slots and first != null_id and 0 <= src < num_slots and src != null_id)
                bad = bad or any(d >= num_slots and d != null_id for d in idx)
            if tree and not bad:
                bad = n >= stash.shape[0]
            if bad:
                out[bos:eos, hv] = 0
                continue
            Tc = T - nleaf if tree else T
            kh = hv // hpk
            h = state[src, hv].float().clone()  # [V, K]
            sq, sk, sv, kq, dec, beta, sdst = [], [], [], [], [], [], []
            for t in range(T):
                row = mixed[bos + t].float()
                q = row[kh * KD:(kh + 1) * KD]
                k = row[H * KD + kh * KD:H * KD + (kh + 1) * KD]
                v = row[2 * H * KD + hv * VD:2 * H * KD + (hv + 1) * VD]
                qn = torch.rsqrt((q * q).sum() + 1e-6) * scale
                kn = torch.rsqrt((k * k).sum() + 1e-6)
                sq.append(q * qn)
                sk.append(k * kn)
                sv.append(v)
                kq.append(((q * qn) * (k * kn)).sum())
                x = float(a[bos + t, hv]) + float(dt_bias[hv])
                sp = math.log1p(math.exp(x)) if x <= 20.0 else x
                dec.append(torch.tensor(math.exp(-math.exp(float(A_log[hv])) * sp), dtype=torch.float32))
                beta.append(torch.sigmoid(b[bos + t, hv].float()))
                d = idx[t] if t < W else -1
                sdst.append(d if (d >= 0 and d != null_id) else -1)
            if tree:
                for l, Rr in enumerate(leaf_rows):
                    stash[n, l, hv, :KD] = sk[Rr]
                    stash[n, l, hv, KD:KD + VD] = sv[Rr]
                    stash[n, l, hv, KD + VD] = dec[Rr]
                    stash[n, l, hv, KD + VD + 1] = beta[Rr]
            so = torch.zeros(T, VD)
            for t in range(Tc):
                if tree:
                    for l, Rr in enumerate(leaf_rows):
                        if leaf_depth[l] != t:
                            continue
                        hk, hq = h @ sk[Rr], h @ sq[Rr]
                        delta = (sv[Rr] - dec[Rr] * hk) * beta[Rr]
                        so[Rr] = f16r(dec[Rr] * hq + delta * kq[Rr])
                hk, hq = h @ sk[t], h @ sq[t]
                delta = (sv[t] - dec[t] * hk) * beta[t]
                hq = dec[t] * hq + delta * kq[t]
                h = dec[t] * h + delta[:, None] * sk[t][None, :]
                so[t] = f16r(hq)
                if sdst[t] >= 0:
                    state[sdst[t], hv] = h.to(state.dtype)
            pre[(n, hv)] = so.clone()
            for t in range(T):
                o = so[t]
                rstd = torch.rsqrt((o * o).sum() / VD + eps)
                g = z[bos + t, hv].float()
                act = torch.sigmoid(g) if sigmoid_gate else g * torch.sigmoid(g)
                out[bos + t, hv] = (((o * rstd) * norm_w.float()) * act).half()
    return pre


def k5_gating(a, b, A_log, dt_bias):
    """g (log decay) and beta exactly as the kernel derives them (fp32)."""
    x = a.float() + dt_bias.float()
    sp = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
    return -torch.exp(A_log.float()) * sp, torch.sigmoid(b.float())


def make_case(spec: TreeSpec, B=2, H=2, HV=4, nslots=None, seed=0, acc=None, state_dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    M, W = spec.M, spec.D + 1
    nslots = nslots or (B * W + 3)
    C = 2 * H * KD + HV * VD
    mixed = (torch.randn(B * M, C, generator=g) * 0.7).half()
    a = (torch.randn(B * M, HV, generator=g)).half()
    bb = (torch.randn(B * M, HV, generator=g)).half()
    A_log = torch.randn(HV, generator=g) * 0.5
    dt_bias = (torch.randn(HV, generator=g) * 0.3).half()
    state = (torch.randn(nslots, HV, VD, KD, generator=g) * 0.05).to(state_dtype)
    z = torch.randn(B * M, HV, VD, generator=g).half()
    norm_w = (1 + 0.1 * torch.randn(VD, generator=g)).half()
    perm = torch.randperm(nslots, generator=g)[: B * W].reshape(B, W).int()
    sidx = perm.contiguous()
    cu = (torch.arange(B + 1) * M).int()
    if acc is None:
        acc = torch.randint(1, W + 1, (B,), generator=g)
    nacc = torch.as_tensor(acc).int()
    return dict(mixed=mixed, a=a, b=bb, A_log=A_log, dt_bias=dt_bias, sidx=sidx, cu=cu, nacc=nacc, state=state,
                z=z, norm_w=norm_w, H=H, HV=HV, B=B, M=M, W=W)


def run_tree(spec, c, state=None):
    st = (c["state"] if state is None else state).clone()
    out = torch.full((c["B"] * c["M"], c["HV"], VD), 7.0).half()
    stash = torch.full((spec.B_tree, max(spec.L, 1), c["HV"], STASH_W), -9.0)
    pre = emulate_k5(c["mixed"], c["a"], c["b"], c["A_log"], c["dt_bias"], c["sidx"], c["cu"], c["nacc"], st,
                     c["z"], c["norm_w"], out, KD ** -0.5, 1e-6, -1, False, spec.leaf_rows, spec.leaf_depth, stash)
    return st, out, stash, pre


def ref_tree(spec, c):
    B, M, H, HV = c["B"], c["M"], c["H"], c["HV"]
    mx = c["mixed"].float().reshape(B, M, -1)
    q = mx[..., : H * KD].reshape(B, M, H, KD)
    k = mx[..., H * KD: 2 * H * KD].reshape(B, M, H, KD)
    v = mx[..., 2 * H * KD:].reshape(B, M, HV, VD)
    g, beta = k5_gating(c["a"], c["b"], c["A_log"], c["dt_bias"])
    src = torch.stack([c["state"][c["sidx"][i, c["nacc"][i] - 1]] for i in range(B)]).float()
    return R.gdn_tree_ref(spec, src, q, k, v, g.reshape(B, M, HV), beta.reshape(B, M, HV), KD ** -0.5)


SPECS = [
    TreeSpec.chain_plus_leaves(3),  # production M=7
    TreeSpec.chain_plus_leaves(3, leaf_depths=(1, 2), width=3),  # 2 leaves at depths 1 and 2, M=8
    TreeSpec.chain_plus_leaves(3, leaf_depths=(2,)),  # one leaf, M=5
    TreeSpec.chain_plus_leaves(2),  # D=2, M=5
]


# ------------------------------------------------------------------------------------------------ P4a: verify kernel
@pytest.mark.parametrize("spec", SPECS, ids=lambda s: f"D{s.D}M{s.M}L{s.L}")
def test_k5_tree_emulation_matches_reference(spec):
    c = make_case(spec, seed=spec.M)
    st, out, stash, pre = run_tree(spec, c)
    ref = ref_tree(spec, c)
    B, HV = c["B"], c["HV"]
    for n in range(B):
        # chain slots: post-state of chain row d in idx[n, d]
        for d in range(spec.D + 1):
            torch.testing.assert_close(st[c["sidx"][n, d]].float(), ref["chain_states"][n, d], atol=2e-5, rtol=1e-4)
        for hv in range(HV):
            torch.testing.assert_close(pre[(n, hv)], ref["o"][n, :, hv], atol=2e-3, rtol=2e-3)
        # stash (exact layout k | v | decay | beta)
        torch.testing.assert_close(stash[n, : spec.L], ref["leaf_stash"][n], atol=1e-6, rtol=1e-5)
    # final gated-norm output = RMSNormGated(o) * silu(z) of the reference o
    o = ref["o"].reshape(B * spec.M, HV, VD)
    y = (o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6)) * c["norm_w"].float() * F.silu(c["z"].float())
    torch.testing.assert_close(out.float(), y, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: f"D{s.D}M{s.M}L{s.L}")
def test_k5_tree_leaves_write_no_slot_and_chain_is_unchanged(spec):
    c = make_case(spec, seed=11 + spec.M, nslots=40)
    st_tree, out_tree, _, _ = run_tree(spec, c)
    # chain-only launch over the first W rows of every request (today's step)
    B, M, W = c["B"], c["M"], c["W"]
    keep = torch.cat([torch.arange(i * M, i * M + W) for i in range(B)])
    st_chain = c["state"].clone()
    out_chain = torch.zeros(B * W, c["HV"], VD).half()
    emulate_k5(c["mixed"][keep], c["a"][keep], c["b"][keep], c["A_log"], c["dt_bias"], c["sidx"],
               (torch.arange(B + 1) * W).int(), c["nacc"], st_chain, c["z"][keep], c["norm_w"], out_chain,
               KD ** -0.5, 1e-6, -1)
    assert torch.equal(st_tree, st_chain)  # identical slots, and nothing else written (leaves store no state)
    assert torch.equal(out_tree[keep], out_chain)
    touched = set(c["sidx"].flatten().tolist())
    untouched = [s for s in range(st_tree.shape[0]) if s not in touched]
    assert torch.equal(st_tree[untouched], c["state"][untouched])


def test_k5_tree_row_count_rule():
    spec = TreeSpec.chain_plus_leaves(3)
    c = make_case(spec, B=1, seed=3)
    # tree launch with T = W (chain-sized) rows -> invalid: zero output rows, no state written
    st = c["state"].clone()
    out = torch.full((spec.D + 1, c["HV"], VD), 5.0).half()
    stash = torch.zeros(spec.B_tree, spec.L, c["HV"], STASH_W)
    emulate_k5(c["mixed"][:4], c["a"][:4], c["b"][:4], c["A_log"], c["dt_bias"], c["sidx"],
               torch.tensor([0, 4]).int(), c["nacc"], st, c["z"][:4], c["norm_w"], out, KD ** -0.5, 1e-6, -1,
               False, spec.leaf_rows, spec.leaf_depth, stash)
    assert torch.equal(st, c["state"]) and not out.float().abs().sum()
    # chain launch with T = M rows -> invalid (T > width), unchanged rule
    st = c["state"].clone()
    out = torch.full((spec.M, c["HV"], VD), 5.0).half()
    emulate_k5(c["mixed"], c["a"], c["b"], c["A_log"], c["dt_bias"], c["sidx"], c["cu"], c["nacc"], st, c["z"],
               c["norm_w"], out, KD ** -0.5, 1e-6, -1)
    assert torch.equal(st, c["state"]) and not out.float().abs().sum()


def test_cu_source_matches_emulated_schedule():
    """The emulation above encodes these facts of the .cu; keep them in lock step."""
    src = K5_CU.read_text()
    assert "constexpr int kMaxTok = 8;" in src
    assert "constexpr int kStashW = kDimK + kDimV + 2;" in src
    # tree row rule and chain rule
    assert "if (T > width || T > kMaxTok || accepted < 1 || accepted > width) return true;" in src
    assert "if (T != width + nleaf || T > kMaxTok || accepted < 1 || accepted > width) return true;" in src
    # leaves are processed at the top of chain iteration t == depth, before the chain update
    loop = src[src.index("for (int t = 0; t < Tc; ++t) {"):]
    assert loop.index("lt.depth[l] != t") < loop.index("h[j][m][i] = dec * h[j][m][i] + kk[m][i] * delta")
    # stash order k | v | decay | beta from the l2-normed (unscaled) k
    assert "c < kDimK ? sk[R][c]" in src and ": c == kDimK + kDimV ? sdecay[R] : sbeta[R];" in src
    # chain launches use TREE=false; only gdn_mtp_tree passes a leaf table
    assert "gdn_mtp_sm75_v2_kernel<S, SIG, false>" in src and "nullptr, nullptr);" in src


# ------------------------------------------------------------------------------------------------ P5: commit kernel
def _flat(t: torch.Tensor) -> tuple[torch.Tensor, int]:
    """1-D view of t's whole storage, and the element offset of a data_ptr in it."""
    base = torch.empty(0, dtype=t.dtype).set_(t.untyped_storage())
    return base, t.element_size()


def emulate_commit(tc, rescue_leaf, slot_mapping):
    """tree_commit_kernel, CTA by CTA, through the int64 layer tables TreeCommit built (pointer = data_ptr)."""
    order, tq = tc._keep
    gdn_tab = tc.gdn_tab.cpu().tolist()
    tq_tab = tc.tq_tab.cpu().tolist()
    W, M, L = tc.W, tc.M, len(tc.leaf_depth)
    gstash, cstash = tc.stash.gdn_leaf, tc.stash.conv_leaf_x
    HV = gstash.shape[3]
    C = cstash.shape[3]
    for b in range(rescue_leaf.shape[0]):
        l = int(rescue_leaf[b])
        if l < 0 or l >= L:
            continue
        j = tc.leaf_depth[l]
        Rr = W + l
        for g, row in enumerate(gdn_tab):
            lay = order[g]
            ssm_flat, es = _flat(lay.ssm_state)
            assert row[0] == lay.ssm_state.data_ptr()
            off = (row[0] - lay.ssm_state.untyped_storage().data_ptr()) // es
            idx_flat, ies = _flat(lay.state_idx)
            ioff = (row[7] - lay.state_idx.untyped_storage().data_ptr()) // ies + b * W
            src, dst = int(idx_flat[ioff + j - 1]), int(idx_flat[ioff + j])
            ok = lambda s: 0 <= s < row[2] and s != -1
            if ok(src) and ok(dst):
                for hv in range(HV):
                    so = off + src * row[1] + hv * VD * KD
                    do = off + dst * row[1] + hv * VD * KD
                    h = ssm_flat[so:so + VD * KD].float().reshape(VD, KD)
                    sp = gstash[g, b, l, hv]
                    k, v, dec, beta = sp[:KD], sp[KD:KD + VD], sp[KD + VD], sp[KD + VD + 1]
                    delta = (v - dec * (h @ k)) * beta
                    ssm_flat[do:do + VD * KD] = (dec * h + delta[:, None] * k[None, :]).reshape(-1).to(ssm_flat.dtype)
            slot0 = int(idx_flat[ioff])
            if ok(slot0):
                conv_flat, ces = _flat(lay.conv_state)
                coff = (row[3] - lay.conv_state.untyped_storage().data_ptr()) // ces
                col = tc.conv_len - W + j
                for ch in range(C):
                    conv_flat[coff + slot0 * row[4] + ch * row[5] + col * row[6]] = cstash[g, b, l, ch]
        for t, row in enumerate(tq_tab):
            kv = tq[t]
            flat, _ = _flat(kv)
            koff = row[0] - kv.untyped_storage().data_ptr()
            sdst, ssrc = int(slot_mapping[b * M + j]), int(slot_mapping[b * M + Rr])
            cap = row[4] * row[5]
            if not (0 <= sdst < cap and 0 <= ssrc < cap):
                continue
            s_at = koff + (ssrc // row[4]) * row[1] + (ssrc % row[4]) * row[2]
            d_at = koff + (sdst // row[4]) * row[1] + (sdst % row[4]) * row[2]
            for hh in range(tc.Hk):
                flat[d_at + hh * row[3]: d_at + hh * row[3] + tc.slot_bytes] = \
                    flat[s_at + hh * row[3]: s_at + hh * row[3] + tc.slot_bytes].clone()


def _tq_cache(nblocks, Hk, N, Cb, gen, layout="LBNHC"):
    """Logical [blocks, Hk, N, C] uint8 view, physically [blocks, N, Hk, C] (as the TQ backend sees it)."""
    phys = torch.randint(0, 256, (nblocks, N, Hk, Cb), generator=gen, dtype=torch.uint8)
    return phys.transpose(1, 2)


def build_commit_world(spec, B=2, NG=3, HV=4, Cc=96, Kw=4, NT=2, Hk=2, Cb=40, seed=0, sd_layout=True):
    gen = torch.Generator().manual_seed(seed)
    W, M = spec.D + 1, spec.M
    conv_len = (Kw - 1) + spec.D
    nslots = B * W + 5
    stash = TreeStash.allocate(spec, [f"l{i}" for i in range(NG)], HV, KD, VD, Cc, torch.float16, "cpu")
    stash.gdn_leaf.copy_(torch.randn(stash.gdn_leaf.shape, generator=gen))
    stash.gdn_leaf[..., KD + VD] = torch.rand(stash.gdn_leaf.shape[:-1], generator=gen) * 0.9 + 0.05
    stash.gdn_leaf[..., KD + VD + 1] = torch.rand(stash.gdn_leaf.shape[:-1], generator=gen)
    stash.conv_leaf_x.copy_(torch.randn(stash.conv_leaf_x.shape, generator=gen).half())
    idx_buf = torch.full((8, W), -1, dtype=torch.int32)  # persistent builder buffer, first B rows live
    idx_buf[:B] = torch.randperm(nslots, generator=gen)[: B * W].reshape(B, W).int()
    layers = []
    for i in range(NG):
        ssm = torch.randn(nslots, HV, VD, KD, generator=gen) * 0.05
        if sd_layout:  # default SD: [slots, state_len, dim]; glue view = transpose -> [slots, C, len]
            conv = torch.randn(nslots, conv_len, Cc, generator=gen).half().transpose(-1, -2)
        else:
            conv = torch.randn(nslots, Cc, conv_len, generator=gen).half()
        from vllm.v1.spec_decode.tree_commit import GDNCommitLayer

        layers.append(GDNCommitLayer(f"l{i}", ssm, conv, idx_buf))
    tq = [_tq_cache(6, Hk, 16, Cb, gen) for _ in range(NT)]
    return stash, layers[::-1], tq, idx_buf, conv_len  # reversed: TreeCommit must order by stash.layer_index


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: f"D{s.D}M{s.M}L{s.L}")
@pytest.mark.parametrize("sd_layout", [True, False], ids=["SD", "DS"])
def test_commit_emulation_matches_references(spec, sd_layout):
    from vllm.v1.spec_decode.tree_commit import TreeCommit

    B = 3
    stash, layers, tq, idx_buf, conv_len = build_commit_world(spec, B=B, seed=spec.M, sd_layout=sd_layout)
    tc = TreeCommit(spec, stash, layers, tq, "cpu")
    W, M, Kw = spec.D + 1, spec.M, conv_len - spec.D + 1
    rescue = torch.tensor([spec.L - 1, -1, 0], dtype=torch.int32)
    gen = torch.Generator().manual_seed(5)
    slot_mapping = torch.randperm(6 * 16, generator=gen)[: B * M].long()
    slot_mapping[M + 2] = -1  # a padded row in a non-rescued request: harmless
    before_ssm = {x.name: x.ssm_state.clone() for x in layers}
    before_conv = {x.name: x.conv_state.clone() for x in layers}
    before_tq = [kv.clone() for kv in tq]
    emulate_commit(tc, rescue, slot_mapping)
    for x in layers:
        g = stash.layer_index[x.name]
        chain = torch.stack([before_ssm[x.name][idx_buf[b]] for b in range(B)])  # [B, W, HV, V, K]
        exp = R.gdn_commit_ref(spec, chain, stash.gdn_leaf[g, :B], rescue)
        got = torch.stack([x.ssm_state[idx_buf[b]] for b in range(B)])
        torch.testing.assert_close(got, exp, atol=1e-6, rtol=1e-5)
        win = torch.stack([before_conv[x.name][idx_buf[b, 0]].t() for b in range(B)]).float()  # [B, len, C]
        exp_w = R.conv_commit_ref(spec, win, stash.conv_leaf_x[g, :B], rescue, Kw)
        got_w = torch.stack([x.conv_state[idx_buf[b, 0]].t() for b in range(B)]).float()
        assert torch.equal(got_w, exp_w)
        # nothing outside the request's slots changed
        live = set(idx_buf[:B].flatten().tolist())
        dead = [s for s in range(x.ssm_state.shape[0]) if s not in live]
        assert torch.equal(x.ssm_state[dead], before_ssm[x.name][dead])
    # TQ: chain view rows (identity, [b, j] = leaf row) of the packed rows == rows after the commit
    cvr = torch.arange(W).repeat(B, 1)
    for b in range(B):
        if rescue[b] >= 0:
            cvr[b, spec.leaf_depth[int(rescue[b])]] = W + int(rescue[b])
    for kv, kv0 in zip(tq, before_tq):
        rows = lambda c: torch.stack([
            torch.stack([c[s // 16, :, s % 16] if s >= 0 else torch.zeros_like(c[0, :, 0])
                         for s in slot_mapping[b * M:(b + 1) * M].tolist()]) for b in range(B)])
        exp = R.gather_chain_view(rows(kv0), cvr)
        got = rows(kv)[:, :W]
        assert torch.equal(got, exp)
        # only rescued requests' row-j slots changed
        changed = (kv != kv0).any(-1).any(1)  # [blocks, N]
        allowed = {int(slot_mapping[b * M + spec.leaf_depth[int(rescue[b])]]) for b in range(B) if rescue[b] >= 0}
        for blk, pos in changed.nonzero().tolist():
            assert blk * 16 + pos in allowed


@pytest.mark.parametrize("spec", SPECS[:3], ids=lambda s: f"D{s.D}M{s.M}L{s.L}")
def test_verify_commit_then_next_step_equals_chain_run_over_accepted_path(spec):
    """The P5 gate on CPU: verify (emulated K5 tree) -> rescue leaf l -> commit (emulated) -> the next step's
    source slot idx[acc-1], acc = j+1, holds the state of a from-scratch chain run over [x0, c1..c_{j-1}, leaf]."""
    from vllm.v1.spec_decode.tree_commit import GDNCommitLayer, TreeCommit

    B = 1
    for l in range(spec.L):
        c = make_case(spec, B=B, seed=100 + l, acc=[2])
        st, out, stash_view, _ = run_tree(spec, c)
        stash = TreeStash.allocate(spec, ["g"], c["HV"], KD, VD, 8, torch.float16, "cpu")
        stash.gdn_leaf[0].copy_(stash_view)
        conv = torch.zeros(st.shape[0], 8, (4 - 1) + spec.D).half()
        idx_buf = c["sidx"].clone()
        tc = TreeCommit(spec, stash, [GDNCommitLayer("g", st, conv, idx_buf)], [], "cpu")
        emulate_commit(tc, torch.tensor([l], dtype=torch.int32), torch.zeros(spec.M, dtype=torch.long))
        j = spec.leaf_depth[l]
        path = list(range(j)) + [spec.leaf_rows[l]]
        # from-scratch chain run (emulated chain kernel) over the accepted path
        st2 = c["state"].clone()
        rows = torch.tensor(path)
        out2 = torch.zeros(len(path), c["HV"], VD).half()
        emulate_k5(c["mixed"][rows], c["a"][rows], c["b"][rows], c["A_log"], c["dt_bias"], c["sidx"],
                   torch.tensor([0, len(path)]).int(), c["nacc"], st2, c["z"][rows], c["norm_w"], out2,
                   KD ** -0.5, 1e-6, -1)
        nxt = int(c["sidx"][0, j])  # acc = j + 1 -> next step reads idx[j]
        torch.testing.assert_close(st[nxt], st2[nxt], atol=1e-5, rtol=1e-5)
        # and the leaf's output from the verify equals the chain run's last output
        torch.testing.assert_close(out[spec.leaf_rows[l]].float(), out2[-1].float(), atol=2e-3, rtol=2e-3)


def test_commit_validates_layers_and_orders_by_stash_index():
    from vllm.v1.spec_decode.tree_commit import GDNCommitLayer, TreeCommit

    spec = TreeSpec.chain_plus_leaves(3)
    stash, layers, tq, idx_buf, _ = build_commit_world(spec)
    tc = TreeCommit(spec, stash, layers, tq, "cpu")
    assert [x.name for x in tc._keep[0]] == ["l0", "l1", "l2"]
    assert tc.gdn_tab.shape == (3, 8) and tc.tq_tab.shape == (2, 6)
    row = tc.tq_tab[0].tolist()
    kv = tc._keep[1][0]
    assert row == [kv.data_ptr(), kv.stride(0), kv.stride(2), kv.stride(1), kv.size(2), kv.size(0)]
    with pytest.raises(ValueError):
        TreeCommit(spec, stash, layers[:2], tq, "cpu")  # must cover every stash layer
    bad = GDNCommitLayer("l0", layers[0].ssm_state, layers[0].conv_state, idx_buf.long())
    with pytest.raises(ValueError):
        TreeCommit(spec, stash, [bad] + layers[1:], tq, "cpu")


def test_commit_call_creates_no_tensors(monkeypatch):
    """Graph-safety: commit() only launches; the tables were built in __init__."""
    from vllm.v1.spec_decode import tree_commit as TC

    spec = TreeSpec.chain_plus_leaves(3)
    stash, layers, tq, _, _ = build_commit_world(spec)
    tc = TC.TreeCommit(spec, stash, layers, tq, "cpu")
    calls = []
    monkeypatch.setattr(TC, "_load", lambda: SimpleNamespace(tree_commit=lambda *a: calls.append(a)))
    for name in ("empty", "zeros", "tensor", "full", "ones", "arange", "as_tensor"):
        monkeypatch.setattr(torch, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError("alloc in commit")))
    out = TreeStepOutputs(
        output_token_ids=None, num_accepted=None,
        rescue_leaf=_I32_2,
        rescue_depth=None, chain_view_rows=None)
    tc.commit_step(out, _SM)
    (args,) = calls
    assert args[1] is tc.gdn_tab and args[4] is tc.tq_tab and args[6] == list(spec.leaf_depth)
    assert (args[7], args[8], args[12], args[13]) == (spec.D + 1, spec.M, -1, True)


_I32_2 = torch.tensor([0, -1], dtype=torch.int32)
_SM = torch.arange(14, dtype=torch.long)


# ------------------------------------------------------------------------------------------------ glue / metadata
GLUE = ROOT / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"


def _method_src(path: Path, cls: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == name:
                    return f
    raise AssertionError(name)


def test_glue_creates_no_layer_state():
    """K5 rule: nothing on the K5 path (chain or tree) assigns layer attributes (no lazy params / tables)."""
    for name in ("_k5_spec_decode", "_k5_spec_decode_md", "_k5_tree_conv"):
        fn = _method_src(GLUE, "QwenGatedDeltaNetAttention", name)
        for node in ast.walk(fn):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    for sub in ast.walk(t):
                        assert not (isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name)
                                    and sub.value.id == "self"), f"{name} assigns self.{sub.attr}"


def test_glue_tree_dispatch_and_fallback(monkeypatch):
    """Chain metadata -> gdn_mtp with today's arguments; tree metadata -> gdn_mtp_tree with the leaf tables and
    this layer's stash view; a tree step the K5 path cannot take raises instead of falling back."""
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as Q
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    calls = []

    def fake_md(self, md, *a):
        calls.append(md.tree_rows)
        return md.tree_rows != 99

    monkeypatch.setattr(Q.QwenGatedDeltaNetAttention, "_k5_spec_decode_md", fake_md)
    layer = Q.QwenGatedDeltaNetAttention.__new__(Q.QwenGatedDeltaNetAttention)
    layer.prefix = "x"
    mk = lambda rows: GDNAttentionMetadata(0, 0, 0, 0, 1, 4, 4, tree_rows=rows)
    for rows, expect in ((0, True), (7, True)):
        monkeypatch.setattr(Q, "get_forward_context",
                            lambda r=rows: SimpleNamespace(attn_metadata={"x": mk(r)}))
        assert layer._k5_spec_decode(None, None, None) is expect
    monkeypatch.setattr(Q, "get_forward_context", lambda: SimpleNamespace(attn_metadata={"x": mk(99)}))
    with pytest.raises(RuntimeError):
        layer._k5_spec_decode(None, None, None)
    assert calls == [0, 7, 99]
    assert Q.QwenGatedDeltaNetAttention.tree_stash is None  # class default; set once by the runner
    src = GLUE.read_text()
    assert "md.tree_spec.leaf_rows,\n                md.tree_spec.leaf_depth,\n" \
           "                self.tree_stash.gdn_view(self.prefix)," in src
    assert "rows_per_request=tree_rows or None" in src


def test_preflight_rows_rule():
    from vllm.model_executor.layers.mamba.gdn import gdn_sm75_cuda as K

    src = Path(K.__file__).read_text()
    assert "rows = width if rows_per_request is None else rows_per_request" in src
    assert "if not width <= rows <= 8 or num_tokens > num_requests * rows:" in src


def _builder(spec, monkeypatch, full_graph=True):
    from vllm.v1.attention.backends import gdn_attn as G

    bld = G.GDNAttentionMetadataBuilder.__new__(G.GDNAttentionMetadataBuilder)
    bld.num_spec = 3
    bld.use_spec_decode = True
    bld.use_full_cuda_graph = full_graph
    bld.tree_spec = spec
    bld.tree_tensors = spec.device_tensors("cpu") if spec else None
    bld.decode_cudagraph_max_bs = 64
    bld.kv_cache_spec = SimpleNamespace()
    bld.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode="align"))
    bld.gdn_prefill_backend = "triton"
    bld._dflash_trace_step = 0
    for n, shape, dt in (("spec_state_indices_tensor", (64, 4), torch.int32),
                         ("non_spec_state_indices_tensor", (64,), torch.int32),
                         ("spec_sequence_masks", (64,), torch.bool),
                         ("spec_token_indx", (256,), torch.int32), ("non_spec_token_indx", (256,), torch.int32),
                         ("spec_query_start_loc", (65,), torch.int32),
                         ("non_spec_query_start_loc", (65,), torch.int32),
                         ("num_accepted_tokens", (64,), torch.int32)):
        setattr(bld, n, torch.full(shape, -7, dtype=dt))
    monkeypatch.setattr(G, "mamba_get_block_table_tensor", lambda bt, *a: bt)
    monkeypatch.setattr(G, "async_tensor_h2d", lambda t, device=None: t.clone())
    return bld


def _cam(B, q, tree):
    qsl = (torch.arange(B + 1) * q).int()
    bt = (torch.arange(B * 8).reshape(B, 8) + 100).int()
    return SimpleNamespace(query_start_loc=qsl, query_start_loc_cpu=qsl.clone(), block_table_tensor=bt,
                           seq_lens=torch.full((B,), 50), num_reqs=B, num_actual_tokens=B * q,
                           slot_mapping=None, tree_step=tree)


def test_builder_chain_unchanged_and_tree_fields(monkeypatch):
    spec = TreeSpec.chain_plus_leaves(3)
    B = 3
    # chain step: tree on but tree_step False == tree off, field for field
    on = _builder(spec, monkeypatch)
    off = _builder(None, monkeypatch)
    acc = torch.tensor([1, 4, 2], dtype=torch.int32)
    drafts = torch.full((B,), 3)
    m_on = on.build(0, _cam(B, 4, False), acc, drafts)
    m_off = off.build(0, _cam(B, 4, False), acc, drafts)
    for f in m_off.__dataclass_fields__:
        x, y = getattr(m_on, f), getattr(m_off, f)
        assert (torch.equal(x, y) if isinstance(x, torch.Tensor) else x == y), f
    assert m_on.tree is None and m_on.tree_rows == 0 and m_on.tree_spec is None
    # tree step: M rows per request, slots stay [B, D+1], persistent buffer filled even without full graphs
    tr = _builder(spec, monkeypatch, full_graph=False)
    m = tr.build(0, _cam(B, spec.M, True), acc, torch.full((B,), spec.M - 1))
    assert m.tree_rows == spec.M and m.tree is tr.tree_tensors and m.tree_spec is spec
    assert m.num_spec_decodes == B and m.num_spec_decode_tokens == B * spec.M
    assert m.spec_query_start_loc.tolist() == [0, 7, 14, 21]
    assert m.spec_state_indices_tensor.shape == (B, 4)
    assert m.spec_state_indices_tensor.data_ptr() == tr.spec_state_indices_tensor.data_ptr()
    assert torch.equal(tr.spec_state_indices_tensor[:B], (torch.arange(B * 8).reshape(B, 8) + 100)[:, :4].int())
    assert m.spec_token_indx.numel() == B * spec.M
    assert m.num_accepted_tokens.tolist() == acc.tolist()
    # a chain-sized batch flagged as tree in the capture path clamps accepted to D+1
    cap = _builder(spec, monkeypatch).build_for_cudagraph_capture(_cam(2, spec.M, True))
    assert cap.num_accepted_tokens.tolist() == [4, 4]


def test_builder_rejects_tree_without_k5(monkeypatch):
    from vllm.v1.attention.backends import gdn_attn as G

    src = Path(G.__file__).read_text()
    assert "speculative_token_tree needs VLLM_K5_GDN_FUSED=1" in src
    assert "(self.use_full_cuda_graph or tree_step)" in src


def test_runner_commit_hook_runs_before_align_postprocess():
    src = (ROOT / "vllm/v1/worker/gpu_model_runner.py").read_text()
    i = src.index('tree_commit = getattr(self, "tree_commit", None)')
    j = src.index("mamba_utils.postprocess_mamba_align_gpu(", i)
    assert 0 < i < j
    assert 'if tree_commit is not None and getattr(scheduler_output, "tree_step", False):' in src


# ------------------------------------------------------------------------------------------------ nvcc (sm_75)
def _nvcc_env():
    nvcc = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else None)
    if nvcc is None:
        pytest.skip("nvcc not found")
    try:
        import sysconfig

        from torch.utils.cpp_extension import include_paths
        inc = [f"-I{p}" for p in include_paths()] + [f"-I{sysconfig.get_paths()['include']}"]
    except Exception as e:  # pragma: no cover
        pytest.skip(f"torch headers unavailable: {e}")
    return nvcc, inc


def _nvcc(src: Path, out: Path, kind: str, name: str):
    nvcc, inc = _nvcc_env()
    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    cmd = [nvcc, "-std=c++17", "-O3", "-gencode=arch=compute_75,code=sm_75", f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
           f"-DTORCH_EXTENSION_NAME={name}", "-DTORCH_API_INCLUDE_EXTENSION_H", "--compiler-options", "-fPIC",
           *inc, kind, str(src), "-o", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-4000:]


def _sass(cubin: Path) -> dict[tuple, list[str]]:
    cuobjdump = Path(shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc").with_name("cuobjdump")
    txt = subprocess.run([str(cuobjdump), "-sass", str(cubin)], capture_output=True, text=True, check=True).stdout
    out, cur = {}, None
    for line in txt.splitlines():
        m = re.match(r"\s+Function : (\S+)", line)
        if m:
            k = re.search(r"(gdn_mtp_sm75_(?:v2_)?kernel)I(6__half|f)Lb(\d)E(?:Lb(\d))?", m.group(1))
            cur = k.groups() if k else None
            if cur:
                out[cur] = []
            continue
        if cur:
            m2 = re.search(r"/\*[0-9a-f]{4}\*/\s+(.*?)\s*;", line)
            if m2:
                out[cur].append(m2.group(1))
    return out


@pytest.mark.skipif(os.environ.get("TREE_SKIP_NVCC") == "1", reason="TREE_SKIP_NVCC=1")
def test_nvcc_sm75_compiles_and_chain_sass_unchanged(tmp_path):
    _nvcc(COMMIT_CU, tmp_path / "commit.o", "-c", "tree_commit_sm75")
    _nvcc(K5_CU, tmp_path / "new.cubin", "-cubin", "gdn_mtp_sm75")
    base = subprocess.run(["git", "-C", str(ROOT), "show", f"{K5_BASE_COMMIT}:{K5_CU.relative_to(ROOT)}"],
                          capture_output=True, text=True)
    if base.returncode != 0:
        pytest.skip("K5 base commit not reachable")
    (tmp_path / "base.cu").write_text(base.stdout)
    _nvcc(tmp_path / "base.cu", tmp_path / "base.cubin", "-cubin", "gdn_mtp_sm75")
    old, new = _sass(tmp_path / "base.cubin"), _sass(tmp_path / "new.cubin")
    assert len(old) == 8
    for k, v in old.items():
        k_new = k if k[0] == "gdn_mtp_sm75_kernel" else (k[0], k[1], k[2], "0")
        assert new[k_new] == v, f"chain kernel {k} changed"
    assert sum(1 for k in new if k[3] == "1") == 4  # tree instantiations: {fp32, fp16} x {silu, sigmoid}
