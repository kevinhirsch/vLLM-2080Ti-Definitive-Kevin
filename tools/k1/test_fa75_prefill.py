"""Lane K1 numerical gate: K1 kernel vs fp32 reference vs the production FlashInfer ragged prefill.

Gate: normwise rel err of O < 1e-3 vs FlashInfer and vs fp32 ref; LSE abs err < 1e-3 (natural log).
Run on a card shared with the live engine: K1_CAP_MB bounds tensor memory (default 260 MiB).
"""

from __future__ import annotations

import itertools
import json
import math
import sys

import torch

from k1_common import K1, flashinfer_run, gpu_guard, preflight, ref_attention, ref_attention_long, smi_free


def rel(a, b):
    b = b.float()
    d = a.float()
    d.sub_(b)
    n = d.norm().item()
    mx = d.abs_().max().item()
    del d
    return n / max(b.norm().item(), 1e-30), mx / max(b.abs().max().item(), 1e-30)


def main():
    torch.cuda.set_device(0)
    gpu_guard(0)
    print("smi:", smi_free(), flush=True)
    dev = "cuda"
    scale = 256**-0.5
    cases = [
        # (Tq, Tkv, Hq, Hk, causal, q_mult)
        (37, 37, 12, 2, True, 1.0),
        (256, 256, 12, 2, True, 1.0),
        (300, 1000, 12, 2, True, 1.0),
        (1000, 1000, 12, 2, True, 4.0),
        (64, 5000, 12, 2, True, 4.0),
        (512, 4096, 12, 2, True, 1.0),
        (512, 4096, 12, 2, False, 4.0),
        (1, 700, 12, 2, True, 1.0),
        (3632, 3632, 6, 1, True, 4.0),
        (2048, 9000, 6, 1, True, 8.0),
        (777, 3001, 6, 1, False, 1.0),
    ]
    variants = [0, 1, 2, 3, 4, 5, 6, 7]
    pc_only = "--pc-only" in sys.argv
    if pc_only:
        cases = []
    results = []
    worst = {vv: [0.0, 0.0, 0.0] for vv in variants}
    for Tq, Tkv, Hq, Hk, causal, qm in cases:
        preflight()
        g = torch.Generator(device=dev).manual_seed(Tq * 7 + Tkv)
        q = (torch.randn(Tq, Hq, 256, device=dev, generator=g) * qm).half()
        k = torch.randn(Tkv, Hk, 256, device=dev, generator=g).half()
        v = torch.randn(Tkv, Hk, 256, device=dev, generator=g).half()
        ref_o, ref_lse = ref_attention(q, k, v, scale, causal)
        _, fi = flashinfer_run(q, k, v, scale, causal, return_lse=True)
        fo, flse = fi()
        fi_rel = rel(fo, ref_o)
        # FlashInfer LSE convention: compare in both bases
        fl_nat = (flse.float() - ref_lse).abs().max().item()
        fl_b2 = (flse.float() * math.log(2) - ref_lse).abs().max().item()
        row = dict(Tq=Tq, Tkv=Tkv, Hq=Hq, Hk=Hk, causal=causal, q_mult=qm, fi_vs_ref=fi_rel,
                   fi_lse_err_if_natural=fl_nat, fi_lse_err_if_base2=fl_b2)
        for vv in variants:
            for bn in (16,):
                o, lse = K1.fa75_prefill(q, k, v, scale=scale, causal=causal, return_lse=True, variant=vv, bn=bn)
                torch.cuda.synchronize()
                r_ref = rel(o, ref_o)
                r_fi = rel(o, fo)
                lerr = (lse - ref_lse).abs().max().item()
                row[f"v{vv}_bn{bn}"] = dict(vs_ref=r_ref, vs_fi=r_fi, lse_err=lerr, finite=bool(torch.isfinite(o).all()))
                del o, lse
                if bn == 16:
                    w = worst[vv]
                    w[0] = max(w[0], r_ref[0]); w[1] = max(w[1], r_fi[0]); w[2] = max(w[2], lerr)
        results.append(row)
        print(json.dumps(row), flush=True)
        del q, k, v, ref_o, ref_lse, fo, flse
        torch.cuda.empty_cache()

    if pc_only:
        return prefix_combine(dev, scale)
    # strided q (head slice of a wider projection), strided o, lse into an [H, T] buffer
    g = torch.Generator(device=dev).manual_seed(5)
    qkv = torch.randn(333, 16, 256, device=dev, generator=g).half()
    q = qkv[:, :12]
    k = qkv[:, 12:14]
    v = qkv[:, 14:16]
    ref_o, ref_lse = ref_attention(q, k, v, scale, True)
    obuf = torch.zeros(333, 20, 256, device=dev, dtype=torch.half)
    lse_ht = torch.zeros(12, 333, device=dev)
    K1.fa75_prefill(q, k, v, scale=scale, causal=True, out=obuf[:, 4:16], lse=lse_ht.transpose(0, 1))
    strided = dict(vs_ref=rel(obuf[:, 4:16], ref_o), lse_err=(lse_ht.T - ref_lse).abs().max().item(),
                   untouched=bool((obuf[:, :4] == 0).all() and (obuf[:, 16:] == 0).all()))
    print("strided:", json.dumps(strided), flush=True)

    # varlen batch: 3 requests, mixed first-chunk and continuation
    lens = [(200, 200), (64, 900), (513, 1500)]
    qs, ks, vs = [], [], []
    for i, (tq, tk) in enumerate(lens):
        g = torch.Generator(device=dev).manual_seed(100 + i)
        qs.append((torch.randn(tq, 12, 256, device=dev, generator=g) * 3).half())
        ks.append(torch.randn(tk, 2, 256, device=dev, generator=g).half())
        vs.append(torch.randn(tk, 2, 256, device=dev, generator=g).half())
    cq = torch.tensor([0] + list(itertools.accumulate(t for t, _ in lens)), dtype=torch.int32, device=dev)
    ck = torch.tensor([0] + list(itertools.accumulate(t for _, t in lens)), dtype=torch.int32, device=dev)
    o, lse = K1.fa75_prefill(torch.cat(qs), torch.cat(ks), torch.cat(vs), scale=scale, causal=True, return_lse=True,
                             cu_seqlens_q=cq, cu_seqlens_k=ck, max_seqlen_q=max(t for t, _ in lens))
    vl = []
    for i, (tq, tk) in enumerate(lens):
        ro, rl = ref_attention(qs[i], ks[i], vs[i], scale, True)
        vl.append(dict(vs_ref=rel(o[cq[i]:cq[i + 1]], ro), lse_err=(lse[cq[i]:cq[i + 1]] - rl).abs().max().item()))
    print("varlen:", json.dumps(vl), flush=True)
    del qs, ks, vs, o, lse, qkv, q, k, v, obuf, lse_ht, ref_o, ref_lse
    torch.cuda.empty_cache()
    # segmented context (exact LSE merge in the kernel epilogue): pieces [0,a) [a,b) non-causal, [b, Tkv) causal last.
    # Valid only when every query row sees all keys of the non-causal pieces: b <= Tkv - Tq (true in the engine:
    # those pieces are cached rows, the chunk comes last)
    seg = []
    for Tq, Tkv, cuts, qm in ((1000, 9000, (3000, 6000), 4.0), (3632, 7300, (1856, 3584), 1.0), (64, 20000, (7136, 14272), 8.0)):
        g = torch.Generator(device=dev).manual_seed(Tkv)
        q = torch.randn(Tq, 6, 256, device=dev, generator=g, dtype=torch.half).mul_(qm)
        k = torch.randn(Tkv, 1, 256, device=dev, generator=g, dtype=torch.half)
        v = torch.randn(Tkv, 1, 256, device=dev, generator=g, dtype=torch.half)
        ref_o, ref_lse = ref_attention(q, k, v, scale, True)
        acc_o = torch.empty(Tq, 6, 256, device=dev)
        acc_l = torch.empty(Tq, 6, device=dev)
        out = torch.empty_like(q)
        lse_o = torch.empty(Tq, 6, device=dev)
        a, b = cuts
        K1.fa75_prefill(q, k[:a], v[:a], scale=scale, causal=False, out=out, acc_o=acc_o, acc_lse=acc_l, acc_mode=1)
        K1.fa75_prefill(q, k[a:b], v[a:b], scale=scale, causal=False, out=out, acc_o=acc_o, acc_lse=acc_l, acc_mode=2)
        K1.fa75_prefill(q, k[b:], v[b:], scale=scale, causal=True, out=out, lse=lse_o, acc_o=acc_o, acc_lse=acc_l,
                        acc_mode=3)
        one = K1.fa75_prefill(q, k, v, scale=scale, causal=True, nsplit=1)
        # the same pieces with split-KV parts + combine kernel on every piece
        out2 = torch.empty_like(q)
        lse2 = torch.empty(Tq, 6, device=dev)
        K1.fa75_prefill(q, k[:a], v[:a], scale=scale, causal=False, out=out2, acc_o=acc_o, acc_lse=acc_l, acc_mode=1, nsplit=3)
        K1.fa75_prefill(q, k[a:b], v[a:b], scale=scale, causal=False, out=out2, acc_o=acc_o, acc_lse=acc_l, acc_mode=2, nsplit=2)
        K1.fa75_prefill(q, k[b:], v[b:], scale=scale, causal=True, out=out2, lse=lse2, acc_o=acc_o, acc_lse=acc_l,
                        acc_mode=3, nsplit=4)
        sp = [rel(K1.fa75_prefill(q, k, v, scale=scale, causal=True, nsplit=S), ref_o)[0] for S in (2, 3, 4)]
        r = dict(Tq=Tq, Tkv=Tkv, cuts=cuts, seg_vs_ref=rel(out, ref_o), seg_vs_single=rel(out, one),
                 lse_err=max((lse_o - ref_lse).abs().max().item(), (lse2 - ref_lse).abs().max().item()),
                 seg_split_vs_ref=rel(out2, ref_o)[0], split_vs_ref=sp,
                 finite=bool(torch.isfinite(out).all() and torch.isfinite(out2).all()))
        seg.append(r)
        print("segmented:", json.dumps(r), flush=True)
        del q, k, v, ref_o, ref_lse, acc_o, acc_l, out, lse_o, one, out2, lse2
        torch.cuda.empty_cache()

    # long context error growth (K9 saw fp32-P.V + TAU 8 drift to 9e-4 at 131K): real per-rank GQA-6, 128 rows
    longc = []
    for Tkv, qm in ((32768, 2.0), (131072, 2.0), (131072, 6.0)):
        g = torch.Generator(device=dev).manual_seed(Tkv + int(qm))
        q = torch.randn(128, 6, 256, device=dev, generator=g, dtype=torch.half).mul_(qm)
        k = torch.randn(Tkv, 1, 256, device=dev, generator=g, dtype=torch.half)
        v = torch.randn(Tkv, 1, 256, device=dev, generator=g, dtype=torch.half)
        ref_o, ref_lse = ref_attention_long(q, k, v, scale, True)
        r = dict(Tkv=Tkv, q_mult=qm)
        for vv in (0, 2, 3, 7, 8, 9):
            o, lse = K1.fa75_prefill(q, k, v, scale=scale, causal=True, return_lse=True, variant=vv)
            r[f"v{vv}"] = (round(rel(o, ref_o)[0], 6), round((lse - ref_lse).abs().max().item(), 6))
            del o, lse
        longc.append(r)
        print("long:", json.dumps(r), flush=True)
        del q, k, v, ref_o, ref_lse
        torch.cuda.empty_cache()

    print("WORST normwise rel err (vs_ref, vs_fi, lse_abs) per variant bn16:", json.dumps(worst))
    ok = all(w[0] < 1e-3 and w[1] < 1e-3 and w[2] < 1e-3 for vv, w in worst.items())
    ok = ok and all(x["seg_vs_ref"][0] < 1e-3 and x["lse_err"] < 1e-3 and x["finite"] and x["seg_split_vs_ref"] < 1e-3
                    and max(x["split_vs_ref"]) < 1e-3 for x in seg)
    ok = ok and all(x[f"v{vv}"][0] < 1e-3 for x in longc for vv in (3, 7))
    ok = ok and strided["vs_ref"][0] < 1e-3 and strided["untouched"] and all(x["vs_ref"][0] < 1e-3 for x in vl)
    print("GATE (all variants, strided, varlen, segmented):", "PASS" if ok else "FAIL")
    print("smi:", smi_free())
    return 0 if ok else 1


def prefix_combine(dev, scale):
    # Production prefix-combine path (turboquant_attn._continuation_prefill, seq_len >= 20480 in "auto"):
    # FlashInfer non-causal over the cached prefix + causal over the chunk, merged with merge_attn_states' rule
    # (natural exp of the LSE). Emulate it with FlashInfer's own LSE, and with K1's LSE, against the fp32 reference.
    pc = []
    for cached, tq, qm in ((20480, 1024, 1.0), (20480, 1024, 4.0), (22000, 1024, 8.0)):
        g = torch.Generator(device=dev).manual_seed(cached + tq)
        q = (torch.randn(tq, 6, 256, device=dev, generator=g) * qm).half()
        k = torch.randn(cached + tq, 1, 256, device=dev, generator=g).half()
        v = torch.randn(cached + tq, 1, 256, device=dev, generator=g).half()
        ref_o, _ = ref_attention(q, k, v, scale, True)

        def merge(po, pl, so, sl):  # vllm merge_attn_states rule (pl/sl [T, H])
            m = torch.maximum(pl, sl)
            pe, se = torch.exp(pl - m), torch.exp(sl - m)
            return (po.float() * (pe / (pe + se))[..., None] + so.float() * (se / (pe + se))[..., None]).half()

        _, fp = flashinfer_run(q, k[:cached], v[:cached], scale, False, return_lse=True)
        po, pl = fp()
        _, fs = flashinfer_run(q, k[cached:], v[cached:], scale, True, return_lse=True)
        so, sl = fs()
        prod_merge = merge(po, pl.float(), so, sl.float())
        fixed_merge = merge(po, pl.float() * math.log(2), so, sl.float() * math.log(2))
        ko, kl = K1.fa75_prefill(q, k[:cached], v[:cached], scale=scale, causal=False, return_lse=True)
        ks_, ksl = K1.fa75_prefill(q, k[cached:], v[cached:], scale=scale, causal=True, return_lse=True)
        k1_merge = merge(ko, kl, ks_, ksl)
        k1_single = K1.fa75_prefill(q, k, v, scale=scale, causal=True)
        r = dict(cached=cached, q_len=tq, q_mult=qm,
                 prod_fi_lse_as_is=rel(prod_merge, ref_o), fi_lse_times_ln2=rel(fixed_merge, ref_o),
                 k1_lse_merge=rel(k1_merge, ref_o), k1_single_call=rel(k1_single, ref_o))
        pc.append(r)
        print("prefix_combine:", json.dumps(r), flush=True)
        del q, k, v, ref_o, po, so, ko, ks_, pl, sl, kl, ksl, prod_merge, fixed_merge, k1_merge, k1_single
        torch.cuda.empty_cache()

    print("smi:", smi_free())
    return 0


if __name__ == "__main__":
    sys.exit(main())
