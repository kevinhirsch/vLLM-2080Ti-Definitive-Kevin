"""Lane E04 GPU numerics gate (run INSIDE a window, engine down): tq_gqa stage-1 C3 layout vs the shipped layout.

The C3 kernel changes only shared-memory addressing; arithmetic and summation order are identical, so the verdict
requires BIT-IDENTICAL out / lse and identical Mid for every valid split (Mid entries of empty splits are never
written by either kernel and stay NaN in both).  Also checks against an fp32 torch reference so a shared bug in
both kernels cannot pass silently.  Prints one JSON line per case and a final {"verdict": PASS|FAIL} line.

usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$REL $REL/.venv/bin/python $REL/tools/e04/e04_numerics.py
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e04_cases import D, HK, HQ, KPS, VB, Case  # noqa: E402


def reference(c: Case, NS: int):
    """fp32 torch reference of the prefix attention for every row (no splits), natural-log LSE."""
    outs, lses = [], []
    G = HQ // HK
    for s in range(c.S):
        L = int(c.sl[s])
        pages = c.bt[s]
        pos = torch.arange(L, device="cuda")
        slots = c.kv[pages[pos // c.BS].long(), pos % c.BS]  # [L, Hk, SLOT]
        kb = slots[..., :96].to(torch.int64)
        # 3-bit codes, 8 per 3 bytes (little-endian bit stream as in the kernel)
        b3 = kb.reshape(L, HK, 32, 3)
        v = b3[..., 0] | (b3[..., 1] << 8) | (b3[..., 2] << 16)
        idx = torch.stack([(v >> (3 * i)) & 7 for i in range(8)], -1).reshape(L, HK, D)
        k = c.cent[idx]
        nrm = slots[..., 96:98].contiguous().view(torch.float16).float()  # [L,Hk,1]
        k = k * (nrm / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-16))
        vb = slots[..., KPS:KPS + VB].to(torch.int64)
        vq = torch.stack([vb & 15, vb >> 4], -1).reshape(L, HK, D).float()
        vs = slots[..., KPS + VB:KPS + VB + 2].contiguous().view(torch.float16).float()
        vz = slots[..., KPS + VB + 2:KPS + VB + 4].contiguous().view(torch.float16).float()
        vv = vq * vs + vz
        for r in range(c.QL):
            q = c.q_rot[s * c.QL + r].reshape(HK, G, D)
            sc = torch.einsum("hgd,lhd->hgl", q, k) * c.scale
            lse = torch.logsumexp(sc, -1)
            o = torch.einsum("hgl,lhd->hgd", torch.softmax(sc, -1), vv)
            outs.append(o.reshape(HQ, D))
            lses.append(lse.reshape(HQ))
    return torch.stack(outs), torch.stack(lses)


def main():
    from vllm.v1.attention.ops import tq_gqa_cuda as C

    ext = C._load()
    cases = [
        # (ctx, S, QL, BS, NS)  production-like first
        (28000, 1, 4, 3568, 128), (4096, 1, 4, 3568, 128), (65536, 1, 4, 3568, 128), (131072, 1, 4, 3568, 128),
        (28000, 4, 1, 3568, 128), (16384, 8, 1, 3568, 128), (1000, 1, 4, 3568, 128),
        (37, 1, 4, 3568, 128), (1, 1, 4, 3568, 128), (5003, 2, 4, 16, 128), (777, 3, 2, 16, 32),
        (28000, 1, 4, 3568, 64), (28000, 1, 4, 3568, 256), (9999, 2, 3, 3568, 128),
    ]
    ok_all = True
    for i, (ctx, S, QL, BS, NS) in enumerate(cases):
        c = Case(ctx, S, QL, BS, seed=100 + i)
        o0, l0, m0 = c.run(ext, "base", NS)
        o1, l1, m1 = c.run(ext, "c3", NS)
        torch.cuda.synchronize()
        bit_out = bool(torch.equal(o0.view(torch.int16), o1.view(torch.int16)))
        bit_lse = bool(torch.equal(l0.view(torch.int32), l1.view(torch.int32)))
        # Mid: compare bit patterns; NaN (unwritten) slots must match too
        bit_mid = bool(torch.equal(m0.view(torch.int32), m1.view(torch.int32)))
        nan = bool(torch.isnan(o1).any() or torch.isnan(l1).any())
        rec = {"case": i, "ctx": ctx, "S": S, "QL": QL, "BS": BS, "NS": NS, "bit_out": bit_out, "bit_lse": bit_lse,
               "bit_mid": bit_mid, "nan": nan}
        if ctx * S <= 70000:
            ro, rl = reference(c, NS)
            rec["ref_out_rel"] = round(((o1.float() - ro).abs().max() / ro.abs().max()).item(), 6)
            rec["ref_lse_abs"] = round((l1 - rl).abs().max().item(), 6)
            ref_ok = rec["ref_out_rel"] < 2e-2 and rec["ref_lse_abs"] < 2e-2
        else:
            ref_ok = True
        rec["ok"] = bit_out and bit_lse and bit_mid and not nan and ref_ok
        ok_all &= rec["ok"]
        print(json.dumps(rec), flush=True)
    print(json.dumps({"verdict": "PASS" if ok_all else "FAIL", "gpu": torch.cuda.get_device_name(0)}), flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
