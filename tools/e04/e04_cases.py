"""Lane E04 shared GPU fixtures: random-but-valid TurboQuant k3v4_nc KV pages at the production per-rank shape.

Per rank (TP=2) Qwen3.8-27B attention: Hq=12, Hk=2 (G=6), head_dim 256, 230-byte slot (98 K bytes incl. fp16 norm,
128 V nibble bytes, fp16 V scale + zero).  Production block_size is 3568 (mamba-aligned attention pages); 16 is the
smallest legal page and exercises the page-straddling chunk path on almost every chunk.
"""
from __future__ import annotations

import torch

D, HQ, HK, KPS, VB = 256, 12, 2, 98, 128
SLOT = KPS + VB + 4


class Case:
    def __init__(self, ctx: int, S: int, QL: int, BS: int = 3568, seed: int = 0, ragged: bool = True):
        g = torch.Generator(device="cuda").manual_seed(seed)
        dev = "cuda"
        self.ctx, self.S, self.QL, self.BS = ctx, S, QL, BS
        nb = (ctx + BS - 1) // BS + 1
        nbt = S * nb + 2
        kv = torch.randint(0, 256, (nbt, BS, HK, SLOT), dtype=torch.uint8, device=dev, generator=g)
        sh = kv.shape[:3]

        def put16(off, vals):
            kv[..., off:off + 2] = vals.to(torch.float16).view(torch.uint8).reshape(*vals.shape, 2)

        put16(96, torch.rand(sh, device=dev, generator=g) * 2 + 0.5)
        put16(KPS + VB, torch.rand(sh, device=dev, generator=g) * 0.2 + 0.05)
        put16(KPS + VB + 2, -torch.rand(sh, device=dev, generator=g))
        self.kv = kv
        self.cent = torch.tensor([-2.15, -1.34, -0.756, -0.245, 0.245, 0.756, 1.34, 2.15], device=dev) / 16
        # random orthogonal rotation (the production Pi is a Hadamard-like orthogonal matrix)
        self.Pi = torch.linalg.qr(torch.randn(D, D, device=dev, generator=g))[0].contiguous()
        self.PiT = self.Pi.T.contiguous()
        perm = torch.randperm(S * nb, device=dev, generator=g).to(torch.int32)  # shuffled pages
        self.bt = perm.reshape(S, nb).contiguous()
        lens = [max(1, ctx - ((i * 37) % 900 if ragged else 0)) for i in range(S)]
        self.sl = torch.tensor(lens, dtype=torch.int32, device=dev)
        self.q = (torch.randn(S * QL, HQ, D, device=dev, generator=g) * 3).to(torch.float16)
        self.scale = 1 / 16
        self.q_rot = (self.q.float() @ self.PiT).contiguous()

    def run(self, ext, variant: str, NS: int = 128, mid=None, out=None, lse=None):
        R = self.S * self.QL
        G = HQ // HK
        QS = max(1, min(self.QL, 16 // G))
        if mid is None:
            mid = torch.full((R, HQ, NS, D + 1), float("nan"), dtype=torch.float32, device="cuda")
        if out is None:
            out = torch.empty(R, HQ, D, dtype=torch.float16, device="cuda")
        if lse is None:
            lse = torch.empty(R, HQ, dtype=torch.float32, device="cuda")
        fn = ext.decode_c3 if variant == "c3" else ext.decode
        fn(self.q_rot, self.kv, self.bt, self.sl, self.cent, mid, out, lse, self.QL, QS, G, self.BS, NS, self.scale, 1)
        return out, lse, mid
