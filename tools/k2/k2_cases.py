"""Lane K2 (L83) test/bench fixtures: a REAL TurboQuant k3v4_nc cache written by the production store kernel, plus
the production-shaped call wrappers for the MTP verifier prefix (tq_gqa, the current path) and tq_imma (K2).

Per-rank shapes of Qwen3.8-27B at TP=2: Hq=12, Hk=2, D=256 (GQA 6), attention block_size 3568, 4 verifier rows
(1 + MTP3).  Everything here runs on the GPU given by CUDA_VISIBLE_DEVICES (run it only inside a window)."""
from __future__ import annotations

import math
import os

import torch

from vllm.model_executor.layers.quantization.turboquant.centroids import get_centroids
from vllm.v1.attention.ops import tq_gqa_cuda, tq_imma_cuda
from vllm.v1.attention.ops.triton_turboquant_decode import _tq_full_dequant_kv
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store

D, HQ, HK, SLOT = 256, 12, 2, 230
MSE_BITS, KPS, VQB, MSE_BYTES, VAL_BYTES = 3, 98, 4, 96, 128
BS = int(os.environ.get("K2_BLOCK_SIZE", "3568"))
SCALE = 1.0 / math.sqrt(D)


def hadamard(d: int, dev) -> torch.Tensor:
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(dev)


class Case:
    """B requests, each with `cached` committed tokens in the TQ cache (randomly scattered pages) and q_len verifier
    rows of raw query/key/value (the current MTP chunk, not yet attended from the cache)."""

    def __init__(self, B: int, ctx: int, q_len: int = 4, seed: int = 0, jitter: int = 300, regime: str = "mixed",
                 dev: str = "cuda"):
        g = torch.Generator(device="cpu").manual_seed(seed * 7919 + ctx * 31 + B)
        self.B, self.q_len, self.dev = B, q_len, dev
        self.H = hadamard(D, dev)
        self.cent = get_centroids(D, MSE_BITS).float().to(dev)
        cs, _ = self.cent.sort()
        self.mids = (cs[:-1] + cs[1:]) / 2
        self.cached = [max(1, ctx - int(torch.randint(0, jitter + 1, (1,), generator=g))) for _ in range(B)]
        npg = [(c + q_len + BS - 1) // BS for c in self.cached]
        self.max_pages = max(npg)
        total = sum(npg) + 1                               # page 0 = the "zero page" used for empty prefixes
        perm = (torch.randperm(total - 1, generator=g) + 1).tolist()
        bt = torch.zeros(B, self.max_pages, dtype=torch.int32)
        k = 0
        for i, n in enumerate(npg):
            bt[i, :n] = torch.tensor(perm[k:k + n], dtype=torch.int32)
            k += n
        self.block_table = bt.to(dev)
        self.kv = torch.zeros(total, BS, HK, SLOT, dtype=torch.uint8, device=dev)
        # structured K: per-head channel bias + a few outlier channels (real post-RoPE keys are far from isotropic)
        kbias = (torch.randn(HK, D, generator=g) * 0.5).to(dev)
        out_ch = torch.randperm(D, generator=g)[:8].to(dev)
        self._gen = torch.Generator(device=dev).manual_seed(seed + 17)
        self.needles = []
        for i, c in enumerate(self.cached):
            for t0 in range(0, c, 8192):
                n = min(8192, c - t0)
                kk, vv = self._kv(n, kbias, out_ch)
                pos = torch.arange(t0, t0 + n, device=dev)
                slots = (self.block_table[i, pos // BS].long() * BS + pos % BS).to(torch.int32)
                triton_turboquant_store(kk, vv, self.kv, slots, self.H, self.mids, mse_bits=MSE_BITS,
                                        key_packed_size=KPS, value_quant_bits=VQB)
        R = B * q_len
        self.query = (torch.randn(R, HQ, D, device=dev, generator=self._gen) * 1.0).half()
        if regime in ("needle", "mixed"):
            # half the query heads point at one cached key each (peaky softmax, like retrieval heads)
            ref_k = self.dequant_prefix_all()
            for i in range(B):
                for h in range(0, HQ, 2):
                    t = int(torch.randint(0, self.cached[i], (1,), generator=g))
                    kr = ref_k[i][0][h // (HQ // HK), t].float() @ self.H   # un-rotate (H symmetric orthonormal)
                    for j in range(q_len):
                        self.query[i * q_len + j, h] = (kr / kr.norm() * 24.0 + 0.3 * self.query[i * q_len + j, h].float()).half()
            self._ref_k = ref_k
        else:
            self._ref_k = None
        self.key_cur, self.val_cur = [x.reshape(B * q_len, HK, D) for x in self._kv(R, kbias, out_ch)]
        self.prefix_lens = torch.tensor(self.cached, dtype=torch.int32, device=dev)

    def _kv(self, n, kbias, out_ch):
        kk = torch.randn(n, HK, D, device=self.dev, generator=self._gen) + kbias
        kk[:, :, out_ch] *= 6.0
        vv = torch.randn(n, HK, D, device=self.dev, generator=self._gen) * (
            0.5 + torch.rand(n, 1, 1, device=self.dev, generator=self._gen))
        return kk.half(), vv.half()

    # ---------------------------------------------------------------- reference (exact w.r.t. the stored codes)
    def dequant_prefix(self, i: int):
        """Production full-dequant kernel: K in the ROTATED basis (fp16, norm-corrected), V fp16.  [Hk, L, D]."""
        L = self.cached[i]
        K = torch.empty(1, HK, L, D, dtype=torch.float16, device=self.dev)
        V = torch.empty(1, HK, L, D, dtype=torch.float16, device=self.dev)
        bt = self.block_table[i:i + 1].contiguous()
        _tq_full_dequant_kv[(L, HK)](
            self.kv, bt, self.cent, K, V, K.stride(0), K.stride(1), K.stride(2), V.stride(0), V.stride(1),
            V.stride(2), self.kv.stride(0), self.kv.stride(1), self.kv.stride(2), bt.stride(0), HEAD_DIM=D,
            BLOCK_SIZE=BS, NUM_KV_HEADS=HK, MSE_BYTES=MSE_BYTES, KPS=KPS, VQB=VQB, VAL_DATA_BYTES=VAL_BYTES,
            MSE_BITS=MSE_BITS, KEY_FP8=0, BLOCK_D=D, NORM_CORRECTION=1, FP8_FORMAT=0, num_warps=4)
        return K[0], V[0]

    def dequant_prefix_all(self):
        return [self.dequant_prefix(i) for i in range(self.B)]

    def reference(self, with_current: bool):
        """fp32 attention. Prefix only (with_current=False): out/lse of every verifier row over the cached tokens.
        Full (True): prefix + the raw current chunk, causal inside the chunk (what the verifier must compute)."""
        refs = self._ref_k or self.dequant_prefix_all()
        G, ql = HQ // HK, self.q_len
        outs, lses = [], []
        for i in range(self.B):
            K, V = refs[i]
            q = self.query[i * ql:(i + 1) * ql].float() @ self.H                    # rotated query [ql, Hq, D]
            Kh = K.float().repeat_interleave(G, 0)                                   # [Hq, L, D]
            Vh = V.float().repeat_interleave(G, 0)
            s = torch.einsum("rhd,hld->hrl", q, Kh) * SCALE                          # [Hq, ql, L]
            vals = Vh
            if with_current:
                kc = (self.key_cur[i * ql:(i + 1) * ql].float() @ self.H).repeat_interleave(G, 1)  # [ql, Hq, D]
                vc = self.val_cur[i * ql:(i + 1) * ql].float().repeat_interleave(G, 1)
                sc = torch.einsum("rhd,chd->hrc", q, kc) * SCALE                     # [Hq, ql, ql]
                causal = torch.ones(ql, ql, device=self.dev, dtype=torch.bool).tril()
                sc = sc.masked_fill(~causal, float("-inf"))
                s = torch.cat([s, sc], -1)
                vals = torch.cat([Vh, vc.transpose(0, 1)], 1)                        # [Hq, L+ql, D]
            lse = torch.logsumexp(s, -1)                                             # [Hq, ql]
            o = torch.einsum("hrl,hld->rhd", torch.softmax(s, -1), vals)
            outs.append(o)
            lses.append(lse.transpose(0, 1))
        return torch.cat(outs), torch.cat(lses)

    # ---------------------------------------------------------------- production-shaped calls
    def gqa_prefix(self, i: int, out=None, lse=None):
        """What production runs today per request (raw-current MTP path): one grouped tq_gqa call, q_per_seq=q_len."""
        ql = self.q_len
        q = self.query[i * ql:(i + 1) * ql]
        out = out if out is not None else torch.empty_like(q)
        lse = lse if lse is not None else torch.empty(ql, HQ, dtype=torch.float32, device=self.dev)
        tq_gqa_cuda.tq_gqa_decode_attention(q, self.kv, self.block_table[i:i + 1], self.prefix_lens[i:i + 1],
                                            self.H, self.cent, SCALE, True, q_per_seq=ql, num_splits=128, PiT=self.H,
                                            output_buf=out, lse_buf=lse)
        return out, lse

    def imma_prefix(self, i: int, out=None, lse=None, num_splits=None):
        """K2 per-request call (same shape as the backend's per-request IMMA branch)."""
        ql = self.q_len
        q = self.query[i * ql:(i + 1) * ql]
        rl = self.prefix_lens[i:i + 1].expand(ql).contiguous()
        out = out if out is not None else torch.empty_like(q)
        lse = lse if lse is not None else torch.empty(ql, HQ, dtype=torch.float32, device=self.dev)
        tq_imma_cuda.tq_imma_decode_attention(q, self.kv, self.block_table[i:i + 1], rl, self.cent, SCALE, True,
                                              q_per_seq=ql, num_seqs=1, num_splits=num_splits, PiT=self.H,
                                              output_buf=out, lse_buf=lse)
        return out, lse

    def imma_batched(self, num_splits=None):
        """K2 batched call (the backend's _imma_spec_prefix_batched): every request's prefix in one launch pair."""
        ql = self.q_len
        rl = self.prefix_lens.reshape(-1, 1).expand(self.B, ql).contiguous().reshape(-1)
        return tq_imma_cuda.tq_imma_decode_attention(self.query, self.kv, self.block_table, rl, self.cent, SCALE,
                                                     True, q_per_seq=ql, num_seqs=self.B, num_splits=num_splits,
                                                     PiT=self.H)
