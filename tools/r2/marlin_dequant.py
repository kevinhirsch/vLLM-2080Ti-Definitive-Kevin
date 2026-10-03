"""Lane R2 / lead L55: dequantise a vLLM Marlin-layout W4A16 weight back to a dense fp16 [K, N] matrix.

Why: on a 2080 Ti the exllama "dequant to fp16 scratch + cublasHgemm" path measured 85 TFLOPS vs Marlin's 63 at
M>=2048 (vLLM PR #29901). We hold weights in Marlin layout only, so a large-M prefill path needs this inverse.

Two implementations, both CPU-verifiable:
  marlin_dequant_torch   exact inverse of marlin_weights/marlin_permute_scales/marlin_zero_points (reference)
  marlin_dequant_triton  one program per 16x64 Marlin tile; table lookup of the 1024-entry weight permutation;
                         writes the dense [K, N] fp16 tile (run under TRITON_INTERPRET=1 on CPU, compile-checked for sm_75)
Layout facts (marlin_utils_test.marlin_weights, 4-bit, fp16 activations): weight tile 16x16; for each k-tile row the n-tiles
are concatenated, each tile flattened (k_in*16 + n_in); then new[j] = old[perm[j]] inside every 1024-chunk (= 4 n-tiles); then 8
consecutive values are packed LSB-first into one int32. Scales/zero points use scale_perm (64) over groups.
"""
from __future__ import annotations

import numpy as np
import torch

from vllm.model_executor.layers.quantization.utils.marlin_utils import get_scale_perms
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import get_weight_perm

TILE_K, TILE_N = 16, 16     # weight layout tile; the permutation acts on 4 consecutive n-tiles (1024 codes)


def _unpack_rows_int32(packed: torch.Tensor) -> torch.Tensor:
    """[R, C] int32 -> [R, C*8] int64 nibbles, LSB first."""
    p = packed.to(torch.int64) & 0xFFFFFFFF
    out = torch.stack([(p >> (4 * i)) & 0xF for i in range(8)], dim=-1)
    return out.reshape(packed.shape[0], packed.shape[1] * 8)


def unpermute_scales(marlin_s: torch.Tensor, size_k: int, size_n: int, group_size: int) -> torch.Tensor:
    scale_perm, scale_perm_single = get_scale_perms()
    perm = scale_perm if (group_size < size_k and group_size != -1) else scale_perm_single
    inv = torch.from_numpy(np.argsort(np.array(perm)))
    s = marlin_s.reshape(-1, len(perm))[:, inv]
    return s.reshape(-1, size_n).contiguous()


def unpermute_zero_points(marlin_zp: torch.Tensor, num_groups: int, size_n: int) -> torch.Tensor:
    """Inverse of marlin_zero_points for 4 bit: [G, N/8] int32 -> [G, N] int64."""
    scale_perm, _ = get_scale_perms()
    zp = _unpack_rows_int32(marlin_zp)                      # [G, N] in permuted+interleaved order
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    undo = torch.from_numpy(np.argsort(interleave))
    zp = zp.reshape(-1, 8)[:, undo].reshape(num_groups, size_n)
    inv = torch.from_numpy(np.argsort(np.array(scale_perm)))
    zp = zp.reshape(-1, len(scale_perm))[:, inv].reshape(num_groups, size_n)
    return zp


def unpack_marlin_codes(marlin_q: torch.Tensor, size_k: int, size_n: int) -> torch.Tensor:
    """[K/16, N*16/8] int32 -> [K, N] int64 codes in 0..15."""
    perm = get_weight_perm(4).to(torch.int64)               # new[j] = old[perm[j]]
    q = _unpack_rows_int32(marlin_q)                        # [K/16, N*16] permuted order
    q = q.reshape(-1, perm.numel())
    old = torch.empty_like(q)
    old[:, perm] = q                                        # invert the gather
    old = old.reshape(size_k // TILE_K, size_n // TILE_N, TILE_K, TILE_N)
    return old.permute(0, 2, 1, 3).reshape(size_k, size_n)


def marlin_dequant_torch(marlin_q, marlin_s, marlin_zp, size_k, size_n, group_size, bias_if_sym: int = 8):
    codes = unpack_marlin_codes(marlin_q, size_k, size_n)
    s = unpermute_scales(marlin_s, size_k, size_n, group_size).to(torch.float32)
    gs = size_k if group_size in (-1, None) else group_size
    s_full = s.repeat_interleave(gs // (size_k // s.shape[0]) if False else size_k // s.shape[0], dim=0)
    if marlin_zp is not None:
        zp = unpermute_zero_points(marlin_zp, size_k // gs, size_n).repeat_interleave(gs, dim=0)
        w = (codes - zp).to(torch.float32) * s_full
    else:
        w = (codes - bias_if_sym).to(torch.float32) * s_full
    return w.to(torch.float16)


# --------------------------------------------------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _marlin_dequant_kernel(q_ptr, s_ptr, zp_ptr, perm_ptr, out_ptr, size_k, size_n, GS: tl.constexpr,
                               HAS_ZP: tl.constexpr, SYM_BIAS: tl.constexpr, OUT_NK: tl.constexpr):
        # program = one (k-tile of 16 rows, group of 4 n-tiles = 64 cols): 1024 codes in permuted order, 128 int32 words
        kt = tl.program_id(0)
        nt = tl.program_id(1)
        j = tl.arange(0, 1024)                                   # stored position inside the tile chunk
        n_tiles = size_n // 64
        word_base = (kt * n_tiles + nt) * 128
        word = tl.load(q_ptr + word_base + (j // 8))
        code = (word >> ((j % 8) * 4)) & 0xF
        old = tl.load(perm_ptr + j)                              # original flat index = ntile4*256 + k_in*16 + n_in
        k_in = (old % 256) // 16
        n_in = old % 16
        k = kt * 16 + k_in
        n = nt * 64 + (old // 256) * 16 + n_in
        g = k // GS
        # scales/zero points are un-permuted on the host (tiny tensors): dense [G, N]
        s = tl.load(s_ptr + g * size_n + n).to(tl.float32)
        if HAS_ZP:
            z = tl.load(zp_ptr + g * size_n + n).to(tl.float32)
            w = (code.to(tl.float32) - z) * s
        else:
            w = (code.to(tl.float32) - SYM_BIAS) * s
        if OUT_NK:   # [N, K] K-contiguous: what cublasGemmEx(opA=T) wants
            tl.store(out_ptr + n * size_k + k, w.to(tl.float16))
        else:        # [K, N] row-major
            tl.store(out_ptr + k * size_n + n, w.to(tl.float16))

    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False


def marlin_dequant_triton(marlin_q, marlin_s, marlin_zp, size_k, size_n, group_size, out=None, out_nk=False):
    """Host side un-permutes the (tiny) scales/zero points once; the kernel handles the weight tile layout."""
    gs = size_k if group_size in (-1, None) else group_size
    s = unpermute_scales(marlin_s, size_k, size_n, group_size).to(torch.float16).contiguous()
    zp = None
    if marlin_zp is not None:
        zp = unpermute_zero_points(marlin_zp, size_k // gs, size_n).to(torch.int32).contiguous()
    perm = get_weight_perm(4).to(torch.int32).contiguous().to(marlin_q.device)
    if out is None:
        out = torch.empty((size_n, size_k) if out_nk else (size_k, size_n), dtype=torch.float16, device=marlin_q.device)
    grid = (size_k // 16, size_n // 64)
    _marlin_dequant_kernel[grid](marlin_q, s, zp if zp is not None else s, perm, out, size_k, size_n, GS=gs,
                                 HAS_ZP=zp is not None, SYM_BIAS=8, OUT_NK=out_nk)
    return out
