# SPDX-License-Identifier: Apache-2.0
"""Lane U2: opt-in int4 (W4A16, group 128, asymmetric) for the bf16 lm_head and MTP block.

Default OFF.  Nothing here runs unless one of these env vars is set:

  VLLM_U2_INT4_HEAD=1   quantize ``lm_head.weight`` (bf16 [V, D]) at load time.
  VLLM_U2_INT4_MTP=1    quantize the MTP block linears (``mtp.fc`` and
                        ``mtp.layers.*.{self_attn.{q,k,v,o}_proj,mlp.{gate,up,down}_proj}``).
  VLLM_U2_INT4_EMBED=1  quantize ``embed_tokens`` (symmetric int4 g128: the CT embedding path has no zero points) -- VRAM only
                        (a lookup reads a few rows, so no bandwidth win); frees ~0.9 GiB/GPU for prefix-cache pages.
  VLLM_U2_CACHE_DIR     where the quantized tensors are cached
                        (default: ``<model_dir>-u2cache``).  The original weight files are never
                        modified; the cache is a separate directory that can simply be deleted.

Why: the MTP drafter shares the target's lm_head, so at MTP-3 the 2.37 GiB bf16 head is read four
times per decode step (plus the 0.79 GiB bf16 MTP block three times).  Decode is weight-bandwidth
bound, so int4 for these tensors cuts roughly a quarter of the step.  Speculative decoding stays
lossless w.r.t. the *target* distribution, but an int4 lm_head changes that distribution slightly,
so ``tools/u2/fidelity.py`` gates it (top-1 agreement / KL / MTP acceptance on recorded prompts).

How it plugs in (all three hooks are no-ops with the env unset):
  * ``CompressedTensorsConfig.from_config``  drops ``lm_head`` / the ``mtp`` regexes from the
    ``ignore`` list (``ignore_filter``);
  * ``CompressedTensorsConfig.get_scheme_dict`` lets a ParallelLMHead fall back to the "Linear"
    target scheme (class name ``ParallelLMHead`` never matches "Linear" on its own);
  * ``DefaultModelLoader.get_all_weights`` wraps the weight iterator (``wrap_weights``) so the bf16
    tensors are replaced by the compressed-tensors ``pack-quantized`` quadruple
    (``.weight_packed`` int32 [out, in/8], ``.weight_scale`` bf16 [out, in/g],
    ``.weight_zero_point`` int32 [out/8, in/g], ``.weight_shape`` int64 [2]) -- exactly the layout
    the checkpoint already uses for every other linear, so the stock Marlin path consumes it.

Quantizer: per-group asymmetric RTN with an MSE clip search (shrink the [min, max] range, pick the
shrink with least squared error per group); scale rounded to bf16 first so dequantization is exact.
This module only needs torch (+ safetensors for the cache) and is importable standalone.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from collections.abc import Iterable, Iterator

import torch

GROUP = 128
_SHIFTS = torch.arange(8, dtype=torch.int64) * 4
# shrink factors tried for the per-group range; 1.0 == plain min/max RTN
CLIP_GRID = (1.0, 0.975, 0.95, 0.925, 0.9, 0.875, 0.85, 0.825, 0.8, 0.75, 0.7)

# MTP linears that are bf16 in model-mtp.safetensors
_MTP_LINEAR = re.compile(
    r"^mtp\.(fc|layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj))\.weight$"
)


def head_enabled() -> bool:
    return os.environ.get("VLLM_U2_INT4_HEAD", "0") == "1"


def mtp_enabled() -> bool:
    return os.environ.get("VLLM_U2_INT4_MTP", "0") == "1"


def embed_enabled() -> bool:
    return os.environ.get("VLLM_U2_INT4_EMBED", "0") == "1"


def enabled() -> bool:
    return head_enabled() or mtp_enabled() or embed_enabled()


_EMBED_NAME = re.compile(r"^model\.(language_model\.)?embed_tokens\.weight$")


def ignore_filter(ignore: list[str]) -> list[str]:
    """Drop the ignore entries that keep lm_head / the MTP block bf16 (only when opted in)."""
    if not enabled():
        return ignore
    out = []
    for entry in ignore:
        if head_enabled() and entry == "lm_head":
            continue
        if mtp_enabled() and ("mtp" in entry and (entry.startswith("re:") or entry.startswith("mtp"))):
            continue
        out.append(entry)
    return out


def wants(name: str) -> bool:
    if head_enabled() and name == "lm_head.weight":
        return True
    if embed_enabled() and _EMBED_NAME.match(name):
        return True
    return mtp_enabled() and _MTP_LINEAR.match(name) is not None


def is_symmetric(name: str) -> bool:
    return _EMBED_NAME.match(name) is not None


# --------------------------------------------------------------------------------------
# quantizer
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _quant_group_chunk(w: torch.Tensor, group: int, grid: tuple[float, ...], symmetric: bool = False):
    """w float32 [R, in] -> (q_signed int8-range int32 [R, in], scale_bf16 [R, ng], zp int32 [R, ng])."""
    R, K = w.shape
    ng = K // group
    wg = w.reshape(R, ng, group)
    if symmetric:  # zero point fixed at 0 (nibble-8 = q), scale = amax*p/7.5
        amax = wg.abs().amax(-1)
        zero = torch.zeros((R, ng), dtype=w.dtype, device=w.device)
        best_err = torch.full((R, ng), float("inf"), dtype=w.dtype, device=w.device)
        best_scale = torch.ones((R, ng), dtype=w.dtype, device=w.device)
        for p in grid:
            scale = (amax * p / 7.5).clamp_min(1e-8).bfloat16().float()
            q = torch.round(wg / scale.unsqueeze(-1)).clamp_(-8, 7)
            err = ((q * scale.unsqueeze(-1) - wg) ** 2).sum(-1)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_scale = torch.where(better, scale, best_scale)
        q = torch.round(wg / best_scale.unsqueeze(-1)).clamp_(-8, 7)
        return q.reshape(R, K).to(torch.int32), best_scale.bfloat16(), zero.to(torch.int32)
    mn = torch.minimum(wg.amin(-1), torch.zeros((), dtype=w.dtype, device=w.device))
    mx = torch.maximum(wg.amax(-1), torch.zeros((), dtype=w.dtype, device=w.device))
    best_err = torch.full((R, ng), float("inf"), dtype=w.dtype, device=w.device)
    best_scale = torch.ones((R, ng), dtype=w.dtype, device=w.device)
    best_zp = torch.zeros((R, ng), dtype=w.dtype, device=w.device)
    for p in grid:
        lo, hi = mn * p, mx * p
        scale = ((hi - lo) / 15.0).clamp_min(1e-8).bfloat16().float()  # exactly what gets stored
        zp = torch.round(-8.0 - lo / scale).clamp_(-8, 7)
        q = torch.round(wg / scale.unsqueeze(-1)).add_(zp.unsqueeze(-1)).clamp_(-8, 7)
        err = (((q - zp.unsqueeze(-1)) * scale.unsqueeze(-1) - wg) ** 2).sum(-1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_scale = torch.where(better, scale, best_scale)
        best_zp = torch.where(better, zp, best_zp)
    q = torch.round(wg / best_scale.unsqueeze(-1)).add_(best_zp.unsqueeze(-1)).clamp_(-8, 7)
    return q.reshape(R, K).to(torch.int32), best_scale.bfloat16(), best_zp.to(torch.int32)


def _pack_nibbles_last(u: torch.Tensor) -> torch.Tensor:
    """u int32/int64 in [0, 15], [..., n*8] -> int32 [..., n] (nibble k of word j = element j*8+k)."""
    u = u.to(torch.int64).reshape(*u.shape[:-1], -1, 8)
    word = (u << _SHIFTS.to(u.device)).sum(-1)
    word = torch.where(word >= 2**31, word - 2**32, word)
    return word.to(torch.int32)


def quantize_linear(
    weight: torch.Tensor,
    group: int = GROUP,
    grid: tuple[float, ...] = CLIP_GRID,
    device: str | torch.device | None = None,
    rows_per_chunk: int = 8192,
    symmetric: bool = False,
) -> dict[str, torch.Tensor]:
    """bf16/fp16/fp32 [out, in] -> compressed-tensors pack-quantized tensors (CPU, contiguous)."""
    out_f, in_f = weight.shape
    assert in_f % group == 0 and out_f % 8 == 0, (out_f, in_f)
    dev = torch.device(device) if device is not None else torch.device("cpu")
    packed, scales, zps = [], [], []
    for r0 in range(0, out_f, rows_per_chunk):
        w = weight[r0 : r0 + rows_per_chunk].to(dev, torch.float32)
        q, s, z = _quant_group_chunk(w, group, grid, symmetric)
        packed.append(_pack_nibbles_last(q + 8).cpu())
        scales.append(s.cpu())
        zps.append(z.cpu())
    packed_t = torch.cat(packed, 0).contiguous()
    scale_t = torch.cat(scales, 0).contiguous()
    out = {
        "weight_packed": packed_t,
        "weight_scale": scale_t,
        "weight_shape": torch.tensor([out_f, in_f], dtype=torch.int64),
    }
    if not symmetric:
        zp_u = (torch.cat(zps, 0) + 8)  # [out, ng], 0..15
        out["weight_zero_point"] = _pack_nibbles_last(zp_u.t().contiguous()).t().contiguous()  # dim-0 pack -> [out/8, ng]
    return out


def dequantize_linear(t: dict[str, torch.Tensor], group: int = GROUP, dtype=torch.float32) -> torch.Tensor:
    """Inverse of quantize_linear (same math as ref_dump.dequant_linear; used by tests)."""
    packed, scale, zpp, shape = t["weight_packed"], t["weight_scale"].float(), t.get("weight_zero_point"), t["weight_shape"]
    out_f, in_f = int(shape[0]), int(shape[1])
    sh = _SHIFTS.to(torch.int32)
    q = ((packed.unsqueeze(-1) >> sh) & 15).reshape(out_f, in_f)
    if zpp is None:  # symmetric (embedding path): value = nibble - 8
        zp = torch.full((out_f, in_f // group), 8, dtype=q.dtype)
    else:
        zp = ((zpp.unsqueeze(1) >> sh.view(1, 8, 1)) & 15).reshape(out_f, -1)
    ng = in_f // group
    w = (q.reshape(out_f, ng, group) - zp.unsqueeze(-1)).to(torch.float32) * scale.unsqueeze(-1)
    return w.reshape(out_f, in_f).to(dtype)


# --------------------------------------------------------------------------------------
# cache + iterator wrapper
# --------------------------------------------------------------------------------------
def cache_dir_for(model_dir: str) -> str:
    return os.environ.get("VLLM_U2_CACHE_DIR") or (model_dir.rstrip("/") + "-u2cache")


def _cache_path(model_dir: str, name: str, src_stamp: str) -> str:
    key = hashlib.sha1(f"{name}|{src_stamp}|g{GROUP}|{CLIP_GRID}".encode()).hexdigest()[:12]
    safe = name.replace("/", "_")
    return os.path.join(cache_dir_for(model_dir), f"{safe}.{key}.safetensors")


def _src_stamp(model_dir: str) -> str:
    parts = []
    for fn in ("model.safetensors", "model-mtp.safetensors"):
        p = os.path.join(model_dir, fn)
        if os.path.exists(p):
            st = os.stat(p)
            parts.append(f"{fn}:{st.st_size}:{int(st.st_mtime)}")
    return ";".join(parts)


def get_quantized(model_dir: str, name: str, weight: torch.Tensor, logger=None) -> dict[str, torch.Tensor]:
    """Quantized tensors for ``name`` (``lm_head.weight`` etc.), from cache or computed + cached."""
    from safetensors.torch import load_file, save_file

    path = _cache_path(model_dir, name, _src_stamp(model_dir))
    if os.path.exists(path):
        return load_file(path)
    t0 = time.time()
    dev = os.environ.get("VLLM_U2_QUANT_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tensors = quantize_linear(weight, device=dev, symmetric=is_symmetric(name))
    finally:
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
    if logger:
        logger.info("U2 int4 %s: quantized %s in %.1fs on %s", name, tuple(weight.shape), time.time() - t0, dev)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        save_file(tensors, tmp)
        os.replace(tmp, path)
    except OSError as e:  # cache is an optimisation only
        if logger:
            logger.warning("U2 int4 cache write failed (%s); continuing without cache", e)
    return tensors


def wrap_weights(
    weights: Iterable[tuple[str, torch.Tensor]], model_dir: str, logger=None
) -> Iterator[tuple[str, torch.Tensor]]:
    """Replace selected bf16 ``X.weight`` with the four pack-quantized tensors ``X.weight_*``."""
    for name, tensor in weights:
        if wants(name) and tensor.dtype in (torch.bfloat16, torch.float16, torch.float32):
            stem = name[: -len(".weight")]
            q = get_quantized(model_dir, name, tensor, logger)
            for suffix, t in q.items():
                yield f"{stem}.{suffix}", t
        else:
            yield name, tensor
