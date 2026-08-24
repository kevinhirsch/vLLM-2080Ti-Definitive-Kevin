#!/usr/bin/env python3
"""F-2 phase 2: TurboQuant decode depth-cost KERNEL-LEVEL timing probe.

Standalone companion to `docs/f2-tq-depth-cost-research.md` (read that first
for the kernel map and the H1/H2/H3 hypothesis table this probe exists to
discriminate between). Attributes the measured TPOT depth STAIRCASE

    ctx:   4K    16K   32K   40K   48K   56K   65K   131K
    TPOT: 16.5  17.6  21.4  25.5  25.4  34.9  49.2  55.4   (ms)

to a specific kernel/stage in the TurboQuant decode path
(`vllm/v1/attention/ops/triton_turboquant_decode.py` +
`vllm/v1/attention/backends/turboquant_attn.py`) by timing each candidate
in isolation across a depth ladder, on synthetic single-sequence
decode-shaped inputs. No engine, no server, no model weights.

Timed arms (see "ARM NOTES" below for what each one stands in for):
    stage1_decode1    _tq_decode_stage1 kernel alone, B=1  (plain decode)
    stage1_mtp4       _tq_decode_stage1 kernel alone, B=4  (MTP verify-width
                       prefix leg, matches _spec_continuation_decode_attention)
    stage2_reduce      _fwd_kernel_stage2 alone, B=4 (LSE reduce across splits)
    fused_wrapper_mtp4 triton_turboquant_decode_attention() end-to-end, B=4
                       (real Python launcher: q_rot GEMM + stage1 + stage2 --
                       this is what VLLM_TURBOQUANT_SPEC_CONTINUATION_DECODE_
                       FASTPATH=1 would dispatch MTP verify steps to)
    full_dequant_kv    _tq_full_dequant_kv kernel alone, B=1 (bulk
                       materialize-to-fp16 -- what MTP verify steps actually
                       hit TODAY, by default, per doc section 1b)
    k_derotate_gemm    the fp16 k_cached @ Pi_half GEMM that follows
                       _tq_full_dequant_kv in _continuation_prefill
                       (turboquant_attn.py ~L2278) to undo the Hadamard
                       rotation on dequanted MSE keys
    downstream_attn    attention over the materialized+concatenated K/V
                       (the flash_attn/flashinfer/SDPA call after dequant)

{full_dequant_kv, k_derotate_gemm, downstream_attn} summed ~= today's
default per-verify-step cost at depth L (H1's "materialize-then-attend"
path). {stage1_mtp4, stage2_reduce} (or fused_wrapper_mtp4 directly) ~=
the fastpath's cost at the same L. Comparing their depth curves is the
decisive H1 test the research doc calls for.

FIDELITY: the KV cache buffer is a REAL `turboquant_k3v4_nc` byte layout
(computed from the real `TurboQuantConfig`), not a stand-in shape -- filled
with random bytes rather than real quantized data. This is safe and
timing-faithful, not a simplification: every bitfield the kernels derive
from those bytes (`mse_idx`, 3-bit/4-bit value codes) is masked to its
valid range in-kernel before being used as a gather/table index (see
`triton_turboquant_decode.py` lines ~178-183, ~283), so random bytes
produce the same memory-access pattern and instruction path as real
quantized data -- only the numeric *values* are garbage, which doesn't
matter for a timing probe. Centroids and the Hadamard rotation matrix are
computed for real (both are cheap, pure-Python/tensor ops, see
`get_centroids` / `_build_hadamard` below).

FIDELITY CAVEAT (the one real simplification): `downstream_attn` prefers
real `flash_attn_varlen_func` (matching `_continuation_prefill`'s actual
call) but this SM75/2080Ti fork's vendored FA2 is typically unavailable
(`fa_utils.py` L30-36: "expected for SM75-only 2080 Ti builds") and this
probe does not wire up FlashInfer's prefill wrapper (workspace buffer +
plan/run split) standalone -- disproportionate import complexity for a
timing probe. Falls back to a manual causal-masked `scaled_dot_product_
attention` call over the same real tensor shapes/dtypes (same O(q_len *
L * D) FLOPs and memory traffic). Whichever path ran is printed at
start. This caveat affects only `downstream_attn`'s absolute ms and not
its depth-scaling *shape*, which is what this probe is attributing.

IMPORTANT -- DO NOT RUN THIS NOW. The GPUs are currently running prod
plus a benchmark window. This file has been authored and syntax/AST
checked only (`python3 -m py_compile tools/tq_depth_probe.py`), never
executed. Run it in the next engine-free window; see the invocation
command at the bottom of this docstring's companion report.

Usage (engine-free window only):
    python3 tools/tq_depth_probe.py [preset] [depths_csv] [warmup] [reps]

    preset      TQ_PRESETS key, default turboquant_k3v4_nc (matches the
                production preset behind the measured staircase)
    depths_csv  comma-separated kv_len ladder, default the 8-point ladder
                below (8192..131072)
    warmup      per-arm warmup iterations, default 20
    reps        per-arm timed iterations (median reported), default 100

Env overrides honored (same names the real kernels read, for H2 sweeps):
    VLLM_TURBOQUANT_DECODE_BLOCK_KV   tokens/tile in stage1 (default 2)
    VLLM_TURBOQUANT_MAX_KV_SPLITS     stage1/stage2 split count (default 32)
"""

import math
import os
import statistics
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Qwen3.8-27B full-attention-layer geometry (per TP rank == no TP here: this
# probe times a single layer, single GPU, per the task's "16 full-attn
# layers-worth is unnecessary -- one layer is fine"). Confirmed against the
# local checkout's real config.json (text_config): num_attention_heads=24,
# num_key_value_heads=4, head_dim=256, full_attention_interval=4 of 64
# layers => 16 full-attention layers total (the "16 layers" the task means).
# ---------------------------------------------------------------------------
HQ = 24  # query heads
HK = 4  # kv heads (GQA group size = HQ // HK = 6)
D = 256  # head_dim
BLOCK_SIZE = 16  # vLLM DEFAULT_BLOCK_SIZE, vllm/config/cache.py:45
Q_LEN = 4  # MTP K=3 verify width -> K+1 = 4 query positions per step

DEPTHS = [8192, 16384, 32768, 49152, 57344, 65536, 98304, 131072]
WARMUP = 20
REPS = 100
DEFAULT_PRESET = "turboquant_k3v4_nc"


def _build_hadamard(d: int, device: torch.device) -> torch.Tensor:
    """Sylvester-construction orthonormal Hadamard matrix, D x D.

    Reimplemented (not imported) to keep this probe import-light and free
    of the engine-context requirements the real
    `turboquant_attn._build_hadamard_cached` module carries as neighbors.
    Mirrors that function exactly (turboquant_attn.py ~L352-357).
    """
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(device)


def time_cuda(fn, warmup: int = WARMUP, reps: int = REPS):
    """Median/min/max wall time (ms) for `fn()`, torch.cuda.Event-based.

    All `reps` iterations are queued back-to-back before a single trailing
    `synchronize()` (not synced per-iteration) so kernel launches aren't
    artificially serialized by host round-trips -- steady-state device
    timing, not launch-latency-inflated timing.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times_ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return {
        "median": statistics.median(times_ms),
        "min": times_ms[0],
        "max": times_ms[-1],
    }


def build_common(preset: str, device: torch.device):
    """One-time (depth-independent) setup: TQ config, centroids, rotation."""
    from vllm.model_executor.layers.quantization.turboquant.centroids import (
        get_centroids,
    )
    from vllm.model_executor.layers.quantization.turboquant.config import (
        TurboQuantConfig,
    )
    from vllm.v1.attention.ops.triton_turboquant_decode import _fp8_format_code

    cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=D)
    if cfg.key_fp8:
        raise ValueError(
            f"preset {preset!r} is an FP8-key preset; this probe targets the "
            "MSE-key staircase (k3v4_nc/4bit_nc/3bit_nc). Pass one of those."
        )

    mse_bytes = math.ceil(D * cfg.key_mse_bits / 8)
    val_data_bytes = math.ceil(D * cfg.effective_value_quant_bits / 8)
    slot_size = cfg.slot_size_aligned

    Pi = _build_hadamard(D, device)  # symmetric: Pi == PiT
    PiT = Pi
    Pi_half = Pi.to(torch.float16)

    centroids = get_centroids(D, cfg.centroid_bits).to(device=device, dtype=torch.float32)

    device_index = device.index if device.index is not None else 0
    fp8_format = _fp8_format_code(device_index)

    return {
        "cfg": cfg,
        "mse_bytes": mse_bytes,
        "val_data_bytes": val_data_bytes,
        "slot_size": slot_size,
        "Pi": Pi,
        "PiT": PiT,
        "Pi_half": Pi_half,
        "centroids": centroids,
        "fp8_format": fp8_format,
        "scale": 1.0 / math.sqrt(D),
    }


def build_depth_inputs(depth: int, common: dict, device: torch.device):
    """Synthetic single-sequence decode-shaped inputs at kv_len == depth.

    Real `turboquant_k3v4_nc` cache byte layout (see module docstring for
    why random bytes are timing-faithful here); real Hadamard/centroids
    from `common`; block_size-aligned depths (all entries in DEPTHS are
    multiples of BLOCK_SIZE=16, so alloc_len == depth exactly, no padding).
    """
    assert depth % BLOCK_SIZE == 0, f"depth {depth} must be a multiple of {BLOCK_SIZE}"
    pages = depth // BLOCK_SIZE

    kv_cache = torch.randint(
        0, 256, (pages, BLOCK_SIZE, HK, common["slot_size"]),
        dtype=torch.uint8, device=device,
    )
    block_table_1 = torch.arange(pages, dtype=torch.int32, device=device).unsqueeze(0)
    block_table_4 = block_table_1.expand(Q_LEN, -1).contiguous()

    seq_lens_1 = torch.full((1,), depth, dtype=torch.int32, device=device)
    seq_lens_4 = torch.full((Q_LEN,), depth, dtype=torch.int32, device=device)

    # Original-space queries (fp16, as the model produces them).
    q_single_orig = torch.randn(1, HQ, D, device=device, dtype=torch.float16)
    q_mtp_orig = torch.randn(Q_LEN, HQ, D, device=device, dtype=torch.float16)

    # MSE path: kernel wants the Hadamard-rotated query in float32
    # (mirrors triton_turboquant_decode.triton_turboquant_decode_attention's
    # `q_rot = (query.float() @ PiT).contiguous()`).
    q_rot_1 = (q_single_orig.float() @ common["PiT"]).contiguous()
    q_rot_4 = (q_mtp_orig.float() @ common["PiT"]).contiguous()

    max_splits = int(os.environ.get("VLLM_TURBOQUANT_MAX_KV_SPLITS", "32"))
    mid_o_1 = torch.empty(1, HQ, max_splits, D + 1, dtype=torch.float32, device=device)
    mid_o_4 = torch.empty(Q_LEN, HQ, max_splits, D + 1, dtype=torch.float32, device=device)

    # Bulk-dequant output buffers (_tq_full_dequant_kv writes these).
    k_out = torch.empty(1, HK, depth, D, dtype=torch.float16, device=device)
    v_out = torch.empty(1, HK, depth, D, dtype=torch.float16, device=device)

    # "Current chunk" raw K/V for the downstream continuation-attention arm
    # (the just-produced, not-yet-quantized tokens the real code appends).
    k_chunk = torch.randn(Q_LEN, HK, D, device=device, dtype=torch.float16)
    v_chunk = torch.randn(Q_LEN, HK, D, device=device, dtype=torch.float16)

    return {
        "depth": depth,
        "pages": pages,
        "kv_cache": kv_cache,
        "block_table_1": block_table_1,
        "block_table_4": block_table_4,
        "seq_lens_1": seq_lens_1,
        "seq_lens_4": seq_lens_4,
        "q_single_orig": q_single_orig,
        "q_mtp_orig": q_mtp_orig,
        "q_rot_1": q_rot_1,
        "q_rot_4": q_rot_4,
        "mid_o_1": mid_o_1,
        "mid_o_4": mid_o_4,
        "max_splits": max_splits,
        "k_out": k_out,
        "v_out": v_out,
        "k_chunk": k_chunk,
        "v_chunk": v_chunk,
    }


# ---------------------------------------------------------------------------
# Per-arm closures. Each returns a zero-arg callable suitable for time_cuda.
# ---------------------------------------------------------------------------


def make_stage1_call(B: int, inp: dict, common: dict):
    from vllm.triton_utils import triton
    from vllm.v1.attention.ops.triton_turboquant_decode import (
        _DECODE_BLOCK_KV,
        _tq_decode_stage1,
    )

    cfg = common["cfg"]
    q_rot = inp["q_rot_1"] if B == 1 else inp["q_rot_4"]
    block_table = inp["block_table_1"] if B == 1 else inp["block_table_4"]
    seq_lens = inp["seq_lens_1"] if B == 1 else inp["seq_lens_4"]
    mid_o = inp["mid_o_1"] if B == 1 else inp["mid_o_4"]
    max_splits = inp["max_splits"]
    kv_cache = inp["kv_cache"]
    kv_group_size = HQ // HK
    grid = (B, HQ, max_splits)
    block_d = triton.next_power_of_2(D)

    def _call():
        _tq_decode_stage1[grid](
            q_rot,
            kv_cache,
            block_table,
            seq_lens,
            common["centroids"],
            mid_o,
            q_rot.stride(0),
            q_rot.stride(1),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_table.stride(0),
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            NUM_KV_HEADS=HK,
            HEAD_DIM=D,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_KV_SPLITS=max_splits,
            KV_GROUP_SIZE=kv_group_size,
            MSE_BITS=cfg.key_mse_bits,
            MSE_BYTES=common["mse_bytes"],
            KPS=cfg.key_packed_size,
            VQB=cfg.effective_value_quant_bits,
            VAL_DATA_BYTES=common["val_data_bytes"],
            ATTN_SCALE=common["scale"],
            SLIDING_WINDOW=0,
            BLOCK_D=block_d,
            BLOCK_KV=_DECODE_BLOCK_KV,
            KEY_FP8=0,
            NORM_CORRECTION=1 if cfg.norm_correction else 0,
            FP8_FORMAT=common["fp8_format"],
            num_warps=1,
            num_stages=1,
        )

    return _call


def make_stage2_call(inp: dict):
    from vllm.triton_utils import triton
    from vllm.v1.attention.ops.triton_decode_attention import _fwd_kernel_stage2

    mid_o = inp["mid_o_4"]
    B = Q_LEN
    max_splits = inp["max_splits"]
    output = torch.empty(B, HQ, D, dtype=torch.float16, device=mid_o.device)
    lse = torch.empty(B, HQ, dtype=torch.float32, device=mid_o.device)
    grid = (B, HQ)
    block_dv = triton.next_power_of_2(D)

    def _call():
        _fwd_kernel_stage2[grid](
            mid_o,
            output,
            lse,
            inp["seq_lens_4"],
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            output.stride(0),
            output.stride(1),
            lse.stride(0),
            NUM_KV_SPLITS=max_splits,
            BLOCK_DV=block_dv,
            Lv=D,
            OUTPUT_FP16=1,
            SLIDING_WINDOW=0,
            num_warps=4,
            num_stages=2,
        )

    return _call


def make_fused_wrapper_call(inp: dict, common: dict):
    from vllm.v1.attention.ops.triton_turboquant_decode import (
        triton_turboquant_decode_attention,
    )

    cfg = common["cfg"]

    def _call():
        triton_turboquant_decode_attention(
            query=inp["q_mtp_orig"],
            kv_cache=inp["kv_cache"],
            block_table=inp["block_table_4"],
            seq_lens=inp["seq_lens_4"],
            Pi=common["Pi"],
            centroids=common["centroids"],
            scale=common["scale"],
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=False,
            norm_correction=cfg.norm_correction,
            PiT=common["PiT"],
            max_num_kv_splits=inp["max_splits"],
            sliding_window=0,
        )

    return _call


def make_full_dequant_call(inp: dict, common: dict):
    from vllm.triton_utils import triton
    from vllm.v1.attention.ops.triton_turboquant_decode import _tq_full_dequant_kv

    cfg = common["cfg"]
    kv_cache = inp["kv_cache"]
    block_table = inp["block_table_1"]
    k_out, v_out = inp["k_out"], inp["v_out"]
    grid = (inp["depth"], 1 * HK)
    block_d = triton.next_power_of_2(D)

    def _call():
        _tq_full_dequant_kv[grid](
            kv_cache,
            block_table,
            common["centroids"],
            k_out,
            v_out,
            k_out.stride(0),
            k_out.stride(1),
            k_out.stride(2),
            v_out.stride(0),
            v_out.stride(1),
            v_out.stride(2),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_table.stride(0),
            HEAD_DIM=D,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_KV_HEADS=HK,
            MSE_BYTES=common["mse_bytes"],
            KPS=cfg.key_packed_size,
            VQB=cfg.effective_value_quant_bits,
            VAL_DATA_BYTES=common["val_data_bytes"],
            MSE_BITS=cfg.key_mse_bits,
            KEY_FP8=0,
            BLOCK_D=block_d,
            NORM_CORRECTION=1 if cfg.norm_correction else 0,
            FP8_FORMAT=common["fp8_format"],
            num_warps=4,
        )

    return _call


def make_derotate_call(inp: dict, common: dict):
    """fp16 GEMM undoing the Hadamard rotation on dequanted MSE keys.

    Mirrors turboquant_attn.py _continuation_prefill's
    `k_flat = k_cached[...].reshape(-1, D); k_flat = k_flat @ Pi_half`
    (~L2278-2286). Depends on `full_dequant_kv` having populated k_out,
    but the GEMM itself doesn't read data values meaningfully -- any
    fp16 buffer of the right shape gives real GEMM timing.
    """
    k_out = inp["k_out"]  # [1, HK, depth, D]
    Pi_half = common["Pi_half"]
    depth = inp["depth"]

    def _call():
        k_flat = k_out[0].reshape(-1, D)  # [HK * depth, D]
        k_rot = k_flat @ Pi_half
        k_rot.reshape(HK, depth, D).transpose(0, 1)

    return _call


def make_downstream_attn_call(inp: dict, common: dict):
    """Attention over materialized+concatenated K/V (post-dequant).

    Prefers real flash_attn_varlen_func (what production actually calls);
    falls back to a manual causal-masked SDPA of matching shape/dtype/FLOP
    count if FA2 isn't importable on this build (see module docstring's
    FIDELITY CAVEAT). Which path is used is decided once at import time
    and reported by main().
    """
    depth = inp["depth"]
    k_out, v_out = inp["k_out"], inp["v_out"]  # [1, HK, depth, D] fp16
    k_chunk, v_chunk = inp["k_chunk"], inp["v_chunk"]  # [Q_LEN, HK, D]
    q = inp["q_mtp_orig"]  # [Q_LEN, HQ, D]
    device = q.device
    scale = common["scale"]

    k_cached_trim = k_out[0].transpose(0, 1)  # [depth, HK, D]
    v_cached_trim = v_out[0].transpose(0, 1)
    k_full = torch.cat([k_cached_trim, k_chunk], dim=0)  # [depth+Q_LEN, HK, D]
    v_full = torch.cat([v_cached_trim, v_chunk], dim=0)

    try:
        from vllm.v1.attention.backends.fa_utils import (
            flash_attn_varlen_func,
            is_flash_attn_varlen_func_available,
        )

        has_fa = is_flash_attn_varlen_func_available()
    except Exception:
        has_fa = False

    if has_fa:
        cu_seqlens_q = torch.tensor([0, Q_LEN], dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor([0, depth + Q_LEN], dtype=torch.int32, device=device)

        def _call():
            flash_attn_varlen_func(
                q=q,
                k=k_full,
                v=v_full,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=Q_LEN,
                max_seqlen_k=depth + Q_LEN,
                softmax_scale=scale,
                causal=True,
            )

        return _call, "flash_attn_varlen_func"

    # SDPA fallback. GQA-expand K/V heads (HK -> HQ) and build the
    # "full prefix + local causal suffix" mask once (outside the timed
    # closure -- the real FA call needs no materialized mask at all, so
    # this fallback carries a bit more work than production; noted in the
    # module docstring's fidelity caveat).
    kv_group_size = HQ // HK
    k_exp = k_full.repeat_interleave(kv_group_size, dim=1)  # [depth+Q_LEN, HQ, D]
    v_exp = v_full.repeat_interleave(kv_group_size, dim=1)
    q_b = q.permute(1, 0, 2).unsqueeze(0).contiguous()  # [1, HQ, Q_LEN, D]
    k_b = k_exp.permute(1, 0, 2).unsqueeze(0).contiguous()  # [1, HQ, depth+Q_LEN, D]
    v_b = v_exp.permute(1, 0, 2).unsqueeze(0).contiguous()

    total_k = depth + Q_LEN
    row = torch.arange(Q_LEN, device=device).view(Q_LEN, 1)
    col = torch.arange(total_k, device=device).view(1, total_k)
    # Column < depth (prefix): always visible. Column >= depth (current
    # chunk): visible iff local-causal (col - depth <= row).
    mask = (col < depth) | ((col - depth) <= row)
    # SDPA requires the additive mask dtype to match the query dtype (half).
    attn_mask = torch.zeros(Q_LEN, total_k, dtype=q_b.dtype, device=device)
    attn_mask.masked_fill_(~mask, float("-inf"))
    attn_mask = attn_mask.view(1, 1, Q_LEN, total_k)

    def _call():
        F.scaled_dot_product_attention(q_b, k_b, v_b, attn_mask=attn_mask, scale=scale)

    return _call, "sdpa_fallback"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

ARM_ORDER = [
    "stage1_decode1",
    "stage1_mtp4",
    "stage2_reduce",
    "fused_wrapper_mtp4",
    "full_dequant_kv",
    "k_derotate_gemm",
    "downstream_attn",
]


def run_depth(depth: int, common: dict, warmup: int, reps: int, results: dict, notes: dict):
    device = common["device"]
    inp = build_depth_inputs(depth, common, device)

    arms = {
        "stage1_decode1": make_stage1_call(1, inp, common),
        "stage1_mtp4": make_stage1_call(Q_LEN, inp, common),
        "stage2_reduce": make_stage2_call(inp),
        "fused_wrapper_mtp4": make_fused_wrapper_call(inp, common),
        "full_dequant_kv": make_full_dequant_call(inp, common),
    }
    # k_derotate_gemm and downstream_attn read k_out/v_out -- run
    # full_dequant_kv once (untimed) first so those buffers hold
    # shape-correct data before those two arms are timed.
    arms["full_dequant_kv"]()
    torch.cuda.synchronize()
    arms["k_derotate_gemm"] = make_derotate_call(inp, common)
    downstream_call, downstream_impl = make_downstream_attn_call(inp, common)
    arms["downstream_attn"] = downstream_call
    notes["downstream_attn_impl"] = downstream_impl

    for name in ARM_ORDER:
        stats = time_cuda(arms[name], warmup=warmup, reps=reps)
        results.setdefault(name, {})[depth] = stats

    del inp, arms
    torch.cuda.empty_cache()


def print_table(results: dict, depths: list):
    header = ["arm"] + [f"{d // 1024}K" for d in depths]
    col_w = max(len(h) for h in header + ARM_ORDER) + 2
    print("\n=== median ms per kernel/stage per depth (torch.cuda.Event, "
          f"{WARMUP} warmup + {REPS} reps) ===")
    print("".join(h.ljust(col_w) for h in header))
    for arm in ARM_ORDER:
        row = [arm]
        for d in depths:
            row.append(f"{results[arm][d]['median']:.3f}")
        print("".join(c.ljust(col_w) for c in row))

    print("\n=== step-over-step growth %  (staircase = flat-then-jump; "
          "smooth O(L) = steadily positive; H2 knee = flat, then jump once "
          "near a specific depth) ===")
    print("".join(h.ljust(col_w) for h in header))
    for arm in ARM_ORDER:
        row = [arm, "--"]
        for i in range(1, len(depths)):
            prev = results[arm][depths[i - 1]]["median"]
            cur = results[arm][depths[i]]["median"]
            pct = (cur - prev) / prev * 100 if prev > 0 else float("nan")
            row.append(f"{pct:+.1f}%")
        print("".join(c.ljust(col_w) for c in row))

    print("\n=== marginal cost: microseconds per additional 1K KV tokens "
          "between consecutive depths (should be ~constant for a clean "
          "O(L) kernel; should be ~0 for a depth-invariant one like "
          "stage2_reduce; a knee here pinpoints an H2-style saturation "
          "point) ===")
    print("".join(h.ljust(col_w) for h in header))
    for arm in ARM_ORDER:
        row = [arm, "--"]
        for i in range(1, len(depths)):
            prev_d, cur_d = depths[i - 1], depths[i]
            prev_t = results[arm][prev_d]["median"]
            cur_t = results[arm][cur_d]["median"]
            d_tok_k = (cur_d - prev_d) / 1000.0
            us_per_1k = (cur_t - prev_t) * 1000.0 / d_tok_k if d_tok_k else float("nan")
            row.append(f"{us_per_1k:.1f}")
        print("".join(c.ljust(col_w) for c in row))


def main():
    preset = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PRESET
    depths = (
        [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else DEPTHS
    )
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else WARMUP
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else REPS

    device = torch.device("cuda:0")
    common = build_common(preset, device)
    common["device"] = device

    block_kv = os.environ.get("VLLM_TURBOQUANT_DECODE_BLOCK_KV", "2 (default)")
    max_splits = os.environ.get("VLLM_TURBOQUANT_MAX_KV_SPLITS", "32 (default)")
    print(f"preset={preset}  Hq={HQ} Hk={HK} D={D} block_size={BLOCK_SIZE} "
          f"q_len={Q_LEN}")
    print(f"VLLM_TURBOQUANT_DECODE_BLOCK_KV={block_kv}  "
          f"VLLM_TURBOQUANT_MAX_KV_SPLITS={max_splits}")
    print(f"depths={depths}  warmup={warmup}  reps={reps}")

    results: dict = {}
    notes: dict = {}
    for depth in depths:
        print(f"... timing depth={depth} ({depth // 1024}K)", file=sys.stderr)
        run_depth(depth, common, warmup, reps, results, notes)

    print(f"\ndownstream_attn implementation used: "
          f"{notes.get('downstream_attn_impl', '?')}")
    print_table(results, depths)

    print(
        "\nReference production staircase (TPOT ms; different depth "
        "ladder, informal cross-check only):\n"
        "  4K=16.5  16K=17.6  32K=21.4  40K=25.5  48K=25.4  56K=34.9  "
        "65K=49.2  131K=55.4"
    )


if __name__ == "__main__":
    # DO NOT RUN: GPUs are serving prod + a benchmark window. Author-time
    # check only:  python3 -m py_compile tools/tq_depth_probe.py
    main()
