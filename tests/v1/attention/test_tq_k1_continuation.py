# SPDX-License-Identifier: Apache-2.0
"""[FORK][LANE K1] TurboQuant continuation prefill on a real k3v4_nc cache (Triton store kernel): the K1 kernel,
unsegmented and segmented (pieces merged by LSE in the kernel epilogue), must match the stock dequant + SDPA
reference path, and the segmented path must stay inside its (piece + chunk) workspace."""

import math
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (7, 5),
    reason="needs an sm_75+ CUDA device",
)


def _impl_and_layer(tqa, D, Hq, Hk, device):
    from vllm.model_executor.layers.quantization.turboquant.centroids import get_centroids
    from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig

    impl = object.__new__(tqa.TurboQuantAttentionImpl)
    impl.num_heads, impl.head_size, impl.num_kv_heads = Hq, D, Hk
    impl.num_kv_groups = Hq // Hk
    impl.scale = D**-0.5
    impl.tq_config = cfg = TurboQuantConfig.from_cache_dtype("turboquant_k3v4_nc", D)
    impl._mse_bytes = math.ceil(D * cfg.key_mse_bits / 8)
    impl._val_data_bytes = math.ceil(D * cfg.effective_value_quant_bits / 8)
    impl._soa_store = False
    impl.sinks = None
    impl.sliding_window = None
    H = tqa._build_hadamard(D, str(device))
    cent = get_centroids(D, cfg.centroid_bits).to(device=device, dtype=torch.float32)
    c_sorted, _ = cent.sort()
    layer = SimpleNamespace(_tq_PiT=H, _tq_Pi=H, _tq_Pi_half=H.to(torch.float16), _tq_centroids=cent,
                            _tq_midpoints=(c_sorted[:-1] + c_sorted[1:]) / 2)
    return impl, layer, cfg


@pytest.mark.parametrize(
    "cached_len, q_len, seg_tokens",
    [(5000, 700, 1024), (4096, 512, 2048), (3000, 1500, 0), (20000, 512, 8192)],  # last: split-KV pieces too
)
def test_k1_continuation_matches_stock_path(monkeypatch, cached_len, q_len, seg_tokens):
    import vllm.v1.attention.backends.turboquant_attn as tqa
    from vllm.v1.attention.ops import fa75_prefill as k1
    from vllm.v1.worker import workspace as ws

    device = torch.device("cuda")
    D, Hq, Hk, bs = 256, 6, 1, 256
    impl, layer, cfg = _impl_and_layer(tqa, D, Hq, Hk, device)
    seq_len = cached_len + q_len
    nblk = math.ceil(seq_len / bs) + 1
    kv_cache = torch.zeros(nblk, bs, Hk, cfg.slot_size_aligned, dtype=torch.uint8, device=device)
    g = torch.Generator(device=device).manual_seed(cached_len + q_len)
    k_all = torch.randn(seq_len, Hk, D, device=device, generator=g, dtype=torch.float16)
    v_all = torch.randn(seq_len, Hk, D, device=device, generator=g, dtype=torch.float16)
    q = torch.randn(q_len, Hq, D, device=device, generator=g, dtype=torch.float16).mul_(2)
    perm = torch.randperm(nblk, generator=torch.Generator().manual_seed(1)).to(device)  # scattered pages
    block_table = perm.to(torch.int32).unsqueeze(0)
    slots = (perm[torch.arange(seq_len, device=device) // bs] * bs + torch.arange(seq_len, device=device) % bs)
    tqa.triton_turboquant_store(
        k_all, v_all, kv_cache, slots.to(torch.int64), layer._tq_PiT, layer._tq_midpoints,
        mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits, key_fp8=cfg.key_fp8,
    )
    if not ws.is_workspace_manager_initialized():
        ws.init_workspace_manager(device)
    kc, vc = k_all[cached_len:], v_all[cached_len:]
    args = (layer, q, kc, vc, kv_cache, block_table, cached_len, seq_len, layer._tq_Pi, layer._tq_centroids)

    monkeypatch.setattr(k1, "_ENABLED", False)
    ref = impl._continuation_prefill(*args).float()  # stock: dequant + SDPA (no wrappers planned)

    monkeypatch.setattr(k1, "_ENABLED", True)
    monkeypatch.setattr(k1, "_SEGMENT_TOKENS", seg_tokens)
    monkeypatch.setattr(k1, "_MIN_SPLIT_KEYS", 2048)  # let the 8K pieces split (wave-balancing path)
    out = impl._continuation_prefill(*args).float()
    err = ((out - ref).norm() / ref.norm()).item()
    assert torch.isfinite(out).all()
    assert err < 1e-3, err
