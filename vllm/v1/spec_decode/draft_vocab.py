# SPDX-License-Identifier: Apache-2.0
"""FR1: opt-in reduced-vocabulary MTP draft head (``VLLM_MTP_DRAFT_VOCAB=<token id list>``).

Why: the MTP drafter reuses the target's lm_head, which is vocab-parallel (TP=2) and int4
(lane U2): 248,320 x 5,120 x 0.5 B = 0.64 GiB total, 0.32 GiB read per rank on EVERY draft step
(3 per decode step at MTP-3), plus an all-gather of [rows, 124,160] logits per rank.  A draft
only has to *propose* a token; the target still verifies against the full vocabulary.  So the
draft step can argmax over a top-K token subset served by a small REPLICATED int4 head
(K=40,960 -> 0.10 GiB read per rank, no all-gather).

Correctness: under rejection sampling a draft token d is accepted with prob min(1, p(d)/q(d)).
Draft sampling here is greedy (draft_probs is None, i.e. q is one-hot on d), which the sampler
treats as deterministic, so ANY proposal rule is lossless: the output distribution is that of the
target.  A subset only changes *which* d is proposed (zero draft probability outside the subset is
a valid q).  The cost is purely acceptance: a target token outside the subset can never be drafted.
If probabilistic draft probs are enabled (draft_sample_method="probabilistic") the feature disables
itself (a subset would need renormalised q); see ``llm_base_proposer._maybe_init_draft_vocab``.

CUDA-graph safety: the head runs in the drafter's eager sampling step (outside the PIECEWISE
graph), and everything (id table, quantized weights, Marlin workspace) is built once at load time;
nothing is allocated lazily on first call.

The pure-torch helpers here (id loading, quantized-row subsetting, index mapping) have CPU tests in
tests/v1/spec_decode/test_draft_vocab.py.
"""

from __future__ import annotations

import json
import os

import torch

ENV_PATH = "VLLM_MTP_DRAFT_VOCAB"
ENV_SELFCHECK = "VLLM_MTP_DRAFT_VOCAB_SELFCHECK"  # default on
PAD_MULTIPLE = 256  # Marlin needs N % 64 == 0; zero-point packing needs N % 8 == 0


def enabled() -> bool:
    return bool(os.environ.get(ENV_PATH))


def load_draft_vocab_ids(path: str, vocab_size: int) -> torch.Tensor:
    """Read a JSON list (or newline-separated text) of token ids -> sorted unique int64 [K],
    padded with the smallest unused ids up to a multiple of PAD_MULTIPLE."""
    with open(path) as f:
        raw = f.read()
    try:
        vals = json.loads(raw)
        if isinstance(vals, dict):
            vals = vals.get("ids") or vals.get("token_ids")
    except json.JSONDecodeError:
        vals = [int(x) for x in raw.split()]
    return normalize_ids(vals, vocab_size)


def normalize_ids(vals, vocab_size: int) -> torch.Tensor:
    ids = torch.as_tensor(sorted({int(v) for v in vals}), dtype=torch.int64)
    if ids.numel() == 0:
        raise ValueError("draft vocab list is empty")
    if int(ids[0]) < 0 or int(ids[-1]) >= vocab_size:
        raise ValueError(
            f"draft vocab ids must be in [0, {vocab_size}); got [{int(ids[0])}, {int(ids[-1])}]"
        )
    n = ids.numel()
    pad = (-n) % PAD_MULTIPLE
    if pad:
        if n + pad > vocab_size:
            raise ValueError("draft vocab larger than the vocabulary after padding")
        present = torch.zeros(vocab_size, dtype=torch.bool)
        present[ids] = True
        filler = torch.nonzero(~present).flatten()[:pad]
        ids = torch.sort(torch.cat([ids, filler])).values
    return ids


def map_subset_argmax(sub_logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """argmax over subset logits [rows, K] -> full-vocab token ids [rows] (ids: int64 [K])."""
    return ids[sub_logits.argmax(dim=-1)]


def subset_quantized(t: dict[str, torch.Tensor], ids: torch.Tensor, group: int = 128):
    """Row-subset a compressed-tensors pack-quantized head (u2_headquant layout).

    Quantization is per output row / input group, so selecting rows of the quantized tensors is
    bit-identical to quantizing the selected rows.  ``weight_zero_point`` is packed 8 rows per
    int32 along dim 0, so it is unpacked, row-selected and repacked.
    """
    from vllm.model_executor.layers.quantization import u2_headquant as u2

    packed, scale, zpp = t["weight_packed"], t["weight_scale"], t["weight_zero_point"]
    out_f, in_f = (int(x) for x in t["weight_shape"])
    sh = u2._SHIFTS.to(torch.int32)
    zp = ((zpp.unsqueeze(1) >> sh.view(1, 8, 1)) & 15).reshape(out_f, -1)  # [out, ng] 0..15
    K = int(ids.numel())
    assert K % 8 == 0, K
    zp_sub = zp[ids]
    zpp_sub = u2._pack_nibbles_last(zp_sub.t().contiguous()).t().contiguous()  # [K/8, ng]
    return {
        "weight_packed": packed[ids].contiguous(),
        "weight_scale": scale[ids].contiguous(),
        "weight_zero_point": zpp_sub,
        "weight_shape": torch.tensor([K, in_f], dtype=torch.int64),
    }


class DraftVocabHead(torch.nn.Module):
    """Replicated top-K head. ``layer`` is a ReplicatedLinear (quantized or plain) of [K, hidden]."""

    def __init__(self, layer: torch.nn.Module, ids: torch.Tensor):
        super().__init__()
        self.layer = layer
        self.register_buffer("ids", ids.to(torch.int64), persistent=False)

    @torch.inference_mode()
    def sub_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.layer.quant_method.apply(self.layer, hidden_states, None)

    @torch.inference_mode()
    def greedy(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return map_subset_argmax(self.sub_logits(hidden_states), self.ids)


def build_draft_vocab_head(
    path: str,
    target_lm_head: torch.nn.Module,
    quant_config,
    model_dir: str,
    hidden_size: int,
    vocab_size: int,
    dtype: torch.dtype,
    device: torch.device,
    logger,
) -> DraftVocabHead | None:
    """Build the replicated subset head from the target lm_head's source weights. Returns None
    (feature off, full-vocab path unchanged) on any problem -- never breaks boot."""
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.layers.quantization import u2_headquant as u2

    ids = load_draft_vocab_ids(path, vocab_size)
    K = int(ids.numel())
    quantized = type(target_lm_head.quant_method).__name__ not in (
        "UnquantizedEmbeddingMethod",
        "UnquantizedLinearMethod",
    )
    if quantized:
        full = u2.load_cached(model_dir, "lm_head.weight")
        if full is None:
            logger.warning("draft vocab: u2 cache for lm_head not found in %s; feature off", model_dir)
            return None
        sub = subset_quantized(full, ids)
        layer = ReplicatedLinear(
            hidden_size, K, bias=False, params_dtype=dtype, quant_config=quant_config,
            prefix="draft_vocab_head", return_bias=False,
        )
        for name, tensor in sub.items():
            param = getattr(layer, name)
            param.weight_loader(param, tensor) if hasattr(param, "weight_loader") else param.data.copy_(tensor)
    else:
        layer = ReplicatedLinear(
            hidden_size, K, bias=False, params_dtype=dtype, quant_config=None,
            prefix="draft_vocab_head", return_bias=False,
        )
        layer.weight.data.copy_(target_lm_head.weight.data[:vocab_size].to("cpu")[ids].to(device, dtype))
    layer.to(device)
    layer.quant_method.process_weights_after_loading(layer)
    logger.info("draft vocab: built %s head K=%d (%.1f%% of vocab) from %s",
                "int4" if quantized else "plain", K, 100.0 * K / vocab_size, path)
    return DraftVocabHead(layer, ids.to(device))


def selfcheck(head: DraftVocabHead, full_logits_fn, hidden_size: int, dtype, device, logger) -> bool:
    """Compare subset logits to the target's full logits restricted to the subset on a fixed random
    batch (runs eagerly at load, all TP ranks symmetric).  Returns True on PASS."""
    g = torch.Generator(device="cpu").manual_seed(1234)
    h = (torch.randn(16, hidden_size, generator=g) * 0.5).to(device, dtype)
    full = full_logits_fn(h)
    sub = head.sub_logits(h).float()
    ref = full.float()[:, head.ids]
    diff = (sub - ref).abs().max().item()
    scale = ref.abs().max().item() + 1.0
    ok = bool(diff <= 0.03 * scale)
    import json as _j
    logger.info("draft vocab selfcheck %s", _j.dumps({"pass": ok, "max_abs_diff": round(diff, 4), "logit_scale": round(scale, 3)}))
    return ok
