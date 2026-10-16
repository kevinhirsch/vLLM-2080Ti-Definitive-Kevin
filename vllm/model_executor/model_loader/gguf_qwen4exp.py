# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp (Qwen3.8-Flash-Next) GGUF -> vLLM weight plan.

llama.cpp's converter (``conversion/qwen4exp.py`` + the Qwen3.5 base classes) changes tensors on the way into
GGUF. The plan below undoes each change so the vLLM model sees exactly the HF checkpoint semantics:

* Gemma-style norms are stored as ``w + 1`` (every ``*norm.weight`` except the GDN ``linear_attn.norm``, plus the
  PLE and indexer norms explicitly). vLLM's ``GemmaRMSNorm`` adds the 1 at runtime -> subtract 1.
* ``ssm_a`` stores ``-exp(A_log)`` -> ``A_log = log(-ssm_a)``.
* Linear-attention V heads are reordered from grouped-by-K-head ``[k0v0, k0v1, k0v2, k1v0, ..]`` to tiled
  ``[k0v0, k1v0, .., k0v1, k1v1, ..]`` in ``in_proj_qkv`` (V rows), ``in_proj_z``, ``in_proj_a/b``, ``A_log``,
  ``dt_bias``, the V channels of ``conv1d`` and the input columns of ``out_proj``. Rows of a quantized matrix can
  be permuted byte-row-wise; ``out_proj`` columns cannot (quant blocks run along them), so ``ssm_out`` is
  dequantized on the host, permuted, and loaded as an fp16 weight.
* The indexer ``index_qk_proj`` is split into ``indexer.q_proj`` / ``indexer.k_proj`` -> concatenated back.
* conv1d weights are squeezed to ``[C, K]`` -> unsqueezed to ``[C, 1, K]``.
* The PLE hash constants live in KV metadata (int64), not tensors.

Everything float (F32/F16/BF16) is fed through the model's own ``load_weights`` under HF names; quantized
tensors go straight into the GGUF quant-method parameters (``GGUFLinearMethod`` / ``GGUFEmbeddingMethod`` /
``GGUFMoEMethod``) through :func:`quant_specs` and the loader in ``gguf_loader.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from vllm.model_executor.model_loader.gguf_reader import GGML_TYPES, GGUFModel

F32, F16, BF16 = 0, 1, 30
FLOAT_TYPES = (F32, F16, BF16)

# transform tags (applied in this order by ``float_tensor``)
NORM_MINUS1 = "norm_minus1"
A_LOG = "a_log"
VPERM_ROWS = "vperm_rows"  # rows grouped by V head (head size in ``vperm_head``)
VPERM_CONV = "vperm_conv"  # conv1d channels: q|k untouched, V part permuted
UNSQUEEZE1 = "unsqueeze1"  # [C, K] -> [C, 1, K]
VPERM_COLS = "vperm_cols"  # input columns grouped by V head (after host dequant)
ROW_VECTOR = "row_vector"  # [N] -> [1, N]


@dataclass
class TensorPlan:
    gguf_name: str
    kind: str  # "hf" | "dequant_hf" | "qlinear" | "qembed" | "qmoe" | "ple_table" | "concat_hf"
    target: str  # HF name (hf kinds) or vLLM module prefix (q kinds)
    ggml_type: int
    shape: tuple[int, ...]  # torch order
    shards: tuple[int, ...] = ()  # qlinear: logical output shards covered, in order
    moe_part: str = ""  # qmoe: "w1" | "w3" | "w2"
    transforms: tuple[str, ...] = ()
    vperm_head: int = 0  # head size for VPERM_ROWS / VPERM_COLS


@dataclass
class LinearSpec:
    shard_types: dict[int, int] = field(default_factory=dict)  # logical shard -> ggml type


@dataclass
class EmbedSpec:
    ggml_type: int


@dataclass
class MoESpec:
    w13_type: int
    w2_type: int


def vhead_grouped_from_tiled(num_k_heads: int, num_v_heads: int) -> np.ndarray:
    """idx with ``grouped[g] = tiled[idx[g]]`` for V-head blocks (llama.cpp _reorder_v_heads inverse)."""
    r = num_v_heads // num_k_heads
    g = np.arange(num_v_heads)
    return (g % r) * num_k_heads + g // r


def _layer_kinds(kv: dict) -> list[str]:
    n = kv["qwen4exp.block_count"]
    ratios = kv["qwen4exp.attention.compress_ratios"]
    interval = kv["qwen4exp.full_attention_interval"]
    kinds = ["full" if (i + 1) % interval == 0 else "linear" for i in range(n)]
    for i, k in enumerate(kinds):
        if (ratios[i] > 0) != (k == "full"):
            raise ValueError(f"layer {i}: compress ratio {ratios[i]} disagrees with layer type {k}")
    return kinds


_BLK = re.compile(r"^blk\.(\d+)\.(.+)$")


def plan(model: GGUFModel) -> list[TensorPlan]:
    """One TensorPlan per GGUF tensor (raises on any tensor it does not understand)."""
    kv = model.kv
    if kv.get("general.architecture") != "qwen4exp":
        raise ValueError(f"not a qwen4exp GGUF: {kv.get('general.architecture')!r}")
    kinds = _layer_kinds(kv)
    ple_layers = set(kv.get("qwen4exp.ple.layers", []))
    v_head = kv["qwen4exp.ssm.inner_size"] // kv["qwen4exp.ssm.time_step_rank"]
    out: list[TensorPlan] = []

    def add(name, kind, target, **kw):
        info = model.tensors[name]
        out.append(TensorPlan(name, kind, target, info.ggml_type, info.shape, **kw))

    hc_mod = {"hc_attn": "attn_hyper_connection", "hc_ffn": "mlp_hyper_connection"}
    hc_part = {"down": "input_mix_weight_down", "up": "input_mix_weight_up", "inject": "block_inject_weight"}
    for name, info in model.tensors.items():
        if name == "token_embd.weight":
            add(name, "qembed" if not info.is_float else "hf", "model.embed_tokens"
                if not info.is_float else "model.embed_tokens.weight")
            continue
        if name == "output.weight":
            add(name, "qembed" if not info.is_float else "hf", "lm_head"
                if not info.is_float else "lm_head.weight")
            continue
        if name == "per_layer_token_embd.weight":
            add(name, "ple_table", "")
            continue
        m = re.match(r"^output_hc_(down|up|norm)\.weight$", name)
        if m:
            part = m.group(1)
            hf = "model.hyper_connection_mixer." + (
                "hc_norm" if part == "norm" else hc_part[part]) + ".weight"
            add(name, "hf", hf, transforms=(NORM_MINUS1,) if part == "norm" else ())
            continue
        m = _BLK.match(name)
        if not m:
            raise ValueError(f"unmapped GGUF tensor {name!r}")
        n, rest = int(m.group(1)), m.group(2)
        L = f"model.layers.{n}"
        m2 = re.match(r"^(hc_attn|hc_ffn)_(down|up|inject|norm)\.weight$", rest)
        if m2:
            mod, part = hc_mod[m2.group(1)], m2.group(2)
            hf = f"{L}.{mod}." + ("hc_norm" if part == "norm" else hc_part[part]) + ".weight"
            add(name, "hf", hf, transforms=(NORM_MINUS1,) if part == "norm" else ())
            continue
        # ---- MoE + shared expert (every layer) ----
        moe = {
            "ffn_gate_inp.weight": ("hf", f"{L}.mlp.gate.weight", ()),
            "ffn_gate_inp_shexp.weight": ("hf", f"{L}.mlp.shared_expert_gate.weight", (ROW_VECTOR,)),
        }
        if rest in moe:
            kind, hf, tr = moe[rest]
            add(name, kind, hf, transforms=tr)
            continue
        m2 = re.match(r"^ffn_(gate|up|down)_shexp\.weight$", rest)
        if m2:
            p = m2.group(1)
            if info.is_float:
                add(name, "hf", f"{L}.mlp.shared_expert.{p}_proj.weight")
            elif p == "down":
                add(name, "qlinear", f"{L}.mlp.shared_expert.down_proj", shards=(0,))
            else:
                add(name, "qlinear", f"{L}.mlp.shared_expert.gate_up_proj", shards=(0 if p == "gate" else 1,))
            continue
        m2 = re.match(r"^ffn_(gate|up|down)_exps\.weight$", rest)
        if m2:
            if info.is_float:
                raise NotImplementedError(f"float routed experts are not supported: {name}")
            add(name, "qmoe", f"{L}.mlp.experts", moe_part={"gate": "w1", "up": "w3", "down": "w2"}[m2.group(1)])
            continue
        # ---- PLE ----
        if rest.startswith("ple_"):
            if n not in ple_layers:
                raise ValueError(f"{name}: layer {n} is not a PLE layer {sorted(ple_layers)}")
            sub = rest[len("ple_"):]
            ple = {
                "key.weight": ("ple.key_proj.weight", ()),
                "value.weight": ("ple.value_proj.weight", ()),
                "norm_key.weight": ("ple.norm_key.weight", (NORM_MINUS1,)),
                "norm_query.weight": ("ple.norm_query.weight", (NORM_MINUS1,)),
                "norm_conv.weight": ("ple.norm_conv.weight", (NORM_MINUS1,)),
                "conv1d.weight": ("ple.conv1d.weight", (UNSQUEEZE1,)),
            }
            if sub not in ple or not info.is_float:
                raise ValueError(f"unexpected PLE tensor {name} ({info.type_name})")
            hf, tr = ple[sub]
            add(name, "hf", f"{L}.{hf}", transforms=tr)
            continue
        # ---- full attention (QSA) ----
        if kinds[n] == "full":
            att = {
                "attn_q.weight": (0,),
                "attn_k.weight": (1,),
                "attn_v.weight": (2,),
            }
            if rest in att:
                if info.is_float:
                    raise NotImplementedError(f"float attention projection {name}")
                add(name, "qlinear", f"{L}.self_attn.qkv_proj", shards=att[rest])
                continue
            if rest == "attn_output.weight":
                add(name, "qlinear" if not info.is_float else "hf",
                    f"{L}.self_attn.o_proj" if not info.is_float else f"{L}.self_attn.o_proj.weight",
                    shards=(0,) if not info.is_float else ())
                continue
            norms = {
                "attn_q_norm.weight": "self_attn.q_norm.weight",
                "attn_k_norm.weight": "self_attn.k_norm.weight",
                "indexer.q_norm.weight": "self_attn.indexer.q_layernorm.weight",
                "indexer.k_norm.weight": "self_attn.indexer.k_layernorm.weight",
            }
            if rest in norms:
                add(name, "hf", f"{L}.{norms[rest]}", transforms=(NORM_MINUS1,))
                continue
            if rest in ("indexer.q_proj.weight", "indexer.k_proj.weight"):
                # concatenated (q rows first) by the loader into index_qk_proj
                add(name, "concat_hf", f"{L}.self_attn.indexer.index_qk_proj.weight",
                    shards=(0 if rest.startswith("indexer.q") else 1,))
                continue
            raise ValueError(f"unmapped attention tensor {name!r}")
        # ---- linear attention (GDN) ----
        G = f"{L}.linear_attn"
        if rest == "attn_qkv.weight":
            # q | k | v rows; only the V rows are head-reordered
            add(name, "qlinear" if not info.is_float else "hf",
                f"{G}.in_proj_qkvz" if not info.is_float else f"{G}.in_proj_qkv.weight",
                shards=(0, 1, 2) if not info.is_float else (), transforms=(VPERM_ROWS,), vperm_head=v_head)
            continue
        if rest == "attn_gate.weight":
            add(name, "qlinear" if not info.is_float else "hf",
                f"{G}.in_proj_qkvz" if not info.is_float else f"{G}.in_proj_z.weight",
                shards=(3,) if not info.is_float else (), transforms=(VPERM_ROWS,), vperm_head=v_head)
            continue
        gdn = {
            "ssm_alpha.weight": ("in_proj_a.weight", (VPERM_ROWS,), 1),
            "ssm_beta.weight": ("in_proj_b.weight", (VPERM_ROWS,), 1),
            "ssm_a": ("A_log", (A_LOG, VPERM_ROWS), 1),
            "ssm_dt.bias": ("dt_bias", (VPERM_ROWS,), 1),
            "ssm_conv1d.weight": ("conv1d.weight", (VPERM_CONV, UNSQUEEZE1), v_head),
            "ssm_norm.weight": ("norm.weight", (), 0),
        }
        if rest in gdn:
            hf, tr, vh = gdn[rest]
            if not info.is_float:
                raise ValueError(f"unexpected quantized GDN tensor {name}")
            add(name, "hf", f"{G}.{hf}", transforms=tr, vperm_head=vh)
            continue
        if rest == "ssm_out.weight":
            add(name, "dequant_hf" if not info.is_float else "hf", f"{G}.out_proj.weight",
                transforms=(VPERM_COLS,), vperm_head=v_head)
            continue
        raise ValueError(f"unmapped GGUF tensor {name!r}")
    return out


def quant_specs(model: GGUFModel, plans: list[TensorPlan] | None = None) -> dict[str, object]:
    """vLLM module prefix -> LinearSpec | EmbedSpec | MoESpec for every quantized module."""
    specs: dict[str, object] = {}
    moe: dict[str, dict[str, int]] = {}
    for p in plans if plans is not None else plan(model):
        if p.kind == "qlinear":
            spec = specs.setdefault(p.target, LinearSpec())
            assert isinstance(spec, LinearSpec)
            for s in p.shards:
                if s in spec.shard_types:
                    raise ValueError(f"{p.target}: shard {s} mapped twice")
                spec.shard_types[s] = p.ggml_type
        elif p.kind == "qembed":
            specs[p.target] = EmbedSpec(p.ggml_type)
        elif p.kind == "qmoe":
            moe.setdefault(p.target, {})[p.moe_part] = p.ggml_type
    for target, parts in moe.items():
        if set(parts) != {"w1", "w2", "w3"}:
            raise ValueError(f"{target}: incomplete expert tensors {sorted(parts)}")
        if parts["w1"] != parts["w3"]:
            raise ValueError(f"{target}: gate/up experts must share one ggml type, got {parts}")
        specs[target] = MoESpec(parts["w1"], parts["w2"])
    return specs


def float_tensor(model: GGUFModel, p: TensorPlan, num_k_heads: int, num_v_heads: int):
    """Load + transform a float GGUF tensor into the HF-semantics torch tensor (fp32/fp16/bf16 kept)."""
    import torch

    arr = np.asarray(model.array(p.gguf_name))
    if p.ggml_type == BF16:
        t = torch.from_numpy(arr.copy()).view(torch.bfloat16)
    else:
        t = torch.from_numpy(arr.copy())
    return apply_transforms(t, p, num_k_heads, num_v_heads)


def apply_transforms(t, p: TensorPlan, num_k_heads: int, num_v_heads: int):
    import torch

    for tr in p.transforms:
        if tr == NORM_MINUS1:
            t = (t.float() - 1.0).to(t.dtype)
        elif tr == A_LOG:
            t = torch.log(-t.float()).to(t.dtype)
        elif tr == VPERM_ROWS:
            t = permute_v_rows(t, num_k_heads, num_v_heads, p.vperm_head, p.gguf_name)
        elif tr == VPERM_CONV:
            vdim = num_v_heads * p.vperm_head
            qk, v = t[: t.shape[0] - vdim], t[t.shape[0] - vdim:]
            idx = torch.from_numpy(vhead_grouped_from_tiled(num_k_heads, num_v_heads))
            v = v.reshape(num_v_heads, p.vperm_head, *v.shape[1:])[idx].reshape(vdim, *v.shape[1:])
            t = torch.cat([qk, v], 0)
        elif tr == UNSQUEEZE1:
            t = t.unsqueeze(1)
        elif tr == VPERM_COLS:
            idx = torch.from_numpy(vhead_grouped_from_tiled(num_k_heads, num_v_heads))
            rows = t.shape[0]
            t = t.reshape(rows, num_v_heads, p.vperm_head)[:, idx].reshape(rows, -1)
        elif tr == ROW_VECTOR:
            t = t.reshape(1, -1)
        else:
            raise ValueError(f"unknown transform {tr}")
    return t.contiguous()


def permute_v_rows(t, num_k_heads: int, num_v_heads: int, head: int, name: str = ""):
    """Rows: [prefix | V part], the V part (last num_v_heads*head rows) goes tiled -> grouped."""
    import torch

    vdim = num_v_heads * head
    if t.shape[0] < vdim:
        raise ValueError(f"{name}: {t.shape[0]} rows < V rows {vdim}")
    idx = torch.from_numpy(vhead_grouped_from_tiled(num_k_heads, num_v_heads))
    pre, v = t[: t.shape[0] - vdim], t[t.shape[0] - vdim:]
    v = v.reshape(num_v_heads, head, *v.shape[1:])[idx].reshape(vdim, *v.shape[1:])
    return torch.cat([pre, v], 0) if pre.shape[0] else v


def quant_rows(model: GGUFModel, p: TensorPlan, num_k_heads: int, num_v_heads: int):
    """Quantized 2-D tensor as uint8 [rows, row_bytes] torch tensor (zero-copy view of the mmap when no
    transform is needed), with V-row permutation applied when planned."""
    import torch

    be, bb = GGML_TYPES[p.ggml_type][1:]
    rows = int(np.prod(p.shape[:-1]))
    row_bytes = p.shape[-1] // be * bb
    raw = model.raw(p.gguf_name)
    t = torch.from_numpy(np.asarray(raw).reshape(*p.shape[:-1], row_bytes))
    if VPERM_ROWS in p.transforms:
        t = permute_v_rows(t.reshape(rows, row_bytes), num_k_heads, num_v_heads, p.vperm_head, p.gguf_name)
    return t


def ple_constants(model: GGUFModel) -> dict[str, list[int]]:
    kv = model.kv
    return {
        "layer_multipliers": list(kv["qwen4exp.ple.layer_multipliers"]),
        "ngram_heads_offsets": list(kv["qwen4exp.ple.head_offsets"]),
        "ngram_heads_vocab_sizes": list(kv["qwen4exp.ple.head_vocab_sizes"]),
    }
