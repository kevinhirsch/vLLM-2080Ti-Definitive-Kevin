#!/usr/bin/env python3
"""K5 engine-path equivalence: fused GDN MTP decode vs the production non-fused path (GPU, engine stopped).

Drives the REAL layer methods of QwenGatedDeltaNetAttention on one fake layer per path:
  - K5:   _forward_cuda_impl with VLLM_K5_GDN_FUSED on (-> _k5_spec_decode: guarded conv + fused kernel + out_proj)
  - prod: the non-fused branch of _forward_cuda_impl (split, split_ba, _forward_core = conv1d update + Triton
          fused_sigmoid_gating_delta_rule_update, _output_projection = RMSNormGated + out_proj)
on identical random inputs and identical random state, at the production per-TP-rank shapes (TP=2: H=8, HV=24,
K=V=128, conv dim 5120, kernel 4, conv state 6 x 5120 fp16 in SD layout, temporal state 24x128x128 fp32, both
views into one shared page per slot), MTP3 (4 slots per request), replicated Marlin ba (96 cols) with the rank-1
head slice, mamba align-mode style slot rows (source slot = one of the destinations), random accepted counts,
graph-padded request rows (-1 slots, accepted=1, cu filled) and token-padded rows.

The K5 layer's FIRST call happens inside torch.cuda.graph capture with capture-time metadata (all requests real,
all accepted), exactly like the engine (capture warmups carry no spec-decode metadata); the real metadata is then
written into the same persistent buffers before every replay. This is the configuration that failed in window
k5-fused-gdn-aba-20261013-075027 (lazy fp32 parameter copies created inside capture were never initialised).

Compares, for every step: the out_proj input (normed core output), the projected output, ALL temporal-state slots,
ALL conv-state slots. Prints one JSON line per case and a final {"verdict": "PASS"|"FAIL", ...} line.
Exit 0 only on PASS. Needs ~1 GB of GPU memory. Run with PYTHONPATH=<release root>.
"""
from __future__ import annotations

import json
import sys
import types

import torch

from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.mamba.gdn import gdn_sm75_cuda as k5
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as Q
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

DEV = torch.device("cuda")
TP = 2
NK, NV, DK, DV, KW = 16, 48, 128, 128, 4  # Qwen3.8-27B GDN (global heads)
H, HV = NK // TP, NV // TP
QKV = (2 * NK * DK + NV * DV) // TP  # 5120 per rank
ZW = NV * DV // TP  # 3072
HIDDEN = 5120
W = 4  # num_spec + 1
CONV_LEN = KW - 1 + W - 1  # 6
NSLOTS = 48
PAD = -1

CTX = {"md": None}
Q.get_forward_context = lambda: types.SimpleNamespace(attn_metadata={"layer.k5": CTX["md"]})
CALLS = {"k5": 0}
_real_gdn_mtp = k5.gdn_mtp


def _counting_gdn_mtp(*args, **kwargs):
    CALLS["k5"] += 1
    return _real_gdn_mtp(*args, **kwargs)


k5.gdn_mtp = _counting_gdn_mtp


def make_norm(weight: torch.Tensor) -> torch.nn.Module:
    try:
        norm = RMSNormGated(DV, eps=1e-6, group_size=None, norm_before_gate=True, activation="silu",
                            device=DEV, dtype=weight.dtype)
    except Exception:  # CustomOp needs a vLLM config in some builds
        from vllm.config import VllmConfig, set_current_vllm_config

        with set_current_vllm_config(VllmConfig()):
            norm = RMSNormGated(DV, eps=1e-6, group_size=None, norm_before_gate=True, activation="silu",
                                device=DEV, dtype=weight.dtype)
    with torch.no_grad():
        norm.weight.copy_(weight)
    return norm


def make_params(g: torch.Generator, param_dtype: torch.dtype):
    r = lambda *s, scale=1.0, dt=torch.float32: (torch.randn(*s, generator=g, device=DEV) * scale).to(dt)
    return dict(
        A_log=r(HV, scale=0.5),
        dt_bias=r(HV, dt=param_dtype),
        conv_w=r(QKV, 1, KW, scale=0.4, dt=torch.float16),
        norm_w=(1 + 0.1 * torch.randn(DV, generator=g, device=DEV)).to(param_dtype),
        out_w=r(ZW, HIDDEN, scale=ZW ** -0.5, dt=torch.float16),
    )


def make_cache(g: torch.Generator, state_dtype: torch.dtype, conv_dim_first: bool):
    conv_bytes = CONV_LEN * QKV * 2
    ssm_bytes = HV * DV * DK * (4 if state_dtype == torch.float32 else 2)
    page = conv_bytes + ssm_bytes + 64  # slot pages share conv + temporal state, like the hybrid allocator
    raw = torch.zeros(NSLOTS, page, dtype=torch.uint8, device=DEV)
    conv = raw[:, :conv_bytes].view(torch.float16)
    conv = conv.view(NSLOTS, QKV, CONV_LEN) if conv_dim_first else conv.view(NSLOTS, CONV_LEN, QKV)
    ssm = raw[:, conv_bytes:conv_bytes + ssm_bytes].view(state_dtype).view(NSLOTS, HV, DV, DK)
    conv.copy_((torch.randn(conv.shape, generator=g, device=DEV) * 0.5).half())
    ssm.copy_((torch.randn(ssm.shape, generator=g, device=DEV) * 0.05).to(state_dtype))
    return raw, conv, ssm


def make_layer(p, conv, ssm, tp_rank: int, core_buf: torch.Tensor) -> Q.QwenGatedDeltaNetAttention:
    L = Q.QwenGatedDeltaNetAttention.__new__(Q.QwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(L)
    for k, v in dict(prefix="layer.k5", tp_size=TP, tp_rank=tp_rank, num_k_heads=NK, num_v_heads=NV,
                     head_k_dim=DK, head_v_dim=DV, key_dim=NK * DK, value_dim=NV * DV, hidden_size=HIDDEN,
                     conv_kernel_size=KW, activation="silu", gqa_interleaved_layout=False,
                     disable_tp_for_ba_proj=True, enable_fused_gdn_decode=False,
                     enable_packed_recurrent_decode=False, kv_cache=(conv, ssm)).items():
        object.__setattr__(L, k, v)
    L.A_log = torch.nn.Parameter(p["A_log"].clone(), requires_grad=False)
    L.dt_bias = torch.nn.Parameter(p["dt_bias"].clone(), requires_grad=False)
    L.conv1d = types.SimpleNamespace(weight=p["conv_w"].clone(), bias=None)
    L.norm = make_norm(p["norm_w"])
    out_w = p["out_w"]

    def out_proj(x):
        core_buf.copy_(x)  # the out_proj input (normed core output); a captured copy under graph replay too
        return x @ out_w, None

    object.__setattr__(L, "out_proj", out_proj)
    return L


def prod_forward(L, mixed_qkvz, ba, output):
    """The non-fused branch of QwenGatedDeltaNetAttention._forward_cuda_impl (Qwen3.5 layout), real methods."""
    qkv_size = (L.key_dim * 2 + L.value_dim) // L.tp_size
    z_size = L.value_dim // L.tp_size
    mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
    z = z.reshape(z.size(0), -1, L.head_v_dim)
    b, a = L.split_ba(ba)
    core = torch.zeros((mixed_qkvz.size(0), L.num_v_heads // L.tp_size, L.head_v_dim),
                       dtype=mixed_qkvz.dtype, device=mixed_qkvz.device)
    L._forward_core(mixed_qkv, b.contiguous(), a.contiguous(), core)
    output.copy_(L._output_projection(core, z))


def k5_forward(L, qkvz_buf, ba_buf, output):
    L.in_proj_qkvz = lambda h: (qkvz_buf, None)
    L.in_proj_ba = lambda h: (ba_buf, None)
    L._forward_cuda_impl(qkvz_buf, output)


class Meta:
    """Persistent metadata buffers (the GDN builder's FULL-graph buffers) + a GDNAttentionMetadata view."""

    def __init__(self, ncap: int):
        self.ncap = ncap
        self.sidx = torch.full((ncap, W), PAD, dtype=torch.int32, device=DEV)
        self.cu = torch.zeros(ncap + 1, dtype=torch.int32, device=DEV)
        self.nacc = torch.ones(ncap, dtype=torch.int32, device=DEV)
        self.masks = torch.zeros(ncap, dtype=torch.bool, device=DEV)
        self.md = GDNAttentionMetadata(
            num_prefills=0, num_prefill_tokens=0, num_decodes=0, num_decode_tokens=0,
            num_spec_decodes=ncap, num_spec_decode_tokens=ncap * W, num_actual_tokens=ncap * W,
            spec_query_start_loc=self.cu, spec_state_indices_tensor=self.sidx,
            spec_sequence_masks=self.masks, num_accepted_tokens=self.nacc,
            spec_token_indx=torch.arange(ncap * W, dtype=torch.int32, device=DEV),
            non_spec_token_indx=torch.empty(0, dtype=torch.int32, device=DEV))

    def capture_values(self, rows):
        self.load(rows, [W] * len(rows))

    def load(self, rows, accepted):
        n = len(rows)
        self.sidx.fill_(PAD)
        self.sidx[:n] = torch.tensor(rows, dtype=torch.int32, device=DEV)
        self.nacc.fill_(1)
        self.nacc[:n] = torch.tensor(accepted, dtype=torch.int32, device=DEV)
        cu = torch.arange(self.ncap + 1, dtype=torch.int32, device=DEV).clamp_(max=n) * W
        self.cu.copy_(cu)
        self.masks.zero_()
        self.masks[:n] = True


def rel(a, b):
    d = (a.float() - b.float()).abs().max().item()
    return d, d / max(b.float().abs().max().item(), 1e-6)


def run_case(name, *, state_dtype, param_dtype, tp_rank, ncap, real_n, graph, conv_dim_first, steps, seed):
    Q.is_conv_state_dim_first = lambda: conv_dim_first
    g = torch.Generator(device=DEV).manual_seed(seed)
    p = make_params(g, param_dtype)
    _, conv_r, ssm_r = make_cache(g, state_dtype, conv_dim_first)
    raw_k, conv_k, ssm_k = make_cache(torch.Generator(device=DEV).manual_seed(seed), state_dtype, conv_dim_first)
    conv_k.copy_(conv_r)
    ssm_k.copy_(ssm_r)
    core_r = torch.full((ncap * W, ZW), float("nan"), dtype=torch.float16, device=DEV)
    core_k = torch.full((ncap * W, ZW), float("nan"), dtype=torch.float16, device=DEV)
    Lr = make_layer(p, conv_r, ssm_r, tp_rank, core_r)
    Lk = make_layer(p, conv_k, ssm_k, tp_rank, core_k)
    perm = torch.randperm(NSLOTS, generator=torch.Generator().manual_seed(seed)).tolist()
    perm.remove(0)
    perm.insert(0, 0)  # slot 0 is a valid GDN slot (null id is -1); make request 0 own it
    rows = [perm[i * W:(i + 1) * W] for i in range(ncap)]
    meta = Meta(ncap)
    ntok = ncap * W
    qkvz_k = torch.empty(ntok, QKV + ZW, dtype=torch.float16, device=DEV)
    ba_k = torch.empty(ntok, 2 * NV, dtype=torch.float16, device=DEV)
    out_k = torch.empty(ntok, HIDDEN, dtype=torch.float16, device=DEV)
    k5._ENABLED = True
    graph_obj = None
    if graph:
        CTX["md"] = meta.md
        meta.capture_values(rows)
        qkvz_k.normal_(generator=g)
        ba_k.normal_(generator=g)
        torch.cuda.synchronize()
        graph_obj = torch.cuda.CUDAGraph()
        before = CALLS["k5"]
        with torch.cuda.graph(graph_obj):
            k5_forward(Lk, qkvz_k, ba_k, out_k)  # FIRST call of this layer = inside capture (as in the engine)
        if CALLS["k5"] == before:
            return dict(case=name, ok=False, error="K5 path not taken during capture")
        torch.cuda.synchronize()
        # capture must not have mutated anything
        if not (torch.equal(ssm_k, ssm_r) and torch.equal(conv_k, conv_r)):
            return dict(case=name, ok=False, error="capture mutated state")
    worst = dict(out_rel=0.0, core_rel=0.0, ssm_abs=0.0, conv_abs=0.0)
    ok = True
    for step in range(steps):
        accepted = torch.randint(1, W + 1, (real_n,), generator=torch.Generator().manual_seed(seed * 100 + step))
        accepted = accepted.tolist()
        if step == 0:
            accepted[0] = W  # all accepted
        if step == 1 and real_n > 1:
            accepted[-1] = 1
        meta.load(rows[:real_n], accepted)
        CTX["md"] = meta.md
        x_qkvz = (torch.randn(ntok, QKV + ZW, generator=g, device=DEV) * 0.7).half()
        x_ba = torch.randn(ntok, 2 * NV, generator=g, device=DEV).half()
        # reference: production path on clones (conv1d writes post-conv qkv into its input in place)
        out_r = torch.empty(ntok, HIDDEN, dtype=torch.float16, device=DEV)
        k5._ENABLED = False
        before = CALLS["k5"]
        prod_forward(Lr, x_qkvz.clone(), x_ba.clone(), out_r)
        assert CALLS["k5"] == before
        k5._ENABLED = True
        qkvz_k.copy_(x_qkvz)
        ba_k.copy_(x_ba)
        core_k.fill_(float("nan"))
        if graph_obj is not None:
            graph_obj.replay()
        else:
            before = CALLS["k5"]
            k5_forward(Lk, qkvz_k, ba_k, out_k)
            if CALLS["k5"] == before:
                return dict(case=name, ok=False, error="K5 path not taken (eager)")
        torch.cuda.synchronize()
        # Production leaves graph-padded token rows uninitialised (the Triton wrapper's o = q.new_empty, rows with
        # T=0 are never stored); the engine discards them. Compare real rows; K5's padded rows must be exactly 0.
        rt = real_n * W
        core_d, core_rel = rel(core_k[:rt], core_r[:rt])
        out_d, out_rel = rel(out_k[:rt], out_r[:rt])
        ssm_d = (ssm_k.float() - ssm_r.float()).abs().max().item()
        smag = ssm_r.float().abs().max().item()
        conv_d = (conv_k.float() - conv_r.float()).abs().max().item()
        nan = bool(torch.isnan(out_k).any() or torch.isnan(core_k).any() or torch.isnan(ssm_k.float()).any()
                   or torch.isnan(conv_k.float()).any() or torch.isnan(out_r[:rt]).any())
        tol_s = (1e-3 * smag + 1e-5) if state_dtype == torch.float32 else (2e-3 * smag + 1e-3)
        pad_rows = max(out_k[rt:].float().abs().max().item(), core_k[rt:].float().abs().max().item()) if real_n < ncap else 0.0
        good = (not nan and out_rel < 1e-2 and core_rel < 1e-2 and ssm_d <= tol_s and conv_d <= 1e-3
                and pad_rows == 0.0)
        ok &= good
        for k_, v_ in (("out_rel", out_rel), ("core_rel", core_rel), ("ssm_abs", ssm_d), ("conv_abs", conv_d)):
            worst[k_] = max(worst[k_], v_)
        if not good:
            print(json.dumps(dict(case=name, step=step, accepted=accepted, out_maxabs=out_d, out_rel=out_rel,
                                  core_rel=core_rel, ssm_maxabs=ssm_d, ssm_tol=tol_s, conv_maxabs=conv_d,
                                  pad_rows_maxabs=pad_rows, nan=nan)), flush=True)
    del graph_obj
    return dict(case=name, ok=bool(ok), state=str(state_dtype).split(".")[-1],
                params=str(param_dtype).split(".")[-1], tp_rank=tp_rank, ncap=ncap, real_n=real_n, graph=graph,
                conv_dim_first=conv_dim_first, steps=steps, **{k_: float(f"{v_:.3g}") for k_, v_ in worst.items()})


def main() -> int:
    if not torch.cuda.is_available():
        print(json.dumps(dict(test="k5_equiv", verdict="FAIL", error="no CUDA device")))
        return 1
    k5._load()
    cases = [
        # name, state, params, rank, ncap, real_n, graph, conv_dim_first
        ("prod-c1-graph", torch.float32, torch.float16, 1, 1, 1, True, False),
        ("prod-c1-padded-graph", torch.float32, torch.float16, 1, 2, 1, True, False),
        ("prod-n3-of-4-graph", torch.float32, torch.float16, 0, 4, 3, True, False),
        ("prod-n5-of-8-graph-rank0", torch.float32, torch.float16, 0, 8, 5, True, False),
        ("prod-n3-eager", torch.float32, torch.float16, 1, 3, 3, False, False),
        ("fp32-params-graph", torch.float32, torch.float32, 1, 4, 2, True, False),
        ("fp16-state-graph", torch.float16, torch.float16, 1, 4, 3, True, False),
        ("ds-conv-layout-graph", torch.float32, torch.float16, 1, 4, 4, True, True),
    ]
    results, ok = [], True
    for i, (name, sdt, pdt, rank, ncap, real_n, graph, dsf) in enumerate(cases):
        try:
            r = run_case(name, state_dtype=sdt, param_dtype=pdt, tp_rank=rank, ncap=ncap, real_n=real_n,
                         graph=graph, conv_dim_first=dsf, steps=4, seed=1000 + i)
        except Exception as e:  # a crash is a FAIL, not a skip
            r = dict(case=name, ok=False, error=f"{type(e).__name__}: {e}")
        results.append(r)
        ok &= bool(r.get("ok"))
        print(json.dumps(r), flush=True)
    verdict = dict(test="k5_equiv", verdict="PASS" if ok else "FAIL", cases=len(results),
                   failed=[r["case"] for r in results if not r.get("ok")], k5_calls=CALLS["k5"])
    print(json.dumps(verdict), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
