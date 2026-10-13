# SPDX-License-Identifier: Apache-2.0
"""E16 P5: tree-step commit (GDN temporal slot, GDN conv column, TurboQuant KV row).

Contract: ``/home/kevin/projects/lanes/coexec/results/tree-contract.md`` §5 P5.

After the tree sampler's walk (P6) and BEFORE ``postprocess_mamba_align_gpu``,
for every request ``b`` that rescued leaf ``l`` at depth ``j`` (leaf row
``R = D + 1 + l``):

* GDN temporal state, all GDN layers, all v-heads:
  ``S[idx[b, j]] = update(S[idx[b, j-1]], stash.gdn_leaf[g, b, l])``
  (``tree_reference.gdn_commit_ref``);
* GDN conv state, all GDN layers:
  ``conv[idx[b, 0]][:, (Kw-2)+j] = stash.conv_leaf_x[g, b, l]``
  (``tree_reference.conv_commit_ref``);
* TurboQuant KV, all attention layers: the packed row of
  ``slot_mapping[b*M + j]`` := the packed row of ``slot_mapping[b*M + R]``
  (``tree_reference.gather_chain_view`` on slots).

After it, ``num_accepted = j + 1`` means exactly what it means for a chain.
One CUDA launch (``tree_commit_ext/tree_commit_sm75.cu``), masked on the
device by ``rescue_leaf >= 0``, so it is graph-safe and may run on every tree
step.  It is never called on chain steps.

Everything device-side that the kernel reads by address (layer tables) is
built ONCE in ``__init__`` (runner init, after KV-cache allocation, before
capture) -- never lazily on the first call (HACKATHON-RULES 08:4x).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from vllm.v1.spec_decode import tree_commit_loader as _L
from vllm.v1.spec_decode.tree_spec import TreeSpec, TreeStash, TreeStepOutputs

_KDIM = 128
_VDIM = 128
_PAD_SLOT_ID = -1



def _load():
    """The commit extension (stamped prebuilt import, else JIT build); the
    loader lives in tree_commit_loader (torch/os only, so release.py's
    --jit-ext prebuild can import it standalone)."""
    return _L._load()

@dataclass
class GDNCommitLayer:
    """One GDN layer as the commit sees it.

    name        the layer prefix, the key of ``TreeStash.layer_index``
    ssm_state   [slots, HV, V=128, K=128] fp32|fp16 (the layer's kv_cache[1])
    conv_state  [slots, C, state_len] view (the same view K5's glue passes:
                kv_cache[0], or kv_cache[0].transpose(-1, -2) in the default
                SD layout); fp16
    state_idx   PERSISTENT int32 [rows, D+1] buffer holding this step's
                ``spec_state_indices_tensor`` in its first B rows (the GDN
                metadata builder's ``spec_state_indices_tensor`` buffer of the
                layer's KV-cache group; the builder fills it on every tree step)
    """

    name: str
    ssm_state: torch.Tensor
    conv_state: torch.Tensor
    state_idx: torch.Tensor


def _gdn_row(layer: GDNCommitLayer, W: int) -> list[int]:
    s, c, i = layer.ssm_state, layer.conv_state, layer.state_idx
    if not (
        s.dim() == 4
        and tuple(s.shape[2:]) == (_VDIM, _KDIM)
        and tuple(s.stride()[1:]) == (_VDIM * _KDIM, _KDIM, 1)
        and s.dtype in (torch.float32, torch.float16)
    ):
        raise ValueError(f"tree commit: {layer.name}: ssm state must be [slots, HV, 128, 128]")
    if not (c.dim() == 3 and c.size(0) == s.size(0) and c.dtype == torch.float16):
        raise ValueError(f"tree commit: {layer.name}: conv state must be fp16 [slots, C, len]")
    if not (
        i.dim() == 2 and i.size(1) == W and i.dtype == torch.int32 and i.is_contiguous()
    ):
        raise ValueError(f"tree commit: {layer.name}: state_idx must be int32 [rows, D+1]")
    return [
        s.data_ptr(),
        s.stride(0),
        s.size(0),
        c.data_ptr(),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        i.data_ptr(),
    ]


def _tq_row(kv: torch.Tensor) -> list[int]:
    """kv: the TurboQuant layer cache in its logical [blocks, Hk, block_size,
    slot] form (any strides; LBNHC/BLHNC), uint8."""
    if not (kv.dim() == 4 and kv.element_size() == 1):
        raise ValueError("tree commit: TQ cache must be a 4-d byte tensor [B, H, N, C]")
    # Store addressing (triton_turboquant_store): slot -> (slot // N, slot % N)
    # on the [B, N, H, C] transpose.
    return [kv.data_ptr(), kv.stride(0), kv.stride(2), kv.stride(1), kv.size(2), kv.size(0)]


class TreeCommit:
    """Owns the commit's device tables.  Construct once (runner init, after
    KV caches and the stash exist, before CUDA-graph capture)."""

    def __init__(
        self,
        spec: TreeSpec,
        stash: TreeStash,
        gdn_layers: list[GDNCommitLayer],
        tq_kv_caches: list[torch.Tensor],
        device: torch.device | str,
    ):
        if spec.L < 1:
            raise ValueError("tree commit needs at least one leaf")
        self.spec = spec
        self.stash = stash
        self.W = spec.D + 1
        self.M = spec.M
        order = sorted(gdn_layers, key=lambda x: stash.layer_index[x.name])
        if [stash.layer_index[x.name] for x in order] != list(range(len(order))):
            raise ValueError("tree commit: GDN layers must cover TreeStash.layer_index exactly")
        if order:
            HV = order[0].ssm_state.size(1)
            C = order[0].conv_state.size(1)
            conv_len = order[0].conv_state.size(2)
            ssm_dtype = order[0].ssm_state.dtype
            for x in order:
                if (
                    x.ssm_state.size(1) != HV
                    or x.conv_state.size(1) != C
                    or x.conv_state.size(2) != conv_len
                    or x.ssm_state.dtype != ssm_dtype
                ):
                    raise ValueError("tree commit: GDN layers must share one shape and dtype")
            if tuple(stash.gdn_leaf.shape[3:]) != (HV, _KDIM + _VDIM + 2):
                raise ValueError("tree commit: stash HV / width mismatch")
            if stash.conv_leaf_x.size(3) != C or stash.conv_leaf_x.dtype != torch.float16:
                raise ValueError("tree commit: conv stash must be fp16 [.., C]")
            if conv_len < self.W:
                raise ValueError("tree commit: conv state shorter than D+1")
        else:
            conv_len, ssm_dtype = self.W, torch.float32
        self.conv_len = conv_len
        self.ssm_fp32 = ssm_dtype == torch.float32
        if tq_kv_caches:
            Hk, slot = tq_kv_caches[0].size(1), tq_kv_caches[0].size(3)
            for kv in tq_kv_caches:
                if kv.size(1) != Hk or kv.size(3) != slot:
                    raise ValueError("tree commit: TQ layers must share Hk / slot size")
        else:
            Hk, slot = 0, 0
        self.Hk, self.slot_bytes = Hk, slot
        # Device tables: built here, once.  Kept alive by self (the kernel reads them by address).
        gdn_rows = [_gdn_row(x, self.W) for x in order]
        tq_rows = [_tq_row(kv) for kv in tq_kv_caches]
        self.gdn_tab = torch.tensor(gdn_rows, dtype=torch.int64).reshape(-1, 8).to(device)
        self.tq_tab = torch.tensor(tq_rows, dtype=torch.int64).reshape(-1, 6).to(device)
        self._keep = (order, list(tq_kv_caches))  # tensors referenced by the tables
        self.leaf_depth = [int(d) for d in spec.leaf_depth]
        # Warmup inputs (all requests masked: rescue_leaf = -1, no write).
        self._warm_rescue = torch.full((spec.B_tree,), -1, dtype=torch.int32, device=device)
        self._warm_slots = torch.full(
            (spec.B_tree * spec.M,), _PAD_SLOT_ID, dtype=torch.int64, device=device
        )

    def warmup(self) -> None:
        """Load the extension and run one fully masked launch, eagerly, at
        init: the first call must never happen inside CUDA-graph capture
        (extension JIT load / first-launch setup invalidate the capture)."""
        if torch.device(self.gdn_tab.device).type != "cuda":
            return  # CPU (tests): no extension
        _load()
        self.commit(self._warm_rescue, self._warm_slots, self.spec.B_tree)

    def commit(
        self, rescue_leaf: torch.Tensor, tq_slot_mapping: torch.Tensor, num_reqs: int
    ) -> None:
        """rescue_leaf int32 [>= num_reqs] (-1 = none); tq_slot_mapping int64
        [>= num_reqs*M] (this step's TQ-group slot mapping, row-indexed)."""
        _load().tree_commit(
            rescue_leaf[:num_reqs],
            self.gdn_tab,
            self.stash.gdn_leaf,
            self.stash.conv_leaf_x,
            self.tq_tab,
            tq_slot_mapping,
            self.leaf_depth,
            self.W,
            self.M,
            self.conv_len,
            self.Hk,
            self.slot_bytes,
            _PAD_SLOT_ID,
            self.ssm_fp32,
        )

    def commit_step(
        self, outputs: TreeStepOutputs, tq_slot_mapping: torch.Tensor
    ) -> None:
        self.commit(outputs.rescue_leaf, tq_slot_mapping, outputs.rescue_leaf.size(0))
