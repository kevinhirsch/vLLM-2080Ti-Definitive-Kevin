# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, replace
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import AttentionSpec


class Mamba1AttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "MAMBA1_ATTN"

    @staticmethod
    def get_builder_cls() -> type["Mamba1AttentionMetadataBuilder"]:
        return Mamba1AttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class Mamba1AttentionMetadata(BaseMambaAttentionMetadata):
    pass


class Mamba1AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[Mamba1AttentionMetadata]
):
    metadata_cls = Mamba1AttentionMetadata

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # [FORK] lane mamba-55178: Mamba1 has no speculative-decode support.
        # MambaMixer keeps no scratch state slots (mamba1_state_shape takes no
        # num_spec) and this builder never receives num_accepted_tokens, so
        # with K > 0 every K+1 decode row is classified as a prefill
        # (decode_threshold stays 1) and selective_scan_fn persists the state
        # after unverified draft tokens -- silent corruption on every step,
        # not just the padded-tail case of vllm-project/vllm#55178. Refuse at
        # engine start instead of serving garbage.
        if self.use_spec_decode:
            raise NotImplementedError(
                "Mamba1 layers (MambaMixer) do not support speculative "
                f"decoding: num_speculative_tokens={self.num_spec_tokens} > 0 "
                "would route every K+1 decode row through the prefill scan "
                "with no rollback. Disable speculative decoding for this model."
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs: Any,
    ) -> Mamba1AttentionMetadata:
        common = self._compute_common_metadata(common_attn_metadata)

        if (
            common.num_prefills > 0
            and self.vllm_config.cache_config.mamba_cache_mode == "all"
        ):
            cu_chunk_seqlen_p, _, last_chunk_indices_p = (
                self._build_chunk_metadata_tensors(
                    self.kv_cache_spec.block_size,
                    common,
                    common_attn_metadata,
                )
            )
            return replace(
                common,
                cu_chunk_seqlen_p=cu_chunk_seqlen_p,
                last_chunk_indices_p=last_chunk_indices_p,
            )

        return common
