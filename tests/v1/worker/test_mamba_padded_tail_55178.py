# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only regression test for the port of upstream vllm-project/vllm#55178
(fork lane ``mamba-55178``).

This line carries upstream #45237: when a request has exactly one prompt
token left (resumed / prefix-cache hit / remote KV tail) and the step already
runs speculative decodes, the scheduler pads it with ``K`` placeholder drafts
(``-1``) so it keeps the uniform ``K + 1`` decode shape. The model runner tags
such a row with ``num_decode_draft_tokens == K`` (``num_scheduled_tokens ==
draft_len + 1``). Before #55178 the Mamba metadata builder only reclassified
*single-token* stateful prefills, so the padded row (query ``K + 1``, still
``is_prefilling``) fell through ``split_decodes_and_prefills(...,
treat_short_extends_as_decodes=False)`` into the prefill/chunk-scan bucket,
which persists state after the unconfirmed placeholders -- silent recurrent
state corruption. #55178 ORs ``padded_prompt_tail_rows`` into
``prefill_to_decode`` so the row takes the transactional decode/spec path.

The builder module is loaded by path so the test always exercises this
checkout's ``mamba_attn.py`` (as ``tests/v1/core/test_spec_decode_workspace.py``
does); its imports resolve through the installed ``vllm``. The builder is
built with ``__new__`` plus the attributes the routine reads (style of
``test_mamba_stale_rows.py``); ``mamba_cache_mode='none'`` keeps the align /
ReplaySSM machinery out of the picture -- bucket routing is what is under
test.
"""

import importlib.util
import os
import sys
from types import SimpleNamespace

import torch

from vllm.v1.attention.backend import CommonAttentionMetadata

K = 3  # MTP K=3 shape -> decode_threshold = 1 + K


def _load_mamba_attn():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
    path = os.path.join(repo, "vllm", "v1", "attention", "backends", "mamba_attn.py")
    name = "_mamba_attn_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


MOD = _load_mamba_attn()

# The prefill branch uploads conv / chunk metadata with pinned-memory H2D
# copies that need a CUDA device (vllm.utils.torch_utils.async_tensor_h2d).
# Bucket routing is what is under test, so keep those uploads on the CPU.
MOD.async_tensor_h2d = lambda data, dtype, device, *a, **k: torch.tensor(data, dtype=dtype)
MOD.compute_causal_conv1d_metadata = lambda query_start_loc_p_cpu, *, device: (
    None, None, None,
)


def _builder():
    cls = MOD.BaseMambaAttentionMetadataBuilder
    b = cls.__new__(cls)
    b.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="none")
    )
    b.kv_cache_spec = SimpleNamespace(block_size=16, num_speculative_blocks=K)
    b.reorder_batch_threshold = 1 + K
    b.num_spec_tokens = K
    b.use_spec_decode = True
    b.use_replayssm = False
    b.decode_bc_pre_scratch = None
    b.metadata_cls = MOD.BaseMambaAttentionMetadata
    b.decode_cudagraph_max_bs = 8
    b.compilation_config = SimpleNamespace(
        cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False)
    )
    return b


def _common(rows):
    """rows: (query_len, seq_len, is_prefilling), in reorder order
    (decode -> short_extend -> long_extend -> prefill)."""
    qls = [r[0] for r in rows]
    qsl = torch.tensor(
        [0] + torch.cumsum(torch.tensor(qls), 0).tolist(), dtype=torch.int32
    )
    seq_lens = torch.tensor([r[1] for r in rows], dtype=torch.int32)
    n, nt = len(rows), int(qsl[-1])
    return CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl,
        seq_lens=seq_lens,
        num_reqs=n,
        num_actual_tokens=nt,
        max_query_len=max(qls),
        max_seq_len=int(seq_lens.max()),
        # mamba_cache_mode='none': (#requests, 1 + num_speculative_blocks)
        block_table_tensor=torch.arange(n * (1 + K), dtype=torch.int32).view(n, 1 + K),
        slot_mapping=torch.zeros(nt, dtype=torch.int64),
        is_prefilling=torch.tensor([r[2] for r in rows], dtype=torch.bool),
        seq_lens_cpu_upper_bound=seq_lens,
    )


def _route(rows, drafts):
    """Run the real _compute_common_metadata; return the bucket split."""
    m = _builder()._compute_common_metadata(
        _common(rows),
        num_accepted_tokens=torch.ones(len(rows), dtype=torch.int32),
        num_decode_draft_tokens_cpu=(
            None if drafts is None else torch.tensor(drafts, dtype=torch.int32)
        ),
    )
    return m.num_decodes, m.num_prefills, m.num_decode_tokens, m.num_prefill_tokens


SPEC_A = (1 + K, 1004, False)  # genuine spec decode rows, prompt long done
SPEC_B = (1 + K, 2004, False)
TAIL_REAL = (1, 1000, True)  # 999/1000 computed, one real token, no padding
TAIL_PADDED = (1 + K, 1003, True)  # #45237 shape: 1 real token + K placeholders


def test_padded_prompt_tail_routes_to_decode_bucket():
    # Runner tag for the padded row: num_scheduled == draft_len + 1 -> K.
    assert _route([SPEC_A, SPEC_B, TAIL_PADDED], [K, K, K]) == (3, 0, 3 * (1 + K), 0)


def test_without_the_draft_tag_the_padded_tail_would_hit_chunk_scan():
    # The pre-port call path (build() did not thread num_decode_draft_tokens_cpu)
    # is equivalent to passing None: the row lands in the prefill bucket.
    assert _route([SPEC_A, SPEC_B, TAIL_PADDED], None) == (2, 1, 2 * (1 + K), 1 + K)


def test_single_token_stateful_tail_still_routes_to_decode():
    # #51483 behaviour preserved: unpadded one-token stateful prefill -> decode.
    assert _route([SPEC_A, SPEC_B, TAIL_REAL], [K, K, -1]) == (3, 0, 2 * (1 + K) + 1, 0)


def test_stateless_first_chunk_with_padded_shape_stays_prefill():
    # seq_len == query_len: no prior Mamba state, must remain a prefill even
    # though the shape matches draft_len + 1 (has_prior_state guard).
    assert _route([SPEC_A, SPEC_B, (1 + K, 1 + K, True)], [K, K, K]) == (
        2, 1, 2 * (1 + K), 1 + K,
    )
