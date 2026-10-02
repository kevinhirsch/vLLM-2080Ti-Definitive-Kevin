# SPDX-License-Identifier: Apache-2.0
"""[FORK][LANE EF] A running spec-decode request that is dropped from the
persistent batch for a step (token budget taken by a prefill chunk) must keep
its accepted-token count when it is re-added; its recurrent state still sits at
slot accepted-1 of the speculative window."""

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch


def _batch():
    return InputBatch(
        max_num_reqs=4,
        max_model_len=256,
        max_num_batched_tokens=256,
        device=torch.device("cpu"),
        pin_memory=False,
        vocab_size=1000,
        block_sizes=[16],
        kernel_block_sizes=[16],
        num_spec_tokens=3,
    )


def _req(req_id: str) -> CachedRequestState:
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(range(20)),
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=8),
        generator=None,
        block_ids=([0, 1],),
        num_computed_tokens=20,
        output_token_ids=[1, 2],
    )


def test_fresh_request_starts_at_one():
    b = _batch()
    r = _req("a")
    b.add_request(r)
    assert int(b.num_accepted_tokens_cpu[b.req_id_to_index["a"]]) == 1


def test_dropped_then_readded_restores_count():
    b = _batch()
    r = _req("a")
    b.add_request(r)
    idx = b.req_id_to_index["a"]
    b.num_accepted_tokens_cpu[idx] = 3  # last step accepted 2 drafts + bonus
    # what GPUModelRunner._update_states does for an unscheduled running req
    r.saved_num_accepted_tokens = int(b.num_accepted_tokens_cpu[idx])
    b.remove_request("a")
    b.condense()
    b.refresh_metadata()
    b.add_request(r)
    b.refresh_metadata()
    assert int(b.num_accepted_tokens_cpu[b.req_id_to_index["a"]]) == 3
    # the stash is single-use: a later re-add without a fresh stash resets
    b.remove_request("a")
    b.condense()
    b.refresh_metadata()
    b.add_request(r)
    b.refresh_metadata()
    assert int(b.num_accepted_tokens_cpu[b.req_id_to_index["a"]]) == 1
