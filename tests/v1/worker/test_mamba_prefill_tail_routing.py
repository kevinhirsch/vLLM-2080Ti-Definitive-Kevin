# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only contract tests for the Mamba1/2 metadata-routing hazard audited
against upstream vllm-project/vllm#55178 (lane ``mamba-55178``).

Upstream #55178 fixes silent Mamba state corruption that appears when the
scheduler pads a one-token prompt tail (a resumed / prefix-cache-hit /
KV-connector request with exactly one real prompt token left) with ``K``
placeholder draft tokens (``-1``) to keep the uniform ``K + 1`` decode shape
(padding introduced upstream by #45237). ``mamba_attn.py::
_compute_common_metadata`` routes rows with
``split_decodes_and_prefills(..., treat_short_extends_as_decodes=False)``,
which sends *every* ``is_prefilling`` row to the prefill/chunk-scan bucket
regardless of query length (``is_prefill |= is_prefilling`` in
``vllm/v1/attention/backends/utils.py``). The chunk-scan kernels persist the
state after the last position unconditionally, so a padded tail row in that
bucket would bake the unconfirmed placeholders into the recurrent state (no
crash; wrong tokens later).

This branch does not carry #45237's padding, and its scheduler guarantees
that a request still inside its prompt is never scheduled with draft tokens
or async placeholders:

* ``Scheduler._update_after_schedule`` derives ``is_prefill_chunk`` from the
  post-step computed count;
* ``Scheduler.update_draft_token_ids`` discards drafts for prefill chunks;
* ``AsyncScheduler._update_after_schedule`` skips placeholder attachment for
  prefill chunks.

Those gates are the *only* reason the Mamba path is safe here (the helper is
not structurally immune), so they are pinned below by calling the real
methods on ``__new__``-constructed scheduler instances with lightweight
stand-ins (same style as ``test_mamba_stale_rows.py``). The routing tests pin
the helper's behaviour on the exact scenario batch so the hazard class stays
visible: if #45237-style padding is ever ported, the padded-tail case must be
re-routed to the decode bucket (upstream #55178 clears ``is_prefilling`` for
``padded_prompt_tail_rows`` before the split, keyed on
``num_decode_draft_tokens_cpu``, which also requires the runner to tag such
rows with ``num_scheduled_tokens == draft_len + 1`` instead of masking every
prefilling row to ``-1``).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mamba1_attn import Mamba1AttentionMetadataBuilder
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.outputs import DraftTokenIds

K = 3  # production MTP shape (K=3) -> decode_threshold = 1 + K

# Batch order mirrors reorder_batch_to_split_decodes_and_prefills:
# decode -> short_extend -> long_extend -> prefill.  Rows are
# (query_len, seq_len, is_prefilling).
SPEC_A = (1 + K, 1004, False)  # genuine MTP decode row, prompt long done
SPEC_B = (1 + K, 2004, False)
TAIL_REAL = (1, 1000, True)  # 999/1000 prompt computed, exactly 1 real token
TAIL_PADDED = (1 + K, 1003, True)  # #45237 shape: 1 real + K placeholders


def _common(rows):
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
        block_table_tensor=torch.zeros((n, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(nt, dtype=torch.int64),
        is_prefilling=torch.tensor([r[2] for r in rows], dtype=torch.bool),
        seq_lens_cpu_upper_bound=seq_lens,
    )


def _mamba2_split(rows):
    # mamba_attn.py::_compute_common_metadata: with num_accepted_tokens set
    # (spec decode on) decode_threshold = reorder_batch_threshold = 1 + K and
    # treat_short_extends_as_decodes=False.
    return split_decodes_and_prefills(
        _common(rows), decode_threshold=1 + K, treat_short_extends_as_decodes=False
    )


# --------------------------------------------------------------------------
# Routing characterization (split_decodes_and_prefills as Mamba2 calls it)
# --------------------------------------------------------------------------


def test_one_token_tail_lands_in_prefill_bucket_and_is_a_single_token():
    """A real one-token tail sits in the prefill bucket. That is safe: a
    chunk-scan over exactly one token with an initial state is the same
    recurrence step as one decode update, so nothing unconfirmed is
    persisted."""
    assert _mamba2_split([SPEC_A, SPEC_B, TAIL_REAL]) == (2, 1, 2 * (1 + K), 1)
    assert _mamba2_split([TAIL_REAL]) == (0, 1, 0, 1)


def test_padded_tail_would_be_misrouted_into_chunk_scan():
    """Characterization of the hazard class: a #45237-shaped padded tail
    (still prefilling, query K+1) lands in the prefill bucket, i.e. the
    chunk-scan path, which cannot roll placeholders back. Safe on this branch
    only because the scheduler never builds such a row (see gate tests)."""
    assert _mamba2_split([SPEC_A, SPEC_B, TAIL_PADDED]) == (2, 1, 2 * (1 + K), 1 + K)


def test_clearing_is_prefilling_is_what_reroutes_a_padded_tail():
    """Upstream #55178's mechanism: clearing is_prefilling for padded tail
    rows before the split moves them to the decode (transactional) bucket."""
    assert _mamba2_split([SPEC_A, SPEC_B, (1 + K, 1003, False)]) == (3, 0, 3 * (1 + K), 0)


def test_mamba1_threshold_sends_every_spec_row_to_prefill_bucket():
    """Mamba1's builder passes no num_accepted_tokens, so decode_threshold is
    1 and every K+1 row becomes a prefill: spec decode is unsupported on the
    Mamba1 path by construction (documented, not a regression)."""
    out = split_decodes_and_prefills(
        _common([SPEC_A, SPEC_B, TAIL_REAL]),
        decode_threshold=1,
        treat_short_extends_as_decodes=False,
    )
    assert out == (0, 3, 0, 2 * (1 + K) + 1)


# --------------------------------------------------------------------------
# Scheduler gates that keep drafts/placeholders off prefilling rows
# --------------------------------------------------------------------------


class _Req:
    def __init__(self, num_computed_tokens, num_tokens, **kw):
        self.num_computed_tokens = num_computed_tokens
        self.num_tokens = num_tokens
        self.num_output_placeholders = kw.get("num_output_placeholders", 0)
        self.spec_token_ids = list(kw.get("spec_token_ids", []))
        self.is_prefill_chunk = kw.get("is_prefill_chunk", False)
        self.use_structured_output = False

    @property
    def num_tokens_with_spec(self):
        return self.num_tokens + len(self.spec_token_ids)

    def is_finished(self):
        return False


def _sched(cls, requests):
    s = cls.__new__(cls)
    s.requests = requests
    s.finished_req_ids = set()
    s.structured_output_manager = SimpleNamespace(should_advance=lambda r: False)
    s._spec_token_placeholders = [-1] * K
    return s


def _running_num_new_tokens(req):
    # scheduler.py running-queue formula (num_tokens_with_spec +
    # num_output_placeholders - num_computed_tokens), no budget caps.
    return req.num_tokens_with_spec + req.num_output_placeholders - req.num_computed_tokens


def test_update_after_schedule_flags_prefill_chunk_from_post_step_count():
    tail = _Req(num_computed_tokens=998, num_tokens=1000)
    s = _sched(Scheduler, {"tail": tail})
    out = SimpleNamespace(num_scheduled_tokens={"tail": 1}, has_structured_output_requests=False)

    Scheduler._update_after_schedule(s, out)  # step leaves exactly 1 token
    assert tail.num_computed_tokens == 999 and tail.is_prefill_chunk is True

    Scheduler._update_after_schedule(s, out)  # the 1-token tail step itself
    assert tail.num_computed_tokens == 1000 and tail.is_prefill_chunk is False


def test_update_draft_token_ids_discards_drafts_for_prefill_chunks():
    tail = _Req(999, 1000, is_prefill_chunk=True, spec_token_ids=[7])
    # Running decode row: the last sampled token is already appended to
    # num_tokens but not yet computed, so num_computed == num_tokens - 1.
    done = _Req(1000, 1001, is_prefill_chunk=False)
    s = _sched(Scheduler, {"tail": tail, "done": done})

    Scheduler.update_draft_token_ids(
        s, DraftTokenIds(["tail", "done"], [[1, 2, 3], [4, 5, 6]])
    )
    assert tail.spec_token_ids == []  # never K+1 while the prompt is unfinished
    assert done.spec_token_ids == [4, 5, 6]
    # Hence the sync scheduler schedules the 1-token tail step with exactly 1.
    assert _running_num_new_tokens(tail) == 1
    assert _running_num_new_tokens(done) == 1 + K


def test_async_scheduler_never_pads_a_prefilling_row_with_placeholders():
    tail = _Req(998, 1000)  # two prompt tokens left before this step
    # Steady-state async decode row: one output placeholder plus K placeholder
    # drafts stand in for the not-yet-returned step, so it schedules 1 + K.
    dec = _Req(1000, 1000, num_output_placeholders=1, spec_token_ids=[-1] * K)
    s = _sched(AsyncScheduler, {"tail": tail, "dec": dec})
    assert _running_num_new_tokens(dec) == 1 + K
    out = SimpleNamespace(
        num_scheduled_tokens={"tail": 1, "dec": 1 + K},
        scheduled_spec_decode_tokens={"dec": [-1] * K},
        has_structured_output_requests=False,
        pending_structured_output_tokens=False,
    )

    AsyncScheduler._update_after_schedule(s, out)  # step S-1: leaves 1 token
    assert tail.is_prefill_chunk is True
    assert tail.spec_token_ids == [] and tail.num_output_placeholders == 0
    assert dec.spec_token_ids == [-1] * K
    assert dec.num_output_placeholders == 1 + (1 + K)
    # Step S (the one-token tail) is therefore scheduled with exactly 1 token,
    # never the padded 1 + K shape that would misroute into chunk-scan.
    assert _running_num_new_tokens(tail) == 1

    out_s = SimpleNamespace(
        num_scheduled_tokens={"tail": 1},
        scheduled_spec_decode_tokens={},
        has_structured_output_requests=False,
        pending_structured_output_tokens=False,
    )
    AsyncScheduler._update_after_schedule(s, out_s)  # step S computes token N
    assert tail.num_computed_tokens == 1000 and tail.is_prefill_chunk is False
    # Placeholders attach only now (one output placeholder plus K placeholder
    # drafts), so step S+1 -- a genuine decode -- is the first 1 + K step.
    assert tail.spec_token_ids == [-1] * K and tail.num_output_placeholders == 1
    assert _running_num_new_tokens(tail) == 1 + K


# --------------------------------------------------------------------------
# Mamba1 + speculative decoding is refused at engine start ([FORK] guard in
# Mamba1AttentionMetadataBuilder.__init__): the Mamba1 path has no scratch
# state slots and never receives num_accepted_tokens, so every K+1 row would
# take the prefill scan (see the threshold test above).
# --------------------------------------------------------------------------


def _vllm_config(num_spec: int):
    spec = (
        None
        if num_spec == 0
        else SimpleNamespace(num_speculative_tokens=num_spec, parallel_drafting=False)
    )
    return SimpleNamespace(
        speculative_config=spec,
        num_speculative_tokens=num_spec,
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=None),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )


def _mamba_spec():
    return MambaSpec(block_size=16, shapes=((1,),), dtypes=(torch.float32,))


def test_mamba1_builder_refuses_speculative_decoding():
    with pytest.raises(NotImplementedError, match="speculative"):
        Mamba1AttentionMetadataBuilder(
            _mamba_spec(), ["layer.0"], _vllm_config(K), torch.device("cpu")
        )


def test_mamba1_builder_constructs_without_speculative_decoding():
    b = Mamba1AttentionMetadataBuilder(
        _mamba_spec(), ["layer.0"], _vllm_config(0), torch.device("cpu")
    )
    assert b.use_spec_decode is False and b.reorder_batch_threshold == 1
