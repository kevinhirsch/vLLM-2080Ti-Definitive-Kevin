"""CPU test: prompt-logprob chunking gives identical results for any chunk size, including "auto".

compute_prompt_logprobs_with_chunking processes the prompt's logits in chunks. NM-9 hit an OOM at the fixed
1024-token chunk (the 486 MiB all-gather did not fit the packed KV pool's headroom), so the chunk is now
VLLM_PROMPT_LOGPROBS_CHUNK_SIZE (default 1024) or "auto" (sized from free memory). The concatenated top-k ids,
scores and ranks must not depend on how the prompt was chunked.
"""
import types

import pytest
import torch

from vllm.v1.worker.gpu.sample import prompt_logprob as P


def _torch_topk_scores(logits, num_logprobs, sampled_token_ids, logits_mode=False, **_):
    """Row-wise stand-in for compute_topk_scores (its Triton logsumexp needs a GPU): the chunking under test
    only concatenates per-row results, so any row-wise scorer must give chunk-invariant output."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    ids = torch.cat((sampled_token_ids.unsqueeze(-1), torch.topk(logits, num_logprobs, dim=-1).indices), dim=1)
    ranks = (logits > logits.gather(-1, sampled_token_ids.unsqueeze(-1))).sum(-1)
    return types.SimpleNamespace(logprob_token_ids=ids, logprobs=lp.gather(-1, ids), selected_token_ranks=ranks)


@pytest.mark.parametrize("chunk", ["1024", "7", "16", "auto"])
def test_chunking_invariant(monkeypatch, chunk):
    monkeypatch.setattr(P, "compute_topk_scores", _torch_topk_scores)
    torch.manual_seed(0)
    n, hidden, vocab = 100, 8, 50
    w = torch.randn(vocab, hidden)
    hs = torch.randn(n, hidden)
    ids = torch.randint(0, vocab, (n,))
    calls = []

    def logits_fn(h):  # row by row: a batched CPU matmul rounds differently per batch size
        calls.append(h.shape[0])
        return torch.stack([w @ row for row in h])

    monkeypatch.delenv("VLLM_PROMPT_LOGPROBS_CHUNK_SIZE", raising=False)
    ref = P.compute_prompt_logprobs_with_chunking(ids, hs, logits_fn, 5)
    monkeypatch.setenv("VLLM_PROMPT_LOGPROBS_CHUNK_SIZE", chunk)
    calls.clear()
    got = P.compute_prompt_logprobs_with_chunking(ids, hs, logits_fn, 5)
    for a, b in zip(ref, got):
        assert torch.equal(a, b)
    assert sum(calls) == n
    if chunk == "7":
        assert max(calls) == 7
    if chunk == "auto":
        assert calls[0] == 16  # the first chunk measures the vocab row, then sizes the rest


def test_auto_chunk_bounds():
    assert P._auto_prompt_logprobs_chunk(torch.empty(1, 248320, dtype=torch.float16)) == 1024  # CPU: no limit
