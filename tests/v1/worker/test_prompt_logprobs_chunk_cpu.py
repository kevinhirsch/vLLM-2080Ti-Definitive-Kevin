"""CPU test: prompt-logprob chunking gives identical results for any chunk size, including "auto".

compute_prompt_logprobs_with_chunking processes the prompt's logits in chunks. NM-9 hit an OOM at the fixed
1024-token chunk (the 486 MiB all-gather did not fit the packed KV pool's headroom), so the chunk is now
VLLM_PROMPT_LOGPROBS_CHUNK_SIZE (default 1024) or "auto" (sized from free memory). The concatenated top-k ids,
scores and ranks must not depend on how the prompt was chunked. "auto" sizes the chunks once per call and agrees
the size across tensor-parallel ranks: each chunk all-gathers, and W2h's ranks (795 vs 1024 rows) hung.
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


def test_auto_chunk_budget(monkeypatch):
    """3 rows of the logits dtype + an fp32 row per token, from 90% of free memory (W2h: rank 1, 0.79 GB)."""
    monkeypatch.setattr(P, "_free_device_bytes", lambda t: 790_000_000)
    per_token = 248320 * (3 * 2 + 4)
    got = P._auto_prompt_logprobs_chunk(torch.empty(1, 248320, dtype=torch.float16))
    assert got == int(790_000_000 * 0.9) // per_token == 286  # the old 2-row estimate gave 795


def test_auto_chunk_agreed_once_across_ranks(monkeypatch):
    """Both ranks run the same chunks: the real sizing, the TP-group minimum, once per call."""
    monkeypatch.setattr(P, "compute_topk_scores", _torch_topk_scores)
    monkeypatch.setenv("VLLM_PROMPT_LOGPROBS_CHUNK_SIZE", "auto")
    torch.manual_seed(1)
    n, hidden, vocab = 300, 8, 50
    w = torch.randn(vocab, hidden)
    hs = torch.randn(n, hidden)
    ids = torch.randint(0, vocab, (n,))
    per_token = vocab * (3 * 4 + 4)  # fp32 logits here
    free = {"rank0": 1 << 40, "rank1": -(-37 * per_token * 10 // 9)}  # alone: 1024 vs 37 rows
    local = {r: min(1024, max(16, int(f * 0.9) // per_token)) for r, f in free.items()}
    assert local == {"rank0": 1024, "rank1": 37}
    proposed = []

    def tp_min(v):  # what the CPU-group MIN all-reduce returns: the smallest proposal of any rank
        proposed.append(v)
        return min(local.values())

    monkeypatch.setattr(P, "_tp_min", tp_min)
    runs = {}
    for rank in ("rank0", "rank1"):
        monkeypatch.setattr(P, "_free_device_bytes", lambda t, r=rank: free[r])
        calls = []

        def logits_fn(h):
            calls.append(h.shape[0])
            return torch.stack([w @ row for row in h])

        runs[rank] = (calls, P.compute_prompt_logprobs_with_chunking(ids, hs, logits_fn, 5))
    assert runs["rank0"][0] == runs["rank1"][0] == [16] + [37] * 7 + [25]  # identical chunk sequence
    assert proposed == [1024, 37]  # each rank agrees once per call
    for a, b in zip(runs["rank0"][1], runs["rank1"][1]):
        assert torch.equal(a, b)


def test_tp_min_single_process():
    assert P._tp_min(123) == 123  # no TP group initialized: the local value
