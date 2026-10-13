# SPDX-License-Identifier: Apache-2.0
"""GPU equivalence gate for E16 P6/P7 (run in a GPU window, not under pytest).

Checks on CUDA, vs the CPU torch reference:
  1. walk kernel == tree_reference.tree_walk_ref (random + exhaustive small
     vocab, M=7 and width-3 M=10 specs, with -1 opt-out leaves);
  2. TreeSampler greedy == ref walk over argmax (V = 248320, B = 1..4);
  3. chain-only TreeSampler greedy == today's RejectionSampler;
  4. drafter leaf topk excludes the chain token (V = 248320).
Prints one JSON line {"test": ..., "pass": bool, ...}; exit 0 only on PASS.
"""

import itertools
import json
import os
import random
import sys

import torch


def main() -> int:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.sample.rejection_sampler import RejectionSampler
    from vllm.v1.sample.sampler import Sampler
    from vllm.v1.spec_decode import tree_reference as R
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
    from vllm.v1.spec_decode.tree_drafter import tree_leaf_tokens
    from vllm.v1.spec_decode.tree_sampler import TreeSampler, tree_walk
    from vllm.v1.spec_decode.tree_spec import TreeSpec

    dev = torch.device("cuda")
    fails: list[str] = []

    def md(B):
        return SamplingMetadata(
            temperature=None, all_greedy=True, all_random=False, top_p=None,
            top_k=None, generators={}, max_num_logprobs=None, no_penalties=True,
            prompt_token_ids=None, frequency_penalties=torch.tensor([]),
            presence_penalties=torch.tensor([]), repetition_penalties=torch.tensor([]),
            output_token_ids=[[] for _ in range(B)], allowed_token_ids_mask=None,
            bad_words_token_ids={}, logitsprocs=LogitsProcessors(), spec_token_ids=None,
        )

    rng = random.Random(0)
    for spec in (TreeSpec.chain_plus_leaves(3), TreeSpec.chain_plus_leaves(3, width=3)):
        lr = torch.tensor(spec.leaf_rows, dtype=torch.int32, device=dev)
        ld = torch.tensor(spec.leaf_depth, dtype=torch.int32, device=dev)
        B, V = 4096, 4
        d = torch.tensor([[rng.randrange(V) for _ in range(spec.M - 1)] for _ in range(B)])
        d[::5, spec.D:] = -1
        y = torch.tensor([[rng.randrange(V) for _ in range(spec.M)] for _ in range(B)])
        got = tree_walk(spec, d.to(dev), y.to(dev), lr, ld)
        ref = R.tree_walk_ref(spec, d, y)
        for k, v in ref.items():
            if not torch.equal(getattr(got, k).cpu(), v):
                fails.append(f"walk M={spec.M} {k}")
    s5 = TreeSpec.chain_plus_leaves(2)
    ys = torch.tensor(list(itertools.product(range(3), repeat=s5.M)))
    d5 = torch.tensor([[0, 1, 1, 2]] * ys.shape[0])
    got = tree_walk(s5, d5.to(dev), ys.to(dev),
                    torch.tensor(s5.leaf_rows, dtype=torch.int32, device=dev),
                    torch.tensor(s5.leaf_depth, dtype=torch.int32, device=dev))
    ref = R.tree_walk_ref(s5, d5, ys)
    if not torch.equal(got.output_token_ids.cpu(), ref["output_token_ids"]):
        fails.append("walk exhaustive")

    V = 248320
    spec = TreeSpec.chain_plus_leaves(3)
    ts = TreeSampler(Sampler(), spec, dev)
    for B in (1, 2, 3, 4):
        logits = torch.randn(B * spec.M, V, device=dev, dtype=torch.float16)
        am = logits.float().argmax(-1).view(B, spec.M)
        drafts = am[:, 1:].clone()  # mostly-accepting drafts, mixed with random
        drafts[:, ::2] = torch.randint(0, V, drafts[:, ::2].shape, device=dev)
        _, res = ts(logits, drafts, md(B))
        ref = R.tree_walk_ref(spec, drafts.cpu(), am.cpu())
        if not torch.equal(res.output_token_ids.cpu(), ref["output_token_ids"]):
            fails.append(f"sampler greedy B={B}")

    chain = TreeSpec.from_choices([(0,), (0, 0), (0, 0, 0)])
    tc = TreeSampler(Sampler(), chain, dev)
    rs = RejectionSampler(Sampler())
    for B in (1, 4):
        logits = torch.randn(B * 4, V, device=dev)
        drafts = [[rng.randrange(V) for _ in range(3)] for _ in range(B)]
        sd = SpecDecodeMetadata.make_dummy(drafts, device=dev)
        sd.target_logits_indices = torch.tensor(
            [b * 4 + i for b in range(B) for i in range(3)], dtype=torch.int32, device=dev)
        sd.bonus_logits_indices = torch.tensor([b * 4 + 3 for b in range(B)],
                                               dtype=torch.int32, device=dev)
        want = rs(sd, None, logits.clone(), md(B)).sampled_token_ids
        got, _ = tc(logits.clone(), torch.tensor(drafts, device=dev), md(B))
        if not torch.equal(got.sampled_token_ids, want.to(torch.int32)):
            fails.append(f"chain==rejection B={B}")

    lg = torch.randn(4, V, device=dev)
    c = lg.argmax(-1)
    leaf = tree_leaf_tokens(lg, c, (1,))
    if not (leaf.view(-1) != c).all():
        fails.append("leaf==chain")

    ok = not fails
    print(json.dumps({"test": "tree_p6_p7_gpu_equiv", "pass": ok, "fails": fails}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
