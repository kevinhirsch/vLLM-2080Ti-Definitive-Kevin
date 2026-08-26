# F-3 phase 1: DFlash2 draft-head port research

Research only. No code changed, no engine run. Scope: fetch and read
vllm-project/vllm HEAD's `qwen3_dflash2.py` + its proposer-side integration,
map it onto our tree, read upstream issue #53477 (prefix-cache reprocessing
bug), inspect the local `Qwen3.8-27B-DFlash2` checkpoint, and pull field
reports.

Repo: `weicj/vLLM-2080Ti-Definitive` (this checkout), branch
`frontier-pastnative-20260816`, 0.21-era base. Local checkpoint:
`/home/kevin/Desktop/models/Qwen3.8-27B-DFlash2/`. Context: production runs
Qwen3.8-27B + GPTQ target weights, MTP K=3 (gains plateau there; K=4 measured
slower), TP=2 on 22GB/card.

**Headline finding, before the detail:** DFlash2 upstream is not a drop-in
model file plus a config flag. Upstream's own PR body for #52816 states it
outright — *"DFlash2 runs on the V2 model runner, which is where its
speculator lives... the V1 `DFlashProposer` has no candidate selector, so a
DFlash2 checkpoint reaching it would draft as DFlash1 without raising."* Our
fork's production path is the V1 `GPUModelRunner` / `SpecDecodeBaseProposer`
stack (`vllm/v1/spec_decode/`), and our fork's V2 runner
(`vllm/v1/worker/gpu/`, gated by `VLLM_USE_V2_MODEL_RUNNER`) only implements
EAGLE (`init_speculator` in `vllm/v1/worker/gpu/spec_decode/__init__.py`
raises `NotImplementedError` for anything else). There is no proposer to
port — we would be authoring one. On top of that, the prefix-caching bug in
#53477 traces back to an architectural property shared by the whole
DFlash/DSpark family, and both of upstream's own proposed fixes for that
property (#47926, #48459) are still **open, unmerged** PRs. See §6 for the
verdict.

## 1. HEAD's DFlash2 implementation

Fetched from `vllm-project/vllm` `main` via `gh api repos/.../contents/...`
(code search `dflash2 repo:vllm-project/vllm` surfaced the full set):

| File | Lines | Role |
| --- | --- | --- |
| `vllm/model_executor/models/qwen3_dflash2.py` | 290 | Draft model: grouped dynamic conv + candidate selector, subclasses our existing `qwen3_dflash.py` |
| `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py` | 215 | Proposer-side: Triton candidate-walk kernel + draft-logits cache kernel, **V2-runner-only** |
| `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` | 216 | `DFlashSpeculator` base class DFlash2's speculator subclasses (V2) |
| `vllm/v1/worker/gpu/spec_decode/speculator.py` | 380 | `BaseSpeculator`/`DraftModelSpeculator` root of the V2 speculator hierarchy |
| `vllm/v1/worker/gpu/spec_decode/__init__.py` | — | Dispatch: `init_speculator()` — EAGLE only in our fork's copy |
| `vllm/model_executor/models/registry.py` | — | `"DFlash2DraftModel": ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM")` |

**Model file (`qwen3_dflash2.py`), what it adds over DFlash1:**

- `DFlashGroupedConv` — a grouped depthwise "dynamic convolution" wrapped
  around attention and MLP in each decoder layer (`prepare()`/`finish()`
  bracketing the sublayer), letting a block position see the ones before it
  without another backbone pass. Sized by `conv_kernel_size` (taps) and
  `conv_group_size`.
- `DFlash2Qwen3DecoderLayer(DFlashQwen3DecoderLayer)` — wraps `self_attn`
  and `mlp` in the conv, `block_size = 1 + num_speculative_tokens` from
  `vllm_config.speculative_config`.
- `CandidateSelector` — after the backbone, keeps the target head's top-K per
  slot, scores adjacent transitions via a low-rank codebook
  (`predecessor_codebook`/`successor_codebook`, `einsum` over `rank`), and
  the proposer walks the best path from the verified anchor.
- `DFlash2Qwen3Model(DFlashQwen3Model)` — adds the selector, an
  `input_embedding_scale` on `embed_input_ids`.
- `DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM)` — adds
  `compute_candidates()`, which calls `LogitsProcessor.get_top_k_tokens()`
  (does not exist in our fork — see §2).

**Proposer file (`dflash2/speculator.py`), what it does:** two Triton
kernels — `_selector_walk_kernel` (per-request program: Gumbel-noised argmax
walk through the K×K transition-score lattice, one loop over
`num_speculative_steps`) and `_cache_draft_logits_kernel` (writes the
realized candidate scores into a cached draft-logits buffer for the
downstream rejection sampler) — plus `_generate_draft()`, which runs the
model, reshapes hidden states to `(num_reqs, num_speculative_steps, -1)`,
calls `model.compute_candidates()` then `model.model.candidate_selector()`,
then the walk kernel. All of this is written against `DraftModelSpeculator`'s
buffer conventions (`self.sample_pos`, `self.sample_idx_mapping`,
`self.temperature`, `self.seeds`, `self.draft_tokens`,
`self.input_buffers.input_ids`) — V2-runner-only state that has no counterpart
in our V1 `SpecDecodeBaseProposer`.

## 2. Port-surface table

| HEAD component | Our equivalent | Status |
| --- | --- | --- |
| `DFlashGroupedConv`, `DFlash2Qwen3DecoderLayer/Model/ForCausalLM` | `qwen3_dflash.py` (`DFlashQwen3DecoderLayer/Model/ForCausalLM` — **already ported and present** in our tree) | **Clean.** Subclasses our existing classes almost 1:1. `set_model_tag`, `ReplicatedLinear(..., return_bias=...)` both exist in our fork already. |
| `CandidateSelector` | none | **Adapt.** New `nn.Module`, no blocker, but its `@support_torch_compile` tag isolation (compile-cache collision — see #53292 below) needs the same care upstream had to retrofit. |
| `LogitsProcessor.get_top_k_tokens()` | `LogitsProcessor.get_top_tokens()` (fork-added, top-**1** only, O(2·tp_size) all-gather for local-argmax reduction) | **Missing API.** Our version does a single-value all-gather-and-reduce across TP ranks; DFlash2 needs top-**K** (K=16) with TP-aware merge — a new, more involved kernel/method, not a small extension. |
| `DFlash2Speculator(DFlashSpeculator)` (V2, Triton walk + logits-cache kernels) | `DFlashProposer(SpecDecodeBaseProposer)` in `vllm/v1/spec_decode/dflash.py` (V1, **already ported**, in production use? — not currently the active method, MTP is) | **Missing API / net-new.** No V1-arch candidate-selector proposer exists anywhere, upstream or ours. The Triton kernels are usable as an *algorithm* reference (the math is portable) but their tensor plumbing (`sample_pos_ptr`, `req_state_ptr`, `InputBuffers`) is V2-specific and must be re-expressed against `SpecDecodeBaseProposer`'s hooks (`set_inputs_first_pass`, `build_model_inputs_first_pass`, `dummy_run` — the same hooks `DFlashProposer` already overrides for DFlash1). This is the single largest item in the port. |
| `init_speculator()` dispatch (V2) | none (V1 dispatch is a different mechanism, inside `gpu_model_runner.py`) | **Adapt.** Needs a new branch wiring a hand-written `DFlash2Proposer` into the V1 dispatch path. |
| `SpeculativeConfig.method` — dispatched via checkpoint architecture (`DFlash2DraftModel`) under the *existing* `"dflash"` method string, per the #53477 repro command (`"method": "dflash"`, model = the DFlash2 checkpoint) | `DFlashModelTypes = Literal["dflash"]` in `vllm/config/speculative.py` — **identical**, no new literal needed | **Clean.** No config-literal change required; the drafter's `hf_config.architectures` decides DFlash1 vs. DFlash2. |
| `dflash_config` schema (`block_size`, `selector_rank`, `selector_top_k`, `conv_group_size`, `conv_kernel_size`, `final_logit_softcapping`, `input_embedding_scale`) | not read anywhere in our `speculative.py`/`qwen3_dflash.py` (DFlash1 has no such block) | **Adapt.** Needs plumbing wherever the checkpoint's `dflash_config` dict is consumed — mostly inside the ported model file itself, so this mostly falls out of §1's clean port. |
| `has_ephemeral_draft_context()` / `eagle_group_is_veto_exempt` (from unmerged PR #48459 — see §3) | none | **Missing, and upstream doesn't have it merged either.** Confirmed via `gh api search/code` — zero hits for either symbol on current `main`. |

## 3. Issue #53477 — mechanism and dodge

**Symptom (as filed, 2026-08-23, zero comments as of this research — no
maintainer triage yet):** serving Qwen3.8-27B with
`{"method": "dflash", "model": ".../Qwen3.8-27B-DFlash2", "num_speculative_tokens": 7}`
and `--enable-prefix-caching`, DFlash2 "force reprocesses context every
reply." MTP on the same server does not.

**Mechanism — well-evidenced from three related, unmerged sibling PRs/issues
against the same DFlash/DSpark architectural property, not a maintainer
diagnosis of #53477 itself (treat as our best-supported hypothesis):**

DFlash-family drafters don't build KV incrementally like a normal causal
attention stack. They build their own separate KV cache by *re-deriving it
from the target's hidden states*, via `precompute_and_store_context_kv` — and
only for tokens that flow through an actual target forward pass **this
turn**. Quoting PR #47926's own description of the bug class directly:

> "Tokens whose KV is restored at request (re)admission — automatic prefix
> caching hits, KV-connector restores, resumption after preemption — never
> run through the target, so their draft context KV slots are never
> written. The draft's attention nevertheless spans the full sequence... so
> it reads uninitialized/stale KV for the whole restored region... degrade
> toward position-0-only acceptance with mean acceptance length ~1.0, while
> MTP/EAGLE-style drafters on the same workload are unaffected because they
> draft from the last decode hidden state and build no context KV."

That PR (#47926, filed 2026-07-07, **still open/unmerged**) fixes it by
masking prefix-cache-restored tokens out of the draft's visible KV window. A
second, narrower, also-**unmerged** PR (#48459, "Exclude DSpark draft-model
KV-cache group from core prefix-cache lookup veto") fixes a related failure
at the scheduler level: a hybrid model's KV-cache-group coordinator requires
every group — including the draft's ephemeral one — to independently confirm
a cache hit before it will honor *any* group's hit, so the draft's
always-empty-at-admission group vetoes the target's real, resident hit and
forces a full recompute. Neither fix has landed on `main`
(`gh api search/code` for `has_ephemeral_draft_context` /
`eagle_group_is_veto_exempt` returns nothing).

Put together: #53477's "force reprocesses" is very likely the *safe* end of
this same bug class — rather than serving stale/uninitialized draft KV
(#47930's "acceptance collapses to ~1.0," also open), something in the
current path is choosing to treat the request as a full cache miss whenever
a `dflash`-family draft is active, buying correctness at the cost of
reprocessing the entire prefix on every turn.

**Does it affect the acceptance path we'd use?** Yes, directly, and it is
the single biggest reason this isn't a quick win for us. Our production
workload is long multi-turn conversation with prefix caching central to
throughput (see `[[Local Context Lane Guarantee]]`). Adopting DFlash2 as-is
today means either eating full-prefix reprocessing on every reply (defeating
the point of long-context chat) or, on some other code path, silently
collapsing acceptance — and there is no merged upstream fix to inherit
either way.

**The dodge — three options, not mutually exclusive:**

1. **`--no-enable-prefix-caching` while DFlash2 is active.** What upstream's
   own DSpark benchmark recipes already do (per #47926's PR body). Zero
   engineering cost, but throws away the thing our workload depends on most —
   a bad trade for us specifically, not a real fix.
2. **Build the #47926 masking fix into our hand-written V1 proposer from day
   one**, since §2 already requires us to author a new `DFlash2Proposer`
   against `SpecDecodeBaseProposer`. Upstream's fix targets the V2 runner's
   `_prepare_dflash_inputs_kernel`/`RequestState.num_cached_tokens`; ours
   would need the equivalent inside `DFlashProposer.set_inputs_first_pass`
   (shorten the draft's visible `seq_lens` by whole cached blocks, shift the
   draft block-table row to match). More work, but converts "wait on
   upstream" into "fixed at the point we're touching anyway" — consistent
   with `[[Patch At Source]]`.
3. **Scope DFlash2 to workloads with no meaningful cache-hit prefix** (fresh
   single-shot completions) and keep MTP K=3 for the long multi-turn
   sessions that are most of our traffic. Cheapest option, but limits
   DFlash2's applicability to a minority of what we actually run.

## 4. Local checkpoint inspection

`config.json` (`/home/kevin/Desktop/models/Qwen3.8-27B-DFlash2/`):
`architectures: ["DFlash2DraftModel"]`, `model_type: "qwen3"`, 5 layers,
`hidden_size 5120`, `intermediate_size 17408`, 32 attention heads / 8 KV
heads, `head_dim 128`, `vocab_size 248320`, `dtype bfloat16`,
`sliding_window 2048` (all 5 layers `sliding_attention`),
`dflash_config: {block_size: 8, conv_group_size: 16, conv_kernel_size: 2,
selector_rank: 256, selector_top_k: 16, target_layer_ids: [5, 19, 33, 47,
61]}`, `num_target_layers: 64`. `model.safetensors` on disk:
3,848,817,896 bytes (3.85 GB decimal / 3.58 GiB).

**VRAM math (bf16, no embedding/LM-head — those are reused from the
target):**

| Component | Formula | Params |
| --- | --- | --- |
| Attention (q/k/v/o), ×5 layers | GQA: q/o project 5120↔4096, k/v project 5120↔1024 | 5 × 52.4M = 262.1M |
| MLP (gate/up/down), ×5 layers | 3 × 5120 × 17408 | 5 × 267.4M = 1,336.9M |
| Grouped conv (attn + mlp), ×5 layers | 2 × (5120 × 1280 + tiny base kernel) | 5 × 13.1M = 65.7M |
| Candidate selector (once) | 2 × (248320 × 256) codebooks + 5120×256 projection | 128.5M |
| **Total** | | **≈1.79B params → ≈3.59 GB bf16** |

That lands within ~7% of the actual 3.85 GB file (the gap is almost
certainly norm weights, RoPE buffers, and safetensors header overhead not
worth modeling exactly) — good enough to trust the order of magnitude:
**call it 3.6–3.9 GB of weights, fixed, on whichever GPU rank loads them.**

**TP handling — this is where it gets uneven for us.** Our fork's
`SpeculativeConfig` (`vllm/config/speculative.py`,
`resolve_draft_tensor_parallel_size`) defaults
`draft_tensor_parallel_size` to **1** whenever the target's TP > 1 (with a
logged warning), unless explicitly set to 1 or the target's own TP size. At
our TP=2, default behavior loads the full 3.6–3.9 GB draft **unsharded, on
rank 0 only** — stacking on top of that rank's ~7 GB target shard
(14 GB GPTQ ÷ 2), giving rank 0 ≈10.6–10.9 GB vs. rank 1's ≈7 GB. Comfortable
headroom under 22 GB either way, but imbalanced, and it eats directly into
the KV-cache pool on rank 0 specifically (see §5's RTX PRO 5000 numbers,
where DFlash2's KV pool measured ~30% smaller than MTP's on the same box for
exactly this reason).

Setting `draft_tensor_parallel_size=2` to balance the load is possible — the
inherited q/k/v/o and MLP linear layers from `qwen3_dflash.py` are standard
TP-sharded layers — but the DFlash2-specific additions
(`DFlashGroupedConv.kernel_projection`, `CandidateSelector.hidden_projection`,
and both codebooks) use `ReplicatedLinear`/plain `nn.Parameter`, i.e. they
are **not** shardable and would duplicate in full on every rank regardless
(≈128.5M selector + ≈65.7M conv ≈ 194M params ≈ 388 MB duplicated at TP=2 —
small, acceptable). Net: TP=2 with `draft_tensor_parallel_size=2` is the
better-balanced configuration to target, at the cost of ~388 MB duplicated
overhead versus a theoretical perfect shard.

**Quantization interaction, flagged but not resolved here:** the draft
checkpoint itself is bf16-unquantized (no need to touch it). But
`compute_candidates()` calls `self.lm_head` — almost certainly the
**target's** (GPTQ) lm_head, shared/tied for vocab consistency — and open
issue #52883 ("DFlash2: accept unquantized linear LM heads in the candidate
selector") implies the selector's top-K call has known rough edges around
quantized LM heads. Re-verify this against our actual GPTQ target before
relying on it; not confirmed broken, but a real open question, not a solved
one.

## 5. Field reports

**Local checkpoint's own model card** (`README.md`, mirrors
`incoai/Qwen3.8-27B-DFlash2`; SGLang on 1×H200, FA3, block size 8 / 7 draft
tokens, Qwen3.8's recommended sampling, `xhigh` reasoning):

| Task | Acceptance length: MTP | DSpark | DFlash2 |
| --- | ---: | ---: | ---: |
| GSM8K | 5.02 | 4.36 | **5.46** |
| MATH-500 | 4.72 | 3.92 | **5.28** |
| HumanEval | 3.91 | 3.30 | **4.39** |
| MBPP | 3.99 | 3.51 | **4.79** |
| MT-Bench | 3.74 | 3.01 | **4.10** |

Throughput at concurrency 1 (tok/s, ×-speedup vs. autoregressive): MTP
1.96–2.59×, DSpark 2.00–2.69×, **DFlash2 2.67–3.43×** across the same five
tasks — i.e. DFlash2 runs roughly **25–35% faster than MTP** at single
stream on this hardware/precision. The gap compresses hard under load:
at concurrency 32, MTP actually drops *below* 1× (scheduling/verification
overhead exceeds its benefit), DSpark similarly, while DFlash2 stays
1.01–1.45×.

**Origin PR #52816's own bench** (1×H200, GSM8K, vs. DSpark and
autoregressive only — no MTP column): DFlash2 3.51× at conc 1 (DSpark
2.79×, **+26% over DSpark**), 2.91× at conc 8, 2.20× at conc 32. Same PR
also profiles the conv+selector overhead directly: 0.174 ms at batch 1
rising to 0.370 ms at batch 32, **0.67–0.84% of the total step time** — the
new machinery itself is cheap; nearly all of the win/cost is in the
backbone forward and the walk's vocabulary top-K (which the PR credits to
FlashInfer's radix top-K kernel, 1.9× `torch.topk` at batch 1 and 4.5× at
batch 32 — **a Hopper/Ada-era kernel path with no stated Turing/SM75
support**, worth treating skeptically for our hardware, see §6).

**Real-world field report — issue #53428** (RTX PRO 5000, sm_120 Blackwell,
Qwen3.8-27B **NVFP4** target, fp8 KV cache, 262K context, single stream,
after applying the two-line `decoder_layer_cls` fix the issue itself
diagnoses and supplies):

| Method | Decode | Prefill @18.7K | KV pool |
| --- | ---: | ---: | ---: |
| MTP spec=3 | 112.5 tok/s | 6,288 tok/s | 515,501 tok |
| **DFlash2 spec=7** | **203.0 tok/s (+80.4%)** | 6,516 tok/s | 359,511 tok (**-30%**) |
| DSpark spec=7 | 141.4 tok/s | 6,440 tok/s | 304,558 tok |

Mean decode-only acceptance length for DFlash2: 4.71. This is the strongest
single data point (real serving numbers, MTP-3 vs. DFlash2-7 head to head)
but on very different silicon/precision than ours (Blackwell, NVFP4+fp8 vs.
our Turing SM75, GPTQ-int4) — and it directly confirms §4's VRAM-imbalance
concern: KV pool shrank ~30% carrying the draft's weights.

**Upstream stability context, relevant to risk not to the number:** #52816
(the DFlash2 origin PR) merged, but generated a cluster of follow-up bugs in
the days since — #53428/#53449 (decoder-layer-cls regression, diagnosed and
fixed by an issue reporter, not yet merged as a PR at research time),
#52883 (quantized LM head), #53292 (torch.compile cache collision between
DFlash/DSpark), #53366 (compile cache key missing `num_speculative_tokens`),
#53122, #53499, #53383, #53543, #53435 — all still **open**. DFlash2 landed
roughly a week before this research and is actively being hardened, not a
settled feature.

## 6. Verdict

**Port size: large.** Not because the model-definition file is hard — §1/§2
show that part is genuinely clean, reusing our already-ported
`qwen3_dflash.py` base classes almost 1:1. It's large because:

- There is no V1-architecture proposer to port (§2) — upstream built DFlash2
  exclusively against a newer worker runner (`vllm/v1/worker/gpu/`) that our
  fork's production path doesn't run and whose V1-equivalent our fork's V2
  runner doesn't yet implement for anything but EAGLE. We would design and
  write `DFlash2Proposer(DFlashProposer)` ourselves, re-deriving the
  candidate-walk/logits-cache logic against `SpecDecodeBaseProposer`'s
  buffer contract — the Triton kernels are a reference, not a diff to apply.
- The prefix-caching bug (§3) has to be actively engineered around, not
  inherited from upstream — both of upstream's own fixes for this bug class
  are unmerged.
- A genuinely missing API (`LogitsProcessor.get_top_k_tokens`, TP-aware,
  §2) has to be built, not just wired up.

**Top 3 riskiest integration points:**

1. **No V1-arch speculator exists anywhere (upstream or ours).** We are
   authoring new proposer logic against a buffer/hook contract that was
   never designed with a candidate-selector graph walk in mind. Upstream's
   own warning applies with extra force to a hand-port: get the wiring
   wrong and the checkpoint loads and runs *silently* as plain DFlash1 (no
   selector, no conv benefit, no error) rather than failing loudly.
2. **Prefix-cache / ephemeral-draft-KV interaction (#53477's bug class).**
   Both known upstream fixes are open, unmerged PRs as of today. Shipping
   without addressing it costs us either the reprocess-every-reply penalty
   or, on some other code path, the acceptance-collapse-to-~1.0 failure
   mode (#47930) — against a workload (long multi-turn, prefix-cache-heavy)
   that is close to worst-case for this specific bug.
3. **VRAM/TP imbalance plus an unresolved quantized-LM-head question.**
   Default `draft_tensor_parallel_size=1` stacks 3.6–3.9 GB unsharded onto
   one rank; the real-world RTX PRO 5000 report shows this costing ~30% of
   the KV-cache pool. `draft_tensor_parallel_size=2` rebalances but doesn't
   fully shard (§4's ReplicatedLinear pieces). Separately, whether the
   candidate selector's top-K call is safe against our GPTQ-quantized
   target lm_head is an open question upstream (#52883), not a solved one.

**Expected gain range vs. our K=3 baseline (72.9 tok/s single-stream):**
Three independent sources put DFlash2 at roughly **+25% to +80% single-stream
decode over MTP** — but every one of them is on newer, non-SM75 silicon
(H200, RTX PRO 5000 Blackwell) with precision/kernel paths (FA3, fp8/NVFP4
KV, FlashInfer's radix top-K) that have no stated support on Turing.
Discount hard for that: FlashInfer's radix top-K specifically is credited by
upstream's own profiling as a real chunk of the selector's speed edge, and
if it falls back to plain `torch.topk` on SM75 that portion of the win
shrinks. A skeptical, SM75-adjusted estimate is **+10% to +35%** over our
72.9 tok/s baseline (roughly 80–98 tok/s) *if the port is done correctly and
the prefix-cache dodge from §3 is applied* — with real downside risk from
the VRAM/TP imbalance (§4) and the missing radix top-K path eroding even
that. Treat every number in §5 as an upper bound, not a forecast.

**A/B protocol:**

- **Baseline:** current production MTP K=3, our standard capacity-tuning
  harness, conc 1/8/32, warm up + 3+ reps per `[[Benchmark Rigor]]`, report
  median and spread plus the existing per-request mean-acceptance-length
  counter.
- **Treatment:** DFlash2 at `num_speculative_tokens=7` (fixed by the
  checkpoint's `target_layer_ids`/`block_size=8` training design, not freely
  tunable like MTP's K — this is a different-K comparison by construction,
  fine for an end-to-end throughput call, not for an apples-to-apples
  per-token one). Two TP variants: `draft_tensor_parallel_size=2` (balanced,
  primary) and `=1` (default, unsharded) to quantify the VRAM/TP tradeoff
  directly rather than assume it.
- **Metrics:** decode tok/s at conc 1/8/32, mean acceptance length
  (`--enable-per-request-metrics`, matching the upstream repro command's own
  instrumentation), prefill tok/s and KV-pool size (to catch the §4/§5
  VRAM-squeeze effect directly), and a greedy-decode determinism spot-check
  against the model card's "decoding is lossless" claim.
- **Correctness gate before any production rollout:** re-test with the §3
  masking dodge actually implemented, not just `--no-enable-prefix-caching`
  — disabling APC outright is not viable against our real traffic, so a
  benchmark that only clears the bar under `--no-enable-prefix-caching`
  hasn't cleared the bar we actually care about.
- **Before writing a line of the V1 proposer:** one more upstream/community
  pass per `[[Research Before Building]]` — specifically check whether
  SGLang's DFlash2 implementation (the checkpoint also documents an SGLang
  serving path) is structured independently of vLLM's V1/V2 runner split;
  if so it may be a cleaner algorithmic reference than vLLM's V2-coupled
  code for the parts we have to write from scratch anyway.

---

## 2026-08-25 addendum: TnzGit independent port — study synthesis (study-only per Kevin)

Source: `TnzGit/vLLM-2080Ti-Definitive-dflash2` (standalone republication of the 0.2.x
branch + 12 agent-authored commits), local read-only clone at
`~/Desktop/.f3-study-tnzgit-dflash2`. Their docs (`docs/dflash2-adaptation/`) are a
complete, honest evidence chain. Materially updates this file's port estimate.

**What they proved (changes our discount):**
- A V1-runner DFlash2 greedy proposer works end-to-end on THIS fork family
  (correctness token-identical vs baseline at 32K) — the "no V1 speculator exists"
  premise of our LARGE-port estimate is now false: `bd57d36` (V1 proposer,
  greedy lattice walk) + `8a7ab04` (drafter port) + `5a9748a` (V2 speculator, opt-in).
- **Central-pool geometry conflict is real and they solved it**: draft 5-layer SWA
  fp16 natural pages (32KB/block16, 2KB/token/layer) get padded 51× to 1.676MB by
  `unify_kv_cache_spec_page_size` next to TQ groups (block 2112-2160,
  ~749 B/token/layer measured), and block sizes 16 vs 2160 are indivisible →
  coordinator assert. Fix that works: **private draft KV pool** owned by the
  proposer (`VLLM_DFLASH_OWN_KV_POOL=1`, extraction pattern mirrors
  HiddenStateCacheSpec at kv_cache_utils.py:1823). This is the architectural
  answer for ANY non-TQ-geometry drafter here — reusable beyond DFlash2.
- Same-code MTP3 baseline arm reproduces an external user benchmark at 0.3%
  (91.81 vs 92.09 tok/s, FP8+MTP3+K8V4 32K-arm method) — their methodology holds.

**Their sole remaining blocker (two faces of FULL-graph × DFlash scheduling):**
- normal/PIECEWISE: correct output but ~505ms target-forward per 8-token step —
  the piecewise decode graph replays at capture width 1024 (= mnbt). py-spy: 43%
  gdn/causal_conv + 24% turboquant_store inside target forward; round-2 suspicion
  of full-context `precompute_and_store_context_kv` rewrites per step (sglang does
  this incrementally).
- fast/FULL decode graph: ~112 tok/s observed BUT Xid31 MMU fault at CAPTURE
  (FAULT_PDE VIRT_READ @0x1000, dual-GPU, no traceback) → NaN logits → token-0
  garbage. They correctly identified garbage-output and Xid31 as one defect.

**Our-side mapping (why the port cost drops if we ever engage F-3):**
1. Their piecewise-1024 mystery is our known class: pin
   `cudagraph_capture_sizes=[K+1]` / `max_cudagraph_capture_size` (our prod uses
   [4]) instead of default-width capture. Likely removes the 505ms face outright.
2. Their capture-time Xid31 signature matches the block-table overrun class our
   guards target (`896b1013f` env-gated write-side bound, `ecced0014` pages>width
   early-out + reader-guard). Arming `VLLM_TQ_XID31_TRACE=1` would name the
   faulting kernel in one run.
3. Their K-bisection + compute-sanitizer + capture-time pool-tensor-visibility
   plan (HANDOVER §5.4) is sound; item 4 (private-pool data_ptr stability across
   capture) is the most likely true root for the capture fault.
4. lued-DFlash2-W8-draft can halve draft weights (3.85→1.9GB) if pool math gets
   tight on a port.

**Revised F-3 posture:** the LARGE-port estimate no longer holds — a port would be
"their commits + our two known fixes," post-0.2.x. Remaining genuine unknowns:
their per-step GDN store semantics under lookahead (their next work item), DFlash
prefix-cache bug class upstream (#47926/#48459) under our multi-turn workload, and
acceptance-length on OUR agentic traces vs their greedy benchmark shapes.
Contact remains OFF (study-only). Their user benchmark also contributes a
Beat-The-Fork cell: 92.09 tok/s decode @ FP8+MTP3+K8V4 128K util .95 on 2×2080Ti.
