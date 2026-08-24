# Upstream vllm-project/vllm harvest — 2026-08-24

Research only, no code changed. Scope: `vllm-project/vllm` (the MAIN project,
not the `weicj` fork), searched via `gh search prs/issues`, focused on the
last ~60 days with merged PRs weighted highest. Target stack: Qwen3.8-27B
(hybrid GDN, 48 linear-attn + 16 full-attn, `qwen3_5` arch), MTP spec decode
K=3, `mamba-cache-mode=align`, TurboQuant KV (fork-tracked, see §0), GPTQ
Marlin on SM75 Turing, TP=2, prefix caching, 524K YaRN context, chunked
prefill, agentic multi-turn workload.

Every candidate below was checked against our tree with
`git log --all --oneline --grep="#NNNNN"` before being listed as new, and PR
bodies/diffs were read (not just titles) via `gh pr view`/`gh pr diff`.
Already-known items from the standing list (#51508, #53479, #53051, #45845,
#52789, #52816, #47926/#48459, #53477, #52883, #52530, #53428, #52895/#52897,
#43447, #45702) are excluded throughout.

## 0. Correction to the brief: TurboQuant is not fork-only

The task brief lists "TurboQuant KV (fork-only)" as one of our differentiators.
That's not accurate as of this sweep. TurboQuant (`turboquant_k8v4` /
`turboquant_4bit_nc` / `turboquant_k3v4_nc` / `turboquant_3bit_nc` — the exact
preset names our `k3v4_nc` config uses) is an **upstream `vllm-project/vllm`
feature**, authored by `vibhavagarwal5`, merged to main 2026-04-15 (#38479,
2963 lines, 27 files) and iterated on since. Our tree already tracks it
closely — `git log` shows #38479's descendants #39931, #39988, #40092,
#40941, #44053, #47609, and #50533 all cherry-picked in
(`aef85aed5`, `fa4321de3`, `183b5f27e`, …), plus fork-original work on top
(SM75 restoration `b9a9fa851`/`d4a5825d7`, the k3v4_nc 3-bit preset, the
Xid31 continuation-workspace reserve chain). The two newest upstream
TurboQuant PRs found this sweep, #51857 (docs-only autorefs fix) and #47896
(ROCm FlyDSL decode kernel, not applicable to CUDA/Turing), are both trivial
or inapplicable — **TurboQuant is fully current, nothing to port.** Correct
the internal framing from "fork-only IP" to "upstream feature we track
closely and have extended (SM75 restore, k3v4_nc)"; that also means anything
we build on TurboQuant is a reasonable upstream contribution candidate, not
just internal tooling.

## 1. Candidate table

State/date/tree-status verified via `gh pr view --json state,mergedAt,body`
and `git log --grep`. "Tree" = already merged into our branch history
(commit shown) vs. `NEW` (not found).

| # | Title | State / merged | Tree | Fixes / gains | Portability | Value for us |
|---|---|---|---|---|---|---|
| [#43650](https://github.com/vllm-project/vllm/pull/43650) | Bugfix(Core): MTP + prefix caching + mamba accuracy fix | **OPEN** since 2026-05-26, review required, mergeable=UNKNOWN | NEW | `MambaManager.find_longest_cache_hit` never drops the final matched block under `use_eagle` (MTP included), unlike `FullAttentionManager`/`SlidingWindowManager` which do. Their GSM8K A/B: 0.916 (no MTP+PC) → 0.900 (old, MTP+PC) → 0.914 (fixed). ~1.6pp systematic accuracy loss from over-reuse of a partially-accepted final Mamba block. | **Clean.** 6 lines, 1 file (`vllm/v1/core/single_type_kv_cache_manager.py`). We diffed our own copy of the same function — confirmed the `use_eagle` branch is simply absent; `FullAttentionManager`/`SlidingWindowManager` in the same file already have it. | **HIGH.** Matches our exact production combo (MTP + `--enable-prefix-caching` + hybrid GDN/mamba). Silent correctness bug, not a crash — would not show up in throughput testing. |
| [#52078](https://github.com/vllm-project/vllm/pull/52078) | [Attention] Avoid redundant mask compute in GDN metadata build | MERGED 2026-08-20 | NEW | `GDNAttentionMetadataBuilder.build()` computed the `num_decode_draft_tokens_cpu >= 0` mask twice and `~spec_sequence_masks_cpu` 4×; collapses to one each. Pure perf, behavior preserved exactly per PR. | Clean, 1 file, 11+/-13. | **MED**, but see cross-ref below — **possible RCA lead**, promoted in §3. |
| [#53077](https://github.com/vllm-project/vllm/pull/53077) | [Bugfix][GDN] Reset speculative decode count for an empty draft schedule | MERGED 2026-08-20 | NEW | Same-day follow-up to #52078: when every scheduled draft-token count is zero, `num_spec_decodes` wasn't reset, so metadata invariants (`num_decodes == num_spec_decodes`) could assert-fail. Nightly CI catch, not found by #52078's own (partially-blocked) CI run. | Clean, 1 line, 1 file. | **MED** standalone; **HIGH** as an RCA lead — see §3. |
| [#51812](https://github.com/vllm-project/vllm/pull/51812) | [Bugfix] Align Qwen GDN gates with speculative tokens | MERGED 2026-08-11 | NEW | In a mixed batch where non-spec tokens precede spec tokens, `mixed_qkv` was gathered by `spec_token_indx` but the `a`/`b` gates were not — the fused recurrent update could apply gates from the wrong token. Repro'd on `Qwen/Qwen3.5-2B` at a `max_model_len` boundary with 2 MTP draft tokens. | Clean, 6+/-2, 1 file. | **HIGH.** Silent state-corruption class bug in exactly our GDN+MTP combo; boundary conditions (short `max_model_len` remainder) are common with 524K context serving many concurrent requests. |
| [#49436](https://github.com/vllm-project/vllm/pull/49436) | [Perf][Hybrid] 3D-grid tiling of the state-copy Triton kernels | MERGED 2026-08-10 | NEW (prereq #48110 already in tree) | Follow-up to our already-ported #48110. Tiles the temporal state copy across a 3rd grid axis so small-batch decode (low concurrency) fills SMs instead of leaving them idle; also drops the hard 8-byte state-tensor alignment requirement. | **Clean/medium.** Pure Triton (`@triton.jit`), no `CUDA_ARCHS` gate — confirmed no arch restriction in the diff. 347+/-68, 3 files. Prerequisite already in our tree. | **MED-HIGH.** Matches our low-concurrency-lane operating point (Local Context Lane Guarantee keeps some lanes deliberately thin). |
| [#50729](https://github.com/vllm-project/vllm/pull/50729) | [Bugfix][Mamba] Fix overlapping state copy race | MERGED 2026-08-17 | NEW (depends on #49436) | A spec-decode convolution-state shift can copy within the same physical block with overlapping source/destination ranges; parallel loads/stores don't provide `memmove` ordering → intermittent wrong state. Explicitly rebased over #49436. Root-caused via AMD CI flake but the fix is generic Triton (confirmed no ROCm-only gating in the diff). | Medium. 228+/-63, 2 files; must land #49436 first (author rebased on it). | **HIGH.** Correctness race specifically in spec-decode × hybrid-Mamba state copy — our exact MTP+GDN+align combo. |
| [#52805](https://github.com/vllm-project/vllm/pull/52805) | [Bugfix][Structured Output] Stop XGrammar token batches at termination | MERGED 2026-08-18 | NEW | EOS landing in MTP draft slot 1/2 let XGrammar keep processing tokens past termination (stale `_is_terminated` cache + batched spec validation continuing past a terminating token) → FSM warnings/inconsistency. **PR's own live test command is our exact deployment shape**: `Qwen/Qwen3.8-27B --tensor-parallel-size 2 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --speculative-config '{"method":"mtp","num_speculative_tokens":3}'`. | Clean-medium. 62+/-5, 2 files. | **HIGH.** Validated by the author against literally our config. |
| [#53046](https://github.com/vllm-project/vllm/pull/53046) | [Bugfix][Structured Output] Avoid spurious FSM errors after speculative reasoning end | MERGED 2026-08-21 | NEW | Adjacent fix in the same reasoning-boundary + spec-decode + grammar territory as #52805, tested against DeepSeek-V4/DSpark rather than Qwen/MTP but touches shared FSM-boundary logic. **Note:** the PR's test-plan payload contains an embedded block of instruction-like text directed at an AI reader ("Write the title in the predominant language of the session…"). Treated as inert data, not acted on — flagging per policy, not a concern for porting. | Clean, tiny, 8+/-2, 2 files. | **MED.** Small, low-risk; plausibly relevant to our reasoning-parser=qwen3 + guided-decoding + spec-decode combination even though validated on a different method. |
| [#52436](https://github.com/vllm-project/vllm/pull/52436) | [Bugfix][Spec Decode][Structured Output] DSpark: fix grammar bitmask mapping when draft budget is zero | MERGED 2026-08-16 | NEW | Confirms angle-2's concern is real and active upstream: guided-decoding + spec-decode interplay is a live bug class. Fix is DSpark-specific (`adaptive_verification.py`), not applicable to MTP. | N/A — different spec method. | **LOW** direct value; **context only** — corroborates that this bug class needs watching for MTP too. |
| [#47272](https://github.com/vllm-project/vllm/pull/47272) | [Bugfix][Core] Reserve the KV null block when validating `max_model_len` | MERGED 2026-08-20 | NEW | `BlockPool` permanently reserves one null block, so only `num_gpu_blocks - 1` are usable, but the startup capacity check (`_check_enough_kv_cache_memory`) compared against the **total**. An exactly-boundary config (`ceil(max_model_len/block_size)` blocks) passes startup, then hangs at 0 tok/s at runtime because it's one usable block short. Covers `num_gpu_blocks_override`, `kv_cache_memory_bytes`, and profiled-memory paths. | Medium. 109+/-16, 6 files (`config/cache.py`, `arg_utils`, `kv_cache_utils`, …). | **MED-HIGH.** We run near-max context (524K YaRN) with hand-tuned reserve arenas already (turboquant continuation-prefill reserve, spec-verify workspace reserve) — exact-boundary hangs are precisely the class of bug that reserve work exists to prevent. |
| [#52419](https://github.com/vllm-project/vllm/pull/52419) | [Bugfix][Spec Decode] Keep EAGLE cache registration on the partial-hash-hit path | MERGED 2026-08-16 | NEW, but **blocked** | Fixes a regression from #50062 in `HybridKVCacheCoordinator.cache_blocks`'s EAGLE branch under `enable_partial_hash_hits`. | **Blocked.** `HybridKVCacheCoordinator` class exists in our tree, but `enable_partial_hash_hits` does not — grepped the whole tree, zero hits. The prerequisite feature (fine-grained sub-block prefix-cache hits) isn't in our tree at all. | **LOW direct** (nothing to fix without the base feature) but **flags a gap**: upstream has a materially better partial-hit prefix-cache mechanism than our block-aligned one that we haven't evaluated as its own harvest item. |
| [#51875](https://github.com/vllm-project/vllm/pull/51875) | [Core] Make prefix-cache `NONE_HASH` deterministic by default | MERGED 2026-08-18 | NEW | `NONE_HASH` (root of the block-hash chain) was seeded from `os.urandom(32)` per process, forcing a shared `PYTHONHASHSEED` across nodes for any multi-instance prefix-cache sharing. Now derives from a fixed default seed (SHA-256 hashing no longer needs the secret-seed collision defense from the #12621 era); `PYTHONHASHSEED` still overrides if set. | Clean-medium, 204+/-77, 10 files (touches call sites broadly). | **LOW** for our single-TP-group deployment; would matter if we ever share a prefix cache across multiple engine instances. Safe, no-risk if ported. |
| [#48915](https://github.com/vllm-project/vllm/pull/48915) | [Frontend][Core][Spec Decode] Per-request acceptance stats in OpenAI API responses | MERGED 2026-08-20 | NEW | Opt-in (`--per-request-spec-decode-metrics summary\|detailed`) per-request `mean_acceptance_length`, `draft_acceptance_rate`, `acceptance_histogram`, etc. in the API response `metrics` object, alongside existing per-request timing metrics. Off by default, zero cost when disabled. | Medium-hard. 798+/-76, 25 files (protocol/serialization touches are broad even though the core logic is small). | **HIGH operationally.** Direct instrumentation for the Frontier Program's empirical MTP-acceptance-rate mapping — currently done by hand/benchmark script; this gets it per-request, for free, from production traffic. |
| [#52966](https://github.com/vllm-project/vllm/pull/52966) | [Bugfix][Quantization] Support CT block FP8 with Marlin | MERGED 2026-08-19 | NEW | Fixes Compressed-Tensors block-FP8 weights failing Marlin's `weight_scale_inv` vs `weight_scale` naming mismatch. | Clean, 41+/-17, 2 files. | **LOW.** Wrong quant format for us — we run GPTQ-Marlin int4, not Compressed-Tensors FP8. |
| [#52041](https://github.com/vllm-project/vllm/pull/52041) | [Core] Skip broadcasting mm tensor data to workers for prefix-cache-covered items | MERGED 2026-08-19 | NEW | EngineCore→TP-worker broadcast ships full multimodal tensors even when an item is fully prefix-cache-covered and provably unused. Explicitly targets **agentic multimodal**: a rolling window of frames resent every turn (live camera/video-style traffic), where the cost is linear in window size per request — their deployment measured ~19ms × 25 images = ~0.5s of EngineCore CPU per turn. Skips the broadcast when cache coverage is proven. | Medium. 130+/-2, 4 files. | **HIGH if we serve multimodal agentic traffic** (repeated-image-window pattern), **LOW if text-only.** Directly named "agentic" workload match either way. |
| [#53016](https://github.com/vllm-project/vllm/pull/53016) | [Bugfix] Skip MM processor cache inserts larger than capacity | MERGED 2026-08-20 | NEW | A single processed MM item exceeding `--mm-processor-cache-gb` crashed EngineCore at startup (`ValueError: value too large`) instead of degrading to uncached. Common with max-size profiling / long-video recipes. | Clean-medium, 108+/-5, 6 files. | **MED.** We do track a "Vision 1MP processor ceiling" — this converts a crash into a graceful degrade at exactly the boundary we operate near. |
| [#50172](https://github.com/vllm-project/vllm/pull/50172) | [Feature] Qwen3-Next (GDN): `mamba_cache_mode="all"` prefix caching with speculative decoding (MTP) on V1 | **OPEN**, draft/WIP, **CONFLICTING** mergeable, since 2026-07-28 | NEW | Adds `all`-mode SSM-state prefix caching (checkpoint at every block, not just chunk ends) composed with MTP. Their production agentic-trace replay (68,266 requests, 1,697 sub-agent groups): **81% cache reuse (all) vs 70% (align)**, with align's shortfall growing as prefill-chunk budget grows. Directly the metric our "agentic multi-turn workload" angle cares about. | **Hard/blocked for now** — draft, 3151+/-115 across 29 files, merge-conflicting with current upstream main, correctness-only per the author (perf/eval "land as follow-up"). Not portable as a clean cherry-pick today. | **HIGH as a watch item, not a port-now candidate.** If it lands and stabilizes, it's a bigger structural win for our agentic workload than anything else in this table — track it, don't attempt to backport the WIP diff. |
| [#36329](https://github.com/vllm-project/vllm/pull/36329) *(already in tree)* | Fix Qwen3.5 GatedDeltaNet `in_proj_ba` Marlin failure at TP>=2 | MERGED, in tree (`e45df8c3f`) | **IN TREE** | Confirms our exact TP=2 + Marlin + GDN combination was a known upstream failure class (`MergedColumnParallelLinear` under-sizes below Marlin's `MIN_THREAD_N=64` at higher TP) and is already fixed in our history. | — | Context only — no action, listed to close the loop on angle 4/1 overlap. |

15 new-to-us rows (2 open/watch-only, 13 mergeable-now), plus 1 confirmed-already-in-tree
reference row and the TurboQuant correction in §0. Also checked and confirmed
**already in our tree** during this sweep (no action needed, listed for
completeness): #48110, #48018, #44297, #46662, #48438, #45295, #48017,
#48860, #44944, #49015, #45413, #45763, #45600, #48816 (GPTQ Qwen3.5 MTP
weight loading — directly matches our GPTQ+MTP combo, already ported).

## 2. Angles that came up mostly empty

- **Marlin SM75/Turing-specific work**: no new Turing-specific Marlin PRs in
  the window; #29901 (Turing Marlin support itself) is old/already-baseline.
  Recent Marlin activity is Compressed-Tensors FP8 (#52966, wrong format for
  us) and MoE-oracle refactors (not our model shape).
- **YaRN/mrope fixes for qwen3.x specifically**: nothing new merged in the
  window that touches Qwen3.x YaRN math; mrope activity was XPU/ROCm-port
  noise or other-model bugfixes (Gemma, Kimi, Hunyuan).
- **Multimodal max_pixels**: #49015 (Qwen3-VL/Qwen-Omni honor max_pixels for
  video) is already in our tree; nothing newer found.
- **Tool-call parser churn past our v0.1.16 baseline**: #45413 (Streaming
  Parser Engine + new Qwen3 parser) is a large frontend refactor, already in
  our tree along with the smaller qwen tool-call fixes (#45763, #45600).
  Nothing new past that baseline surfaced in the window.
- **Model Runner V2 items** (dozens found: #49811, #46849, #48261, #50062,
  #48290, #53176, #53093, …): excluded per instructions — MRV2 is a newer
  runner generation than our V1-based 0.21-era tree targets even after the
  planned 0.2.x/v0.27.1 migration (that migration stays on "V1 contracts").
  None were trivially adaptable; several (#50062) are the actual root cause
  of a bug (#52419) we can't use either, for the same reason.

## 3. Cross-reference: possible lead for the open K3-tiny-prompt-hang RCA

`docs/k3-tiny-prompt-hang-rca.md` (written today, unresolved) documents a
silent deadlock: MTP K=3 with a ~4-token prompt hangs decode forever (0
errors, 0 tok/s); K=2 with the same prompt is fine; K=3 with a 4096-token
prompt is fine. The RCA's own top candidates are (a) PIECEWISE cudagraph
capturing the GDN/mamba scan against warmup-only state pointers, and (b)
mamba-align × MTP short-prefix block-boundary bookkeeping — both flagged
"not yet reduced to a specific offending line."

This sweep independently found #52078 and #53077 — both merged **2026-08-20,
four days before the RCA was written**, both patching
`GDNAttentionMetadataBuilder.build()`, the exact metadata builder that
computes per-request spec/non-spec classification for hybrid GDN models.
\#53077's fixed bug ("every scheduled draft-token count is zero" →
`num_decodes`/`num_spec_decodes` invariant mismatch) is adjacent territory
to a K=3 + tiny-prompt scenario, where a short prompt is a plausible trigger
for a degenerate per-step draft schedule. The RCA's own hang symptom (silent
wedge, not an assertion) doesn't literally match #53077's symptom (an
`AssertionError`), so **this is not a confirmed fix** — but it's upstream
activity in the identical code path, four days prior, that the RCA's author
did not have visibility into. Worth a cheap check before the next hardware
window: does upstream HEAD (with #52078+#53077 applied) still reproduce the
K=3/4-token hang on our config?

## 4. Top-5 shortlist

1. **#43650 — MTP + prefix caching + Mamba accuracy fix.** Port now; this is
   the highest-confidence, lowest-risk item in the whole sweep. We
   independently diffed our own `single_type_kv_cache_manager.py` and
   confirmed the gap directly: `FullAttentionManager` and
   `SlidingWindowManager` both already special-case `use_eagle` to drop the
   final matched block (a partially-accepted block shouldn't be trusted as a
   full cache hit); `MambaManager.find_longest_cache_hit` — the function we
   actually use for our hybrid GDN model — has no such branch. Upstream's own
   A/B shows ~1.6pp GSM8K accuracy loss from exactly this gap under MTP +
   prefix caching, which is our default production configuration, not an
   edge case. It's still open upstream (review pending), but the fix is 6
   lines in 1 file with an obvious, checkable invariant — cherry-pick
   directly. Port size: trivial (under an hour including a targeted eval
   check).

2. **#52078 + #53077 — GDN metadata builder pair.** Port together (they're
   same-day companion commits upstream) both for their own merits — one
   removes redundant per-step boolean-tensor recomputation in the hot
   decode-metadata path, the other is a 1-line invariant fix for the
   zero-draft-schedule case — and because they are the single strongest
   external lead for the still-open K3-tiny-prompt-hang RCA (§3). Even in the
   likely case they don't fully explain the hang, they're free, small,
   already-CI'd fixes to the exact metadata builder our hybrid GDN + MTP
   combo depends on every decode step. Port size: small (two small diffs,
   ~30 lines combined, no dependencies).

3. **#51812 — Align Qwen GDN gates with speculative tokens.** A silent
   state-corruption bug, not a crash: in a mixed batch where non-speculative
   rows precede speculative rows, the fused recurrent update could apply
   gate values (`a`/`b`) from the wrong token because `mixed_qkv` was
   index-gathered but the gates weren't. Reproduced by the author at a
   `max_model_len` boundary condition with 2 MTP draft tokens on
   Qwen3.5-2B — the boundary-condition framing (short remaining context) is
   generic enough to plausibly hit under our 524K-context, many-concurrent-
   request agentic workload. 6 lines, 1 file. Port size: trivial.

4. **#49436 + #50729 — hybrid state-copy perf + race fix, as a pair.** #49436
   is a direct, already-in-our-tree-prerequisite (#48110) follow-up: 3D-grid
   tiling that fills SMs at small batch sizes and drops the 8-byte alignment
   requirement, confirmed pure Triton with no CUDA-arch gate (portable to
   Turing). #50729 is explicitly rebased on top of it and fixes a genuine
   overlapping-memcpy race in the spec-decode convolution-state shift path —
   a correctness bug that would manifest as silent wrong state under load,
   intermittently, exactly in our spec-decode × hybrid-Mamba combo. Port
   #49436 first, then #50729 on top, in that order (matches upstream's own
   sequencing). Port size: medium — roughly 575 combined lines across 5
   files, plus their test suites; budget a half-day including our own align-
   mode regression pass.

5. **#52805 — Stop XGrammar token batches at termination.** Singled out over
   the similar #53046 because the author's own live-model test command in
   the PR body is not just "similar to" but reproduces our exact deployment:
   `Qwen/Qwen3.8-27B`, `--tensor-parallel-size 2`,
   `--tool-call-parser qwen3_coder --reasoning-parser qwen3`,
   `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`. It
   fixes an EOS-lands-in-a-draft-slot edge case where XGrammar kept
   processing tokens past termination, producing FSM warnings and possible
   inconsistency — the exact "does spec decode stay well-behaved under
   guided/JSON mode" question this sweep was asked to check. Port size:
   small (62+/-5 lines, 2 files); low risk given it's guarded by a
   termination check, not a structural change.
