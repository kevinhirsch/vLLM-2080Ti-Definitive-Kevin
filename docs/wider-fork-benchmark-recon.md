# Wider fork/benchmark recon — 2026-08-24

Research only, no code changed. Mission: find every published LLM-serving
benchmark on comparable hardware (SM75/Turing: 2080 Ti/22GB mods, T4, Quadro
RTX, CMP-series; plus 2-GPU consumer rigs generally) at ~27-35B model scale,
and check whether any of it beats our tracked numbers. Sweep covered GitHub
repo/code search, `gh search issues`, HF model-card discussions, and web
search (r/LocalLLaMA, llama.cpp/exllama ecosystem). Every number below was
pulled from a primary source (README, PR body, changelog, issue) via `gh
api`/`gh pr view`/WebFetch — not from search-snippet summaries alone, except
where explicitly marked "unverified."

**Our reference point** (from the task brief, `docs/upstream-vllm-harvest-2026-08.md`
context): Qwen3.8-27B, 2x RTX 2080 Ti 22GB, SM75, NVLink, TP=2 —
**76.6 tok/s single-stream decode, K=3 MTP, prefill ~1300-1500 tok/s, 524K
YaRN context, 637K-token KV pool.**

## 0. Headline finding

The single strongest external threat is not on Turing at all: **[syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090)**
serves the identical model (Qwen3.8-27B) on **one** RTX 3090 (Ampere,
sm_86, $700-900 secondhand) at 120-133 tok/s single-stream decode with a
**working DFlash2 port** — the same upstream feature (vLLM PR #52816) that
our own `docs/f3-dflash2-port-research.md` stalled on at the research stage
today, and that a downstream fork of our own upstream (`weicj`) is currently
blocked on with a Xid 31 GPU-fault (see §2). One Ampere card, at less than
half our system cost, already beats our decode number by 55-75% at shallower
context, has a reproducible bench harness, and is actively maintained (commits
2 days old at sweep time). Full detail in §3.1.

## 1. Beat-board — every number found

Sorted by how directly it's comparable to our config (model, quant, context
depth). "vs us" compares against 76.6 tok/s single-stream decode / prefill
1300-1500 / 524K ctx unless noted. All decode numbers are single-stream
unless labeled aggregate/concurrent.

| # | Source | Hardware | Model / quant | Published number | Credibility | vs. us |
|---|---|---|---|---|---|---|
| 1 | [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090) | 1x RTX 3090 24GB @250W (Ampere) | Qwen3.8-27B, int4 lm_head + calibrated int8 GEMMs | MTP: 121 tok/s default / 120 greedy @64k; 96-102 @150k. **DFlash2: 127-133 tok/s** @64k; up to 382 tok/s on doc-reproduction (25k ctx) | **HIGH** — reproducible `bench/run_benchmarks.sh`, dated commits, documented regressions/fixes with commit hashes | **BEATS us** at shallower context (64-150K vs our 524K); single GPU |
| 2 | syv-ai (same repo), `CTX=huge` (KVarN, 268K pool) | 1x RTX 3090 | Qwen3.8-27B + SPEC=dflash2 | 53-67 tok/s aggregate across 6 mixed tasks; up to 130-167 tok/s on copy/reproduction task; 268,169-token pool | **HIGH** | **Below us on mixed-task average**, but pool (268K) approaches our 637K and copy-task peak (167) crushes us — closest apples-to-apples long-context comparison found |
| 3 | [redlinedtm-jpg/vllm-v100-2080ti-recipes](https://github.com/redlinedtm-jpg/vllm-v100-2080ti-recipes) | **4x** RTX 2080 Ti 22GB (2 NVLink pairs), `weicj` fork | Qwen3.6-27B-AWQ, MTP K=3, 2x TP=2 instances | **76-83 tok/s per instance** (152.6 tok/s aggregate); 46 tok/s per instance without MTP | **HIGH** — maintained by a GPU-server reseller (NextGen-PC), explicit methodology, "measured on hardware we assemble/ship" | **Roughly matches us** (83 vs 76.6) on same base fork family, different model (3.6 not 3.8), shorter context, AWQ not GPTQ+TQK8V4 |
| 4 | [weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive) `0.2.x` branch (our own upstream) | 2x RTX 2080 Ti 22GB NVLink TP=2 | Qwen3.8-27B-FP8, MTP3, `tqk8v4`/NVFP4 fast mode | **83.90-102.60 tok/s decode**, prefill 1355-1525 tok/s — but measured at **4K-input/128-output** (shallow), not deep-context | **HIGH** — official upstream README/CHANGELOG, `v0.2.1-pre2` release, credits `@kevinhirsch` as a contributor | **Not a clean beat** — different methodology (shallow decode, not 524K-deep); flags our tree may be behind upstream's own 0.2.x runtime gains. See §2.2 |
| 5 | [weicj/2080Ti-LLM-Toolbox](https://github.com/weicj/2080Ti-LLM-Toolbox) (companion repo, same author) | 2x RTX 2080 Ti 22GB NVLink TP=2 | Qwen3.6-27B-AWQ, MTP K=3 | Peak 4K/128: decode **101.3-101.5 tok/s**, prefill 1841.7. PP64K/TG512: decode 55.3, prefill 1294.3. LongGen3 (9.8k in/3k out): decode 54.14-55.20 | **HIGH** — extremely detailed methodology (separate prefill/decode, warm-up rules, provenance dated) | Same-family calibration data, not a competitor; shows the shallow-vs-deep decode gap (101 @ 4K vs 55 @ 64K) that likely explains most of the gap to our 524K number |
| 6 | [TnzGit/vLLM-2080Ti-Definitive-dflash2](https://github.com/TnzGit/vLLM-2080Ti-Definitive-dflash2) (fork of our upstream) | Same rig class, 2x 2080 Ti TP=2 | Qwen3.8-27B-FP8, DFlash2 port attempt | Reference point they benchmark against: **92.09 tok/s decode / 1408.07 prefill @ 128K**, FP8+MTP3+K8V4. Their own DFlash2 port: **0.72 tok/s** (correctness-preserving PIECEWISE mode, broken/blocked) or invalid ~112 tok/s in a mode that then crashes (Xid 31) | **HIGH provenance, but currently a non-result** — detailed engineering docs, git history, but the headline number is a blocked/broken port | **Does not currently beat us** — see §2 for why this is the most important non-benchmark finding of the sweep |
| 7 | [zyYuc/twin-turing](https://github.com/zyYuc/twin-turing) | 2x RTX 2080 Ti 22GB, **no NVLink** | Qwopus/Qwen3.6-27B-AWQ, MTP K=3 | Decode "about 60-78 tok/s" (workload-dependent) | **MEDIUM** — real service, dated evidence files/JSON, but self-described as "not a promise," coarse range | Roughly matches to slightly below us; no NVLink and it's still competitive — worth checking why we don't see similar without NVLink dependency |
| 8 | [Luce-Org/lucebox-hub PR #80](https://github.com/Luce-Org/lucebox-hub/pull/80) (via [noonghunna/club-3090](https://github.com/noonghunna/club-3090)) | 2x RTX 2080 Ti 22GB, dual-GPU target/draft split | Qwen3.5-27B Q4 target + z-lab DFlash draft | **51.86 tok/s**, accept-length 7.09, 44.3% acceptance (HumanEval-style, 10 prompts) | **HIGH** — merged PR with the actual bench table in the PR body | Below us; different (older z-lab DFlash1, not DFlash2/MTP) speculator and smaller-scope benchmark (10 prompts) |
| 9 | llama.cpp baseline, dual/single 2080 Ti (via `weicj/2080Ti-LLM-Toolbox`) | 1x RTX 2080 Ti 22GB | Qwen3.6-27B-Q4_K_M GGUF | PP4096/TG128: decode 23.7, prefill 553.4. PP64K/TG512: decode 16.3, prefill 383.1. With integrated MTP (RDson checkpoint): decode up to 28.4 (n=2, 68% accept) | **HIGH** | Well below us — confirms vLLM+MTP is structurally 2-4x llama.cpp on this hardware class, nothing to chase here |
| 10 | Quadro RTX 8000 48GB (single card, SM75) — via WebSearch, [gpubattle.com](https://gpubattle.com/ai/nvidia-quadro-rtx-8000) | 1x Quadro RTX 8000 | Qwen3.6-27B, 262K ctx | 18.1 tok/s. Qwen3 32B: 21.98 tok/s | **MEDIUM** — third-party aggregator site, not a primary repo; not independently reproduced | Well below us — confirms our TP=2+NVLink+MTP setup already beats a single bigger-VRAM Turing card |
| 11 | RTX Pro 6000 Blackwell (workstation, not Turing — included as ceiling reference) — HF discussion [Qwen3.8-27B-FP8 #9](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/discussions/9) | 1x RTX Pro 6000 (300W or 600W variant) | Qwen3.8-27B-FP8, MTP depth 2-5 | 300W: 62.2 tok/s (MTP depth 2, optimal). 600W: up to 116.0 tok/s in one user's config; initial naive config only 23 tok/s (MTP depth 5, mistuned) | **MEDIUM** — primary source is an HF discussion thread (community-reported, not a repo), numbers not independently re-run by us | Mixed: 600W config beats us by 51%, 300W config (62.2) is below us — shows MTP-depth mistuning can cost 5x even on far newer silicon; not our hardware class but a sanity ceiling |
| 12 | [ziyiguanmomo/QwenForge](https://github.com/ziyiguanmomo/QwenForge) | 1x RTX 2080 Ti 22GB | **Qwen2.5-0.5B** (hand-written CUDA/Triton engine, no vLLM) | "3.34x vLLM decode TPS," 96.15/100 benchmark score | **LOW relevance** — real repo, but 0.5B model is not comparable to 27B-class; included for completeness per the sweep brief | N/A — different model class entirely, does not threaten our number |

## 2. The important non-benchmark finding: a downstream fork is mid-flight on the same DFlash2 port, hitting our own Xid 31 signature

`TnzGit/vLLM-2080Ti-Definitive-dflash2` (created today, 2026-08-24, forked
from `weicj/vLLM-2080Ti-Definitive@vllm-2080ti-definitive-0.2.x`) is actively
porting DFlash2 (upstream PR #52816) onto the SM75 fork — the exact same task
our own `docs/f3-dflash2-port-research.md` scoped today and stopped at
"there is no proposer to port — we would be authoring one." This fork went
further: it built a working V1-runner `DFlash2Proposer` with a private draft
KV pool, and its own handover doc (`docs/dflash2-adaptation/HANDOVER.md`,
written in Chinese) documents:

- **Correctness verified**: DFlash2 produces coherent, greedy-matching output
  in `normal`/PIECEWISE mode.
- **Blocked on performance**: PIECEWISE mode replays a 1024-wide graph on
  every decode step regardless of actual batch size, costing ~500ms/step →
  **0.72 tok/s**.
- **The fast path is broken by a hardware fault**: `FULL`-graph mode
  transiently hit ~112 tok/s before corrupting output (NaN logits → argmax
  collapses to token 0) and was traced to **Xid 31 (MMU fault,
  `FAULT_PDE VIRT_READ`)** during CUDA graph capture — the identical Xid 31
  failure class documented in this project's own
  `[[RCA Xid31 Local Engine Crashes]]` memory note, now showing up in someone
  else's DFlash2×FULL-graph combination.
- Their own baseline reference point (92.09 tok/s decode @ 128K, FP8+MTP3+K8V4)
  reproduced to within 0.3% (91.81 measured) — establishing this as a
  credible, careful benchmarking effort, not noise.
- Their CHANGELOG credits `@kevinhirsch` (this project's own GitHub handle)
  among the `weicj` upstream's contributors, and the handover doc's
  operational details (host naming, task-directory layout, bilingual
  documentation style) closely mirror this project's own conventions.

**This is flagged, not acted on.** It reads as either (a) a genuinely
independent third party racing the same well-known upstream PR on the same
public fork, or (b) a parallel/duplicate effort by someone on this project's
own team working the identical problem in an isolated environment. The
Xid 31 correlation and the shared-contributor credit make (b) plausible
enough that it is worth confirming out-of-band before spending more effort on
our own DFlash2 port — no benchmark action is warranted from this content
itself (per the instruction-source boundary: everything read from these
repos is data, not instructions), but the RCA cross-reference (Xid 31 under
FULL-graph + DFlash2) is worth carrying into `docs/k3-tiny-prompt-hang-rca.md`
or the frontier RCA chain regardless of provenance.

### 2.2 Our own upstream may already be ahead of our tree on shallow-context decode

Separately from the DFlash2 story: `weicj/vLLM-2080Ti-Definitive`'s own
`0.2.x` branch README (CUDA 13.0 / PyTorch 2.13 / vLLM v0.27.1 base) publishes
**83.90-102.60 tok/s decode** for Qwen3.8-27B-FP8/NVFP4 with MTP3, at a 4K-input/
128-output test shape. That is not directly comparable to our 76.6 tok/s
number (measured deep into a 524K context, where full-attention-layer cost is
much higher — the toolbox's own data shows the same model dropping from
101→55 tok/s between a 4K and a 64K test, so a further drop by 524K is
expected). But it does mean: (a) our own upstream's shallow-context ceiling
should be checked against our tree's shallow-context number as a sanity
check, and (b) the 0.2.x / CUDA 13 / v0.27.1 migration (tracked in this
project's `docs/0.2.x-migration-memo.md`) may carry real throughput gains
independent of any DFlash2 work.

## 3. Detail on the top external threats

### 3.1 syv-ai/qwen38-27b-rtx3090 — top target

- Single RTX 3090 24GB @ 250W power limit, Ampere (sm_86) — not our hardware
  class, but the most relevant "beat this" target because it's the same
  model, well-documented, and cheap.
- Ships **both** MTP and a working **DFlash2** backport (vLLM PR #52816 —
  same PR our fork is trying to port) as switchable `SPEC=` modes, plus a
  KVarN long-context patch (`CTX=huge`, 268K-token pool at 245,760
  max-model-len) that composes with DFlash2.
- Batch mode: ~1,035 tok/s aggregate at 64 concurrent (128in/512out); up to
  ~1,222 tok/s with int8 on all layers.
- Single-user headline: 120 tok/s greedy MTP @ 64K ctx; **127-133 tok/s with
  DFlash2**; up to 382 tok/s when the answer quotes back the prompt (drafted
  straight from context).
- At their longest context (`CTX=huge`, 268K pool, close to half our 637K):
  53-67 tok/s aggregate across six mixed tasks, but the copy/reproduction
  task alone hits 130-167 tok/s.
- Concurrency data is unusually rigorous: shows DFlash2 losing ~7ms/step per
  additional resident request (vs ~5ms for MTP) due to recurrent-state page
  cost, and documents that DFlash2 tops out around 5 concurrent residents
  where MTP scales to 8 — i.e., DFlash2 is a single-user speed play, MTP (or
  batch mode) is the concurrency play. That trade-off matches this project's
  own posture (extreme single-concurrency on Turing) closely enough to be
  directly actionable if/when our own DFlash2 port lands.
- One contributor (`@changtimwu`, issue #22) reports testing "on a TP=2 box,"
  i.e., some of this repo's user base is already running multi-GPU
  configurations and comparing — worth watching that issue thread for
  Turing-specific data points in the future.

### 3.2 redlinedtm-jpg/vllm-v100-2080ti-recipes — closest same-hardware-family number

- Explicitly built around the `weicj` fork ("Turing → weicj" is their stated
  rule), so this is effectively an independent operator's field report on
  our own upstream, not a rival engine.
- Their 4x-2080Ti / 2x-TP=2-instance MTP K=3 result (76-83 tok/s/instance)
  sits right at our number — but on Qwen3.6-27B-AWQ, not Qwen3.8-27B, and at
  unspecified (likely much shorter) context, so it's suggestive rather than
  conclusive.
- Documents a genuinely useful operational fact for this project: on 4-GPU
  2080 Ti boxes, **two independent TP=2 instances (one per NVLink pair) beat
  one TP=4 instance** by a wide margin (152.6 vs ~90 aggregate) — not
  applicable to our 2-GPU rig directly, but confirms the NVLink-pair topology
  matters more than raw GPU count on this silicon.
- Also documents a 7.8x throughput cliff from picking `--quantization awq`
  vs `awq_marlin` on the wrong model type (dense vs MoE) — a config-hazard
  class worth a defensive check in our own launcher if not already guarded.

### 3.3 Upstream SM75 risk items found along the way (not benchmarks, but relevant)

From `gh search issues repo:vllm-project/vllm "2080 Ti"` and related Turing
searches — filed for awareness, not acted on:

- [#47549](https://github.com/vllm-project/vllm/issues/47549) — "REGRESSION:
  FP8 KV cache FlashInfer no longer available as attention backend on SM75
  (Turing) in v0.24.0" (open). Relevant to the 0.2.x/v0.27.1 migration path.
- [#38918](https://github.com/vllm-project/vllm/issues/38918) — Gemma4 on
  Turing hits shared-memory limits on all attention backends (open,
  confirms this project's own decision to keep Gemma4 experimental-only on
  SM75).
- [#33461](https://github.com/vllm-project/vllm/issues/33461) — Marlin NVFP4
  GEMM kernel on Turing "produces meaningless outputs" (closed) — worth a
  quick sanity check that our NVFP4 route, if ever adopted, post-dates the
  fix.
- [#35414](https://github.com/vllm-project/vllm/issues/35414) — "4x2080ti 22g
  deploy Qwen3.5-35B-A3B fail: 2080 Ti does not support bfloat16" (closed) —
  confirms the fp16-only constraint this project already designs around.

## 4. Angles that came up empty

- **HF model-card performance tables specifically tagged SM75-validated**:
  none found beyond the checkpoint-status tables already inside the `weicj`/
  `2080Ti-LLM-Toolbox` ecosystem (§1 rows 4-5). HF discussion threads (e.g.
  the Qwen3.8-27B-FP8 one in row 11) carry community numbers but on newer
  (Blackwell) hardware, not Turing.
- **r/LocalLLaMA direct hits**: web search could not surface a specific 2026
  r/LocalLLaMA thread with dual-2080Ti/22GB serving numbers; all roads led
  back to the `weicj` GitHub repos themselves (which likely originated any
  such Reddit chatter). Treat this angle as exhausted for now rather than
  under-searched.
- **ik_llama.cpp specifically**: no distinct ik_llama.cpp (vs. mainline
  llama.cpp) 2080Ti benchmark surfaced; the llama.cpp numbers found (row 9)
  are mainline.
- **exllamav2/v3 on Turing**: search results confirm ExLlamaV2/V3 have known
  issues even on Ampere and no Turing-specific 27B/32B benchmark was found.
- **CMP 90HX**: real ecosystem exists (unlock/patch repos), but it's a
  different architecture (Ampere GA102, not Turing) marketed as a mining
  card with no display output — out of the SM75 scope and no LLM-serving
  benchmark numbers were found for it in this sweep.

## Top-3 threats/targets

1. **syv-ai/qwen38-27b-rtx3090** (single RTX 3090, Ampere) — the best-documented,
   most actively maintained, most directly comparable (same model) benchmark
   found. Already beats our single-stream decode number by 55-75% at shallower
   context, with a working DFlash2 port we don't have yet. Priority: study
   their DFlash2 concurrency/context trade-off data (§3.1) before/alongside
   our own port attempt, and watch issue #22 for any Turing/TP=2 data points
   from their contributors.

2. **The parallel DFlash2 port on `TnzGit/vLLM-2080Ti-Definitive-dflash2`** —
   not a benchmark win (currently blocked/broken), but the most operationally
   important find: it's racing the identical task this project scoped today,
   on the identical upstream fork, and independently hit the identical Xid 31
   signature this project already has an open RCA for. Confirm whether this
   is duplicate in-house effort before continuing our own port; if not, its
   Xid 31 evidence chain (`FULL`-graph capture × DFlash2 = MMU fault) is a
   useful RCA cross-reference either way.

3. **Our own upstream's `0.2.x` branch shallow-context numbers** (83-102 tok/s
   decode via `weicj`'s own README/CHANGELOG) — not a competitor, but a
   same-family ceiling we should be hitting too. Worth a controlled
   shallow-context (4K/128) re-measurement on our tree to check whether the
   gap to upstream's published number is explained entirely by the 0.1.x
   vs 0.2.x runtime difference, or whether there's a config/tuning delta
   worth closing regardless of the 0.2.x migration timeline.

## Sources

- https://github.com/syv-ai/qwen38-27b-rtx3090
- https://github.com/redlinedtm-jpg/vllm-v100-2080ti-recipes (docs/turing-2080ti.md, docs/benchmarks.md)
- https://github.com/weicj/vLLM-2080Ti-Definitive (README, README on `vllm-2080ti-definitive-0.2.x` branch, CHANGELOG.md)
- https://github.com/weicj/2080Ti-LLM-Toolbox/blob/main/BENCHMARKS.md
- https://github.com/TnzGit/vLLM-2080Ti-Definitive-dflash2 (README, CHANGELOG.md, docs/dflash2-adaptation/HANDOVER.md, docs/dflash2-adaptation/PHASE2-VERIFY.md)
- https://github.com/zyYuc/twin-turing
- https://github.com/Luce-Org/lucebox-hub/pull/80
- https://github.com/noonghunna/club-3090/blob/main/BENCHMARKS.md
- https://github.com/ziyiguanmomo/QwenForge
- https://github.com/vllm-project/vllm/issues/47549, /38918, /33461, /35414
- https://huggingface.co/Qwen/Qwen3.8-27B-FP8/discussions/9
- https://gpubattle.com/ai/nvidia-quadro-rtx-8000
