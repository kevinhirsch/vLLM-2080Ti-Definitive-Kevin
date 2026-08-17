# vLLM-2080Ti-Definitive — Kevin's fork

Fork of [weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive)
(vLLM for Turing SM75), serving **Qwen3.8-27B** as a fully-local agentic stack on
**2× RTX 2080 Ti 22 GB (TP=2, NVLink)**. Everything below is measured on that
hardware, not estimated.

## Headline numbers (2026-08-17)

| Axis | Production | Best proven |
|---|---|---|
| Context | **440K-token fence** (crash-free by construction) | **491,563-token** uncached prefill, needle-verified |
| KV pool | **741,146 tokens**, deterministic across boots | — |
| Decode (single stream) | **~67 tok/s** (MTP K=2) | **811.5 tok/s** on verbatim-copy spans (scoped drafter, branch) |
| Decode (mixed agent work) | 72 tok/s | **102 tok/s** (K=16 MTP chain, branch) |
| Concurrency | **16 lanes**, 369 tok/s aggregate (agent-shaped) | knee mapped: 24 lanes regresses |
| Hot-prompt TTFT (repeat prefix) | **2.2 s** (was 4.1 s) | via MTP mamba-cache retention |
| Quality | evalkit **92/100** — parity with the pre-optimization baseline on every ship | 100-item deterministic gate in `deploy/evalkit/` |

Serving config: GPTQ-Int4 (self-quant, gptqmodel), `turboquant_k3v4_nc` KV
(3-bit K / 4-bit V), mamba-cache-mode `align`, `max-num-batched-tokens 3584`,
MTP K=2 with the dense-draft fix, YaRN past-native RoPE, chunked prefill,
prefix caching, froggeric v22 chat template, `qwen3_xml` tools.

## What this fork adds over upstream

**Fixes (measured, root-caused):**
- **Workspace-reserve KV sizing** ([upstream PR #111](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/111)):
  the turboquant continuation-prefill workspace was invisible to KV-cache sizing —
  high-`max-model-len` configs over-committed VRAM and died with Xid 31 illegal-access
  instead of a clean error. Reserve-before-sizing + deterministic arena accounting
  (two review rounds). Confirmed by intervention: the same 380-450K prefills that
  reliably crashed run clean, and boot-to-boot KV variance (~1.1 GiB) is gone.
- **MTP dense-draft load fix**: the GPTQ layer matcher wrapped the (dense-BF16)
  MTP head in a quantized kernel → silently corrupted drafts → the historical
  "MTP is broken on this box" verdict. `VLLM_QWOPUS_MTP_BF16_DRAFT=1` restores it:
  acceptance p0=0.79/p1=0.56, **+54 % single-stream decode**.
- **RoPE cache sizing past native** (`VLLM_ROPE_MAX_POSITION`): positions beyond
  262,144 no longer index out of bounds — unlocked 352K → 491K serving.
- **k3v4 align-mode floor**: 3-bit K requires `max-num-batched-tokens ≥ 3488`
  (align block); documented + configured.
- Named-tool-choice streaming: truncated tool calls now report `finish_reason="length"`
  (never `"tool_calls"` — clients must not execute cut-off JSON), fallback
  deduplicated onto the shared helper, with regression tests.
- Community cherry-picks, verified here: upstream **#92** (MTP mamba cache
  retention — the hot-prompt TTFT halving above) and **#108** (custom-allreduce
  profiling IPC leak).

**Serving primitives (branch `feat-gdn-snapshot`, env-gated, stages 0-4 hardware-proven):**
- **Hybrid-state snapshot / pin / restore / fork**: prefill a context once, fork it
  into N children with divergent sampling — 98 % cache reuse per child, byte-exact
  greedy continuation, zero leaked blocks. Exposed over HTTP (`/tq/pin|fork|unpin`)
  and consumed by the `deploy/localflow` orchestrator (fork-research demo runs
  end-to-end). The primitive no cloud API offers.

**Speed (branch `feat-s4-scoped-drafter`):**
- **FSM-gated scoped re-emission drafter**: when generation is verbatim-copying the
  prompt (44.9 % of agent-workload tokens by census), draft the copied span in
  16-token chunks — **811.5 tok/s** measured on copy-shaped work, no effect on
  free generation. Plus K=16 MTP chaining (no drafter): mixed work 72 → 102 tok/s.
  Ship pending a spec-decode memory-profiling fix (in progress on the branch).

**Ops (`deploy/`):**
- `bin/keepalive-shim.py` — local-first gateway with DeepSeek overflow, lane
  guarantees (admission budget + per-request cap + monster-in-flight bypass),
  think/repetition guards, crash-adaptive limits, live-tunable via HTTP.
- `localflow/` — deterministic multi-agent orchestrator (a local
  Claude-Code-Workflows clone): `agent/parallel/pipeline/phase`, guided-JSON,
  journal+resume, overflow-by-design, `fork_agents` over the fork primitive.
- `evalkit/` — the 100-item deterministic quality gate every ship must pass.
- `bench/` — decode/tool-call/equivalence/MTP-requalification benches
  (fail-closed on missing telemetry).
- Serve scripts, systemd units (watchdog, restart-always), chat templates.

## Docs
`docs/past-native-context.md` (the 262K→491K recipe) ·
`docs/frontier-changelog-20260816.md`, `docs/frontier-session2-20260816.md`,
`docs/frontier-session3-20260817.md` (measured session logs) ·
`docs/exp038-*.md`, `docs/exp039-*.md`, `docs/exp045b-*.md` (designs + evidence) ·
`docs/UPSTREAM-PR-PLAN.md`.

## Known open items
- Residual crash class: MTP + `max-model-len ≥ ~508K` faults under large requests
  (threshold bracketed to (460800, 507904]; production fenced below it;
  instrumentation hunt in progress).
- Scoped-drafter prod ship blocked on the K>2 spec memory-profiling fix.
- Fork-HTTP v2 (non-blocking, stop-token-correct children) designed, not built.

## Branch map
`frontier-pastnative-20260816` — **the serving branch** (all shipped fixes + deploy/).
`feat-gdn-snapshot` — fork primitive + HTTP. `feat-s4-scoped-drafter` — drafter.
`feat-kv-workspace-reserve` — the PR #111 patch. `steal-pr108-pr92` — community
cherry-picks (now merged to serving). `upstream-pr-workspace-reserve` — PR #111 head.

---
*Upstream credit: [weicj](https://github.com/weicj/vLLM-2080Ti-Definitive) for the
SM75 fork this builds on; community PRs #92/#108 by their respective authors.*
