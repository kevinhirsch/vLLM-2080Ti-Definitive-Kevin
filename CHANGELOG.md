# Changelog

This changelog tracks the fork release version for vLLM 2080 Ti Definitive
Edition. It is separate from the upstream vLLM package version.

## Unreleased: branch `frontier-pastnative-20260816`

Not a fork release. This branch carries the HNET00 estate's gateway, window-controller and release tooling.
**Complete list:** every commit on the branch since the last changelog entry (274 commits, 2026-08-17 to
2026-10-20) is in [`deploy/docs/CHANGELOG-frontier-pastnative-20260816.md`](deploy/docs/CHANGELOG-frontier-pastnative-20260816.md),
grouped by component.

### Pushed 2026-10-21 (the live ops layer under git, EB-103)

On 2026-10-21 the serving host's live ops layer (`~/.local/share/vllm-qwen27b`) was re-hashed against all 323 refs.
The gateway was already in sync: the live 13,804-line `keepalive-shim.py` has been byte-identical to this branch's
since `765600de46`. Still off this branch, now on it:
- **`gpu-boot-gate.sh`**: production's `serve-hauhaucs-v02.sh` sources it, but it existed only on the unmerged
  `lane-a6`/`lane-bs3` chain and on the host, so a rebuild from this branch booted without the gate.
  - Added the live copy, identical to `lane-a6`, with its test.
  - `serve-profile-v02.sh` gets the live copy, which wires the same gate.
  - `test_serve_v02_pythonpath.py` and `test_serve_v02_model_selection.py` get `lane-bs3`'s hermetic
    `V02_GPU_WAIT_S=0`. Without it, 5 tests wait on the real GPUs.
- **`serve-tqk8v4-fg.sh`**: the copy that last ran on the 0.1.x line (2026-09-28, EXP-048 r6). Not on today's path.
- **Engine env: the live file is the truth** (decided 2026-10-21; no production change).
  - `deploy/env/vllm-qwen27b.env` now matches what production has run on. This branch's copy (synced 2026-09-04
    from the 08-27 state) lacked:
    - `VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE=off` (the 2026-09-07 Xid-31 mitigation)
    - `VLLM_PREFIX_CACHE_USE_RETAINED_MTP_BLOCK=0`
    - `VLLM_TURBOQUANT_FLASHINFER_PREFILL_PLAN_CACHE_MAXSIZE=128`
    - `VLLM_SCHED_SHORT_FIRST_RUN=3`
  - The five variables only the old copy set are **not deployed and need an A/B to adopt** (EB-104). On 2026-10-21,
    production release `w8x-r4full-adepth-af95f8d682` read none of them in its Python, C/C++/CUDA sources or built
    libraries, so discarding them is likely:
    - `VLLM_FLASHQLA_VARLEN_LOOP=1`
    - `VLLM_TURBOQUANT_STAGE1_QTILE=1`
    - `VLLM_TURBOQUANT_CONTINUATION_BOUNDS_CHECK=1`
    - `VLLM_TURBOQUANT_CONTINUATION_SDPA_MAX_QK_CELLS=16777216`
    - `VLLM_TURBOQUANT_CONTINUATION_SDPA_Q_CHUNK=512`
- **Code public, config private** (Kevin, 2026-10-21). The serving host's own files stay on the host and in its
  nightly backup: `clients.map`, `lanes.dirs`, `MODEL-SWITCHING.md`, `v02.override.env`, `v02.model.json`, and the
  secret carriers `shim.env` and `gateway-aliases.json`. Only redacted examples with RFC 5737 placeholders are in git,
  in `deploy/env/host-local/`. `shim.env.example` is rebuilt from the live settings (103 were missing), with
  `SHIM_REMOTE_KEY` blank and client addresses replaced by placeholders.
- **`deploy/bin/live_drift_check.py`**: a read-only check that the live layer still matches a git ref (exit 1 on drift
  or untracked files), with 7 tests.
  - Its allowlist, `deploy/env/live-layer-allow.txt`, is read from the ref, so it is reviewed.
  - It marks state, logs, banked copies and retired launchers `ignore`, and the private host files `local`.
  - It runs nightly inside the host's backup job and reports to estate-events only when the result changes.
- **The spend tests no longer need the 27B running (EB-122).**
  - `test_gateway_spend_authority.py` (Seam) and `test_gateway_spend_v6.py` (Routing) let the shim probe the real
    engine on :8001. So whenever the 27B was down for a GPU window, three tests got 503 on their local path, and so did
    the production-ledger ratchet that reruns them.
  - They now stub the engine's health the way the suite's other routing tests do. The full suite passes with the
    engine down.

### Pushed 2026-10-20 (gateway test debt cleared)

Tests only; no gateway code changed. The full `deploy/bin` suite now passes: **1,093 passed, 657 subtests, about
2 min**. Before this push, 4 tests failed and 2 files hung for 5+ minutes each (the "known test debt" below). Every
fix updates a stale test to behaviour Kevin decided earlier; none relaxes what a test pins.
- **`test_gateway_spend_authority.py`** (3 tests) and **`test_gateway_spend_v6.py`** (1 test): they predated the
  remote-alias degrade (GW3, "Local First Remote For Spikes", on by default), so they asserted a 429 unconditionally.
  Each now pins both settings. With the degrade off (`SHIM_REMOTE_ALIAS_DEGRADE=0`, the kill switch), the request
  gets a 429 and nothing is forwarded. With it on, an explicit remote alias, route-intent overflow or an exhausted
  pro alias is served by the local engine and never by the paid remote. An operator's forced-remote window still
  answers 429: an explicit choice is never silently degraded. 43 and 24 pass.
- **`test_gateway_local_first.py`** (3 tests that hung): since 2026-10-19, a caller that cannot overflow
  (`estate-local`, local pin) waits for a local slot, or is held through a planned window for up to
  `LOCAL_ONLY_HOLD_SECS` (1,800 s), instead of being refused at once. So these tests each waited about 30 minutes.
  - `test_estate_local_never_overflows_when_admission_fills` and
    `test_halo_control_uses_protected_place_when_ordinary_budget_is_full` patch `admission_wait_seconds` to 0.05 s.
  - `test_stall_brake_never_refuses_work_that_only_remote_can_serve` patches `LOCAL_ONLY_HOLD_SECS` to 0.01 s.
  - They still pin the same outcomes: the request ends in a 503, unforwarded and never remote, and Halo's control
    request takes the protected place.
  - The file passes, 31 tests in 2.3 s.
- **`test_no_production_ledger_touch.py`** (the ledger ratchet) passes, 2 tests in about 57 s. It had two separate
  problems:
  - **Hang:** it re-runs every test module in a subprocess, so it hung on the local-first hangs above.
  - **Temp dir left behind:** `conftest.py` created its flight-recorder temp dir (`SHIM_FLIGHTREC_DIR`) without
    removing it. The ratchet requires every temp dir a run creates to be cleaned up, so `conftest.py` now removes
    that dir at exit.
- **`test_rep_retry.py`** now imports its fakes from `conftest_rep_retry` (the module that defines them).
  - It used to import them from `conftest`, which never defined them; the pytest plugin
    `conftest_rep_retry_compat.py` published them there at startup.
  - The ratchet runs the modules under plain unittest, which loads no pytest plugin, so the import failed there.
  - The plugin keeps only the `shim` / `patch_stream` fixtures.

### Pushed 2026-10-20 (gateway update)

- **Strict tool-call history on the remote path** (lane GW-STRICT, via gw-combined): requests forwarded to the paid
  remote get their tool-call history repaired to the strict form DeepSeek requires (orphaned tool results, missing
  call ids), logged as `strict tool history: client=... changes=N`. Local requests are untouched. No switch.
- **Repetition retry** (lane GW-REP-RETRY, via gw-combined): `SHIM_REPETITION_RETRY=off|shadow|live` (default `off`).
  When a local request carrying tools comes back cut by the repetition guard with no tool call, `live` re-issues it once
  with thinking off and hands the client the retry. Turned on in production 2026-10-20 at Kevin's request.
- **`SHIM_LOCAL_ONLY_CLIENTS`** (new): a comma-separated list of X-Client values served exactly like the `estate-local`
  alias. They never reach the paid remote: not for congestion, not during a planned window, not when the local engine
  is down (they are held instead). Set to the agent lanes (`openhands-coordinator`, `openhands-coordinator-deep`,
  `lane-medic`) because Kevin revoked remote use for lanes on 2026-10-20.
- **`gateway_config_schema.py`**: registers six settings the code already read but the schema lacked:
  `SHIM_LOCAL_ONLY_CLIENTS`, `SHIM_LOCAL_ONLY_HOLD_SECS`, `SHIM_DRAIN_WAIT_S`, `SHIM_REPETITION_RETRY`,
  `SHIM_AGENT_CONTEXT_TOKENS`, `SHIM_AGENT_CONTEXT_MODELS`. `test_gateway_config_schema` passes again; it was red since
  `b19cb65f4f`, my miss.
- **Tests:**
  - `test_gateway_flow.py`: two tests updated to the 2026-10-19 behaviour (a local-only caller is held through a planned
    window instead of refused at once), and two added (served locally when the window ends; listed clients never reach
    the remote). They take 7 s instead of 3.5 min.
  - `golden/routing_golden.json`: regenerated. Exactly three scenarios changed, all to the intended hold behaviour
    (`local_down_estate_local_503`, `offline_window_local_pin_503`, `offline_window_no_remote_503`).
  - `test_strict_tool_history.py` and `test_rep_retry.py` (56 tests) come with the two fixes, plus
    `conftest_rep_retry*.py`. In `conftest.py` the compat plugin is registered and the flight recorder goes to a temp dir.
- **Known test debt, failing identically on the code before this change** (all fixed in the next push, "gateway test
  debt cleared", above):
  - `test_gateway_spend_authority.py`: 3 failures, explicit-remote 429 cases;
  - `test_gateway_spend_v6.py`: 1 failure, pro alias;
  - `test_gateway_local_first.py` and `test_no_production_ledger_touch.py`: hang (each over 5 min).
  - The cause is the same: the live gateway was edited outside the repo for weeks and synced back on 2026-10-19 without
    updating these tests.

### Pushed 2026-10-20 (8 commits, `c467b05dad..a4c72e8bb8`)

- **Gateway, bounded graceful stop** (`b19cb65f4f`). On the stop signal, every queued request that was never admitted
  gets an immediate 503 with `Retry-After`. The stop then waits only for requests already sent upstream, for at most
  `SHIM_DRAIN_WAIT_S` (default 600 s). After that it cancels the remaining handler tasks. A restart on 2026-10-19 had
  held the stop for 7 minutes.
- **windowctl rejects specs that shadow built-in variables** (`b19cb65f4f`): `RESULTS`, `WINDOW`, `LANE`, `STEP`,
  `PROD_ROOT`. Test added.
- **Gateway synced with the live HNET00 copy** (`c18b8e8a5c`). The repo copy had been stale since 2026-10-03. This
  brings in `keepalive-shim.py`, `gateway_config_schema.py` and the new `experiment_controls.py`, including:
  - local-only requests are held through planned windows instead of refused;
  - an anti-starvation rule: a class's oldest waiter, held only by the capacity check for `FLOW_STARVE_S`, blocks
    younger admissions until it fits.
  - `spend_reconcile.py` was deliberately not synced: the repo version is newer than the live one.
- **windowctl drain floor** (`c18b8e8a5c`). Accepted local work gets at least `WINDOWCTL_DRAIN_FLOOR_S` (default
  300 s) before the engine stops. The drain time is added to the window budget, and the lease outlives the drain.
- **release.py refuses a build whose source tree has no FlashQLA** (`873d5fbd8d`). Such releases crashed at GDN
  init. `RELEASE_ALLOW_NO_FLASHQLA=1` opts out.
- **windowctl spec preflight before production stops** (`b76f80968a`). `WINDOWCTL_PREFLIGHT=<checker>` refuses a
  window before the engine stops. Read-only script snapshots are made writable before pinning.
- **windowctl script snapshots keep their own directory** (`36f6d67b32`). `$(dirname "$0")` and
  `$(dirname "${BASH_SOURCE[0]}")` resolve to the original script's directory.
- **windowctl timing marker** (`a4ee7b0418`). `WINDOWCTL_BENCH_FILE` exists while a run step measures; steps opt out
  with `"timing": false`.
- **windowctl records the 2026-10-17 lift of overnight-only windows** (`532a427439`). The window daemon sets
  `WINDOWCTL_ALLOW_DAYTIME` and pauses on `windows/HOLD`.
- **Model switching and gateway dashboard** (`a4c72e8bb8`). This work was found uncommitted in the working tree,
  last edited 2026-10-10. It covers `model_switch.py`, the rewritten `switch-model.sh`, model selection in
  `serve-hauhaucs-v02.sh`, the dashboard changes, and their tests. Committed as found; 138 tests pass.

## v0.1.17 - 2026-08-24

- Merges [PR #125](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/125), deduplicating named-tool streaming fallback output and preserving `finish_reason=length` for truncated tool calls.
- Merges [PR #128](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/128), disabling implicit repetition detection for tool-call arguments by default so valid repeated Markdown or code is not truncated.
- Merges [PR #129](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/129), emitting Qwen XML function names once per tool call and omitting repeated `name` fields from continuation deltas.
- Merges [PR #130](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/130), making Qwen XML close-tag recovery call-scoped, cross-chunk safe, and bounded for long streamed arguments.
- Merges [PR #132](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/132), bounding the TurboQuant FlashInfer prefill wrapper cache while preserving CUDA Graph-safe wrapper lifetimes.
- Merges [PR #133](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/133), re-aligning Mamba offload hit boundaries after per-group clamps to preserve hybrid KV state consistency.
- Merges [PR #134](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/134), reserving TurboQuant continuation workspace for the configured maximum context before KV-cache sizing.
- Release credit: @kevinhirsch and @weicj.

## v0.1.16 - 2026-08-20

- Merges [PR #101](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/101), bounding Qwen reasoning blocks without dropping split markers.
- Merges [PR #108](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/108), disabling custom all-reduce during CUDA Graph profiling to avoid the SM75 128K IPC leak path.
- Merges [PR #111](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/111), reserving TurboQuant continuation-prefill workspace before KV-cache sizing to avoid high-context Xid 31 crashes.
- Merges [PR #116](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/116), preventing `None` function names in Qwen3 XML streaming tool-call chunks.
- Merges [PR #120](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/120), adding interactive and non-interactive TP/PP layout selection to the 0.1.x launcher with GPU-factorization validation.
- Release credit: @YuYue1208, @superniker, @kevinhirsch, @hotwa, and @weicj.

## v0.1.15 - 2026-08-15

> Deployment status on this fork: MTP speculative decoding remains **OFF** in
> the live serve config (2026-08-15 campaign decision after the 57-64K
> incident). The MTP correctness fixes below make re-enabling *plausible*, not
> automatic — it stays off until every gate in
> `deploy/docs/QWEN-3.8-MTP-REQUALIFICATION.md` (G1-G5) is green.

- Validates the official
  [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) checkpoint
  end to end on the dual RTX 2080 Ti TP=2 runtime. The Qwen3.x 27B FP8 route
  now explicitly covers official Qwen3.6 and Qwen3.8 checkpoints.
- Merges [PR #85](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/85),
  improving build download-route preflight, PyTorch fallback
  behavior, and automatic `MAX_JOBS` limits for reliable source builds.
- Merges [PR #89](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/89),
  preserving hybrid Mamba prefix-cache correctness when MTP
  speculative decoding is enabled.
- Merges [PR #91](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/91),
  adding native CPU KV offload support for the Mamba align
  cache path.
- Merges [PR #93](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/93),
  preserving the GDN causal-convolution state slot zero during
  SM75 decode and adding its regression coverage.
- Merges [PR #98](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/98),
  completing named tool-choice response handling for full and
  streaming OpenAI-compatible chat completions, including Mistral-compliant
  IDs and empty-argument handling.
- Release credit: @weicj, @YuYue1208, and @0xYYP.

## v0.1.14 - 2026-07-07

- Merges [PR #78](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/78),
  improving the validated SM75 TurboQuant long-context route with
  the tested continuation prefix-combine path, tuned decode `BLOCK_KV=2`
  defaults, reproducible long-context benchmark controls, and launcher/runtime
  plumbing for the shipped TurboQuant throughput lane.
- Merges [PR #81](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/81),
  finalizing the non-interactive launcher override path with
  explicit `CLI > ENV > PROFILE > default` precedence, `CUDA_VISIBLE_DEVICES`
  mapping, mode-derived override hygiene, and matching English / Simplified
  Chinese launcher documentation.
- Release credit: @0xYYP and @weicj.

## v0.1.13 - 2026-07-04

- Merges [PR #71](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/71),
  fixing the SM75 TurboQuant TQK8V4 FP8 key-format path and the
  launcher submenu numeric-selection regression.
- Merges [PR #72](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/72),
  adding the first validated Docker runtime packaging path for
  the SM75 fork, including the runtime Dockerfile, compose example, entrypoint
  flow, and packaged helper assets needed to launch the shipped profiles
  inside a container.
- Extends the merged Docker path with the validated host-side `docker/build.sh`
  wrapper so Docker builds follow the repository `build.sh` behavior for
  automatic `MAX_JOBS` selection, download-route preflight, and consistent
  PyPI/Git mirror propagation through Docker and compose builds.
- Release credit: @0xYYP and @hotwa.

## v0.1.12 - 2026-07-02

- Integrates the SM75 custom all-reduce graph-input auto policy so the official
  Qwen3.6 35B FP8 release route can keep the validated fast decode path while
  avoiding the PIECEWISE graph-capture startup crash.
- Finalizes the current build/install reliability fixes, including safer build
  parallelism defaults, broken virtualenv self-heal, dependency checkout reuse,
  and rebuild recovery for fresh source installs.
- Adds the validated official Qwen3.6 35B FP8 profile set: 256K text-only
  `normal` and `aggressive`, 136K text+image `normal` and `aggressive`, and a
  178K `fast` MTP3 route.
- Fixes the shipped Qwen3.6 27B `fast` TQK8V4 256K prefix-cache continuation
  route by promoting the validated `GPU_UTIL=0.96` profile values and making
  the launcher auto-reserve the larger continuation workspace for long
  single-sequence TurboQuant lanes.

## v0.1.11 - 2026-06-29

- Updates the launcher defaults for long-context serving: prefix cache is now a
  launcher-level default, prompt token details are enabled for status
  visibility, and Qwen routes automatically use the cache mode required by the
  validated prefix-cache path.
- Adds launcher-side startup safeguards for large profiles, including
  cold-compile prewarm retry and display-GPU occupancy warnings.
- Retunes the shipped Qwen3.6 TQK8V4 `fast` profiles for the validated prefix
  cache path by raising their max batched tokens to `2560`.

## v0.1.10 - 2026-06-17

- Fixes the Qwen reasoning startup path by keeping `reasoning_content`
  compatible with clients that still read the legacy field name.
- Unifies chat reasoning and tool-call parsing so `content=None` responses do
  not break named tool choice handling, and adds regression coverage for the
  reasoning-only null-content path.
- Fixes Qwen GDN mixed prefill/decode handling when mixed batches transition
  through the recurrent attention path, including the TurboQuant attention state
  handling needed by that route.
- Adds the explicit GDN prefill metadata used by the validated mixed
  prefill/decode path, and re-validates the FP8 FP16-KV 128K MTP3 route on
  dual RTX 2080 Ti.

## v0.1.9 - 2026-06-15

- Restores the validated Qwen3.6 27B FP8 + TurboQuant TQK8V4 `fast` route on
  SM75 by pinning the working FlashInfer 0.6.8 runtime path and guarding
  unvalidated newer FlashInfer ragged-prefill behavior.
- Adds TurboQuant launcher defaults for the validated FlashInfer `fa2` prefill,
  workspace reservation, and speculative decode fast path used by the tested
  dual 2080 Ti route.
- Carries FlashQLA legacy SM70/SM75 patch assets into source builds so fresh
  builds reproduce the validated GDN prefill path.
- Improves one-click build reliability with network preflight, package/git
  mirror fallback, local wheelhouse support, CUDA toolkit sanity checks, and
  clearer hardware/environment hints.
- Adds `update.sh` for release-version checks and optional rebuild after an
  update.
- Documents the recommended host runtime as Ubuntu 22.04/24.04 LTS or Debian
  12 on Linux kernel 6.x. Ubuntu 26.04 / CUDA 13 remains a community
  experiment until runtime profiles are validated.

## v0.1.8 - 2026-06-14

- Improves source-build reliability for non-Miniclaw dual 2080 Ti hosts,
  including automatic FlashQLA SM70/SM75 install and clearer runtime path logs.
- Fixes local launch reliability around service stop, failed-start cleanup,
  generated-kernel cache isolation, and default `normal` mode handling.
- Adds an explicit `aggressive` launcher mode: a more aggressive mode with the
  highest performance and quality risk.
- Tightens SM75 backend selection so Turing hosts do not accidentally prefer
  unsuitable FlashAttention paths over the validated FlashInfer/TurboQuant
  routes.
- Carries the SM75 TurboQuant compatibility fixes for workspace reservation,
  sliding-window decode, and legacy runtime environment handling.
- Refines Qwen GDN / linear-attention decode correctness and removes debug-only
  overhead from the normal serving path.
- Updates hardware notes for CPU/platform latency after local build-to-launch
  validation showed old Xeon hosts can underperform newer low-end desktop CPUs.

## v0.1.7 - 2026-06-13

- Updates the `safe`, `normal`, and `fast` launch modes. `normal` is now the
  default recommended mode.
- Fix OpenAI-compatible tool calling, including automatic tool choice and
  structured tool-call output.
- Fixes SM75 TurboQuant compatibility gaps reported against the legacy
  `fast` + INT8-KV route: all ubatch workspace reservation, decode
  `sliding_window` handling, and fork-specific runtime environment variables.
- Improves the current Qwen and Gemma serving routes with updated stability,
  long-context, and profile validation work.
- Includes the Gemma4 checkpoint-name remap fix for official QAT and
  multimodal-style checkpoints.


## v0.1.6 - 2026-06-09

- Adds explicit W8A8 checkpoint support documentation for the Quark INT8 route,
  including the tested `nameistoken/Qwen3.6-27B-Quark-W8A8-INT8` checkpoint.
- Updates the launcher display to separate the real vLLM `--quantization`
  value from the display-only W/A type (`W4A16`, `W8A16`, `W8A8`).
- Adds launcher service-status cache reporting for running services. Live
  `used` values and 30-second refresh are shown only when vLLM exposes real
  cache-usage metrics; otherwise the launcher reports total cache capacity only.
- Improves launcher startup preflight for large single-file checkpoints and
  cleans up residual vLLM processes after failed launches.
- Fixes launcher stop handling so orphaned vLLM API servers, worker processes,
  and resource trackers are discovered and cleaned up instead of leaving VRAM
  occupied.

## v0.1.5 - 2026-06-08

- Renames the public service manager to `launcher.sh` and keeps `build.sh` as
  the one-click source build entry point.
- Updates launcher modes to `safe`, `normal`, and `fast`, with route profiles
  split by model, mode, and weight precision.
- Adds chat template presets and service-level thinking budget defaults while
  keeping global runtime controls out of route profile files.
- Refreshes the Qwen3.6 profile documentation and restores the KV throughput
  sweep SVG charts.

## v0.1.4 - 2026-06-06

- Slims the public repository down to the focused SM75 runtime source tree,
  launcher scripts, validated profiles, and project documentation.
- Keeps Docker artifacts out of this source release; Docker packaging remains a
  separate future deployment layer.
- Adds the interactive `launcher.sh` service manager and one-click `build.sh`
  source build entry point.
- Uses the public launcher modes `safe`, `normal`, and `fast`, with validated
  profiles organized under model-specific profile directories.
- Carries forward the `v0.1.3` graph-safety runtime fixes while removing
  upstream CI/docs/test bulk from the public source tree.

## v0.1.3 - 2026-06-05

- Adds the issue #24 MTP graph-safety fix for hybrid Mamba/GDN models.
- Makes production profiles safer by default: Native MTP + hybrid recurrent KV
  layers fall back from full decode CUDA Graph replay to PIECEWISE/NONE.
- Keeps the old peak-throughput route available for explicit speed benchmarking
  via `VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=1`.

## v0.1.2 - 2026-06-04

- Public stable snapshot for the SM75 TP=2 CUDA 12.8 runtime.
- Keeps the upstream vLLM base at `0.21.0` while versioning this fork as an
  independent 2080 Ti runtime distribution.
- Updates the documented Qwen3.6 and Gemma4 runtime routes, tested checkpoint
  list, launcher profile guidance, and benchmark evidence links.

## v0.1.1

- Follow-up compatibility fixes for editable/source bu
