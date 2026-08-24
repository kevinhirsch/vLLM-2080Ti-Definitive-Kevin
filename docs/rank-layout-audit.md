# M-4: Rank-Layout Hazard Audit

**Scope.** vLLM lays out global ranks as `(ExternalDP x) DP x PP x PCP x TP`,
DP outermost among the model-parallel dims, TP innermost
(`vllm/distributed/parallel_state.py`, `initialize_model_parallel`, the
`all_ranks = torch.arange(world_size).reshape(-1, dp, pp, pcp, tp)` call
around line 1568). `ParallelConfig.world_size` is normally `PP*TP*PCP`, but
when `distributed_executor_backend == "external_launcher"` it is multiplied
by `data_parallel_size` (`vllm/config/parallel.py::__post_init__`, ~line
762-770). Any code that receives a **per-worker Python list** (one entry per
executor worker, e.g. `available_memory`, `kv_cache_specs`) and assumes "the
last PP stage is a contiguous tail slice of that list" breaks once DP is
folded into that list's length, because DP is the *outer* dimension, not the
inner one — the last-PP-stage workers are scattered one-per-DP-group, not
bunched at the end.

We already fixed one such site: `spec_verify_last_stage_mask()` in
`vllm/v1/core/spec_decode_workspace.py`, consumed by
`vllm/v1/core/kv_cache_utils.py::get_kv_cache_configs`. It derives stage
membership as `(idx // (pcp*tp)) % pp_size == pp_size - 1`, which is correct
under both the normal and the DP-folded layout, and falls back to "reserve on
every worker" if `n_workers` isn't a multiple of `inner_size * pp_size`
(safe-over-reserve rather than a silently-wrong tail slice).

**Method.** Grepped `vllm/` for tail-slice-shaped and rank-arithmetic patterns
(`per_stage`, `world_size // `, `pp_size - 1`, `[-tp_size`, `last_stage`,
`is_last_rank`, `n_workers`, `projected_groups_per_worker`, `pp_tp_workers`,
FORK-marked sites), then read each hit's surrounding function to determine
whether it (a) is a **positional list index/slice** keyed by executor-worker
rank, or (b) a **local, per-rank boolean/scalar** derived from the process's
own group membership (e.g. `get_pp_group().is_last_rank`) or from
multi-dimensional tensor reshape/transpose (which is layout-order-agnostic by
construction). Only (a)-shaped sites can exhibit the M-4 hazard.

**Key structural finding.** The hazard as stated (`available_memory`-shaped
list, sliced positionally, length == world_size including DP) is only
reachable if an executor can present a per-worker Python list whose length
spans multiple DP groups. It cannot, currently: `distributed_executor_backend
== "external_launcher"` always resolves to `ExecutorWithExternalLauncher`
(`vllm/v1/executor/abstract.py::Executor.get_class`), a `UniProcExecutor`
subclass that manages **exactly one** local `driver_worker` per OS process
(one process per torchrun rank). Its `determine_available_memory()`
(`vllm/v1/executor/uniproc_executor.py`) MIN-reduces over
`get_world_group()` (the *entire* DP-folded world) and returns a
single-element list. `MultiprocExecutor` and `RayExecutorV2` — the executors
that *do* manage multi-worker per-rank lists — both assert
`world_size == tp_size * pp_size * pcp_size` at init
(`vllm/v1/executor/multiproc_executor.py:117`,
`vllm/v1/executor/ray_executor_v2.py:264`), i.e. they refuse to run with a
DP-folded world_size at all; DP for those backends is realized as multiple
independent `EngineCore` processes (one per DP rank), each with its own
un-folded `world_size = PP*TP*PCP`. So today, no `collective_rpc`-sourced
per-worker list ever actually has DP folded into its length — the
`spec_verify_last_stage_mask` fix hardens `kv_cache_utils.py` for this shape
regardless (defense in depth / correct-by-construction), but per this audit
no *other* currently-live site is exposed to it either. This finding doesn't
reduce the value of the audit — it explains *why* the hazard class is narrow
today and flags exactly what would make it live (a future executor that
fans out per-worker RPC across DP ranks within one process, e.g. a
multi-DP-in-one-process variant of `ExecutorWithExternalLauncher`).

## Findings

| # | Site | Pattern | Verdict | Why |
|---|------|---------|---------|-----|
| 1 | `vllm/v1/core/spec_decode_workspace.py::spec_verify_last_stage_mask` | Per-worker-list stage derivation, `(idx // inner) % pp == pp-1` | SAFE (reference fix) | Derives stage from the PCP*TP inner block size, which holds under both normal and DP-folded layouts; falls back to reserve-on-every-worker if `n_workers` isn't a clean multiple. This is the pattern the rest of the audit is checked against. |
| 2 | `vllm/v1/core/kv_cache_utils.py::get_kv_cache_configs` (all `zip(projected_groups_per_worker, available_memory)` loops, `_auto_fit_max_model_len`, `num_gpu_blocks_override` loop, final per-worker config loop) | Per-worker list consumption | SAFE | Every consumer of `available_memory` / `kv_cache_specs` / `projected_groups_per_worker` here is an element-wise `zip()` over same-length lists (order-preserving, no positional tail-slice assumption) except the one spot that needs stage membership, which delegates to site #1. |
| 3 | `vllm/v1/executor/uniproc_executor.py::ExecutorWithExternalLauncher.determine_available_memory` | MIN-reduce across `get_world_group()`, returns 1-element list | SAFE | Structural reason hazard class #1 is narrow: this executor always has exactly one local worker, regardless of PP/TP/PCP/DP size, so no in-process per-worker list ever has length > 1 here. |
| 4 | `vllm/v1/executor/multiproc_executor.py:117` (`assert self.world_size == tp_size * pp_size * pcp_size`) | Init-time invariant guard | SAFE | Explicitly refuses a DP-folded `world_size`; `MultiprocExecutor` is never selected for `external_launcher` (`Executor.get_class` routing), so this path never sees DP folded in. |
| 5 | `vllm/v1/executor/ray_executor_v2.py:264` (same assert) | Init-time invariant guard | SAFE | Same reasoning as #4; Ray backends are a disjoint code path from `external_launcher`. |
| 6 | `vllm/v1/executor/ray_executor.py::pp_tp_workers`, `execute_model_ray` (`last_pp_rank = len(self.pp_tp_workers) - 1`) | Looks like tail-slice arithmetic | SAFE | `pp_tp_workers` is built as an explicit 2D `[pp_rank][tp_rank]` grouping (not a flat per-global-rank list), so "last PP stage" is `pp_tp_workers[-1]` by construction, not an inferred tail slice of a flat list. Also Ray is never used for `external_launcher`, so DP-folding never applies here regardless. |
| 7 | `vllm/config/parallel.py::ParallelConfig.__post_init__` — `self.data_parallel_rank = int(os.environ["RANK"]) // (self.world_size // self.data_parallel_size)` | Rank-to-DP-group arithmetic | SAFE | Upstream code (no `[FORK]` marker), only reached when `distributed_executor_backend == "external_launcher"`. `world_size` is already DP-multiplied at this point, so `world_size // data_parallel_size == PP*TP*PCP`; `RANK // that` correctly recovers the DP-group index because DP is the outer dimension of the multiplied layout (matches the `all_ranks.reshape(-1, dp, pp, pcp, tp)` order, `ExternalDP` defaults to size 1). |
| 8 | `vllm/distributed/parallel_state.py::initialize_model_parallel` (`_TP`/`_DCP`/`_PCP`/`_PP`/`_DP`/`_EP`/`_EPLB` group construction) | Multi-dim tensor `reshape`/`transpose`/`unbind` over `all_ranks` | SAFE | This is the layout ground truth every other site must agree with. It indexes ranks via genuine 5D tensor ops (reshape to `(-1, dp, pp, pcp, tp)`, transpose the target dim to last, unbind), which is layout-order-agnostic by construction — there is no flat positional tail-slice here to get wrong. |
| 9 | ~150 occurrences of `get_pp_group().is_last_rank` across `vllm/model_executor/models/*.py` (every PP-aware model definition) | Per-rank local boolean | SAFE | `is_last_rank` is a property of the calling rank's own membership in the `_PP` process group (`vllm/distributed/parallel_state.py:446`), evaluated independently on each rank. It is never used to index into a per-worker Python list, so DP-folding is irrelevant — each rank asks "am I last?" about itself. |
| 10 | `is_last_pp_rank` / `pp.is_last_rank` cached-at-init uses in `vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/worker/gpu_worker.py`, `vllm/v1/worker/gpu/model_runner.py`, `vllm/v1/worker/gpu/cudagraph_utils.py`, `vllm/v1/worker/gpu/pp_utils.py`, `vllm/v1/worker/gpu/warmup.py` | Per-rank local boolean (cached) | SAFE | Same reasoning as #9 — cached once from `get_pp_group().is_last_rank` at construction time, always a local per-rank fact, never a list index. |
| 11 | `vllm/v1/executor/ray_utils.py::_is_last_rank` | Wraps `get_pp_group().is_last_rank` | SAFE | Same reasoning as #9; also Ray-only, disjoint from `external_launcher`. |
| 12 | Fork-added VRAM reserve formulas: `spec_decode_workspace.py` (`spec_verify_reserve*`), `kv_cache_utils.py::_turboquant_prefill_workspace_reserve_bytes`, `gpu_worker.py` TQ workspace add-back (~line 502-530) | Scalar reserve math | SAFE / not applicable | All operate on scalars local to the current rank/process (`vllm_config`, `self.*`); none index a per-worker list. |
| 13 | GDN/#51508-port sites: `vllm/v1/attention/backends/gdn_attn.py` (~317-337), `vllm/v1/worker/mamba_utils.py` | Per-request row masking (`num_accepted_tokens <= 0`) | SAFE / not applicable | Operates on per-request rows within a single rank's local batch, not per-worker/rank lists. Orthogonal hazard class; covered separately under Task 2's test debt. |
| 14 | `vllm/model_executor/layers/fla/ops/chunk_delta_h.py`, `chunk_o.py` (GDN kernel launch config), `vllm/model_executor/layers/mamba/gdn_linear_attn.py` (EXP-046 varlen unbatching) | Kernel/tile config | SAFE / not applicable | Per-sequence/kernel-tiling logic, no per-worker rank list. |
| 15 | `vllm/distributed/device_communicators/cuda_communicator.py`, `custom_all_reduce.py` (custom-AR profiling disable flag) | Module-level bool flag toggle | SAFE / not applicable | Boolean env-gated flag, not a list/rank computation. |
| 16 | `vllm/envs.py` FORK entries, `vllm/v1/core/sched/scheduler.py` FORK entry (MTP/EAGLE cache-block retention) | Env var declarations; per-request block-alignment logic | SAFE / not applicable | No per-worker list or rank arithmetic involved. |

## Verdict summary

- **Sites audited:** 16 (1 reference fix + 15 candidates found via the grep patterns and `[FORK]` marker sweep).
- **HAZARDs found:** 0.
- All fork-added and upstream sites that touch per-worker lists or PP-stage
  membership either (a) delegate to the already-fixed
  `spec_verify_last_stage_mask` pattern, (b) use per-rank local booleans
  (`is_last_rank`) that are layout-agnostic, (c) use genuine multi-dimensional
  tensor reshapes that are layout-order-agnostic, or (d) run on executors
  (`MultiprocExecutor`, `RayExecutorV2`, Ray backend generally) that assert
  away the DP-folded-`world_size` shape entirely and are never selected for
  `external_launcher`.
- **Residual risk (not a current HAZARD, but worth tracking):** the hazard
  class is real in principle and only closed today because
  `ExecutorWithExternalLauncher` happens to manage exactly one worker per
  process. If a future executor variant fans out `collective_rpc` across
  multiple DP ranks from a single process under `external_launcher` (or DP
  folding is extended to `mp`/`ray` backends), every per-worker-list consumer
  in `kv_cache_utils.py` would need re-auditing against that new shape — and
  any *new* PP-stage-membership helper added elsewhere should follow the
  `spec_verify_last_stage_mask` pattern (`idx // inner_size % pp_size`) rather
  than tail-slicing.
