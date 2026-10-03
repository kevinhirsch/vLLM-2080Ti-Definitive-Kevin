# K6 lead M (L91): intra-GPU prefill/decode multiplexing

Status 2026-10-03: the model is done, the lane plumbing has landed (env-gated, default OFF, CPU-tested), and the GPU measurements are queued for window row 4. Code lives on branch `k6/pd-disagg`.

## Why

Measured on the trace (`tools/k6/pdsim.py`, 8,193 local requests, calibrated to TTFT p50/p90 7.6/55 s), today's TP2 engine time-slices.

- A decoding stream waits through every 1,856-token prefill chunk, which takes 1.2-2.5 s.
- `VLLM_SCHED_PREFILL_SHARE=0.75` then pauses prefill 25% of the time so decodes can catch up.
- The result is a per-stream decode p50 of 8.5 tok/s, with gaps of 1.8-2.7 s.

Decode is memory-bound and prefill is compute-bound. They can overlap on the same SMs if decode sits on a high-priority stream.

Modelled gain on the 37 h trace, at 1.3-2.0x decode slowdown and 10-30% prefill loss:

| Metric | Today | With M |
|---|---|---|
| e2e mean | 42 s | 21-31 s |
| decode p50 | 14 tok/s | 23-35 tok/s |
| TTFT p50 | 7.8 s | 4.0-6.4 s |

1P1D fails for two reasons that M avoids:

- **It halves prefill FLOPs.** M keeps both cards on prefill.
- **It replicates the weights.** That shrinks the KV/prefix cache 3-4x. M keeps one shared pool.

## Shape

- **Decode lane.** This is the existing engine main loop, unchanged. It runs FULL CUDA graphs on a high-priority stream, together with short prompts (the existing short-first path).
- **Prefill lane.** This is a background thread with a second, scheduler-less `GPUModelRunner`. It shares `model` and `kv_caches` with the main runner and has its own input batch, attention-metadata builders and persistent buffers.
  - It is fed synthetic `SchedulerOutput`s containing one long prompt, cut into block-aligned chunks. Mamba-align mode needs whole 1,856-token blocks.
  - It runs eager or piecewise-compiled on a low-priority stream. Chunks are larger than the 64-token graph sizes.
- **Handoff.** The handoff reuses the KV-connector async-load path (`WAITING_FOR_REMOTE_KVS`).
  - The scheduler allocates the prompt's blocks, with prefix hits honoured, and parks the request.
  - The lane fills the KV blocks and GDN state slots.
  - `get_finished()` releases the request with `num_computed_tokens = prompt - 1`, using the same N-1 rule NIXL applies to Mamba models.
  - The main lane then computes the last token and samples, so sampling, logprobs and MTP stay in one place.
  - Prefix-cache blocks are committed by the normal connector flow.
- **Routing.** A request goes to the lane when its uncached prompt is at least `VLLM_K6_MUX_MIN_TOKENS`. Default: one attention block.

## Shared-state audit

Two lanes run the same layers at the same time. This is what must not be shared.

| State | Owner | Status |
|---|---|---|
| Weights | shared, read-only | ok |
| KV blocks, GDN/conv state slots | per request, disjoint | ok (scheduler allocates) |
| Marlin split-K lock workspace (per layer) | **per lane** | hook landed: `k6_mux.marlin_workspace`. The decode lane keeps the captured address; the prefill lane gets private zeroed copies. Sharing it would corrupt or deadlock concurrent GEMMs of the same layer. |
| ForwardContext (module global) | **per lane** | hook landed: thread-local for the prefill lane. DBO ubatching writes the global directly, so DBO is incompatible (it is off here). |
| TP GroupCoordinator, incl. compiled `vllm.all_reduce/all_gather/reduce_scatter` ops (resolve by name) | **per lane** | hook landed: `get_tp_group()` and `_resolve_group()` return the lane's group |
| Attention metadata builders, persistent input buffers, block tables | per runner | second runner instance (to build) |
| TurboQuant continuation-prefill workspace (1.0 GiB/rank reserve) | prefill only | prefill lane only. The main lane must not run long continuation prefills while M is on. |
| S2 `tq_gqa_ext` decode scratch, MTP drafter, sampler | decode only | main lane only |
| Caching allocator | per stream | lane tensors stay on the lane stream; `record_stream` at any cross-lane handoff |
| Legacy default stream | nobody | any legacy-stream op serialises both lanes; audit with `CUDA_LAUNCH_BLOCKING=0` plus an nsys check in the prototype |

## CUDA graphs

- **Capture.** Decode FULL graphs are captured at boot on the graph-capture stream, with CustomAllreduce **A** buffers registered.
- **Replay.** Replaying them into the high-priority stream inherits that stream's priority. We do not instantiate with `cudaGraphInstantiateFlagUseNodePriority`.
- **The prefill lane never captures.** So CustomAllreduce **B** needs no graph-buffer registration. Its eager path copies into B's pre-registered IPC buffer.
- **Capture must never overlap lane activity.** `torch.cuda.graph` uses global capture mode, so CUDA work from another thread during capture raises an error.
  - V1 captures every size at startup, so the lane thread must start after capture.
  - If anything ever re-captures, the lane must be paused first. The prototype will add an assert.
- **Piecewise graphs are unaffected.** The main lane's piecewise graphs only cover sizes up to 64 tokens. They are not used for lane chunks.

## Communicators (K3 constraint: one collective in flight per communicator)

### Option (a), chosen: a second communicator for the prefill lane

- **Group assignment.**
  - The decode lane keeps group **A**: CustomAllreduce A plus pynccl A. Its graphs are bound to A's buffers.
  - The prefill lane gets group **B** (`tp:0-k6prefill`): a second CustomAllreduce plus a second pynccl comm. B is built on the same CPU group at worker init, after capture.
- **Cost.**
  - CustomAllreduce B: ~32 MiB/GPU (meta + 8 MiB uncached, 16 MiB IPC buffer, 8 MiB rank_data).
  - NCCL comm B: about 50-200 MiB/GPU, estimated from its channel buffers.
  - Both are **measured** by `mux_ar_bench.py` as `second_custom_ar.mib` and `second_nccl_comm.mib`, together with any per-op overhead (decode alone on A vs B).
  - 32-232 MiB is 0.3-2.5% of the 9.3 GiB pool, i.e. 3-25K tokens.
- **K3 compatibility.**
  - K3's overlap op calls `get_tp_group()`, so in the prefill lane it automatically uses group B.
  - K3 issues its chunk all-reduces sequentially on one side stream, so the one-in-flight rule holds per communicator.
  - The decode lane never enters K3's path. Its sizes are below `VLLM_K3_AR_OVERLAP_MIN_TOKENS`, and it runs under graph capture.
  - Decode all-reduces are at most 64 rows x 5120 x 2 B = 655 KB, which is below CustomAllreduce's 8 MiB limit. The decode lane therefore never falls through to NCCL, and NCCL comm A sees no traffic while M is on. B's NCCL comm carries the 19-37 MB prefill all-reduces.

### Option (b), rejected: one shared communicator plus a lock

- A CPU lock cannot cover graph-replayed collectives: one replay launches 128 all-reduces.
- A GPU-side lock needs cross-stream event waits per all-reduce. Inside a FULL graph that means capturing external event nodes, and the graph would then wait on prefill.
- Decode all-reduces would queue behind 19-37 MB prefill all-reduces of about 0.5-0.9 ms each. Prefill all-reduce duty is about 5%, which costs roughly 3 ms per decode step on average with a far worse tail. Compute would still overlap, but the latency win shrinks.

### Deadlock analysis

- Custom all-reduce kernels spin on peer flags. A spinning block on rank 0 waits for rank 1's kernel to get SMs.
- With decode at the device's greatest priority, rank 1's decode blocks are dispatched as soon as running prefill blocks retire (milliseconds). The wait is bounded, so no cycle forms.
- **Mitigations:**
  - `NCCL_MAX_NCHANNELS=2` on comm B so its spinners never fill the SMs.
  - The bench watchdog reports a DEADLOCK instead of hanging.
  - A constant all-reduce checksum in every lane detects barrier corruption.

## Stream priorities

`k6_mux.choose_priorities()` enforces this order (lower number = higher priority):

| Stream | Priority |
|---|---|
| decode lane | greatest the device offers, e.g. -5; never below K3 |
| K3 all-reduce side stream | `VLLM_K3_AR_OVERLAP_PRIO`, default -1 |
| prefill compute | 0; never above K3 |

Overrides: `VLLM_K6_MUX_DECODE_PRIO`, `VLLM_K6_MUX_PREFILL_PRIO`.

## CPU / GIL

- The prefill lane's eager launch of a 1,856-token chunk takes roughly 15-30 ms of Python across 64 layers. The decode step's Python (schedule + inputs + sampling) takes roughly 5-10 ms.
- They interleave under the GIL. Expect a few ms of ITL jitter; it is measured in the prototype.
- If it hurts, torch.compile'd regions already cut launch counts. Move the lane launch to a C++ thread only if the measurements say so.

## Gates

1. **Window (row 4).**
   - `mux_bench.py` on GPU0: single-GPU stream-priority overlap of real Marlin W4 GEMMs at 64 layers and TP2 per-rank shapes.
   - `mux_ar_bench.py` on both GPUs: second-communicator cost and safety, three concurrent arrangements, checksums, watchdog.
   - Build the prototype only if all of these hold:
     - decode slowdown is at most 2x;
     - prefill loss during overlap is at most 30%;
     - no DEADLOCK;
     - every checksum is ok.
2. **Prototype** (next, ~3-5 days): second runner, connector-style handoff, lane thread, group B at init.
   - It must pass all of these:
     - ITL under a 16K cold prefill (`itl_under_prefill.py`): during-prefill chunk rate at least 3x today, and max gap at most 0.5 s;
     - 12/16/20-body estate pass wall no worse than +5%;
     - evalkit at least 59/60;
     - greedy determinism vs reference in the same class as the stack;
     - continuation soak 24/24;
     - Xid 0.
3. **Production trial**, with the same guards as the S4 trial.

## Landed (default OFF)

- `vllm/k6_mux.py`: lane state, per-lane group resolution, Marlin workspace, priorities.
- Hooks:
  - `vllm/forward_context.py`
  - `vllm/distributed/parallel_state.py` (`get_tp_group`, `_resolve_group` used by the compiled collective ops)
  - `vllm/model_executor/kernels/linear/mixed_precision/marlin.py`
- `tests/k6/test_k6_mux.py`: 9 CPU tests covering default-off semantics, private forward context across threads, per-lane groups, private workspaces and the priority order.
