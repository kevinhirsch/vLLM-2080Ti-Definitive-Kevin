# Frontier Changelog — 2026-08-16

This is a terse index of the fork work done in the 2026-08-16 frontier session.
It records each branch, what it contains, and the honest validation status of
each item. Negative results are marked as negatives on purpose — a perf-neutral
or unconfirmed outcome is more useful recorded than dressed up.

All branches are local to this fork. Nothing here has been pushed to the
upstream remote (`origin` = `weicj/vLLM-2080Ti-Definitive`). The upstream
separation plan lives in [Upstream PR Plan](UPSTREAM-PR-PLAN.md).

## Branch map

| Branch | Head | Contents |
|---|---|---|
| `frontier-pastnative-20260816` | `721ad64` | Session branch: Xid31 guard → EXP-040 allreduce → EXP-017 rope, stacked |
| `fix-xid31-guard` | `22f4c2b` | Full 3-patch Xid31 instrumentation |
| `merged-v0115-regression` | `24d758d` | Banked upstream v0.1.15 merge |
| `feat-retention-interval` | `3277ffb` | Upstream #45845 retention-interval port (in worktree) |
| `exp040-allreduce-32mib` | `6a82ec3` | SM75 custom-allreduce cap raise |
| `exp017-rope-extend` | `721ad64` | Past-native RoPE context extension |

The session branch `frontier-pastnative-20260816` is the stacked chain
`eb34502` (Xid31 reader-guard) → `6a82ec3` (EXP-040 allreduce) → `721ad64`
(EXP-017 rope). The single-purpose branches above isolate each change for clean
upstream cherry-picking.

## EXP-017 — Past-native context (`exp017-rope-extend`, `721ad64`)

Serve Qwen3.8-27B past its native `262,144` window. Correct deep-needle
retrieval at `352,247` tokens (`90,103` past native) on 2×2080Ti-22GB.

- Patch: `vllm/model_executor/models/qwen3_next.py` — size the RoPE cos/sin
  cache from `VLLM_ROPE_MAX_POSITION` (env-or-default; inert when unset).
- Envs: `VLLM_ROPE_MAX_POSITION=524288`,
  `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`,
  `VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS=393216`.
- Flags: YaRN `factor=2.0` over `original_max_position_embeddings=262144`,
  `--max-model-len 393216`.

**Status: positive, partial.** Needle at `~88%` depth retrieved correctly and
coherently past native; short-context gate 5/5. Clean because Qwen3-Next is
hybrid — only 16/64 layers carry RoPE, the other 48 are position-agnostic GDN.
**Pending:** fresh-unique-prefix reconfirm, full evalkit past native, stability
soak. Full recipe in [Past-Native Context](past-native-context.md).

## EXP-040 — SM75 allreduce cap (`exp040-allreduce-32mib`, `6a82ec3`)

`"7.5"` was missing from `CUSTOM_ALL_REDUCE_MAX_SIZES`
(`vllm/distributed/device_communicators/all_reduce_utils.py`), so SM75 took the
`8MiB` constructor default. Allreduces `≥8MiB` — chunked prefill emits ones near
`~26MiB` — fell back to NCCL. Raised the constructor default `max_size` to
`32MiB` in `custom_all_reduce.py` so those stay on the NVLink custom kernel.

- Correctness: needle retrieval correct at 20k / 40k / 80k.
- Decode speed: unchanged.
- **Prefill speed: unchanged (honest negative).** 3-rep measurement showed no
  prefill speedup — allreduce is not the prefill bottleneck on this path. The
  change is kept for correctness/stability routing, not for a speed claim.

**Status: correct, perf-neutral.** Deployed as a stability soak. Any speed
benefit is unproven; do not represent it as a prefill optimization.

## Xid31 instrumentation (`fix-xid31-guard`, `22f4c2b`)

Instrumentation and guards around the TurboQuant continuation dequant path,
hunting an Xid31 (GPU memory-fault) crash. Four commits:

- `28458c6` — env-gated reader-side bounds guard
  (`VLLM_TURBOQUANT_CONTINUATION_BOUNDS_CHECK`) before continuation dequant.
- `db65d0a` — always-on `pages > width` early-out mirroring the `:998` sibling,
  plus reader-guard width fix.
- `27c064a` — write-side OOB assert in `BlockTable.append_row` (P1 review fix:
  correct + actually-live bound).
- `22f4c2b` — shrink `seq_len` with `cached_len` in the `pages > width`
  early-out (P2 review fix).

**Status: instrumentation, root cause not closed.** The crash is racy and
state-dependent; the crash floor descends with successive un-rebooted crash
cycles. Refuted hypotheses: A/B path divergence, torn-read windows. Surviving
suspects: write-side OOB block id, `pages > width`, GDN kernels, and non-ECC
bit-flip on this hardware. The guards make the failure observable and add a
structural early-out; they are not a confirmed fix.

## Retention-interval port (`feat-retention-interval`, `3277ffb`)

Ported upstream #45845 (plus #43447 scaffolding) for Mamba/GDN cache groups:
env `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`. Branch is checked out in the
worktree `/home/kevin/Desktop/.ftree-retention` and stacks on the Xid31 guard
chain (`3277ffb` parent is `22f4c2b`).

**Status: ported, parity-pending.** Brought over for Mamba/GDN cache-group
parity with upstream; not yet independently validated on this fork's routes.

## v0.1.15 merge (`merged-v0115-regression`, `24d758d`)

Banked merge of upstream `v0.1.15` (`origin/vllm-2080ti-deifinitive`) onto the
deploy line. Held on its own branch as a regression-watch checkpoint rather than
fast-forwarded onto the session chain.

**Status: banked, not promoted.** Kept isolated so the frontier experiments
above sit on a known pre-merge base.
