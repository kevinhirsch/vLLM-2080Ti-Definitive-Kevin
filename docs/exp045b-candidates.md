# EXP-045b — Static hunt: the max-model-len-gated structure behind the residual Xid31 crash

**Goal:** name the structure whose size or stride derives from `max_model_len` and
that overflows / goes out-of-bounds when `max_model_len ≥ ~half-a-million`, producing
the residual `Xid31 FAULT_PDE` crash with MTP on. Staged code only; no engine runs.

## The fingerprint (from the EXP-045 constraint matrix)

| max-model-len | pool (tok) | result |
|---|---|---|
| 460800 (`450·1024`) | 615–741K | 380K / 400K / **449K / 450K** requests **ALL CLEAN** |
| 507904 (`496·1024`) | 514K | **449K crashes** (67s), 462K (72s), 483K (36s) |
| 524288 (`512·1024 = 2^19`) | 622–758K | 375K–483K crash (36–478s); **≤352K clean** |
| 524288, **MTP-OFF** | — | 465K / 491K **clean** |

Five facts constrain the culprit:

1. **Scales with `max_model_len`, NOT the pool.** A **449K** request is CLEAN at
   `max_model_len=460800` (large 615–741K pool) but CRASHES at `max_model_len=507904`
   (smaller 514K pool). Same request, larger pool → clean; smaller pool → crash. So the
   faulting structure is sized/strided by `max_model_len` (or `cdiv(max_model_len,
   block_size)`), **not** by the physical block count. **This exonerates every
   pool-indexed structure** (they scale *down* at 507904 yet it crashes).
2. **Threshold ∈ (460800, 507904].** *This interval is the fingerprint.* A candidate
   whose behaviour does not change across it is exonerated.
3. **Spec-config dependent.** MTP-OFF @524288 is clean at 465K/491K; MTP-ON crashes.
4. **Prefill path**, faults **async** (cudagraph-replay OR eager), at
   **allocator-layout-dependent depths** → an out-of-bounds *device* access into an
   unmapped page, not an OOM (3.2 GB free at death).
5. Request must be **large enough to reach** the faulting region (≤352K clean @524288).

Facts 1+2 mean: `offset = f(request_depth) · g(max_model_len)` crossing a hardware
boundary — and the only boundary that produces an unmapped-page fault at a *specific*
`max_model_len` is the **int32 2 GiB byte-offset wrap** (`2^31 = 2,147,483,648`): a
CUDA/Triton kernel that accumulates a linear byte address in `int32` wraps negative
once the address reaches 2 GiB, dereferencing garbage → `Xid31 FAULT_PDE`.

## The arithmetic signature

Config (from `deploy/bin/serve-tqk8v4-fg.sh`): `kv-cache-dtype=turboquant_k8v4`,
`mamba-cache-mode=align`, `max-num-batched-tokens=2560`, MTP (K=2 in the probes),
TP=2, `block_size` = default 16 (no `--block-size`), PIECEWISE cudagraphs.

`cdiv(max_model_len, 16)`: **460800 → 28800**, **507904 → 31744**, **524288 → 32768**.

**Per-token byte-stride band that puts the 2 GiB crossing inside (460800, 507904]:**

```
2^31 / 507904 = 2097152 / 496 = 4228.13 bytes/token   (crossing AT 507904)
2^31 / 460800 = 2097152 / 450 = 4660.34 bytes/token   (crossing AT 460800)
```

So **a per-token byte stride in `[4228.13, 4660.34]` crosses 2 GiB exactly inside the
interval** — that is the number to look for. Two corroborating anchors:

- **4096 bytes/token crosses 2 GiB at exactly `max_model_len = 2^31/4096 = 524288 =
  2^19`** — *precisely* the MTP-OFF-clean / MTP-ON-crash config, with the MTP-ON
  threshold sitting *just below* it. A ~4 KB/token structure whose effective index is
  nudged up by the extra MTP draft tokens (spec-dependency, fact 3) would have its 2 GiB
  crossing pushed *down* from `2^19` into `(460800, 507904]`. This single mechanism
  explains facts 1–5 at once.
- A stride ≥ ~4661 B/token would already cross at ≤460800 → would crash the CLEAN
  460800 config → exonerated; a stride ≤ ~4096 B/token crosses at ≥524288 → 507904
  would be clean → exonerated. **Only the `[4228, 4660]` band fits.**

`~4.1–4.7 KB/token` is the footprint of a **full-context K+V dequant / paged-attention
row** for the turboquant *full-attention* layers: `num_kv_heads · head_dim · 2(K,V) ·
2(fp16)` (e.g. `Hk·D ≈ 1024–1150` → 4096–4600 B). The continuation-prefill path
(`_continuation_prefill`, engaged at seq_len ≥ 20480) dequants the whole cached prefix
into `[1, Hk, cached_len, D]` fp16 and feeds it to a FlashInfer/Triton attention over
up to ~449K tokens — a per-token-strided kernel over a `max_model_len`-scaled extent.

## Ranked candidates

### Exonerated by arithmetic (with the numbers)

| # | structure | why it scales | verdict |
|---|---|---|---|
| E1 | **turboquant KV-cache physical index** `block_num·stride_kv_block` (`_tq_decode_stage1`, `triton_turboquant_decode.py:313`) | `block_num ≤ num_gpu_blocks ∝ pool` | Pool-gated → *worse* at 460800 (46 312 blocks) than 507904 (32 125). 460800 is CLEAN. **EXONERATED.** |
| E2 | **block_table** `[max_num_reqs, cdiv(mml,16)]` int32 (`block_table.py:70`) | width ∝ `max_model_len` | Worst byte offset `(8·31744−1)·4 = 1.02 MiB`; kernel index `req·stride+col`, `req ≤ 8`. 3 orders of magnitude below 2 GiB. **EXONERATED.** |
| E3 | **spec slot-mapping kernel** `eagle_step_slot_mapping_metadata_kernel` (`spec_decode/utils.py:73`) | reads block_table | `block_number = min(block_number, n_blocks_per_req−1)` (bounds-clamped), `req_idx ≤ 8`. int32-safe. **EXONERATED.** |
| E4 | **mamba "align" gather** `torch.gather(block_table, 1, (seq−1)//16 + [0..num_spec])` (`utils.py:897`) | width ∝ `max_model_len` | OOB only when `seq_len ≥ max_model_len − 16·num_spec` (within ~32 tok of the ceiling). Crashes seen at **449K ≪ 507873**. **EXONERATED** for the observed crashes (latent near-ceiling bug — see below). |
| E5 | **spec proposer buffers** `input_ids / positions / hidden_states / mrope_positions / inputs_embeds / _slot_mapping_buffer` (`llm_base_proposer.py:154–241`) | sized by `max_num_batched_tokens (2560)` / `max_positions` | **Invariant across the interval** — identical at 460800 and 507904. **EXONERATED by fingerprint.** |
| E6 | **rope `cos_sin_cache`** `[max_position_embeddings, rotary_dim]` (`rotary_embedding/base.py:86`) | rows = `max_pos` | `index_select` with **int64** positions (no int32 wrap); runs with spec OFF too (not fact-3). Draft rope-cache already eliminated (session 2). **EXONERATED.** |
| E7 | **ngram** `np.zeros((1024, max_model_len))` (`ngram_proposer.py:60`) | cols = `max_model_len` | CPU numpy → cannot raise a GPU MMU fault; inactive under MTP. **EXONERATED.** |
| E8 | **continuation dequant workspace** `[1, Hk, alloc_len, D]` (`turboquant_attn.py:2107`) | reserve ∝ `max_num_batched_tokens`; runtime `alloc_len ∝ cached_len` | Sized by request/batch, **not** `max_model_len`; "fixed & accounted" (PR #111). Request-gated ≠ interval-gated. **EXONERATED** (matches the task's note). |

### Surviving candidates (need runtime confirmation — ranked)

**C1 (top): an `int32` linear byte offset in the continuation-prefill / full-attention
kernel over the dequanted full-context K+V, per-token stride `Hk·D·2·2 ∈ [4228,4660]`
B, effective token count bumped by the MTP draft tokens.**
Fits *all five facts*: per-token-strided (fact 1 via the `cached_len ∝ request` index ×
a stride set by the model's KV width), 2 GiB crossing in-interval (fact 2 / the
`[4228,4660]` band), MTP nudges the crossing down from `2^19` (fact 3), prefill-path
async fault (fact 4), needs depth to reach 2 GiB (fact 5). The **exact** per-token
stride — hence whether the crossing lands in-interval — depends on the checkpoint's
`num_kv_heads·head_dim` and the FlashInfer ragged/paged offset dtype, which source
reading cannot resolve. **This is what the instrumentation resolves at runtime.**

**C2: a 2-D `[rows ∝ request_blocks, cols = cdiv(max_model_len,16)]` structure indexed
`row·width` in int32**, product ∝ `request · max_model_len`. The eliminated "3584×128
chunk-table" (session 2) was a *fixed-width* sibling; a `max_model_len`-scaled-width
variant survives fact 1. Lower-ranked: no such structure was found in the enumerated
fork code, but a FLA/GDN or FlashInfer-internal one cannot be excluded from source.

**C3: FlashInfer prefill internal workspace/index** for the continuation attention over
the up-to-449K-token paged context (int32 offsets inside the library, `q/kv_indptr`
scaled by `num_heads·head_dim` — see `_flashinfer_indptr`, `turboquant_attn.py:866`).

### Honest verdict

**NOT CONFIRMED-BY-ARITHMETIC.** The static hunt *exonerates every enumerable fork
structure* and *pins the arithmetic signature* (`[4228.13, 4660.34]` B/token; anchored
at `4096 B/tok → 2^19`), but the surviving culprit lives in a kernel whose per-token
stride is set by model dims + FlashInfer internals not visible in source. The
instrumentation is built to name it in one run.

*Note (latent, out of scope):* E4's mamba-align gather **is** a real OOB within
~`16·num_spec` tokens of `max_model_len`; harmless today because the gateway caps
requests at 440K, but worth a `torch.clamp` on the gather index.

## What the instrumentation logs (`vllm/v1/worker/xid31_trace.py`)

Armed by `VLLM_TQ_XID31_TRACE=1` (default off; negligible overhead when off — one bool
check per step). When on:

- **At init (after `lock_workspace`)** — every registered persistent tensor ≥
  `VLLM_TQ_XID31_TRACE_MIN_MB` (default 64) MiB: `name / shape / dtype / device /
  data-ptr range / worst-case linear byte offset`, flagging any within 75% of 2^31
  (`<== NEAR-2GiB-INT32-LIMIT`). Roots scanned: `kv_caches`, `attn_groups` (metadata
  builders — block-table / state-index tensors), `drafter`, `input_batch.block_table`.
  Also logs the `max_model_len / block_size / cdiv / num_speculative_tokens` and the
  `[4228,4660]`-byte band.
- **Every `VLLM_TQ_XID31_TRACE_EVERY_N` (default 64) steps** — projects the current
  high-water sequence position (`hw_block = hw_pos // block_size`) onto each
  `max_model_len`-scaled suspect: `proj_off = hw_block · row_stride · itemsize`, and
  log-asserts (never raises) it stays under both the buffer extent and 2^31, printing
  `OVER-2GiB` / `near-limit` when it does not.
- **On any exception (incl. the CUDA error after an async Xid31)** — dumps a
  `torch.cuda.memory._record_memory_history` snapshot to
  `VLLM_TQ_XID31_TRACE_SNAPSHOT.<tag>` so the faulting address maps back to the owning
  allocation (name + Python allocation stack).

## Window command — 507904 boot + ~460K probe, tracing on

Edit `deploy/bin/serve-tqk8v4-fg.sh` so the `--max-model-len` value is `507904`
(currently `256000`), then boot with tracing armed:

```bash
# boot (window 1) — arm tracing, dense cadence, snapshot to Desktop
VLLM_TQ_XID31_TRACE=1 \
VLLM_TQ_XID31_TRACE_EVERY_N=16 \
VLLM_TQ_XID31_TRACE_MIN_MB=32 \
VLLM_TQ_XID31_TRACE_SNAPSHOT=/home/kevin/Desktop/xid31_snap.pickle \
  bash /home/kevin/Desktop/vLLM-2080Ti-Definitive/deploy/bin/serve-tqk8v4-fg.sh \
  2>&1 | tee /home/kevin/Desktop/xid31_507904_boot.log

# probe (window 2) — ~460K-token single request through the front-door shim (:8000).
# Reuses the evalkit corpus; repeat/​concat to ~460K tokens (~1.8 MB of text).
python3 - <<'PY'
import requests, itertools
base = open("/home/kevin/Desktop/vLLM-2080Ti-Definitive/deploy/evalkit/corpus/corpus_128000tok.txt").read()
# ~128K tok per copy; 4 copies ≈ 460–500K tokens.
prompt = "".join(itertools.islice(itertools.cycle([base]), 4))
r = requests.post("http://127.0.0.1:8000/v1/chat/completions", json={
    "model": "qwen-local",
    "messages": [{"role": "user", "content": prompt + "\n\nSummarize in one sentence."}],
    "max_tokens": 32, "temperature": 0,
}, timeout=900)
print(r.status_code, r.text[:300])
PY
```

Expect in `xid31_507904_boot.log`: the `[XID31] config:` line, the `[XID31] buf …`
registry (watch for `NEAR-2GiB-INT32-LIMIT` on a ~4 KB/token or `cdiv(mml,16)`-wide
buffer), the periodic `proj_off=…GiB` lines climbing as the probe deepens, and — at the
crash — `[XID31] memory snapshot dumped -> /home/kevin/Desktop/xid31_snap.pickle.excepthook`.
Load that snapshot (`torch.cuda.memory._snapshot` viewer) to read the allocation that
owned the faulting address.
