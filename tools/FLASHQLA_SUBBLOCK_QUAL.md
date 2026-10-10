# FlashQLA sub-block continuation fixture

This source-only candidate extends the existing module-level qualification.
It is based on upstream vLLM and weicj/vLLM-2080Ti-Definitive at
`b7ea65b3ee8a60bcb0b7f1a256c0915badc77697`. Existing vLLM Apache-2.0 and
FlashQLA MIT attribution remain intact. It has not executed on a GPU.

`flashqla_subblock_source_binding.json` pins the actual immutable release's
numerical source files, checkpoint metadata, pre-existing native GDN ELF,
quiet-window helper and measurement module. Description mode imports only
stdlib modules and reads these files. CPU contract tests likewise avoid
Torch, vLLM and Triton imports, network observations and service actions.

```bash
CUDA_VISIBLE_DEVICES='' python3 tools/test_flashqla_subblock_qual_cpu.py
CUDA_VISIBLE_DEVICES='' python3 tools/flashqla_subblock_qual.py
```

The optional numerical arm uses production **local TP2 geometry** on one
SM75 device: key/value heads8/24, dimensions128, mixed-QKV width5120, conv
kernel width4 with six FP16 history/speculation positions, FP32 VK recurrent
state, MTP3's four physical queries, and previous accepted counts1..4. It
does not instantiate TP collectives, the model, or production weights.

At lengths511/512/513,1023/1024/1025,3567/3568/3569 and7135/7136/7137,
independently generated threshold512/1024 plans stop at mandatory physical
block junctions and the hash16/Eagle-drop partial-tail checkpoint. These are
explicit fixture contracts, not assertions that the actual scheduler chose
these dispatches. Threshold0 is one dense native numerical reference call;
it is not a valid whole-model scheduling arm. The split arm calls actual
causal convolution, post-conv preparation and the actual packed-varlen GDN
ELF, then actual speculative conv/GDN continuation. Every output and state
element is finite and compared against independent CPU FP64 recurrence
with declared FP16/FP32 rounding and tolerances. Conv histories and unrelated
cache slots are compared byte exactly. Native supplied h0/final-state alias
is checked explicitly. No warning percentage exempts failed elements.

At the physical3568 checkpoint, the fixture also assigns two private fork
slots from one shared checkpoint through the real pre-copy kernel. It runs
one writer while preserving the held sibling/shared slot, then unequal
17/31-token packed continuations. It checks the full pool against the
existing scalar copy contract and all outputs/states against independent
oracles. This does not exercise allocator ownership/refcounts, an actual
APC lookup/hit, request preemption/refund or model logits. The receipt keeps
those qualifications false even when numerical checks pass.

The state orientation is **VK `[value,key]`**. The actual native kernel
indexes `col=value,row=key`, and the current speculative kernel multiplies
`state[v,k] * k[k]`. The older vendor `_reference` uses transposed state and
`outer(k,delta)`; it cannot be reused unchanged with non-symmetric initial
state. The new CPU discriminator and nonzero random h0 make this observable.
Normalization is `x/sqrt(sum(x*x)+1e-6)` with an additive epsilon, matching
the actual preparation and speculative kernels. `F.normalize`'s default
maximum-norm epsilon has different behavior near zero.

The frozen37c helper independently uses the correct VK recurrence and
additive epsilon. Its four GDN, four conv and eight pre-copy checks remain
the documented synthetic module qualification: they do not test FlashQLA
chunked prefill, whole-model state, allocator CoW/APC or logits. Its files
and prior receipts are unchanged by this candidate.

Only the root-owned controller may execute the optional GPU command in an
exclusive quiet window. The following templates have **not** been executed:

```bash
ACTUALROOT=/home/kevin/.local/share/vllm-releases/halo-prefill-global-quota-20261010-b7ea65b3ee-20261010T1943
"$ACTUALROOT/.venv/bin/python" tools/flashqla_subblock_qual.py --run-gpu --length 3568 --out "$UNIQUE_RESULT_PATH"
"$ACTUALROOT/.venv/bin/python" tools/flashqla_subblock_qual.py --run-gpu --out "$OTHER_UNIQUE_RESULT_PATH"
```

The controller must supply `WINDOW_ENGINE_PID` and
`WINDOW_ENGINE_IDENTITY_FILE`. The gate fixes the known SYSTEM engine unit,
direct engine8001/gateway8000, pins `ACTUALROOT/RELEASE.json`, checks that
all8001 listeners belong to the attested PID/start process tree, and requires
two spaced queue-zero samples plus gateway drain counters and a finite
fence. Actual checkpoint, FP16, MTP3 and FlashQLA backend must match before
Torch import. Actual TP2 and the known cache CLI controls must select FP16
conv and explicit FP32 SSM: checkpoint `mamba_ssm_dtype` metadata alone does
not determine vLLM's cache dtype. Known actual QKV mode `native`/`fp32` is read without saving
arbitrary environment or credentials. The ELF must already be mapped by
the attested live engine. It is imported directly and never source-built;
Triton may JIT during a separately authorized future numerical run.

Pre/post admission samples do not protect the interval. The existing
controller must pause direct consumers, renew its finite gateway fence,
retain deadman ownership, and allow enough bounded time for the CPU oracle
and first Triton compile. A missing gate, changed identity/helper/source,
nonfinite value, or failed element comparison fails the receipt. No service
pause/restart, engine generation, profile, model workload, throughput or
latency claim belongs to this tool. A single-length receipt explicitly
labels the boundary matrix incomplete.

Before any production promotion, remaining separate gates are actual
worker/scheduler dispatch evidence, both production TP ranks, shared-tail
allocator/APC/CoW/preemption replay, accepted-prefix/logit trajectories and
matched full-model output quality at every threshold/quota arm. The
observed threshold512 first-action latency is a measurement hypothesis;
this numerical candidate cannot turn it into a quality result or speedup.
