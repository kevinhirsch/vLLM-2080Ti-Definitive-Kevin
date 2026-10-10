# E06: cross-request TurboQuant prefix launch prototype

This default-off source prototype is based on commit
`7004df6ae82c020c8a8e01c51f2f18f8889ab796` of vLLM 2080 Ti Definitive
Edition (`github.com/weicj`), with upstream vLLM and Marlin attribution and
Apache-2.0 notices preserved. It extends the existing lane S2 grouped TQ path;
it does not claim to invent speculative-row/GQA grouping or FP16 accumulation.

`VLLM_TQ_BATCH_SPEC_PREFIX=1` enables grouping of compatible independent
requests inside the raw-current speculative decode path, only when the
existing CUDA TQ backend is enabled and eligible. It requires already
available CPU query-start metadata, equal query lengths in 1..4, at least two
requests, the same layer/cache/backend and NS in 1..128. Singleton, ineligible,
incompatible and oversized query lengths retain the old path. It adds no
device scalar read or device-to-CPU synchronization.

Each request first executes its **original** FP32 rotation expression and GEMM
dimensions: `(query.float() @ PiT).contiguous()`. Only the rotated rows, block
tables and lengths are concatenated before the existing extension's two
kernel launches. Prefix outputs/LSE are mapped back by request ownership,
empty-prefix page selection/masks are retained, and raw-current causal merges
stay request-local with the existing output alias. Cache contents, recurrent
states, scheduler and acceptance logic are not changed.

The new packed temporaries and graph topology can cost more than the launches
saved. No speedup, acceptance, full-model quality, runtime activation or GPU
numerical result is claimed. Trailing graph-padding rows remain unclaimed as
in the original backend; the caller must retain its original valid-row mask.
The prototype uses per-call allocations and CUDA Graph capture allocations,
not a new global workspace or shared mutable batch state.

CPU validation (no torch/vLLM import, CUDA initialization or build):

```bash
CUDA_VISIBLE_DEVICES='' python3 tools/test_tq_prefix_batch_cpu.py
python3 -m py_compile vllm/v1/attention/ops/tq_prefix_batch_plan.py vllm/v1/attention/ops/tq_gqa_cuda.py vllm/v1/attention/backends/turboquant_attn.py tools/compare_tq_prefix_batch_sm75.py
git diff --check
```

The tests execute AST-extracted wrapper/backend methods with tensor metadata
spies. They cover equal/mixed lengths, C1/C4, zero-length requests, malformed
CPU starts, zero prefixes, padded block-table strides, row ownership,
unchanged request-local merge alias, fallback, and unchanged original wrapper
and compressed-cache method. This is a control-flow/metadata proof only.

`tools/compare_tq_prefix_batch_sm75.py` defaults to a non-GPU description and
imports torch only after explicit `--run-gpu`, source HEAD and existing-ELF SHA
qualification. It never JITs/builds. The following template was **not run**:

```bash
python3 tools/compare_tq_prefix_batch_sm75.py --run-gpu --graph \
  --expected-source-head <THIS_PROTOTYPE_COMMIT> \
  --extension /home/kevin/Desktop/frontier-lab/reports/2026-10-10/static-cuda-audit/tq/tq_gqa_sm75.so \
  --expected-extension-sha256 29bc4aebae68697b7af9e59e7124fef54307940c425e1a5cdf9fd36eeedc1cd2 \
  --prefix-lengths 0,1,17,257 --ql 4 --hq 16 --hk 2 --num-splits 128
```

Root owns its later exclusive GPU commissioning. The fixture binds committed
source bytes and the unchanged extension ELF, generates disjoint synthetic
request pages, and compares prefix output/LSE bitwise to per-request calls.
Optional graph mode captures zero prefix metadata, then updates the same
input tensors and replays actual lengths/pages. Limits are eight requests,
four query rows, maximum prefix 32769, NS<=128 and 128 MiB synthetic KV cache.
It checks SM75 and does not run a model or touch services. It does not prove
production graph padding, mixed-prefill routing, actual MTP acceptance,
GDN/conv rollback, APC/CoW, logits or end-to-end quality.

A later serving comparison must use real MTP3 accepted-token counts (including
bonus), C1/C4, cold/warm cache identity, actual graph buckets and prefix/split
configurations. Report draft+verifier+sampling critical-path cost per actually
accepted token plus ITL/throughput. Count packing/masking/new allocations as
part of the candidate; do not sum overlapping GPU operations as wall time.

Research source/binary provenance and the other experiment shortlist:
`/home/kevin/Desktop/frontier-lab/reports/2026-10-10/source-audit/marlin-verifier-research/`.
