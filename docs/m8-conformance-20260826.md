# M-8 route-claim conformance — 2026-08-26 (post v0.1.17-rebase cutover)

Base: frontier-pastnative-20260816 @ v0.1.17 rebase tip. Smoke config per arm:
Qwen3.8-27B-GPTQ-Int4, TP=2, max-model-len 8192, mnbt 3584, seqs 2, noMTP,
PIECEWISE graphs, deterministic probe (temp 0). Full transcript:
frontier-queue/results/m8_conformance.txt.

| Route | --kv-cache-dtype / config | Result | Pool @8K | Notes |
| --- | --- | --- | --- | --- |
| FP16 KV | (default) | **PASS** | 174,441 | clean probe |
| INT8 KV | `int8_per_token_head` | **PASS** | 246,442 | clean probe |
| FP8 weights | `--quantization fp8` | SKIP | — | no local FP8 checkpoint; re-run when one lands |
| Offload | — | SKIP | — | fork documents no offload serve route (upstream `--cpu-offload-gb` unvalidated here); no claim exists |

Prod (turboquant_k3v4_nc + MTP3) is continuously validated by the ship gates.
Re-run this tier after any kv/align/GDN patch wave.
