# Lane U2 / U2b notes (2026-10-02) -- speed + memory unlocks, all env-gated, default OFF

## What is in the tree
| file | purpose |
|---|---|
| `vllm/model_executor/layers/quantization/u2_headquant.py` | quantizer (RTN + per-group MSE clip, g128, asym for head/MTP, sym for embeddings), CT pack-quantized layout, cache, loader wrapper |
| hooks | `compressed_tensors.py` (ignore filter + Linear-scheme fallback for ParallelLMHead / VocabParallelEmbedding), `default_loader.py` (wrap weight iterator), `qwen3_5.py` + `qwen3_5_mtp.py` (pass quant_config to embed_tokens only when VLLM_U2_INT4_EMBED=1) |
| env | `VLLM_U2_INT4_HEAD=1`, `VLLM_U2_INT4_MTP=1`, `VLLM_U2_INT4_EMBED=1`, `VLLM_U2_CACHE_DIR`, `VLLM_U2_QUANT_DEVICE` |
| `prebuild_headquant.py` | builds `<model>-u2cache/` (done on this box: head, 8 MTP linears, embeddings); original weights never touched |
| `serve-hauhaucs-v02-u2b-{head,headmtp,all}.sh` | wrappers that set the envs and exec the LIVE v02 serve script unmodified (`all` also adds --language-model-only) |
| `serve-hauhaucs-v02-u2a-lmo.sh` | `--language-model-only` alone (vision tower off) |
| `test_headquant.py` | CPU unit check vs the compressed-tensors library (pack bit-identical to CT pack_to_int32, CT decompress round trip, shipped-layer reader equality, sym/embedding kernel arithmetic, env gating) |
| `gpu_layer_check.py` | tiny GPU check (<=0.5 GiB, refuses if <1.2 GiB free): REAL ParallelLMHead / VocabParallelEmbedding with real weight_loader under emulated TP=2, real Marlin repack + gemm |
| `ref_dump.py`, `fidelity.py`, `fidelity_embed.py` | CPU fp32 layerwise reference forward over recorded estate prompts; top-1/KL/dCE + MTP acceptance gate |
| `prefill_gemm_bench.py` | Lane S3 microbench (engine STOPPED): Marlin vs cuBLAS fp32-acc vs cublasGemmEx COMPUTE_16F vs W4A8-int8 Marlin, model shapes, M=512..3584 |

## Findings
* lm_head (2.37 GiB), embeddings (2.37), MTP block (0.79) are bf16 in the checkpoint; MTP shares the target lm_head (4 reads/step at MTP-3).
* `ParallelLMHead` never matched the checkpoint's "Linear" target by class name, so dropping `lm_head` from `ignore` alone would NOT have quantized it; hence the get_scheme_dict fallback.
* The embed path (CompressedTensorsEmbeddingWNA16Int) is symmetric-only (no zero points) -> embeddings are quantized symmetric.
* Marlin on sm_75 already accumulates in fp16 (`use_fp16_accum`, marlin_template.h:269) for fp16-activation group-quantized int4. The "fp16-accumulate prefill GEMM" unlock is therefore mostly already taken; what remains is Marlin's efficiency vs the 107.5 TF/s fp16-acc roof at M=3584 (bench arms).
* W4A8-INT8 Marlin (`VLLM_MARLIN_INPUT_DTYPE=int8`, s8 kernels ARE compiled for sm_75) asserts symmetric uint4b8 weights; the shipped checkpoint is asymmetric, so it is not usable without a symmetric re-quantization of every layer, and the env is global (decode too).
