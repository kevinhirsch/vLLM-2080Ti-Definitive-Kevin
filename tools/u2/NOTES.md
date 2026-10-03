# Lane U2 WIP (2026-10-02) - checkpoint, nothing here is validated on GPU
Findings (code-read, no GPU used):
- Build already supports quantized heads: compressed_tensors.py get_quant_method handles ParallelLMHead (as linear, if not in `ignore`) and VocabParallelEmbedding via CompressedTensorsEmbeddingWNA16Int. config.json quantization_config.ignore contains "lm_head" (line ~339); embeddings are bf16 (2.37 GiB), lm_head bf16 (2.37 GiB), MTP block bf16 (0.79 GiB), vision bf16 (0.86 GiB). MTP shares target embed/lm_head (llm_base_proposer._maybe_share_*); drafter runs lm_head once per draft step (4 reads/step at MTP-3).
- Vision tower is auto-skipped with --language-model-only or limit-mm-per-prompt 0 for image AND video.
- Marlin (sm_75 capable) is the W4A16 kernel. ncu is not installed; nsys is (/usr/local/bin/nsys).
- Plan for int4 heads: env-gated weight-iterator transform (RTN + MSE clip, group 128, CT pack-quantized layout) cached under a *-u2cache dir, plus env removing "lm_head" from CT ignore. Fidelity gate needs hidden states: CPU layerwise reference harness (subagent) at ~/projects/lanes/u2-quant/ (ref_dump.py, ref.pt) - may be incomplete.
- Prefill GEMM plan: microbench Marlin vs cuBLAS fp16 (fp32-acc) vs cublasGemmEx COMPUTE_16F (fp16-acc) at M=3584 when the engine is idle (needs ~0.5 GB free VRAM); then dequant->fp16 scratch + fp16-acc GEMM for M>=512.
