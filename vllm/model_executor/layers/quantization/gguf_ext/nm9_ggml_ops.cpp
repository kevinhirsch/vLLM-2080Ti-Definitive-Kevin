// NM-9: GGUF (ggml) CUDA kernels as a standalone torch op library for the SM75 fork.
// The prod build ships only *_stable_libtorch.abi3.so (no ggml ops); these kernels use the
// regular torch C++ API, so they are JIT-built by vllm/model_executor/layers/quantization/
// gguf_ops.py (release.py --jit-ext) and registered as torch.ops._nm9_ggml.*.
#include <torch/all.h>
#include <torch/library.h>

torch::Tensor ggml_dequantize(torch::Tensor W, int64_t type, int64_t m, int64_t n,
                              std::optional<at::ScalarType> const& dtype);
torch::Tensor ggml_mul_mat_vec_a8(torch::Tensor W, torch::Tensor X, int64_t type,
                                  int64_t row);
torch::Tensor ggml_mul_mat_a8(torch::Tensor W, torch::Tensor X, int64_t type,
                              int64_t row);
torch::Tensor ggml_moe_a8(torch::Tensor X, torch::Tensor W,
                          torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded, int64_t type,
                          int64_t row, int64_t top_k, int64_t tokens);
torch::Tensor ggml_moe_a8_vec(torch::Tensor X, torch::Tensor W, torch::Tensor topk_ids,
                              int64_t top_k, int64_t type, int64_t row,
                              int64_t tokens);
int64_t ggml_moe_get_block_size(int64_t type);

TORCH_LIBRARY(_nm9_ggml, ops) {
  ops.def(
      "ggml_dequantize(Tensor W, int type, SymInt m, SymInt n, ScalarType? "
      "dtype) -> Tensor");
  ops.def(
      "ggml_mul_mat_vec_a8(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
  ops.def("ggml_mul_mat_a8(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
  ops.def(
      "ggml_moe_a8(Tensor X, Tensor W, Tensor sorted_token_ids, Tensor "
      "expert_ids, Tensor num_tokens_post_padded, int type, SymInt row, SymInt "
      "top_k, SymInt tokens) -> Tensor");
  ops.def(
      "ggml_moe_a8_vec(Tensor X, Tensor W, Tensor topk_ids, int top_k, int "
      "type, SymInt row, SymInt tokens) -> Tensor");
  ops.def("ggml_moe_get_block_size(int type) -> int", &ggml_moe_get_block_size);
}

TORCH_LIBRARY_IMPL(_nm9_ggml, CUDA, ops) {
  ops.impl("ggml_dequantize", &ggml_dequantize);
  ops.impl("ggml_mul_mat_vec_a8", &ggml_mul_mat_vec_a8);
  ops.impl("ggml_mul_mat_a8", &ggml_mul_mat_a8);
  ops.impl("ggml_moe_a8", &ggml_moe_a8);
  ops.impl("ggml_moe_a8_vec", &ggml_moe_a8_vec);
}
