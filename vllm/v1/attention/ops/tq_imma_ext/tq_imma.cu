// TurboQuant k3v4_nc (head_dim 256) decode attention on Turing INT8 tensor cores (IMMA, mma.m8n8k16). Lane K2, 2026-10-03.
//
// Math (exact apart from the three int8 roundings named below):
//   K: code c in 0..7 -> centroid cent[c].  With norm correction k_hat = |k| * cent[codes] / ||cent[codes]||, so only the
//      RATIOS of the 8 centroids matter.  We use an int8 LUT whose ratios match the Lloyd-Max centroids to <=0.26%
//      (rounding #1, host-chosen).  Scores: s = (sigma_q * |k| * attn_scale / ||lut[codes]||) * sum_d q8[d]*lut[code_d],
//      the sum runs on IMMA s8*s8 -> s32.  ||lut[codes]||^2 is an exact integer (dp4a).
//   Q: int8 per row (sigma_q = absmax/127, rounding #2).
//   V: v = code*vs + vz with 4-bit codes, EXACT as u8.  O = sum_t p_t*vs_t*code_t + sum_t p_t*vz_t.  a_t = p_t*vs_t is
//      quantized to u8 per (row, 64-token chunk) (rounding #3), the code sum runs on IMMA u8*u8 -> s32, the zero term
//      sum_t p_t*vz_t and the softmax denominator stay fp32.
// One CTA = (sequence, kv head, KV split) and ALL its query rows (QL rows x GQA group <= 8*MT): every cached token is
// read and decoded exactly once per kv head.  Rows may have different lengths (MTP rows: causal over the cache).
// Output: Mid [R, Hq, NS, D+1] (normalized partial O, lse) then a split reduce -> Out fp16 [R, Hq, D], Lse fp32 [R, Hq].
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "tq_imma_kernels.cuh"


template <int MT>
static void launch1(dim3 grid, size_t smem, cudaStream_t st, const int8_t* q8, const float* qs, const uint8_t* kv,
                    const int* bt, const int* rl, float* mid, long sq8r, long sq8h, long scb, long scp, long sch,
                    long sbt, long smr, long smh, long sms, int QL, int G, int bs, int NS, float scale, float cscale,
                    int nc, int nbt, uint32_t lo, uint32_t hi) {
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(tq_imma_stage1<MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr = true;
  }
  tq_imma_stage1<MT><<<grid, NTHR, smem, st>>>(q8, qs, kv, bt, rl, mid, sq8r, sq8h, scb, scp, sch, sbt, smr, smh, sms,
                                                QL, G, bs, NS, scale, cscale, nc, nbt, lo, hi);
}

// q_rot [S*QL, Hq, D] fp32 (rotated); kv [nb, bs, Hk, slot] u8; bt [S, nbt] i32; row_lens [S*QL] i32 (per-row causal
// length); q8/qs/mid/out/lse caller-provided workspaces.
void tq_imma_decode(torch::Tensor q_rot, torch::Tensor kv, torch::Tensor bt, torch::Tensor row_lens, torch::Tensor q8,
                    torch::Tensor qs, torch::Tensor mid, torch::Tensor out, torch::Tensor lse, int64_t QL, int64_t NS,
                    double scale, double cscale, int64_t norm_corr, int64_t lut_lo, int64_t lut_hi) {
  const c10::cuda::CUDAGuard guard(q_rot.device());
  TORCH_CHECK(q_rot.size(2) == HD && q_rot.stride(2) == 1 && q_rot.stride(1) == HD, "q_rot layout");
  const int S = bt.size(0), Hk = kv.size(2), Hq = q_rot.size(1), R = q_rot.size(0);
  const int G = Hq / Hk;
  const int rows = (int)QL * G;
  TORCH_CHECK(rows <= 32, "QL*G must be <= 32");
  TORCH_CHECK(R == S * QL, "R == S*QL");
  TORCH_CHECK(kv.size(1) >= T, "block_size >= 64");
  TORCH_CHECK(mid.size(2) >= NS, "mid splits");
  auto st = at::cuda::getCurrentCUDAStream();
  tq_imma_qquant<<<R * Hq, 64, 0, st>>>(q_rot.data_ptr<float>(), q8.data_ptr<int8_t>(), qs.data_ptr<float>(), Hq,
                                         q_rot.stride(0), q_rot.stride(1));
  const int MT = (rows + 7) / 8;
  const size_t smem = tq_imma_smem_bytes(MT);
  dim3 grid(S, Hk, NS);
#define L1(mt)                                                                                                       \
  launch1<mt>(grid, smem, st, q8.data_ptr<int8_t>(), qs.data_ptr<float>(), kv.data_ptr<uint8_t>(), bt.data_ptr<int>(), \
              row_lens.data_ptr<int>(), mid.data_ptr<float>(), (long)Hq * HD, (long)HD, kv.stride(0), kv.stride(1),    \
              kv.stride(2), bt.stride(0), mid.stride(0), mid.stride(1), mid.stride(2), (int)QL, G, (int)kv.size(1),     \
              (int)NS, (float)scale, (float)cscale, (int)norm_corr, (int)bt.size(1), (uint32_t)lut_lo, (uint32_t)lut_hi)
  switch (MT) {
    case 1: L1(1); break;
    case 2: L1(2); break;
    case 3: L1(3); break;
    default: L1(4); break;
  }
#undef L1
  tq_imma_stage2<<<R * Hq, HD, NS * sizeof(float), st>>>(mid.data_ptr<float>(),
                                                         reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                                                         lse.data_ptr<float>(), Hq, (int)NS, mid.stride(0),
                                                         mid.stride(1), mid.stride(2), out.stride(0), out.stride(1),
                                                         lse.stride(0));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("decode", &tq_imma_decode, "tq imma decode"); }
