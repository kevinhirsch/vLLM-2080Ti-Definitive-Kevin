// Lane K7: W4A4 prefill GEMM on Turing INT4 tensor cores (mma.m8n8k32 s4, measured 2008 MAC/clk/SM = 3.92x fp16).
//
//   y[M,N] (fp16) = sa[M] * sb[N] * sum_k A_s4[M,k] * B_s4[N,k]
//
// A = activations, symmetric per-token int4 after an online block-Hadamard rotation (k7_act_quant_had below).
// B = weights, W.H^T re-quantized offline to symmetric per-output-channel int4 (python side, tools/k7/rotquant.py).
// GEMM: CUTLASS 2.x sm75 int4 tensor-op mainloop + EVT epilogue (per-row x per-col scale -> fp16), same skeleton
// as vLLM's c2x scaled_mm for sm75 int8 (csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c2x.cuh).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/arch/mma_sm75.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"

using namespace cute;

namespace k7 {

template <typename TileShape, typename WarpShape, int Stages, typename Swizzle>
struct W4A4Gemm {
  using ElementAB = cutlass::int4b_t;
  using ElementD = cutlass::half_t;
  using ElementAcc = int32_t;
  using OutputTileThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<TileShape, WarpShape, float, 4, 1>;

  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
  using ScaleA = cutlass::epilogue::threadblock::VisitorColBroadcast<OutputTileThreadMap, float,
                                                                    Stride<Int<1>, Int<0>, Int<0>>>;
  using ScaleB = cutlass::epilogue::threadblock::VisitorRowBroadcast<OutputTileThreadMap, float,
                                                                    Stride<Int<0>, Int<1>, Int<0>>>;
  using Compute0 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, float, float,
                                                                  cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<Compute0, ScaleB, Accum>;
  using Compute1 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementD, float,
                                                                  cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTCompute = cutlass::epilogue::threadblock::Sm80EVT<Compute1, ScaleA, EVT0>;
  using D = cutlass::epilogue::threadblock::VisitorAuxStore<OutputTileThreadMap, ElementD,
                                                            cutlass::FloatRoundStyle::round_to_nearest,
                                                            Stride<int64_t, Int<1>, Int<0>>>;
  using EVTD = cutlass::epilogue::threadblock::Sm80EVT<D, EVTCompute>;

  static constexpr int AlignmentAB = 32;  // 128 bits of int4
  static constexpr int AlignmentCD = 4;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementAB, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, AlignmentAB,
      ElementAB, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, AlignmentAB,
      float, cutlass::layout::RowMajor, AlignmentCD, ElementAcc, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75, TileShape, WarpShape, cutlass::gemm::GemmShape<8, 8, 32>, EVTD, Swizzle, Stages,
      cutlass::arch::OpMultiplyAddSaturate, 1>::GemmKernel;
  using Op = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static void run(torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b, torch::Tensor const& sa,
                  torch::Tensor const& sb) {
    int m = a.size(0), k = a.size(1) * 2, n = b.size(0);
    cutlass::gemm::GemmCoord problem{m, n, k};
    int64_t lda = k, ldb = k, ldc = out.stride(0);
    typename D::Arguments d_args{reinterpret_cast<ElementD*>(out.data_ptr()), {ldc, Int<1>{}, Int<0>{}}};
    typename ScaleA::Arguments sa_args{sa.data_ptr<float>()};
    typename ScaleB::Arguments sb_args{sb.data_ptr<float>()};
    typename EVT0::Arguments e0{sb_args, {}, {}};
    typename EVTCompute::Arguments ec{sa_args, e0, {}};
    typename EVTD::Arguments epi{ec, d_args};
    typename Op::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, problem, 1, epi,
                                a.data_ptr(), b.data_ptr(), nullptr, nullptr, 0, 0, 0, 0, lda, ldb, ldc, ldc};
    Op op;
    size_t ws = op.get_workspace_size(args);
    auto workspace = torch::empty({(int64_t)std::max<size_t>(ws, 1)}, a.options().dtype(torch::kUInt8));
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(op.can_implement(args) == cutlass::Status::kSuccess, "k7 w4a4: can_implement failed");
    auto st = op(args, workspace.data_ptr(), stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "k7 w4a4: run failed ", cutlassGetStatusString(st));
  }
};

using SwzId = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
using SwzSK = cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;
using G = cutlass::gemm::GemmShape<1, 1, 1>;
template <int a, int b, int c> using S = cutlass::gemm::GemmShape<a, b, c>;

// config table (tile M,N,K / warp M,N,K). Turing has no cp.async -> 2-stage mainloop.
using C0 = W4A4Gemm<S<128, 256, 128>, S<64, 64, 128>, 2, SwzId>;
using C1 = W4A4Gemm<S<256, 128, 128>, S<64, 64, 128>, 2, SwzId>;
using C2 = W4A4Gemm<S<128, 128, 128>, S<64, 64, 128>, 2, SwzId>;
using C3 = W4A4Gemm<S<128, 128, 256>, S<64, 64, 256>, 2, SwzId>;
using C4 = W4A4Gemm<S<64, 128, 128>, S<32, 64, 128>, 2, SwzId>;
using C5 = W4A4Gemm<S<128, 256, 128>, S<64, 64, 128>, 2, SwzSK>;
using C6 = W4A4Gemm<S<128, 128, 128>, S<64, 64, 128>, 2, SwzSK>;

}  // namespace k7

torch::Tensor w4a4_gemm(torch::Tensor a, torch::Tensor b, torch::Tensor sa, torch::Tensor sb, int64_t cfg) {
  TORCH_CHECK(a.dtype() == torch::kInt8 && b.dtype() == torch::kInt8, "packed int4 as int8 bytes");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && a.size(1) == b.size(1));
  TORCH_CHECK(sa.dtype() == torch::kFloat32 && sb.dtype() == torch::kFloat32);
  const at::cuda::OptionalCUDAGuard guard(a.device());
  auto out = torch::empty({a.size(0), b.size(0)}, a.options().dtype(torch::kFloat16));
  switch (cfg) {
    case 0: k7::C0::run(out, a, b, sa, sb); break;
    case 1: k7::C1::run(out, a, b, sa, sb); break;
    case 2: k7::C2::run(out, a, b, sa, sb); break;
    case 3: k7::C3::run(out, a, b, sa, sb); break;
    case 4: k7::C4::run(out, a, b, sa, sb); break;
    case 5: k7::C5::run(out, a, b, sa, sb); break;
    case 6: k7::C6::run(out, a, b, sa, sb); break;
    default: TORCH_CHECK(false, "bad cfg");
  }
  return out;
}

// ---------------------------------------------------------------------------------------------------------------
// Activation path: x fp16 [M,K] -> block-Hadamard (size HB, orthonormal, along K) -> per-token symmetric int4.
// One CTA per row; row staged in smem as fp32 (K<=12288 -> 48KB).  Output: packed int8 [M,K/2] (low nibble =
// even k, as cutlass int4b_t), scale fp32 [M] (absmax/7).  HB=1 disables the rotation.
// ---------------------------------------------------------------------------------------------------------------
template <int THREADS>
__global__ void act_quant_had_kernel(const __half* __restrict__ x, int8_t* __restrict__ q, float* __restrict__ s,
                                     int K, int HB, float qmax) {
  extern __shared__ float row[];
  __shared__ float red[THREADS / 32];
  const int r = blockIdx.x;
  const __half* xr = x + (size_t)r * K;
  for (int i = threadIdx.x * 2; i < K; i += THREADS * 2) {
    float2 v = __half22float2(*reinterpret_cast<const __half2*>(xr + i));
    row[i] = v.x; row[i + 1] = v.y;
  }
  __syncthreads();
  // in-place FWHT on every HB-block: log2(HB) butterfly passes, K/2 butterflies per pass
  for (int h = 1; h < HB; h <<= 1) {
    for (int t = threadIdx.x; t < K / 2; t += THREADS) {
      int blk = t / h, off = t % h;
      int i = blk * 2 * h + off, j = i + h;
      float u = row[i], v = row[j];
      row[i] = u + v; row[j] = u - v;
    }
    __syncthreads();
  }
  const float norm = rsqrtf((float)HB);
  float amax = 0.f;
  for (int i = threadIdx.x; i < K; i += THREADS) amax = fmaxf(amax, fabsf(row[i]));
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < THREADS / 32 ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  amax = red[0] * norm;
  const float scale = amax > 0.f ? amax / qmax : 1.f;
  const float inv = norm / scale;
  if (threadIdx.x == 0) s[r] = scale;
  int8_t* qr = q + (size_t)r * (K / 2);
  for (int i = threadIdx.x * 2; i < K; i += THREADS * 2) {
    int a0 = __float2int_rn(row[i] * inv), a1 = __float2int_rn(row[i + 1] * inv);
    a0 = max(-8, min(7, a0)); a1 = max(-8, min(7, a1));
    qr[i >> 1] = (int8_t)((a0 & 15) | (a1 << 4));
  }
}

std::vector<torch::Tensor> act_quant_had(torch::Tensor x, int64_t hb, double qmax) {
  TORCH_CHECK(x.dtype() == torch::kFloat16 && x.is_contiguous() && x.dim() == 2);
  int M = x.size(0), K = x.size(1);
  TORCH_CHECK(K % hb == 0 && (hb & (hb - 1)) == 0 && K % 64 == 0 && K <= 12288);
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto q = torch::empty({M, K / 2}, x.options().dtype(torch::kInt8));
  auto s = torch::empty({M}, x.options().dtype(torch::kFloat32));
  constexpr int TH = 512;
  size_t smem = (size_t)K * 4;
  auto kern = act_quant_had_kernel<TH>;
  if (smem > 48 * 1024) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  kern<<<M, TH, smem, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const __half*>(x.data_ptr()),
                                                          q.data_ptr<int8_t>(), s.data_ptr<float>(), K, (int)hb,
                                                          (float)qmax);
  return {q, s};
}

// v2: HB=128 warp-register FWHT. Lane l holds elements 4l..4l+3 of a 128-block (one 8-byte load); stages h=1,2 in
// registers, h=4..64 via shfl_xor(1..16).  8 warps per row, each warp owns blocks w, w+8, ...  (<= NBMAX per warp).
template <int NBMAX>
__global__ void __launch_bounds__(256) act_quant_h128_kernel(const __half* __restrict__ x, int8_t* __restrict__ q,
                                                             float* __restrict__ s, int K, float qmax) {
  __shared__ float red[8];
  const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nb = K >> 7;
  const __half* xr = x + (size_t)r * K;
  float v[NBMAX][4];
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < NBMAX; j++) {
    int b = warp + j * 8;
    if (b < nb) {
      uint2 raw = *reinterpret_cast<const uint2*>(xr + b * 128 + lane * 4);
      float2 p0 = __half22float2(*reinterpret_cast<__half2*>(&raw.x)), p1 = __half22float2(*reinterpret_cast<__half2*>(&raw.y));
      float a0 = p0.x + p0.y, a1 = p0.x - p0.y, a2 = p1.x + p1.y, a3 = p1.x - p1.y;   // h=1
      float e[4] = {a0 + a2, a1 + a3, a0 - a2, a1 - a3};                                // h=2
#pragma unroll
      for (int m = 1; m < 32; m <<= 1) {                                                // h=4..64
        bool up = lane & m;
#pragma unroll
        for (int t = 0; t < 4; t++) {
          float o = __shfl_xor_sync(0xffffffff, e[t], m);
          e[t] = up ? (o - e[t]) : (e[t] + o);
        }
      }
#pragma unroll
      for (int t = 0; t < 4; t++) { v[j][t] = e[t]; amax = fmaxf(amax, fabsf(e[t])); }
    }
  }
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
  if (lane == 0) red[warp] = amax;
  __syncthreads();
  amax = red[0];
#pragma unroll
  for (int w = 1; w < 8; w++) amax = fmaxf(amax, red[w]);
  const float norm = 0.08838834764831845f;  // 1/sqrt(128)
  amax *= norm;
  const float scale = amax > 0.f ? amax / qmax : 1.f;
  const float inv = norm / scale;
  if (threadIdx.x == 0) s[r] = scale;
  uint16_t* qr = reinterpret_cast<uint16_t*>(q + (size_t)r * (K / 2));
#pragma unroll
  for (int j = 0; j < NBMAX; j++) {
    int b = warp + j * 8;
    if (b < nb) {
      uint32_t pk = 0;
#pragma unroll
      for (int t = 0; t < 4; t++) {
        int c = max(-8, min(7, __float2int_rn(v[j][t] * inv)));
        pk |= (uint32_t)(c & 15) << (4 * t);
      }
      qr[b * 32 + lane] = (uint16_t)pk;
    }
  }
}

std::vector<torch::Tensor> act_quant_h128(torch::Tensor x, double qmax) {
  TORCH_CHECK(x.dtype() == torch::kFloat16 && x.is_contiguous() && x.dim() == 2);
  int M = x.size(0), K = x.size(1);
  TORCH_CHECK(K % 128 == 0 && K / 128 <= 8 * 12, "K must be a multiple of 128 and <= 12288");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto q = torch::empty({M, K / 2}, x.options().dtype(torch::kInt8));
  auto s = torch::empty({M}, x.options().dtype(torch::kFloat32));
  auto st = at::cuda::getCurrentCUDAStream();
  int nbw = (K / 128 + 7) / 8;
  auto xp = reinterpret_cast<const __half*>(x.data_ptr());
  if (nbw <= 4) act_quant_h128_kernel<4><<<M, 256, 0, st>>>(xp, q.data_ptr<int8_t>(), s.data_ptr<float>(), K, (float)qmax);
  else if (nbw <= 6) act_quant_h128_kernel<6><<<M, 256, 0, st>>>(xp, q.data_ptr<int8_t>(), s.data_ptr<float>(), K, (float)qmax);
  else if (nbw <= 9) act_quant_h128_kernel<9><<<M, 256, 0, st>>>(xp, q.data_ptr<int8_t>(), s.data_ptr<float>(), K, (float)qmax);
  else act_quant_h128_kernel<12><<<M, 256, 0, st>>>(xp, q.data_ptr<int8_t>(), s.data_ptr<float>(), K, (float)qmax);
  return {q, s};
}

// =================================================================================================================
// k7 w4a4g: GROUP-SCALED activations, hand-written sm_75 kernel (no CUTLASS).
//   y[m,n] = smax[m]/2^SB * sw[n] * sum_g s_int[g,m] * sum_{k in g} a[m,k] * w[n,k]
// a = int4 codes per (row, 128-group) after Hadamard-128; s_int = round(s_g/s_max * 2^SB) (int, <= 2^SB).
// The per-group scale is folded into the int32 accumulator with ONE IMAD per accumulator per group (no float epilogue
// per group): |sum_k a*w| <= 128*64 = 8192 per group, so G*8192*2^SB < 2^31 bounds the accumulator (SB=11: K <= 8704*... G<=127).
// Tile 128x128x128 (one group per K-step), 8 warps (2 x 4), warp tile 64x32, register-staged double buffer (no cp.async on
// Turing), XOR-swizzled smem (16B chunk ^= (row>>1)&3) -> conflict-free ldmatrix.  mma.m8n8k32.row.col.s32.s4.s4.s32.
// =================================================================================================================
__device__ __forceinline__ int swz(int row, int chunk) { return row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4); }

__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

__global__ void __launch_bounds__(256, 1) w4a4g_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                                      const int* __restrict__ sint, const float* __restrict__ smax,
                                                      const float* __restrict__ sw, __half* __restrict__ C,
                                                      int M, int N, int K, float inv2sb) {
  __shared__ __align__(128) uint8_t sA[2][128 * 64];
  __shared__ __align__(128) uint8_t sB[2][128 * 64];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 2, wn = warp & 3;  // 2 x 4 warps
  const int m0 = blockIdx.y * 128, n0 = blockIdx.x * 128;
  const int G = K >> 7;
  const size_t ldk = K >> 1;  // bytes per row
  // global -> smem mapping: 512 x 16B per tile, thread handles idx = tid and tid+256: row = idx>>2, chunk = idx&3
  int ra0 = min(m0 + (tid >> 2), M - 1), ra1 = min(m0 + 64 + (tid >> 2), M - 1);
  const int ch = tid & 3;
  const uint4* gA0 = reinterpret_cast<const uint4*>(A + ra0 * ldk) + ch;
  const uint4* gA1 = reinterpret_cast<const uint4*>(A + ra1 * ldk) + ch;
  const uint4* gB0 = reinterpret_cast<const uint4*>(B + (size_t)(n0 + (tid >> 2)) * ldk) + ch;
  const uint4* gB1 = reinterpret_cast<const uint4*>(B + (size_t)(n0 + 64 + (tid >> 2)) * ldk) + ch;
  const int so0 = swz(tid >> 2, ch), so1 = swz(64 + (tid >> 2), ch);
  // this thread's 8 accumulator rows (one per m8 tile)
  int rows[8];
#pragma unroll
  for (int mt = 0; mt < 8; mt++) rows[mt] = min(m0 + wm * 64 + mt * 8 + (lane >> 2), M - 1);

  int acc[8][4][2], tot[8][4][2];
#pragma unroll
  for (int i = 0; i < 8; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) { tot[i][j][0] = tot[i][j][1] = 0; }

  uint4 pa0 = gA0[0], pa1 = gA1[0], pb0 = gB0[0], pb1 = gB1[0];
  *reinterpret_cast<uint4*>(&sA[0][so0]) = pa0; *reinterpret_cast<uint4*>(&sA[0][so1]) = pa1;
  *reinterpret_cast<uint4*>(&sB[0][so0]) = pb0; *reinterpret_cast<uint4*>(&sB[0][so1]) = pb1;
  __syncthreads();

  const int lr = lane & 7, lm = lane >> 3;
  for (int g = 0; g < G; g++) {
    const int buf = g & 1;
    if (g + 1 < G) {  // prefetch next group's tiles into registers (4 x 16B chunks per row per group = 64B)
      const int off = (g + 1) * 4;
      pa0 = gA0[off]; pa1 = gA1[off]; pb0 = gB0[off]; pb1 = gB1[off];
    }
    int sc[8];
#pragma unroll
    for (int mt = 0; mt < 8; mt++) sc[mt] = __ldg(sint + (size_t)g * M + rows[mt]);
    uint32_t af[2][8], bf[2][4];
    ldsm_x4(af[0][0], af[0][1], af[0][2], af[0][3], &sA[buf][swz(wm * 64 + (0 + lm) * 8 + lr, 0)]);
    ldsm_x4(af[0][4], af[0][5], af[0][6], af[0][7], &sA[buf][swz(wm * 64 + (4 + lm) * 8 + lr, 0)]);
    ldsm_x4(bf[0][0], bf[0][1], bf[0][2], bf[0][3], &sB[buf][swz(wn * 32 + lm * 8 + lr, 0)]);
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
      const int cur = ks & 1;
      if (ks < 3) {  // software-pipelined fragment load for the next k32 step
        ldsm_x4(af[cur ^ 1][0], af[cur ^ 1][1], af[cur ^ 1][2], af[cur ^ 1][3], &sA[buf][swz(wm * 64 + (0 + lm) * 8 + lr, ks + 1)]);
        ldsm_x4(af[cur ^ 1][4], af[cur ^ 1][5], af[cur ^ 1][6], af[cur ^ 1][7], &sA[buf][swz(wm * 64 + (4 + lm) * 8 + lr, ks + 1)]);
        ldsm_x4(bf[cur ^ 1][0], bf[cur ^ 1][1], bf[cur ^ 1][2], bf[cur ^ 1][3], &sB[buf][swz(wn * 32 + lm * 8 + lr, ks + 1)]);
      }
#pragma unroll
      for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
          if (ks == 0)
            asm("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0,%1}, {%2}, {%3}, {%4,%4};\n"
                : "=r"(acc[i][j][0]), "=r"(acc[i][j][1]) : "r"(af[cur][i]), "r"(bf[cur][j]), "r"(0));
          else
            asm("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
                : "+r"(acc[i][j][0]), "+r"(acc[i][j][1]) : "r"(af[cur][i]), "r"(bf[cur][j]));
        }
    }
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
      for (int j = 0; j < 4; j++) { tot[i][j][0] += acc[i][j][0] * sc[i]; tot[i][j][1] += acc[i][j][1] * sc[i]; }
    if (g + 1 < G) {
      *reinterpret_cast<uint4*>(&sA[buf ^ 1][so0]) = pa0; *reinterpret_cast<uint4*>(&sA[buf ^ 1][so1]) = pa1;
      *reinterpret_cast<uint4*>(&sB[buf ^ 1][so0]) = pb0; *reinterpret_cast<uint4*>(&sB[buf ^ 1][so1]) = pb1;
    }
    __syncthreads();
  }
  // epilogue: y = tot * smax[row]/2^SB * sw[col]
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int r = m0 + wm * 64 + i * 8 + (lane >> 2);
    if (r >= M) continue;
    const float sr = smax[r] * inv2sb;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int c = n0 + wn * 32 + j * 8 + (lane & 3) * 2;
      __half2 v = __floats2half2_rn((float)tot[i][j][0] * sr * sw[c], (float)tot[i][j][1] * sr * sw[c + 1]);
      *reinterpret_cast<__half2*>(C + (size_t)r * N + c) = v;
    }
  }
}

// v2: CTA 128(M) x 256(N), warp tile 64x64 processed as two 64x32 halves per group (int acc 64 regs + int total 128 regs),
// grouped tile rasterization (GM=8 M-tiles per band) for L2 reuse of the weight tiles.  Arithmetic intensity per stage
// 4.2M MAC / 24 KB = 175 MAC/B (v1 128x128: 128 MAC/B, which needed ~1.65 TB/s of L2 at the IMMA roof).
__global__ void __launch_bounds__(256, 1) w4a4g2_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                                       const int* __restrict__ sint, const float* __restrict__ smax,
                                                       const float* __restrict__ sw, __half* __restrict__ C,
                                                       int M, int N, int K, float inv2sb) {
  __shared__ __align__(128) uint8_t sA[2][128 * 64];
  __shared__ __align__(128) uint8_t sB[2][256 * 64];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 2, wn = warp & 3;
  // grouped rasterization
  const int nM = (M + 127) / 128, nN = N / 256, GMB = 8;
  const int id = blockIdx.x;
  const int band = id / (GMB * nN), first_m = band * GMB, bm = min(GMB, nM - first_m);
  const int mt_ = first_m + (id % (GMB * nN)) % bm, nt_ = (id % (GMB * nN)) / bm;
  const int m0 = mt_ * 128, n0 = nt_ * 256;
  const int G = K >> 7;
  const size_t ldk = K >> 1;
  const int ch = tid & 3, rr = tid >> 2;
  const uint4* gA0 = reinterpret_cast<const uint4*>(A + min(m0 + rr, M - 1) * ldk) + ch;
  const uint4* gA1 = reinterpret_cast<const uint4*>(A + min(m0 + 64 + rr, M - 1) * ldk) + ch;
  const uint4* gB0 = reinterpret_cast<const uint4*>(B + (size_t)(n0 + rr) * ldk) + ch;
  const size_t gBs = (size_t)64 * ldk / 16;  // 64 rows, in uint4
  const int soA0 = swz(rr, ch), soA1 = swz(64 + rr, ch);
  const int rbase = m0 + wm * 64 + (lane >> 2);
  int tot[8][8][2];
#pragma unroll
  for (int i = 0; i < 8; i++)
#pragma unroll
    for (int j = 0; j < 8; j++) { tot[i][j][0] = tot[i][j][1] = 0; }
  uint4 pa0 = gA0[0], pa1 = gA1[0], pb[4];
#pragma unroll
  for (int u = 0; u < 4; u++) pb[u] = gB0[u * gBs];
  *reinterpret_cast<uint4*>(&sA[0][soA0]) = pa0; *reinterpret_cast<uint4*>(&sA[0][soA1]) = pa1;
#pragma unroll
  for (int u = 0; u < 4; u++) *reinterpret_cast<uint4*>(&sB[0][swz(u * 64 + rr, ch)]) = pb[u];
  __syncthreads();
  const int lr = lane & 7, lm = lane >> 3;
  for (int g = 0; g < G; g++) {
    const int buf = g & 1;
    const bool more = g + 1 < G;
    const int off = (g + 1) * 4;
    if (more) { pa0 = gA0[off]; pa1 = gA1[off]; pb[0] = gB0[off]; pb[1] = gB0[gBs + off]; }
    int sc[8];
#pragma unroll
    for (int mt = 0; mt < 8; mt++) sc[mt] = __ldg(sint + (size_t)g * M + min(rbase + mt * 8, M - 1));
#pragma unroll
    for (int h = 0; h < 2; h++) {  // two 32-wide N halves of the 64-wide warp tile
      if (h == 1 && more) {  // the other smem buffer is idle during this whole group: drain staged regs early
        *reinterpret_cast<uint4*>(&sA[buf ^ 1][soA0]) = pa0; *reinterpret_cast<uint4*>(&sA[buf ^ 1][soA1]) = pa1;
        *reinterpret_cast<uint4*>(&sB[buf ^ 1][swz(rr, ch)]) = pb[0]; *reinterpret_cast<uint4*>(&sB[buf ^ 1][swz(64 + rr, ch)]) = pb[1];
        pb[2] = gB0[2 * gBs + off]; pb[3] = gB0[3 * gBs + off];
      }
      int acc[8][4][2];
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        uint32_t af[8], bf[4];
        ldsm_x4(af[0], af[1], af[2], af[3], &sA[buf][swz(wm * 64 + (0 + lm) * 8 + lr, ks)]);
        ldsm_x4(af[4], af[5], af[6], af[7], &sA[buf][swz(wm * 64 + (4 + lm) * 8 + lr, ks)]);
        ldsm_x4(bf[0], bf[1], bf[2], bf[3], &sB[buf][swz(wn * 64 + h * 32 + lm * 8 + lr, ks)]);
#pragma unroll
        for (int i = 0; i < 8; i++)
#pragma unroll
          for (int j = 0; j < 4; j++) {
            if (ks == 0)
              asm("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0,%1}, {%2}, {%3}, {%4,%4};\n"
                  : "=r"(acc[i][j][0]), "=r"(acc[i][j][1]) : "r"(af[i]), "r"(bf[j]), "r"(0));
            else
              asm("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
                  : "+r"(acc[i][j][0]), "+r"(acc[i][j][1]) : "r"(af[i]), "r"(bf[j]));
          }
      }
#pragma unroll
      for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
          tot[i][h * 4 + j][0] += acc[i][j][0] * sc[i];
          tot[i][h * 4 + j][1] += acc[i][j][1] * sc[i];
        }
    }
    if (more) {
      *reinterpret_cast<uint4*>(&sB[buf ^ 1][swz(128 + rr, ch)]) = pb[2]; *reinterpret_cast<uint4*>(&sB[buf ^ 1][swz(192 + rr, ch)]) = pb[3];
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int r = m0 + wm * 64 + i * 8 + (lane >> 2);
    if (r >= M) continue;
    const float sr = smax[r] * inv2sb;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int c = n0 + wn * 64 + j * 8 + (lane & 3) * 2;
      __half2 v = __floats2half2_rn((float)tot[i][j][0] * sr * sw[c], (float)tot[i][j][1] * sr * sw[c + 1]);
      *reinterpret_cast<__half2*>(C + (size_t)r * N + c) = v;
    }
  }
}

template <bool FOLD>
__global__ void __launch_bounds__(256, 1) w4a4g3_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                                       const int* __restrict__ sint, const float* __restrict__ smax,
                                                       const float* __restrict__ sw, __half* __restrict__ C,
                                                       int M, int N, int K, float inv2sb) {
  __shared__ __align__(128) uint8_t sA[2][128 * 64];
  __shared__ __align__(128) uint8_t sB[2][256 * 64];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 2, wn = warp & 3;
  // grouped rasterization
  const int nM = (M + 127) / 128, nN = N / 256, GMB = 8;
  const int id = blockIdx.x;
  const int band = id / (GMB * nN), first_m = band * GMB, bm = min(GMB, nM - first_m);
  const int mt_ = first_m + (id % (GMB * nN)) % bm, nt_ = (id % (GMB * nN)) / bm;
  const int m0 = mt_ * 128, n0 = nt_ * 256;
  const int G = K >> 7;
  const size_t ldk = K >> 1;
  const int ch = tid & 3, rr = tid >> 2;
  const uint4* gA0 = reinterpret_cast<const uint4*>(A + min(m0 + rr, M - 1) * ldk) + ch;
  const uint4* gA1 = reinterpret_cast<const uint4*>(A + min(m0 + 64 + rr, M - 1) * ldk) + ch;
  const uint4* gB0 = reinterpret_cast<const uint4*>(B + (size_t)(n0 + rr) * ldk) + ch;
  const size_t gBs = (size_t)64 * ldk / 16;  // 64 rows, in uint4
  const int soA0 = swz(rr, ch), soA1 = swz(64 + rr, ch);
  const int rbase = m0 + wm * 64 + (lane >> 2);
  int tot[8][8][2];
#pragma unroll
  for (int i = 0; i < 8; i++)
#pragma unroll
    for (int j = 0; j < 8; j++) { tot[i][j][0] = tot[i][j][1] = 0; }
  uint4 pa0 = gA0[0], pa1 = gA1[0], pb[4];
#pragma unroll
  for (int u = 0; u < 4; u++) pb[u] = gB0[u * gBs];
  *reinterpret_cast<uint4*>(&sA[0][soA0]) = pa0; *reinterpret_cast<uint4*>(&sA[0][soA1]) = pa1;
#pragma unroll
  for (int u = 0; u < 4; u++) *reinterpret_cast<uint4*>(&sB[0][swz(u * 64 + rr, ch)]) = pb[u];
  __syncthreads();
  const int lr = lane & 7, lm = lane >> 3;
  int scn[8];
#pragma unroll
  for (int mt = 0; mt < 8; mt++) scn[mt] = __ldg(sint + min(rbase + mt * 8, M - 1));
  for (int g = 0; g < G; g++) {
    const int buf = g & 1;
    const bool more = g + 1 < G;
    const int off = (g + 1) * 4;
    if (more) { pa0 = gA0[off]; pa1 = gA1[off]; pb[0] = gB0[off]; pb[1] = gB0[gBs + off]; pb[2] = gB0[2 * gBs + off]; pb[3] = gB0[3 * gBs + off]; }
    int sc[8];
#pragma unroll
    for (int mt = 0; mt < 8; mt++) { sc[mt] = scn[mt]; if (more) scn[mt] = __ldg(sint + (size_t)(g + 1) * M + min(rbase + mt * 8, M - 1)); }
#pragma unroll
    // A fragments for all 4 k32 steps of this group stay in registers (32 regs); B is streamed per 16-wide N quarter.
    uint32_t af[4][8];
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
      ldsm_x4(af[ks][0], af[ks][1], af[ks][2], af[ks][3], &sA[buf][swz(wm * 64 + (0 + lm) * 8 + lr, ks)]);
      ldsm_x4(af[ks][4], af[ks][5], af[ks][6], af[ks][7], &sA[buf][swz(wm * 64 + (4 + lm) * 8 + lr, ks)]);
    }
#pragma unroll
    for (int q = 0; q < 4; q++) {
      int acc[8][2][2];
#pragma unroll
      for (int kp = 0; kp < 2; kp++) {  // k32-step pairs
        uint32_t bf[4];  // (ntile0, ks), (ntile1, ks), (ntile0, ks+1), (ntile1, ks+1)
        ldsm_x4(bf[0], bf[1], bf[2], bf[3], &sB[buf][swz(wn * 64 + q * 16 + (lm & 1) * 8 + lr, kp * 2 + (lm >> 1))]);
#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          const int ks = kp * 2 + kk;
#pragma unroll
          for (int i = 0; i < 8; i++)
#pragma unroll
            for (int j = 0; j < 2; j++) {
              if (ks == 0)
                asm("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0,%1}, {%2}, {%3}, {%4,%4};\n"
                    : "=r"(acc[i][j][0]), "=r"(acc[i][j][1]) : "r"(af[ks][i]), "r"(bf[kk * 2 + j]), "r"(0));
              else
                asm("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
                    : "+r"(acc[i][j][0]), "+r"(acc[i][j][1]) : "r"(af[ks][i]), "r"(bf[kk * 2 + j]));
            }
        }
      }
#pragma unroll
      for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) {
          if (FOLD) {
            tot[i][q * 2 + j][0] += acc[i][j][0] * sc[i];
            tot[i][q * 2 + j][1] += acc[i][j][1] * sc[i];
          } else {
            tot[i][q * 2 + j][0] += acc[i][j][0];
            tot[i][q * 2 + j][1] += acc[i][j][1];
          }
        }
    }
    if (more) {
      *reinterpret_cast<uint4*>(&sA[buf ^ 1][soA0]) = pa0; *reinterpret_cast<uint4*>(&sA[buf ^ 1][soA1]) = pa1;
      *reinterpret_cast<uint4*>(&sB[buf ^ 1][swz(rr, ch)]) = pb[0]; *reinterpret_cast<uint4*>(&sB[buf ^ 1][swz(64 + rr, ch)]) = pb[1];
      *reinterpret_cast<uint4*>(&sB[buf ^ 1][swz(128 + rr, ch)]) = pb[2]; *reinterpret_cast<uint4*>(&sB[buf ^ 1][swz(192 + rr, ch)]) = pb[3];
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int r = m0 + wm * 64 + i * 8 + (lane >> 2);
    if (r >= M) continue;
    const float sr = smax[r] * inv2sb;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int c = n0 + wn * 64 + j * 8 + (lane & 3) * 2;
      __half2 v = __floats2half2_rn((float)tot[i][j][0] * sr * sw[c], (float)tot[i][j][1] * sr * sw[c + 1]);
      *reinterpret_cast<__half2*>(C + (size_t)r * N + c) = v;
    }
  }
}

torch::Tensor w4a4g_gemm(torch::Tensor a, torch::Tensor b, torch::Tensor sint, torch::Tensor smax, torch::Tensor sw, int64_t sbits, int64_t ver) {
  TORCH_CHECK(a.dtype() == torch::kInt8 && b.dtype() == torch::kInt8 && a.is_contiguous() && b.is_contiguous());
  TORCH_CHECK(sint.dtype() == torch::kInt32 && smax.dtype() == torch::kFloat32 && sw.dtype() == torch::kFloat32);
  int M = a.size(0), K = a.size(1) * 2, N = b.size(0);
  TORCH_CHECK(b.size(1) * 2 == K && K % 128 == 0 && N % 128 == 0, "K, N must be multiples of 128");
  TORCH_CHECK(sint.size(0) == K / 128 && sint.size(1) == M, "sint must be [K/128, M]");
  TORCH_CHECK((double)(K / 128) * 8192.0 * (double)(1 << sbits) < 2147483648.0, "int32 accumulator bound violated");
  const at::cuda::OptionalCUDAGuard guard(a.device());
  auto out = torch::empty({M, N}, a.options().dtype(torch::kFloat16));
  if (ver == 4 && N % 256 == 0) {  // PERF PROBE ONLY: v3 without the group-scale fold (wrong math)
    dim3 grid((N / 256) * ((M + 127) / 128));
    w4a4g3_kernel<false><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), sint.data_ptr<int>(),
        smax.data_ptr<float>(), sw.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr()), M, N, K, 1.f / (float)(1 << sbits));
  } else if (ver == 3 && N % 256 == 0) {
    dim3 grid((N / 256) * ((M + 127) / 128));
    w4a4g3_kernel<true><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), sint.data_ptr<int>(),
        smax.data_ptr<float>(), sw.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr()), M, N, K, 1.f / (float)(1 << sbits));
  } else if (ver == 2 && N % 256 == 0) {
    dim3 grid((N / 256) * ((M + 127) / 128));
    w4a4g2_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), sint.data_ptr<int>(),
        smax.data_ptr<float>(), sw.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr()), M, N, K, 1.f / (float)(1 << sbits));
  } else {
    dim3 grid(N / 128, (M + 127) / 128);
    w4a4g_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), sint.data_ptr<int>(),
        smax.data_ptr<float>(), sw.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr()), M, N, K, 1.f / (float)(1 << sbits));
  }
  return out;
}

// activation path for w4a4g: Hadamard-128 (warp-register FWHT, as v2), per-(row,128) int4 codes, integer group scales
// s_int[g, row] = round(s_g / s_max * 2^SB) (>= 1), smax[row] = max_g s_g.   s_g = absmax_g / qmax.
template <int NBMAX>
__global__ void __launch_bounds__(256) act_quant_h128g_kernel(const __half* __restrict__ x, int8_t* __restrict__ q,
                                                              int* __restrict__ sint, float* __restrict__ smax, int M, int K,
                                                              float qmax, float two_sb) {
  __shared__ float red[8];
  const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nb = K >> 7;
  const __half* xr = x + (size_t)r * K;
  float v[NBMAX][4], gmax[NBMAX];
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < NBMAX; j++) {
    int b = warp + j * 8;
    gmax[j] = 0.f;
    if (b < nb) {
      uint2 raw = *reinterpret_cast<const uint2*>(xr + b * 128 + lane * 4);
      float2 p0 = __half22float2(*reinterpret_cast<__half2*>(&raw.x)), p1 = __half22float2(*reinterpret_cast<__half2*>(&raw.y));
      float a0 = p0.x + p0.y, a1 = p0.x - p0.y, a2 = p1.x + p1.y, a3 = p1.x - p1.y;
      float e[4] = {a0 + a2, a1 + a3, a0 - a2, a1 - a3};
#pragma unroll
      for (int m = 1; m < 32; m <<= 1) {
        bool up = lane & m;
#pragma unroll
        for (int t = 0; t < 4; t++) {
          float o = __shfl_xor_sync(0xffffffff, e[t], m);
          e[t] = up ? (o - e[t]) : (e[t] + o);
        }
      }
      float gm = 0.f;
#pragma unroll
      for (int t = 0; t < 4; t++) { v[j][t] = e[t] * 0.08838834764831845f; gm = fmaxf(gm, fabsf(v[j][t])); }
      for (int o = 16; o; o >>= 1) gm = fmaxf(gm, __shfl_xor_sync(0xffffffff, gm, o));
      gmax[j] = gm;
      amax = fmaxf(amax, gm);
    }
  }
  if (lane == 0) red[warp] = amax;
  __syncthreads();
  amax = red[0];
#pragma unroll
  for (int w = 1; w < 8; w++) amax = fmaxf(amax, red[w]);
  const float smx = amax > 0.f ? amax / qmax : 1.f;
  if (threadIdx.x == 0) smax[r] = smx;
  uint16_t* qr = reinterpret_cast<uint16_t*>(q + (size_t)r * (K / 2));
#pragma unroll
  for (int j = 0; j < NBMAX; j++) {
    int b = warp + j * 8;
    if (b < nb) {
      float sg = gmax[j] > 0.f ? gmax[j] / qmax : smx;
      int si = max(1, __float2int_rn(sg / smx * two_sb));
      float inv = 1.f / sg;
      uint32_t pk = 0;
#pragma unroll
      for (int t = 0; t < 4; t++) {
        int c = max(-8, min(7, __float2int_rn(v[j][t] * inv)));
        pk |= (uint32_t)(c & 15) << (4 * t);
      }
      qr[b * 32 + lane] = (uint16_t)pk;
      if (lane == 0) sint[(size_t)b * M + r] = si;
    }
  }
}

std::vector<torch::Tensor> act_quant_h128g(torch::Tensor x, double qmax, int64_t sbits) {
  TORCH_CHECK(x.dtype() == torch::kFloat16 && x.is_contiguous() && x.dim() == 2);
  int M = x.size(0), K = x.size(1);
  TORCH_CHECK(K % 128 == 0 && K / 128 <= 96);
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto q = torch::empty({M, K / 2}, x.options().dtype(torch::kInt8));
  auto si = torch::empty({K / 128, M}, x.options().dtype(torch::kInt32));
  auto sm = torch::empty({M}, x.options().dtype(torch::kFloat32));
  auto st = at::cuda::getCurrentCUDAStream();
  int nbw = (K / 128 + 7) / 8;
  auto xp = reinterpret_cast<const __half*>(x.data_ptr());
  float t = (float)(1 << sbits);
#define K7L(NB) act_quant_h128g_kernel<NB><<<M, 256, 0, st>>>(xp, q.data_ptr<int8_t>(), si.data_ptr<int>(), sm.data_ptr<float>(), M, K, (float)qmax, t)
  if (nbw <= 4) K7L(4); else if (nbw <= 6) K7L(6); else if (nbw <= 9) K7L(9); else K7L(12);
#undef K7L
  return {q, si, sm};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w4a4_gemm", &w4a4_gemm, "W4A4 s4 tensor-core GEMM, per-token x per-channel scales -> fp16");
  m.def("w4a4g_gemm", &w4a4g_gemm, "group-scaled-activation W4A4 (hand-written sm75)");
  m.def("act_quant_h128g", &act_quant_h128g, "Hadamard-128 + per-(row,128) int4 + integer group scales");
  m.def("act_quant_h128", &act_quant_h128, "v2 warp-register FWHT-128 + per-token int4");
  m.def("act_quant_had", &act_quant_had, "fp16 -> block-Hadamard -> per-token int4 (packed) + scale");
}
