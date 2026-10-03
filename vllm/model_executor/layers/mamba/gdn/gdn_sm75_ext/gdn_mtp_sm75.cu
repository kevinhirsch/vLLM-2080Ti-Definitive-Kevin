// Fused GatedDeltaNet post-conv MTP decode for sm_75 / fp16 (Lane K5, 2026-10-03; decode-megakernel step 1).
//
// Port of upstream vLLM `fused_gdn_decode_post_conv_mtp` (#51674, bf16 + sm80 + cp.async; pruned from this fork)
// to Turing: fp16 activations, fp16 or fp32 recurrent state, no cp.async (each thread streams its own state rows
// straight into registers, which on sm_75 is the cheaper path anyway), HV/H in {1,2,3,4,8}.
//
// One CTA = (request, value head).  Replaces, per GDN layer and verify step, the eager chain
//   a/b .contiguous() x2, rearrange_mixed_qkv cat/copies, fused_sigmoid_gating_delta_rule_update (Triton),
//   output scatter copies, RMSNormGated native (pow/mean/add/rsqrt/mul/mul/silu/mul/cast = 9 kernels)
// with ONE kernel.  Semantics follow the Triton kernel + RMSNormGated(norm_before_gate=True) exactly:
//   - source state  = state[state_indices[n, accepted-1]]; invalid (idx<0 or idx==null_block_id, or accepted out
//     of [1, width]) -> output rows are zero (the Triton path zeroes o, and RMSNorm(0) * gate = 0).
//   - per token: q,k l2-normalised (eps 1e-6), q *= scale; h *= exp(-exp(A_log) * softplus(a + dt_bias));
//     v' = (v - h k) * sigmoid(b); h += v' k^T; o = h q  (o rounded to fp16 like the Triton store);
//     state for token t stored to state[state_indices[n, t]] when that index is valid.
//   - out = fp16( (o * rsqrt(mean(o^2) + eps)) * w * act(z) ),  act = silu or sigmoid.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace {

constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int kThreads = 512;
constexpr int kWarps = kThreads / 32;      // 16
constexpr int kRows = kDimV / kWarps;      // 8 value rows per warp
constexpr int kMaxTok = 8;

template <typename T> __device__ __forceinline__ float ld1(const T* p);
template <> __device__ __forceinline__ float ld1<float>(const float* p) { return *p; }
template <> __device__ __forceinline__ float ld1<__half>(const __half* p) { return __half2float(*p); }

template <typename S> __device__ __forceinline__ void ld4(const S* p, float* o);
template <> __device__ __forceinline__ void ld4<float>(const float* p, float* o) {
  const float4 v = *reinterpret_cast<const float4*>(p);
  o[0] = v.x; o[1] = v.y; o[2] = v.z; o[3] = v.w;
}
template <> __device__ __forceinline__ void ld4<__half>(const __half* p, float* o) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
  const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
  o[0] = a.x; o[1] = a.y; o[2] = b.x; o[3] = b.y;
}

template <typename S> __device__ __forceinline__ void st4(S* p, const float* v);
template <> __device__ __forceinline__ void st4<float>(float* p, const float* v) {
  *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
}
template <> __device__ __forceinline__ void st4<__half>(__half* p, const float* v) {
  __half2 a = __floats2half2_rn(v[0], v[1]);
  __half2 b = __floats2half2_rn(v[2], v[3]);
  uint2 u;
  u.x = *reinterpret_cast<uint32_t*>(&a);
  u.y = *reinterpret_cast<uint32_t*>(&b);
  *reinterpret_cast<uint2*>(p) = u;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

struct Strides {
  int64_t mixed_row, a_row, b_row, z_row, state_slot, out_row;
};

// DT: dt_bias element type, NW: norm weight element type, S: state element type
template <typename S, typename DT, typename NW, int HPK, bool SIGMOID>
__global__ __launch_bounds__(kThreads, 1) void gdn_mtp_sm75_kernel(
    const __half* __restrict__ mixed_qkv, const __half* __restrict__ a, const __half* __restrict__ b,
    const float* __restrict__ a_log, const DT* __restrict__ dt_bias, const int* __restrict__ state_indices,
    const int* __restrict__ cu_seqlens, const int* __restrict__ num_accepted, S* __restrict__ state,
    const __half* __restrict__ z, const NW* __restrict__ norm_w, __half* __restrict__ out, int H, int HV,
    int width, int null_block_id, float scale, float eps, Strides st) {
  const int n = blockIdx.x;
  const int hv = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int bos = cu_seqlens[n];
  const int T = cu_seqlens[n + 1] - bos;
  if (T <= 0) return;

  const int acc = num_accepted[n];
  int src = -1;
  if (acc >= 1 && acc <= width) src = state_indices[(int64_t)n * width + acc - 1];
  if (src < 0 || src == null_block_id || T > kMaxTok) {
    for (int i = tid; i < T * kDimV; i += kThreads) {
      const int t = i / kDimV, v = i % kDimV;
      out[(int64_t)(bos + t) * st.out_row + (int64_t)hv * kDimV + v] = __float2half(0.0f);
    }
    return;
  }

  const int kh = hv / HPK;
  __shared__ __align__(16) float sq[kMaxTok][kDimK];
  __shared__ __align__(16) float sk[kMaxTok][kDimK];
  __shared__ float sv[kMaxTok][kDimV];
  __shared__ float so[kMaxTok][kDimV];
  __shared__ float sdecay[kMaxTok];
  __shared__ float sbeta[kMaxTok];
  __shared__ int sdst[kMaxTok];

  // Issue the state loads first (the dominant bytes); everything else overlaps with them.
  const int k0 = lane * 4;
  const S* src_state = state + (int64_t)src * st.state_slot + (int64_t)hv * kDimV * kDimK;
  float h[kRows][4];
#pragma unroll
  for (int r = 0; r < kRows; ++r) ld4<S>(src_state + (warp + r * kWarps) * kDimK + k0, h[r]);

  if (warp < T) {
    const int t = warp;
    const int64_t mb = (int64_t)(bos + t) * st.mixed_row;
    float qv[4], kv[4], qs = 0.f, ks = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + i * 32;
      qv[i] = __half2float(mixed_qkv[mb + kh * kDimK + d]);
      kv[i] = __half2float(mixed_qkv[mb + (int64_t)H * kDimK + kh * kDimK + d]);
      sv[t][d] = __half2float(mixed_qkv[mb + (int64_t)2 * H * kDimK + (int64_t)hv * kDimV + d]);
      qs += qv[i] * qv[i];
      ks += kv[i] * kv[i];
    }
    qs = warp_sum(qs);
    ks = warp_sum(ks);
    const float qn = rsqrtf(qs + 1e-6f) * scale;
    const float kn = rsqrtf(ks + 1e-6f);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + i * 32;
      sq[t][d] = qv[i] * qn;
      sk[t][d] = kv[i] * kn;
    }
    if (lane == 0) {
      const float x = __half2float(a[(int64_t)(bos + t) * st.a_row + hv]) + ld1<DT>(dt_bias + hv);
      const float sp = x <= 20.0f ? log1pf(expf(x)) : x;
      sdecay[t] = expf(-expf(a_log[hv]) * sp);
      sbeta[t] = sigmoidf_(__half2float(b[(int64_t)(bos + t) * st.b_row + hv]));
      const int d = t < width ? state_indices[(int64_t)n * width + t] : -1;
      sdst[t] = (d >= 0 && d != null_block_id) ? d : -1;
    }
  }
  __syncthreads();

  for (int t = 0; t < T; ++t) {
    float kk[4], qq[4];
    {
      const float4 k4 = *reinterpret_cast<const float4*>(&sk[t][k0]);
      const float4 q4 = *reinterpret_cast<const float4*>(&sq[t][k0]);
      kk[0] = k4.x; kk[1] = k4.y; kk[2] = k4.z; kk[3] = k4.w;
      qq[0] = q4.x; qq[1] = q4.y; qq[2] = q4.z; qq[3] = q4.w;
    }
    const float dec = sdecay[t], beta = sbeta[t];
    float hk[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      float s = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        h[r][i] *= dec;
        s += h[r][i] * kk[i];
      }
      hk[r] = s;
    }
#pragma unroll
    for (int r = 0; r < kRows; ++r) hk[r] = warp_sum(hk[r]);
    float hq[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      const float delta = (sv[t][warp + r * kWarps] - hk[r]) * beta;
      float s = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        h[r][i] += kk[i] * delta;
        s += h[r][i] * qq[i];
      }
      hq[r] = s;
    }
#pragma unroll
    for (int r = 0; r < kRows; ++r) hq[r] = warp_sum(hq[r]);
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < kRows; ++r) so[t][warp + r * kWarps] = __half2float(__float2half(hq[r]));
    }
    const int dst = sdst[t];
    if (dst >= 0) {
      S* d = state + (int64_t)dst * st.state_slot + (int64_t)hv * kDimV * kDimK;
#pragma unroll
      for (int r = 0; r < kRows; ++r) st4<S>(d + (warp + r * kWarps) * kDimK + k0, h[r]);
    }
  }
  __syncthreads();

  if (warp < T) {
    const int t = warp;
    float ov[4], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      ov[i] = so[t][lane + i * 32];
      ss += ov[i] * ov[i];
    }
    ss = warp_sum(ss);
    const float rstd = rsqrtf(ss / (float)kDimV + eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int v = lane + i * 32;
      const float g = __half2float(z[(int64_t)(bos + t) * st.z_row + (int64_t)hv * kDimV + v]);
      const float act = SIGMOID ? sigmoidf_(g) : g * sigmoidf_(g);
      const float y = ((ov[i] * rstd) * ld1<NW>(norm_w + v)) * act;
      out[(int64_t)(bos + t) * st.out_row + (int64_t)hv * kDimV + v] = __float2half(y);
    }
  }
}

template <typename S, typename DT, typename NW, int HPK>
void launch(bool sigmoid, dim3 grid, cudaStream_t s, const torch::Tensor& mixed, const torch::Tensor& a,
            const torch::Tensor& b, const torch::Tensor& a_log, const torch::Tensor& dt_bias,
            const torch::Tensor& sidx, const torch::Tensor& cu, const torch::Tensor& nacc, torch::Tensor& state,
            const torch::Tensor& z, const torch::Tensor& nw, torch::Tensor& out, int H, int HV, int width,
            int null_id, float scale, float eps, Strides st) {
#define K5_ARGS                                                                                                     \
  reinterpret_cast<const __half*>(mixed.data_ptr()), reinterpret_cast<const __half*>(a.data_ptr()),                \
      reinterpret_cast<const __half*>(b.data_ptr()), a_log.data_ptr<float>(),                                      \
      reinterpret_cast<const DT*>(dt_bias.data_ptr()), sidx.data_ptr<int>(), cu.data_ptr<int>(),                    \
      nacc.data_ptr<int>(), reinterpret_cast<S*>(state.data_ptr()), reinterpret_cast<const __half*>(z.data_ptr()), \
      reinterpret_cast<const NW*>(nw.data_ptr()), reinterpret_cast<__half*>(out.data_ptr()), H, HV, width,         \
      null_id, scale, eps, st
  if (sigmoid)
    gdn_mtp_sm75_kernel<S, DT, NW, HPK, true><<<grid, kThreads, 0, s>>>(K5_ARGS);
  else
    gdn_mtp_sm75_kernel<S, DT, NW, HPK, false><<<grid, kThreads, 0, s>>>(K5_ARGS);
#undef K5_ARGS
}

template <typename S, typename DT, typename NW>
void dispatch_hpk(int hpk, bool sigmoid, dim3 grid, cudaStream_t s, const torch::Tensor& mixed,
                  const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& a_log,
                  const torch::Tensor& dt_bias, const torch::Tensor& sidx, const torch::Tensor& cu,
                  const torch::Tensor& nacc, torch::Tensor& state, const torch::Tensor& z, const torch::Tensor& nw,
                  torch::Tensor& out, int H, int HV, int width, int null_id, float scale, float eps, Strides st) {
#define K5_CALL(N)                                                                                                \
  launch<S, DT, NW, N>(sigmoid, grid, s, mixed, a, b, a_log, dt_bias, sidx, cu, nacc, state, z, nw, out, H, HV, \
                       width, null_id, scale, eps, st)
  switch (hpk) {
    case 1: K5_CALL(1); break;
    case 2: K5_CALL(2); break;
    case 3: K5_CALL(3); break;
    case 4: K5_CALL(4); break;
    default: K5_CALL(8); break;
  }
#undef K5_CALL
}

}  // namespace

// mixed_qkv [L, 2*H*128 + HV*128] fp16 (row stride free, channel stride 1); a, b [L, HV] fp16 (row stride free);
// z [L, HV, 128] fp16 (row stride free, head rows contiguous); state [slots, HV, 128, 128] fp16|fp32;
// out [L, HV, 128] fp16 (row stride free, head rows contiguous); state_indices [N, W] int32; cu_seqlens [N+1];
// num_accepted [N] int32.  Only requests 0..N-1 are processed; rows outside their [bos, eos) are untouched.
void gdn_mtp_sm75(torch::Tensor mixed_qkv, torch::Tensor a, torch::Tensor b, torch::Tensor a_log,
                  torch::Tensor dt_bias, torch::Tensor state_indices, torch::Tensor cu_seqlens,
                  torch::Tensor num_accepted, torch::Tensor state, torch::Tensor z, torch::Tensor norm_w,
                  torch::Tensor out, double scale, double eps, int64_t null_block_id, bool sigmoid_gate) {
  TORCH_CHECK(mixed_qkv.is_cuda() && mixed_qkv.scalar_type() == at::kHalf && mixed_qkv.dim() == 2 &&
              mixed_qkv.stride(1) == 1, "mixed_qkv: CUDA fp16 [L, C], channel-contiguous");
  TORCH_CHECK(a.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf && a.dim() == 2 && b.dim() == 2 &&
              a.stride(1) == 1 && b.stride(1) == 1, "a/b: fp16 [L, HV] with contiguous heads");
  TORCH_CHECK(z.scalar_type() == at::kHalf && z.dim() == 3 && z.size(2) == kDimV && z.stride(2) == 1 &&
              z.stride(1) == kDimV, "z: fp16 [L, HV, 128] with contiguous head rows");
  TORCH_CHECK(out.scalar_type() == at::kHalf && out.dim() == 3 && out.size(2) == kDimV && out.stride(2) == 1 &&
              out.stride(1) == kDimV, "out: fp16 [L, HV, 128] with contiguous head rows");
  TORCH_CHECK(a_log.scalar_type() == at::kFloat && a_log.is_contiguous(), "A_log: fp32 contiguous");
  TORCH_CHECK(dt_bias.is_contiguous() && (dt_bias.scalar_type() == at::kFloat || dt_bias.scalar_type() == at::kHalf),
              "dt_bias: fp32|fp16 contiguous");
  TORCH_CHECK(norm_w.is_contiguous() && norm_w.numel() == kDimV &&
              (norm_w.scalar_type() == at::kFloat || norm_w.scalar_type() == at::kHalf), "norm_w: 128 fp32|fp16");
  TORCH_CHECK(state.dim() == 4 && state.size(2) == kDimV && state.size(3) == kDimK && state.stride(3) == 1 &&
              state.stride(2) == kDimK && state.stride(1) == kDimV * kDimK, "state: [slots, HV, 128, 128]");
  TORCH_CHECK(state.scalar_type() == at::kFloat || state.scalar_type() == at::kHalf, "state: fp32|fp16");
  const int esz = state.scalar_type() == at::kFloat ? 4 : 2;
  TORCH_CHECK(reinterpret_cast<uintptr_t>(state.data_ptr()) % 16 == 0 && (state.stride(0) * esz) % 16 == 0,
              "state: 16-byte aligned slots");
  TORCH_CHECK(state_indices.scalar_type() == at::kInt && state_indices.dim() == 2 && state_indices.is_contiguous(),
              "state_indices: int32 [N, W] contiguous");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt && cu_seqlens.is_contiguous(), "cu_seqlens: int32");
  TORCH_CHECK(num_accepted.scalar_type() == at::kInt && num_accepted.is_contiguous(), "num_accepted: int32");
  const int N = state_indices.size(0);
  const int W = state_indices.size(1);
  TORCH_CHECK(cu_seqlens.numel() >= N + 1 && num_accepted.numel() >= N, "cu_seqlens/num_accepted too short");
  TORCH_CHECK(W >= 1 && W <= kMaxTok, "state_indices width must be 1..8");
  const int HV = state.size(1);
  const int64_t kw = mixed_qkv.size(1) - (int64_t)HV * kDimV;
  TORCH_CHECK(kw > 0 && kw % (2 * kDimK) == 0, "mixed_qkv width inconsistent with state heads");
  const int H = kw / (2 * kDimK);
  TORCH_CHECK(HV % H == 0, "HV % H");
  const int hpk = HV / H;
  TORCH_CHECK(hpk == 1 || hpk == 2 || hpk == 3 || hpk == 4 || hpk == 8, "HV/H must be in {1,2,3,4,8}");
  TORCH_CHECK(a.size(1) == HV && b.size(1) == HV && z.size(1) == HV && out.size(1) == HV, "head count mismatch");
  if (N == 0) return;
  const Strides st{mixed_qkv.stride(0), a.stride(0), b.stride(0), z.stride(0), state.stride(0), out.stride(0)};
  const at::cuda::OptionalCUDAGuard guard(mixed_qkv.device());
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  const dim3 grid(N, HV);
  const bool s32 = state.scalar_type() == at::kFloat;
  const bool d32 = dt_bias.scalar_type() == at::kFloat;
  const bool w32 = norm_w.scalar_type() == at::kFloat;
  const float sc = (float)scale, ep = (float)eps;
  const int nid = (int)null_block_id;
#define K5_D(S, DT, NW)                                                                                          \
  dispatch_hpk<S, DT, NW>(hpk, sigmoid_gate, grid, s, mixed_qkv, a, b, a_log, dt_bias, state_indices, cu_seqlens, \
                          num_accepted, state, z, norm_w, out, H, HV, W, nid, sc, ep, st)
  if (s32) {
    if (d32) { if (w32) K5_D(float, float, float); else K5_D(float, float, __half); }
    else     { if (w32) K5_D(float, __half, float); else K5_D(float, __half, __half); }
  } else {
    if (d32) { if (w32) K5_D(__half, float, float); else K5_D(__half, float, __half); }
    else     { if (w32) K5_D(__half, __half, float); else K5_D(__half, __half, __half); }
  }
#undef K5_D
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gdn_mtp", &gdn_mtp_sm75, "K5 fused GDN post-conv MTP decode (sm_75)"); }
