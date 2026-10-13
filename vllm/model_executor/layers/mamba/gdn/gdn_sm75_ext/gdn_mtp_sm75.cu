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
#include <cmath>
#include <limits>

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

// dt_bias and the gated-norm weight are read in the model's own parameter dtype (fp16 or fp32). The layer must not
// keep derived fp32 copies: the first K5 call happens inside FULL CUDA-graph capture (warmups carry no spec-decode
// metadata), where a lazily created copy is allocated but its cast kernel is only recorded, never run.
__device__ __forceinline__ float ld_param(const void* p, int i, bool f32) {
  return f32 ? static_cast<const float*>(p)[i] : __half2float(static_cast<const __half*>(p)[i]);
}

struct Strides {
  int64_t mixed_row, a_row, b_row, z_row, state_slot, out_row;
};

// Reject malformed metadata before reading or writing either recurrent state.
// The convolution guard applies the same request predicate before its cache
// update. Negative/null destinations remain skippable; a bad positive slot
// invalidates the whole request rather than partially committing it.
__device__ __forceinline__ bool invalid_request(
    const int* indices, int width, int T, int accepted, int num_slots, int null_id, int* src) {
  *src = -1;
  if (T > width || T > kMaxTok || accepted < 1 || accepted > width) return true;
  const int first = indices[0];
  *src = indices[accepted - 1];
  if (first < 0 || first >= num_slots || first == null_id ||
      *src < 0 || *src >= num_slots || *src == null_id) return true;
  for (int t = 0; t < width; ++t) {
    const int dst = indices[t];
    if (dst >= num_slots && dst != null_id) return true;
  }
  return false;
}

__device__ __forceinline__ void zero_interval(
    __half* out, int bos, int eos, int num_tokens, int hv, int tid, int64_t row_stride) {
  const int start = bos < 0 ? 0 : (bos > num_tokens ? num_tokens : bos);
  const int end = eos < 0 ? 0 : (eos > num_tokens ? num_tokens : eos);
  for (int64_t i = tid; i < (int64_t)(end - start) * kDimV; i += kThreads) {
    const int t = i / kDimV, v = i % kDimV;
    out[(int64_t)(start + t) * row_stride + (int64_t)hv * kDimV + v] = __float2half(0.0f);
  }
}

// S: state element type (fp16 or fp32); A_log arrives as fp32, dt_bias and the norm weight as fp16 or fp32
template <typename S, bool SIGMOID>
__global__ __launch_bounds__(kThreads, 1) void gdn_mtp_sm75_kernel(
    const __half* __restrict__ mixed_qkv, const __half* __restrict__ a, const __half* __restrict__ b,
    const float* __restrict__ a_log, const void* __restrict__ dt_bias, const int* __restrict__ state_indices,
    const int* __restrict__ cu_seqlens, const int* __restrict__ num_accepted, S* __restrict__ state,
    const __half* __restrict__ z, const void* __restrict__ norm_w, __half* __restrict__ out, int H, int HV, int HPK,
    int width, int num_tokens, int num_slots, int null_block_id, float scale, float eps, Strides st, bool dt_f32,
    bool nw_f32) {
  const int n = blockIdx.x;
  const int hv = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int bos = cu_seqlens[n];
  const int eos = cu_seqlens[n + 1];
  // Clamp even rejection stores: malformed intervals never form OOB pointers.
  if (bos < 0 || eos < bos || eos > num_tokens) {
    zero_interval(out, bos, eos, num_tokens, hv, tid, st.out_row);
    return;
  }
  const int T = eos - bos;
  if (T <= 0) return;

  const int acc = num_accepted[n];
  int src = -1;
  if (invalid_request(state_indices + (int64_t)n * width, width, T, acc,
                      num_slots, null_block_id, &src)) {
    for (int64_t i = tid; i < (int64_t)T * kDimV; i += kThreads) {
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
      const float x = __half2float(a[(int64_t)(bos + t) * st.a_row + hv]) + ld_param(dt_bias, hv, dt_f32);
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
      const float y = ((ov[i] * rstd) * ld_param(norm_w, v, nw_f32)) * act;
      out[(int64_t)(bos + t) * st.out_row + (int64_t)hv * kDimV + v] = __float2half(y);
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// v2 layout (reduction-light).  v1 gives every value row to a whole warp (32 lanes x 4 K-elements), so each row
// dot product costs a 5-step shuffle and a thread does 8 rows x 2 dots x 5 = 80 shuffles per token; the recurrence
// is instruction/reduction-bound, not memory-bound (K8 measured the stock Triton kernel the same way).  v2 gives each
// row to an 8-lane group (16 K-elements per lane, interleaved as k = m*32 + seg*4 + i so shared-memory float4 reads
// are conflict-free and each 8-lane group reads a contiguous 128 B / 64 B global segment): 2 rows x 2 dots x 3 =
// 12 shuffles per thread per token, the same FMA count, the same register footprint.
constexpr int kGrp = 8;                               // lanes per row
constexpr int kRowsPerPass = kWarps * (32 / kGrp);    // 64
constexpr int kRowsV2 = kDimV / kRowsPerPass;         // 2 rows per thread
constexpr int kSegs = kDimK / (kGrp * 4);             // 4 float4 segments per row per lane

template <typename S, bool SIGMOID>
__global__ __launch_bounds__(kThreads, 1) void gdn_mtp_sm75_v2_kernel(
    const __half* __restrict__ mixed_qkv, const __half* __restrict__ a, const __half* __restrict__ b,
    const float* __restrict__ a_log, const void* __restrict__ dt_bias, const int* __restrict__ state_indices,
    const int* __restrict__ cu_seqlens, const int* __restrict__ num_accepted, S* __restrict__ state,
    const __half* __restrict__ z, const void* __restrict__ norm_w, __half* __restrict__ out, int H, int HV, int HPK,
    int width, int num_tokens, int num_slots, int null_block_id, float scale, float eps, Strides st, bool dt_f32,
    bool nw_f32) {
  const int n = blockIdx.x;
  const int hv = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int grp = lane >> 3;
  const int seg = lane & 7;
  const int bos = cu_seqlens[n];
  const int eos = cu_seqlens[n + 1];
  if (bos < 0 || eos < bos || eos > num_tokens) {
    zero_interval(out, bos, eos, num_tokens, hv, tid, st.out_row);
    return;
  }
  const int T = eos - bos;
  if (T <= 0) return;

  const int acc = num_accepted[n];
  int src = -1;
  if (invalid_request(state_indices + (int64_t)n * width, width, T, acc,
                      num_slots, null_block_id, &src)) {
    for (int64_t i = tid; i < (int64_t)T * kDimV; i += kThreads) {
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
  __shared__ float skq[kMaxTok];

  int rows[kRowsV2];
#pragma unroll
  for (int j = 0; j < kRowsV2; ++j) rows[j] = warp * (32 / kGrp) + grp + j * kRowsPerPass;
  const S* src_state = state + (int64_t)src * st.state_slot + (int64_t)hv * kDimV * kDimK;
  float h[kRowsV2][kSegs][4];
#pragma unroll
  for (int j = 0; j < kRowsV2; ++j)
#pragma unroll
    for (int m = 0; m < kSegs; ++m) ld4<S>(src_state + rows[j] * kDimK + m * 32 + seg * 4, h[j][m]);

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
    float kq = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + i * 32;
      sq[t][d] = qv[i] * qn;
      sk[t][d] = kv[i] * kn;
      kq += (qv[i] * qn) * (kv[i] * kn);
    }
    kq = warp_sum(kq);
    if (lane == 0) {
      skq[t] = kq;
      const float x = __half2float(a[(int64_t)(bos + t) * st.a_row + hv]) + ld_param(dt_bias, hv, dt_f32);
      const float sp = x <= 20.0f ? log1pf(expf(x)) : x;
      sdecay[t] = expf(-expf(a_log[hv]) * sp);
      sbeta[t] = sigmoidf_(__half2float(b[(int64_t)(bos + t) * st.b_row + hv]));
      const int d = t < width ? state_indices[(int64_t)n * width + t] : -1;
      sdst[t] = (d >= 0 && d != null_block_id) ? d : -1;
    }
  }
  __syncthreads();

  for (int t = 0; t < T; ++t) {
    float kk[kSegs][4], qq[kSegs][4];
#pragma unroll
    for (int m = 0; m < kSegs; ++m) {
      const float4 k4 = *reinterpret_cast<const float4*>(&sk[t][m * 32 + seg * 4]);
      const float4 q4 = *reinterpret_cast<const float4*>(&sq[t][m * 32 + seg * 4]);
      kk[m][0] = k4.x; kk[m][1] = k4.y; kk[m][2] = k4.z; kk[m][3] = k4.w;
      qq[m][0] = q4.x; qq[m][1] = q4.y; qq[m][2] = q4.z; qq[m][3] = q4.w;
    }
    // K8's single-pass form: both dot products against the OLD state (independent, no decay pass):
    //   sk = h.k, sq = h.q;  d = beta * (v - dec*sk);  o = dec*sq + d*(k.q);  h' = dec*h + d*k
    const float dec = sdecay[t], beta = sbeta[t], kq = skq[t];
    float hk[kRowsV2], hq[kRowsV2];
#pragma unroll
    for (int j = 0; j < kRowsV2; ++j) {
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int m = 0; m < kSegs; ++m)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          s0 += h[j][m][i] * kk[m][i];
          s1 += h[j][m][i] * qq[m][i];
        }
      hk[j] = s0;
      hq[j] = s1;
    }
#pragma unroll
    for (int o = kGrp / 2; o > 0; o >>= 1)
#pragma unroll
      for (int j = 0; j < kRowsV2; ++j) {
        hk[j] += __shfl_xor_sync(0xffffffffu, hk[j], o);
        hq[j] += __shfl_xor_sync(0xffffffffu, hq[j], o);
      }
#pragma unroll
    for (int j = 0; j < kRowsV2; ++j) {
      const float delta = (sv[t][rows[j]] - dec * hk[j]) * beta;
      hq[j] = dec * hq[j] + delta * kq;
#pragma unroll
      for (int m = 0; m < kSegs; ++m)
#pragma unroll
        for (int i = 0; i < 4; ++i) h[j][m][i] = dec * h[j][m][i] + kk[m][i] * delta;
    }
    if (seg == 0) {
#pragma unroll
      for (int j = 0; j < kRowsV2; ++j) so[t][rows[j]] = __half2float(__float2half(hq[j]));
    }
    const int dst = sdst[t];
    if (dst >= 0) {
      S* d = state + (int64_t)dst * st.state_slot + (int64_t)hv * kDimV * kDimK;
#pragma unroll
      for (int j = 0; j < kRowsV2; ++j)
#pragma unroll
        for (int m = 0; m < kSegs; ++m) st4<S>(d + rows[j] * kDimK + m * 32 + seg * 4, h[j][m]);
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
      const float y = ((ov[i] * rstd) * ld_param(norm_w, v, nw_f32)) * act;
      out[(int64_t)(bos + t) * st.out_row + (int64_t)hv * kDimV + v] = __float2half(y);
    }
  }
}

template <typename S, bool SIG>
void launch_one(int variant, dim3 grid, cudaStream_t s, const __half* mixed, const __half* a, const __half* b,
                const float* a_log, const void* dt_bias, const int* sidx, const int* cu, const int* nacc, S* state,
                const __half* z, const void* nw, __half* out, int H, int HV, int hpk, int width,
                int num_tokens, int num_slots, int null_id,
                float scale, float eps, Strides st, bool dt_f32, bool nw_f32) {
  if (variant == 1)
    gdn_mtp_sm75_kernel<S, SIG><<<grid, kThreads, 0, s>>>(mixed, a, b, a_log, dt_bias, sidx, cu, nacc, state, z, nw,
                                                         out, H, HV, hpk, width, num_tokens, num_slots,
                                                         null_id, scale, eps, st, dt_f32, nw_f32);
  else
    gdn_mtp_sm75_v2_kernel<S, SIG><<<grid, kThreads, 0, s>>>(mixed, a, b, a_log, dt_bias, sidx, cu, nacc, state, z,
                                                            nw, out, H, HV, hpk, width, num_tokens, num_slots,
                                                            null_id, scale, eps, st, dt_f32, nw_f32);
}

}  // namespace

// mixed_qkv [L, 2*H*128 + HV*128] fp16 (row stride free, channel stride 1); a, b [L, HV] fp16 (row stride free);
// z [L, HV, 128] fp16 (row stride free, head rows contiguous); state [slots, HV, 128, 128] fp16|fp32;
// out [L, HV, 128] fp16 (row stride free, head rows contiguous); A_log [HV] fp32; dt_bias [HV], norm_w [128] fp16|fp32;
// state_indices [N, W] int32; cu_seqlens [N+1]; num_accepted [N] int32.  Only requests 0..N-1 are processed;
// rows outside their [bos, eos) are untouched.  variant: 1 = 32 lanes/row, 2 = 8 lanes/row single-pass.
void gdn_mtp_sm75(torch::Tensor mixed_qkv, torch::Tensor a, torch::Tensor b, torch::Tensor a_log,
                  torch::Tensor dt_bias, torch::Tensor state_indices, torch::Tensor cu_seqlens,
                  torch::Tensor num_accepted, torch::Tensor state, torch::Tensor z, torch::Tensor norm_w,
                  torch::Tensor out, double scale, double eps, int64_t null_block_id, bool sigmoid_gate,
                  int64_t variant) {
  TORCH_CHECK(mixed_qkv.is_cuda() && mixed_qkv.scalar_type() == at::kHalf && mixed_qkv.dim() == 2 &&
              mixed_qkv.stride(1) == 1 && mixed_qkv.stride(0) >= mixed_qkv.size(1),
              "mixed_qkv: CUDA fp16 [L, C], non-overlapping channel-contiguous rows");
  for (const auto& tensor : {a, b, a_log, dt_bias, state_indices, cu_seqlens,
                            num_accepted, state, z, norm_w, out}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == mixed_qkv.device(),
                "all K5 tensors must be CUDA tensors on the same device");
  }
  TORCH_CHECK(variant == 1 || variant == 2, "K5 variant must be 1 or 2");
  TORCH_CHECK(std::isfinite(scale) && scale > 0 && std::isfinite(eps) && eps > 0,
              "scale and eps must be finite and positive");
  TORCH_CHECK(a.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf && a.dim() == 2 && b.dim() == 2 &&
              a.stride(1) == 1 && b.stride(1) == 1, "a/b: fp16 [L, HV] with contiguous heads");
  TORCH_CHECK(z.scalar_type() == at::kHalf && z.dim() == 3 && z.size(2) == kDimV && z.stride(2) == 1 &&
              z.stride(1) == kDimV, "z: fp16 [L, HV, 128] with contiguous head rows");
  TORCH_CHECK(out.scalar_type() == at::kHalf && out.dim() == 3 && out.size(2) == kDimV && out.stride(2) == 1 &&
              out.stride(1) == kDimV, "out: fp16 [L, HV, 128] with contiguous head rows");
  TORCH_CHECK(a_log.scalar_type() == at::kFloat && a_log.dim() == 1 && a_log.is_contiguous(), "A_log: fp32 contiguous vector");
  TORCH_CHECK((dt_bias.scalar_type() == at::kFloat || dt_bias.scalar_type() == at::kHalf) && dt_bias.dim() == 1 &&
              dt_bias.is_contiguous(), "dt_bias: fp16|fp32 contiguous vector");
  TORCH_CHECK((norm_w.scalar_type() == at::kFloat || norm_w.scalar_type() == at::kHalf) && norm_w.dim() == 1 &&
              norm_w.is_contiguous() && norm_w.numel() == kDimV, "norm_w: 128 fp16|fp32");
  TORCH_CHECK(state.dim() == 4 && state.size(2) == kDimV && state.size(3) == kDimK && state.stride(3) == 1 &&
              state.stride(2) == kDimK && state.stride(1) == kDimV * kDimK, "state: [slots, HV, 128, 128]");
  TORCH_CHECK(state.scalar_type() == at::kFloat || state.scalar_type() == at::kHalf, "state: fp32|fp16");
  const int esz = state.scalar_type() == at::kFloat ? 4 : 2;
  TORCH_CHECK(reinterpret_cast<uintptr_t>(state.data_ptr()) % 16 == 0 && (state.stride(0) * esz) % 16 == 0,
              "state: 16-byte aligned slots");
  TORCH_CHECK(state_indices.scalar_type() == at::kInt && state_indices.dim() == 2 && state_indices.is_contiguous(),
              "state_indices: int32 [N, W] contiguous");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt && cu_seqlens.dim() == 1 && cu_seqlens.is_contiguous(), "cu_seqlens: int32 vector");
  TORCH_CHECK(num_accepted.scalar_type() == at::kInt && num_accepted.dim() == 1 && num_accepted.is_contiguous(), "num_accepted: int32 vector");
  const auto max_int = std::numeric_limits<int>::max();
  TORCH_CHECK(state_indices.size(0) < max_int && state.size(0) <= max_int &&
              state.size(1) <= 65535 && mixed_qkv.size(0) <= max_int,
              "K5 dimensions exceed int32 metadata range");
  const int N = state_indices.size(0);
  const int W = state_indices.size(1);
  TORCH_CHECK(cu_seqlens.numel() >= N + 1 && num_accepted.numel() >= N, "cu_seqlens/num_accepted too short");
  TORCH_CHECK(W >= 1 && W <= kMaxTok, "state_indices width must be 1..8");
  const int HV = state.size(1);
  TORCH_CHECK(HV > 0 && state.size(0) > 0 && state.stride(0) >= (int64_t)HV * kDimV * kDimK,
              "state: positive, non-overlapping slots and heads");
  const int64_t kw = mixed_qkv.size(1) - (int64_t)HV * kDimV;
  TORCH_CHECK(kw > 0 && kw % (2 * kDimK) == 0 && kw / (2 * kDimK) <= max_int,
              "mixed_qkv width inconsistent with state heads or int32 range");
  const int H = kw / (2 * kDimK);
  TORCH_CHECK(HV % H == 0, "HV % H");
  TORCH_CHECK(a.size(1) == HV && b.size(1) == HV && z.size(1) == HV && out.size(1) == HV, "head count mismatch");
  const int L = mixed_qkv.size(0);
  TORCH_CHECK(a.size(0) == L && b.size(0) == L && z.size(0) == L && out.size(0) == L,
              "a/b/z/out token counts must match mixed_qkv");
  TORCH_CHECK(a.stride(0) >= HV && b.stride(0) >= HV &&
              z.stride(0) >= (int64_t)HV * kDimV && out.stride(0) >= (int64_t)HV * kDimV,
              "a/b/z/out rows must not overlap");
  TORCH_CHECK(a_log.numel() == HV && dt_bias.numel() == HV, "A_log/dt_bias head count mismatch");
  TORCH_CHECK(HV / H == 1 || HV / H == 2 || HV / H == 3 || HV / H == 4 || HV / H == 8,
              "K5 HV/H ratio unsupported");
  if (N == 0) return;
  const Strides st{mixed_qkv.stride(0), a.stride(0), b.stride(0), z.stride(0), state.stride(0), out.stride(0)};
  const at::cuda::OptionalCUDAGuard guard(mixed_qkv.device());
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  const dim3 grid(N, HV);
  const auto* mp = reinterpret_cast<const __half*>(mixed_qkv.data_ptr());
  const auto* ap = reinterpret_cast<const __half*>(a.data_ptr());
  const auto* bp = reinterpret_cast<const __half*>(b.data_ptr());
  const auto* zp = reinterpret_cast<const __half*>(z.data_ptr());
  auto* op = reinterpret_cast<__half*>(out.data_ptr());
  const int v = (int)variant;
  const bool dt_f32 = dt_bias.scalar_type() == at::kFloat;
  const bool nw_f32 = norm_w.scalar_type() == at::kFloat;
#define K5_L(S, SIG)                                                                                              \
  launch_one<S, SIG>(v, grid, s, mp, ap, bp, a_log.data_ptr<float>(), dt_bias.data_ptr(),                         \
                     state_indices.data_ptr<int>(), cu_seqlens.data_ptr<int>(), num_accepted.data_ptr<int>(),      \
                     reinterpret_cast<S*>(state.data_ptr()), zp, norm_w.data_ptr(), op, H, HV, HV / H, W,          \
                     L, (int)state.size(0), (int)null_block_id, (float)scale, (float)eps, st, dt_f32, nw_f32)
  if (state.scalar_type() == at::kFloat) {
    if (sigmoid_gate) K5_L(float, true); else K5_L(float, false);
  } else {
    if (sigmoid_gate) K5_L(__half, true); else K5_L(__half, false);
  }
#undef K5_L
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gdn_mtp", &gdn_mtp_sm75, "K5 fused GDN post-conv MTP decode (sm_75)"); }
