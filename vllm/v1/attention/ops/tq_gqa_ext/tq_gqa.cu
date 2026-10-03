// TurboQuant (k3v4_nc, head_dim 256) GQA/shared-prefix grouped decode attention for sm_75. Lane S2, 2026-10-02.
// One CTA = (sequence, kv head, query-row group, KV split).  Each cached token is dequantized ONCE per CTA and
// scored against up to 16 query rows (speculative rows x GQA group) with mma.m16n8k8 tensor cores.
// Output layout == stock Triton stage1/stage2 (mid_o [R, Hq, NS, D+1], then reduced to out [R,Hq,D] + lse [R,Hq]).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define HD 256
#define MSE_BYTES 96
#define KPS 98
#define VAL_BYTES 128
#define CH 16
#define NTHR 256
#define LDQ 264
#define LDK 264
#define LDV 26
#define LDS 20
#define LDP 24

__device__ __forceinline__ void mma_16816(float* c, uint32_t a0, uint32_t a1, uint32_t b0) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(b0));
}


struct Pref {
  uint32_t kb[2][3];       // K MSE bytes for this warp's 2 tokens (3 bytes per lane)
  unsigned short nrm[2];   // K norms
  unsigned char vb[CH / 2];  // V bytes for this thread's 8 tokens
  uint32_t vs, vz;         // V scale/zero raw fp16 bits (tid < CH only)
};

// Per-chunk page math done ONCE (the chunk's 16 positions span at most two attention pages).
struct ChunkAddr {
  long base0, base1m;  // slot(o) = (o < bs ? base0 : base1m) + o * cp   (base1m is pre-shifted by -bs*cp)
  int off0;
};

__device__ __forceinline__ ChunkAddr chunk_addr(const int* __restrict__ BT, long btrow, int nbt, int p0, int block_size,
                                                long stride_cb, long stride_cp, long koff) {
  ChunkAddr A;
  const int page0 = p0 / block_size;
  A.off0 = p0 - page0 * block_size;
  A.base0 = (long)BT[btrow + page0] * stride_cb + koff;
  const long b1 = (page0 + 1 < nbt) ? (long)BT[btrow + page0 + 1] * stride_cb + koff : A.base0;
  A.base1m = b1 - (long)block_size * stride_cp;
  return A;
}

__device__ __forceinline__ long tok_slot(const ChunkAddr& A, int tk, int block_size, int cp32) {
  const int o = A.off0 + tk;
  return ((o < block_size) ? A.base0 : A.base1m) + (long)(o * cp32);
}

// Issue every global load a chunk needs at once (K bytes, norms, V bytes, V meta); no dependent smem round trips.
// Invalid tokens (tk >= nvalid) load nothing and stay zero => they decode to all-zero K/V rows with zero V meta.
__device__ __forceinline__ void load_chunk(Pref& R, const uint8_t* __restrict__ KV, const ChunkAddr& A, int nvalid,
                                           int block_size, int cp32, int tid, int lane, int warp) {
  const int j = tid & 127, th = tid >> 7;
  const bool cross = (A.off0 + CH > block_size);  // warp-uniform: chunk straddles two attention pages (rare)
  R.vs = 0u;
  R.vz = 0u;
  if (!cross) {
    const uint8_t* base = KV + A.base0 + (long)A.off0 * cp32;
#pragma unroll
    for (int r = 0; r < 2; r++) {
      const int tk = warp + 8 * r;
      R.kb[r][0] = R.kb[r][1] = R.kb[r][2] = 0;
      R.nrm[r] = 0;
      if (tk < nvalid) {
        const uint8_t* kp = base + (long)tk * cp32;
        R.kb[r][0] = kp[3 * lane];
        R.kb[r][1] = kp[3 * lane + 1];
        R.kb[r][2] = kp[3 * lane + 2];
        R.nrm[r] = *reinterpret_cast<const unsigned short*>(kp + MSE_BYTES);
      }
    }
    const uint8_t* vp = base + (long)(th * (CH / 2)) * cp32 + KPS + j;
#pragma unroll
    for (int i = 0; i < CH / 2; i++) {
      R.vb[i] = (th * (CH / 2) + i < nvalid) ? vp[(long)i * cp32] : 0;
    }
    if (tid < CH && tid < nvalid) {
      const uint8_t* mp = base + (long)tid * cp32 + KPS + VAL_BYTES;
      R.vs = *reinterpret_cast<const unsigned short*>(mp);
      R.vz = *reinterpret_cast<const unsigned short*>(mp + 2);
    }
  } else {
#pragma unroll
    for (int r = 0; r < 2; r++) {
      const int tk = warp + 8 * r;
      R.kb[r][0] = R.kb[r][1] = R.kb[r][2] = 0;
      R.nrm[r] = 0;
      if (tk < nvalid) {
        const uint8_t* kp = KV + tok_slot(A, tk, block_size, cp32);
        R.kb[r][0] = kp[3 * lane];
        R.kb[r][1] = kp[3 * lane + 1];
        R.kb[r][2] = kp[3 * lane + 2];
        R.nrm[r] = *reinterpret_cast<const unsigned short*>(kp + MSE_BYTES);
      }
    }
#pragma unroll
    for (int i = 0; i < CH / 2; i++) {
      const int tk = th * (CH / 2) + i;
      R.vb[i] = (tk < nvalid) ? KV[tok_slot(A, tk, block_size, cp32) + KPS + j] : 0;
    }
    if (tid < CH && tid < nvalid) {
      const uint8_t* mp = KV + tok_slot(A, tid, block_size, cp32) + KPS + VAL_BYTES;
      R.vs = *reinterpret_cast<const unsigned short*>(mp);
      R.vz = *reinterpret_cast<const unsigned short*>(mp + 2);
    }
  }
}

__global__ void __launch_bounds__(NTHR, 2) tq_gqa_stage1(
    const float* __restrict__ Q, const uint8_t* __restrict__ KV, const int* __restrict__ BT,
    const int* __restrict__ SL, const float* __restrict__ Cent, float* __restrict__ Mid,
    long stride_qr, long stride_qh, long stride_cb, long stride_cp, long stride_ch, long stride_btb,
    long stride_mr, long stride_mh, long stride_ms, int QL, int QS, int G, int block_size, int NS,
    float attn_scale, int norm_corr, int nbt) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __half* Q_s = reinterpret_cast<__half*>(smem_raw);                       // [16][LDQ]
  __half* KV_s = Q_s + 16 * LDQ;                                           // K_s [CH][LDK] | V_t [HD][LDV]
  const int kv_elems = (HD * LDV > CH * LDK) ? HD * LDV : CH * LDK;
  float* S_s = reinterpret_cast<float*>(KV_s + kv_elems);                  // [4][16][LDS]
  __half* P_s = reinterpret_cast<__half*>(S_s + 4 * 16 * LDS);             // [16][LDP]
  float* vs_s = reinterpret_cast<float*>(P_s + 16 * LDP);                  // [CH]
  float* vz_s = vs_s + CH;                                                 // [CH]  (vs_s/vz_s hold half2 bit patterns)
  float* alpha_s = vz_s + CH;                                              // [16]
  float* l_s = alpha_s + 16;                                               // [16]
  float* m_s = l_s + 16;                                                   // [16]
  float* cent_s = m_s + 16;                                                // [8]

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const int seq = blockIdx.x;
  const int NQG = (QL + QS - 1) / QS;
  const int kvh = blockIdx.y / NQG, qg = blockIdx.y % NQG;
  const int sid = blockIdx.z;

  const int L = SL[seq];
  const int split_len = (L + NS - 1) / NS;
  const int t0 = split_len * sid;
  const int t1 = min(t0 + split_len, L);
  if (t0 >= t1) return;

  if (tid < 8) cent_s[tid] = Cent[tid];
  // ---- stage Q (rotated, fp32 -> fp16) for this CTA's 16 rows, float4 loads
  {
    const int c4 = tid & 63, m0 = tid >> 6;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int m = m0 + 4 * i;
      const int qi = qg * QS + m / G;
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if (m < QS * G && qi < QL)
        v = *reinterpret_cast<const float4*>(Q + (long)(seq * QL + qi) * stride_qr + (long)(kvh * G + (m % G)) * stride_qh + c4 * 4);
      __half2* dst = reinterpret_cast<__half2*>(Q_s + m * LDQ + c4 * 4);
      dst[0] = __floats2half2_rn(v.x, v.y);
      dst[1] = __floats2half2_rn(v.z, v.w);
    }
  }

  float O[4][4];
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) O[i][j] = 0.f;
  float m_run = -INFINITY, l_run = 0.f;
  const int srow = tid >> 4, scol = tid & 15;  // softmax mapping

  const long btrow = (long)seq * stride_btb;
  const long koff = (long)kvh * stride_ch;
  Pref cur, nxt;
  load_chunk(cur, KV, chunk_addr(BT, btrow, nbt, t0, block_size, stride_cb, stride_cp, koff), min(CH, t1 - t0), block_size, (int)stride_cp, tid, lane, warp);

  for (int c0 = t0; c0 < t1; c0 += CH) {
    const int nvalid = min(CH, t1 - c0);
    __syncthreads();  // previous chunk fully consumed (V_t / P_s / vs_s reuse)
    if (tid < CH) {
      reinterpret_cast<unsigned short*>(vs_s)[tid] = (unsigned short)cur.vs;
      reinterpret_cast<unsigned short*>(vz_s)[tid] = (unsigned short)cur.vz;
    }

    // ---- K decode from prefetched registers: warp w -> tokens w, w+8 ; lane -> dims 8*lane..8*lane+7
#pragma unroll
    for (int r = 0; r < 2; r++) {
      const int tk = warp + 8 * r;
      uint4 outv = make_uint4(0, 0, 0, 0);
      {
        const uint32_t v = cur.kb[r][0] | (cur.kb[r][1] << 8) | (cur.kb[r][2] << 16);
        float c[8];
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
          c[i] = cent_s[(v >> (3 * i)) & 7];
          ss += c[i] * c[i];
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float vnorm = __half2float(__ushort_as_half(cur.nrm[r]));
        const float f = vnorm * attn_scale * (norm_corr ? rsqrtf(ss + 1e-16f) : 1.f);
        __half2 h0 = __floats2half2_rn(c[0] * f, c[1] * f), h1 = __floats2half2_rn(c[2] * f, c[3] * f);
        __half2 h2 = __floats2half2_rn(c[4] * f, c[5] * f), h3 = __floats2half2_rn(c[6] * f, c[7] * f);
        outv.x = *reinterpret_cast<uint32_t*>(&h0);
        outv.y = *reinterpret_cast<uint32_t*>(&h1);
        outv.z = *reinterpret_cast<uint32_t*>(&h2);
        outv.w = *reinterpret_cast<uint32_t*>(&h3);
      }
      *reinterpret_cast<uint4*>(KV_s + tk * LDK + 8 * lane) = outv;
    }
    // prefetch the NEXT chunk now: its latency hides behind mma / softmax / V decode / PV below
    if (c0 + CH < t1)
      load_chunk(nxt, KV, chunk_addr(BT, btrow, nbt, c0 + CH, block_size, stride_cb, stride_cp, koff), min(CH, t1 - c0 - CH), block_size, (int)stride_cp, tid, lane, warp);
    __syncthreads();

    // ---- scores: warp -> (token tile nt, k quarter kq)
    {
      const int nt = warp & 1, kq = warp >> 1;
      float cc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int ks = 0; ks < 8; ks++) {
        const int k0 = kq * 64 + ks * 8;
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&Q_s[g * LDQ + k0 + 2 * t4]);
        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&Q_s[(g + 8) * LDQ + k0 + 2 * t4]);
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&KV_s[(nt * 8 + g) * LDK + k0 + 2 * t4]);
        mma_16816(cc, a0, a1, b0);
      }
      float* sp = S_s + kq * 16 * LDS;
      sp[g * LDS + nt * 8 + 2 * t4] = cc[0];
      sp[g * LDS + nt * 8 + 2 * t4 + 1] = cc[1];
      sp[(g + 8) * LDS + nt * 8 + 2 * t4] = cc[2];
      sp[(g + 8) * LDS + nt * 8 + 2 * t4 + 1] = cc[3];
    }
    __syncthreads();

    // ---- online softmax (16 rows x 16 tokens, one element per thread)
    {
      float s = S_s[0 * 16 * LDS + srow * LDS + scol] + S_s[1 * 16 * LDS + srow * LDS + scol] +
                S_s[2 * 16 * LDS + srow * LDS + scol] + S_s[3 * 16 * LDS + srow * LDS + scol];
      if (scol >= nvalid) s = -INFINITY;
      float cmax = s;
#pragma unroll
      for (int o = 8; o > 0; o >>= 1) cmax = fmaxf(cmax, __shfl_xor_sync(0xffffffffu, cmax, o));
      const float m_new = fmaxf(m_run, cmax);
      const float alpha = expf(m_run - m_new);
      const float p = (scol < nvalid) ? expf(s - m_new) : 0.f;
      float ps = p;
#pragma unroll
      for (int o = 8; o > 0; o >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, o);
      l_run = l_run * alpha + ps;
      m_run = m_new;
      P_s[srow * LDP + scol] = __float2half(p);
      if (scol == 0) alpha_s[srow] = alpha;
    }
    __syncthreads();

    // ---- V decode from prefetched registers (transposed into V_t[d][token]); token PAIRS packed as half2
    {
      __half* Vt = KV_s;
      const int j = tid & 127, th = tid >> 7;
      const __half2 k1024 = __float2half2_rn(1024.f);
      const unsigned short* vsh = reinterpret_cast<const unsigned short*>(vs_s);
      const unsigned short* vzh = reinterpret_cast<const unsigned short*>(vz_s);
#pragma unroll
      for (int q = 0; q < CH / 4; q++) {
        const int tk = th * (CH / 2) + 2 * q;
        const uint32_t x = cur.vb[2 * q] | ((uint32_t)cur.vb[2 * q + 1] << 16);
        const uint32_t lo2 = (x & 0x000F000Fu) | 0x64006400u;
        const uint32_t hi2 = ((x >> 4) & 0x000F000Fu) | 0x64006400u;
        const uint32_t vs2 = *reinterpret_cast<const uint32_t*>(vsh + tk), vz2 = *reinterpret_cast<const uint32_t*>(vzh + tk);
        const __half2 vsv = *reinterpret_cast<const __half2*>(&vs2), vzv = *reinterpret_cast<const __half2*>(&vz2);
        const __half2 v_lo = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&lo2), k1024), vsv, vzv);
        const __half2 v_hi = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&hi2), k1024), vsv, vzv);
        *reinterpret_cast<__half2*>(Vt + (2 * j) * LDV + tk) = v_lo;
        *reinterpret_cast<__half2*>(Vt + (2 * j + 1) * LDV + tk) = v_hi;
      }
    }
    __syncthreads();

    // ---- rescale O, then O += P @ V  (warp owns 4 d-tiles = 32 dims)
    {
      const float ag = alpha_s[g], ag8 = alpha_s[g + 8];
#pragma unroll
      for (int dt = 0; dt < 4; dt++) {
        O[dt][0] *= ag;
        O[dt][1] *= ag;
        O[dt][2] *= ag8;
        O[dt][3] *= ag8;
      }
#pragma unroll
      for (int ks = 0; ks < CH / 8; ks++) {
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&P_s[g * LDP + ks * 8 + 2 * t4]);
        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&P_s[(g + 8) * LDP + ks * 8 + 2 * t4]);
#pragma unroll
        for (int dt = 0; dt < 4; dt++) {
          const int d = (warp * 4 + dt) * 8 + g;
          const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&KV_s[d * LDV + ks * 8 + 2 * t4]);
          mma_16816(O[dt], a0, a1, b0);
        }
      }
    }
    cur = nxt;
  }

  // (cur = nxt handled at loop tail)
  // ---- epilogue
  if (scol == 0) {
    l_s[srow] = l_run;
    m_s[srow] = m_run;
  }
  __syncthreads();
#pragma unroll
  for (int half_i = 0; half_i < 2; half_i++) {
    const int m = g + 8 * half_i;
    const int qi = qg * QS + m / G;
    if (m < QS * G && qi < QL) {
      const float l = l_s[m] > 0.f ? l_s[m] : 1.f;
      const long base = (long)(seq * QL + qi) * stride_mr + (long)(kvh * G + (m % G)) * stride_mh + (long)sid * stride_ms;
#pragma unroll
      for (int dt = 0; dt < 4; dt++) {
        const int d = (warp * 4 + dt) * 8 + 2 * t4;
        Mid[base + d] = O[dt][2 * half_i] / l;
        Mid[base + d + 1] = O[dt][2 * half_i + 1] / l;
      }
      if (warp == 0 && t4 == 0) Mid[base + HD] = m_s[m] + logf(l);
    }
  }
}

// Reduce over KV splits. grid (R*Hq), block HD threads (thread = dim).
__global__ void tq_gqa_stage2(const float* __restrict__ Mid, const int* __restrict__ SL, __half* __restrict__ Out,
                              float* __restrict__ Lse, int Hq, int QL, int NS, long stride_mr, long stride_mh,
                              long stride_ms) {
  extern __shared__ float w_s[];  // [NS]
  const int r = blockIdx.x / Hq, h = blockIdx.x % Hq;
  const int d = threadIdx.x;
  const int L = SL[r / QL];
  const int split_len = (L + NS - 1) / NS;
  const float* base = Mid + (long)r * stride_mr + (long)h * stride_mh;
  float mx = -INFINITY;
  for (int s = d; s < NS; s += HD) {
    const bool valid = (long)split_len * s < L;
    const float lse = valid ? base[(long)s * stride_ms + HD] : -INFINITY;
    w_s[s] = lse;
  }
  __syncthreads();
  __shared__ float red[HD / 32];
  float lm = -INFINITY;
  for (int s = d; s < NS; s += HD) lm = fmaxf(lm, w_s[s]);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) lm = fmaxf(lm, __shfl_xor_sync(0xffffffffu, lm, o));
  if ((d & 31) == 0) red[d >> 5] = lm;
  __syncthreads();
  mx = red[0];
#pragma unroll
  for (int i = 1; i < HD / 32; i++) mx = fmaxf(mx, red[i]);
  __syncthreads();
  for (int s = d; s < NS; s += HD) w_s[s] = (w_s[s] == -INFINITY) ? 0.f : expf(w_s[s] - mx);
  __syncthreads();
  float acc = 0.f, sum = 0.f;
  int s = 0;
  for (; s + 4 <= NS; s += 4) {
    const float w0 = w_s[s], w1 = w_s[s + 1], w2 = w_s[s + 2], w3 = w_s[s + 3];
    const float v0 = w0 > 0.f ? base[(long)s * stride_ms + d] : 0.f;
    const float v1 = w1 > 0.f ? base[(long)(s + 1) * stride_ms + d] : 0.f;
    const float v2 = w2 > 0.f ? base[(long)(s + 2) * stride_ms + d] : 0.f;
    const float v3 = w3 > 0.f ? base[(long)(s + 3) * stride_ms + d] : 0.f;
    acc += w0 * v0 + w1 * v1 + w2 * v2 + w3 * v3;
    sum += w0 + w1 + w2 + w3;
  }
  for (; s < NS; s++) {
    const float w = w_s[s];
    if (w > 0.f) acc += w * base[(long)s * stride_ms + d];
    sum += w;
  }
  Out[(long)r * Hq * HD + (long)h * HD + d] = __float2half(acc / sum);
  if (d == 0) Lse[(long)r * Hq + h] = mx + logf(sum);
}

void tq_gqa_decode(torch::Tensor q_rot, torch::Tensor kv, torch::Tensor bt, torch::Tensor seq_lens,
                   torch::Tensor cent, torch::Tensor mid, torch::Tensor out, torch::Tensor lse, int64_t QL,
                   int64_t QS, int64_t G, int64_t block_size, int64_t NS, double scale, int64_t norm_corr) {
  const c10::cuda::CUDAGuard guard(q_rot.device());
  const int S = bt.size(0), Hk = kv.size(2), Hq = q_rot.size(1), R = q_rot.size(0);
  const int NQG = (QL + QS - 1) / QS;
  const size_t kv_elems = (HD * LDV > CH * LDK) ? HD * LDV : CH * LDK;
  const size_t smem = 16 * LDQ * 2 + kv_elems * 2 + 4 * 16 * LDS * 4 + 16 * LDP * 2 + CH * 8 + CH * 4 * 2 + 16 * 4 * 3 + 8 * 4 + 64;
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(S, Hk * NQG, NS);
  tq_gqa_stage1<<<grid, NTHR, smem, stream>>>(
      q_rot.data_ptr<float>(), kv.data_ptr<uint8_t>(), bt.data_ptr<int>(), seq_lens.data_ptr<int>(),
      cent.data_ptr<float>(), mid.data_ptr<float>(), q_rot.stride(0), q_rot.stride(1), kv.stride(0), kv.stride(1),
      kv.stride(2), bt.stride(0), mid.stride(0), mid.stride(1), mid.stride(2), (int)QL, (int)QS, (int)G,
      (int)block_size, (int)NS, (float)scale, (int)norm_corr, (int)bt.size(1));
  tq_gqa_stage2<<<R * Hq, HD, NS * sizeof(float), stream>>>(
      mid.data_ptr<float>(), seq_lens.data_ptr<int>(), reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
      lse.data_ptr<float>(), Hq, (int)QL, (int)NS, mid.stride(0), mid.stride(1), mid.stride(2));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("decode", &tq_gqa_decode, "tq gqa decode"); }
