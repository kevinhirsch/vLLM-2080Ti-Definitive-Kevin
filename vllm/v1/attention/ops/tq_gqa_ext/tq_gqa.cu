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

// ---- lane E04 (2026-10-13): compact shared-memory layout -> 3 resident CTAs/SM instead of 2 -------------------------
// tq_gqa_stage1 (above, unchanged) needs 28192 B dynamic smem/CTA. NCU of S2 (28K ctx, grid 512): the 64 KiB carveout is
// ALREADY selected (Shared Memory Configuration Size 65.54 KB, driver smem/block 0) and shared memory is the binding
// limit (Block Limit Shared Mem 2, Registers 3, Warps 4) -> 50% theoretical occupancy, 59.6% of cycles with no eligible
// warp, 3.76 waves.  So a carveout preference cannot help; a smaller footprint can.
// tq_gqa_stage1_c3 is a copy of tq_gqa_stage1 with ONLY the shared-memory layout changed (21600 B; 3 x 21760 granule-
// rounded = 65280 <= 65536):
//   * Q_s / K_s rows unpadded (256 halves) with a 16-byte-chunk XOR swizzle (chunk ^ (row & 7)): mma fragment loads
//     stay conflict-free (as with the padded layout); 128-bit K stores stay full-row permutations.
//   * V_t rows 24 halves (was 26) with a token-pair XOR swizzle (pair ^ ((d >> 3) & 7)): transposed V writes become
//     conflict-free (were 2-way) and PV fragment loads stay conflict-free.
//   * S_s (4 k-quarter partial scores, 16 floats/row, column ^ 8 when (row >> 1) & 1) ALIASES the unused tail of the
//     K_s/V_t buffer: K_s occupies bytes [0, 8192) of the 12288-byte buffer during the score phase, S_s [8192, 12288).
//     S_s is dead (softmax read it, then __syncthreads) before V decode overwrites the buffer with V_t, and the next
//     chunk's scores are written only after the K-decode __syncthreads, i.e. after PV has finished reading V_t.
// Every arithmetic op and the summation order are identical -> bit-identical Mid / out / lse (GPU test:
// tools/e04/e04_numerics.py).  __launch_bounds__(256, 3) caps registers at 80 = what the 2-CTA build already uses
// (ptxas: 80 regs, 0 spill for both).  Opt-in: VLLM_TQ_GQA_C3=1 (Python wrapper selects decode_c3).
#define C3_Q_OFF 0
#define C3_KV_OFF (16 * 256 * 2)
#define C3_KV_BYTES (HD * 24 * 2)                        /* max(V_t 256x24, K_s 16x256) halves * 2 */
#define C3_S_OFF (C3_KV_OFF + CH * 256 * 2)              /* tail of the K_s/V_t buffer */
#define C3_P_OFF (C3_KV_OFF + C3_KV_BYTES)
#define C3_SMEM_BYTES (C3_P_OFF + 16 * LDP * 2 + CH * 8 + 16 * 4 * 3 + 8 * 4)
static_assert(HD * 24 >= CH * 256, "C3 V_t is the larger tenant of the K/V buffer");
static_assert(C3_S_OFF + 4 * 16 * 16 * 4 <= C3_P_OFF, "C3 S_s alias must fit after K_s");
static_assert(C3_SMEM_BYTES <= 21760, "C3 layout must fit 3 CTAs in 64 KiB (256-byte allocation granules)");

// half index of (row, col) in Q_s / K_s; col even, an access never crosses an 8-half (16-byte) chunk
__device__ __forceinline__ int c3_qk(int row, int col) { return row * 256 + (((col >> 3) ^ (row & 7)) << 3) + (col & 7); }
// half index of V_t[d][tk], tk even (token pair)
__device__ __forceinline__ int c3_v(int d, int tk) { return d * 24 + (((tk >> 1) ^ ((d >> 3) & 7)) << 1); }
// float index of S_s[kq][row][col]
__device__ __forceinline__ int c3_s(int kq, int row, int col) { return kq * 256 + row * 16 + (col ^ (((row >> 1) & 1) << 3)); }

__global__ void __launch_bounds__(NTHR, 3) tq_gqa_stage1_c3(
    const float* __restrict__ Q, const uint8_t* __restrict__ KV, const int* __restrict__ BT,
    const int* __restrict__ SL, const float* __restrict__ Cent, float* __restrict__ Mid,
    long stride_qr, long stride_qh, long stride_cb, long stride_cp, long stride_ch, long stride_btb,
    long stride_mr, long stride_mh, long stride_ms, int QL, int QS, int G, int block_size, int NS,
    float attn_scale, int norm_corr, int nbt) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __half* Q_s = reinterpret_cast<__half*>(smem_raw + C3_Q_OFF);           // [16][256], chunk-swizzled
  __half* KV_s = reinterpret_cast<__half*>(smem_raw + C3_KV_OFF);         // K_s [CH][256] swz | V_t [HD][24] swz
  float* S_s = reinterpret_cast<float*>(smem_raw + C3_S_OFF);             // [4][16][16] swz, aliases K_s/V_t tail
  __half* P_s = reinterpret_cast<__half*>(smem_raw + C3_P_OFF);           // [16][LDP]
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
      __half2* dst = reinterpret_cast<__half2*>(Q_s + c3_qk(m, c4 * 4));
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
      *reinterpret_cast<uint4*>(KV_s + c3_qk(tk, 8 * lane)) = outv;
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
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&Q_s[c3_qk(g, k0 + 2 * t4)]);
        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&Q_s[c3_qk(g + 8, k0 + 2 * t4)]);
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&KV_s[c3_qk(nt * 8 + g, k0 + 2 * t4)]);
        mma_16816(cc, a0, a1, b0);
      }
      // 64-bit stores: conflict-free per half-warp with the column swizzle
      *reinterpret_cast<float2*>(S_s + c3_s(kq, g, nt * 8 + 2 * t4)) = make_float2(cc[0], cc[1]);
      *reinterpret_cast<float2*>(S_s + c3_s(kq, g + 8, nt * 8 + 2 * t4)) = make_float2(cc[2], cc[3]);
    }
    __syncthreads();

    // ---- online softmax (16 rows x 16 tokens, one element per thread)
    {
      float s = S_s[c3_s(0, srow, scol)] + S_s[c3_s(1, srow, scol)] + S_s[c3_s(2, srow, scol)] +
                S_s[c3_s(3, srow, scol)];
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
        *reinterpret_cast<__half2*>(Vt + c3_v(2 * j, tk)) = v_lo;
        *reinterpret_cast<__half2*>(Vt + c3_v(2 * j + 1, tk)) = v_hi;
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
          const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&KV_s[c3_v(d, ks * 8 + 2 * t4)]);
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

// lane E04: compact 3-CTA/SM stage-1 layout (bit-identical results); stage 2 unchanged.
void tq_gqa_decode_c3(torch::Tensor q_rot, torch::Tensor kv, torch::Tensor bt, torch::Tensor seq_lens,
                   torch::Tensor cent, torch::Tensor mid, torch::Tensor out, torch::Tensor lse, int64_t QL,
                   int64_t QS, int64_t G, int64_t block_size, int64_t NS, double scale, int64_t norm_corr) {
  const c10::cuda::CUDAGuard guard(q_rot.device());
  const int S = bt.size(0), Hk = kv.size(2), Hq = q_rot.size(1), R = q_rot.size(0);
  const int NQG = (QL + QS - 1) / QS;
  const size_t smem = C3_SMEM_BYTES;
  auto stream = at::cuda::getCurrentCUDAStream();
  // Hint the max-shared carveout once per process (each TP worker owns one device). The driver default already
  // picks the occupancy-maximizing 64 KiB configuration (NCU: 65.54 KB for the 2-CTA kernel); this only removes
  // doubt. cudaFuncSetAttribute is not a stream operation; failure is non-fatal (the default config fits 3 CTAs).
  static bool carveout_hinted = false;
  if (!carveout_hinted) {
    carveout_hinted = true;
    if (cudaFuncSetAttribute(tq_gqa_stage1_c3, cudaFuncAttributePreferredSharedMemoryCarveout,
                             (int)cudaSharedmemCarveoutMaxShared) != cudaSuccess)
      (void)cudaGetLastError();
  }
  dim3 grid(S, Hk * NQG, NS);
  tq_gqa_stage1_c3<<<grid, NTHR, smem, stream>>>(
      q_rot.data_ptr<float>(), kv.data_ptr<uint8_t>(), bt.data_ptr<int>(), seq_lens.data_ptr<int>(),
      cent.data_ptr<float>(), mid.data_ptr<float>(), q_rot.stride(0), q_rot.stride(1), kv.stride(0), kv.stride(1),
      kv.stride(2), bt.stride(0), mid.stride(0), mid.stride(1), mid.stride(2), (int)QL, (int)QS, (int)G,
      (int)block_size, (int)NS, (float)scale, (int)norm_corr, (int)bt.size(1));
  tq_gqa_stage2<<<R * Hq, HD, NS * sizeof(float), stream>>>(
      mid.data_ptr<float>(), seq_lens.data_ptr<int>(), reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
      lse.data_ptr<float>(), Hq, (int)QL, (int)NS, mid.stride(0), mid.stride(1), mid.stride(2));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode", &tq_gqa_decode, "tq gqa decode");
  m.def("decode_c3", &tq_gqa_decode_c3, "tq gqa decode, compact 3-CTA/SM stage-1 layout (lane E04, opt-in)");
}
