// tq_imma kernels (torch-free; included by tq_imma.cu and by the standalone K2 bench). Lane K2, 2026-10-03.
#pragma once
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

#define HD 256
#define MSE_BYTES 96
#define KPS 98
#define T 64       // tokens per chunk
#define NTHR 256   // 8 warps
#define LDS_S 72   // fp32 score row stride (floats)
#define LDP 80     // u8 P row stride (bytes)
#ifdef TQ_IMMA_PROF
__device__ unsigned long long tq_prof[10];
__device__ unsigned int tq_sink;
#define TQP(i) do { if (tid == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { long long _n = clock64(); tq_prof[i] += _n - _pt; _pt = _n; } } while (0)
#else
#define TQP(i) do {} while (0)
#endif

__device__ __forceinline__ void imma_s8(int* c, uint32_t a, uint32_t b) {
  asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
               : "+r"(c[0]), "+r"(c[1]) : "r"(a), "r"(b));
}
__device__ __forceinline__ void imma_u8(int* c, uint32_t a, uint32_t b) {
  asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.u8.u8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
               : "+r"(c[0]), "+r"(c[1]) : "r"(a), "r"(b));
}

// 8 packed 3-bit codes (24 bits) -> 8 nibbles (32 bits): byte_perm selectors.
__device__ __forceinline__ uint32_t spread3(uint32_t x) {
  uint32_t y = (x & 0x000FFFu) | ((x & 0xFFF000u) << 4);         // fields 0-3 @0..11, 4-7 @16..27
  y = (y & 0x003F003Fu) | ((y & 0x0FC00FC0u) << 2);              // per 16b: fields (0,1) @0..5, (2,3) @8..13
  y = (y & 0x07070707u) | ((y & 0x38383838u) << 1);              // per byte: fields @0..2 and @4..6
  return y;
}

// 16-byte chunk swizzle for [row][256 B] int8 tiles read as u128 by (g, t4) lanes: odd rows flip chunk bit 2.
__device__ __forceinline__ int kq_off(int row, int chunk) { return row * HD + ((chunk ^ ((row & 1) << 2)) << 4); }
// V tile: [jp 0..63][tq 0..15] u64 (bytes 2jp, 2jp+1 of tokens 4tq..4tq+3)
__device__ __forceinline__ int vt_off(int jp, int tq) { return jp * 16 + (tq ^ (((jp & 3) << 2) | ((jp >> 2) & 3))); }

struct Pref {
  uint32_t k[4][3];   // K: 4 tasks x 3 u16 (6 code bytes = 16 dims)
  uint32_t kn[4];     // K norm (gp==0 tasks only)
  uint32_t v[4][4];   // V: 4 tasks x 4 tokens x u16 (byte pair jp)
  uint32_t meta;      // tid<64: v scale of token tid, 64<=tid<128: v zero of token tid-64
};

struct ChunkAddr {
  long base0, base1m;
  int off0;
};
__device__ __forceinline__ ChunkAddr chunk_addr(const int* __restrict__ BT, long btrow, int nbt, int p0, int bs,
                                                long scb, long scp, long koff) {
  ChunkAddr A;
  const int page0 = p0 / bs;
  A.off0 = p0 - page0 * bs;
  A.base0 = (long)BT[btrow + page0] * scb + koff;
  const long b1 = (page0 + 1 < nbt) ? (long)BT[btrow + page0 + 1] * scb + koff : A.base0;
  A.base1m = b1 - (long)bs * scp;
  return A;
}
__device__ __forceinline__ long tok_slot(const ChunkAddr& A, int tk, int bs, long scp) {
  const int o = A.off0 + tk;
  return ((o < bs) ? A.base0 : A.base1m) + (long)o * scp;
}

__device__ __forceinline__ void load_chunk(Pref& R, const uint8_t* __restrict__ KV, const ChunkAddr& A, int nvalid,
                                           int bs, long scp, int tid) {
#pragma unroll
  for (int i = 0; i < 4; i++) {  // K tasks: tau = tid + 256 i -> token tau>>4, gp tau&15
    const int tau = tid + 256 * i, tk = tau >> 4, gp = tau & 15;
    R.k[i][0] = R.k[i][1] = R.k[i][2] = 0u;
    R.kn[i] = 0u;
    if (tk < nvalid) {
      const unsigned short* p = reinterpret_cast<const unsigned short*>(KV + tok_slot(A, tk, bs, scp)) + 3 * gp;
      R.k[i][0] = p[0];
      R.k[i][1] = p[1];
      R.k[i][2] = p[2];
      if (gp == 0) R.kn[i] = p[MSE_BYTES / 2];
    }
  }
#pragma unroll
  for (int i = 0; i < 4; i++) {  // V tasks: tau = tid + 256 i -> jp tau&63, quad tq tau>>6
    const int tau = tid + 256 * i, jp = tau & 63, tq = tau >> 6;
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const int tk = 4 * tq + u;
      R.v[i][u] = (tk < nvalid) ? (uint32_t)reinterpret_cast<const unsigned short*>(KV + tok_slot(A, tk, bs, scp) + KPS)[jp] : 0u;
    }
  }
  R.meta = 0u;
  if (tid < 128) {
    const int tk = tid & 63;
    if (tk < nvalid) R.meta = reinterpret_cast<const unsigned short*>(KV + tok_slot(A, tk, bs, scp) + KPS + 128)[tid >> 6];
  }
}

template <int MT, bool QSPLIT>
__global__ void __launch_bounds__(NTHR, 1) tq_imma_stage1(
    const int8_t* __restrict__ Q8, const float* __restrict__ QS, const uint8_t* __restrict__ KV,
    const int* __restrict__ BT, const int* __restrict__ RL, float* __restrict__ Mid, long sq8r, long sq8h,
    long scb, long scp, long sch, long sbt, long smr, long smh, long sms, int QL, int G, int bs, int NS,
    float attn_scale, float cscale, int norm_corr, int nbt, uint32_t lut_lo, uint32_t lut_hi) {
  constexpr int MR = MT * 8;
  extern __shared__ __align__(16) unsigned char smem[];
  int8_t* Ks = reinterpret_cast<int8_t*>(smem);                        // [T][256] swizzled
  int8_t* Qs = Ks + T * HD;                                            // [MR][256] swizzled
  int8_t* Qs2 = Qs + MR * HD;                                          // [MR][256] low plane (QSPLIT)
  uint2* Vt = reinterpret_cast<uint2*>(Qs2 + (QSPLIT ? MR * HD : 0));  // [64][16] u64 swizzled
  float* Ss = reinterpret_cast<float*>(Vt + 64 * 16);                  // [MR][LDS_S]
  uint8_t* Ps = reinterpret_cast<uint8_t*>(Ss + MR * LDS_S);           // [MR][LDP]
  float* f_s = reinterpret_cast<float*>(Ps + MR * LDP);                // [T]
  float* vs_s = f_s + T;                                               // [T]
  float* vz_s = vs_s + T;                                              // [T]
  float* qs_s = vz_s + T;                                              // [MR]
  float* al_s = qs_s + MR;                                             // [MR]
  float* sa_s = al_s + MR;                                             // [MR]
  float* zs_s = sa_s + MR;                                             // [MR]
  float* l_s = zs_s + MR;                                              // [MR]
  float* m_s = l_s + MR;                                               // [MR]

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const int seq = blockIdx.x, kvh = blockIdx.y, sid = blockIdx.z;
  const int R = QL * G;

  // ---- split range over the longest row of this sequence
  int Lmax = 0;
  for (int i = 0; i < QL; i++) Lmax = max(Lmax, RL[seq * QL + i]);
  int split_len = (Lmax + NS - 1) / NS;
  split_len = (split_len + T - 1) / T * T;
  const int t0 = split_len * sid;
  const int t1 = min(t0 + split_len, Lmax);

  // ---- softmax-row ownership: row r = tid>>3, 8 lanes per row, lane sub owns tokens sub*8..sub*8+7
  const int sr = tid >> 3, sub = tid & 7;
  const bool srow_ok = sr < R;
  const int s_qi = srow_ok ? sr / G : 0;
  const int s_len = srow_ok ? RL[seq * QL + s_qi] : 0;

  if (t0 >= t1) {  // empty split: lse = -inf for every row
    if (sub == 0 && srow_ok) {
      const long base = (long)(seq * QL + s_qi) * smr + (long)(kvh * G + sr % G) * smh + (long)sid * sms;
      Mid[base + HD] = -INFINITY;
    }
    return;
  }

  // ---- stage int8 Q (pre-quantized by the wrapper) for MR rows
  {
    const int row = tid >> 3, part = tid & 7;  // 32 rows x 8 parts of 32 B
    if (row < MR) {
      uint4 a = make_uint4(0, 0, 0, 0), b = make_uint4(0, 0, 0, 0);
      float sc = 0.f;
      if (row < R) {
        const int qi = row / G, h = row % G;
        const int8_t* src = Q8 + (long)(seq * QL + qi) * sq8r + (long)(kvh * G + h) * sq8h + part * 32;
        a = *reinterpret_cast<const uint4*>(src);
        b = *reinterpret_cast<const uint4*>(src + 16);
        sc = QS[(long)(seq * QL + qi) * (sq8r / sq8h) + (kvh * G + h)];
      }
      *reinterpret_cast<uint4*>(Qs + kq_off(row, 2 * part)) = a;
      *reinterpret_cast<uint4*>(Qs + kq_off(row, 2 * part + 1)) = b;
      if (QSPLIT) {
        uint4 c = make_uint4(0, 0, 0, 0), d = make_uint4(0, 0, 0, 0);
        if (row < R) {
          const int qi = row / G, h = row % G;
          const int8_t* src = Q8 + (long)(seq * QL + qi) * sq8r + (long)(kvh * G + h) * sq8h + HD + part * 32;
          c = *reinterpret_cast<const uint4*>(src);
          d = *reinterpret_cast<const uint4*>(src + 16);
        }
        *reinterpret_cast<uint4*>(Qs2 + kq_off(row, 2 * part)) = c;
        *reinterpret_cast<uint4*>(Qs2 + kq_off(row, 2 * part + 1)) = d;
      }
      if (part == 0) qs_s[row] = sc;
    }
  }

  float O[MT][4][2];
#pragma unroll
  for (int m = 0; m < MT; m++)
#pragma unroll
    for (int n = 0; n < 4; n++) O[m][n][0] = O[m][n][1] = 0.f;
  float m_run = -INFINITY, l_run = 0.f;

  const long btrow = (long)seq * sbt;
  const long koff = (long)kvh * sch;
  Pref cur;
#ifdef TQ_IMMA_PROF
  long long _pt = clock64();
#endif
  load_chunk(cur, KV, chunk_addr(BT, btrow, nbt, t0, bs, scb, scp, koff), min(T, t1 - t0), bs, scp, tid);

  for (int c0 = t0; c0 < t1; c0 += T) {
    const int nvalid = min(T, t1 - c0);
    __syncthreads();  // previous chunk's PV done with Vt / Ps; QK done with Ks
    TQP(0);
#ifdef TQ_IMMA_PROF
    {
      uint32_t x = cur.meta;
#pragma unroll
      for (int i = 0; i < 4; i++) x ^= cur.k[i][0] ^ cur.k[i][1] ^ cur.k[i][2] ^ cur.kn[i] ^ cur.v[i][0] ^ cur.v[i][1] ^ cur.v[i][2] ^ cur.v[i][3];
      if (x == 0x12345679u) tq_sink = x;
    }
    TQP(8);
#endif
    // ---- K: codes -> int8 LUT (byte_perm), ||lut||^2 by dp4a, per-token score factor
    {
      float fac = 0.f;
      int ssum = 0;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int tau = tid + 256 * i, tk = tau >> 4, gp = tau & 15;
        const uint32_t x0 = cur.k[i][0] | (cur.k[i][1] << 16);
        const uint32_t ga = x0 & 0xFFFFFFu, gb = (x0 >> 24) | (cur.k[i][2] << 8);
        const uint32_t sa = spread3(ga), sb = spread3(gb);
        uint4 w;
        w.x = __byte_perm(lut_lo, lut_hi, sa & 0xFFFFu);
        w.y = __byte_perm(lut_lo, lut_hi, sa >> 16);
        w.z = __byte_perm(lut_lo, lut_hi, sb & 0xFFFFu);
        w.w = __byte_perm(lut_lo, lut_hi, sb >> 16);
        *reinterpret_cast<uint4*>(Ks + kq_off(tk, gp)) = w;
        int ss = __dp4a((int)w.x, (int)w.x, 0);
        ss = __dp4a((int)w.y, (int)w.y, ss);
        ss = __dp4a((int)w.z, (int)w.z, ss);
        ss = __dp4a((int)w.w, (int)w.w, ss);
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if (gp == 0) {
          const float nrm = __half2float(__ushort_as_half((unsigned short)cur.kn[i]));
          fac = (tk < nvalid) ? nrm * attn_scale * (norm_corr ? rsqrtf((float)ss + 1e-30f) : cscale) : 0.f;
          f_s[tk] = fac;
        }
        (void)ssum;
      }
    }
    // ---- V: transpose 4 tokens x byte pair into the swizzled u64 tile
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int tau = tid + 256 * i, jp = tau & 63, tq = tau >> 6;
      const uint32_t lo = cur.v[i][0] | (cur.v[i][1] << 16), hi = cur.v[i][2] | (cur.v[i][3] << 16);
      Vt[vt_off(jp, tq)] = make_uint2(__byte_perm(lo, hi, 0x6420), __byte_perm(lo, hi, 0x7531));
    }
    if (tid < 64) vs_s[tid] = __half2float(__ushort_as_half((unsigned short)cur.meta));
    else if (tid < 128) vz_s[tid - 64] = __half2float(__ushort_as_half((unsigned short)cur.meta));
    // prefetch next chunk (hidden behind QK / softmax / PV)
    TQP(9);
    if (c0 + T < t1)
      load_chunk(cur, KV, chunk_addr(BT, btrow, nbt, c0 + T, bs, scb, scp, koff), min(T, t1 - c0 - T), bs, scp, tid);
    TQP(1);
    __syncthreads();
    TQP(2);

    // ---- QK^T on IMMA: warp -> 8 tokens, all MR rows, k = 256 (16 steps)
    {
      const int n0 = warp * 8;
      int acc[MT][2], acc2[MT][2];
#pragma unroll
      for (int m = 0; m < MT; m++) acc[m][0] = acc[m][1] = acc2[m][0] = acc2[m][1] = 0;
#pragma unroll
      for (int sq = 0; sq < 4; sq++) {
        const uint4 b = *reinterpret_cast<const uint4*>(Ks + kq_off(n0 + g, 4 * sq + t4));
#pragma unroll
        for (int m = 0; m < MT; m++) {
          const uint4 a = *reinterpret_cast<const uint4*>(Qs + kq_off(m * 8 + g, 4 * sq + t4));
          imma_s8(acc[m], a.x, b.x);
          imma_s8(acc[m], a.y, b.y);
          imma_s8(acc[m], a.z, b.z);
          imma_s8(acc[m], a.w, b.w);
          if (QSPLIT) {
            const uint4 a2 = *reinterpret_cast<const uint4*>(Qs2 + kq_off(m * 8 + g, 4 * sq + t4));
            imma_s8(acc2[m], a2.x, b.x);
            imma_s8(acc2[m], a2.y, b.y);
            imma_s8(acc2[m], a2.z, b.z);
            imma_s8(acc2[m], a2.w, b.w);
          }
        }
      }
      const float f0 = f_s[n0 + 2 * t4], f1 = f_s[n0 + 2 * t4 + 1];
#pragma unroll
      for (int m = 0; m < MT; m++) {
        const float qsc = qs_s[m * 8 + g];
        *reinterpret_cast<float2*>(Ss + (m * 8 + g) * LDS_S + n0 + 2 * t4) =
            make_float2(((float)acc[m][0] + (QSPLIT ? (float)acc2[m][0] * (1.f / 254.f) : 0.f)) * qsc * f0,
                        ((float)acc[m][1] + (QSPLIT ? (float)acc2[m][1] * (1.f / 254.f) : 0.f)) * qsc * f1);
      }
    }
    TQP(3);
    __syncthreads();
    TQP(4);

    // ---- online softmax; P' = p * v_scale quantized to u8 per (row, chunk)
    {
      float s[8];
      if (sr < MR) {
        const float4 x = *reinterpret_cast<const float4*>(Ss + sr * LDS_S + sub * 8);
        const float4 y = *reinterpret_cast<const float4*>(Ss + sr * LDS_S + sub * 8 + 4);
        s[0] = x.x; s[1] = x.y; s[2] = x.z; s[3] = x.w; s[4] = y.x; s[5] = y.y; s[6] = y.z; s[7] = y.w;
      }
      float cmax = -INFINITY;
#pragma unroll
      for (int k = 0; k < 8; k++) {
        const int tk = sub * 8 + k;
        const bool ok = srow_ok && tk < nvalid && (c0 + tk) < s_len;
        s[k] = ok ? s[k] : -INFINITY;
        cmax = fmaxf(cmax, s[k]);
      }
#pragma unroll
      for (int o = 4; o > 0; o >>= 1) cmax = fmaxf(cmax, __shfl_xor_sync(0xffffffffu, cmax, o));
      const float m_new = fmaxf(m_run, cmax);
      float alpha, a[8], psum = 0.f, zsum = 0.f, amax = 0.f;
      if (m_new == -INFINITY) {
        alpha = 1.f;
#pragma unroll
        for (int k = 0; k < 8; k++) a[k] = 0.f;
      } else {
        alpha = (m_run == -INFINITY) ? 0.f : __expf(m_run - m_new);
#pragma unroll
        for (int k = 0; k < 8; k++) {
          const float p = __expf(s[k] - m_new);  // exp(-inf) = 0 for masked
          const int tk = sub * 8 + k;
          a[k] = p * vs_s[tk];
          psum += p;
          zsum += p * vz_s[tk];
          amax = fmaxf(amax, a[k]);
        }
      }
#pragma unroll
      for (int o = 4; o > 0; o >>= 1) {
        psum += __shfl_xor_sync(0xffffffffu, psum, o);
        zsum += __shfl_xor_sync(0xffffffffu, zsum, o);
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
      }
      l_run = l_run * alpha + psum;
      m_run = m_new;
      const float inv = amax > 0.f ? 255.f / amax : 0.f;
      uint32_t w0 = 0, w1 = 0;
#pragma unroll
      for (int k = 0; k < 4; k++) {
        w0 |= (uint32_t)__float2uint_rn(a[k] * inv) << (8 * k);
        w1 |= (uint32_t)__float2uint_rn(a[k + 4] * inv) << (8 * k);
      }
      if (sr < MR) {
        *reinterpret_cast<uint2*>(Ps + sr * LDP + sub * 8) = make_uint2(w0, w1);
        if (sub == 0) {
          al_s[sr] = alpha;
          sa_s[sr] = amax * (1.f / 255.f);
          zs_s[sr] = zsum;
        }
      }
    }
    TQP(5);
    __syncthreads();
    TQP(6);

    // ---- PV on IMMA: warp -> byte pairs jp0..jp0+7 (dims 4*jp0 .. 4*jp0+31), all MR rows, k = 64 tokens
    {
      const int jp0 = warp * 8;
      int C[MT][4][2];
#pragma unroll
      for (int m = 0; m < MT; m++)
#pragma unroll
        for (int n = 0; n < 4; n++) C[m][n][0] = C[m][n][1] = 0;
#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        const uint2 w = Vt[vt_off(jp0 + g, 4 * kk + t4)];
        const uint32_t b0 = w.x & 0x0F0F0F0Fu, b1 = (w.x >> 4) & 0x0F0F0F0Fu;
        const uint32_t b2 = w.y & 0x0F0F0F0Fu, b3 = (w.y >> 4) & 0x0F0F0F0Fu;
#pragma unroll
        for (int m = 0; m < MT; m++) {
          const uint32_t a = *reinterpret_cast<const uint32_t*>(Ps + (m * 8 + g) * LDP + 16 * kk + 4 * t4);
          imma_u8(C[m][0], a, b0);
          imma_u8(C[m][1], a, b1);
          imma_u8(C[m][2], a, b2);
          imma_u8(C[m][3], a, b3);
        }
      }
#pragma unroll
      for (int m = 0; m < MT; m++) {
        const int row = m * 8 + g;
        const float al = al_s[row], sa = sa_s[row], zs = zs_s[row];
#pragma unroll
        for (int n = 0; n < 4; n++) {
          O[m][n][0] = O[m][n][0] * al + sa * (float)C[m][n][0] + zs;
          O[m][n][1] = O[m][n][1] * al + sa * (float)C[m][n][1] + zs;
        }
      }
    }
    TQP(7);
  }

  // ---- epilogue: normalized partial O and lse per row
  if (sub == 0 && sr < MR) {
    l_s[sr] = l_run;
    m_s[sr] = m_run;
  }
  __syncthreads();
  const int jp0 = warp * 8;
#pragma unroll
  for (int m = 0; m < MT; m++) {
    const int row = m * 8 + g;
    if (row < R) {
      const int qi = row / G, h = row % G;
      const long base = (long)(seq * QL + qi) * smr + (long)(kvh * G + h) * smh + (long)sid * sms;
      const float l = l_s[row];
      const float il = l > 0.f ? 1.f / l : 0.f;
#pragma unroll
      for (int i = 0; i < 2; i++) {
        const int jp = jp0 + 2 * t4 + i;
        *reinterpret_cast<float4*>(Mid + base + 4 * jp) =
            make_float4(O[m][0][i] * il, O[m][1][i] * il, O[m][2][i] * il, O[m][3][i] * il);
      }
      if (warp == 0 && t4 == 0) Mid[base + HD] = l > 0.f ? m_s[row] + logf(l) : -INFINITY;
    }
  }
}

// ===================================================================================================================
// v2: aligned u32 record loads (funnel-shifted, 16 loads/thread/chunk instead of 33 u16), K codes decoded inside the
// QK warp straight into IMMA B fragments (no int8 K tile, no decode sync), V B fragments gathered from the raw tile,
// score tile aliased onto the warp's own raw-K rows.  Smem ~26 KB (MT=3) -> 2 CTAs/SM.
// ===================================================================================================================
#define KW 33  // raw K row (words): 24 code words + norm + pad; 8 rows (1056 B) also hold the warp's score tile
#define VW 33  // raw V row (words): 32 code words + meta

struct Pref2 {
  uint32_t w0[8], w1[8];
  uint32_t shb;  // bit u: record of token u starts at 2 mod 4
};

__device__ __forceinline__ void load_chunk2(Pref2& R, const uint8_t* __restrict__ KV, const ChunkAddr& A, int nvalid,
                                            int bs, long scp, int lane, int warp) {
  R.shb = 0u;
#pragma unroll
  for (int u = 0; u < 8; u++) {
    const int tk = warp * 8 + u;
    R.w0[u] = 0u;
    R.w1[u] = 0u;
    if (tk < nvalid) {
      const uint8_t* a = KV + tok_slot(A, tk, bs, scp);
      const uint32_t s = (uint32_t)((uintptr_t)a >> 1) & 1u;
      const uint32_t* a0 = reinterpret_cast<const uint32_t*>((uintptr_t)a & ~(uintptr_t)3);
      R.shb |= s << u;
      R.w0[u] = a0[lane];
      if (lane < 26) {
        if (lane == 25 && s == 0u)
          R.w1[u] = *reinterpret_cast<const unsigned short*>(a0 + 57);  // never read past the record end
        else
          R.w1[u] = a0[32 + lane];
      }
    }
  }
}

template <int MT, bool QSPLIT>
__global__ void __launch_bounds__(NTHR, 2) tq_imma2_stage1(
    const int8_t* __restrict__ Q8, const float* __restrict__ QS, const uint8_t* __restrict__ KV,
    const int* __restrict__ BT, const int* __restrict__ RL, float* __restrict__ Mid, long sq8r, long sq8h,
    long scb, long scp, long sch, long sbt, long smr, long smh, long sms, int QL, int G, int bs, int NS,
    float attn_scale, float cscale, int norm_corr, int nbt, uint32_t lut_lo, uint32_t lut_hi) {
  constexpr int MR = MT * 8;
  static_assert(MR * 8 * 4 <= 8 * KW * 4, "score tile must fit in the warp's raw-K rows");
  extern __shared__ __align__(16) unsigned char smem[];
  uint32_t* Kraw = reinterpret_cast<uint32_t*>(smem);                  // [T][KW]
  uint32_t* Vraw = Kraw + T * KW;                                      // [T][VW]
  int8_t* Qs = reinterpret_cast<int8_t*>(Vraw + T * VW);               // [MR][256] swizzled
  int8_t* Qs2 = Qs + MR * HD;                                          // [MR][256] low plane (QSPLIT)
  uint8_t* Ps = reinterpret_cast<uint8_t*>(Qs2 + (QSPLIT ? MR * HD : 0));  // [MR][LDP]
  float* vs_s = reinterpret_cast<float*>(Ps + MR * LDP);               // [T]
  float* vz_s = vs_s + T;                                              // [T]
  float* qs_s = vz_s + T;                                              // [MR]
  float* al_s = qs_s + MR;
  float* sa_s = al_s + MR;
  float* zs_s = sa_s + MR;
  float* l_s = zs_s + MR;
  float* m_s = l_s + MR;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const int seq = blockIdx.x, kvh = blockIdx.y, sid = blockIdx.z;
  const int R = QL * G;

  int Lmax = 0;
  for (int i = 0; i < QL; i++) Lmax = max(Lmax, RL[seq * QL + i]);
  int split_len = (Lmax + NS - 1) / NS;
  split_len = (split_len + T - 1) / T * T;
  const int t0 = split_len * sid;
  const int t1 = min(t0 + split_len, Lmax);

  const int sr = tid >> 3, sub = tid & 7;
  const bool srow_ok = sr < R;
  const int s_qi = srow_ok ? sr / G : 0;
  const int s_len = srow_ok ? RL[seq * QL + s_qi] : 0;

  if (t0 >= t1) {
    if (sub == 0 && srow_ok) {
      const long base = (long)(seq * QL + s_qi) * smr + (long)(kvh * G + sr % G) * smh + (long)sid * sms;
      Mid[base + HD] = -INFINITY;
    }
    return;
  }

  {  // stage int8 Q
    const int row = tid >> 3, part = tid & 7;
    if (row < MR) {
      uint4 a = make_uint4(0, 0, 0, 0), b = a, c = a, d = a;
      float sc = 0.f;
      if (row < R) {
        const int qi = row / G, h = row % G;
        const int8_t* src = Q8 + (long)(seq * QL + qi) * sq8r + (long)(kvh * G + h) * sq8h + part * 32;
        a = *reinterpret_cast<const uint4*>(src);
        b = *reinterpret_cast<const uint4*>(src + 16);
        if (QSPLIT) {
          c = *reinterpret_cast<const uint4*>(src + HD);
          d = *reinterpret_cast<const uint4*>(src + HD + 16);
        }
        sc = QS[(long)(seq * QL + qi) * (sq8r / sq8h) + (kvh * G + h)];
      }
      *reinterpret_cast<uint4*>(Qs + kq_off(row, 2 * part)) = a;
      *reinterpret_cast<uint4*>(Qs + kq_off(row, 2 * part + 1)) = b;
      if (QSPLIT) {
        *reinterpret_cast<uint4*>(Qs2 + kq_off(row, 2 * part)) = c;
        *reinterpret_cast<uint4*>(Qs2 + kq_off(row, 2 * part + 1)) = d;
      }
      if (part == 0) qs_s[row] = sc;
    }
  }

  float O[MT][4][2];
#pragma unroll
  for (int m = 0; m < MT; m++)
#pragma unroll
    for (int n = 0; n < 4; n++) O[m][n][0] = O[m][n][1] = 0.f;
  float m_run = -INFINITY, l_run = 0.f;

  const long btrow = (long)seq * sbt;
  const long koff = (long)kvh * sch;
  Pref2 cur;
  load_chunk2(cur, KV, chunk_addr(BT, btrow, nbt, t0, bs, scb, scp, koff), min(T, t1 - t0), bs, scp, lane, warp);
#ifdef TQ_IMMA_PROF
  long long _pt = clock64();
#endif

  for (int c0 = t0; c0 < t1; c0 += T) {
    const int nvalid = min(T, t1 - c0);
    __syncthreads();  // previous chunk: PV done with Vraw / Ps, softmax done with the score tiles
    TQP(0);
    // ---- stage this warp's 8 records: funnel-shift to record alignment, split K codes / V codes / metadata
#pragma unroll
    for (int u = 0; u < 8; u++) {
      const int tk = warp * 8 + u;
      const uint32_t r0 = cur.w0[u], r1 = cur.w1[u];
      uint32_t n0 = __shfl_down_sync(0xffffffffu, r0, 1);
      const uint32_t x = __shfl_sync(0xffffffffu, r1, 0);
      const uint32_t n1 = __shfl_down_sync(0xffffffffu, r1, 1);
      if (lane == 31) n0 = x;
      const bool s2 = (cur.shb >> u) & 1u;
      const uint32_t ksh = s2 ? 16u : 0u, vsh = s2 ? 0u : 16u;
      const int dl = s2 ? 1 : 0;
      if (lane < 24) Kraw[tk * KW + lane] = __funnelshift_r(r0, n0, ksh);
      if (lane == 24) Kraw[tk * KW + 24] = s2 ? (r0 >> 16) : (r0 & 0xFFFFu);
      const int i0 = lane - 24 - dl;
      if (i0 >= 0) Vraw[tk * VW + i0] = __funnelshift_r(r0, n0, vsh);
      const int i1 = 8 + lane - dl;
      if (i1 <= 31) Vraw[tk * VW + i1] = __funnelshift_r(r1, n1, vsh);
      if (i1 == 32) {
        const uint32_t meta = __funnelshift_r(r1, n1, vsh);
        vs_s[tk] = __half2float(__ushort_as_half((unsigned short)(meta & 0xFFFFu)));
        vz_s[tk] = __half2float(__ushort_as_half((unsigned short)(meta >> 16)));
      }
    }
    TQP(9);
    if (c0 + T < t1)
      load_chunk2(cur, KV, chunk_addr(BT, btrow, nbt, c0 + T, bs, scb, scp, koff), min(T, t1 - c0 - T), bs, scp, lane, warp);
    TQP(1);
    __syncwarp();
    TQP(2);

    // ---- QK^T: warp -> its own 8 tokens; B fragments decoded from raw 3-bit codes in registers
    {
      const int tk = warp * 8 + g;
      const uint32_t* Krow = Kraw + tk * KW;
      int acc[MT][2], acc2[MT][2];
#pragma unroll
      for (int m = 0; m < MT; m++) acc[m][0] = acc[m][1] = acc2[m][0] = acc2[m][1] = 0;
      int ss = 0;
#pragma unroll
      for (int sq = 0; sq < 4; sq++) {
        const int wi = 6 * sq + 3 * (t4 >> 1) + (t4 & 1);
        const uint32_t ka = Krow[wi], kb = Krow[wi + 1];
        const uint32_t x0 = (t4 & 1) ? __funnelshift_r(ka, kb, 16) : ka;
        const uint32_t x1 = (t4 & 1) ? (kb >> 16) : (kb & 0xFFFFu);
        const uint32_t sa = spread3(x0 & 0xFFFFFFu), sb = spread3((x0 >> 24) | (x1 << 8));
        uint4 b;
        b.x = __byte_perm(lut_lo, lut_hi, sa & 0xFFFFu);
        b.y = __byte_perm(lut_lo, lut_hi, sa >> 16);
        b.z = __byte_perm(lut_lo, lut_hi, sb & 0xFFFFu);
        b.w = __byte_perm(lut_lo, lut_hi, sb >> 16);
        ss = __dp4a((int)b.x, (int)b.x, ss);
        ss = __dp4a((int)b.y, (int)b.y, ss);
        ss = __dp4a((int)b.z, (int)b.z, ss);
        ss = __dp4a((int)b.w, (int)b.w, ss);
#pragma unroll
        for (int m = 0; m < MT; m++) {
          const uint4 a = *reinterpret_cast<const uint4*>(Qs + kq_off(m * 8 + g, 4 * sq + t4));
          imma_s8(acc[m], a.x, b.x);
          imma_s8(acc[m], a.y, b.y);
          imma_s8(acc[m], a.z, b.z);
          imma_s8(acc[m], a.w, b.w);
          if (QSPLIT) {
            const uint4 a2 = *reinterpret_cast<const uint4*>(Qs2 + kq_off(m * 8 + g, 4 * sq + t4));
            imma_s8(acc2[m], a2.x, b.x);
            imma_s8(acc2[m], a2.y, b.y);
            imma_s8(acc2[m], a2.z, b.z);
            imma_s8(acc2[m], a2.w, b.w);
          }
        }
      }
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      const float nrm = __half2float(__ushort_as_half((unsigned short)Krow[24]));
      const float fg = (ss > 0) ? nrm * attn_scale * (norm_corr ? rsqrtf((float)ss) : cscale) : 0.f;
      const float f0 = __shfl_sync(0xffffffffu, fg, 8 * t4), f1 = __shfl_sync(0xffffffffu, fg, 8 * t4 + 4);
      __syncwarp();  // every lane is done reading this warp's raw-K rows: reuse them as the score tile
      float* St = reinterpret_cast<float*>(Kraw + warp * 8 * KW);  // [MR][8]
#pragma unroll
      for (int m = 0; m < MT; m++) {
        const float qsc = qs_s[m * 8 + g];
        *reinterpret_cast<float2*>(St + (m * 8 + g) * 8 + 2 * t4) =
            make_float2(((float)acc[m][0] + (QSPLIT ? (float)acc2[m][0] * (1.f / 254.f) : 0.f)) * qsc * f0,
                        ((float)acc[m][1] + (QSPLIT ? (float)acc2[m][1] * (1.f / 254.f) : 0.f)) * qsc * f1);
      }
    }
    TQP(3);
    __syncthreads();
    TQP(4);

    // ---- online softmax (row sr, 8 lanes, lane sub = token group sub = warp sub's score tile)
    {
      float s[8];
      if (sr < MR) {
        const float* St = reinterpret_cast<const float*>(Kraw + sub * 8 * KW) + sr * 8;
        const float4 x = *reinterpret_cast<const float4*>(St);
        const float4 y = *reinterpret_cast<const float4*>(St + 4);
        s[0] = x.x; s[1] = x.y; s[2] = x.z; s[3] = x.w; s[4] = y.x; s[5] = y.y; s[6] = y.z; s[7] = y.w;
      }
      float cmax = -INFINITY;
#pragma unroll
      for (int k = 0; k < 8; k++) {
        const int tk = sub * 8 + k;
        const bool ok = srow_ok && tk < nvalid && (c0 + tk) < s_len;
        s[k] = ok ? s[k] : -INFINITY;
        cmax = fmaxf(cmax, s[k]);
      }
#pragma unroll
      for (int o = 4; o > 0; o >>= 1) cmax = fmaxf(cmax, __shfl_xor_sync(0xffffffffu, cmax, o));
      const float m_new = fmaxf(m_run, cmax);
      float alpha, a[8], psum = 0.f, zsum = 0.f, amax = 0.f;
      if (m_new == -INFINITY) {
        alpha = 1.f;
#pragma unroll
        for (int k = 0; k < 8; k++) a[k] = 0.f;
      } else {
        alpha = (m_run == -INFINITY) ? 0.f : __expf(m_run - m_new);
#pragma unroll
        for (int k = 0; k < 8; k++) {
          const float p = __expf(s[k] - m_new);
          const int tk = sub * 8 + k;
          a[k] = p * vs_s[tk];
          psum += p;
          zsum += p * vz_s[tk];
          amax = fmaxf(amax, a[k]);
        }
      }
#pragma unroll
      for (int o = 4; o > 0; o >>= 1) {
        psum += __shfl_xor_sync(0xffffffffu, psum, o);
        zsum += __shfl_xor_sync(0xffffffffu, zsum, o);
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
      }
      l_run = l_run * alpha + psum;
      m_run = m_new;
      const float inv = amax > 0.f ? 255.f / amax : 0.f;
      uint32_t w0 = 0, w1 = 0;
#pragma unroll
      for (int k = 0; k < 4; k++) {
        w0 |= (uint32_t)__float2uint_rn(a[k] * inv) << (8 * k);
        w1 |= (uint32_t)__float2uint_rn(a[k + 4] * inv) << (8 * k);
      }
      if (sr < MR) {
        *reinterpret_cast<uint2*>(Ps + sr * LDP + sub * 8) = make_uint2(w0, w1);
        if (sub == 0) {
          al_s[sr] = alpha;
          sa_s[sr] = amax * (1.f / 255.f);
          zs_s[sr] = zsum;
        }
      }
    }
    TQP(5);
    __syncthreads();
    TQP(6);

    // ---- PV: warp -> V bytes j in [16w, 16w+16) (dims 32w..32w+31); B fragments gathered from raw V rows
    {
      int C[MT][4][2];
#pragma unroll
      for (int m = 0; m < MT; m++)
#pragma unroll
        for (int n = 0; n < 4; n++) C[m][n][0] = C[m][n][1] = 0;
      const uint8_t* Vb = reinterpret_cast<const uint8_t*>(Vraw);
#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        const int tb = 16 * kk + 4 * t4;
        uint32_t bf[4];
#pragma unroll
        for (int jt = 0; jt < 2; jt++) {
          const int j = 16 * warp + 8 * jt + g;
          const uint32_t v0 = Vb[(tb + 0) * VW * 4 + j], v1 = Vb[(tb + 1) * VW * 4 + j];
          const uint32_t v2 = Vb[(tb + 2) * VW * 4 + j], v3 = Vb[(tb + 3) * VW * 4 + j];
          const uint32_t v = v0 | (v1 << 8) | (v2 << 16) | (v3 << 24);
          bf[2 * jt] = v & 0x0F0F0F0Fu;
          bf[2 * jt + 1] = (v >> 4) & 0x0F0F0F0Fu;
        }
#pragma unroll
        for (int m = 0; m < MT; m++) {
          const uint32_t a = *reinterpret_cast<const uint32_t*>(Ps + (m * 8 + g) * LDP + 16 * kk + 4 * t4);
          imma_u8(C[m][0], a, bf[0]);
          imma_u8(C[m][1], a, bf[1]);
          imma_u8(C[m][2], a, bf[2]);
          imma_u8(C[m][3], a, bf[3]);
        }
      }
#pragma unroll
      for (int m = 0; m < MT; m++) {
        const int row = m * 8 + g;
        const float al = al_s[row], sa = sa_s[row], zs = zs_s[row];
#pragma unroll
        for (int n = 0; n < 4; n++) {
          O[m][n][0] = O[m][n][0] * al + sa * (float)C[m][n][0] + zs;
          O[m][n][1] = O[m][n][1] * al + sa * (float)C[m][n][1] + zs;
        }
      }
    }
    TQP(7);
  }

  if (sub == 0 && sr < MR) {
    l_s[sr] = l_run;
    m_s[sr] = m_run;
  }
  __syncthreads();
#pragma unroll
  for (int m = 0; m < MT; m++) {
    const int row = m * 8 + g;
    if (row < R) {
      const int qi = row / G, h = row % G;
      const long base = (long)(seq * QL + qi) * smr + (long)(kvh * G + h) * smh + (long)sid * sms;
      const float l = l_s[row];
      const float il = l > 0.f ? 1.f / l : 0.f;
#pragma unroll
      for (int jt = 0; jt < 2; jt++)
#pragma unroll
        for (int i = 0; i < 2; i++) {
          const int j = 16 * warp + 8 * jt + 2 * t4 + i;  // C column 2*t4+i <-> byte j -> dims 2j, 2j+1
          *reinterpret_cast<float2*>(Mid + base + 2 * j) = make_float2(O[m][2 * jt][i] * il, O[m][2 * jt + 1][i] * il);
        }
      if (warp == 0 && t4 == 0) Mid[base + HD] = l > 0.f ? m_s[row] + logf(l) : -INFINITY;
    }
  }
}

static inline size_t tq_imma2_smem_bytes(int MT, bool qsplit = false) {
  const int MR = MT * 8;
  return (size_t)T * KW * 4 + T * VW * 4 + MR * HD * (qsplit ? 2 : 1) + MR * LDP + 2 * T * 4 + 6 * MR * 4;
}

// Reduce over KV splits using only the per-split lse (-inf = empty). grid (R*Hq), block HD threads.
__global__ void tq_imma_stage2(const float* __restrict__ Mid, __half* __restrict__ Out, float* __restrict__ Lse, int Hq,
                               int NS, long smr, long smh, long sms, long sor, long soh, long slr) {
  extern __shared__ float w_s[];
  __shared__ float red[HD / 32];
  const int r = blockIdx.x / Hq, h = blockIdx.x % Hq, d = threadIdx.x;
  const float* base = Mid + (long)r * smr + (long)h * smh;
  float lm = -INFINITY;
  for (int s = d; s < NS; s += HD) {
    const float v = base[(long)s * sms + HD];
    w_s[s] = v;
    lm = fmaxf(lm, v);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) lm = fmaxf(lm, __shfl_xor_sync(0xffffffffu, lm, o));
  if ((d & 31) == 0) red[d >> 5] = lm;
  __syncthreads();
  float mx = red[0];
#pragma unroll
  for (int i = 1; i < HD / 32; i++) mx = fmaxf(mx, red[i]);
  __syncthreads();
  for (int s = d; s < NS; s += HD) w_s[s] = (w_s[s] == -INFINITY || mx == -INFINITY) ? 0.f : __expf(w_s[s] - mx);
  __syncthreads();
  float acc = 0.f, sum = 0.f;
  for (int s = 0; s < NS; s++) {
    const float w = w_s[s];
    if (w > 0.f) acc += w * base[(long)s * sms + d];
    sum += w;
  }
  Out[(long)r * sor + (long)h * soh + d] = __float2half(sum > 0.f ? acc / sum : 0.f);
  if (d == 0) Lse[(long)r * slr + h] = sum > 0.f ? mx + logf(sum) : -INFINITY;
}

// q_rot fp32 [R, Hq, D] -> int8 [R, Hq, D] + per-(row, head) scale. grid R*Hq, block 64 (4 dims per thread).
__global__ void tq_imma_qquant(const float* __restrict__ Qr, int8_t* __restrict__ Q8, float* __restrict__ QS, int Hq, int split,
                               long sqr, long sqh) {
  const int r = blockIdx.x / Hq, h = blockIdx.x % Hq, t = threadIdx.x;
  const float4 v = *reinterpret_cast<const float4*>(Qr + (long)r * sqr + (long)h * sqh + 4 * t);
  float am = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
  __shared__ float red[2];
  if ((t & 31) == 0) red[t >> 5] = am;
  __syncthreads();
  am = fmaxf(red[0], red[1]);
  const float inv = am > 0.f ? 127.f / am : 0.f;
  const float x[4] = {v.x * inv, v.y * inv, v.z * inv, v.w * inv};
  char4 q, ql;
  int hi[4], lo[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    hi[i] = __float2int_rn(x[i]);
    lo[i] = max(-127, min(127, __float2int_rn((x[i] - (float)hi[i]) * 254.f)));
  }
  q = make_char4(hi[0], hi[1], hi[2], hi[3]);
  ql = make_char4(lo[0], lo[1], lo[2], lo[3]);
  const long qrow = ((long)r * Hq + h) * (split ? 2 * HD : HD);
  *reinterpret_cast<char4*>(Q8 + qrow + 4 * t) = q;
  if (split) *reinterpret_cast<char4*>(Q8 + qrow + HD + 4 * t) = ql;
  if (t == 0) QS[(long)r * Hq + h] = am / 127.f;
}


static inline size_t tq_imma_smem_bytes(int MT, bool qsplit = false) {
  const int MR = MT * 8;
  return (size_t)T * HD + MR * HD * (qsplit ? 2 : 1) + 64 * 16 * 8 + MR * LDS_S * 4 + MR * LDP + 3 * T * 4 + 6 * MR * 4;
}
