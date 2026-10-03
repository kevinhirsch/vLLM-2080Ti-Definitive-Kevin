// K1 fa75 prefill: flash-attention forward for Turing (sm_75), head_dim 256, fp16 in/out, fp32 LSE out.
// Lane K1, 2026-10-03.
//
// Derived from the "fa75" kernel of the ExLlamaV3 Turing fork (rafatxf/exllamav3, branch turing,
// exllamav3_ext/turing/fa75.cu, MIT licence, Copyright (c) turboderp and contributors). Changes here:
//   * log-sum-exp output (natural log, the convention of vllm merge_attn_states), any strides
//   * variable-length batch (cu_seqlens_q / cu_seqlens_k, one request per grid.z), bottom-right causal per request
//   * strided output, compile-time variants (P V accumulate fp16-per-slice or fp32, causal)
//   * lazy O rescale (warp-uniform skip when no row max moved), wider key tiles (BN template)
//
// Layout: q [Tq_total, Hq, 256], k/v [Tkv_total, Hkv, 256] (row and head strides, last dim contiguous),
// o [Tq_total, Hq, 256] (row/head strides), lse [Tq_total, Hq] fp32 (row/head strides) or null.
// Request b: rows cu_q[b]..cu_q[b+1], keys cu_k[b]..cu_k[b+1]. Causal is bottom-right aligned
// (row i sees keys j <= i + Tkv - Tq), as for a prefill chunk appended to a cache. GQA: head h -> kv head h/(Hq/Hkv).
//
// Block = 64 query rows of one head, 4 warps x 16 rows, two blocks per SM. Q dims 0..127 live in mma A fragments,
// dims 128..255 in shared memory (ldmatrix per tile), O in fp32 accumulators (128 regs). S -> P stays in registers
// (the m16n8 accumulator layout is the m16n8k8 A layout). K and V tiles of BN keys sit in XOR-swizzled shared memory
// and are read with ldmatrix(.trans); the global loads of V(kt) are in flight during Q K^T, those of K(kt+1) during
// softmax and P V (ordinary loads staged through registers: Turing has no cp.async, register staging gives the same
// overlap).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define HD 256
#define BM 64
#define NWARPS 4
#define NTHREADS (NWARPS * 32)
#define ROW_CHUNKS (HD / 8)  // 32 x 16-byte chunks per 512-byte row

// K/V tiles: [BN rows][32 chunks], chunk ^= row & 7
static __device__ __forceinline__ uint32_t tile_off(int r, int c) { return (uint32_t)((r * ROW_CHUNKS + (c ^ (r & 7))) * 16); }
// Q dims 128..255: 64 rows x 16 chunks, swizzled, after the K and V tiles
static __device__ __forceinline__ uint32_t qh_off(uint32_t base, int r, int c) { return base + (uint32_t)((r * 16 + (c ^ (r & 7))) * 16); }
static __device__ __forceinline__ uint32_t ql_off(int r, int c) { return (uint32_t)((r * 16 + (c ^ (r & 7))) * 16); }

static __device__ __forceinline__ void mma1688(float* c, uint32_t a0, uint32_t a1, uint32_t b) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(b));
}
// fp16-accumulate HMMA: twice the fp32-accumulate rate on GeForce Turing
static __device__ __forceinline__ void mma1688h(uint32_t* c, uint32_t a0, uint32_t a1, uint32_t b) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
               : "+r"(c[0]), "+r"(c[1])
               : "r"(a0), "r"(a1), "r"(b));
}
static __device__ __forceinline__ void ldsm_x4(uint32_t* r, uint32_t saddr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(saddr));
}
static __device__ __forceinline__ void ldsm_x4_t(uint32_t* r, uint32_t saddr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(saddr));
}
static __device__ __forceinline__ uint32_t pack_h2(float a, float b) {
  half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}

// Segment accumulation (attention over a context processed in pieces, merged exactly by LSE in the epilogue):
// mode 0 = plain (o fp16 [+ lse]); 1 = first piece (write acc_o fp32 = O, acc_lse); 2 = middle piece (merge into
// acc_o / acc_lse); 3 = last piece (merge with acc and write the final o fp16 [+ lse]). acc_* use natural-log LSE.
struct AccArgs {
  float* o;
  float* lse;
  int64_t o_row, o_head, lse_row, lse_head;
  int mode;
  // split-KV (wave balance): nsplit > 1 -> each CTA covers keys [s * split_len, (s + 1) * split_len) of its request
  // and writes normalised O (fp32) + LSE to part_o [S, Tq_total, Hq, 256] / part_lse [S, Tq_total, Hq]; a combine
  // kernel then merges the S parts (and applies the acc mode)
  int nsplit, split_len;
  float* part_o;
  float* part_lse;
  int64_t part_rows;  // Tq_total
};

// PV16: P V accumulates each BN-key slice in fp16 then adds into fp32 O (fa75's scheme), else straight fp32.
// LAZY: rescale O only when some row max of the warp grows by more than LAZY_TAU (log2 units); P is then bounded
// by 2^LAZY_TAU instead of 1 (the running max that defines the exponent is kept stale).
template <int BN, bool CAUSAL, bool PV16, bool LAZY, bool QK4, int ABL = 0, int QKC = 0>
__global__ void __launch_bounds__(NTHREADS, 2) k1fa_kernel(
    const half* __restrict__ q, int64_t q_row, int64_t q_head,
    const half* __restrict__ k, int64_t k_row, int64_t k_head,
    const half* __restrict__ v, int64_t v_row, int64_t v_head,
    half* __restrict__ o, int64_t o_row, int64_t o_head,
    float* __restrict__ lse, int64_t lse_row, int64_t lse_head, AccArgs acc,
    const int* __restrict__ cu_q, const int* __restrict__ cu_k,
    int Hq, int Hkv, float scale_log2, float q_prescale) {
  static_assert(QKC == 0 || (QKC % 16 == 0 && HD % QKC == 0 && !QK4), "QKC: 0 or a multiple of 16 dividing 256");
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750)
  __trap();
#else
  constexpr uint32_t TILE_BYTES = BN * HD * 2;
  constexpr uint32_t QH_BASE = 2 * TILE_BYTES;           // K tile, V tile, then Q high half (16 KB)
  constexpr uint32_t SMEM = QH_BASE + BM * HD;          // BM rows x 128 dims x 2 B
  static_assert(SMEM >= BM * HD, "Q low staging must fit before QH");
  constexpr float LAZY_TAU = PV16 ? 3.0f : 8.0f;  // P <= 2^TAU; fp16 P V slices stay far from overflow
  extern __shared__ __align__(128) uint8_t sm[];
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(sm);
  const uint32_t sK = sbase, sV = sbase + TILE_BYTES, sQH = sbase + QH_BASE;

  const int b = blockIdx.z;
  const int q_beg = cu_q[b], Tq = cu_q[b + 1] - q_beg;
  const int k_beg = cu_k[b], Tkv = cu_k[b + 1] - k_beg;
  const int n_mtiles = (Tq + BM - 1) / BM;
  const int split = (int)blockIdx.x % acc.nsplit;
  const int m_tile = n_mtiles - 1 - (int)blockIdx.x / acc.nsplit;  // heaviest (last) query tiles first
  if (m_tile < 0) return;

  const int t_id = threadIdx.x;
  const int warp = t_id >> 5;
  const int lane = t_id & 31;
  const int gq = lane >> 2;
  const int cq = lane & 3;
  const int lr = lane & 7;
  const int lm = lane >> 3;

  const int hq = blockIdx.y;
  const int hk = hq / (Hq / Hkv);
  const int m0 = m_tile * BM;
  const int offs = Tkv - Tq;  // causal: key j visible to row i iff j <= i + offs

  const half* q_ = q + (int64_t)q_beg * q_row + hq * q_head;
  const half* k_ = k + (int64_t)k_beg * k_row + hk * k_head;
  const half* v_ = v + (int64_t)k_beg * v_row + hk * v_head;

  // Stage Q (64 x 256): dims 0..127 through the tile area into A fragments, dims 128..255 into QH (kept)
  uint32_t qf[HD / 16][2];
  {
#pragma unroll
    for (int j = 0; j < (BM * ROW_CHUNKS) / NTHREADS; ++j) {
      int idx = t_id + j * NTHREADS;
      int r = idx / ROW_CHUNKS, c = idx % ROW_CHUNKS;
      uint4 val = make_uint4(0, 0, 0, 0);
      if (m0 + r < Tq) val = *reinterpret_cast<const uint4*>(q_ + (int64_t)(m0 + r) * q_row + c * 8);
      if (QKC > 0) {  // (Lane K9) fold the softmax scale into Q so fp16 partial scores hold scaled logits
        const half2 ps = __float2half2_rn(q_prescale);
        half2* hv = reinterpret_cast<half2*>(&val);
#pragma unroll
        for (int e = 0; e < 4; ++e) hv[e] = __hmul2(hv[e], ps);
      }
      if (c < ROW_CHUNKS / 2) *reinterpret_cast<uint4*>(sm + ql_off(r, c)) = val;
      else *reinterpret_cast<uint4*>(sm + (qh_off(QH_BASE, r, c - ROW_CHUNKS / 2))) = val;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < HD / 16; kk += 2) {
      uint32_t r4[4];
      int row = 16 * warp + lr + (lm & 1) * 8;
      ldsm_x4(r4, sbase + ql_off(row, kk + (lm >> 1)));
      qf[kk][0] = r4[0]; qf[kk][1] = r4[1];
      qf[kk + 1][0] = r4[2]; qf[kk + 1][1] = r4[3];
    }
    __syncthreads();
  }

  float oacc[HD / 8][4];
#pragma unroll
  for (int dn = 0; dn < HD / 8; ++dn)
#pragma unroll
    for (int i = 0; i < 4; ++i) oacc[dn][i] = 0.0f;
  float mrow[2] = {-1e30f, -1e30f};
  float lrow[2] = {0.0f, 0.0f};

  const int row0 = m0 + 16 * warp + gq;  // this thread's rows: row0, row0 + 8
  int kv_end = CAUSAL ? min(Tkv, m0 + BM - 1 + offs + 1) : Tkv;
  kv_end = max(kv_end, 0);
  const int kv_lo = acc.nsplit > 1 ? split * acc.split_len : 0;  // split_len is a multiple of BN
  if (acc.nsplit > 1) kv_end = min(kv_end, kv_lo + acc.split_len);
  const int n_tiles = kv_end > kv_lo ? (kv_end - kv_lo + BN - 1) / BN : 0;
  // this warp's last visible key (causal): tiles beyond it contribute nothing to the warp
  const int warp_kv_end = CAUSAL ? min(Tkv, m0 + 16 * warp + 15 + offs + 1) : Tkv;

  constexpr int LD_PER_T = (BN * ROW_CHUNKS) / NTHREADS;  // 16-byte loads per thread per tile
  auto load_tile = [&](const half* src, int64_t row_stride, int n0, uint4* reg) {
#pragma unroll
    for (int j = 0; j < LD_PER_T; ++j) {
      int idx = t_id + j * NTHREADS;
      int r = idx / ROW_CHUNKS, c = idx % ROW_CHUNKS;
      reg[j] = n0 + r < Tkv ? *reinterpret_cast<const uint4*>(src + (int64_t)(n0 + r) * row_stride + c * 8)
                            : make_uint4(0, 0, 0, 0);
    }
  };
  auto store_tile = [&](uint32_t off, const uint4* reg) {
#pragma unroll
    for (int j = 0; j < LD_PER_T; ++j) {
      int idx = t_id + j * NTHREADS;
      int r = idx / ROW_CHUNKS, c = idx % ROW_CHUNKS;
      *reinterpret_cast<uint4*>(sm + off + tile_off(r, c)) = reg[j];
    }
  };

  uint4 stage[LD_PER_T];
  if (n_tiles > 0) {
    load_tile(k_, k_row, kv_lo, stage);
    store_tile(0, stage);
  }
  __syncthreads();

  constexpr int NT8 = BN / 8;  // n8 score tiles per warp per key tile
  for (int kt = 0; kt < n_tiles; ++kt) {
    const int n0 = kv_lo + kt * BN;
    if (!(ABL & 1)) load_tile(v_, v_row, n0, stage);  // V(kt) in flight during Q K^T

    const bool warp_active = n0 < warp_kv_end;  // warp-uniform
    float s[NT8][4];
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) s[nt][i] = 0.0f;
    uint32_t sh[NT8][2];  // QKC > 0: fp16 partial scores of the current head-dim chunk
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt) { sh[nt][0] = 0u; sh[nt][1] = 0u; }
    if (warp_active && !(ABL & 2)) {
      if (QK4) {
        // two independent accumulator sets (dims 0..127 from registers, 128..255 from smem), interleaved:
        // 4 HMMA dependency chains per warp instead of 2, and the smem Q loads overlap register-Q HMMAs
        float s2[NT8][4];
#pragma unroll
        for (int nt = 0; nt < NT8; ++nt)
#pragma unroll
          for (int i = 0; i < 4; ++i) s2[nt][i] = 0.0f;
#pragma unroll
        for (int kk = 0; kk < HD / 16; kk += 2) {
          const int kh = kk + HD / 16;
          uint32_t ah[4];
          ldsm_x4(ah, qh_off(sQH, 16 * warp + lr + (lm & 1) * 8, kk + (lm >> 1)));
#pragma unroll
          for (int np = 0; np < BN / 16; ++np) {
            uint32_t bl[4], bh[4];
            ldsm_x4(bl, sK + tile_off(16 * np + 8 * (lm & 1) + lr, kk + (lm >> 1)));
            ldsm_x4(bh, sK + tile_off(16 * np + 8 * (lm & 1) + lr, kh + (lm >> 1)));
            mma1688(s[2 * np], qf[kk][0], qf[kk][1], bl[0]);
            mma1688(s2[2 * np], ah[0], ah[1], bh[0]);
            mma1688(s[2 * np + 1], qf[kk][0], qf[kk][1], bl[1]);
            mma1688(s2[2 * np + 1], ah[0], ah[1], bh[1]);
            mma1688(s[2 * np], qf[kk + 1][0], qf[kk + 1][1], bl[2]);
            mma1688(s2[2 * np], ah[2], ah[3], bh[2]);
            mma1688(s[2 * np + 1], qf[kk + 1][0], qf[kk + 1][1], bl[3]);
            mma1688(s2[2 * np + 1], ah[2], ah[3], bh[3]);
          }
        }
#pragma unroll
        for (int nt = 0; nt < NT8; ++nt)
#pragma unroll
          for (int i = 0; i < 4; ++i) s[nt][i] += s2[nt][i];
      } else {
#pragma unroll
      for (int kk = 0; kk < HD / 8; kk += 2) {
        uint32_t a[4];
        if (kk < HD / 16) {
          a[0] = qf[kk][0]; a[1] = qf[kk][1]; a[2] = qf[kk + 1][0]; a[3] = qf[kk + 1][1];
        } else {
          ldsm_x4(a, qh_off(sQH, 16 * warp + lr + (lm & 1) * 8, kk - HD / 16 + (lm >> 1)));
        }
#pragma unroll
        for (int np = 0; np < BN / 16; ++np) {
          // matrices: (keys 16np+0-7, chunk kk), (16np+8-15, kk), (16np+0-7, kk+1), (16np+8-15, kk+1)
          uint32_t bb[4];
          ldsm_x4(bb, sK + tile_off(16 * np + 8 * (lm & 1) + lr, kk + (lm >> 1)));
          if (QKC > 0) {  // (Lane K9) full-rate fp16-accumulate HMMA, promoted to fp32 every QKC head dims
            mma1688h(sh[2 * np], a[0], a[1], bb[0]);
            mma1688h(sh[2 * np + 1], a[0], a[1], bb[1]);
            mma1688h(sh[2 * np], a[2], a[3], bb[2]);
            mma1688h(sh[2 * np + 1], a[2], a[3], bb[3]);
          } else {
            mma1688(s[2 * np], a[0], a[1], bb[0]);
            mma1688(s[2 * np + 1], a[0], a[1], bb[1]);
            mma1688(s[2 * np], a[2], a[3], bb[2]);
            mma1688(s[2 * np + 1], a[2], a[3], bb[3]);
          }
        }
        if (QKC > 0 && ((kk + 2) % (QKC / 8)) == 0) {
#pragma unroll
          for (int nt = 0; nt < NT8; ++nt) {
            const float2 lo = __half22float2(*reinterpret_cast<half2*>(&sh[nt][0]));
            const float2 hi = __half22float2(*reinterpret_cast<half2*>(&sh[nt][1]));
            s[nt][0] += lo.x; s[nt][1] += lo.y; s[nt][2] += hi.x; s[nt][3] += hi.y;
            sh[nt][0] = 0u; sh[nt][1] = 0u;
          }
        }
      }
      }
    }
    store_tile(TILE_BYTES, stage);

    const bool more = kt + 1 < n_tiles;
    if (more && !(ABL & 1)) load_tile(k_, k_row, n0 + BN, stage);  // K(kt+1) in flight during softmax and P V

    if (warp_active) {
      const bool need_mask = (n0 + BN > Tkv) || (CAUSAL && n0 + BN - 1 > m0 + 16 * warp + offs);
      float mnew[2], alpha[2];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int row = row0 + hh * 8;
        float mx = -1e30f;
#pragma unroll
        for (int nt = 0; nt < NT8; ++nt)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            float x = s[nt][2 * hh + e] * scale_log2;
            if (need_mask) {
              int key = n0 + 8 * nt + 2 * cq + e;
              if (key >= Tkv || (CAUSAL && key > row + offs)) x = -1e30f;
            }
            s[nt][2 * hh + e] = x;
            mx = fmaxf(mx, x);
          }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
        mnew[hh] = fmaxf(mrow[hh], mx);
      }
      bool rescale = true;
      if (LAZY) {
        // keep the stale max while no row of the warp grew by more than LAZY_TAU (P <= 2^TAU)
        const bool grow = (mnew[0] > mrow[0] + LAZY_TAU) || (mnew[1] > mrow[1] + LAZY_TAU);
        rescale = __any_sync(0xffffffff, grow);
      }
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const float m_use = rescale ? mnew[hh] : mrow[hh];
        alpha[hh] = exp2f(mrow[hh] - m_use);
        mrow[hh] = m_use;
        float sum = 0.0f;
#pragma unroll
        for (int nt = 0; nt < NT8; ++nt)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            float p = (ABL & 8) ? s[nt][2 * hh + e] : exp2f(s[nt][2 * hh + e] - m_use);
            s[nt][2 * hh + e] = p;
            sum += p;
          }
        lrow[hh] = lrow[hh] * alpha[hh] + sum;
      }
      if (rescale) {
#pragma unroll
        for (int dn = 0; dn < HD / 8; ++dn) {
          oacc[dn][0] *= alpha[0]; oacc[dn][1] *= alpha[0];
          oacc[dn][2] *= alpha[1]; oacc[dn][3] *= alpha[1];
        }
      }
    }
    if (!(ABL & 16)) __syncthreads();  // V(kt) visible, K(kt) no longer read

    if (warp_active) {
      uint32_t pa[NT8][2];
#pragma unroll
      for (int j = 0; j < NT8; ++j) {
        pa[j][0] = pack_h2(s[j][0], s[j][1]);
        pa[j][1] = pack_h2(s[j][2], s[j][3]);
      }
#pragma unroll
      for (int dn = 0; dn < HD / 8; dn += 4) {
        if (ABL & 4) {
          oacc[dn][0] += __uint_as_float(pa[0][0]);  // keep P live
        } else if (PV16) {
          uint32_t t[4][2] = {};
#pragma unroll
          for (int j = 0; j < NT8; ++j) {
            uint32_t bb[4];
            ldsm_x4_t(bb, sV + tile_off(8 * j + lr, dn + lm));
#pragma unroll
            for (int x = 0; x < 4; ++x) mma1688h(t[x], pa[j][0], pa[j][1], bb[x]);
          }
#pragma unroll
          for (int x = 0; x < 4; ++x) {
            float2 lo = __half22float2(*reinterpret_cast<half2*>(&t[x][0]));
            float2 hi = __half22float2(*reinterpret_cast<half2*>(&t[x][1]));
            oacc[dn + x][0] += lo.x; oacc[dn + x][1] += lo.y;
            oacc[dn + x][2] += hi.x; oacc[dn + x][3] += hi.y;
          }
        } else {
#pragma unroll
          for (int j = 0; j < NT8; ++j) {
            uint32_t bb[4];
            ldsm_x4_t(bb, sV + tile_off(8 * j + lr, dn + lm));
#pragma unroll
            for (int x = 0; x < 4; ++x) mma1688(oacc[dn + x], pa[j][0], pa[j][1], bb[x]);
          }
        }
      }
    }
    if (more) store_tile(0, stage);
    if (!(ABL & 16)) __syncthreads();  // K(kt+1) visible, V(kt) no longer read
  }

  // Normalize, store O and LSE (optionally merged with / into the segment accumulator)
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    float l = lrow[hh];
    l += __shfl_xor_sync(0xffffffff, l, 1);
    l += __shfl_xor_sync(0xffffffff, l, 2);
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    const int row = row0 + hh * 8;
    // scores were in log2 units: lse_e = (m + log2 l) * ln 2
    const float lse_new = l > 0.0f ? (mrow[hh] + __log2f(l)) * 0.6931471805599453f : -INFINITY;
    float wa = 0.0f, wn = inv, lse_tot = lse_new;  // final value = acc_o * wa + oacc * wn
    float* ao = nullptr;
    float* al = nullptr;
    if (acc.nsplit == 1 && acc.mode != 0 && row < Tq) {
      ao = acc.o + (int64_t)(q_beg + row) * acc.o_row + (int64_t)hq * acc.o_head;
      al = acc.lse + (int64_t)(q_beg + row) * acc.lse_row + (int64_t)hq * acc.lse_head;
      if (acc.mode >= 2) {
        const float la = *al;
        const float mx = fmaxf(la, lse_new);
        if (mx == -INFINITY) {
          wa = 0.0f; wn = 0.0f; lse_tot = -INFINITY;
        } else {
          const float ea = __expf(la - mx), en = __expf(lse_new - mx);
          const float den = ea + en;
          wa = ea / den; wn = inv * en / den; lse_tot = mx + __logf(den);
        }
      }
    }
    __syncwarp();  // every lane of the quad has read acc_lse before lane cq == 0 rewrites it
    if (acc.nsplit > 1) {
      if (row < Tq) {
        const int64_t pr = (int64_t)split * acc.part_rows + q_beg + row;
        float* po = acc.part_o + (pr * Hq + hq) * HD;
#pragma unroll
        for (int dn = 0; dn < HD / 8; ++dn)
          *reinterpret_cast<float2*>(po + 8 * dn + 2 * cq) = make_float2(oacc[dn][2 * hh] * inv, oacc[dn][2 * hh + 1] * inv);
        if (cq == 0) acc.part_lse[pr * Hq + hq] = lse_new;
      }
      continue;
    }
    if (row < Tq) {
      if (acc.mode == 1 || acc.mode == 2) {
#pragma unroll
        for (int dn = 0; dn < HD / 8; ++dn) {
          float2* p2 = reinterpret_cast<float2*>(ao + 8 * dn + 2 * cq);
          float2 a2 = acc.mode == 2 ? *p2 : make_float2(0.0f, 0.0f);
          *p2 = make_float2(a2.x * wa + oacc[dn][2 * hh] * wn, a2.y * wa + oacc[dn][2 * hh + 1] * wn);
        }
        if (cq == 0) *al = lse_tot;
      } else {
        half* o_ = o + (int64_t)(q_beg + row) * o_row + (int64_t)hq * o_head;
#pragma unroll
        for (int dn = 0; dn < HD / 8; ++dn) {
          float x0 = oacc[dn][2 * hh] * wn, x1 = oacc[dn][2 * hh + 1] * wn;
          if (acc.mode == 3) {
            const float2 a2 = *reinterpret_cast<const float2*>(ao + 8 * dn + 2 * cq);
            x0 += a2.x * wa; x1 += a2.y * wa;
          }
          *reinterpret_cast<uint32_t*>(o_ + 8 * dn + 2 * cq) = pack_h2(x0, x1);
        }
        if (lse != nullptr && cq == 0) lse[(int64_t)(q_beg + row) * lse_row + (int64_t)hq * lse_head] = lse_tot;
      }
    }
  }
#endif
}

// Merge the S split-KV parts of every (row, head) exactly by LSE, then apply the segment-accumulation mode.
// grid (Tq_total, Hq), 64 threads x 4 dims.
__global__ void __launch_bounds__(64) k1fa_combine_kernel(const float* __restrict__ part_o, const float* __restrict__ part_lse,
                                                        int S, int64_t rows, int Hq, half* __restrict__ o, int64_t o_row,
                                                        int64_t o_head, float* __restrict__ lse, int64_t lse_row,
                                                        int64_t lse_head, AccArgs acc) {
  const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x * 4;
  float mx = -INFINITY;
  for (int s = 0; s < S; ++s) mx = fmaxf(mx, part_lse[((int64_t)s * rows + r) * Hq + h]);
  float den = 0.0f;
  float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
  if (mx != -INFINITY) {
    for (int s = 0; s < S; ++s) {
      const int64_t pr = ((int64_t)s * rows + r) * Hq + h;
      const float w = __expf(part_lse[pr] - mx);
      if (w == 0.0f) continue;
      const float4 x = *reinterpret_cast<const float4*>(part_o + pr * HD + d);
      a.x += w * x.x; a.y += w * x.y; a.z += w * x.z; a.w += w * x.w;
      den += w;
    }
  }
  const float inv = den > 0.0f ? 1.0f / den : 0.0f;
  float lse_new = den > 0.0f ? mx + __logf(den) : -INFINITY;
  float wa = 0.0f, wn = inv;
  float* ao = nullptr;
  float* al = nullptr;
  if (acc.mode != 0) {
    ao = acc.o + (int64_t)r * acc.o_row + (int64_t)h * acc.o_head + d;
    al = acc.lse + (int64_t)r * acc.lse_row + (int64_t)h * acc.lse_head;
    if (acc.mode >= 2) {
      const float la = *al, m2 = fmaxf(la, lse_new);
      if (m2 == -INFINITY) { wa = 0.0f; wn = 0.0f; }
      else {
        const float ea = __expf(la - m2), en = __expf(lse_new - m2), dd = ea + en;
        wa = ea / dd; wn = inv * en / dd; lse_new = m2 + __logf(dd);
      }
    }
  }
  __syncthreads();  // all threads read acc_lse before thread 0 rewrites it
  float4 v = make_float4(a.x * wn, a.y * wn, a.z * wn, a.w * wn);
  if (acc.mode >= 2) {
    const float4 p = *reinterpret_cast<const float4*>(ao);
    v.x += p.x * wa; v.y += p.y * wa; v.z += p.z * wa; v.w += p.w * wa;
  }
  if (acc.mode == 1 || acc.mode == 2) {
    *reinterpret_cast<float4*>(ao) = v;
    if (threadIdx.x == 0) *al = lse_new;
  } else {
    half* op = o + (int64_t)r * o_row + (int64_t)h * o_head + d;
    *reinterpret_cast<uint32_t*>(op) = pack_h2(v.x, v.y);
    *reinterpret_cast<uint32_t*>(op + 2) = pack_h2(v.z, v.w);
    if (lse != nullptr && threadIdx.x == 0) lse[(int64_t)r * lse_row + (int64_t)h * lse_head] = lse_new;
  }
}

template <int BN, bool CAUSAL, bool PV16, bool LAZY, bool QK4, int ABL = 0, int QKC = 0>
static void launch(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& o,
                   float* lse_ptr, int64_t lse_row, int64_t lse_head, const AccArgs& acc,
                   const at::Tensor& cu_q, const at::Tensor& cu_k, int max_q, float scale_log2, float q_prescale = 1.0f) {
  constexpr uint32_t SMEM = 2 * BN * HD * 2 + BM * HD;
  auto kern = k1fa_kernel<BN, CAUSAL, PV16, LAZY, QK4, ABL, QKC>;
  static bool attr_set[64] = {};
  const int dev = q.get_device();
  if (!attr_set[dev]) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    attr_set[dev] = true;
  }
  const int B = cu_q.size(0) - 1;
  const int Hq = q.size(1), Hkv = k.size(1);
  dim3 grid(((max_q + BM - 1) / BM) * acc.nsplit, Hq, B);
  kern<<<grid, NTHREADS, SMEM, at::cuda::getCurrentCUDAStream()>>>(
      (const half*)q.data_ptr(), q.stride(0), q.stride(1),
      (const half*)k.data_ptr(), k.stride(0), k.stride(1),
      (const half*)v.data_ptr(), v.stride(0), v.stride(1),
      (half*)o.data_ptr(), o.stride(0), o.stride(1),
      lse_ptr, lse_row, lse_head, acc,
      cu_q.data_ptr<int>(), cu_k.data_ptr<int>(), Hq, Hkv, QKC > 0 ? 1.4426950408889634f : scale_log2, q_prescale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// variant: bit0 = PV16, bit1 = LAZY, bit2 = QK4 (two interleaved QK accumulator sets); 8/9 = K9 QK16 c64/c32;
// 64*k = timing ablations; bn = 16
void k1fa_fwd(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o, c10::optional<at::Tensor> lse,
              at::Tensor cu_q, at::Tensor cu_k, int64_t max_q, double scale, bool causal, int64_t bn,
              int64_t variant, c10::optional<at::Tensor> acc_o, c10::optional<at::Tensor> acc_lse, int64_t acc_mode,
              int64_t nsplit, int64_t split_len, c10::optional<at::Tensor> part_o, c10::optional<at::Tensor> part_lse) {
  const at::cuda::OptionalCUDAGuard guard(q.device());
  TORCH_CHECK(q.dtype() == at::kHalf && k.dtype() == at::kHalf && v.dtype() == at::kHalf && o.dtype() == at::kHalf,
              "fp16 q/k/v/o");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3 && o.dim() == 3, "3-d tensors");
  TORCH_CHECK(q.size(2) == HD && k.size(2) == HD && v.size(2) == HD && o.size(2) == HD, "head_dim 256");
  TORCH_CHECK(q.stride(2) == 1 && k.stride(2) == 1 && v.stride(2) == 1 && o.stride(2) == 1, "contiguous head dim");
  TORCH_CHECK(q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0 && k.stride(0) % 8 == 0 && k.stride(1) % 8 == 0 &&
                  v.stride(0) % 8 == 0 && v.stride(1) % 8 == 0 && o.stride(0) % 2 == 0 && o.stride(1) % 2 == 0,
              "16-byte aligned strides");
  TORCH_CHECK(q.size(1) % k.size(1) == 0 && v.size(1) == k.size(1) && v.size(0) == k.size(0), "shapes");
  TORCH_CHECK(cu_q.dtype() == at::kInt && cu_k.dtype() == at::kInt && cu_q.numel() == cu_k.numel() && cu_q.is_cuda(),
              "cu_seqlens int32 cuda");
  TORCH_CHECK(o.size(0) == q.size(0) && o.size(1) == q.size(1), "o shape");
  float* lse_ptr = nullptr;
  int64_t lse_row = 0, lse_head = 0;
  if (lse.has_value()) {
    TORCH_CHECK(lse->dtype() == at::kFloat && lse->dim() == 2, "lse fp32 2-d [T, H] view");
    TORCH_CHECK(lse->size(0) == q.size(0) && lse->size(1) == q.size(1), "lse shape");
    lse_ptr = lse->data_ptr<float>();
    lse_row = lse->stride(0);
    lse_head = lse->stride(1);
  }
  AccArgs acc{nullptr, nullptr, 0, 0, 0, 0, (int)acc_mode, 1, 0, nullptr, nullptr, 0};
  if (acc_mode != 0) {
    TORCH_CHECK(acc_mode >= 1 && acc_mode <= 3 && acc_o.has_value() && acc_lse.has_value(), "acc_mode needs acc_o/acc_lse");
    TORCH_CHECK(acc_o->dtype() == at::kFloat && acc_o->dim() == 3 && acc_o->size(0) == q.size(0) &&
                    acc_o->size(1) == q.size(1) && acc_o->size(2) == HD && acc_o->stride(2) == 1 &&
                    acc_o->stride(0) % 2 == 0 && acc_o->stride(1) % 2 == 0, "acc_o fp32 [Tq, Hq, 256]");
    TORCH_CHECK(acc_lse->dtype() == at::kFloat && acc_lse->dim() == 2 && acc_lse->size(0) == q.size(0) &&
                    acc_lse->size(1) == q.size(1), "acc_lse fp32 [Tq, Hq]");
    acc = AccArgs{acc_o->data_ptr<float>(), acc_lse->data_ptr<float>(), acc_o->stride(0), acc_o->stride(1),
                  acc_lse->stride(0), acc_lse->stride(1), (int)acc_mode, 1, 0, nullptr, nullptr, 0};
  }
  if (nsplit > 1) {
    TORCH_CHECK(split_len > 0 && split_len % 16 == 0 && part_o.has_value() && part_lse.has_value(), "split-KV needs part buffers");
    TORCH_CHECK(part_o->dtype() == at::kFloat && part_o->is_contiguous() && part_o->numel() >= nsplit * q.size(0) * q.size(1) * HD &&
                    part_lse->dtype() == at::kFloat && part_lse->is_contiguous() && part_lse->numel() >= nsplit * q.size(0) * q.size(1),
                "part_o fp32 >= [S, Tq, Hq, 256], part_lse fp32 >= [S, Tq, Hq], contiguous");
    acc.nsplit = (int)nsplit;
    acc.split_len = (int)split_len;
    acc.part_o = part_o->data_ptr<float>();
    acc.part_lse = part_lse->data_ptr<float>();
    acc.part_rows = q.size(0);
  }
  if (max_q <= 0) return;
  const float sl2 = (float)(scale * 1.4426950408889634);
#define K1_DISPATCH(C, P, L, Q) \
  launch<16, C, P, L, Q>(q, k, v, o, lse_ptr, lse_row, lse_head, acc, cu_q, cu_k, (int)max_q, sl2)
  TORCH_CHECK(bn == 16, "bn must be 16");
  [&]() {
  const int vv = (int)variant & 7;
#define K1_V(C)                                                  \
  switch (vv) {                                                  \
    case 0: K1_DISPATCH(C, false, false, false); break;          \
    case 1: K1_DISPATCH(C, true, false, false); break;           \
    case 2: K1_DISPATCH(C, false, true, false); break;           \
    case 3: K1_DISPATCH(C, true, true, false); break;            \
    case 4: K1_DISPATCH(C, false, false, true); break;           \
    case 5: K1_DISPATCH(C, true, false, true); break;            \
    case 6: K1_DISPATCH(C, false, true, true); break;            \
    default: K1_DISPATCH(C, true, true, true); break;            \
  }
  if (variant == 8 || variant == 9) {
    // (Lane K9) QK16: v3 base (fp16-slice P.V, lazy rescale) + Q K^T on fp16-accumulate HMMA, promoted to fp32
    // every 64 head dims (8) or 32 (9); Q pre-scaled by the softmax scale. Normwise error ~1e-3: default off.
    if (causal) {
      if (variant == 8) launch<16, true, true, true, false, 0, 64>(q, k, v, o, lse_ptr, lse_row, lse_head, acc, cu_q, cu_k, (int)max_q, sl2, (float)scale);
      else launch<16, true, true, true, false, 0, 32>(q, k, v, o, lse_ptr, lse_row, lse_head, acc, cu_q, cu_k, (int)max_q, sl2, (float)scale);
    } else {
      if (variant == 8) launch<16, false, true, true, false, 0, 64>(q, k, v, o, lse_ptr, lse_row, lse_head, acc, cu_q, cu_k, (int)max_q, sl2, (float)scale);
      else launch<16, false, true, true, false, 0, 32>(q, k, v, o, lse_ptr, lse_row, lse_head, acc, cu_q, cu_k, (int)max_q, sl2, (float)scale);
    }
    return;
  }
  if (variant >= 64) {
    // timing ablations (results are wrong on purpose), causal v7 base: bit3 no global loads, bit4 no QK,
    // bit5 no PV, bit6 no exp, bit7 no barriers
    TORCH_CHECK(causal, "ablations are causal only");
    switch ((int)(variant >> 6)) {
#define K1_ABL(A) case A: launch<16, true, true, true, true, A>(q, k, v, o, lse_ptr, lse_row, lse_head, acc, cu_q, cu_k, (int)max_q, sl2); break;
      K1_ABL(1) K1_ABL(2) K1_ABL(4) K1_ABL(8) K1_ABL(16) K1_ABL(6) K1_ABL(3) K1_ABL(5) K1_ABL(17)
#undef K1_ABL
      default: TORCH_CHECK(false, "unknown ablation");
    }
    return;
  }
  TORCH_CHECK(variant >= 0 && variant <= 7, "variant 0..7, 8/9 (QK16), or 64*k (ablations)");
  if (causal) { K1_V(true) } else { K1_V(false) }
  }();
  if (acc.nsplit > 1) {
    k1fa_combine_kernel<<<dim3((unsigned)q.size(0), (unsigned)q.size(1)), 64, 0, at::cuda::getCurrentCUDAStream()>>>(
        acc.part_o, acc.part_lse, acc.nsplit, acc.part_rows, (int)q.size(1), (half*)o.data_ptr(), o.stride(0), o.stride(1),
        lse_ptr, lse_row, lse_head, acc);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
#undef K1_V
#undef K1_DISPATCH
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fwd", &k1fa_fwd, "K1 fa75 prefill attention (sm_75, hd256)"); }
