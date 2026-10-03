// Lane K9 (L102), 2026-10-03: chunked gated-delta-rule prefill forward on Turing (sm_75) tensor cores.
//
// Replaces the FlashQLA-legacy token recurrence (fp32 CUDA cores, 3,632 dependent steps per chunk of prompt) with the
// UT/WY chunk form (chunk C = 64): per chunk only 64-row matrix products on HMMA (mma.sync m16n8k8) plus a 64-step
// forward substitution on NC state columns. Same conventions as FlashQLA legacy / vLLM:
//   state M [dk, dv] stored as [N, Hv, dv, dk] fp32; per token kv = M^T k; delta = beta (v - e^g kv);
//   M = e^g M + k delta^T; o = scale M^T q. q, k: [T, Hk, 128] fp16 (l2-normalised upstream), v: [T, Hv, 128] fp16,
//   g (log decay), beta: [T, Hv] fp32, packed varlen via cu_seqlens.
// Chunk algebra (tools/k9/gdn_chunk_ref.py, exact vs the recurrence to 5e-16):
//   G = cumsum_chunk(g); L_ji = beta_j e^{G_j-G_i} k_j.k_i (i<j); R = beta (V - e^G (K M0)); U = (I+L)^{-1} R;
//   O = scale (e^G (Q M0) + P U), P_ti = e^{G_t-G_i} q_t.k_i (i<=t); M1 = e^{G_C} M0 + K^T (e^{G_C-G} U).
// Block = 4 warps, one (sequence, v-head, NC-column slice of the state). The state slice lives in fp32 mma
// accumulators across all chunks (warp w owns dk rows 32w..32w+31). F16QK: Q K^T and K K^T accumulate in fp16 (full
// HMMA rate on GeForce; safe because |q.k| <= 1 for unit vectors), everything else accumulates in fp32.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define D 128
#define C 64
#define NW 4
#define NT (NW * 32)

static __device__ __forceinline__ void mma_f32(float* c, uint32_t a0, uint32_t a1, uint32_t b) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(b));
}
static __device__ __forceinline__ void mma_f16(uint32_t* c, uint32_t a0, uint32_t a1, uint32_t b) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
               : "+r"(c[0]), "+r"(c[1]) : "r"(a0), "r"(a1), "r"(b));
}
static __device__ __forceinline__ void ldsm4(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
static __device__ __forceinline__ void ldsm4t(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
static __device__ __forceinline__ uint32_t pack2(float a, float b) {
  half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}
// K tile [64 rows][128 halves] = 16 chunks of 16 B per row, chunk ^= row & 7
static __device__ __forceinline__ uint32_t koff(int r, int ch) { return (uint32_t)(r * 256 + ((ch ^ (r & 7)) * 16)); }
// [rows][NC halves] tiles (M0 fp16, U, U'): NC/8 chunks per row, chunk ^= (row >> 1) & (NC/8 - 1)
template <int NC>
static __device__ __forceinline__ uint32_t noff(int r, int ch) {
  constexpr int NCH = NC / 8;
  return (uint32_t)(r * NC * 2 + ((ch ^ ((r >> 1) & (NCH - 1))) * 16));
}

template <int NC, bool F16QK>
__global__ void __launch_bounds__(NT, 1) k9_gdn_chunk_fwd(
    const half* __restrict__ q, int64_t q_row, int64_t q_head,
    const half* __restrict__ k, int64_t k_row, int64_t k_head,
    const half* __restrict__ v, int64_t v_row, int64_t v_head,
    const float* __restrict__ g, const float* __restrict__ beta, int64_t gb_row,  // [T, Hv] (head stride 1)
    const float* __restrict__ state_in, float* __restrict__ state_out,  // [N, Hv, dv, dk]
    half* __restrict__ o, int64_t o_row, int64_t o_head,
    const int* __restrict__ cu, int Hk, int Hv, float scale) {
  static_assert(NC == 32, "this version: NC = 32 (4 warps x 8 columns in the substitution)");
  constexpr int NT8 = NC / 8;           // n8 tiles over the column slice
  constexpr uint32_t SK = 0;            // K tile, 16384 B
  constexpr uint32_t SL = SK + 16384;   // L fp32 [64][65], 16640 B
  constexpr uint32_t SM = SL + 64 * 65 * 4;  // M0 fp16 [128][NC] / R fp32 [64][NC] (union), 8192 B
  constexpr uint32_t SU = SM + 8192;    // U fp16 [64][NC]
  constexpr uint32_t SU2 = SU + 64 * NC * 2;  // U' = e^{G_C-G} U fp16
  constexpr uint32_t SG = SU2 + 64 * NC * 2;  // G fp32 [64], beta fp32 [64]
  extern __shared__ __align__(128) uint8_t sm[];
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(sm);
  float* sL = reinterpret_cast<float*>(sm + SL);
  float* sR = reinterpret_cast<float*>(sm + SM);
  float* sG = reinterpret_cast<float*>(sm + SG);
  float* sB = sG + 64;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int gq = lane >> 2, cq = lane & 3, lr = lane & 7, lm = lane >> 3;
  const int nsl = D / NC;
  const int hv = blockIdx.x / nsl, cs = blockIdx.x % nsl;
  const int hq = hv / (Hv / Hk);
  const int b = blockIdx.y;
  const int t0 = cu[b], t1 = cu[b + 1];
  const int c0 = cs * NC;

  // state slice M0 [dk rows 32w + 16mt + (gq | gq+8)][cols 8nt + 2cq (+1)] in fp32 accumulators
  float Mr[2][NT8][4];
  {
    const float* st = state_in + ((int64_t)b * Hv + hv) * D * D;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int d = 32 * warp + 16 * mt + gq + (e >> 1) * 8, c = c0 + 8 * nt + 2 * cq + (e & 1);
          Mr[mt][nt][e] = st[(int64_t)c * D + d];
        }
  }

  for (int s = t0; s < t1; s += C) {
    const int n = min(C, t1 - s);
    // ---- stage: K tile, g/beta -> G, M0 fp16 ----
#pragma unroll
    for (int j = 0; j < (C * 16) / NT; ++j) {
      const int idx = tid + j * NT, r = idx >> 4, ch = idx & 15;
      uint4 val = make_uint4(0, 0, 0, 0);
      if (r < n) val = *reinterpret_cast<const uint4*>(k + (int64_t)(s + r) * k_row + (int64_t)hq * k_head + ch * 8);
      *reinterpret_cast<uint4*>(sm + SK + koff(r, ch)) = val;
    }
    if (warp == 0) {
      float ga = 0.f, gb2 = 0.f, ba = 0.f, bb = 0.f;
      const int r0 = 2 * lane, r1 = 2 * lane + 1;
      if (r0 < n) { ga = g[(int64_t)(s + r0) * gb_row + hv]; ba = beta[(int64_t)(s + r0) * gb_row + hv]; }
      if (r1 < n) { gb2 = g[(int64_t)(s + r1) * gb_row + hv]; bb = beta[(int64_t)(s + r1) * gb_row + hv]; }
      float x = ga + gb2;
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        const float y = __shfl_up_sync(0xffffffffu, x, off);
        if (lane >= off) x += y;
      }
      sG[r1] = x; sG[r0] = x - gb2; sB[r0] = ba; sB[r1] = bb;
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT8; ++nt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int d = 32 * warp + 16 * mt + gq + hh * 8;
          *reinterpret_cast<uint32_t*>(sm + SM + noff<NC>(d, nt) + 4 * cq) = pack2(Mr[mt][nt][2 * hh], Mr[mt][nt][2 * hh + 1]);
        }
    __syncthreads();

    const int ra = 16 * warp + gq, rb = ra + 8;  // this thread's chunk rows
    const float Ga = sG[ra], Gb = sG[rb], Ba = sB[ra], Bb = sB[rb];
    const float GC = sG[C - 1];

    // ---- Q fragments (rows 16w..16w+15) straight from global ----
    uint32_t qa[D / 8][2];
    {
      const half* qp = q + (int64_t)hq * q_head;
#pragma unroll
      for (int kk = 0; kk < D / 8; ++kk) {
        qa[kk][0] = ra < n ? *reinterpret_cast<const uint32_t*>(qp + (int64_t)(s + ra) * q_row + 8 * kk + 2 * cq) : 0u;
        qa[kk][1] = rb < n ? *reinterpret_cast<const uint32_t*>(qp + (int64_t)(s + rb) * q_row + 8 * kk + 2 * cq) : 0u;
      }
    }
    // ---- S_qk = Q K^T, S_kk = K K^T (16 x 64 per warp) ----
    float sqk[8][4], skk[8][4];
    if (F16QK) {
      uint32_t hq2[8][2] = {}, hk2[8][2] = {};
#pragma unroll
      for (int kk = 0; kk < D / 8; kk += 2) {
        uint32_t ka[4];
        ldsm4(ka, sb + SK + koff(16 * warp + lr + (lm & 1) * 8, kk + (lm >> 1)));
#pragma unroll
        for (int np = 0; np < 4; ++np) {
          uint32_t bb[4];
          ldsm4(bb, sb + SK + koff(16 * np + 8 * (lm & 1) + lr, kk + (lm >> 1)));
          mma_f16(hq2[2 * np], qa[kk][0], qa[kk][1], bb[0]);
          mma_f16(hq2[2 * np + 1], qa[kk][0], qa[kk][1], bb[1]);
          mma_f16(hq2[2 * np], qa[kk + 1][0], qa[kk + 1][1], bb[2]);
          mma_f16(hq2[2 * np + 1], qa[kk + 1][0], qa[kk + 1][1], bb[3]);
          mma_f16(hk2[2 * np], ka[0], ka[1], bb[0]);
          mma_f16(hk2[2 * np + 1], ka[0], ka[1], bb[1]);
          mma_f16(hk2[2 * np], ka[2], ka[3], bb[2]);
          mma_f16(hk2[2 * np + 1], ka[2], ka[3], bb[3]);
        }
      }
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        float2 a = __half22float2(*reinterpret_cast<half2*>(&hq2[nt][0])), c2 = __half22float2(*reinterpret_cast<half2*>(&hq2[nt][1]));
        sqk[nt][0] = a.x; sqk[nt][1] = a.y; sqk[nt][2] = c2.x; sqk[nt][3] = c2.y;
        a = __half22float2(*reinterpret_cast<half2*>(&hk2[nt][0])); c2 = __half22float2(*reinterpret_cast<half2*>(&hk2[nt][1]));
        skk[nt][0] = a.x; skk[nt][1] = a.y; skk[nt][2] = c2.x; skk[nt][3] = c2.y;
      }
    } else {
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) { sqk[nt][e] = 0.f; skk[nt][e] = 0.f; }
#pragma unroll
      for (int kk = 0; kk < D / 8; kk += 2) {
        uint32_t ka[4];
        ldsm4(ka, sb + SK + koff(16 * warp + lr + (lm & 1) * 8, kk + (lm >> 1)));
#pragma unroll
        for (int np = 0; np < 4; ++np) {
          uint32_t bb[4];
          ldsm4(bb, sb + SK + koff(16 * np + 8 * (lm & 1) + lr, kk + (lm >> 1)));
          mma_f32(sqk[2 * np], qa[kk][0], qa[kk][1], bb[0]);
          mma_f32(sqk[2 * np + 1], qa[kk][0], qa[kk][1], bb[1]);
          mma_f32(sqk[2 * np], qa[kk + 1][0], qa[kk + 1][1], bb[2]);
          mma_f32(sqk[2 * np + 1], qa[kk + 1][0], qa[kk + 1][1], bb[3]);
          mma_f32(skk[2 * np], ka[0], ka[1], bb[0]);
          mma_f32(skk[2 * np + 1], ka[0], ka[1], bb[1]);
          mma_f32(skk[2 * np], ka[2], ka[3], bb[2]);
          mma_f32(skk[2 * np + 1], ka[2], ka[3], bb[3]);
        }
      }
    }
    // P (decayed causal Q K^T) -> fp16 A fragments; L (strictly lower, beta-scaled decayed K K^T) -> smem fp32
    uint32_t pa[8][2];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      float p[4], l[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int row = (e < 2) ? ra : rb, col = 8 * nt + 2 * cq + (e & 1);
        const float Gr = (e < 2) ? Ga : Gb, Br = (e < 2) ? Ba : Bb;
        const float dec = __expf(fminf(Gr - sG[col], 0.f));
        p[e] = col <= row ? dec * sqk[nt][e] : 0.f;
        l[e] = col < row ? Br * dec * skk[nt][e] : 0.f;
      }
      pa[nt][0] = pack2(p[0], p[1]); pa[nt][1] = pack2(p[2], p[3]);
      sL[ra * 65 + 8 * nt + 2 * cq] = l[0]; sL[ra * 65 + 8 * nt + 2 * cq + 1] = l[1];
      sL[rb * 65 + 8 * nt + 2 * cq] = l[2]; sL[rb * 65 + 8 * nt + 2 * cq + 1] = l[3];
    }
    // ---- Y = K_w M0, QM = Q_w M0 (16 x NC per warp), fp32 accumulate ----
    float y[NT8][4], qm[NT8][4];
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) { y[nt][e] = 0.f; qm[nt][e] = 0.f; }
#pragma unroll
    for (int kk = 0; kk < D / 8; kk += 2) {
      uint32_t ka[4];
      ldsm4(ka, sb + SK + koff(16 * warp + lr + (lm & 1) * 8, kk + (lm >> 1)));
#pragma unroll
      for (int np = 0; np < NT8 / 2; ++np) {
        uint32_t bm[4];  // (kk, 2np), (kk+1, 2np), (kk, 2np+1), (kk+1, 2np+1)
        ldsm4t(bm, sb + SM + noff<NC>(8 * (kk + (lm & 1)) + lr, 2 * np + (lm >> 1)));
        mma_f32(y[2 * np], ka[0], ka[1], bm[0]);
        mma_f32(y[2 * np], ka[2], ka[3], bm[1]);
        mma_f32(y[2 * np + 1], ka[0], ka[1], bm[2]);
        mma_f32(y[2 * np + 1], ka[2], ka[3], bm[3]);
        mma_f32(qm[2 * np], qa[kk][0], qa[kk][1], bm[0]);
        mma_f32(qm[2 * np], qa[kk + 1][0], qa[kk + 1][1], bm[1]);
        mma_f32(qm[2 * np + 1], qa[kk][0], qa[kk][1], bm[2]);
        mma_f32(qm[2 * np + 1], qa[kk + 1][0], qa[kk + 1][1], bm[3]);
      }
    }
    // R = beta (V - e^G Y); O starts as e^G QM
    float rr[NT8][4];
    {
      const half* vp = v + (int64_t)hv * v_head + c0;
      const float ea = __expf(Ga), eb = __expf(Gb);
#pragma unroll
      for (int nt = 0; nt < NT8; ++nt) {
        float2 va = make_float2(0.f, 0.f), vb = make_float2(0.f, 0.f);
        if (ra < n) va = __half22float2(*reinterpret_cast<const half2*>(vp + (int64_t)(s + ra) * v_row + 8 * nt + 2 * cq));
        if (rb < n) vb = __half22float2(*reinterpret_cast<const half2*>(vp + (int64_t)(s + rb) * v_row + 8 * nt + 2 * cq));
        rr[nt][0] = Ba * (va.x - ea * y[nt][0]); rr[nt][1] = Ba * (va.y - ea * y[nt][1]);
        rr[nt][2] = Bb * (vb.x - eb * y[nt][2]); rr[nt][3] = Bb * (vb.y - eb * y[nt][3]);
        qm[nt][0] *= ea; qm[nt][1] *= ea; qm[nt][2] *= eb; qm[nt][3] *= eb;
      }
    }
    __syncthreads();  // all warps done reading M0 fp16 and writing L
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt) {
      sR[ra * NC + 8 * nt + 2 * cq] = rr[nt][0]; sR[ra * NC + 8 * nt + 2 * cq + 1] = rr[nt][1];
      sR[rb * NC + 8 * nt + 2 * cq] = rr[nt][2]; sR[rb * NC + 8 * nt + 2 * cq + 1] = rr[nt][3];
    }
    __syncthreads();
    // ---- forward substitution U = (I + L)^{-1} R: warp w owns columns 8w..8w+7, 4 lanes per column ----
    {
      const int col = 8 * warp + (lane >> 2), p = lane & 3;
      float ur[16];
#pragma unroll
      for (int m = 0; m < 16; ++m) ur[m] = 0.f;
#pragma unroll
      for (int j = 0; j < C; ++j) {
        float part = 0.f;
#pragma unroll
        for (int m = 0; m < 16; ++m)
          if (4 * m < j) {  // compile-time prune; i = 4m + p < j checked at run time
            const int i = 4 * m + p;
            if (i < j) part = fmaf(sL[j * 65 + i], ur[m], part);
          }
        part += __shfl_xor_sync(0xffffffffu, part, 1);
        part += __shfl_xor_sync(0xffffffffu, part, 2);
        const float uj = sR[j * NC + col] - part;
        if (p == (j & 3)) ur[j >> 2] = uj;
        if (p == 0) {
          *reinterpret_cast<half*>(sm + SU + noff<NC>(j, col >> 3) + 2 * (col & 7)) = __float2half_rn(uj);
          *reinterpret_cast<half*>(sm + SU2 + noff<NC>(j, col >> 3) + 2 * (col & 7)) = __float2half_rn(__expf(GC - sG[j]) * uj);
        }
      }
    }
    __syncthreads();
    // ---- O = scale (e^G QM + P U) ----
#pragma unroll
    for (int kk = 0; kk < C / 8; kk += 2) {
#pragma unroll
      for (int np = 0; np < NT8 / 2; ++np) {
        uint32_t bu[4];
        ldsm4t(bu, sb + SU + noff<NC>(8 * (kk + (lm & 1)) + lr, 2 * np + (lm >> 1)));
        mma_f32(qm[2 * np], pa[kk][0], pa[kk][1], bu[0]);
        mma_f32(qm[2 * np], pa[kk + 1][0], pa[kk + 1][1], bu[1]);
        mma_f32(qm[2 * np + 1], pa[kk][0], pa[kk][1], bu[2]);
        mma_f32(qm[2 * np + 1], pa[kk + 1][0], pa[kk + 1][1], bu[3]);
      }
    }
    {
      half* op = o + (int64_t)hv * o_head + c0;
#pragma unroll
      for (int nt = 0; nt < NT8; ++nt) {
        if (ra < n) *reinterpret_cast<uint32_t*>(op + (int64_t)(s + ra) * o_row + 8 * nt + 2 * cq) = pack2(scale * qm[nt][0], scale * qm[nt][1]);
        if (rb < n) *reinterpret_cast<uint32_t*>(op + (int64_t)(s + rb) * o_row + 8 * nt + 2 * cq) = pack2(scale * qm[nt][2], scale * qm[nt][3]);
      }
    }
    // ---- M = e^{G_C} M + K^T U' (warp w: dk rows 32w..32w+31) ----
    {
      const float eC = __expf(GC);
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT8; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e) Mr[mt][nt][e] *= eC;
#pragma unroll
      for (int kp = 0; kp < C / 16; ++kp) {
        uint32_t bu[NT8 / 2][4];
#pragma unroll
        for (int np = 0; np < NT8 / 2; ++np)
          ldsm4t(bu[np], sb + SU2 + noff<NC>(8 * (2 * kp + (lm & 1)) + lr, 2 * np + (lm >> 1)));
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          uint32_t at[4];  // (d 0-7, i 0-7) a0(k0), (d 8-15, i 0-7) a1(k0), (d 0-7, i 8-15) a0(k1), (d 8-15, i 8-15) a1(k1)
          ldsm4t(at, sb + SK + koff(16 * kp + 8 * (lm >> 1) + lr, (32 * warp + 16 * mt) / 8 + (lm & 1)));
#pragma unroll
          for (int np = 0; np < NT8 / 2; ++np) {
            mma_f32(Mr[mt][2 * np], at[0], at[1], bu[np][0]);
            mma_f32(Mr[mt][2 * np], at[2], at[3], bu[np][1]);
            mma_f32(Mr[mt][2 * np + 1], at[0], at[1], bu[np][2]);
            mma_f32(Mr[mt][2 * np + 1], at[2], at[3], bu[np][3]);
          }
        }
      }
    }
    __syncthreads();  // smem reused by the next chunk
  }
  {
    float* st = state_out + ((int64_t)b * Hv + hv) * D * D;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int d = 32 * warp + 16 * mt + gq + (e >> 1) * 8, c = c0 + 8 * nt + 2 * cq + (e & 1);
          st[(int64_t)c * D + d] = Mr[mt][nt][e];
        }
  }
}

// q, k [T, Hk, 128] fp16; v [T, Hv, 128] fp16; g, beta [T, Hv] fp32; cu [N+1] int32; state_in [N, Hv, 128, 128] fp32
// returns (o [T, Hv, 128] fp16, state_out [N, Hv, 128, 128] fp32)
std::vector<at::Tensor> k9_gdn_fwd(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor g, at::Tensor beta,
                                   at::Tensor cu, at::Tensor state_in, double scale, bool f16qk) {
  const at::cuda::OptionalCUDAGuard guard(q.device());
  TORCH_CHECK(q.dtype() == at::kHalf && k.dtype() == at::kHalf && v.dtype() == at::kHalf, "fp16 q/k/v");
  TORCH_CHECK(g.dtype() == at::kFloat && beta.dtype() == at::kFloat && state_in.dtype() == at::kFloat, "fp32 g/beta/state");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3 && q.size(2) == D && v.size(2) == D, "[T, H, 128]");
  TORCH_CHECK(q.stride(2) == 1 && k.stride(2) == 1 && v.stride(2) == 1, "contiguous head dim");
  TORCH_CHECK(q.stride(0) % 8 == 0 && k.stride(0) % 8 == 0 && k.stride(1) % 8 == 0 && v.stride(0) % 2 == 0, "aligned strides");
  TORCH_CHECK(g.is_contiguous() && beta.is_contiguous() && g.size(1) == v.size(1), "g/beta [T, Hv] contiguous");
  TORCH_CHECK(cu.dtype() == at::kInt && state_in.is_contiguous(), "cu int32, contiguous state");
  const int Hk = q.size(1), Hv = v.size(1), N = cu.size(0) - 1;
  TORCH_CHECK(Hv % Hk == 0 && state_in.size(0) == N && state_in.size(1) == Hv, "shapes");
  auto o = at::empty({v.size(0), Hv, D}, v.options());
  auto st = at::empty_like(state_in);
  if (N <= 0) return {o, st};
  constexpr int NC = 32;
  constexpr uint32_t SMEM = 16384 + 64 * 65 * 4 + 8192 + 2 * 64 * NC * 2 + 512;
  auto kern = f16qk ? k9_gdn_chunk_fwd<NC, true> : k9_gdn_chunk_fwd<NC, false>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
  dim3 grid(Hv * (D / NC), N);
  kern<<<grid, NT, SMEM, at::cuda::getCurrentCUDAStream()>>>(
      (const half*)q.data_ptr(), q.stride(0), q.stride(1), (const half*)k.data_ptr(), k.stride(0), k.stride(1),
      (const half*)v.data_ptr(), v.stride(0), v.stride(1), g.data_ptr<float>(), beta.data_ptr<float>(), g.stride(0),
      state_in.data_ptr<float>(), st.data_ptr<float>(), (half*)o.data_ptr(), o.stride(0), o.stride(1),
      cu.data_ptr<int>(), Hk, Hv, (float)scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {o, st};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fwd", &k9_gdn_fwd, "K9 chunked GDN prefill forward (sm_75 HMMA)"); }
