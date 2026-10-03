// Lane K7: silicon roof of Turing (sm_75) tensor-core datapaths, measured per SM per clock.
// Arms: fp16 m16n8k8 (f16 acc), fp16 m16n8k8 (f32 acc), int8 m8n8k16 (s32 acc), int4 m8n8k32 (s32 acc).
// Each warp runs NCH independent accumulator chains; per-block clock64() deltas give MACs/clk/SM that are
// independent of time-slicing with a co-resident process (the live engine), plus wall-clock ops/s for reference.
// Build: /usr/local/cuda-13/bin/nvcc -O3 -arch=sm_75 -o mma_roof mma_roof.cu
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define NCH 8
#define ITERS 4096

__device__ __forceinline__ void mma_f16f16(uint32_t* c, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
               : "+r"(c[0]), "+r"(c[1]) : "r"(a[0]), "r"(a[1]), "r"(b[0]));
}
__device__ __forceinline__ void mma_f16f32(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(b[0]));
}
__device__ __forceinline__ void mma_s8(int* c, uint32_t a, uint32_t b) {
  asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
               : "+r"(c[0]), "+r"(c[1]) : "r"(a), "r"(b));
}
__device__ __forceinline__ void mma_s4(int* c, uint32_t a, uint32_t b) {
  asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
               : "+r"(c[0]), "+r"(c[1]) : "r"(a), "r"(b));
}
__device__ __forceinline__ void mma_s4u4(int* c, uint32_t a, uint32_t b) {
  asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.u4.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
               : "+r"(c[0]), "+r"(c[1]) : "r"(a), "r"(b));
}

// MACs per mma: f16 m16n8k8 = 1024; s8 m8n8k16 = 1024; s4 m8n8k32 = 2048
template <int ARM>
__global__ void roof(long long* clk, int* sink, uint32_t seed) {
  uint32_t a[2] = {seed ^ threadIdx.x, seed * 3u + threadIdx.x};
  uint32_t b[1] = {seed * 7u ^ (threadIdx.x << 3)};
  uint32_t ch[NCH][4];
  float cf[NCH][4];
  for (int i = 0; i < NCH; i++) for (int j = 0; j < 4; j++) { ch[i][j] = 0; cf[i][j] = 0.f; }
  __syncthreads();
  long long t0 = clock64();
#pragma unroll 1
  for (int it = 0; it < ITERS; it++) {
#pragma unroll
    for (int i = 0; i < NCH; i++) {
      if (ARM == 0) mma_f16f16(ch[i], a, b);
      if (ARM == 1) mma_f16f32(cf[i], a, b);
      if (ARM == 2) mma_s8((int*)ch[i], a[0], b[0]);
      if (ARM == 3) mma_s4((int*)ch[i], a[0], b[0]);
      if (ARM == 4) mma_s4u4((int*)ch[i], a[0], b[0]);
    }
  }
  __syncthreads();
  long long t1 = clock64();
  int s = 0;
  for (int i = 0; i < NCH; i++) for (int j = 0; j < 4; j++) s += (int)ch[i][j] + (int)cf[i][j];
  if (s == 0x7fffffff) sink[0] = s;
  if (threadIdx.x == 0) clk[blockIdx.x] = t1 - t0;
}

int main() {
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  int nsm = p.multiProcessorCount;
  int clk_khz = 0; cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0);
  const char* names[] = {"fp16 m16n8k8 f16acc", "fp16 m16n8k8 f32acc", "int8 m8n8k16 s32acc", "int4 m8n8k32 s32acc", "s4xu4 m8n8k32 s32acc"};
  const double macs_per_mma[] = {1024, 1024, 1024, 2048, 2048};
  int warps_per_block = 8, threads = 32 * warps_per_block;
  int blocks = nsm;  // one block per SM (8 warps = 2 per SMSP)
  long long* dclk; int* dsink; cudaMalloc(&dclk, blocks * sizeof(long long)); cudaMalloc(&dsink, 4);
  long long* hclk = new long long[blocks];
  printf("%s  SMs=%d  boost clk=%d MHz\n", p.name, nsm, clk_khz / 1000);
  for (int rep = 0; rep < 3; rep++) {
    for (int arm = 0; arm < 5; arm++) {
      cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
      cudaEventRecord(e0);
      switch (arm) {
        case 0: roof<0><<<blocks, threads>>>(dclk, dsink, 1234u + rep); break;
        case 1: roof<1><<<blocks, threads>>>(dclk, dsink, 1234u + rep); break;
        case 2: roof<2><<<blocks, threads>>>(dclk, dsink, 1234u + rep); break;
        case 3: roof<3><<<blocks, threads>>>(dclk, dsink, 1234u + rep); break;
        case 4: roof<4><<<blocks, threads>>>(dclk, dsink, 1234u + rep); break;
      }
      cudaEventRecord(e1); cudaEventSynchronize(e1);
      float ms; cudaEventElapsedTime(&ms, e0, e1);
      cudaMemcpy(hclk, dclk, blocks * sizeof(long long), cudaMemcpyDeviceToHost);
      double mean_clk = 0; long long mn = hclk[0];
      for (int i = 0; i < blocks; i++) { mean_clk += hclk[i]; if (hclk[i] < mn) mn = hclk[i]; }
      mean_clk /= blocks;
      double macs_block = (double)warps_per_block * ITERS * NCH * macs_per_mma[arm];
      double macs_per_clk_sm = macs_block / (double)mn;  // best block = least preempted
      double tops_at_boost = 2.0 * macs_per_clk_sm * nsm * clk_khz * 1e3 / 1e12;
      double wall_tops = 2.0 * macs_block * blocks / (ms * 1e-3) / 1e12;
      printf("rep%d %-22s MAC/clk/SM %7.1f (min-clk block)  -> %6.1f T(op)/s @boost   wall %6.1f T/s (shared GPU)  "
             "clk spread mean/min %.2f\n",
             rep, names[arm], macs_per_clk_sm, tops_at_boost, wall_tops, mean_clk / mn);
    }
  }
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { printf("CUDA error %s\n", cudaGetErrorString(e)); return 1; }
  return 0;
}
