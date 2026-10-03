// Lane K9: f16-accumulate vs f32-accumulate tensor-core rate on sm_75, per SM per clock (clock64) + wall clock.
// Shapes: m16n8k8 (native Turing HMMA.1688) and m8n8k4 (Volta shape, HMMA.884 on sm_75).
// Build: /usr/local/cuda-13/bin/nvcc -O3 -arch=sm_75 -o mma_acc mma_acc.cu
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#define NCH 8
#define ITERS 65536
__device__ __forceinline__ void k8_h(uint32_t* c, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
               : "+r"(c[0]), "+r"(c[1]) : "r"(a[0]), "r"(a[1]), "r"(b[0]));
}
__device__ __forceinline__ void k8_f(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(b[0]));
}
// m8n8k4: per-thread A 2 regs, B 2 regs, C f16 4 regs / f32 8 regs
__device__ __forceinline__ void k4_h(uint32_t* c, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void k4_f(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
               : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]));
}
template <int ARM>
__global__ void roof(long long* clk, int* sink, uint32_t seed, long long* gt) {
  uint32_t a[2] = {0x3c003c00u ^ (seed & 0x00ff00ffu), 0x3c003c00u};  // small finite halves
  uint32_t b[2] = {0x3c003c00u, 0x38003800u};
  uint32_t ch[NCH][4]; float cf[NCH][8];
  for (int i = 0; i < NCH; i++) { for (int j = 0; j < 4; j++) ch[i][j] = 0; for (int j = 0; j < 8; j++) cf[i][j] = 0.f; }
  __syncthreads();
  long long g0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  long long t0 = clock64();
#pragma unroll 1
  for (int it = 0; it < ITERS; it++) {
#pragma unroll
    for (int i = 0; i < NCH; i++) {
      if (ARM == 0) k8_h(ch[i], a, b);
      if (ARM == 1) k8_f(cf[i], a, b);
      if (ARM == 2) k4_h(ch[i], a, b);
      if (ARM == 3) k4_f(cf[i], a, b);
    }
  }
  __syncthreads();
  long long t1 = clock64();
  long long g1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
  int s = 0;
  for (int i = 0; i < NCH; i++) { for (int j = 0; j < 4; j++) s += (int)ch[i][j]; for (int j = 0; j < 8; j++) s += (int)cf[i][j]; }
  if (s == 0x7fffffff) sink[0] = s;
  if (threadIdx.x == 0) { clk[blockIdx.x] = t1 - t0; gt[blockIdx.x] = g1 - g0; }
}
int main(int argc, char** argv) {
  int reps = argc > 1 ? atoi(argv[1]) : 5;
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  int nsm = p.multiProcessorCount;
  const char* names[] = {"m16n8k8 f16acc", "m16n8k8 f32acc", "m8n8k4  f16acc", "m8n8k4  f32acc"};
  const double macs[] = {16 * 8 * 8, 16 * 8 * 8, 4 * 8 * 8 * 4 /* 4 quadpairs x 8x8x4 per warp */, 4 * 8 * 8 * 4};
  int threads = 256, blocks = nsm;
  long long* dclk; int* dsink; cudaMalloc(&dclk, blocks * sizeof(long long)); cudaMalloc(&dsink, 4);
  long long* hclk = new long long[blocks]; long long* dgt; cudaMalloc(&dgt, blocks*8); long long* hgt = new long long[blocks];
  printf("%s SMs=%d\n", p.name, nsm);
  for (int rep = 0; rep < reps + 1; rep++) {
    for (int arm = 0; arm < 4; arm++) {
      cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
      cudaEventRecord(e0);
      switch (arm) {
        case 0: roof<0><<<blocks, threads>>>(dclk, dsink, rep, dgt); break;
        case 1: roof<1><<<blocks, threads>>>(dclk, dsink, rep, dgt); break;
        case 2: roof<2><<<blocks, threads>>>(dclk, dsink, rep, dgt); break;
        case 3: roof<3><<<blocks, threads>>>(dclk, dsink, rep, dgt); break;
      }
      cudaEventRecord(e1); cudaEventSynchronize(e1);
      float ms; cudaEventElapsedTime(&ms, e0, e1);
      cudaMemcpy(hclk, dclk, blocks * sizeof(long long), cudaMemcpyDeviceToHost);
      cudaMemcpy(hgt, dgt, blocks*8, cudaMemcpyDeviceToHost); double mhz=0; for(int i=0;i<blocks;i++) mhz += (double)hclk[i]/hgt[i]*1e3; mhz/=blocks;
      long long mn = hclk[0], mx = hclk[0]; for (int i = 1; i < blocks; i++) { if (hclk[i] < mn) mn = hclk[i]; if (hclk[i] > mx) mx = hclk[i]; }
      double mma_per_sm = 8.0 * ITERS * NCH;  // 8 warps/block, 1 block/SM
      double mac_clk_sm = mma_per_sm * macs[arm] / (double)mn;
      double wall_tf = 2.0 * mma_per_sm * nsm * macs[arm] / (ms * 1e-3) / 1e12;
      if (rep > 0) printf("rep%d %s  MAC/clk/SM=%.1f (best SM) worst=%.1f  wall=%.2f ms  wallTFLOPS=%.1f  impliedTF@1545=%.1f  SMclk=%.0fMHz  TF@SMclk=%.1f\n",
                          rep, names[arm], mac_clk_sm, mma_per_sm * macs[arm] / (double)mx, ms, wall_tf, 2 * mac_clk_sm * nsm * 1.545e9 / 1e12, mhz, 2*mac_clk_sm*nsm*mhz*1e6/1e12);
    }
  }
  return 0;
}
