// E16 P5: tree-step commit for the Qwen3.8 hybrid (GDN + TurboQuant) target, sm_75 (2026-10-13).
//
// Contract: /home/kevin/projects/lanes/coexec/results/tree-contract.md §5 P5 and §6 (TreeStash).
// Runs after the tree sampler/walk and BEFORE postprocess_mamba_align_gpu.  For every request b whose walk rescued
// leaf l (rescue_leaf[b] = l >= 0) at depth j = leaf_depth[l] (leaf row R = W + l, W = D + 1 chain rows):
//   GDN temporal, every GDN layer g, every v-head:  S[idx[b, j]] = update(S[idx[b, j-1]], stash.gdn_leaf[g, b, l])
//       with the K5 single-pass algebra  hk = S.k;  delta = beta * (v - decay * hk);  S' = decay * S + delta k^T
//       (no q: the leaf's output was already produced by the verify kernel).
//   GDN conv, every GDN layer:  conv[idx[b, 0]][:, col] = stash.conv_leaf_x[g, b, l, :],  col = (Kw-2) + j
//       = conv_len - W + j (c_j's column of the stored window, dead once c_j is rejected).
//   TurboQuant KV, every attention layer:  packed row of slot_mapping[b*M + j] := packed row of slot_mapping[b*M + R]
//       (all Hk heads; TQ quantisation is per token, so the copy is bit-exact).
// Requests with rescue_leaf < 0 (and graph-padded rows) do nothing, so the launch is graph-safe and may run
// unconditionally on every tree step.  After it, num_accepted = j + 1 means exactly what it means for a chain.
//
// One launch: grid (B, NG*HV + NG*conv_blocks + NT), 256 threads.  Layer tables are device int64 arrays built ONCE
// at init (no lazy state, see HACKATHON-RULES 08:4x); the static leaf depths are passed by value.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <vector>

namespace {

constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int kStashW = kDimK + kDimV + 2;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxLeaves = 16;
constexpr int kConvChunk = 1024;  // channels per conv CTA

// GDN layer table row (int64): ssm ptr, ssm slot stride (elements), num slots, conv ptr, conv slot / channel /
// column strides (elements), idx ptr (int32 rows of W).
constexpr int kGdnTab = 8;
// TQ layer table row (int64): ptr (bytes), block / pos / head strides (bytes), block size, num blocks.
constexpr int kTqTab = 6;

struct Leaves {
  int n;
  int depth[kMaxLeaves];
};

struct Args {
  const int* rescue_leaf;
  const int64_t* gdn_tab;
  const float* gdn_stash;   // [NG, cap_b, L, HV, 258] contiguous
  const __half* conv_stash; // [NG, cap_b, L, C] contiguous
  const int64_t* tq_tab;
  const int64_t* slot_mapping;  // [B*M] (TQ group)
  int64_t gs_layer, gs_b, gs_l, gs_h;
  int64_t cs_layer, cs_b, cs_l;
  int NG, HV, C, W, conv_len, conv_blocks, NT, Hk, slot_bytes, M, null_id;
};

template <typename S> __device__ __forceinline__ float ld(const S* p);
template <> __device__ __forceinline__ float ld<float>(const float* p) { return *p; }
template <> __device__ __forceinline__ float ld<__half>(const __half* p) { return __half2float(*p); }
template <typename S> __device__ __forceinline__ void st(S* p, float v);
template <> __device__ __forceinline__ void st<float>(float* p, float v) { *p = v; }
template <> __device__ __forceinline__ void st<__half>(__half* p, float v) { *p = __float2half(v); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ bool slot_ok(int s, int64_t num_slots, int null_id) {
  return s >= 0 && s != null_id && s < num_slots;
}

template <typename S>
__global__ __launch_bounds__(kThreads) void tree_commit_kernel(Args a, Leaves lv) {
  const int b = blockIdx.x;
  const int l = a.rescue_leaf[b];
  if (l < 0 || l >= lv.n) return;
  const int j = lv.depth[l];
  if (j < 1 || j >= a.W) return;
  const int R = a.W + l;  // chain-first row of leaf l
  const int job = blockIdx.y;
  const int tid = threadIdx.x;
  const int n_ssm = a.NG * a.HV;
  const int n_conv = a.NG * a.conv_blocks;

  if (job < n_ssm) {
    // ---- GDN temporal state: slot j <- update(slot j-1, leaf stash)
    const int g = job / a.HV, hv = job - g * a.HV;
    const int64_t* t = a.gdn_tab + (int64_t)g * kGdnTab;
    const int* idx = reinterpret_cast<const int*>(t[7]) + (int64_t)b * a.W;
    const int src = idx[j - 1], dst = idx[j];
    if (!slot_ok(src, t[2], a.null_id) || !slot_ok(dst, t[2], a.null_id)) return;
    S* base = reinterpret_cast<S*>(t[0]);
    const S* hs = base + (int64_t)src * t[1] + (int64_t)hv * kDimV * kDimK;
    S* hd = base + (int64_t)dst * t[1] + (int64_t)hv * kDimV * kDimK;
    const float* sp = a.gdn_stash + g * a.gs_layer + b * a.gs_b + l * a.gs_l + hv * a.gs_h;
    const float dec = sp[kDimK + kDimV], beta = sp[kDimK + kDimV + 1];
    const int lane = tid & 31, warp = tid >> 5;
    float kk[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) kk[i] = sp[lane * 4 + i];
    for (int r = warp; r < kDimV; r += kWarps) {
      float h[4], s = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        h[i] = ld<S>(hs + (int64_t)r * kDimK + lane * 4 + i);
        s += h[i] * kk[i];
      }
      s = warp_sum(s);  // every lane has read its elements of row r before any lane writes (in-place safe)
      const float delta = (sp[kDimK + r] - dec * s) * beta;
#pragma unroll
      for (int i = 0; i < 4; ++i) st<S>(hd + (int64_t)r * kDimK + lane * 4 + i, dec * h[i] + kk[i] * delta);
    }
    return;
  }
  if (job < n_ssm + n_conv) {
    // ---- GDN conv window: column (Kw-2)+j of slot idx[b, 0] <- leaf pre-conv x
    const int cj = job - n_ssm;
    const int g = cj / a.conv_blocks, blk = cj - g * a.conv_blocks;
    const int64_t* t = a.gdn_tab + (int64_t)g * kGdnTab;
    const int* idx = reinterpret_cast<const int*>(t[7]) + (int64_t)b * a.W;
    const int slot = idx[0];
    if (!slot_ok(slot, t[2], a.null_id)) return;
    const int col = a.conv_len - a.W + j;
    __half* cs = reinterpret_cast<__half*>(t[3]) + (int64_t)slot * t[4] + (int64_t)col * t[6];
    const __half* xs = a.conv_stash + g * a.cs_layer + b * a.cs_b + l * a.cs_l;
    const int c0 = blk * kConvChunk;
    const int c1 = min(a.C, c0 + kConvChunk);
    for (int c = c0 + tid; c < c1; c += kThreads) cs[(int64_t)c * t[5]] = xs[c];
    return;
  }
  // ---- TurboQuant packed KV row: slot of row j <- slot of leaf row R (all heads, whole padded slot)
  const int tl = job - n_ssm - n_conv;
  if (tl >= a.NT) return;
  const int64_t* t = a.tq_tab + (int64_t)tl * kTqTab;
  const int64_t sdst = a.slot_mapping[(int64_t)b * a.M + j];
  const int64_t ssrc = a.slot_mapping[(int64_t)b * a.M + R];
  const int64_t cap = t[4] * t[5];
  if (sdst < 0 || ssrc < 0 || sdst >= cap || ssrc >= cap) return;
  uint8_t* base = reinterpret_cast<uint8_t*>(t[0]);
  const uint8_t* src = base + (ssrc / t[4]) * t[1] + (ssrc % t[4]) * t[2];
  uint8_t* dst = base + (sdst / t[4]) * t[1] + (sdst % t[4]) * t[2];
  for (int i = tid; i < a.Hk * a.slot_bytes; i += kThreads) {
    const int h = i / a.slot_bytes, c = i - h * a.slot_bytes;
    dst[(int64_t)h * t[3] + c] = src[(int64_t)h * t[3] + c];
  }
}

}  // namespace

// rescue_leaf int32 [B]; gdn_tab int64 [NG, 8]; gdn_stash fp32 [NG, cap_b, L, HV, 258] contiguous;
// conv_stash fp16 [NG, cap_b, L, C] contiguous; tq_tab int64 [NT, 6]; slot_mapping int64 [>= B*M];
// leaf_depth: host list (TreeSpec.leaf_depth).  ssm_fp32 selects the temporal-state dtype (fp32 | fp16).
void tree_commit(torch::Tensor rescue_leaf, torch::Tensor gdn_tab, torch::Tensor gdn_stash, torch::Tensor conv_stash,
                 torch::Tensor tq_tab, torch::Tensor slot_mapping, std::vector<int64_t> leaf_depth, int64_t W,
                 int64_t M, int64_t conv_len, int64_t Hk, int64_t slot_bytes, int64_t null_id, bool ssm_fp32) {
  for (const auto& x : {rescue_leaf, gdn_tab, gdn_stash, conv_stash, tq_tab, slot_mapping})
    TORCH_CHECK(x.is_cuda() && x.device() == rescue_leaf.device(), "tree_commit: CUDA tensors on one device");
  TORCH_CHECK(rescue_leaf.scalar_type() == at::kInt && rescue_leaf.dim() == 1 && rescue_leaf.is_contiguous(),
              "rescue_leaf: int32 [B]");
  TORCH_CHECK(gdn_tab.scalar_type() == at::kLong && gdn_tab.dim() == 2 && gdn_tab.size(1) == kGdnTab &&
              gdn_tab.is_contiguous(), "gdn_tab: int64 [NG, 8]");
  TORCH_CHECK(tq_tab.scalar_type() == at::kLong && tq_tab.dim() == 2 && tq_tab.size(1) == kTqTab &&
              tq_tab.is_contiguous(), "tq_tab: int64 [NT, 6]");
  TORCH_CHECK(gdn_stash.scalar_type() == at::kFloat && gdn_stash.dim() == 5 && gdn_stash.is_contiguous() &&
              gdn_stash.size(4) == kStashW && gdn_stash.size(0) == gdn_tab.size(0),
              "gdn_stash: fp32 contiguous [NG, B_tree, L, HV, 258]");
  TORCH_CHECK(conv_stash.scalar_type() == at::kHalf && conv_stash.dim() == 4 && conv_stash.is_contiguous() &&
              conv_stash.size(0) == gdn_tab.size(0) && conv_stash.size(1) == gdn_stash.size(1) &&
              conv_stash.size(2) == gdn_stash.size(2), "conv_stash: fp16 contiguous [NG, B_tree, L, C]");
  TORCH_CHECK(slot_mapping.scalar_type() == at::kLong && slot_mapping.dim() == 1 && slot_mapping.is_contiguous(),
              "slot_mapping: int64 [B*M]");
  const int64_t B = rescue_leaf.size(0);
  const int64_t L = (int64_t)leaf_depth.size();
  TORCH_CHECK(L >= 1 && L <= kMaxLeaves && L <= gdn_stash.size(2), "tree_commit: 1 <= L <= 16 and <= stash L");
  TORCH_CHECK(W >= 2 && M == W + L && conv_len >= W, "tree_commit: M = W + L, conv_len >= W");
  TORCH_CHECK(B <= gdn_stash.size(1), "tree_commit: more requests than the stash holds (B_tree)");
  TORCH_CHECK(slot_mapping.numel() >= B * M, "tree_commit: slot_mapping shorter than B*M");
  TORCH_CHECK(tq_tab.size(0) == 0 || (Hk >= 1 && slot_bytes >= 1), "tree_commit: TQ head count / slot bytes");
  Leaves lv{};
  lv.n = (int)L;
  for (int64_t i = 0; i < L; ++i) {
    TORCH_CHECK(leaf_depth[i] >= 1 && leaf_depth[i] < W, "tree_commit: leaf depth in [1, W-1]");
    lv.depth[i] = (int)leaf_depth[i];
  }
  if (B == 0) return;
  Args a{};
  a.rescue_leaf = rescue_leaf.data_ptr<int>();
  a.gdn_tab = gdn_tab.data_ptr<int64_t>();
  a.gdn_stash = gdn_stash.data_ptr<float>();
  a.conv_stash = reinterpret_cast<const __half*>(conv_stash.data_ptr());
  a.tq_tab = tq_tab.data_ptr<int64_t>();
  a.slot_mapping = slot_mapping.data_ptr<int64_t>();
  a.gs_layer = gdn_stash.stride(0); a.gs_b = gdn_stash.stride(1); a.gs_l = gdn_stash.stride(2);
  a.gs_h = gdn_stash.stride(3);
  a.cs_layer = conv_stash.stride(0); a.cs_b = conv_stash.stride(1); a.cs_l = conv_stash.stride(2);
  a.NG = (int)gdn_tab.size(0); a.HV = (int)gdn_stash.size(3); a.C = (int)conv_stash.size(3); a.W = (int)W;
  a.conv_len = (int)conv_len; a.conv_blocks = (a.C + kConvChunk - 1) / kConvChunk; a.NT = (int)tq_tab.size(0);
  a.Hk = (int)Hk; a.slot_bytes = (int)slot_bytes; a.M = (int)M; a.null_id = (int)null_id;
  const int64_t jobs = (int64_t)a.NG * a.HV + (int64_t)a.NG * a.conv_blocks + a.NT;
  if (jobs == 0) return;
  TORCH_CHECK(jobs <= 65535 && B <= 2147483647, "tree_commit: grid too large");
  const at::cuda::OptionalCUDAGuard guard(rescue_leaf.device());
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  const dim3 grid((unsigned)B, (unsigned)jobs);
  if (ssm_fp32)
    tree_commit_kernel<float><<<grid, kThreads, 0, s>>>(a, lv);
  else
    tree_commit_kernel<__half><<<grid, kThreads, 0, s>>>(a, lv);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("tree_commit", &tree_commit, "E16 P5 tree-step commit (sm_75)"); }
