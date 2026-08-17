<!-- markdownlint-disable MD001 MD041 -->
# ⚡ vLLM 2080 Ti Definitive Edition

![vLLM 2080 Ti Definitive Edition cover](docs/assets/vllm-2080ti-cover.jpg)

The definitive vLLM runtime for dual RTX 2080 Ti / SM75 serving.

This is a hardware-focused fork that preserves the patched source, launch
profiles, and runtime notes needed to reproduce the working 2080 Ti vLLM stack.

Fork release: `v0.1.15`
Base vLLM: `0.21.0`

Headline evidence: the Qwen3.x 27B FP8 route is validated on official Qwen3.6
and Qwen3.8 checkpoints, with the Qwen3.6 baseline reaching `100+ tok/s`
single-request decode. The Qwen3.x 35B route also validates 256K text-only
normal/aggressive noMTP routes, 136K text+image normal/aggressive routes, and a
178K fast/MTP3 route on the same dual 2080 Ti TP=2 runtime.

Language: English | [简体中文](README.zh-CN.md)

![Live single-request throughput demo](docs/assets/vllmspeed.gif)

## 💡 Why RTX 2080 Ti for LLM Inference?

In August 2018, NVIDIA launched the RTX 2080 Ti and moved the enthusiast GPU
line from GTX into the RTX era. Years later, the card is still remembered as a
landmark Turing design. With 22GB memory mods, NVLink, high memory bandwidth,
and enough raw compute to remain relevant, dual 2080 Ti cards turn out to be a
surprisingly strong local AI inference platform.

| Metric | 2x 2080 Ti 22GB + NVLink | 3090 Ti 24GB baseline | Ratio |
|---|---:|---:|---:|
| Physical CUDA core count | 8,704 | 5,376 | 1.62x |
| SM count | 136 | 84 | 1.62x |
| Physical Tensor Core count | 1,088 | 336 | 3.24x |
| Dense Tensor FP16 matrix throughput | 228 TFLOPS | 160 TFLOPS | 1.43x |
| Total physical memory bandwidth | 1,232 GB/s | 1,008 GB/s | 1.22x |
| Total VRAM capacity | 44GB | 24GB | 1.83x |
| Secondary-market price anchor | about $550 with NVLink | about $1,100 | about 0.5x |

The project is built around a simple cost/performance bet: use roughly half the
secondary-market price of an RTX 3090 Ti to get a dual 22GB RTX 2080 Ti setup
that can match or exceed it on the physical resources that matter for LLM
serving, then use vLLM runtime work to turn those resources into real tokens.

That is the first value of this fork: take old but strong Turing silicon and
make it behave like a serious 27B/31B/35B-class inference platform through
Marlin, FlashQLA/FlashInfer, TurboQuant/INT8 KV, MTP, and CUDAGraph
integration.

## 🧩 Core Routes

Serving shape:

- This project optimizes for extreme single-concurrency performance on dual
  2080 Ti: one personal-agent style workload, one serious 27B/31B/35B model,
  and the largest practical context window this hardware can sustain.
- It is not a multi-tenant serving stack. Multi-agent use is supported best as
  queued workspace isolation, not as parallel long-prefill throughput. Long
  prefill work is capacity-safe when tuned, but it is effectively serialized by
  the runtime scheduler on this TP=2 profile.

Status: 🟢 validated support; 🟡 experimental or partial support; 🔴 known
failure or clear regression; ⚪ not a target preset or not yet validated.

### Qwen3.x 27B Mature Route

Qwen3.x-family 27B is the primary production route for this fork. It has the
broadest tested coverage across FP8/INT4/NVFP4 weights, MTP, FP16/INT8/
TurboQuant KV, 256K native context, YaRN capacity, and image serving.

| Feature | FP16 KV | INT8 KV | TurboQuant KV |
|---|---|---|---|
| Marlin weight route | 🟢 FP8/INT4/NVFP4 | 🟢 FP8/INT4/NVFP4 | 🟢 FP8/INT4/NVFP4 |
| MTP decoding | 🟢 supported | 🟢 supported | 🟢 supported |
| Native 256K context | 🟢 supported | 🟢 supported | 🟢 supported |
| YaRN extension | ⚪ not the target route | 🟢 supported | ⚪ not a target preset |
| No-eager / CUDAGraph | 🟢 supported | 🟡 partial support | 🟢 supported |
| Fast prefill path | 🟢 FlashQLA / FlashInfer | 🟢 FlashQLA / FlashInfer | 🟢 FlashQLA / FlashInfer |
| Multimodal image serving | 🟢 supported | 🟢 supported | 🟢 supported |
| Current preset status | 🟢 normal / fast / safe | 🟢 normal / safe | 🟢 fast |

### Qwen3.x 35B Mature Secondary Route

Qwen3.x 35B MoE is the mature secondary route on the same validated dual
2080 Ti runtime. It broadly inherits the same support surface as the 27B lane:
MTP, FP16-KV long-context serving, FlashQLA / FlashInfer fast prefill, and
multimodal image serving are all supported on the shipped 35B presets.

The current shipped set covers FP16-KV 256K text-only `normal` /
`aggressive`, FP16-KV 136K text+image `normal` / `aggressive`, and a 178K
`fast` MTP3 preset.

### Gemma4 31B Experimental Route

Gemma4 31B is kept as a secondary experimental route. The official QAT target
with the matching QAT assistant is now the most promising Gemma path, with
better FP16/default-KV headroom than earlier Gemma checkpoints.

| Feature | FP16 KV | INT8 KV | TurboQuant KV |
|---|---|---|---|
| Marlin weight route | 🟢 GPTQ / QAT | 🟡 GPTQ / QAT | 🟡 GPTQ / QAT |
| MTP decoding | 🟡 QAT assistant MTP3 | ⚪ no preset | ⚪ no preset |
| Validated context | 🟡 about 170K KV headroom observed | 🔴 init issue | 🔴 capacity shortfall |
| No-eager / CUDAGraph | 🟢 supported | 🟡 fallback issue | 🟡 admission limited |
| Fast prefill path | 🟢 FlashInfer | 🟡 FlashInfer | 🟡 FlashInfer |
| Multimodal image serving | ⚪ no validated preset | ⚪ no validated preset | ⚪ no validated preset |
| Current preset status | 🟡 experimental only | ⚪ no preset | ⚪ no preset |

## 🧪 Tested Model Checkpoints

This section records checkpoint-level validation. It is intentionally stricter
than "vLLM can load it": a supported checkpoint can start and generate, while a
recommended checkpoint also has a useful throughput/context tradeoff on dual 2080 Ti.
The generic Qwen3.x 27B FP8 route currently covers the official Qwen3.6 and
Qwen3.8 checkpoints listed below; other quantized rows remain checkpoint-specific.

| Model route | Weight route | Model cards | Status |
|---|---|---|---|
| Qwen3.x 27B | FP8 | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)<br>[Qwen/Qwen3.6-27B-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)<br>[Jackrong/Qwopus3.6-27B-v2-FP8](https://huggingface.co/Jackrong/Qwopus3.6-27B-v2-FP8) | 🟢 Recommended |
| Qwen3.x 35B | FP8 | [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)<br>[Jackrong/Qwopus3.6-35B-A3B-Coder-FP8](https://huggingface.co/Jackrong/Qwopus3.6-35B-A3B-Coder-FP8)<br>[kyr0/Ornith-35B-FP8-E4M3-MTP](https://huggingface.co/kyr0/Ornith-35B-FP8-E4M3-MTP) | 🟢 Recommended |
| Qwen3.x 27B | AWQ-INT4 | [QuantTrio/Qwen3.6-27B-AWQ](https://huggingface.co/QuantTrio/Qwen3.6-27B-AWQ)<br>[mconcat/Qwopus3.6-27B-v2-AWQ-4bit](https://huggingface.co/mconcat/Qwopus3.6-27B-v2-AWQ-4bit) | 🟢 Recommended |
| Qwen3.x 27B | GPTQ-INT4 | [llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4](https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4) | 🟢 Recommended |
| Qwen3.x 27B | NVFP4 | [unsloth/Qwen3.6-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4) | 🟡 Supported |
| Qwen3.x 27B | Quark-INT8 | [nameistoken/Qwen3.6-27B-Quark-W8A8-INT8](https://huggingface.co/nameistoken/Qwen3.6-27B-Quark-W8A8-INT8) | 🟡 Supported |
| Qwen3.x 27B | AutoGPTQ-INT8 | [Minachist/Qwen3.6-27B-INT8-AutoRound](https://huggingface.co/Minachist/Qwen3.6-27B-INT8-AutoRound)<br>[Minachist/Qwen3.6-27B-INT8-AutoRound W8A16-GS128](https://huggingface.co/Minachist/Qwen3.6-27B-INT8-AutoRound/tree/W8A16-GS128) | 🟡 Supported |
| Gemma4 31B QAT | QAT + QAT assistant draft | [google/gemma-4-31B-it-qat-w4a16-ct](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct)<br>[google/gemma-4-31B-it-qat-q4_0-unquantized-assistant](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized-assistant) | 🟡 Supported |
| Gemma4 31B GPTQ | GPTQ-INT4 + assistant draft | [ebircak/gemma-4-31B-it-4bit-W4A16-GPTQ](https://huggingface.co/ebircak/gemma-4-31B-it-4bit-W4A16-GPTQ) | 🟡 Supported |

## 🛠️ Target Hardware & Runtime

- Validated GPU profile: dual RTX 2080 Ti 22GB, SM75, NVLink, tensor parallel
  size 2
- Validated host OS: Ubuntu 22.04/24.04 LTS or Debian 12, on Linux kernel 6.x
- CUDA/PyTorch: CUDA 12.8, `torch 2.11.0+cu128`
- Fork release: `v0.1.15`
- Base vLLM: `0.21.0`
- Repository identity: `vllm-2080ti-definitive`
- Runtime identity: `vllm-sm75-tp2-cu128`
- Compatibility target: NVIDIA Turing / SM75 GPUs. Other Turing cards still
  need profile validation for VRAM capacity, P2P/NVLink behavior, model
  head_dim, KV dtype, and CUDAGraph/MTP settings.

## 🚀 How To Use

Clone the repository and build the runtime:

```bash
git clone https://github.com/weicj/vLLM-2080Ti-Definitive.git
cd vLLM-2080Ti-Definitive
./build.sh
```

`build.sh` creates the local `.venv`, installs dependencies, builds the CUDA
extensions, and prints a clear success or failure result with a build log path.
Before the install starts, it benchmarks the available PyPI, Git, and PyTorch
wheel routes and switches to a faster mirror path automatically when needed.

Then start and manage the service:

```bash
./launcher.sh
```

The launcher is the interactive service manager. From the menu you can choose
the checkpoint directory, apply or edit a profile, select `safe` / `normal` /
`fast` / `aggressive`, choose GPU/TP devices, set the port, switch local-only
or LAN access, configure chat templates and tool calling, start the server,
stop the server, or save a custom profile.

Prefix cache is enabled by default as a global launcher setting. It is not saved
inside route profiles. For Qwen routes, the launcher also applies the matching
cache mode needed by the validated prefix-cache path.

Tool calling is supported through the OpenAI-compatible serving API. The
launcher exposes automatic tool choice, tool parser selection, and strict
structured tool output as global runtime settings.

After a successful launch, the status panel shows `RUNNING`, the served model
name, PID, OpenAI-compatible API URL, log file, prefix-cache state, prompt token
detail state, and cache capacity when vLLM reports it.

For scripted use:

```bash
MODEL_DIR=/path/to/qwen-or-gemma-checkpoint \
PROFILE=qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env \
MODE=normal \
PORT=8000 \
SERVICE_SCOPE=lan \
CUDA_VISIBLE_DEVICES=0,1 \
./launcher.sh --non-interactive
```

Profiles declare compatible modes, not a recommended launch mode. Set
`MODE=safe`, `MODE=normal`, `MODE=fast`, or `MODE=aggressive` explicitly when
you want a specific mode; the launcher validates that choice against the
profile.

3. Update an existing checkout:

```bash
./update.sh
```

`update.sh` checks the latest GitHub Release against the local fork version,
downloads the newer release archive when available, preserves local runtime
state such as `.venv`, `.deps`, logs, results, caches, and user profiles, then
asks whether to run `build.sh` immediately.

## 🧭 Profiles

Start from [Profile Guide](profiles/README.md). Profiles are organized as
`profiles/<model>/<mode>/<weight>/<route>.env`, for example
`qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env`,
`qwen35b/aggressive/fp8/fp16kv-256K-nomtp-text-only.env`, and
`qwen35b/normal/fp8/fp16kv-136K-nomtp-text-image.env`.

Available modes:

- `normal`: recommended default mode for regular production deployments.
- `fast`: high-performance mode, not recommended for stable production
  deployments.
- `aggressive`: more aggressive mode with the highest performance and quality risk.
- `safe`: safety mode. It is slower, but prioritizes highly stable output
  quality for troubleshooting.

## 🚀 MTP And KV Precision

Use the bundled profiles instead of hand-tuning MTP and KV settings first.
MTP is already set to the best practical value for each route. Choose KV by
intent first: FP16/default KV for maximum output quality, INT8 KV for balanced
long-context service, and TurboQuant K8V4 for fast compression routes.

Detailed benchmark notes are kept in
[MTP Task Sensitivity](docs/mtp-task-sensitivity.md) and
[Qwen3.6 KV Throughput Sweep](docs/qwen36-kv-throughput-sweep.md).

## ❓ Hardware Q&A

**Q: What GPU interconnect is required?**

A: NVLink is recommended, but PCIe P2P is the real baseline requirement. The
validated system uses NVLink and an intentionally non-ideal PCIe topology, with
one card at PCIe 3.0 x1 and the other at PCIe 3.0 x4. With NVLink carrying
GPU-to-GPU traffic, PCIe slot bandwidth is not the main bottleneck. Without
NVLink, do not treat narrow PCIe links as proven sufficient; confirm P2P
behavior and benchmark the actual topology.

**Q: Does the host need a strong CPU or a lot of RAM?**

A: It does not need a high-end CPU, but it does prefer a modern CPU with strong
single-core performance and low platform latency. The validated path has run on
an Intel Core i3-9100T with 16GB RAM; in contrast, a much older dual Xeon X5675
host measured about 56 tok/s decode versus about 91 tok/s on the i3-9100T under
the same 4096/128 GPTQ-INT4 MTP3 route. More RAM mainly helps builds, downloads,
and compile cache. Because vLLM has a Python/service control plane, very old CPUs
may be better matched to minimal C++ runtimes such as llama.cpp.

**Q: Which Turing GPUs make sense? Can I mix 11GB and 22GB cards?**

A: The fully validated target is dual RTX 2080 Ti 22GB. Other good candidates
are high-VRAM TU102-class cards: TITAN RTX 24GB, Quadro RTX 6000 24GB, and
Quadro RTX 8000 48GB, preferably in pairs with NVLink or confirmed PCIe P2P.
Mixed 11GB + 22GB RTX 2080 Ti setups are not recommended for these 27B/31B
profiles because vLLM TP=2 is effectively constrained by the smaller rank.
Smaller Turing cards can run smaller models, but they are not the main target
for this stack.

**Q: Which CUDA, PyTorch, and driver versions are validated?**

A: The validated runtime is CUDA 12.8 + `torch 2.11.0+cu128`. The reference
validation host used NVIDIA driver `590.48.01`. Use a recent NVIDIA driver that
supports your host GPUs and is compatible with the CUDA runtime. Do not mix
build/runtime assumptions casually: keep the PyTorch CUDA lane, local CUDA
toolkit, FlashInfer/FlashQLA builds, and launch profile aligned.

**Q: What other hardware risks matter?**

A: Cooling, power stability, and enough SSD space for model files and compile
caches. Thermal throttling can hide as a software regression, especially during
long prefill or repeated CUDAGraph/AOT compilation runs.

## 🔗 Related Project

- [2080Ti-LLM-Toolbox](https://github.com/weicj/2080Ti-LLM-Toolbox): companion
  toolbox for dual 2080 Ti model routes, benchmark summaries, model notes, and
  operational guidance. This repository focuses on the patched vLLM runtime
  itself.

## 🙏 Credits / Upstream Projects

This repository is a hardware-focused fork of upstream
[vLLM](https://github.com/vllm-project/vllm), licensed under Apache-2.0. The
fork keeps the upstream project structure and adds local SM75 runtime patches,
launch profiles, and validation notes for the dual 2080 Ti route.

Acceleration components used or integrated by this runtime include:

- [vLLM](https://github.com/vllm-project/vllm): base inference engine and
  serving stack.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer): attention,
  sampling, and quantized kernel paths used by vLLM.
- [QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA): upstream FlashQLA
  Gated DeltaNet / Qwen3.5 linear-attention implementation.
- [weicj/FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75):
  SM70/SM75 adaptation used by the validated Qwen3.6 prefill profile.
- TurboQuant, Marlin, CUTLASS, Triton, and related vLLM
  acceleration kernels: existing open-source acceleration work integrated and
  profiled for this hardware target.

While this vLLM-2080Ti-Definitive fork will not strictly follow upstream vLLM,
patches merged from upstream updates will be re-validated within the SM75-specific
scope of this fork.
