# Experimental TQ stage-1 shared/L1 preference

Source-only candidate based on immutable `7004df6ae82c020c8a8e01c51f2f18f8889ab796`.
Credit: upstream vLLM and its original license; vLLM 2080 Ti Definitive Edition;
github.com/weicj. Existing notices and license remain unchanged.

Set `VLLM_TQ_GQA_STAGE1_SHARED_KIB` before boot to `default`, `32`, or `64`.
Absent means `default`. Empty, whitespace, percentages, and other values refuse
at first decode. It is cached per process and cannot be changed per request.
Default performs no new CUDA API, logging, device query, or explicit carveout lock;
first-use function-local static initialization can synchronize. Explicit modes
configure the exact non-templated `tq_gqa_stage1` once per device in the worker,
after the original device guard and before its first launch. Normal worker
primary-context lifetime is assumed; in-process context replacement/reset is
unsupported. First use during graph capture refuses: eager warmup must configure
the preference before capture. Both TP workers need an actual success marker.

This is restricted to SM75. Request 32 means preferred percentage 50 of its 64 KiB
maximum shared capacity; request 64 means 100. It does not pass KiB values to the
percentage API. CUDA return values and the returned stored preference are checked.
No kernel math, launch arguments, shared bytes, output buffers or stage 2 changes.
The once marker declares the request, returned preference and function resources;
it does not prove selected physical partition, occupancy, kernel dispatch, numerical
parity or speed. The driver may override the preference. Default may already use64.

Saved stage 1 resources are 256 threads / 80 registers per thread / 28192 B dynamic shared.
One versus two blocks at 32 versus 64 KiB is a capacity calculation. More L1 can win
over extra blocks; no speedup is implied. A subsequent native build must compare
the candidate's exact stage 1/stage 2 device SASS and resources with 7004: unchanged
device bytes are expected from unchanged device source, but NOT verified here.
No compiled-C++ test, CUDA invocation, native build, graph smoke, profiler trace,
worker-state/quality gate or performance measurement has occurred in this package.
The stdlib tests inspect source contracts and an extracted literal policy only.

Minimum root-owned qualification: publish one candidate with its own TQ JIT root
and seven native pins; compare same candidate default versus 64 (then 32 only if
useful), matching geometry 3568 / retention 10704 and the existing 40 request / 16 wave
C1/C4 greedy/Halo suite. Confirm both worker markers and actual module/function
binding. Preserve actual KV capacity/native admission, costs, counted delivered
tokens, cache usage, all 20 tool contracts, output hashes, sampler and original
restoration. Profiled timings are diagnostic; unprofiled waves establish speed.
If source default controls differ from old binaries, same-new-release arms control
that variation. Do not infer physical carveout from `cudaFuncGetAttributes` alone.

NVIDIA references: [Turing partition and resources](https://docs.nvidia.com/cuda/archive/12.8.1/turing-tuning-guide/index.html),
[CUDA13.2 attribute API](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-runtime-api/group__CUDART__HIGHLEVEL.html).
This is an E04/E06 and REA03/04 variant, not a new stable idea ID.
