#!/usr/bin/env bash
# Build the K1 JIT extension (no GPU needed) and print registers/spills per kernel instantiation.
cd "$(dirname "$0")/../.."
export CUDA_HOME=/usr/local/cuda-13 PATH=/home/kevin/Desktop/wt-integrate/.venv/bin:/usr/local/cuda-13/bin:$PATH TORCH_CUDA_ARCH_LIST=7.5 CUDA_VISIBLE_DEVICES= VLLM_TQ_FA75_PTXAS_V=1 VLLM_TQ_FA75_VERBOSE=1 MAX_JOBS=4
python -c "import sys; sys.path.insert(0,'vllm/v1/attention/ops'); import fa75_prefill as f; f._load(); print('BUILD OK')" > /tmp/k1_build.log 2>&1
grep -E "error|BUILD OK" /tmp/k1_build.log | head -20
grep -E "Compiling entry|spill|registers" /tmp/k1_build.log | grep -v EmptyKernel | paste - - - | sed -E 's/.*k1fa_kernelILi([0-9]+)ELb([01])ELb([01])ELb([01])ELb([01])E\S*/BN\1 C\2 P\3 L\4 Q\5/; s/ptxas info *: //g' | awk '{print $1,$2,$3,$4,$5,"|",$0}' | sed -E 's/\| BN.*for .sm_75.//' | cut -c1-200
