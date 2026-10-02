#!/usr/bin/env bash
# ============================================================================
# Fix the CUDA 12.8 + glibc 2.43 (Ubuntu 26.04) header clash that breaks the
# vLLM build at cmake's CUDA-compiler probe:
#   crt/math_functions.h: "exception specification is incompatible" for
#   cospi/sinpi/rsqrt (+f variants) vs glibc's noexcept declarations.
# CUDA 12.9 fixed this upstream; for pinned 12.8 we add noexcept(true) to the
# 6 offending declarations so they match glibc. Reversible + idempotent.
#
# RUN AS ROOT (the header is root-owned):   sudo ~/Desktop/vllm-setup/fix-cuda128-glibc.sh
# ============================================================================
set -euo pipefail
H=/usr/local/cuda-12.8/targets/x86_64-linux/include/crt/math_functions.h
[ -f "$H" ] || { echo "ABORT: $H not found — is cuda-toolkit-12-8 installed?"; exit 1; }
[ "$(id -u)" -eq 0 ] || { echo "ABORT: run with sudo (header is root-owned)."; exit 1; }

BAK="$H.orig-precospi"
[ -f "$BAK" ] || { cp -a "$H" "$BAK"; echo ">> backed up original -> $BAK"; }

# Anchored on '__device_builtin__ <type> <name>(<type> x);' — the trailing ');'
# means an already-patched line ('...) noexcept(true);') will NOT re-match.
sed -i -E \
  -e 's/(__device_builtin__[[:space:]]+double[[:space:]]+(rsqrt|sinpi|cospi)\(double x\));/\1 noexcept(true);/' \
  -e 's/(__device_builtin__[[:space:]]+float[[:space:]]+(rsqrtf|sinpif|cospif)\(float x\));/\1 noexcept(true);/' \
  "$H"

echo ">> patched declarations now carrying noexcept(true):"
N=$(grep -cE '__device_builtin__[[:space:]]+(double|float)[[:space:]]+(rsqrt|rsqrtf|sinpi|sinpif|cospi|cospif)\((double|float) x\) noexcept\(true\);' "$H")
grep -nE '(rsqrt|rsqrtf|sinpi|sinpif|cospi|cospif)\((double|float) x\) noexcept' "$H"
[ "$N" -eq 6 ] || { echo "ABORT: expected 6 patched decls, found $N. Restore: sudo cp -a '$BAK' '$H'"; exit 1; }

echo; echo ">> nvcc smoke test (the exact thing cmake's probe does)..."
T=$(mktemp -d)
printf '#include <cuda_runtime.h>\n#include <math.h>\nint main(){return 0;}\n' > "$T/t.cu"
if /usr/local/cuda-12.8/bin/nvcc -ccbin g++-13 -c "$T/t.cu" -o "$T/t.o" 2>"$T/err"; then
  echo -e "\033[1;32m>> SMOKE TEST PASSED — CUDA can compile again.\033[0m"
else
  echo "SMOKE TEST FAILED:"; cat "$T/err"; rm -rf "$T"; exit 1
fi
rm -rf "$T"

cat <<'EOF'

Fixed. Now resume the build + serve (safe to re-run; it skips CUDA/model and
rebuilds vLLM, then installs the service and verifies):

    ~/Desktop/vllm-setup/phase2-build-serve.sh
EOF
