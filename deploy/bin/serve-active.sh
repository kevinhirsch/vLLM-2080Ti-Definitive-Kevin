#!/usr/bin/env bash
# serve-active.sh -- exec whichever model profile is active. The pointer file
# 'active-serve' names the target script; switch-model.sh repoints it. Defaults
# to the incumbent if the pointer is missing/garbage, so nothing breaks.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
target="$(cat "$D/active-serve" 2>/dev/null || true)"
case "$target" in
  serve-abliterated.sh) exec bash "$D/serve-abliterated.sh" ;;
  serve-hauhaucs.sh)    exec bash "$D/serve-hauhaucs.sh" ;;
  serve-hauhaucs-v02.sh) exec bash "$D/serve-hauhaucs-v02.sh" ;;   # integrated weicj v0.2.2-post3 build (2026-10-02)
  serve-profile-v02.sh) exec bash "$D/serve-profile-v02.sh" ;;
  *)                    exec bash "$D/serve-tqk8v4-fg.sh" ;;   # default = incumbent
esac
