#!/usr/bin/env bash
# Switch an installed Qwen checkpoint without rewriting the validated runtime flags.
# Usage: switch-model.sh <default|hauhaucs|dft1|abliterated|/path/to/model|status|list|rollback>
#        switch-model.sh --dry-run /path/to/model   # validation + JSON plan, no writes/restart
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$D/model_switch.py" "$@"
