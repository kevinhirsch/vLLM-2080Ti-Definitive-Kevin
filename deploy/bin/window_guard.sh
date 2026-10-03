#!/usr/bin/env bash
# window_guard.sh -- the ONE way a legacy engine-window shell script owns the engine (leads L111 + L153, lane A6).
#
# Why: window scripts used to `systemctl stop vllm-qwen27b-watchdog.timer` and `start` it again at the end. That is wrong
# twice. The timer is the clock of the one engine-liveness authority (engine-actuator.py tick), so stopping it blinds the
# whole authority, not only the "restart a dead engine" action (L153). And every script that dies between the stop and the
# start (set -e, a failed restart, SIGTERM, SIGKILL of the parent) leaves it stopped: DFT left it stopped 02:49-07:14 and the
# engine ran 4.5 h unwatched through the 06:27 crash (L111). A script that "restores" by starting unconditionally also turns
# a timer the owner had stopped back on.
#
# Use (first lines of the window script, before it touches the engine):
#     . "$(dirname "$(readlink -f "$0")")/window_guard.sh"        # path to this file
#     window_guard_enter MYLANE "what the window does (>= 8 chars)" 7200 -- "$@"
# window_guard_enter re-executes the script under `engine-actuator.py hold run --kind engine` (Lane LV's TTL-bounded,
# owner-pid-scoped liveness hold: the liveness authority and the watchdog take no automatic action, nobody else's planned
# restart is accepted, the hold is void the moment the wrapper dies, and the engine is brought back at exit). The second pass
# sees ENGINE_HOLD_LEASE and returns. If another hold is already held the script never runs (hold run exits 75).
#
# Fallback (the deployed actuator has no `hold` subcommand yet): the script runs as a child of a wrapper that pauses the
# timer, and restores its EXACT prior state (started only if it was active) on every exit path, including signals.
#
# Env (tests): WINDOW_GUARD_ACTUATOR, WINDOW_GUARD_TIMER, WINDOW_GUARD_TTL_MAX.
WG_ACTUATOR=${WINDOW_GUARD_ACTUATOR:-$HOME/.local/share/vllm-qwen27b/engine-actuator.py}
WG_TIMER=${WINDOW_GUARD_TIMER:-vllm-qwen27b-watchdog.timer}
WG_FILE=$(readlink -f "${BASH_SOURCE[0]}")

_wg_has_hold() {
  [ -f "$WG_ACTUATOR" ] && /usr/bin/python3 "$WG_ACTUATOR" hold status >/dev/null 2>&1
}

# Run "$@" with the watchdog timer paused; restore its exact prior state on every exit path.
_wg_fallback_run() {
  local child rc=0
  WG_PRIOR=$(systemctl is-active "$WG_TIMER" 2>/dev/null)
  echo "[window_guard] no liveness hold available: pausing $WG_TIMER (was: ${WG_PRIOR:-unknown}) and restoring that exact state at exit" >&2
  [ "$WG_PRIOR" = active ] && sudo -n systemctl stop "$WG_TIMER"
  _wg_restore() {                              # runs at EXIT, after this function returned: WG_PRIOR is global on purpose
    trap - EXIT
    if [ "$WG_PRIOR" = active ] && [ "$(systemctl is-active "$WG_TIMER" 2>/dev/null)" != active ]; then
      sudo -n systemctl start "$WG_TIMER" && echo "[window_guard] $WG_TIMER restored (active)" >&2
    fi
  }
  trap _wg_restore EXIT
  "$@" &
  child=$!
  trap 'kill -TERM "$child" 2>/dev/null' TERM INT HUP
  while :; do
    wait "$child"; rc=$?
    kill -0 "$child" 2>/dev/null || break      # a trapped signal interrupts wait while the child lives on: wait again
  done
  return "$rc"
}

window_guard_enter() {
  local by=$1 reason=$2 ttl=$3
  shift 3
  [ "${1:-}" = "--" ] && shift
  [ -n "${ENGINE_HOLD_LEASE:-}" ] && return 0          # already under `hold run` (second pass)
  [ -n "${WINDOW_GUARD_FALLBACK_PASS:-}" ] && return 0 # already under the fallback wrapper
  local script=${BASH_SOURCE[1]:-$0}
  script=$(readlink -f "$script")
  if _wg_has_hold; then
    exec /usr/bin/python3 "$WG_ACTUATOR" hold run --kind engine --by "$by" --reason "$reason" --ttl "$ttl" -- bash "$script" "$@"
  fi
  WINDOW_GUARD_FALLBACK_PASS=1 export WINDOW_GUARD_FALLBACK_PASS
  exec bash "$WG_FILE" --fallback-run bash "$script" "$@"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then               # executed, not sourced: the fallback wrapper
  case "${1:-}" in
    --fallback-run) shift; _wg_fallback_run "$@"; exit $? ;;
    *) echo "usage: . window_guard.sh; window_guard_enter BY REASON TTL -- \"\$@\"" >&2; exit 2 ;;
  esac
fi
