#!/usr/bin/env bash
# estate-watchdog.sh -- deterministic, dependency-light estate health check.
#
# WHY THIS EXISTS (2026-09-04 incident): agents-prod (10.0.1.10) lost internet
# for 8d19h because a systemd-networkd restart flushed a foreign (wg-quick
# PostUp-pinned) route for the VPS endpoint; with 173/8 inside AllowedIPs the
# endpoint then routed into wg0 itself, and DNS (pinned through the tunnel)
# died with it. Nothing alerted: the previous "egress-reputation-watchdog" was
# an LLM-driven scheduled Claude task that silently stopped logging on
# 2026-08-09 (see ~/egress-watchdog.log). Per Kevin's standing policy, standing
# jobs must be DETERMINISTIC (no LLM) -- this script is plain bash, calls no
# model, and must keep working even if every AI system on the estate is down.
#
# Checks the VPS wireguard hub, agents-prod's egress/DNS/routing, HNET00's own
# wg tunnel + policy routing + container egress, the research service, the
# local vLLM gateway/engine, disk, GPUs, the frontier-queue cron pipeline, and
# vault sync health. Never fails the cron job itself (always exits 0); signals
# via state.json / ALERT / an optional Discord webhook instead.
#
# Usage:
#   estate-watchdog.sh           # read-only checks only (safe, default)
#   estate-watchdog.sh --fix     # also attempt narrow, rate-limited remediation
#
# Env overrides (all optional; defaults match the live 2026-09-04 estate):
#   WATCHDOG_STATE_DIR   default $HOME/.local/share/vllm-qwen27b/watchdog
#                        (log/state/ALERT/PAUSE/discord_webhook all live here)
#   WATCHDOG_SKIP_DOCKER set to 1 to skip the docker-exec egress probe outright
#                        (used for the sandboxed authoring/validation run; the
#                        real cron invocation should NOT set this)
#   VPS_HOST, VPS_USER, AGENTSPROD_HOST, AGENTSPROD_USER, RESEARCH_URL,
#   GATEWAY_URL, ENGINE_URL, FRONTIER_QUEUE_DIR, VAULT_MEMORY,
#   DOCKER_CONTAINER, DISK_PATH, DISK_WARN_GB, DISK_CRIT_GB,
#   HANDSHAKE_CRIT_SEC, QUEUE_STALE_SEC, FIX_COOLDOWN_SEC,
#   PEER_AGENTSPROD_PUBKEY, PEER_HNET00_PUBKEY
#
# Kill switch: touch "$WATCHDOG_STATE_DIR/PAUSE" to skip cycles (mirrors the
# frontier-queue runner.sh PAUSE convention already in use on this box).
#
# Exit status is always 0 (this is a monitoring probe, not a gate); severity
# is communicated via state.json, the ALERT file, and the log -- never via a
# non-zero exit that would just spam cron's own error handling.

set -uo pipefail

# ---------------------------------------------------------------------------
# Config (env-overridable)
# ---------------------------------------------------------------------------
VPS_HOST="${VPS_HOST:-173.254.204.32}"
VPS_USER="${VPS_USER:-root}"
AGENTSPROD_HOST="${AGENTSPROD_HOST:-10.0.1.10}"
AGENTSPROD_USER="${AGENTSPROD_USER:-kevin}"
RESEARCH_URL="${RESEARCH_URL:-http://10.0.1.10:8790}"
GATEWAY_URL="${GATEWAY_URL:-http://127.0.0.1:8000}"
ENGINE_URL="${ENGINE_URL:-http://127.0.0.1:8001}"
FRONTIER_QUEUE_DIR="${FRONTIER_QUEUE_DIR:-$HOME/.local/share/vllm-qwen27b/frontier-queue}"
VAULT_MEMORY="${VAULT_MEMORY:-$HOME/Obsidian/Memory/MEMORY.md}"
DOCKER_CONTAINER="${DOCKER_CONTAINER:-docker-a0-1}"
DISK_PATH="${DISK_PATH:-/}"
DISK_WARN_GB="${DISK_WARN_GB:-20}"
DISK_CRIT_GB="${DISK_CRIT_GB:-10}"
HANDSHAKE_CRIT_SEC="${HANDSHAKE_CRIT_SEC:-300}"
QUEUE_STALE_SEC="${QUEUE_STALE_SEC:-10800}"
FIX_COOLDOWN_SEC="${FIX_COOLDOWN_SEC:-3600}"
# Known wg pubkeys (from `wg show all dump` on the VPS, captured 2026-09-04).
# Override if keys ever rotate; any peer not matching either is still
# reported (see check_vps_wireguard) but cannot by itself force CRIT.
PEER_AGENTSPROD_PUBKEY="${PEER_AGENTSPROD_PUBKEY:-bXbSkAyiX7sB8tjCMqOrU7+xkm0f/WNFYnJNbLWVODc=}"
PEER_HNET00_PUBKEY="${PEER_HNET00_PUBKEY:-7J57AuXIfejQckRUXKrXL6v+yzkPOMP7obdD3cFsyVU=}"

STATE_DIR="${WATCHDOG_STATE_DIR:-$HOME/.local/share/vllm-qwen27b/watchdog}"
LOG_FILE="$STATE_DIR/estate-watchdog.log"
STATE_FILE="$STATE_DIR/state.json"
ALERT_FILE="$STATE_DIR/ALERT"
PAUSE_FILE="$STATE_DIR/PAUSE"
DISCORD_WEBHOOK_FILE="$STATE_DIR/discord_webhook"
FIX_STATE_FILE="$STATE_DIR/last-wg-restart-epoch"
LOG_MAX_BYTES=5242880

if ! command -v jq >/dev/null 2>&1; then
  echo "estate-watchdog: FATAL jq not found on PATH, cannot produce state.json" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
status_rank() { case "$1" in OK) echo 0 ;; WARN) echo 1 ;; CRIT) echo 2 ;; *) echo 1 ;; esac; }
rank_status() { case "$1" in 0) echo OK ;; 1) echo WARN ;; 2) echo CRIT ;; *) echo WARN ;; esac; }

# Milliseconds since epoch. NOTE: GNU `date +%s%3N` does NOT truncate to 3
# digits on this box's coreutils (verified empirically -- it silently returns
# the full 9-digit nanosecond field), so we take the full %s%N string and
# slice it ourselves: first 10 chars = epoch seconds, next 9 = nanoseconds.
# The 10# forces base-10 so a leading-zero nanosecond field is never
# misread as octal.
now_ms() {
  local raw
  raw=$(date +%s%N)
  printf '%d' $(( ${raw:0:10} * 1000 + 10#${raw:10:9} / 1000000 ))
}

log_line() {
  local line
  line="$(date '+%F %T %Z') $*"
  echo "$line"
  echo "$line" >> "$LOG_FILE"
}

rotate_log_if_needed() {
  [ -f "$LOG_FILE" ] || return 0
  local sz
  sz=$(stat -c %s "$LOG_FILE" 2>/dev/null || echo 0)
  case "$sz" in ''|*[!0-9]*) sz=0 ;; esac
  if [ "$sz" -gt "$LOG_MAX_BYTES" ]; then
    mv -f "$LOG_FILE" "${LOG_FILE}.1" 2>/dev/null || true
  fi
}

write_state() {
  local tmp
  tmp=$(mktemp "${STATE_DIR}/.state.json.XXXXXX")
  printf '%s\n' "$1" > "$tmp"
  mv -f "$tmp" "$STATE_FILE"
}

frontier_benchmark_active() {
  compgen -G "${FRONTIER_QUEUE_DIR}/done/*.running" > /dev/null 2>&1
}

# Result bookkeeping (associative arrays keyed by check name).
declare -A ST=() DET=() MS=()
CHECK_ORDER=()

add_result() {
  local name="$1" status="$2" detail="$3" ms="${4:-0}"
  if [ -z "${ST[$name]+x}" ]; then CHECK_ORDER+=("$name"); fi
  ST["$name"]="$status"
  DET["$name"]="$detail"
  MS["$name"]="$ms"
}

compute_overall() {
  local worst=0 name r
  for name in "${CHECK_ORDER[@]}"; do
    r=$(status_rank "${ST[$name]}")
    [ "$r" -gt "$worst" ] && worst=$r
  done
  rank_status "$worst"
}

build_checks_json() {
  local objs=() name obj
  for name in "${CHECK_ORDER[@]}"; do
    obj=$(jq -nc --arg n "$name" --arg s "${ST[$name]}" --arg d "${DET[$name]}" --argjson m "${MS[$name]:-0}" \
      '{name:$n,status:$s,detail:$d,ms:$m}')
    objs+=("$obj")
  done
  (IFS=,; echo "[${objs[*]}]")
}

# ---------------------------------------------------------------------------
# Checks -- each sets CHECK_STATUS (OK|WARN|CRIT) and CHECK_DETAIL (one line,
# no embedded newlines). Runs in a forked subshell (see main loop), so these
# may freely use "local" and never leak state between checks.
# ---------------------------------------------------------------------------

# (a) VPS reachable on tcp/22 + wg handshake age per peer.
check_vps_wireguard() {
  if ! timeout 3 nc -z -w3 "$VPS_HOST" 22 2>/dev/null; then
    CHECK_STATUS="CRIT"; CHECK_DETAIL="VPS ${VPS_HOST}:22 tcp connect failed"
    return
  fi
  local out rc
  out=$(timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=5 "${VPS_USER}@${VPS_HOST}" 'wg show all dump' 2>&1)
  rc=$?
  if [ $rc -ne 0 ] || [ -z "$out" ]; then
    CHECK_STATUS="CRIT"
    CHECK_DETAIL="tcp/22 ok but ssh/wg-show failed rc=${rc}: $(printf '%s' "$out" | tr '\n' ' ' | cut -c1-160)"
    return
  fi
  local now worst=0 parts=() seen_agentsprod=0 seen_hnet00=0
  now=$(date +%s)
  while IFS=$'\t' read -r iface pub _psk _endpoint _aips hs _rx _tx _keepalive; do
    [ "$iface" = "wg0" ] || continue
    case "${hs:-}" in ''|*[!0-9]*) continue ;; esac   # skips the interface-info line (no handshake field)
    local label="other" age st r
    case "$pub" in
      "$PEER_AGENTSPROD_PUBKEY") label="agentsprod(.10)"; seen_agentsprod=1 ;;
      "$PEER_HNET00_PUBKEY") label="HNET00"; seen_hnet00=1 ;;
    esac
    if [ "$hs" -eq 0 ]; then age=-1; else age=$(( now - hs )); fi
    if [ "$label" = "other" ]; then
      # Visibility only, per spec ("report all peers"): a retired/unused
      # tunnel (e.g. the long-dead Hermes .18 peer, or an unconnected spare)
      # must never permanently pin this check to WARN. Only the two named
      # peers below can move the check off OK.
      st="INFO"
    elif [ "$hs" -eq 0 ] || [ "$age" -gt "$HANDSHAKE_CRIT_SEC" ]; then
      st="CRIT"
    else
      st="OK"
    fi
    if [ "$st" != "INFO" ]; then
      r=$(status_rank "$st")
      [ "$r" -gt "$worst" ] && worst=$r
    fi
    parts+=("${label}(${pub:0:8}..)=${age}s:${st}")
  done <<< "$out"
  # A named peer entirely absent from the VPS's peer list (removed, not just
  # stale) would otherwise never be scored -- that is strictly worse than
  # stale, so force CRIT rather than silently staying OK/INFO-only.
  if [ "$seen_agentsprod" -eq 0 ]; then
    worst=2; parts+=("agentsprod(.10) NOT IN wg show output:CRIT")
  fi
  if [ "$seen_hnet00" -eq 0 ]; then
    worst=2; parts+=("HNET00 NOT IN wg show output:CRIT")
  fi
  CHECK_STATUS=$(rank_status "$worst")
  CHECK_DETAIL="peers: $(IFS='; '; echo "${parts[*]}")"
}

# (b) agents-prod (.10) egress must be the VPS; DNS must resolve; route to the
# VPS must go via eth0, not wg0 (the exact 2026-09-04 failure signature).
check_agentsprod_egress() {
  local out rc
  out=$(timeout 14 ssh -o BatchMode=yes -o ConnectTimeout=5 "${AGENTSPROD_USER}@${AGENTSPROD_HOST}" '
    echo "IP=$(curl -s -m 8 https://ifconfig.me)"
    echo "DNSLINE=$(getent hosts duckduckgo.com)"
    echo "ROUTELINE=$(ip route get 173.254.204.32 2>&1 | head -1)"
  ' 2>&1)
  rc=$?
  if [ $rc -ne 0 ]; then
    CHECK_STATUS="CRIT"; CHECK_DETAIL="ssh to ${AGENTSPROD_HOST} failed rc=${rc}: $(printf '%s' "$out" | tr '\n' ' ' | cut -c1-160)"
    return
  fi
  local ip dnsline routeline st="OK" parts=()
  ip=$(printf '%s\n' "$out" | sed -n 's/^IP=//p')
  dnsline=$(printf '%s\n' "$out" | sed -n 's/^DNSLINE=//p')
  routeline=$(printf '%s\n' "$out" | sed -n 's/^ROUTELINE=//p')
  if [ "$ip" = "$VPS_HOST" ]; then
    parts+=("egress-ip=OK(${ip})")
  else
    st="CRIT"; parts+=("egress-ip=CRIT(got '${ip:-empty}' want ${VPS_HOST})")
  fi
  if [ -n "$dnsline" ]; then
    parts+=("dns=OK(${dnsline})")
  else
    st="CRIT"; parts+=("dns=CRIT(duckduckgo.com did not resolve)")
  fi
  if printf '%s' "$routeline" | grep -q "dev wg0"; then
    st="CRIT"; parts+=("route=CRIT(via wg0: ${routeline})")
  elif printf '%s' "$routeline" | grep -Eq "via .+ dev "; then
    parts+=("route=OK(${routeline})")
  else
    [ "$st" = "OK" ] && st="WARN"
    parts+=("route=WARN(unparsed: ${routeline})")
  fi
  CHECK_STATUS="$st"
  CHECK_DETAIL="$(IFS='; '; echo "${parts[*]}")"
}

# (c1) HNET00's own wg-quick@wg0 unit must be active.
check_wg_active() {
  local st
  st=$(systemctl is-active wg-quick@wg0 2>&1)
  if [ "$st" = "active" ]; then CHECK_STATUS="OK"; else CHECK_STATUS="CRIT"; fi
  CHECK_DETAIL="wg-quick@wg0=${st}"
}

# (c2) docker bridge subnets must have their table-200 policy routes (these
# are what keep container egress correctly split from host egress).
check_ip_rules() {
  local rules miss=()
  rules=$(ip rule 2>&1)
  printf '%s' "$rules" | grep -q "172\.17\.0\.0/16 lookup 200" || miss+=("172.17.0.0/16->200")
  printf '%s' "$rules" | grep -q "172\.18\.0\.0/16 lookup 200" || miss+=("172.18.0.0/16->200")
  if [ "${#miss[@]}" -eq 0 ]; then
    CHECK_STATUS="OK"; CHECK_DETAIL="both docker-subnet -> table 200 policy rules present"
  else
    CHECK_STATUS="CRIT"; CHECK_DETAIL="missing ip rule(s): $(IFS=,; echo "${miss[*]}")"
  fi
}

# (c3) HNET00's own default egress must be the home IP, never the VPS
# (a HNET00 leak into its own tunnel would be a routing misconfiguration).
# Deliberately does NOT hardcode "the" home IP (residential IPs rotate) --
# only asserts it is not the VPS IP, and reports the observed value for a
# human to eyeball.
check_desktop_egress() {
  local ip
  ip=$(curl -s -m 8 https://ifconfig.me 2>/dev/null)
  if [ -z "$ip" ]; then
    CHECK_STATUS="WARN"; CHECK_DETAIL="curl ifconfig.me failed/timed out from HNET00"
  elif [ "$ip" = "$VPS_HOST" ]; then
    CHECK_STATUS="CRIT"; CHECK_DETAIL="HNET00 egress is the VPS (${ip}) -- unexpected tunnel leak"
  else
    CHECK_STATUS="OK"; CHECK_DETAIL="egress=${ip} (not VPS, as expected)"
  fi
}

# (c4) docker-a0-1 container egress must be the VPS (it's meant to run
# through the tunnel). Skips gracefully if docker or the container is absent.
check_docker_egress() {
  if [ "${WATCHDOG_SKIP_DOCKER:-0}" = "1" ]; then
    CHECK_STATUS="OK"
    CHECK_DETAIL="skipped: WATCHDOG_SKIP_DOCKER=1 (set for the sandboxed authoring/validation run; unset for the real cron job)"
    return
  fi
  if ! command -v docker >/dev/null 2>&1; then
    CHECK_STATUS="OK"; CHECK_DETAIL="docker not installed, skipped gracefully"
    return
  fi
  local running
  running=$(timeout 4 docker inspect -f '{{.State.Running}}' "$DOCKER_CONTAINER" 2>/dev/null)
  if [ "$running" != "true" ]; then
    CHECK_STATUS="WARN"; CHECK_DETAIL="container ${DOCKER_CONTAINER} not running or not found, skipped"
    return
  fi
  local ip rc
  ip=$(timeout 12 docker exec "$DOCKER_CONTAINER" curl -s -m 8 https://ifconfig.me 2>/dev/null)
  rc=$?
  if [ $rc -ne 0 ] || [ -z "$ip" ]; then
    CHECK_STATUS="WARN"; CHECK_DETAIL="docker exec curl failed rc=${rc} (container may lack curl, or egress broken)"
    return
  fi
  if [ "$ip" = "$VPS_HOST" ]; then
    CHECK_STATUS="OK"; CHECK_DETAIL="egress=${ip} (VPS, correct)"
  else
    CHECK_STATUS="CRIT"; CHECK_DETAIL="egress=${ip}, expected ${VPS_HOST} (container bypassing tunnel)"
  fi
}

# (d) research service health + at least one of the last 3 done jobs actually
# produced claims (claims_total > 0) -- catches a silently-degenerate run.
check_research_service() {
  local code
  code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "${RESEARCH_URL}/health" 2>/dev/null)
  if [ "$code" != "200" ]; then
    CHECK_STATUS="CRIT"; CHECK_DETAIL="GET ${RESEARCH_URL}/health -> HTTP ${code:-timeout}"
    return
  fi
  local list
  list=$(curl -s -m 6 "${RESEARCH_URL}/research?limit=5" 2>/dev/null)
  if [ -z "$list" ]; then
    CHECK_STATUS="WARN"; CHECK_DETAIL="health=200 but GET /research?limit=5 returned empty/unparseable"
    return
  fi
  local ids
  ids=$(printf '%s' "$list" | jq -r '(.jobs // []) | map(select(.status=="done")) | .[0:3][].job_id' 2>/dev/null)
  if [ -z "$ids" ]; then
    CHECK_STATUS="WARN"; CHECK_DETAIL="health=200, no done jobs found in the last 5 to sample"
    return
  fi
  local any_nonzero=0 checked=0 parts=() ct id
  while IFS= read -r id; do
    [ -z "$id" ] && continue
    checked=$((checked + 1))
    ct=$(curl -s -m 5 "${RESEARCH_URL}/research/${id}" 2>/dev/null | jq -r '.result.claims_total // 0' 2>/dev/null)
    case "$ct" in ''|*[!0-9]*) ct=0 ;; esac
    parts+=("${id}=${ct}")
    [ "$ct" -gt 0 ] && any_nonzero=1
  done <<< "$ids"
  if [ "$any_nonzero" -eq 1 ]; then CHECK_STATUS="OK"; else CHECK_STATUS="WARN"; fi
  CHECK_DETAIL="sampled ${checked} done job(s), claims_total[$(IFS=,; echo "${parts[*]}")]"
}

# (e1/e2) gateway + engine health, tolerant of a DEGRADED body / non-200 IFF
# a frontier-queue benchmark window is actively running (done/*.running).
_check_health_common() {
  local url="$1" resp code body
  resp=$(curl -s -m 5 -w $'\n%{http_code}' "$url" 2>/dev/null)
  code=$(printf '%s' "$resp" | tail -n1)
  body=$(printf '%s' "$resp" | sed '$d' | tr '\n' ' ' | cut -c1-200)
  if [ "$code" = "200" ]; then
    if printf '%s' "$body" | grep -qi "degraded"; then
      if frontier_benchmark_active; then
        CHECK_STATUS="WARN"; CHECK_DETAIL="HTTP 200 DEGRADED (benchmark window active via done/*.running, tolerated): ${body}"
      else
        CHECK_STATUS="WARN"; CHECK_DETAIL="HTTP 200 DEGRADED (no benchmark-window flag found): ${body}"
      fi
    else
      CHECK_STATUS="OK"; CHECK_DETAIL="HTTP 200: ${body:-<empty body>}"
    fi
  else
    if frontier_benchmark_active; then
      CHECK_STATUS="WARN"; CHECK_DETAIL="HTTP ${code:-timeout} (benchmark window active via done/*.running, tolerated)"
    else
      CHECK_STATUS="CRIT"; CHECK_DETAIL="HTTP ${code:-timeout}"
    fi
  fi
}
check_gateway_health() { _check_health_common "${GATEWAY_URL}/health"; }
check_engine_health() { _check_health_common "${ENGINE_URL}/health"; }

# (e3) core vLLM systemd units.
check_vllm_services() {
  local unit st worst="OK" parts=() note=""
  for unit in vllm-qwen27b vllm-qwen27b-watchdog.timer vllm-keepalive-shim; do
    st=$(systemctl is-active "$unit" 2>&1)
    if [ "$st" != "active" ]; then
      # 2026-09-05: every benchmark/promotion window stops the ENGINE watchdog timer on purpose
      # (so it cannot restart the engine mid-arm) and re-arms it in its restore step. While a
      # window marker (done/*.running) exists that is expected -> WARN with a reason, not CRIT.
      if [ "$unit" = "vllm-qwen27b-watchdog.timer" ] && engine_window_active; then
        [ "$worst" = "CRIT" ] || worst="WARN"; note=" (engine watchdog timer stopped by an active window, tolerated; windows should take a liveness hold instead)"
      else
        worst="CRIT"
      fi
    fi
    parts+=("${unit}=${st}")
  done
  CHECK_STATUS="$worst"
  CHECK_DETAIL="$(IFS='; '; echo "${parts[*]}")${note}"
}

# (e4) LV 2026-10-03: the ONE engine-liveness authority (engine-actuator.py tick, run by the watchdog timer each minute) publishes
# the declared engine state. This check READS it -- it never probes or acts on the engine itself. A stale file means the authority
# is not ticking (timer stopped, or the actuator crashing): that is the one thing nothing else would notice.
LIVENESS_STATE="${LIVENESS_STATE:-$HOME/.local/share/vllm-qwen27b/liveness-state.json}"
LIVENESS_HOLDS="${LIVENESS_HOLDS:-$HOME/.local/share/vllm-qwen27b/liveness-holds.json}"
LIVENESS_STALE_SEC="${LIVENESS_STALE_SEC:-300}"
TIMER_REARM_SEC="${TIMER_REARM_SEC:-1800}"
check_engine_liveness() {
  if [ ! -s "$LIVENESS_STATE" ]; then
    CHECK_STATUS="WARN"; CHECK_DETAIL="no ${LIVENESS_STATE} (liveness authority not deployed yet?)"; return
  fi
  local st as_of reason age
  st=$(jq -r '.state // "?"' "$LIVENESS_STATE" 2>/dev/null)
  as_of=$(jq -r '(.as_of // 0) | floor' "$LIVENESS_STATE" 2>/dev/null)
  reason=$(jq -r '.reason // ""' "$LIVENESS_STATE" 2>/dev/null | cut -c1-160)
  case "$as_of" in ''|*[!0-9]*) as_of=0 ;; esac
  age=$(( $(date +%s) - as_of ))
  if [ "$age" -gt "$LIVENESS_STALE_SEC" ]; then
    CHECK_STATUS="CRIT"; CHECK_DETAIL="liveness authority has not ticked for ${age}s (last state ${st}; watchdog timer $(systemctl is-active vllm-qwen27b-watchdog.timer 2>&1))"; return
  fi
  case "$st" in
    UP|BOOTING|PLANNED|HELD|OFFLINE_WINDOW|RECOVERING|STOPPING) CHECK_STATUS="OK" ;;
    CRASH_LOOP|BREAKER_OPEN) CHECK_STATUS="CRIT" ;;
    *) CHECK_STATUS="WARN" ;;
  esac
  CHECK_DETAIL="${st} (${age}s ago): ${reason}"
}

# Timer re-arm eligibility: the watchdog timer is the authority's clock. Windows take a TTL-bounded hold now instead of stopping it;
# a timer still found stopped long after every window has ended (DFT 10-02: 4.5 h) is re-armed by --fix.
engine_window_active() {
  frontier_benchmark_active && return 0
  [ -s "$LIVENESS_HOLDS" ] && jq -e --argjson now "$(date +%s)" 'any(.[]?; .kind == "engine" and (.until // 0) > $now)' "$LIVENESS_HOLDS" >/dev/null 2>&1 && return 0
  [ "$(curl -s -m 3 "${GATEWAY_URL}/gateway/offline" 2>/dev/null | jq -r '.offline // false' 2>/dev/null)" = "true" ] && return 0
  pgrep -f "bash [^ ]*(_window|_driver|/window)\.sh" >/dev/null 2>&1 && return 0
  return 1
}

timer_stopped_for() {   # seconds the engine watchdog timer has been inactive (0 when active / unknown)
  local st since
  st=$(systemctl is-active vllm-qwen27b-watchdog.timer 2>/dev/null)
  [ "$st" = "active" ] && { echo 0; return; }
  since=$(systemctl show vllm-qwen27b-watchdog.timer -p InactiveEnterTimestamp --timestamp=unix --value 2>/dev/null | tr -d '@')
  case "$since" in ''|*[!0-9]*) echo 0; return ;; esac
  echo $(( $(date +%s) - since ))
}

# (e2) vault MCP hub reachability (2026-09-05): the shared-brain write path for every agent.
# Its host 10.0.1.221 went down at ~13:5x on 2026-09-05 and nothing alerted -- the vault-sync check
# only inspects the local replica's frontmatter. TCP 27124 + an MCP initialize round-trip.
check_vault_mcp() {
  local host="${VAULT_MCP_HOST:-10.0.1.221}" port="${VAULT_MCP_PORT:-27124}" code
  if ! timeout 4 bash -c "exec 3<>/dev/tcp/${host}/${port}" 2>/dev/null; then
    CHECK_STATUS="CRIT"; CHECK_DETAIL="vault MCP ${host}:${port} unreachable (TCP) -- all agent vault writes are failing"; return
  fi
  code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' -X POST "http://${host}:${port}/mcp" -H 'Content-Type: application/json' \
         -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"estate-watchdog","version":"1"}}}' 2>/dev/null)
  case "$code" in
    200|202|400|406) CHECK_STATUS="OK"; CHECK_DETAIL="vault MCP ${host}:${port} answers (HTTP ${code})" ;;
    *) CHECK_STATUS="WARN"; CHECK_DETAIL="vault MCP ${host}:${port} TCP open but HTTP ${code:-timeout}" ;;
  esac
}

# (f) disk free space on /.
check_disk_free() {
  local avail_kb avail_gb
  avail_kb=$(df -Pk "$DISK_PATH" 2>/dev/null | awk 'NR==2{print $4}')
  case "$avail_kb" in ''|*[!0-9]*) avail_kb=0 ;; esac
  avail_gb=$(( avail_kb / 1024 / 1024 ))
  if [ "$avail_gb" -lt "$DISK_CRIT_GB" ]; then CHECK_STATUS="CRIT"
  elif [ "$avail_gb" -lt "$DISK_WARN_GB" ]; then CHECK_STATUS="WARN"
  else CHECK_STATUS="OK"
  fi
  CHECK_DETAIL="${avail_gb}GB free on ${DISK_PATH} (warn<${DISK_WARN_GB}, crit<${DISK_CRIT_GB})"
}

# (g1) both GPUs visible.
check_gpu_count() {
  local n
  n=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
  case "$n" in ''|*[!0-9]*) n=0 ;; esac
  if [ "$n" -eq 2 ]; then CHECK_STATUS="OK"
  elif [ "$n" -eq 0 ]; then CHECK_STATUS="CRIT"
  else CHECK_STATUS="WARN"
  fi
  CHECK_DETAIL="nvidia-smi -L reports ${n} GPU(s), expected 2"
}

# (g2) no Xid kernel errors in the last 10 minutes.
check_kernel_xid() {
  local hits newest marker archived=""
  hits=$(journalctl -k --since "-10 min" 2>/dev/null | grep -c "NVRM: Xid")
  case "$hits" in ''|*[!0-9]*) hits=0 ;; esac
  newest=$(journalctl -k -o short-iso --since "-10 min" 2>/dev/null | grep "NVRM: Xid" | tail -n1 | awk '{print $1}')
  if [ "${WATCHDOG_XID_TEST:-0}" = "1" ]; then hits=1; newest="TEST-$(date +%s)"; fi   # self-test hook, inert otherwise
  if [ "$hits" -gt 0 ]; then
    CHECK_STATUS="WARN"; CHECK_DETAIL="${hits} Xid line(s) in journalctl -k --since -10min"
    # 2026-09-06 incident archive (Kevin's estate item 9): the shim's flight recorder rotates (~40 bodies) and the
    # 2026-09-03 crash-night payloads were lost that way. On each NEW Xid event, freeze the evidence once:
    # newest 30 request bodies, engine journal, kernel Xid lines, last 20 min of gateway telemetry, GPU snapshot.
    marker="$STATE_DIR/last-xid-archived"
    if [ -n "$newest" ] && [ "$(cat "$marker" 2>/dev/null)" != "$newest" ] && [ "${WATCHDOG_INCIDENT_ARCHIVE:-1}" = "1" ]; then
      local inc="$HOME/.local/share/vllm-qwen27b/incidents/xid-$(date +%Y%m%d-%H%M%S)"
      mkdir -p "$inc/flightrec"
      ls -t "$HOME/.local/share/vllm-qwen27b/flightrec"/*.json 2>/dev/null | head -n 30 | xargs -r -I{} cp -p {} "$inc/flightrec/" 2>/dev/null || true
      journalctl -k -o short-iso --since "-20 min" 2>/dev/null | grep -E "NVRM: (Xid|GPU at PCI)" > "$inc/kernel-xid.txt" || true
      journalctl -u vllm-qwen27b --since "-20 min" --no-pager 2>/dev/null | tail -n 3000 > "$inc/engine-journal.txt" || true
      nvidia-smi -q 2>/dev/null | head -n 400 > "$inc/nvidia-smi-q.txt" || true
      ( cd "$HOME/.local/share/vllm-qwen27b/telemetry" 2>/dev/null && ls -t requests-*.jsonl 2>/dev/null | head -n 2 | xargs -r cat | jq -c --argjson t0 "$(( $(date +%s) - 1200 ))" 'select((.t // 0) >= $t0)' > "$inc/telemetry-last20min.jsonl" 2>/dev/null ) || true
      printf 'newest_xid=%s\narchived_at=%s\nhits_10min=%s\n' "$newest" "$(date -Is)" "$hits" > "$inc/META.txt"
      echo "$newest" > "$marker"
      archived=" -- evidence archived to ${inc}"
    fi
    CHECK_DETAIL="${CHECK_DETAIL}${archived}"
  else
    CHECK_STATUS="OK"; CHECK_DETAIL="no Xid errors in the last 10 minutes"
  fi
}

# (h) frontier-queue stuck-job detection (mirrors runner.sh's own 3h
# staleness definition for the RUNNING flag, applied consistently here).
check_frontier_queue() {
  local st="OK" parts=() stuck_n flag_age now
  stuck_n=$(find "${FRONTIER_QUEUE_DIR}/done" -maxdepth 1 -name '*.running' -mmin +180 2>/dev/null | wc -l)
  case "$stuck_n" in ''|*[!0-9]*) stuck_n=0 ;; esac
  if [ "$stuck_n" -gt 0 ]; then
    st="WARN"; parts+=("stuck-done-running=${stuck_n}(>3h)")
  else
    parts+=("stuck-done-running=0")
  fi
  if [ -f "${FRONTIER_QUEUE_DIR}/RUNNING" ]; then
    now=$(date +%s)
    flag_age=$(( now - $(stat -c %Y "${FRONTIER_QUEUE_DIR}/RUNNING" 2>/dev/null || echo "$now") ))
    if [ "$flag_age" -gt "$QUEUE_STALE_SEC" ]; then
      st="WARN"; parts+=("RUNNING-flag-age=${flag_age}s(stale>${QUEUE_STALE_SEC}s)")
    else
      parts+=("RUNNING-flag-age=${flag_age}s(fresh)")
    fi
  else
    parts+=("RUNNING-flag=absent")
  fi
  CHECK_STATUS="$st"
  CHECK_DETAIL="$(IFS='; '; echo "${parts[*]}")"
}

# (i) vault MEMORY.md exists and its frontmatter carries no old:/new: keys
# (the 2026-09-04 Dreams patch_note corruption signature).
check_vault_sync() {
  if [ ! -f "$VAULT_MEMORY" ]; then
    CHECK_STATUS="WARN"; CHECK_DETAIL="missing: ${VAULT_MEMORY}"
    return
  fi
  local fm bad
  fm=$(awk 'NR==1 && $0=="---"{f=1; next} f && $0=="---"{exit} f' "$VAULT_MEMORY" 2>/dev/null)
  bad=$(printf '%s\n' "$fm" | grep -Ec '^(old|new):')
  case "$bad" in ''|*[!0-9]*) bad=0 ;; esac
  if [ "$bad" -gt 0 ]; then
    CHECK_STATUS="WARN"; CHECK_DETAIL="frontmatter has ${bad} old:/new: key(s) -- corruption signature"
  else
    CHECK_STATUS="OK"; CHECK_DETAIL="frontmatter clean ($(wc -l < "$VAULT_MEMORY" 2>/dev/null | tr -d ' ') lines total)"
  fi
}

# ---------------------------------------------------------------------------
# Opt-in remediation (--fix only; default off). Narrow, rate-limited, and
# degrades to alert-only whenever sudo -n is refused -- never assumes sudo.
# ---------------------------------------------------------------------------
fix_signature() { printf '%s' "$1" | sha256sum | cut -d' ' -f1; }

fix_allowed() {
  local kind="$1" sig="$2" state last_file
  state="$STATE_DIR/fix-${kind}-circuit"
  last_file="$STATE_DIR/last-${kind}-fix-epoch"
  local old_sig="" failures=0 last=0 now
  [ -f "$state" ] && read -r old_sig failures < "$state" || true
  case "$failures" in ''|*[!0-9]*) failures=0 ;; esac
  if [ "$old_sig" = "$sig" ] && [ "$failures" -ge 2 ]; then
    log_line "FIX: ${kind} no-progress circuit open after ${failures} failed same-condition attempts; alert-only until evidence changes"
    return 1
  fi
  [ -f "$last_file" ] && last=$(cat "$last_file" 2>/dev/null || echo 0)
  case "$last" in ''|*[!0-9]*) last=0 ;; esac
  now=$(date +%s)
  if [ $(( now - last )) -lt "$FIX_COOLDOWN_SEC" ]; then
    return 1
  fi
  printf '%s\n' "$now" > "$last_file"
  return 0
}

fix_record() {
  local kind="$1" sig="$2" recovered="$3" state
  state="$STATE_DIR/fix-${kind}-circuit"
  local old_sig="" failures=0
  if [ "$recovered" = "1" ]; then
    rm -f "$state"
    return
  fi
  [ -f "$state" ] && read -r old_sig failures < "$state" || true
  case "$failures" in ''|*[!0-9]*) failures=0 ;; esac
  [ "$old_sig" = "$sig" ] || failures=0
  printf '%s %s\n' "$sig" "$((failures + 1))" > "${state}.tmp"
  mv -f "${state}.tmp" "$state"
}

do_fix() {
  # LV: re-arm the liveness authority's clock when it was left stopped and no window owns the engine any more.
  local tstop; tstop=$(timer_stopped_for)
  if [ "$tstop" -ge "$TIMER_REARM_SEC" ] && ! engine_window_active; then
    local tsig trec=0
    tsig=$(fix_signature "watchdog-timer-stopped|$(sha256sum "$0" 2>/dev/null | cut -d' ' -f1)")
    if fix_allowed "watchdog-timer" "$tsig"; then
      log_line "FIX: vllm-qwen27b-watchdog.timer stopped for ${tstop}s with no window owning the engine; re-arming"
      timeout 10 sudo -n systemctl start vllm-qwen27b-watchdog.timer >/dev/null 2>&1 && \
        [ "$(systemctl is-active vllm-qwen27b-watchdog.timer 2>/dev/null)" = "active" ] && trec=1
      fix_record "watchdog-timer" "$tsig" "$trec"
      [ "$trec" = "1" ] || log_line "FIX: watchdog timer re-arm did not verify; retained alert"
    fi
  fi

  if [ "${ST[agentsprod-egress]:-}" = "CRIT" ]; then
    local sig out recovered=0 _t0 _t1
    sig=$(fix_signature "${DET[agentsprod-egress]}|$(sha256sum "$0" 2>/dev/null | cut -d' ' -f1)")
    if fix_allowed "agentsprod-egress" "$sig"; then
      log_line "FIX: agentsprod-egress is CRIT, attempting .10 pin-route remediation"
      out=$(timeout 14 ssh -o BatchMode=yes -o ConnectTimeout=5 "${AGENTSPROD_USER}@${AGENTSPROD_HOST}" \
        "sudo -n ip route replace ${VPS_HOST}/32 via 10.0.1.1 dev eth0 2>&1; echo RC=\$?" 2>&1)
      if printf '%s' "$out" | grep -q "RC=0"; then
        _t0=$(now_ms); check_agentsprod_egress; _t1=$(now_ms)
        add_result "agentsprod-egress" "$CHECK_STATUS" "${CHECK_DETAIL} (post-fix recheck)" "$(( _t1 - _t0 ))"
        [ "$CHECK_STATUS" = "OK" ] && recovered=1
      fi
      fix_record "agentsprod-egress" "$sig" "$recovered"
      [ "$recovered" = "1" ] || log_line "FIX: .10 pin route did not verify; retained alert and no-progress evidence"
    fi
  fi

  if [ "${ST[hnet00-wg-active]:-}" = "CRIT" ]; then
    local sig out recovered=0 _t0 _t1
    sig=$(fix_signature "${DET[hnet00-wg-active]}|$(sudo -n sha256sum /etc/wireguard/wg0.conf 2>/dev/null | cut -d' ' -f1)|$(sha256sum "$0" 2>/dev/null | cut -d' ' -f1)")
    if fix_allowed "hnet00-wg-active" "$sig"; then
      log_line "FIX: hnet00-wg-active is CRIT, attempting sudo -n systemctl restart wg-quick@wg0"
      out=$(timeout 10 sudo -n systemctl restart wg-quick@wg0 2>&1; echo "RC=$?")
      if printf '%s' "$out" | grep -q "RC=0"; then
        sleep 2
        _t0=$(now_ms); check_wg_active; _t1=$(now_ms)
        add_result "hnet00-wg-active" "$CHECK_STATUS" "${CHECK_DETAIL} (post-fix recheck)" "$(( _t1 - _t0 ))"
        [ "$CHECK_STATUS" = "OK" ] && recovered=1
      fi
      fix_record "hnet00-wg-active" "$sig" "$recovered"
      [ "$recovered" = "1" ] || log_line "FIX: wg-quick@wg0 did not verify; retained alert and no-progress evidence"
    fi
  fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
FIX=0
for arg in "$@"; do
  case "$arg" in
    --fix) FIX=1 ;;
    *) echo "estate-watchdog: unknown argument '$arg' (ignored)" >&2 ;;
  esac
done

mkdir -p "$STATE_DIR"

if [ -f "$PAUSE_FILE" ]; then
  log_line "PAUSED (kill switch present: ${PAUSE_FILE}) -- skipping this cycle"
  exit 0
fi

rotate_log_if_needed

RUNDIR=$(mktemp -d "${STATE_DIR}/.run.XXXXXX")
trap 'rm -rf "$RUNDIR"' EXIT

# name:function pairs, run concurrently so total wall time is bounded by the
# SLOWEST single check's internal timeout, not the sum of all of them.
CHECK_LIST=(
  "vps-wireguard:check_vps_wireguard"
  "agentsprod-egress:check_agentsprod_egress"
  "hnet00-wg-active:check_wg_active"
  "hnet00-ip-rules:check_ip_rules"
  "hnet00-desktop-egress:check_desktop_egress"
  "docker-a0-egress:check_docker_egress"
  "research-service:check_research_service"
  "gateway-health:check_gateway_health"
  "engine-health:check_engine_health"
  "vllm-services:check_vllm_services"
  "engine-liveness:check_engine_liveness"
  "vault-mcp:check_vault_mcp"
  "disk-free:check_disk_free"
  "gpu-count:check_gpu_count"
  "kernel-xid:check_kernel_xid"
  "frontier-queue:check_frontier_queue"
  "vault-sync:check_vault_sync"
)

for entry in "${CHECK_LIST[@]}"; do
  name="${entry%%:*}"
  fn="${entry#*:}"
  (
    t0=$(now_ms)
    CHECK_STATUS="WARN"; CHECK_DETAIL="check did not set a result"
    "$fn"
    t1=$(now_ms)
    printf '%s\n%s\n%s\n' "$CHECK_STATUS" "$CHECK_DETAIL" "$(( t1 - t0 ))" > "$RUNDIR/$name.out" 2>/dev/null
  ) &
done
wait

for entry in "${CHECK_LIST[@]}"; do
  name="${entry%%:*}"
  f="$RUNDIR/$name.out"
  if [ -s "$f" ]; then
    st=$(sed -n '1p' "$f"); det=$(sed -n '2p' "$f"); ms=$(sed -n '3p' "$f")
  else
    st="WARN"; det="check crashed / produced no output"; ms=0
  fi
  case "$ms" in ''|*[!0-9]*) ms=0 ;; esac
  add_result "$name" "$st" "$det" "$ms"
done

if [ "$FIX" -eq 1 ]; then
  do_fix
fi

OVERALL=$(compute_overall)
PREV_OVERALL=""
[ -f "$STATE_FILE" ] && PREV_OVERALL=$(jq -r '.overall // empty' "$STATE_FILE" 2>/dev/null)

for name in "${CHECK_ORDER[@]}"; do
  line=$(printf '%s %-22s %s' "${ST[$name]}" "$name" "${DET[$name]}")
  echo "$line"
  echo "$(date '+%F %T %Z') $line" >> "$LOG_FILE"
done

SUMMARY_LINE="OVERALL=${OVERALL} (prev=${PREV_OVERALL:-none}) at $(date '+%F %T %Z')"
echo "$SUMMARY_LINE"
echo "$(date '+%F %T %Z') $SUMMARY_LINE" >> "$LOG_FILE"

checks_json=$(build_checks_json)
state_json=$(jq -nc --arg ts "$(date -u +%FT%TZ)" --arg overall "$OVERALL" --argjson checks "$checks_json" \
  '{ts:$ts, overall:$overall, checks:$checks}')
write_state "$state_json"

if [ "$OVERALL" != "OK" ]; then
  {
    echo "estate-watchdog ALERT overall=${OVERALL} at $(date '+%F %T %Z') on $(hostname)"
    for name in "${CHECK_ORDER[@]}"; do
      [ "${ST[$name]}" != "OK" ] && printf '%s %s: %s\n' "${ST[$name]}" "$name" "${DET[$name]}"
    done
  } > "$ALERT_FILE"
else
  [ -f "$ALERT_FILE" ] && rm -f "$ALERT_FILE"
fi

if [ "$OVERALL" != "${PREV_OVERALL:-OK}" ] && [ -f "$DISCORD_WEBHOOK_FILE" ]; then
  webhook_url=$(tr -d ' \t\n\r' < "$DISCORD_WEBHOOK_FILE")
  if [ -n "$webhook_url" ]; then
    bad=""
    if [ "$OVERALL" != "OK" ]; then
      bad=" -- $(for n in "${CHECK_ORDER[@]}"; do [ "${ST[$n]}" != "OK" ] && echo "${ST[$n]} ${n}"; done | paste -sd '; ' -)"
    fi
    msg="estate-watchdog: ${PREV_OVERALL:-none} -> ${OVERALL} on $(hostname) at $(date '+%F %T %Z')${bad}"
    payload=$(jq -nc --arg content "$msg" '{content:$content}')
    if ! curl -s -m 6 -H 'Content-Type: application/json' -d "$payload" "$webhook_url" >/dev/null 2>&1; then
      log_line "discord webhook POST failed (non-fatal)"
    fi
  fi
fi

exit 0
