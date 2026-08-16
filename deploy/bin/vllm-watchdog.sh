#!/usr/bin/env bash
# vllm-qwen27b generation watchdog.
#
# Runs as a systemd oneshot service on a 60s timer (mirrors the az-watchdog.service/.timer
# pattern already in use on this box). Each invocation does exactly ONE probe cycle:
#
#   1. GET  ENDPOINT_URL/v1/models        (cheap liveness of the vLLM engine on :8000)
#   2. POST ENDPOINT_URL/v1/chat/completions, max_tokens=5, enable_thinking=false, hard timeout
#      (the real signal: is the engine actually GENERATING, not just answering metadata?)
#
# NOTE (2026-08-11): :8000 is now served by vLLM DIRECTLY — the model-router was retired.
# This watchdog probes/guards the vLLM engine itself. The var is ENDPOINT_URL; the old
# ROUTER_URL env name is still honored for back-compat.
#
# Only declares a "generation wedge" — and only then considers restarting the service —
# when the generation probe has failed CONSEC_FAIL_THRESHOLD times *in a row* while
# /v1/models kept returning 200 the whole time (alive-but-not-generating, matching the
# 2026-08-10 incident signature). A single slow/failed generation probe under legitimate
# heavy load is expected and normal on this box (max-num-seqs=2, up to 256K context) — see
# WEDGE-ROOTCAUSE-2026-08-10.md, which documents this exact false-positive risk being
# reproduced live. That is why this script requires several consecutive failures, not one.
#
# Safety rails:
#   - COOLDOWN_SEC: minimum gap between automated restarts (default 1800s / 30 min).
#   - MAX_RESTARTS_PER_HOUR: hard cap on restarts/hour (default 2).
#   - --dry-run: never actually restarts; logs "[DRY-RUN] would restart ..." instead.
#     State mutations related to restart bookkeeping are skipped in dry-run so testing never
#     pollutes the cooldown/rate-limit state the real (future, enabled) watchdog depends on.
#
# TUNING NOTE (empirical, 2026-08-10): while validating this script against the live, healthy
# service, a live re-score job was saturating the backend (max-num-seqs=2, long-context
# requests, GPU util 93-100%). A tiny 5-token test probe failed to win a scheduling slot even
# with generous 45-120s single-request budgets, while the backend's OWN request log showed a
# continuous stream of OTHER requests completing successfully the whole time -- i.e. genuinely
# healthy, just deeply queued for a low-priority newcomer. CONSEC_FAIL_THRESHOLD defaults to 5
# (5 probe cycles = ~5 min of continuous inability to complete even a tiny request) specifically
# because of this: a lower threshold (e.g. 3) would very plausibly false-positive and restart a
# healthy-but-busy engine during ordinary heavy production traffic like the observed re-score
# job. Retune CONSEC_FAIL_THRESHOLD / GEN_TIMEOUT upward if this box's normal worst-case queuing
# depth is known to exceed ~5 minutes under legitimate load -- see WEDGE-ROOTCAUSE-2026-08-10.md.
#
# State persists between invocations in a small JSON file (STATE_FILE). All probe results
# and all restart decisions are appended to ACTION_LOG (never truncated).
#
# INSTALLED DORMANT: this script and its systemd units exist on disk but the timer is not
# enabled/started. See the ENABLE section at the bottom of this file's header comment, or
# the root-cause doc, for the exact command to turn it on.
#
# ENABLE LATER WITH:
#   sudo systemctl enable --now vllm-qwen27b-watchdog.timer
#
set -uo pipefail

# ---------- config (override via env) ----------
ENDPOINT_URL="${ENDPOINT_URL:-${ROUTER_URL:-http://localhost:8000}}"   # :8000 = vLLM direct (router retired 2026-08-11); ROUTER_URL still honored
# Auto-discover the served model from the endpoint so a model swap (e.g. Qwen 3.6 -> 3.8)
# doesn't make the generation probe request a now-404 model id and false-positive as a wedge.
# Falls back to qwen3.6:27b only if discovery fails. Override with WATCHDOG_MODEL.
MODEL="${WATCHDOG_MODEL:-$(curl -s -m 5 "$ENDPOINT_URL/v1/models" 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo qwen3.6:27b)}"
MODELS_TIMEOUT="${WATCHDOG_MODELS_TIMEOUT:-5}"       # seconds, GET /v1/models budget
GEN_TIMEOUT="${WATCHDOG_GEN_TIMEOUT:-20}"            # seconds, hard timeout for the generation probe
CONSEC_FAIL_THRESHOLD="${WATCHDOG_CONSEC_FAIL_THRESHOLD:-5}"   # consecutive gen failures required
COOLDOWN_SEC="${WATCHDOG_COOLDOWN_SEC:-1800}"        # 30 min minimum between automated restarts
MAX_RESTARTS_PER_HOUR="${WATCHDOG_MAX_RESTARTS_PER_HOUR:-2}"
SERVICE="${WATCHDOG_SERVICE:-vllm-qwen27b.service}"

D="/home/kevin/.local/share/vllm-qwen27b"
STATE_FILE="${WATCHDOG_STATE_FILE:-$D/watchdog-state.json}"
ACTION_LOG="${WATCHDOG_ACTION_LOG:-$D/watchdog.log}"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) ;;
  esac
done

# ---------- helpers ----------
now_epoch() { date +%s; }
ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

log() {
  # Append-only action/probe log. Never truncated.
  printf '%s %s\n' "$(ts)" "$1" >> "$ACTION_LOG"
}

init_state() {
  if [ ! -f "$STATE_FILE" ]; then
    printf '{"consecutive_failures":0,"last_restart_epoch":0,"restart_timestamps":[]}\n' > "$STATE_FILE"
  fi
}

state_get() { jq -r ".$1" "$STATE_FILE"; }

state_set_consecutive_failures() {
  local n="$1"
  jq --argjson n "$n" '.consecutive_failures = $n' "$STATE_FILE" > "$STATE_FILE.tmp" && mv "$STATE_FILE.tmp" "$STATE_FILE"
}

state_record_restart() {
  local t="$1"
  jq --argjson t "$t" \
     '.last_restart_epoch = $t | .restart_timestamps = ((.restart_timestamps + [$t]) | map(select(. > ($t - 3600))))' \
     "$STATE_FILE" > "$STATE_FILE.tmp" && mv "$STATE_FILE.tmp" "$STATE_FILE"
}

restarts_in_last_hour() {
  local t; t=$(now_epoch)
  jq --argjson t "$t" '[.restart_timestamps[] | select(. > ($t - 3600))] | length' "$STATE_FILE"
}

# ---------- probes ----------
# Returns 0 and prints "<http_code> <elapsed_s>" on completion (any HTTP code, even error);
# prints "000 <elapsed_s>" if curl itself failed/timed out.
probe_models() {
  local out
  out=$(curl -s -o /dev/null -m "$MODELS_TIMEOUT" -w '%{http_code} %{time_total}' \
        "$ENDPOINT_URL/v1/models" 2>/dev/null) || out="000 $MODELS_TIMEOUT.000"
  echo "$out"
}

probe_generation() {
  local out
  out=$(curl -s -o /dev/null -m "$GEN_TIMEOUT" -w '%{http_code} %{time_total}' \
        "$ENDPOINT_URL/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: OK\"}],\"max_tokens\":5,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
        2>/dev/null) || out="000 $GEN_TIMEOUT.000"
  echo "$out"
}

# ---------- main ----------
init_state

models_result=$(probe_models)
models_code="${models_result%% *}"
models_time="${models_result#* }"

gen_result=$(probe_generation)
gen_code="${gen_result%% *}"
gen_time="${gen_result#* }"

models_ok=0; [ "$models_code" = "200" ] && models_ok=1
gen_ok=0; [ "$gen_code" = "200" ] && gen_ok=1

prev_failures=$(state_get consecutive_failures)

if [ "$gen_ok" = "1" ]; then
  log "PROBE ok models=${models_code}(${models_time}s) gen=${gen_code}(${gen_time}s) consecutive_failures=0 (reset from ${prev_failures}) -> HEALTHY"
  [ "$prev_failures" != "0" ] && state_set_consecutive_failures 0
  exit 0
fi

# Generation probe failed.
if [ "$models_ok" != "1" ]; then
  # Router itself isn't answering /v1/models either -> not the "alive-but-not-generating"
  # signature this watchdog targets (could be a full outage, a fresh cold-load in progress,
  # or a network blip). Do not count it toward the wedge threshold; just log it.
  log "PROBE fail models=${models_code}(${models_time}s) gen=${gen_code}(${gen_time}s) -> NOT the targeted wedge signature (models also down); no action, consecutive_failures unchanged (${prev_failures})"
  exit 0
fi

# models OK, generation failed -> this is the targeted signature. Count it.
new_failures=$((prev_failures + 1))
state_set_consecutive_failures "$new_failures"
log "PROBE fail models=${models_code}(${models_time}s) gen=${gen_code}(${gen_time}s) -> generation-wedge signature, consecutive_failures=${new_failures}/${CONSEC_FAIL_THRESHOLD}"

if [ "$new_failures" -lt "$CONSEC_FAIL_THRESHOLD" ]; then
  log "DECISION below threshold (${new_failures}/${CONSEC_FAIL_THRESHOLD}), no action"
  exit 0
fi

# Threshold reached -> confirmed wedge. Check safety rails before acting.
t=$(now_epoch)
last_restart=$(state_get last_restart_epoch)
since_last=$((t - last_restart))

if [ "$last_restart" != "0" ] && [ "$since_last" -lt "$COOLDOWN_SEC" ]; then
  remaining=$((COOLDOWN_SEC - since_last))
  log "DECISION wedge CONFIRMED (consecutive_failures=${new_failures}) but within cooldown (${remaining}s remaining of ${COOLDOWN_SEC}s) -> SKIPPING restart"
  exit 0
fi

rph=$(restarts_in_last_hour)
if [ "$rph" -ge "$MAX_RESTARTS_PER_HOUR" ]; then
  log "DECISION wedge CONFIRMED (consecutive_failures=${new_failures}) but max-restarts-per-hour reached (${rph}/${MAX_RESTARTS_PER_HOUR}) -> SKIPPING restart"
  exit 0
fi

if [ "$DRY_RUN" = "1" ]; then
  log "DECISION wedge CONFIRMED (consecutive_failures=${new_failures}, models healthy, gen timed out ${new_failures}x consecutively) -> [DRY-RUN] would restart ${SERVICE} via: sudo systemctl restart ${SERVICE} (state NOT updated, so this test run does not consume real cooldown/rate-limit budget)"
  exit 0
fi

log "DECISION wedge CONFIRMED (consecutive_failures=${new_failures}, models healthy, gen timed out ${new_failures}x consecutively) -> RESTARTING ${SERVICE}"
# Clear any StartLimitBurst latch FIRST. systemd stops honouring `restart` once a unit has
# failed too many times in the interval, and a latched unit silently ignores the command --
# the watchdog then logs a successful restart that never happened. This is not hypothetical:
# it stranded the engine on 2026-08-14 during the quantization window.
sudo -n systemctl reset-failed "$SERVICE" >> "$ACTION_LOG" 2>&1 || true
if sudo -n systemctl restart "$SERVICE" >> "$ACTION_LOG" 2>&1; then
  state_record_restart "$t"
  state_set_consecutive_failures 0
  log "ACTION restart of ${SERVICE} issued successfully (sudo systemctl restart exit 0)"
else
  rc=$?
  log "ACTION restart of ${SERVICE} FAILED (sudo systemctl restart exit ${rc}) -- state NOT updated, will retry next cycle if still wedged"
fi
