#!/usr/bin/env bash
# vllm-qwen27b generation watchdog.
#
# Runs as a systemd oneshot service on a 60s timer (mirrors the az-watchdog.service/.timer
# pattern already in use on this box). Each invocation does exactly ONE probe cycle:
#
#   1. GET  ENDPOINT_URL/v1/models        (cheap liveness of the vLLM engine on :8001)
#   2. POST ENDPOINT_URL/v1/chat/completions, max_tokens=5, enable_thinking=false, hard timeout
#      (the real signal: is the engine actually GENERATING, not just answering metadata?)
#
# NOTE (2026-09-05): probe the ENGINE DIRECTLY on :8001, never the gateway shim on :8000.
# The shim fails over to DeepSeek when it marks local-down, so a probe through :8000 gets a
# remote 200 and this watchdog logs HEALTHY while the engine is dead -- exactly what happened
# 18:43-18:48 today (Xid 31 on TP1, EngineCore blocked on shm_broadcast for 4 min, watchdog
# reset to 0 failures four times). The var is ENDPOINT_URL; ROUTER_URL is still honored.
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
ENDPOINT_URL="${ENDPOINT_URL:-${ROUTER_URL:-http://127.0.0.1:8001}}"   # :8001 = the vLLM engine itself (:8000 is the failover shim -- see NOTE above)
# Auto-discover the served model from the endpoint so a model swap (e.g. Qwen 3.6 -> 3.8)
# doesn't make the generation probe request a now-404 model id and false-positive as a wedge.
# Falls back to qwen3.6:27b only if discovery fails. Override with WATCHDOG_MODEL.
MODEL="${WATCHDOG_MODEL:-$(curl -s -m 5 "$ENDPOINT_URL/v1/models" 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo qwen3.6:27b)}"
MODELS_TIMEOUT="${WATCHDOG_MODELS_TIMEOUT:-5}"       # seconds, GET /v1/models budget
GEN_TIMEOUT="${WATCHDOG_GEN_TIMEOUT:-20}"            # seconds, hard timeout for the generation probe
CONSEC_FAIL_THRESHOLD="${WATCHDOG_CONSEC_FAIL_THRESHOLD:-5}"   # consecutive gen failures required
XID_FAST_THRESHOLD="${WATCHDOG_XID_FAST_THRESHOLD:-2}"   # RS: consecutive gen failures required when a kernel Xid corroborates
XID_WINDOW_MIN="${WATCHDOG_XID_WINDOW_MIN:-8}"           # RS: how recent the Xid must be
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

# A queued watchdog request is not evidence of a wedged engine. Require the
# engine's own counters to stay flat for the entire failed probe. Unreadable
# counters fail closed: they cannot authorize a destructive restart.
probe_progress() {
  local metrics
  metrics=$(curl -fsS -m 5 "$ENDPOINT_URL/metrics" 2>/dev/null) || return 1
  printf '%s\n' "$metrics" | awk '
    /^vllm:prompt_tokens_total[{ ]/ { prompt += $NF; have_prompt = 1 }
    /^vllm:generation_tokens_total[{ ]/ { generation += $NF; have_generation = 1 }
    END { if (have_prompt && have_generation) print prompt, generation; else exit 1 }
  '
}

# ---------- main ----------
init_state

models_result=$(probe_models)
models_code="${models_result%% *}"
models_time="${models_result#* }"

progress_before=$(probe_progress) || progress_before=""
gen_result=$(probe_generation)
gen_code="${gen_result%% *}"
gen_time="${gen_result#* }"
progress_after=$(probe_progress) || progress_after=""

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

if [ -z "$progress_before" ] || [ -z "$progress_after" ]; then
  log "PROBE fail models=${models_code} gen=${gen_code} -> engine progress metrics unreadable; restart not authorized"
  [ "$prev_failures" != "0" ] && state_set_consecutive_failures 0
  exit 0
fi

if ! awk -v before="$progress_before" -v after="$progress_after" '
  BEGIN {
    split(before, b, " "); split(after, a, " ");
    # A counter reset means the engine may have restarted independently.
    exit (a[1] >= b[1] && a[2] >= b[2]) ? 0 : 1
  }
'; then
  log "PROBE fail models=${models_code} gen=${gen_code} -> engine counters reset; restart not authorized"
  [ "$prev_failures" != "0" ] && state_set_consecutive_failures 0
  exit 0
fi

if awk -v before="$progress_before" -v after="$progress_after" '
  BEGIN { split(before, b, " "); split(after, a, " "); exit (a[1] > b[1] || a[2] > b[2]) ? 0 : 1 }
'; then
  log "PROBE fail models=${models_code} gen=${gen_code} -> engine progressing (${progress_before} to ${progress_after}); watchdog request queued, no restart"
  [ "$prev_failures" != "0" ] && state_set_consecutive_failures 0
  exit 0
fi

# models OK, generation failed -> this is the targeted signature. Count it.
new_failures=$((prev_failures + 1))
state_set_consecutive_failures "$new_failures"
log "PROBE fail models=${models_code}(${models_time}s) gen=${gen_code}(${gen_time}s) -> generation-wedge signature, consecutive_failures=${new_failures}/${CONSEC_FAIL_THRESHOLD}"

# RS (2026-10-02): a kernel Xid (13/31/43/45/79) in the last XID_WINDOW_MIN minutes corroborates a generation failure: the
# 10:34:40 and 12:00:30 CUDA faults each left a zombie engine (API up, workers dead) that sat out ALL 5 probes = 4 min 52 s of
# full outage (local-only: nothing else can serve) before the kill. With an Xid on record the wedge is CONFIRMED by 2 probes
# (still two consecutive failures, so a transient is not killed). The cooldown and restarts/hour rails below still apply.
CONSEC_EFFECTIVE="$CONSEC_FAIL_THRESHOLD"
xid_recent=$(journalctl -k --since "-${XID_WINDOW_MIN} min" --no-pager 2>/dev/null | grep -cE 'NVRM: Xid .*: (13|31|43|45|79),' || true)
if [ "${xid_recent:-0}" -gt 0 ] && [ "$XID_FAST_THRESHOLD" -lt "$CONSEC_EFFECTIVE" ]; then
  CONSEC_EFFECTIVE="$XID_FAST_THRESHOLD"
  log "XID-CORROBORATED ${xid_recent} kernel Xid line(s) in the last ${XID_WINDOW_MIN} min -> wedge threshold ${CONSEC_EFFECTIVE} (instead of ${CONSEC_FAIL_THRESHOLD})"
fi
if [ "$new_failures" -lt "$CONSEC_EFFECTIVE" ]; then
  log "DECISION below threshold (${new_failures}/${CONSEC_EFFECTIVE}), no action"
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
# 2026-09-05 (wedge RCA): a CONFIRMED wedge has never honoured SIGTERM (0-for-2: 2026-09-02 and
# 2026-09-05 both sat out the full TimeoutStopSec=180 before systemd's SIGKILL). Kill the whole
# control group up front so the restart starts immediately; the 180 s grace stays for operator
# restarts, where a clean TP=2 teardown is worth waiting for.
# EF2: tell the fault collector this death is a confirmed generation wedge (a FAULT for Halo, not a planned stop).
printf '{"ts":"%s","by":"watchdog","wedge":true,"consecutive_failures":%s}\n' "$(date -Is)" "${new_failures}" > "$(dirname "$STATE_FILE")/wedge-restart.json" 2>/dev/null || true
# RS (2026-10-02): record the death BEFORE killing. Two wedge kills (10:39, 12:05) never reached the fault ledger: the
# ExecStopPost collector raced the `systemctl restart` issued 3 s after this kill and its record was lost. --pre-kill writes
# ledger + incident dir + Halo hand-off now, with the journal still intact, and drops a dedupe marker so the later ExecStopPost
# does not double-count. Bounded (timeout) and best-effort: a collector failure never delays recovery beyond the bound.
timeout 60 /usr/bin/python3 "$(dirname "$STATE_FILE")/engine-fault-collector.py" --pre-kill >> "$ACTION_LOG" 2>&1 || true
sudo -n systemctl kill -s KILL "$SERVICE" >> "$ACTION_LOG" 2>&1 || true
sleep 3
log "ACTION SIGKILL sent to ${SERVICE} control group (wedged engines never exit on SIGTERM); restarting now"
if sudo -n systemctl restart "$SERVICE" >> "$ACTION_LOG" 2>&1; then
  state_record_restart "$t"
  state_set_consecutive_failures 0
  log "ACTION restart of ${SERVICE} issued successfully (sudo systemctl restart exit 0)"
else
  rc=$?
  log "ACTION restart of ${SERVICE} FAILED (sudo systemctl restart exit ${rc}) -- state NOT updated, will retry next cycle if still wedged"
fi
