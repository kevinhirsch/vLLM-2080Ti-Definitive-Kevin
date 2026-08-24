#!/usr/bin/env bash
# update-from-remote.sh — fast local update from the GitHub fork (+ upstream status).
#
# Remotes (as configured in this clone):
#   kevin  = https://github.com/kevinhirsch/vLLM-2080Ti-Definitive-Kevin  (our fork)
#   origin = https://github.com/weicj/vLLM-2080Ti-Definitive              (upstream)
# Canonical branch: vllm-2080ti-definitive-0.1.x-kevin (fork default).
#
# Usage:
#   update-from-remote.sh              # status only: fetch + ahead/behind report
#   update-from-remote.sh --apply      # ff-only update of the current branch from the fork default
#   update-from-remote.sh --apply --restart
#                                      # ...then restart the engine under the safe window:
#                                      # gateway->DeepSeek, restart, health-wait, smoke, restore
#
# Deterministic, no LLM (cron policy). Safe by construction:
#   - --apply refuses non-fast-forward (never rewrites local work; prints what to do)
#   - --restart uses the exact window sequence proven in ops (Hermes stays live on
#     the DeepSeek failover; watchdog stopped/restarted around the engine restart)
set -euo pipefail

REPO=/home/kevin/Desktop/vLLM-2080Ti-Definitive
BRANCH_REMOTE=kevin
BRANCH=vllm-2080ti-definitive-0.1.x-kevin
UPSTREAM_REMOTE=origin
UPSTREAM_BRANCH=vllm-2080ti-definitive-0.1.x
GATEWAY=http://127.0.0.1:8000
ENGINE=http://127.0.0.1:8001

cd "$REPO"
APPLY=0; RESTART=0
for a in "$@"; do case "$a" in --apply) APPLY=1;; --restart) RESTART=1;; *) echo "unknown arg: $a"; exit 2;; esac; done

echo "== fetch =="
git fetch "$BRANCH_REMOTE" "$BRANCH" 2>&1 | tail -1 || true
git fetch "$UPSTREAM_REMOTE" "$UPSTREAM_BRANCH" 2>&1 | tail -1 || true

CUR=$(git branch --show-current)
LOCAL=$(git rev-parse --short HEAD)
FORK=$(git rev-parse --short "$BRANCH_REMOTE/$BRANCH")
read -r BEHIND AHEAD <<<"$(git rev-list --left-right --count "$BRANCH_REMOTE/$BRANCH...HEAD" | tr '\t' ' ')"
echo "== status =="
echo "  branch: $CUR @ $LOCAL | fork default: $FORK"
echo "  vs fork default : ahead $AHEAD, behind $BEHIND"
UP_NEW=$(git rev-list --count "HEAD..$UPSTREAM_REMOTE/$UPSTREAM_BRANCH" 2>/dev/null || \
         git rev-list --count HEAD..FETCH_HEAD 2>/dev/null || echo "?")
echo "  upstream commits not in local: $UP_NEW  (upstream sync is a manual merge, not --apply)"

if [ "$APPLY" = 1 ]; then
  echo "== apply (ff-only) =="
  if [ "$BEHIND" = 0 ]; then
    echo "  already up to date."
  elif [ "$AHEAD" != 0 ]; then
    echo "  REFUSING: local is ahead by $AHEAD commit(s) — push or reconcile first:"
    git log --oneline "$BRANCH_REMOTE/$BRANCH..HEAD" | sed 's/^/    /'
    exit 1
  else
    git merge --ff-only "$BRANCH_REMOTE/$BRANCH"
    echo "  updated to $(git rev-parse --short HEAD)"
    echo "  NOTE: the running engine still executes the OLD code until restarted."
  fi
fi

if [ "$RESTART" = 1 ]; then
  echo "== engine restart window =="
  echo "  gateway -> remote (Hermes stays live on DeepSeek failover)"
  curl -s -X POST "$GATEWAY/gateway/config" -H "Content-Type: application/json" \
       -d '{"force_remote": 1}' >/dev/null
  sudo systemctl stop vllm-qwen27b-watchdog.timer
  sudo systemctl restart vllm-qwen27b.service
  echo -n "  waiting for health"
  ok=0
  for i in $(seq 1 60); do
    if curl -sf -m3 "$ENGINE/health" >/dev/null 2>&1; then ok=1; echo " HEALTHY (~$((i*10))s)"; break; fi
    echo -n "."; sleep 10
  done
  if [ "$ok" = 1 ]; then
    SMOKE=$(curl -s "$ENGINE/v1/chat/completions" -H 'Content-Type: application/json' \
      -d '{"model":"qwen-local","max_tokens":8,"temperature":0,"messages":[{"role":"user","content":"Reply with exactly: OK"}]}' \
      | grep -c '"content"' || true)
    echo "  smoke: $([ "$SMOKE" -ge 1 ] && echo PASS || echo FAIL)"
    curl -s -X POST "$GATEWAY/gateway/config" -H "Content-Type: application/json" \
         -d '{"force_remote": 0}' >/dev/null
    sudo systemctl start vllm-qwen27b-watchdog.timer
    echo "  gateway -> local, watchdog -> active. DONE."
  else
    echo "  ENGINE NOT HEALTHY after 10 min — leaving gateway on remote (traffic keeps working)."
    echo "  Investigate: journalctl -u vllm-qwen27b -n 50 | rollback: git checkout pre-consolidation-rollback && $0 --restart"
    exit 1
  fi
fi
