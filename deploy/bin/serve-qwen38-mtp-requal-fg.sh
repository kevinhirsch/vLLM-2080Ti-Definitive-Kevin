#!/usr/bin/env bash
# serve-qwen38-mtp-requal-fg.sh — the MTP RE-QUALIFICATION variant of the live
# serve script. Identical to serve-tqk8v4-fg.sh (2026-08-15 MTP-off config)
# EXCEPT:
#
#   1. MTP speculative decoding is back ON (k = MTP_K, default 3).
#   2. Structured outputs pinned to the "guidance" (llguidance) backend —
#      guided/named tool calls are grammar-enforced WITHOUT xgrammar, which is
#      the backend with the documented spec-decode+Int4 crash (vLLM #11484).
#
# WHY RE-TRY MTP (it garbled/crashed at ~57-64K on 2026-08-14):
#   - upstream v0.1.15 40129ea: preserve hybrid Mamba prefix-cache correctness
#     with MTP  <- we run prefix caching + MTP + GDN, i.e. exactly this bug
#   - upstream v0.1.15 c256ad2: preserve GDN state slot zero during decode
#   - this fork 746d8b1: clamp negative draft-token ids before embedding lookup
#     (the cudagraph-replay illegal-address crash)
# Together these cover both halves of the incident (garble + crash). MTP-off
# costs ~40-60% single-stream decode; this variant exists to win that back.
#
# QUALIFY BEFORE ADOPTING — never point clients at this without a green run of:
#   deploy/bench/bench_equivalence.py  (record on MTP-off first, check here)
#   deploy/bench/bench_mtp_requal.py   (the context ladder across 57-64K)
#   deploy/bench/bench_toolcalls.sh    (incl. named tool_choice + code args)
#   deploy/bench/bench_decode.py       (the reference matrix)
# Full procedure: deploy/docs/QWEN-3.8-MTP-REQUALIFICATION.md
#
# ACTIVATE (drop-in, no edits to the live script):
#   sudo install -m644 deploy/systemd/vllm-qwen27b.service.d/mtp-requal.conf.example \
#        /etc/systemd/system/vllm-qwen27b.service.d/mtp-requal.conf
#   sudo systemctl daemon-reload && sudo systemctl restart vllm-qwen27b
# ROLLBACK (back to MTP-off in one move):
#   sudo rm /etc/systemd/system/vllm-qwen27b.service.d/mtp-requal.conf
#   sudo systemctl daemon-reload && sudo systemctl restart vllm-qwen27b
#
# Tunables (env): MTP_K=3   speculative tokens (sweep 2/3 via bench)
set -euo pipefail
export VLLM_SUFFIX_OVERLAY=0
export VLLM_MTP_DRAFT_CAP=3
export VLLM_SUFFIX_COVER_MIN=4
export VLLM_SUFFIX_OVERLAY_MIN=2

MTP_K="${MTP_K:-3}"

# During qualification: surface the open Xid31 TQ-continuation fault hunt
# (see fix-xid31-guard / QWEN-3.8-MTP-REQUALIFICATION.md prereq A) instead of
# letting it masquerade as an MTP failure. Harmless no-op on builds without
# the guard commits.
export VLLM_TURBOQUANT_CONTINUATION_BOUNDS_CHECK="${VLLM_TURBOQUANT_CONTINUATION_BOUNDS_CHECK:-1}"

ARGS=(
  /home/kevin/Desktop/vLLM-2080Ti-Definitive/.venv/bin/python
  -m
  vllm.entrypoints.openai.api_server
  --host
  0.0.0.0
  --port
  8001
  --model
  /home/kevin/Desktop/models/Qwen3.8-27B-GPTQ-Int4
  --served-model-name
  qwen-local
  qwen3.6:27b
  qwen3.8-27b-gptq-int4
  "qwen38-int4-tqk8v4-mtp${MTP_K}-requal"
  --dtype
  half
  --tensor-parallel-size
  2
  --generation-config
  /home/kevin/.local/share/vllm-qwen27b/gencfg
  --gpu-memory-utilization
  0.82
  --quantization
  gptq_marlin
  --compilation-config
  '{"cudagraph_mode":"PIECEWISE"}'
  --max-model-len
  256000
  --enable-chunked-prefill
  --max-num-seqs
  8
  --max-num-batched-tokens
  2560
  --kv-cache-dtype
  turboquant_k8v4
  --mamba-cache-mode
  align
  --enable-prefix-caching
  --enable-prompt-tokens-details
  --language-model-only
  --skip-mm-profiling
  --speculative-config
  "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_K}}"
  --structured-outputs-config
  '{"backend":"guidance"}'
  --chat-template
  /home/kevin/.local/share/vllm-qwen27b/chat_template-froggeric-v22-official.jinja
  --reasoning-parser
  qwen3
  --tool-call-parser
  qwen3_xml
  --enable-auto-tool-choice
  --additional-config
  '{"gdn_prefill_backend":"flashqla_legacy"}'
)

exec "${ARGS[@]}"
