#!/usr/bin/env python3
"""Typed, validated, observable config for the gateway (keepalive-shim.py) -- lane CFG, 2026-10-03.

WHY THIS EXISTS
The gateway reads ~200 SHIM_* environment variables, loaded by systemd from shim.env
(EnvironmentFile=, last assignment wins). Before this module nothing knew the whole set, so:
  * shim.env carried DUPLICATE keys (SHIM_BG_XCLIENTS, SHIM_FORCE_REMOTE) -- the earlier line was dead
    and every reader of the file had to know systemd's last-wins rule;
  * a value the reader could not parse (SHIM_TOKEN_BUDGET=auto under the old int() reader) crashed the
    gateway at import -- one typo in shim.env takes the estate's front door down;
  * _persist_config writes str(True)/str(False), and several import-time readers test
    `v not in ("0", "false", "")` WITHOUT lowercasing, so a knob switched OFF on the dashboard
    ("False" on disk) came back ON after the next restart. Same writer/reader asymmetry as the
    _CFG_SEP bug documented at the top of keepalive-shim.py, for booleans this time;
  * the dashboard and the vault showed numbers nobody could check against the running process
    (500k shown while 922k was live; an audit found 26 of 63 dashboard items wrong or stale).

WHAT IT DOES
  SCHEMA        every key the gateway reads: type, default (exactly as the code passes it), unit, range or
                choices, kill-switch / money / secret flags, owning lane, one-line description.
                test_gateway_config_schema.py AST-scans keepalive-shim.py and FAILS when the code reads a
                key that is not here, or when a default here differs from the code's default.
  sanitize_environ()
                called by the gateway BEFORE its import-time reads. Fail safe, never raises:
                  - a numeric value the reader would crash on -> removed, so the code's own default
                    applies, with a loud warning (an integral float such as "6.0" for an int key is
                    healed to "6" instead);
                  - a boolean in any spelling -> canonical "1"/"0", correct for every reader variant;
                  - out-of-range / unknown enum / malformed url -> warning only (behaviour unchanged).
                It also lints the env file: duplicates (and which line wins), unknown keys, invalid values.
  effective()   the value table behind GET /gateway/config/effective: live value, its source
                (env / default / env-invalid->default / live-edit), the default, whether it differs, what
                shim.env holds now. Secrets are masked; secret values are never returned or logged.
  CLI           python3 gateway_config_schema.py lint [shim.env]     duplicates / unknown / invalid (exit 1 on errors)
                python3 gateway_config_schema.py scan [keepalive-shim.py]  keys the code reads but the schema lacks
                python3 gateway_config_schema.py doc                  the schema as a markdown table

Adding a knob: add one _k(...) line below (type + the same default string the code uses). The test tells you
if you forgot.
"""
from __future__ import annotations

import ast
import json
import logging
import math
import os
import re
import sys
import time

SCHEMA_VERSION = 1
log = logging.getLogger("gateway-shim.config")

TRUE_TOKENS = frozenset({"1", "true", "on", "yes", "y", "t"})
FALSE_TOKENS = frozenset({"0", "false", "off", "no", "n", "f"})
_SECRET_RE = re.compile(r"(_KEY|_SECRET|_PASSWORD|_PASSWD|_ADMIN_TOKEN|_API_TOKEN|_BEARER)$|^SHIM_REMOTE_KEY$")


class Key:
    __slots__ = ("name", "type", "default", "unit", "lo", "hi", "choices", "sep", "desc", "owner",
                 "tunable", "kill", "money", "secret", "gname", "computed_default", "restart")

    def __init__(self, name, type, default, desc="", unit="", lo=None, hi=None, choices=None, sep=",",
                 owner="", tunable=False, kill=False, money=False, secret=False, gname=None,
                 computed_default=False, restart=False):
        self.name, self.type, self.default, self.desc = name, type, default, desc
        self.unit, self.lo, self.hi, self.choices, self.sep = unit, lo, hi, choices, sep
        self.owner, self.tunable, self.kill, self.money = owner, tunable, kill, money
        self.secret = secret or bool(_SECRET_RE.search(name))
        self.gname, self.computed_default, self.restart = gname, computed_default, restart

    def as_dict(self):
        return {s: getattr(self, s) for s in self.__slots__}


SCHEMA: dict[str, Key] = {}


def _k(name, type, default, desc, **kw):
    if name in SCHEMA:
        raise ValueError("duplicate schema entry " + name)
    if kw.get("lo") is None and type in ("int", "float") and default not in (None, "") and not kw.get("computed_default"):
        try:
            if float(default) >= 0:
                kw["lo"] = 0          # every non-negative knob: a negative value is a typo (warn only)
        except ValueError:
            pass
    SCHEMA[name] = Key(name, type, default, desc, **kw)


# Flags: tunable = live-editable from the dashboard (in the shim's _CFG table); kill = kill switch / rollback
# lever; money = touches spend (Only Money Pauses: automation never changes these); secret = value never shown.
# ---- process / paths (restart-only, deployment facts) ----
_k("SHIM_PORT", "int", "8000", "Gateway listen port.", unit="port", lo=1, hi=65535, gname="PORT", restart=True)
_k("SHIM_UPSTREAM", "url", "http://127.0.0.1:8001", "Local vLLM engine base URL.", gname="LOCAL", restart=True)
_k("SHIM_ENV_FILE", "path", "/home/kevin/.local/share/vllm-qwen27b/shim.env", "File live dashboard edits are persisted to.", gname="SHIM_ENV_FILE", restart=True)
_k("SHIM_ALIASES_FILE", "path", "/home/kevin/.local/share/vllm-qwen27b/gateway-aliases.json", "Named remote-endpoint alias registry (mode 600).", gname="ALIASES_FILE", restart=True)
_k("SHIM_ADMIN_TOKEN", "str", None, "Admin token for POST mutations (env override of the token file).", secret=True, restart=True)
_k("SHIM_ADMIN_TOKEN_FILE", "path", "/home/kevin/.local/share/vllm-qwen27b/admin.token", "File holding the admin token for POST mutations.", gname="SHIM_ADMIN_TOKEN_FILE", restart=True)
_k("SHIM_STATS_FILE", "path", "/home/kevin/.local/share/vllm-qwen27b/gateway-stats.json", "Persisted counters file.", gname="STATS_FILE", restart=True)
_k("SHIM_TELEMETRY_DIR", "path", "/home/kevin/.local/share/vllm-qwen27b/telemetry", "Request/hardware telemetry JSONL directory.", gname="TELEMETRY_DIR", restart=True)
_k("SHIM_FLIGHTREC_DIR", "path", "/home/kevin/.local/share/vllm-qwen27b/flightrec", "Flight-recorder dump directory (big remote responses).", restart=True)
_k("SHIM_DASHBOARD_FILE", "path", None, "Dashboard HTML path (default: gateway_dashboard.html beside the shim).", gname="DASHBOARD_FILE", restart=True, computed_default=True)
_k("SHIM_MODELS_DIR", "path", "/home/kevin/Desktop/models", "Model directory listed by the model-switch page.", gname="MODELS_DIR", restart=True)
_k("SHIM_SERVE_SCRIPT", "path", "/home/kevin/.local/share/vllm-qwen27b/serve-tqk8v4-fg.sh", "Serve script the model-switch page reports/edits.", gname="SERVE_SH", restart=True)
_k("SHIM_SWITCH_SCRIPT", "path", "/home/kevin/.local/share/vllm-qwen27b/switch-model.sh", "Model switch helper script.", gname="SWITCH_SH", restart=True)
_k("SHIM_PRICING_FILE", "path", None, "Remote pricing table (default remote-pricing.json beside the shim).", gname="PRICING_FILE", money=True, restart=True, computed_default=True)
_k("SHIM_PRICING_STALE_DAYS", "int", "30", "Pricing table age that raises a stale-pricing warning.", unit="days", gname="PRICING_STALE_DAYS", money=True)
_k("SHIM_REMOTE_DEAD_FILE", "path", "~/.local/share/vllm-qwen27b/remote-dead-until.json", "Persisted 'remote provider refused us until' marker.", gname="_REMOTE_DEAD_FILE", restart=True)
_k("SHIM_FLOW_EVENTS_FILE", "path", None, "Capacity-mode change log (default incidents/capacity-mode.jsonl).", gname="_FLOW_EVENTS_FILE", owner="CF", restart=True, computed_default=True)
_k("SHIM_OUTCOME_ALARM_PATH", "path", "", "Estate outcome-alarm JSON; a stalled outcome caps auto-overflow spend.", gname="OUTCOME_ALARM_PATH", money=True)
_k("SHIM_SPEND_FILE", "path", "/home/kevin/.local/share/vllm-qwen27b/gateway-spend.json", "Spend authority ledger.", gname="SPEND_FILE", money=True, owner="R2", restart=True)
_k("SHIM_SPEND_CLIENTS_FILE", "path", "/home/kevin/.local/share/vllm-qwen27b/spend-clients.json", "Per-client spend attribution file.", gname="SPEND_CLIENTS_FILE", money=True, owner="R2", restart=True)
_k("SHIM_SPEND_RECONCILIATION_FILE", "path", "/home/kevin/.local/share/vllm-qwen27b/spend-reconciliation.json", "Provider-bill reconciliation fact file.", gname="RECONCILIATION_FACT_FILE", money=True, owner="R2", restart=True)
_k("GATEWAY_DRAIN_LEDGER", "path", "~/.local/share/vllm-qwen27b/incidents/drains.jsonl", "Append-only ledger of publish drains.", gname="_DRAIN_LEDGER", restart=True)
_k("GATEWAY_TIMEOUT_LEDGER", "path", "~/.local/share/vllm-qwen27b/incidents/timeouts.jsonl", "Append-only ledger of which layer timed out.", gname="_TIMEOUT_LEDGER", restart=True)
# ---- remote provider (money) ----
_k("SHIM_REMOTE_BASE", "url", "", "Remote OpenAI-compatible overflow base URL (empty = no remote).", gname="REMOTE_BASE", tunable=True, money=True)
_k("SHIM_REMOTE_KEY", "str", "", "Remote provider API key.", gname="REMOTE_KEY", tunable=True, money=True, secret=True)
_k("SHIM_REMOTE_MODEL", "str", "deepseek-v4-flash", "Model name requests are rewritten to on overflow.", gname="REMOTE_MODEL", tunable=True, money=True)
_k("SHIM_REMOTE_PRO_MODEL", "str", "deepseek-v4-pro", "estate-remote-pro model (refused until priced in SHIM_REMOTE_PRICES_JSON).", gname="REMOTE_PRO_MODEL", money=True, owner="R2")
_k("SHIM_REMOTE_PROVIDER", "str", None, "Provider id for pricing (default derived from SHIM_REMOTE_BASE host).", money=True)
_k("SHIM_REMOTE_CONTEXT_LIMIT", "int", "128000", "Remote provider context window admitted against.", unit="tokens", gname="REMOTE_CONTEXT_LIMIT", tunable=True)
_k("SHIM_REMOTE_NO_THINK", "bool", "1", "Force non-thinking on the remote failover.", gname="REMOTE_NO_THINK", kill=True)
_k("SHIM_REMOTE_VISION", "bool", "0", "Declare that the default remote accepts images.", gname="REMOTE_VISION", tunable=True, owner="GW")
_k("SHIM_REMOTE_DEAD_SECS", "float", "300", "After a hard provider refusal, treat remote as unavailable this long.", unit="s", gname="REMOTE_DEAD_SECS")
_k("SHIM_FORCE_REMOTE", "bool", "0", "Send everything remote (lease-bounded; see FORCE_REMOTE_UNTIL_EPOCH).", gname="FORCE_REMOTE", tunable=True, money=True, kill=True)
_k("SHIM_FORCE_REMOTE_UNTIL_EPOCH", "float", "0", "Expiry of the force-remote lease (written by the gateway).", unit="epoch s", gname="FORCE_REMOTE_UNTIL_EPOCH", money=True)
_k("SHIM_LOCAL_ONLY", "bool", "0", "Never overflow to remote.", gname="LOCAL_ONLY", tunable=True, kill=True)
_k("SHIM_PEAK_HOURS_UTC", "hours", "1-4,6-10", "Provider peak-price hours (UTC ranges).", gname="PEAK_HOURS", tunable=True, money=True)
# ---- spend (money; Only Money Pauses) ----
_k("SHIM_SPEND_CAP_USD", "float", "25.0", "Hard daily remote spend cap.", unit="USD/day", gname="SPEND_CAP_USD", tunable=True, money=True, owner="R2")
_k("SHIM_SPEND_ENFORCE", "bool", "1", "Enforce the spend cap (0 = observe only).", gname="SPEND_ENFORCE", tunable=True, money=True, kill=True, owner="R2")
_k("SHIM_SPEND_TZ", "str", "America/Phoenix", "Timezone of the spend day boundary.", gname="SPEND_TZ", money=True, owner="R2")
_k("SHIM_SPEND_RESERVATION_TTL_MAX", "int", None, "Longest a spend reservation may live.", unit="s", gname="SPEND_RESERVATION_TTL_MAX", money=True, owner="R2", computed_default=True)
_k("SHIM_SPEND_REQUEST_HOLD_TTL", "int", "1800", "A crashed request's hold is charged in full after this.", unit="s", gname="SPEND_REQUEST_HOLD_TTL", money=True, owner="R2")
_k("SHIM_SPEND_DEFAULT_OUT_TOKENS", "int", "16384", "Output tokens reserved when the caller gives no max_tokens.", unit="tokens", gname="SPEND_DEFAULT_OUT_TOKENS", money=True, owner="R2")
_k("SHIM_STALLED_AUTO_OVERFLOW_CAP_USD", "float", "2.0", "Auto-overflow spend cap while the estate outcome alarm is stalled.", unit="USD", gname="STALLED_AUTO_OVERFLOW_CAP_USD", money=True)
_k("SHIM_REMOTE_COST_IN_PER_MTOK", "float", "0.28", "Legacy input price used by telemetry cost estimates.", unit="USD/Mtok", gname="REMOTE_COST_IN_PER_MTOK", tunable=True, money=True)
_k("SHIM_REMOTE_COST_OUT_PER_MTOK", "float", "1.10", "Legacy output price used by telemetry cost estimates.", unit="USD/Mtok", gname="REMOTE_COST_OUT_PER_MTOK", tunable=True, money=True)
_k("SHIM_REMOTE_COST_IN_PER_MTOK_PEAK", "float", None, "Legacy peak input price (default = off-peak).", unit="USD/Mtok", gname="REMOTE_COST_IN_PER_MTOK_PEAK", tunable=True, money=True, computed_default=True)
_k("SHIM_REMOTE_COST_OUT_PER_MTOK_PEAK", "float", None, "Legacy peak output price (default = off-peak).", unit="USD/Mtok", gname="REMOTE_COST_OUT_PER_MTOK_PEAK", tunable=True, money=True, computed_default=True)
_k("SHIM_REMOTE_PRICE_CACHE_HIT_PER_MTOK", "float", "0.003", "Spend-authority price: cached input.", unit="USD/Mtok", gname="REMOTE_PRICE_CACHE_HIT_PER_MTOK", tunable=True, money=True, owner="R2")
_k("SHIM_REMOTE_PRICE_CACHE_MISS_PER_MTOK", "float", "0.15", "Spend-authority price: uncached input.", unit="USD/Mtok", gname="REMOTE_PRICE_CACHE_MISS_PER_MTOK", tunable=True, money=True, owner="R2")
_k("SHIM_REMOTE_PRICE_OUTPUT_PER_MTOK", "float", "0.6", "Spend-authority price: output.", unit="USD/Mtok", gname="REMOTE_PRICE_OUTPUT_PER_MTOK", tunable=True, money=True, owner="R2")
_k("SHIM_REMOTE_PRICE_CACHE_HIT_PER_MTOK_PEAK", "float", "0.006", "Spend-authority peak price: cached input.", unit="USD/Mtok", gname="REMOTE_PRICE_CACHE_HIT_PER_MTOK_PEAK", tunable=True, money=True, owner="R2")
_k("SHIM_REMOTE_PRICE_CACHE_MISS_PER_MTOK_PEAK", "float", "0.3", "Spend-authority peak price: uncached input.", unit="USD/Mtok", gname="REMOTE_PRICE_CACHE_MISS_PER_MTOK_PEAK", tunable=True, money=True, owner="R2")
_k("SHIM_REMOTE_PRICE_OUTPUT_PER_MTOK_PEAK", "float", "1.2", "Spend-authority peak price: output.", unit="USD/Mtok", gname="REMOTE_PRICE_OUTPUT_PER_MTOK_PEAK", tunable=True, money=True, owner="R2")
_k("SHIM_REMOTE_PRICES_JSON", "json", "", "Per-model price table JSON (overrides the flat prices).", gname="REMOTE_PRICES_JSON", tunable=True, money=True, owner="R2")
# ---- context / capacity ----
_k("SHIM_LOCAL_CONTEXT_LIMIT", "int", "524288", "Local context window fallback until the engine's max_model_len is read.", unit="tokens", lo=1024, gname="LOCAL_CONTEXT_LIMIT", tunable=True)
_k("SHIM_MAX_LOCAL_TOKENS", "int", "80000", "Single-request prompt+output ceiling for local (fallback; live = engine context).", unit="tokens", gname="MAX_LOCAL_TOKENS", tunable=True)
_k("SHIM_CONTEXT_SAFETY_MARGIN", "int", "1024", "Tokens held back from the context window at admission.", unit="tokens", gname="CONTEXT_SAFETY_MARGIN", tunable=True)
_k("SHIM_CONTEXT_COMPACTION", "bool", "1", "Allow explicit context compaction for over-window transcripts.", gname="CONTEXT_COMPACTION_ENABLED", tunable=True, kill=True)
_k("SHIM_CONTEXT_COMPACTION_KEEP", "int", "12", "Most recent messages compaction always keeps.", unit="messages", gname="CONTEXT_COMPACTION_KEEP", tunable=True)
_k("SHIM_CAPACITY_LIVE", "bool", "1", "Read pool/budget/prefill capacity from the live engine (0 = configured numbers only).", gname="CAPACITY_LIVE", tunable=True, kill=True, owner="GW")
_k("SHIM_CONTEXT_LIVE", "bool", "1", "Admit against the engine's live max_model_len (0 = configured only).", gname="CONTEXT_LIVE", tunable=True, kill=True, owner="GW")
_k("SHIM_MODELS_POLL_SECS", "float", "30", "How often the engine's /v1/models is polled for context facts.", unit="s", gname="MODELS_POLL_S", tunable=True, owner="GW")
_k("SHIM_POOL_TOKENS", "int", "754068", "KV pool size fallback (live value is scraped from the engine when CAPACITY_LIVE).", unit="tokens", gname="POOL_TOKENS", tunable=True, owner="GW")
_k("SHIM_TOKEN_BUDGET", "budget", None, "In-flight reserved-token cap: auto (derive from live pool) | 0 (off) | explicit int override.", unit="tokens", gname="TOKEN_BUDGET", tunable=True, owner="GW")
_k("SHIM_TOKEN_BUDGET_FRAC", "float", None, "Token budget as a fraction of the live KV pool (default 500000/754068 = 0.663).", lo=0, hi=1, gname="TOKEN_BUDGET_FRAC", tunable=True, owner="GW", computed_default=True)
_k("SHIM_TOKEN_BUDGET_CEIL", "int", "680000", "Absolute token-budget ceiling (largest in-flight context benched). 0 = none.", unit="tokens", gname="TOKEN_BUDGET_CEIL", tunable=True, owner="GW")
_k("SHIM_LOCAL_MODALITIES", "list", "text", "Modalities the local engine accepts (text[,image]).", gname="LOCAL_MODALITIES", tunable=True, owner="GW")
_k("SHIM_PREFILL_TPS", "float", "500", "Configured prefill rate fallback (live measure preferred).", unit="tok/s", gname="PREFILL_TPS", tunable=True)
_k("SHIM_PREFILL_MEASURE_WINDOW_S", "float", "1800", "Window of the measured prefill rate.", unit="s", gname="PREFILL_MEASURE_WINDOW_S", owner="GW")
_k("SHIM_PREFILL_MEASURE_BUCKET_S", "float", "60", "Bucket width of the measured prefill rate.", unit="s", gname="PREFILL_MEASURE_BUCKET_S", owner="GW")
_k("SHIM_PREFILL_MEASURE_MIN_BUCKETS", "int", "5", "Buckets with prefill required before the measured rate is trusted.", gname="PREFILL_MEASURE_MIN_BUCKETS", owner="GW")
_k("SHIM_PREFILL_MEASURE_QUANTILE", "float", "0.75", "Quantile of bucket rates used as the measured prefill rate.", lo=0, hi=1, gname="PREFILL_MEASURE_QUANTILE", owner="GW")
_k("SHIM_PREFILL_MEASURE_SETTLE_S", "float", "180", "Ignore prefill samples this long after an engine (re)start.", unit="s", gname="PREFILL_MEASURE_SETTLE_S", owner="GW")
_k("SHIM_PREFILL_MEASURE_OFFLINE_PAD_S", "float", "180", "Ignore prefill samples this long around a planned-offline window.", unit="s", gname="PREFILL_MEASURE_OFFLINE_PAD_S", owner="GW")
# ---- lanes / admission ----
_k("SHIM_LOCAL_BUDGET", "int", "2", "Local lane budget in units (production 14).", unit="units", lo=1, hi=64, gname="BUDGET", tunable=True)
_k("SHIM_LOCAL_WAIT_SECS", "float", "8", "How long a request waits for a local lane before overflow.", unit="s", gname="LOCAL_WAIT", tunable=True)
_k("SHIM_SLOT_POLL_SECS", "float", "0.05", "Lane wait poll interval.", unit="s", gname="SLOT_POLL")
_k("SHIM_OOM_BACKOFF_SECS", "int", "120", "After a local crash/OOM, budget=1 for this long.", unit="s", gname="OOM_BACKOFF", tunable=True)
_k("SHIM_BIG_TOKENS", "int", "40000", "Requests at/above this cost ceil(tokens/TOKENS_PER_UNIT) units.", unit="tokens", gname="BIG_TOKENS", tunable=True)
_k("SHIM_TOKENS_PER_UNIT", "int", "12000", "Tokens per lane unit for big requests.", unit="tokens", lo=1, gname="TOKENS_PER_UNIT", tunable=True)
_k("SHIM_BIG_OUTPUT", "int", "32000", "max_tokens at/above this is a big-output request (routing guard). 0 = off.", unit="tokens", gname="BIG_OUTPUT", tunable=True)
_k("SHIM_BIG_PROMPT", "int", "24000", "Prompt at/above this is a big-prompt request (routing guard). 0 = off.", unit="tokens", gname="BIG_PROMPT", tunable=True)
_k("SHIM_MONSTER_INFLIGHT", "int", "120000", "In-flight tokens that mark a monster prefill (new arrivals routed away).", unit="tokens", gname="MONSTER_INFLIGHT", tunable=True)
_k("SHIM_LOCAL_MAX_OUT", "int", "8192", "Clamp on local max_tokens.", unit="tokens", gname="LOCAL_MAX_OUT", tunable=True)
_k("SHIM_DEFAULT_MAX_OUT", "int", "8192", "max_tokens assumed when the caller sends none.", unit="tokens", gname="DEFAULT_MAX_OUT")
_k("SHIM_FIRST_TOKEN_BASE", "float", "15", "First-token deadline base.", unit="s", gname="FIRST_TOKEN_BASE")
_k("SHIM_FIRST_TOKEN_MAX", "float", "45", "First-token deadline ceiling.", unit="s", gname="FIRST_TOKEN_MAX", tunable=True)
_k("SHIM_FT_CONCURRENCY_SCALE", "int", "1", "Scale the first-token deadline by in-flight concurrency (0 = flat).", lo=0, hi=1, gname="FT_CONCURRENCY_SCALE", kill=True)
_k("SHIM_FG_RESERVED", "int", "2", "Lanes kept free for interactive traffic.", unit="units", gname="FG_RESERVED", tunable=True)
_k("SHIM_TINY_TOKENS", "int", "1500", "prompt+max_tokens at/below this is a tiny call (extra lanes).", unit="tokens", gname="TINY_TOKENS", tunable=True)
_k("SHIM_TINY_EXTRA_LANES", "int", "2", "Extra lanes for tiny calls above the budget.", unit="lanes", gname="TINY_EXTRA_LANES", tunable=True)
_k("SHIM_INTERACTIVE_NEVER_OVERFLOW", "int", "1", "Interactive traffic waits for local instead of overflowing (0 = old behaviour).", lo=0, hi=1, gname="INTERACTIVE_NEVER_OVERFLOW", tunable=True, kill=True)
_k("SHIM_FOREIGN_LOAD_GUARD", "int", "1", "Route away when the engine runs more than the gateway admitted (0 = off).", lo=0, hi=1, gname="FOREIGN_LOAD_GUARD", kill=True)
_k("SHIM_EXACT_TOKENS", "bool", "1", "Count prompt tokens with the engine tokenizer (0 = char estimate).", gname="EXACT_TOKENS", kill=True)
_k("SHIM_MIN_CHARS_PER_TOK", "float", "2.0", "Char/token ratio floor of the estimate.", gname="MIN_CHARS_PER_TOK")
_k("SHIM_TOKENIZE_TIMEOUT", "float", "5", "Engine /tokenize timeout.", unit="s", gname="TOKENIZE_TIMEOUT")
_k("SHIM_CRASH_ADAPTIVE", "bool", "0", "Lower BIG_PROMPT after a local crash, restore after clean completions.", gname="CRASH_ADAPTIVE", kill=True)
_k("SHIM_BIG_PROMPT_RESTORE_N", "int", "5", "Clean big-prompt completions that restore BIG_PROMPT (crash-adaptive).", gname="BIG_PROMPT_RESTORE_N")
_k("SHIM_BIG_PROMPT_QUALIFY_FRAC", "float", "0.8", "Fraction of the floor a prompt must reach to count toward restore.", lo=0, hi=1, gname="BIG_PROMPT_QUALIFY_FRAC")
# ---- prefill-aware admission ----
_k("SHIM_MONSTER_PREFILL_SECS", "float", "30", "Uncached prefill in flight >= this marks a monster.", unit="s", gname="MONSTER_PREFILL_SECS", tunable=True)
_k("SHIM_HEAVY_PREFILL_SECS", "float", "20", "A request whose own uncached prefill is >= this is heavy.", unit="s", gname="HEAVY_PREFILL_SECS", tunable=True)
_k("SHIM_HEAVY_ADMIT_BACKLOG_SECS", "float", "15", "A heavy request runs locally only if the prefill backlog is <= this.", unit="s", gname="HEAVY_ADMIT_BACKLOG_SECS", tunable=True)
_k("SHIM_PREFILL_ADMIT_SECS", "float", "45", "Admit only when total prefill backlog stays under this. 0 = off.", unit="s", gname="PREFILL_ADMIT_SECS", tunable=True)
_k("SHIM_LIGHT_PREFILL_SECS", "float", "5", "Prefill this short always fits.", unit="s", gname="LIGHT_PREFILL_SECS", tunable=True)
# ---- prefix-cache cost model ----
_k("SHIM_PREFIX_HIT_MARGIN_TOKENS", "int", "512", "Safety margin on the predicted prefix-cache credit.", unit="tokens", gname="PREFIX_HIT_MARGIN_TOKENS", tunable=True)
_k("SHIM_USE_COMPUTED_COST", "int", "0", "Charge lanes by predicted uncached tokens (1) instead of raw prompt (0).", lo=0, hi=1, gname="USE_COMPUTED_COST", tunable=True)
_k("SHIM_PREFIX_ALIGN_TOKENS", "int", "3568", "Engine attention block size the credit is rounded down to.", unit="tokens", gname="PREFIX_ALIGN_TOKENS", tunable=True)
_k("SHIM_PREFIX_MODEL_TTL_SECS", "float", "900", "Prefix model entry lifetime.", unit="s", gname="PREFIX_MODEL_TTL_SECS", tunable=True)
_k("SHIM_PREFIX_MODEL_MAX_NODES", "int", "60000", "Prefix model size bound.", unit="nodes", gname="PREFIX_MODEL_MAX_NODES")
_k("SHIM_PREFIX_CREDIT_UNIT", "int", "0", "Round the credit to N tokens instead of whole blocks (0 = blocks).", unit="tokens", gname="PREFIX_CREDIT_UNIT")
_k("SHIM_CHAIN_TELEMETRY", "bool", "0", "Record prefix-chain telemetry.", gname="CHAIN_TELEMETRY", tunable=True, owner="GW2")
_k("SHIM_WARM_PRIORITY", "bool", "0", "Give warm-prefix requests engine priority.", gname="WARM_PRIORITY", kill=True, tunable=True, owner="GW2")
_k("SHIM_WARM_PRIORITY_MIN_CREDIT", "int", "4096", "Minimum predicted credit for warm priority.", unit="tokens", gname="WARM_PRIORITY_MIN_CREDIT", tunable=True, owner="GW2")
_k("SHIM_WARM_PRIORITY_MAX_COMPUTED", "int", "1700", "Maximum uncached tokens for warm priority.", unit="tokens", gname="WARM_PRIORITY_MAX_COMPUTED", tunable=True, owner="GW2")
_k("SHIM_WARM_PRIORITY_VALUE", "int", "-10", "Engine priority value given to warm requests (lower = sooner).", lo=-1000, hi=1000, gname="WARM_PRIORITY_VALUE")
# ---- background traffic ----
_k("SHIM_BG_MARKERS", "list", "scheduled cron job", "Prompt phrases that mark background traffic ('|'-separated).", sep="|", gname="BG_MARKERS", tunable=True)
_k("SHIM_BG_XCLIENTS", "list", "cron,batch,workflow-bg,research-feeder,digester", "X-Client names treated as background.", gname="BG_XCLIENTS", tunable=True)
_k("SHIM_BG_WAIT_SECS", "float", "5", "Background wait before overflow.", unit="s", gname="BG_WAIT", tunable=True)
_k("SHIM_BG_LOCAL_ONLY", "bool", "1", "Background never overflows to paid remote.", gname="BG_LOCAL_ONLY", tunable=True, money=True)
_k("SHIM_BG_WAIT_LOCAL_SECS", "float", "120", "Background local-only wait before a retry-later refusal.", unit="s", gname="BG_WAIT_LOCAL", tunable=True)
_k("SHIM_BG_REJECT_RETRY_SECS", "int", "300", "Retry-After given to refused background requests.", unit="s", gname="BG_REJECT_RETRY_SECS", tunable=True)
_k("SHIM_BG_BIG_LOCAL_WHEN_IDLE", "int", "1", "A big background request may take the whole budget when idle (0 = always remote).", lo=0, hi=1, gname="BG_BIG_LOCAL_WHEN_IDLE", tunable=True)
_k("SHIM_BG_NO_THINK", "bool", "1", "Inject enable_thinking=false for local background requests.", gname="BG_NO_THINK", tunable=True)
_k("SHIM_NO_THINK_IPS", "list", "", "Client IPs whose requests get enable_thinking=false.", gname="NO_THINK_IPS", tunable=True)
# ---- thinking / output guards ----
_k("SHIM_THINK_GUARD", "bool", "1", "Bound thinking budget by max_tokens.", gname="THINK_GUARD", tunable=True, kill=True)
_k("SHIM_THINK_BUDGET_FRAC", "float", "0.5", "Thinking budget as a fraction of max_tokens.", lo=0, hi=1, gname="THINK_BUDGET_FRAC", tunable=True)
_k("SHIM_THINK_BUDGET_MIN", "int", "128", "Thinking budget floor.", unit="tokens", gname="THINK_BUDGET_MIN", tunable=True)
_k("SHIM_THINK_BUDGET_MAX", "int", "4096", "Thinking budget ceiling.", unit="tokens", gname="THINK_BUDGET_MAX", tunable=True)
_k("SHIM_THINK_OFF_UNDER", "int", "600", "max_tokens under this disables thinking.", unit="tokens", gname="THINK_OFF_UNDER", tunable=True)
_k("SHIM_THINK_LOW_UNDER", "int", "1400", "max_tokens under this uses a low thinking budget.", unit="tokens", gname="THINK_LOW_UNDER", tunable=True)
_k("SHIM_CREDIT_ANCHOR", "enum", "off", "Engine-anchored prefix-cache credit (shadow = log pm_credit_anchored only).", choices=("off", "shadow"), gname="CREDIT_ANCHOR", tunable=True, kill=True, owner="GW2")
_k("SHIM_THINK_BUDGET_EXPLICIT", "int", "0", "Bound thinking even when the caller set it explicitly (1 = on).", lo=0, hi=1, gname="THINK_BUDGET_EXPLICIT", tunable=True, owner="GW2")
_k("SHIM_REASONING_WATCHDOG", "enum", "off", "Runaway-reasoning watchdog on local streams.", choices=("off", "shadow", "retry"), gname="REASONING_WATCHDOG", tunable=True, kill=True, owner="GW2")
_k("SHIM_REASONING_BUDGET_TOKENS", "int", "8192", "Streamed reasoning tokens that trip the watchdog.", unit="tokens", gname="REASONING_BUDGET_TOKENS", tunable=True, owner="GW2")
_k("SHIM_REASONING_CHARS_PER_TOKEN", "float", "4.0", "Chars/token estimate for streamed reasoning.", gname="REASONING_CHARS_PER_TOKEN", tunable=True, owner="GW2")
_k("SHIM_EMPTY_RETRY", "bool", "1", "Retry an empty completion once.", gname="EMPTY_RETRY", tunable=True, kill=True)
_k("SHIM_REP_GUARD", "bool", "1", "Abort degenerate repetition loops.", gname="REP_GUARD", tunable=True, kill=True)
_k("SHIM_REP_MIN_PATTERN", "int", "8", "Shortest repeated pattern detected.", unit="chars", gname="REP_MIN_PATTERN", tunable=True)
_k("SHIM_REP_MAX_PATTERN", "int", "64", "Longest repeated pattern detected.", unit="chars", gname="REP_MAX_PATTERN", tunable=True)
_k("SHIM_REP_MIN_COUNT", "int", "6", "Repeats that count as a loop.", gname="REP_MIN_COUNT", tunable=True)
_k("SHIM_NONTHINK_PROFILE", "bool", "1", "Apply the non-thinking sampling profile (EXP-026).", gname="NONTHINK_PROFILE", tunable=True, kill=True)
_k("SHIM_NONTHINK_PP", "float", "1.5", "Non-thinking presence penalty.", gname="NONTHINK_PP", tunable=True)
_k("SHIM_NONTHINK_TOP_P", "float", "0.80", "Non-thinking top_p.", lo=0, hi=1, gname="NONTHINK_TOP_P", tunable=True)
_k("SHIM_NONTHINK_TEMP", "float", "0.7", "Non-thinking temperature.", lo=0, hi=2, gname="NONTHINK_TEMP", tunable=True)
_k("SHIM_NONTHINK_TOP_K", "int", "20", "Non-thinking top_k.", gname="NONTHINK_TOP_K", tunable=True)
_k("SHIM_STREAM_IDLE_TIMEOUT_SECS", "float", "45", "Abort a stream silent this long.", unit="s", gname="STREAM_IDLE_TIMEOUT_SECS", tunable=True)
# ---- congestion / perf breaker ----
_k("SHIM_PREDICTED_OCCUPANCY_SECS", "float", "180", "Predicted lane occupancy that counts as congestion.", unit="s", gname="PREDICTED_OCCUPANCY_SECS", tunable=True)
_k("SHIM_DECODE_TPS_FLOOR", "float", "20", "Decode rate floor used by the occupancy prediction.", unit="tok/s", gname="DECODE_TPS_FLOOR", tunable=True)
_k("SHIM_PERF_BREAKER_ENABLED", "bool", "1", "Performance circuit breaker.", gname="PERF_BREAKER_ENABLED", tunable=True, kill=True)
_k("SHIM_PERF_BREAKER_TTFT_P95_SECS", "float", "20", "TTFT p95 that trips the breaker.", unit="s", gname="PERF_BREAKER_TTFT_P95_SECS", tunable=True)
_k("SHIM_PERF_BREAKER_GPU_UTIL_PCT", "float", "95", "GPU utilisation that trips the breaker.", unit="%", hi=100, gname="PERF_BREAKER_GPU_UTIL_PCT", tunable=True)
_k("SHIM_PERF_BREAKER_KV_PCT", "float", "85", "KV usage that trips the breaker.", unit="%", hi=100, gname="PERF_BREAKER_KV_PCT", tunable=True)
_k("SHIM_PERF_BREAKER_MIN_SAMPLES", "int", "3", "Samples required before the breaker may trip.", gname="PERF_BREAKER_MIN_SAMPLES", tunable=True)
_k("SHIM_PERF_BREAKER_HOLD_SECS", "float", "45", "Breaker hold time.", unit="s", gname="PERF_BREAKER_HOLD_SECS", tunable=True)
# ---- local-first (L1 / LF) ----
_k("SHIM_LOCAL_FIRST", "bool", "1", "Local-first: overflow reasons in LOCAL_FIRST_REASONS wait for local instead.", gname="LOCAL_FIRST", tunable=True, kill=True, owner="L1")
_k("SHIM_LOCAL_FIRST_REASONS", "list", "big-prompt,perf,predicted,big-out,monster", "Overflow reasons local-first converts into a local wait.", gname="LOCAL_FIRST_REASONS", tunable=True, owner="L1")
_k("SHIM_LOCAL_FIRST_QUEUE_WAIT_SECS", "float", "5", "Local-first queue wait (fallback when derive is off).", unit="s", gname="LOCAL_FIRST_QUEUE_WAIT_SECS", tunable=True, owner="L1")
_k("SHIM_LOCAL_FIRST_WAIT_WINDOW_SECS", "float", "60", "Local-first wait window.", unit="s", gname="LOCAL_FIRST_WAIT_WINDOW_SECS", tunable=True, owner="L1")
_k("SHIM_LOCAL_FIRST_FIRST_TOKEN_MAX", "float", "300", "First-token wait the local relay grants under local-first.", unit="s", gname="LOCAL_FIRST_FIRST_TOKEN_MAX", tunable=True, owner="L1")
_k("SHIM_LOCAL_FIRST_INTERACTIVE_TTFT_SECS", "float", "0", "Interactive TTFT target under local-first (0 = none).", unit="s", gname="LOCAL_FIRST_INTERACTIVE_TTFT_SECS", tunable=True, owner="L1")
_k("SHIM_LOCAL_FIRST_DERIVE", "bool", "1", "Derive local-first waits from the measured prefill rate (rollback: 0).", gname="LOCAL_FIRST_DERIVE", tunable=True, kill=True, owner="LF")
_k("SHIM_EXPECTED_OUTPUT", "bool", "1", "Big-out rule uses the learned expected output (rollback: 0).", gname="EXPECTED_OUTPUT", tunable=True, kill=True, owner="LF")
_k("SHIM_EXPECTED_OUTPUT_QUANTILE", "float", "0.95", "Quantile of a signature's output history used as its expectation.", lo=0, hi=1, gname="EXPECTED_OUTPUT_QUANTILE", tunable=True, owner="LF")
_k("SHIM_EXPECTED_OUTPUT_MIN_SAMPLES", "int", "20", "History required before the expectation is used.", gname="EXPECTED_OUTPUT_MIN_SAMPLES", tunable=True, owner="LF")
_k("SHIM_EXPECTED_OUTPUT_HISTORY", "int", "200", "Output history kept per signature.", gname="EXPECTED_OUTPUT_HISTORY", owner="LF")
# ---- micro lane / probes ----
_k("SHIM_MICRO_LEARN", "bool", "1", "Learn short-output signatures and admit them to the tiny lane.", gname="MICRO_LEARN", kill=True)
_k("SHIM_MICRO_SIG_CHARS", "int", "32", "Prompt-head chars in a micro signature.", unit="chars", gname="MICRO_SIG_CHARS")
_k("SHIM_MICRO_MIN_SAMPLES", "int", "6", "Samples before a signature is trusted as micro.", gname="MICRO_MIN_SAMPLES")
_k("SHIM_MICRO_HISTORY", "int", "16", "Outputs remembered per signature.", gname="MICRO_HISTORY")
_k("SHIM_MICRO_MAX_OUT", "int", "96", "Output length that counts as micro.", unit="tokens", gname="MICRO_MAX_OUT")
_k("SHIM_MICRO_MAX_PROMPT", "int", "6000", "Largest prompt a micro admission may carry.", unit="tokens", gname="MICRO_MAX_PROMPT")
_k("SHIM_PROBE_SYNTH", "bool", "1", "Answer liveness probes from fresh engine evidence (kill: 0).", gname="PROBE_SYNTH", kill=True)
_k("SHIM_PROBE_FRESH_S", "float", "120", "Engine evidence fresher than this lets a probe be synthesised.", unit="s", gname="PROBE_FRESH_S")
_k("SHIM_PROBE_REAL_EVERY", "int", "12", "Every Nth probe still goes through for real.", lo=1, gname="PROBE_REAL_EVERY")
_k("SHIM_LOCAL_MODEL_NAME", "str", "qwen-local", "Model name used for gateway-originated local calls.")
# ---- flow (CF) ----
_k("SHIM_FLOW_MODE", "enum", "enforce", "Capacity-aware flow scheduler mode (rollback: off).", choices=("off", "shadow", "enforce"), gname="FLOW_MODE", tunable=True, kill=True, owner="CF")
_k("SHIM_FLOW_SHARES", "map", "kevin=1000,halo=60,runner=25,background=10", "Work-class weighted shares.", gname="FLOW_SHARES", tunable=True, owner="CF")
_k("SHIM_FLOW_CEIL", "map", "kevin=0,halo=1.5,runner=1,background=0.5", "Per-class engine-queue ceiling as a fraction of FLOW_BACKLOG_S (0 = none).", gname="FLOW_CEIL", tunable=True, owner="CF")
_k("SHIM_FLOW_BACKLOG_S", "float", "40", "Engine prefill backlog the ceilings are fractions of.", unit="s", gname="FLOW_BACKLOG_S", tunable=True, owner="CF")
_k("SHIM_FLOW_DEADLINES", "map", "kevin=0,halo=0,runner=900,background=600", "Default must-start-within per class (0 = never refused).", unit="s", gname="FLOW_DEADLINES", tunable=True, owner="CF")
_k("SHIM_FLOW_CLASS_MAP", "map", None, "Client-name pattern -> work class.", gname="FLOW_CLASS_MAP", tunable=True, owner="CF", computed_default=True)
_k("SHIM_FLOW_MARKERS", "map", "local-lane-runner batch job=runner", "Prompt phrase -> work class.", gname="FLOW_MARKERS", tunable=True, owner="CF")
_k("SHIM_FLOW_MODEL_MAP", "map", "", "Request model -> work class.", gname="FLOW_MODEL_MAP", tunable=True, owner="CF")
_k("SHIM_FLOW_DEMAND_WINDOW_S", "float", "300", "Window of the demand_5m facts.", unit="s", gname="FLOW_DEMAND_WINDOW_S", owner="CF")
_k("SHIM_FLOW_AFFINITY_MAX", "int", "6", "Same-prefix grants in a row before FIFO resumes.", gname="FLOW_AFFINITY_MAX", tunable=True, owner="CF")
_k("SHIM_FLOW_STARVE_S", "float", "300", "A class unserved this long ignores its ceiling once.", unit="s", gname="FLOW_STARVE_S", tunable=True, owner="CF")
_k("SHIM_FLOW_PREFIX_HOLD_MAX_S", "float", "60", "Longest a same-prefix hold may delay others.", unit="s", gname="FLOW_PREFIX_HOLD_MAX_S", tunable=True, owner="CF")
_k("SHIM_FLOW_URGENT_SLACK_S", "float", "30", "Deadline closer than this goes first (EDF).", unit="s", gname="FLOW_URGENT_SLACK_S", tunable=True, owner="CF")
_k("SHIM_FLOW_RESTORE", "bool", "1", "Rebuild the 24 h routing ring from the request log at startup.", gname="FLOW_RESTORE", owner="DB2")
# ---- telemetry / logging / caches ----
_k("SHIM_LOG_REQUESTS", "bool", "1", "Log one line per request (0 for prompt privacy).", gname="LOG_REQUESTS", tunable=True)
_k("SHIM_LOG_PREVIEW_CHARS", "int", "70", "Prompt preview chars in request logs.", unit="chars", gname="LOG_PREVIEW_CHARS")
_k("SHIM_TELEM_SAMPLE_SECS", "float", "2", "Hardware telemetry sample interval.", unit="s", gname="TELEM_SAMPLE_SECS", tunable=True)
_k("SHIM_TELEM_SLOW_EVERY", "int", "30", "Slow-ring sample every N fast samples.", lo=1, gname="TELEM_SLOW_EVERY", tunable=True)
_k("SHIM_TELEM_SCRAPE_TIMEOUT", "float", "3", "Engine /metrics scrape timeout.", unit="s", gname="TELEM_SCRAPE_TIMEOUT")
_k("SHIM_TELEMETRY_JSONL_MAX_MB", "float", "200", "Request telemetry JSONL size bound.", unit="MB", gname="TELEMETRY_JSONL_MAX_MB", tunable=True)
_k("SHIM_TELEMETRY_FLUSH_SECS", "float", "3", "Telemetry flush interval.", unit="s", gname="TELEMETRY_FLUSH_SECS", tunable=True)
_k("SHIM_TELEMETRY_RETENTION_DAYS", "int", "30", "Telemetry retention.", unit="days", gname="TELEMETRY_RETENTION_DAYS", tunable=True)
_k("SHIM_TELEMETRY_QUEUE_MAX", "int", "10000", "Telemetry in-memory queue bound.", unit="rows", gname="TELEMETRY_QUEUE_MAX")
_k("SHIM_HISTORY_SUMMARY_CACHE_TTL", "float", "5", "History summary cache TTL.", unit="s", gname="HISTORY_SUMMARY_CACHE_TTL")
_k("SHIM_HW_HISTORY", "bool", "1", "Persist hardware history (kill: 0).", gname="HW_HISTORY", kill=True, owner="TL")
_k("SHIM_HW_JSONL_MAX_MB", "float", "20", "Hardware history JSONL size bound.", unit="MB", gname="HW_JSONL_MAX_MB", owner="TL")
_k("SHIM_HW_QUEUE_MAX", "int", "2880", "Hardware history queue bound.", unit="rows", gname="HW_QUEUE_MAX", owner="TL")
_k("SHIM_HW_SEED_HOURS", "float", "24", "Hours of hardware history re-seeded at startup.", unit="h", gname="HW_SEED_HOURS", owner="TL")
_k("SHIM_HW_QUERY_CACHE_TTL", "float", "10", "Hardware history query cache TTL.", unit="s", gname="HW_QUERY_CACHE_TTL", owner="TL")
_k("SHIM_WINDOWS_CACHE_TTL", "float", "5", "/gateway/windows cache TTL.", unit="s", gname="WINDOWS_CACHE_TTL")
_k("SHIM_CLAIMS_CACHE_MAX", "int", "500", "Research claims cache bound.", unit="entries", gname="CLAIMS_CACHE_MAX")
_k("SHIM_FLIGHTREC_MIN_TOK", "int", "15000", "Prompt size from which remote responses are flight-recorded.", unit="tokens")

_SETTABLE_BY_GATEWAY = {"SHIM_FORCE_REMOTE_UNTIL_EPOCH"}   # written by _persist_config, not by humans


# ------------------------------------------------------------------ value checks
def canonical_bool(raw):
    """'1' / '0' for any recognised spelling, None when unrecognised (or empty)."""
    s = str(raw).strip().lower()
    if s in TRUE_TOKENS:
        return "1"
    if s in FALSE_TOKENS:
        return "0"
    return None


def check_value(key: Key, raw: str):
    """Return (ok, problem, healed). ok=False means the gateway's own reader would CRASH on this value
    (numeric types only) -> the caller drops it so the code default applies. problem without ok=False is a
    warning only. healed is a replacement string that keeps the intent (e.g. "6.0" -> "6" for an int)."""
    s = str(raw).strip()
    t = key.type
    if t in ("int", "float", "budget"):
        if t == "budget" and s.lower() in ("", "auto", "none", "live"):
            return True, None, None
        if s == "" and t != "budget":
            return False, "empty value for a numeric key", None
        try:
            f = float(s)
        except ValueError:
            return False, "not a number: %r" % s, None
        if not math.isfinite(f):
            return False, "not finite: %r" % s, None
        healed = None
        if t == "int":
            try:
                int(s)
            except ValueError:
                if f.is_integer():
                    healed = str(int(f))
                else:
                    return False, "not an integer: %r" % s, None
        problem = None
        if key.lo is not None and f < key.lo:
            problem = "below minimum %s: %s" % (key.lo, s)
        elif key.hi is not None and f > key.hi:
            problem = "above maximum %s: %s" % (key.hi, s)
        if healed:
            problem = (problem + "; " if problem else "") + "integral float healed %r -> %r" % (s, healed)
        return True, problem, healed
    if t == "bool":
        if s == "":
            return True, "empty boolean (readers disagree on its meaning; set 0 or 1)", None
        c = canonical_bool(s)
        if c is None:
            return True, "not a boolean: %r (readers treat it as true)" % s, None
        return True, None, (c if c != s else None)
    if t == "enum":
        if s.lower() not in key.choices:
            return True, "not one of %s: %r" % ("|".join(key.choices), s), None
        return True, None, None
    if t == "url":
        if s and not re.match(r"^https?://[^\s/]+", s):
            return True, "not an http(s) URL: %r" % s, None
        return True, None, None
    if t == "hours":
        ok = all(re.fullmatch(r"(\d{1,2})(-(\d{1,2}))?", p.strip()) and all(0 <= int(x) <= 24 for x in re.findall(r"\d+", p))
                 for p in s.split(",") if p.strip())
        return True, (None if ok else "not UTC hour ranges like 1-4,6-10: %r" % s), None
    if t == "map":
        bad = [p for p in s.split(",") if p.strip() and "=" not in p]
        return True, ("entries without '=': %s" % bad[:3] if bad else None), None
    if t == "json":
        if s:
            try:
                json.loads(s)
            except ValueError as e:
                return True, "invalid JSON (%s)" % e, None
        return True, None, None
    return True, None, None


# ------------------------------------------------------------------ env file parsing
def parse_env_file(path):
    """systemd EnvironmentFile semantics, enough for shim.env: KEY=VALUE lines, '#'/';' comments, optional
    surrounding quotes, LAST assignment wins. Returns (entries, error). entries: list of (lineno, key, value)."""
    try:
        with open(path) as fh:
            text = fh.read()
    except OSError as e:
        return [], "%s: %s" % (path, e)
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s or s[0] in "#;":
            continue
        if "=" not in s:
            out.append((n, s, None))
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        if k.startswith("export "):
            k = k[7:].strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        out.append((n, k, v))
    return out, None


def lint_entries(entries, schema=SCHEMA):
    """Findings for parsed env-file entries. Each finding: {level, code, key, lines, msg}. Never includes a
    secret value."""
    findings = []
    by_key = {}
    for n, k, v in entries:
        if v is None:
            findings.append(dict(level="error", code="malformed", key=k[:40], lines=[n], msg="line has no '='"))
            continue
        by_key.setdefault(k, []).append((n, v))
    for k, rows in sorted(by_key.items()):
        key = schema.get(k)
        secret = key.secret if key else bool(_SECRET_RE.search(k))
        if len(rows) > 1:
            vals = {v for _, v in rows}
            same = len(vals) == 1
            findings.append(dict(level="warn", code="duplicate", key=k, lines=[n for n, _ in rows],
                                 msg="%d assignments; systemd uses the LAST (line %d)%s" % (
                                     len(rows), rows[-1][0], "" if same else (
                                         "; values differ" if secret else "; values differ: " + " | ".join(
                                             "L%d=%s" % (n, v[:60]) for n, v in rows)))))
        if key is None:
            if k.startswith("SHIM_") or k.startswith("GATEWAY_"):
                findings.append(dict(level="warn", code="unknown", key=k, lines=[rows[-1][0]],
                                     msg="no code reads this key (typo, or a knob that was removed)"))
            continue
        n, v = rows[-1]
        ok, problem, healed = check_value(key, v)
        if not ok:
            findings.append(dict(level="error", code="invalid", key=k, lines=[n],
                                 msg=(problem if not secret else "invalid value") + "; the gateway ignores it and uses the code default"))
        elif problem:
            findings.append(dict(level="warn", code="range" if "minimum" in problem or "maximum" in problem else "suspicious",
                                 key=k, lines=[n], msg=problem if not secret else "suspicious value"))
        elif healed is not None and key.type == "bool":
            findings.append(dict(level="info", code="noncanonical-bool", key=k, lines=[n],
                                 msg="%r read as %s (canonical form is 0/1)" % (v, healed)))
    return findings


def last_wins(entries):
    d = {}
    for n, k, v in entries:
        if v is not None:
            d[k] = (n, v)
    return d


# ------------------------------------------------------------------ startup sanitiser
STARTUP = {"at": None, "env_file": None, "raw": {}, "changed": {}, "dropped": {}, "findings": [], "error": None}


def sanitize_environ(environ=None, env_file=None, schema=SCHEMA, logger=None):
    """Validate the SHIM_* environment BEFORE the gateway's import-time reads. Mutates environ only to (a) drop
    a numeric value the reader would crash on (code default then applies) and (b) canonicalise booleans to
    1/0. Never raises. Returns (and records in STARTUP) a report; the original raw values are kept in
    STARTUP['raw'] so the config-ownership check can compare against what systemd actually loaded."""
    lg = logger or log
    environ = os.environ if environ is None else environ
    rep = STARTUP
    rep.update(at=time.time(), env_file=env_file, raw={}, changed={}, dropped={}, findings=[], error=None)
    try:
        for name, key in schema.items():
            if name not in environ:
                continue
            raw = environ[name]
            rep["raw"][name] = raw
            ok, problem, healed = check_value(key, raw)
            shown = "<secret>" if key.secret else repr(raw)
            if not ok:
                del environ[name]
                rep["dropped"][name] = problem
                lg.warning("CONFIG INVALID %s=%s: %s -> IGNORED, using the code default %r", name, shown, problem,
                           key.default if not key.computed_default else "(computed)")
                rep["findings"].append(dict(level="error", code="invalid", key=name, msg=problem))
                continue
            if healed is not None:
                environ[name] = healed
                rep["changed"][name] = healed
                if key.type != "bool":
                    lg.warning("CONFIG HEALED %s=%s -> %r (%s)", name, shown, healed, problem)
            if problem:
                lg.warning("CONFIG WARNING %s=%s: %s", name, shown, problem)
                rep["findings"].append(dict(level="warn", code="suspicious", key=name, msg=problem))
        for name in sorted(environ):
            if (name.startswith("SHIM_") or name.startswith("GATEWAY_")) and name not in schema:
                lg.warning("CONFIG UNKNOWN %s is set but no gateway code reads it (typo or removed knob)", name)
                rep["findings"].append(dict(level="warn", code="unknown", key=name, msg="set in environment, not read"))
        if env_file:
            entries, err = parse_env_file(env_file)
            if err:
                rep["findings"].append(dict(level="warn", code="env-file", key="", msg=err))
            for f in lint_entries(entries, schema):
                if f["code"] in ("duplicate", "malformed"):
                    lg.warning("CONFIG FILE %s %s: %s (lines %s)", f["code"], f["key"], f["msg"], f["lines"])
                    rep["findings"].append(f)
        n_err = sum(1 for f in rep["findings"] if f["level"] == "error")
        n_warn = sum(1 for f in rep["findings"] if f["level"] == "warn")
        lg.warning("CONFIG schema v%d: %d keys set, %d invalid (ignored), %d warnings, %d booleans canonicalised",
                   SCHEMA_VERSION, len(rep["raw"]), n_err, n_warn,
                   sum(1 for k in rep["changed"] if schema[k].type == "bool"))
    except Exception as e:   # fail safe: validation must never take the gateway down
        rep["error"] = repr(e)
        try:
            lg.error("CONFIG schema validation failed (gateway continues unvalidated): %r", e)
        except Exception:
            pass
    return rep


def same_value(name, a, b):
    """Type-aware equality of two raw strings for name ('True' == '1' for a boolean, '6' == '6.0' for a number)."""
    key = SCHEMA.get(name)
    if key is None or a is None or b is None:
        return a == b
    try:
        return _norm(a, key) == _norm(b, key)
    except Exception:
        return a == b


def raw_value(name, environ=None):
    """What systemd loaded for name (before sanitising), falling back to the current environment."""
    environ = os.environ if environ is None else environ
    return STARTUP["raw"].get(name, environ.get(name))


# ------------------------------------------------------------------ effective view
def _render(v, key: Key | None):
    if v is None:
        return "auto" if key is not None and key.type == "budget" else None
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (set, frozenset)):
        return (key.sep if key else ",").join(sorted(str(x) for x in v))
    if isinstance(v, (list, tuple)):
        return (key.sep if key else ",").join(str(x) for x in v)
    if isinstance(v, (int, float, str)):
        return v
    return str(v)


def _norm(v, key: Key):
    """Comparable form of a value or default string for differs-from-default."""
    if v is None:
        return None
    t = key.type
    s = str(v).strip()
    try:
        if t in ("int", "float"):
            return float(s)
        if t == "budget":
            return None if s.lower() in ("", "auto", "none", "live") else float(s)
        if t == "bool":
            return canonical_bool(s) or s.lower()
    except ValueError:
        return s
    if t == "list":
        sep = key.sep if t == "list" else ","
        parts = [p.strip().lower() if key.name != "SHIM_BG_MARKERS" else p.strip() for p in s.replace("|", sep).split(sep)]
        return tuple(sorted(p for p in parts if p))
    if t == "enum":
        return s.lower()
    if t == "path":
        return os.path.expanduser(s)
    return s


def mask(key: Key | None, value):
    if value is None or value == "":
        return value
    return "<set, %d chars>" % len(str(value))


def effective(module_globals=None, environ=None, env_file=None, schema=SCHEMA, only_changed=False, key_filter=None):
    """The table behind GET /gateway/config/effective. module_globals = the gateway's globals() (live values);
    environ = current process environment (after sanitising); env_file = shim.env (what a restart would load)."""
    environ = os.environ if environ is None else environ
    g = module_globals or {}
    entries, err = parse_env_file(env_file) if env_file else ([], None)
    persisted = last_wins(entries)
    rows = []
    for name in sorted(schema):
        key = schema[name]
        if key_filter and key_filter.upper() not in name:
            continue
        in_env = name in environ
        env_val = environ.get(name)
        dropped = name in STARTUP["dropped"]
        if key.gname and key.gname in g:
            live = _render(g[key.gname], key)
            have_live = True
        else:
            live, have_live = (env_val if in_env else key.default), False
        if dropped:
            source = "default (env value invalid: %s)" % STARTUP["dropped"][name]
        elif in_env:
            source = "env"
        else:
            source = "default"
        # a dashboard live edit moved the global away from what the environment gave it at startup
        if have_live and in_env and not dropped and _norm(live, key) != _norm(env_val, key) and key.tunable:
            source = "live-edit (env had %s)" % ("<secret>" if key.secret else env_val)
        differs = None if key.computed_default else (_norm(live, key) != _norm(key.default, key))
        if not in_env and not have_live and not dropped:
            differs = False
        p = persisted.get(name)
        pval = p[1] if p else None
        restart_changes = None
        if p is not None or in_env:
            restart_changes = _norm(pval, key) != _norm(STARTUP["raw"].get(name, env_val), key) if (p is not None) else True
        row = dict(key=name, type=key.type, unit=key.unit, value=live, default=key.default if not key.computed_default else "(computed)",
                   source=source, differs_from_default=differs, persisted=pval, persisted_line=p[0] if p else None,
                   restart_would_change=bool(restart_changes), tunable=key.tunable, kill_switch=key.kill, money=key.money,
                   secret=key.secret, owner=key.owner, desc=key.desc)
        if key.secret:
            row["value"] = mask(key, live)
            row["persisted"] = mask(key, pval)
        if only_changed and not (differs or row["restart_would_change"] or source != "default"):
            continue
        rows.append(row)
    unknown = sorted(k for k in persisted if k not in schema and (k.startswith("SHIM_") or k.startswith("GATEWAY_")))
    lint = lint_entries(entries, schema) if env_file else []
    summary = dict(keys=len(schema), set_in_env=sum(1 for k in schema if k in environ),
                   differs_from_default=sum(1 for r in rows if r["differs_from_default"]),
                   invalid_ignored=sorted(STARTUP["dropped"]), unknown_in_file=unknown,
                   duplicates_in_file=sorted({f["key"] for f in lint if f["code"] == "duplicate"}),
                   restart_would_change=sorted(r["key"] for r in rows if r["restart_would_change"]),
                   live_edits=sorted(r["key"] for r in rows if r["source"].startswith("live-edit")),
                   file_findings=len(lint))
    return dict(schema_version=SCHEMA_VERSION, as_of=time.strftime("%Y-%m-%dT%H:%M:%S%z"), env_file=env_file,
                env_file_error=err, startup=dict(at=STARTUP["at"], error=STARTUP["error"],
                                                 findings=STARTUP["findings"], booleans_canonicalised=sorted(STARTUP["changed"])),
                summary=summary, file_findings=lint, keys=rows)


# ------------------------------------------------------------------ code scan (used by the test and `scan`)
def scan_code(path):
    """{key: [(lineno, default_literal_or_marker), ...]} for every literal-named env read in a Python file:
    os.environ.get("K", d), os.getenv("K", d), os.environ["K"], "K" in os.environ."""
    src = open(path).read()
    tree = ast.parse(src)
    hits = {}

    def lit(n):
        return n.value if isinstance(n, ast.Constant) and isinstance(n.value, str) else None
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and ast.unparse(n.func) in ("os.environ.get", "os.getenv", "environ.get") and n.args:
            k = lit(n.args[0])
            if k:
                d = None
                if len(n.args) > 1:
                    d = n.args[1].value if isinstance(n.args[1], ast.Constant) else "<expr>"
                hits.setdefault(k, []).append((n.lineno, d))
        elif isinstance(n, ast.Subscript) and ast.unparse(n.value) == "os.environ":
            k = lit(n.slice)
            if k:
                hits.setdefault(k, []).append((n.lineno, "<required>"))
        elif isinstance(n, ast.Compare) and len(n.comparators) == 1 and ast.unparse(n.comparators[0]) == "os.environ":
            k = lit(n.left)
            if k:
                hits.setdefault(k, []).append((n.lineno, "<presence>"))
    return hits


def markdown_table(schema=SCHEMA):
    out = ["| key | type | default | unit | range | flags | owner | description |", "|---|---|---|---|---|---|---|---|"]
    for name in sorted(schema):
        k = schema[name]
        rng = ("|".join(k.choices) if k.choices else
               ("" if k.lo is None and k.hi is None else "%s..%s" % ("" if k.lo is None else k.lo, "" if k.hi is None else k.hi)))
        flags = ",".join(f for f, on in (("tunable", k.tunable), ("kill", k.kill), ("money", k.money),
                                          ("secret", k.secret), ("restart", k.restart)) if on)
        d = "(computed)" if k.computed_default else ("" if k.default is None else k.default)
        out.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (name, k.type, str(d).replace("|", "\\|"), k.unit, rng, flags, k.owner, k.desc))
    return "\n".join(out)


def _main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lint", help="lint a shim.env (read-only)")
    p.add_argument("path", nargs="?", default=os.path.expanduser("~/.local/share/vllm-qwen27b/shim.env"))
    p.add_argument("--json", action="store_true")
    p.add_argument("--quiet-info", action="store_true", help="hide info-level findings")
    s = sub.add_parser("scan", help="env keys the code reads that the schema lacks")
    s.add_argument("path", nargs="?", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "keepalive-shim.py"))
    sub.add_parser("doc", help="print the schema as a markdown table")
    a = ap.parse_args(argv)
    if a.cmd == "doc":
        print(markdown_table())
        return 0
    if a.cmd == "scan":
        hits = scan_code(a.path)
        missing = sorted(k for k in hits if k not in SCHEMA and (k.startswith("SHIM_") or k.startswith("GATEWAY_")))
        stale = sorted(k for k in SCHEMA if k not in hits)
        for k in missing:
            print("MISSING  %s  read at %s" % (k, ", ".join("L%d default=%r" % h for h in hits[k])))
        for k in stale:
            print("STALE    %s  in the schema but no longer read" % k)
        print("%d keys read, %d in schema, %d missing, %d stale" % (len(hits), len(SCHEMA), len(missing), len(stale)))
        return 1 if missing else 0
    entries, err = parse_env_file(a.path)
    if err:
        print("ERROR", err)
        return 2
    f = lint_entries(entries)
    if a.quiet_info:
        f = [x for x in f if x["level"] != "info"]
    if a.json:
        print(json.dumps(dict(path=a.path, lines=len(entries), findings=f), indent=1))
    else:
        for x in f:
            print("%-5s %-17s %-40s lines %-10s %s" % (x["level"].upper(), x["code"], x["key"], ",".join(map(str, x["lines"])), x["msg"]))
        print("%s: %d assignments, %d errors, %d warnings, %d info" % (
            a.path, len(entries), sum(1 for x in f if x["level"] == "error"), sum(1 for x in f if x["level"] == "warn"),
            sum(1 for x in f if x["level"] == "info")))
    return 1 if any(x["level"] == "error" for x in f) else 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
