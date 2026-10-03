#!/usr/bin/env python3
"""
gateway-shim — capacity-routing front door on :8000 for the 2x2080Ti box.

Local-first, remote-overflow, self-healing. Everything is passed through to the
local vLLM (:8001) DYNAMICALLY — no model names are hardcoded, so any model you
add to vLLM tomorrow just works. When local is at capacity or unhealthy, requests
transparently overflow to a remote OpenAI-compatible endpoint (DeepSeek).

Placement (per request):
  - Estimate cost in "units": under SHIM_BIG_TOKENS = 1; at or above it,
    ceil(tokens / SHIM_TOKENS_PER_UNIT), capped at budget-SHIM_FG_RESERVED so one big
    request can never claim every lane (2026-09-11: it used to cost the WHOLE budget,
    which let a single 100K+ prompt lock out every other caller -- see
    gw-admission-proportional-units in the vault's Qwen3.8 Capacity Tuning Backlog).
  - Local budget = SHIM_LOCAL_BUDGET units (default 14 in production), matching the
    box's chaos-tested envelope; the per-request KV/token safety net is TOKEN_BUDGET
    (total in-flight PROMPT tokens across ALL admitted requests), independent of units.
  - If local is healthy AND admitting this request stays within budget => LOCAL.
    Otherwise => REMOTE (model rewritten to SHIM_REMOTE_MODEL).
Self-heal:
  - Any local OOM / EngineDead / 5xx / connection error => this request FAILS OVER
    to remote, AND local budget is cut to 1 for SHIM_OOM_BACKOFF_SECS, then recovers.
    So the box finds its own safe concurrency without anyone configuring the number.
Cost-first: a single request is always local (free); only real overflow costs money.

NEVER manages the vLLM process (that's vllm-qwen27b.service + its watchdog).
"""
import asyncio, os, sys, json, time, logging, collections, subprocess, re, hmac, math, hashlib
import fcntl
import threading
import copy
import datetime
import urllib.request
import aiohttp
from aiohttp import web

# -------- config sequences: ONE separator per field, on both the read and the write --------
# 2026-09-09 RCA. NO_THINK_IPS was dead for ~3 weeks and said nothing about it. h_chat gates
# the no-think policy on `request.remote in NO_THINK_IPS`; the set held exactly one member --
# this entire run of garbage -- so neither 10.0.1.10 (Hermes) nor 10.0.1.250 (the scheduler
# peer) could ever match it:
#     {{'''"{'10.0.1.10'"}'}'}'''}'}'}|{'{'{'''{'{'{"'10.0.1.250'}"'''}'}'''
#
# This file corrupted its own config. No shell was involved. Two halves of THIS module
# disagreed: the readers below split on "," while _persist_config joined on "|", so the first
# dashboard save of ANY field rewrote a good value into one the reader could not parse. An
# older _persist_config was worse -- it wrote str(g[gname]) for every type, so a set reached
# disk as its Python repr, braces and quotes included, and each save then re-serialised the
# previous parse of the previous repr. The punctuation COMPOUNDED. The shim.env backups in
# this directory are that ratchet, one doubling per save:
#     08-14  {'10.0.1.10', '10.0.1.250'}                     <- str(set) written
#     08-16  {"'10.0.1.250'}", "{'10.0.1.10'"}               <- comma-split of that
#     08-24  {'{\'\'\'"{\'10.0.1.10\'"}\'}\'}\'', ...        <- and again
#     09-05  {{'''"{'10.0.1.10'"}...}|{'{'{...               <- the "|" join froze it there
# It has the shape the vault files under [[Workflow Interpolation Footgun]], reached with no
# shell at all. SHIM_BG_XCLIENTS was dead the same way, from the same asymmetry.
#
# The fix is both halves, because a tolerant parser alone would just let the writer lay down
# the next bad value. _CFG_SEP pins the canonical separator per field -- the one that field's
# reader has always used -- and BOTH writers (_persist_config to shim.env, current_config to
# the dashboard form) now use it, so the round trip is lossless. _parse_seq additionally
# accepts the other separator and strips repr punctuation, so a value already corrupt on disk
# self-heals on the next read instead of staying dead until a human happens to notice.
# Covered by test_no_think_ips.py beside this file.
_CFG_SEP = {
    "SHIM_NO_THINK_IPS": ",",
    "SHIM_BG_XCLIENTS":  ",",
    "SHIM_BG_MARKERS":   "|",   # free-text phrases: a marker may itself contain a comma
    "SHIM_LOCAL_FIRST_REASONS": ",",   # read back by _parse_reason_set's comma split
}
_CFG_SEP_DEFAULT = "|"
# Punctuation a Python container repr leaves around a member. Stripped from the ENDS only,
# never from the middle, so a legitimate value keeps its interior characters.
_REPR_JUNK = " \t\r\n{}[]()'\""


def _parse_seq(value, sep=",", heal=True):
    """Split a persisted config scalar into its member tokens.

    sep  -- the field's canonical separator, from _CFG_SEP.
    heal -- also accept the OTHER separator and strip container-repr punctuation, so a value
            written by the old broken writer comes back to life. Turn this OFF for free-text
            fields (BG_MARKERS), where a quote or a comma may be part of the value itself.

    Returns a list: order preserved, duplicates dropped, empties dropped. Callers that want a
    set wrap it, so this one function serves both shapes.
    """
    if value is None:
        return []
    if isinstance(value, (set, frozenset, list, tuple)):
        # A caster can be handed a real container (a dashboard POST, or a re-cast of a value
        # already parsed). Flatten it through the same path, so a member that is itself a
        # corrupt joined string still splits.
        items = sorted(value) if isinstance(value, (set, frozenset)) else list(value)
        out = []
        for item in items:
            out.extend(_parse_seq(item, sep, heal))
    else:
        s = str(value)
        if heal and sep != _CFG_SEP_DEFAULT:
            s = s.replace(_CFG_SEP_DEFAULT, sep)
        out = []
        for tok in s.split(sep):
            tok = tok.strip(_REPR_JUNK) if heal else tok.strip()
            if tok:
                out.append(tok)
    seen, uniq = set(), []
    for tok in out:
        if tok not in seen:
            seen.add(tok)
            uniq.append(tok)
    return uniq


def _fmt_seq(value, sep):
    """Serialise a config sequence with the separator its own reader expects.

    Sets are sorted so a save is deterministic -- otherwise shim.env churns on every write and
    a diff can never tell you whether the value actually changed.
    """
    items = sorted(value) if isinstance(value, (set, frozenset)) else list(value)
    return sep.join(str(x) for x in items)


# Lane CFG (2026-10-03): typed config schema (gateway_config_schema.py, shipped beside this file). It validates every
# SHIM_* value BEFORE the import-time reads below: a number the reader would crash on is dropped (code default applies,
# loud warning), booleans are canonicalised to 1/0 (a dashboard-persisted "False" used to come back as ON after a
# restart), and shim.env duplicates / unknown keys are logged. Fail safe: if the module is missing or raises, the
# gateway runs exactly as before. Effective values: GET /gateway/config/effective.
def _load_cfg_schema():
    try:
        import importlib.util as _ilu
        _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gateway_config_schema.py")
        _spec = _ilu.spec_from_file_location("gateway_config_schema", _p)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _mod.sanitize_environ(os.environ, env_file=os.environ.get(
            "SHIM_ENV_FILE", "/home/kevin/.local/share/vllm-qwen27b/shim.env"))
        return _mod
    except Exception as _e:
        print("gateway-shim: config schema unavailable (%r); running unvalidated" % (_e,), file=sys.stderr)
        return None


_cfgschema = _load_cfg_schema()


# Lane SH (2026-10-03): ONE boolean reader. Before, 26 import-time readers and 19 dashboard casts used four different
# spellings: `not in ("0","false","")` turned "off"/"no"/"False " ON, `in ("1","true","on")` turned "yes" OFF, and ""
# meant ON for two keys but OFF for the rest. Vocabulary = the config schema's (fallback copy if it is absent);
# an empty or unrecognised value means the code default (and is reported), never a silent guess.
_FLAG_TRUE = frozenset(getattr(_cfgschema, "TRUE_TOKENS", None) or {"1", "true", "on", "yes", "y", "t"})
_FLAG_FALSE = frozenset(getattr(_cfgschema, "FALSE_TOKENS", None) or {"0", "false", "off", "no", "n", "f"})


def _flag_value(raw):
    """True / False for a recognised spelling, None otherwise (None, "", junk)."""
    if raw is None or isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    return True if s in _FLAG_TRUE else (False if s in _FLAG_FALSE else None)


def _env_flag(raw, default):
    """Import-time boolean: wrap the literal read, `_env_flag(os.environ.get("SHIM_X", "1"), True)`."""
    v = _flag_value(raw)
    if v is None:
        if raw is not None and str(raw).strip():
            print("gateway-shim: not a boolean: %r -> code default %s" % (raw, bool(default)), file=sys.stderr)
        return bool(default)
    return v


def _cfg_flag(v):
    """Dashboard/API cast (apply_config): an unrecognised value is refused (logged, live value unchanged)."""
    out = _flag_value(v)
    if out is None:
        raise ValueError("not a boolean: %r" % (v,))
    return out
PORT         = int(os.environ.get("SHIM_PORT", "8000"))
LOCAL        = os.environ.get("SHIM_UPSTREAM", "http://127.0.0.1:8001").rstrip("/")
# where live config edits (via the dashboard) are persisted so they survive a restart
SHIM_ENV_FILE = os.environ.get("SHIM_ENV_FILE", "/home/kevin/.local/share/vllm-qwen27b/shim.env")
REMOTE_BASE  = os.environ.get("SHIM_REMOTE_BASE", "").rstrip("/")
REMOTE_KEY   = os.environ.get("SHIM_REMOTE_KEY", "")
REMOTE_MODEL = os.environ.get("SHIM_REMOTE_MODEL", "deepseek-v4-flash")
# Provider context capabilities.  These are admission limits, not prompt truncation
# instructions: every provider receives an explicit, intact transcript or an explicit
# context-capacity error.  The local value is kept separate from MAX_LOCAL_TOKENS so a
# deployment can change the engine's output guard without lying about its context window.
LOCAL_CONTEXT_LIMIT = int(os.environ.get("SHIM_LOCAL_CONTEXT_LIMIT", "524288"))
REMOTE_CONTEXT_LIMIT = int(os.environ.get("SHIM_REMOTE_CONTEXT_LIMIT", "128000"))
CONTEXT_SAFETY_MARGIN = int(os.environ.get("SHIM_CONTEXT_SAFETY_MARGIN", "1024"))
CONTEXT_COMPACTION_ENABLED = _env_flag(os.environ.get("SHIM_CONTEXT_COMPACTION", "1"), True)
CONTEXT_COMPACTION_KEEP = int(os.environ.get("SHIM_CONTEXT_COMPACTION_KEEP", "12"))
ALIASES_FILE = os.environ.get("SHIM_ALIASES_FILE", "/home/kevin/.local/share/vllm-qwen27b/gateway-aliases.json")
BUDGET       = int(os.environ.get("SHIM_LOCAL_BUDGET", "2"))
BIG_TOKENS   = int(os.environ.get("SHIM_BIG_TOKENS", "40000"))
OOM_BACKOFF  = int(os.environ.get("SHIM_OOM_BACKOFF_SECS", "120"))
# Queue-first: when local is healthy but at capacity, WAIT up to LOCAL_WAIT secs for a slot
# to free before overflowing to remote. Turns bursty harness turns (a few quick calls) into
# local-serial instead of DeepSeek-overflow. Concurrency stays <= budget, so NO new OOM risk.
# Set SHIM_LOCAL_WAIT_SECS=0 to restore instant-overflow.
LOCAL_WAIT   = float(os.environ.get("SHIM_LOCAL_WAIT_SECS", "8"))
SLOT_POLL    = float(os.environ.get("SHIM_SLOT_POLL_SECS", "0.05"))
HEALTH_TTL   = 5
CHARS_PER_TOK = 3.5
# Exact tokenisation for routing decisions. See _tokenize_exact().
EXACT_TOKENS      = _env_flag(os.environ.get("SHIM_EXACT_TOKENS", "1"), True)
MIN_CHARS_PER_TOK = float(os.environ.get("SHIM_MIN_CHARS_PER_TOK", "2.0"))
TOKENIZE_TIMEOUT  = float(os.environ.get("SHIM_TOKENIZE_TIMEOUT", "5"))
# Adaptive first-token deadline: base + est_prompt_tokens/prefill_rate. Big-context prefills
# legitimately take a while, so this scales with prompt size; a WEDGED backend (accepts the
# connection but never emits a token) still fails over in ~base seconds instead of hanging.
FIRST_TOKEN_BASE = float(os.environ.get("SHIM_FIRST_TOKEN_BASE", "15"))
PREFILL_TPS      = float(os.environ.get("SHIM_PREFILL_TPS", "500"))
# Hard ceiling on the first-token deadline: no matter prompt size / concurrency, if local hasn't
# produced a token within this many seconds, fail the request over to remote. Keeps interactive
# clients (Hermes) responsive on a saturated box instead of hanging for minutes. (2026-08-13)
FIRST_TOKEN_MAX  = float(os.environ.get("SHIM_FIRST_TOKEN_MAX", "45"))
# Big-OUTPUT requests (max_tokens >= this) are long generations that saturate this slow box and
# starve everything else — route them straight to remote (DeepSeek is far faster for them anyway).
# Hermes fires max_tokens=65536; pi uses <=8192, so this cleanly separates them. 0 = disabled.
BIG_OUTPUT       = int(os.environ.get("SHIM_BIG_OUTPUT", "32000"))
# Big-PROMPT requests (est. prompt tokens >= this) can't prefill within the first-token cap on this
# slow box (~PREFILL_TPS tok/s), so they'd waste a local lane then fail over anyway — AND a client
# whose context GROWS over a long session (e.g. pi) would otherwise silently start saturating/OOM-ing
# local. Route them straight to remote up front. Default ≈ FIRST_TOKEN_MAX×PREFILL_TPS (what local
# can actually prefill in time). 0 = disabled. Raise it (with FIRST_TOKEN_MAX) to keep more on local.
BIG_PROMPT       = int(os.environ.get("SHIM_BIG_PROMPT", "24000"))
# MONSTER-IN-FLIGHT bypass (2026-08-16, Local Context Lane Guarantee): while a huge prefill is
# chewing engine steps, concurrent decode advances ~1 tok per chunk-step (~4s at mnbt 3584 —
# a one-line reply measured 118s). While shim-tracked in-flight prompt tokens >= this, NEW
# arrivals route straight to remote until the monster drains. Only sees shim-routed monsters
# (direct-:8001 probes are invisible — acceptable: production traffic enters here). 0 = off.
MONSTER_INFLIGHT = int(os.environ.get("SHIM_MONSTER_INFLIGHT", "120000"))
# FOREIGN-LOAD sensing (2026-08-17): direct-:8001 traffic (probes, benches) is
# invisible to the shim's own admission tracking, so a foreign monster prefill
# starved gateway traffic at ~1 tok per chunk-step (observed: Hermes scheduler
# "unreasonably slow" during probe campaigns). Scrape the engine's own running
# count on the health poll; if it exceeds what WE admitted, treat it like a
# monster in flight and route new arrivals remote. 0 = off.
FOREIGN_LOAD_GUARD = int(os.environ.get("SHIM_FOREIGN_LOAD_GUARD", "1"))
# 2026-09-05 rework: the guard used to fire on ANY foreign request (engine running > shim admitted),
# so a sequential evalkit pass of tiny items against :8001 diverted every gateway arrival to the
# remote provider for 17 minutes (121 requests, incl. 6-token probes). A foreign request only
# hurts gateway traffic when it is BIG (a monster prefill), so the guard now requires the engine's
# KV usage to exceed what the shim itself admitted by at least BIG_TOKENS worth of tokens.
# POOL_TOKENS = the engine's KV pool size (GPU KV cache size line); hot-configurable.
POOL_TOKENS = int(os.environ.get("SHIM_POOL_TOKENS", "754068"))
# --- CRASH-ADAPTIVE big_prompt guard (2026-08-16) ---
# Kevin's manual pattern after a local crash has been to hand-lower SHIM_BIG_PROMPT to reduce
# OOM risk, and never raise it back -- a one-way crash -> degrade -> lower-the-floor spiral.
# When enabled, this automates both halves: on a detected local crash (see trigger_backoff(),
# fed by real EngineDead/OOM/503 failures only -- NOT routine busy/wedged failovers), if
# BIG_PROMPT is currently above the safe floor it is saved to _big_prompt_restore and dropped
# to CRASH_ADAPTIVE_FLOOR immediately. It is restored automatically once local has proven
# itself stable again: BIG_PROMPT_RESTORE_N consecutive LOCAL completions whose prompt size is
# "big" relative to the (lowered) floor land cleanly. Master-switched off by default so there
# is no behaviour change until Kevin opts in.
CRASH_ADAPTIVE        = _env_flag(os.environ.get("SHIM_CRASH_ADAPTIVE", "0"), False)
BIG_PROMPT_RESTORE_N  = int(os.environ.get("SHIM_BIG_PROMPT_RESTORE_N", "5"))
CRASH_ADAPTIVE_FLOOR  = 24000
# A completion with ptok >= the CURRENT (possibly-floored) BIG_PROMPT never reaches local -- the
# big-prompt guard above routes it straight to remote -- so the "clean streak" evidence has to be
# gathered from prompts sized close to (but under) the active floor, not at/above it. Fraction of
# CRASH_ADAPTIVE_FLOOR a local completion's prompt must reach to count as restore-evidence.
BIG_PROMPT_QUALIFY_FRAC = float(os.environ.get("SHIM_BIG_PROMPT_QUALIFY_FRAC", "0.8"))
# Bound local generation: a request with max_tokens ABSENT or 0 is unbounded (vLLM would generate up
# to the context limit) — a runaway long generation on local that saturates the box. When we route
# such a request to LOCAL, inject this cap so local only ever does bounded generations. Explicit
# max_tokens the client set (and < BIG_OUTPUT) are respected. pi + Hermes both send max_tokens=0.
LOCAL_MAX_OUT    = int(os.environ.get("SHIM_LOCAL_MAX_OUT", "8192"))
# --- PRIORITY LANES (2026-08-13): interactive-first ---
# The user runs 1-2 INTERACTIVE agents; the latency pain comes from BACKGROUND cron jobs
# (Hermes fires one every ~2 min) saturating all lanes so interactive turns queue behind
# robot busywork and decode at 4-way shared bandwidth. Classify background requests
# (X-Client containing "cron", or a prompt carrying a BG marker like Hermes's literal
# "scheduled cron job" preamble) and make them YIELD: they may only use lanes beyond
# FG_RESERVED (kept free for interactive), and they wait only BG_WAIT before overflowing
# to the cheap remote. Interactive traffic keeps the full budget and queue-first wait.
BG_MARKERS  = [m for m in os.environ.get("SHIM_BG_MARKERS", "scheduled cron job").split("|") if m]
FG_RESERVED = int(os.environ.get("SHIM_FG_RESERVED", "2"))
# 2026-09-05 (Dreams forensics): a "big" request costs the WHOLE budget (estimate_units), while
# background admission is capped at budget-FG_RESERVED -- so a background request >= BIG_TOKENS
# could NEVER run locally and always overflowed to the remote provider (26 of 35 Dreams requests
# on 09-05 went to DeepSeek at 41K-132K tokens). Local-first is the product rule: when the engine
# is completely idle (nothing in flight, nobody queued) a big background request may take the
# whole budget, exactly like a big foreground one does. Set 0 to restore the old always-remote rule.
BG_BIG_LOCAL_WHEN_IDLE = int(os.environ.get("SHIM_BG_BIG_LOCAL_WHEN_IDLE", "1"))
# 2026-09-11 (Kevin: "the uncensored local model IS the point of the estate"; measured: his
# interactive pi session was served locally only 7% of the time, 448/482 requests overflowed
# to paid DeepSeek in 2 days). gw-interactive-never-overflows: an INTERACTIVE (non-background)
# request that is merely WAITING for a lane -- local is healthy, just busy -- no longer times
# out into an overflow. It keeps waiting for a local lane instead (same 1e9 sentinel already
# used when no remote is configured at all). This does NOT touch the local-DOWN path above
# (a genuinely dead/unhealthy engine still overflows interactive immediately, same as before --
# waiting for a lane that will never open is not "the point"), and does not touch background,
# which still yields and overflows on its own short timer. Default on; set 0 to restore the
# previous LOCAL_WAIT-then-overflow behaviour for interactive traffic.
INTERACTIVE_NEVER_OVERFLOW = int(os.environ.get("SHIM_INTERACTIVE_NEVER_OVERFLOW", "1"))
# 2026-09-11 (Kevin: "response time... is unbearably slow"; gw-admission-proportional-units).
# estimate_units() used to charge ANY request >= BIG_TOKENS the WHOLE current budget, on the
# theory that a big prefill "runs alone." Measured effect: one of Kevin's own 111K-token pi
# turns holds 14/14 lanes for the length of its wait+prefill+decode, so every OTHER request --
# including his own next interactive turn -- queues up to LOCAL_WAIT and then overflows to
# paid remote (his pi session was served locally only 7% of the time). The engine already has
# an independent, finer-grained memory-safety net for concurrent big prefills: TOKEN_BUDGET
# bounds total in-flight PROMPT TOKENS across ALL admitted requests regardless of unit count
# (see the `_inflight_tokens + ptok <= TOKEN_BUDGET` check below) -- "units" is a proxy for
# concurrent SEQUENCE SLOTS (vLLM's max_num_seqs), not for KV/token budget, so a big request
# does not need every slot just because it is big. Units are now proportional to estimated
# size, capped so a single request can never claim more than budget-FG_RESERVED (at least
# FG_RESERVED lanes free for the NEXT caller, always, regardless of class) and floored at 1
# (a request is never inadmissible). SHIM_TOKENS_PER_UNIT is hot-reloadable via
# /gateway/config so the ratio can be tuned from the bake-off without a restart. This does NOT
# relax the TOKEN_BUDGET check, which remains the real KV-safety gate.
TOKENS_PER_UNIT = int(os.environ.get("SHIM_TOKENS_PER_UNIT", "12000"))
# 2026-09-11 (gw-admission-computed-token-cost). estimate_units() charges a big request by its
# PROMPT LENGTH, but prefix caching means a growing multi-turn conversation (pi's own pattern --
# each turn resends the whole history plus one new message) only actually costs the engine the
# NEW suffix, not the whole resent prefix: a 111K-token turn that is 98% cached is ~2K tokens of
# real prefill work. predict_computed_tokens() tracks a one-deep per-client prefix hash and
# predicts "only the delta since last turn" on a hit, full cost on any miss (new client, first
# turn, edited/branched history). PREFIX_HIT_MARGIN_TOKENS pads that delta for tokenizer/
# block-alignment slop at the cache boundary. USE_COMPUTED_COST is a KILL SWITCH, default OFF:
# est_computed/computed_actual are measured and logged from the moment this deploys (so the
# card's own 200-live-turn accuracy gate has something to grade), but admission cost keeps using
# the existing raw-token math until that gate is actually checked against real telemetry --
# shipping the measurement ahead of the behaviour change on a brand-new heuristic that touches
# live admission, not after it, per Kevin's own benchmark-rigor standard. Flipping it to 1 does
# NOT relax TOKEN_BUDGET (unchanged, still raw-prompt-token-based, still the real KV-safety net)
# -- it only changes how many LANES a correctly-predicted-cheap request is charged.
PREFIX_HIT_MARGIN_TOKENS = int(os.environ.get("SHIM_PREFIX_HIT_MARGIN_TOKENS", "512"))
USE_COMPUTED_COST = int(os.environ.get("SHIM_USE_COMPUTED_COST", "0"))
PREFIX_CACHE_MAX_CLIENTS = 200   # (legacy one-deep model bound; the model below is content-addressed)
# 2026-10-01 (LS lane, local-serving quick wins): CACHE-AWARE PREFILL COST MODEL.
# Measured today: Halo (58% of requests) resends a 40-120K-token prompt on every tool-loop turn, and
# ~95% of it is the previous turn's prompt. The engine's prefix cache serves that prefix when it
# was prefilled LOCALLY (probe: 50K prompt, 92% cached, TTFT 3.9s vs 48s cold), but the old
# one-deep per-CLIENT predictor lost the thread as soon as Halo interleaved >1 conversation, and it
# recorded requests that went REMOTE (which never warm the local cache). Result: every turn was
# costed at its raw size (a 58K turn = 5 lanes), lanes read "saturated" with 4-5 requests actually
# running and KV at ~40%, and Halo went to the paid remote while the engine's prefill queue was
# in fact the real constraint. The model below is content-addressed (a rolling hash over
# tools+messages, one node per message boundary), global (the cache is global), learns ONLY from
# requests actually admitted to the local engine, expires on a TTL, clears on engine restart and
# self-corrects from the engine's own usage.prompt_tokens_details.cached_tokens. The engine only
# caches whole attention blocks (3568 tokens on this build: "attention block size 3568"), so the
# credit is rounded DOWN to PREFIX_ALIGN_TOKENS. USE_COMPUTED_COST above decides whether
# admission/routing USES it (default off); prediction + telemetry always run.
PREFIX_ALIGN_TOKENS = int(os.environ.get("SHIM_PREFIX_ALIGN_TOKENS", "3568"))
PREFIX_MODEL_TTL_SECS = float(os.environ.get("SHIM_PREFIX_MODEL_TTL_SECS", "900"))
PREFIX_MODEL_MAX_NODES = int(os.environ.get("SHIM_PREFIX_MODEL_MAX_NODES", "60000"))
# [LANE CR 2026-10-03] Lost tool-loop continuations. All default OFF.
#  SHIM_CHAIN_TELEMETRY=1  log, per request, the route that last served this conversation's deepest known
#                          message prefix (chain_prev_route/_age_s) and the unshaved matched prefix
#                          (pm_match_tok): the exact "previous turn went remote" (cause b) signal.
#  SHIM_WARM_PRIORITY=1    a warm continuation (credit >= MIN_CREDIT, predicted uncached <= MAX_COMPUTED)
#                          is sent to vLLM with priority WARM_PRIORITY_VALUE (scheduling-policy priority) so it
#                          is admitted ahead of cold prefills (its remainder fits the step budget a running
#                          cold chunk leaves), and local-first judges it on queue + own prefill, not on the
#                          cold backlog it no longer waits behind. Keeps warm chains local instead of
#                          splitting them to remote under monster/prefill pressure.
#  SHIM_PREFIX_CREDIT_UNIT=N  round the cost model's credit to N tokens instead of whole attention blocks
#                          (set to the engine's --prefix-match-unit when it runs VLLM_GDN_TAIL_PUBLISH=copy).
CHAIN_TELEMETRY = _env_flag(os.environ.get("SHIM_CHAIN_TELEMETRY", "0"), False)
WARM_PRIORITY = _env_flag(os.environ.get("SHIM_WARM_PRIORITY", "0"), False)
WARM_PRIORITY_MIN_CREDIT = int(os.environ.get("SHIM_WARM_PRIORITY_MIN_CREDIT", "4096"))
WARM_PRIORITY_MAX_COMPUTED = int(os.environ.get("SHIM_WARM_PRIORITY_MAX_COMPUTED", "1700"))
WARM_PRIORITY_VALUE = int(os.environ.get("SHIM_WARM_PRIORITY_VALUE", "-10"))
PREFIX_CREDIT_UNIT = int(os.environ.get("SHIM_PREFIX_CREDIT_UNIT", "0"))
# Prefill is the engine's scarce resource (one cold 50K prompt = ~45s of chunk steps that every
# younger request waits behind). These express the guards in SECONDS OF PREFILL at PREFILL_TPS
# (so they follow the measured rate instead of a hard-coded token count):
#   monster       -- uncached prefill already in flight >= this many seconds: new arrivals go remote
#   heavy         -- a request whose OWN uncached prefill is >= this many seconds
#   heavy backlog -- a heavy request runs locally only if the in-flight backlog is <= this
MONSTER_PREFILL_SECS = float(os.environ.get("SHIM_MONSTER_PREFILL_SECS", "30"))
HEAVY_PREFILL_SECS = float(os.environ.get("SHIM_HEAVY_PREFILL_SECS", "20"))
HEAVY_ADMIT_BACKLOG_SECS = float(os.environ.get("SHIM_HEAVY_ADMIT_BACKLOG_SECS", "15"))
# PREFILL ADMISSION WINDOW (LS lane, 2026-10-02). Measured with the cost model live: the first six
# hours kept in-flight uncached prefill at 100-150 s (14 requests, avg ~12K tokens each -- none
# "heavy" on its own) and local TTFT p50 sat at 27-53 s; after an engine restart emptied the
# queue it was 7.5 s. vLLM prefills in arrival order, so the queue the shim lets in IS every later
# request's wait. A request is admitted to the local engine only when its own prefill seconds fit
# under PREFILL_ADMIT_SECS together with what is already prefilling -- or when it is LIGHT (a few
# seconds of prefill), which always fits. A request that does not fit waits for a lane like any
# other (local-wait), then overflows. 0 = off.
PREFILL_ADMIT_SECS = float(os.environ.get("SHIM_PREFILL_ADMIT_SECS", "45"))
LIGHT_PREFILL_SECS = float(os.environ.get("SHIM_LIGHT_PREFILL_SECS", "5"))
# 2026-09-05 (evalkit run-2 forensics): background classification by X-Client only matched the
# substrings "cron"/"batch", so the research service ("workflow-bg"), the research feeder and the
# local digester all ran as FOREGROUND and could fill every lane, queueing genuinely interactive
# turns behind robot busywork -- the exact thing FG_RESERVED exists to prevent. Substrings (case-
# insensitive) that mark an X-Client as background; hot-reloadable via /gateway/config.
BG_XCLIENTS = [m.lower() for m in _parse_seq(
    os.environ.get("SHIM_BG_XCLIENTS", "cron,batch,workflow-bg,research-feeder,digester"),
    _CFG_SEP["SHIM_BG_XCLIENTS"])]
BG_WAIT     = float(os.environ.get("SHIM_BG_WAIT_SECS", "5"))
# 2026-09-06 (background-remote-cost incident): an engine fault/restart used to fail EVERY
# in-flight request over to the paid remote provider the instant local_healthy() went False --
# including background traffic (cron/batch/research-feeder/digester) that can always wait for
# the next cycle. Measured 2026-09-05: ~$0.50 of remote spend per fault was PURELY background
# traffic riding out a 40s-4min local restart. When BG_LOCAL_ONLY is on (default), a background
# request that would otherwise overflow remote for a LOCAL-AVAILABILITY reason (engine down, or
# an operator's SHIM_FORCE_REMOTE maintenance window) is HELD (poll local health, serve local
# once it recovers) or REJECTED (503 + Retry-After) instead of ever reaching the remote
# provider -- see the "MASTER SWITCH" and "local DOWN" branches in _route_completions().
# Legitimate SIZE-based remote routing (size/big-out/big-prompt/monster) is deliberately
# untouched: those requests can't run local regardless of engine health, remote or not, and
# interactive traffic's behavior is completely unchanged either way.
BG_LOCAL_ONLY    = _env_flag(os.environ.get("SHIM_BG_LOCAL_ONLY", "1"), True)
BG_WAIT_LOCAL    = float(os.environ.get("SHIM_BG_WAIT_LOCAL_SECS", "120"))
BG_REJECT_RETRY_SECS = int(os.environ.get("SHIM_BG_REJECT_RETRY_SECS", "300"))
# Background turns don't need chain-of-thought: thinking mode makes a cron status report
# generate 2-3K reasoning tokens and hold a local lane for minutes. Injecting
# enable_thinking=false for LOCAL background requests cuts their lane-hold ~10x.
BG_NO_THINK = _env_flag(os.environ.get("SHIM_BG_NO_THINK", "1"), True)
# thinking-budget guard: below THINK_OFF_UNDER tokens disable thinking entirely, below
# THINK_LOW_UNDER downgrade to reasoning_effort=low. Measured starvation point ~1315 tok.
THINK_GUARD     = _env_flag(os.environ.get("SHIM_THINK_GUARD", "1"), True)
EMPTY_RETRY     = _env_flag(os.environ.get("SHIM_EMPTY_RETRY", "1"), True)
REP_GUARD       = _env_flag(os.environ.get("SHIM_REP_GUARD", "1"), True)
REP_MIN_PATTERN = int(os.environ.get("SHIM_REP_MIN_PATTERN", "8"))
REP_MAX_PATTERN = int(os.environ.get("SHIM_REP_MAX_PATTERN", "64"))
REP_MIN_COUNT   = int(os.environ.get("SHIM_REP_MIN_COUNT", "6"))
# [LANE RA / L143] cross-turn identical tool-call loop note (see repeated_tool_call_note()).
REPEAT_NOTE     = os.environ.get("SHIM_REPEAT_NOTE", "1") not in ("0", "false", "")
REPEAT_NOTE_MIN = int(os.environ.get("SHIM_REPEAT_NOTE_MIN", "3"))
THINK_BUDGET_FRAC = float(os.environ.get("SHIM_THINK_BUDGET_FRAC", "0.5"))
THINK_BUDGET_MIN  = int(os.environ.get("SHIM_THINK_BUDGET_MIN", "128"))
THINK_BUDGET_MAX  = int(os.environ.get("SHIM_THINK_BUDGET_MAX", "4096"))
THINK_OFF_UNDER = int(os.environ.get("SHIM_THINK_OFF_UNDER", "600"))
THINK_LOW_UNDER = int(os.environ.get("SHIM_THINK_LOW_UNDER", "1400"))
# --- non-thinking sampling profile (EXP-026, 2026-08-16) ---
# Qwen3's official NON-THINKING sampling profile fixes long-generation repetition loops
# (measured EXP-026: loop_probe 5/10->8/10, dupes 5->1) that our gencfg default (temp 0.6,
# top_p 0.95, pp 0) falls into. Injected for LOCAL non-thinking requests ONLY (chat_template_kwargs
# .enable_thinking=False); THINKING-mode requests must keep presence_penalty=0 per Qwen's official
# guidance, so this never touches them. Each param is applied only when the caller did NOT already
# provide it -- an explicit caller value always wins. Disable with SHIM_NONTHINK_PROFILE=0; tune the
# individual values via SHIM_NONTHINK_{PP,TOP_P,TEMP,TOP_K}.
NONTHINK_PROFILE = _env_flag(os.environ.get("SHIM_NONTHINK_PROFILE", "1"), True)
NONTHINK_PP      = float(os.environ.get("SHIM_NONTHINK_PP", "1.5"))
NONTHINK_TOP_P   = float(os.environ.get("SHIM_NONTHINK_TOP_P", "0.80"))
NONTHINK_TEMP    = float(os.environ.get("SHIM_NONTHINK_TEMP", "0.7"))
NONTHINK_TOP_K   = int(os.environ.get("SHIM_NONTHINK_TOP_K", "20"))
# MASTER SWITCH: 1 = FULL REMOTE (every completion -> DeepSeek; local engine untouched —
# for maintenance/repro/debugging), 0 = normal local-first. Toggle live from the dashboard.
FORCE_REMOTE = 1 if _env_flag(os.environ.get("SHIM_FORCE_REMOTE", "0"), False) else 0
# A paid full-remote maintenance mode is a lease, not a permanent routing state.
# An old persisted FORCE_REMOTE=1 without a lease is inert after an upgrade.
FORCE_REMOTE_LEASE_S = 3600
try:
    FORCE_REMOTE_UNTIL_EPOCH = float(os.environ.get("SHIM_FORCE_REMOTE_UNTIL_EPOCH", "0"))
except ValueError:
    FORCE_REMOTE_UNTIL_EPOCH = 0.0


def effective_force_remote():
    return bool(FORCE_REMOTE and time.time() < FORCE_REMOTE_UNTIL_EPOCH)
# The third mode (Kevin 2026-09-10: "LOCAL FIRST // FULL REMOTE // FULL LOCAL"):
# 1 = FULL LOCAL — never overflow to the paid remote for ANY request; when every local lane is
# busy a request QUEUES for a lane instead of spending money ($0 guaranteed, latency unbounded).
# Better than unsetting the remote credentials because it is reversible from the dashboard and
# leaves the remote configured for the moment it is wanted again. It OUTRANKS FORCE_REMOTE: if
# both are somehow set, local wins, because the mode that cannot spend money is the safe one.
LOCAL_ONLY = 1 if _env_flag(os.environ.get("SHIM_LOCAL_ONLY", "0"), False) else 0
# PEAK-AWARE overflow bias (2026-08-13, DeepSeek peak/off-peak pricing eff. Aug 16):
# during remote-provider PEAK hours (UTC ranges like "1-4,6-10"), BACKGROUND requests
# wait the full LOCAL_WAIT for a local lane instead of fast-overflowing at BG_WAIT —
# biasing robot busywork away from 2x-priced remote. Interactive routing unchanged.
PEAK_HOURS = os.environ.get("SHIM_PEAK_HOURS_UTC", "1-4,6-10")

def is_peak(ts=None):
    """Is `ts` (epoch seconds, default now) inside the provider's PEAK pricing window?
    The schedule comes from the dated pricing table (remote-pricing.json: UTC windows, weekdays
    only, Chinese public holidays off-peak -- see _pricing_is_peak); SHIM_PEAK_HOURS_UTC is only
    the fallback when no table can be loaded."""
    try:
        return _pricing_is_peak(ts)
    except Exception:
        pass
    try:
        utc = time.gmtime(ts)
        if utc.tm_wday >= 5:
            return False
        h = utc.tm_hour
        for part in PEAK_HOURS.split(","):
            a, b = (part.split("-") + [part])[:2]
            if int(a) <= h < int(b):
                return True
    except Exception:
        pass
    return False
# Also strip thinking for LOCAL requests from these client IPs (comma-separated; e.g. the
# Hermes boxes, whose 1-3K-token chain-of-thought per turn is the user-felt latency).
NO_THINK_IPS = set(_parse_seq(os.environ.get("SHIM_NO_THINK_IPS", ""), _CFG_SEP["SHIM_NO_THINK_IPS"]))


def _parse_budget(v):
    """SHIM_TOKEN_BUDGET: a positive integer is an EXPLICIT override, 0 disables the cap, and ''/auto/None means
    "derive it from the live engine" (see token_budget_info)."""
    if v is None or str(v).strip().lower() in ("", "auto", "none", "live"):
        return None
    return int(float(v))


# ---------------- live config (editable from the dashboard, no restart) ----------------
# env-var name -> (global name, caster). Only these are runtime-tunable.
_CFG = {
    # remote overflow provider (any OpenAI-compatible endpoint)
    "SHIM_REMOTE_BASE":      ("REMOTE_BASE",  lambda v: str(v).rstrip("/")),
    "SHIM_REMOTE_KEY":       ("REMOTE_KEY",   str),
    "SHIM_REMOTE_MODEL":     ("REMOTE_MODEL", str),
    "SHIM_LOCAL_CONTEXT_LIMIT": ("LOCAL_CONTEXT_LIMIT", int),
    "SHIM_REMOTE_CONTEXT_LIMIT": ("REMOTE_CONTEXT_LIMIT", int),
    "SHIM_CONTEXT_SAFETY_MARGIN": ("CONTEXT_SAFETY_MARGIN", int),
    "SHIM_CONTEXT_COMPACTION": ("CONTEXT_COMPACTION_ENABLED", lambda v: _cfg_flag(v)),
    "SHIM_CONTEXT_COMPACTION_KEEP": ("CONTEXT_COMPACTION_KEEP", int),
    "SHIM_FORCE_REMOTE":     ("FORCE_REMOTE", lambda v: 1 if _cfg_flag(v) else 0),
    "SHIM_LOCAL_ONLY":       ("LOCAL_ONLY",   lambda v: 1 if _cfg_flag(v) else 0),
    # local capacity
    "SHIM_LOCAL_BUDGET":     ("BUDGET",       int),
    "SHIM_LOCAL_WAIT_SECS":  ("LOCAL_WAIT",   float),
    "SHIM_TOKEN_BUDGET":     ("TOKEN_BUDGET", _parse_budget),
    "SHIM_OOM_BACKOFF_SECS": ("OOM_BACKOFF",  int),
    # routing guards
    "SHIM_BIG_TOKENS":       ("BIG_TOKENS",       int),
    "SHIM_BIG_OUTPUT":       ("BIG_OUTPUT",       int),
    "SHIM_BIG_PROMPT":       ("BIG_PROMPT",       int),
    "SHIM_MONSTER_INFLIGHT": ("MONSTER_INFLIGHT", int),
    "SHIM_MAX_LOCAL_TOKENS": ("MAX_LOCAL_TOKENS", int),
    "SHIM_LOCAL_MAX_OUT":    ("LOCAL_MAX_OUT",    int),
    "SHIM_FIRST_TOKEN_MAX":  ("FIRST_TOKEN_MAX",  float),
    "SHIM_PREFILL_TPS":      ("PREFILL_TPS",      float),
    # lanes / priority
    "SHIM_TINY_TOKENS":      ("TINY_TOKENS",      int),
    "SHIM_TINY_EXTRA_LANES": ("TINY_EXTRA_LANES", int),
    "SHIM_FG_RESERVED":      ("FG_RESERVED",      int),
    "SHIM_BG_BIG_LOCAL_WHEN_IDLE": ("BG_BIG_LOCAL_WHEN_IDLE", int),
    "SHIM_TOKENS_PER_UNIT":  ("TOKENS_PER_UNIT", int),
    "SHIM_INTERACTIVE_NEVER_OVERFLOW": ("INTERACTIVE_NEVER_OVERFLOW", int),
    "SHIM_PREFIX_HIT_MARGIN_TOKENS": ("PREFIX_HIT_MARGIN_TOKENS", int),
    "SHIM_USE_COMPUTED_COST": ("USE_COMPUTED_COST", int),
    "SHIM_PREFIX_ALIGN_TOKENS": ("PREFIX_ALIGN_TOKENS", int),
    "SHIM_PREFIX_MODEL_TTL_SECS": ("PREFIX_MODEL_TTL_SECS", float),
    "SHIM_MONSTER_PREFILL_SECS": ("MONSTER_PREFILL_SECS", float),
    "SHIM_HEAVY_PREFILL_SECS": ("HEAVY_PREFILL_SECS", float),
    "SHIM_HEAVY_ADMIT_BACKLOG_SECS": ("HEAVY_ADMIT_BACKLOG_SECS", float),
    "SHIM_PREFILL_ADMIT_SECS": ("PREFILL_ADMIT_SECS", float),
    "SHIM_LIGHT_PREFILL_SECS": ("LIGHT_PREFILL_SECS", float),
    "SHIM_BG_XCLIENTS":      ("BG_XCLIENTS", lambda v: [m.lower() for m in _parse_seq(v, _CFG_SEP["SHIM_BG_XCLIENTS"])]),
    "SHIM_POOL_TOKENS":      ("POOL_TOKENS",      int),
    # live capacity model (lane GW, 2026-10-02) -- see the CAPACITY MODEL section
    "SHIM_CAPACITY_LIVE":    ("CAPACITY_LIVE",    lambda v: _cfg_flag(v)),
    "SHIM_CONTEXT_LIVE":     ("CONTEXT_LIVE",     lambda v: _cfg_flag(v)),
    "SHIM_MODELS_POLL_SECS": ("MODELS_POLL_S",    float),
    "SHIM_TOKEN_BUDGET_FRAC": ("TOKEN_BUDGET_FRAC", float),
    "SHIM_TOKEN_BUDGET_CEIL": ("TOKEN_BUDGET_CEIL", int),
    "SHIM_LOCAL_MODALITIES": ("LOCAL_MODALITIES", lambda v: ",".join(sorted({x.strip().lower() for x in str(v).split(",") if x.strip()})) or "text"),
    "SHIM_REMOTE_VISION":    ("REMOTE_VISION", lambda v: 1 if _cfg_flag(v) else 0),
    "SHIM_BG_WAIT_SECS":     ("BG_WAIT",          float),
    "SHIM_BG_MARKERS":       ("BG_MARKERS", lambda v: [m for m in str(v).split("|") if m]),
    "SHIM_BG_LOCAL_ONLY":    ("BG_LOCAL_ONLY", lambda v: _cfg_flag(v)),
    "SHIM_BG_WAIT_LOCAL_SECS": ("BG_WAIT_LOCAL", float),
    "SHIM_BG_REJECT_RETRY_SECS": ("BG_REJECT_RETRY_SECS", int),
    "SHIM_PEAK_HOURS_UTC":   ("PEAK_HOURS",       str),
    # behaviour toggles
    "SHIM_BG_NO_THINK":      ("BG_NO_THINK",  lambda v: _cfg_flag(v)),
    "SHIM_THINK_GUARD":      ("THINK_GUARD",  lambda v: _cfg_flag(v)),
    "SHIM_EMPTY_RETRY":      ("EMPTY_RETRY",  lambda v: _cfg_flag(v)),
    "SHIM_REP_GUARD":        ("REP_GUARD",    lambda v: _cfg_flag(v)),
    "SHIM_REP_MIN_PATTERN":  ("REP_MIN_PATTERN", int),
    "SHIM_REP_MAX_PATTERN":  ("REP_MAX_PATTERN", int),
    "SHIM_REP_MIN_COUNT":    ("REP_MIN_COUNT", int),
    "SHIM_REPEAT_NOTE":      ("REPEAT_NOTE",  lambda v: str(v).lower() not in ("0","false","")),
    "SHIM_REPEAT_NOTE_MIN":  ("REPEAT_NOTE_MIN", int),
    "SHIM_THINK_BUDGET_FRAC":("THINK_BUDGET_FRAC", float),
    "SHIM_THINK_BUDGET_MIN": ("THINK_BUDGET_MIN", int),
    "SHIM_THINK_BUDGET_MAX": ("THINK_BUDGET_MAX", int),
    "SHIM_THINK_OFF_UNDER":  ("THINK_OFF_UNDER", int),
    "SHIM_THINK_LOW_UNDER":  ("THINK_LOW_UNDER", int),
    "SHIM_NO_THINK_IPS":     ("NO_THINK_IPS", lambda v: set(_parse_seq(v, _CFG_SEP["SHIM_NO_THINK_IPS"]))),
    "SHIM_LOG_REQUESTS":     ("LOG_REQUESTS", lambda v: _cfg_flag(v)),
    # non-thinking sampling profile (EXP-026)
    "SHIM_NONTHINK_PROFILE": ("NONTHINK_PROFILE", lambda v: _cfg_flag(v)),
    "SHIM_NONTHINK_PP":      ("NONTHINK_PP",    float),
    "SHIM_NONTHINK_TOP_P":   ("NONTHINK_TOP_P", float),
    "SHIM_NONTHINK_TEMP":    ("NONTHINK_TEMP",  float),
    "SHIM_NONTHINK_TOP_K":   ("NONTHINK_TOP_K", int),
    "SHIM_PREDICTED_OCCUPANCY_SECS": ("PREDICTED_OCCUPANCY_SECS", float),
    "SHIM_DECODE_TPS_FLOOR": ("DECODE_TPS_FLOOR", float),
    "SHIM_PERF_BREAKER_ENABLED": ("PERF_BREAKER_ENABLED", lambda v: _cfg_flag(v)),
    "SHIM_PERF_BREAKER_TTFT_P95_SECS": ("PERF_BREAKER_TTFT_P95_SECS", float),
    "SHIM_PERF_BREAKER_GPU_UTIL_PCT": ("PERF_BREAKER_GPU_UTIL_PCT", float),
    "SHIM_PERF_BREAKER_KV_PCT": ("PERF_BREAKER_KV_PCT", float),
    "SHIM_PERF_BREAKER_MIN_SAMPLES": ("PERF_BREAKER_MIN_SAMPLES", int),
    "SHIM_PERF_BREAKER_HOLD_SECS": ("PERF_BREAKER_HOLD_SECS", float),
    "SHIM_STREAM_IDLE_TIMEOUT_SECS": ("STREAM_IDLE_TIMEOUT_SECS", float),
    # [GW2 / L100] Lane CR knobs, hot-reloadable so the warm-priority A/B can alternate arms without restarts
    "SHIM_CHAIN_TELEMETRY": ("CHAIN_TELEMETRY", lambda v: _cfg_flag(v)),
    "SHIM_WARM_PRIORITY": ("WARM_PRIORITY", lambda v: _cfg_flag(v)),
    "SHIM_WARM_PRIORITY_MIN_CREDIT": ("WARM_PRIORITY_MIN_CREDIT", int),
    "SHIM_WARM_PRIORITY_MAX_COMPUTED": ("WARM_PRIORITY_MAX_COMPUTED", int),
    "SHIM_REASONING_WATCHDOG": ("REASONING_WATCHDOG", lambda v: str(v).strip().lower()),
    "SHIM_REASONING_BUDGET_TOKENS": ("REASONING_BUDGET_TOKENS", int),
    "SHIM_REASONING_CHARS_PER_TOKEN": ("REASONING_CHARS_PER_TOKEN", float),
    "SHIM_THINK_BUDGET_EXPLICIT": ("THINK_BUDGET_EXPLICIT", int),
    "SHIM_CREDIT_ANCHOR": ("CREDIT_ANCHOR", lambda v: str(v).strip().lower()),
    "SHIM_CREDIT_PROBE": ("CREDIT_PROBE", lambda v: str(v).strip().lower()),
    "SHIM_CREDIT_PROBE_TIMEOUT_S": ("CREDIT_PROBE_TIMEOUT_S", float),
    "SHIM_CREDIT_PROBE_MIN_TOKENS": ("CREDIT_PROBE_MIN_TOKENS", int),
    # LOCAL-FIRST (L1, 2026-09-25) -- see the block after STREAM_IDLE_TIMEOUT_SECS below.
    "SHIM_LOCAL_FIRST":      ("LOCAL_FIRST", lambda v: _cfg_flag(v)),
    "SHIM_LOCAL_FIRST_REASONS": ("LOCAL_FIRST_REASONS", lambda v: _parse_reason_set(v)),
    "SHIM_LOCAL_FIRST_QUEUE_WAIT_SECS": ("LOCAL_FIRST_QUEUE_WAIT_SECS", float),
    "SHIM_LOCAL_FIRST_WAIT_WINDOW_SECS": ("LOCAL_FIRST_WAIT_WINDOW_SECS", float),
    "SHIM_LOCAL_FIRST_FIRST_TOKEN_MAX": ("LOCAL_FIRST_FIRST_TOKEN_MAX", float),
    "SHIM_LOCAL_FIRST_INTERACTIVE_TTFT_SECS": ("LOCAL_FIRST_INTERACTIVE_TTFT_SECS", float),
    # LF (10-03): deadline-derived local-first + expected-output big-out rule (see the LF block below).
    "SHIM_LOCAL_FIRST_DERIVE": ("LOCAL_FIRST_DERIVE", lambda v: _cfg_flag(v)),
    "SHIM_EXPECTED_OUTPUT":  ("EXPECTED_OUTPUT", lambda v: _cfg_flag(v)),
    "SHIM_EXPECTED_OUTPUT_QUANTILE": ("EXPECTED_OUTPUT_QUANTILE", float),
    "SHIM_EXPECTED_OUTPUT_MIN_SAMPLES": ("EXPECTED_OUTPUT_MIN_SAMPLES", int),
}
# A single request whose est. (prompt + max_tokens) exceeds this will OOM local even at
# budget=1 (context + generation peaks past this box's tiny free VRAM), so route it straight
# to remote. Prevents single-big-request OOM crashes. (2026-08-12: observed a solo OOM here.)
MAX_LOCAL_TOKENS = int(os.environ.get("SHIM_MAX_LOCAL_TOKENS", "80000"))
# Size-aware admission: cap reserved prompt + bounded output tokens across local lanes.
# Prefix-cache compute savings do not reduce this conservative KV reservation.
# Historical prompt-only calibration (activation ∝
# concurrent context). Benchmark (2026-08-12, util 0.82) held 4x170K=680K with 611MB margin;
# 500K default leaves comfortable headroom while allowing generous concurrency. 0 = disabled.
# 2026-10-02 (lane GW): the token budget is a DECLARED FRACTION OF THE LIVE KV POOL, not a number frozen against an old
# engine. TOKEN_BUDGET stays as an optional explicit override (int) or None = auto.
TOKEN_BUDGET     = _parse_budget(os.environ.get("SHIM_TOKEN_BUDGET"))
# CAPACITY MODEL (lane GW, 2026-10-02). The gateway's capacity numbers are FACTS OF THE RUNNING ENGINE, so they are read
# from it (see the "CAPACITY MODEL" section: pool_info / token_budget_info / prefill_info) and the configured values
# below are only fallbacks / optional overrides. SHIM_CAPACITY_LIVE=0 restores configured-only behaviour (kill switch).
CAPACITY_LIVE = _env_flag(os.environ.get("SHIM_CAPACITY_LIVE", "1"), True)
# The context window the gateway admits against (SHIM_LOCAL_CONTEXT_LIMIT, SHIM_MAX_LOCAL_TOKENS) is the engine's own
# max_model_len, read from its /v1/models every MODELS_POLL_S (and at once after an engine restart); the configured values are
# only the fallback until the engine has answered. SHIM_CONTEXT_LIVE=0 (or SHIM_CAPACITY_LIVE=0) = configured-only (kill switch).
CONTEXT_LIVE = _env_flag(os.environ.get("SHIM_CONTEXT_LIVE", "1"), True)
MODELS_POLL_S = float(os.environ.get("SHIM_MODELS_POLL_SECS", "30"))
# The token budget as a fraction of the KV pool. CALIBRATION (the only hand-set capacity datum, with its provenance):
# 500,000 reserved tokens was set 2026-08-12 from the breaking-point bench (4 x 170K = 680K held with 611 MB VRAM margin at
# util 0.82) and has run in production ever since, on the 637,560-token pool and (since the 2026-09-05 R4 promotion) the
# 754,068-token pool. The LARGEST pool it was proven against sets the fraction (500,000 / 754,068 = 0.663): a bigger pool
# earns a bigger budget only in proportion, never more than that.
TOKEN_BUDGET_CALIBRATION = (500000, 754068)       # (reserved tokens proven safe, KV pool they were proven against)
TOKEN_BUDGET_FRAC = float(os.environ.get("SHIM_TOKEN_BUDGET_FRAC") or 0) or (TOKEN_BUDGET_CALIBRATION[0] / TOKEN_BUDGET_CALIBRATION[1])
# Absolute ceiling = the largest in-flight context the bench actually HELD (680K). The KV pool is not the only memory that
# grows with concurrent context (activations do), so a larger pool alone must not raise the budget past what was exercised;
# raise this only after a new breaking-point bench. 0 = no ceiling.
TOKEN_BUDGET_CEIL = int(os.environ.get("SHIM_TOKEN_BUDGET_CEIL", "680000"))
# MODALITIES (lane GW, 2026-10-02). The default engine stack serves --language-model-only: an image/video/audio part gets
# HTTP 400 "At most 0 image(s)" from the engine, and before this guard the gateway passed that bare 400 to the caller
# on every local alias. LOCAL_MODALITIES declares what the local engine accepts ("text"; add "image" when a vision stack
# is served); REMOTE_VISION=1 declares that the default remote provider accepts images (DeepSeek's chat API is text-only,
# so the default is 0). A media request the local engine cannot take goes to the remote provider when it can take it and
# the hard daily cap has room, otherwise it gets one clear 400 -- never the engine's bare error.
LOCAL_MODALITIES = ",".join(sorted({x.strip().lower() for x in os.environ.get("SHIM_LOCAL_MODALITIES", "text").split(",") if x.strip()})) or "text"
REMOTE_VISION = 1 if _env_flag(os.environ.get("SHIM_REMOTE_VISION", "0"), False) else 0
_MEDIA_PART_TYPES = {"image_url": "image", "input_image": "image", "image": "image", "video_url": "video", "input_video": "video",
                     "video": "video", "input_audio": "audio", "audio_url": "audio", "audio": "audio"}
DEFAULT_MAX_OUT  = int(os.environ.get("SHIM_DEFAULT_MAX_OUT", "8192"))
# --- TINY fast-lane (2026-08-13) ---
# Micro-calls (title-gen, classification, keepalive probes: est. prompt+max_out <= TINY_TOKENS)
# are VRAM-negligible. Give them a fast-lane: they SKIP the queue-first wait and may use up to
# TINY_EXTRA_LANES slots BEYOND the big-request budget (staying within the engine's max-num-seqs),
# so a trivial call never eats a 15s wait or gets starved behind big generations / during a
# budget=1 backoff. If local is busy past that headroom, they fast-overflow to remote immediately
# (a tiny call on DeepSeek is near-free + fast) rather than waiting.
TINY_TOKENS       = int(os.environ.get("SHIM_TINY_TOKENS", "1500"))
TINY_EXTRA_LANES  = int(os.environ.get("SHIM_TINY_EXTRA_LANES", "2"))
# --- MICRO fast lane v2 (lane K4, 2026-10-03) ---
# Measured on 7 days of gateway telemetry (40.8k local completions): the static is_tiny() rule (prompt + max_tokens
# <= TINY_TOKENS) catches 5.2% of requests, but a further ~10% finish in <=64 output tokens and queue in the full
# lanes because their max_tokens is unset or large (research-service fact-checks, Halo pong probes, JSON triage).
# MICRO_LEARN lets the gateway LEARN that a (client, prompt-head) signature is short-output and admit it to the tiny
# lane. Unlike static-tiny, a learned-tiny request that finds the tiny lanes full WAITS locally (never overflows to
# paid remote). Kill switch: SHIM_MICRO_LEARN=0.
MICRO_LEARN       = _env_flag(os.environ.get("SHIM_MICRO_LEARN", "1"), True)
MICRO_SIG_CHARS   = int(os.environ.get("SHIM_MICRO_SIG_CHARS", "32"))      # normalized prompt-head chars in a signature
MICRO_MIN_SAMPLES = int(os.environ.get("SHIM_MICRO_MIN_SAMPLES", "6"))     # completions seen before a signature qualifies
MICRO_HISTORY     = int(os.environ.get("SHIM_MICRO_HISTORY", "16"))        # most recent completions kept per signature
MICRO_MAX_OUT     = int(os.environ.get("SHIM_MICRO_MAX_OUT", "96"))        # EVERY recent completion must be <= this
MICRO_MAX_PROMPT  = int(os.environ.get("SHIM_MICRO_MAX_PROMPT", "6000"))   # prompts larger than this are never micro
MICRO_MAX_SIGS    = 2000
# --- PROBE synth (lane K4): Halo's front-door "Reply with the single word: pong" liveness probe fires every ~5 min,
# carries 3-35k tokens of injected session context (2.9% of all uncached prefill in a week, 0.8% of requests) and
# 391 of 1558 probes in that week were billed to the paid remote. When the engine has produced real tokens within
# PROBE_FRESH_S, the gateway answers the probe itself; every PROBE_REAL_EVERY-th probe still goes through for real
# so end-to-end generation stays proven. Only local-capable aliases (never estate-remote). Kill: SHIM_PROBE_SYNTH=0.
PROBE_SYNTH       = _env_flag(os.environ.get("SHIM_PROBE_SYNTH", "1"), True)
PROBE_FRESH_S     = float(os.environ.get("SHIM_PROBE_FRESH_S", "120"))
PROBE_REAL_EVERY  = int(os.environ.get("SHIM_PROBE_REAL_EVERY", "12"))
# --- concurrency-aware first-token deadline (2026-08-13) ---
# Prefill compute is SHARED across concurrent requests on this box, so a big request queued behind
# N others emits its first token only after ~N prefills complete. Scaling the wedge-detection
# deadline by in-flight concurrency stops legit slow prefills being misread as a wedged backend
# (which was needlessly failing big requests over to DeepSeek AND tripping a 120s budget=1 backoff
# that cascaded tiny requests to overflow). 1 = scale by concurrency (default); 0 = old flat behavior.
FT_CONCURRENCY_SCALE = int(os.environ.get("SHIM_FT_CONCURRENCY_SCALE", "1"))
# --- per-request logging (2026-08-13) ---
# Log source (ip/UA), model, size and a short prompt preview for each completion, to attribute
# traffic (which client fires the tiny bursts / the slow big prefills). Set SHIM_LOG_REQUESTS=0
# to disable (e.g. for prompt privacy).
LOG_REQUESTS      = _env_flag(os.environ.get("SHIM_LOG_REQUESTS", "1"), True)
LOG_PREVIEW_CHARS = int(os.environ.get("SHIM_LOG_PREVIEW_CHARS", "70"))
# vLLM-only params that a remote OpenAI endpoint would reject — stripped on overflow.
REMOTE_STRIP = ("chat_template_kwargs", "mamba_cache_mode", "guided_decoding_backend")
# Force non-thinking on the DeepSeek failover (see remap_for_remote). Disable with SHIM_REMOTE_NO_THINK=0.
REMOTE_NO_THINK = _env_flag(os.environ.get("SHIM_REMOTE_NO_THINK", "1"), True)

# Congestion controls.  These guards run inside the gateway before local admission.  Callers
# never select a provider; they may only express an optional route intent.
PREDICTED_OCCUPANCY_SECS = float(os.environ.get("SHIM_PREDICTED_OCCUPANCY_SECS", "180"))
DECODE_TPS_FLOOR = float(os.environ.get("SHIM_DECODE_TPS_FLOOR", "20"))
PERF_BREAKER_ENABLED = _env_flag(os.environ.get("SHIM_PERF_BREAKER_ENABLED", "1"), True)
PERF_BREAKER_TTFT_P95_SECS = float(os.environ.get("SHIM_PERF_BREAKER_TTFT_P95_SECS", "20"))
PERF_BREAKER_GPU_UTIL_PCT = float(os.environ.get("SHIM_PERF_BREAKER_GPU_UTIL_PCT", "95"))
PERF_BREAKER_KV_PCT = float(os.environ.get("SHIM_PERF_BREAKER_KV_PCT", "85"))
PERF_BREAKER_MIN_SAMPLES = int(os.environ.get("SHIM_PERF_BREAKER_MIN_SAMPLES", "3"))
PERF_BREAKER_HOLD_SECS = float(os.environ.get("SHIM_PERF_BREAKER_HOLD_SECS", "45"))
STREAM_IDLE_TIMEOUT_SECS = float(os.environ.get("SHIM_STREAM_IDLE_TIMEOUT_SECS", "45"))
# [LANE GW2 / L95, 2026-10-03] Long-empty-generation watchdog. Measured: pi sends
# chat_template_kwargs.enable_thinking=true (pi models.json thinkingFormat=qwen-chat-template), which
# thinking_budget_guard() reads as "caller handled thinking" and so leaves UNBOUNDED; with max_tokens
# clamped to LOCAL_MAX_OUT=16384 a turn can think for 16384 tokens (~18 min at contended decode) and return
# nothing (3 such pi turns 2026-10-03 05:21-05:23, 529-1076 s). Two independent, default-OFF fixes:
#  SHIM_REASONING_WATCHDOG=off|shadow|retry  committed LOCAL stream that has streamed >= BUDGET estimated
#      reasoning tokens with no content and no tool call yet. shadow: log + count what retry would have done.
#      retry: abort the upstream generation and re-issue the same request with thinking OFF, piping its answer
#      into the SAME client stream (the client has only seen reasoning deltas so far, so this is seamless).
#  SHIM_REASONING_BUDGET_TOKENS  the budget, in tokens estimated as reasoning chars / SHIM_REASONING_CHARS_PER_TOKEN.
#  SHIM_THINK_BUDGET_EXPLICIT=N  (0=off) the sampler-level source fix: an explicit enable_thinking=true without a
#      thinking_token_budget gets thinking_token_budget=N (vLLM force-closes </think> at N -- no wasted work).
REASONING_WATCHDOG = os.environ.get("SHIM_REASONING_WATCHDOG", "off").strip().lower()
REASONING_BUDGET_TOKENS = int(os.environ.get("SHIM_REASONING_BUDGET_TOKENS", "8192"))
REASONING_CHARS_PER_TOKEN = float(os.environ.get("SHIM_REASONING_CHARS_PER_TOKEN", "4.0"))
THINK_BUDGET_EXPLICIT = int(os.environ.get("SHIM_THINK_BUDGET_EXPLICIT", "0"))
# [LANE GW2 2026-10-03] Engine-anchored cache credit, SHADOW only. The cost model's credit is a guess (matched chars /
# total chars x the ESTIMATED prompt, shaved by trust, block-rounded); measured over 7,687 local rows it under-credits by
# up to 14.3K tokens at p05 and over-credits by 3.6K at p95, and that error -- not the prompt-size estimate -- drives the
# light/heavy/big-prompt threshold flips. SHIM_CREDIT_ANCHOR=shadow remembers, per conversation chain end, the ENGINE's
# own usage.prompt_tokens of the finished local request, and logs pm_credit_anchored = that exact prefix rounded down to
# the engine's cache block for the next turn of the chain. Routing still uses pm_credit; the log sizes the switch.
CREDIT_ANCHOR = os.environ.get("SHIM_CREDIT_ANCHOR", "off").strip().lower()
# [LANE GW2 / L105] Engine-reported cache credit. With the engine booted with VLLM_FORK_PREFIX_PROBE=1 (fork branch
# lane-gw2-engine), POST /v1/fork/prefix_cache_probe answers from the engine's own block table: exact prompt_tokens
# (same render as /tokenize) and the prefix-cache hit an admission would get right now.
#  SHIM_CREDIT_PROBE=off|shadow|live   shadow logs pm_credit_probe beside pm_credit; live routes on it (credit and
#      uncached computed = probe prompt_tokens - cached), falling back to the anchored credit (SHIM_CREDIT_ANCHOR=live)
#      and then to the model. A probe that is late, fails or is absent never delays routing past the timeout.
CREDIT_PROBE = os.environ.get("SHIM_CREDIT_PROBE", "off").strip().lower()
CREDIT_PROBE_TIMEOUT_S = float(os.environ.get("SHIM_CREDIT_PROBE_TIMEOUT_S", "0.3"))
CREDIT_PROBE_MIN_TOKENS = int(os.environ.get("SHIM_CREDIT_PROBE_MIN_TOKENS", "4096"))

# ---------------- LOCAL-FIRST overflow policy (L1, 2026-09-25) ----------------
# Kevin 2026-09-25: "Is the GPU being used? Otherwise ... we're wasting money and/or time
# without using the local model." Measured from the gateway's own request log (24h to 13:20
# MST): 10,211 remote routes, and at the start of EVERY one of the capacity/latency-predictive
# ones (big-prompt 4,500, perf 3,556, monster 24, big-out 22, predicted 11) the local engine had
# free lanes and free token budget (reconstructed lane occupancy: mean 1.3 of 14, >=7 lanes
# only 3.2% of the time, never 14). The engine was idle ~50% of wall time while they went out.
#   * big-prompt fired at a fixed 24K-token threshold that predates prefix caching: the gateway's
#     own predictor put the p50 COMPUTED (uncached) prefill of those prompts at 2,334 tokens
#     (5% of the prompt) -- incremental pi turns whose prefix local already holds.
#   * perf fired on "gpu-saturation" 60% of the time: nvidia-smi util is a duty cycle, ~88% with
#     ONE stream running, so avg>=95% with running>=2 is "the GPU is doing work", not "full".
#     Local requests admitted while it was active: TTFT p95 18.4s vs 10.4s, >20s for 4.4%;
#     interactive local TTFT over the whole day never exceeded 20.05s (1 of 2,244).
# So these reasons now PREDICT; they no longer DECIDE. A request that one of them would have
# sent remote stays LOCAL unless local is measurably saturated right now:
#   lanes      -- this class's lanes are full (inflight + units > lane limit),
#   tokens     -- the KV token budget cannot fit its reservation,
#   queued     -- requests of this class are already waiting for a lane,
#   queue-wait -- the mean admission wait over the last LOCAL_FIRST_WAIT_WINDOW_SECS reached
#                 LOCAL_FIRST_QUEUE_WAIT_SECS.
# Kept requests then go through normal admission, which still overflows ("cap"/"tokens"/
# "bg-yield") if lanes do not free within the class's wait -- the saturation fallback.
# UNCHANGED: explicit remote intents (estate-remote/custom aliases, route-intent, the forced
# window), safety routes (local-down, failover), hard impossibilities (size, tokens, context).
# ROLLBACK: SHIM_LOCAL_FIRST=0 restores the previous behavior exactly (dashboard-editable).
def _parse_reason_set(v):
    if isinstance(v, (set, frozenset, list, tuple)):
        v = ",".join(str(x) for x in v)
    parts = str(v or "").replace("|", ",").split(",")
    return frozenset(x.strip(_REPR_JUNK) for x in parts if x.strip(_REPR_JUNK))


LOCAL_FIRST = _env_flag(os.environ.get("SHIM_LOCAL_FIRST", "1"), True)
LOCAL_FIRST_REASONS = _parse_reason_set(os.environ.get(
    "SHIM_LOCAL_FIRST_REASONS", "big-prompt,perf,predicted,big-out,monster"))
LOCAL_FIRST_QUEUE_WAIT_SECS = float(os.environ.get("SHIM_LOCAL_FIRST_QUEUE_WAIT_SECS", "5"))
LOCAL_FIRST_WAIT_WINDOW_SECS = float(os.environ.get("SHIM_LOCAL_FIRST_WAIT_WINDOW_SECS", "60"))
# A prompt >= BIG_PROMPT kept local may need a cold prefill longer than FIRST_TOKEN_MAX
# (100K tokens at the measured ~1,600 tok/s is ~62s); without a wider cap it would be misread
# as a wedge and fail over after wasting the prefill. Applies only to LOCAL relays.
LOCAL_FIRST_FIRST_TOKEN_MAX = float(os.environ.get("SHIM_LOCAL_FIRST_FIRST_TOKEN_MAX", "300"))
# Optional latency ceiling for INTERACTIVE requests: > 0 sends an interactive request remote
# (under its original reason) when its predicted local TTFT -- predicted computed tokens /
# PREFILL_TPS, scaled by concurrency -- exceeds this. 0 = off: the data above shows local does
# not breach a 20s interactive ceiling, so it is off by default.
LOCAL_FIRST_INTERACTIVE_TTFT_SECS = float(os.environ.get("SHIM_LOCAL_FIRST_INTERACTIVE_TTFT_SECS", "0"))
# ---- LF (lane LF, 2026-10-03): "are we routing to remote what local could serve?" ----
# MEASURED 10-02 23:21 -> 10-03 05:10 (11,879 requests): of 4,179 "discretionary" remote trips, 2,822 happened INSIDE
# declared offline windows (the big-out/big-prompt guards sat before the _local_offline() check, so a window's
# traffic wore a discretionary label), and in the 95 live minutes local prefill was already 94% busy.
# What was left was decided by flat constants, not by the caller's patience:
#   * big-out compared the REQUESTED ceiling (Halo sends max_tokens=16384 on every turn; its real outputs are p50 336 /
#     p90 846 tokens) -> every Halo turn tripped it;
#   * the heavy-prefill / queue-wait / monster tests used 15 s / 5 s / 30 s whatever the class and its deadline;
#   * a remote-served turn never warms the local prefix model, so a patient caller was predicted "cold" forever and
#     its cold prediction sent it remote again (self-fulfilling).
# With LOCAL_FIRST_DERIVE (default on) a predictive reason keeps the request local while its PREDICTED time to first
# token -- queue wait for its class + the uncached prefill already ahead in the engine + its own uncached prefill, all
# at the live measured prefill rate -- fits the class's patience: kevin = FIRST_TOKEN_MAX (a person is waiting),
# halo = LOCAL_FIRST_FIRST_TOKEN_MAX (no deadline), runner/background = their flow deadline; every budget is capped at
# LOCAL_FIRST_FIRST_TOKEN_MAX, the first-token wait the local relay will actually grant. The flat
# QUEUE_WAIT_SECS / HEAVY_* / MONSTER_PREFILL_SECS values remain the configured FALLBACK (derive off, or no measured
# prefill rate yet). ROLLBACK: SHIM_LOCAL_FIRST_DERIVE=0 and SHIM_EXPECTED_OUTPUT=0 restore the previous rules.
LOCAL_FIRST_DERIVE = _env_flag(os.environ.get("SHIM_LOCAL_FIRST_DERIVE", "1"), True)
# big-out is decided on the output a caller ACTUALLY produces (this quantile of its recent completions) once there
# are enough samples; the requested max_tokens stays the hard upper bound and the fallback.
EXPECTED_OUTPUT = _env_flag(os.environ.get("SHIM_EXPECTED_OUTPUT", "1"), True)
EXPECTED_OUTPUT_QUANTILE = float(os.environ.get("SHIM_EXPECTED_OUTPUT_QUANTILE", "0.95"))
EXPECTED_OUTPUT_MIN_SAMPLES = int(os.environ.get("SHIM_EXPECTED_OUTPUT_MIN_SAMPLES", "20"))
EXPECTED_OUTPUT_HISTORY = int(os.environ.get("SHIM_EXPECTED_OUTPUT_HISTORY", "200"))

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s [gateway] %(levelname)s %(message)s")
log = logging.getLogger("gateway-shim")

# ---- lane SH (2026-10-03): swallowed exceptions are counted, never invisible ----
# ~100 `except Exception: pass` blocks keep bookkeeping from breaking a request, which is right, but a handler that
# fires on EVERY request (an unwritable ledger, an unreadable outcome alarm that silently keeps the stalled brake on,
# a learning table that always raises) looked exactly like health. State/IO sites now call _swallowed(site, exc):
# per-site count + last error, a warning on the 1st, 2nd, 4th, 8th... occurrence, exported at
# /gateway/internal-errors and as gateway_swallowed_errors_total{site} on /metrics. Parse-tolerance sites
# (untrusted request bodies, optional files, regex scrapes) stay silent on purpose.
_SWALLOWED = {}                 # site -> [count, last_epoch, last_repr]; keys are fixed source sites (bounded)


def _swallowed(site, exc):
    rec = _SWALLOWED.get(site)
    if rec is None:
        rec = _SWALLOWED[site] = [0, 0.0, ""]
    rec[0] += 1
    rec[1] = time.time()
    rec[2] = repr(exc)[:240]
    n = rec[0]
    if n & (n - 1) == 0:          # 1, 2, 4, 8, ...: loud at first, never a log flood
        log.warning("swallowed exception #%d at %s: %s", n, site, rec[2])

# Optional admin gate for mutating gateway endpoints (POST /gateway/config, POST
# /gateway/models/local). Default OFF: loaded once at import. SHIM_ADMIN_TOKEN env
# overrides; otherwise read from SHIM_ADMIN_TOKEN_FILE (0600 file, one line, whitespace
# stripped). Neither present -> token is "" -> _admin_ok() always True -> byte-for-byte
# the same behavior as before this patch existed. The /v1 proxy and every GET endpoint
# are untouched either way -- this only ever gates the two POST handlers that change
# live config or trigger a model switch.
SHIM_ADMIN_TOKEN_FILE = os.environ.get(
    "SHIM_ADMIN_TOKEN_FILE", "/home/kevin/.local/share/vllm-qwen27b/admin.token")


def _load_admin_token():
    env_tok = os.environ.get("SHIM_ADMIN_TOKEN")
    if env_tok:
        return env_tok
    try:
        with open(SHIM_ADMIN_TOKEN_FILE) as fh:
            return fh.read().strip()
    except Exception:
        return ""


SHIM_ADMIN_TOKEN = _load_admin_token()
if not SHIM_ADMIN_TOKEN:
    log.warning("SHIM_ADMIN_TOKEN not set (no SHIM_ADMIN_TOKEN env, no token at %s) -- "
                "admin endpoints OPEN: POST /gateway/config and POST /gateway/models/local "
                "accept requests with no authentication", SHIM_ADMIN_TOKEN_FILE)


def _admin_ok(request):
    """Constant-time header check. Token unset/empty -> always True (default-off)."""
    if not SHIM_ADMIN_TOKEN:
        return True
    return hmac.compare_digest(request.headers.get("X-Admin-Token", ""), SHIM_ADMIN_TOKEN)


_inflight = 0            # local capacity units currently in flight
_inflight_tokens = 0    # sum of est prompt tokens of in-flight local requests (size-aware cap)
_inflight_reserved_tokens = 0  # prompt + maximum bounded generation, across all local lanes
_inflight_computed = 0   # sum of PREDICTED UNCACHED prefill tokens of in-flight local requests
_waiting  = 0           # requests currently blocked in the queue-first wait loop (backlog)
# 2026-09-11 (gw-queue-position-header): per-class split of the SAME count above. Kevin's
# dashboard showed one aggregate "waiting" number with no way to tell "am I, personally,
# waiting" from "a cron job is waiting" -- exactly the ambiguity behind his "response time...
# is unbearably slow" complaint, since a page full of background waiters looks identical to
# one interactive waiter. Mutated at the SAME two sites as _waiting, by construction (see the
# comment there): never drifts from it because it is updated in the same breath.
_waiting_by_class = {"interactive": 0, "background": 0}
# gw-admission-computed-token-cost: client name -> {"hashes": per-message sha256 list of that
# client's last request, "prompt_tokens": that request's estimated prompt tokens, "ts":
# time.time()}. One entry per client (not per request) -- a one-deep prefix-cache model, matching
# the shape of a growing multi-turn conversation where each new turn's messages list starts with
# ALL of the previous turn's messages (see _is_prefix_of() -- a real conversation typically grows
# by 2+ messages per turn, since the client echoes the assistant's own reply back alongside the
# next user message, not by exactly 1). Bounded by PREFIX_CACHE_MAX_CLIENTS (FIFO eviction) so an
# attacker or a runaway number of distinct X-Client values can't grow this without bound.
_prefix_seen = {}
_backoff_until = 0.0
_health = {"ok": False, "at": 0.0}
REMOTE_ENABLED = bool(REMOTE_BASE and REMOTE_KEY)


# LS lane 2026-10-02: the paid provider answered HTTP 402 "Insufficient Balance" for ~1 hour
# (154 requests, all returned to callers as empty/failed) while the gateway kept choosing it as the
# overflow target. A hard provider refusal is not a capacity signal: treat the remote as unavailable
# for REMOTE_DEAD_SECS (requests wait for a local lane instead, exactly like FULL LOCAL), then let one
# request probe it again. Explicit remote aliases still try the provider -- their callers handle refusal.
REMOTE_DEAD_SECS = float(os.environ.get("SHIM_REMOTE_DEAD_SECS", "300"))
_remote_dead_until = 0.0
_remote_dead_count = 0
# RS (2026-10-02 verification of 854b51ab62): the breaker lived only in memory, so every gateway publish reset it. The first requests
# after the 12:49 publish saw remote_ok()==True, got the SHORT first-token deadline and were aborted ('held/local-failed', 50.7 s,
# 13k tokens of prefill thrown away) although the remote was still empty. Persist the dead-until stamp across restarts.
_REMOTE_DEAD_FILE = os.path.expanduser(os.environ.get("SHIM_REMOTE_DEAD_FILE", "~/.local/share/vllm-qwen27b/remote-dead-until.json"))


def _remote_dead_load():
    global _remote_dead_until
    try:
        with open(_REMOTE_DEAD_FILE) as fh:
            until = float(json.load(fh).get("until") or 0)
        if until > time.time():
            _remote_dead_until = max(_remote_dead_until, until)
    except Exception:
        pass


def _remote_dead_save():
    if not _REMOTE_DEAD_PERSIST:
        return
    try:
        tmp = _REMOTE_DEAD_FILE + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"until": _remote_dead_until, "count": _remote_dead_count}, fh)
        os.replace(tmp, _REMOTE_DEAD_FILE)
    except Exception as _e:
        _swallowed("_remote_dead_save", _e)


_REMOTE_DEAD_PERSIST = False       # armed by _on_startup only: importing the module (tests) must never read or write live state


def _note_remote_status(base, status):
    """Called with every upstream HTTP status; arms the remote breaker on 402 (balance exhausted)."""
    global _remote_dead_until, _remote_dead_count
    try:
        if status == 402 and str(base).rstrip("/") != str(LOCAL).rstrip("/"):
            _remote_dead_count += 1
            if time.time() >= _remote_dead_until:
                log.error("remote provider refused with 402 (balance): remote overflow OFF for %ds; "
                          "requests wait for local until it recovers", int(REMOTE_DEAD_SECS))
            _remote_dead_until = time.time() + REMOTE_DEAD_SECS
            _remote_dead_save()
        elif status == 200 and str(base).rstrip("/") != str(LOCAL).rstrip("/") and _remote_dead_until:
            _remote_dead_until = 0.0
            _remote_dead_save()
    except Exception as _e:
        _swallowed("_note_remote_status", _e)


def remote_ok():
    """May this request overflow to the PAID remote at all?

    One question, one place. FULL LOCAL (Kevin 2026-09-10) is exactly "the answer is no,
    for every request" -- so every routing decision asks this rather than reading
    REMOTE_ENABLED itself, and a request that would have overflowed waits for a local lane
    instead (the queue-first wait loop already treats "no remote" as an unbounded deadline).
    Reading the globals live is deliberate: both flags are hot-reloadable from the dashboard,
    so the mode changes without a restart and without dropping in-flight work."""
    return bool(REMOTE_ENABLED) and not LOCAL_ONLY and time.time() >= _remote_dead_until


def routing_mode():
    """The dashboard's three states, derived from the same two flags the router uses so the
    badge can never disagree with the behaviour."""
    if LOCAL_ONLY:
        return "full_local"
    if REMOTE_ENABLED and effective_force_remote():
        return "full_remote"
    return "local_first"


STATS_FILE = os.environ.get("SHIM_STATS_FILE", "/home/kevin/.local/share/vllm-qwen27b/gateway-stats.json")

# ---------------- live metrics (for the /gateway/dashboard status page) ----------------
_PROCESS_STARTED = time.time()   # THIS process (the persisted _stats["started"] below survives restarts)
_stats = {"started": time.time(), "total": 0, "local": 0, "remote": 0,
          "waited_total": 0.0, "waited_n": 0, "peak_inflight": 0, "peak_waiting": 0,
          "overflowed_after_wait": 0, "held": 0, "rejected_bg": 0}
_remote_reasons = collections.Counter()
_events = collections.deque(maxlen=200)   # most-recent-first ring buffer of routing decisions
_gpu_cache = {"at": 0.0, "data": []}

def _client_label(request):
    # harness self-id via X-Client/X-Title header, else source IP. Capped defensively: this
    # lands in the _events ring buffer (maxlen=200, but NOT per-entry-size-bounded) and an
    # oversized/hostile header would otherwise sit in memory 200x over.
    v = (request.headers.get("X-Client") or request.headers.get("X-Title")
         or getattr(request, "remote", None) or "?")
    return str(v)[:80]

# ---- live request registry: the dashboard's "agents in progress" for a fully local operation ----
# Every /v1 completion request is registered while in flight (queued, generating on the local model, or
# overflowed to the remote provider) and dropped when it finishes. Nothing here is persisted.
_ACTIVE = {}          # id(request) -> {name, ip, ua, ep, model, ptok, maxtok, stream, t0, phase, route, reason, preview, bg, tiny}
# A deployment fence: set on the event loop before observing _ACTIVE. It has a
# lease so a crashed deployer cannot leave the gateway rejecting work forever.
_DRAIN_UNTIL = 0.0
_DRAIN_LEASE = None
_DRAIN_REASON = None
# RS (2026-10-02): every drain is one durable record. Before this a drain left NO trace of who raised it, why, how long it held,
# or how many requests it refused (the 503 reason text was a constant), so "why was the estate down 20 min?" was unanswerable.
# Events are append-only JSONL: {"event":"open"|"close", ...}; `close.how` is delete | expired | gateway-restart.
_DRAIN_LEDGER = os.path.expanduser(os.environ.get("GATEWAY_DRAIN_LEDGER", "~/.local/share/vllm-qwen27b/incidents/drains.jsonl"))
_DRAIN_REC = None    # live record of the open drain: {t0, reason, by, ttl_s, active_at_open, refused, refused_by_client{}}


def _drain_ledger_write(row):
    try:
        os.makedirs(os.path.dirname(_DRAIN_LEDGER), exist_ok=True)
        with open(_DRAIN_LEDGER, "a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    except Exception as _e:       # never let bookkeeping affect admission
        _swallowed("_drain_ledger_write", _e)


def _drain_close_record(how, now=None):
    """Write the close event for the open drain record (idempotent)."""
    global _DRAIN_REC
    rec, _DRAIN_REC = _DRAIN_REC, None
    if not rec:
        return
    now = time.time() if now is None else now
    end = min(now, rec["t0"] + rec["ttl_s"]) if how == "expired" else now
    _drain_ledger_write({"event": "close", "how": how, "t": round(end, 3), "t0": round(rec["t0"], 3), "duration_s": round(end - rec["t0"], 1),
                         "reason": rec["reason"], "by": rec["by"], "ttl_s": rec["ttl_s"], "active_at_open": rec["active_at_open"],
                         "active_at_close": len(_ACTIVE), "refused": rec["refused"],
                         "refused_by_client": dict(sorted(rec["refused_by_client"].items(), key=lambda kv: -kv[1])[:12])})


def _drain_reap(now=None):
    """Close the record of a lease that expired without a DELETE (a crashed deployer)."""
    if _DRAIN_REC and not _draining(now):
        _drain_close_record("expired", now)


def _drain_startup_recover():
    """A previous gateway process may have died/been replaced while a drain was open (every publish does exactly that):
    close it in the ledger as `gateway-restart` so open/close always pair."""
    try:
        last = None
        with open(_DRAIN_LEDGER) as fh:
            for ln in fh:
                if ln.strip():
                    last = json.loads(ln)
        if last and last.get("event") == "open":
            _drain_ledger_write({"event": "close", "how": "gateway-restart", "t": round(time.time(), 3), "t0": last.get("t"),
                                 "duration_s": round(time.time() - float(last.get("t") or time.time()), 1), "reason": last.get("reason"),
                                 "by": last.get("by"), "ttl_s": last.get("ttl_s"), "active_at_open": last.get("active"),
                                 "refused": None, "note": "refused count unknown: the process that held the fence is gone"})
    except Exception as _e:
        _swallowed("_drain_startup_recover", _e)


def _draining(now=None):
    return (time.time() if now is None else now) < _DRAIN_UNTIL


async def gateway_drain(request):
    """Admin-controlled, leased admission fence for lossless gateway publication."""
    global _DRAIN_UNTIL, _DRAIN_LEASE, _DRAIN_REASON, _DRAIN_REC
    if request.method == "GET":
        _drain_reap()
        return web.json_response({"draining": _draining(), "active": len(_ACTIVE),
                                  "until": _DRAIN_UNTIL if _draining() else None,
                                  "reason": _DRAIN_REASON if _draining() else None,
                                  "by": _DRAIN_REC["by"] if _DRAIN_REC else None,
                                  "since": _DRAIN_REC["t0"] if _DRAIN_REC else None,
                                  "refused": _DRAIN_REC["refused"] if _DRAIN_REC else None})
    if not _admin_ok(request):
        return web.json_response({"error": "admin token required"}, status=401)
    if request.method == "DELETE":
        try:
            body = await request.json()
        except (ValueError, TypeError):
            body = None
        if not isinstance(body, dict) or body.get("lease") != _DRAIN_LEASE or not _DRAIN_LEASE:
            return web.json_response({"error": "drain lease mismatch"}, status=409)
        _DRAIN_UNTIL, _DRAIN_LEASE, _DRAIN_REASON = 0.0, None, None
        _drain_close_record("delete")
        return web.json_response({"draining": False, "active": len(_ACTIVE)})
    try:
        body = await request.json()
        ttl = int(body.get("ttl_s"))
        if not isinstance(body, dict) or not 30 <= ttl <= 1800:
            raise ValueError("ttl_s outside [30, 1800]")
    except (ValueError, TypeError, AttributeError):
        return web.json_response({"error": "ttl_s must be 30..1800 seconds"}, status=400)
    _drain_reap()
    if _draining():
        return web.json_response({"error": "another drain lease is active", "active": len(_ACTIVE),
                                  "until": _DRAIN_UNTIL, "reason": _DRAIN_REASON,
                                  "by": _DRAIN_REC["by"] if _DRAIN_REC else None}, status=409)
    _DRAIN_LEASE = os.urandom(16).hex()
    _DRAIN_UNTIL = time.time() + ttl
    _DRAIN_REASON = str(body.get("reason") or "deployment")[:120]
    by = str(body.get("by") or request.headers.get("X-Client") or request.headers.get("User-Agent") or "?")[:80]
    _DRAIN_REC = {"t0": time.time(), "reason": _DRAIN_REASON, "by": by, "ttl_s": ttl, "active_at_open": len(_ACTIVE),
                  "refused": 0, "refused_by_client": {}}
    _drain_ledger_write({"event": "open", "t": round(_DRAIN_REC["t0"], 3), "reason": _DRAIN_REASON, "by": by, "ttl_s": ttl,
                         "active": len(_ACTIVE)})
    return web.json_response({"draining": True, "active": len(_ACTIVE),
                              "until": _DRAIN_UNTIL, "lease": _DRAIN_LEASE})
_CLIENT_NAMES_FILE = "/home/kevin/.local/share/vllm-qwen27b/clients.map"   # "<ip or X-Client> = <friendly name>"
_client_names_cache = {"t": 0.0, "map": {}}


def _client_names():
    now = time.time()
    if now - _client_names_cache["t"] > 10:
        m = {}
        try:
            with open(_CLIENT_NAMES_FILE) as fh:
                for ln in fh:
                    ln = ln.strip()
                    if ln and not ln.startswith("#") and "=" in ln:
                        k, v = ln.split("=", 1)
                        m[k.strip()] = v.strip()
        except Exception:
            pass
        _client_names_cache.update(t=now, map=m)
    return _client_names_cache["map"]


def _friendly_client(request):
    ip = getattr(request, "remote", None) or "?"
    xc = (request.headers.get("X-Client") or request.headers.get("X-Title") or "")[:40]
    ua = (request.headers.get("User-Agent") or "")[:40]
    names = _client_names()
    # a harness that names itself (X-Client / X-Title) wins over the per-IP mapping
    return {"ip": ip, "xclient": xc, "ua": ua, "name": names.get(xc) or xc or names.get(ip) or ip}


def _active_set(request, **kw):
    a = _ACTIVE.get(id(request))
    if a is not None:
        a.update(kw)


def _request_identity(request, body):
    """Privacy-safe request identity for retry/duplicate forensics.

    Callers may supply Idempotency-Key or X-Request-ID. We persist only its
    digest, plus a digest of the exact request body, never the raw identifier or
    prompt. This makes repeated paid submissions measurable without changing
    routing or pretending identical prompts are always accidental.
    """
    headers = getattr(request, "headers", None) or {}
    raw = (headers.get("Idempotency-Key") or headers.get("X-Request-ID") or "")[:512]
    return {
        "request_id": hashlib.sha256(raw.encode()).hexdigest()[:24] if raw else None,
        "request_id_source": ("idempotency-key" if headers.get("Idempotency-Key")
                              else "x-request-id" if headers.get("X-Request-ID") else None),
        "request_body_sha256": hashlib.sha256(body).hexdigest(),
    }


def _active_snapshot(now):
    out = []
    for a in list(_ACTIVE.values()):
        d = dict(a)
        d["elapsed_s"] = round(now - a["t0"], 1)
        out.append(d)
    out.sort(key=lambda x: -x["elapsed_s"])
    return out


# ======================================================================================
# TELEMETRY (added). Full design: LANE/DESIGN.md. Self-contained module: its own state,
# its own sampler task (started in _on_startup / cancelled in _on_cleanup, mirroring the
# existing _stats_saver() task below), its own two routes (/gateway/telemetry,
# /gateway/metrics). The only touches OUTSIDE this block are: record_event() and
# _gpu_stats() just below (edited in place -- not duplicated), a handful of one-line hooks
# in _relay() / _forward_remote() / the three local-success branches / handle_completions'
# existing finally (each individually justified in DESIGN.md (c)), and the dashboard's new
# Telemetry section appended near the end of DASHBOARD_HTML.
# ======================================================================================

TELEM_SAMPLE_SECS    = float(os.environ.get("SHIM_TELEM_SAMPLE_SECS", "2"))
TELEM_SLOW_EVERY     = int(os.environ.get("SHIM_TELEM_SLOW_EVERY", "30"))     # 30*2s = 60s/slow pt
TELEM_SCRAPE_TIMEOUT = float(os.environ.get("SHIM_TELEM_SCRAPE_TIMEOUT", "3"))
# Placeholder $/Mtok figures -- NOT scraped from the provider. Kevin/Fable should set the real
# current rate from the dashboard (same live-edit mechanism as every other knob, see _CFG below).
REMOTE_COST_IN_PER_MTOK       = float(os.environ.get("SHIM_REMOTE_COST_IN_PER_MTOK", "0.28"))
REMOTE_COST_OUT_PER_MTOK      = float(os.environ.get("SHIM_REMOTE_COST_OUT_PER_MTOK", "1.10"))
REMOTE_COST_IN_PER_MTOK_PEAK  = float(os.environ.get("SHIM_REMOTE_COST_IN_PER_MTOK_PEAK", str(REMOTE_COST_IN_PER_MTOK)))
REMOTE_COST_OUT_PER_MTOK_PEAK = float(os.environ.get("SHIM_REMOTE_COST_OUT_PER_MTOK_PEAK", str(REMOTE_COST_OUT_PER_MTOK)))
# vLLM mounts /metrics at the root (vllm/entrypoints/serve/instrumentator/metrics.py), not under
# /v1 -- same normalisation the existing FOREIGN_LOAD_GUARD probe in local_healthy() uses below,
# duplicated here rather than factored out to avoid touching that function for a one-liner.
_METRICS_URL = (LOCAL.rsplit("/v1", 1)[0] if LOCAL.endswith("/v1") else LOCAL) + "/metrics"
_MODELS_URL = (LOCAL.rsplit("/v1", 1)[0] if LOCAL.endswith("/v1") else LOCAL) + "/v1/models"

_CFG.update({
    "SHIM_TELEM_SAMPLE_SECS":             ("TELEM_SAMPLE_SECS", float),
    "SHIM_TELEM_SLOW_EVERY":              ("TELEM_SLOW_EVERY", int),
    "SHIM_REMOTE_COST_IN_PER_MTOK":       ("REMOTE_COST_IN_PER_MTOK", float),
    "SHIM_REMOTE_COST_OUT_PER_MTOK":      ("REMOTE_COST_OUT_PER_MTOK", float),
    "SHIM_REMOTE_COST_IN_PER_MTOK_PEAK":  ("REMOTE_COST_IN_PER_MTOK_PEAK", float),
    "SHIM_REMOTE_COST_OUT_PER_MTOK_PEAK": ("REMOTE_COST_OUT_PER_MTOK_PEAK", float),
})   # extends the existing live-config dict -- these become dashboard-editable like every other knob

# ---- (e) Retention: append-only JSONL request log. Design: telemetry/DESIGN.md (e) (the prior
# telemetry patch explicitly did NOT write here -- "design only, nothing written to the real
# path"). This lane (telemetry-history) implements exactly that design. See REPORT.md for the
# full event-loop-safety argument and disk math; short version: _telemetry_note_request() below
# enqueues one compact dict per finished request onto a small bounded in-memory list (plain
# list.append(), no I/O); _jsonl_flusher() (new task, sibling to _stats_saver(), same start/cancel
# lifecycle) wakes every TELEMETRY_FLUSH_SECS and hands whatever accumulated to a thread-pool
# executor for the actual open()/write()/close() -- never on the request path.
TELEMETRY_DIR             = os.environ.get("SHIM_TELEMETRY_DIR", "/home/kevin/.local/share/vllm-qwen27b/telemetry")
TELEMETRY_JSONL_MAX_MB    = float(os.environ.get("SHIM_TELEMETRY_JSONL_MAX_MB", "200"))
TELEMETRY_FLUSH_SECS      = float(os.environ.get("SHIM_TELEMETRY_FLUSH_SECS", "3"))
TELEMETRY_RETENTION_DAYS  = int(os.environ.get("SHIM_TELEMETRY_RETENTION_DAYS", "30"))
TELEMETRY_QUEUE_MAX       = int(os.environ.get("SHIM_TELEMETRY_QUEUE_MAX", "10000"))
HISTORY_SUMMARY_CACHE_TTL = float(os.environ.get("SHIM_HISTORY_SUMMARY_CACHE_TTL", "5"))

_CFG.update({
    "SHIM_TELEMETRY_JSONL_MAX_MB":   ("TELEMETRY_JSONL_MAX_MB", float),
    "SHIM_TELEMETRY_FLUSH_SECS":     ("TELEMETRY_FLUSH_SECS", float),
    "SHIM_TELEMETRY_RETENTION_DAYS": ("TELEMETRY_RETENTION_DAYS", int),
})   # dashboard-editable like every other knob. SHIM_TELEMETRY_DIR is deliberately NOT in here --
     # it's a deployment path, same category as SHIM_STATS_FILE/SHIM_MODELS_DIR (neither of which
     # is live-editable either), not a runtime tuning knob.

_JSONL_PENDING = []   # plain list; appended to only from the event loop (see _telemetry_log_enqueue)
_JSONL_STATE = {"date": None, "path": None, "bytes": 0, "capped": False,
                "written": 0, "dropped_cap": 0, "dropped_queue": 0, "last_err": None,
                # lane SH: rows lost AFTER leaving the queue used to vanish uncounted (and a failed write still
                # counted as written): unserialisable row / failed write / flusher error.
                "dropped_bad": 0, "dropped_io": 0, "dropped_err": 0}
_HISTORY_SUMMARY_CACHE = {"key": None, "at": 0.0, "data": None}

# ---- rings: bounded by construction (deque maxlen), see DESIGN.md (e) ----
_TELEM_FAST   = collections.deque(maxlen=1800)   # 1h @ TELEM_SAMPLE_SECS=2s
_TELEM_SLOW   = collections.deque(maxlen=1440)   # 24h @ ~1min, downsampled from _TELEM_FAST
_TELEM_WINDOW = []                                # fast points since the last slow-ring append
_TELEM_TICK   = 0
_ENGINE_METRICS = {"ok": False, "at": 0.0, "err": None, "text": "", "families": {}}
_PER_CLIENT = collections.defaultdict(lambda: {
    "requests": 0, "local": 0, "remote": 0, "tokens_out": 0, "tokens_out_exact": 0,
    "tokens_out_lb": 0, "wait_sum": 0.0, "wait_n": 0, "ttft_sum": 0.0, "ttft_n": 0,
    "errors": 0, "cost_est_usd": 0.0, "classes": collections.Counter()})
# lane SH: keyed by the caller-supplied X-Client/X-Title (or IP), so a client that varies its header grew this table
# -- and the per-client /metrics label set -- without bound. Past CLIENT_KEYS_MAX distinct names, new names share
# one "(other)" row; existing rows keep counting.
CLIENT_KEYS_MAX = 512
CLIENT_OVERFLOW_KEY = "(other)"


def _client_key(table, name):
    """`name` if `table` already has it or still has room, else the shared overflow key."""
    if name in table or len(table) < CLIENT_KEYS_MAX:
        return name
    return CLIENT_OVERFLOW_KEY


_ERROR_FEED = collections.deque(maxlen=200)
# gw-admission-computed-token-cost safety AC: a request whose actual computed tokens exceed
# what predict_computed_tokens() would have charged it by > 2x, regardless of whether
# USE_COMPUTED_COST is even on -- this must be visible BEFORE the switch is flipped, not after.
_MISESTIMATE_FEED = collections.deque(maxlen=200)
_cpu_prev = {"t": 0.0, "total": 0, "idle": 0}
_telem_lr_prev = {"local": None, "remote": None}   # previous tick's cumulative local/remote counts
_PERF_STATE = {"bad_streak": 0, "good_streak": 0, "until": 0.0, "reason": ""}


def _remote_share_delta():
    """Remote-overflow share over the last sample interval (not all-time) -- diff of the
    existing cumulative _stats['local']/['remote'] counters between two ticks. None on the
    first tick (nothing to diff yet) or a tick where nothing completed."""
    pl, pr = _telem_lr_prev["local"], _telem_lr_prev["remote"]
    cl, cr = _stats["local"], _stats["remote"]
    _telem_lr_prev.update(local=cl, remote=cr)
    if pl is None:
        return None
    dl, dr = cl - pl, cr - pr
    tot = dl + dr
    return round(100 * dr / tot, 1) if tot > 0 else None


def _perf_sample_bad(sample):
    """Return measurable overload reasons from one telemetry sample."""
    if not PERF_BREAKER_ENABLED or not sample:
        return []
    eng = sample.get("engine") or {}
    reasons = []
    waiting = eng.get("waiting")
    ttft = eng.get("ttft_p95")
    kv = eng.get("kv_cache_pct")
    if isinstance(waiting, (int, float)) and waiting > 0:
        reasons.append("engine-queue")
    if isinstance(ttft, (int, float)) and ttft >= PERF_BREAKER_TTFT_P95_SECS:
        reasons.append("ttft-p95")
    if isinstance(kv, (int, float)) and kv >= PERF_BREAKER_KV_PCT:
        reasons.append("kv-pressure")
    gpu = [g.get("util") for g in (sample.get("gpu") or []) if isinstance(g.get("util"), (int, float))]
    running = eng.get("running") or 0
    if gpu and sum(gpu) / len(gpu) >= PERF_BREAKER_GPU_UTIL_PCT and running >= 2:
        reasons.append("gpu-saturation")
    return reasons


def _update_perf_breaker(sample):
    """Small hysteretic circuit breaker, driven only by sampled runtime evidence."""
    reasons = _perf_sample_bad(sample)
    if reasons:
        _PERF_STATE["bad_streak"] += 1
        _PERF_STATE["good_streak"] = 0
        _PERF_STATE["reason"] = ",".join(reasons)
        if _PERF_STATE["bad_streak"] >= max(1, PERF_BREAKER_MIN_SAMPLES):
            _PERF_STATE["until"] = max(_PERF_STATE["until"], time.time() + PERF_BREAKER_HOLD_SECS)
    else:
        _PERF_STATE["good_streak"] += 1
        _PERF_STATE["bad_streak"] = 0
        if _PERF_STATE["good_streak"] >= max(1, PERF_BREAKER_MIN_SAMPLES):
            _PERF_STATE["until"] = 0.0
            _PERF_STATE["reason"] = ""


def perf_breaker_active():
    return bool(PERF_BREAKER_ENABLED and time.time() < _PERF_STATE.get("until", 0.0))


# ---- LOCAL-FIRST state + decision (see the LOCAL_FIRST config block) ----
_ADMISSION_WAITS = collections.deque(maxlen=512)       # (t, waited_s) per admission outcome
_local_first_kept = collections.Counter()              # reason -> kept local
_local_first_remote = collections.Counter()            # "reason:why" -> still sent remote
# LF: client -> remote trips taken while its prefix was predicted (almost) entirely cold. A remote-served turn never warms the
# local prefix model, so a caller that keeps going remote stays predicted cold; a rising count here is that loop (the fix is
# capacity -- the derived budget keeps patient callers local whenever their first token fits -- or rewarm-prefix.py).
_local_first_cold_remote = collections.Counter()


def _note_admission_wait(waited, now=None):
    """Record how long one request waited in the admission loop (0 for an immediate slot)."""
    _ADMISSION_WAITS.append((time.time() if now is None else now, float(waited or 0.0)))


def recent_admission_wait(now=None, window=None):
    """Mean admission wait (s) over the last `window` seconds; 0.0 with no samples."""
    now = time.time() if now is None else now
    window = LOCAL_FIRST_WAIT_WINDOW_SECS if window is None else window
    vals = [w for (t, w) in _ADMISSION_WAITS if now - t <= window]
    return sum(vals) / len(vals) if vals else 0.0


def local_reservation_estimate(ptok, maxtok):
    """KV reservation the local path will charge (prompt + bounded output), mirroring
    _prepare_local_body's LOCAL_MAX_OUT clamp + local_memory_reservation()."""
    out = maxtok if maxtok and maxtok > 0 else 0
    if LOCAL_MAX_OUT > 0 and (out <= 0 or out > LOCAL_MAX_OUT):
        out = LOCAL_MAX_OUT
    elif out <= 0:
        out = max(0, max_local_tokens() - ptok)
    return max(0, int(ptok)) + int(out)


def local_saturation(background, units, reservation, now=None):
    """Measured reasons local cannot take this request NOW ([] = it has capacity)."""
    why = []
    lane_limit = admission_lane_limit(background, effective_budget(), FG_RESERVED)
    if _inflight + max(1, units) > lane_limit:
        why.append("lanes")
    if not _memory_available(reservation):
        why.append("tokens")
    if _waiting_by_class.get("background" if background else "interactive", 0) > 0:
        why.append("queued")
    if LOCAL_FIRST_QUEUE_WAIT_SECS > 0 and recent_admission_wait(now) >= LOCAL_FIRST_QUEUE_WAIT_SECS:
        why.append("queue-wait")
    return why


def local_first_budget_s(cls):
    """Seconds this work class will wait for its FIRST TOKEN before remote is the better answer (lane LF).
    kevin / halo = LOCAL_FIRST_FIRST_TOKEN_MAX (no deadline: they have priority in the flow queue, so the queue term is
    small); runner / background = their flow deadline. Capped at LOCAL_FIRST_FIRST_TOKEN_MAX: the longest first-token
    wait the local relay grants (first_token_timeout), so a request kept local is never one the relay would abort.
    Kevin additionally never queues behind more than HEAVY_ADMIT_BACKLOG_SECS of prefill (see local_first_decision)."""
    cap = max(1.0, float(LOCAL_FIRST_FIRST_TOKEN_MAX))
    try:
        d = float(_flow_deadlines().get(cls) or 0.0)
    except Exception:
        d = 0.0
    return max(1.0, min(d if d > 0 else cap, cap))


def big_prompt_limit(cls=None):
    """The prompt size (tokens, same unit the big-prompt guard compares) above which the prefill alone cannot fit the
    class's first-token patience at the live measured prefill rate. Derived (prefill rate x patience) when local-first
    derivation is on and the engine is not under the crash-adaptive floor; otherwise the configured BIG_PROMPT. Never
    lower than the configured BIG_PROMPT."""
    if BIG_PROMPT <= 0:
        return 0
    if (not (LOCAL_FIRST and LOCAL_FIRST_DERIVE and "big-prompt" in LOCAL_FIRST_REASONS and cls in FLOW_CLASSES)
            or _big_prompt_restore is not None):
        return BIG_PROMPT
    ptps = prefill_tps()
    if ptps <= 0:
        return BIG_PROMPT
    return max(BIG_PROMPT, int(ptps * local_first_budget_s(cls)))


def local_first_predicted_ttft(cls, est_computed, *, background=False, units=1, now=None):
    """(ttft_s, parts): predicted seconds to first token if this request is kept local now. queue = the class's
    expected wait to be admitted; backlog = uncached prefill already admitted to the engine that is still ahead
    of it (after admission the engine backlog is at most the class ceiling); own = its own uncached prefill.
    All at the live measured prefill rate. None when no prefill rate is known (callers use the flat fallbacks)."""
    ptps = prefill_tps()
    if ptps <= 0:
        return None, {}
    own = max(0, est_computed or 0) / ptps
    backlog = _prefill_backlog_secs()
    queue = None
    if FLOW_MODE != "off":
        try:
            queue = float(flow_expected_wait(cls, own, units=max(1, units),
                                             lane_limit=admission_lane_limit(background, effective_budget(), FG_RESERVED)))
            ceil = _flow_ceils()[cls] * FLOW_BACKLOG_S
            if ceil > 0:
                backlog = min(backlog, ceil)
        except Exception:
            queue = None
    if queue is None:
        queue = recent_admission_wait(now)
    return queue + backlog + own, {"queue_s": round(queue, 1), "backlog_s": round(backlog, 1), "own_s": round(own, 1)}


def local_first_decision(reason, *, background, units, reservation, est_computed=0, now=None, cls=None, warm=False):
    """(keep_local, why) for a capacity/latency-PREDICTIVE remote reason.

    keep_local=False means: route remote under `reason` exactly as before this policy
    existed (policy off, reason not covered, local saturated, or the interactive ceiling).
    With `cls` (the work class) and LOCAL_FIRST_DERIVE the saturation test is the class's patience, not flat constants."""
    if not LOCAL_FIRST or reason not in LOCAL_FIRST_REASONS:
        return False, "policy-off"
    if not _health.get("ok", False):
        return False, "local-unhealthy"
    if LOCAL_FIRST_DERIVE and cls in FLOW_CLASSES:
        ttft, parts = local_first_predicted_ttft(cls, est_computed, background=background, units=units, now=now)
        if ttft is not None and warm and WARM_PRIORITY:
            # [LANE CR] a warm continuation is sent with engine priority and its uncached remainder fits the step
            # budget a running cold chunk leaves, so it does not wait behind the cold backlog.
            ttft = ttft - parts.get("backlog_s", 0.0)
            parts = dict(parts, backlog_s=0.0)
        if ttft is not None:
            sat = []
            if not _memory_available(reservation):
                sat.append("tokens")                       # a hard fact: the KV reservation does not fit
            budget = local_first_budget_s(cls)
            if ttft > budget:
                sat.append("deadline")                     # it would not get a first token inside its class patience
            elif cls == "kevin" and HEAVY_ADMIT_BACKLOG_SECS > 0 and parts["backlog_s"] > HEAVY_ADMIT_BACKLOG_SECS:
                sat.append("prefill-backlog")              # a person is waiting: not behind more than the configured backlog bound
            if sat:
                return False, "saturated:" + "+".join(sat)
            if LOCAL_FIRST_INTERACTIVE_TTFT_SECS > 0 and not background and parts["own_s"] > LOCAL_FIRST_INTERACTIVE_TTFT_SECS:
                return False, "interactive-ttft"
            return True, "capacity"
    if USE_COMPUTED_COST and reason == "monster":
        # Cache-aware mode: "monster" means a real uncached prefill is already chewing the engine's
        # chunk steps, and every younger request queues behind it (vLLM schedules in arrival order).
        # That is not a capacity question local-first can answer with free lanes: stay remote.
        return False, "prefill-backlog"
    sat = local_saturation(background, units, reservation, now)
    _ptps = prefill_tps()
    if USE_COMPUTED_COST and _ptps > 0 and HEAVY_PREFILL_SECS > 0:
        # A HEAVY cold prefill (many seconds of the engine's chunk steps) runs locally only when it
        # will not queue behind another one; otherwise the engine serializes them and everything
        # younger waits on both. The bound is in seconds at the measured prefill rate.
        if (max(0, est_computed or 0) / _ptps >= HEAVY_PREFILL_SECS
                and _prefill_backlog_secs() > HEAVY_ADMIT_BACKLOG_SECS):
            sat = list(sat) + ["prefill-backlog"]
    if sat:
        return False, "saturated:" + "+".join(sat)
    if LOCAL_FIRST_INTERACTIVE_TTFT_SECS > 0 and not background:
        conc = max(1, _inflight + 1) if FT_CONCURRENCY_SCALE else 1
        predicted_ttft = (max(0, est_computed or 0) / max(1.0, _ptps)) * conc
        if predicted_ttft > LOCAL_FIRST_INTERACTIVE_TTFT_SECS:
            return False, "interactive-ttft"
    return True, "capacity"


# ---------------- QoL INTERACTIVE OVERFLOW (lane CFG, 2026-10-03) ----------------
# Kevin, 2026-10-03: interactive overflow "should be based on quality of life". An INTERACTIVE request goes remote only
# when the PREDICTED local time-to-first-token is long (> QOL_TTFT_S, counting what it already waited) AND the remote is
# predicted to be meaningfully faster (by >= QOL_MIN_GAIN_S, from MEASURED remote TTFT in its prompt-size band). A request
# about to start locally, or a small one that will be fast locally, never overflows just because it waited; one stuck
# behind a monster prefill overflows at once instead of after a timer. Prediction = local_first_predicted_ttft() (class
# queue wait + engine prefill backlog + own uncached prefill at the live prefill rate), with GW2's engine-anchored cache
# credit for the request's own size when the chain has one (the cache-credit guess is the largest input error).
#   SHIM_QOL_OVERFLOW=off     no effect (default)
#                     shadow  decide and log (telemetry qol_* fields) next to what actually happened; routing unchanged
#                     on      the decision governs interactive overflow for QOL_REASONS and the lane wait
# Background traffic, planned-offline windows, local-down, size caps, pins, aliases and the spend authority are untouched.
QOL_OVERFLOW = os.environ.get("SHIM_QOL_OVERFLOW", "off").strip().lower()
QOL_TTFT_S = float(os.environ.get("SHIM_QOL_TTFT_S", "15"))
QOL_MIN_GAIN_S = float(os.environ.get("SHIM_QOL_MIN_GAIN_S", "5"))
QOL_REMOTE_QUANTILE = float(os.environ.get("SHIM_QOL_REMOTE_QUANTILE", "0.75"))
QOL_REMOTE_MIN_SAMPLES = int(os.environ.get("SHIM_QOL_REMOTE_MIN_SAMPLES", "5"))
QOL_REMOTE_WINDOW_S = float(os.environ.get("SHIM_QOL_REMOTE_WINDOW_S", "21600"))
QOL_REASONS = os.environ.get("SHIM_QOL_REASONS", "perf,big-prompt,monster,predicted")
QOL_BANDS = (8192, 32768, 65536, 131072)          # prompt-token band upper edges; the last band is open-ended
_QOL_REMOTE = collections.deque(maxlen=4000)      # (t, prompt_tokens, ttft_s) of finished streamed remote requests
_QOL_SEEDED = [False]
_QOL_STATS = collections.Counter()
_CFG.update({
    "SHIM_QOL_OVERFLOW": ("QOL_OVERFLOW", lambda v: _qol_cast_mode(v)),
    "SHIM_QOL_TTFT_S": ("QOL_TTFT_S", float),
    "SHIM_QOL_MIN_GAIN_S": ("QOL_MIN_GAIN_S", float),
    "SHIM_QOL_REASONS": ("QOL_REASONS", lambda v: ",".join(sorted({x.strip().lower() for x in str(v).split(",") if x.strip()}))),
})


def _qol_cast_mode(v):
    v = str(v).strip().lower()
    if v not in ("off", "shadow", "on"):
        raise ValueError("qol_overflow must be off|shadow|on")
    return v


def qol_band(ptok):
    for i, edge in enumerate(QOL_BANDS):
        if (ptok or 0) <= edge:
            return i
    return len(QOL_BANDS)


def qol_note_remote(ptok, ttft, now=None):
    if isinstance(ttft, (int, float)) and ttft > 0 and ptok:
        _QOL_REMOTE.append((now or time.time(), int(ptok), float(ttft)))


def _qol_seed():
    """Restore recent remote TTFT samples from the on-disk request log once (a restart must not blind the model)."""
    if _QOL_SEEDED[0]:
        return
    _QOL_SEEDED[0] = True
    try:
        files = sorted(f for f in os.listdir(TELEMETRY_DIR) if f.startswith("requests-") and f.endswith(".jsonl"))[-2:]
        cutoff = time.time() - QOL_REMOTE_WINDOW_S
        rows = []
        for f in files:
            p = os.path.join(TELEMETRY_DIR, f)
            with open(p, "rb") as fh:
                size = os.path.getsize(p)
                fh.seek(max(0, size - 16 * 1024 * 1024))
                for line in fh.read().splitlines()[1:]:
                    if b'"route": "remote"' not in line and b'"route":"remote"' not in line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("t", 0) >= cutoff and r.get("ttft"):
                        rows.append((r["t"], int(r.get("ptok_exact") or r.get("ptok") or 0), float(r["ttft"])))
        for row in sorted(rows)[-_QOL_REMOTE.maxlen:]:
            _QOL_REMOTE.append(row)
    except Exception as e:
        log.warning("qol: remote TTFT seed failed: %s", e)


def qol_remote_ttft(ptok, now=None):
    """(seconds, band, n): the QOL_REMOTE_QUANTILE of measured remote TTFT for this prompt-size band over the last
    QOL_REMOTE_WINDOW_S, from samples in the same provider price period (peak / off-peak hours: the provider's latency
    follows its load) when that period alone has enough, else all hours. seconds is None with fewer than
    QOL_REMOTE_MIN_SAMPLES samples (no claim that remote is faster)."""
    _qol_seed()
    now = now or time.time()
    b = qol_band(ptok)
    rows = [(t, s) for t, p, s in _QOL_REMOTE if now - t <= QOL_REMOTE_WINDOW_S and qol_band(p) == b]
    try:
        pk = is_peak(now)
        same = sorted(s for t, s in rows if is_peak(t) == pk)
    except Exception:
        same = []
    v = same if len(same) >= max(1, QOL_REMOTE_MIN_SAMPLES) else sorted(s for _, s in rows)
    if len(v) < max(1, QOL_REMOTE_MIN_SAMPLES):
        return None, b, len(v)
    return v[min(len(v) - 1, int(QOL_REMOTE_QUANTILE * len(v)))], b, len(v)


def qol_choice(local_remaining_s, waited_s, remote_s, ttft_target_s=None, min_gain_s=None):
    """The pure QoL rule: ("remote"|"local"|"legacy", why). Remote only when the person's total wait for a local first
    token would exceed the target AND the remote is predicted to answer at least min_gain sooner from now."""
    target = QOL_TTFT_S if ttft_target_s is None else ttft_target_s
    gain = QOL_MIN_GAIN_S if min_gain_s is None else min_gain_s
    if local_remaining_s is None:
        return "legacy", "no-local-prediction"
    total = max(0.0, waited_s or 0.0) + max(0.0, local_remaining_s)
    if total <= target:
        return "local", "fast-local"
    if remote_s is None:
        return "legacy", "no-remote-samples"
    if local_remaining_s - remote_s < gain:
        return "local", "remote-not-faster"
    return "remote", "remote-faster"


def qol_decide(cls, est_computed, ptok, *, units=1, waited=0.0, pm_chain=None, now=None):
    """Predict and decide for one interactive request. Never raises; returns a flat dict for telemetry."""
    out = {"qol_mode": QOL_OVERFLOW}
    try:
        own_tok = est_computed
        if pm_chain:
            ac, _age = anchored_credit(pm_chain, ptok, now=now)
            if ac:
                own_tok = max(0, int(ptok or 0) - int(ac))
                out["qol_anchored"] = int(ac)
        pred, parts = local_first_predicted_ttft(cls if cls in FLOW_CLASSES else "kevin", own_tok, units=units, now=now)
        remote, band, n = qol_remote_ttft(ptok, now=now)
        would, why = qol_choice(pred, waited, remote)
        out.update(qol_would=would, qol_why=why, qol_pred_local_s=None if pred is None else round(pred, 1),
                   qol_pred_remote_s=None if remote is None else round(remote, 2), qol_band=band, qol_remote_n=n,
                   qol_waited_s=round(waited or 0.0, 1), **{"qol_" + k: v for k, v in (parts or {}).items()})
    except Exception as e:
        out.update(qol_would="legacy", qol_why="error:%s" % type(e).__name__)
    return out


_QOL_FIELDS = ("qol_mode", "qol_would", "qol_why", "qol_at", "qol_legacy", "qol_pred_local_s", "qol_pred_remote_s",
               "qol_band", "qol_remote_n", "qol_waited_s", "qol_queue_s", "qol_backlog_s", "qol_own_s", "qol_anchored")


def qol_governs(reason):
    return QOL_OVERFLOW in ("shadow", "on") and reason in {x.strip() for x in str(QOL_REASONS).split(",") if x.strip()}


def _latest_decode_tps():
    """Use the most recent measured engine decode rate, with a conservative floor."""
    for sample in reversed(_TELEM_FAST):
        value = ((sample.get("engine") or {}).get("gen_tok_s"))
        if isinstance(value, (int, float)) and value > 0:
            # Do not let a quiet/partially sampled engine report 0--20 tok/s become a
            # self-fulfilling remote-routing loop.  The measured 30-day local baseline is
            # ~74 tok/s, so clamp only the warm-start estimate; the performance breaker handles
            # genuinely sustained degradation separately.
            return max(60.0, float(value))
    # The live 30-day Qwen measurements cluster around 70--75 tok/s.  Use a modestly
    # conservative warm-start value instead of the hard safety floor; otherwise every first
    # 4K-token request after a restart would be pessimistically classified as a 200s job.
    return max(DECODE_TPS_FLOOR, 60.0)


_OUT_HIST = {}                      # client -> deque[(t, outtok)] of recent completed outputs, any route


def note_output_tokens(client, outtok, t=None):
    """Remember how long this client's completed answers really are (its max_tokens is only a ceiling)."""
    try:
        if not client or outtok is None or int(outtok) < 0:
            return
        h = _OUT_HIST.get(client)
        if h is None:
            h = _OUT_HIST[client] = collections.deque(maxlen=max(10, EXPECTED_OUTPUT_HISTORY))
        h.append((time.time() if t is None else t, int(outtok)))
        if len(_OUT_HIST) > 500:                       # bounded: drop the longest-idle client
            _OUT_HIST.pop(min(_OUT_HIST, key=lambda k: _OUT_HIST[k][-1][0] if _OUT_HIST[k] else 0), None)
    except Exception as _e:
        _swallowed("note_output_tokens", _e)


def expected_output_tokens(client, maxtok):
    """(tokens, source). The output this caller is EXPECTED to produce: its recent EXPECTED_OUTPUT_QUANTILE completion
    length, never above the requested max_tokens (the hard bound) and never inferred from fewer than
    EXPECTED_OUTPUT_MIN_SAMPLES completions -- then the ceiling itself is used (the previous rule)."""
    maxtok = int(maxtok or 0)
    if not EXPECTED_OUTPUT or maxtok <= 0:
        return maxtok, "ceiling"
    h = _OUT_HIST.get(client)
    if not h or len(h) < max(1, EXPECTED_OUTPUT_MIN_SAMPLES):
        return maxtok, "ceiling"
    vals = sorted(v for _, v in h)
    q = vals[min(len(vals) - 1, int(len(vals) * min(0.999, max(0.5, EXPECTED_OUTPUT_QUANTILE))))]
    return min(maxtok, max(1, q)), "history-p%d" % round(EXPECTED_OUTPUT_QUANTILE * 100)


def output_history_restore_blocking(now=None, before=None, per_file_cap=16 * 1024 * 1024, per_client=None):
    """Blocking (executor): rebuild the per-client output history from the last day of the request log so a gateway
    restart does not forget how long Halo's answers are. Returns {client: [(t, outtok), ...]} oldest first."""
    now = time.time() if now is None else now
    before = now if before is None else before
    per_client = per_client or max(10, EXPECTED_OUTPUT_HISTORY)
    out = {}
    for day_i in range(2):
        for raw in _read_lines_reverse(_history_day_file(now - day_i * 86400), per_file_cap):
            try:
                rec = json.loads(raw)
            except Exception:
                continue
            t, c, o = rec.get("t"), rec.get("client"), rec.get("outtok")
            if not isinstance(t, (int, float)) or t >= before or t < now - 86400 or not c or o is None:
                continue
            st = rec.get("status")
            if st is not None and st >= 400:
                continue
            lst = out.setdefault(c, [])
            if len(lst) < per_client:
                lst.append((t, int(o)))
    return {c: sorted(v) for c, v in out.items()}


def output_history_restore(data):
    n = 0
    for c, rows in (data or {}).items():
        for t, o in rows:
            note_output_tokens(c, o, t)
            n += 1
    return n


# ---- MICRO fast lane v2 + PROBE synth (lane K4, 2026-10-03) ----
_MICRO_HIST = collections.OrderedDict()   # (client, signature) -> deque[outtok] of recent completed local answers
_MICRO_STATS = {"admitted": 0, "observed": 0, "probe_synth": 0, "probe_real": 0}
_LAST_LOCAL_OK = 0.0                      # last time the local engine really produced tokens for someone
_PROBE_SEEN = 0


def _micro_sig(client, text):
    """(client, normalized prompt head). Digits fold to '#', whitespace collapses: ids, counts and timestamps in a
    prompt head must not split one logical call type into endless one-sample signatures."""
    t = re.sub(r"\s+", " ", re.sub(r"\d+", "#", (text or "").lower())).strip()
    return (client or "?", t[:max(8, MICRO_SIG_CHARS)])


def _note_local_ok(now):
    global _LAST_LOCAL_OK
    _LAST_LOCAL_OK = now


def micro_observe(client, text, outtok):
    """Record one completed LOCAL answer for its signature. Never raises."""
    try:
        if not MICRO_LEARN or not text or outtok is None or int(outtok) < 0:
            return
        sig = _micro_sig(client, text)
        h = _MICRO_HIST.get(sig)
        if h is None:
            h = _MICRO_HIST[sig] = collections.deque(maxlen=max(2, MICRO_HISTORY))
            while len(_MICRO_HIST) > MICRO_MAX_SIGS:
                _MICRO_HIST.popitem(last=False)
        else:
            _MICRO_HIST.move_to_end(sig)
        h.append(int(outtok))
        _MICRO_STATS["observed"] += 1
    except Exception as _e:
        _swallowed("micro_observe", _e)


def micro_predict(client, text, ptok):
    """True when this signature has repeatedly produced short answers (every one of the last MICRO_HISTORY, at least
    MICRO_MIN_SAMPLES of them, <= MICRO_MAX_OUT) and the prompt is modest. One long answer in the window disqualifies."""
    try:
        if not MICRO_LEARN or not text or ptok > MICRO_MAX_PROMPT:
            return False
        h = _MICRO_HIST.get(_micro_sig(client, text))
        return bool(h) and len(h) >= MICRO_MIN_SAMPLES and max(h) <= MICRO_MAX_OUT
    except Exception:
        return False


def micro_history_restore_blocking(now=None, before=None, per_file_cap=16 * 1024 * 1024):
    """Blocking (executor): rebuild the signature history from the last day of the request log so a restart does not
    forget which call types are short. Returns [(client, preview, outtok), ...] oldest first (local completions only)."""
    now = time.time() if now is None else now
    before = now if before is None else before
    rows = []
    for day_i in range(2):
        for raw in _read_lines_reverse(_history_day_file(now - day_i * 86400), per_file_cap):
            try:
                rec = json.loads(raw)
            except Exception:
                continue
            t, c, o = rec.get("t"), rec.get("client"), rec.get("outtok")
            if not isinstance(t, (int, float)) or t >= before or t < now - 86400 or not c or o is None:
                continue
            if rec.get("route") not in ("local", "held") or (rec.get("status") or 200) >= 400 or not rec.get("preview"):
                continue
            rows.append((t, c, rec["preview"], int(o)))
    rows.sort()
    return [(c, p, o) for _, c, p, o in rows]


def micro_history_restore(rows):
    for c, p, o in rows or []:
        micro_observe(c, p, o)
    return len(rows or [])


# _resolve_alias kinds a local engine can answer: "default" (model=estate, local-first), "builtin-local" (estate-local) and
# "native" (any other model name the engine serves). builtin-remote / custom-remote / disabled must exercise THEIR path.
PROBE_SYNTH_ALIAS_KINDS = frozenset({None, "", "default", "builtin-local", "native"})
_PROBE_RE = re.compile(r"^\s*reply with (?:the )?(?:single|one|only the) word:?\s*[\"'`]?([A-Za-z0-9_-]{1,24})[\"'`]?(?=\s|$)", re.I)
# What Hermes/the front door append to the probe: injected live state, truncated plugin output, the clock line.
_PROBE_CONTEXT_MARKERS = ("##", "[plugin hook", "(awareness", "now (")


def probe_word(body):
    """The word a trivial liveness probe asks for, or None. Matches only when the LAST user message is the probe
    sentence followed by nothing, or by recognised injected-context markers -- never "pong, then explain ...". """
    try:
        msgs = json.loads(body).get("messages") or []
        for m in reversed(msgs):
            if (m or {}).get("role") != "user":
                continue
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type", "text") == "text")
            if not isinstance(c, str):
                return None
            mt = _PROBE_RE.match(c)
            if not mt:
                return None
            rest = c[mt.end():].strip().lower()
            if rest and not rest.startswith(_PROBE_CONTEXT_MARKERS):
                return None
            return mt.group(1)
    except Exception:
        pass
    return None


def probe_synth_decision(word, alias_kind, local_ok_age_s, health_ok, offline_or_forced):
    """(serve_synthetic, why). Pure so it is testable. The probe asks "is the front door alive and generating?":
    real engine output within PROBE_FRESH_S answers that without spending a slot, a prefill or paid remote tokens."""
    global _PROBE_SEEN
    if not PROBE_SYNTH or not word:
        return False, "off"
    if alias_kind not in PROBE_SYNTH_ALIAS_KINDS:
        return False, "alias-%s" % alias_kind            # an estate-remote / custom probe must exercise ITS provider
    if offline_or_forced:
        return False, "window"
    if not health_ok:
        return False, "unhealthy"
    if local_ok_age_s > PROBE_FRESH_S:
        return False, "stale"
    _PROBE_SEEN += 1
    if PROBE_REAL_EVERY > 0 and _PROBE_SEEN % PROBE_REAL_EVERY == 0:
        _MICRO_STATS["probe_real"] += 1
        return False, "sampled-real"
    return True, "fresh"


def probe_synth_response(body, word, streaming):
    """OpenAI-shaped answer for the synthetic probe. Marked X-Shim-Synth so nothing can mistake it for generation."""
    try:
        model = str(json.loads(body).get("model") or "estate")
    except Exception:
        model = "estate"
    rid, created = "chatcmpl-synth-%s" % os.urandom(6).hex(), int(time.time())
    usage = {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1}
    hdr = {"X-Shim-Synth": "probe"}
    if not streaming:
        return web.json_response({
            "id": rid, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": word}, "finish_reason": "stop"}],
            "usage": usage}, headers=hdr)

    def chunk(delta, finish=None, u=None):
        d = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
             "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if u:
            d["usage"] = u
        return "data: " + json.dumps(d) + "\n\n"
    payload = (chunk({"role": "assistant", "content": ""}) + chunk({"content": word})
               + chunk({}, "stop", usage) + "data: [DONE]\n\n")
    return web.Response(body=payload.encode(), content_type="text/event-stream",
                        headers={**hdr, "Cache-Control": "no-cache"})


def predicted_occupancy_seconds(ptok, maxtok, concurrency=1):
    """Conservative local occupancy estimate used before admission.

    It intentionally applies only when the caller supplies a positive max_tokens value.  An
    omitted max_tokens request is already bounded by LOCAL_MAX_OUT on the local path and is too
    often a short tool call to classify from its nominal ceiling alone.
    """
    if not maxtok or PREDICTED_OCCUPANCY_SECS <= 0:
        return None
    prefill = (max(0, ptok) / max(1.0, prefill_tps())) * max(1, concurrency)
    decode = max(0, maxtok) / _latest_decode_tps()
    return round(prefill + decode, 3)


# ---- (b) GPU: extended nvidia-smi fields, off-loop refresh. _gpu_stats() (edited below)
# keeps its existing synchronous/cached contract for its existing callers unchanged. ----
_GPU_FIELDS = ("index,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total,"
               "temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,"
               "pcie.link.gen.current,fan.speed")
# Lane TL: the active clock-throttle bitmask (0x4 = software power cap, 0x20 = software thermal slowdown,
# 0x40 = hardware thermal slowdown, ...). Kept OUT of _GPU_FIELDS so a driver that rejects the field name cannot
# blank the whole query: _gpu_query_blocking tries the extended list first and, if nvidia-smi returns nothing,
# falls back to the base list once and remembers that (None = not probed yet).
_GPU_FIELDS_THROTTLE = _GPU_FIELDS + ",clocks_throttle_reasons.active"
_GPU_THROTTLE_OK = None


def _gpu_hex(x):
    x = (x or "").strip()
    if not x or x.startswith("["):
        return None
    try:
        return int(x, 16) if x.lower().startswith("0x") else int(x)
    except ValueError:
        return None


def _gpu_num(x, cast=float):
    x = (x or "").strip()
    if not x or x.startswith("["):     # "[Not Supported]" / "[N/A]" on cards without a sensor
        return None
    try:
        return cast(x)
    except ValueError:
        return None


def _gpu_query_blocking():
    """Blocking nvidia-smi call -- only ever invoked via run_in_executor (the sampler, below)
    or, once, as a same-thread fallback the very first time _gpu_stats() is called before the
    sampler has produced its first sample. See DESIGN.md (b) for why this used to be a hazard."""
    global _GPU_THROTTLE_OK
    data = []
    try:
        fields = _GPU_FIELDS_THROTTLE if _GPU_THROTTLE_OK is not False else _GPU_FIELDS
        out = subprocess.run(["nvidia-smi", "--query-gpu=" + fields,
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=2).stdout.strip()
        if not out and _GPU_THROTTLE_OK is None:
            # the extended list was rejected (or nvidia-smi is down): retry the base list once, and only
            # stop asking for the throttle field if the base list answers where the extended one did not.
            out = subprocess.run(["nvidia-smi", "--query-gpu=" + _GPU_FIELDS,
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=2).stdout.strip()
            if out:
                _GPU_THROTTLE_OK = False
        elif out and _GPU_THROTTLE_OK is None:
            _GPU_THROTTLE_OK = True
        for line in out.splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) < 13:
                continue
            data.append({
                "index": _gpu_num(p[0], int), "util": _gpu_num(p[1], int),
                "mem_util": _gpu_num(p[2], int), "used": _gpu_num(p[3], int),
                "free": _gpu_num(p[4], int), "total": _gpu_num(p[5], int),
                "temp_c": _gpu_num(p[6], int), "power_w": _gpu_num(p[7]),
                "power_limit_w": _gpu_num(p[8]), "clock_sm_mhz": _gpu_num(p[9], int),
                "clock_mem_mhz": _gpu_num(p[10], int), "pcie_gen": _gpu_num(p[11], int),
                "fan_pct": _gpu_num(p[12], int),
                "throttle_reasons": _gpu_hex(p[13]) if len(p) > 13 else None,
            })
    except Exception:
        pass
    return data


async def _gpu_stats_async():
    """Off-loop refresh, called every tick by the telemetry sampler -- keeps _gpu_cache warm so
    the existing synchronous _gpu_stats() (used on the hot /gateway/stats path) essentially never
    has to fall back to a direct blocking call itself."""
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, _gpu_query_blocking)
    _gpu_cache.update(at=time.time(), data=data)
    return data


# ---- (d) host telemetry: pure stdlib (/proc + statvfs). Deliberately NOT psutil, even though
# it happens to be importable in the shim's venv -- it's a transitive dependency of vLLM, never
# imported by the shim itself today, and this is a handful of lines either way. ----
def _host_stats_blocking():
    out = {}
    try:
        with open("/proc/stat") as f:
            vals = [int(x) for x in f.readline().split()[1:]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        total = sum(vals)
        if _cpu_prev["total"]:
            dt, di = total - _cpu_prev["total"], idle - _cpu_prev["idle"]
            out["cpu_pct"] = round(100.0 * (1 - di / dt), 1) if dt > 0 else None
        else:
            out["cpu_pct"] = None
        _cpu_prev.update(t=time.time(), total=total, idle=idle)
    except Exception:
        out["cpu_pct"] = None
    try:
        mem = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                mem[k] = int(v.strip().split()[0])
        out["ram_total_gb"] = round(mem.get("MemTotal", 0) / 1048576, 2)
        out["ram_avail_gb"] = round(mem.get("MemAvailable", 0) / 1048576, 2)
        out["ram_used_gb"] = round(out["ram_total_gb"] - out["ram_avail_gb"], 2)
    except Exception:
        pass
    try:
        st = os.statvfs("/")
        out["disk_free_gb"] = round(st.f_bavail * st.f_frsize / 1e9, 1)
        out["disk_total_gb"] = round(st.f_blocks * st.f_frsize / 1e9, 1)
    except Exception:
        pass
    try:
        st2 = os.statvfs(MODELS_DIR)     # defined later in the file (model-switch feature);
        out["models_disk_free_gb"] = round(st2.f_bavail * st2.f_frsize / 1e9, 1)   # safe: only
    except Exception:                                                              # read at
        pass                                                                       # runtime
    return out


# ---- (a) vLLM /metrics: hand-rolled Prometheus text-exposition parser. Deliberately not the
# `prometheus_client` library (same "not a declared shim dependency" reasoning as psutil above).
# Metric names are quoted verbatim from vllm/v1/metrics/loggers.py and
# vllm/v1/spec_decode/metrics.py -- see DESIGN.md (a) for the exact source lines. ----
_PROM_LINE_RE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{([^}]*)\})?\s+(\S+)\s*$')
_PROM_LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def _parse_prom_text(text):
    fam = collections.defaultdict(list)
    for line in text.splitlines():
        if not line or line[0] == '#':
            continue
        m = _PROM_LINE_RE.match(line)
        if not m:
            continue
        name, _grp, labelstr, raw = m.groups()
        try:
            val = float(raw)
        except ValueError:
            continue
        labels = dict(_PROM_LABEL_RE.findall(labelstr)) if labelstr else {}
        fam[name].append((labels, val))
    return fam


def _fv(fam, name, default=None):
    """First value of a gauge/counter family (fine here: one model, one engine index)."""
    v = fam.get(name)
    return v[0][1] if v else default


def _hist_points(fam, base_name):
    """[(le_float, cumulative_count), ...] ascending, or None if absent (e.g. spec-decode
    counters when speculative decoding is off -- they're only registered when configured)."""
    buckets = fam.get(base_name + "_bucket")
    if not buckets:
        return None
    pts = []
    for labels, cum in buckets:
        le = labels.get("le")
        if le is None:
            continue
        pts.append((float("inf") if le == "+Inf" else float(le), cum))
    pts.sort(key=lambda p: p[0])
    return pts or None


def _quantile_from_points(pts, q):
    """Linear interpolation within the bucket containing the target rank -- the same method
    Prometheus's histogram_quantile() uses. A value that actually falls in the +Inf overflow
    bucket is reported at the last finite edge (can't interpolate past it) -- see DESIGN.md (a)."""
    if not pts:
        return None
    total = pts[-1][1]
    if total <= 0:
        return None
    target = q * total
    lo_le, lo_cum = 0.0, 0.0
    for le, cum in pts:
        if cum >= target:
            if le == float("inf") or cum <= lo_cum:
                return lo_le
            return lo_le + (target - lo_cum) / (cum - lo_cum) * (le - lo_le)
        lo_le, lo_cum = le, cum
    return lo_le


def _hist_quantile(fam, base_name, q):
    """Cumulative (since engine start) quantile straight from one scrape."""
    return _quantile_from_points(_hist_points(fam, base_name), q)


def _hist_quantile_delta(cur_fam, prev_fam, base_name, q):
    """Windowed quantile: per-bucket delta between two scrapes (clamped >=0 so a counter reset
    on engine restart can't go negative) run through the same interpolation."""
    cur = _hist_points(cur_fam, base_name)
    if cur is None:
        return None
    if prev_fam is None:
        return _quantile_from_points(cur, q)     # first scrape ever -- nothing to diff against
    prev = dict(_hist_points(prev_fam, base_name) or [])
    running = 0.0
    fixed = []
    for le, c in cur:
        d = max(0.0, c - prev.get(le, 0.0))
        running = max(running, d)    # cumulative-by-construction; max() only guards edge cases
        fixed.append((le, running))
    return _quantile_from_points(fixed, q)


def _counter_rate(cur_fam, prev_fam, name, dt):
    """Δcounter/Δt between two scrapes. None if either scrape lacks the series or dt<=0."""
    if prev_fam is None or dt <= 0:
        return None
    cur, prev = _fv(cur_fam, name), _fv(prev_fam, name)
    if cur is None or prev is None:
        return None
    return max(0.0, cur - prev) / dt


async def _scrape_engine_metrics():
    """Async HTTP GET -- aiohttp is natively non-blocking, no executor needed (unlike the
    nvidia-smi subprocess above). Bounded timeout; any failure degrades this one field, never
    the proxy path. See DESIGN.md (h)."""
    prev_fam = _ENGINE_METRICS["families"] if _ENGINE_METRICS["ok"] else None
    prev_at = _ENGINE_METRICS["at"]
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TELEM_SCRAPE_TIMEOUT)) as s:
            async with s.get(_METRICS_URL) as r:
                text = await r.text()
                if r.status != 200:
                    raise RuntimeError(f"http {r.status}")
    except Exception as e:
        _ENGINE_METRICS.update(ok=False, err=str(e)[:200])
        age = round(time.time() - prev_at, 1) if prev_at else None
        return {"ok": False, "age_s": age, "err": str(e)[:200]}
    fam = _parse_prom_text(text)
    now = time.time()
    dt = (now - prev_at) if prev_fam is not None else 0
    _ENGINE_METRICS.update(ok=True, at=now, err=None, text=text, families=fam)
    capacity_note_scrape(fam, now)     # live KV pool / block size / engine generation (lane GW)
    await _poll_engine_models(now)     # live max_model_len -> context window (lane FX)
    kv = _fv(fam, "vllm:kv_cache_usage_perc")
    dq = _counter_rate(fam, prev_fam, "vllm:prefix_cache_queries_total", dt)
    dh = _counter_rate(fam, prev_fam, "vllm:prefix_cache_hits_total", dt)
    prefix_hit_rate = round(dh / dq, 4) if (dq and dh is not None and dq > 0) else None
    prompt_tok_s = _counter_rate(fam, prev_fam, "vllm:prompt_tokens_total", dt)
    gen_tok_s = _counter_rate(fam, prev_fam, "vllm:generation_tokens_total", dt)
    drafted = _counter_rate(fam, prev_fam, "vllm:spec_decode_num_draft_tokens_total", dt)
    accepted = _counter_rate(fam, prev_fam, "vllm:spec_decode_num_accepted_tokens_total", dt)
    spec_rate = round(accepted / drafted, 4) if (drafted and accepted is not None and drafted > 0) else None
    flow_meter_update(fam, prev_fam, dt, _fv(fam, "vllm:num_requests_running"), _fv(fam, "vllm:num_requests_waiting"),
                      prefix_hit_rate, gen_tok_s)
    return {
        "ok": True, "age_s": 0.0,
        "running": _fv(fam, "vllm:num_requests_running"), "waiting": _fv(fam, "vllm:num_requests_waiting"),
        "kv_cache_pct": round(kv * 100, 1) if kv is not None else None,
        "prefix_hit_rate": prefix_hit_rate,
        "prompt_tok_s": round(prompt_tok_s, 1) if prompt_tok_s is not None else None,
        "gen_tok_s": round(gen_tok_s, 1) if gen_tok_s is not None else None,
        "spec_decode_enabled": bool(fam.get("vllm:spec_decode_num_drafts_total")),
        "spec_accept_rate": spec_rate,
        "ttft_p50": _hist_quantile_delta(fam, prev_fam, "vllm:time_to_first_token_seconds", 0.50),
        "ttft_p95": _hist_quantile_delta(fam, prev_fam, "vllm:time_to_first_token_seconds", 0.95),
        "ttft_p50_cum": _hist_quantile(fam, "vllm:time_to_first_token_seconds", 0.50),
        "ttft_p95_cum": _hist_quantile(fam, "vllm:time_to_first_token_seconds", 0.95),
        "tpot_p50": _hist_quantile_delta(fam, prev_fam, "vllm:inter_token_latency_seconds", 0.50),
        "tpot_p95": _hist_quantile_delta(fam, prev_fam, "vllm:inter_token_latency_seconds", 0.95),
        "e2e_p50": _hist_quantile_delta(fam, prev_fam, "vllm:e2e_request_latency_seconds", 0.50),
        "e2e_p95": _hist_quantile_delta(fam, prev_fam, "vllm:e2e_request_latency_seconds", 0.95),
    }


# ---- Lane DB2: windowed latency quantiles from the gateway's OWN per-request telemetry ----
# The engine's TTFT / inter-token windowed quantile is a delta over one 2 s scrape interval, so it is null whenever no
# request finished inside it (non-null in about 20% of samples). The gateway sees every request, so it keeps one small
# tuple per finished streaming request and computes quantiles over a real 60 s (and 5 min, 15 min) window.
#   ttft = gateway-observed time from the upstream POST to the first streamed byte (engine queue + prefill + one hop);
#   itl  = decode_time / (output tokens - 1): mean gap between tokens of ONE request, only when the output token
#          count is exact (usage trailer) and the request produced at least 2 tokens.
# Only requests whose first byte was timed are counted (streaming); a non-streaming response has no ttft and is skipped.
LAT_WINDOWS_S = (60, 300, 900)
_REQ_LAT = collections.deque(maxlen=20000)       # (t_end, route, ttft_s, itl_s_or_None, class)


def lat_note_request(now, route, ttft, duration, waited, outtok, flow_class=None):
    """Called once per finished request from _telemetry_note_request(). Never raises."""
    try:
        if route not in ("local", "remote") or ttft is None or ttft < 0:
            return
        itl = None
        if outtok is not None and outtok >= 2 and duration is not None:
            dec = decompose_timing(waited, ttft, duration).get("decode_time")
            if dec is not None and dec > 0:
                v = dec / (outtok - 1)
                if v < 120:
                    itl = round(v, 5)
        _REQ_LAT.append((now, route, round(float(ttft), 4), itl, flow_class))
    except Exception as _e:
        _swallowed("lat_note_request", _e)


def _quantile(vals, q):
    """Linear-interpolated quantile of an unsorted list; None for an empty list."""
    v = sorted(vals)
    if not v:
        return None
    if len(v) == 1:
        return v[0]
    pos = q * (len(v) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def req_latency_windows(now=None, windows=LAT_WINDOWS_S):
    """{"60s": {"local": {"n","ttft_p50","ttft_p95","itl_n","itl_p50","itl_p95"}, "remote": {...}}, ...}"""
    now = time.time() if now is None else now
    rows = list(_REQ_LAT)
    out = {}
    for w in windows:
        inwin = [r for r in rows if now - r[0] <= w]
        d = {}
        for route in ("local", "remote"):
            rr = [r for r in inwin if r[1] == route]
            tt = [r[2] for r in rr]
            it = [r[3] for r in rr if r[3] is not None]
            f = lambda x, nd: None if x is None else round(x, nd)
            d[route] = {"n": len(tt),
                        "ttft_p50": f(_quantile(tt, 0.50), 3), "ttft_p95": f(_quantile(tt, 0.95), 3),
                        "itl_n": len(it),
                        "itl_p50": f(_quantile(it, 0.50), 4), "itl_p95": f(_quantile(it, 0.95), 4)}
        out["%ds" % w] = d
    return out


def req_latency_facts(now=None):
    return {"source": "gateway per-request telemetry: every streaming request that finished inside the window",
            "ttft": "seconds from the upstream POST to the first streamed byte (engine queue + prefill + one hop); excludes the "
                    "gateway's own admission wait",
            "itl": "mean seconds between tokens within one request = decode time / (output tokens - 1); exact token counts only",
            "windows": req_latency_windows(now)}


async def _take_sample():
    loop = asyncio.get_running_loop()
    gpu, host, engine = await asyncio.gather(
        _gpu_stats_async(),
        loop.run_in_executor(None, _host_stats_blocking),
        _scrape_engine_metrics(),
    )
    _now = time.time()
    try:
        _l60 = req_latency_windows(_now, (60,))["60s"]["local"]
    except Exception:
        _l60 = {}
    return {
        "t": _now, "gpu": gpu, "host": host, "engine": engine,
        "gateway": {"ttft60_p50": _l60.get("ttft_p50"), "ttft60_p95": _l60.get("ttft_p95"),
                    "itl60_p50": _l60.get("itl_p50"), "itl60_p95": _l60.get("itl_p95"), "lat60_n": _l60.get("n"),
                    "inflight": _inflight, "budget": effective_budget(), "waiting": _waiting,
                    "backoff_s": max(0, int(_backoff_until - time.time())),
                    "local_healthy": _health.get("ok", False),
                    "remote_share_pct": _remote_share_delta(),
                    "perf_breaker": perf_breaker_active(),
                    "perf_reason": _PERF_STATE.get("reason", "")},
    }


def _avg(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 3) if vals else None


def _downsample(samples):
    """Collapse ~TELEM_SLOW_EVERY fast points into one slow-ring point for the 24h/1min ring.
    Numeric leaves average; 'ok'/'local_healthy' booleans use 'true if any sample was true' (an
    averaged-away blip is the opposite of what a 24h trend view is for); everything else takes
    the latest sample's value."""
    if not samples:
        return None
    last = samples[-1]
    out = {"t": last["t"]}
    out["gateway"] = {
        "inflight": _avg(s["gateway"]["inflight"] for s in samples),
        "budget": last["gateway"]["budget"],
        "waiting": _avg(s["gateway"]["waiting"] for s in samples),
        "backoff_s": max(s["gateway"]["backoff_s"] for s in samples),
        "local_healthy": any(s["gateway"]["local_healthy"] for s in samples),
        "remote_share_pct": _avg(s["gateway"].get("remote_share_pct") for s in samples),
        "perf_breaker": any(s["gateway"].get("perf_breaker", False) for s in samples),
        "perf_reason": last["gateway"].get("perf_reason", ""),
        **{k: _avg(s["gateway"].get(k) for s in samples)
           for k in ("ttft60_p50", "ttft60_p95", "itl60_p50", "itl60_p95")},
        "lat60_n": last["gateway"].get("lat60_n"),
    }
    out["host"] = {k: _avg(s["host"].get(k) for s in samples) for k in
                   ("cpu_pct", "ram_used_gb", "ram_total_gb", "disk_free_gb", "models_disk_free_gb")}
    gpu_n = max((len(s["gpu"]) for s in samples), default=0)
    out["gpu"] = []
    for i in range(gpu_n):
        cards = [s["gpu"][i] for s in samples if i < len(s["gpu"])]
        if not cards:
            continue
        row = {k: _avg(c.get(k) for c in cards) for k in
               ("util", "mem_util", "used", "free", "total", "temp_c", "power_w",
                "power_limit_w", "clock_sm_mhz", "clock_mem_mhz", "fan_pct")}
        # Lane TL: a one-minute mean hides the peak and the throttle that explain "how hot did it get".
        temps = [c.get("temp_c") for c in cards if isinstance(c.get("temp_c"), (int, float))]
        row["temp_max_c"] = max(temps) if temps else None
        thr = [c.get("throttle_reasons") for c in cards if isinstance(c.get("throttle_reasons"), int)]
        row["throttle_reasons"] = None
        if thr:
            row["throttle_reasons"] = 0
            for bits in thr:
                row["throttle_reasons"] |= bits
        out["gpu"].append(row)
    ok_samples = [s["engine"] for s in samples if s["engine"].get("ok")]
    out["engine"] = {"ok": bool(ok_samples)}
    if ok_samples:
        for k in ("running", "waiting", "kv_cache_pct", "prefix_hit_rate", "prompt_tok_s",
                  "gen_tok_s", "spec_accept_rate", "ttft_p50", "ttft_p95", "tpot_p50",
                  "tpot_p95", "e2e_p50", "e2e_p95"):
            out["engine"][k] = _avg(e.get(k) for e in ok_samples)
    return out


# ---- Lane TL: persistent hardware/engine history (hw-YYYYMMDD.jsonl) -------------------------------------
# Kevin asked "how has the temperature been?" overnight and the only history was the in-memory 4 h slow ring.
# Every slow-ring point (one per ~minute) is now ALSO appended as ONE compact JSON line to
# TELEMETRY_DIR/hw-YYYYMMDD.jsonl (UTC day, exactly like requests-YYYYMMDD.jsonl), so the history survives a
# restart and spans days. No new cron/unit: the sampler that already builds the point enqueues it; the write is a
# few hundred bytes, done in the default executor, and any failure only increments a counter (the record is kept
# for the next attempt, bounded). Retention is the same sweep as requests-*.jsonl. On startup the slow ring is
# re-seeded from the last 24 h of these files so a restart does not blank the charts.
# Kill switch: SHIM_HW_HISTORY=0 (no writes, no seeding; the read endpoint still serves existing files).
HW_HISTORY            = os.environ.get("SHIM_HW_HISTORY", "1") != "0"
HW_JSONL_MAX_MB       = float(os.environ.get("SHIM_HW_JSONL_MAX_MB", "20"))
HW_QUEUE_MAX          = int(os.environ.get("SHIM_HW_QUEUE_MAX", "2880"))
HW_SEED_HOURS         = float(os.environ.get("SHIM_HW_SEED_HOURS", "24"))
HW_QUERY_CACHE_TTL    = float(os.environ.get("SHIM_HW_QUERY_CACHE_TTL", "10"))
HW_MAX_POINTS         = 2000
HW_HOT_C              = 83          # same "hot" line the dashboard draws
# nvidia-smi clocks_throttle_reasons bits: 0x4 sw power cap, 0x8 hw slowdown, 0x20 sw thermal, 0x40 hw thermal, 0x80 power brake
HW_THERMAL_MASK       = 0x08 | 0x20 | 0x40
HW_POWERCAP_MASK      = 0x04 | 0x80
_HW_PENDING = []
_HW_STATE = {"date": None, "path": None, "written": 0, "failed": 0, "dropped": 0, "dropped_cap": 0,
             "last_err": None, "seeded": 0}
_HW_TASK = None
_HW_QUERY_CACHE = {}

_HW_GPU_MAP = (("util", "util"), ("mem_util", "mem_util"), ("temp", "temp_c"), ("power", "power_w"),
               ("power_limit", "power_limit_w"), ("sm_clock", "clock_sm_mhz"), ("mem_clock", "clock_mem_mhz"),
               ("fan", "fan_pct"), ("vram_used", "used"), ("vram_total", "total"))
_HW_ENG_MAP = (("running", "running"), ("waiting", "waiting"), ("kv_pct", "kv_cache_pct"),
               ("prefix_hit", "prefix_hit_rate"), ("prefill_tps", "prompt_tok_s"),
               ("decode_tps", "gen_tok_s"), ("spec_accept", "spec_accept_rate"))
_HW_GW_MAP = (("inflight", "inflight"), ("waiting", "waiting"), ("remote_pct", "remote_share_pct"))


def _hw_r(v):
    return round(v, 2) if isinstance(v, float) else v


def _hw_record(ds):
    """One slow-ring point (see _downsample) -> the compact line stored on disk. None for an empty point."""
    if not ds or not isinstance(ds.get("t"), (int, float)):
        return None
    gpus = []
    for i, g in enumerate(ds.get("gpu") or []):
        row = {"i": i}
        for k, src in _HW_GPU_MAP:
            row[k] = _hw_r(g.get(src))
        row["temp_max"] = g.get("temp_max_c")
        row["throttle"] = g.get("throttle_reasons")
        gpus.append(row)
    eng = ds.get("engine") or {}
    gw = ds.get("gateway") or {}
    return {"t": round(ds["t"], 1), "gpu": gpus,
            "engine": ({k: _hw_r(eng.get(src)) for k, src in _HW_ENG_MAP} if eng.get("ok") else None),
            "gateway": {k: _hw_r(gw.get(src)) for k, src in _HW_GW_MAP}}


def _hw_to_slow(rec):
    """Inverse of _hw_record: a stored line -> a slow-ring point of the shape _downsample produces, so the
    dashboard's existing charts read a seeded point exactly like a live one. Unknown fields are None."""
    gw = rec.get("gateway") or {}
    gpu = []
    for g in rec.get("gpu") or []:
        row = {src: g.get(k) for k, src in _HW_GPU_MAP}
        row["index"] = g.get("i")
        row["free"] = (g["vram_total"] - g["vram_used"]) if isinstance(g.get("vram_total"), (int, float)) \
            and isinstance(g.get("vram_used"), (int, float)) else None
        row["temp_max_c"] = g.get("temp_max")
        row["throttle_reasons"] = g.get("throttle")
        gpu.append(row)
    e = rec.get("engine")
    engine = {"ok": bool(e)}
    if e:
        engine.update({src: e.get(k) for k, src in _HW_ENG_MAP})
    return {"t": rec["t"],
            "gateway": {"inflight": gw.get("inflight"), "budget": None, "waiting": gw.get("waiting"),
                        "backoff_s": 0, "local_healthy": True, "remote_share_pct": gw.get("remote_pct"),
                        "perf_breaker": False, "perf_reason": "", "seeded": True},
            "host": {}, "gpu": gpu, "engine": engine}


def _hw_day_path(epoch_seconds):
    return os.path.join(TELEMETRY_DIR, "hw-%s.jsonl" % time.strftime("%Y%m%d", time.gmtime(epoch_seconds)))


def _hw_append_blocking(recs):
    """Blocking (executor only). Appends each record to the file of ITS OWN UTC day, so a batch that straddles
    midnight lands correctly. Returns {"ok", "written", "dropped_cap", "err", "rotated"}; never raises."""
    res = {"ok": True, "written": 0, "dropped_cap": 0, "err": None, "rotated": False}
    try:
        os.makedirs(TELEMETRY_DIR, mode=0o700, exist_ok=True)
    except OSError:
        pass
    max_bytes = int(HW_JSONL_MAX_MB * 1024 * 1024)
    by_path = {}
    for r in recs:
        try:
            by_path.setdefault(_hw_day_path(r["t"]), []).append(json.dumps(r, separators=(",", ":")) + "\n")
        except Exception:
            continue   # one malformed record must not lose the rest
    for path, lines in by_path.items():
        try:
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
                res["rotated"] = True    # a brand-new day file: time for the retention sweep
            keep = []
            for ln in lines:
                n = len(ln.encode("utf-8"))
                if size + n > max_bytes:
                    res["dropped_cap"] += 1
                    continue
                keep.append(ln)
                size += n
            if keep:
                fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                with os.fdopen(fd, "a", encoding="utf-8") as f:
                    f.write("".join(keep))
                res["written"] += len(keep)
            _HW_STATE["path"] = path
        except OSError as e:
            res["ok"] = False
            res["err"] = str(e)
            log.warning("hw history: write %s failed: %s", path, e)
    if res["rotated"]:
        _telemetry_retention_sweep()
    return res


async def _hw_flush():
    recs = list(_HW_PENDING)
    _HW_PENDING.clear()
    if not recs:
        return
    try:
        res = await asyncio.get_running_loop().run_in_executor(None, _hw_append_blocking, recs)
    except asyncio.CancelledError:
        _HW_PENDING[:0] = recs
        raise
    except Exception as e:
        res = {"ok": False, "written": 0, "dropped_cap": 0, "err": str(e)}
    _HW_STATE["written"] += res["written"]
    _HW_STATE["dropped_cap"] += res["dropped_cap"]
    if res["ok"]:
        _HW_STATE["last_err"] = None
    else:
        # keep the records for the next minute's attempt (bounded; the oldest are shed first)
        _HW_STATE["failed"] += 1
        _HW_STATE["last_err"] = res["err"]
        _HW_PENDING[:0] = recs
        over = len(_HW_PENDING) - HW_QUEUE_MAX
        if over > 0:
            del _HW_PENDING[:over]
            _HW_STATE["dropped"] += over


def _hw_enqueue(ds):
    """Called from the sampler once per slow-ring point. Event-loop only; no I/O here."""
    global _HW_TASK
    if not HW_HISTORY:
        return
    try:
        rec = _hw_record(ds)
        if rec is None:
            return
        _HW_PENDING.append(rec)
        if len(_HW_PENDING) > HW_QUEUE_MAX:
            del _HW_PENDING[:len(_HW_PENDING) - HW_QUEUE_MAX]
            _HW_STATE["dropped"] += 1
        if _HW_TASK is None or _HW_TASK.done():
            _HW_TASK = asyncio.get_running_loop().create_task(_hw_flush())
    except Exception as e:
        log.warning("hw history enqueue: %s", e)


def hw_read_blocking(t0, t1, telemetry_dir=None):
    """Stored records with t0 <= t <= t1, oldest first, from the UTC-day files that cover the range. Executor only.
    Unreadable files and malformed lines are skipped; returns (records, files_read)."""
    tdir = telemetry_dir or TELEMETRY_DIR
    out, files = [], 0
    day = int(t0 // 86400)
    while day * 86400 <= t1:
        path = os.path.join(tdir, "hw-%s.jsonl" % time.strftime("%Y%m%d", time.gmtime(day * 86400)))
        day += 1
        try:
            with open(path, "r", encoding="utf-8") as f:
                files += 1
                for line in f:
                    try:
                        r = json.loads(line)
                        if t0 <= r["t"] <= t1:
                            out.append(r)
                    except Exception:
                        continue
        except OSError:
            continue
    out.sort(key=lambda r: r["t"])
    return out, files


def _hw_agg(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 2) if vals else None


def _hw_bucket(recs):
    """Collapse stored records of one time bucket: means, except temp_max (max) and throttle (OR of bits)."""
    out = {"t": round(sum(r["t"] for r in recs) / len(recs), 1), "n": len(recs), "gpu": []}
    for i in range(max((len(r.get("gpu") or []) for r in recs), default=0)):
        cards = [r["gpu"][i] for r in recs if i < len(r.get("gpu") or [])]
        row = {"i": i}
        for k, _src in _HW_GPU_MAP:
            row[k] = _hw_agg(c.get(k) for c in cards)
        tm = [c.get("temp_max") for c in cards if isinstance(c.get("temp_max"), (int, float))]
        row["temp_max"] = max(tm) if tm else None
        th = [c.get("throttle") for c in cards if isinstance(c.get("throttle"), int)]
        row["throttle"] = None
        if th:
            row["throttle"] = 0
            for b in th:
                row["throttle"] |= b
        out["gpu"].append(row)
    es = [r["engine"] for r in recs if r.get("engine")]
    out["engine"] = {k: _hw_agg(e.get(k) for e in es) for k, _s in _HW_ENG_MAP} if es else None
    out["gateway"] = {k: _hw_agg((r.get("gateway") or {}).get(k) for r in recs) for k, _s in _HW_GW_MAP}
    return out


def hw_summary(recs):
    """Per-GPU headline numbers over the raw (one-minute) records: how hot, how long hot, how long throttled."""
    out = []
    for i in range(max((len(r.get("gpu") or []) for r in recs), default=0)):
        cards = [r["gpu"][i] for r in recs if i < len(r.get("gpu") or [])]
        temps = [c["temp"] for c in cards if isinstance(c.get("temp"), (int, float))]
        peaks = [c["temp_max"] for c in cards if isinstance(c.get("temp_max"), (int, float))]
        thr = [c["throttle"] for c in cards if isinstance(c.get("throttle"), int)]
        out.append({
            "i": i, "minutes": len(cards),
            "temp_min": min(temps) if temps else None,
            "temp_avg": _hw_agg(temps),
            "temp_max": max(peaks or temps) if (peaks or temps) else None,
            "hot_minutes": sum(1 for c in cards if (c.get("temp_max") or c.get("temp") or 0) >= HW_HOT_C),
            "thermal_throttle_minutes": sum(1 for b in thr if b & HW_THERMAL_MASK),
            "power_cap_minutes": sum(1 for b in thr if b & HW_POWERCAP_MASK),
            "throttle_known": bool(thr),
            "power_avg": _hw_agg(c.get("power") for c in cards),
            "power_max": max((c["power"] for c in cards if isinstance(c.get("power"), (int, float))), default=None),
            "sm_clock_avg": _hw_agg(c.get("sm_clock") for c in cards),
            "sm_clock_min": min((c["sm_clock"] for c in cards if isinstance(c.get("sm_clock"), (int, float))), default=None),
        })
    return out


def hw_history(hours, step, now=None, telemetry_dir=None):
    """Blocking. The body of /gateway/telemetry/history: downsampled points + per-GPU summary for the last `hours`."""
    now = time.time() if now is None else now
    max_h = max(1, TELEMETRY_RETENTION_DAYS * 24) if TELEMETRY_RETENTION_DAYS > 0 else 24 * 365
    hours = min(max(hours, 1 / 60), max_h)
    t0 = now - hours * 3600
    # at most HW_MAX_POINTS points and never finer than the one-minute resolution that is stored
    step = max(60.0, float(step) if step else hours * 3600 / 360, hours * 3600 / HW_MAX_POINTS)
    recs, files = hw_read_blocking(t0, now, telemetry_dir)
    buckets, cur, cur_k = [], [], None
    for r in recs:
        k = int((r["t"] - t0) // step)
        if cur and k != cur_k:
            buckets.append(_hw_bucket(cur))
            cur = []
        cur.append(r)
        cur_k = k
    if cur:
        buckets.append(_hw_bucket(cur))
    return {"at": now, "hours": hours, "step_s": step, "from": t0, "to": now,
            "files": files, "raw_samples": len(recs),
            "first_t": recs[0]["t"] if recs else None, "last_t": recs[-1]["t"] if recs else None,
            "retention_days": TELEMETRY_RETENTION_DAYS,
            "fields": {"temp": "deg C (mean over the bucket)", "temp_max": "deg C (peak minute-mean in the bucket)",
                       "power": "W", "power_limit": "W", "sm_clock": "MHz", "mem_clock": "MHz", "fan": "% of max",
                       "util": "% busy", "vram_used": "MiB", "throttle": "OR of nvidia-smi clocks_throttle_reasons bits"},
            "summary": {"gpu": hw_summary(recs)}, "points": buckets}


async def gateway_telemetry_history(request):
    """/gateway/telemetry/history?hours=N&step=S -- persisted hardware/engine history, downsampled. Read-only."""
    try:
        hours = float(request.query.get("hours", "24"))
    except ValueError:
        hours = 24.0
    try:
        step = float(request.query.get("step", "0"))
    except ValueError:
        step = 0.0
    if not (hours == hours and step == step) or hours in (float("inf"), float("-inf")):
        return web.json_response({"error": "hours and step must be finite numbers"}, status=400)
    now = time.time()
    key = (round(hours, 3), round(step, 1))
    hit = _HW_QUERY_CACHE.get(key)
    if hit and now - hit[0] < HW_QUERY_CACHE_TTL:
        return web.json_response(hit[1])
    try:
        body = await asyncio.get_running_loop().run_in_executor(None, hw_history, hours, step, now)
    except Exception as e:
        log.warning("telemetry history: %s", e)
        return web.json_response({"error": "history unreadable: %s" % e}, status=500)
    body["store"] = hw_store_state()
    if len(_HW_QUERY_CACHE) > 16:
        _HW_QUERY_CACHE.clear()
    _HW_QUERY_CACHE[key] = (now, body)
    return web.json_response(body)


def hw_store_state():
    return {"enabled": HW_HISTORY, "dir": TELEMETRY_DIR, "current_file": _HW_STATE["path"],
            "written": _HW_STATE["written"], "failed_flushes": _HW_STATE["failed"],
            "pending": len(_HW_PENDING), "dropped": _HW_STATE["dropped"],
            "dropped_cap": _HW_STATE["dropped_cap"], "last_err": _HW_STATE["last_err"],
            "seeded_points": _HW_STATE["seeded"], "max_mb_per_day": HW_JSONL_MAX_MB}


async def _hw_seed_slow():
    """Startup: refill the (empty) slow ring from the last HW_SEED_HOURS of stored lines so a restart does not
    blank the 24 h charts. Never blocks or fails startup."""
    if not HW_HISTORY or _TELEM_SLOW:
        return
    try:
        now = time.time()
        recs, _files = await asyncio.get_running_loop().run_in_executor(
            None, hw_read_blocking, now - HW_SEED_HOURS * 3600, now)
        pts = [_hw_to_slow(r) for r in recs[-_TELEM_SLOW.maxlen:]]
        if pts and not _TELEM_SLOW:
            _TELEM_SLOW.extend(pts)
            _HW_STATE["seeded"] = len(pts)
        log.info("telemetry slow ring seeded: %d points from hw-*.jsonl", len(pts))
    except Exception as e:
        log.warning("telemetry slow-ring seed failed: %s", e)


async def _telemetry_sampler():
    global _TELEM_TICK
    while True:
        await asyncio.sleep(TELEM_SAMPLE_SECS)
        try:
            sample = await _take_sample()
            _TELEM_FAST.append(sample)
            _TELEM_WINDOW.append(sample)
            _update_perf_breaker(sample)
            _TELEM_TICK += 1
            if _TELEM_TICK % 5 == 0:
                _offline_reap()
                flow_note_mode()       # CF: a capacity-mode change is an event
            if _TELEM_TICK % max(1, TELEM_SLOW_EVERY) == 0:
                ds = _downsample(_TELEM_WINDOW)
                if ds:
                    _TELEM_SLOW.append(ds)
                    _hw_enqueue(ds)        # Lane TL: persist the minute point (hw-YYYYMMDD.jsonl)
                _TELEM_WINDOW.clear()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("telemetry sampler: %s", e)


# R2 v5 (2026-09-25): ACTUAL provider prices, per token type, from Kevin's DeepSeek billing
# export for 2026-09-25 (model deepseek-flash): input cache hit $0.003/Mtok, input cache miss
# $0.15/Mtok, output $0.60/Mtok. 96.4% of today's input tokens were cache hits, so the
# no-cache list estimate below overstated the real bill 12.7x ($41.90 vs $3.31 for
# 00:00-13:00). Settlement, the dashboard's per-client cost and the telemetry use these
# prices on the provider's own usage counts (prompt_cache_hit_tokens / prompt_cache_miss_tokens
# / completion_tokens). DeepSeek doubles all three rates in its weekday UTC peak windows;
# record peak/off-peak at forwarding so settlement and telemetry agree even across a boundary.
# Per-model overrides: SHIM_REMOTE_PRICES_JSON, e.g.
# {"deepseek-flash": {"cache_hit": 0.003, "cache_miss": 0.15, "output": 0.6,
#                     "cache_hit_peak": 0.006, "cache_miss_peak": 0.3, "output_peak": 1.2}}
# ($/Mtok). A model without explicit peak fields defaults to twice its off-peak rates.
REMOTE_PRICE_CACHE_HIT_PER_MTOK = float(os.environ.get("SHIM_REMOTE_PRICE_CACHE_HIT_PER_MTOK", "0.003"))
REMOTE_PRICE_CACHE_MISS_PER_MTOK = float(os.environ.get("SHIM_REMOTE_PRICE_CACHE_MISS_PER_MTOK", "0.15"))
REMOTE_PRICE_OUTPUT_PER_MTOK = float(os.environ.get("SHIM_REMOTE_PRICE_OUTPUT_PER_MTOK", "0.6"))
REMOTE_PRICE_CACHE_HIT_PER_MTOK_PEAK = float(os.environ.get("SHIM_REMOTE_PRICE_CACHE_HIT_PER_MTOK_PEAK", "0.006"))
REMOTE_PRICE_CACHE_MISS_PER_MTOK_PEAK = float(os.environ.get("SHIM_REMOTE_PRICE_CACHE_MISS_PER_MTOK_PEAK", "0.3"))
REMOTE_PRICE_OUTPUT_PER_MTOK_PEAK = float(os.environ.get("SHIM_REMOTE_PRICE_OUTPUT_PER_MTOK_PEAK", "1.2"))
REMOTE_PRICES_JSON = os.environ.get("SHIM_REMOTE_PRICES_JSON", "")
_CFG.update({
    "SHIM_REMOTE_PRICE_CACHE_HIT_PER_MTOK": ("REMOTE_PRICE_CACHE_HIT_PER_MTOK", float),
    "SHIM_REMOTE_PRICE_CACHE_MISS_PER_MTOK": ("REMOTE_PRICE_CACHE_MISS_PER_MTOK", float),
    "SHIM_REMOTE_PRICE_OUTPUT_PER_MTOK": ("REMOTE_PRICE_OUTPUT_PER_MTOK", float),
    "SHIM_REMOTE_PRICE_CACHE_HIT_PER_MTOK_PEAK": ("REMOTE_PRICE_CACHE_HIT_PER_MTOK_PEAK", float),
    "SHIM_REMOTE_PRICE_CACHE_MISS_PER_MTOK_PEAK": ("REMOTE_PRICE_CACHE_MISS_PER_MTOK_PEAK", float),
    "SHIM_REMOTE_PRICE_OUTPUT_PER_MTOK_PEAK": ("REMOTE_PRICE_OUTPUT_PER_MTOK_PEAK", float),
    "SHIM_REMOTE_PRICES_JSON": ("REMOTE_PRICES_JSON", str),
})


# ---------------- dated pricing table (lane SL, 2026-10-02) ----------------
# The price of a remote token is a FUNCTION OF (model, token type, time), read from a small
# dated table -- remote-pricing.json beside this file, hot-reloaded when it changes -- not from
# constants. Provenance (source_url, fetched) travels with the numbers; refresh with
# remote_pricing_refresh.py (reports drift as a fact; --apply rewrites the file). If the file is
# absent or invalid the embedded copy below (identical to the committed file; a test pins that)
# is used and the status says so. Source: https://api-docs.deepseek.com/quick_start/pricing --
# off-peak is half of peak; PEAK = 01:00-04:00 and 06:00-10:00 UTC, Monday-Friday, EXCLUDING
# Chinese public holidays (those are off-peak all day). Legacy model names are billed at Flash.
# 2026-10-02 was inside China's National Day holiday (Oct 1-7), so every hour of Kevin's export
# carries the off-peak price even though 07:00-09:00 UTC on a Friday looks like peak.
PRICING_FILE = os.environ.get("SHIM_PRICING_FILE") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "remote-pricing.json")
PRICING_STALE_DAYS = int(os.environ.get("SHIM_PRICING_STALE_DAYS", "30"))
_PRICE_TYPES = ("cache_hit", "cache_miss", "output")
_PRICING_DEFAULT = {
    "schema": 1,
    "provider": "deepseek",
    "unit": "usd_per_mtok",
    "source_url": "https://api-docs.deepseek.com/quick_start/pricing",
    "fetched": "2026-10-02",
    "peak": {
        "tz": "UTC",
        "windows": [[1, 4], [6, 10]],
        "weekdays": [0, 1, 2, 3, 4],
        "off_peak_dates": [["2026-01-01", "2026-01-03"], ["2026-02-15", "2026-02-23"],
                           ["2026-04-04", "2026-04-06"], ["2026-05-01", "2026-05-05"],
                           ["2026-06-19", "2026-06-21"], ["2026-09-25", "2026-09-27"],
                           ["2026-10-01", "2026-10-07"]],
        "holidays_through": "2026-12-31",
        "holiday_source": "https://www.china-briefing.com/news/china-2026-public-holiday-schedule/",
        "note": "Chinese public holidays are off-peak all day (provider page, footnote 2). "
                "Make-up working Saturdays/Sundays are treated as weekend (off-peak). "
                "Dates after holidays_through are treated as ordinary weekdays (the dearer side).",
    },
    "models": {
        "deepseek-flash": {
            "aliases": ["deepseek-v4-flash", "deepseek-v4-flash-vision-exp"],
            "off_peak": {"cache_hit": 0.003, "cache_miss": 0.15, "output": 0.6},
            "peak": {"cache_hit": 0.006, "cache_miss": 0.3, "output": 1.2},
        },
        "deepseek-v4-pro": {
            "aliases": [],
            "off_peak": {"cache_hit": 0.022, "cache_miss": 0.66, "output": 1.98},
            "peak": {"cache_hit": 0.044, "cache_miss": 1.32, "output": 3.96},
        },
    },
}
_PRICING_CACHE = {"checked": 0.0, "mtime": None, "table": None, "source": "embedded", "error": None}


def _pricing_validate(raw):
    """A raw table dict -> the normalised table, or ValueError naming the first problem."""
    if not isinstance(raw, dict) or raw.get("schema") != 1:
        raise ValueError("pricing table: schema must be 1")
    pk = raw.get("peak")
    if not isinstance(pk, dict):
        raise ValueError("pricing table: peak section missing")
    windows = []
    for w in pk.get("windows") or []:
        a, b = float(w[0]), float(w[1])
        if not (0 <= a < b <= 24):
            raise ValueError(f"pricing table: bad peak window {w}")
        windows.append((a, b))
    weekdays = {int(d) for d in pk.get("weekdays") or []}
    if not weekdays <= set(range(7)):
        raise ValueError("pricing table: weekdays must be 0..6 (Monday=0)")
    off_dates = []
    for pair in pk.get("off_peak_dates") or []:
        lo = datetime.date.fromisoformat(pair[0]).toordinal()
        hi = datetime.date.fromisoformat(pair[-1]).toordinal()
        if hi < lo:
            raise ValueError(f"pricing table: off-peak range reversed {pair}")
        off_dates.append((lo, hi))
    through = pk.get("holidays_through")
    through = datetime.date.fromisoformat(through).toordinal() if through else None
    models, alias = {}, {}
    for name, m in (raw.get("models") or {}).items():
        entry = {}
        for period in ("off_peak", "peak"):
            row = (m or {}).get(period)
            if not isinstance(row, dict):
                raise ValueError(f"pricing table: {name}.{period} missing")
            vals = {}
            for t in _PRICE_TYPES:
                v = float(row[t])
                if not (v >= 0) or v != v or v == float("inf"):
                    raise ValueError(f"pricing table: {name}.{period}.{t} must be a finite price >= 0")
                vals[t] = v
            entry[period] = vals
        models[str(name).lower()] = entry
        alias[str(name).lower()] = str(name).lower()
        for a in (m or {}).get("aliases") or []:
            alias[str(a).lower()] = str(name).lower()
    if not models:
        raise ValueError("pricing table: no models")
    return {"raw": raw, "windows": windows, "weekdays": weekdays, "off_dates": off_dates,
            "holidays_through": through, "models": models, "alias": alias,
            "fetched": raw.get("fetched"), "source_url": raw.get("source_url")}


def _pricing_table(force=False):
    """The live normalised pricing table. Re-stats the file at most every 30 s; a file that is
    missing or invalid never breaks pricing (the embedded copy, or the last good file, is used)."""
    c = _PRICING_CACHE
    now = time.monotonic()
    if c["table"] is not None and not force and now - c["checked"] < 30:
        return c["table"]
    c["checked"] = now
    try:
        st = os.stat(PRICING_FILE)
        if c["table"] is not None and c["source"] == "file" and c["mtime"] == st.st_mtime_ns:
            return c["table"]
        with open(PRICING_FILE, encoding="utf-8") as fh:
            table = _pricing_validate(json.load(fh))
        c.update(table=table, mtime=st.st_mtime_ns, source="file", error=None)
        log.info("pricing table loaded from %s (fetched %s)", PRICING_FILE, table["fetched"])
    except FileNotFoundError:
        if c["table"] is None or c["source"] == "file":
            c.update(table=_pricing_validate(_PRICING_DEFAULT), mtime=None, source="embedded", error=None)
    except Exception as exc:
        c["error"] = f"{type(exc).__name__}: {exc}"[:200]
        log.error("pricing table %s unusable (%s): keeping %s", PRICING_FILE, exc,
                  "the last good file" if c["source"] == "file" else "the embedded table")
        if c["table"] is None:
            c.update(table=_pricing_validate(_PRICING_DEFAULT), mtime=None, source="embedded")
    return c["table"]


def _pricing_is_peak(ts=None):
    t = _pricing_table()
    g = time.gmtime(time.time() if ts is None else ts)
    if g.tm_wday not in t["weekdays"]:
        return False
    day = datetime.date(g.tm_year, g.tm_mon, g.tm_mday).toordinal()
    if any(lo <= day <= hi for lo, hi in t["off_dates"]):
        return False
    hour = g.tm_hour + g.tm_min / 60.0
    return any(a <= hour < b for a, b in t["windows"])


def remote_price(model, token_type, ts=None, *, peak=None):
    """$/Mtok for (model, token type, time) from the dated table; None if the table does not
    know the model. token_type: cache_hit | cache_miss | output. `peak` overrides the clock."""
    if token_type not in _PRICE_TYPES:
        raise ValueError(f"unknown token type {token_type!r}")
    t = _pricing_table()
    entry = t["models"].get(t["alias"].get(str(model or "").lower(), ""))
    if entry is None:
        return None
    if peak is None:
        peak = _pricing_is_peak(ts)
    return entry["peak" if peak else "off_peak"][token_type]


def pricing_status(now=None):
    """Provenance + freshness of the live table (an input to Halo's reconciliation fact)."""
    now = time.time() if now is None else now
    t = _pricing_table()
    today = datetime.datetime.fromtimestamp(now, datetime.timezone.utc).date()
    age = None
    try:
        age = (today - datetime.date.fromisoformat(str(t["fetched"]))).days
    except Exception:
        pass
    beyond = bool(t["holidays_through"] and today.toordinal() > t["holidays_through"])
    return {"source": _PRICING_CACHE["source"], "path": PRICING_FILE, "fetched": t["fetched"],
            "source_url": t["source_url"], "age_days": age, "holidays_through":
            datetime.date.fromordinal(t["holidays_through"]).isoformat() if t["holidays_through"] else None,
            "period_now": "peak" if _pricing_is_peak(now) else "off_peak",
            "models": sorted(t["models"]), "error": _PRICING_CACHE["error"],
            "stale": bool(beyond or age is None or age > PRICING_STALE_DAYS),
            "stale_reason": ("holiday calendar ended" if beyond else
                             "table older than %d days" % PRICING_STALE_DAYS if (age is None or age > PRICING_STALE_DAYS)
                             else None)}


def _remote_prices(model=None, *, peak=None, when=None):
    """($/Mtok cache hit, cache miss, output) for a remote model at a time. Precedence: an
    operator override row in SHIM_REMOTE_PRICES_JSON, then the dated table, then the legacy knobs."""
    model = model or REMOTE_MODEL
    try:
        row = (json.loads(REMOTE_PRICES_JSON) if REMOTE_PRICES_JSON else {}).get(str(model or "")) or {}
    except Exception:
        row = {}
    if peak is None:
        peak = is_peak() if when is None else is_peak(when)
    if not row:
        try:
            got = tuple(remote_price(model, t, peak=peak) for t in _PRICE_TYPES)
            if None not in got:
                return got
        except Exception as exc:
            log.warning("pricing table lookup failed (%s): legacy price knobs used", exc)
    base = (float(row.get("cache_hit", REMOTE_PRICE_CACHE_HIT_PER_MTOK)),
            float(row.get("cache_miss", REMOTE_PRICE_CACHE_MISS_PER_MTOK)),
            float(row.get("output", REMOTE_PRICE_OUTPUT_PER_MTOK)))
    if not peak:
        return base
    return (float(row.get("cache_hit_peak", REMOTE_PRICE_CACHE_HIT_PER_MTOK_PEAK if not row else base[0] * 2)),
            float(row.get("cache_miss_peak", REMOTE_PRICE_CACHE_MISS_PER_MTOK_PEAK if not row else base[1] * 2)),
            float(row.get("output_peak", REMOTE_PRICE_OUTPUT_PER_MTOK_PEAK if not row else base[2] * 2)))


def _remote_hold_rates(model=None):
    """($/Mtok cache-miss input, output) at the DEARER period -- the rates an upfront hold must
    assume because nobody knows yet what the call will hit or when it will be billed."""
    return (max(_remote_prices(model, peak=False)[1], _remote_prices(model, peak=True)[1]),
            max(_remote_prices(model, peak=False)[2], _remote_prices(model, peak=True)[2]))


def _remote_cost_actual(model, cache_hit, cache_miss, outtok, *, peak=None, when=None):
    hit_p, miss_p, out_p = _remote_prices(model, peak=peak, when=when)
    return round((max(0, cache_hit or 0) / 1e6) * hit_p + (max(0, cache_miss or 0) / 1e6) * miss_p
                 + (max(0, outtok or 0) / 1e6) * out_p, 9)


# Calibration for the one case that has no usage to read (a client that hung up, a cut stream):
# the expected cache-hit share of a prompt, learned token-weighted from the last ~200 calls that
# DID report usage. Below 50 observations nothing is assumed (all-miss, the dearer bound).
_CACHE_RATIO = {"hit": 0.0, "miss": 0.0, "n": 0}
CACHE_RATIO_DECAY = 0.995
CACHE_RATIO_MIN_SAMPLES = 50


def _cache_ratio_note(hit, miss):
    """Fold one usage-bearing call into the rolling hit/miss prompt-token ratio."""
    c = _CACHE_RATIO
    c["hit"] = c["hit"] * CACHE_RATIO_DECAY + max(0, hit or 0)
    c["miss"] = c["miss"] * CACHE_RATIO_DECAY + max(0, miss or 0)
    c["n"] += 1


def _cache_hit_share():
    c = _CACHE_RATIO
    tot = c["hit"] + c["miss"]
    return (c["hit"] / tot) if (c["n"] >= CACHE_RATIO_MIN_SAMPLES and tot > 0) else 0.0


def _request_remote_cost(info):
    """(usd, basis) for one finished remote request -- the ONE pricing function behind
    settlement, the per-client dashboard and the telemetry. Bases:
      'actual'   the provider reported cache hit/miss counts (priced per type, at the time sent);
      'usage'    it reported only prompt_tokens (priced as cache misses -- an upper bound);
      'failed'   the call returned an HTTP error / never produced a completion: $0, the
                 provider does not bill it (a 402 Insufficient Balance cost nothing);
      'estimate' the call was answered or begun but no usage arrived (client hung up, stream
                 cut): prompt split by the recently observed cache-hit share (all-miss until
                 enough calls reported usage) + streamed chunks as output tokens.
    Failure is decided by `remote_outcome` (set where the relay result is known) only when no
    provider usage is present -- usage always wins."""
    out = info.get("outtok") if info.get("outtok") is not None else info.get("outtok_lb")
    model = info.get("remote_model") or REMOTE_MODEL
    prices = info.get("remote_prices")                 # a metered custom endpoint's own prices
    hit, miss = info.get("remote_cache_hit"), info.get("remote_cache_miss")
    peak, when = info.get("remote_price_peak"), info.get("remote_sent_at")

    def priced(h, m):
        if prices:
            return round((max(0, h or 0) * prices[0] + max(0, m or 0) * prices[1]
                          + max(0, out or 0) * prices[2]) / 1e6, 9)
        return _remote_cost_actual(model, h, m, out, peak=peak, when=when)
    if hit is not None and miss is not None:
        return priced(hit, miss), "actual"
    if info.get("ptok_exact") is not None:
        return priced(0, info.get("ptok_exact")), "usage"
    if info.get("remote_outcome") == "failed":
        return 0.0, "failed"
    ptok = int(info.get("ptok") or 0)
    share = _cache_hit_share()                         # 0.0 until enough calls have reported usage
    hit_est = int(ptok * share)
    return priced(hit_est, ptok - hit_est), "estimate"


def _remote_cost_estimate(ptok, outtok, model=None, when=None):
    """est. remote cost for an unmetered/unknown call: all-miss input + output at the table's
    price for the time (the dearer-than-actual side; the real bill is mostly cache hits)."""
    if not ptok and not outtok:
        return 0.0
    return _remote_cost_actual(model or REMOTE_MODEL, 0, ptok, outtok, when=when)


# ---------------- remote spend authority (R2, 2026-09-25; v2 after Terra's 12:38 review) ----------------
# ONE atomic daily cap for EVERY paid call to the gateway's remote provider. Before R2 the
# estate's "$25/day DeepSeek cap" was two unconnected SpendGuard files (brain on .10 at $25,
# Halo on HNET00 at a $5 default) and neither saw gateway traffic. The gateway prices every
# remote request, so the cap lives here, checked and reserved atomically under one lock.
#
#   * EVERY paid route holds first -- _forward_remote() to the default (metered) provider,
#     whatever the reason (estate-remote alias, route intent, overflow, size, big-prompt,
#     forced window, local-down): the request's upper-bound cost (prompt + max_tokens at the
#     higher of normal/peak rates) is held before anything is sent, from its reservation first,
#     then from the shared pool. No room -> 429 spend_cap_exhausted, nothing forwarded. There
#     is NO uncapped emergency route. (Custom aliases target other, separately configured
#     providers and are outside this cap; today the only one is openrouter/free.)
#   * reservation -- POST /gateway/spend/reserve {key, amount, ttl_s}, authenticated with a
#     dedicated spend-client credential (X-Spend-Token; sha256 hashes in SPEND_CLIENTS_FILE,
#     mode 0600 -- never the dashboard admin token). The owner IS the authenticated client;
#     the reservation is bound to the reserving source IP, so its id (which rides on the model
#     name `estate-remote.<id>`) is useless from anywhere else. Idempotent on (owner, key).
#   * settle -- handle_completions' finally replaces the hold with the gateway's own price.
#   * finalize -- POST /gateway/spend/finalize, same credential, owner-only, idempotent.
#   * durability -- every mutation is written to a 0600 temp file, fsynced, atomically
#     replaced, and the directory fsynced before the call succeeds; a failed write rolls the
#     in-memory state back and refuses (the hold/reserve is never granted undurably).
#   * corruption -- an unreadable ledger is copied aside (never overwritten) and ALL paid
#     forwarding and every mutation are refused, observe mode included, until an operator
#     runs the governed recovery (POST /gateway/spend/recover, admin token).
#   * day completeness -- the ledger knows whether it saw today's spend since 00:00 Phoenix.
#     A ledger that started mid-day imports the day's already-recorded remote cost from the
#     gateway's own request telemetry before it may enforce; if that import is impossible,
#     enforcement starts at the next Phoenix day boundary (observe until then).
#   * enforce transition -- turning enforcement on expires the newest observe-mode
#     reservations until the day is no longer oversubscribed.
#   * expiry/crash -- reservations expire at ttl; a request hold left by a crashed process or
#     older than SPEND_REQUEST_HOLD_TTL is charged in full, exactly once.
#   * read -- GET /gateway/spend (unauthenticated, read-only): the one total.
SPEND_FILE = os.environ.get("SHIM_SPEND_FILE", "/home/kevin/.local/share/vllm-qwen27b/gateway-spend.json")
_RUNTIME_SPEND_LOCK = None


def _claim_spend_authority(path=SPEND_FILE):
    """Only one gateway process may mutate the daily ledger at a time.

    SpendLedger's in-process mutex and atomic rename do not serialize two
    processes. A second process would load a stale total, orphan another
    process's holds, and could each spend up to the $25 cap. The lock is held
    for this process's entire lifetime and released by the kernel on death.
    """
    global _RUNTIME_SPEND_LOCK
    if _RUNTIME_SPEND_LOCK is not None:
        return
    lock_path = path + ".owner.lock"
    os.makedirs(os.path.dirname(os.path.abspath(lock_path)), mode=0o700, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise RuntimeError("another gateway process already owns the spend ledger") from exc
    except BaseException:
        os.close(fd)
        raise
    _RUNTIME_SPEND_LOCK = fd
SPEND_CLIENTS_FILE = os.environ.get("SHIM_SPEND_CLIENTS_FILE",
                                    "/home/kevin/.local/share/vllm-qwen27b/spend-clients.json")
SPEND_CAP_USD = float(os.environ.get("SHIM_SPEND_CAP_USD", "25.0"))
SPEND_ENFORCE = _env_flag(os.environ.get("SHIM_SPEND_ENFORCE", "1"), True)
SPEND_TZ = os.environ.get("SHIM_SPEND_TZ", "America/Phoenix")   # the estate host's day (SpendGuard's)
SPEND_RESERVATION_TTL_MAX = int(os.environ.get("SHIM_SPEND_RESERVATION_TTL_MAX", str(4 * 3600)))
SPEND_REQUEST_HOLD_TTL = int(os.environ.get("SHIM_SPEND_REQUEST_HOLD_TTL", "1800"))
SPEND_DEFAULT_OUT_TOKENS = int(os.environ.get("SHIM_SPEND_DEFAULT_OUT_TOKENS", "16384"))
# An optional verified-outcome brake for automatic overflow. The operator's
# full-remote lease and explicit estate-remote incident tier remain deliberate
# routes under the shared $25 hard cap. A stalled/unknown estate outcome cannot
# silently spend a fresh day on ordinary gateway-chosen overflow.
OUTCOME_ALARM_PATH = os.environ.get("SHIM_OUTCOME_ALARM_PATH", "")
STALLED_AUTO_OVERFLOW_CAP_USD = float(os.environ.get("SHIM_STALLED_AUTO_OVERFLOW_CAP_USD", "2.0"))
SPEND_KEEP_SECS = 2 * 86400          # finalized/expired reservations kept for idempotent replays
_SPEND_RID = re.compile(r"^[0-9a-f]{32}$")
_CFG.update({
    "SHIM_SPEND_CAP_USD": ("SPEND_CAP_USD", float),
    "SHIM_SPEND_ENFORCE": ("SPEND_ENFORCE", lambda v: _cfg_flag(v)),
})
try:
    with open(os.path.abspath(__file__), "rb") as _fh:
        GATEWAY_SHA256 = hashlib.sha256(_fh.read()).hexdigest()
except Exception:
    GATEWAY_SHA256 = ""


def _spend_zone(tz=None):
    from zoneinfo import ZoneInfo
    return ZoneInfo(tz or SPEND_TZ)


def _spend_day(now, tz=None):
    try:
        return datetime.datetime.fromtimestamp(now, _spend_zone(tz)).strftime("%Y-%m-%d")
    except Exception:
        return time.strftime("%Y-%m-%d", time.gmtime(now))


def _spend_day_start(now, tz=None):
    d = datetime.datetime.fromtimestamp(now, _spend_zone(tz))
    return d.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def _spend_hold_estimate(ptok, maxtok, model=None):
    """Upper bound for one paid call: the prompt-token bound at the dearer period's CACHE-MISS
    price plus the maximum output at the dearer period's output price, from the dated table
    (legacy knobs only when the table cannot price the model). Settlement replaces it with the
    call's actual usage, so the hold only has to be an upper bound, never the charge."""
    try:
        rin, rout = _remote_hold_rates(model)
    except Exception:
        rin = max(REMOTE_COST_IN_PER_MTOK, REMOTE_COST_IN_PER_MTOK_PEAK)
        rout = max(REMOTE_COST_OUT_PER_MTOK, REMOTE_COST_OUT_PER_MTOK_PEAK)
    out = maxtok if maxtok and maxtok > 0 else SPEND_DEFAULT_OUT_TOKENS
    return round((max(0, ptok or 0) / 1e6) * rin + (out / 1e6) * rout, 6)


def _spend_prompt_token_upper(body, estimated):
    """Conservatively reserve paid input before provider usage can be known.

    `_est_tokens` is a capacity estimate, not a billing upper bound: non-English
    or dense text can tokenize far above its characters/4 heuristic. The JSON
    wire body's byte length bounds ordinary tokenized content and leaves room
    for provider chat-template markers. This affects the temporary hold only;
    settlement still charges actual provider usage and releases the difference.
    """
    return max(max(0, int(estimated or 0)), len(body) + 4096)


def telemetry_remote_cost(start, end, telemetry_dir=None):
    """(usd, requests, estimated_requests) the gateway itself recorded as remote spend in
    [start, end), from its request telemetry (requests-YYYYMMDD.jsonl, UTC-named). Same rule as
    settle: default provider only, status < 400. Rows written since R2 v5 carry the
    cache-aware ACTUAL cost (cost_basis 'actual'/'usage'); older rows carry only the no-cache
    list estimate and are counted in `estimated_requests` (their sum is an upper bound --
    replace it with the provider's bill via spend_reconcile.py). None when unreadable."""
    tdir = telemetry_dir or TELEMETRY_DIR
    usd, n, est = 0.0, 0, 0
    day = int(start // 86400)
    while day * 86400 < end:
        path = os.path.join(tdir, "requests-%s.jsonl" % time.strftime("%Y%m%d", time.gmtime(day * 86400)))
        day += 1
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    t = float(r.get("t") or 0)
                    status = r.get("status")
                    # v6: every paid row counts (custom metered included); only a declared-free
                    # endpoint is excluded. Rows since v6 count at any status (the provider
                    # bills errors that carried usage); pre-v6 rows keep the old status rule.
                    if "remote_sent" in r:
                        counted = (r.get("remote_sent") and r.get("cost_policy") != "free"
                                   and r.get("remote_outcome") != "failed")   # a failed call is never billed
                    else:
                        counted = status is None or int(status) < 400
                    if start <= t < end and r.get("route") == "remote" and counted:
                        usd += float(r.get("cost_est") or 0.0)
                        n += 1
                        if r.get("cost_basis") not in ("actual", "usage"):
                            est += 1
        except OSError:
            return None
    return round(usd, 6), n, est


class SpendLedgerError(RuntimeError):
    pass


SPEND_ATTRIBUTION_MAX_GROUPS = 512


def _spend_attribution_meta(info, basis=None):
    """Small, prompt-free dimensions persisted with the same write as spend."""
    info = info or {}
    return {k: str(v or "unknown")[:64] for k, v in {
        "client": info.get("client") or info.get("name") or info.get("xclient") or info.get("ip"),
        "model": info.get("remote_model") or info.get("model"),
        "alias": info.get("alias"),
        "reason": info.get("reason"),
        "basis": basis or info.get("basis") or info.get("cost_basis") or "hold",
    }.items()}


def _spend_attribution_empty(day):
    return {"day": day, "groups": {}, "unattributed_usd": 0.0,
            "reconciliation_usd": 0.0, "overflow_usd": 0.0, "overflow_calls": 0}


def _spend_attribution_view(audit):
    if not isinstance(audit, dict):
        return None
    groups = sorted((audit.get("groups") or {}).values(),
                    key=lambda row: -float(row.get("usd") or 0))
    view = {k: v for k, v in audit.items() if k != "groups"}
    view["groups"] = groups
    view["accounted_usd"] = round(
        sum(float(row.get("usd") or 0) for row in groups)
        + float(audit.get("unattributed_usd") or 0)
        + float(audit.get("overflow_usd") or 0)
        + float(audit.get("reconciliation_usd") or 0), 6)
    return view


class SpendLedger:
    """The atomic, durable daily remote-spend ledger. Pure apart from its own file."""

    def __init__(self, path, cap=lambda: SPEND_CAP_USD, clock=time.time, tz=None,
                 process_token=None, importer=telemetry_remote_cost, enforce=lambda: SPEND_ENFORCE):
        import threading
        self.path, self._cap, self._clock, self.tz = path, cap, clock, tz
        self._importer, self._enforce = importer, enforce
        self.process = process_token or hashlib.sha256(
            f"{os.getpid()}:{time.time_ns()}".encode()).hexdigest()[:16]
        self._lock = threading.Lock()
        self._settled = collections.OrderedDict()   # request keys already settled (bounded)
        self._dirty = False
        self._import_tried = None
        self.state = self._load()
        with self._lock:
            self._roll_and_sweep(self._clock())
            self._flush_quietly()

    # -- persistence --
    def _fresh(self, now):
        day = _spend_day(now, self.tz)
        return {"version": 1, "day": day, "spent": 0.0, "since": now,
                "complete_day": None, "orphans_charged": 0.0, "reservations": {}, "holds": {},
                "process": self.process, "enforcing": False,
                "attribution": _spend_attribution_empty(day)}

    def _load(self):
        now = self._clock()
        try:
            with open(self.path, "rb") as fh:
                raw = fh.read()
        except FileNotFoundError:
            self._dirty = True
            return self._fresh(now)
        except OSError as exc:
            return self._corrupt_state(now, f"unreadable: {exc}", None)
        try:
            st = json.loads(raw)
            if not isinstance(st, dict) or st.get("version") != 1 or not isinstance(st.get("spent", 0), (int, float)):
                raise ValueError("not a version-1 spend ledger")
        except Exception as exc:
            return self._corrupt_state(now, str(exc)[:200], raw)
        st.setdefault("reservations", {}); st.setdefault("holds", {})
        st.setdefault("since", None); st.setdefault("complete_day", None)
        st.setdefault("enforcing", False)
        if not isinstance(st.get("attribution"), dict) or st["attribution"].get("day") != st.get("day"):
            st["attribution"] = _spend_attribution_empty(st["day"])
            # Older ledgers had only a total. Preserve the gap explicitly;
            # never manufacture attribution from delayed telemetry.
            st["attribution"]["unattributed_usd"] = round(float(st.get("spent") or 0), 6)
            self._dirty = True
        if st.get("process") != self.process and st["holds"]:
            for key, h in list(st["holds"].items()):
                self._settle_locked(st, key, float(h.get("amount") or 0.0), orphan=True)
            self._dirty = True
        st["process"] = self.process
        return st

    def _corrupt_state(self, now, why, raw):
        """Preserve the bad artifact beside the ledger; never write over it until recovery."""
        artifact = None
        if raw is not None:
            artifact = f"{self.path}.corrupt-{int(now)}"
            try:
                fd = os.open(artifact, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, "wb") as fh:
                    fh.write(raw)
                    fh.flush()
                    os.fsync(fh.fileno())
            except FileExistsError:
                pass
            except OSError as exc:
                artifact = f"(copy failed: {exc})"
        log.error("spend ledger %s corrupt (%s): every paid forward and mutation is refused until "
                  "POST /gateway/spend/recover; artifact kept at %s", self.path, why, artifact)
        st = self._fresh(now)
        st.update(corrupt=why, corrupt_artifact=artifact)
        return st

    def _save(self):
        """Durable before return: 0600 O_EXCL temp, fsync, atomic replace, fsync the dir."""
        if self.state.get("corrupt"):
            raise SpendLedgerError("ledger is corrupt; recover it before writing")
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, mode=0o700, exist_ok=True)
        tmp = f"{self.path}.tmp.{os.getpid()}.{time.time_ns()}"
        self.state["revision"] = int(self.state.get("revision") or 0) + 1   # CAS token for REPLACE
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(self.state, fh, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
            dfd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self._dirty = False

    def _flush_quietly(self):
        if self._dirty and not self.state.get("corrupt"):
            try:
                self._save()
            except Exception as exc:  # noqa: BLE001 -- stays dirty; the next mutation retries
                log.warning("spend ledger flush failed (kept dirty): %s", exc)

    def _commit_or_rollback(self, before):
        """Make a mutation durable, or restore `before` and report failure."""
        try:
            self._save()
            return True
        except Exception as exc:  # noqa: BLE001
            self.state = before
            log.error("spend ledger write failed, mutation refused: %s", exc)
            return False

    # -- bookkeeping --
    def enforcing(self):
        st = self.state
        return bool(self._enforce()) and not st.get("corrupt") and st.get("complete_day") == st.get("day")

    def _attribution_gap(self):
        audit = _spend_attribution_view(self.state.get("attribution"))
        return round(float(self.state.get("spent") or 0)
                     - float((audit or {}).get("accounted_usd") or 0), 6)

    def _enforce_blocker(self):
        if not self._enforce():
            return "observe mode (SHIM_SPEND_ENFORCE=0)"
        if self.state.get("corrupt"):
            return "ledger corrupt"
        if abs(self._attribution_gap()) > 0.000001:
            return "spend attribution does not match the durable ledger total"
        if self.state.get("complete_day") != self.state.get("day"):
            return "today's prior remote spend is not imported; enforcement starts at the next Phoenix day"
        return None

    def _import_day(self, now):
        st = self.state
        if self._importer is None or self._import_tried == st["day"]:
            return
        self._import_tried = st["day"]          # at most one telemetry read per process per day
        got = self._importer(_spend_day_start(now, self.tz), now)
        if got is None:
            return
        usd, n = got[0], got[1]
        est = got[2] if len(got) > 2 else 0
        old_spent = float(st.get("spent") or 0)
        st["spent"] = round(max(float(st.get("spent") or 0), float(usd)), 6)
        imported_gap = round(float(st["spent"]) - old_spent, 6)
        if imported_gap > 0:
            st["attribution"]["unattributed_usd"] = round(
                float(st["attribution"].get("unattributed_usd") or 0) + imported_gap, 6)
        st["imported"] = {"day": st["day"], "telemetry_usd": usd, "requests": n,
                          "estimated_requests": est, "upper_bound": bool(est), "at": now}
        st["complete_day"] = st["day"]
        self._dirty = True

    def _roll_and_sweep(self, now):
        st = self.state
        if st.get("corrupt"):
            return
        day = _spend_day(now, self.tz)
        if st.get("day") != day:
            # Every remote request passes through this ledger, so a ledger that was live (or the
            # gateway down) across midnight has seen the whole new day: it starts complete.
            st["previous_attribution"] = st.get("attribution")
            st["attribution"] = _spend_attribution_empty(day)
            st["day"], st["spent"], st["orphans_charged"] = day, 0.0, 0.0
            st["complete_day"] = day
            self._dirty = True
        if st.get("complete_day") != day:
            self._import_day(now)
        for key, h in list(st["holds"].items()):
            if now - float(h.get("created") or 0) >= SPEND_REQUEST_HOLD_TTL:
                self._settle_locked(st, key, float(h.get("amount") or 0.0), orphan=True)
                self._dirty = True
        for rid, r in list(st["reservations"].items()):
            if r.get("status") == "active" and now >= float(r.get("expires") or 0):
                self._close_reservation(st, r, "expired", now)
                self._dirty = True
            elif r.get("status") != "active" and now - float(r.get("closed_at") or now) > SPEND_KEEP_SECS:
                del st["reservations"][rid]
                self._dirty = True
        enforcing = self.enforcing()
        if enforcing and not st.get("enforcing"):
            # Observe-mode reservations must not oversubscribe the cap once it binds: expire
            # the newest until the day fits (their holders see a refusal and reconcile).
            active = sorted((r for r in st["reservations"].values() if r.get("status") == "active"),
                            key=lambda r: -float(r.get("created") or 0))
            for r in active:
                if self._totals_raw()["available_raw"] >= 0:
                    break
                self._close_reservation(st, r, "expired-oversubscribed", now)
        if bool(st.get("enforcing")) != enforcing:
            st["enforcing"] = enforcing
            self._dirty = True

    @staticmethod
    def _close_reservation(st, r, status, now):
        for h in st["holds"].values():
            if h.get("rid") == r["id"] and h.get("from_res"):
                h["from_global"] = round(float(h.get("from_global") or 0) + float(h["from_res"]), 6)
                h["from_res"] = 0.0
        r["held"] = 0.0
        r["status"], r["closed_at"] = status, now

    @staticmethod
    def _settle_locked(st, key, cost, orphan=False, meta=None):
        h = st["holds"].pop(key, None)
        cost = round(max(0.0, float(cost or 0.0)), 6)
        if h is not None and h.get("rid"):
            r = st["reservations"].get(h["rid"])
            if r is not None:
                r["held"] = round(max(0.0, float(r.get("held") or 0) - float(h.get("from_res") or 0)), 6)
                r["used"] = round(float(r.get("used") or 0) + cost, 6)
        st["spent"] = round(float(st.get("spent") or 0) + cost, 6)
        if orphan:
            st["orphans_charged"] = round(float(st.get("orphans_charged") or 0) + cost, 6)
        audit = st.setdefault("attribution", _spend_attribution_empty(st["day"]))
        if cost:
            dims = meta or ((h or {}).get("meta") if h else None)
            if not isinstance(dims, dict) or not dims:
                audit["unattributed_usd"] = round(float(audit.get("unattributed_usd") or 0) + cost, 6)
            else:
                dims = {k: str(dims.get(k) or "unknown")[:64]
                        for k in ("client", "model", "alias", "reason", "basis")}
                if orphan:
                    dims["basis"] = "orphan-held"
                group_key = json.dumps(dims, sort_keys=True, separators=(",", ":"))
                groups = audit.setdefault("groups", {})
                if group_key not in groups and len(groups) >= SPEND_ATTRIBUTION_MAX_GROUPS:
                    audit["overflow_usd"] = round(float(audit.get("overflow_usd") or 0) + cost, 6)
                    audit["overflow_calls"] = int(audit.get("overflow_calls") or 0) + 1
                else:
                    group = groups.setdefault(group_key, {**dims, "usd": 0.0, "calls": 0})
                    group["usd"] = round(float(group["usd"]) + cost, 6)
                    group["calls"] += 1
        return cost

    def _totals_raw(self):
        st = self.state
        cap = float(self._cap())
        outstanding = sum(max(0.0, float(r.get("amount") or 0) - float(r.get("used") or 0))
                          for r in st["reservations"].values() if r.get("status") == "active")
        held = sum(float(h.get("from_global") or 0) for h in st["holds"].values())
        spent = float(st.get("spent") or 0)
        return {"cap": cap, "spent": spent, "reserved": outstanding, "held": held,
                "available_raw": cap - spent - outstanding - held}

    def _totals(self):
        t = self._totals_raw()
        available = 0.0 if self.state.get("corrupt") else max(0.0, t["available_raw"])
        return {"cap": round(t["cap"], 6), "spent": round(t["spent"], 6), "reserved": round(t["reserved"], 6),
                "held": round(t["held"], 6), "available": round(available, 6)}

    # -- public API (each call is one atomic step) --
    def snapshot(self):
        with self._lock:
            now = self._clock()
            self._roll_and_sweep(now)
            self._flush_quietly()
            st = self.state
            active = [{k: v for k, v in r.items() if k != "bound_ip"}
                      for r in st["reservations"].values() if r.get("status") == "active"]
            audit_view = _spend_attribution_view(st.get("attribution"))
            return {"day": st["day"], "tz": self.tz or SPEND_TZ,
                    "enforce": self.enforcing(), "enforce_configured": bool(self._enforce()),
                    "enforce_blocked": self._enforce_blocker(),
                    "day_complete": st.get("complete_day") == st.get("day"),
                    "imported": st.get("imported"), "since": st.get("since"),
                    "corrupt": st.get("corrupt"), "corrupt_artifact": st.get("corrupt_artifact"),
                    "durable": not self._dirty, "in_flight": len(st["holds"]),
                    "revision": int(st.get("revision") or 0), "last_settle_at": st.get("last_settle_at"),
                    "orphans_charged": st.get("orphans_charged", 0.0),
                    "gateway_sha256": GATEWAY_SHA256, "reservations": active,
                    "attribution": audit_view,
                    "attribution_gap_usd": round(float(st.get("spent") or 0)
                                                  - float((audit_view or {}).get("accounted_usd") or 0), 6),
                    "previous_attribution": _spend_attribution_view(st.get("previous_attribution")),
                    **self._totals()}

    def get(self, rid):
        with self._lock:
            r = self.state["reservations"].get(str(rid or ""))
            return dict(r) if r else None

    def reserve(self, owner, key, amount, ttl_s, meta=None, client_ip=None):
        owner, key = str(owner or "").strip()[:64], str(key or "").strip()[:128]
        if not owner or not key:
            return False, None, "owner and key are required"
        try:
            amount, ttl_s = float(amount), int(ttl_s)
        except (TypeError, ValueError):
            return False, None, "amount and ttl_s must be numbers"
        if not (0 < amount <= 1000) or not (60 <= ttl_s <= SPEND_RESERVATION_TTL_MAX):
            return False, None, f"amount must be in (0, 1000] and ttl_s in [60, {SPEND_RESERVATION_TTL_MAX}]"
        with self._lock:
            now = self._clock()
            self._roll_and_sweep(now)
            if self.state.get("corrupt"):
                return False, None, "spend ledger corrupt: refused until recovered"
            for r in self.state["reservations"].values():
                if r.get("owner") == owner and r.get("key") == key:
                    self._flush_quietly()
                    return True, dict(r, replayed=True), "replayed"      # retry-idempotent
            tot = self._totals()
            if self.enforcing() and amount > tot["available"]:
                self._flush_quietly()
                return False, None, (f"daily remote cap: ${tot['spent']:.2f} spent + ${tot['reserved']:.2f} reserved "
                                     f"+ ${tot['held']:.2f} in flight leaves ${tot['available']:.2f} of "
                                     f"${tot['cap']:.2f}; ${amount:.2f} requested")
            before = copy.deepcopy(self.state)
            rid = hashlib.sha256(f"{owner}\0{key}\0{now}\0{self.process}".encode()).hexdigest()[:32]
            self.state["reservations"][rid] = {
                "id": rid, "owner": owner, "key": key, "amount": round(amount, 6), "used": 0.0,
                "held": 0.0, "status": "active", "created": now, "expires": now + ttl_s,
                "bound_ip": client_ip, "meta": {k: str(v)[:128] for k, v in (meta or {}).items()}}
            if not self._commit_or_rollback(before):
                return False, None, "spend ledger write failed: reservation refused"
            return True, dict(self.state["reservations"][rid]), "reserved"

    def finalize(self, rid=None, owner=None, key=None):
        """-> (record | None, error | None). Owner-scoped when `owner` is given."""
        with self._lock:
            now = self._clock()
            self._roll_and_sweep(now)
            if self.state.get("corrupt"):
                return None, "spend ledger corrupt: refused until recovered"
            r = self.state["reservations"].get(str(rid or ""))
            if r is None and owner and key:
                r = next((x for x in self.state["reservations"].values()
                          if x.get("owner") == owner and x.get("key") == key), None)
            if r is None:
                return None, "unknown reservation"
            if owner is not None and r.get("owner") != owner:
                return None, "forbidden: reservation belongs to another client"
            if r.get("status") == "active":
                before = copy.deepcopy(self.state)
                self._close_reservation(self.state, self.state["reservations"][r["id"]], "finalized", now)
                if not self._commit_or_rollback(before):
                    return None, "spend ledger write failed: finalize not recorded"
                r = self.state["reservations"][r["id"]]
            else:
                self._flush_quietly()
            return dict(r), None

    def hold(self, req_key, amount, rid=None, client_ip=None, meta=None):
        """(ok, reason). Atomic check-and-hold for one paid request before it is forwarded."""
        amount = round(max(0.0, float(amount or 0.0)), 6)
        with self._lock:
            now = self._clock()
            self._roll_and_sweep(now)
            if self.state.get("corrupt"):
                return False, "spend ledger corrupt: paid forwarding refused until recovered"
            if abs(self._attribution_gap()) > 0.000001:
                return False, "spend attribution mismatch: paid forwarding refused until recovered"
            if req_key in self.state["holds"]:
                return True, "already held"
            enforcing = self.enforcing()
            from_res, r = 0.0, None
            if rid:
                r = self.state["reservations"].get(rid)
                if r is None or r.get("status") != "active":
                    return False, f"spend reservation {rid} is {'unknown' if r is None else r.get('status')}"
                if r.get("bound_ip") and r["bound_ip"] != client_ip:
                    return False, f"spend reservation {rid} is bound to another client"
                remaining = max(0.0, float(r["amount"]) - float(r.get("used") or 0) - float(r.get("held") or 0))
                from_res = min(amount, remaining)
            need = round(amount - from_res, 6)
            tot = self._totals()
            if enforcing and need > tot["available"]:
                self._flush_quietly()
                return False, (f"daily remote cap: ${tot['available']:.2f} of ${tot['cap']:.2f} left, "
                               f"this request needs up to ${need:.4f} beyond its reservation")
            before = copy.deepcopy(self.state)
            if r is not None:
                r["held"] = round(float(r.get("held") or 0) + from_res, 6)
            self.state["holds"][req_key] = {"amount": amount, "from_res": from_res, "from_global": need,
                                            "rid": r["id"] if r is not None else None, "created": now,
                                            "meta": _spend_attribution_meta(meta) if meta else None}
            if not self._commit_or_rollback(before):
                if enforcing:
                    return False, "spend ledger write failed: paid forwarding refused"
                # observe mode: keep the hold in memory (still counted) and retry durability later
                self.state["holds"][req_key] = {"amount": amount, "from_res": 0.0, "from_global": amount,
                                                "rid": None, "created": now,
                                                "meta": _spend_attribution_meta(meta) if meta else None}
                self._dirty = True
            return True, "held"

    def can_hold(self, amount):
        """Non-mutating: would a hold of `amount` without a reservation be granted now?"""
        with self._lock:
            now = self._clock()
            self._roll_and_sweep(now)
            self._flush_quietly()
            if self.state.get("corrupt"):
                return False
            if abs(self._attribution_gap()) > 0.000001:
                return False
            return (not self.enforcing()) or round(float(amount), 6) <= self._totals()["available"]

    def held_amount(self, req_key):
        """The amount currently held for a request (None if it holds nothing)."""
        with self._lock:
            h = self.state["holds"].get(req_key)
            return None if h is None else float(h.get("amount") or 0.0)

    def settle(self, req_key, cost, meta=None):
        """Replace a request's hold with its priced cost, or charge an unheld remote request.
        Idempotent per request key. A failed write keeps the charge in memory (still counted)
        and marks the ledger not durable until a later write succeeds."""
        with self._lock:
            now = self._clock()
            self._roll_and_sweep(now)
            if req_key in self._settled or (req_key not in self.state["holds"] and not cost):
                self._flush_quietly()
                return 0.0
            if cost is None:                   # no trustworthy usage: charge the full hold
                cost = float((self.state["holds"].get(req_key) or {}).get("amount") or 0.0)
            charged = self._settle_locked(self.state, req_key, cost,
                                          meta=_spend_attribution_meta(meta) if meta else None)
            self.state["last_settle_at"] = now
            self._settled[req_key] = charged
            while len(self._settled) > 20000:
                self._settled.popitem(last=False)
            self._dirty = True
            self._flush_quietly()
            return charged

    def recover(self, spent=None, reason="", replace=False, source=None, expected_revision=None):
        """Governed recovery from a corrupt ledger (or a re-import): a fresh ledger whose
        spend is the operator's figure or, if none, the gateway's own telemetry for today.
        The corrupt artifact is left untouched. -> (ok, detail)."""
        with self._lock:
            now = self._clock()
            prior = self.state
            if replace and not prior.get("corrupt"):
                # v6: a REPLACE may lower today's figure, so it is compare-and-set: bound to the
                # exact revision the caller computed against, with nothing in flight.
                if expected_revision is None or int(expected_revision) != int(prior.get("revision") or 0):
                    return False, (f"conflict: ledger is at revision {int(prior.get('revision') or 0)}, "
                                   f"not {expected_revision}; recompute")
                if prior.get("holds"):
                    return False, f"conflict: {len(prior['holds'])} paid request(s) in flight; recompute"
            fresh = self._fresh(now)
            fresh["process"] = self.process
            if spent is None:
                got = self._importer(_spend_day_start(now, self.tz), now) if self._importer else None
                if got is None:
                    return False, "no spend figure given and today's telemetry is unreadable"
                spent, n = got
                source = f"telemetry ({n} remote requests)"
            else:
                spent = float(spent)
                if spent < 0:
                    return False, "spent must be >= 0"
                source = ("operator-replace" if replace else "operator") + (f": {str(source)[:120]}" if source else "")
            if not prior.get("corrupt"):
                # A re-import never lowers today's figure unless the operator explicitly REPLACES
                # it with an attested provider figure (spend_reconcile.py, from the bill).
                if not replace:
                    spent = max(spent, float(prior.get("spent") or 0))
                fresh["reservations"], fresh["holds"] = prior.get("reservations", {}), prior.get("holds", {})
                fresh["since"] = prior.get("since")
                if prior.get("day") == fresh["day"]:
                    fresh["attribution"] = copy.deepcopy(prior.get("attribution") or
                                                          _spend_attribution_empty(fresh["day"]))
                    fresh["previous_attribution"] = copy.deepcopy(prior.get("previous_attribution"))
                else:
                    fresh["previous_attribution"] = copy.deepcopy(prior.get("attribution"))
            if replace and spent is not None and not str(source).startswith("operator-replace"):
                return False, "replace needs an explicit operator figure"
            fresh["revision"] = int(prior.get("revision") or 0)
            fresh.update(spent=round(spent, 6), complete_day=fresh["day"],
                         recovered={"at": now, "source": source, "reason": str(reason)[:200],
                                    "prior_corrupt": prior.get("corrupt"),
                                    "artifact": prior.get("corrupt_artifact")})
            audit = fresh["attribution"]
            explained = (sum(float(row.get("usd") or 0) for row in (audit.get("groups") or {}).values())
                         + float(audit.get("unattributed_usd") or 0)
                         + float(audit.get("overflow_usd") or 0))
            audit["reconciliation_usd"] = round(float(fresh["spent"]) - explained, 6)
            self.state = fresh
            if not self._commit_or_rollback(prior):
                return False, "spend ledger write failed: recovery not recorded"
            return True, fresh["recovered"]


_SPEND_LEDGER = None


def _spend():
    """The process's ledger. Only the SERVICE (run as __main__) or a caller that sets
    SHIM_SPEND_FILE explicitly ever opens the production path. An imported copy (tests,
    tooling) that merely routes a request gets a private throwaway ledger with no telemetry
    import -- 2026-09-25: a v5 test run that imported this module without SHIM_SPEND_FILE
    rewrote the live ledger file for ~7 minutes (the live process then overwrote it back)."""
    global _SPEND_LEDGER
    if _SPEND_LEDGER is None:
        if __name__ == "__main__" or "SHIM_SPEND_FILE" in os.environ:
            _SPEND_LEDGER = SpendLedger(SPEND_FILE)
        else:
            _SPEND_LEDGER = SpendLedger(os.path.join(_spend_scratch_dir(), "gateway-spend.json"),
                                        importer=None)
    return _SPEND_LEDGER


_SPEND_SCRATCH = None


def _spend_scratch_dir():
    """ONE managed temp dir per imported (non-service) process, removed at exit (v6)."""
    global _SPEND_SCRATCH
    if _SPEND_SCRATCH is None:
        import atexit
        import tempfile
        _SPEND_SCRATCH = tempfile.TemporaryDirectory(prefix="shim-spend-")
        atexit.register(_SPEND_SCRATCH.cleanup)
    return _SPEND_SCRATCH.name


_SPEND_CLIENTS = {"mtime": None, "map": {}}


def _spend_clients():
    """{sha256(token) hex: client name} from SPEND_CLIENTS_FILE. The file must be a regular,
    owner-only file; anything else yields no clients (every mutation refused)."""
    try:
        st = os.lstat(SPEND_CLIENTS_FILE)
        if not (st.st_mode & 0o170000 == 0o100000) or st.st_mode & 0o077:
            return {}
        if _SPEND_CLIENTS["mtime"] != st.st_mtime_ns:
            with open(SPEND_CLIENTS_FILE) as fh:
                raw = json.load(fh)
            clients = (raw.get("clients") if isinstance(raw, dict) else None) or {}
            _SPEND_CLIENTS["map"] = {str(h).lower(): str(n) for n, h in clients.items()
                                     if re.fullmatch(r"[0-9a-fA-F]{64}", str(h))}
            _SPEND_CLIENTS["mtime"] = st.st_mtime_ns
        return _SPEND_CLIENTS["map"]
    except Exception:
        return {}


def _spend_client(request):
    token = request.headers.get("X-Spend-Token", "") if getattr(request, "headers", None) else ""
    if not token:
        return None
    digest = hashlib.sha256(token.encode()).hexdigest()
    for h, name in _spend_clients().items():
        if hmac.compare_digest(h, digest):
            return name
    return None


def _spend_refusal(reason, kind="spend_cap_exhausted"):
    return web.json_response({"error": {"message": f"gateway spend authority: {reason}", "type": kind}},
                             status=429, headers={"Retry-After": "600", "X-Gateway-Spend-Refused": kind})


def _cap_exhausted_unavailable(why):
    return web.json_response({"error": {"message": f"{why}; retry later", "type": "local_unavailable_cap_exhausted"}},
                             status=503, headers={"Retry-After": "60"})


def _spend_allows_overflow(ptok, maxtok):
    """May an OVERFLOW (gateway-chosen, not caller-requested) request go remote right now?
    False when the cap is exhausted or the ledger is unusable -- the router then serves it
    locally (queueing for a lane) instead of refusing it."""
    try:
        return _spend().can_hold(_spend_hold_estimate(ptok, maxtok))
    except Exception as _e:
        _swallowed("_spend_allows_overflow", _e)
        return False


def _automatic_remote_budget_allows(ptok, maxtok):
    """Bound automatic paid work while accepted delivery is stalled or unknown.

    This is a conservative routing brake, not the hard spend authority: every
    paid request still passes SpendLedger's atomic $25 hold. If the configured
    independent outcome alarm cannot be read, automatic overflow stays local.
    Explicit incident escalation remains available through estate-remote.
    """
    if not OUTCOME_ALARM_PATH:
        return True
    try:
        with open(OUTCOME_ALARM_PATH, encoding="utf-8") as fh:
            outcome = json.load(fh)
        if not isinstance(outcome, dict):
            return False
        if outcome.get("status") == "producing":
            return True
        spend = _spend().snapshot()
        exposure = sum(float(spend.get(k) or 0) for k in ("spent", "held", "reserved"))
        return exposure + _spend_hold_estimate(ptok, maxtok) <= STALLED_AUTO_OVERFLOW_CAP_USD
    except Exception as _e:
        _swallowed("_automatic_remote_budget_allows", _e)
        return False


_STALL_BRAKE_NOT_APPLIED = collections.Counter()


def _stall_brake_not_applied(why, client):
    """The stalled-delivery brake would have refused this request, but remote was the only server (`why`).
    Counted so the capacity surface shows how much stalled-state traffic the gateway carried rather than refused."""
    _STALL_BRAKE_NOT_APPLIED[why] += 1
    log.info("stalled-delivery brake not applied (%s, client=%s): remote is the only place this can be served", why, client)


def _spend_hold_for(request, body, prices=None, model=None):
    """Hold this paid request's upper-bound cost before forwarding. None = go ahead.
    `prices` ($/Mtok hit, miss, out) for a metered custom endpoint; default provider otherwise
    (`model` selects its row in the dated pricing table)."""
    info = _ACTIVE.get(id(request))
    key = (info or {}).get("spend_key") or ("anon-" + os.urandom(8).hex())
    try:
        maxtok = int(json.loads(body).get("max_tokens") or 0)
    except Exception:
        maxtok = 0
    ptok = _spend_prompt_token_upper(body, (info or {}).get("ptok") or _est_tokens(body))
    rid = _alias_for_request(body).get("reservation")
    if prices:
        out = maxtok if maxtok and maxtok > 0 else SPEND_DEFAULT_OUT_TOKENS
        amount = round((max(0, ptok or 0) * prices[1] + out * prices[2]) / 1e6, 6)   # no-cache bound
    else:
        amount = _spend_hold_estimate(ptok, maxtok, model)
    ok, reason = _spend().hold(key, amount, rid=rid, client_ip=getattr(request, "remote", None),
                               meta=_spend_attribution_meta(info))
    if not ok:
        log.info("spend refused %s: %s", key, reason)
        kind = ("spend_reservation_invalid" if reason.startswith("spend reservation")
                else "spend_ledger_unavailable" if "ledger" in reason else "spend_cap_exhausted")
        return _spend_refusal(reason, kind)
    if info is not None:
        info["spend_held"] = True
    return None


def _spend_settle(info, resp):
    """handle_completions' finally: charge what the provider bills, release the hold.

    The charge is the call's ACTUAL usage (SL lane, 2026-10-02), priced per token type at the
    time it was sent:
      * provider usage present (any HTTP status) -> actual (cache hit / miss / output);
      * the call failed -- HTTP >= 400 (a 402 Insufficient Balance, 4xx, 5xx), a relay that died
        before any completion, or a request that was held but never sent -> $0, hold released;
      * the call was answered but no usage arrived (client hung up, stream cut) -> a bounded
        estimate (prompt split by the recent cache-hit share + streamed chunks), never more than
        the hold;
      * the outcome is unknown (the handler itself was torn down mid-call) -> the full hold,
        the one conservative case left.
    A free-policy endpoint is never charged. The upfront hold is unchanged: it bounds what
    in-flight calls can spend, then this replaces it with the real figure."""
    try:
        key = info.get("spend_key")
        if not key:
            return
        if info.get("cost_policy") == "free":
            if info.get("spend_held"):
                _spend().settle(key, 0.0, meta=_spend_attribution_meta(info, "free"))
            return
        sent = bool(info.get("remote_sent"))
        cost, basis = _request_remote_cost(info) if sent else (0.0, None)
        outcome = info.get("remote_outcome")
        if sent and basis in ("actual", "usage"):
            charge, mbasis = cost, basis
            if basis == "actual":
                _cache_ratio_note(info.get("remote_cache_hit"), info.get("remote_cache_miss"))
        elif sent and basis == "failed":
            charge, mbasis = 0.0, "failed"
        elif sent and basis == "estimate" and outcome == "ok":
            charge, mbasis = cost, "aborted-estimate"
            held = _spend().held_amount(key) if info.get("spend_held") else None
            if held is not None:
                charge = min(charge, held)
        elif sent and info.get("spend_held"):
            charge, mbasis = None, "held-fallback"     # outcome unknown: the held amount
        elif sent and info.get("route") == "remote":
            charge, mbasis = cost, "estimated"         # unheld (pre-v6 path) estimate
        else:
            charge, mbasis = 0.0, "unsent"
        if info.get("spend_held") or charge:
            charged = _spend().settle(key, charge, meta=_spend_attribution_meta(info, mbasis))
            info["charged_usd"] = charged
    except Exception as exc:
        log.warning("spend settle failed (hold expires and is charged): %s", exc)


RECONCILIATION_FACT_FILE = os.environ.get(
    "SHIM_SPEND_RECONCILIATION_FILE", "/home/kevin/.local/share/vllm-qwen27b/spend-reconciliation.json")


def _reconciliation_fact():
    """The latest ledger-vs-provider drift fact (written by spend_reconcile.py --fact-out), as
    a SUMMARY for Halo -- read-only, no hours. None until someone has run the reconciliation.
    Nothing here changes the cap or the ledger."""
    try:
        with open(RECONCILIATION_FACT_FILE, encoding="utf-8") as fh:
            f = json.load(fh)
        keys = ("generated_at", "window", "provider_usd", "ledger_usd", "drift_usd", "ledger_over_provider",
                "failed_calls_charged_usd", "hours_drifted", "verdict")
        out = {k: f.get(k) for k in keys}
        out["age_s"] = round(time.time() - float(f.get("generated_at") or 0))
        return out
    except Exception:
        return None


async def gateway_spend(request):
    """GET /gateway/spend: the one daily remote-spend total (read-only, unauthenticated)."""
    snap = _spend().snapshot()
    snap["reconciliation"] = _reconciliation_fact()
    try:
        snap["pricing"] = pricing_status()
    except Exception as exc:
        snap["pricing"] = {"error": str(exc)[:120]}
    return web.json_response(snap)


async def _spend_body(request):
    try:
        f = await request.json()
    except Exception:
        return None
    return f if isinstance(f, dict) else None


async def gateway_spend_reserve(request):
    client = _spend_client(request)
    if client is None:
        return web.json_response({"ok": False, "error": "unauthorized: X-Spend-Token required"}, status=401)
    f = await _spend_body(request)
    if f is None:
        return web.json_response({"ok": False, "error": "invalid json"}, status=400)
    if f.get("owner") not in (None, client):
        return web.json_response({"ok": False, "error": "forbidden: owner must be the authenticated client"},
                                 status=403)
    ok, rec, reason = _spend().reserve(client, f.get("key"), f.get("amount"), f.get("ttl_s", 3600),
                                       f.get("meta") or {}, client_ip=getattr(request, "remote", None))
    if not ok:
        return web.json_response({"ok": False, "reason": reason},
                                 status=429 if reason.startswith(("daily remote cap", "spend ledger")) else 400)
    rec = {k: v for k, v in rec.items() if k != "bound_ip"}
    return web.json_response({"ok": True, "reservation": rec, "model": f"estate-remote.{rec['id']}",
                              "outcome": reason})


async def gateway_spend_finalize(request):
    client = _spend_client(request)
    if client is None:
        return web.json_response({"ok": False, "error": "unauthorized: X-Spend-Token required"}, status=401)
    f = await _spend_body(request)
    if f is None:
        return web.json_response({"ok": False, "error": "invalid json"}, status=400)
    rec, err = _spend().finalize(f.get("reservation_id"), client, f.get("key"))
    if rec is None:
        code = 403 if err.startswith("forbidden") else 404 if err == "unknown reservation" else 503
        return web.json_response({"ok": False, "error": err}, status=code)
    return web.json_response({"ok": True, "reservation": {k: v for k, v in rec.items() if k != "bound_ip"}})


async def gateway_spend_recover(request):
    """Governed recovery (dashboard admin token): replace a corrupt ledger, or re-import."""
    if not _admin_ok(request):
        return web.json_response({"ok": False, "error": "unauthorized: X-Admin-Token required"}, status=401)
    f = await _spend_body(request) or {}
    ok, detail = _spend().recover(f.get("spent"), f.get("reason") or "",
                                  replace=bool(f.get("replace")), source=f.get("source"),
                                  expected_revision=f.get("expected_revision"))
    log.warning("spend ledger recovery requested: ok=%s detail=%s", ok, detail)
    return web.json_response({"ok": ok, "detail": detail, "spend": _spend().snapshot()},
                             status=200 if ok else 409)


def _note_payload_outcome(request, payload, stream):
    """Best-effort: stash the finished response's HTTP status and (non-streaming) exact
    completion-token count onto the live request registry entry (_ACTIVE), for the
    finally-hook / rollups below and for record_event()'s _events enrichment. Streaming
    outtok/outtok_lb is instead set inside _relay() itself (see DESIGN.md (c) for why: this
    helper only ever sees an already-fully-buffered non-streaming body or, for the remote
    streaming case, the same StreamResponse _relay already finished writing). Never raises."""
    try:
        kw = {"http_status": getattr(payload, "status", None)}
        if not stream:
            body = getattr(payload, "body", None)
            if isinstance(body, (bytes, bytearray)):
                u = (json.loads(body).get("usage") or {})
                if "completion_tokens" in u:
                    kw["outtok"] = int(u["completion_tokens"])
                if (_ACTIVE.get(id(request)) or {}).get("route") == "local" and "prompt_tokens" in u:
                    cached = int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
                    kw["computed_actual"] = max(0, int(u["prompt_tokens"]) - cached)
                    kw["cached_actual"] = cached
                    kw["ptok_exact_local"] = int(u["prompt_tokens"])
        _active_set(request, **kw)
    except Exception as _e:
        _swallowed("_note_payload_outcome", _e)


_EST_ERR = collections.deque(maxlen=2000)    # [GW2] (client, est_tokens, exact, est_computed, computed_actual)


def _est_err_note(route, name, info):
    """Grade the gateway's prompt-size estimate against the engine's own usage on a finished local request."""
    try:
        exact = info.get("ptok_exact_local")
        if route not in ("local", "held") or exact is None:
            return
        est = info.get("est_tokens") or info.get("ptok")
        if not est:
            return
        _EST_ERR.append((name, int(est), int(exact), info.get("est_computed"), info.get("computed_actual")))
    except Exception:
        pass


def _err_dist(pairs):
    """Signed error (estimate - actual) distribution for [(estimate, actual)]."""
    if not pairs:
        return None
    errs = sorted(e - a for e, a in pairs)
    n = len(errs)
    q = lambda f: errs[min(n - 1, int(n * f))]
    return {"n": n, "bias_mean": round(sum(errs) / n, 1), "abs_mean": round(sum(abs(x) for x in errs) / n, 1),
            "p05": q(0.05), "p50": q(0.5), "p95": q(0.95), "min": errs[0], "max": errs[-1],
            "ratio_mean": round(sum(e / max(1, a) for e, a in pairs) / n, 4)}


def _est_err_summary():
    rows = list(_EST_ERR)
    by_client = collections.defaultdict(list)
    for c, e, a, _, _ in rows:
        by_client[c].append((e, a))
    top = sorted(by_client.items(), key=lambda kv: -len(kv[1]))[:8]
    return {"window": len(rows), "basis": "estimate - engine usage, last %d local requests" % _EST_ERR.maxlen,
            "prompt_tokens": _err_dist([(e, a) for _, e, a, _, _ in rows]),
            "computed_tokens": _err_dist([(int(ec), int(ca)) for _, _, _, ec, ca in rows
                                          if ec is not None and ca is not None]),
            "by_client": {c: _err_dist(p) for c, p in top}}


_SHAPE_STATS = collections.defaultdict(collections.Counter)   # [GW2 / L94] route -> response-shape counters


def _shape_stats_note(route, status, info):
    """Response-shape counters per route for /gateway/stats (successful chat responses only). Never raises."""
    try:
        if route not in ("local", "remote", "held") or (status is not None and status >= 400):
            return
        if info.get("content_empty") is None and info.get("finish_reason") is None:
            return                     # not a chat completion we classified
        c = _SHAPE_STATS["local" if route == "held" else route]
        c["responses"] += 1
        c["finish:%s" % (info.get("finish_reason") or "unknown")] += 1
        if info.get("content_empty") and not info.get("has_tool_calls"):
            c["empty_no_tool"] += 1
        if info.get("has_tool_calls") or info.get("tool_calls_n"):
            c["tool_responses"] += 1
            v = info.get("tool_args_valid")
            c["tool_args_valid" if v is True else ("tool_args_invalid" if v is False else "tool_args_unchecked")] += 1
        if info.get("reasoning_watchdog"):
            c["reasoning_watchdog_%s" % info["reasoning_watchdog"]] += 1
    except Exception:
        pass


def _shape_stats_summary():
    out = {}
    for route, c in _SHAPE_STATS.items():
        n = c["responses"] or 0
        checked = c["tool_args_valid"] + c["tool_args_invalid"]
        out[route] = {
            "responses": n,
            "finish_reason": {k[7:]: v for k, v in sorted(c.items()) if k.startswith("finish:")},
            "finish_length_rate": round(c["finish:length"] / n, 4) if n else None,
            "empty_no_tool": c["empty_no_tool"],
            "empty_no_tool_rate": round(c["empty_no_tool"] / n, 4) if n else None,
            "tool_responses": c["tool_responses"],
            "tool_args_valid": c["tool_args_valid"], "tool_args_invalid": c["tool_args_invalid"],
            "tool_args_unchecked": c["tool_args_unchecked"],
            "tool_args_valid_rate": round(c["tool_args_valid"] / checked, 4) if checked else None,
            "reasoning_watchdog": {k[19:]: v for k, v in c.items() if k.startswith("reasoning_watchdog_")},
        }
    return {"since_process_start": True, "by_route": out,
            "reasoning_watchdog": {"mode": REASONING_WATCHDOG, "budget_tokens": REASONING_BUDGET_TOKENS,
                                   "chars_per_token": REASONING_CHARS_PER_TOKEN, "triggers": dict(_RW_STATS)},
            "think_budget_explicit": THINK_BUDGET_EXPLICIT}


def _telemetry_note_request(info, resp=None):
    """Runs exactly once per request, from handle_completions' existing finally block --
    AFTER the response has already been returned to the client, regardless of which of the
    routing branches handled it. Updates per-client rollups and the error feed. Never raises:
    telemetry bookkeeping must not be able to affect a request that has already been answered.

    LIVE-TESTED FINDING: status was originally read only from info["http_status"], which
    _note_payload_outcome() stashes on the *success* path of _relay()/_forward_remote() -- an
    upstream failure (e.g. remote overflow itself failing, tested live with a closed dummy
    port) returns an error web.Response directly from _forward_remote()/_route_completions()
    without ever going through that helper, so http_status stayed unset and real errors were
    invisible to the error feed. Reading resp.status directly -- the actual object already
    returned to the client, always available here -- catches every path, instrumented or not,
    with info["http_status"] kept only as a fallback for the (none currently known) case resp
    is unavailable."""
    try:
        now = time.time()
        name = info.get("name") or info.get("ip") or "?"
        route = info.get("route") or "?"
        reason = info.get("reason") or ""
        outtok, outtok_lb = info.get("outtok"), info.get("outtok_lb")
        waited = info.get("waited") or 0.0
        duration = round(now - info["t0"], 3) if info.get("t0") else None
        status = getattr(resp, "status", None)
        if status is None:
            status = info.get("http_status")
        c = _PER_CLIENT[_client_key(_PER_CLIENT, name)]
        c["requests"] += 1
        c["classes"][info.get("flow_class") or "?"] += 1          # Lane DB2: the real work class, per client
        if route in ("local", "remote"):
            c[route] += 1
        if outtok is not None and (status is None or status < 400):
            note_output_tokens(name, outtok, now)           # LF: what this client's answers really weigh
            if route in ("local", "held"):
                micro_observe(name, info.get("preview"), outtok)   # K4: which call types are short
        if route in ("local", "held") and (status is None or status < 400) and (outtok or outtok_lb):
            _note_local_ok(now)                             # K4: proof the engine is really generating
        if outtok is not None:
            c["tokens_out"] += outtok; c["tokens_out_exact"] += outtok
        elif outtok_lb is not None:
            c["tokens_out"] += outtok_lb; c["tokens_out_lb"] += outtok_lb
        c["wait_sum"] += waited; c["wait_n"] += 1
        ttft = info.get("ttft")
        if ttft is not None and route == "remote" and (status is None or status < 400):
            qol_note_remote(info.get("ptok_exact") or info.get("ptok") or info.get("est_tokens"), ttft, now)   # lane CFG QoL
        if ttft is not None:
            c["ttft_sum"] += ttft; c["ttft_n"] += 1
            if status is None or status < 400:
                lat_note_request(now, route, ttft, duration, waited, outtok, info.get("flow_class"))
        req_cost, cost_basis = 0.0, None
        if route == "remote" and info.get("remote_sent") and info.get("cost_policy") != "free":
            req_cost, cost_basis = _request_remote_cost(info)
            c["cost_est_usd"] += req_cost
        if (status is not None and status >= 400) or reason == "failover" or info.get("stream_watchdog"):
            c["errors"] += 1
            _ERROR_FEED.appendleft({"t": round(now, 1), "client": name, "ep": info.get("ep"),
                                    "route": route, "reason": reason, "status": status,
                                    "preview": info.get("preview")})
        est_computed = info.get("est_computed")
        computed_actual = info.get("computed_actual")
        if est_computed is not None and computed_actual is not None and computed_actual > 2 * est_computed:
            log.warning("mis-estimate: client=%s est_computed=%d computed_actual=%d (%.1fx)",
                        name, est_computed, computed_actual, computed_actual / max(1, est_computed))
            _MISESTIMATE_FEED.appendleft({"t": round(now, 1), "client": name,
                                          "est_computed": est_computed, "computed_actual": computed_actual})
        if (CREDIT_ANCHOR != "off" or CREDIT_PROBE != "off") and route in ("local", "held") and info.get("ptok_exact_local"):
            try:
                _anchor_note((_PM_INFLIGHT.get(info.get("pm_ref")) or {}).get("chain") or [], info["ptok_exact_local"], now)
            except Exception:
                pass
        if CHAIN_TELEMETRY or WARM_PRIORITY:
            try:
                _chain_route_note((_PM_INFLIGHT.get(info.get("pm_ref")) or {}).get("chain") or [], route, now)
            except Exception:
                pass
        _pm_feedback(info)        # LS lane: grade + self-correct the cache-aware cost model
        _shape_stats_note(route, status, info)
        _est_err_note(route, name, info)
        # (e) append-only JSONL request log -- see DESIGN.md (e) / REPORT.md. Never write on the
        # request path: this only appends a small dict to a bounded in-memory list;
        # _jsonl_flusher() (sibling to _stats_saver()) does the actual blocking file I/O off the
        # event loop. Reuses every local already computed above -- no new work on this path
        # beyond building one small dict and a bounds-checked list.append().
        _telemetry_log_enqueue({
            "t": round(now, 3), "client": name, "ip": info.get("ip"),
            "xclient": info.get("xclient"), "ua": info.get("ua"),
            "ep": info.get("ep"), "model": info.get("model"),
            "request_id": info.get("request_id"),
            "request_id_source": info.get("request_id_source"),
            "request_body_sha256": info.get("request_body_sha256"),
            "route": route, "reason": reason,
            "alias": info.get("alias"), "alias_kind": info.get("alias_kind"),
            "waited": round(waited, 3) if waited else 0.0,
            "ttft": ttft, "ptok": info.get("ptok"), "maxtok": info.get("maxtok"),
            "outtok": outtok, "outtok_lb": outtok_lb,
            # RESPONSE-SHAPE fields. content_empty/has_tool_calls/finish_reason:
            #   - non-streaming remote: _forward_remote -> _note_remote_response ->
            #     classify_remote_response (shim-remote-observability lane, 2026-09-05).
            #   - ANY streaming response, local or remote (gw-streaming-content-classifier,
            #     2026-09-11): _relay()'s _sse_content_shape() scans every delta for
            #     non-whitespace content / tool_calls WITHOUT reassembling the full message --
            #     content_len is the one field that stays None for streaming (a running char
            #     count was judged not worth the extra per-chunk work; content_empty already
            #     answers the question content_len existed to answer).
            #   - non-streaming local: still None -- the local path already has its own
            #     real-time empty-response check (_is_empty_thinking_response()/EMPTY_RETRY),
            #     it just never persisted the verdict here; out of THIS lane's scope.
            "finish_reason": info.get("finish_reason"),
            "content_len": info.get("content_len"),
            "content_empty": info.get("content_empty"),
            "has_tool_calls": info.get("has_tool_calls"),
            # [GW2 / L94+L95] tool-call argument validity (JSON object + request tool schema), reasoning volume,
            # reasoning streamed before the first content/tool call (sizes the reasoning watchdog), its action.
            "tool_calls_n": info.get("tool_calls_n"), "tool_args_valid": info.get("tool_args_valid"),
            "tool_args_error": info.get("tool_args_error"),
            "reasoning_chars": info.get("reasoning_chars"),
            "reasoning_chars_at_output": info.get("reasoning_chars_at_output"),
            "reasoning_watchdog": info.get("reasoning_watchdog"),
            "reasoning_watchdog_at_tok": info.get("reasoning_watchdog_at_tok"),
            "reasoning_watchdog_ok": info.get("reasoning_watchdog_ok"),
            # [GW2] remote: the provider's usage; local: the ENGINE's usage.prompt_tokens (row `ptok` is the
            # gateway's ESTIMATE -- CR2 found it off by -5200..+2334 tokens, which read as a phantom engine bug).
            "ptok_exact": (info.get("ptok_exact") if info.get("ptok_exact") is not None
                           else (info.get("ptok_exact_local") if route in ("local", "held") else None)),
            "ptok_exact_src": ("provider" if info.get("ptok_exact") is not None else
                               ("engine" if route in ("local", "held") and info.get("ptok_exact_local") is not None
                                else None)),
            "ptok_est_err": ((info.get("est_tokens") or info.get("ptok") or 0) - info["ptok_exact_local"]
                             if route in ("local", "held") and info.get("ptok_exact_local") is not None else None),
            "predicted_occupancy_s": info.get("predicted_occupancy_s"),
            "context_provider": info.get("context_provider"),
            "context_limit": info.get("context_limit"),
            "context_prompt_tokens": info.get("context_prompt_tokens"),
            "context_compacted": bool(info.get("context_compacted")),
            "context_omitted": info.get("context_omitted", 0),
            # [RA / L143] identical-call loop note: present only when it fired (keeps every other row byte-identical)
            **({"repeat_tool_note": info["repeat_tool_note"]} if info.get("repeat_tool_note") else {}),
            "stream_watchdog": bool(info.get("stream_watchdog")),
            "stream_idle_timeout_s": info.get("stream_idle_timeout_s"),
            "duration": duration,
            "status": status, "stream": bool(info.get("stream")),
            "bg": bool(info.get("bg")), "tiny": bool(info.get("tiny")),
            "preview": info.get("preview"), "cost_est": req_cost, "cost_basis": cost_basis,
            "charged_usd": info.get("charged_usd"),
            "remote_model": info.get("remote_model"), "remote_cache_hit": info.get("remote_cache_hit"),
            "cost_policy": info.get("cost_policy"), "remote_provider": info.get("remote_provider"),
            "remote_sent": bool(info.get("remote_sent")),
            "remote_outcome": info.get("remote_outcome"), "remote_status": info.get("remote_status"),
            "remote_cache_miss": info.get("remote_cache_miss"),
            # gw-ttft-decomposition-telemetry: admission_wait duplicates `waited` under the
            # AC-named field so the JSONL is self-auditable against the card without a lookup
            # table; decode_time is new (see decompose_timing()'s docstring for why engine_queue
            # and prefill_time aren't split out here).
            **decompose_timing(waited, ttft, duration),
            # gw-admission-computed-token-cost: est_tokens/est_computed set unconditionally in
            # _route_completions (regardless of USE_COMPUTED_COST); computed_actual only when
            # the engine's usage trailer carried prompt_tokens (local only -- see _relay()).
            # None (not 0) whenever the engine side isn't known, same "missing != zero" rule as
            # decompose_timing() -- a null here must never be mistaken for "0 tokens computed".
            "est_tokens": info.get("est_tokens"), "est_computed": info.get("est_computed"),
            "computed_actual": info.get("computed_actual"),
            "cached_actual": info.get("cached_actual"),
            "pm_credit": info.get("pm_credit"), "pm_age_s": info.get("pm_age_s"),
            **({"pm_credit_anchored": info.get("pm_credit_anchored"), "pm_anchor_age_s": info.get("pm_anchor_age_s"),
                "pm_credit_probe": info.get("pm_credit_probe"), "probe_prompt_tokens": info.get("probe_prompt_tokens"),
                "probe_ms": info.get("probe_ms"), "credit_source": info.get("credit_source"),
                "pm_credit_model": info.get("pm_credit_model")}
               if (CREDIT_ANCHOR != "off" or CREDIT_PROBE != "off") else {}),
            **({k: info.get(k) for k in _QOL_FIELDS} if QOL_OVERFLOW != "off" else {}),
            "flow_class": info.get("flow_class"), "flow_expected_wait": info.get("flow_expected_wait_s"),
            "flow_held": info.get("flow_held"), "flow_adjacent": info.get("flow_adjacent"),
            **({"chain_prev_route": info.get("chain_prev_route"), "chain_prev_age_s": info.get("chain_prev_age_s"),
                "pm_match_tok": info.get("pm_match_tok"), "warm_priority": bool(info.get("warm_priority"))}
               if (CHAIN_TELEMETRY or WARM_PRIORITY) else {}),
        })
    except Exception as e:
        log.warning("telemetry note_request: %s", e)
def _telemetry_log_enqueue(rec):
    """Append one finished-request record to the pending list for _jsonl_flusher(). Called
    synchronously from _telemetry_note_request() -- already on the event loop, already inside
    that function's own try/except. A plain list.append() does no I/O (microseconds), so this
    never adds request-path latency. Bounded by TELEMETRY_QUEUE_MAX so a stalled disk / stuck
    executor can't grow this list without limit -- past the bound, new records are dropped and
    counted (_JSONL_STATE['dropped_queue']), never silently blocked or unbounded."""
    if len(_JSONL_PENDING) >= TELEMETRY_QUEUE_MAX:
        _JSONL_STATE["dropped_queue"] += 1
        return
    _JSONL_PENDING.append(rec)


def _history_day_file(epoch_seconds):
    return os.path.join(TELEMETRY_DIR, "requests-%s.jsonl" % time.strftime("%Y%m%d", time.gmtime(epoch_seconds)))


def _flush_jsonl_blocking(lines, cur_date, cur_path, cur_bytes, cur_capped):
    """Blocking -- only ever invoked via loop.run_in_executor (or, once, synchronously from
    _on_cleanup for a best-effort final flush at shutdown, exactly like _save_stats() already
    does there). Pure-ish: takes the flusher's last-known {date,path,bytes,capped} and returns
    the updated values; the caller applies them to _JSONL_STATE AFTER the await, mirroring the
    existing _gpu_stats_async() pattern (compute off-loop, mutate shared state back on the loop
    thread) instead of writing module globals from inside the executor thread.

    Size cap: once a day's cumulative bytes would exceed TELEMETRY_JSONL_MAX_MB, this STOPS
    writing for the rest of that UTC day and counts every further record as dropped_cap (the
    brief's chosen behaviour -- simpler and easier to reason about than DESIGN.md (e)'s original
    suggestion of rolling to a suffixed requests-YYYYMMDD.2.jsonl file; the trade-off, spelled out
    in REPORT.md, is that a single freak chatty day loses its tail instead of growing another
    file -- flip TELEMETRY_JSONL_MAX_MB up, or reinstate the suffix-roll design, if that's wrong.)
    """
    today = time.strftime("%Y%m%d", time.gmtime())
    rotated = False
    if today != cur_date:
        cur_date, cur_path, cur_bytes, cur_capped, rotated = today, None, 0, False, True
    if cur_path is None:
        try:
            os.makedirs(TELEMETRY_DIR, mode=0o700, exist_ok=True)
            os.chmod(TELEMETRY_DIR, 0o700)   # belt-and-braces: makedirs' mode is umask-masked
        except OSError:
            pass
        cur_path = os.path.join(TELEMETRY_DIR, "requests-%s.jsonl" % today)
        # Re-derive the true on-disk size instead of trusting the caller's cur_bytes (which is
        # always 0 here on a fresh process -- _JSONL_STATE starts at {"bytes": 0} on every
        # restart). Without this, a process that restarts N times in one day would enforce
        # N * TELEMETRY_JSONL_MAX_MB for that day instead of one real cap. FileNotFoundError
        # (first flush of a genuinely new day) just means the file doesn't exist yet -> 0, same
        # as before.
        try:
            cur_bytes = os.path.getsize(cur_path)
        except OSError:
            cur_bytes = 0
        # ...and if that pre-existing file is already at/over the cap (e.g. the process
        # restarted mid-way through an already-capped day), mark it capped immediately instead
        # of needing one more over-cap write attempt to notice.
        if cur_bytes >= int(TELEMETRY_JSONL_MAX_MB * 1024 * 1024):
            cur_capped = True
    if cur_capped:
        return {"date": cur_date, "path": cur_path, "bytes": cur_bytes, "capped": True,
                "written": 0, "dropped_cap": len(lines), "rotated": rotated, "dropped_bad": 0, "dropped_io": 0}
    max_bytes = int(TELEMETRY_JSONL_MAX_MB * 1024 * 1024)
    written = dropped_cap = dropped_bad = dropped_io = 0
    start_bytes = cur_bytes
    buf = []
    for rec in lines:
        try:
            s = json.dumps(rec, separators=(",", ":")) + "\n"
        except Exception:
            dropped_bad += 1   # one broken record must not lose the rest of the batch (lane SH: but it is counted)
            continue
        n = len(s.encode("utf-8"))
        if cur_bytes + n > max_bytes:
            cur_capped = True
            dropped_cap += 1
            continue
        buf.append(s)
        cur_bytes += n
        written += 1
    if buf:
        try:
            with open(cur_path, "a", encoding="utf-8") as f:
                f.write("".join(buf))
            os.chmod(cur_path, 0o600)   # previews can contain prompt text -- same as flightrec
        except OSError as e:
            log.warning("jsonl flush: write %s failed: %s", cur_path, e)
            # lane SH: nothing was written -- do not report the batch as written or advance the day's size
            dropped_io, written, cur_bytes = written, 0, start_bytes
    return {"date": cur_date, "path": cur_path, "bytes": cur_bytes, "capped": cur_capped,
            "written": written, "dropped_cap": dropped_cap, "rotated": rotated,
            "dropped_bad": dropped_bad, "dropped_io": dropped_io}


def _telemetry_retention_sweep():
    """Off-loop, called at most once per day (only when _jsonl_flusher notices the UTC date
    rolled over) -- deletes request-log files older than TELEMETRY_RETENTION_DAYS. Narrowly
    scoped on purpose: only files matching 'requests-YYYYMMDD.jsonl' / 'hw-YYYYMMDD.jsonl' inside TELEMETRY_DIR, never
    recursive, never touches anything else in that directory. Set
    SHIM_TELEMETRY_RETENTION_DAYS=0 to disable entirely."""
    if TELEMETRY_RETENTION_DAYS <= 0:
        return
    cutoff = time.time() - TELEMETRY_RETENTION_DAYS * 86400
    try:
        names = os.listdir(TELEMETRY_DIR)
    except OSError:
        return
    for name in names:
        if not (name.startswith(("requests-", "hw-")) and name.endswith(".jsonl")):
            continue
        fp = os.path.join(TELEMETRY_DIR, name)
        try:
            if os.path.isfile(fp) and os.stat(fp).st_mtime < cutoff:
                os.unlink(fp)
                log.info("telemetry retention: deleted %s (older than %d days)", name, TELEMETRY_RETENTION_DAYS)
        except OSError as e:
            log.warning("telemetry retention: %s: %s", name, e)


async def _jsonl_flusher():
    """Sibling task to _stats_saver(): wakes every TELEMETRY_FLUSH_SECS and, if anything is
    pending, hands it to a thread-pool executor for the actual blocking file I/O -- never on the
    request path. Started in _on_startup, cancelled in _on_cleanup: same lifecycle as every other
    background task in this file."""
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(TELEMETRY_FLUSH_SECS)
        if not _JSONL_PENDING:
            continue
        lines = list(_JSONL_PENDING)
        _JSONL_PENDING.clear()
        try:
            result = await loop.run_in_executor(
                None, _flush_jsonl_blocking, lines,
                _JSONL_STATE["date"], _JSONL_STATE["path"], _JSONL_STATE["bytes"], _JSONL_STATE["capped"])
            _JSONL_STATE["date"] = result["date"]
            _JSONL_STATE["path"] = result["path"]
            _JSONL_STATE["bytes"] = result["bytes"]
            _JSONL_STATE["capped"] = result["capped"]
            _JSONL_STATE["written"] += result["written"]
            _JSONL_STATE["dropped_cap"] += result["dropped_cap"]
            _JSONL_STATE["dropped_bad"] += result.get("dropped_bad", 0)
            _JSONL_STATE["dropped_io"] += result.get("dropped_io", 0)
            _JSONL_STATE["last_err"] = None if not result.get("dropped_io") else "write failed"
            if result.get("rotated"):
                await loop.run_in_executor(None, _telemetry_retention_sweep)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("jsonl flusher: %s (%d rows lost)", e, len(lines))
            _JSONL_STATE["last_err"] = str(e)
            _JSONL_STATE["dropped_err"] += len(lines)      # lane SH: the batch left the queue; count its loss


def _read_lines_reverse(path, max_bytes):
    """Yield a text file's lines from the end backward (last line first), reading bounded chunks
    from the tail so a caller that stops early (e.g. once it has `limit` matches) never pays for
    more than it needed -- the 'no full-file scan over 50MB' requirement. `max_bytes` is a hard
    cap on total bytes read from the tail (a worst-case guard for a filter that matches almost
    nothing in a huge file), not a chunk size -- chunks start at 64KB and double up to 1MB."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            scanned = 0
            chunk_size = 65536
            carry = b""
            while pos > 0 and scanned < max_bytes:
                read_size = min(chunk_size, pos, max_bytes - scanned)
                if read_size <= 0:
                    break
                pos -= read_size
                f.seek(pos)
                chunk = f.read(read_size)
                scanned += read_size
                data = chunk + carry
                parts = data.split(b"\n")
                carry = parts[0]   # possibly-partial first fragment; prefixed onto the next (earlier) chunk
                for ln in reversed(parts[1:]):
                    if ln:
                        yield ln
                chunk_size = min(chunk_size * 2, 1 << 20)
            if pos == 0 and carry:
                yield carry
    except FileNotFoundError:
        return
    except OSError as e:
        log.warning("history: read %s failed: %s", path, e)
        return


def _history_match(rec, client_q, route_q, since_ts):
    if since_ts is not None and (rec.get("t") or 0) < since_ts:
        return False
    if client_q and client_q not in str(rec.get("client") or "").lower():
        return False
    if route_q:
        hay = (str(rec.get("route") or "") + " " + str(rec.get("reason") or "")).lower()
        if route_q not in hay:
            return False
    return True


def _history_scan_blocking(client_q, route_q, since_ts, limit, max_files=7, per_file_cap=8 * 1024 * 1024):
    """Blocking (only ever called via run_in_executor). Walks requests-YYYYMMDD.jsonl newest-day
    first, tail-first within each day (see _read_lines_reverse), until `limit` matches are
    collected, `max_files` calendar days have been checked, or the day being examined already
    predates `since` -- whichever comes first. Newest-first order is preserved across files
    because an older day is only consulted after the newer one is exhausted/capped. Worst case
    (a filter that matches almost nothing) reads at most max_files * per_file_cap bytes -- see
    REPORT.md disk-math; the common case (last 50, no filter) returns after a few KB of today's
    file."""
    out = []
    files_scanned = []
    now = time.time()
    for day_i in range(max_files):
        day_epoch = now - day_i * 86400
        if since_ts is not None and day_epoch + 86400 < since_ts:
            break   # this whole day (and everything older) predates the caller's `since`
        fn = _history_day_file(day_epoch)
        if os.path.exists(fn):
            files_scanned.append(os.path.basename(fn))
            for raw in _read_lines_reverse(fn, per_file_cap):
                try:
                    rec = json.loads(raw)
                except Exception:
                    continue
                if _history_match(rec, client_q, route_q, since_ts):
                    out.append(rec)
                    if len(out) >= limit:
                        break
        if len(out) >= limit:
            break
    return {"items": out[:limit], "files_scanned": files_scanned}


def _parse_since(s):
    """Accepts a unix epoch (int/float string), a relative shorthand ('2h', '45m', '3d', '90s'),
    or 'YYYY-MM-DD[ HH:MM[:SS]]' / 'YYYY-MM-DDTHH:MM:SS'. Returns unix seconds, or None (treated
    as 'no filter', not a 400 -- this feeds a 10s dashboard poll; a bad query param should
    degrade, not break it)."""
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([smhd])$", s, re.I)
    if m:
        n, unit = float(m.group(1)), m.group(2).lower()
        return time.time() - n * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return time.mktime(time.strptime(s, fmt))
        except ValueError:
            continue
    return None


def _raw_pctl(sorted_vals, q):
    """Nearest-rank percentile over an already-sorted list of floats. None if empty. Hand-rolled
    (no `statistics` import) -- same stdlib-only, roll-it-yourself posture this file already
    applies to _hist_quantile() above, for the same reason (one small pure function, not worth a
    new dependency)."""
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))
    return round(sorted_vals[idx], 3)


def decompose_timing(waited, ttft, duration):
    """Split one request's shim-observed wall-clock into the phases the SHIM can actually see.

    gw-ttft-decomposition-telemetry (Kevin 2026-09-11) asked for admission_wait, engine_queue,
    prefill_time and decode_time as four separately-measured fields. The shim can only deliver
    three: engine_queue and prefill_time both happen between the upstream POST and the first
    streamed byte, entirely inside the engine's own scheduler -- the shim has no vantage point
    between them (the engine tracks them separately as vllm:request_queue_time_seconds /
    vllm:request_prefill_time_seconds, but those are process-wide Prometheus histograms, not
    attributable to one request under concurrency, and the per-request values that do exist
    internally -- see vllm's output_processor.do_tracing()/RequestStateStats -- are never
    surfaced on the OpenAI-compatible response; exposing them needs an engine-side patch, which
    needs a restart). What IS real and shim-only: admission_wait (this shim's own queue, always
    known), queue_plus_prefill (the combined engine phase -- exactly today's `ttft` field, kept
    under its existing name in the JSONL rather than duplicated), and decode_time (derived).

    ttft is None for every non-streaming response (see _relay()'s own TELEMETRY comment) --
    decode_time must then be None too, never a wrong 0."""
    admission_wait = round(waited or 0.0, 3)
    if ttft is None or duration is None:
        return {"admission_wait": admission_wait, "queue_plus_prefill": None, "decode_time": None}
    queue_plus_prefill = round(ttft, 3)
    decode_time = round(max(0.0, duration - admission_wait - queue_plus_prefill), 3)
    return {"admission_wait": admission_wait, "queue_plus_prefill": queue_plus_prefill, "decode_time": decode_time}


def _latency_class_pctls(rows):
    """rows: iterable of (is_background, admission_wait, queue_plus_prefill, decode_time)
    already-observed tuples (any of the three may be None). Buckets by class exactly like
    _waiting_by_class (bg=True -> background, else interactive) and returns p50/p90 -- p90, not
    the p95 the pre-existing duration/ttft aggregate uses, matching this card's own AC and
    gw-slo-panel's target definitions (p50/p90 per class)."""
    buckets = {"interactive": {"admission_wait": [], "queue_plus_prefill": [], "decode_time": []},
               "background": {"admission_wait": [], "queue_plus_prefill": [], "decode_time": []}}
    for is_bg, aw, qp, dt in rows:
        b = buckets["background" if is_bg else "interactive"]
        if aw is not None:
            b["admission_wait"].append(aw)
        if qp is not None:
            b["queue_plus_prefill"].append(qp)
        if dt is not None:
            b["decode_time"].append(dt)
    out = {}
    for cls, metrics in buckets.items():
        out[cls] = {}
        for metric, vals in metrics.items():
            vals.sort()
            out[cls][metric] = {"p50": _raw_pctl(vals, 0.50), "p90": _raw_pctl(vals, 0.90), "n": len(vals)}
    return out


def _history_summary_blocking(since_ts, max_files, hard_line_cap=500000):
    """Off-loop full read (forward this time -- aggregation needs every line in the window
    anyway, so the reverse tail-reader's early-exit trick buys nothing here) of each day-file
    touching [since_ts, now]. `hard_line_cap` is a last-resort circuit breaker for a
    misconfigured huge `hours` value, not expected to bite at the documented 200MB/day cap."""
    now = time.time()
    per_client = collections.defaultdict(lambda: {"requests": 0, "local": 0, "remote": 0,
                                                    "tokens_in": 0, "tokens_out": 0, "errors": 0,
                                                    "remote_cache_hit_tokens": 0,
                                                    "remote_cache_miss_tokens": 0,
                                                    "remote_cost_usd": 0.0,
                                                    "exact_body_repeats": 0})
    per_route = collections.defaultdict(lambda: {"requests": 0, "tokens_out": 0, "errors": 0})
    remote_by_reason = collections.Counter()      # Lane DB2: why requests went to the paid provider, from the on-disk log
    per_class = collections.defaultdict(lambda: {"requests": 0, "remote": 0})     # work class (flow_class) -> counts
    durations, ttfts = [], []
    latency_rows = []   # gw-ttft-decomposition-telemetry: (is_bg, admission_wait, queue_plus_prefill, decode_time)
    files_scanned = []
    lines_seen = 0
    truncated = False
    seen_bodies = set()
    for day_i in range(max_files):
        day_epoch = now - day_i * 86400
        if day_epoch + 86400 < since_ts:
            break
        fn = _history_day_file(day_epoch)
        if not os.path.exists(fn):
            continue
        files_scanned.append(os.path.basename(fn))
        try:
            with open(fn, "r", encoding="utf-8") as f:
                for raw in f:
                    lines_seen += 1
                    if lines_seen > hard_line_cap:
                        truncated = True
                        break
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        rec = json.loads(raw)
                    except Exception:
                        continue
                    t = rec.get("t")
                    if t is None or t < since_ts:
                        continue
                    name = rec.get("client") or "?"
                    route = rec.get("route") or "?"
                    status = rec.get("status")
                    is_err = ((status is not None and status >= 400)
                              or rec.get("reason") == "failover"
                              or rec.get("stream_watchdog"))
                    outtok = rec.get("outtok")
                    if outtok is None:
                        outtok = rec.get("outtok_lb") or 0
                    pc = per_client[name]
                    pc["requests"] += 1
                    if route in ("local", "remote"):
                        pc[route] += 1
                    pc["tokens_in"] += rec.get("ptok") or 0
                    pc["tokens_out"] += outtok or 0
                    body_sha = rec.get("request_body_sha256")
                    if body_sha:
                        body_key = (name, body_sha)
                        if body_key in seen_bodies:
                            pc["exact_body_repeats"] += 1
                        else:
                            seen_bodies.add(body_key)
                    if route == "remote" and rec.get("remote_sent"):
                        pc["remote_cache_hit_tokens"] += rec.get("remote_cache_hit") or 0
                        pc["remote_cache_miss_tokens"] += rec.get("remote_cache_miss") or 0
                        pc["remote_cost_usd"] += float(rec.get("cost_est") or 0.0)
                    if is_err:
                        pc["errors"] += 1
                    pcl = per_class[rec.get("flow_class") or "?"]
                    pcl["requests"] += 1
                    if route == "remote":
                        pcl["remote"] += 1
                        remote_by_reason[rec.get("reason") or "?"] += 1
                    pr = per_route[route]
                    pr["requests"] += 1
                    pr["tokens_out"] += outtok or 0
                    if is_err:
                        pr["errors"] += 1
                    if rec.get("duration") is not None:
                        durations.append(rec["duration"])
                    if rec.get("ttft") is not None:
                        ttfts.append(rec["ttft"])
                    latency_rows.append((rec.get("bg"), rec.get("admission_wait"),
                                         rec.get("queue_plus_prefill"), rec.get("decode_time")))
        except OSError as e:
            log.warning("history summary: read %s failed: %s", fn, e)
        if truncated:
            break
    durations.sort()
    ttfts.sort()
    total_requests = sum(c["requests"] for c in per_client.values())
    for pc in per_client.values():
        pc["remote_cost_usd"] = round(pc["remote_cost_usd"], 6)
    return {
        "per_client": dict(per_client), "per_route": dict(per_route),
        "remote_by_reason": dict(remote_by_reason.most_common()), "per_class": dict(per_class),
        "latency_by_class": _latency_class_pctls(latency_rows),
        "duration_p50": _raw_pctl(durations, 0.50), "duration_p95": _raw_pctl(durations, 0.95),
        "ttft_p50": _raw_pctl(ttfts, 0.50), "ttft_p95": _raw_pctl(ttfts, 0.95),
        "requests": total_requests, "files_scanned": files_scanned, "truncated": truncated,
    }
# ---------------- end TELEMETRY module; record_event() and _gpu_stats() below are the
# existing functions, edited in place to plug into it -------------------------------------


def record_event(decision, reason, request, units, waited, ptok=0, maxtok=0, stream=False):
    # route values: "local" / "remote" (unchanged), plus two BG-LOCAL-ONLY additions --
    # "held" (a background request that waited out a local-availability event and was served
    # LOCALLY once it recovered) and "rejected-bg" (503'd instead of ever reaching remote).
    # Neither ever touched the remote provider, so neither is counted in _remote_reasons
    # (that breakdown is specifically "why did we pay for remote").
    prior = _ACTIVE.get(id(request)) or {}
    # A post-commit stream watchdog has a more specific outcome than the admission reason
    # (usually "-" for a normal local request). Preserve it in the receipt and history.
    final_reason = "stream-idle-timeout" if prior.get("stream_watchdog") else reason
    _active_set(request, route=decision, reason=final_reason,
                phase=("done" if decision in ("local", "held") else "remote"))
    _stats["total"] += 1
    if decision == "local":
        _stats["local"] += 1
    elif decision == "held":
        _stats["held"] += 1
    elif decision == "rejected-bg":
        _stats["rejected_bg"] += 1
    elif decision == "gone":
        _stats["client_gone"] = _stats.get("client_gone", 0) + 1
    else:
        _stats["remote"] += 1
        _remote_reasons[reason] += 1
    if waited and waited > 0:
        _stats["waited_total"] += waited
        _stats["waited_n"] += 1
    _stats["peak_inflight"] = max(_stats["peak_inflight"], _inflight)
    flow_note_route(decision, final_reason, request)
    # TELEMETRY: pull whatever _relay()/_forward_remote()/the local-success branches have
    # already stashed on this request's live-registry entry (ttft/outtok are None for the
    # "record-before-forward" remote branches -- see DESIGN.md (c) for exactly why, and where
    # the accurate post-hoc numbers live instead: _telemetry_note_request(), fed from the same
    # _ACTIVE entry but read later, in handle_completions' finally, after the request finishes).
    _rinfo = _ACTIVE.get(id(request)) or {}
    _outtok, _outtok_lb = _rinfo.get("outtok"), _rinfo.get("outtok_lb")
    _events.appendleft({"t": round(time.time(), 1), "d": decision, "r": final_reason,
                        "client": _client_label(request), "units": units,
                        "waited": round(waited or 0, 1),
                        "ep": request.path.rsplit("/", 1)[-1],
                        "ptok": ptok, "maxtok": maxtok, "stream": stream,
                        "alias": _rinfo.get("alias"), "alias_kind": _rinfo.get("alias_kind"),
                        "cls": _rinfo.get("flow_class"),
                        "ttft": _rinfo.get("ttft"), "outtok": _outtok, "outtok_lb": _outtok_lb,
                        "cost_est": (_remote_cost_estimate(ptok, _outtok if _outtok is not None else _outtok_lb)
                                     if decision == "remote" else 0.0)})

def _gpu_stats():
    now = time.time()
    if now - _gpu_cache["at"] < 1.5:
        return _gpu_cache["data"]
    if _gpu_cache["at"] == 0.0:      # never sampled yet (startup race) -- one-time blocking
        _gpu_cache.update(at=now, data=_gpu_query_blocking())   # fallback; see _gpu_stats_async()
    return _gpu_cache["data"]

# ---------------- metrics persistence (survive gateway restarts) ----------------
def _save_stats():
    try:
        with open(STATS_FILE + ".tmp", "w") as f:
            json.dump({"stats": _stats, "reasons": dict(_remote_reasons), "events": list(_events)}, f)
        os.replace(STATS_FILE + ".tmp", STATS_FILE)
    except Exception as e:
        log.warning("save stats failed: %s", e)

def _load_stats():
    try:
        if not os.path.exists(STATS_FILE):
            return
        d = json.load(open(STATS_FILE))
        st = d.get("stats", {})
        for k in list(_stats.keys()):
            if k in st:
                _stats[k] = st[k]          # includes original "started" -> cumulative uptime
        _remote_reasons.update(d.get("reasons", {}))
        for e in reversed(d.get("events", [])):
            _events.appendleft(e)
        log.info("restored stats: total=%d (%d events)", _stats.get("total", 0), len(_events))
    except Exception as e:
        log.warning("load stats failed: %s", e)

async def _stats_saver():
    while True:
        await asyncio.sleep(10)
        _save_stats()


# ---------------- CF: capacity-aware flow (lane CF, 2026-10-02) ----------------
# Kevin 10-02: "it needs to be aware of how to optimize its flow for local only versus local plus
# remote versus remote. Queuing? ... we might have to get creative here."
#
# MEASURED before this block (telemetry 2026-10-01 17:00 -> 10-02 12:07, 9,064 requests): the gateway's
# admission wait was ~0 (budget 14 lanes) while engine TTFT p95 was 79-117 s per class. The queue lived
# INSIDE the engine, FIFO: vllm request_queue_time mean 34.6 s vs prefill 8.9 s, with 7 requests
# 'deferred'. Priority, deadlines and prefix affinity were therefore impossible -- by the time the gateway
# had admitted a Halo call it already stood behind whatever background work had been let in earlier, and
# the caller with the shortest timeout gave up first, discarding the prefill already done for it.
#
# The mechanics here (facts + ordering + a hard admission ceiling; the JUDGEMENT about shares is Halo's,
# set through /gateway/config flow_* fields by the capacity_set_flow tool):
#   * every request gets a WORK CLASS: kevin (interactive) > halo (mind) > runner (cards) > background.
#   * the engine's prefill queue is kept SHORT: a class is admitted only while the uncached prefill already
#     in the engine (seconds at the measured rate) is under that class's ceiling. The excess waits HERE,
#     where it can be ordered, refused with Retry-After, or overflowed.
#   * the waiters are served by start-time fair queuing over per-class shares (kevin's share is large
#     enough to be strict priority), earliest-deadline first inside a class when slack is short, then
#     same-prefix first (cache affinity), then FIFO.
#   * a request whose prefix is being prefilled RIGHT NOW is held until that prefill finishes when waiting
#     is cheaper than recomputing the shared part (it then reads the cache instead of duplicating it).
#   * deadline-aware admission: a refusable caller (background, runner) that cannot START before its
#     declared deadline is refused 429 + Retry-After BEFORE any prefill. Nothing prefilling is discarded.
#   * /gateway/capacity publishes the facts: mode, measured throughput, demand and queues by class.
# SHIM_FLOW_MODE: off | shadow | enforce.  off = the previous behaviour byte for byte; shadow = everything
# measured and reported, nothing gated (counts what enforce would have held).
FLOW_CLASSES = ("kevin", "halo", "runner", "background")     # priority order, highest first
FLOW_REFUSABLE = ("runner", "background")                     # classes that may be refused for a missed deadline


def _flow_parse_map(v, cast=float, positive=False):
    out = {}
    for part in str(v or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        k, _, val = part.partition("=")
        k = k.strip().lower()
        if k not in FLOW_CLASSES:
            raise ValueError("unknown work class %r (classes: %s)" % (k, ", ".join(FLOW_CLASSES)))
        x = cast(val.strip())
        if x < 0 or (positive and x <= 0):
            raise ValueError("%s must be %s" % (k, "> 0" if positive else ">= 0"))
        out[k] = x
    return out


def _flow_norm(cast=float, positive=False):
    def f(v):
        d = _flow_parse_map(v, cast, positive)
        return ",".join("%s=%g" % (k, d[k]) for k in FLOW_CLASSES if k in d)
    return f


def _flow_parse_classmap(v):
    out = []
    for part in str(v or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        pat, _, cls = part.rpartition("=")
        pat, cls = pat.strip().lower(), cls.strip().lower()
        if not pat or cls not in FLOW_CLASSES:
            raise ValueError("bad class map entry %r (pattern=class, class in %s)" % (part, ", ".join(FLOW_CLASSES)))
        out.append((pat, cls))
    return out


def _flow_norm_classmap(v):
    return ",".join("%s=%s" % pc for pc in _flow_parse_classmap(v))


def _flow_cast_mode(v):
    v = str(v).strip().lower()
    if v not in ("off", "shadow", "enforce"):
        raise ValueError("flow_mode must be off|shadow|enforce")
    return v


FLOW_MODE = os.environ.get("SHIM_FLOW_MODE", "enforce").strip().lower()
FLOW_SHARES = os.environ.get("SHIM_FLOW_SHARES", "kevin=1000,halo=60,runner=25,background=10")
# Fraction of FLOW_BACKLOG_S of uncached prefill a class may leave queued in the engine. 0 = no ceiling.
FLOW_CEIL = os.environ.get("SHIM_FLOW_CEIL", "kevin=0,halo=1.5,runner=1,background=0.5")
FLOW_BACKLOG_S = float(os.environ.get("SHIM_FLOW_BACKLOG_S", "40"))
# Default 'must START within' seconds for a caller that declared nothing (0 = never refused).
FLOW_DEADLINES = os.environ.get("SHIM_FLOW_DEADLINES", "kevin=0,halo=0,runner=900,background=600")
FLOW_CLASS_MAP = os.environ.get("SHIM_FLOW_CLASS_MAP", ",".join((
    "halo-=halo", "estate-entity=halo", "card-repair=runner", "work-verifier=runner",
    "control=kevin", "brain-passthrough=kevin", "pi /=kevin", "opencode=kevin", "openhands=kevin", "desktop=kevin",
    "overseer-=background", "vault-dreams=background", "digester=background", "acceptance-review=background",
    "research=background", "workflow-bg=background", "applicant=background", "m1c-probe=background",
    "outcome-alarm=background", "cron=background", "batch=background")))
# Phrases (first 500 chars of the first two messages) that mark a class when the client name cannot: the card
# runner's pi sessions arrive from the same host/provider as Kevin's own pi, 100% of them opening with this line.
FLOW_MARKERS = os.environ.get("SHIM_FLOW_MARKERS", "local-lane-runner batch job=runner")
# Request `model` -> class, checked before the client map: lets a harness that cannot send a per-request header (Hermes,
# LibreChat) put Kevin's own chats in a distinct model alias and so in the interactive class. Empty by default.
FLOW_MODEL_MAP = os.environ.get("SHIM_FLOW_MODEL_MAP", "")
FLOW_DEMAND_WINDOW_S = float(os.environ.get("SHIM_FLOW_DEMAND_WINDOW_S", "300"))      # window of the demand_5m facts (env only; the drill shortens it)
FLOW_AFFINITY_MAX = int(os.environ.get("SHIM_FLOW_AFFINITY_MAX", "6"))      # same-prefix grants in a row before FIFO order resumes
FLOW_STARVE_S = float(os.environ.get("SHIM_FLOW_STARVE_S", "300"))          # a waiting class unserved this long ignores its ceiling once
FLOW_PREFIX_HOLD_MAX_S = float(os.environ.get("SHIM_FLOW_PREFIX_HOLD_MAX_S", "60"))
FLOW_URGENT_SLACK_S = float(os.environ.get("SHIM_FLOW_URGENT_SLACK_S", "30"))   # a deadline closer than this goes first (EDF)
_CFG.update({
    "SHIM_FLOW_MODE":         ("FLOW_MODE", _flow_cast_mode),
    "SHIM_FLOW_SHARES":       ("FLOW_SHARES", _flow_norm(float, positive=True)),
    "SHIM_FLOW_CEIL":         ("FLOW_CEIL", _flow_norm(float)),
    "SHIM_FLOW_BACKLOG_S":    ("FLOW_BACKLOG_S", float),
    "SHIM_FLOW_DEADLINES":    ("FLOW_DEADLINES", _flow_norm(float)),
    "SHIM_FLOW_CLASS_MAP":    ("FLOW_CLASS_MAP", _flow_norm_classmap),
    "SHIM_FLOW_MARKERS":      ("FLOW_MARKERS", _flow_norm_classmap),
    "SHIM_FLOW_MODEL_MAP":    ("FLOW_MODEL_MAP", _flow_norm_classmap),
    "SHIM_FLOW_AFFINITY_MAX": ("FLOW_AFFINITY_MAX", int),
    "SHIM_FLOW_STARVE_S":     ("FLOW_STARVE_S", float),
    "SHIM_FLOW_PREFIX_HOLD_MAX_S": ("FLOW_PREFIX_HOLD_MAX_S", float),
    "SHIM_FLOW_URGENT_SLACK_S": ("FLOW_URGENT_SLACK_S", float),
})
_FLOW_DEFAULT_SHARES = {"kevin": 1000.0, "halo": 60.0, "runner": 25.0, "background": 10.0}
_FLOW_EVENTS_FILE = os.environ.get("SHIM_FLOW_EVENTS_FILE") or os.path.join(
    os.path.dirname(STATS_FILE), "incidents", "capacity-mode.jsonl")

_FLOW = {
    "waiters": [],                                    # FlowTicket, arrival order
    "vtime": {c: 0.0 for c in FLOW_CLASSES}, "vclock": 0.0,
    "last_prefix": None, "run": 0,                    # last admitted prefix and how many in a row
    "last_admit": {c: 0.0 for c in FLOW_CLASSES},
    "prefilling": {},                                 # id(request) -> {keys,cum,total,est_s,t0,prefix}
    "version": 0, "head": (-1, 0.0, None),
    "seq": 0,
}
_FLOW_STATS = collections.Counter()
_FLOW_DEMAND = collections.deque(maxlen=20000)       # (t, class, ptok, est_computed_tokens)
_FLOW_WAITS = collections.deque(maxlen=4000)         # (t, class, waited_s) per ticket that left the queue
_FLOW_METER = collections.deque(maxlen=900)          # (t, compute_tok_s, decode_tok_s, running, waiting, hit_rate)
_FLOW_PURE = collections.deque(maxlen=3000)          # (t, computed tokens, prefill seconds) deltas between scrapes
_FLOW_SERVICE = collections.deque(maxlen=400)        # (t, class, duration_s) local completions
_FLOW_CACHE = {"adjacent": [0, 0, 0], "other": [0, 0, 0]}   # [requests, cached_tokens, prompt_tokens]
_FLOW_MODE_STATE = {"mode": None, "since": None, "basis": None, "events": collections.deque(maxlen=200)}


_FLOW_MEMO = {}


def _flow_get(name, raw, base, positive=False):
    """Parse a per-class map once per distinct configured string; a bad string falls back to the defaults."""
    key = (name, raw)
    hit = _FLOW_MEMO.get(key)
    if hit is None:
        try:
            d = _flow_parse_map(raw, float, positive)
        except ValueError:
            d = {}
        hit = {c: d.get(c, base[c]) for c in FLOW_CLASSES}
        if len(_FLOW_MEMO) > 64:
            _FLOW_MEMO.clear()
        _FLOW_MEMO[key] = hit
    return hit


def _flow_shares():
    return _flow_get("shares", FLOW_SHARES, _FLOW_DEFAULT_SHARES, True)


def _flow_ceils():
    return _flow_get("ceil", FLOW_CEIL, {"kevin": 0.0, "halo": 1.5, "runner": 1.0, "background": 0.5})


def _flow_deadlines():
    return _flow_get("deadlines", FLOW_DEADLINES, {"kevin": 0.0, "halo": 0.0, "runner": 900.0, "background": 600.0})


def flow_class_of(request, body, background, halo_control):
    """Work class: caller's X-Work-Class (valid names only) > Halo control turn > the configured
    X-Client/friendly-name patterns > the legacy background test > interactive."""
    h = (request.headers.get("X-Work-Class") or "").strip().lower()
    if h in FLOW_CLASSES:
        return h
    try:
        models = _flow_parse_classmap(FLOW_MODEL_MAP)
        if models:
            mdl = str(json.loads(body).get("model") or "").strip().lower()
            for pat, cls in models:
                if mdl == pat:
                    return cls
    except Exception:
        pass
    if halo_control:
        return "halo"
    try:
        marks = _flow_parse_classmap(FLOW_MARKERS)
        if marks:
            msgs = (json.loads(body).get("messages") or [])[:2]
            for m in msgs:
                c = m.get("content")
                text = c if isinstance(c, str) else " ".join(b.get("text", "") for b in (c or [])[:2] if isinstance(b, dict))
                head = (text or "")[:500].lower()
                for pat, cls in marks:
                    if pat in head:
                        return cls
    except Exception:
        pass
    try:
        fc = _friendly_client(request)
        key = ("%s\n%s" % (fc.get("xclient") or "", fc.get("name") or "")).lower()
        for pat, cls in _flow_parse_classmap(FLOW_CLASS_MAP):
            if pat in key:
                return cls
    except Exception:
        pass
    return "background" if background else "kevin"


class FlowTicket:
    """One request waiting for (or holding) a place in the local engine."""
    __slots__ = ("cls", "t_enq", "seq", "deadline_at", "declared", "pm", "prefix", "cost_s", "units", "ptok",
                 "est_computed", "fits", "xclient", "rid", "expected_wait_s", "adjacent", "held")

    def __init__(self, cls, pm, ptok, units, fits, deadline_at, declared, xclient, rid, cost_s):
        self.cls, self.pm, self.ptok, self.units, self.fits = cls, pm, ptok, units, fits
        self.deadline_at, self.declared, self.xclient, self.rid = deadline_at, declared, xclient, rid
        self.est_computed = (pm or {}).get("computed", ptok)
        chain = (pm or {}).get("chain") or []
        self.prefix = chain[0][0].hex()[:12] if chain else None
        self.cost_s, self.t_enq, self.seq = cost_s, time.time(), 0
        self.expected_wait_s, self.adjacent, self.held = None, False, ""


_FLOW_ERR = {"n": 0, "last": 0.0}


def _flow_failopen(default):
    """CF can never take the gateway down: any exception inside a flow hook is logged (rate-limited) and counted, and
    the hook answers `default` -- which is always the legacy behaviour (admit / no refusal / no-op)."""
    def deco(fn):
        def wrapper(*a, **k):
            try:
                return fn(*a, **k)
            except Exception as e:          # noqa: BLE001
                _FLOW_ERR["n"] += 1
                _FLOW_STATS["failopen_" + fn.__name__] += 1
                if time.time() - _FLOW_ERR["last"] > 60:
                    _FLOW_ERR["last"] = time.time()
                    log.exception("flow hook %s failed (%d so far); serving with legacy admission: %s", fn.__name__, _FLOW_ERR["n"], e)
                return default
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper
    return deco


def flow_prefill_tps():
    """Measured uncached-prefill rate (tok/s): median over recent samples in which the engine had a queue
    (so it was working at capacity), else the configured PREFILL_TPS."""
    vals = sorted(r[1] for r in list(_FLOW_METER)[-120:] if r[1] and r[1] > 0 and (r[4] or 0) > 0)
    if len(vals) >= 5:
        return max(50.0, vals[len(vals) // 2])
    return prefill_tps()


def flow_prefill_pure_tps(now=None, window=300.0):
    """Per-request prefill speed with queue and admission wait excluded (see flow_meter_update), or None when the last
    `window` seconds hold too little prefill to say (< 10 prefill-seconds or < 1,000 computed tokens)."""
    now = time.time() if now is None else now
    rows = [r for r in _FLOW_PURE if now - r[0] <= window]
    toks, secs = sum(r[1] for r in rows), sum(r[2] for r in rows)
    if secs < 10.0 or toks < 1000:
        return None
    return toks / secs


# ================= CAPACITY MODEL: derived from the LIVE engine (lane GW, 2026-10-02) =================
# Why: shim.env carried hand-set capacity numbers (SHIM_POOL_TOKENS=637560, SHIM_TOKEN_BUDGET=500000, SHIM_PREFILL_TPS=1100)
# calibrated against an engine that no longer exists; the engine now has a 922,358-token KV pool and prefills at ~1,270
# tok/s, and the gateway kept admitting as if neither had changed. Rule: code supplies facts and mechanics. Every number
# here is read from the running engine's own /metrics (scraped every TELEM_SAMPLE_SECS by the telemetry sampler); the
# configured value is only the fallback used when the engine has never answered (or SHIM_CAPACITY_LIVE=0).
#
#   KV pool        vllm:cache_config_info{kv_cache_size_tokens}  (== the boot log's "GPU KV cache size: N tokens")
#   token budget   TOKEN_BUDGET_FRAC x live pool, capped at TOKEN_BUDGET_CEIL; SHIM_TOKEN_BUDGET (int) is an explicit override
#   prefill rate   p75 of per-minute pure per-request rates (computed tokens / prefill seconds, queue EXCLUDED) over the
#                  CURRENT engine generation only (after a settle period, planned offline windows excluded)
#   attention block vllm:cache_config_info{block_size}  (the prefix-cache credit is rounded to whole blocks)
_CAPLIVE = {"pool": None, "pool_at": 0.0, "pool_src": None, "block": None, "gen_start": None, "gen_src": None,
            "scrapes": 0, "events": collections.deque(maxlen=20), "measure": None,
            "ctx": None, "ctx_at": 0.0, "ctx_poll_at": 0.0, "ctx_err": None, "ctx_models": None}
PREFILL_MEASURE_WINDOW_S = float(os.environ.get("SHIM_PREFILL_MEASURE_WINDOW_S", "1800"))
PREFILL_MEASURE_BUCKET_S = float(os.environ.get("SHIM_PREFILL_MEASURE_BUCKET_S", "60"))
PREFILL_MEASURE_MIN_BUCKETS = int(os.environ.get("SHIM_PREFILL_MEASURE_MIN_BUCKETS", "5"))
PREFILL_MEASURE_QUANTILE = float(os.environ.get("SHIM_PREFILL_MEASURE_QUANTILE", "0.75"))
# Same settle/pad as tools/agent_config_standard.py (lane AC2): samples start this long after the engine generation began
# (CUDA graphs, cold prefix cache) and this long after a planned offline window closed (a benchmark shared the engine).
PREFILL_MEASURE_SETTLE_S = float(os.environ.get("SHIM_PREFILL_MEASURE_SETTLE_S", "180"))
PREFILL_MEASURE_OFFLINE_PAD_S = float(os.environ.get("SHIM_PREFILL_MEASURE_OFFLINE_PAD_S", "180"))
_PREFILL_BUCKET_MIN_SECS = 2.0         # a minute with less prefill than this says nothing about the rate
_PREFILL_BUCKET_MIN_TOKENS = 1000
_PREFILL_MEASURE_TTL_S = 5.0
_OFFLINE_SPANS = collections.deque(maxlen=64)       # (t0, t1) planned local-offline windows that closed (see _flow_event)
_POOL_STALE_FRAC = 0.05


def _capacity_engine_changed(now, why):
    """The engine process behind :8001 is a new generation (restart / came back after being down): everything measured
    on the previous one stops counting."""
    _CAPLIVE["events"].appendleft({"at": round(now, 1), "event": why})
    _CAPLIVE["measure"] = None
    _CAPLIVE["ctx_poll_at"] = 0.0          # a new engine may serve a different max_model_len: re-read /v1/models at once
    if _CAPLIVE["gen_src"] != "process_start_time_seconds":
        _CAPLIVE["gen_start"], _CAPLIVE["gen_src"] = now, "health transition"


def capacity_note_scrape(fam, now=None):
    """Called by the engine scrape with the parsed /metrics families: refresh the live KV pool, the attention block size and
    the engine generation. Cheap (one info line); runs on every sample, so a restart or a changed serve config is seen
    within one sample interval."""
    try:
        now = time.time() if now is None else now
        _CAPLIVE["scrapes"] += 1
        info = (fam.get("vllm:cache_config_info") or [({}, 1.0)])[0][0]
        pool, src = None, None
        try:
            pool = int(float(info.get("kv_cache_size_tokens")))
            src = "kv_cache_size_tokens"
        except (TypeError, ValueError):
            try:
                pool = int(info["num_gpu_blocks"]) * int(info["block_size"])
                src = "num_gpu_blocks x block_size (upper bound on hybrid models)"
            except (KeyError, TypeError, ValueError):
                pass
        if pool and pool > 0:
            if _CAPLIVE["pool"] != pool:
                _CAPLIVE["events"].appendleft({"at": round(now, 1), "event": "kv pool %s -> %s tokens" % (_CAPLIVE["pool"], pool)})
                log.info("capacity: live engine KV pool %s -> %s tokens (%s); configured SHIM_POOL_TOKENS=%s",
                         _CAPLIVE["pool"], pool, src, POOL_TOKENS)
                _CAPLIVE["measure"] = None
            _CAPLIVE["pool"], _CAPLIVE["pool_at"], _CAPLIVE["pool_src"] = pool, now, src
        try:
            blk = int(info.get("block_size"))
            _CAPLIVE["block"] = blk if blk > 0 else _CAPLIVE["block"]
        except (TypeError, ValueError):
            pass
        start = _fv(fam, "process_start_time_seconds")
        if start and start > 0:
            if _CAPLIVE["gen_src"] == "process_start_time_seconds" and abs(start - (_CAPLIVE["gen_start"] or 0)) > 2.0:
                _capacity_engine_changed(now, "engine restarted (process start %s -> %s)" % (
                    time.strftime("%H:%M:%S", time.localtime(_CAPLIVE["gen_start"])), time.strftime("%H:%M:%S", time.localtime(start))))
            _CAPLIVE["gen_start"], _CAPLIVE["gen_src"] = float(start), "process_start_time_seconds"
        elif _CAPLIVE["gen_start"] is None:
            _CAPLIVE["gen_start"], _CAPLIVE["gen_src"] = now, "first scrape"
    except Exception as e:      # noqa: BLE001 -- the capacity model must never break the scrape
        log.warning("capacity_note_scrape: %s", e)


def pool_info(now=None):
    """The KV pool the admission math should use, with where it came from: live (this engine answered within the last
    scrapes), live-last-known (it answered earlier and is unreachable now: still a better guess than a hand-set number),
    or configured (SHIM_POOL_TOKENS: the engine has never answered, or SHIM_CAPACITY_LIVE=0)."""
    now = time.time() if now is None else now
    cfg, live = int(POOL_TOKENS), _CAPLIVE["pool"]
    out = {"configured_tokens": cfg, "live_tokens": live}
    if CAPACITY_LIVE and live:
        reachable = bool(_ENGINE_METRICS.get("ok"))
        out.update(tokens=live, source="live" if reachable else "live-last-known", detail=_CAPLIVE["pool_src"],
                   age_s=round(max(0.0, now - _CAPLIVE["pool_at"]), 1), engine_reachable=reachable,
                   configured_stale=abs(live - cfg) > _POOL_STALE_FRAC * live)
    else:
        out.update(tokens=cfg, source="configured", detail=("SHIM_CAPACITY_LIVE=0" if not CAPACITY_LIVE else "engine has not answered yet"),
                   age_s=None, engine_reachable=bool(_ENGINE_METRICS.get("ok")), configured_stale=None)
    return out


_CTX_MIN_SANE = 1024                   # a max_model_len below this is a garbage reading, not a window


def capacity_note_models(payload, now=None):
    """Called with the engine's parsed /v1/models: the context window is the SMALLEST max_model_len any served model
    reports (aliases of one engine agree; if they ever differ the smaller one is the one a request can rely on)."""
    try:
        now = time.time() if now is None else now
        lens = []
        for m in (payload.get("data") or []) if isinstance(payload, dict) else []:
            try:
                v = int(m.get("max_model_len"))
            except (TypeError, ValueError, AttributeError):
                continue
            if v >= _CTX_MIN_SANE:
                lens.append(v)
        if not lens:
            _CAPLIVE["ctx_err"] = "no model in /v1/models reported a usable max_model_len"
            return
        win = min(lens)
        if _CAPLIVE["ctx"] != win:
            _CAPLIVE["events"].appendleft({"at": round(now, 1), "event": "context window %s -> %s tokens" % (_CAPLIVE["ctx"], win)})
            log.info("capacity: live engine max_model_len %s -> %s tokens; configured SHIM_LOCAL_CONTEXT_LIMIT=%s SHIM_MAX_LOCAL_TOKENS=%s",
                     _CAPLIVE["ctx"], win, LOCAL_CONTEXT_LIMIT, MAX_LOCAL_TOKENS)
        _CAPLIVE["ctx"], _CAPLIVE["ctx_at"], _CAPLIVE["ctx_err"], _CAPLIVE["ctx_models"] = win, now, None, len(lens)
    except Exception as e:      # noqa: BLE001 -- the capacity model must never break the scrape
        log.warning("capacity_note_models: %s", e)


async def _poll_engine_models(now=None):
    """GET <engine>/v1/models at most every MODELS_POLL_S (the window changes only when the engine is restarted with another
    --max-model-len, which also resets the poll timer). Bounded; any failure keeps the last-known value and is reported."""
    now = time.time() if now is None else now
    if not (CAPACITY_LIVE and CONTEXT_LIVE) or now - _CAPLIVE["ctx_poll_at"] < max(1.0, MODELS_POLL_S):
        return
    _CAPLIVE["ctx_poll_at"] = now
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TELEM_SCRAPE_TIMEOUT)) as s:
            async with s.get(_MODELS_URL) as r:
                if r.status != 200:
                    raise RuntimeError("http %s" % r.status)
                payload = await r.json(content_type=None)
    except Exception as e:      # noqa: BLE001
        _CAPLIVE["ctx_err"] = str(e)[:200] or type(e).__name__
        return
    capacity_note_models(payload, now)


def context_window_info(configured, now=None):
    """The context window to admit against for one configured knob (SHIM_LOCAL_CONTEXT_LIMIT or SHIM_MAX_LOCAL_TOKENS), with
    where it came from: live (the engine's max_model_len), live-last-known (it answered earlier, unreachable now), or
    configured (never answered, or the kill switch). A configured value <= 0 means 'guard disabled' and is never replaced."""
    now = time.time() if now is None else now
    cfg, live = int(configured), _CAPLIVE["ctx"]
    out = {"configured": cfg, "live": live}
    if CAPACITY_LIVE and CONTEXT_LIVE and live and cfg > 0:
        age = max(0.0, now - _CAPLIVE["ctx_at"])
        fresh = age <= max(90.0, 3 * MODELS_POLL_S)          # the engine itself answered /v1/models recently
        out.update(tokens=live, source="live" if fresh else "live-last-known", age_s=round(age, 1),
                   configured_stale=(cfg != live), engine_reachable=fresh)
    else:
        why = ("SHIM_CAPACITY_LIVE=0" if not CAPACITY_LIVE else "SHIM_CONTEXT_LIVE=0" if not CONTEXT_LIVE
               else "guard disabled (configured <= 0)" if cfg <= 0 else "engine has not answered yet")
        out.update(tokens=cfg, source="configured", detail=why, age_s=None, configured_stale=None)
    return out


def local_context_limit():
    """Effective SHIM_LOCAL_CONTEXT_LIMIT: the engine's live max_model_len, else the configured value."""
    return context_window_info(LOCAL_CONTEXT_LIMIT)["tokens"]


def max_local_tokens():
    """Effective SHIM_MAX_LOCAL_TOKENS (the per-request size cap): the engine's live max_model_len, else the configured value."""
    return context_window_info(MAX_LOCAL_TOKENS)["tokens"]


def token_budget_info(now=None):
    """Effective cap on in-flight RESERVED tokens (prompt + bounded output across local lanes): the explicit SHIM_TOKEN_BUDGET
    override if one is set (0 = disabled), else TOKEN_BUDGET_FRAC x the live pool, never above TOKEN_BUDGET_CEIL."""
    pool = pool_info(now)
    derived = int(pool["tokens"] * TOKEN_BUDGET_FRAC)
    capped = bool(TOKEN_BUDGET_CEIL > 0 and derived > TOKEN_BUDGET_CEIL)
    if capped:
        derived = int(TOKEN_BUDGET_CEIL)
    out = {"derived_tokens": derived, "fraction": round(TOKEN_BUDGET_FRAC, 4), "ceiling": TOKEN_BUDGET_CEIL or None,
           "ceiling_applied": capped, "pool_tokens": pool["tokens"], "pool_source": pool["source"],
           "calibration": "%d reserved tokens proven against a %d-token pool" % TOKEN_BUDGET_CALIBRATION}
    if TOKEN_BUDGET is not None:
        out.update(tokens=int(TOKEN_BUDGET), source="override",
                   detail="explicit SHIM_TOKEN_BUDGET=%d (set it to 'auto' to follow the engine)" % int(TOKEN_BUDGET),
                   override_vs_derived_pct=(round(100.0 * (int(TOKEN_BUDGET) - derived) / derived, 1) if derived and TOKEN_BUDGET else None))
    else:
        out.update(tokens=derived, source=("live-derived" if pool["source"].startswith("live") else "configured-pool-derived"),
                   detail="%.1f%% of the %s KV pool%s" % (100 * TOKEN_BUDGET_FRAC, pool["source"], ", capped at the benchmarked ceiling" if capped else ""))
    return out


def token_budget():
    return token_budget_info()["tokens"]


def _quantile(vals, q):
    vals = sorted(vals)
    if not vals:
        return None
    x = max(0.0, min(1.0, q)) * (len(vals) - 1)
    lo = int(x)
    hi = min(len(vals) - 1, lo + 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (x - lo)


def _offline_pad_spans(now):
    spans = list(_OFFLINE_SPANS)
    if _OFFLINE.get("t0"):
        spans.append((_OFFLINE["t0"], max(now, _OFFLINE.get("until") or now)))
    return [(a, b + PREFILL_MEASURE_OFFLINE_PAD_S) for a, b in spans]


def measured_prefill_pure(now=None, force=False):
    """The pure per-request prefill rate (computed tokens / prefill seconds, queue and admission wait excluded) that the
    CURRENT engine generation delivered: the PREFILL_MEASURE_QUANTILE (p75) of per-minute rates over the last
    PREFILL_MEASURE_WINDOW_S, counting only minutes that carried real prefill. A transient dip (decode contention, a thermal
    throttle blip, a foreign probe) therefore cannot collapse the budgets, and a rate measured on a previous engine, during
    the post-restart settle, or inside a planned offline window never counts. Report dict; tok_s is None when the evidence is
    not there (status says why) and callers fall back to the configured value."""
    explicit_now = now is not None
    cached = _CAPLIVE.get("measure")
    t_now = time.time()
    if not explicit_now and not force and cached and t_now - cached["at"] < _PREFILL_MEASURE_TTL_S:
        return cached["rep"]
    now = t_now if now is None else now
    gen = _CAPLIVE["gen_start"]
    rep = {"tok_s": None, "status": "no-engine-generation", "valid_minutes": 0, "min_minutes": PREFILL_MEASURE_MIN_BUCKETS,
           "quantile": PREFILL_MEASURE_QUANTILE, "generation_start": gen, "generation_source": _CAPLIVE["gen_src"],
           "window_start": None, "samples_in_window": 0, "excluded_offline": 0, "excluded_pre_generation": 0}
    if gen:
        lo = max(now - PREFILL_MEASURE_WINDOW_S, gen + PREFILL_MEASURE_SETTLE_S)
        rep["window_start"] = lo
        pads = _offline_pad_spans(now)
        buckets = {}
        for t, toks, secs in list(_FLOW_PURE):
            if t > now:
                continue
            if t < gen + PREFILL_MEASURE_SETTLE_S and t >= now - PREFILL_MEASURE_WINDOW_S:
                rep["excluded_pre_generation"] += 1
            if t < lo:
                continue
            if any(a <= t <= b for a, b in pads):
                rep["excluded_offline"] += 1
                continue
            b = buckets.setdefault(int(t // PREFILL_MEASURE_BUCKET_S), [0.0, 0.0])
            b[0] += toks
            b[1] += secs
            rep["samples_in_window"] += 1
        rates = [tk / sc for tk, sc in buckets.values() if sc >= _PREFILL_BUCKET_MIN_SECS and tk >= _PREFILL_BUCKET_MIN_TOKENS]
        rep["valid_minutes"] = len(rates)
        if now < gen + PREFILL_MEASURE_SETTLE_S:
            rep["status"] = "settling"
        elif len(rates) < PREFILL_MEASURE_MIN_BUCKETS:
            rep["status"] = "insufficient"
        else:
            rep["status"] = "ok"
            rep["tok_s"] = round(_quantile(rates, PREFILL_MEASURE_QUANTILE), 1)
            rep["range_tok_s"] = [round(min(rates), 1), round(max(rates), 1)]
    if not explicit_now:
        _CAPLIVE["measure"] = {"at": t_now, "rep": rep}
    return rep


def prefill_info(now=None):
    """Effective prefill rate for admission seconds, first-token deadlines, heavy-prefill routing and predicted TTFT."""
    rep = measured_prefill_pure(now)
    cfg = float(PREFILL_TPS)
    if CAPACITY_LIVE and rep["tok_s"]:
        return {"tok_s": rep["tok_s"], "source": "measured", "configured_tok_s": cfg, "measurement": rep,
                "detail": "p%d of per-minute pure per-request prefill rate over %d minutes of this engine generation (queue excluded)" % (
                    round(rep["quantile"] * 100), rep["valid_minutes"])}
    why = "SHIM_CAPACITY_LIVE=0" if not CAPACITY_LIVE else "%s (%d of %d minutes with prefill)" % (
        rep["status"], rep["valid_minutes"], rep["min_minutes"])
    return {"tok_s": cfg, "source": "configured", "configured_tok_s": cfg, "measurement": rep,
            "detail": "configured SHIM_PREFILL_TPS: %s" % why}


def prefill_tps():
    return prefill_info()["tok_s"]


def prefix_align_tokens():
    """Whole attention blocks are what the engine's prefix cache serves: the live block size, else the configured one."""
    if CAPACITY_LIVE and _CAPLIVE["block"]:
        return int(_CAPLIVE["block"])
    return int(PREFIX_ALIGN_TOKENS)


def capacity_model_facts(now=None):
    """Everything the dashboard and /gateway/capacity say about the gateway's capacity model, each number with its source."""
    now = time.time() if now is None else now
    pool, tb, pf = pool_info(now), token_budget_info(now), prefill_info(now)
    align = prefix_align_tokens()
    cl, ml = context_window_info(LOCAL_CONTEXT_LIMIT, now), context_window_info(MAX_LOCAL_TOKENS, now)
    warns = []
    for env, w in (("SHIM_LOCAL_CONTEXT_LIMIT", cl), ("SHIM_MAX_LOCAL_TOKENS", ml)):
        if w.get("configured_stale"):
            warns.append("%s=%d differs from the engine's max_model_len %d; the live value is used, the configured one is only a fallback" % (
                env, w["configured"], w["live"]))
    if CAPACITY_LIVE and CONTEXT_LIVE and _CAPLIVE["ctx"] is None and _CAPLIVE["ctx_err"]:
        warns.append("context window is the configured value: %s" % _CAPLIVE["ctx_err"])
    if pool.get("configured_stale"):
        warns.append("SHIM_POOL_TOKENS=%d is stale (the engine reports %d); the configured value is only a fallback" % (
            pool["configured_tokens"], pool["live_tokens"]))
    if tb["source"] == "override" and tb.get("override_vs_derived_pct") is not None and abs(tb["override_vs_derived_pct"]) > 10:
        warns.append("SHIM_TOKEN_BUDGET=%d pins the budget %+.0f%% away from the live-derived %d; set SHIM_TOKEN_BUDGET=auto to follow the engine" % (
            tb["tokens"], tb["override_vs_derived_pct"], tb["derived_tokens"]))
    if pf["source"] == "configured" and CAPACITY_LIVE:
        m = pf["measurement"]
        if m.get("tok_s") is None and m["status"] != "ok":
            warns.append("prefill rate is the configured SHIM_PREFILL_TPS=%g: %s" % (pf["configured_tok_s"], m["status"]))
    gen = _CAPLIVE["gen_start"]
    return {
        "enabled": bool(CAPACITY_LIVE), "kill_switch": "SHIM_CAPACITY_LIVE=0",
        "principle": "capacity numbers come from the running engine; configured values are fallbacks or explicit overrides",
        "kv_pool_tokens": {"effective": pool["tokens"], "source": pool["source"], "live": pool["live_tokens"],
                           "configured": pool["configured_tokens"], "configured_stale": pool.get("configured_stale"),
                           "read_from": "vllm:cache_config_info kv_cache_size_tokens", "age_s": pool.get("age_s"),
                           "engine_reachable": pool.get("engine_reachable")},
        "token_budget": {"effective": tb["tokens"], "source": tb["source"], "detail": tb["detail"],
                         "derived_from_live": tb["derived_tokens"], "fraction_of_pool": tb["fraction"], "ceiling": tb["ceiling"],
                         "ceiling_applied": tb["ceiling_applied"], "override": (int(TOKEN_BUDGET) if TOKEN_BUDGET is not None else None),
                         "calibration": tb["calibration"], "halo_control_reserve": _halo_control_token_reserve()},
        "prefill_tok_s": {"effective": pf["tok_s"], "source": pf["source"], "detail": pf["detail"],
                          "configured": pf["configured_tok_s"], "measured_p75": pf["measurement"].get("tok_s"),
                          "measured_range": pf["measurement"].get("range_tok_s"),
                          "valid_minutes": pf["measurement"]["valid_minutes"], "min_minutes": pf["measurement"]["min_minutes"],
                          "status": pf["measurement"]["status"], "window_start": pf["measurement"].get("window_start")},
        "context_window": {
            "local_context_limit": {k: v for k, v in {**cl, "effective": cl["tokens"], "env": "SHIM_LOCAL_CONTEXT_LIMIT"}.items() if k != "tokens"},
            "max_local_tokens": {k: v for k, v in {**ml, "effective": ml["tokens"], "env": "SHIM_MAX_LOCAL_TOKENS"}.items() if k != "tokens"},
            "kill_switch": "SHIM_CONTEXT_LIVE=0", "read_from": "engine /v1/models max_model_len (min over served models)",
            "poll_s": MODELS_POLL_S, "last_error": _CAPLIVE["ctx_err"], "models_reporting": _CAPLIVE["ctx_models"]},
        "prefix_align_tokens": {"effective": align, "source": "live" if (CAPACITY_LIVE and _CAPLIVE["block"]) else "configured",
                                "configured": int(PREFIX_ALIGN_TOKENS), "read_from": "vllm:cache_config_info block_size"},
        "engine_generation": {"start": gen, "source": _CAPLIVE["gen_src"],
                              "age_s": round(now - gen, 1) if gen else None,
                              "settled": bool(gen and now >= gen + PREFILL_MEASURE_SETTLE_S),
                              "recent_changes": list(_CAPLIVE["events"])[:8]},
        "warnings": warns,
    }


def flow_decode_tps():
    vals = sorted(r[2] for r in list(_FLOW_METER)[-120:] if r[2] and r[2] > 0 and (r[3] or 0) > 0)
    return vals[len(vals) // 2] if vals else None


def flow_backlog_s():
    return _inflight_computed / max(1.0, flow_prefill_tps())


def flow_meter_update(fam, prev_fam, dt, running, waiting, hit_rate, gen_tok_s):
    """Called by the engine scrape: the uncached-prompt rate (prompt_tokens_by_source local_compute)."""
    try:
        if prev_fam is None or dt <= 0:
            return
        def src(f):
            for lab, val in f.get("vllm:prompt_tokens_by_source_total") or []:
                if lab.get("source") == "local_compute":
                    return val
            return None
        cur, prev = src(fam), src(prev_fam)
        rate = max(0.0, cur - prev) / dt if cur is not None and prev is not None else None
        _FLOW_METER.append((time.time(), rate, gen_tok_s, running, waiting, hit_rate))
        # PURE per-request prefill speed, queue wait excluded: computed tokens / prefill seconds, both from the engine's
        # own per-request histograms (request_prefill_kv_computed_tokens, request_prefill_time_seconds -- queue time is a
        # separate histogram). This is what a request experiences once started; budgets should derive from THIS one.
        def hsum(f, name):
            return _fv(f, name + "_sum")
        ct, pt = hsum(fam, "vllm:request_prefill_kv_computed_tokens"), hsum(fam, "vllm:request_prefill_time_seconds")
        ct0, pt0 = hsum(prev_fam, "vllm:request_prefill_kv_computed_tokens"), hsum(prev_fam, "vllm:request_prefill_time_seconds")
        if None not in (ct, pt, ct0, pt0) and pt >= pt0 and ct >= ct0 and (pt > pt0 or ct > ct0):
            _FLOW_PURE.append((time.time(), ct - ct0, pt - pt0))
    except Exception as _e:
        _swallowed("flow_meter_update", _e)


# ---- the mode, as a fact ----
def flow_mode_now():
    """local-only | local+remote | remote-only | none, with the facts it was derived from. Same flags the
    router reads, so the label can never disagree with behaviour."""
    local_up = bool(_health.get("ok"))
    remote_cfg = bool(REMOTE_ENABLED)
    dead_s = max(0, int(_remote_dead_until - time.time()))
    forced = bool(effective_force_remote())
    budget_ok = None
    if remote_cfg and not LOCAL_ONLY and not dead_s:
        try:
            budget_ok = bool(_spend_allows_overflow(20000, 2000))
        except Exception:
            budget_ok = None
    remote_usable = bool(remote_cfg and not LOCAL_ONLY and not dead_s and budget_ok is not False)
    offline = _local_offline()
    if offline:
        mode = "remote-only" if remote_usable else "none"
    elif forced and remote_usable:
        mode = "remote-only"
    elif local_up and remote_usable:
        mode = "local+remote"
    elif local_up:
        mode = "local-only"
    elif remote_usable:
        mode = "remote-only"
    else:
        mode = "none"
    why = []
    if LOCAL_ONLY:
        why.append("SHIM_LOCAL_ONLY (full-local mode)")
    if not remote_cfg:
        why.append("no remote provider configured")
    if dead_s:
        why.append("remote provider refused (402), breaker open %ds" % dead_s)
    if budget_ok is False:
        why.append("daily remote spend cap has no room")
    if forced:
        why.append("force-remote window")
    if offline:
        why.append("planned local-offline window: %s (by %s, %ds left)" % (
            _OFFLINE["reason"], _OFFLINE["by"], max(0, int(_OFFLINE["until"] - time.time()))))
    if not local_up:
        why.append("local engine unhealthy")
    return {"planned_offline": offline, "mode": mode, "local_up": local_up, "remote_configured": remote_cfg, "remote_usable": remote_usable,
            "remote_dead_for_s": dead_s, "remote_budget_ok": budget_ok, "forced_remote": forced,
            "full_local_flag": bool(LOCAL_ONLY), "why": why}


def flow_note_mode(now=None):
    """Record a mode change (an event for Halo/the estate to catch up from). Cheap; called by the sampler."""
    now = time.time() if now is None else now
    m = flow_mode_now()
    st = _FLOW_MODE_STATE
    if st["mode"] != m["mode"]:
        ev = {"t": round(now, 3), "from": st["mode"], "to": m["mode"], "why": m["why"],
              "local_up": m["local_up"], "remote_usable": m["remote_usable"]}
        st["events"].appendleft(ev)
        if st["mode"] is not None:           # the first observation is a baseline, not a change
            try:
                os.makedirs(os.path.dirname(_FLOW_EVENTS_FILE), exist_ok=True)
                with open(_FLOW_EVENTS_FILE, "a") as fh:
                    fh.write(json.dumps(ev, sort_keys=True) + "\n")
            except Exception as e:
                log.warning("flow mode event write failed: %s", e)
            log.warning("capacity mode %s -> %s (%s)", st["mode"], m["mode"], "; ".join(m["why"]) or "-")
        st["mode"], st["since"], st["basis"] = m["mode"], now, m
    return m


# ---- tickets and the queue ----
def _flow_bump():
    _FLOW["version"] += 1


def flow_make_ticket(request, body, cls, pm, ptok, units, fits):
    now = time.time()
    declared = False
    dl = None
    try:
        s = request.headers.get("X-Gateway-Deadline-S")
        a = request.headers.get("X-Gateway-Deadline-At")
        if s not in (None, ""):
            dl, declared = now + max(0.0, float(s)), True
        elif a not in (None, ""):
            dl, declared = float(a), True
    except (TypeError, ValueError):
        dl, declared = None, False
    if dl is None:
        d = _flow_deadlines()[cls]
        if d > 0:
            dl = now + d
    cost_s = max(0.5, ((pm or {}).get("computed") or ptok) / max(1.0, flow_prefill_tps()))
    return FlowTicket(cls, pm, ptok, units, fits, dl, declared, _friendly_client(request).get("name"),
                      id(request), cost_s)


@_flow_failopen(None)
def flow_note_arrival(cls, ptok, est_computed):
    _FLOW_DEMAND.append((time.time(), cls, int(ptok or 0), int(est_computed or 0)))


@_flow_failopen(None)
def flow_enqueue(t):
    _FLOW["seq"] += 1
    t.seq = _FLOW["seq"]
    if not any(w.cls == t.cls for w in _FLOW["waiters"]):
        # a class that was idle starts level with the classes already waiting: no banked credit, no debt
        others = [_FLOW["vtime"][w.cls] for w in _FLOW["waiters"]]
        if others:
            _FLOW["vtime"][t.cls] = max(_FLOW["vtime"][t.cls], min(others))
    _FLOW["waiters"].append(t)
    _flow_bump()


@_flow_failopen(None)
def flow_dequeue(t):
    try:
        _FLOW["waiters"].remove(t)
    except ValueError:
        return
    _flow_bump()


def _flow_prefix_hold(t, now):
    """Hold t while a request sharing its prefix is still prefilling, iff the recompute it avoids costs more
    than the wait (no threshold: the two are compared). Bounded by FLOW_PREFIX_HOLD_MAX_S."""
    pm = t.pm or {}
    chain, total, est, credit = pm.get("chain") or [], pm.get("total") or 0, pm.get("est") or 0, 0
    if not chain or total <= 0 or est <= 0:
        return None
    credit = max(0, est - (pm.get("computed") or est))
    tps = flow_prefill_tps()
    best = None
    for rid, p in _FLOW["prefilling"].items():
        if rid == t.rid or now - p["t0"] >= FLOW_PREFIX_HOLD_MAX_S:
            continue
        pk = p["cum"]
        deepest = None
        for key, cum in chain:
            if key in pk:
                deepest = cum
        if deepest is None:
            continue
        matched = deepest / total * est - PREFIX_HIT_MARGIN_TOKENS
        saving_s = max(0.0, matched - credit) / tps
        remaining_s = max(0.0, p["est_s"] - (now - p["t0"]))
        if saving_s > remaining_s and (best is None or remaining_s < best[0]):
            best = (remaining_s, saving_s)
    return best


def _flow_eligible(t, now, backlog):
    """Could t be admitted right now, ignoring order? (legacy fit + class ceiling + prefix hold)"""
    if not t.fits():
        return False, "fits"
    ceil = _flow_ceils()[t.cls] * FLOW_BACKLOG_S
    if ceil > 0 and backlog > ceil and _inflight > 0 and backlog > LIGHT_PREFILL_SECS:
        starving = (now - max(_FLOW["last_admit"][t.cls], t.t_enq)) > FLOW_STARVE_S
        if not starving:
            return False, "ceiling"
    hold = _flow_prefix_hold(t, now)
    if hold is not None:
        return False, "prefix-hold"
    return True, ""


def flow_pick(now=None):
    """The ticket that should be admitted next, or None. Cached per queue/engine state change."""
    now = time.time() if now is None else now
    ver, at, head = _FLOW["head"]
    if ver == _FLOW["version"] and now - at < 0.1:
        return head
    backlog = flow_backlog_s()
    by_cls = {}
    blocked = collections.Counter()
    for t in _FLOW["waiters"]:
        ok, why = _flow_eligible(t, now, backlog)
        if ok:
            by_cls.setdefault(t.cls, []).append(t)
        else:
            blocked[(t.cls, why)] += 1
            t.held = why
    pick = None
    if by_cls:
        order = {c: i for i, c in enumerate(FLOW_CLASSES)}
        shares = _flow_shares()
        best_key = None
        for cls, cand in by_cls.items():
            cand.sort(key=lambda x: x.seq)
            choice = cand[0]                                    # FIFO ...
            urgent = [x for x in cand if x.deadline_at is not None and x.deadline_at - now < FLOW_URGENT_SLACK_S]
            if urgent:                                          # ... unless a deadline is close (EDF) ...
                choice = min(urgent, key=lambda x: x.deadline_at)
            elif _FLOW["last_prefix"] and _FLOW["run"] < FLOW_AFFINITY_MAX:
                aff = [x for x in cand if x.prefix == _FLOW["last_prefix"]]
                if aff:                                         # ... or a warm prefix is waiting (affinity)
                    choice = aff[0]
            # weighted fair queuing on FINISH tags: a heavy class wins a near-tie, light classes still
            # get their share (the heavier the weight, the sooner a class's next request finishes)
            key = (_FLOW["vtime"][cls] + choice.cost_s / shares[cls], order[cls])
            if best_key is None or key < best_key:
                best_key, pick = key, choice
    _FLOW["head"] = (_FLOW["version"], now, pick)
    return pick


@_flow_failopen(True)
def flow_turn(t):
    """May t take the engine place now? True whenever flow is off; in shadow mode counts what enforce would hold."""
    if FLOW_MODE == "off":
        return True
    head = flow_pick()
    if FLOW_MODE == "shadow":
        if head is not None and head is not t:
            _FLOW_STATS["shadow_would_hold"] += 1
        return True
    return head is t


@_flow_failopen(None)
def flow_on_admit(t, request=None):
    """t just claimed its place: advance fair-queuing time, remember the prefix, start the prefill watch."""
    now = time.time()
    shares = _flow_shares()
    _FLOW["vtime"][t.cls] += t.cost_s / shares[t.cls]
    t.adjacent = bool(t.prefix and t.prefix == _FLOW["last_prefix"])
    if t.prefix and t.prefix == _FLOW["last_prefix"]:
        _FLOW["run"] += 1
        _FLOW_STATS["affinity_adjacent"] += 1
    else:
        _FLOW["run"] = 0
    _FLOW["last_prefix"] = t.prefix
    _FLOW["last_admit"][t.cls] = now
    # affinity that actually reordered: a same-prefix ticket jumped an older one of its class
    older = [w for w in _FLOW["waiters"] if w is not t and w.cls == t.cls and w.seq < t.seq]
    if older and t.adjacent:
        _FLOW_STATS["affinity_reorders"] += 1
    _FLOW_STATS["granted_" + t.cls] += 1
    _FLOW_WAITS.append((now, t.cls, now - t.t_enq))
    pm = t.pm or {}
    if pm.get("chain"):
        _FLOW["prefilling"][t.rid] = {"cum": {k for k, _ in pm["chain"]}, "t0": now, "prefix": t.prefix,
                                      "est_s": (pm.get("computed") or 0) / max(1.0, flow_prefill_tps())}
    _flow_bump()


@_flow_failopen(None)
def flow_prefill_done(rid):
    if _FLOW["prefilling"].pop(rid, None) is not None:
        _flow_bump()


@_flow_failopen(None)
def flow_note_cache(info):
    """Grade the cache outcome of back-to-back same-prefix requests against all others."""
    try:
        cached, ptok = info.get("cached_actual"), info.get("ptok_exact_local")
        if cached is None or not ptok or info.get("route") != "local":
            return
        row = _FLOW_CACHE["adjacent" if info.get("flow_adjacent") else "other"]
        row[0] += 1
        row[1] += int(cached)
        row[2] += int(ptok)
    except Exception as _e:
        _swallowed("flow_note_cache", _e)


@_flow_failopen(None)
def flow_note_service(cls, duration_s):
    if duration_s and duration_s > 0:
        _FLOW_SERVICE.append((time.time(), cls or "kevin", float(duration_s)))


# ---- expected wait and deadline-aware admission ----
def _flow_service_p50(default=60.0):
    v = sorted(d for _, _, d in list(_FLOW_SERVICE)[-80:])
    return v[len(v) // 2] if v else default


def flow_expected_wait(cls, cost_s=0.0, lane_limit=None, units=1, exclude=None):
    """Seconds until a request of this class starts, given the engine backlog and who is ahead. An ESTIMATE:
    the engine backlog must fall to the class ceiling after every higher-order waiter has been admitted."""
    backlog = flow_backlog_s()
    ceil = _flow_ceils()[cls] * FLOW_BACKLOG_S
    vt = _FLOW["vtime"]
    ahead = [w for w in _FLOW["waiters"] if w is not exclude and (w.cls == cls or vt[w.cls] <= vt[cls])]
    ahead_cost = sum(w.cost_s for w in ahead)
    prefill_term = 0.0 if ceil <= 0 else max(0.0, backlog + ahead_cost - ceil)
    lane_term = 0.0
    lim = lane_limit or effective_budget()
    if _inflight + units > lim:
        lane_term = _flow_service_p50() * (len(ahead) + 1) / max(1, lim)
    return round(max(prefill_term, lane_term), 1)


def _flow_class_service_p50(cls, default=30.0):
    v = sorted(d for _, c, d in list(_FLOW_SERVICE)[-200:] if c == cls)
    if len(v) < 5:
        v = sorted(d for _, _, d in list(_FLOW_SERVICE)[-200:])
    return v[len(v) // 2] if v else default


@_flow_failopen(None)
def flow_admission_check(t, remote_can_take):
    """Deadline-aware admission. Returns None (go on) or a dict describing the refusal. Only refusable classes
    that cannot START before their deadline are refused, and only when remote cannot absorb them."""
    t.expected_wait_s = flow_expected_wait(t.cls, t.cost_s, units=t.units, exclude=t)
    if FLOW_MODE == "off" or t.cls not in FLOW_REFUSABLE or t.deadline_at is None:
        return None
    budget_s = t.deadline_at - time.time()
    # A deadline the CALLER declared (X-Gateway-Deadline-S) is its total patience for the answer, so the typical
    # service time of its class (prefill + decode, measured) must fit as well; a class default is a start deadline.
    need = t.expected_wait_s + (max(t.cost_s, _flow_class_service_p50(t.cls)) if t.declared else t.cost_s)
    if need <= budget_s:
        return None
    if FLOW_MODE == "shadow":
        _FLOW_STATS["shadow_would_refuse_" + t.cls] += 1
        return None
    if remote_can_take:
        # local+remote: this is the abnormal-spike case the remote valve exists for -- cannot start in time
        # locally. Send it now instead of letting it age in the queue.
        return {"overflow": True, "class": t.cls, "expected_wait_s": t.expected_wait_s,
                "own_prefill_s": round(t.cost_s, 1), "deadline_in_s": round(budget_s, 1)}
    return {"class": t.cls, "expected_wait_s": t.expected_wait_s, "own_prefill_s": round(t.cost_s, 1),
            "deadline_in_s": round(budget_s, 1), "retry_after": int(max(5, min(900, t.expected_wait_s)))}


def _flow_refusal_response(r):
    payload = json.dumps({"error": {
        "message": ("local capacity: this %s request cannot start before its deadline (expected wait %ss + "
                    "prefill %ss > %ss left); nothing was prefilled. Retry after %ss." %
                    (r["class"], r["expected_wait_s"], r["own_prefill_s"], r["deadline_in_s"], r["retry_after"])),
        "type": "capacity_deadline", "code": "flow_deadline", **r}}).encode()
    return web.Response(body=payload, status=429, content_type="application/json",
                        headers={"Retry-After": str(r["retry_after"]), "X-Gateway-Refused": "flow-deadline",
                                 "X-Gateway-Expected-Wait": str(r["expected_wait_s"])})


# ---- planned local-offline window (Kevin 10-02: benchmarks and engine upgrades must not be outages) ----
# MEASURED 2026-10-02: ~85% of the day's fenced time was lane benchmark/restart drain fences that refused ALL
# traffic (503), although DeepSeek could have carried the estate. A planned local-offline window does not
# refuse: it routes new work to the remote valve (inside the daily cap), keeps pinned-local callers waiting
# with Retry-After, lets accepted local work finish, and ends by itself (lease) or by DELETE. The admission
# FENCE (/gateway/drain) stays for the instant of a gateway code swap only.
_OFFLINE = {"until": 0.0, "lease": None, "reason": None, "by": None, "t0": None, "ttl_s": None, "refused": 0}
OFFLINE_MAX_TTL_S = 3600


def _local_offline(now=None):
    return (time.time() if now is None else now) < _OFFLINE["until"]


def _flow_event(row):
    if row.get("event") == "offline-close":
        try:
            _OFFLINE_SPANS.append((float(row.get("t0")), float(row.get("t"))))     # capacity model: not a sample window
        except (TypeError, ValueError):
            pass
    try:
        os.makedirs(os.path.dirname(_FLOW_EVENTS_FILE), exist_ok=True)
        with open(_FLOW_EVENTS_FILE, "a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    except Exception as e:
        log.warning("flow event write failed: %s", e)


def _offline_reap(now=None):
    """Write the close row of a window whose lease ran out (a crashed benchmark must not leave local off forever)."""
    now = time.time() if now is None else now
    if _OFFLINE["t0"] and not _local_offline(now):
        _flow_event({"event": "offline-close", "how": "expired", "t": round(min(now, _OFFLINE["until"]), 3),
                     "t0": round(_OFFLINE["t0"], 3), "reason": _OFFLINE["reason"], "by": _OFFLINE["by"],
                     "refused": _OFFLINE["refused"]})
        _OFFLINE.update(t0=None, lease=None, reason=None, by=None, ttl_s=None, refused=0)
        flow_note_mode(now)


def _offline_status():
    _offline_reap()
    on = _local_offline()
    return {"offline": on, "until": _OFFLINE["until"] if on else None, "reason": _OFFLINE["reason"] if on else None,
            "by": _OFFLINE["by"] if on else None, "since": _OFFLINE["t0"] if on else None,
            "remaining_s": max(0, int(_OFFLINE["until"] - time.time())) if on else 0,
            "local_active": sum(1 for a in _ACTIVE.values() if a.get("phase") == "local"),
            "local_inflight_units": _inflight, "active": len(_ACTIVE), "mode": flow_mode_now()["mode"]}


async def gateway_offline(request):
    """GET status; POST {ttl_s, reason, by} opens (or, with {lease}, extends) a window; DELETE {lease} closes it."""
    if request.method == "GET":
        return web.json_response(_offline_status())
    if not _admin_ok(request):
        return web.json_response({"error": "admin token required"}, status=401)
    try:
        body = await request.json()
    except (ValueError, TypeError):
        body = None
    if not isinstance(body, dict):
        return web.json_response({"error": "JSON body required"}, status=400)
    now = time.time()
    _offline_reap(now)
    if request.method == "DELETE":
        if not _OFFLINE["lease"] or body.get("lease") != _OFFLINE["lease"]:
            return web.json_response({"error": "offline lease mismatch"}, status=409)
        _flow_event({"event": "offline-close", "how": "delete", "t": round(now, 3), "t0": round(_OFFLINE["t0"] or now, 3),
                     "reason": _OFFLINE["reason"], "by": _OFFLINE["by"], "refused": _OFFLINE["refused"]})
        _OFFLINE.update(until=0.0, t0=None, lease=None, reason=None, by=None, ttl_s=None, refused=0)
        flow_note_mode(now)
        return web.json_response(_offline_status())
    try:
        ttl = int(body.get("ttl_s") or 900)
        if not 30 <= ttl <= OFFLINE_MAX_TTL_S:
            raise ValueError
    except (ValueError, TypeError):
        return web.json_response({"error": "ttl_s must be 30..%d seconds" % OFFLINE_MAX_TTL_S}, status=400)
    if _local_offline(now):
        if body.get("lease") and body.get("lease") == _OFFLINE["lease"]:           # extend
            _OFFLINE["until"] = now + ttl
            return web.json_response({**_offline_status(), "lease": _OFFLINE["lease"]})
        return web.json_response({"error": "another offline window is open", **_offline_status()}, status=409)
    by = str(body.get("by") or request.headers.get("X-Client") or request.headers.get("User-Agent") or "?")[:80]
    reason = str(body.get("reason") or "planned local work")[:120]
    _OFFLINE.update(until=now + ttl, lease=os.urandom(16).hex(), reason=reason, by=by, t0=now, ttl_s=ttl, refused=0)
    _flow_event({"event": "offline-open", "t": round(now, 3), "reason": reason, "by": by, "ttl_s": ttl,
                 "local_active": _offline_status()["local_active"]})
    flow_note_mode(now)
    return web.json_response({**_offline_status(), "lease": _OFFLINE["lease"]})


_FLOW_ROUTES = collections.deque(maxlen=30000)       # (t, decision, reason, class, local_had_headroom)
_FLOW_EXPLICIT = frozenset({"alias", "intent", "forced", "local-down", "local-offline", "failover", "full-local-remote-alias", "vision"})


@_flow_failopen(None)
def flow_note_route(decision, reason, request):
    """Remote-as-a-valve accounting (Kevin 10-02: remote is for abnormal spikes, never the normal path)."""
    try:
        info = _ACTIVE.get(id(request)) or {}
        headroom = bool(_health.get("ok") and not _local_offline() and _inflight < effective_budget()
                        and flow_backlog_s() <= LIGHT_PREFILL_SECS)
        _FLOW_ROUTES.append((time.time(), decision, reason, info.get("flow_class"), headroom))
    except Exception as _e:
        _swallowed("flow_note_route", _e)


# Lane DB2: the routing ring is in memory, so every restart emptied the "24 h" window (live: 6,439 of the 17,568 requests
# the on-disk log held). At startup the last 24 h are rebuilt from the request log. A restored row cannot know whether
# local had headroom at the time, so its headroom flag is None (never counted as a defect) and it is reported as `restored`.
FLOW_RESTORE = _env_flag(os.environ.get("SHIM_FLOW_RESTORE", "1"), True)
_FLOW_RESTORED = {"from": None, "n": 0}


def flow_routes_restore_blocking(now=None, before=None, max_rows=30000, per_file_cap=64 * 1024 * 1024):
    """Blocking (run in an executor). Rows (t, decision, reason, class, None) from the last 24 h of the request log that
    finished before `before` (the process start), oldest first, newest `max_rows` kept."""
    now = time.time() if now is None else now
    before = now if before is None else before
    since = now - 86400
    rows = []
    for day_i in range(2):
        fn = _history_day_file(now - day_i * 86400)
        for raw in _read_lines_reverse(fn, per_file_cap):
            try:
                rec = json.loads(raw)
            except Exception:
                continue
            t = rec.get("t")
            if not isinstance(t, (int, float)) or t >= before:
                continue
            if t < since:
                break
            rows.append((t, rec.get("route") or "?", rec.get("reason") or "?", rec.get("flow_class"), None))
            if len(rows) >= max_rows:
                break
        if len(rows) >= max_rows:
            break
    rows.sort(key=lambda r: r[0])
    return rows


def flow_routes_restore(rows):
    """Prepend restored rows (older than anything live) to the ring."""
    if not rows:
        return 0
    live = list(_FLOW_ROUTES)
    _FLOW_ROUTES.clear()
    _FLOW_ROUTES.extend(rows)
    _FLOW_ROUTES.extend(live)
    _FLOW_RESTORED.update(**{"from": rows[0][0], "n": len(rows)})
    return len(rows)


def flow_remote_use(now=None):
    now = time.time() if now is None else now
    out = {}
    # how far back the ring really reaches: the oldest restored row, else the process start
    reach_from = min(_PROCESS_STARTED, _FLOW_RESTORED["from"]) if _FLOW_RESTORED["from"] else _PROCESS_STARTED
    for label, secs in (("15m", 900), ("1h", 3600), ("24h", 86400)):
        rows = [r for r in _FLOW_ROUTES if now - r[0] <= secs]
        remote = [r for r in rows if r[1] == "remote"]
        avoidable = [r for r in remote if r[2] not in _FLOW_EXPLICIT]
        # the ring lives in memory: a window longer than the process has been up is only partly covered
        out[label] = {"covered_s": round(min(secs, now - reach_from), 0), "restored": sum(1 for r in rows if r[4] is None),
                      "requests": len(rows), "remote": len(remote),
                      "remote_share": round(len(remote) / len(rows), 3) if rows else None,
                      "by_reason": dict(collections.Counter(r[2] for r in remote).most_common()),
                      "by_class": dict(collections.Counter(r[3] or "?" for r in remote)),
                      "gateway_chosen_remote": len(avoidable),
                      "remote_while_local_had_headroom": sum(1 for r in avoidable if r[4])}
    spend = None
    try:
        snap = _spend().snapshot()
        spend = {k: snap.get(k) for k in ("spent", "held", "reserved", "cap", "remaining") if k in snap}
    except Exception as _e:
        _swallowed("flow_remote_use", _e)
    return {"principle": "local-first: remote is a valve for abnormal spikes and planned local-offline windows, not the "
                         "normal path. gateway_chosen_remote counts routes the gateway chose (not forced/aliased/"
                         "local-down/offline); remote_while_local_had_headroom is the defect signal: remote used although "
                         "the engine had a free place and no prefill queue.",
            "windows": out, "spend_today": spend}


# ---- the facts endpoint ----
def _pct(vals, q):
    vals = sorted(vals)
    return round(vals[min(len(vals) - 1, int(q * len(vals)))], 1) if vals else None


def flow_capacity_facts(now=None, cls=None, ptok=None):
    now = time.time() if now is None else now
    m = flow_note_mode(now)
    tps, dtps = flow_prefill_tps(), flow_decode_tps()
    meter = [r for r in list(_FLOW_METER) if now - r[0] <= 300]
    eng = _ENGINE_METRICS.get("ok")
    last = meter[-1] if meter else None
    shares, ceils, dls = _flow_shares(), _flow_ceils(), _flow_deadlines()
    demand, queue = {}, {}
    wsec = max(10.0, FLOW_DEMAND_WINDOW_S)
    win5 = [d for d in _FLOW_DEMAND if now - d[0] <= wsec]
    win60 = [d for d in _FLOW_DEMAND if now - d[0] <= 3600]
    tot_s5 = 0.0
    for c in FLOW_CLASSES:
        r5 = [d for d in win5 if d[1] == c]
        r60 = [d for d in win60 if d[1] == c]
        s5 = sum(d[3] for d in r5) / max(1.0, tps)
        tot_s5 += s5
        demand[c] = {"req_per_min_5m": round(len(r5) / (wsec / 60.0), 2), "req_per_min_60m": round(len(r60) / 60.0, 2),
                     "prompt_tok_per_min_5m": int(sum(d[2] for d in r5) / (wsec / 60.0)),
                     "uncached_prefill_s_per_min_5m": round(s5 / (wsec / 60.0), 1),
                     "uncached_prefill_s_per_min_60m": round(sum(d[3] for d in r60) / max(1.0, tps) / 60.0, 1)}
        ws = [w for (t_, c_, w) in _FLOW_WAITS if c_ == c and now - t_ <= 900]
        waiting = [t for t in _FLOW["waiters"] if t.cls == c]
        queue[c] = {"waiting": len(waiting),
                    "oldest_wait_s": round(max((now - t.t_enq for t in waiting), default=0.0), 1),
                    "expected_wait_s": flow_expected_wait(c),
                    "wait_p50_s_15m": _pct(ws, .5), "wait_p95_s_15m": _pct(ws, .95), "admitted_15m": len(ws),
                    "held_by": dict(collections.Counter(t.held for t in waiting if t.held)),
                    "refused_total": _FLOW_STATS.get("refused_" + c, 0),
                    "share": shares[c], "ceiling_backlog_s": round(ceils[c] * FLOW_BACKLOG_S, 1) or None,
                    "default_deadline_s": dls[c] or None}
    out = {
        "as_of": datetime_iso(now), "flow_mode": FLOW_MODE,
        **m,
        "mode_since": _FLOW_MODE_STATE["since"], "mode_changes": list(_FLOW_MODE_STATE["events"])[:20],
        "throughput": {
            "prefill_pure_tok_s": None if flow_prefill_pure_tps(now) is None else round(flow_prefill_pure_tps(now), 1),
            "prefill_pure_basis": ("computed tokens / prefill seconds per request over the last 5 min (queue and admission wait "
                                   "EXCLUDED; derive per-request budgets from this)" if flow_prefill_pure_tps(now) is not None
                                   else "too little prefill in the last 5 min to say; admission uses prefill_effective_tok_s (see capacity_model.prefill_tok_s)"),
            "prefill_effective_tok_s": round(prefill_tps(), 1),
            "prefill_effective_basis": prefill_info()["detail"],
            "prefill_uncached_tok_s": round(tps, 1),
            "prefill_basis_note": "prefill_uncached_tok_s is the engine's AGGREGATE uncached-prompt throughput while it had a queue (it falls when "
                                  "decode or a benchmark shares the engine); it sizes the engine backlog, not per-request budgets",
            "prefill_basis": "measured" if len([r for r in list(_FLOW_METER)[-120:] if r[1] and (r[4] or 0) > 0]) >= 5
                             else "configured fallback (too few busy samples yet; the effective rate is prefill_effective_tok_s)",
            "decode_tok_s_aggregate": None if dtps is None else round(dtps, 1),
            "decode_tok_s_per_stream": None if (dtps is None or not last or not (last[3] or 0)) else round(dtps / max(1.0, last[3]), 1),
            "engine_running": last[3] if last else None, "engine_waiting": last[4] if last else None,
            "prefix_cache_hit_rate_now": last[5] if last else None, "engine_metrics_ok": bool(eng)},
        "pressure": {
            "engine_prefill_backlog_s": round(flow_backlog_s(), 1), "backlog_target_s": FLOW_BACKLOG_S,
            "demand_over_capacity_5m": round(tot_s5 / wsec, 2),
            "meaning": ">1 means uncached prefill is arriving faster than the engine can compute it; queues grow without bound "
                       "unless demand is shed, delayed or sent remote",
            "inflight": _inflight, "lane_budget": effective_budget()},
        "capacity_model": capacity_model_facts(now),
        "demand": demand, "queue": queue,
        "affinity": {"adjacent_grants": _FLOW_STATS.get("affinity_adjacent", 0),
                     "reorders": _FLOW_STATS.get("affinity_reorders", 0),
                     "prefix_holds_now": sum(1 for t in _FLOW["waiters"] if t.held == "prefix-hold"),
                     "hit_rate_adjacent": (round(_FLOW_CACHE["adjacent"][1] / _FLOW_CACHE["adjacent"][2], 3)
                                           if _FLOW_CACHE["adjacent"][2] else None),
                     "hit_rate_other": (round(_FLOW_CACHE["other"][1] / _FLOW_CACHE["other"][2], 3)
                                        if _FLOW_CACHE["other"][2] else None),
                     "graded": [_FLOW_CACHE["adjacent"][0], _FLOW_CACHE["other"][0]]},
        "config": {"shares": shares, "ceilings": {c: ceils[c] for c in FLOW_CLASSES}, "default_deadlines_s": dls,
                   "backlog_target_s": FLOW_BACKLOG_S, "class_map": FLOW_CLASS_MAP,
                   "set_with": "POST /gateway/config {flow_shares, flow_ceil, flow_deadlines, flow_backlog_s, flow_class_map, flow_mode}"},
        "remote_use": flow_remote_use(now),
        "offline_window": _offline_status(),
        "counters": dict(_FLOW_STATS),
        "stall_brake_not_applied": dict(_STALL_BRAKE_NOT_APPLIED),
        "cannot_measure": ([] if eng else ["engine /metrics scrape failing: throughput and engine queue unknown"]),
    }
    if cls in FLOW_CLASSES:
        c0 = max(0, int(ptok or 0))
        out["estimate"] = {"class": cls, "ptok": c0, "expected_wait_s": flow_expected_wait(cls, c0 / max(1.0, tps))}
    return out


def datetime_iso(now):
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now))


async def gateway_capacity(request):
    cls = (request.query.get("class") or "").strip().lower() or None
    try:
        ptok = int(request.query.get("ptok") or 0)
    except ValueError:
        ptok = 0
    return web.json_response(flow_capacity_facts(cls=cls, ptok=ptok))


# form-field key <-> env-var (what the dashboard sends)
_FIELD_ENV = {k.lower().replace("shim_", ""): k for k in _CFG}

def current_config(masked=True):
    """Every runtime-tunable knob, keyed by dashboard field name (env minus SHIM_)."""
    g = globals()
    out = {}
    for env, (gname, _) in _CFG.items():
        field = env.lower().replace("shim_", "")
        v = g[gname]
        if v is None:
            v = "auto"      # SHIM_TOKEN_BUDGET unset = follow the live engine
        if isinstance(v, (set, frozenset, list, tuple)):
            # Same separator table the env-file writer uses. This value is rendered straight
            # into the dashboard form, which POSTs it back into apply_config -- so if the two
            # ever disagree again, merely opening the page and pressing Save re-corrupts the
            # field. That is precisely how no_think_ips died.
            v = _fmt_seq(v, _CFG_SEP.get(env, _CFG_SEP_DEFAULT))
        elif isinstance(v, bool):
            v = 1 if v else 0
        out[field] = v
    out["force_remote"] = int(effective_force_remote())
    out["force_remote_lease_remaining_s"] = max(0, int(FORCE_REMOTE_UNTIL_EPOCH - time.time())) if effective_force_remote() else 0
    k = g["REMOTE_KEY"]
    out["remote_key_display"] = ("set (" + k[:5] + "\u2026" + k[-4:] + ")") if (masked and k and len(k) > 12) else ("set" if k else "")
    out["remote_key_set"] = bool(k)
    out["admin_token_set"] = bool(SHIM_ADMIN_TOKEN)  # dashboard auth-state badge (never the value itself)
    out["mode"] = routing_mode()  # the three-way badge reads this, never its own guess
    out["aliases_count"] = len(_ALIASES)
    out["aliases_url"] = "/gateway/aliases/page"
    if masked:
        out.pop("remote_key", None)
    return out


def apply_config(fields):
    """fields = dashboard form dict (subset). Reassigns globals live + persists to SHIM_ENV_FILE."""
    g = globals()
    changed = []
    for fk, val in fields.items():
        env = _FIELD_ENV.get(fk)
        if not env or val in (None, ""):
            continue
        gname, cast = _CFG[env]
        try:
            g[gname] = cast(val)
            changed.append(fk)
        except Exception as e:
            log.warning("config: bad value for %s: %r (%s)", fk, val, e)
    if "force_remote" in changed:
        g["FORCE_REMOTE_UNTIL_EPOCH"] = time.time() + FORCE_REMOTE_LEASE_S if FORCE_REMOTE else 0.0
    g["REMOTE_ENABLED"] = bool(g["REMOTE_BASE"] and g["REMOTE_KEY"])
    if changed and _config_owner():
        _persist_config()
    return changed


# ---------------- model aliases and provider registry ----------------
# Aliases are resolved here, at the gateway boundary.  Callers never receive credentials and
# never choose a provider by URL; they choose a stable model name and this process resolves it.
_BUILTIN_ALIASES = {"estate", "estate-local", "estate-remote", "estate-remote-pro"}
# R2 v6: the default provider's top model under the SAME spend authority. It is refused (fail
# closed) until its per-type prices are configured in SHIM_REMOTE_PRICES_JSON.
REMOTE_PRO_MODEL = os.environ.get("SHIM_REMOTE_PRO_MODEL", "deepseek-v4-pro")
_ALIASES = {}
_ALIAS_NAME_RE = re.compile(r"^[a-z][a-z0-9._-]{1,63}$", re.I)


def _alias_name(value):
    return str(value or "").strip().lower()


def _validate_alias_record(name, record):
    name = _alias_name(name)
    if not _ALIAS_NAME_RE.fullmatch(name) or name in _BUILTIN_ALIASES:
        raise ValueError("alias must be 2-64 characters and cannot be a built-in alias")
    if not isinstance(record, dict):
        raise ValueError("alias record must be an object")
    base = str(record.get("base") or "").strip().rstrip("/")
    if not re.match(r"^https?://[^\s]+$", base, re.I):
        raise ValueError("base must be an http(s) OpenAI-compatible endpoint")
    model = str(record.get("model") or "").strip()
    if not model or len(model) > 256:
        raise ValueError("model is required")
    key = str(record.get("key") or "")
    try:
        context_limit = int(record.get("context_limit", REMOTE_CONTEXT_LIMIT))
        max_output = int(record.get("max_output", 65536))
    except (TypeError, ValueError):
        raise ValueError("context_limit and max_output must be integers")
    if context_limit < 1024 or max_output < 1 or max_output >= context_limit:
        raise ValueError("context_limit must exceed max_output and both must be positive")
    return {"name": name, "base": base, "key": key, "model": model,
            "context_limit": context_limit, "max_output": max_output,
            "enabled": bool(record.get("enabled", True)),
            "cost": _validate_cost_policy(record.get("cost"))}


def _validate_cost_policy(cost):
    """R2 v6: every custom endpoint declares what it costs. None = undeclared -> the alias
    fails CLOSED for paid routing. {"policy": "free"}, or {"policy": "metered", "provider":
    "<id>", "cache_hit": $/Mtok, "cache_miss": $/Mtok, "output": $/Mtok} (held, settled and
    capped by the same authority as the default provider)."""
    if cost in (None, ""):
        return None
    if not isinstance(cost, dict):
        raise ValueError("cost must be an object")
    policy = str(cost.get("policy") or "").strip().lower()
    if policy == "free":
        return {"policy": "free"}
    if policy != "metered":
        raise ValueError("cost.policy must be 'free' or 'metered'")
    provider = str(cost.get("provider") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", provider):
        raise ValueError("metered cost needs a provider identity")
    out = {"policy": "metered", "provider": provider}
    for k in ("cache_hit", "cache_miss", "output"):
        try:
            v = float(cost[k])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"metered cost needs a numeric {k} price ($/Mtok)")
        if v < 0:
            raise ValueError(f"{k} price must be >= 0")
        out[k] = v
    return out


def _load_aliases():
    global _ALIASES
    try:
        with open(ALIASES_FILE) as fh:
            raw = json.load(fh)
        rows = raw.get("aliases", raw) if isinstance(raw, dict) else raw
        loaded = {}
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    name = row.get("name")
                    if name:
                        try:
                            rec = _validate_alias_record(name, row)
                            loaded[rec["name"]] = rec
                        except ValueError as exc:
                            log.warning("ignoring invalid gateway alias %r: %s", name, exc)
        _ALIASES = loaded
    except FileNotFoundError:
        _ALIASES = {}
    except Exception as exc:
        log.warning("gateway alias registry load failed: %s", exc)
        _ALIASES = {}


def _save_aliases():
    os.makedirs(os.path.dirname(ALIASES_FILE) or ".", mode=0o700, exist_ok=True)
    tmp = ALIASES_FILE + ".tmp"
    payload = {"version": 1, "aliases": list(_ALIASES.values())}
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, ALIASES_FILE)


def _public_alias(record, masked=True):
    out = dict(record)
    if masked:
        key = out.get("key") or ""
        out["key_display"] = ("set (" + key[:5] + "…" + key[-4:] + ")") if len(key) > 12 else ("set" if key else "")
        out.pop("key", None)
    return out


def _resolve_alias(model):
    name = _alias_name(model)
    if name == "estate":
        return {"name": name, "kind": "default"}
    if name == "estate-remote":
        return {"name": name, "kind": "builtin-remote"}
    if name == "estate-remote-pro":
        return {"name": name, "kind": "builtin-remote", "model": REMOTE_PRO_MODEL}
    for alias_name, model in (("estate-remote-pro", REMOTE_PRO_MODEL), ("estate-remote", None)):
        prefix = alias_name + "."
        if name.startswith(prefix) and _SPEND_RID.match(name[len(prefix):]):
            # R2: a reservation-backed request (POST /gateway/spend/reserve returns this name).
            out = {"name": alias_name, "kind": "builtin-remote", "reservation": name[len(prefix):]}
            if model:
                out["model"] = model
            return out
    if name == "estate-local":
        return {"name": name, "kind": "builtin-local"}
    rec = _ALIASES.get(name)
    if rec:
        if rec.get("enabled", True):
            return {"name": name, "kind": "custom-remote", "endpoint": rec}
        return {"name": name, "kind": "disabled"}
    return {"name": name, "kind": "native"}


def _alias_for_request(body):
    try:
        model = json.loads(body).get("model", "")
    except Exception:
        model = ""
    return _resolve_alias(model)


_load_aliases()

def _config_owner():
    """May THIS process rewrite SHIM_ENV_FILE?

    Learned the hard way 2026-09-10: a second shim started for a smoke test on another port
    inherited none of the production env, so its globals were library DEFAULTS -- and its first
    dashboard POST called _persist_config(), which rewrites the WHOLE file from those globals.
    It silently replaced the tuned production values (lane budget 14 -> 2) and emptied the
    remote base+key; the next restart of the real service loaded them.

    Ownership test: a process that actually loaded this env file agrees with most of what is in
    it. A foreign process does not. Refusing to write is always safe (the operator's live change
    still applies in memory, it just does not outlive a restart), while writing when we are not
    the owner destroys a hand-tuned production config."""
    path = SHIM_ENV_FILE
    try:
        with open(path) as fh:
            rows = [l.strip().split("=", 1) for l in fh if l.strip() and not l.startswith("#") and "=" in l]
    except OSError:
        return True                      # no file yet -> we are creating it
    tunable = [(k, v) for k, v in rows if k in _CFG]
    if not tunable:
        return True
    # Compare against what systemd LOADED (the schema canonicalises booleans in os.environ at startup).
    _raw = _cfgschema.raw_value if _cfgschema else (lambda k: os.environ.get(k))
    _same = _cfgschema.same_value if _cfgschema else (lambda k, a, b: a == b)
    agree = sum(1 for k, v in tunable if _raw(k) is not None and _same(k, str(_raw(k)), v))
    ok = agree >= max(1, int(0.6 * len(tunable)))
    if not ok:
        log.error("_persist_config REFUSED: this process matches only %d/%d tunables in %s, so it did "
                  "not load that file and must not rewrite it (a foreign instance once clobbered the "
                  "production gateway config this way). Live change applied in memory only.",
                  agree, len(tunable), path)
    return ok


def _persist_config():
    """Rewrite SHIM_ENV_FILE with current tunable values (atomic, mode 600)."""
    g = globals()
    # Each sequence field is written with ITS OWN separator (_CFG_SEP), not a single global
    # one. The old unconditional "|".join is what killed NO_THINK_IPS and BG_XCLIENTS: both
    # are read back with a comma split, so every save handed the reader one unsplittable
    # token and the policy silently stopped matching anything.
    vals = {env: (_fmt_seq(g[gname], _CFG_SEP.get(env, _CFG_SEP_DEFAULT))
                  if isinstance(g[gname], (list, tuple, set, frozenset)) else ("auto" if g[gname] is None else
                  # lane CFG: booleans as 1/0 -- str(False) == "False" was read back as ON by the import-time readers
                  ("1" if g[gname] else "0") if isinstance(g[gname], bool) else str(g[gname])))
            for env, (gname, _) in _CFG.items()}
    vals["SHIM_FORCE_REMOTE_UNTIL_EPOCH"] = str(FORCE_REMOTE_UNTIL_EPOCH)
    try:
        lines, seen = [], set()
        if os.path.exists(SHIM_ENV_FILE):
            for ln in open(SHIM_ENV_FILE):
                key = ln.split("=", 1)[0].strip()
                if key in vals:
                    lines.append(f"{key}={vals[key]}\n"); seen.add(key)
                else:
                    lines.append(ln)
        for k, v in vals.items():
            if k not in seen:
                lines.append(f"{k}={v}\n")
        tmp = SHIM_ENV_FILE + ".tmp"
        with open(tmp, "w") as f:
            f.write("".join(lines))
        os.chmod(tmp, 0o600)
        os.replace(tmp, SHIM_ENV_FILE)
    except Exception as e:
        log.warning("config persist failed: %s", e)


def effective_budget():
    return 1 if time.time() < _backoff_until else BUDGET


def trigger_backoff(reason):
    global _backoff_until
    _backoff_until = time.time() + OOM_BACKOFF
    log.warning("LOCAL backoff -> budget=1 for %ss (%s)", OOM_BACKOFF, reason)
    _pm_reset("local fault: " + str(reason)[:60])
    # CRASH-ADAPTIVE big_prompt guard: trigger_backoff() only fires on a real detected local
    # crash (OOM/EngineDead/503 -- see _is_oom()), never on a routine busy/wedged failover, so
    # it's the correct single hook for "a local crash just happened".
    if CRASH_ADAPTIVE:
        _crash_adaptive_on_crash(reason)


# ---------------- CRASH-ADAPTIVE big_prompt guard (SHIM_CRASH_ADAPTIVE=1, default off) --------
_big_prompt_restore = None   # prior BIG_PROMPT value while the crash-adaptive floor is active
                              # (None = no drop in effect)
_big_prompt_clean_n  = 0     # consecutive clean big-prompt local completions since the drop


def _crash_adaptive_on_crash(reason):
    """On a detected local crash: if BIG_PROMPT is currently above the safe floor, save it as
    _big_prompt_restore and drop BIG_PROMPT to CRASH_ADAPTIVE_FLOOR. If a drop is already in
    effect and another crash lands before the restore streak completes, just reset the streak
    (the ceiling stays floored -- local hasn't earned it back yet)."""
    global _big_prompt_restore, _big_prompt_clean_n, BIG_PROMPT
    if BIG_PROMPT > CRASH_ADAPTIVE_FLOOR:
        _big_prompt_restore = BIG_PROMPT
        BIG_PROMPT = CRASH_ADAPTIVE_FLOOR
        _big_prompt_clean_n = 0
        log.warning("CRASH-ADAPTIVE: local crash (%s) -> big_prompt %d -> %d; will restore after "
                    "%d consecutive clean big-prompt local completions",
                    reason, _big_prompt_restore, CRASH_ADAPTIVE_FLOOR, BIG_PROMPT_RESTORE_N)
    elif _big_prompt_restore is not None:
        _big_prompt_clean_n = 0
        log.warning("CRASH-ADAPTIVE: another local crash (%s) while big_prompt already floored "
                    "at %d -- clean streak reset", reason, BIG_PROMPT)


def _crash_adaptive_note_local_completion(ptok):
    """Call after a CLEAN (non-failover) local completion. Counts it toward the restore streak
    if a drop is in effect and the prompt was big enough (relative to the floor) to be
    meaningful evidence; once BIG_PROMPT_RESTORE_N land in a row, restores the prior ceiling."""
    global _big_prompt_restore, _big_prompt_clean_n, BIG_PROMPT
    if _big_prompt_restore is None:
        return  # no drop in effect -- nothing to track
    if ptok < CRASH_ADAPTIVE_FLOOR * BIG_PROMPT_QUALIFY_FRAC:
        return  # too small to count as evidence at this scale
    _big_prompt_clean_n += 1
    log.info("CRASH-ADAPTIVE: clean big-prompt local completion %d/%d (ptok=%d)",
             _big_prompt_clean_n, BIG_PROMPT_RESTORE_N, ptok)
    if _big_prompt_clean_n >= BIG_PROMPT_RESTORE_N:
        restored = _big_prompt_restore
        BIG_PROMPT = restored
        _big_prompt_restore = None
        _big_prompt_clean_n = 0
        log.warning("CRASH-ADAPTIVE: big_prompt restored -> %d after %d consecutive clean "
                    "big-prompt local completions", restored, BIG_PROMPT_RESTORE_N)


def _est_tokens(body):
    try:
        j = json.loads(body)
    except Exception:
        return 0
    if not isinstance(j, dict):
        return 0                 # lane SH: a JSON array/string body used to raise here -> bare 500, no telemetry
    chars = 0
    msgs = j.get("messages")
    for m in (msgs if isinstance(msgs, list) else []):
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict):
                    chars += len(b.get("text", "") or "")
    # tool definitions are part of the actual prompt the engine processes — omitting them
    # undercounts by ~20K tokens for a 34-tool Hermes request, producing a first-token
    # timeout that expires before prefill completes
    tools = j.get("tools")
    if tools:
        chars += len(json.dumps(tools))
    if not EXACT_TOKENS:
        return int(chars / CHARS_PER_TOK)
    # A char count can only ever correspond to FEWER tokens than chars/MIN_CHARS_PER_TOK.
    # If even that pessimistic bound is under every decision threshold, the exact number
    # cannot change any routing decision, so skip the round-trip. This is an optimisation
    # with a proof, not a heuristic about the answer.
    if chars / MIN_CHARS_PER_TOK < _min_decision_threshold():
        return int(chars / CHARS_PER_TOK)
    exact = _tokenize_exact(_prompt_text(j))
    # add tool-definition estimate on top of exact message-content count — /tokenize
    # only counts message text, not tool schemas
    if exact is not None and tools:
        exact += int(len(json.dumps(tools)) / CHARS_PER_TOK)
    return exact if exact is not None else int(chars / CHARS_PER_TOK)


def _min_decision_threshold():
    vals = [v for v in (TINY_TOKENS, BIG_TOKENS, BIG_PROMPT, max_local_tokens()) if v and v > 0]
    return min(vals) if vals else 1500


def _prompt_text(j):
    out = []
    for m in (j.get("messages") or []):
        c = m.get("content")
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("text"):
                    out.append(b["text"])
    return "\n".join(out)


_tok_fail_until = 0.0
_TOK_MEMO = collections.OrderedDict()          # [GW2] sha1(prompt text) -> engine token count (bounded LRU)
_TOK_STATS = collections.Counter()
_TOK_LOCK = threading.Lock()                   # the warm step runs on a worker thread


async def _warm_token_estimate(body):
    """[GW2] Run this request's one /tokenize call on a worker thread BEFORE routing, so the ~7 synchronous
    _est_tokens() calls on the routing path all hit the memo instead of blocking the event loop."""
    if not EXACT_TOKENS:
        return
    try:
        await asyncio.get_running_loop().run_in_executor(None, _est_tokens, body)
    except Exception:
        pass

def _tokenize_exact(text):
    """Ask the engine for the REAL token count instead of guessing chars/3.5.

    The estimator was measured +9% to +19% off on real code and 1.77x off on repetitive
    text, and every routing decision (tiny lane, big-prompt guard, local size cap) keys off
    it. /tokenize costs 10ms on a small prompt and 400ms on a 1.2M-char one -- 0.2-0.8% of
    those requests' own latency, i.e. cheapest exactly where precision matters most.

    Falls back to the estimate (and stops trying for a minute) if the endpoint misbehaves,
    so routing never depends on it being up."""
    global _tok_fail_until
    # [GW2 2026-10-03] Memoised: _est_tokens runs ~7 times per request (registration, size cap, tiny lane, units,
    # flow class, route...) and each call was a SYNCHRONOUS /tokenize round trip on the gateway's event loop (10 ms
    # small, 140-270 ms for a 34K-token prompt) -- measured 4,456 /tokenize calls for 642 requests in 15 min. Every
    # one of those blocked every other stream the gateway was relaying. Same text -> same count, so one call each.
    key = hashlib.sha1(text.encode("utf-8", "surrogatepass")).digest()
    with _TOK_LOCK:
        hit = _TOK_MEMO.get(key)
        if hit is not None:
            _TOK_MEMO.move_to_end(key)
            _TOK_STATS["memo_hits"] += 1
            return hit
    if time.time() < _tok_fail_until:
        return None
    try:
        t0 = time.perf_counter()
        req = urllib.request.Request(
            LOCAL.rstrip("/") + "/tokenize",
            json.dumps({"model": _local_model_name(), "prompt": text}).encode(),
            {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=TOKENIZE_TIMEOUT) as r:
            n = int(json.load(r).get("count"))
        dt = time.perf_counter() - t0
        _TOK_STATS["calls"] += 1
        _TOK_STATS["ms_sum"] += dt * 1000
        if threading.current_thread() is threading.main_thread():
            _TOK_STATS["calls_on_loop"] += 1          # a call the warm step did not pre-empt (blocked the loop)
            _TOK_STATS["ms_on_loop"] += dt * 1000
        with _TOK_LOCK:
            _TOK_MEMO[key] = n
            while len(_TOK_MEMO) > 512:
                _TOK_MEMO.popitem(last=False)
        return n
    except Exception as e:
        _tok_fail_until = time.time() + 60
        log.warning("tokenize failed (%s) -> falling back to estimate for 60s", str(e)[:80])
        return None


def _local_model_name():
    return os.environ.get("SHIM_LOCAL_MODEL_NAME", "qwen-local")


# Sentinel meaning "never time out and overflow" -- the same value the code already relied on
# for "no remote configured at all" long before this file added a second reason to want it.
NO_OVERFLOW_SECONDS = 1e9


def admission_lane_limit(background, budget, fg_reserved, *, halo_control=False,
                         tiny=False, tiny_extra_lanes=0):
    """Class limit with one physical admission place protected for Halo control.

    Tiny calls may use spare engine places, but only Halo may take the final one.
    With no spare place configured, ordinary foreground yields one budget unit.
    The independent token reservation guard still applies to every class.
    """
    extra = max(0, tiny_extra_lanes)
    if halo_control:
        return budget + extra
    if background:
        return max(1, budget - fg_reserved)
    if tiny:
        return max(1, budget + extra - 1)
    return max(1, budget - (extra == 0))


def admission_wait_seconds(background, remote_available, peak, local_wait, bg_wait,
                            interactive_never_overflow):
    """How long a queued request waits for a local lane before the caller overflows it to
    remote. Pulled out of _route_completions as a pure function so the policy is testable
    without an event loop (gw-interactive-never-overflows, 2026-09-11).

    - No remote configured at all: nobody can overflow -- wait forever, for any class
      (unchanged, pre-existing behaviour).
    - Background: unchanged -- BG_WAIT normally, biased up to LOCAL_WAIT during peak hours
      (peak-hours DeepSeek pricing makes queueing locally cheaper than robot busywork on the
      2x-priced remote).
    - Interactive, with the policy on (default): NEVER overflow on a timeout. Local merely
      being busy is not local being unavailable, and "the uncensored local model is the point"
      (Kevin, 2026-09-11) -- so it keeps waiting for a lane instead of paying for remote. This
      does not touch the SEPARATE local-DOWN check earlier in _route_completions, which still
      overflows interactive immediately on a genuinely dead engine.
    - Interactive, with the policy off: the previous behaviour (LOCAL_WAIT then overflow).
    """
    if not remote_available:
        return NO_OVERFLOW_SECONDS
    if background:
        return local_wait if peak else bg_wait
    return NO_OVERFLOW_SECONDS if interactive_never_overflow else local_wait


def _parse_messages(body):
    try:
        return json.loads(body).get("messages") or []
    except Exception:
        return []


def _message_hashes(msgs):
    """Per-message content hash -- one sha256 per message, not one hash of the whole list.
    This is what makes "is request N a prefix of request N+1" a cheap elementwise comparison
    (see _is_prefix_of) instead of needing to store or re-hash raw conversation text, which for
    a 100K-token pi turn would mean holding that much text in memory per tracked client. None
    for an unhashable message -- treated as a guaranteed non-match, never a crash."""
    out = []
    for m in msgs:
        try:
            out.append(hashlib.sha256(json.dumps(m, sort_keys=True, default=str)
                                       .encode("utf-8", "replace")).hexdigest())
        except Exception:
            out.append(None)
    return out


def _is_prefix_of(prev_hashes, this_hashes):
    """True iff prev_hashes is a non-empty, strictly-shorter, elementwise-equal prefix of
    this_hashes. A real multi-turn conversation grows by MORE than one message per turn (the
    client echoes back the assistant's own reply alongside the next user message), so this
    checks list-prefix containment at any growth amount, not "exactly one message longer"."""
    if not prev_hashes or len(prev_hashes) >= len(this_hashes):
        return False
    return all(a is not None and a == b for a, b in zip(prev_hashes, this_hashes))


# ---------------- CACHE-AWARE PREFILL COST MODEL (see PREFIX_ALIGN_TOKENS) ----------------
_PM_NODES = collections.OrderedDict()      # rolling chain key -> [committed_at, ready_at|None]
_PM_STATS = collections.Counter()
_PM_PAIRS = collections.deque(maxlen=500)  # (predicted_credit, actual_cached, ptok) for local requests
_PM_INFLIGHT = {}                          # id(request) -> prediction dict (chain etc.) until the request ends


_CHAIN_ROUTE = collections.OrderedDict()   # [LANE CR] full-chain key of a finished request -> [route, t]


def _chain_route_lookup(chain, now=None):
    """(route, age_s, depth) of the deepest message prefix of `chain` that a finished request was served
    on, within the prefix-model TTL; (None, None, -1) when this conversation has no known predecessor."""
    now = time.time() if now is None else now
    for i in range(len(chain) - 1, -1, -1):
        node = _CHAIN_ROUTE.get(chain[i][0])
        if node is not None and now - node[1] <= PREFIX_MODEL_TTL_SECS:
            return node[0], now - node[1], i
    return None, None, -1


def _chain_route_note(chain, route, now=None):
    """Remember which route served this request's whole conversation (its deepest chain node)."""
    if not chain or route not in ("local", "remote"):
        return
    k = chain[-1][0]
    _CHAIN_ROUTE[k] = [route, time.time() if now is None else now]
    _CHAIN_ROUTE.move_to_end(k)
    while len(_CHAIN_ROUTE) > max(1000, PREFIX_MODEL_MAX_NODES):
        _CHAIN_ROUTE.popitem(last=False)


_PM_ANCHOR = collections.OrderedDict()     # [GW2] chain-end key of a finished LOCAL request -> [engine prompt_tokens, t]


def _anchor_note(chain, prompt_tokens, now=None):
    """Remember the engine's exact prompt size for this request's whole conversation (its deepest chain node)."""
    if not chain or not prompt_tokens:
        return
    k = chain[-1][0]
    _PM_ANCHOR[k] = [int(prompt_tokens), time.time() if now is None else now]
    _PM_ANCHOR.move_to_end(k)
    while len(_PM_ANCHOR) > max(1000, PREFIX_MODEL_MAX_NODES):
        _PM_ANCHOR.popitem(last=False)


def anchored_credit(chain, est, now=None):
    """(credit, age_s) from the deepest chain node a finished local request ended on, within the prefix-model TTL:
    that request's exact engine prompt_tokens rounded DOWN to the engine's cache block, capped below `est`.
    (0, None) when the conversation has no anchored predecessor."""
    now = time.time() if now is None else now
    for i in range(len(chain) - 1, -1, -1):
        node = _PM_ANCHOR.get(chain[i][0])
        if node is not None and now - node[1] <= PREFIX_MODEL_TTL_SECS:
            a = max(1, PREFIX_CREDIT_UNIT if PREFIX_CREDIT_UNIT > 0 else prefix_align_tokens())
            return max(0, min((node[0] // a) * a, int(est or 0) - 1)), now - node[1]
    return 0, None


_PROBE = {"disabled_until": 0.0, "why": None, "session": None}
_PROBE_STATS = collections.Counter()
_PROBE_LAT = collections.deque(maxlen=1000)
_PROBE_ERR = collections.deque(maxlen=2000)     # (model credit, probe cached, anchored credit) -- the replay record


def _probe_session():
    s = _PROBE.get("session")
    if s is None or s.closed:
        s = _PROBE["session"] = aiohttp.ClientSession()
    return s


async def credit_probe(body, now=None):
    """(cached_tokens, prompt_tokens, ms) from the engine's prefix-cache probe, or None (late, failed, absent,
    disabled). Never raises and never waits past CREDIT_PROBE_TIMEOUT_S. An engine without the endpoint (404) or
    with it switched off backs the probe off for 10 minutes instead of paying the timeout on every request."""
    now = time.time() if now is None else now
    if now < _PROBE["disabled_until"]:
        _PROBE_STATS["skipped_backoff"] += 1
        return None
    try:
        j = json.loads(body)
        payload = {"model": _local_model_name(), "messages": j["messages"]}
        for k in ("tools", "chat_template_kwargs", "add_generation_prompt", "continue_final_message"):
            if j.get(k) is not None:
                payload[k] = j[k]
    except Exception:
        _PROBE_STATS["unprobeable"] += 1
        return None
    t0 = time.perf_counter()
    try:
        async with _probe_session().post(LOCAL.rstrip("/") + "/v1/fork/prefix_cache_probe", json=payload,
                                         timeout=aiohttp.ClientTimeout(total=CREDIT_PROBE_TIMEOUT_S)) as r:
            if r.status == 404:
                _PROBE.update(disabled_until=now + 600, why="endpoint absent (engine without VLLM_FORK_PREFIX_PROBE=1)")
                _PROBE_STATS["absent"] += 1
                return None
            if r.status != 200:
                _PROBE_STATS["http_%d" % r.status] += 1
                return None
            d = await r.json()
    except asyncio.TimeoutError:
        _PROBE_STATS["timeout"] += 1
        return None
    except Exception as e:
        _PROBE.update(disabled_until=now + 30, why="error: %s" % str(e)[:80])
        _PROBE_STATS["error"] += 1
        return None
    if not d.get("enabled"):
        _PROBE.update(disabled_until=now + 600, why="engine reports the probe disabled")
        _PROBE_STATS["disabled"] += 1
        return None
    ms = (time.perf_counter() - t0) * 1000
    _PROBE_STATS["ok"] += 1
    _PROBE_LAT.append(ms)
    _PROBE.update(why=None)
    return int(d.get("cached_tokens") or 0), int(d.get("prompt_tokens") or 0), ms


async def _credit_sources(request, pm, body, ptok):
    """Shadow-log (and in live mode apply) the engine-reported / engine-anchored credit for this request.
    Mutates pm["credit"] / pm["computed"] only in live mode. Returns the credit source used for routing."""
    model_credit = int(pm.get("credit") or 0)
    src = "model"
    anch = None
    if CREDIT_ANCHOR != "off":
        try:
            anch, _aa = anchored_credit(pm["chain"], ptok)
            _active_set(request, pm_credit_anchored=anch, pm_anchor_age_s=None if _aa is None else round(_aa, 1))
            if _aa is None:
                anch = None
        except Exception:
            anch = None
    pr = None
    if CREDIT_PROBE in ("shadow", "live") and ptok >= CREDIT_PROBE_MIN_TOKENS and not _local_offline():
        pr = await credit_probe(body)
        if pr is not None:
            _active_set(request, pm_credit_probe=pr[0], probe_prompt_tokens=pr[1], probe_ms=round(pr[2], 1))
            _PROBE_ERR.append((model_credit, pr[0], anch))
    if CREDIT_PROBE == "live" and pr is not None:
        pm["credit"], pm["computed"], src = pr[0], max(1, pr[1] - pr[0]), "probe"
    elif CREDIT_ANCHOR == "live" and anch:
        pm["credit"], pm["computed"], src = anch, max(1, int(pm.get("est") or ptok) - anch), "anchor"
    if src != "model":
        _active_set(request, pm_credit_model=model_credit)
    _active_set(request, credit_source=src)
    return src


def _credit_probe_summary():
    lat = sorted(_PROBE_LAT)
    n = len(lat)
    errs = list(_PROBE_ERR)
    return {"mode": CREDIT_PROBE, "anchor_mode": CREDIT_ANCHOR, "timeout_s": CREDIT_PROBE_TIMEOUT_S,
            "min_tokens": CREDIT_PROBE_MIN_TOKENS, "stats": dict(_PROBE_STATS),
            "backoff_s": max(0, round(_PROBE["disabled_until"] - time.time())), "why": _PROBE["why"],
            "latency_ms": {"n": n, "p50": round(lat[n // 2], 1) if n else None,
                           "p95": round(lat[min(n - 1, int(n * 0.95))], 1) if n else None},
            "model_minus_probe": _err_dist([(m, p) for m, p, _ in errs]),
            "anchor_minus_probe": _err_dist([(a, p) for _, p, a in errs if a is not None])}


def warm_continuation(pm):
    """[LANE CR] True when the cost model says this request is a warm continuation of a local chain."""
    return (WARM_PRIORITY and int(pm.get("credit") or 0) >= WARM_PRIORITY_MIN_CREDIT
            and int(pm.get("computed") or 0) <= WARM_PRIORITY_MAX_COMPUTED)


def _pm_chain(body):
    """[(chain_key, cumulative_serialized_chars)] -- one entry per message boundary of the request,
    plus the grand total. The root folds in everything that changes the rendered PREFIX before the
    first message: the tool list and the template switches that put text ahead of it (thinking
    on/off, reasoning effort). Two requests share a chain node iff their rendered prompts share
    that whole prefix. Never raises; an unparsable body has an empty chain."""
    try:
        j = json.loads(body)
    except Exception:
        return [], 0
    msgs = j.get("messages") or []
    if not isinstance(msgs, list) or not msgs:
        return [], 0
    ctk = j.get("chat_template_kwargs")
    root = [j.get("tools"), ctk.get("enable_thinking") if isinstance(ctk, dict) else None,
            j.get("reasoning_effort")]
    try:
        root_s = json.dumps(root, sort_keys=True, default=str)
    except Exception:
        return [], 0
    h = hashlib.sha256(root_s.encode("utf-8", "replace")).digest()
    cum = len(root_s)
    out = []
    for m in msgs:
        try:
            ms = json.dumps(m, sort_keys=True, default=str)
        except Exception:
            break
        h = hashlib.sha256(h + ms.encode("utf-8", "replace")).digest()
        cum += len(ms)
        out.append((h, cum))
    return out, cum


def _pm_predict(body, est_tokens, now=None):
    """Predict the UNCACHED prefill tokens of this request against what the local engine has been
    fed recently. Returns a dict: computed (never < 1), credit (tokens predicted cached), best
    (deepest matched chain index or -1), age (s since that node was written), chain."""
    now = time.time() if now is None else now
    chain, total = _pm_chain(body)
    est = max(1, int(est_tokens or 0))
    best, age = -1, None
    for i, (k, _) in enumerate(chain):
        node = _PM_NODES.get(k)
        # A node counts only once the request that wrote it has produced its FIRST TOKEN (its
        # prefill is done and the blocks are in the cache); a still-prefilling prefix is a miss.
        if node is None or node[1] is None or now - node[1] > PREFIX_MODEL_TTL_SECS:
            break
        best, age = i, now - node[1]
    credit = 0
    if best >= 0 and total > 0:
        # chars->tokens is not uniform (tool-schema JSON vs prose), so the matched fraction is an
        # estimate: pad it by the fixed margin plus 3%, shave it by how well past predictions held
        # up against the engine's own cached_tokens, then round DOWN to whole attention blocks.
        matched = chain[best][1] / total * est
        a = max(1, PREFIX_CREDIT_UNIT if PREFIX_CREDIT_UNIT > 0 else prefix_align_tokens())
        credit = (int(max(0.0, (matched - PREFIX_HIT_MARGIN_TOKENS - 0.03 * matched) * _pm_trust())) // a) * a
    return {"computed": max(1, est - credit), "credit": credit, "best": best,
            "matched": int(chain[best][1] / total * est) if best >= 0 and total > 0 else 0,
            "age": age, "chain": chain, "total": total, "est": est}


def _pm_trust():
    """Fraction of predicted credit the engine has actually delivered lately (1.0 with no history,
    never below 0.3): the model's own eviction/staleness error, measured, not assumed."""
    c = d = 0
    for credit, cached, _ in _PM_PAIRS:
        if credit > 0:
            c += credit
            d += min(cached, credit)
    if c < 4 * max(1, prefix_align_tokens()):
        return 1.0
    return max(0.3, min(1.0, d / c))


def predict_computed_tokens(client, body, est_tokens):
    """Predicted uncached prefill tokens (>= 1) for this request. `client` is accepted for API
    compatibility only: the engine's cache is global, not per client."""
    return _pm_predict(body, est_tokens)["computed"]


def _pm_commit(chain, now=None):
    """Record that these prefixes are being prefilled on the local engine. Deepest key first so the
    SHALLOW nodes are the freshest and an LRU eviction can never cut the middle out of a chain."""
    if not chain:
        return
    now = time.time() if now is None else now
    for k, _ in reversed(chain):
        node = _PM_NODES.get(k)
        _PM_NODES[k] = [now, node[1] if node else None]
        _PM_NODES.move_to_end(k)
    while len(_PM_NODES) > max(1000, PREFIX_MODEL_MAX_NODES):
        _PM_NODES.popitem(last=False)


def _pm_prefill_done(request):
    """First token out of the local engine: the prompt is prefilled, so (a) its prefix is in the
    cache and (b) it no longer occupies the engine's prefill queue -- only its decode remains."""
    global _inflight_computed
    pm = _PM_INFLIGHT.get(id(request)) or {}
    flow_prefill_done(id(request))      # CF: its prefix is cached now -- held same-prefix waiters may go
    _pm_ready(pm.get("chain") or [])
    held = pm.pop("backlog_held", 0)
    if held:
        _inflight_computed = max(0, _inflight_computed - held)


def _pm_ready(chain, now=None):
    """The committing request produced its first token: its prefix is now in the engine's cache."""
    now = time.time() if now is None else now
    for k, _ in chain:
        node = _PM_NODES.get(k)
        if node is not None:
            node[1] = now


def _pm_reset(reason):
    """The engine's cache is gone (restart / crash / model switch): forget everything."""
    if _PM_NODES:
        log.info("prefix model reset (%s): %d nodes dropped", reason, len(_PM_NODES))
    _PM_NODES.clear()
    _PM_ANCHOR.clear()                  # [GW2] the engine's cache is gone, so are its exact anchors
    _PM_STATS["resets"] += 1


def _pm_feedback(info):
    """Grade one finished LOCAL request's prediction against the engine's own cached_tokens and
    unlearn prefixes the engine did not in fact have (over-prediction), so a stale or evicted node
    costs at most one mis-admitted request, not a run of them. Never raises."""
    try:
        flow_note_cache(info)
        pm = _PM_INFLIGHT.pop(info.get("pm_ref"), None) or {}
        credit, cached = info.get("pm_credit"), info.get("cached_actual")
        ptok_exact = info.get("ptok_exact_local")
        if credit is None or cached is None or info.get("route") != "local":
            return
        _PM_PAIRS.append((int(credit), int(cached), int(ptok_exact or 0)))
        a = max(1, prefix_align_tokens())
        if credit - cached > 2 * a:
            _PM_STATS["overpredict"] += 1
            chain, best, total = pm.get("chain") or [], pm.get("best", -1), pm.get("total") or 0
            est = info.get("est_tokens") or 0
            if chain and best >= 0 and total > 0 and est > 0:
                for i in range(best, -1, -1):
                    if chain[i][1] / total * est <= cached + a:
                        break
                    _PM_NODES.pop(chain[i][0], None)
        elif cached - credit > 2 * a:
            _PM_STATS["underpredict"] += 1
        else:
            _PM_STATS["accurate"] += 1
    except Exception as _e:
        _swallowed("_pm_feedback", _e)


def _pm_summary():
    pairs = list(_PM_PAIRS)
    n = len(pairs)
    out = {"nodes": len(_PM_NODES), "ttl_secs": PREFIX_MODEL_TTL_SECS, "align": prefix_align_tokens(),
           "trust": round(_pm_trust(), 3), "graded": n, **{k: v for k, v in _PM_STATS.items()}}
    if n:
        err = sorted(abs(c - a) for c, a, _ in pairs)
        out["abs_err_p50"] = err[n // 2]
        out["abs_err_p90"] = err[min(n - 1, int(n * 0.9))]
        out["credit_sum"] = sum(c for c, _, _ in pairs)
        out["cached_sum"] = sum(a for _, a, _ in pairs)
        out["ptok_sum"] = sum(p for _, _, p in pairs)
    return out


def _prefix_cache_observe(client, body, prompt_tokens):
    """Legacy hook, now a no-op. The old model learned from EVERY request regardless of route
    (a remote-served turn does not warm the local cache); learning now happens in _pm_commit(),
    called only when a request is admitted to the local engine."""
    return None


def _prefill_backlog_secs():
    """Seconds of uncached prefill already admitted to the local engine, at the measured rate."""
    return _inflight_computed / max(1.0, prefill_tps())


def prefill_window_ok(est_computed, halo_control=False):
    """May a request with this predicted uncached prefill join the engine's prefill queue now?"""
    if not USE_COMPUTED_COST or halo_control or PREFILL_ADMIT_SECS <= 0 or prefill_tps() <= 0:
        return True
    own = max(0, est_computed or 0) / prefill_tps()
    backlog = _prefill_backlog_secs()
    # An (almost) empty queue admits anything -- a request bigger than the whole window would
    # otherwise never run; the heavy-request backlog rule and the monster guard bound the rest.
    return own <= LIGHT_PREFILL_SECS or backlog <= LIGHT_PREFILL_SECS or backlog + own <= PREFILL_ADMIT_SECS


def _desired_units(body, client=None, computed=None):
    """The size-implied unit cost with NO ceiling applied -- how many lanes this request would
    take if it could have as many as it wants. Used two ways: estimate_units() clamps it to
    the normal reserved cap; the bg-idle-bypass below reads it directly to decide whether a
    truly enormous background job should be allowed MORE than that cap when nothing else is
    running (see BG_BIG_LOCAL_WHEN_IDLE).

    `client` is optional and, while USE_COMPUTED_COST stays at its default (0), unused --
    existing callers/tests that don't pass it see byte-identical behaviour. When flipped on,
    a request at or above BIG_TOKENS is costed by its PREDICTED computed tokens instead of its
    raw size (see _pm_predict()). `computed` lets the caller pass the prediction it already
    made, so the admission wait loop does not re-hash a 300 KB body on every poll."""
    tokens = _est_tokens(body)
    if tokens < BIG_TOKENS:
        return 1
    if USE_COMPUTED_COST and (client is not None or computed is not None):
        tokens = computed if computed is not None else predict_computed_tokens(client, body, tokens)
        if tokens < BIG_TOKENS:
            return 1
    return max(1, math.ceil(tokens / max(1, TOKENS_PER_UNIT)))


def estimate_units(body, budget=None, allow_full_budget=False, client=None, computed=None):
    """How many of the `budget` lane-slots this request should claim.

    2026-09-11 (gw-admission-proportional-units): proportional to estimated size, not
    all-or-nothing. A request under BIG_TOKENS still costs 1 (unchanged -- most traffic).
    A request at or above BIG_TOKENS costs ceil(tokens / TOKENS_PER_UNIT), floored at 1 (never
    inadmissible) and capped at `budget - FG_RESERVED` (at least FG_RESERVED lanes stay free
    for the NEXT caller, of any class, no matter how big this one is) -- UNLESS
    allow_full_budget is set, which raises that ceiling to `budget` itself; that escape hatch
    exists only for the bg-idle-bypass path, which already independently confirms nothing else
    is running or waiting before it ever asks for allow_full_budget=True. `budget` is
    injectable for testing; defaults to the live effective_budget() so callers need not import
    it too.
    """
    budget = effective_budget() if budget is None else budget
    ceiling = max(1, budget if allow_full_budget else budget - FG_RESERVED)
    return min(ceiling, _desired_units(body, client, computed))


def first_token_timeout(body, concurrency=1, local=False):
    # Prefill throughput is shared across concurrent local requests, so time-to-first-token scales
    # with how many are in flight. Without this, a legit big prefill queued behind others is misread
    # as a "wedged backend" (false failover + a 120s budget=1 backoff cascade). concurrency=1 for
    # remote/uncontended calls preserves the original tight deadline.
    # LOCAL-FIRST: a big prompt (>= BIG_PROMPT) on a LOCAL relay may take longer than
    # FIRST_TOKEN_MAX to prefill cold; its cap widens to LOCAL_FIRST_FIRST_TOKEN_MAX.
    factor = max(1, concurrency) if FT_CONCURRENCY_SCALE else 1
    est = _est_tokens(body)
    cap = FIRST_TOKEN_MAX
    if local and LOCAL_FIRST and BIG_PROMPT > 0 and est >= BIG_PROMPT:
        cap = max(FIRST_TOKEN_MAX, LOCAL_FIRST_FIRST_TOKEN_MAX)
    if local and not remote_ok():
        # No remote to fail over to (FULL LOCAL, or the 402 breaker is open): a first-token
        # deadline would only abort local work already in prefill and return a 503 the caller
        # retries from scratch. Measured 2026-10-02 10:39-12:03: ~50% of requests ended
        # 'held/local-failed' 503 after ~60 s this way while DeepSeek was empty by Kevin's choice.
        # Give local the long cap instead; a truly wedged engine is the engine watchdog's job.
        return max(cap, LOCAL_FIRST_FIRST_TOKEN_MAX, FIRST_TOKEN_BASE + (est / prefill_tps()) * factor)
    legacy = min(cap, FIRST_TOKEN_BASE + (est / prefill_tps()) * factor)
    if local and LOCAL_FIRST and LOCAL_FIRST_DERIVE:
        # LF: a request local-first kept local may sit behind the uncached prefill already in the engine's FIFO; that
        # wait is expected, not a wedge. Own prompt + the backlog ahead at the measured rate, capped at the local-first cap.
        expected = FIRST_TOKEN_BASE + (est + max(0, _inflight_computed)) / max(1.0, prefill_tps())
        return max(legacy, min(LOCAL_FIRST_FIRST_TOKEN_MAX, expected))
    return legacy


def is_background(body, request):
    """Background (cron/batch) request? Checked via X-Client header or prompt markers
    (Hermes cron turns literally open with 'scheduled cron job'). Background yields
    lanes to interactive traffic and fast-overflows instead of queueing."""
    xc = (request.headers.get("X-Client") or "").lower()
    if xc and any(k in xc for k in BG_XCLIENTS):
        return True
    if not BG_MARKERS:
        return False
    try:
        msgs = json.loads(body).get("messages") or []
    except Exception:
        return False
    if not isinstance(msgs, list):
        return False
    for m in (msgs[:2] + msgs[-1:]):
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, str) and any(k in c[:500] for k in BG_MARKERS):
            return True
        if isinstance(c, list):
            for b in c[:2]:
                t = b.get("text", "") if isinstance(b, dict) else ""
                if any(k in t[:500] for k in BG_MARKERS):
                    return True
    return False


def is_tiny(body):
    """VRAM-negligible micro-call: est. prompt + max_out <= TINY_TOKENS. Eligible for the fast-lane."""
    try:
        mt = int(json.loads(body).get("max_tokens") or DEFAULT_MAX_OUT)
    except Exception:
        mt = DEFAULT_MAX_OUT
    return (_est_tokens(body) + mt) <= TINY_TOKENS


def _preview(body):
    """Short, single-line preview of the last user message (for source-attribution logging)."""
    try:
        msgs = json.loads(body).get("messages") or []
        for m in reversed(msgs):
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                return c.replace("\n", " ")[:LOG_PREVIEW_CHARS]
            if isinstance(c, list):
                t = " ".join(b.get("text", "") for b in c if isinstance(b, dict)).strip()
                if t:
                    return t.replace("\n", " ")[:LOG_PREVIEW_CHARS]
    except Exception:
        pass
    return ""


def over_local_cap(body):
    try:
        mt = int(json.loads(body).get("max_tokens") or DEFAULT_MAX_OUT)
    except Exception:
        mt = DEFAULT_MAX_OUT
    # Never advertise more capacity than the engine actually exposes.  Older deployments had a
    # 700K shim cap beside a 524K vLLM -- the provider capability limit is authoritative here.
    return (_est_tokens(body) + mt + max(0, CONTEXT_SAFETY_MARGIN)) > min(max_local_tokens(), local_context_limit())


def strip_thinking(body):
    """Disable chain-of-thought for a LOCAL background request (cron/batch): status
    reports don't need 2-3K reasoning tokens holding a lane for minutes."""
    try:
        j = json.loads(body)
    except Exception:
        return body
    ctk = j.get("chat_template_kwargs") or {}
    ctk["enable_thinking"] = False
    j["chat_template_kwargs"] = ctk
    return json.dumps(j).encode()


def _no_think_policy(request, background):
    """Should chain-of-thought be stripped for this LOCAL request?

    Two triggers, unchanged: a background request when BG_NO_THINK is on, or a client IP in
    NO_THINK_IPS. Extracted 2026-09-09 only so both local lanes ask the SAME question -- see
    _prepare_local_body.
    """
    return bool((background and BG_NO_THINK)
                or (getattr(request, "remote", None) in NO_THINK_IPS))


def _halo_control_request(request, body):
    """True only for Halo's Hermes local-only control turn from its host.

    Hermes terminates the caller's control header, but its provider request is
    still identifiable by source host, X-Client and the estate-local alias.
    Other Hermes turns use estate; other local callers do not get priority.
    """
    try:
        xclient = (request.headers.get("X-Client") or "").strip().lower()
        model = str(json.loads(body).get("model") or "").strip().lower()
        return (getattr(request, "remote", None) == "10.0.1.95"
                and xclient == "halo-hermes" and model == "estate-local")
    except Exception:
        return False


def _prepare_local_body(request, body, background):
    """The one and only body transform for a request being sent to the LOCAL engine.

    2026-09-09. This chain used to be written out twice -- once on the main admission path and
    once in the TINY fast-lane -- and the two copies drifted: the tiny copy applied
    bound_local_output/thinking_budget_guard/nonthinking_sampling_profile/repetition_guard but
    never strip_thinking. So the entire no-think policy (NO_THINK_IPS *and* the background
    trigger) was silently skipped for every call small enough to take the fast lane, which is
    most of the traffic the policy exists for: the listed hosts are precisely the ones firing
    short status turns under TINY_TOKENS.

    Measured on the live gateway before this fix, same prompt, same minute:
        main lane, from 10.0.1.10  -> reasoning_content None      (policy applied)
        TINY lane, from 10.0.1.10  -> reasoning_content 114 chars (policy skipped)

    Duplication was the mechanism, so the fix is one function rather than a corrected copy.
    test_no_think_ips.py asserts the chain appears exactly once in this file.
    """
    halo_control = _halo_control_request(request, body)
    try:
        alias = _alias_for_request(body)
        if alias["kind"] in ("default", "builtin-local"):
            local_alias_body = json.loads(body)
            local_alias_body["model"] = _local_model_name()
            if WARM_PRIORITY and not halo_control and "priority" not in local_alias_body:
                try:
                    if request.get("cr_warm_priority"):
                        local_alias_body["priority"] = WARM_PRIORITY_VALUE     # [LANE CR] warm continuation
                except Exception as _e:
                    _swallowed("_prepare_local_body", _e)
            if halo_control:
                # vLLM priority scheduling preempts bulk FCFS work for the one
                # control decision that keeps the estate supervised. A bounded
                # non-thinking answer is sufficient for ACTION/REASONING/ORDER.
                local_alias_body["priority"] = -100
                local_alias_body["max_tokens"] = min(
                    int(local_alias_body.get("max_tokens") or 1024), 1024)
            body = json.dumps(local_alias_body).encode()
    except Exception as _e:
        _swallowed("_prepare_local_body", _e)
    prepared = repetition_guard(nonthinking_sampling_profile(thinking_budget_guard(bound_local_output(body))))
    prepared, repeat_n = repeated_tool_call_note(prepared)
    if repeat_n:
        _active_set(request, repeat_tool_note=repeat_n)
    if halo_control or _no_think_policy(request, background):
        prepared = strip_thinking(prepared)
    return prepared


def _is_empty_thinking_response(resp):
    """True if a NON-STREAMING local response came back with finish_reason=length and no
    content and no tool calls -- i.e. the model spent its entire budget thinking and
    returned NOTHING. Returns False for anything we cannot parse, so this can only ever
    trigger on a clearly-identified failure."""
    try:
        if not isinstance(getattr(resp, "body", None), (bytes, bytearray)):
            return False
        d = json.loads(resp.body)
        ch = (d.get("choices") or [{}])[0]
        if ch.get("finish_reason") != "length":
            return False
        m = ch.get("message") or {}
        if m.get("tool_calls"):
            return False
        return not (m.get("content") or "").strip()
    except Exception:
        return False


def repetition_guard(body):
    """Stop degenerate output loops at the SAMPLER instead of waiting for a timeout.

    vLLM's RepetitionDetectionParams watches for an N-gram pattern repeating and ends the
    sequence when it does. Without it, a looping model holds a lane until max_tokens or the
    client gives up -- the usual workaround is a client-side timeout, which wastes the whole
    generation and the lane.

    Deliberately CONSERVATIVE defaults: legitimate output repeats itself (code, tables,
    lists, JSON arrays), so we require a fairly long pattern repeated several times rather
    than trying to catch every loop. Better to miss a loop than truncate real work.
    Disable with SHIM_REP_GUARD=0; tune via SHIM_REP_MIN/MAX_PATTERN and SHIM_REP_MIN_COUNT.
    """
    if not REP_GUARD:
        return body
    try:
        j = json.loads(body)
    except Exception:
        return body
    if j.get("repetition_detection") is not None:      # caller was explicit
        return body
    j["repetition_detection"] = {
        "min_pattern_size": REP_MIN_PATTERN,
        "max_pattern_size": REP_MAX_PATTERN,
        "min_count": REP_MIN_COUNT,
    }
    return json.dumps(j).encode()


#: [LANE RA / L143] marker that makes repeated_tool_call_note() idempotent (the local prep chain can run twice
#: on one request: once on the body, once on its compacted form).
REPEAT_NOTE_MARKER = "[gateway note: repeated identical tool call]"


def _tool_exchange_tail(messages):
    """PURE. Split `messages` into trailing (assistant tool_calls -> tool results) exchanges, newest first.

    Each exchange is (key, last_tool_index) where key = (calls, results): the calls as sorted
    (function name, canonical arguments) pairs and the results as the tool contents in call order. Stops at
    the first message that is not part of such an exchange (a user turn, a plain assistant answer, ...)."""
    out = []
    i = len(messages) - 1
    while i >= 0:
        tools = []
        while i >= 0 and isinstance(messages[i], dict) and messages[i].get("role") == "tool":
            tools.append(messages[i])
            i -= 1
        if not tools or i < 0:
            break
        asst = messages[i]
        calls = asst.get("tool_calls") if isinstance(asst, dict) and asst.get("role") == "assistant" else None
        if not calls or len(calls) != len(tools):
            break
        tools.reverse()
        by_id = {t.get("tool_call_id"): t for t in tools}
        sig, res = [], []
        for n, c in enumerate(calls):
            fn = (c or {}).get("function") or {}
            raw = fn.get("arguments")
            try:
                args = json.dumps(json.loads(raw) if isinstance(raw, str) else raw, sort_keys=True)
            except Exception:
                args = str(raw)
            sig.append((str(fn.get("name")), args))
            t = by_id.get((c or {}).get("id"), tools[n])
            content = t.get("content")
            res.append(content if isinstance(content, str) else json.dumps(content, sort_keys=True))
        # results are compared without a note this gateway may have appended to them on an earlier turn
        res = [r.split("\n\n" + REPEAT_NOTE_MARKER)[0] for r in res]
        out.append(((tuple(sorted(sig)), tuple(res)), i + len(tools)))
        i -= 1
    return out


def repeated_tool_call_note(body):
    """Tell the model, inside the newest tool result, that it is looping on one identical tool call.

    ROOT (L143, measured on the runner's pi transcripts 2026-10-03): Qwen 27B can re-issue the SAME tool call
    with the SAME arguments, get the SAME result, and do it again -- 21 times in a row mid-task (repair-c93dd3e57b,
    `grep ... | head -20` x21), and 7-54 times at the end of finished work ("final git status" loops) in 11 of
    514 PASS attempts; 55 of 2825 runner attempts had >=4 consecutive identical calls. Nothing in the request
    tells it so: presence/repetition penalties act within ONE generation, not across turns, and the harness
    feeds the identical result back verbatim, which reinforces the loop.

    Fires only when the newest REPEAT_NOTE_MIN (default 3) tool exchanges are byte-identical in BOTH the calls
    (name + canonical arguments) and their results, so a re-run whose output changed (tests after an edit, a
    poll that progressed) never counts. It appends one short factual note to the last tool result; it never
    drops, reorders or rewrites anything else, never blocks the call, and is idempotent. Returns
    (body, repeats) where repeats is 0 when nothing was changed. Disable with SHIM_REPEAT_NOTE=0."""
    if not REPEAT_NOTE or REPEAT_NOTE_MIN < 2:
        return body, 0
    try:
        j = json.loads(body)
        messages = j.get("messages")
        if not isinstance(messages, list) or not messages:
            return body, 0
        last = messages[-1]
        if not isinstance(last, dict) or last.get("role") != "tool":
            return body, 0
        tail = _tool_exchange_tail(messages)
        if len(tail) < REPEAT_NOTE_MIN:
            return body, 0
        first = tail[0][0]
        n = 1
        for key, _ in tail[1:]:
            if key != first:
                break
            n += 1
        if n < REPEAT_NOTE_MIN:
            return body, 0
        content = last.get("content")
        flat = content if isinstance(content, str) else json.dumps(content)
        if REPEAT_NOTE_MARKER in (flat or ""):
            return body, 0
        note = (REPEAT_NOTE_MARKER + " You have now made this exact tool call %d times in a row and received "
                "exactly the same result each time. Running it again will return the same result. Use the result "
                "above: take a different next step, or, if the task is complete, stop calling tools and give your "
                "final answer." % n)
        if isinstance(content, str):
            last["content"] = content + "\n\n" + note
        elif isinstance(content, list):
            last["content"] = content + [{"type": "text", "text": "\n\n" + note}]
        else:
            return body, 0
        return json.dumps(j).encode(), n
    except Exception:
        return body, 0


def thinking_budget_guard(body):
    """Bound chain-of-thought at the SAMPLER so a request cannot spend its whole budget
    thinking and return nothing.

    Uses vLLM's `thinking_token_budget` sampling param, which counts thinking tokens and
    force-injects the reasoning end token when the budget is hit. That is a HARD bound in
    the sampler -- unlike `reasoning_effort=low`, which is only a prompt suggestion and
    demonstrably does not bound anything (measured 2026-08-14: max_tokens=4000 WITH
    reasoning_effort=low returned empty, while max_tokens=8000 with full thinking completed
    in 3519 tokens).

    Measured effect at the source:
        no budget                  -> 2341 tok, 6324ch thinking, 543ch answer, 25.5s
        thinking_token_budget=200  ->  292 tok,  812ch thinking, 512ch answer,  3.6s
    Same answer, 7x faster, 8x fewer tokens.

    Policy: give thinking a fixed share of the caller's budget (THINK_BUDGET_FRAC), floored
    and capped, so a small max_tokens always leaves room for an actual answer. Never
    override a caller who set thinking behaviour explicitly. Disable with SHIM_THINK_GUARD=0.
    """
    if not THINK_GUARD:
        return body
    try:
        j = json.loads(body)
    except Exception:
        return body
    # Only a REAL bound counts as the caller having handled this. reasoning_effort is a
    # prompt-level hint that demonstrably does NOT bound thinking (measured: max_tokens=4000
    # with reasoning_effort=low still returned empty), so treating it as "caller knows best"
    # silently disabled the budget for anyone who sets it. Both Hermes instances send
    # reasoning_effort=medium on every request, so they were bypassing this entirely.
    _et = (j.get("chat_template_kwargs") or {}).get("enable_thinking")
    if j.get("thinking_token_budget") is not None or _et is False:
        return body
    if _et is not None:
        # [LANE GW2 / L95] "thinking ON" is not "thinking BOUNDED": pi says enable_thinking=true on every turn
        # and was left unbounded (16384-token empty turns). THINK_BUDGET_EXPLICIT>0 bounds it at the sampler;
        # 0 (default) keeps the old "caller was explicit" behaviour byte for byte.
        try:
            mt = int(j.get("max_tokens") or 0)
        except Exception:
            return body
        if THINK_BUDGET_EXPLICIT <= 0 or mt <= 0 or THINK_BUDGET_EXPLICIT >= mt:
            return body
        j["thinking_token_budget"] = THINK_BUDGET_EXPLICIT
        return json.dumps(j).encode()
    try:
        mt = int(j.get("max_tokens") or 0)
    except Exception:
        return body
    if mt <= 0:
        return body
    budget = int(mt * THINK_BUDGET_FRAC)
    budget = max(THINK_BUDGET_MIN, min(budget, THINK_BUDGET_MAX))
    if budget >= mt:                      # nothing left for an answer -> no thinking at all
        ctk = j.get("chat_template_kwargs") or {}
        ctk["enable_thinking"] = False
        j["chat_template_kwargs"] = ctk
    else:
        j["thinking_token_budget"] = budget
    return json.dumps(j).encode()


def nonthinking_sampling_profile(body):
    """Inject Qwen3's official NON-THINKING sampling profile for LOCAL non-thinking requests.

    EXP-026 (measured 2026-08-16): the official non-thinking profile (temperature 0.7, top_p
    0.80, top_k 20, presence_penalty 1.5) fixed Qwen3's long-generation repetition (loop_probe
    5/10 -> 8/10, dupes 5 -> 1) where our gencfg default (temp 0.6, top_p 0.95, pp 0) looped.

    Fires ONLY when the request is non-thinking -- chat_template_kwargs.enable_thinking is False
    (which thinking_budget_guard may itself have just set, so this must run AFTER it in the chain).
    THINKING-mode requests must keep presence_penalty=0 per Qwen's official guidance, so they are
    left untouched. Each of presence_penalty/top_p/temperature/top_k is applied ONLY if the caller
    did not already provide that key -- an explicit caller value always wins. Exception-safe: any
    error returns the body unchanged so it can never break forwarding. Disable with
    SHIM_NONTHINK_PROFILE=0; tune via SHIM_NONTHINK_{PP,TOP_P,TEMP,TOP_K}.
    """
    if not NONTHINK_PROFILE:
        return body
    try:
        j = json.loads(body)
        if (j.get("chat_template_kwargs") or {}).get("enable_thinking") is not False:
            return body                                    # thinking / unspecified -> untouched
        applied = False
        for key, val in (("presence_penalty", NONTHINK_PP), ("top_p", NONTHINK_TOP_P),
                         ("temperature", NONTHINK_TEMP), ("top_k", NONTHINK_TOP_K)):
            if key not in j:                               # never override an explicit caller value
                j[key] = val
                applied = True
        if not applied:                                    # caller set everything -> nothing to do
            return body
        if not getattr(nonthinking_sampling_profile, "_logged", False):
            nonthinking_sampling_profile._logged = True
            log.info("non-thinking sampling profile active (pp=%s top_p=%s temp=%s top_k=%s)",
                     NONTHINK_PP, NONTHINK_TOP_P, NONTHINK_TEMP, NONTHINK_TOP_K)
        return json.dumps(j).encode()
    except Exception:
        return body


def bound_local_output(body):
    """Ensure a request routed to LOCAL has a bounded max_tokens. Absent/0 (unbounded) OR an
    oversized explicit ceiling (e.g. Hermes's 65536 — a ceiling, not real usage: typical turns
    finish at EOS in a few K tokens) is CLAMPED to LOCAL_MAX_OUT so the request runs local-first
    instead of being rerouted; only genuinely >LOCAL_MAX_OUT generations hit the cap
    (finish_reason=length). Returns (possibly-rewritten) body bytes."""
    if LOCAL_MAX_OUT <= 0:
        return body
    try:
        j = json.loads(body)
    except Exception:
        return body
    # Normalize the alternate OpenAI output limit before reservation/relay so
    # the engine cannot prefer a larger uncapped max_completion_tokens value.
    mt = j.get("max_completion_tokens", j.get("max_tokens"))
    had_alias = "max_completion_tokens" in j
    j.pop("max_completion_tokens", None)
    if not mt or int(mt) <= 0 or int(mt) > LOCAL_MAX_OUT:
        j["max_tokens"] = LOCAL_MAX_OUT
        return json.dumps(j).encode()
    if had_alias:
        j["max_tokens"] = int(mt)
        return json.dumps(j).encode()
    return body


def local_memory_reservation(body):
    """Conservative KV-token ceiling for the exact prepared body sent locally.

    Count every requested sequence; cached prefixes still occupy KV memory.
    With output bounding disabled, an unspecified limit reserves the complete
    configured local context ceiling instead of assuming a short generation.
    """
    data = json.loads(body)
    count = data.get("n", 1)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("n must be a positive integer")
    best_of = data.get("best_of") or count
    if isinstance(best_of, bool) or not isinstance(best_of, int) or best_of < count:
        raise ValueError("best_of must be an integer at least n")
    count = best_of
    prompt = _est_tokens(body)
    output = data.get("max_completion_tokens", data.get("max_tokens"))
    if output is None or output == 0:
        mlt = max_local_tokens()
        if mlt <= 0:
            raise ValueError("local output requires a finite token limit")
        return max(prompt, mlt) * count, count
    if isinstance(output, bool) or not isinstance(output, int) or output < 0:
        raise ValueError("local output token limit must be a positive integer")
    return (prompt + output) * count, count


def _halo_control_token_reserve():
    tb = token_budget()
    return min(65536, tb // 8) if tb >= 100000 else 0


def _memory_available(reservation, *, halo_control=False):
    tb = token_budget()
    if tb <= 0:
        return True
    # Keep one bounded Halo control prompt's KV reservation available even
    # when ordinary work fills the local engine. The reserve is a fraction of
    # the validated token pool and does not narrow small test/backoff pools.
    limit = tb if halo_control else tb - _halo_control_token_reserve()
    return _inflight_reserved_tokens + reservation <= limit


def wants_stream(body):
    try:
        return bool(json.loads(body).get("stream"))
    except Exception:
        return False


def _requested_output_tokens(body, default=DEFAULT_MAX_OUT):
    try:
        data = json.loads(body)
        value = data.get("max_completion_tokens", data.get("max_tokens"))
        value = int(value or 0)
        return value if value > 0 else int(default)
    except (TypeError, ValueError, json.JSONDecodeError):
        return int(default)


def _context_fits(body, context_limit, provider_max_output=None):
    """Return (fits, prompt_tokens, output_tokens, total_tokens) for one provider.

    This is deliberately conservative.  A provider-specific context limit includes both input
    and output, and the safety margin absorbs tokenizer/serialization overhead.  No provider is
    selected by this helper; it only answers whether an intact request can fit.
    """
    try:
        limit = int(context_limit)
    except (TypeError, ValueError):
        return False, 0, 0, 0
    ptok = int(_est_tokens(body))
    requested = _requested_output_tokens(body, DEFAULT_MAX_OUT)
    if provider_max_output:
        requested = min(requested, int(provider_max_output))
    total = ptok + requested + max(0, int(CONTEXT_SAFETY_MARGIN))
    return total <= limit, ptok, requested, total


def _compact_context_body(body, context_limit, provider_max_output=None):
    """Deterministically compact old turns while preserving instructions and active tool state.

    The gateway never silently drops context: it inserts an explicit system marker containing the
    number of omitted messages.  System/developer instructions and a contiguous recent tail are
    retained.  If even that cannot fit, return (None, metadata) and the caller returns a precise
    413 rather than sending a provider an invalid request.
    """
    try:
        data = json.loads(body)
        messages = data.get("messages")
        if not isinstance(messages, list) or len(messages) <= CONTEXT_COMPACTION_KEEP:
            return None, {"omitted": 0, "reason": "no-compaction-candidate"}
    except Exception:
        return None, {"omitted": 0, "reason": "invalid-json"}
    prefix = [m for m in messages if isinstance(m, dict) and m.get("role") in ("system", "developer")]
    non_prefix = [m for m in messages if not (isinstance(m, dict) and m.get("role") in ("system", "developer"))]
    keep = max(2, int(CONTEXT_COMPACTION_KEEP))
    for tail_count in range(min(keep, len(non_prefix)), 1, -1):
        tail = non_prefix[-tail_count:]
        # Never start with an orphaned tool result. Include its assistant tool-call turn and the
        # immediately preceding user turn when present.
        start = len(non_prefix) - tail_count
        while start > 0 and isinstance(non_prefix[start], dict) and non_prefix[start].get("role") == "tool":
            start -= 1
        if start < len(non_prefix) - tail_count:
            tail = non_prefix[start:]
        omitted = len(messages) - len(prefix) - len(tail)
        marker = {"role": "system", "content":
                  f"[Gateway context compaction: {omitted} older messages omitted to fit the selected provider; "
                  "the retained instructions, recent turns, and active tool state are authoritative.]"}
        candidate = dict(data)
        candidate["messages"] = prefix + ([marker] if omitted else []) + tail
        candidate.pop("max_completion_tokens", None)
        raw = json.dumps(candidate, ensure_ascii=False).encode()
        fits, ptok, outtok, total = _context_fits(raw, context_limit, provider_max_output)
        if fits:
            return raw, {"omitted": omitted, "prompt_tokens": ptok, "output_tokens": outtok, "total_tokens": total}
    return None, {"omitted": len(messages), "reason": "instructions-exceed-context"}


def _prepare_provider_context(body, context_limit, provider_max_output=None):
    fits, ptok, outtok, total = _context_fits(body, context_limit, provider_max_output)
    if fits:
        return body, {"compacted": False, "prompt_tokens": ptok, "output_tokens": outtok, "total_tokens": total}
    if not CONTEXT_COMPACTION_ENABLED:
        return None, {"compacted": False, "reason": "context-exceeded", "prompt_tokens": ptok,
                       "output_tokens": outtok, "total_tokens": total, "limit": int(context_limit)}
    compacted, meta = _compact_context_body(body, context_limit, provider_max_output)
    if compacted is None:
        return None, {"compacted": False, "reason": meta.get("reason", "context-exceeded"),
                       "prompt_tokens": ptok, "output_tokens": outtok, "total_tokens": total,
                       "limit": int(context_limit)}
    return compacted, {"compacted": True, **meta, "limit": int(context_limit)}


def _context_error(meta, provider):
    return web.json_response({"error": {"message":
        f"context exceeds {provider} capacity: prompt={meta.get('prompt_tokens', 0)} + "
        f"output={meta.get('output_tokens', 0)} + margin={CONTEXT_SAFETY_MARGIN} > "
        f"limit={meta.get('limit', 0)}", "type": "context_length_exceeded",
        "provider": provider, "compaction_attempted": CONTEXT_COMPACTION_ENABLED}}, status=413)


def remap_for_remote(body, remote_model=None, remote_max_output=None):
    """Rewrite the model to a remote model and strip vLLM-only params."""
    try:
        j = json.loads(body)
    except Exception:
        return body
    j["model"] = remote_model or REMOTE_MODEL
    if remote_max_output:
        try:
            requested = int(j.get("max_completion_tokens", j.get("max_tokens")) or 0)
            if requested <= 0 or requested > int(remote_max_output):
                j.pop("max_completion_tokens", None)
                j["max_tokens"] = int(remote_max_output)
        except (TypeError, ValueError):
            j.pop("max_completion_tokens", None)
            j["max_tokens"] = int(remote_max_output)
    for k in REMOTE_STRIP:
        j.pop(k, None)
    rf = j.get("response_format")
    if isinstance(rf, dict) and rf.get("type") == "json_schema":
        j["response_format"] = {"type": "json_object"}
    # DeepSeek's THINKING mode (V4) rejects any conversation that ends on a tool result whose
    # preceding assistant tool_call lacks reasoning_content — "The `reasoning_content` in the
    # thinking mode must be passed back to the API" (verified 2026-08-19). Hermes/pi histories are
    # generated by the LOCAL model (Qwen) and never carry DeepSeek reasoning_content, and Hermes
    # sends reasoning_effort=medium on every turn, which puts DeepSeek in thinking mode. Any tool-
    # using agent that failed over mid-loop hit an unconditional 400 → "model provider failed".
    # Force non-thinking on the DeepSeek failover: it removes the passback requirement entirely,
    # matches the gateway's existing no-think policy for Hermes (NO_THINK_IPS), and a degraded-mode
    # overflow wants a fast correct answer over deep CoT. thinking:disabled beats reasoning_effort.
    if REMOTE_NO_THINK and "deepseek" in (REMOTE_BASE or "").lower():
        j["thinking"] = {"type": "disabled"}
    return json.dumps(j).encode()


async def local_healthy():
    now = time.time()
    if now - _health["at"] < HEALTH_TTL:
        return _health["ok"]
    ok = False
    foreign = 0
    foreign_tokens = 0
    foreign_heavy = False
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{LOCAL}/health", timeout=aiohttp.ClientTimeout(total=2)) as r:
                ok = (r.status == 200)
            if ok and FOREIGN_LOAD_GUARD:
                try:
                    async with s.get(f"{LOCAL.rsplit('/v1',1)[0] if LOCAL.endswith('/v1') else LOCAL}/metrics",
                                     timeout=aiohttp.ClientTimeout(total=2)) as m:
                        if m.status == 200:
                            running = None
                            kv_pct = None
                            for line in (await m.text()).splitlines():
                                if line.startswith("vllm:num_requests_running{") or line.startswith("vllm:num_requests_running "):
                                    running = int(float(line.rsplit(" ", 1)[-1]))
                                elif line.startswith("vllm:kv_cache_usage_perc{") or line.startswith("vllm:kv_cache_usage_perc "):
                                    kv_pct = float(line.rsplit(" ", 1)[-1])
                                if running is not None and kv_pct is not None:
                                    break
                            if running is not None:
                                foreign = max(0, running - _inflight)
                            if kv_pct is not None:
                                # KV the engine holds beyond what this gateway admitted (prompt tokens);
                                # a foreign monster prefill shows up here as tens of thousands of tokens,
                                # a foreign tiny probe as ~0.
                                foreign_tokens = max(0, int(kv_pct * pool_info()["tokens"]) - _inflight_tokens)
                            foreign_heavy = bool(foreign > 0 and BIG_TOKENS > 0 and foreign_tokens >= BIG_TOKENS)
                except Exception:
                    foreign = 0
                    foreign_tokens = 0
                    foreign_heavy = False
    except Exception:
        ok = False
    if ok and not _health.get("ok", False) and _health.get("at", 0.0) > 0:
        _pm_reset("engine back after being down")      # its prefix cache did not survive the outage
        _capacity_engine_changed(now, "engine back after being down")
    _health.update(ok=ok, at=now, foreign=foreign, foreign_tokens=foreign_tokens, foreign_heavy=foreign_heavy)
    return ok


def _is_oom(status, text):
    t = (text or "").lower()
    return (status == 503) or ("enginedead" in t or "out of memory" in t
            or "outofmemory" in t or "enginecore" in t)


# ---------------- upstream forwarding ----------------
async def _open(session, base, path, body, key, streaming):
    """POST to an upstream and return the aiohttp response (caller manages ctx)."""
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    to = aiohttp.ClientTimeout(total=None, sock_read=600) if streaming else aiohttp.ClientTimeout(total=600)
    base = str(base or "").rstrip("/")
    path = "/" + str(path or "").lstrip("/")
    # Callers address the gateway as /v1/chat/completions.  Accept both endpoint styles in the
    # registry: https://host (append /v1) and https://host/v1 (avoid /v1/v1 duplication).
    if base.lower().endswith("/v1") and path.lower().startswith("/v1/"):
        url = base + path[3:]
    else:
        url = base + path
    return await session.post(url, data=body, headers=headers, timeout=to)


# First-token gate: hold a streaming response before committing to the client, so a
# local crash DURING prefill (before any token) can fail over cleanly.
FIRST_GATE_MAX_BYTES = 16384
FIRST_GATE_MAX_CHUNKS = 8


def _looks_meaningful(text):
    """True once the upstream emitted real generated progress (a content/reasoning token,
    a tool call, or a finish_reason) — the safe point to commit the response to the client."""
    i = text.find('"reasoning_content":"')
    if i != -1 and text[i + 21:i + 22] not in ("", '"'):
        return True
    i = text.find('"reasoning":"')      # [GW2] vLLM >= 0.29 streams reasoning as delta.reasoning
    if i != -1 and text[i + 13:i + 14] not in ("", '"'):
        return True
    i = text.find('"content":"')
    if i != -1 and text[i + 11:i + 12] not in ("", '"'):
        return True
    # tool-call responses stream function name/args in tool_calls, not content — without
    # this check, a tool-call-only response hits the first-token timeout and false-failovers
    if '"tool_calls":[' in text or '"function_call":{' in text:
        return True
    if '"finish_reason":"' in text:
        return True
    return False


def _sse_content_shape(data, saw_content, saw_tool_call):
    """Update (saw_content, saw_tool_call) from one chunk of raw SSE bytes -- possibly several
    'data: {...}' lines, possibly a partial line at either end (like every other scanner in
    _relay, it is fed overlapping/repeated chunks and is safe to call on the same bytes more
    than once: OR-ing booleans is idempotent). Never raises; an unparseable line is skipped,
    not fatal, exactly like _scan_usage's existing tolerance for the same input shape.

    2026-09-11 (gw-streaming-content-classifier): pulled out of _relay's closure so the
    content-shape LOGIC is testable without mocking the aiohttp streaming machinery around it.
    """
    try:
        for ln in bytes(data).split(b"\n"):
            ln = ln.strip()
            if not ln.startswith(b"data:"):
                continue
            raw = ln[5:].strip()
            if raw in (b"", b"[DONE]"):
                continue
            delta = ((json.loads(raw).get("choices") or [{}])[0].get("delta") or {})
            c = delta.get("content")
            if c and str(c).strip():
                saw_content = True
            if delta.get("tool_calls"):
                saw_tool_call = True
    except Exception:
        pass
    return saw_content, saw_tool_call


# [LANE GW2 / L94, 2026-10-03] Response-shape telemetry that a hard number can be read from: finish_reason and
# tool-call ARGUMENT validity for local rows (before this, has_tool_calls recorded presence only and every local
# row had finish_reason=null, so neither stop-vs-length nor broken tool calls were measurable). Telemetry only.
TOOL_ARGS_MAX_CHARS = 1 << 20      # per response; a bigger argument blob is reported as unchecked (None), not invalid


class _SSEShape:
    """Lossless running facts over one streamed chat completion: line-buffered (an SSE line split across network
    chunks is reassembled, unlike _sse_content_shape's best-effort scan), choice 0 only. Tracks finish_reason,
    the tool calls' names + concatenated argument fragments, reasoning/content character counts, and how much
    reasoning had streamed before the first real output (content or tool call). Never raises."""

    __slots__ = ("tail", "finish", "tools", "reasoning_chars", "content_chars", "reasoning_at_output",
                 "args_chars", "overflow", "content_nonblank")

    def __init__(self):
        self.tail = b""
        self.finish = None
        self.tools = {}
        self.reasoning_chars = 0
        self.content_chars = 0
        self.reasoning_at_output = None
        self.args_chars = 0
        self.overflow = False
        self.content_nonblank = False

    @property
    def saw_output(self):
        return self.reasoning_at_output is not None

    def reasoning_tokens_est(self):
        return int(self.reasoning_chars / max(0.5, REASONING_CHARS_PER_TOKEN))

    def feed(self, chunk):
        try:
            complete, self.tail = _sse_split_tail(self.tail, chunk)
            if complete:
                self._scan(complete)
        except Exception:
            pass

    def close(self):
        try:
            if self.tail:
                tail, self.tail = self.tail, b""
                self._scan(tail)
        except Exception:
            pass

    def _output(self):
        if self.reasoning_at_output is None:
            self.reasoning_at_output = self.reasoning_chars

    def _scan(self, data):
        for ln in bytes(data).split(b"\n"):
            ln = ln.strip()
            if not ln.startswith(b"data:"):
                continue
            raw = ln[5:].strip()
            if raw in (b"", b"[DONE]"):
                continue
            try:
                ev = json.loads(raw)
            except Exception:
                continue
            for ch in (ev.get("choices") or []) if isinstance(ev, dict) else []:
                if not isinstance(ch, dict) or (ch.get("index") or 0) != 0:
                    continue
                if ch.get("finish_reason"):
                    self.finish = ch["finish_reason"]
                d = ch.get("delta") or {}
                r = d.get("reasoning") or d.get("reasoning_content")
                if r:
                    self.reasoning_chars += len(r)
                c = d.get("content") if d else ch.get("text")     # legacy /v1/completions streams `text`
                if c:
                    self.content_chars += len(c)
                    if str(c).strip():
                        self.content_nonblank = True
                        self._output()
                for tc in d.get("tool_calls") or []:
                    self._output()
                    slot = self.tools.setdefault(tc.get("index") or 0, {"name": "", "args": []})
                    f = tc.get("function") or {}
                    if f.get("name") and not slot["name"]:
                        slot["name"] = f["name"]
                    a = f.get("arguments")
                    if a:
                        self.args_chars += len(a)
                        if self.args_chars > TOOL_ARGS_MAX_CHARS:
                            self.overflow = True
                        else:
                            slot["args"].append(a)

    def calls(self):
        return [{"name": v["name"], "arguments": "".join(v["args"])} for _, v in sorted(self.tools.items())]


def _schema_error(value, schema, path="$", depth=0):
    """First violation of `value` against a JSON-schema subset (type, enum, const, required, properties,
    additionalProperties, items, anyOf/oneOf, minItems/maxItems), or None. Keywords outside the subset ($ref,
    pattern, formats, ...) are ignored -- i.e. treated as satisfied -- so this can under-report, never over-report."""
    if not isinstance(schema, dict) or depth > 32:
        return None
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        ok = False
        for ty in types:
            if (ty == "object" and isinstance(value, dict)) or (ty == "array" and isinstance(value, list)) \
                    or (ty == "string" and isinstance(value, str)) or (ty == "null" and value is None) \
                    or (ty == "boolean" and isinstance(value, bool)) \
                    or (ty == "integer" and isinstance(value, int) and not isinstance(value, bool)) \
                    or (ty == "integer" and isinstance(value, float) and value.is_integer()) \
                    or (ty == "number" and isinstance(value, (int, float)) and not isinstance(value, bool)) \
                    or ty not in ("object", "array", "string", "null", "boolean", "integer", "number"):
                ok = True
                break
        if not ok:
            return "%s: expected %s, got %s" % (path, "|".join(map(str, types)), type(value).__name__)
    if "enum" in schema and isinstance(schema["enum"], list) and value not in schema["enum"]:
        return "%s: not in enum" % path
    if "const" in schema and value != schema["const"]:
        return "%s: != const" % path
    for kw in ("anyOf", "oneOf"):
        alts = schema.get(kw)
        if isinstance(alts, list) and alts and all(_schema_error(value, a, path, depth + 1) for a in alts):
            return "%s: matches no %s branch" % (path, kw)
    if isinstance(value, dict):
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        for k in schema.get("required") or []:
            if k not in value:
                return "%s: missing required '%s'" % (path, k)
        extra = schema.get("additionalProperties")
        for k, v in value.items():
            if k in props:
                e = _schema_error(v, props[k], "%s.%s" % (path, k), depth + 1)
                if e:
                    return e
            elif extra is False:
                return "%s: unexpected property '%s'" % (path, k)
            elif isinstance(extra, dict):
                e = _schema_error(v, extra, "%s.%s" % (path, k), depth + 1)
                if e:
                    return e
    if isinstance(value, list):
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            return "%s: fewer than minItems" % path
        if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
            return "%s: more than maxItems" % path
        if isinstance(schema.get("items"), dict):
            for i, v in enumerate(value):
                e = _schema_error(v, schema["items"], "%s[%d]" % (path, i), depth + 1)
                if e:
                    return e
    return None


def _request_tool_schemas(body):
    """{tool name: parameters schema} from a request body (bytes/str/dict); None when it declares no tools."""
    try:
        j = json.loads(body) if isinstance(body, (bytes, bytearray, str)) else body
        tools = j.get("tools") if isinstance(j, dict) else None
        if not tools:
            return None
        out = {}
        for t in tools:
            f = (t or {}).get("function") or {}
            if f.get("name"):
                out[f["name"]] = f.get("parameters") if isinstance(f.get("parameters"), dict) else {}
        return out
    except Exception:
        return None


def tool_args_check(calls, schemas):
    """(tool_args_valid, error) for a response's tool calls [{"name", "arguments": str|dict}].
    None when there are no calls. A call is valid when its arguments parse as a JSON OBJECT (an empty string
    counts as {}), and, when the request declared tools, it names one of them and its arguments satisfy that
    tool's parameters schema. error is the FIRST problem, short ('json: ...', 'unknown_tool: x', 'schema: ...')."""
    if not calls:
        return None, None
    for c in calls:
        name, a = c.get("name") or "", c.get("arguments")
        if isinstance(a, dict):
            args = a
        else:
            a = (a or "").strip()
            try:
                args = json.loads(a) if a else {}
            except Exception as e:
                return False, ("json: %s" % e)[:160]
            if not isinstance(args, dict):
                return False, "json: arguments not an object"
        if schemas is not None:
            if name not in schemas:
                return False, ("unknown_tool: %s" % name)[:160]
            e = _schema_error(args, schemas[name])
            if e:
                return False, ("schema: %s: %s" % (name, e))[:160]
    return True, None


def _shape_kw(shape, body, local):
    """_active_set kwargs for a finished stream's _SSEShape (L94 + L95 telemetry)."""
    kw = {"reasoning_chars": shape.reasoning_chars, "reasoning_chars_at_output": shape.reasoning_at_output}
    # Same class as L94: _sse_content_shape skips any SSE line split across network chunks, so it can MISS the
    # only content/tool line of a response and report a false empty. The lossless scan corrects false negatives only.
    if shape.content_nonblank:
        kw["content_empty"] = False
    if shape.tools:
        kw["has_tool_calls"] = True
    if local and shape.finish is not None:
        kw["finish_reason"] = shape.finish
    if shape.tools:
        kw["tool_calls_n"] = len(shape.tools)
        if shape.overflow:
            kw["tool_args_valid"], kw["tool_args_error"] = None, "unchecked: arguments over %d chars" % TOOL_ARGS_MAX_CHARS
        else:
            kw["tool_args_valid"], kw["tool_args_error"] = tool_args_check(shape.calls(), _request_tool_schemas(body))
    return kw


def _nonstream_shape_kw(data, body, local):
    """Same fields for a buffered (non-streaming) chat completion body."""
    try:
        d = json.loads(data)
        ch = (d.get("choices") or [{}])[0]
        m = ch.get("message") or ({"content": ch.get("text")} if "text" in ch else {})   # legacy completions
        kw = {}
        r = m.get("reasoning") or m.get("reasoning_content") or ""
        kw["reasoning_chars"] = len(r)
        if local:
            kw["finish_reason"] = ch.get("finish_reason")
            kw["content_len"] = len(m.get("content") or "")
            kw["content_empty"] = not (m.get("content") or "").strip()
            kw["has_tool_calls"] = bool(m.get("tool_calls"))
        tcs = m.get("tool_calls") or []
        if tcs:
            calls = [{"name": (t.get("function") or {}).get("name") or "",
                      "arguments": (t.get("function") or {}).get("arguments")} for t in tcs]
            kw["tool_calls_n"] = len(calls)
            kw["tool_args_valid"], kw["tool_args_error"] = tool_args_check(calls, _request_tool_schemas(body))
        return kw
    except Exception:
        return {}


async def _prepare_stream_response(resp, request, initial, session):
    """A disconnected caller must not strand the upstream ClientSession."""
    try:
        try:
            request["gw_stream_prepared"] = True     # lane SH: _router_crash_response must not answer twice
        except Exception:
            pass
        await resp.prepare(request)
        if initial:
            await resp.write(initial)
        return True
    except (ConnectionResetError, BrokenPipeError):
        _active_set(request, client_disconnected=True)
        await session.close()
        return False
    except BaseException:
        await session.close()
        raise


async def _finish_stream_response(resp, request, session):
    """Close upstream even when write_eof sees a closed client transport."""
    try:
        await resp.write_eof()
    except (ConnectionResetError, BrokenPipeError):
        _active_set(request, client_disconnected=True)
    finally:
        await session.close()


# RS (2026-10-02): one durable record per GATEWAY-layer timeout (first response header, first token, stream idle). Without it a
# timeout only showed up as a generic 'local-failed' 503 / failover, with no deadline, no wait, no size - so nobody could say
# which layer gave up first or how much prefill the abort threw away. Same incidents/ dir as drains.jsonl; read by tools/timeout_audit.py.
_TIMEOUT_LEDGER = os.path.expanduser(os.environ.get("GATEWAY_TIMEOUT_LEDGER", "~/.local/share/vllm-qwen27b/incidents/timeouts.jsonl"))


def _timeout_note(request, layer, deadline_s, base, body, **extra):
    try:
        a = _ACTIVE.get(id(request)) or {}
        t0 = a.get("t0")
        row = {"t": round(time.time(), 3), "layer": layer, "deadline_s": round(float(deadline_s), 1),
               "waited_s": round(time.time() - t0, 1) if t0 else None, "base": "local" if base == LOCAL else "remote",
               "ptok_est": _est_tokens(body), "client": a.get("name") or _client_label(request), "bg": bool(a.get("bg")),
               "tiny": bool(a.get("tiny")), "stream": bool(a.get("stream")), "route": a.get("route"), "phase": a.get("phase")}
        row.update(extra)
        os.makedirs(os.path.dirname(_TIMEOUT_LEDGER), exist_ok=True)
        with open(_TIMEOUT_LEDGER, "a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    except Exception as _e:       # bookkeeping must never touch the request path
        _swallowed("_timeout_note", _e)


class _ClientGone(Exception):
    pass


def _client_gone(request):
    """Has the caller hung up? aiohttp does NOT cancel a handler when the client disconnects (handler_cancellation is
    off), so before CF a queued or prefilling request of a caller that had already timed out was still admitted, still
    prefilled by the engine, and only discovered dead at the first write -- the engine's work thrown away."""
    tr = getattr(request, "transport", "absent")
    if tr == "absent":
        return False                      # a test double with no connection
    return tr is None or tr.is_closing()


async def _await_unless_gone(request, awaitable, timeout=None, poll=0.5):
    """Await `awaitable` (an upstream call) but give up -- and cancel it, which closes the upstream connection and so
    aborts the engine's prefill -- as soon as the caller hangs up. Same TimeoutError contract as asyncio.wait_for."""
    task = asyncio.ensure_future(awaitable)
    t_end = None if timeout is None else time.time() + timeout
    try:
        while True:
            wait = poll if t_end is None else max(0.01, min(poll, t_end - time.time()))
            done, _ = await asyncio.wait({task}, timeout=wait)
            if done:
                return task.result()
            if _client_gone(request):
                raise _ClientGone()
            if t_end is not None and time.time() >= t_end:
                raise asyncio.TimeoutError()
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except BaseException:       # noqa: BLE001 -- the cancelled upstream call
                pass


def _gone_response(request):
    _active_set(request, client_disconnected=True, flow_held="client-gone")
    _FLOW_STATS["client_gone"] += 1
    return web.Response(status=499, text="client closed request")


async def _relay(request, base, path, body, key, streaming, concurrency=1, provider_name=None):
    """
    Forward to (base) and relay the response to the client.
    Returns ("ok", web.Response|StreamResponse) on success,
            ("fail", (status, text, is_oom)) if the upstream failed BEFORE the client was
            committed (so the caller may fail over) — including a streaming upstream that
            dies during prefill before producing any token (the FIRST-TOKEN GATE).
    """
    t_relay_start = time.time()   # TELEMETRY: TTFT reference point -- see DESIGN.md (c)
    session = aiohttp.ClientSession()

    def _route_receipt():
        info = _ACTIVE.get(id(request)) or {}
        route = "local" if base.rstrip("/") == LOCAL.rstrip("/") else "remote"
        provider = (_local_model_name() if route == "local" else (provider_name or REMOTE_MODEL))
        return {
            "X-Gateway-Route": route,
            "X-Gateway-Provider": str(provider),
            "X-Gateway-Reason": str(info.get("reason") or "admitted"),
            "X-Gateway-Queue-Wait": str(round(info.get("waited") or 0.0, 3)),
            "X-Gateway-Work-Class": str(info.get("flow_class") or ""),
            "X-Gateway-Expected-Wait": str(info.get("flow_expected_wait_s") if info.get("flow_expected_wait_s") is not None else ""),
            "X-Gateway-Predicted-Occupancy": str(info.get("predicted_occupancy_s") or ""),
            "X-Gateway-Context-Provider": str(info.get("context_provider") or provider),
            "X-Gateway-Context-Limit": str(info.get("context_limit") or ""),
            "X-Gateway-Context-Prompt": str(info.get("context_prompt_tokens") or ""),
            "X-Gateway-Context-Compacted": "1" if info.get("context_compacted") else "0",
            "X-Gateway-Context-Omitted": str(info.get("context_omitted") or "0"),
        }
    try:
        if streaming:
            # bound time-to-response-headers for streaming so a backend that accepts the
            # connection but never responds (a wedge) fails over instead of hanging.
            up = await _await_unless_gone(request, _open(session, base, path, body, key, streaming),
                                          timeout=first_token_timeout(body, concurrency, local=(base == LOCAL)))
        else:
            up = await _await_unless_gone(request, _open(session, base, path, body, key, streaming))
    except _ClientGone:
        await session.close()
        return "ok", _gone_response(request)
    except asyncio.TimeoutError:
        await session.close()
        _timeout_note(request, "gateway:response-headers", first_token_timeout(body, concurrency, local=(base == LOCAL)), base, body,
                      concurrency=concurrency, outcome="failover")
        # slow/wedged under load -> fail over, but do NOT flag as OOM (no 120s budget backoff:
        # local is busy, not crashed; a real crash returns 5xx/EngineDead below and DOES backoff).
        return "fail", (0, "no response headers within first-token deadline (busy/wedged)", False)
    except Exception as e:
        await session.close()
        return "fail", (0, f"connect error: {e}", False)

    if up.status >= 500:
        text = ""
        try:
            text = (await up.read()).decode("utf-8", "replace")[:500]
        except Exception:
            pass
        await up.release(); await session.close()
        return "fail", (up.status, text, _is_oom(up.status, text))

    # streaming 4xx: the response is an error body, not a real event stream — log it
    # and return it directly instead of entering the first-token gate (which would hang
    # waiting for SSE content that will never come)
    if streaming and 400 <= up.status < 500:
        try:
            data = await up.read()
            log.warning("upstream %s returned streaming %d: %s", base, up.status,
                        data[:500].decode("utf-8", "replace"))
            _note_remote_status(base, up.status)
            ct = up.headers.get("Content-Type", "application/json").split(";")[0]
            await session.close()
            resp = web.Response(body=data, status=up.status, content_type=ct,
                                headers=_route_receipt())
            return "ok", resp
        except Exception as e:
            await session.close()
            return "fail", (up.status, f"streaming 4xx read error: {e}", False)

    if not streaming:
        try:
            data = await up.read()
            if up.status >= 400:
                # capture WHY an upstream 4xx happened (e.g. DeepSeek 400 on overflow -> the
                # "model provider failed after retries" the client shows). 5xx already handled above.
                log.warning("upstream %s returned %d: %s", base, up.status,
                            data[:400].decode("utf-8", "replace"))
            _note_remote_status(base, up.status)
            if up.status < 400:     # [GW2 / L94] finish_reason + tool-arg validity for buffered responses
                _skw = _nonstream_shape_kw(data, body, local=(base.rstrip("/") == LOCAL.rstrip("/")))
                if _skw:
                    _active_set(request, **_skw)
            ct = up.headers.get("Content-Type", "application/json").split(";")[0]
            return "ok", web.Response(body=data, status=up.status, content_type=ct,
                                       headers=_route_receipt())
        finally:
            await session.close()

    # streaming with FIRST-TOKEN GATE: buffer upstream chunks WITHOUT committing to the
    # client until real generation has started. If the upstream dies before that (e.g. an
    # OOM crash during prefill), nothing reached the client yet -> return "fail" so the
    # caller fails the whole request over to remote seamlessly (no error surfaced).
    def _mk_resp():
        return web.StreamResponse(status=up.status, headers={
            "Content-Type": up.headers.get("Content-Type", "text/event-stream"),
            "Cache-Control": "no-cache", "Connection": "keep-alive",
            **_route_receipt()})

    buf = bytearray()
    # TELEMETRY: closure-shared counters (not new params/return values -- see DESIGN.md (c) for
    # why this is the safe way to surface these out of _relay without touching its contract).
    _chunk_ct = [0]
    _t_first = [None]
    _exact_outtok = [None]
    # TELEMETRY (shim-remote-observability lane, 2026-09-05): best-effort finish_reason capture
    # for streaming responses, same scan as _exact_outtok below. _is_remote_relay gates only
    # whether it's SURFACED into telemetry (see the two _active_set(**_outkw) call sites further
    # down) -- local's streaming behavior/telemetry is deliberately left exactly as it was; this
    # lane's mandate is remote-path observability (see REPORT.md). content_len/content_empty/
    # has_tool_calls are NOT captured for streaming at all: a streamed response's content arrives
    # as per-token deltas that are never reassembled anywhere in this relay (it only counts/
    # forwards chunks), so recovering a full message here would need materially more surgery
    # than this lane's scope -- documented limitation, see REPORT.md.
    _exact_finish = [None]
    # gw-admission-computed-token-cost: prompt_tokens + prefix-cache hit, from the SAME usage
    # trailer _exact_outtok already scans (live-measured: 96% of streaming rows already carry
    # it -- pi/OpenCode/OpenHands request stream_options.include_usage today, no shim-side
    # injection needed). cached_tokens is only ever present (per vLLM's chat_completion/
    # serving.py) when it is truthy, i.e. a genuine cache hit -- a miss or a non-hit prompt
    # leaves this None, which computed_actual below must NOT treat as "0 cached" by accident.
    _exact_ptok = [None]
    _exact_cached = [None]
    _exact_hit, _exact_miss = [None], [None]     # R2 v5: provider cache accounting (remote)
    _is_remote_relay = (base.rstrip("/") != LOCAL.rstrip("/"))

    def _scan_usage(data):
        """Best-effort exact completion-token count from stream_options.include_usage's final
        SSE event, scanned out of raw (possibly multi-event) bytes. LIVE-TESTED FINDING: for a
        short response the whole SSE stream -- including the usage trailer -- often arrives
        within the FIRST-TOKEN GATE's own buffer (aiohttp's iter_any() coalesces multiple SSE
        events that land close together into one physical chunk, and _looks_meaningful() commits
        as soon as it sees finish_reason, which can be the very chunk the usage event rode in on)
        -- so this must run against the gate's accumulated `buf`, not only against later
        post-commit chunks. Never raises; leaves _exact_outtok[0]/_exact_finish[0] untouched on
        any failure.

        TELEMETRY (shim-remote-observability lane): also opportunistically captures
        finish_reason the same way, into _exact_finish[0] -- cheap (same line-split, same
        data: prefix check), and _looks_meaningful() above already establishes that a
        '\"finish_reason\":\"' substring check on this exact buffer is a safe, proven pattern."""
        if (_exact_outtok[0] is not None and _exact_finish[0] is not None) or (
                b'"usage"' not in data and b'"finish_reason"' not in data):
            return
        try:
            for ln in bytes(data).split(b"\n"):
                ln = ln.strip()
                if not ln.startswith(b"data:"):
                    continue
                raw = ln[5:].strip()
                if raw in (b"", b"[DONE]"):
                    continue
                if _exact_outtok[0] is None and b'"usage"' in ln:
                    uu = (json.loads(raw).get("usage") or {})
                    if "completion_tokens" in uu:
                        _exact_outtok[0] = int(uu["completion_tokens"])
                    if "prompt_tokens" in uu:
                        _exact_ptok[0] = int(uu["prompt_tokens"])
                    ptd = uu.get("prompt_tokens_details")
                    if ptd and "cached_tokens" in ptd:
                        _exact_cached[0] = int(ptd["cached_tokens"])
                    hit, miss = _usage_cache_split(uu)
                    if hit is not None:
                        _exact_hit[0], _exact_miss[0] = hit, miss
                if _exact_finish[0] is None and b'"finish_reason"' in ln:
                    ch = (json.loads(raw).get("choices") or [{}])[0]
                    fr = ch.get("finish_reason")
                    if fr is not None:
                        _exact_finish[0] = fr
                if _exact_outtok[0] is not None and _exact_finish[0] is not None:
                    break
        except Exception:
            pass

    # TELEMETRY (gw-streaming-content-classifier, 2026-09-11): content_empty/has_tool_calls
    # for STREAMING responses. Kevin's coding runner (pi/OpenCode/OpenHands) streams 100% of
    # its traffic, so the empty-content-on-200 case reproduced live in the gateway black-box
    # audit (forced thinking + a max_tokens too small for the reasoning trace to finish, which
    # leaves `content` empty while `reasoning_content` is populated) was structurally invisible
    # for exactly the client it hurts -- classify_remote_response() only ever ran on the
    # non-streaming path. _sse_content_shape() (module-level, unit-tested independently of this
    # closure -- see test_streaming_content_classifier.py) does NOT reassemble the full message
    # (the 2026-09-05 comment above is right that doing so would be real surgery); it keeps only
    # two cheap running facts across every delta: whether ANY non-whitespace content character
    # has been seen, and whether ANY tool_call has been seen -- exactly what content_empty/
    # has_tool_calls reduce to.
    _saw_content = [False]
    _saw_tool_call = [False]
    _scan_tail = [b""]     # SL lane: the unterminated last SSE line, carried into the next chunk
    _shape = _SSEShape()   # [GW2 / L94+L95] lossless finish_reason / tool-call args / reasoning counter

    def _scan_usage_stream(chunk):
        """_scan_usage for a chunk stream: a usage trailer that straddles a network chunk
        boundary used to be missed (the call was then settled without its real usage)."""
        complete, _scan_tail[0] = _sse_split_tail(_scan_tail[0], chunk)
        if complete:
            _scan_usage(complete)

    def _scan_content_shape(data):
        _saw_content[0], _saw_tool_call[0] = _sse_content_shape(data, _saw_content[0], _saw_tool_call[0])

    async def _read_until_commit():
        async for chunk in up.content.iter_any():
            buf.extend(chunk)
            _chunk_ct[0] += 1
            if _t_first[0] is None:
                _t_first[0] = time.time()
            if (_looks_meaningful(buf.decode("utf-8", "replace"))
                    or len(buf) >= FIRST_GATE_MAX_BYTES or _chunk_ct[0] >= FIRST_GATE_MAX_CHUNKS):
                return "commit"
        return "clean_end"

    # Adaptive first-token deadline: fail over from a wedged backend without stalling, while
    # allowing a legit big-context prefill the time it actually needs (scaled by concurrency).
    deadline = first_token_timeout(body, concurrency, local=(base == LOCAL))
    try:
        phase = await _await_unless_gone(request, _read_until_commit(), timeout=deadline)
    except _ClientGone:
        await session.close()
        return "ok", _gone_response(request)
    except asyncio.TimeoutError:
        await session.close()
        _timeout_note(request, "gateway:first-token", deadline, base, body, concurrency=concurrency, outcome="failover",
                      bytes_before=len(buf))
        return "fail", (up.status, f"no first token within {deadline:.0f}s (busy/wedged)", False)
    except Exception as e:
        await session.close()
        return "fail", (up.status, f"pre-commit stream error: {e}", True)

    # TELEMETRY: time to first streamed byte after the upstream POST was issued (excludes the
    # shim's own admission/queue wait, which is tracked separately as `waited`) -- see DESIGN.md
    # (c) for why this is the right spot and why non-streaming has no equivalent insertion point.
    if _t_first[0] is not None:
        _active_set(request, ttft=round(_t_first[0] - t_relay_start, 3))
        if not _is_remote_relay:
            _pm_prefill_done(request)
    _scan_usage(buf)   # TELEMETRY: covers the (common, for short responses) case where the whole
                        # stream -- usage trailer included -- already arrived within the gate
    _scan_content_shape(buf)
    _scan_tail[0] = bytes(buf[buf.rfind(b"\n") + 1:])[-(1 << 20):]   # partial last line -> next chunk
    _shape.feed(buf)

    if phase == "clean_end":
        # upstream finished before any 'meaningful' chunk — deliver as-is (short/empty response),
        # don't fail over (avoids double-generating a real-but-tiny answer).
        resp = _mk_resp()
        if not await _prepare_stream_response(resp, request, bytes(buf), session):
            return "ok", resp
        await _finish_stream_response(resp, request, session)
        _outkw = {"outtok_lb": _chunk_ct[0],
                  "content_empty": not _saw_content[0], "has_tool_calls": _saw_tool_call[0]}
        if _exact_outtok[0] is not None:
            _outkw["outtok"] = _exact_outtok[0]
        if _exact_finish[0] is not None:      # [GW2 / L94] local rows too (was remote-only)
            _outkw["finish_reason"] = _exact_finish[0]
        _shape.close()
        _outkw.update(_shape_kw(_shape, body, local=not _is_remote_relay))
        # gw-admission-computed-token-cost: local only -- prefix caching is a local-engine
        # concept, and remote's own cache accounting (if any) isn't comparable to it.
        if not _is_remote_relay and _exact_ptok[0] is not None:
            _outkw["computed_actual"] = max(0, _exact_ptok[0] - (_exact_cached[0] or 0))
            _outkw["cached_actual"] = int(_exact_cached[0] or 0)
            _outkw["ptok_exact_local"] = _exact_ptok[0]
        if _is_remote_relay:
            _outkw.update(_remote_usage_kw(_exact_ptok[0], _exact_hit[0], _exact_miss[0]))
        _active_set(request, **_outkw)
        for k, v in _route_receipt().items():
            resp.headers[k] = v
        return "ok", resp

    # committed to the client: flush buffered first chunk(s), then stream the rest. A mid-stream
    # error past this point can only be reported inline (client already receiving output).
    resp = _mk_resp()
    if not await _prepare_stream_response(resp, request, bytes(buf), session):
        return "ok", resp
    # [GW2 / L95] long-empty-generation watchdog: armed for a LOCAL stream whose request still has thinking on.
    _rw = {"armed": (not _is_remote_relay and REASONING_WATCHDOG in ("shadow", "retry")
                     and REASONING_BUDGET_TOKENS > 0 and not _thinking_disabled(body)),
           "retried": False}

    async def _reasoning_retry():
        """Abort the thinking-only generation (closing the connection makes vLLM abort the request and free its
        lane) and stream the same request with thinking OFF into the already-open client stream."""
        try:
            up.close()
        except Exception:
            pass
        _scan_tail[0] = b""
        _shape.tail = b""
        s2 = aiohttp.ClientSession()
        try:
            up2 = await _open(s2, base, path, strip_thinking(body), key, True)
            if up2.status >= 400:
                txt = (await up2.read())[:300].decode("utf-8", "replace")
                await resp.write(('data: {"error":{"message":%s}}\n\n' % json.dumps(
                    "reasoning watchdog retry failed: %s %s" % (up2.status, txt))).encode())
                return False
            while True:
                ch2 = await asyncio.wait_for(up2.content.readany(), timeout=STREAM_IDLE_TIMEOUT_SECS)
                if not ch2:
                    break
                await resp.write(ch2)
                _chunk_ct[0] += 1
                _scan_usage_stream(ch2)
                _scan_content_shape(ch2)
                _shape.feed(ch2)
            return True
        except Exception as e:
            try:
                await resp.write(('data: {"error":{"message":%s}}\n\n' % json.dumps(
                    "reasoning watchdog retry interrupted: %s" % e)).encode())
            except Exception:
                pass
            return False
        finally:
            await s2.close()

    # TELEMETRY: exact output-token count when the client asked for stream_options.include_usage
    # and the usage trailer didn't already arrive within the gate phase above (long streams) --
    # same cheap per-chunk scan, applied to whatever's left of the stream.
    try:
        while True:
            try:
                # A connected stream that stops producing bytes is a stuck generation, not a
                # healthy long run.  Abort it after the bounded idle window; the client already
                # received output, so migration is unsafe, but the lane is released and the
                # event is recorded for the next admission decision.
                chunk = await asyncio.wait_for(up.content.readany(), timeout=STREAM_IDLE_TIMEOUT_SECS)
            except asyncio.TimeoutError:
                _active_set(request, stream_watchdog=True, stream_idle_timeout_s=STREAM_IDLE_TIMEOUT_SECS,
                            reason="stream-idle-timeout")
                _timeout_note(request, "gateway:stream-idle", STREAM_IDLE_TIMEOUT_SECS, base, body, outcome="aborted-after-output",
                              out_chunks=_chunk_ct[0])
                log.warning("stream idle for %.1fs -> aborting %s request", STREAM_IDLE_TIMEOUT_SECS,
                            "remote" if _is_remote_relay else "local")
                try:
                    await up.release()
                except Exception:
                    pass
                try:
                    await resp.write(b'data: {"error":{"message":"stream idle timeout"}}\n\n')
                except Exception:
                    pass
                break
            if not chunk:
                break
            await resp.write(chunk)
            _chunk_ct[0] += 1
            _scan_usage_stream(chunk)
            _scan_content_shape(chunk)
            _shape.feed(chunk)
            if _rw["armed"] and not _shape.saw_output and _shape.reasoning_tokens_est() >= REASONING_BUDGET_TOKENS:
                _rw["armed"] = False
                act = _reasoning_watchdog_note(request, base, body, _shape, REASONING_WATCHDOG)
                if act == "retry":
                    _rw["retried"] = True
                    _active_set(request, reasoning_watchdog_ok=await _reasoning_retry())
                    break
    except Exception as e:
        try:
            await resp.write(f'data: {{"error":{{"message":"stream interrupted: {e}"}}}}\n\n'.encode())
        except Exception:
            pass
    await _finish_stream_response(resp, request, session)
    if _scan_tail[0]:
        _scan_usage(_scan_tail[0])       # a final line the stream ended without a newline on
    _outkw = {"outtok_lb": _chunk_ct[0],
              "content_empty": not _saw_content[0], "has_tool_calls": _saw_tool_call[0]}
    if _exact_outtok[0] is not None:
        _outkw["outtok"] = _exact_outtok[0]
    if _exact_finish[0] is not None:          # [GW2 / L94] local rows too (was remote-only)
        _outkw["finish_reason"] = _exact_finish[0]
    _shape.close()
    _outkw.update(_shape_kw(_shape, body, local=not _is_remote_relay))
    # LS lane 2026-10-01: this committed-stream path (every response longer than the first-token
    # gate -- i.e. nearly all real traffic) never recorded computed_actual, so the cost model's
    # predicted-vs-actual gate had no data (0 of 40,000 local rows). Same local-only rule as above.
    if not _is_remote_relay and _exact_ptok[0] is not None:
        _outkw["computed_actual"] = max(0, _exact_ptok[0] - (_exact_cached[0] or 0))
        _outkw["cached_actual"] = int(_exact_cached[0] or 0)
        _outkw["ptok_exact_local"] = _exact_ptok[0]
    if _is_remote_relay:
        _outkw.update(_remote_usage_kw(_exact_ptok[0], _exact_hit[0], _exact_miss[0]))
    _active_set(request, **_outkw)
    return "ok", resp


_RW_STATS = collections.Counter()      # [GW2 / L95] reasoning-watchdog triggers since start, by action


def _thinking_disabled(body):
    try:
        j = json.loads(body)
        return (j.get("chat_template_kwargs") or {}).get("enable_thinking") is False
    except Exception:
        return False


def _reasoning_watchdog_note(request, base, body, shape, mode):
    """Record one watchdog trigger (shadow or real) and return the action to take: 'retry' or 'shadow'."""
    act = "retry" if mode == "retry" else "shadow"
    est = shape.reasoning_tokens_est()
    _RW_STATS[act] += 1
    _active_set(request, reasoning_watchdog=act, reasoning_watchdog_at_tok=est)
    _timeout_note(request, "gateway:reasoning-budget", REASONING_BUDGET_TOKENS, base, body,
                  outcome=("aborted-retry-no-think" if act == "retry" else "shadow-would-retry"),
                  reasoning_chars=shape.reasoning_chars, reasoning_tok_est=est)
    log.warning("reasoning watchdog (%s): %d est reasoning tokens, no content/tool call yet -> %s", mode, est,
                "abort + retry no-think" if act == "retry" else "would retry (shadow)")
    return act


def _sse_split_tail(tail, chunk, cap=1 << 20):
    """(complete lines, unterminated tail) for a byte stream arriving in arbitrary chunks: the
    tail is prepended to the next chunk, so an SSE line (the usage trailer above all) that a
    network read split in two is still seen whole. The tail is bounded by `cap`."""
    data = bytes(tail) + bytes(chunk)
    nl = data.rfind(b"\n")
    if nl < 0:
        return b"", data[-cap:]
    return data[:nl + 1], data[nl + 1:][-cap:]


def _usage_cache_split(usage):
    """(cache_hit, cache_miss) prompt tokens from a provider usage object, or (None, None).
    DeepSeek reports prompt_cache_hit_tokens / prompt_cache_miss_tokens (verified on a live
    response 2026-09-25); OpenAI-compatible providers report prompt_tokens_details.cached_tokens."""
    try:
        if "prompt_cache_hit_tokens" in usage and "prompt_cache_miss_tokens" in usage:
            return int(usage["prompt_cache_hit_tokens"]), int(usage["prompt_cache_miss_tokens"])
        ptd = usage.get("prompt_tokens_details") or {}
        if "cached_tokens" in ptd and "prompt_tokens" in usage:
            hit = int(ptd["cached_tokens"])
            return hit, max(0, int(usage["prompt_tokens"]) - hit)
    except Exception:
        pass
    return None, None


def _remote_usage_kw(ptok_exact, hit, miss):
    kw = {}
    if ptok_exact is not None:
        kw["ptok_exact"] = ptok_exact
    if hit is not None and miss is not None:
        kw["remote_cache_hit"], kw["remote_cache_miss"] = hit, miss
    return kw


def classify_remote_response(payload):
    """Pure helper -- no I/O, no aiohttp, no event loop. Given an already-JSON-decoded
    /v1/chat/completions (or /v1/completions) response body, extract the response-shape facts
    the LOCAL path has effectively had since _is_empty_thinking_response() (finish_reason,
    content presence, tool_calls presence) plus the two exact token counts from `usage` --
    none of which _forward_remote() captured before this lane.

    Added by the shim-remote-observability lane (2026-09-05); see
    lanes/dreams-empty-turns/REPORT.md (b)/(c) fix #5 and its 'Gap' note: the 3 fatal Dreams
    turns that night ran entirely on the remote (DeepSeek) overflow path and left no record
    anywhere -- not telemetry, not the flight recorder -- of what the model actually returned.

    Never raises. Always returns every key; on anything unparseable every value stays at its
    safe default (content_empty=True, has_tool_calls=False, the rest None/0) instead of
    raising, so callers never need their own try/except around this. Kept as a standalone pure
    function of a plain dict (not the raw aiohttp response/payload object) specifically so it
    can be unit-tested without aiohttp, asyncio, or a running shim -- see LANE/tests/."""
    out = {"finish_reason": None, "content_len": 0, "content_empty": True,
           "has_tool_calls": False, "prompt_tokens": None, "completion_tokens": None}
    if not isinstance(payload, dict):
        return out
    try:
        ch = (payload.get("choices") or [{}])[0]
        out["finish_reason"] = ch.get("finish_reason")
        m = ch.get("message") or {}
        content = m.get("content") or ""
        out["content_len"] = len(content)
        out["content_empty"] = not content.strip()
        out["has_tool_calls"] = bool(m.get("tool_calls"))
    except Exception:
        pass
    try:
        u = payload.get("usage") or {}
        if "prompt_tokens" in u:
            out["prompt_tokens"] = int(u["prompt_tokens"])
        if "completion_tokens" in u:
            out["completion_tokens"] = int(u["completion_tokens"])
    except Exception:
        pass
    return out


def _flightrec_min_tok():
    """SHIM_FLIGHTREC_MIN_TOK, read per call (as before). Lane SH: a bad value used to raise inside the local
    path's try/finally -> every request >= that size died as a bare 500 when the schema module was absent; it
    now means the default (15000) and is counted."""
    raw = os.environ.get("SHIM_FLIGHTREC_MIN_TOK", "15000")
    try:
        return int(raw)
    except (TypeError, ValueError) as e:
        _swallowed("_flightrec_min_tok", e)
        return 15000


def _flightrec_dir():
    """Resolve the flight-recorder directory. Configurable via SHIM_FLIGHTREC_DIR (added by
    the shim-remote-observability lane, 2026-09-05 -- the path was previously hardcoded inline
    at the local-branch call site below, with no way for a sandboxed/lane copy of this shim to
    redirect it away from the production directory). Unset in production, so this returns
    exactly the hardcoded path it replaces -- zero behavior change there."""
    return os.environ.get("SHIM_FLIGHTREC_DIR", "/home/kevin/.local/share/vllm-qwen27b/flightrec")


def _write_remote_flightrec(fr, req_fn, req_body, resp_fn, resp_body):
    """Blocking file I/O for the REMOTE-path flight recorder -- must run off the event loop
    (ASYNC230), exactly like _write_flightrec() below it. Shares that function's directory and
    40-file ring (both local's and remote's recordings compete for the same 40 slots, trimmed
    oldest-first) rather than keeping a separate cap -- see classify_remote_response's callers
    for why remote filenames still lead with the epoch-seconds timestamp (matching local's
    '<ts>_<ptok>tok.json' shape) and only put 'remote' right after it, rather than literally
    prefixing the whole filename with 'remote_': a leading 'remote_' would sort lexically AFTER
    every all-digit local filename regardless of actual age (ASCII 'r' > any digit), so the
    ring-cleanup's plain sorted(os.listdir(...))[:-40] below would always evict local's older
    files first, starving them out of the ring -- a real correctness bug, not a style choice.

    Writes the request body always; the response body only for the empty-response case that
    earns it (resp_fn/resp_body are None otherwise) -- one executor hop, one cleanup pass, for
    both files of a single event. Creates the directory on first use (exist_ok) since, unlike
    the production flightrec path, a lane's SHIM_FLIGHTREC_DIR may not exist yet; the real
    production directory already exists, so this is a no-op there."""
    try:
        os.makedirs(fr, mode=0o700, exist_ok=True)
    except OSError:
        pass
    for fn, data in ((req_fn, req_body), (resp_fn, resp_body)):
        if fn is None or data is None:
            continue
        with open(fn, "wb") as f:
            f.write(data)
        os.chmod(fn, 0o600)
    olds = sorted(os.listdir(fr))
    for o in olds[:-40]:
        os.unlink(os.path.join(fr, o))


async def _note_remote_response(request, payload, body):
    """Remote-path counterpart to the local branch's flightrec block + _is_empty_thinking_
    response()/EMPTY_RETRY (see _route_completions below) -- added by the shim-remote-
    observability lane (2026-09-05) because NEITHER existed for _forward_remote() before: see
    lanes/dreams-empty-turns/REPORT.md (b)/(c) fix #5 and its 'Gap' note. Called only for a
    non-streaming 'ok' remote response (the caller guards streaming out -- see REPORT.md for
    why streaming content can't be reconstructed here). Never raises into the request path --
    mirrors _note_payload_outcome()'s own contract exactly, and reuses that exact same
    telemetry sink (_active_set() -> the live _ACTIVE entry -> _telemetry_note_request()'s
    JSONL record, read back in handle_completions' finally, strictly after this coroutine has
    already returned) rather than inventing a second one.

    UNLIKE the local path's EMPTY_RETRY, an empty remote response is never retried here:
    (1) COST -- a remote retry is a second paid DeepSeek call, whereas the local retry is free
    compute on hardware that's already sitting idle; (2) it would very likely just repeat --
    remap_for_remote() already force-injects 'thinking: {type: disabled}' on every DeepSeek
    request whenever REMOTE_NO_THINK is on (the default), which is the ENTIRE mechanism the
    local retry relies on (stripping thinking) to get a different, non-empty answer. Thinking
    is already off here, so retrying would spend a second paid call to very likely reproduce
    the same empty answer instead of fixing it."""
    try:
        resp_body = getattr(payload, "body", None)
        if not isinstance(resp_body, (bytes, bytearray)):
            return
        try:
            parsed = json.loads(resp_body)
        except Exception:
            return
        cls = classify_remote_response(parsed)
        _active_set(request, finish_reason=cls["finish_reason"], content_len=cls["content_len"],
                    content_empty=cls["content_empty"], has_tool_calls=cls["has_tool_calls"],
                    ptok_exact=cls["prompt_tokens"])
        hit, miss = _usage_cache_split(parsed.get("usage") or {}) if isinstance(parsed, dict) else (None, None)
        if hit is not None:
            _active_set(request, remote_cache_hit=hit, remote_cache_miss=miss)
        empty_no_tool = cls["content_empty"] and not cls["has_tool_calls"]
        if empty_no_tool:
            log.warning("remote returned EMPTY content (finish_reason=%s, completion_tokens=%s)",
                        cls["finish_reason"], cls["completion_tokens"])
        ptok = _est_tokens(body)
        min_tok = _flightrec_min_tok()
        if ptok >= min_tok or empty_no_tool:
            fr = _flightrec_dir()
            stem = f"{int(time.time())}_remote_{ptok}tok"
            req_fn = f"{fr}/{stem}.json"
            resp_fn = f"{fr}/{stem}.resp.json" if empty_no_tool else None
            resp_data = resp_body if empty_no_tool else None
            await asyncio.get_running_loop().run_in_executor(
                None, _write_remote_flightrec, fr, req_fn, body, resp_fn, resp_data)
    except Exception as e:
        log.warning("remote response note: %s", e)


async def _forward_remote(request, path, body, streaming, endpoint=None, model=None):
    """Forward intact or deterministically compacted context to the selected remote endpoint."""
    ep = endpoint or {"base": REMOTE_BASE, "key": REMOTE_KEY, "model": model or REMOTE_MODEL,
                     "context_limit": REMOTE_CONTEXT_LIMIT, "max_output": 65536,
                     "name": "default"}
    base = str(ep.get("base") or "").rstrip("/")
    key = str(ep.get("key") or "")
    model = str(ep.get("model") or REMOTE_MODEL)
    limit = int(ep.get("context_limit") or REMOTE_CONTEXT_LIMIT)
    max_output = int(ep.get("max_output") or 65536)
    if not base or (endpoint is None and not remote_ok()):
        return web.json_response(
            {"error": {"message": "local unavailable and no remote overflow configured"}}, status=503)
    # R2: every paid call -- the default provider on any route (alias, route intent, overflow,
    # size, big-prompt, forced window, local-down) AND every metered custom endpoint -- passes
    # the one spend authority BEFORE anything is sent. v6: a custom endpoint with no declared
    # cost policy, or a default-provider model with no configured price, fails CLOSED.
    policy = _endpoint_cost_policy(endpoint, model)
    if policy is None:
        what = (f"gateway alias {ep.get('name')} declares no cost policy" if endpoint is not None
                else f"model {model} has no configured price (SHIM_REMOTE_PRICES_JSON)")
        return web.json_response({"error": {"message": f"gateway spend authority: {what}; refused",
                                            "type": "cost_policy_missing"}}, status=409)
    _active_set(request, cost_policy=policy["policy"], remote_provider=policy.get("provider"),
                remote_prices=policy.get("prices"))
    if policy["policy"] == "metered":
        refused = _spend_hold_for(request, body, prices=policy.get("prices") if endpoint is not None else None,
                                  model=model)
        if refused is not None:
            return refused
    prepared, ctx = _prepare_provider_context(body, limit, max_output)
    if prepared is None:
        log.warning("remote %s context rejected provider=%s prompt=%s total=%s limit=%s",
                    path, ep.get("name", "default"), ctx.get("prompt_tokens"),
                    ctx.get("total_tokens"), ctx.get("limit"))
        _active_set(request, context_provider=ep.get("name", "default"),
                    context_limit=ctx.get("limit"), context_prompt_tokens=ctx.get("prompt_tokens"),
                    context_compacted=False)
        return _context_error(ctx, ep.get("name", "default"))
    _active_set(request, context_provider=ep.get("name", "default"),
                context_limit=limit, context_prompt_tokens=ctx.get("prompt_tokens"),
                context_compacted=bool(ctx.get("compacted")),
                context_omitted=int(ctx.get("omitted", 0) or 0))
    _sent_at = time.time()
    _active_set(request, remote_model=model, remote_sent_at=_sent_at, remote_price_peak=is_peak())
    relay_body = remap_for_remote(prepared, model, max_output)
    if streaming:
        relay_body = _with_stream_usage(relay_body)   # v6: settle needs the provider's usage
    _active_set(request, remote_sent=True)            # v6: from here the provider may bill
    kind, payload = await _relay(request, base, path, relay_body, key, streaming,
                                 provider_name=model)
    # Settlement needs to know whether the provider produced a completion: a relay that failed
    # before the client was committed, or an HTTP >= 400 answer (402 Insufficient Balance, 4xx,
    # 5xx), is never billed. An ok answer that reaches settlement without usage (hung-up client,
    # cut stream) is billed what it consumed, estimated.
    _status = getattr(payload, "status", None) if kind == "ok" else (payload[0] if payload else None)
    _active_set(request, remote_status=_status,
                remote_outcome="failed" if (kind != "ok" or (isinstance(_status, int) and _status >= 400))
                else "ok")
    if kind == "ok":
        _note_payload_outcome(request, payload, streaming)   # TELEMETRY: see DESIGN.md (c)
        if not streaming:
            # TELEMETRY + FLIGHT RECORDER (shim-remote-observability lane, 2026-09-05): see
            # _note_remote_response()'s docstring. Streaming remote responses only get the
            # finish_reason capture inside _relay() above (best-effort, SSE-scan-based);
            # content-shape telemetry and the flight recorder are non-streaming-only.
            await _note_remote_response(request, payload, prepared)
        return payload
    status, text, _ = payload
    return web.json_response({"error": {"message": f"remote overflow failed: {status} {text}"}}, status=502)


def _remote_provider_id():
    host = re.sub(r"^https?://", "", str(REMOTE_BASE or "")).split("/")[0].lower()
    return os.environ.get("SHIM_REMOTE_PROVIDER") or ("deepseek" if "deepseek" in host else host or "default")


def _price_configured(model):
    """The default provider's prices: the knobs price REMOTE_MODEL; any other model needs an
    explicit SHIM_REMOTE_PRICES_JSON entry."""
    if not model or model == REMOTE_MODEL:
        return True
    try:
        # Deliberately NOT the dated table: serving a non-default model (e.g. the 13x dearer pro
        # model) stays an explicit operator decision -- a SHIM_REMOTE_PRICES_JSON row.
        return str(model) in (json.loads(REMOTE_PRICES_JSON) if REMOTE_PRICES_JSON else {})
    except Exception:
        return False


def _endpoint_cost_policy(endpoint, model=None):
    """{"policy": "free"|"metered", "provider": id, "prices": ($/Mtok hit, miss, out)|None} or
    None when the endpoint's cost is undeclared (-> refuse)."""
    if endpoint is None:
        if not _price_configured(model):
            return None
        return {"policy": "metered", "provider": _remote_provider_id(), "prices": None}
    cost = endpoint.get("cost")
    if not isinstance(cost, dict):
        return None
    if cost.get("policy") == "free":
        return {"policy": "free", "provider": None, "prices": None}
    if cost.get("policy") == "metered":
        return {"policy": "metered", "provider": cost.get("provider"),
                "prices": (float(cost["cache_hit"]), float(cost["cache_miss"]), float(cost["output"]))}
    return None


def _with_stream_usage(relay_body):
    """Ask the provider for its usage trailer on a streamed response (OpenAI-compatible
    stream_options.include_usage) so settlement prices the provider's own counts."""
    try:
        j = json.loads(relay_body)
        so = j.get("stream_options") if isinstance(j.get("stream_options"), dict) else {}
        if so.get("include_usage") is True:
            return relay_body
        j["stream_options"] = {**so, "include_usage": True}
        return json.dumps(j).encode()
    except Exception:
        return relay_body


async def handle_completions(request):
    """Register the request as live work for the dashboard, then run the router (below)."""
    body = await request.read()          # aiohttp caches the body; the router re-reads it for free
    await _warm_token_estimate(body)     # [GW2] the one /tokenize round trip, off the event loop
    # Check after the await and before _ACTIVE registration, with no await in
    # between. The drain endpoint runs on this same event loop: every request
    # accepted before its fence is visible to the deployer, and every later
    # request is refused before either local work or a paid hold can begin.
    if _draining():
        who = _client_label(request)
        reason, since = _DRAIN_REASON, (_DRAIN_REC or {}).get("t0")
        if _DRAIN_REC:
            _DRAIN_REC["refused"] += 1
            _DRAIN_REC["refused_by_client"][who] = _DRAIN_REC["refused_by_client"].get(who, 0) + 1
        # RS: Retry-After no longer overshoots the end of the fence, and the message names the reason + age so a caller's
        # error text (Halo's `last_error`) says WHY, not just "draining".
        wait = max(1, min(30, int(_DRAIN_UNTIL - time.time())))
        msg = ("gateway deployment is draining accepted calls; retry shortly"
               + (f" (reason: {reason}; held {int(time.time() - since)}s; by {_DRAIN_REC['by']})" if since else ""))
        return web.json_response({"error": {"type": "gateway_draining", "message": msg}},
                                 status=503, headers={"Retry-After": str(wait), "X-Gateway-Drain": "active"})
    _drain_reap()
    try:
        j = json.loads(body)
        if not isinstance(j, dict):
            j = {}
    except Exception:
        j = {}
    info = _friendly_client(request)
    try:
        maxtok = int(j.get("max_tokens") or 0)
    except Exception:
        maxtok = 0
    # lane SH: a body the router cannot read is still registered (so telemetry sees its 400), but the classifiers
    # below read an empty object instead of crashing on it.
    _bad = _body_shape_error(body)
    _cb = b"{}" if _bad else body
    info.update({"ep": request.path.rsplit("/", 1)[-1], "model": str(j.get("model") or "")[:40],
                 "ptok": _est_tokens(_cb), "maxtok": maxtok, "stream": wants_stream(_cb),
                 "t0": time.time(), "phase": "routing", "route": None, "reason": None,
                 "preview": _preview(_cb)[:100], "bg": is_background(_cb, request), "tiny": is_tiny(_cb),
                 "spend_key": os.urandom(16).hex(), **_request_identity(request, body)})
    _ACTIVE[id(request)] = info
    _resp = None
    try:
        if _bad:
            _active_set(request, route="rejected", reason="bad-body")
            _resp = web.json_response({"error": {"message": _bad, "type": "invalid_request_error",
                                                 "code": "invalid_body"}}, status=400)
            return _resp
        try:
            _resp = await _route_completions(request)
        except Exception as e:                     # noqa: BLE001 -- lane SH: see _router_crash_response
            _resp = _router_crash_response(request, e)
        return _resp
    finally:
        _info = _ACTIVE.pop(id(request), None)
        if _info is not None:
            _note_unrouted_outcome(_info, _resp)     # lane SH: an early-return refusal still names its outcome
            _spend_settle(_info, _resp)             # R2: price what went remote, release its hold
            _telemetry_note_request(_info, _resp)   # TELEMETRY: per-client rollups + error feed
            _PM_INFLIGHT.pop(id(request), None)     # cost-model side table: never outlive the request


# ---- lane SH (2026-10-03): every request gets the gateway's own answer, and every answer names its outcome ----
# Before: (a) a body that parses as JSON but is not an object (or whose messages are not a list of objects) raised
# inside handle_completions -> aiohttp's bare 500, which never reached telemetry (row status None, not in the error
# feed); (b) any other exception escaping the router did the same; (c) early-return refusals (alias disabled,
# estate-local while local is down, cap exhausted while local is down, no remote for an over-budget reservation,
# context errors, cost-policy refusals) logged route "?" with no reason.
_ROUTER_CRASHES = collections.Counter()      # exception type -> count (bounded: exception classes)


def _body_shape_error(body):
    """The reason a completion body cannot be routed, or None. Only shapes the router itself would crash on are
    refused; a body that is not JSON at all keeps its existing 400 from the body preparation below."""
    try:
        j = json.loads(body)
    except Exception:
        return None
    if not isinstance(j, dict):
        return "request body must be a JSON object, got %s" % type(j).__name__
    msgs = j.get("messages")
    if msgs is not None and not isinstance(msgs, list):
        return "messages must be a list of message objects, got %s" % type(msgs).__name__
    if isinstance(msgs, list) and any(not isinstance(m, dict) for m in msgs):
        return "every item of messages must be a message object"
    return None


def _router_crash_response(request, exc):
    """An unhandled router exception becomes a counted, logged, telemetry-visible JSON 500. A stream that was
    already committed to the client cannot be answered twice: re-raise (aiohttp closes it) after recording."""
    if isinstance(exc, (ConnectionResetError, BrokenPipeError)):
        _active_set(request, client_disconnected=True)   # the caller hung up: nothing to answer, not a gateway bug
        raise exc
    name = type(exc).__name__
    _ROUTER_CRASHES[name] += 1
    log.exception("router crashed (%s) on %s: %s", name, getattr(request, "path", "?"), exc)
    _active_set(request, route="error", reason="gateway-exception:" + name, phase="done")
    try:
        committed = bool(request.get("gw_stream_prepared"))
    except Exception:
        committed = False
    if committed:
        raise exc
    return web.json_response({"error": {"message": "gateway internal error (%s); the request was not served" % name,
                                        "type": "gateway_internal_error"}}, status=500)


def _note_unrouted_outcome(info, resp):
    """A response returned before any route was recorded gets route "rejected" and a reason taken from the error
    it carries (its type or code, else http-<status>), so telemetry and the error feed never show route "?"."""
    if info.get("route") not in (None, "", "?"):
        return
    status = getattr(resp, "status", None)
    reason = None
    try:
        err = json.loads(getattr(resp, "body", b"") or b"{}").get("error")
        if isinstance(err, dict):
            reason = err.get("code") or err.get("type")
    except Exception:
        reason = None
    if resp is None:                                  # cancelled / aborted before any response existed
        info["route"], info["reason"] = "aborted", info.get("reason") or "no-response"
        return
    info["route"] = "rejected" if status is None or status >= 400 else "?"
    info["reason"] = info.get("reason") or reason or "http-%s" % status


def _write_flightrec(fr, fn, body):
    """Blocking file I/O for the flight recorder — must run off the event loop (ASYNC230):
    a multi-100K-token body written synchronously here would stall every OTHER in-flight
    request (including tiny-lane calls expected to return in milliseconds) for the
    duration of the write + the ring-cleanup listdir/unlink calls.

    Creates `fr` on first use (shim-remote-observability lane, 2026-09-05, alongside making
    the directory itself configurable via SHIM_FLIGHTREC_DIR -- see _flightrec_dir()): a
    no-op in production, where this directory already exists, but needed for a lane/sandbox
    copy pointed at a fresh scratch directory."""
    try:
        os.makedirs(fr, mode=0o700, exist_ok=True)
    except OSError:
        pass
    with open(fn, "wb") as f:
        f.write(body)
    os.chmod(fn, 0o600)
    olds = sorted(os.listdir(fr))
    for o in olds[:-40]:
        os.unlink(os.path.join(fr, o))


def _bg_reject_response():
    """BG-LOCAL-ONLY: immediate hold-refusal for a background request caught in a
    local-availability event (engine down, or an operator's SHIM_FORCE_REMOTE maintenance
    window) -- background work can wait for the next cycle, so it is NEVER charged to the
    paid remote provider for this. 503 + Retry-After so a well-behaved batch caller (cron,
    the research feeder, the digester) backs off and retries later instead of hammering the
    gateway or a human mistaking it for a hang."""
    payload = json.dumps({
        "error": "local engine unavailable; background requests are never served remotely",
        "retry_after": BG_REJECT_RETRY_SECS,
    }).encode()
    return web.Response(body=payload, status=503, content_type="application/json",
                         headers={"Retry-After": str(BG_REJECT_RETRY_SECS)})


async def _route_completions(request, _no_overflow=False):
    global _inflight, _waiting, _inflight_tokens, _inflight_reserved_tokens, _inflight_computed
    path = request.path
    body = await request.read()
    client = _friendly_client(request)["name"]
    streaming = wants_stream(body)
    ptok = _est_tokens(body)
    # No configured provider can accept a prompt above this ceiling. Refuse it
    # before cache prediction, routing, queue admission, or paid forwarding. A
    # stale agent session once rebuilt a ~1.9M-token health probe on every retry.
    # The caller must rotate/compact its session; retrying this body is not work.
    provider_ceiling = max(
        int(local_context_limit()), int(REMOTE_CONTEXT_LIMIT),
        *(int(a.get("context_limit") or 0) for a in _ALIASES.values()
          if isinstance(a, dict) and a.get("enabled")),
    )
    if ptok > provider_ceiling:
        _active_set(request, est_tokens=ptok, route="rejected", reason="prompt-exceeds-all-providers")
        return web.json_response({"error": {"message":
            f"prompt estimate {ptok} exceeds every configured provider context ({provider_ceiling}); "
            "rotate or compact the client session before retrying",
            "type": "context_length_exceeded", "code": "prompt_exceeds_all_providers"}}, status=413)
    try:
        maxtok = int(json.loads(body).get("max_tokens") or 0)
    except Exception:
        maxtok = 0
    ev = dict(ptok=ptok, maxtok=maxtok, stream=streaming)
    tiny = is_tiny(body)
    # K4 micro lane v2: a call type that has repeatedly answered in <= MICRO_MAX_OUT tokens gets the tiny lane even when
    # its max_tokens is unset/large. It never takes the paid tiny-fast overflow (see the tiny branch below).
    tiny_learned = False
    if not tiny and MICRO_LEARN:
        tiny_learned = micro_predict(client, _preview(body), ptok)
        if tiny_learned:
            tiny = True
            _active_set(request, tiny=True, micro="learned")
    background = is_background(body, request)
    halo_control = _halo_control_request(request, body)
    # Lane DB2: record the work class on the live request BEFORE any routing decision, so every route -- remote, alias,
    # vision, forced, local-offline, failover, rejected -- reaches record_event()/flow_note_route() with a real class
    # (it used to be set only on the local admission path: 407 of 410 remote requests showed class "?").
    try:
        _active_set(request, flow_class=flow_class_of(request, body, background, halo_control))
    except Exception:
        pass
    # gw-admission-computed-token-cost: predicted UNCONDITIONALLY (not gated on
    # USE_COMPUTED_COST, which only decides whether admission COST uses this number) so the
    # card's own accuracy gate has real predicted-vs-actual data to grade from the moment this
    # deploys, before that switch is ever flipped on. Observe must run AFTER predict, against
    # the prefix state predict just read -- a miss this turn still becomes next turn's hit.
    # Key the model on the body the ENGINE will actually see (think-guard / no-think policy /
    # alias rewrite change the rendered prefix), not the body the client sent.
    try:
        _pm_body = _prepare_local_body(request, body, background)
    except Exception:
        _pm_body = body          # malformed request: it is rejected further down; just don't crash here
    _pm = _pm_predict(_pm_body, ptok)
    if CREDIT_ANCHOR != "off" or CREDIT_PROBE != "off":
        try:
            await _credit_sources(request, _pm, _pm_body, ptok)     # [GW2 / L105] engine-reported credit
        except Exception:
            pass
    est_computed = _pm["computed"]
    units = estimate_units(body, client=client, computed=est_computed)
    # The chain holds raw digests (not JSON-serialisable), so it lives in a side table keyed by the
    # request and is released by _telemetry_note_request(); _ACTIVE only carries scalars.
    _PM_INFLIGHT[id(request)] = _pm
    _active_set(request, est_tokens=ptok, est_computed=est_computed, pm_credit=_pm["credit"],
                pm_age_s=None if _pm["age"] is None else round(_pm["age"], 1), pm_ref=id(request))
    if CHAIN_TELEMETRY or WARM_PRIORITY:
        try:
            _cr_route, _cr_age, _ = _chain_route_lookup(_pm["chain"])
            _active_set(request, chain_prev_route=_cr_route,
                        chain_prev_age_s=None if _cr_age is None else round(_cr_age, 1), pm_match_tok=_pm.get("matched"))
        except Exception:
            pass
    _cr_warm = warm_continuation(_pm)
    if _cr_warm:
        request["cr_warm_priority"] = True
        _active_set(request, warm_priority=True)
    _prefix_cache_observe(client, body, ptok)
    # What the routing guards below treat as this request's "size": the predicted UNCACHED prefill
    # when the cache-aware cost model is on, else the raw prompt (previous behaviour, byte for byte).
    _cost_tokens = est_computed if USE_COMPUTED_COST else ptok
    alias = _alias_for_request(body)
    alias_kind = alias.get("kind")
    alias_local_only = alias_kind == "builtin-local"
    alias_force_remote = alias_kind == "builtin-remote"
    custom_endpoint = alias.get("endpoint") if alias_kind == "custom-remote" else None
    _active_set(request, alias=alias.get("name"), alias_kind=alias_kind)
    # K4 PROBE synth: a trivial "Reply with the single word: X" liveness probe is answered here while the engine has
    # demonstrably produced tokens for real traffic in the last PROBE_FRESH_S (see probe_synth_decision).
    if PROBE_SYNTH and not _no_overflow:
        _pw = probe_word(body)
        if _pw:
            _ok, _why = probe_synth_decision(_pw, alias_kind, time.time() - _LAST_LOCAL_OK, bool(_health["ok"]),
                                             bool(_local_offline() or (remote_ok() and effective_force_remote())))
            if _ok:
                _MICRO_STATS["probe_synth"] += 1
                _active_set(request, route="synth", reason="probe", phase="done", synth=True)
                log.info("route %s PROBE %r answered by the gateway (engine fresh)", path, _pw)
                return probe_synth_response(body, _pw, streaming)
            log.info("route %s PROBE %r goes through for real (%s)", path, _pw, _why)
    # R2 v5 THROUGHPUT GUARD (Kevin, 2026-09-25): the daily remote cap is hard, but an OVERFLOW
    # -- a remote trip the GATEWAY chose (size, big-prompt, big-out, local-down, monster, perf,
    # predicted, tiny-fast, admission timeout) -- must never die of it. When the cap has no room
    # for this request (or the ledger is unusable), every overflow branch below sees
    # overflow_ok=False and the request is served LOCALLY instead: it queues for a lane (the
    # engine serves 524K context; latency is acceptable). Only explicit remote -- the
    # estate-remote alias, route-intent remote, the forced window -- gets the 429, because those
    # callers handle refusal themselves. Local genuinely DOWN with no budget -> 503 + Retry-After.
    # 2026-10-02: the stalled-delivery brake (default $2) must not block the REMEDIATOR. With outcome 'stalled'
    # and $7.74 spent it refused overflow to Halo's mind during a planned local-offline window -> 'rejected-bg'
    # 503s, so the one actor that can end the stall could not think (deadlock). Kevin's own turns and Halo's
    # turns bypass the brake; they remain inside the hard $25 spend authority (_spend_allows_overflow).
    # Unlabelled callers default to class 'kevin', so only an EXPLICIT Kevin signal counts here: the X-Work-Class
    # header or his LibreChat client. Halo is recognised by flow_class_of (control turn / client map).
    _early_cls = flow_class_of(request, body, background, halo_control)
    _explicit_kevin = ((request.headers.get("X-Work-Class") or "").strip().lower() == "kevin"
                       or "librechat" in (request.headers.get("X-Client") or "").lower())
    automatic_paid_ok = (_early_cls == "halo" or _explicit_kevin
                         or _automatic_remote_budget_allows(ptok, maxtok))
    overflow_ok = ((not _no_overflow) and automatic_paid_ok and remote_ok()
                   and _spend_allows_overflow(ptok, maxtok))
    # 2026-10-02 (lane NO): the stalled-delivery brake guards OPTIONAL overflow -- a remote trip taken although local
    # could serve the request, only slower. It must never refuse work that ONLY remote can serve: during a planned
    # local-offline window or an engine-down moment the alternative to the valve is a 503, and refusing the estate's
    # own repair/authoring calls is what keeps it stalled (stalled -> brake -> 503 -> no remediation -> stalled;
    # 35+ 'stuck-intervene-deferred: HTTP 503' events, 3000+ 503s in one day). There the hard $25 authority alone
    # governs, exactly as the offline-window design above promises ("inside the daily cap").
    remote_only_ok = ((not _no_overflow) and remote_ok() and _spend_allows_overflow(ptok, maxtok))

    async def _overflow_forward(reentry=True):
        """Forward an overflow; if the spend authority refuses it after all (a race with other
        spenders), serve it locally (reentry) or, when local has already failed, report 503."""
        resp = await _forward_remote(request, path, body, streaming)
        if resp is not None and getattr(resp, "headers", {}).get("X-Gateway-Spend-Refused"):
            if reentry:
                log.info("route %s overflow refused by the spend authority -> local", path)
                return await _route_completions(request, _no_overflow=True)
            return _cap_exhausted_unavailable("local failed and the daily remote cap is exhausted")
        return resp
    # BG-LOCAL-ONLY: set to the triggering reason ("local-down") once a background request has
    # been HELD for an engine-health recovery below. When set, a local success further down in
    # this function (tiny fast-lane / empty-retry / normal admission) is recorded as
    # route="held" instead of route="local" -- same response to the client either way, just
    # telemetry that distinguishes "recovered from a held outage" from an ordinary local hit.
    # None for every request that never went through the hold -- i.e. everything today already
    # covers, byte-for-byte the same recorded route as before.
    bg_held_reason = None

    # LOCAL-FIRST (L1): a predictive overflow reason routes remote only when local is measurably
    # saturated. _lf_keep() is evaluated LAST in each guard below, so it only runs (and only
    # counts) when that guard would otherwise have sent the request remote.
    _lf_res = local_reservation_estimate(ptok, maxtok)
    _lf_kept = []

    def _lf_keep(reason):
        keep, why = local_first_decision(reason, background=background, units=units,
                                         reservation=_lf_res, est_computed=est_computed, cls=_early_cls,
                                         warm=_cr_warm)
        if not background and qol_governs(reason):       # lane CFG: QoL interactive overflow (shadow logs, on decides)
            _q = qol_decide(_early_cls, est_computed, ptok, units=units, pm_chain=_pm.get("chain"))
            _q.update(qol_at="guard:" + reason, qol_legacy="local" if keep else "remote")
            _active_set(request, **_q)
            _QOL_STATS["%s:%s->%s" % (reason, _q["qol_legacy"], _q["qol_would"])] += 1
            if QOL_OVERFLOW == "on" and _q["qol_would"] in ("local", "remote"):
                keep, why = _q["qol_would"] == "local", "qol:" + _q["qol_why"]
        if keep:
            _local_first_kept[reason] += 1
            _lf_kept.append(reason)
            log.info("route %s %s predicted but local has capacity -> local-first", path, reason)
        elif why != "policy-off":
            _local_first_remote[f"{reason}:{why}"] += 1
            try:                                   # LF: cold-cache loop signal (remote turns never warm the local prefix)
                if _pm["credit"] < 0.1 * max(1, ptok) and why.startswith("saturated"):
                    _local_first_cold_remote[_client_key(_local_first_cold_remote, client)] += 1
            except Exception:
                pass
        return keep

    if LOG_REQUESTS:
        try:
            model_req = json.loads(body).get("model", "?")
        except Exception:
            model_req = "?"
        log.info("REQ ip=%s ua=%r model=%s ptok=%d maxtok=%d stream=%s tiny=%s bg=%s preview=%r",
                 getattr(request, "remote", "?"), request.headers.get("User-Agent", "?")[:45],
                 model_req, ptok, maxtok, streaming, tiny, background, _preview(body))

    # FULL LOCAL is the operator's $0 safety mode for every request. Explicit
    # remote aliases cannot bypass it; callers must wait until the mode changes.
    if LOCAL_ONLY and (custom_endpoint or alias_force_remote):
        record_event("rejected", "full-local-remote-alias", request, units, 0, **ev)
        return web.json_response({"error": {"message": "full-local mode disables paid remote aliases",
                                            "type": "full_local_remote_disabled"}}, status=503)
    # Explicit aliases are gateway-owned routing contracts.  Custom aliases target their
    # configured OpenAI-compatible endpoint; built-ins are stable local/remote modes.
    # They are evaluated before the normal local-first and full-remote choices.
    if custom_endpoint:
        log.info("route %s alias=%s -> remote(alias)", path, alias.get("name"))
        record_event("remote", "alias", request, units, 0, **ev)
        return await _forward_remote(request, path, body, streaming, endpoint=custom_endpoint)
    if alias_kind == "disabled":
        return web.json_response({"error": {"message": f"gateway alias {alias.get('name')} is disabled",
                                              "type": "alias_disabled"}}, status=409)
    if alias_force_remote:
        log.info("route %s alias=estate-remote -> remote(alias)", path)
        record_event("remote", "alias", request, units, 0, **ev)
        if alias.get("model"):
            return await _forward_remote(request, path, body, streaming, model=alias["model"])
        return await _forward_remote(request, path, body, streaming)

    # TEXT-ONLY GUARD (lane GW): the local engine does not take image/video/audio parts. Decide here, once, instead of
    # letting the engine answer 400 "At most 0 image(s)": the remote vision provider when one is declared and the hard
    # daily cap has room (never for callers pinned to local), else one clear 400 that names the reason.
    _media = request_media(body)
    _unsupported = _media - set(local_input_modalities())
    if _unsupported:
        _pinned = alias_local_only or "local-pin" in (request.headers.get("X-Client") or "").lower() or LOCAL_ONLY
        if REMOTE_VISION and _unsupported <= {"image"} and not _pinned and remote_ok():
            if _spend_allows_overflow(ptok, maxtok):
                log.info("route %s %s content, local is text-only -> remote(vision)", path, "/".join(sorted(_unsupported)))
                record_event("remote", "vision", request, units, 0, **ev)
                return await _overflow_forward(reentry=False)
            _active_set(request, route="rejected", reason="vision-cap-exhausted")
            return _cap_exhausted_unavailable(
                "this request carries %s content the local engine cannot read and the daily remote cap has no room for the vision provider"
                % "/".join(sorted(_unsupported)))
        if _pinned:
            why = "this caller/alias is pinned to the local engine; send it to a vision-capable alias or strip the media"
        elif not remote_ok():
            why = "no remote provider is available to take it; strip the media or send it to a vision-capable alias"
        elif not REMOTE_VISION or not _unsupported <= {"image"}:
            why = "the configured remote provider does not accept %s content either; send it to a vision-capable alias or strip the media" % "/".join(sorted(_unsupported))
        else:
            why = "strip the media or send it to a vision-capable alias"
        _active_set(request, route="rejected", reason="text-only")
        log.info("route %s %s content refused: local model is text-only", path, "/".join(sorted(_unsupported)))
        return web.json_response({"error": {"message": text_only_guard_message(_unsupported, why),
                                            "type": "invalid_request_error", "code": "local_model_text_only",
                                            "param": "messages"}}, status=400)

    # MASTER SWITCH: full-remote mode (maintenance/debug) — everything -> DeepSeek.  The
    # estate-local alias is the one explicit per-request exception.
    if remote_ok() and effective_force_remote() and not alias_local_only:
        # BG-LOCAL-ONLY: a maintenance window is a deliberate, operator-chosen full-remote
        # mode -- not a transient outage worth waiting out -- so background traffic is
        # REJECTED immediately (no wait, no paid remote) rather than held or forwarded.
        if background and BG_LOCAL_ONLY:
            log.info("route %s bg + force_remote window -> rejected-bg(force-remote)", path)
            record_event("rejected-bg", "force-remote", request, units, 0, **ev)
            return _bg_reject_response()
        record_event("remote", "forced", request, units, 0, **ev)
        return await _forward_remote(request, path, body, streaming)

    # LOCAL-PIN: X-Client containing "local-pin" means the caller (coder_loop,
    # localflow local lane) exists to burn FREE local tokens — never divert it to
    # paid remote for congestion reasons (size/big-out/big-prompt/monster/foreign).
    # It queues behind whatever is running instead. A truly dead local still
    # falls through to the local-down branch (remote beats a hard failure).
    local_pin = "local-pin" in (request.headers.get("X-Client") or "").lower()
    route_intent = (request.headers.get("X-Gateway-Route-Intent") or "").strip().lower()
    remote_intent = route_intent in {"remote", "overflow", "deepseek"}

    # ROUTE-INTENT: callers can ask the gateway to treat this as overflow work (for example an
    # explicit brain escalation), but the gateway still checks overflow_ok and local-pin before
    # making the provider decision.  This keeps routing authority in one place.
    if remote_intent and automatic_paid_ok and remote_ok() and not local_pin and not alias_local_only:
        log.info("route %s gateway route intent=%s -> remote(intent)", path, route_intent)
        record_event("remote", "intent", request, units, 0, **ev)
        return await _forward_remote(request, path, body, streaming)

    # size cap: too-big-for-this-box requests OOM local even at budget=1 -> send straight to remote
    if overflow_ok and not local_pin and not alias_local_only and over_local_cap(body):
        log.info("route %s est prompt+max > %d -> remote(size)", path, max_local_tokens())
        record_event("remote", "size", request, units, 0, **ev)
        return await _overflow_forward()

    # LF: checked BEFORE the predictive guards below (it used to follow them), so a window's traffic is labelled
    # local-offline and not big-out/big-prompt/monster: 2,822 of 4,179 "discretionary" remote trips on 10-02/03 were window traffic.
    # CF: PLANNED local-offline window (benchmark / engine upgrade): the remote valve carries the estate inside
    # the daily cap; callers that must stay local (pinned, estate-local) and everything when no remote can take
    # work are told to retry -- the engine is being worked on, so nothing is admitted to it.
    if _local_offline():
        if remote_only_ok and not alias_local_only and not local_pin:
            if not overflow_ok:
                _stall_brake_not_applied("local-offline", client)
            log.info("route %s planned local-offline window (%s) -> remote(local-offline)", path, _OFFLINE["reason"])
            record_event("remote", "local-offline", request, units, 0, **ev)
            return await _overflow_forward()
        _OFFLINE["refused"] += 1
        record_event("rejected-bg", "local-offline", request, units, 0, **ev)
        left = max(5, min(60, int(_OFFLINE["until"] - time.time())))
        return web.json_response({"error": {"message": "local engine is offline for planned work (%s); retry after %ds" % (
            _OFFLINE["reason"], left), "type": "local_offline_window"}}, status=503,
            headers={"Retry-After": str(left), "X-Gateway-Offline": "active"})

    # big-OUTPUT requests (e.g. Hermes max_tokens=65536): long generations that saturate this slow
    # box and hang interactive clients -> straight to remote (DeepSeek serves them far faster).
    # LF: judged on the output this caller is EXPECTED to produce (its recent p95 completion length), the requested
    # max_tokens being only the ceiling -- Halo asks for 16384 on every turn and writes p90 846.
    _expected_out, _expected_src = expected_output_tokens(client, maxtok)
    if (overflow_ok and not local_pin and not alias_local_only and BIG_OUTPUT > 0 and _expected_out >= BIG_OUTPUT
            and not _lf_keep("big-out")):
        log.info("route %s maxtok=%d expected_out=%d (%s) >= %d -> remote(big-out)", path, maxtok, _expected_out, _expected_src, BIG_OUTPUT)
        record_event("remote", "big-out", request, units, 0, **ev)
        return await _overflow_forward()

    # big-PROMPT requests (e.g. a pi session whose context has grown huge): can't prefill within the
    # first-token cap on this slow box and would saturate/OOM local -> straight to remote up front.
    if (overflow_ok and not local_pin and not alias_local_only and BIG_PROMPT > 0 and _cost_tokens >= big_prompt_limit(_early_cls)
            and not _lf_keep("big-prompt")):
        log.info("route %s ptok=%d cost_tokens=%d >= %d -> remote(big-prompt)", path, ptok, _cost_tokens, big_prompt_limit(_early_cls))
        record_event("remote", "big-prompt", request, units, 0, **ev)
        return await _overflow_forward()

    # local DOWN -> overflow immediately (waiting for a slot won't help a dead engine) --
    # UNLESS this is background traffic under BG_LOCAL_ONLY: hold it and poll for recovery
    # instead (background can wait; it must never pay for remote to cover an engine restart).
    if not await local_healthy():
        if background and BG_LOCAL_ONLY and remote_ok():
            _active_set(request, phase="held")
            held_deadline = time.time() + BG_WAIT_LOCAL
            held_waited = 0.0
            while not _health["ok"] and time.time() < held_deadline:
                await asyncio.sleep(SLOT_POLL)
                held_waited += SLOT_POLL
                await local_healthy()   # refresh cached health while holding
            if not _health["ok"]:
                log.info("route %s bg held %.1fs, local still down -> rejected-bg(local-down)",
                         path, held_waited)
                record_event("rejected-bg", "local-down", request, units, held_waited, **ev)
                return _bg_reject_response()
            log.info("route %s bg held %.1fs, local recovered -> resuming normal routing",
                     path, held_waited)
            bg_held_reason = "local-down"
            # FALL THROUGH: local is healthy again, so every check below (monster bypass, tiny
            # fast-lane, admission wait, relay) runs exactly as it would have if
            # local_healthy() had returned True on the very first check above.
        elif remote_only_ok and not alias_local_only:
            if not overflow_ok:
                _stall_brake_not_applied("local-down", client)
            log.info("route %s local unhealthy -> remote(local-down)", path)
            record_event("remote", "local-down", request, units, 0, **ev)
            return await _overflow_forward()
        elif alias_local_only:
            return web.json_response({"error": {"message": "estate-local requires the local engine, which is currently unavailable",
                                                  "type": "local_unavailable"}}, status=503)
        elif remote_ok():
            # local DOWN and the remote cap has no room for this overflow: nothing can serve it now
            return _cap_exhausted_unavailable("local engine is down and the daily remote cap is exhausted")

    # MONSTER-IN-FLIGHT bypass: a huge prefill is monopolizing engine steps; anything admitted
    # now would crawl (~1 tok per chunk-step). Route new arrivals remote until it drains.
    _foreign = _health.get("foreign", 0) if FOREIGN_LOAD_GUARD else 0
    _foreign_heavy = bool(_health.get("foreign_heavy", False)) if FOREIGN_LOAD_GUARD else False
    # Cache-aware mode measures the monster in UNCACHED prefill seconds at the measured rate (warm
    # multi-turn prompts, which share their KV with the cache, no longer count as a monster);
    # legacy mode keeps the raw in-flight prompt-token threshold.
    _monster_now = ((MONSTER_PREFILL_SECS > 0 and _prefill_backlog_secs() >= MONSTER_PREFILL_SECS)
                    if USE_COMPUTED_COST else (MONSTER_INFLIGHT > 0 and _inflight_tokens >= MONSTER_INFLIGHT))
    if overflow_ok and not local_pin and not alias_local_only and (
            _monster_now or _foreign_heavy
    ) and not _lf_keep("monster"):
        log.info("route %s monster/foreign inflight tok=%d foreign=%d foreign_tok=%d -> remote(monster)",
                 path, _inflight_tokens, _foreign, int(_health.get("foreign_tokens", 0) or 0))
        record_event("remote", "monster", request, units, 0, **ev)
        return await _overflow_forward()

    # PERFORMANCE BREAKER: sustained measured queue/prefill/KV/GPU pressure sends new work to
    # the configured remote provider. In-flight local work is allowed to finish; no unsafe
    # mid-generation migration is attempted.
    if overflow_ok and not local_pin and not alias_local_only and perf_breaker_active() and not _lf_keep("perf"):
        log.info("route %s performance breaker (%s) -> remote(perf)",
                 path, _PERF_STATE.get("reason", "overload"))
        record_event("remote", "perf", request, units, 0, **ev)
        return await _overflow_forward()

    predicted = predicted_occupancy_seconds(_cost_tokens, _expected_out, max(1, _inflight + 1))
    _active_set(request, predicted_occupancy_s=predicted)
    if (overflow_ok and not local_pin and not alias_local_only and predicted is not None
            and predicted >= PREDICTED_OCCUPANCY_SECS and not _lf_keep("predicted")):
        log.info("route %s predicted occupancy %.1fs >= %.1fs -> remote(predicted)",
                 path, predicted, PREDICTED_OCCUPANCY_SECS)
        record_event("remote", "predicted", request, units, 0, **ev)
        return await _overflow_forward()

    try:
        local_body = _prepare_local_body(request, body, background)
        local_context_body, local_ctx = _prepare_provider_context(
            local_body, local_context_limit(), LOCAL_MAX_OUT)
        if local_context_body is None:
            if alias_local_only or LOCAL_ONLY:
                return _context_error(local_ctx, "local")
            if overflow_ok and not local_pin:
                record_event("remote", "context", request, units, 0, **ev)
                return await _overflow_forward()
            return _context_error(local_ctx, "local")
        if local_ctx.get("compacted"):
            local_body = _prepare_local_body(request, local_context_body, background)
            _active_set(request, context_provider="local", context_limit=local_context_limit(),
                        context_prompt_tokens=local_ctx.get("prompt_tokens"),
                        context_compacted=True, context_omitted=int(local_ctx.get("omitted", 0) or 0))
        else:
            _active_set(request, context_provider="local", context_limit=local_context_limit(),
                        context_prompt_tokens=local_ctx.get("prompt_tokens"), context_compacted=False)
        reservation, sequences = local_memory_reservation(local_body)
    except (ValueError, TypeError, AttributeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
    _tb = token_budget()
    if _tb > 0 and reservation > _tb:
        # Waiting cannot make a request larger than the entire pool admissible.
        if overflow_ok and not local_pin and not alias_local_only and not LOCAL_ONLY:
            record_event("remote", "tokens", request, units, 0, **ev)
            return await _overflow_forward()
        return web.json_response({"error": "request exceeds local token reservation budget"}, status=503)
    units *= sequences
    flow_cls = flow_class_of(request, body, background, halo_control)       # CF: work class
    if FLOW_MODE != "off":
        flow_note_arrival(flow_cls, ptok, est_computed)
        _active_set(request, flow_class=flow_cls)
    if halo_control:
        # Units estimate scheduler pressure from prompt size. Halo's one
        # control sequence must fit the protected place even with a long
        # awareness prompt; the exact prompt/output KV reservation above
        # remains charged and guarded independently.
        units = sequences
    reserved = False
    claimed_at = 0.0

    def claim_local():
        nonlocal reserved, claimed_at
        global _inflight, _inflight_tokens, _inflight_reserved_tokens, _inflight_computed
        claimed_at = time.time()
        _flow_bump()
        _inflight += units
        _inflight_tokens += ptok
        _inflight_reserved_tokens += reservation
        _inflight_computed += est_computed
        _pm["backlog_held"] = est_computed          # released at first token (prefill done), not at stream end
        reserved = True
        # The engine will now prefill this prompt: from here on its prefix is (being) cached, so
        # the next turn of this conversation is cheap. Learn ONLY from requests that go local.
        _pm_commit(_pm["chain"])

    def release_local():
        nonlocal reserved
        global _inflight, _inflight_tokens, _inflight_reserved_tokens, _inflight_computed
        if reserved:
            _inflight -= units
            _inflight_tokens -= ptok
            _inflight_reserved_tokens -= reservation
            _inflight_computed = max(0, _inflight_computed - _pm.pop("backlog_held", 0))
            reserved = False
            flow_note_service(flow_cls, time.time() - claimed_at)
            flow_prefill_done(id(request))
            _flow_bump()

    # TINY fast-lane: small calls skip the queue, but never the KV memory limit.
    # The final extra place belongs to Halo control, even when routine tiny
    # requests are busy. Other tiny work keeps any remaining extra places.
    if tiny:
        tiny_limit = admission_lane_limit(background, effective_budget(), FG_RESERVED,
                                          halo_control=halo_control, tiny=True,
                                          tiny_extra_lanes=TINY_EXTRA_LANES)
        if (_health["ok"] and (_inflight + units) <= tiny_limit
                and _memory_available(reservation, halo_control=halo_control)):
            claim_local()
            log.info("route %s TINY units=%d inflight=%d/%d -> local(tiny)",
                     path, units, _inflight, tiny_limit)
            _active_set(request, phase="local", route="local")
            try:
                # tiny fast-lane needs the same body preparation as the main path: these are
                # exactly the small-max_tokens calls that get starved to an empty response,
                # AND (2026-09-09) they are most of the traffic NO_THINK_IPS is meant to
                # cover. It used to inline its own copy of the chain and omitted
                # strip_thinking, so the no-think policy was dead on this lane specifically.
                kind, payload = await _relay(request, LOCAL, path,
                                             local_body,
                                             None, streaming, concurrency=1)
                if kind == "ok":
                    _note_payload_outcome(request, payload, streaming)   # TELEMETRY
                    record_event("held" if bg_held_reason else "local",
                                 bg_held_reason or ("tiny-learned" if tiny_learned else "tiny"), request, units, 0, **ev)
                    if tiny_learned:
                        _MICRO_STATS["admitted"] += 1
                    return payload
                status, text, oom = payload
                if oom:
                    trigger_backoff(f"local {status}: {text[:120]}")
                log.warning("local(tiny) failed (%s) -> failover to remote", status)
                if alias_local_only:
                    record_event("held", "local-failed", request, units, 0, **ev)
                    return web.json_response({"error": {
                        "message": "estate-local local attempt failed; paid failover is disabled",
                        "type": "local_only_unavailable"}}, status=503)
                if not remote_ok():
                    record_event("held", "local-failed", request, units, 0, **ev)
                    release_local()
                    return _cap_exhausted_unavailable("local attempt failed and paid failover is unavailable")
                record_event("remote", "failover", request, units, 0, **ev)
                release_local()
                return await _overflow_forward(reentry=False)
            finally:
                release_local()
        elif overflow_ok and not alias_local_only and not tiny_learned:
            log.info("route %s TINY inflight=%d/%d full -> remote(tiny-fast)",
                     path, _inflight, tiny_limit)
            record_event("remote", "tiny-fast" if _memory_available(reservation, halo_control=halo_control) else "tokens", request, units, 0, **ev)
            return await _overflow_forward()
        # no remote configured -> fall through to the normal local wait loop

    # local UP: claim a slot, WAITING for capacity instead of instant-overflow.
    # PRIORITY LANES: background (cron/batch) may only fill lanes beyond FG_RESERVED — those
    # stay free so an interactive turn NEVER queues behind robot busywork — and background
    # waits only BG_WAIT before overflowing to the cheap remote. Interactive traffic, by
    # policy (2026-09-11), does not overflow on a mere timeout at all -- see admission_wait().
    # The check-and-increment is done with no await in between, so it's race-free under asyncio.
    lane_limit = admission_lane_limit(background, effective_budget(), FG_RESERVED,
                                      halo_control=halo_control,
                                      tiny_extra_lanes=TINY_EXTRA_LANES)
    deadline = time.time() + admission_wait_seconds(
        background, overflow_ok, is_peak(), LOCAL_WAIT, BG_WAIT, INTERACTIVE_NEVER_OVERFLOW)
    admitted = False
    t_admit0 = time.time()
    # lane CFG QoL: an interactive request that may overflow is judged by its PREDICTED wait, not a timer.
    _qol_live = (QOL_OVERFLOW in ("shadow", "on") and not background and overflow_ok and not alias_local_only
                 and not local_pin)
    _qol_broke, _qol_next = False, 0.0
    if _qol_live:
        _q = qol_decide(_early_cls, est_computed, ptok, units=units, pm_chain=_pm.get("chain"))
        _q.update(qol_at="admission")
        _active_set(request, **_q)
        if QOL_OVERFLOW == "on":
            deadline = t_admit0 + max(1.0, float(LOCAL_FIRST_FIRST_TOKEN_MAX))   # hard bound; the prediction decides first
    queued = False
    # The size-implied unit cost does not change while we wait (est_computed is fixed at arrival), so
    # compute it once rather than re-hashing the body on every 50 ms poll.
    _desired_const = _desired_units(body, client, est_computed) * sequences

    def _legacy_fits():
        """The admission predicate exactly as it was before CF (lanes, bg-idle bypass, KV reservation, prefill
        window). The flow ticket asks it too, so the ticket chosen to go next is always one that CAN go."""
        bg_big_idle = (background and BG_BIG_LOCAL_WHEN_IDLE and _desired_const > lane_limit
                       and _inflight == 0 and _waiting <= (1 if queued else 0))
        admit_units = min(effective_budget(), _desired_const) if bg_big_idle else units
        win_ok = bg_big_idle or prefill_window_ok(est_computed, halo_control)
        ok = bool(_health["ok"] and not _local_offline() and admit_units >= sequences
                  and ((_inflight + admit_units) <= lane_limit or bg_big_idle)
                  and _memory_available(reservation, halo_control=halo_control) and win_ok)
        return ok, admit_units, bg_big_idle, win_ok

    flow_t = flow_make_ticket(request, body, flow_cls, _pm, ptok, units, lambda: _legacy_fits()[0])
    if FLOW_MODE == "enforce" and flow_cls in FLOW_REFUSABLE and flow_t.deadline_at is not None:
        # Local-first (Kevin 10-02): robot work waits for local until it would miss its own deadline (bounded by
        # BG_WAIT_LOCAL) instead of overflowing to the paid remote after a few seconds.
        # The flow class (not the legacy is_background() test, which does not know the overseer-* clients) owns the wait
        # bound of robot work whenever remote is a possible valve.
        if overflow_ok and not alias_local_only:
            deadline = min(flow_t.deadline_at - flow_t.cost_s, t_admit0 + BG_WAIT_LOCAL)
    if FLOW_MODE != "off":
        _refusal = flow_admission_check(flow_t, bool(overflow_ok and not alias_local_only))
        _active_set(request, flow_expected_wait_s=flow_t.expected_wait_s)
        if _refusal and _refusal.get("overflow"):
            _FLOW_STATS["valve_" + flow_cls] += 1
            log.info("route %s %s cannot start locally in time (wait %.1fs + prefill %.1fs > %.1fs) -> remote(flow-deadline)",
                     path, flow_cls, _refusal["expected_wait_s"], _refusal["own_prefill_s"], _refusal["deadline_in_s"])
            flow_dequeue(flow_t)
            record_event("remote", "flow-deadline", request, units, 0, **ev)
            return await _overflow_forward()
        if _refusal:
            _FLOW_STATS["refused_" + flow_cls] += 1
            log.info("route %s %s refused before prefill: expected wait %.1fs + prefill %.1fs > %.1fs left",
                     path, flow_cls, _refusal["expected_wait_s"], _refusal["own_prefill_s"], _refusal["deadline_in_s"])
            record_event("rejected-bg", "flow-deadline", request, units, 0, **ev)
            return _flow_refusal_response(_refusal)
    flow_enqueue(flow_t)
    _local_reason = "-"      # "bg-big-idle" when the idle-engine rule admitted a big background request
    if _lf_kept:             # LOCAL-FIRST: telemetry shows which predictive reason was overridden
        _local_reason = "lf-" + _lf_kept[0]
    admitted_conc = 1        # local concurrency at admission -> scales the first-token deadline
    waited = 0.0
    client_gone_queued = False
    try:
        while True:
            # big background request + idle engine: nothing in flight and nobody else queued
            # (this request counts itself in _waiting once queued) -> may take MORE than the
            # bg-only reserved cap (up to the whole budget), since nothing is competing for the
            # reserved lanes anyway. 2026-09-11: gated on the UNCAPPED desired size now that
            # `units` itself is already capped by estimate_units() -- under proportional units
            # most big bg requests fit under lane_limit without ever needing this bypass; it
            # only fires for requests so large that even the proportional estimate would still
            # exceed budget-FG_RESERVED.
            if _client_gone(request):
                client_gone_queued = True
                break           # the caller hung up while queued: admit nothing, prefill nothing
            if ((_local_offline() or not _health["ok"]) and overflow_ok and not alias_local_only and not local_pin
                    and not (background and BG_LOCAL_ONLY and not _local_offline())):
                break           # a planned offline window opened, or local died, under a waiter: the valve takes it
            _fits, _admit_units, _bg_big_idle, _win_ok = _legacy_fits()
            if _fits and flow_turn(flow_t):
                flow_on_admit(flow_t)
                _active_set(request, flow_adjacent=flow_t.adjacent, flow_held=flow_t.held or None)
                if _bg_big_idle and (_inflight + _admit_units) > lane_limit:
                    _stats["bg_big_idle_local"] = _stats.get("bg_big_idle_local", 0) + 1
                    _local_reason = "bg-big-idle"
                units = _admit_units   # reassign the OUTER units: the release below (and any
                                       # telemetry/record_event after this block) must charge
                                       # and refund the SAME amount that was actually admitted.
                claim_local()
                admitted_conc = _inflight
                admitted = True
                break
            if (background and not _win_ok and _health["ok"] and (_inflight + _admit_units) <= lane_limit
                    and remote_ok()):
                # Only the engine's prefill queue is full: background work is patient and has no
                # reason to pay for remote because of it -- keep waiting (bounded) for it to drain.
                deadline = max(deadline, t_admit0 + BG_WAIT_LOCAL)
            if _qol_live and QOL_OVERFLOW == "on" and waited >= _qol_next:
                _qol_next = waited + 1.0
                _q = qol_decide(_early_cls, est_computed, ptok, units=units, waited=waited, pm_chain=_pm.get("chain"))
                if _q.get("qol_would") == "remote":
                    _q.update(qol_at="wait", qol_legacy=None)
                    _active_set(request, **_q)
                    _qol_broke = True
                    break
            if time.time() >= deadline:
                break
            if not queued:                       # first time we couldn't get a slot -> we're backlogged
                queued = True
                _active_set(request, phase="queued",
                            queue_position=_waiting_by_class["background" if background else "interactive"] + 1)
                _waiting += 1
                _waiting_by_class["background" if background else "interactive"] += 1
                _stats["peak_waiting"] = max(_stats["peak_waiting"], _waiting)
            await asyncio.sleep(SLOT_POLL)
            waited += SLOT_POLL
            await local_healthy()  # refresh cached health while waiting
    finally:
        flow_dequeue(flow_t)
        if queued:
            _waiting -= 1
            _waiting_by_class["background" if background else "interactive"] -= 1
        _note_admission_wait(waited)   # LOCAL-FIRST queue-wait signal (admitted or overflowed)

    if client_gone_queued:
        log.info("route %s caller hung up after %.1fs queued -> dropped before any prefill", path, waited)
        record_event("gone", "queued", request, units, waited, **ev)
        return _gone_response(request)

    if not admitted:
        # distinguish WHY we couldn't admit: lane-count vs total-context (size-aware) cap
        if _local_offline():
            reason = "local-offline"
        elif not _health["ok"]:
            reason = "local-down"
        elif (_inflight + units) <= lane_limit and not prefill_window_ok(est_computed, halo_control):
            reason = "prefill"
        elif (_inflight + units) <= lane_limit:
            reason = "tokens"
        elif background and (_inflight + units) <= effective_budget():
            reason = "bg-yield"      # lanes exist but are reserved for interactive
        else:
            reason = "cap"
        if _qol_broke:
            reason = "qol"                # predicted local first token too late and remote measurably faster
        elif _qol_live and not _local_offline() and _health["ok"]:
            _q = qol_decide(_early_cls, est_computed, ptok, units=units, waited=waited, pm_chain=_pm.get("chain"))
            _q.update(qol_at="wait-timeout", qol_legacy="remote")
            _active_set(request, **_q)
            _QOL_STATS["wait-timeout:remote->%s" % _q["qol_would"]] += 1
        where = f"remote({reason})" if overflow_ok and not alias_local_only else "local-only(wait-exhausted)"
        log.info("route %s units=%d inflight=%d/%d tok=%d/%d waited=%.1fs -> %s",
                 path, units, _inflight, effective_budget(), _inflight_tokens, token_budget(), waited, where)
        if queued and not alias_local_only:
            _stats["overflowed_after_wait"] += 1
        if alias_local_only:
            record_event("held", reason, request, units, waited, **ev)
            return web.json_response({"error": {
                "message": "estate-local capacity unavailable; paid overflow is disabled",
                "type": "local_only_unavailable"}}, status=503)
        record_event("remote", reason, request, units, waited, **ev)
        return await _overflow_forward()

    log.info("route %s units=%d inflight=%d/%d waited=%.1fs -> local%s",
             path, units, _inflight, effective_budget(), waited,
             "(bg-big-idle)" if _local_reason == "bg-big-idle" else "")
    _active_set(request, phase="local", route="local", waited=round(waited, 1))
    # FLIGHT RECORDER (RCA, 2026-08-13): persist big local-routed request bodies so the
    # next Xid-31 crash leaves a deterministic repro payload. Ring of 40 files, 0600.
    # Directory made configurable (SHIM_FLIGHTREC_DIR, default unchanged) by the
    # shim-remote-observability lane, 2026-09-05 -- see _flightrec_dir()'s docstring.
    try:
        if ptok >= _flightrec_min_tok():
            try:
                fr = _flightrec_dir()
                fn = f"{fr}/{int(time.time())}_{ptok}tok.json"
                await asyncio.get_running_loop().run_in_executor(None, _write_flightrec, fr, fn, body)
            except Exception as e:
                log.warning("flightrec: %s", e)
        _lb = local_body
        kind, payload = await _relay(request, LOCAL, path, _lb, None, streaming, concurrency=admitted_conc)
        if kind == "ok":
            _note_payload_outcome(request, payload, streaming)   # TELEMETRY (see DESIGN.md (c))
            # EMPTY-RESPONSE RETRY. Thinking length is prompt-dependent and unbounded, so no
            # max_tokens threshold can guarantee an answer: measured 2026-08-14, the SAME
            # budget that answered one prompt returned finish_reason=length with empty
            # content on another, and reasoning_effort=low did NOT bound it (mt=4000 low ->
            # empty, while mt=8000 full -> completed in 3519 tok). Predicting is hopeless;
            # detecting is trivial. Retry once with thinking OFF, which reliably answers in
            # ~100 tokens. Non-streaming only -- a streamed response is already committed.
            if (EMPTY_RETRY and not streaming and _is_empty_thinking_response(payload)
                    and b'"enable_thinking": false' not in _lb):
                log.warning("local returned EMPTY (all budget spent thinking) -> retry no-think")
                kind2, payload2 = await _relay(request, LOCAL, path, strip_thinking(_lb), None,
                                               False, concurrency=admitted_conc)
                if kind2 == "ok" and not _is_empty_thinking_response(payload2):
                    _note_payload_outcome(request, payload2, False)   # TELEMETRY: retry's real answer
                    record_event("held" if bg_held_reason else "local",
                                 bg_held_reason or "empty-retry", request, units, waited, **ev)
                    if CRASH_ADAPTIVE:
                        _crash_adaptive_note_local_completion(ptok)
                    return payload2
            record_event("held" if bg_held_reason else "local",
                         bg_held_reason or _local_reason, request, units, waited, **ev)
            if CRASH_ADAPTIVE:
                _crash_adaptive_note_local_completion(ptok)
            return payload
        status, text, oom = payload
        if oom:
            trigger_backoff(f"local {status}: {text[:120]}")
        log.warning("local failed (%s) -> failover to remote", status)
        if alias_local_only:
            record_event("held", "local-failed", request, units, waited, **ev)
            return web.json_response({"error": {
                "message": "estate-local local attempt failed; paid failover is disabled",
                "type": "local_only_unavailable"}}, status=503)
        if not remote_ok():
            record_event("held", "local-failed", request, units, waited, **ev)
            release_local()
            return _cap_exhausted_unavailable("local attempt failed and paid failover is unavailable")
        record_event("remote", "failover", request, units, waited, **ev)
        release_local()
        return await _overflow_forward(reentry=False)
    finally:
        release_local()


# ---------------- passthrough (dynamic; no hardcoded models) ----------------
async def _passthrough(request):
    body = await request.read()
    # /tq/* (EXP-038 snapshot pin/fork) can carry a multi-100K-token prefill;
    # 30s would abort it mid-prefill. Everything else keeps the tight timeout.
    total = 900 if request.path.startswith("/tq/") else 30
    async with aiohttp.ClientSession() as s:
        async with s.request(request.method, f"{LOCAL}{request.rel_url}", data=body or None,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=total)) as up:
            data = await up.read()
            ct = up.headers.get("Content-Type", "application/json").split(";")[0]
            return web.Response(body=data, status=up.status, content_type=ct)


def request_media(body):
    """Set of non-text modalities ('image', 'video', 'audio') present in the content parts of a chat/responses request."""
    found = set()
    try:
        data = json.loads(body)
    except Exception:
        return found
    if not isinstance(data, dict):
        return found

    def scan(items):
        for it in items if isinstance(items, list) else []:
            if not isinstance(it, dict):
                continue
            content = it.get("content")
            for part in content if isinstance(content, list) else []:
                kind = _MEDIA_PART_TYPES.get(str(part.get("type", "")).lower()) if isinstance(part, dict) else None
                if kind:
                    found.add(kind)
            if _MEDIA_PART_TYPES.get(str(it.get("type", "")).lower()):          # responses API: a bare input_image item
                found.add(_MEDIA_PART_TYPES[str(it["type"]).lower()])
    scan(data.get("messages"))
    scan(data.get("input"))
    return found


def local_input_modalities():
    return sorted(set(LOCAL_MODALITIES.split(",")) | {"text"})


def remote_input_modalities():
    return ["text", "image"] if REMOTE_VISION else ["text"]


def text_only_guard_message(media, why):
    return ("local model is text-only (%s content is not accepted by the local engine, which serves with --language-model-only); %s"
            % ("/".join(sorted(media)), why))


def _modalities_fields(mods):
    mods = sorted(set(mods))
    return {"input_modalities": mods, "output_modalities": ["text"], "modalities": {"input": mods, "output": ["text"]},
            "capabilities": {"vision": "image" in mods}}


def _alias_model_rows():
    """The gateway's own routable model names, in OpenAI /v1/models shape.

    These are real models from a client's point of view -- `estate-remote`
    reaches DeepSeek, a custom alias reaches its configured provider -- but they
    exist only in the gateway, so a pure passthrough of vLLM's list never
    mentioned them. Open WebUI builds its picker from this endpoint, so every
    proxy alias was unreachable from the UI even though POSTing to it worked
    (2026-09-17). Disabled custom aliases are deliberately omitted.
    """
    now = int(time.time())
    rows = []
    for name, kind in (("estate", "gateway-default"),
                       ("estate-local", "gateway-local"),
                       ("estate-remote", "gateway-remote"),
                       ("estate-remote-pro", "gateway-remote-pro")):
        mods = local_input_modalities() if name == "estate-local" else sorted(set(local_input_modalities()) | set(remote_input_modalities()))
        if name in ("estate-remote", "estate-remote-pro"):
            mods = remote_input_modalities()
        rows.append({"id": name, "object": "model", "created": now,
                     "owned_by": "gateway", "gateway_alias": kind, **_modalities_fields(mods)})
    for name, rec in sorted(_ALIASES.items()):
        if not rec.get("enabled", True):
            continue
        rows.append({"id": name, "object": "model", "created": now,
                     "owned_by": "gateway", "gateway_alias": "custom-remote",
                     "gateway_target": rec.get("model") or "",
                     **_modalities_fields(["text", "image"] if rec.get("vision") else ["text"])})
    return rows


async def h_models(request):
    # Passthrough of whatever vLLM serves (new models appear automatically),
    # UNIONED with the gateway's own aliases so clients can discover the proxy
    # routes as well as the local engine's names.
    local_rows, local_ok = [], False
    try:
        resp = await _passthrough(request)
        body = json.loads(resp.body.decode() or "{}") if getattr(resp, "body", None) else {}
        local_rows = body.get("data") or []
        local_ok = True
    except Exception as e:
        log.warning("h_models: local unreachable (%r)", e)

    for r in local_rows:                    # the engine's own rows: advertise what it accepts
        if isinstance(r, dict):
            for k, v in _modalities_fields(local_input_modalities()).items():
                r.setdefault(k, v)
    seen = {r.get("id") for r in local_rows}
    rows = list(local_rows) + [r for r in _alias_model_rows() if r["id"] not in seen]
    if not rows:
        return web.json_response({"object": "list", "data": []}, status=503)
    # Aliases can still serve while the engine is down, so this is not a 503.
    return web.json_response({"object": "list", "data": rows},
                             status=200 if (local_ok or rows) else 503)


async def h_health(request):
    """Is the GATEWAY able to serve? Not "is local up".

    These are different questions and conflating them causes false alarms: during an engine
    restart (model swap, config test, crash recovery) local is down for ~40-90s while every
    request is still answered via remote overflow. The old handler returned 503 there, so
    Kevin's Hermes cron paged "DOWN HTTP 503" for a service that was working fine.

    A monitor acts on "can it serve", so that is what /health answers now:
      200 "OK"                 local is healthy
      200 "DEGRADED: ..."      local down, remote available -> requests are still served
      503                      nothing can serve -> a real outage
    Anything that specifically needs local status has /health/local, which keeps the old
    strict behaviour."""
    if await local_healthy():
        return web.Response(text="OK")
    if remote_ok() and REMOTE_BASE:
        return web.Response(text="DEGRADED: local down, serving via remote")
    return web.Response(status=503, text="local down, no remote configured")


async def h_health_local(request):
    """Strict local-only health, for callers that need to distinguish (503 when local down)."""
    return web.Response(text="OK") if await local_healthy() else web.Response(status=503, text="local down")


async def h_catchall(request):
    try:
        return await _passthrough(request)
    except Exception as e:
        return web.json_response({"error": {"message": f"gateway: {e}"}}, status=502)


async def gateway_stats(request):
    up = _health["ok"] if (time.time() - _health["at"] < HEALTH_TTL) else await local_healthy()
    total = _stats["total"] or 1
    return web.json_response({
        "uptime": int(time.time() - _stats["started"]),
        # "uptime" above is the age of the PERSISTED counters (they survive restarts), not of this process;
        # the dashboard needs both to label its lifetime counters honestly.
        "process_uptime": int(time.time() - _PROCESS_STARTED), "stats_since": _stats["started"],
        "local_healthy": up,
        "budget": effective_budget(), "configured_budget": BUDGET,
        "inflight": _inflight, "peak_inflight": _stats["peak_inflight"],
        "waiting": _waiting, "peak_waiting": _stats["peak_waiting"],
        # K4 micro lane v2 + probe synth: what the learned tiny lane and the liveness-probe shortcut did since start.
        "micro_lane": {"learn": MICRO_LEARN, "signatures": len(_MICRO_HIST), "qualifying": sum(
            1 for h in _MICRO_HIST.values() if len(h) >= MICRO_MIN_SAMPLES and max(h) <= MICRO_MAX_OUT),
            "probe_synth_enabled": PROBE_SYNTH, **_MICRO_STATS},
        # gw-queue-position-header (2026-09-11): per-class queue depth, so "am I personally
        # waiting" is answerable at a glance instead of one aggregate number that background
        # traffic can dominate. Deliberately NOT paired with a fake per-class wait estimate --
        # avg_wait below stays a single honest aggregate rather than two numbers dressed up as
        # a per-class split that record_event() does not actually track. See
        # gw-ttft-decomposition-telemetry for the real per-class latency breakdown this wants.
        "waiting_by_class": dict(_waiting_by_class),
        "flow_mode": FLOW_MODE, "flow_waiting_by_work_class": {c: sum(1 for w in _FLOW["waiters"] if w.cls == c) for c in FLOW_CLASSES},
        "overflowed_after_wait": _stats["overflowed_after_wait"],
        "inflight_tokens": _inflight_tokens, "token_budget": token_budget(),
        "capacity_model": capacity_model_facts(),
        "inflight_computed": _inflight_computed,
        "remote_dead_for_s": max(0, int(_remote_dead_until - time.time())),
        "remote_402_count": _remote_dead_count,
        "prefill_backlog_secs": round(_prefill_backlog_secs(), 1),
        "cache_model": _pm_summary(),
        "response_shape": _shape_stats_summary(),      # [GW2 / L94+L95]
        "estimate_error": _est_err_summary(),          # [GW2] gateway prompt/computed estimate vs engine usage
        "credit_probe": _credit_probe_summary(),       # [GW2 / L105] engine-reported cache credit
        "tokenize": dict(_TOK_STATS, memo_size=len(_TOK_MEMO)),   # [GW2] /tokenize calls, memo hits, loop-blocking ms
        "inflight_reserved_tokens": _inflight_reserved_tokens,
        "halo_control_lane_limit": admission_lane_limit(
            False, effective_budget(), FG_RESERVED, halo_control=True,
            tiny_extra_lanes=TINY_EXTRA_LANES),
        "halo_control_token_reserve": _halo_control_token_reserve(),
        "backoff": max(0, int(_backoff_until - time.time())),
        "remote_model": REMOTE_MODEL, "remote_enabled": REMOTE_ENABLED,
        "local_only": bool(LOCAL_ONLY), "mode": routing_mode(),
        "local_wait": LOCAL_WAIT,
        "monster_inflight": MONSTER_INFLIGHT,
        "total": _stats["total"], "local": _stats["local"], "remote": _stats["remote"],
        "held": _stats["held"], "rejected_bg": _stats["rejected_bg"],
        "local_pct": round(100 * _stats["local"] / total, 1),
        "remote_pct": round(100 * _stats["remote"] / total, 1),
        "perf_breaker": perf_breaker_active(),
        "perf_breaker_reason": _PERF_STATE.get("reason", ""),
        "avg_wait": round(_stats["waited_total"] / (_stats["waited_n"] or 1), 1),
        "remote_reasons": dict(_remote_reasons),
        "local_first": {
            "enabled": bool(LOCAL_FIRST), "reasons": sorted(LOCAL_FIRST_REASONS),
            "queue_wait_secs": LOCAL_FIRST_QUEUE_WAIT_SECS,
            "derive": bool(LOCAL_FIRST_DERIVE), "cold_remote_by_client": dict(_local_first_cold_remote),
            "budgets_s": {c: local_first_budget_s(c) for c in FLOW_CLASSES},
            "big_prompt_limit": {c: big_prompt_limit(c) for c in FLOW_CLASSES},
            "expected_output": {"enabled": bool(EXPECTED_OUTPUT), "quantile": EXPECTED_OUTPUT_QUANTILE,
                                "min_samples": EXPECTED_OUTPUT_MIN_SAMPLES,
                                "clients": {c: {"n": len(h), "p": expected_output_tokens(c, 10 ** 9)[0]}
                                            for c, h in list(_OUT_HIST.items())[:50]}},
            "recent_admission_wait": round(recent_admission_wait(), 2),
            "kept_local": dict(_local_first_kept), "still_remote": dict(_local_first_remote),
        },
        "gpu": _gpu_stats(),
        "events": list(_events)[:60],
    })


# ---- TELEMETRY (added): /gateway/telemetry (JSON) + /gateway/metrics (Prometheus text).
# See LANE/DESIGN.md (f). Both are read-only, additive routes with no interaction with the
# routing/failover path -- worst case (engine down, as observed live all session) they report
# a degraded field, never a 5xx of their own. ----
def _telem_tail(series, raw):
    """?fast=N / ?slow=N: only the newest N samples (0 = none). Absent or malformed = everything (back-compat)."""
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return list(series)
    if n < 0:
        return list(series)
    return list(series)[-n:] if n else []


async def gateway_telemetry(request):
    latest = _TELEM_FAST[-1] if _TELEM_FAST else None
    pct = {}
    if _ENGINE_METRICS["ok"]:
        fam = _ENGINE_METRICS["families"]
        for key, base in (("ttft", "vllm:time_to_first_token_seconds"),
                          ("tpot", "vllm:inter_token_latency_seconds"),
                          ("e2e", "vllm:e2e_request_latency_seconds")):
            pct[key] = {"cumulative_p50": _hist_quantile(fam, base, 0.50),
                        "cumulative_p95": _hist_quantile(fam, base, 0.95)}
            if latest and (latest.get("engine") or {}).get("ok"):
                pct[key]["p50"] = latest["engine"].get(key + "_p50")
                pct[key]["p95"] = latest["engine"].get(key + "_p95")
    per_client = {}
    for name, c in _PER_CLIENT.items():
        per_client[name] = {
            "requests": c["requests"], "local": c["local"], "remote": c["remote"],
            "tokens_out": c["tokens_out"], "tokens_out_exact": c["tokens_out_exact"],
            "tokens_out_lb": c["tokens_out_lb"],
            "wait_avg_s": round(c["wait_sum"] / c["wait_n"], 2) if c["wait_n"] else None,
            "ttft_avg_s": round(c["ttft_sum"] / c["ttft_n"], 3) if c["ttft_n"] else None,
            "errors": c["errors"], "cost_est_usd": round(c["cost_est_usd"], 4),
            "by_class": dict(c["classes"]),
        }
    return web.json_response({
        "at": time.time(),
        "latest": latest,
        "series": {"fast": _telem_tail(_TELEM_FAST, request.query.get("fast")),
                   "slow": _telem_tail(_TELEM_SLOW, request.query.get("slow"))},
        "per_client": per_client,
        "errors": list(_ERROR_FEED)[:60],
        "mis_estimates": list(_MISESTIMATE_FEED)[:60],
        "percentiles": pct,
        "windowed_latency": req_latency_facts(),
        "engine_scrape": {"ok": _ENGINE_METRICS["ok"],
                          "age_s": round(time.time() - _ENGINE_METRICS["at"], 1) if _ENGINE_METRICS["at"] else None,
                          "err": _ENGINE_METRICS["err"]},
        "hw_history": hw_store_state(),
    })


def _prom_escape(s):
    return str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _prom_line(name, value, labels=None):
    if value is None:
        return None
    if labels:
        lbl = ",".join(f'{k}="{_prom_escape(v)}"' for k, v in labels.items())
        return f"{name}{{{lbl}}} {value}"
    return f"{name} {value}"


async def gateway_internal_errors(request):
    """Lane SH: every suppressed exception (per source site: count, last time, last error) and every router crash
    since start. Read-only; empty objects mean nothing has been swallowed."""
    now = time.time()
    return web.json_response({
        "swallowed": {site: {"n": r[0], "last_t": round(r[1], 1), "last_age_s": round(now - r[1], 1), "last": r[2]}
                      for site, r in sorted(_SWALLOWED.items())},
        "router_crashes": dict(_ROUTER_CRASHES),
        "telemetry_log": {k: _JSONL_STATE.get(k) for k in ("written", "dropped_queue", "dropped_cap", "dropped_bad",
                                                            "dropped_io", "dropped_err", "last_err")},
    })


async def gateway_metrics(request):
    """One scrape target for both layers: this gateway's own counters/gauges, hand-formatted
    (deliberately not the prometheus_client library -- see the TELEMETRY module docstring far
    above), followed by a verbatim passthrough of the engine's last successful /metrics scrape.
    Never re-derives quantiles as gauges here -- the raw histograms are in the passthrough for
    a real Prometheus consumer to run histogram_quantile() on; see DESIGN.md (f)."""
    L = []

    def emit(name, help_text, mtype, value, labels=None):
        ln = _prom_line(name, value, labels)
        if ln is None:
            return
        L.append(f"# HELP {name} {help_text}")
        L.append(f"# TYPE {name} {mtype}")
        L.append(ln)

    emit("gateway_up", "Gateway process is serving.", "gauge", 1)
    emit("gateway_local_healthy", "1 if the local vLLM engine answered its last health check.",
         "gauge", 1 if _health.get("ok") else 0)
    emit("gateway_inflight_lanes", "Local capacity units currently in flight.", "gauge", _inflight)
    emit("gateway_lane_budget", "Effective local lane budget right now (may be backed off).",
         "gauge", effective_budget())
    emit("gateway_waiting_requests", "Requests currently blocked in the admission wait loop.",
         "gauge", _waiting)
    emit("gateway_inflight_reserved_tokens", "Reserved prompt plus bounded output tokens across local requests.",
         "gauge", _inflight_reserved_tokens)
    emit("gateway_inflight_tokens", "Sum of estimated prompt tokens across in-flight local requests.",
         "gauge", _inflight_tokens)
    emit("gateway_backoff_seconds", "Seconds remaining in an OOM-triggered budget=1 backoff.",
         "gauge", max(0, int(_backoff_until - time.time())))

    L.append("# HELP gateway_requests_total Completed requests by route.")
    L.append("# TYPE gateway_requests_total counter")
    for route, n in (("local", _stats["local"]), ("remote", _stats["remote"]),
                     ("held", _stats["held"]), ("rejected-bg", _stats["rejected_bg"])):
        ln = _prom_line("gateway_requests_total", n, {"route": route})
        if ln:
            L.append(ln)

    if _remote_reasons:
        L.append("# HELP gateway_remote_reason_total Remote-overflow completions by reason.")
        L.append("# TYPE gateway_remote_reason_total counter")
        for reason, n in _remote_reasons.items():
            ln = _prom_line("gateway_remote_reason_total", n, {"reason": reason})
            if ln:
                L.append(ln)

    gpu_fields = [("util", "gateway_gpu_util_percent", "GPU compute utilization, percent."),
                  ("mem_util", "gateway_gpu_mem_util_percent", "GPU memory-controller utilization, percent."),
                  ("used", "gateway_gpu_mem_used_mib", "GPU memory used, MiB."),
                  ("total", "gateway_gpu_mem_total_mib", "GPU memory total, MiB."),
                  ("temp_c", "gateway_gpu_temp_celsius", "GPU temperature, Celsius."),
                  ("power_w", "gateway_gpu_power_watts", "GPU power draw, watts."),
                  ("power_limit_w", "gateway_gpu_power_limit_watts", "GPU power limit, watts."),
                  ("clock_sm_mhz", "gateway_gpu_clock_sm_mhz", "GPU SM clock, MHz."),
                  ("clock_mem_mhz", "gateway_gpu_clock_mem_mhz", "GPU memory clock, MHz."),
                  ("fan_pct", "gateway_gpu_fan_percent", "GPU fan speed, percent.")]
    gpu_now = _gpu_stats()
    for key, name, help_text in gpu_fields:
        rows = [(g.get("index"), g.get(key)) for g in gpu_now if g.get(key) is not None and g.get("index") is not None]
        if not rows:
            continue
        L.append(f"# HELP {name} {help_text}")
        L.append(f"# TYPE {name} gauge")
        for idx, v in rows:
            L.append(_prom_line(name, v, {"gpu": str(idx)}))

    latest = _TELEM_FAST[-1] if _TELEM_FAST else None
    host = (latest or {}).get("host") or {}
    emit("gateway_host_cpu_percent", "Host CPU utilization, percent (delta since previous sample).",
         "gauge", host.get("cpu_pct"))
    emit("gateway_host_ram_used_bytes", "Host RAM used, bytes.", "gauge",
         int(host["ram_used_gb"] * 1e9) if host.get("ram_used_gb") is not None else None)
    emit("gateway_host_ram_total_bytes", "Host RAM total, bytes.", "gauge",
         int(host["ram_total_gb"] * 1e9) if host.get("ram_total_gb") is not None else None)
    if host.get("disk_free_gb") is not None:
        L += ["# HELP gateway_host_disk_free_bytes Free disk space, bytes.",
              "# TYPE gateway_host_disk_free_bytes gauge",
              _prom_line("gateway_host_disk_free_bytes", int(host["disk_free_gb"] * 1e9), {"path": "/"})]
    if host.get("models_disk_free_gb") is not None:
        L.append(_prom_line("gateway_host_disk_free_bytes", int(host["models_disk_free_gb"] * 1e9),
                            {"path": "models"}))

    if _PER_CLIENT:
        L.append("# HELP gateway_client_requests_total Completed requests per client.")
        L.append("# TYPE gateway_client_requests_total counter")
        for name, c in _PER_CLIENT.items():
            L.append(_prom_line("gateway_client_requests_total", c["requests"], {"client": name}))
        # sum/count pairs (not a pre-averaged gauge) -- the standard Prometheus idiom for
        # letting a real query engine compute a correct windowed average via rate(sum)/rate(count).
        L.append("# HELP gateway_client_wait_seconds_sum Sum of admission-wait seconds per client.")
        L.append("# TYPE gateway_client_wait_seconds_sum counter")
        for name, c in _PER_CLIENT.items():
            L.append(_prom_line("gateway_client_wait_seconds_sum", round(c["wait_sum"], 3), {"client": name}))
        L.append("# HELP gateway_client_wait_seconds_count Count of requests contributing to the sum above.")
        L.append("# TYPE gateway_client_wait_seconds_count counter")
        for name, c in _PER_CLIENT.items():
            L.append(_prom_line("gateway_client_wait_seconds_count", c["wait_n"], {"client": name}))
        L.append("# HELP gateway_client_cost_est_usd_total Estimated remote-overflow cost per client.")
        L.append("# TYPE gateway_client_cost_est_usd_total counter")
        for name, c in _PER_CLIENT.items():
            L.append(_prom_line("gateway_client_cost_est_usd_total", round(c["cost_est_usd"], 6), {"client": name}))

    if _SWALLOWED:                                    # lane SH: hidden failures are scrapeable
        L.append("# HELP gateway_swallowed_errors_total Exceptions caught and suppressed at a state/IO site.")
        L.append("# TYPE gateway_swallowed_errors_total counter")
        for site, rec in sorted(_SWALLOWED.items()):
            L.append(_prom_line("gateway_swallowed_errors_total", rec[0], {"site": site}))
    if _ROUTER_CRASHES:
        L.append("# HELP gateway_router_crashes_total Unhandled router exceptions answered with a JSON 500.")
        L.append("# TYPE gateway_router_crashes_total counter")
        for typ, n in sorted(_ROUTER_CRASHES.items()):
            L.append(_prom_line("gateway_router_crashes_total", n, {"type": typ}))

    if _ENGINE_METRICS["ok"]:
        L.append(f"# engine metrics passthrough: last good scrape {round(time.time() - _ENGINE_METRICS['at'], 1)}s ago")
        L.append(_ENGINE_METRICS["text"])
    else:
        age = f"{round(time.time() - _ENGINE_METRICS['at'], 1)}s ago" if _ENGINE_METRICS["at"] else "never"
        L.append(f"# engine metrics unavailable (last good scrape: {age}; error: {_ENGINE_METRICS['err']})")

    return web.Response(text="\n".join(L) + "\n", content_type="text/plain")


# ---- TELEMETRY-HISTORY (added): /gateway/history + /gateway/history/summary, reading the
# JSONL request log written by _jsonl_flusher() above. Both read-only, additive, off-loop (all
# file I/O runs via run_in_executor -- see REPORT.md); neither ever returns a 5xx of its own --
# an internal scan failure degrades to an empty/error-flagged 200, same posture as
# /gateway/telemetry (DESIGN.md (f) point 3). ----
async def gateway_history(request):
    """GET /gateway/history?client=&route=&since=&limit=&page= -- newest-first matches from
    the on-disk request log. `client` matches substring (case-insensitive) against the client
    name; `route` matches substring against "<route> <reason>" (so both "local"/"remote" and a
    reason like "tiny"/"failover" work from the one filter box); `since` accepts a unix
    timestamp, a relative shorthand ('2h','45m','3d'), or 'YYYY-MM-DD[ HH:MM[:SS]]'; `limit`
    defaults to 50, capped at 500; `page` is 1-indexed (default 1). Deep paging is bounded so
    the disk scan can't run away: (page-1)*limit + limit must fit within 500."""
    q = request.query
    client_q = (q.get("client") or "").strip().lower() or None
    route_q = (q.get("route") or "").strip().lower() or None
    since_ts = _parse_since(q.get("since"))
    try:
        limit = max(1, min(int(q.get("limit") or 50), 500))
    except (TypeError, ValueError):
        limit = 50
    try:
        page = max(1, int(q.get("page") or 1))
    except (TypeError, ValueError):
        page = 1
    offset = (page - 1) * limit
    # Hard cap on how far the scan is willing to walk. This is not a UI limit -- it's a disk-
    # budget guard for the paged log-scan. Deep pages that exceed this get clamped to the last
    # valid page.
    HARD_MAX = 500
    if offset >= HARD_MAX:
        offset = max(0, HARD_MAX - limit)
        page = offset // limit + 1
    # Fetch offset+limit+1 to know whether a next page exists without paying for a second scan.
    fetch_n = min(HARD_MAX, offset + limit + 1)
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, _history_scan_blocking, client_q, route_q, since_ts, fetch_n)
    except Exception as e:
        return web.json_response({"error": f"history scan failed: {e}", "items": [], "count": 0}, status=200)
    scanned = result["items"]
    items = scanned[offset:offset + limit]
    has_more = len(scanned) > (offset + limit)
    return web.json_response({
        "at": time.time(), "items": items, "count": len(items),
        "limit": limit, "page": page, "offset": offset, "has_more": has_more,
        "files_scanned": result["files_scanned"],
        "log": {"written": _JSONL_STATE["written"], "dropped_cap": _JSONL_STATE["dropped_cap"],
                "dropped_queue": _JSONL_STATE["dropped_queue"], "capped_today": _JSONL_STATE["capped"],
                "last_err": _JSONL_STATE["last_err"]},
    })


async def gateway_history_summary(request):
    """GET /gateway/history/summary?hours=24 -- per-client/per-route counts, p50/p95
    duration+TTFT, tokens in/out, error counts, and (gw-ttft-decomposition-telemetry) per-class
    (interactive/background) p50/p90/n for admission_wait, queue_plus_prefill and decode_time
    under `latency_by_class` -- over the trailing `hours` (default 24, capped at 30 days). Pass
    hours=1 for the "last hour" window the card asks for. Cached for HISTORY_SUMMARY_CACHE_TTL
    seconds (default 5s): the History dashboard section polls this every 10s and a full scan of
    one-to-several ~200MB/day files on every single poll is real disk+CPU work not worth
    repeating for back-to-back callers -- see REPORT.md disk-math."""
    try:
        hours = float(request.query.get("hours") or 24)
    except (TypeError, ValueError):
        hours = 24.0
    hours = max(0.1, min(hours, 24 * 30))
    cache_key = round(hours, 2)
    now = time.time()
    if _HISTORY_SUMMARY_CACHE["key"] == cache_key and (now - _HISTORY_SUMMARY_CACHE["at"]) < HISTORY_SUMMARY_CACHE_TTL:
        data = _HISTORY_SUMMARY_CACHE["data"]
    else:
        since_ts = now - hours * 3600
        max_files = max(2, min(int(hours // 24) + 2, 32))
        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(None, _history_summary_blocking, since_ts, max_files)
        except Exception as e:
            return web.json_response({"error": f"history summary failed: {e}"}, status=200)
        _HISTORY_SUMMARY_CACHE["key"] = cache_key
        _HISTORY_SUMMARY_CACHE["at"] = now
        _HISTORY_SUMMARY_CACHE["data"] = data
    return web.json_response({"at": now, "hours": hours, **data})


MODELS_DIR = os.environ.get("SHIM_MODELS_DIR", "/home/kevin/Desktop/models")
SWITCH_SH = os.environ.get("SHIM_SWITCH_SCRIPT", "/home/kevin/.local/share/vllm-qwen27b/switch-model.sh")
SERVE_SH  = os.environ.get("SHIM_SERVE_SCRIPT", "/home/kevin/.local/share/vllm-qwen27b/serve-tqk8v4-fg.sh")


def _detect_format(d):
    """Model-agnostic: classify any HF checkpoint dir by servability on THIS box (SM75)."""
    try:
        c = json.load(open(os.path.join(d, "config.json")))
    except Exception:
        return None, "no config.json"
    q = c.get("quantization_config") or {}
    m = (q.get("quant_method") or q.get("method") or "").lower()
    dt = str(c.get("torch_dtype") or c.get("dtype") or "").lower()
    arch = (c.get("architectures") or ["?"])[0]
    if "gptq" in m or "compressed" in m:
        return "gptq_marlin", f"Int4/GPTQ · {arch}"
    if "awq" in m:
        return "awq_marlin", f"Int4/AWQ · {arch}"
    if any(k in m for k in ("fp8", "modelopt", "nvfp4")):
        return None, f"FP8/NVFP4 — BLOCKED on SM75 · {arch}"
    if dt in ("bfloat16", "float16") or not q:
        return None, f"{dt or 'bf16'} unquantized — quantize first · {arch}"
    return None, f"unknown quant '{m}' · {arch}"


def _scan_local_models():
    out = []
    try:
        entries = sorted(os.listdir(MODELS_DIR))
    except Exception:
        return out
    try:
        live = open(SERVE_SH).read()
    except Exception:
        live = ""
    for name in entries:
        d = os.path.join(MODELS_DIR, name)
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "config.json")):
            continue
        quant, desc = _detect_format(d)
        sz = 0
        try:
            for f in os.listdir(d):
                if f.endswith((".safetensors", ".bin", ".gguf")):
                    sz += os.path.getsize(os.path.join(d, f))
        except Exception:
            pass
        out.append({"name": name, "path": d, "servable": quant is not None,
                    "quant": quant, "desc": desc, "gb": round(sz / 1e9, 1),
                    "live": d in live})
    return out


async def gateway_models_local(request):
    if request.method == "GET":
        return web.json_response({"models": _scan_local_models(), "models_dir": MODELS_DIR})
    if not _admin_ok(request):
        return web.json_response({"error": "unauthorized: X-Admin-Token required"}, status=401)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)
    target = (body or {}).get("path") or (body or {}).get("name")
    if not target:
        return web.json_response({"error": "path or name required"}, status=400)
    if not os.path.isabs(target):
        target = os.path.join(MODELS_DIR, target)
    if not os.path.exists(os.path.join(target, "config.json")):
        return web.json_response({"error": f"not a model dir: {target}"}, status=400)
    quant, desc = _detect_format(target)
    if quant is None:
        return web.json_response({"error": f"not servable on this box: {desc}"}, status=400)
    log.warning("MODEL SWITCH requested via dashboard -> %s (%s)", target, quant)

    async def _run():
        p = await asyncio.create_subprocess_exec(
            "/usr/bin/env", "bash", SWITCH_SH, target,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        out, _ = await p.communicate()
        log.warning("MODEL SWITCH finished rc=%s: %s", p.returncode,
                    (out or b"").decode("utf-8", "replace")[-400:])

    asyncio.create_task(_run())
    return web.json_response({"switching_to": target, "quant": quant, "desc": desc,
                              "note": "engine restarting; gateway serves via remote overflow "
                                      "until :8001 is healthy (~40s warm, ~4-5 min cold)"})


async def gateway_config(request):
    if request.method == "GET":
        return web.json_response(current_config(masked=True))
    if not _admin_ok(request):
        return web.json_response({"error": "unauthorized: X-Admin-Token required"}, status=401)
    try:
        fields = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)
    changed = apply_config(fields if isinstance(fields, dict) else {})
    log.info("config updated via dashboard: %s", ",".join(changed) or "(none)")
    return web.json_response({"changed": changed, "config": current_config(masked=True)})


async def gateway_config_effective(request):
    """Lane CFG: every SHIM_* key with its live value, source (env/default/invalid->default/live-edit), default,
    differs-from-default, and what shim.env holds now (what a restart would load). Secrets masked. Read-only.
    ?changed=1 -> only keys that are set, differ, or would change on restart; ?key=SUBSTR filters."""
    if _cfgschema is None:
        return web.json_response({"error": "config schema module not installed beside the gateway"}, status=503)
    try:
        out = _cfgschema.effective(globals(), os.environ, SHIM_ENV_FILE,
                                   only_changed=request.query.get("changed") in ("1", "true"),
                                   key_filter=request.query.get("key") or None)
    except Exception as e:
        return web.json_response({"error": "effective config failed: %r" % (e,)}, status=500)
    return web.json_response(out)


async def gateway_aliases(request):
    """Manage named OpenAI-compatible remote endpoints.

    GET is safe for the dashboard and masks credentials. POST/DELETE require the same admin
    token as other gateway mutations and persist atomically in a mode-600 registry file.
    """
    if request.method == "GET":
        return web.json_response({"builtins": sorted(_BUILTIN_ALIASES),
                                  "aliases": [_public_alias(v) for v in sorted(_ALIASES.values(), key=lambda x: x["name"])]})
    if not _admin_ok(request):
        return web.json_response({"error": "unauthorized: X-Admin-Token required"}, status=401)
    if request.method == "DELETE":
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        name = _alias_name((payload or {}).get("name") or request.query.get("name"))
        if name in _BUILTIN_ALIASES:
            return web.json_response({"error": "built-in aliases cannot be deleted"}, status=400)
        if name not in _ALIASES:
            return web.json_response({"error": "alias not found"}, status=404)
        del _ALIASES[name]
        _save_aliases()
        return web.json_response({"deleted": name})
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)
    try:
        name = (payload or {}).get("name")
        record = _validate_alias_record(name, payload or {})
    except (ValueError, TypeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
    # An omitted key on update means keep the existing credential; an explicit empty key clears it.
    if "key" not in (payload or {}) and record["name"] in _ALIASES:
        record["key"] = _ALIASES[record["name"]].get("key", "")
    _ALIASES[record["name"]] = record
    _save_aliases()
    log.info("gateway alias saved: %s -> %s model=%s", record["name"], record["base"], record["model"])
    return web.json_response({"saved": record["name"], "alias": _public_alias(record)})


async def gateway_aliases_page(request):
    return web.Response(text=ALIASES_HTML, content_type="text/html")

# ---------------- research & lanes aggregator (for the dashboard) ----------------
_LANES_CACHE = {"t": 0.0, "data": None}
# job_id -> "survived/total" (finished jobs never change, so once cached a value is final).
# Bounded to CLAIMS_CACHE_MAX (LRU by insertion order via OrderedDict): the research service
# is long-running and accumulates jobs indefinitely, so an unbounded dict here is a slow memory
# leak in a process that otherwise runs for weeks between restarts.
_RESEARCH_CLAIMS_CACHE = collections.OrderedDict()
CLAIMS_CACHE_MAX = int(os.environ.get("SHIM_CLAIMS_CACHE_MAX", "500"))


def _claims_cache_set(jid, val):
    _RESEARCH_CLAIMS_CACHE[jid] = val
    _RESEARCH_CLAIMS_CACHE.move_to_end(jid)
    while len(_RESEARCH_CLAIMS_CACHE) > CLAIMS_CACHE_MAX:
        _RESEARCH_CLAIMS_CACHE.popitem(last=False)
_QUEUE_DIR = "/home/kevin/.local/share/vllm-qwen27b/frontier-queue"
_RESEARCH_URL = "http://10.0.1.10:8790/research"
_SCRATCH_LANES = "/tmp/claude-1000/-home-kevin-Desktop/bca5cde6-e554-43d7-befb-8acb99b93810/scratchpad"
_LANES_DIRS_FILE = "/home/kevin/.local/share/vllm-qwen27b/lanes.dirs"   # one lane-root dir per line


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _lane_roots():
    roots = []
    try:
        with open(_LANES_DIRS_FILE) as fh:
            roots = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
    except Exception:
        pass
    return roots or [_SCRATCH_LANES]


_CLAUDE_PROJECTS = "/home/kevin/.claude/projects"


def _jsonl_first_last(path, head=20000, tail=60000):
    """First JSON line and the last user/assistant JSON line of a transcript, reading only both ends."""
    first, last = None, None
    try:
        with open(path, "rb") as fh:
            try:
                first = json.loads(fh.readline(head))
            except Exception:
                first = None
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - tail))
            chunk = fh.read()
        for ln in reversed([l for l in chunk.split(b"\n") if l.strip()]):
            try:
                d = json.loads(ln)
            except Exception:
                continue
            if d.get("type") in ("user", "assistant"):
                last = d
                break
    except Exception:
        pass
    return first, last


def _msg_parts(d):
    c = ((d or {}).get("message") or {}).get("content")
    if isinstance(c, str):
        return [{"type": "text", "text": c}]
    return [p for p in (c or []) if isinstance(p, dict)]


def _agent_label(first, meta):
    text = ""
    for p in _msg_parts(first):
        if p.get("type") == "text" and p.get("text"):
            text = p["text"]
            break
    m = re.search(r"LANE=(/\S+)", text)
    lane = os.path.basename(m.group(1).rstrip("/")) if m else None
    t = re.search(r"\bTASK\b[^:\n]*:\s*(.+)", text)
    task = (t.group(1) if t else text.strip().split("\n")[0]).strip()
    desc = (meta or {}).get("description") or ""
    return (desc or lane or task[:120]), task[:240]


def _agent_step(last):
    """What the agent is doing right now, from its newest transcript line."""
    if not last:
        return ""
    if last.get("type") == "assistant":
        parts = _msg_parts(last)
        for p in parts:
            if p.get("type") == "tool_use":
                inp = p.get("input") or {}
                arg = (inp.get("description") or inp.get("command") or inp.get("file_path")
                       or inp.get("query") or inp.get("pattern") or inp.get("prompt") or inp.get("path") or "")
                return f"{p.get('name')}: {str(arg).strip()[:130]}"
        for p in parts:
            if p.get("type") == "text" and p.get("text"):
                return "wrote: " + p["text"].strip()[:130]
        return "thinking…"
    return "waiting on a tool result…"


def _scan_claude_agents(now, max_age_s=86400, cap=40):
    """Claude Code background agents on this box: one row per subagent transcript touched in the last 24 h."""
    found = []
    try:
        for proj in os.listdir(_CLAUDE_PROJECTS):
            pdir = os.path.join(_CLAUDE_PROJECTS, proj)
            if not os.path.isdir(pdir):
                continue
            for sess in os.listdir(pdir):
                sdir = os.path.join(pdir, sess, "subagents")
                if not os.path.isdir(sdir):
                    continue
                for f in os.listdir(sdir):
                    if f.startswith("agent-") and f.endswith(".jsonl"):
                        fp = os.path.join(sdir, f)
                        try:
                            st = os.stat(fp)
                        except OSError:
                            continue
                        age = now - st.st_mtime
                        if age <= max_age_s:
                            found.append((age, fp, st, sess, proj, f[6:-6]))
    except Exception:
        return []
    found.sort(key=lambda x: x[0])
    res = []
    for age, fp, st, sess, proj, aid in found[:cap]:
        meta = {}
        try:
            with open(fp[:-6] + ".meta.json") as fh:
                meta = json.load(fh)
        except Exception:
            pass
        first, last = _jsonl_first_last(fp)
        label, task = _agent_label(first, meta)
        res.append({"id": aid, "session": sess[:8], "project": proj.replace("-home-kevin-", "~/").strip("-"),
                    "label": label, "task": task, "step": _agent_step(last),
                    "state": "working" if age < 20 else ("active" if age < 120 else "finished"),
                    "age_s": int(age), "started": (first or {}).get("timestamp"), "size": st.st_size,
                    "model": meta.get("model")})
    return res


def _scan_claude_sessions(now, max_age_s=6 * 3600):
    """Claude Code sessions (main transcripts) touched in the last 6 h, freshest first."""
    res = []
    try:
        for proj in os.listdir(_CLAUDE_PROJECTS):
            pdir = os.path.join(_CLAUDE_PROJECTS, proj)
            if not os.path.isdir(pdir):
                continue
            for f in os.listdir(pdir):
                if f.endswith(".jsonl"):
                    try:
                        st = os.stat(os.path.join(pdir, f))
                    except OSError:
                        continue
                    age = now - st.st_mtime
                    if age <= max_age_s:
                        res.append({"id": f[:8], "project": proj.replace("-home-kevin-", "~/").strip("-"),
                                    "age_s": int(age), "size": st.st_size,
                                    "state": "working" if age < 20 else ("active" if age < 300 else "idle")})
    except Exception:
        pass
    res.sort(key=lambda x: x["age_s"])
    return res[:12]


_WATCHDOG_DIR = "/home/kevin/.local/share/vllm-qwen27b/watchdog"


def _parse_ts_epoch(ts, now):
    """Seconds since `ts`, which may be an epoch number or an ISO-8601 string (Z or offset). None if unparseable."""
    if ts is None:
        return None
    try:
        return int(now - float(ts))
    except (TypeError, ValueError):
        pass
    try:
        import datetime as _dt
        t = str(ts).strip().replace("Z", "+00:00")
        d = _dt.datetime.fromisoformat(t)
        if d.tzinfo is None:
            d = d.replace(tzinfo=_dt.timezone.utc)
        return int(now - d.timestamp())
    except Exception:
        return None


def _read_watchdog(now):
    """Latest estate-watchdog result: {ts, age_s, overall, checks:[{name,status,detail,ms}], alert, paused}."""
    h = {"present": False}
    try:
        with open(os.path.join(_WATCHDOG_DIR, "state.json")) as fh:
            st = json.load(fh)
        h.update(present=True, overall=st.get("overall"), ts=st.get("ts"),
                 checks=(st.get("checks") or [])[:40])
        h["age_s"] = _parse_ts_epoch(st.get("ts"), now)
    except Exception:
        pass
    try:
        with open(os.path.join(_WATCHDOG_DIR, "ALERT"), errors="replace") as fh:
            h["alert"] = fh.read(2000).strip()
    except Exception:
        h["alert"] = None
    h["paused"] = os.path.exists(os.path.join(_WATCHDOG_DIR, "PAUSE"))
    return h


def _collect_local_lanes(now, out, dev=False):
    """Blocking filesystem scan (runs in the default executor, never on the event loop)."""
    try:
        out["health"] = _read_watchdog(now)
    except Exception as exc:
        out["errors"].append(f"health:{type(exc).__name__}")
    if dev:
        # Anthropic Claude Code subagents/sessions are DEV tooling, not part of the local product:
        # only exposed on /gateway/lanes?dev=1 for oversight, never on the dashboard.
        try:
            out["agents"] = _scan_claude_agents(now)
            out["sessions"] = _scan_claude_sessions(now)
        except Exception as exc:
            out["errors"].append(f"agents:{type(exc).__name__}")
    # frontier queue: the runner pops *.sh from queue/, flags done/<name>.running while a window runs
    try:
        qd = os.path.join(_QUEUE_DIR, "queue")
        dn = os.path.join(_QUEUE_DIR, "done")
        rs = os.path.join(_QUEUE_DIR, "results")
        out["queue"]["queued"] = sorted(f for f in (os.listdir(qd) if os.path.isdir(qd) else []) if f.endswith(".sh"))
        out["queue"]["running"] = [f for f in (os.listdir(dn) if os.path.isdir(dn) else []) if f.endswith(".running")]
        out["queue"]["paused"] = os.path.exists(os.path.join(_QUEUE_DIR, "PAUSE"))
        res = []
        if os.path.isdir(rs):
            files = [(f, os.path.getmtime(os.path.join(rs, f))) for f in os.listdir(rs) if f.endswith(".txt")]
            for f, mt in sorted(files, key=lambda x: -x[1])[:6]:
                try:
                    with open(os.path.join(rs, f), "rb") as fh:
                        tail = fh.read()[-400:].decode("utf-8", "replace")
                    lines = [l for l in tail.splitlines() if l.strip()][-3:]
                except Exception:
                    lines = []
                res.append({"file": f, "age_s": int(now - mt), "tail": lines})
        out["queue"]["results"] = res
    except Exception as exc:
        out["errors"].append(f"queue:{type(exc).__name__}")
    # agent lanes: every immediate subdir of each lane root; a STATUS file's first line is the heartbeat
    try:
        for root in _lane_roots():
            if not os.path.isdir(root):
                continue
            for d in sorted(os.listdir(root)):
                dp = os.path.join(root, d)
                if not os.path.isdir(dp) or d.startswith("."):
                    continue
                newest, newest_mt, status = None, 0.0, None
                for f in os.listdir(dp):
                    fp = os.path.join(dp, f)
                    if os.path.isfile(fp):
                        mt = os.path.getmtime(fp)
                        if mt > newest_mt:
                            newest, newest_mt = f, mt
                sp = os.path.join(dp, "STATUS")
                if os.path.isfile(sp):
                    try:
                        # heartbeat = the LAST non-empty line (workers may append instead of overwrite)
                        with open(sp, "rb") as fh:
                            fh.seek(0, 2); size = fh.tell(); fh.seek(max(0, size - 4096))
                            tail = fh.read().decode("utf-8", "replace")
                        lines = [l.strip() for l in tail.splitlines() if l.strip()]
                        status = (lines[-1] if lines else "")[:200] or None
                    except Exception:
                        status = None
                if newest:
                    out["lanes"].append({"lane": d, "root": root, "newest": newest,
                                          "age_s": int(now - newest_mt), "status": status})
        # one row per lane name across roots: prefer the copy with a STATUS heartbeat, then the freshest
        best = {}
        for l in out["lanes"]:
            cur = best.get(l["lane"])
            if (cur is None or (l["status"] and not cur["status"])
                    or (bool(l["status"]) == bool(cur["status"]) and l["age_s"] < cur["age_s"])):
                best[l["lane"]] = l
        out["lanes"] = sorted(best.values(), key=lambda x: x["age_s"])
    except Exception as exc:
        out["errors"].append(f"lanes:{type(exc).__name__}")


async def gateway_research_detail(request):
    """Read-only proxy: one research job's full report (HTML) or ?raw=1 for the JSON."""
    jid = request.match_info["jid"]
    if len(jid) > 64 or not all(c.isalnum() or c in "-_" for c in jid):
        raise web.HTTPBadRequest(text="bad job id")
    try:
        timeout = aiohttp.ClientTimeout(total=8)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.get(f"{_RESEARCH_URL}/{jid}") as r:
                if r.status != 200:
                    return web.Response(status=r.status, text=f"research service returned {r.status}")
                d = await r.json()
    except Exception as exc:
        return web.Response(status=502, text=f"research service unreachable: {type(exc).__name__}")
    if request.query.get("raw") == "1":
        return web.json_response(d)
    res = d.get("result") or {}
    md = res if isinstance(res, str) else (res.get("report_md") or json.dumps(res, indent=1))
    # cap unconditionally: only the json.dumps fallback above was bounded, so a huge string
    # result or a huge report_md from the (trusted but not size-limited) research service could
    # otherwise render an unbounded HTML page.
    md = md[:200000]

    def esc(s):
        return str("" if s is None else s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    meta = " · ".join(f"{k} {esc(d.get(k))}" for k in ("status", "phase", "depth", "submitted", "ended",
                                                        "elapsed", "agents_total", "tokens_total")
                      if d.get(k) is not None)
    html = ("<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>research {esc(jid)}</title>"
            "<style>body{background:#0d1117;color:#e6edf3;font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;max-width:1000px;margin:0 auto;padding:18px}"
            "pre{white-space:pre-wrap;word-wrap:break-word;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}"
            "a{color:#58a6ff}.dim{color:#8b949e;font-size:12px}h2{font-size:15px}</style></head><body>"
            f"<div class=dim><a href=/gateway/dashboard>&larr; dashboard</a> · job {esc(jid)} · {meta} · <a href='?raw=1'>raw json</a></div>"
            f"<h2>{esc(d.get('question'))}</h2><pre>{esc(md)}</pre></body></html>")
    return web.Response(text=html, content_type="text/html")


_LANE_STAGE_RE = re.compile(r"^(RUNNING|DONE|BLOCKED)\s*\|\s*([^|]*)\|\s*(.*)$", re.ASCII)
_LANE_TERMINAL_RESEARCH = frozenset({"done", "failed", "error", "degraded", "cancelled"})


def lane_state(status, age_s, newest=None):
    """Server-side mirror of DashLib.laneStage() in gateway_dashboard.html (a test compares the two on a table of
    statuses). A lane's STATUS heartbeat is free text written by many agents; classify on the first 90 characters
    so a long note that merely mentions a word later does not change its state.
    -> run | stale | done | blocked | empty"""
    stt = status or ""
    m = _LANE_STAGE_RE.match(stt)
    head = stt[:90]
    if m:
        stg = m.group(1)
    elif re.search(r"\bSTALE\b", head, re.ASCII):
        stg = "STALE"
    elif re.search(r"\bBLOCKED\b", head, re.ASCII):
        stg = "BLOCKED"
    elif re.search(r"\bDONE\b|\bAPPLIED\b", head, re.ASCII):
        stg = "DONE"
    else:
        stg = "RUNNING" if stt else "UNKNOWN"
    stale = stg == "STALE" or (stg == "RUNNING" and (age_s or 0) > 900)
    if stg == "DONE":
        return "done"
    if stg == "BLOCKED":
        return "blocked"
    if stale:
        return "stale"
    return "run" if stg == "RUNNING" else "empty"


def _q_int(q, name, default, lo=0, hi=5000):
    try:
        v = int(str(q.get(name)).strip())
    except (TypeError, ValueError):
        return default
    if v < lo:
        return default
    return min(hi, v)


def _lanes_view(data, q):
    """Trim a /gateway/lanes payload server-side (Lane DB2: the full feed is ~400 KB -- 1,200 agent-lane rows and 200
    research rows -- and the dashboard polls it every 5 s).

      (no parameters)      the full list, exactly as before, plus a `summary` object
      ?active=1            only what is going on: lanes not done and not silent for over `max_age_s` (default 1 day),
                           research jobs that have not finished
      ?limit=N             at most N lanes, freshest first (default 200 with active=1, unlimited otherwise)
      ?research_limit=N    at most N research rows, newest first
      ?max_age_s=S         lane age cutoff for active=1

    `summary` always carries the totals (by state, by age, research counts), so a trimmed reader still knows what it did
    not receive. The cached payload is never modified."""
    lanes_all = data.get("lanes") or []
    research_all = data.get("research") or []
    active = str(q.get("active", "")).lower() in ("1", "true", "yes")
    has_limit = "limit" in q
    has_rlimit = "research_limit" in q
    by_state = collections.Counter()
    by_age = {"le_5m": 0, "le_1h": 0, "le_1d": 0, "le_7d": 0, "older": 0}
    states = []
    for l in lanes_all:
        st = lane_state(l.get("status"), l.get("age_s"), l.get("newest"))
        states.append(st)
        by_state[st] += 1
        a = l.get("age_s") or 0
        by_age["le_5m" if a <= 300 else "le_1h" if a <= 3600 else "le_1d" if a <= 86400 else "le_7d" if a <= 604800 else "older"] += 1
    max_age = _q_int(q, "max_age_s", 86400, lo=0, hi=10 * 365 * 86400)
    lanes, research = lanes_all, research_all
    if active:
        lanes = [l for l, st in zip(lanes_all, states) if st != "done" and (l.get("age_s") or 0) <= max_age]
        research = [j for j in research_all if j.get("status") not in _LANE_TERMINAL_RESEARCH]
    lim = _q_int(q, "limit", 200 if active else 0)
    if lim:
        lanes = lanes[:lim]
    rlim = _q_int(q, "research_limit", 0)
    if rlim or has_rlimit:
        research = research[:rlim]
    out = dict(data)
    out["lanes"], out["research"] = lanes, research
    out["summary"] = {
        "form": "light" if (active or has_limit or has_rlimit) else "full",
        "filter": {"active": active, "limit": lim or None, "research_limit": rlim if has_rlimit else None,
                   "max_age_s": max_age if active else None},
        "lanes_total": len(lanes_all), "lanes_returned": len(lanes), "lanes_truncated": len(lanes) < len(lanes_all),
        "lanes_by_state": {k: by_state.get(k, 0) for k in ("run", "stale", "blocked", "empty", "done")},
        "lanes_by_age": by_age,
        "research_total": len(research_all), "research_returned": len(research),
        "research_running": sum(1 for j in research_all if j.get("status") not in _LANE_TERMINAL_RESEARCH),
    }
    return out


async def gateway_lanes(request):
    """Read-only aggregate of ongoing work: research jobs (remote service),
    frontier-queue windows, and agent-lane scratchpad artifacts. Cached 5s.

    Top-level guarantee: this handler must NEVER raise into aiohttp — it shares the process
    with the /v1 proxy, and a dashboard-polling endpoint misbehaving must not be able to take
    the front door down or wedge a worker. Every failure mode below degrades to a 200/500 JSON
    body with an `errors` list instead."""
    try:
        return await _gateway_lanes_impl(request)
    except Exception as exc:
        log.warning("gateway_lanes: unhandled %s: %s", type(exc).__name__, exc)
        return web.json_response(
            {"error": f"gateway/lanes failed: {type(exc).__name__}: {exc}", "errors": [str(exc)]},
            status=500)


async def _gateway_lanes_impl(request):
    now = time.time()
    dev = request.query.get("dev") == "1"
    if not dev and _LANES_CACHE["data"] is not None and now - _LANES_CACHE["t"] < 2:
        data = dict(_LANES_CACHE["data"])
        data["active"] = _active_snapshot(now)          # live requests are never served stale
        data.update(inflight=_inflight, waiting=_waiting, budget=effective_budget())
        return web.json_response(_lanes_view(data, request.query))
    out = {"ts": now, "research": [], "research_counts": {}, "queue": {}, "lanes": [],
           "active": _active_snapshot(now), "inflight": _inflight, "waiting": _waiting,
           "budget": effective_budget(), "errors": []}
    # research jobs — ALL of them (service default list is capped at 20; limit=500 returns everything).
    # Never let the remote call hurt the gateway: bounded timeout, errors reported not raised.
    try:
        timeout = aiohttp.ClientTimeout(total=4)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.get(_RESEARCH_URL, params={"limit": "500"}) as r:
                if r.status == 200:
                    jobs = (await r.json()).get("jobs", [])
                    for j in jobs:
                        q = (j.get("question") or "").strip()
                        if len(q) < 12:      # research-service self-test probes ("q", "What is X?")
                            continue
                        out["research"].append({
                            "id": j.get("job_id"),
                            "status": j.get("status"),
                            "phase": j.get("phase"),
                            "progress": j.get("phase_progress"),
                            "depth": j.get("depth"),
                            "submitted": j.get("submitted"),
                            "ended": j.get("ended"),
                            "elapsed_s": _to_float(j.get("elapsed")),
                            "agents": j.get("agents_total"),
                            "tokens": j.get("tokens_total"),
                            "q": q[:400],
                        })
                    out["research"].sort(key=lambda x: x.get("submitted") or "", reverse=True)
                    out["research_counts"] = dict(collections.Counter(x["status"] for x in out["research"]))
                else:
                    out["errors"].append(f"research:{r.status}")
            # Claim counts (survived/total) live only in the per-job detail; finished jobs are immutable,
            # so fetch a handful per refresh and cache them for the life of the process.
            pending = [x for x in out["research"]
                       if x["status"] in ("done", "failed", "error") and x["id"]
                       and x["id"] not in _RESEARCH_CLAIMS_CACHE][:8]

            async def _claims(jid):
                try:
                    async with sess.get(f"{_RESEARCH_URL}/{jid}") as rr:
                        if rr.status != 200:
                            return
                        res = (await rr.json()).get("result") or {}
                    if isinstance(res, dict) and "claims_total" in res:
                        _claims_cache_set(jid, f"{res.get('claims_survived', '?')}/{res.get('claims_total', '?')}")
                    else:
                        _claims_cache_set(jid, "-")
                except Exception:
                    pass

            if pending:
                await asyncio.gather(*(_claims(x["id"]) for x in pending))
            for x in out["research"]:
                x["claims"] = _RESEARCH_CLAIMS_CACHE.get(x["id"])
    except Exception as exc:
        out["errors"].append(f"research:{type(exc).__name__}")
    # frontier queue + agent lanes: blocking filesystem work off the event loop, bounded so a
    # stuck scan (an unreachable NFS lane root, a huge queue dir) can't hang this endpoint
    # forever. run_in_executor cannot be cancelled once started — a straggling thread keeps
    # running to completion in the background and is simply discarded — so it is handed its
    # OWN dict, never `out`, which we may serialize and return before that thread finishes;
    # writing into a shared dict here would be a cross-thread race on the response we already sent.
    lanes_out = {"queue": {}, "lanes": [], "errors": [], "health": {"present": False}}
    if dev:
        lanes_out["agents"] = []
        lanes_out["sessions"] = []
    try:
        await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(None, _collect_local_lanes, now, lanes_out, dev),
            timeout=4.0)
        out["queue"] = lanes_out["queue"]
        out["lanes"] = lanes_out["lanes"]
        if dev:
            out["agents"] = lanes_out.get("agents", [])
            out["sessions"] = lanes_out.get("sessions", [])
        out["errors"].extend(lanes_out["errors"])
        out["health"] = lanes_out.get("health", {"present": False})   # estate-watchdog state (health strip)
    except asyncio.TimeoutError:
        out["errors"].append("lanes:timeout>4.0s")
    if not dev:
        _LANES_CACHE["t"] = now
        _LANES_CACHE["data"] = out
    return web.json_response(_lanes_view(out, request.query))


# ---- WINDOWS (added): /gateway/windows + /gateway/windows/{name}/log -- a per-window drill-down
# of the frontier-queue engine-benchmark chain (queue/, done/, results/*.txt), richer than the
# one-line-per-window summary _collect_local_lanes already puts on the Background-tasks card
# above. Read-only, additive, off-loop, same never-raise posture as /gateway/lanes: a parse
# failure on one window's log degrades that window to its raw tail; it never takes the whole
# endpoint down. Cached WINDOWS_CACHE_TTL seconds. ----
WINDOWS_CACHE_TTL = float(os.environ.get("SHIM_WINDOWS_CACHE_TTL", "5"))
_WINDOWS_CACHE = {"t": 0.0, "data": None}
_WIN_RESULT_RE = re.compile(r"\$OUT/([A-Za-z0-9_.\-]+\.(?:txt|log))")
_WIN_RESULT_RE2 = re.compile(r"frontier-queue/results/([A-Za-z0-9_.\-]+\.(?:txt|log))")
_WIN_ARM_RESULT_RE = re.compile(r"^\[([^\]\n]+)\]\s*RESULT\s+(.+)$", re.M)
_WIN_KV_RE = re.compile(r"(\w+)=(\S+)")
_WIN_EVALKIT_RE = re.compile(r"^\[([^\]\n]+)\][^\n]*?\bevalkit\b(.*)$", re.M | re.I)
_WIN_EVALKIT_SCORE_RE = re.compile(r"(?<![\d=>])(\d+)\s*/\s*45")
_WIN_EVALKIT_MEDIAN_RE = re.compile(r"\bmedian[=\s]+(\d+)")
_WIN_POOL_RE = re.compile(r"^\[([^\]\n]+)\][^\n]*?GPU KV cache size:\s*([\d,]+)\s*tokens", re.M)
_WIN_VRAM_RE = re.compile(r"^\[vram\]\s*(.+)$", re.M)
_WIN_DONE_RE = re.compile(r"^([A-Z][A-Z0-9 _\-]{0,80}\bDONE)\s*$", re.M)
_WIN_STARTED_RE = re.compile(r"\[window\][^\n]*?\bstarted at (\S+)")
_WIN_FINISHED_RE = re.compile(r"\[window\][^\n]*?\bfinished at (\S+)")
_WIN_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,120}$")
_WIN_EMPTY = {"arms": {}, "evalkit": {}, "pool": {}, "vram": [], "verdict_lines": [],
              "done_marker": None, "started_at": None, "finished_at": None, "failed": False}


def _window_header(text):
    """First comment block of a queue/done script, trimmed: shebang dropped, pure '===' / '---'
    decorator lines dropped, remaining '#' lines joined with spaces. Never raises."""
    try:
        lines = text.splitlines()
        if lines and lines[0].startswith("#!"):
            lines = lines[1:]
        out = []
        for ln in lines:
            s = ln.strip()
            if not s.startswith("#"):
                break
            body = s[1:].strip()
            if body and set(body) <= set("=-"):
                continue
            if body:
                out.append(body)
        return " ".join(out)[:4000]
    except Exception:
        return ""


def _window_result_name(text):
    try:
        m = _WIN_RESULT_RE.search(text) or _WIN_RESULT_RE2.search(text)
        return m.group(1) if m else None
    except Exception:
        return None


def _win_parse_iso(ts):
    """Epoch seconds for an ISO-8601 timestamp (as written by `date -Is`), or None."""
    if not ts:
        return None
    try:
        import datetime as _dt
        t = str(ts).strip()
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        return _dt.datetime.fromisoformat(t).timestamp()
    except Exception:
        return None


def _parse_window_log(text):
    """Tolerant metrics extraction from one window's result .txt. No exception escapes: any
    regex/parse hiccup just leaves that piece at its default; the raw tail (computed separately
    by _window_tail) always still renders, per the 'unknown shapes fall back to raw tails' rule."""
    out = {k: (v.copy() if isinstance(v, (dict, list)) else v) for k, v in _WIN_EMPTY.items()}
    if not text:
        return out
    try:
        for m in _WIN_ARM_RESULT_RE.finditer(text):
            arm, rest = m.group(1).strip(), m.group(2)
            kv = {}
            for k, v in _WIN_KV_RE.findall(rest):
                try:
                    kv[k] = float(v) if re.match(r"^-?\d+(\.\d+)?$", v) else v
                except Exception:
                    kv[k] = v
            out["arms"].setdefault(arm, {}).update(kv)
    except Exception:
        pass
    try:
        for m in _WIN_EVALKIT_RE.finditer(text):
            arm, rest = m.group(1).strip(), m.group(2).strip(" :-")
            e = out["evalkit"].setdefault(arm, {"scores": [], "lines": []})
            e["lines"].append(rest[:200])
            # A real score is "S/45" (e.g. "evalkit: 45/45", "run 2: 44/45") or "median=S". The
            # (?<![\d=>]) guard specifically excludes the category-bar narrative line's "bar
            # >=44/45" -- that 44 is the PASS BAR, not a measured score, and matching it would
            # show up as a fake extra evalkit result (it has no leading digit/'='/'>' before it
            # only because "bar >=" ends in '='/'>', which the guard checks for).
            sc = _WIN_EVALKIT_SCORE_RE.search(rest) or _WIN_EVALKIT_MEDIAN_RE.search(rest)
            if sc:
                e["scores"].append(int(sc.group(1)))
    except Exception:
        pass
    try:
        for m in _WIN_POOL_RE.finditer(text):
            arm, tok = m.group(1).strip(), m.group(2)
            try:
                out["pool"][arm] = int(tok.replace(",", ""))
            except Exception:
                out["pool"][arm] = tok
    except Exception:
        pass
    try:
        out["vram"] = [m.group(1).strip()[:200] for m in _WIN_VRAM_RE.finditer(text)][-8:]
    except Exception:
        pass
    try:
        vlines = []
        for ln in text.splitlines():
            s = ln.strip()
            if s and (s.startswith("[verdict]") or re.search(r"\bGATES?\s+(GREEN|FAILED)\b", s, re.I)
                      or re.search(r"\bno-ship\b", s, re.I) or re.search(r"\bSHIP\b", s)):
                vlines.append(s[:300])
        out["verdict_lines"] = vlines[-8:]
    except Exception:
        pass
    try:
        dm = _WIN_DONE_RE.search(text)
        out["done_marker"] = dm.group(1).strip() if dm else None
    except Exception:
        pass
    try:
        out["failed"] = bool(("FAILED" in text or "CRASH" in text) and not out["done_marker"])
    except Exception:
        pass
    try:
        sm = _WIN_STARTED_RE.search(text)
        out["started_at"] = sm.group(1) if sm else None
        fm = _WIN_FINISHED_RE.search(text)
        out["finished_at"] = fm.group(1) if fm else None
    except Exception:
        pass
    return out


def _window_tail(text, n=5):
    try:
        return [l for l in text.splitlines() if l.strip()][-n:]
    except Exception:
        return []


def _new_window_entry(stem, state, script_path):
    return {"name": stem, "script": os.path.basename(script_path), "state": state,
            "description": "", "result_file": None, "started_at": None, "finished_at": None,
            "duration_s": None, "sort_ts": 0.0, "arms": {}, "evalkit": {}, "pool": {},
            "vram": [], "verdict_lines": [], "done_marker": None, "failed": False,
            "tail": [], "log_url": None}


def _build_window_entry(stem, state, script_path, results_dir, now):
    """One window's full entry: header off its script, metrics off its matched results/*.txt (if
    one exists yet). Every sub-step degrades gracefully -- see _parse_window_log."""
    w = _new_window_entry(stem, state, script_path)
    try:
        w["sort_ts"] = os.stat(script_path).st_mtime
    except Exception:
        pass
    try:
        with open(script_path, "r", errors="replace") as fh:
            script_text = fh.read(60000)
    except Exception:
        script_text = ""
    w["description"] = _window_header(script_text)
    rname = _window_result_name(script_text)
    if not rname:
        return w
    w["result_file"] = rname
    rpath = os.path.join(results_dir, rname)
    if not os.path.isfile(rpath):
        return w                      # named in the script, just not written yet (fresh queued window)
    w["log_url"] = f"/gateway/windows/{stem}/log"
    try:
        rst = os.stat(rpath)
        with open(rpath, "r", errors="replace") as fh:
            text = fh.read(500000)
    except Exception:
        return w
    parsed = _parse_window_log(text)
    for k in ("arms", "evalkit", "pool", "vram", "verdict_lines", "done_marker", "failed",
              "started_at", "finished_at"):
        w[k] = parsed[k]
    w["tail"] = _window_tail(text)
    # Only trust real timestamps parsed out of the log for duration -- st_ctime is NOT creation
    # time on Linux (it's "inode last changed", which converges to st_mtime for a file that gets
    # written repeatedly through the run), so using it as a start-time guess silently produced a
    # ~0s duration for any window whose script doesn't emit "[window] started at ..." itself
    # (e.g. the older f1-gate-rN family). Unknown start => duration_s stays None (honest) rather
    # than a plausible-looking wrong number; st_mtime is still fine as a FINISH proxy (last write
    # really is close to "when it stopped") and, separately, as a sort key below.
    s_ep = _win_parse_iso(parsed["started_at"])
    f_ep = _win_parse_iso(parsed["finished_at"])
    if f_ep is None and (parsed["done_marker"] or state == "done"):
        f_ep = rst.st_mtime           # best-effort finish proxy: the log's last write
    if s_ep and f_ep and f_ep >= s_ep:
        w["duration_s"] = round(f_ep - s_ep, 1)
    elif s_ep and state == "running":
        w["duration_s"] = round(max(0.0, now - s_ep), 1)
    w["sort_ts"] = max(w["sort_ts"], f_ep or 0.0, s_ep or 0.0, rst.st_mtime)
    return w


def _collect_windows_blocking(now, out):
    """Blocking filesystem scan (runs in the default executor, never on the event loop). One
    entry per queue/done script -- queued (queue/*.sh), running (done/*.sh.running), or done
    (done/*.sh) -- matched to its results/*.txt via the script's own `LOG=$OUT/<name>` line (the
    one convention every frontier-queue window script follows; see REPORT.md). A .running copy
    always wins over a same-named .sh in done/ (shouldn't coexist, but be deterministic if it
    ever does). Every window is built by _build_window_entry with its own try/except fallback
    below, so one unreadable script or malformed log degrades that window, never the whole scan."""
    qd = os.path.join(_QUEUE_DIR, "queue")
    dn = os.path.join(_QUEUE_DIR, "done")
    rs = os.path.join(_QUEUE_DIR, "results")
    seen = {}
    try:
        for f in (os.listdir(qd) if os.path.isdir(qd) else []):
            if f.endswith(".sh"):
                seen[f[:-3]] = ("queued", os.path.join(qd, f))
    except Exception as exc:
        out["errors"].append(f"windows:queue:{type(exc).__name__}")
    try:
        entries = os.listdir(dn) if os.path.isdir(dn) else []
    except Exception as exc:
        out["errors"].append(f"windows:done:{type(exc).__name__}")
        entries = []
    for f in entries:
        if f.endswith(".sh.running"):
            seen[f[: -len(".sh.running")]] = ("running", os.path.join(dn, f))
    for f in entries:
        if f.endswith(".sh") and not f.endswith(".sh.running"):
            stem = f[:-3]
            if seen.get(stem, (None,))[0] != "running":
                seen[stem] = ("done", os.path.join(dn, f))
    windows = []
    for stem, (state, script_path) in seen.items():
        try:
            windows.append(_build_window_entry(stem, state, script_path, rs, now))
        except Exception as exc:
            out["errors"].append(f"windows:{stem}:{type(exc).__name__}")
            windows.append(_new_window_entry(stem, state, script_path))
    windows.sort(key=lambda w: w["sort_ts"], reverse=True)
    out["windows"] = windows


async def gateway_windows(request):
    """Read-only per-window drill-down for the frontier-queue engine-benchmark chain. Same
    never-raise posture as /gateway/lanes: any unhandled failure degrades to a 200/500 JSON body
    with an `errors` list, never a stack trace to the dashboard poller."""
    try:
        return await _gateway_windows_impl(request)
    except Exception as exc:
        log.warning("gateway_windows: unhandled %s: %s", type(exc).__name__, exc)
        return web.json_response(
            {"error": f"gateway/windows failed: {type(exc).__name__}: {exc}", "windows": [],
             "errors": [str(exc)]}, status=500)


async def _gateway_windows_impl(request):
    now = time.time()
    if _WINDOWS_CACHE["data"] is not None and now - _WINDOWS_CACHE["t"] < WINDOWS_CACHE_TTL:
        return web.json_response(_WINDOWS_CACHE["data"])
    # own dict, never the cached one -- run_in_executor cannot be cancelled once started, so a
    # straggling thread past the timeout keeps writing into whatever dict it was handed; see the
    # identical note on _gateway_lanes_impl above.
    win_out = {"windows": [], "errors": []}
    try:
        await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(None, _collect_windows_blocking, now, win_out),
            timeout=4.0)
    except asyncio.TimeoutError:
        win_out["errors"].append("windows:timeout>4.0s")
    out = {"ts": now, "windows": win_out["windows"], "errors": win_out["errors"]}
    _WINDOWS_CACHE["t"] = now
    _WINDOWS_CACHE["data"] = out
    return web.json_response(out)


def _read_window_log_blocking(name):
    """Resolve <name> (a window's script stem) to its results/*.txt via the same LOG=$OUT/...
    convention used everywhere else, then return its content capped at 200 KB. None if the
    window or its log doesn't exist."""
    qd = os.path.join(_QUEUE_DIR, "queue")
    dn = os.path.join(_QUEUE_DIR, "done")
    rs = os.path.join(_QUEUE_DIR, "results")
    for path in (os.path.join(dn, name + ".sh.running"), os.path.join(qd, name + ".sh"),
                 os.path.join(dn, name + ".sh")):
        if os.path.isfile(path):
            try:
                with open(path, "r", errors="replace") as fh:
                    rname = _window_result_name(fh.read(60000))
            except Exception:
                return None
            if not rname:
                return None
            rpath = os.path.join(rs, rname)
            if not os.path.isfile(rpath):
                return None
            try:
                with open(rpath, "r", errors="replace") as fh:
                    return fh.read(200000)
            except Exception:
                return None
    return None


async def gateway_window_log(request):
    """GET /gateway/windows/<name>/log -- the full result .txt for one window, text/plain,
    capped at 200 KB (these are the small per-window result logs, not the multi-hundred-KB
    companion server-transcript *.log files some windows also leave in results/)."""
    name = request.match_info.get("name", "")
    if not _WIN_NAME_RE.match(name):
        raise web.HTTPBadRequest(text="bad window name")
    try:
        text = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(None, _read_window_log_blocking, name),
            timeout=4.0)
    except asyncio.TimeoutError:
        return web.Response(status=504, text="timed out reading log")
    except Exception as exc:
        return web.Response(status=500, text=f"log read failed: {type(exc).__name__}: {exc}")
    if text is None:
        raise web.HTTPNotFound(text="no result file for this window")
    return web.Response(text=text, content_type="text/plain")


# The dashboard page lives in its own file (deploy/bin/gateway_dashboard.html, installed next to this script by
# gateway_safe_publish.py). It is read per request and cached by mtime, so a page-only change needs no restart.
# DASHBOARD_HTML below is the legacy inline copy, served only if the file is missing or unreadable.
DASHBOARD_FILE = os.environ.get("SHIM_DASHBOARD_FILE") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "gateway_dashboard.html")
_DASH_CACHE = {"mtime": None, "path": None, "text": None}


def dashboard_html():
    """(html, source): source is 'file' or 'inline-fallback'."""
    path = DASHBOARD_FILE
    try:
        mtime = os.stat(path).st_mtime_ns
        if _DASH_CACHE["path"] != path or _DASH_CACHE["mtime"] != mtime:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            if "<html" not in text[:2000].lower():
                raise ValueError("not an html document")
            _DASH_CACHE.update(path=path, mtime=mtime, text=text)
        return _DASH_CACHE["text"], "file"
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        if _DASH_CACHE["text"] is None or _DASH_CACHE["path"] != path:
            log.warning("dashboard file %s unusable (%s); serving the inline fallback", path, exc)
        return DASHBOARD_HTML, "inline-fallback"


async def gateway_dashboard(request):
    html, source = dashboard_html()
    return web.Response(text=html, content_type="text/html", headers={"Cache-Control": "no-store", "X-Dashboard-Source": source})

DASHBOARD_HTML = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>vLLM Gateway</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--bd:#30363d;--fg:#e6edf3;--dim:#8b949e;--grn:#3fb950;--amb:#d29922;--red:#f85149;--blu:#58a6ff;--acc:#a371f7;--hover:#1c2129}
@media (prefers-color-scheme:light){
 :root{--bg:#f6f8fa;--card:#ffffff;--bd:#d0d7de;--fg:#1f2328;--dim:#57606a;--grn:#1a7f37;--amb:#9a6700;--red:#cf222e;--blu:#0969da;--acc:#8250df;--hover:#eef1f4}
}
*{box-sizing:border-box}
html,body{margin:0}
body{background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif}
.mono,.v,td.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
.wrap{max-width:1280px;margin:0 auto;padding:0 16px 40px}
.rz{color:var(--dim)}
a{color:var(--blu);text-decoration:none}a:hover{opacity:.85}
:focus-visible{outline:2px solid var(--blu);outline-offset:2px;border-radius:3px}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.001ms!important;transition-duration:.001ms!important}}

/* header */
header.top{position:sticky;top:0;z-index:6;background:rgba(13,17,23,.94);backdrop-filter:blur(8px);border-bottom:1px solid var(--bd);margin:0 -16px 0;padding:10px 16px;display:flex;flex-wrap:wrap;align-items:center;gap:8px 12px}
@media (prefers-color-scheme:light){header.top{background:rgba(246,248,250,.94)}}
header h1{font-size:15px;margin:0;font-weight:650;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;vertical-align:middle}
.dot.up{background:var(--grn);box-shadow:0 0 6px var(--grn)}
.dot.down{background:var(--red);box-shadow:0 0 6px var(--red)}
.dot.warn{background:var(--amb);box-shadow:0 0 6px var(--amb)}
.stamp{font-size:11px;padding:2px 9px;border-radius:10px;font-weight:600}
.stamp.local{background:rgba(63,185,80,.2);color:var(--grn)}
.stamp.remote{background:rgba(210,153,34,.25);color:var(--amb)}
.stamp.locked{background:rgba(88,166,255,.22);color:var(--acc)}
.seg{display:inline-flex;gap:0;border:1px solid var(--bd);border-radius:8px;overflow:hidden;margin-left:4px}
.segbtn{background:transparent;border:0;border-right:1px solid var(--bd);color:var(--dim);font:inherit;font-size:11px;
 font-weight:600;letter-spacing:.02em;padding:3px 10px;cursor:pointer}
.segbtn:last-child{border-right:0}
.segbtn:hover{background:var(--hover)}
.segbtn.on{background:rgba(63,185,80,.22);color:var(--grn)}
.segbtn.on[data-mode=full_remote]{background:rgba(210,153,34,.25);color:var(--amb)}
.segbtn.on[data-mode=full_local]{background:rgba(88,166,255,.22);color:var(--acc)}
button.ghost{background:transparent;color:var(--dim);border:1px solid var(--bd);font-weight:500;padding:4px 10px;font-size:12px;border-radius:6px;cursor:pointer}
button.ghost:hover{color:var(--fg);background:var(--hover)}
button.ghost.on{color:var(--blu);border-color:var(--blu)}
button.primary{background:var(--blu);color:#04101f;border:0;border-radius:6px;padding:7px 16px;font-weight:600;cursor:pointer;font-size:13px}
button.primary:hover{opacity:.92}
.consequence{width:100%;font-size:11.5px;color:var(--dim);padding:2px 0 0}

/* auth badge */
.authbadge{font-size:11px;padding:2px 8px;border-radius:10px;font-weight:600;cursor:default}
.authbadge.on{background:rgba(63,185,80,.15);color:var(--grn)}
.authbadge.off{background:rgba(248,81,73,.15);color:var(--red)}

/* sub-header row: subtitle + glossary toggle */
.subrow{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px;font-size:12px;margin:10px 0 12px}

/* glossary */
#glossary{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:10px 14px;margin-bottom:12px;font-size:12.5px}
#glossary dl{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:6px 18px;margin:6px 0 0}
#glossary dt{color:var(--fg);font-weight:600;display:inline}
#glossary dd{color:var(--dim);display:inline;margin:0 0 0 4px}
#glossary .row{margin:0}

/* global banner (errors + backoff + estate alert, consolidated) */
#banner{display:flex;flex-direction:column;gap:6px;margin:10px 0}
.bnln{padding:8px 12px;border-radius:8px;font-size:12.5px;display:flex;gap:8px;align-items:baseline}
.bnln.err{background:rgba(248,81,73,.12);border:1px solid rgba(248,81,73,.4)}
.bnln.warn{background:rgba(210,153,34,.12);border:1px solid rgba(210,153,34,.4)}
.bnln b.src{font-size:10.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--dim);flex:none}

/* right-now summary strip */
#now_summary{display:flex;flex-wrap:wrap;gap:14px 22px;align-items:baseline;background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:10px 16px;margin:12px 0;font-size:13px}
#now_summary b{font-size:17px;font-variant-numeric:tabular-nums}
#now_summary .lbl{color:var(--dim);font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;margin-left:4px}
#longest{font-size:12px;color:var(--amb);margin-top:2px}

/* tab bar */
.tabbar{display:flex;gap:4px;border-bottom:1px solid var(--bd);margin:6px 0 14px;flex-wrap:wrap}
.tabbtn{background:transparent;border:0;border-bottom:2px solid transparent;color:var(--dim);font:inherit;font-size:13.5px;font-weight:600;padding:8px 14px 9px;cursor:pointer;margin-bottom:-1px}
.tabbtn:hover{color:var(--fg)}
.tabbtn.on{color:var(--fg);border-bottom-color:var(--blu)}
.tabbtn .n{background:#21262d;border-radius:9px;padding:0 6px;font-size:10.5px;margin-left:6px;color:var(--dim)}
.tabpane{display:none}
.tabpane.on{display:block}

/* native tooltips on every metric label + knob (secondary reinforcement; text is never hover-only) */
abbr[title]{text-decoration:none;border-bottom:1px dotted var(--dim);cursor:help}

/* live-health strip */
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-bottom:14px}
.tile{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px 14px;min-width:0}
.tile .k{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.tile .v{font-size:22px;font-weight:600;margin-top:2px;line-height:1.2}
.tile .v small{font-size:12px;color:var(--dim);font-weight:400}
.tile .sub{color:var(--dim);font-size:11.5px;margin-top:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

/* collapsible sections (still used inside tabs for optional/secondary content) */
details.sec{background:var(--card);border:1px solid var(--bd);border-radius:10px;margin:10px 0;overflow:hidden}
details.sec>summary{list-style:none;padding:11px 14px;cursor:pointer;display:flex;align-items:center;gap:10px;font-size:14px;font-weight:600;color:var(--fg);user-select:none;flex-wrap:wrap}
details.sec>summary::-webkit-details-marker{display:none}
details.sec>summary::after{content:'\25B8';margin-left:auto;color:var(--dim);font-size:14px;transition:transform .12s}
details.sec[open]>summary::after{transform:rotate(90deg)}
details.sec>summary .sub{color:var(--dim);font-size:12px;font-weight:400}
details.sec>.body{padding:0 14px 14px;border-top:1px solid var(--bd);padding-top:12px}
h2.h{font-size:14px;font-weight:600;margin:18px 0 8px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
h2.h .sub{color:var(--dim);font-size:12px;font-weight:400}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px 14px;margin:10px 0}

/* meter/bar */
.bar{height:8px;border-radius:4px;background:#21262d;overflow:hidden;display:flex;margin-top:8px}
.bar i{display:block;height:100%}
.bl{background:var(--grn)}.br{background:var(--amb)}.bblu{background:var(--blu)}

/* tables */
.tw{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{text-align:left;color:var(--dim);font-weight:500;padding:6px 8px;border-bottom:1px solid var(--bd);white-space:nowrap}
th.sortable{cursor:pointer;user-select:none}
th.sortable:hover{color:var(--fg)}
th.sortable .arrow{display:inline-block;width:9px;color:var(--blu);font-size:10px}
td{padding:6px 8px;border-bottom:1px solid #21262d;white-space:nowrap}
@media (prefers-color-scheme:light){td{border-bottom-color:#e7ebee}}

/* tags + badges */
.tag{padding:1px 7px;border-radius:10px;font-size:11px;font-weight:600}
.tag.local{background:rgba(63,185,80,.15);color:var(--grn)}
.tag.remote{background:rgba(210,153,34,.15);color:var(--amb)}
.tag.held{background:rgba(88,166,255,.15);color:var(--blu)}
.tag.reject{background:rgba(248,81,73,.15);color:var(--red)}
.badge{display:inline-block;font-size:11px;padding:0 6px;border-radius:8px;background:#21262d;color:var(--dim);margin-left:6px;vertical-align:1px;font-weight:500}
.badge.ok{color:var(--grn);background:rgba(63,185,80,.12)}
.badge.warn{color:var(--amb);background:rgba(210,153,34,.14)}
.badge.run{color:var(--blu);background:rgba(88,166,255,.14)}
.badge.err{color:var(--red);background:rgba(248,81,73,.14)}
.badge.kind{color:var(--acc);background:rgba(163,113,247,.15);text-transform:uppercase;font-size:9.5px;letter-spacing:.03em}

/* task rows */
.tasks{display:flex;flex-direction:column}
.task{display:grid;grid-template-columns:12px minmax(0,1fr) auto;gap:10px;align-items:start;padding:6px 8px;border-radius:6px;text-decoration:none;color:inherit}
a.task:hover,.task.hov:hover{background:var(--hover)}
.sq{width:9px;height:9px;border-radius:2px;margin-top:6px;background:#484f58;display:inline-block;flex:none}
.sq.run{background:var(--blu);animation:sqpulse 1.3s ease-in-out infinite}
@keyframes sqpulse{0%,100%{opacity:1}50%{opacity:.25}}
.sq.done{background:var(--grn)}.sq.blocked{background:var(--red)}
.sq.stale{background:var(--amb)}
.sq.queued{background:transparent;border:1.5px solid var(--dim)}.sq.empty{background:#30363d}
.tname{font-weight:600;line-height:1.35;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;word-break:break-word}
.tstat{color:var(--dim);font-size:12.5px;margin-top:2px;word-break:break-word;white-space:normal}
.tmeta{color:var(--dim);font-size:12px;white-space:nowrap;text-align:right;line-height:1.35;padding-top:1px}
.tmeta b{color:var(--fg);font-weight:600}
.role{color:var(--fg)}
.rawid{color:var(--dim);font-size:11px}

/* group headers inside a section */
.thead{display:flex;align-items:center;gap:8px;margin:14px 0 4px;color:var(--dim);font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;flex-wrap:wrap}
.thead .n{background:#21262d;border-radius:9px;padding:0 7px;font-size:11px;color:var(--fg)}
.thead .lg{text-transform:none;letter-spacing:0;font-weight:400}
.empty{color:var(--dim);font-size:12.5px;padding:8px 4px}

/* pagination */
.pager{display:inline-flex;align-items:center;gap:6px;color:var(--dim);font-size:12px;margin-left:auto}
.pager button{background:transparent;color:var(--dim);border:1px solid var(--bd);border-radius:6px;padding:2px 9px;font-size:12px;cursor:pointer;line-height:1.4}
.pager button:hover:not(:disabled){color:var(--fg);background:var(--hover)}
.pager button:disabled{opacity:.35;cursor:not-allowed}
.pager .lbl{color:var(--dim);min-width:44px;text-align:center;font-variant-numeric:tabular-nums}

/* tools row */
.tools{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:4px 0 8px}
.tools input[type=text]{background:var(--bg);border:1px solid var(--bd);color:var(--fg);border-radius:6px;padding:5px 9px;font-size:12.5px;min-width:180px}
.tools label{display:flex;align-items:center;gap:6px;color:var(--dim);font-size:12px}
.tools select{background:var(--bg);border:1px solid var(--bd);color:var(--fg);border-radius:6px;padding:3px 7px;font-size:12px}
.chip{background:transparent;border:1px solid var(--bd);color:var(--dim);border-radius:14px;padding:3px 11px;font-size:12px;cursor:pointer;font-weight:600}
.chip:hover{color:var(--fg);background:var(--hover)}
.chip.on{background:rgba(88,166,255,.16);color:var(--blu);border-color:var(--blu)}

/* config form */
form.cfg .row{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px 14px}
form.cfg label{display:flex;flex-direction:column;gap:3px}
form.cfg input,form.cfg select{background:var(--bg);border:1px solid var(--bd);color:var(--fg);border-radius:6px;padding:6px 9px;font-size:13px;font-family:inherit}
form.cfg input:focus,form.cfg select:focus{outline:none;border-color:var(--blu)}
form.cfg .k{color:var(--fg);font-size:12.5px;font-weight:500}
form.cfg .hint{color:var(--dim);font-size:11px;font-weight:400}
.fieldflash{animation:flash 1.6s ease-out}
@keyframes flash{0%{background:rgba(88,166,255,.25)}100%{background:transparent}}

/* sparklines */
.spark{width:100%;height:56px;display:block;overflow:visible}
.spark polyline{fill:none;stroke-width:1.6;vector-effect:non-scaling-stroke}
.s1{stroke:var(--blu)}.s2{stroke:var(--amb)}.s3{stroke:var(--grn)}
.spark-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px;margin-top:8px}
.spark-cell{background:var(--bg);border:1px solid var(--bd);border-radius:8px;padding:10px 12px}
.telemwrap{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:2px;gap:6px;flex-wrap:wrap}
.telemwrap .legend{font-size:11px;color:var(--dim)}
.l1{color:var(--blu)}.l2{color:var(--amb)}.l3{color:var(--grn)}
.sparkfoot{font-size:11px;color:var(--dim);margin-top:3px}

/* details.tail (used inside task rows) */
details.tail summary{cursor:pointer;color:var(--dim);font-size:12px;list-style:none}
details.tail summary::-webkit-details-marker{display:none}
details.tail pre{margin:6px 0 0;font-size:11.5px;color:var(--dim);white-space:pre-wrap;background:var(--bg);border:1px solid var(--bd);border-radius:6px;padding:8px}

/* windows cards */
.wcard{background:var(--bg);border:1px solid var(--bd);border-radius:8px;padding:10px 12px;margin-bottom:8px}
.armwrap{overflow-x:auto}
.armtbl{margin:6px 0 0;width:100%}
.armtbl th,.armtbl td{padding:3px 6px;font-size:11.5px;white-space:nowrap}

/* reason -> knob explainer rows */
.reasonrow{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:10px;align-items:baseline;padding:8px 0;border-bottom:1px solid #21262d}
@media (prefers-color-scheme:light){.reasonrow{border-bottom-color:#e7ebee}}
.reasonrow:last-child{border-bottom:0}
.reasonrow .rtitle{font-weight:600}
.reasonrow .rwhy{color:var(--dim);font-size:12.5px;margin-top:2px}
.reasonrow .rfix{color:var(--acc);font-size:12px;margin-top:3px}
.reasonrow .rcount{text-align:right;white-space:nowrap}
.reasonrow .rcount b{font-size:16px}
.rbar{height:5px;border-radius:3px;background:#21262d;margin-top:6px;overflow:hidden}
.rbar i{display:block;height:100%;background:var(--amb)}

/* mobile */
@media(max-width:640px){
  .wrap{padding:0 12px 32px}
  header.top{margin:0 -12px;padding:10px 12px}
  header h1{font-size:14px}
  details.sec>summary{padding:11px 12px;font-size:13.5px}
  details.sec>.body{padding:0 12px 12px;padding-top:12px}
  .tmeta{font-size:11.5px}
  .tile .v{font-size:19px}
  .strip{grid-template-columns:repeat(auto-fit,minmax(140px,1fr))}
  .pager{margin-left:0;margin-top:4px}
  .reasonrow{grid-template-columns:1fr}
  .reasonrow .rcount{text-align:left}
}
</style></head><body><div class=wrap>

<header class=top>
 <h1>
  <span id=live class="dot down" title="Overall gateway status. Green = live, red = last poll failed."></span>
  vLLM Gateway
  <span id=modebadge class=stamp title="Routing mode">...</span>
  <span id=modeseg class=seg>
   <button class=segbtn data-mode=local_first>LOCAL FIRST</button>
   <button class=segbtn data-mode=full_remote>FULL REMOTE</button>
   <button class=segbtn data-mode=full_local>FULL LOCAL</button>
  </span>
  <span id=authbadge class=authbadge title="Whether POST /gateway/config and POST /gateway/models/local require the X-Admin-Token header.">...</span>
 </h1>
 <span class=rz id=updated style="margin-left:auto;font-size:12px">connecting...</span>
 <div class=consequence id=modeconsequence></div>
</header>

<div class=subrow>
 <span class=rz><abbr title="This gateway listens on :8000 and forwards to the local vLLM engine on :8001. It queues, serialises, and (when local is full) overflows to a paid remote provider.">:8000 capacity-routing gateway</abbr> &rarr; local engine :8001 &middot; overflow provider <b id=rm class=mono>--</b> &middot; <a href=/gateway/aliases/page>custom aliases</a></span>
 <button class=chip id=glossary_btn type=button>? Glossary</button>
</div>

<div id=glossary hidden>
 <b>Glossary</b> &mdash; every term used on this page, in one place (not hover-only).
 <dl>
  <div class=row><dt>Lane</dt><dd>one concurrent request slot on the local engine. "Budget" = how many lanes exist.</dd></div>
  <div class=row><dt>TTFT</dt><dd>time to first token &mdash; how long before streaming starts.</dd></div>
  <div class=row><dt>TPOT / inter-token latency</dt><dd>seconds between output tokens once streaming has started.</dd></div>
  <div class=row><dt>KV-cache %</dt><dd>how full the engine's attention-cache memory is.</dd></div>
  <div class=row><dt>ctx</dt><dd>context &mdash; total prompt+output tokens a request occupies while in flight.</dd></div>
  <div class=row><dt>prefill</dt><dd>the engine reading your prompt, before it can generate the first token.</dd></div>
  <div class=row><dt>OOM backoff</dt><dd>the local engine ran out of memory; the gateway holds off sending it new work for a bit.</dd></div>
  <div class=row><dt>Serialize-solo</dt><dd>a request big enough that it claims the *entire* lane budget alone.</dd></div>
  <div class=row><dt>Tiny fast-lane</dt><dd>very small requests get their own reserved headroom so they never queue behind big ones.</dd></div>
  <div class=row><dt>Background</dt><dd>a request tagged (by user-agent or IP) as non-interactive &mdash; cron/batch traffic that waits less patiently and never blocks a human's turn.</dd></div>
  <div class=row><dt>FULL REMOTE / FULL LOCAL</dt><dd>routing modes &mdash; see the tooltip on the mode buttons above.</dd></div>
  <div class=row><dt>p50 / p95</dt><dd>median, then the worst 5% &mdash; a shorthand for "typical" vs "bad case".</dd></div>
  <div class=row><dt>Evalkit score</dt><dd>an automated quality check on an engine-benchmark window's output, out of 45.</dd></div>
  <div class=row><dt>Arm</dt><dd>one configuration variant being A/B-tested in a frontier-queue benchmark window.</dd></div>
  <div class=row><dt>Pool</dt><dd>the KV-cache block pool size an arm was tested with.</dd></div>
  <div class=row><dt>MTP</dt><dd>multi-token prediction &mdash; the engine's speculative-decoding scheme.</dd></div>
  <div class=row><dt>Claims</dt><dd>research-job output: how many extracted claims survived verification, out of how many drafted.</dd></div>
  <div class=row><dt>Degraded</dt><dd>a research job finished without enough usable evidence to trust.</dd></div>
 </dl>
</div>

<div id=banner></div>

<!-- Live health strip: always visible, whichever tab is open -->
<div class=strip id=strip></div>

<div class=tabbar id=tabbar>
 <button class="tabbtn on" data-tab=now>Now <span class=n id=tab_n_now>&middot;</span></button>
 <button class=tabbtn data-tab=traffic>Traffic <span class=n id=tab_n_traffic>&middot;</span></button>
 <button class=tabbtn data-tab=settings>Settings</button>
</div>

<!-- ================= NOW ================= -->
<div class="tabpane on" id=pane-now>

 <div id=now_summary>loading&hellip;</div>
 <div id=longest hidden></div>

 <div class=tools>
  <span class=chip id=k_all data-k="" style="border-color:var(--blu);color:var(--blu)">All</span>
  <span class=chip data-k=request>Requests</span>
  <span class=chip data-k=lane>Agent lanes</span>
  <span class=chip data-k=window>Windows</span>
  <span class=chip data-k=research>Research</span>
  <label><input type=checkbox id=n_showdone> include finished</label>
  <span class=rz style="font-size:11.5px" id=n_state_legend></span>
  <span class=pager>
   <button id=n_prev disabled>&lsaquo;</button>
   <span class=lbl id=n_page>1/1</span>
   <button id=n_next disabled>&rsaquo;</button>
  </span>
 </div>
 <div class=tw>
  <table id=n_table>
   <thead><tr>
    <th></th>
    <th class=sortable data-sort=kind>kind<span class=arrow></span></th>
    <th class=sortable data-sort=who>who / what<span class=arrow></span></th>
    <th>status</th>
    <th class=sortable data-sort=age>age<span class=arrow></span></th>
    <th></th>
   </tr></thead>
   <tbody id=n_rows><tr><td colspan=6 class=empty>loading...</td></tr></tbody>
  </table>
 </div>
 <div id=n_err style="color:var(--red);font-size:12px;margin-top:6px"></div>
 <div style="margin-top:6px"><button class="ghost" id=n_csv type=button>Export CSV</button></div>

 <details class=sec id=sec-health>
  <summary>Estate watchdog <span class=sub>separate service on 10.0.1.10 &middot; not part of this gateway &middot; refreshes every 2 s</span></summary>
  <div class=body>
   <div id=health_alert hidden style="margin:0 0 10px;padding:8px 12px;border-radius:8px;background:rgba(248,81,73,.12);border:1px solid rgba(248,81,73,.4);color:var(--fg);font-size:13px"></div>
   <div class=thead>Checks <span class=n id=n_health>...</span><span class="lg rz" id=health_line>loading...</span></div>
   <div id=t_health style="display:flex;flex-wrap:wrap;gap:6px 14px;padding:4px 0 6px;font-size:12.5px"><span class=empty>loading...</span></div>
  </div>
 </details>

</div>

<!-- ================= TRAFFIC ================= -->
<div class=tabpane id=pane-traffic>

 <h2 class=h>Routing mix <span class=sub>local vs. paid overflow, and exactly why each overflow happened</span></h2>
 <div class=card>
  <div class=bar id=lrbar></div>
  <div id=lrtxt class=rz style="margin-top:8px;font-size:12.5px"></div>
 </div>

 <div class=card>
  <div class=thead>Why requests went to the paid provider <span class="lg rz">ranked by how often &middot; each links to the setting that causes it</span></div>
  <div id=reasons><span class=empty>loading...</span></div>
  <div class=thead style="margin-top:14px">Outcome legend</div>
  <div class="lg rz" style="font-size:12px;line-height:2">
   <span class="tag local">local</span> served by the local engine &nbsp;&middot;&nbsp;
   <span class="tag remote">remote</span> sent to the paid overflow provider &nbsp;&middot;&nbsp;
   <span class="tag held">held</span> background traffic paused, waiting for local to recover (never billed) &nbsp;&middot;&nbsp;
   <span class="tag reject">rejected</span> background traffic refused outright (a deliberate maintenance window, no wait, no bill)
  </div>
 </div>

 <h2 class=h>Telemetry <span class=sub>engine internals, GPU/host trends, per-client rollups &middot; live, refreshes every 2 s</span></h2>
 <div class=card>
  <div class=strip id=telem_tiles></div>
  <div class=spark-grid>
   <div class=spark-cell><div class=telemwrap><div class=k><abbr title="Prompt tokens/s (input) and generation tokens/s (output) as reported by the engine's Prometheus metrics.">Tokens/s (prompt / generation)</abbr></div><div class=legend><span class=l1>&#9632;</span> prompt <span class=l2>&#9632;</span> gen</div></div><svg class=spark id=sp_toks viewBox="0 0 300 56" preserveAspectRatio=none></svg><div class=sparkfoot id=sf_toks></div></div>
   <div class=spark-cell><div class=telemwrap><div class=k><abbr title="Time to first token, seconds. p50 = median wait before streaming starts; p95 = worst 5%.">TTFT p50 / p95 (s)</abbr></div><div class=legend><span class=l1>&#9632;</span> p50 <span class=l2>&#9632;</span> p95</div></div><svg class=spark id=sp_ttft viewBox="0 0 300 56" preserveAspectRatio=none></svg><div class=sparkfoot id=sf_ttft></div></div>
   <div class=spark-cell><div class=telemwrap><div class=k><abbr title="Inter-token latency, seconds. Gap between consecutive output tokens after the first -- streaming smoothness.">Inter-token latency (s)</abbr></div><div class=legend><span class=l1>&#9632;</span> p50 <span class=l2>&#9632;</span> p95</div></div><svg class=spark id=sp_tpot viewBox="0 0 300 56" preserveAspectRatio=none></svg><div class=sparkfoot id=sf_tpot></div></div>
   <div class=spark-cell><div class=telemwrap><div class=k><abbr title="KV-cache utilisation % (engine) and gateway lanes currently in flight.">KV-cache % / lanes</abbr></div><div class=legend><span class=l1>&#9632;</span> KV% <span class=l2>&#9632;</span> lanes</div></div><svg class=spark id=sp_kv viewBox="0 0 300 56" preserveAspectRatio=none></svg><div class=sparkfoot id=sf_kv></div></div>
   <div class=spark-cell><div class=telemwrap><div class=k><abbr title="GPU 0 utilisation %, temperature C, and power draw as % of card cap.">GPU 0 &mdash; util / temp / power</abbr></div><div class=legend><span class=l1>&#9632;</span> util% <span class=l2>&#9632;</span> temp&deg;C <span class=l3>&#9632;</span> pwr%cap</div></div><svg class=spark id=sp_gpu0 viewBox="0 0 300 56" preserveAspectRatio=none></svg><div class=sparkfoot id=sf_gpu0></div></div>
   <div class=spark-cell><div class=telemwrap><div class=k><abbr title="GPU 1 utilisation %, temperature C, and power draw as % of card cap.">GPU 1 &mdash; util / temp / power</abbr></div><div class=legend><span class=l1>&#9632;</span> util% <span class=l2>&#9632;</span> temp&deg;C <span class=l3>&#9632;</span> pwr%cap</div></div><svg class=spark id=sp_gpu1 viewBox="0 0 300 56" preserveAspectRatio=none></svg><div class=sparkfoot id=sf_gpu1></div></div>
   <div class=spark-cell><div class=telemwrap><div class=k><abbr title="Percentage of recent requests routed to the paid overflow provider.">Remote-overflow share %</abbr></div><div class=legend></div></div><svg class=spark id=sp_remote viewBox="0 0 300 56" preserveAspectRatio=none></svg><div class=sparkfoot id=sf_remote></div></div>
   <div class=spark-cell><div class=k><abbr title="Whether the shim's periodic scrape of the engine's Prometheus /metrics is succeeding.">Engine metrics scrape</abbr></div><div class=v style="font-size:14px;margin-top:6px" id=telem_engok>...</div></div>
  </div>
 </div>

 <h2 class=h>Per-client usage <span class=sub>who is calling this gateway, and how much</span></h2>
 <div class=card>
  <div class=tools>
   <span class=rz style="font-size:12px">host key: <span id=hostkey></span></span>
   <label>window <select id=pc_window><option value=uptime selected>since gateway start</option><option value=day>last 24h</option></select></label>
   <span class=rz style="font-size:11.5px" id=pc_filtered_note></span>
   <span class=pager>
    <button id=pc_prev disabled>&lsaquo;</button>
    <span class=lbl id=pc_page>1/1</span>
    <button id=pc_next disabled>&rsaquo;</button>
   </span>
  </div>
  <div class=tw><table><thead><tr>
   <th class=sortable data-sort=client>client<span class=arrow></span></th>
   <th class=sortable data-sort=requests>requests<span class=arrow></span></th><th>local</th><th>remote</th>
   <th><abbr title="Total output tokens served (exact where the engine reported them; else lower-bound from chunk count, marked ~).">tokens out</abbr></th>
   <th><abbr title="Average seconds spent waiting for a free lane before starting.">avg wait</abbr></th>
   <th><abbr title="Average time to first token, seconds.">avg TTFT</abbr></th>
   <th>errors</th>
   <th class=sortable data-sort=cost><abbr title="Rough USD estimate for what the remote-overflow calls have cost (uptime window only).">est. cost</abbr><span class=arrow></span></th>
  </tr></thead><tbody id=telem_clients><tr><td colspan=9 class=empty>loading...</td></tr></tbody></table></div>
 </div>

 <h2 class=h>Error feed <span class=sub>failovers and non-2xx completions, newest first</span></h2>
 <div class=card>
  <div class=pager style="margin-bottom:6px">
   <button id=ef_prev disabled>&lsaquo;</button>
   <span class=lbl id=ef_page>1/1</span>
   <button id=ef_next disabled>&rsaquo;</button>
  </div>
  <div class=tw><table><thead><tr>
   <th>time</th><th>client</th><th>endpoint</th>
   <th><abbr title="Route the request took: local, remote (overflow), held, or rejected.">route</abbr></th>
   <th>reason</th><th>status</th>
  </tr></thead><tbody id=telem_errors><tr><td colspan=6 class=empty>loading...</td></tr></tbody></table></div>
 </div>

 <h2 class=h>Request history <span class=sub>on-disk log, survives restarts &middot; server-side paginated</span></h2>
 <div class=card>
  <div class=tools>
   <input type=text id=h_client placeholder="filter by client...">
   <input type=text id=h_route placeholder="filter by route/reason (local, remote, tiny...)">
   <label>per page <select id=h_limit>
    <option>10</option><option selected>25</option><option>50</option><option>100</option>
   </select></label>
   <span class=rz id=h_logstate></span>
   <span class=pager>
    <button id=h_prev disabled>&lsaquo;</button>
    <span class=lbl id=h_page>1</span>
    <button id=h_next disabled>&rsaquo;</button>
   </span>
  </div>
  <div class=tw><table><thead><tr>
   <th>time</th><th>client</th><th>route</th>
   <th><abbr title="First few chars of the user prompt.">preview</abbr></th>
   <th>in&rarr;out</th>
   <th><abbr title="Time to first token, seconds.">TTFT</abbr></th>
   <th>duration</th><th>status</th>
  </tr></thead><tbody id=h_rows><tr><td colspan=8 class=empty>loading...</td></tr></tbody></table></div>
 </div>

 <h2 class=h>Recent requests <span class=sub>in-memory ring buffer, this process's uptime only &middot; &#9889; = streamed</span></h2>
 <div class=card>
  <div class=tools><span class="lg rz" id=req_summary></span>
   <span class=pager>
    <button id=ev_prev disabled>&lsaquo;</button>
    <span class=lbl id=ev_page>1/1</span>
    <button id=ev_next disabled>&rsaquo;</button>
   </span>
  </div>
  <div class=tw><table><thead><tr>
   <th>time</th><th>endpoint</th><th>client</th>
   <th>route</th><th>reason</th>
   <th><abbr title="Prompt tokens in -> max output tokens requested. streamed = lightning icon.">size (in&rarr;out)</abbr></th>
   <th>waited</th>
  </tr></thead><tbody id=ev><tr><td colspan=7 class=empty>loading...</td></tr></tbody></table></div>
 </div>

</div>

<!-- ================= SETTINGS ================= -->
<div class=tabpane id=pane-settings>

 <h2 class=h>Local model <span class=sub>switch the served checkpoint &middot; engine restarts &middot; overflow covers the gap</span></h2>
 <div class=card>
  <div class=rz style="font-size:12.5px">Scans <span id=mdir class=mono></span>. Any HF checkpoint works. Switching restarts the engine (~40 s warm / 4-5 min cold); traffic falls back to remote overflow until it is healthy.</div>
  <div class=tw><table><thead><tr><th></th><th>model</th><th>size</th><th>format</th><th></th></tr></thead><tbody id=lm></tbody></table></div>
  <div style="margin-top:8px"><span id=lmmsg class=rz></span></div>
 </div>

 <h2 class=h>Admin access <span class=sub>who can change settings below</span></h2>
 <div class=card id=authcard>
  <div id=authtext class=rz style="font-size:12.5px"></div>
  <div style="margin-top:8px"><button class=ghost id=forget_token type=button>Forget saved admin token in this browser</button></div>
 </div>

 <h2 class=h>Provider &amp; routing settings <span class=sub>applied live, saved to shim.env</span></h2>
 <div class=card>
 <form class=cfg autocomplete=off>

  <div class=thead>Overflow provider &mdash; where paid requests go</div>
  <label class=k>Provider preset
   <select id=f_preset>
    <option value="">-- pick to autofill base + model --</option>
    <option value="https://api.deepseek.com|deepseek-v4-flash">DeepSeek v4-flash</option>
    <option value="https://api.deepseek.com|deepseek-v4-pro">DeepSeek v4-pro</option>
    <option value="https://api.minimax.io/v1|MiniMax-M3">MiniMax M3</option>
    <option value="https://dashscope-intl.aliyuncs.com/compatible-mode/v1|qwen3-coder-next">Qwen3-Coder-Next (DashScope intl)</option>
    <option value="https://dashscope-intl.aliyuncs.com/compatible-mode/v1|qwen3.7-flash">qwen3.7-flash (DashScope intl)</option>
    <option value="https://openrouter.ai/api/v1|minimax/minimax-m3">OpenRouter &rarr; MiniMax M3</option>
   </select>
  </label>
  <div class=row>
   <label><span class=k>Overflow provider URL</span><span class=hint>base URL of the OpenAI-compatible endpoint</span>
    <input id=f_remote_base placeholder=https://api.minimax.io/v1></label>
   <label><span class=k>Overflow model name</span><span class=hint>model id that provider expects on /v1/chat/completions</span>
    <input id=f_remote_model placeholder=MiniMax-M3></label>
   <label><span class=k>Overflow API key</span><span class=hint id=keystate>&nbsp;</span>
    <input id=f_remote_key type=password placeholder="leave blank to keep current"></label>
   <label><span class=k>Send everything to the overflow provider?</span><span class=hint>1 = yes, bypass the local engine entirely &middot; 0 = local-first (normal)</span>
    <input id=f_force_remote type=number min=0 max=1></label>
  </div>

  <div class=thead>Local capacity &mdash; how much the local engine can take at once</div>
  <div class=row>
   <label><span class=k>How many requests can run locally at once?</span><span class=hint>concurrent lanes &middot; 1 = strictly one at a time</span>
    <input id=f_local_budget type=number min=1 max=8></label>
   <label><span class=k>How long should a request wait for a free lane?</span><span class=hint>seconds &middot; used for background during peak hours, and for interactive only if "never overflow" below is off</span>
    <input id=f_local_wait_secs type=number min=0 step=1></label>
   <label><span class=k>Should interactive traffic ever overflow while waiting?</span><span class=hint>1 = never, it queues until a lane is free (default) &middot; 0 = restores the wait-above-then-overflow behaviour</span>
    <input id=f_interactive_never_overflow type=number min=0 max=1></label>
   <label><span class=k>Prompt and output tokens all lanes may reserve</span><span class=hint>estimated tokens &middot; admission memory limit</span>
    <input id=f_token_budget type=number min=0 step=50000></label>
   <label><span class=k>After an out-of-memory crash, how long to back off?</span><span class=hint>seconds before probing the local engine again</span>
    <input id=f_oom_backoff_secs type=number min=0 step=10></label>
  </div>

  <div class=thead>Routing guards &mdash; when a request skips the local engine</div>
  <div class=row>
   <label><span class=k>Send to overflow if requested output is at least&hellip;</span><span class=hint>tokens &middot; caused reason "big-out"</span>
    <input id=f_big_output type=number min=0 step=1000></label>
   <label><span class=k>Send to overflow if the prompt is at least&hellip;</span><span class=hint>tokens, 0 = off &middot; caused reason "big-prompt"</span>
    <input id=f_big_prompt type=number min=0 step=1000></label>
   <label><span class=k>Hard size cap: prompt + output above this always goes to overflow</span><span class=hint>tokens &middot; caused reason "size"</span>
    <input id=f_max_local_tokens type=number min=0 step=10000></label>
   <label><span class=k>Clamp any local request's output to at most&hellip;</span><span class=hint>tokens</span>
    <input id=f_local_max_out type=number min=0 step=1024></label>
   <label><span class=k>Above this size, a request's lane cost scales with its size</span><span class=hint>tokens &middot; below this, every request costs exactly 1 lane</span>
    <input id=f_big_tokens type=number min=0 step=1000></label>
   <label><span class=k>How many tokens equal one lane, for a big request?</span><span class=hint>tokens/unit &middot; e.g. a 100K-token request costs ceil(100000&divide;this) lanes, capped by the budget</span>
    <input id=f_tokens_per_unit type=number min=1000 step=1000></label>
   <label><span class=k>Charge lanes for PREDICTED computed tokens instead of raw prompt size?</span><span class=hint>0 = off, default &middot; 1 = a repeated-prefix turn with a small new suffix costs ~1 lane instead of its full size &middot; check the mis-estimate feed before flipping this on</span>
    <input id=f_use_computed_cost type=number min=0 max=1></label>
   <label><span class=k>Safety margin added to a predicted-cheap request's cost</span><span class=hint>tokens &middot; padding for tokenizer/cache-boundary slop</span>
    <input id=f_prefix_hit_margin_tokens type=number min=0 step=128></label>
   <label><span class=k>Prefill admission window (cache-aware mode)</span><span class=hint>seconds of uncached prefill the engine may hold in its queue &middot; a request that does not fit waits for a lane, then overflows &middot; reason "prefill"</span>
    <input id=f_prefill_admit_secs type=number min=0 step=5></label>
   <label><span class=k>A request this small always fits the window</span><span class=hint>seconds of prefill</span>
    <input id=f_light_prefill_secs type=number min=0 step=1></label>
   <label><span class=k>Monster = this many seconds of prefill already in flight (cache-aware mode)</span><span class=hint>seconds &middot; new arrivals go remote &middot; reason "monster"</span>
    <input id=f_monster_prefill_secs type=number min=0 step=5></label>
   <label><span class=k>Treat this much in-flight context as "a monster is running"</span><span class=hint>tokens &middot; caused reason "monster"</span>
    <input id=f_monster_inflight type=number min=0 step=10000></label>
   <label><span class=k>Upper bound on the first-token wait budget</span><span class=hint>seconds</span>
    <input id=f_first_token_max type=number min=5 step=5></label>
   <label><span class=k>Estimated prefill speed</span><span class=hint>tokens/s, used to size first-token deadlines</span>
    <input id=f_prefill_tps type=number min=100 step=100></label>
  </div>

  <div class=thead>Priority lanes &amp; behaviour</div>
  <div class=row>
   <label><span class=k>Treat requests this small as "tiny"</span><span class=hint>tokens &middot; tiny requests get their own fast lane</span>
    <input id=f_tiny_tokens type=number min=0 step=100></label>
   <label><span class=k>Extra lanes reserved just for tiny requests</span><span class=hint>beyond the normal budget &middot; caused reason "tiny-fast" when full</span>
    <input id=f_tiny_extra_lanes type=number min=0 max=4></label>
   <label><span class=k>Lanes always kept free for interactive (non-background) traffic</span><span class=hint>caused reason "bg-yield"</span>
    <input id=f_fg_reserved type=number min=0 max=4></label>
   <label><span class=k>How long background traffic waits for a lane</span><span class=hint>seconds, before overflowing</span>
    <input id=f_bg_wait_secs type=number min=0 step=1></label>
   <label><span class=k>User-agent substrings that mark a request "background"</span><span class=hint>pipe-separated, e.g. scheduled|cron|job</span>
    <input id=f_bg_markers placeholder="scheduled cron job"></label>
   <label><span class=k>Peak hours (UTC) &mdash; bias background traffic to wait for local</span><span class=hint>comma ranges, e.g. 1-4,6-10</span>
    <input id=f_peak_hours_utc></label>
   <label><span class=k>Strip /think for background traffic?</span><span class=hint>0/1</span>
    <input id=f_bg_no_think type=number min=0 max=1></label>
   <label><span class=k>Client IPs that always get /think stripped</span><span class=hint>comma-separated</span>
    <input id=f_no_think_ips placeholder="10.0.1.10,10.0.1.250"></label>
   <label><span class=k>Log every request body to disk?</span><span class=hint>0/1 &middot; needed for the History tab</span>
    <input id=f_log_requests type=number min=0 max=1></label>
  </div>

  <div style="margin-top:12px">
   <button type=button id=save class=primary>Save settings</button>
   <span id=savemsg class=rz></span>
  </div>
 </form>
 </div>

</div>

</div>
<script>
(function(){
  // ---- shared helpers ----
  const $=s=>document.querySelector(s);
  const $$=s=>document.querySelectorAll(s);
  const esc=t=>String(t==null?'':t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  // ONE family of time formatters, consistently suffixed, so the page never mixes "14s elapsed" /
  // "0s ago" / "48m ago" / "up 719h25m" styles for the same underlying quantity again.
  const ago=s=>s==null?'?':s<60?Math.round(s)+'s ago':s<5400?Math.round(s/60)+'m ago':s<172800?Math.round(s/3600)+'h ago':Math.round(s/86400)+'d ago';
  const fmtAgo=t=>{const s=Math.max(0,Date.now()/1000-t);return s<60?s.toFixed(0)+'s ago':(s/60).toFixed(0)+'m ago';};
  const dur=s=>s==null?'--':(s<90?Math.round(s)+'s':s<5400?Math.round(s/60)+'m':(s/3600).toFixed(1)+'h');
  const fmtDur=s=>s==null?'--':(s<1?Math.round(s*1000)+'ms':s<90?s.toFixed(1)+'s':(s/60).toFixed(1)+'m');
  const when=t=>{if(!t)return'';let x=String(t).replace(' ','T');if(!/Z$|[+-]\d\d:\d\d$/.test(x))x+='Z';const d=new Date(x);if(isNaN(d))return esc(t);const s=(Date.now()-d.getTime())/1000;return s<86400?ago(s):d.toLocaleDateString(undefined,{month:'short',day:'numeric'})+' '+d.toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit'});};
  const tile=(k,v,sub)=>`<div class=tile><div class=k>${k}</div><div class=v>${v}</div>${sub?`<div class=sub>${sub}</div>`:''}</div>`;
  const routeTag=r=>r==='local'?'local':r==='held'?'held':r==='rejected-bg'?'reject':'remote';

  // ---- host -> role map (item 42/43): who is actually calling this gateway ----
  const HOSTS={'10.0.1.10':'agents-prod / Hermes','10.0.1.11':'Applicant','10.0.1.12':'ubuntuide01 / pi',
               '10.0.1.225':'this box (local)','10.0.1.250':'Hermes scheduler','127.0.0.1':'this box (local)'};
  function hostRole(ip){for(const k in HOSTS){if(ip&&ip.indexOf(k)===0)return HOSTS[k];}return null;}
  function clientDisplay(name,ip,ua){
    const role=hostRole(ip||name);
    const primary=role?role:esc(name||ip||'?');
    const secondary=[ip&&ip!==primary?esc(ip):null, ua?esc(ua):null].filter(Boolean).join(' &middot; ');
    return '<span class=role>'+primary+'</span>'+(secondary?' <span class=rawid>'+secondary+'</span>':'');
  }
  $('#hostkey').innerHTML=Object.entries(HOSTS).map(([ip,r])=>'<span class=mono>'+ip+'</span>='+esc(r)).join(' &middot; ');

  // ---- admin-token fetch (for mutating endpoints) ----
  function adminToken(){try{return localStorage.getItem('shim_admin_token')||'';}catch(e){return'';}}
  async function adminFetch(url,opts){
    opts=opts||{};opts.headers=Object.assign({'Content-Type':'application/json'},opts.headers||{});
    const t=adminToken();if(t)opts.headers['X-Admin-Token']=t;
    let r=await fetch(url,opts);
    if(r.status===401){
      const entered=prompt('Admin token required for this action (kept in this browser only):');
      if(entered){try{localStorage.setItem('shim_admin_token',entered);}catch(e){}opts.headers['X-Admin-Token']=entered;r=await fetch(url,opts);}
    }
    return r;
  }
  $('#forget_token').addEventListener('click',()=>{try{localStorage.removeItem('shim_admin_token');}catch(e){}$('#forget_token').textContent='forgotten -- next save will re-prompt';});

  // ---- global error/alert banner (item 63): every subsystem writes ONE line here instead of
  // scattering #err/#t_err/#win_err/#telem_engok/#health_alert around the page. ----
  const BANNER={};   // key -> {text, level}
  function setBanner(key,text,level){ // level 'err'|'warn'|null(clear)
    if(!text){delete BANNER[key];}else{BANNER[key]={text,level:level||'err'};}
    renderBanner();
  }
  function renderBanner(){
    const keys=Object.keys(BANNER);
    $('#banner').innerHTML=keys.map(k=>{const b=BANNER[k];
      return '<div class="bnln '+b.level+'"><b class=src>'+esc(k)+'</b><span>'+esc(b.text)+'</span></div>';
    }).join('');
  }

  // ---- generic client-side pagination + sort helper ----
  const PAGE={now:1,pc:1,ef:1,ev:1};
  const SORT={now:{key:null,dir:1},pc:{key:null,dir:-1}};
  function sortArr(arr,key,dir){
    if(!key)return arr;
    return arr.slice().sort((a,b)=>{const av=a[key],bv=b[key];
      if(typeof av==='number'||typeof bv==='number')return((av||0)-(bv||0))*dir;
      return String(av||'').localeCompare(String(bv||''))*dir;});
  }
  function paginate(arr,key,per){
    const total=arr.length,pages=Math.max(1,Math.ceil(total/per));
    if(PAGE[key]>pages)PAGE[key]=pages;
    const p=PAGE[key],from=(p-1)*per,to=Math.min(from+per,total);
    return {slice:arr.slice(from,to),page:p,pages,total,from,to};
  }
  function wirePager(key,prevId,nextId,lblId,pages,onchange){
    const prev=$(prevId),next=$(nextId),lbl=$(lblId);if(!prev||!next||!lbl)return;
    prev.disabled=PAGE[key]<=1;next.disabled=PAGE[key]>=pages;
    lbl.textContent=PAGE[key]+'/'+pages;
    prev.onclick=()=>{if(PAGE[key]>1){PAGE[key]--;onchange();}};
    next.onclick=()=>{if(PAGE[key]<pages){PAGE[key]++;onchange();}};
  }
  function wireSort(tableSel,sortKey,onchange){
    $$(tableSel+' th.sortable').forEach(th=>{
      th.addEventListener('click',()=>{
        const k=th.dataset.sort,s=SORT[sortKey];
        s.dir=(s.key===k)?-s.dir:1;s.key=k;
        $$(tableSel+' th.sortable .arrow').forEach(a=>a.textContent='');
        th.querySelector('.arrow').textContent=s.dir>0?'\u25B2':'\u25BC';
        onchange();
      });
    });
  }
  function csvDownload(filename,rows){
    const esc2=v=>{v=String(v==null?'':v);return /[",\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v;};
    const csv=rows.map(r=>r.map(esc2).join(',')).join('\n');
    const blob=new Blob([csv],{type:'text/csv'});
    const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=filename;
    document.body.appendChild(a);a.click();a.remove();
    setTimeout(()=>URL.revokeObjectURL(a.href),4000);
  }

  // change-detection: skip a full innerHTML re-render when the underlying data hasn't
  // changed (item 58) -- cheap JSON-string compare, keyed per render target.
  const _lastRender={};
  function renderIfChanged(key,data,fn){
    const sig=JSON.stringify(data);
    if(_lastRender[key]===sig)return false;
    _lastRender[key]=sig;fn();return true;
  }

  // Global blink phase kept only as a fallback flag; the actual pulse is a CSS animation now
  // (item 57) so it costs no forced layout, and prefers-reduced-motion turns it off for free.

  // ==================== TAB BAR (items 34/35/50/53/54) ====================
  let TAB='now';
  function applyTab(t,push){
    TAB=t;
    $$('.tabbtn').forEach(b=>b.classList.toggle('on',b.dataset.tab===t));
    $$('.tabpane').forEach(p=>p.classList.toggle('on',p.id==='pane-'+t));
    try{localStorage.setItem('shim_tab',t);}catch(e){}
    if(push)history.replaceState(null,'','#'+t);
  }
  $$('.tabbtn').forEach(b=>b.addEventListener('click',()=>applyTab(b.dataset.tab,true)));
  (function initTab(){
    let t=(location.hash||'').replace('#','');
    if(!['now','traffic','settings'].includes(t)){try{t=localStorage.getItem('shim_tab')||'now';}catch(e){t='now';}}
    applyTab(t,false);
  })();
  function goToSetting(fieldId){
    applyTab('settings',true);
    const el=document.getElementById(fieldId);
    if(el){el.scrollIntoView({block:'center'});el.focus();
      const label=el.closest('label');(label||el).classList.add('fieldflash');
      setTimeout(()=>(label||el).classList.remove('fieldflash'),1700);}
  }

  // ---- glossary (item 24): a real panel, not a hover-only tooltip ----
  $('#glossary_btn').addEventListener('click',()=>{
    const g=$('#glossary');g.hidden=!g.hidden;
    $('#glossary_btn').classList.toggle('on',!g.hidden);
    try{localStorage.setItem('shim_glossary',g.hidden?'0':'1');}catch(e){}
  });
  try{if(localStorage.getItem('shim_glossary')==='1'){$('#glossary').hidden=false;$('#glossary_btn').classList.add('on');}}catch(e){}

  // ==================== TOP + STATS (uses /gateway/stats) ====================
  let modelName='?', FR=0, MODE='local_first', lastOk=0, CFG={};
  async function loadModels(){try{const r=await fetch('/v1/models');const d=await r.json();modelName=(d.data&&d.data[0]&&d.data[0].id)||'?';}catch(e){}}
  const MODE_LABEL={local_first:'LOCAL-FIRST',full_remote:'FULL REMOTE',full_local:'FULL LOCAL (queue, $0)'};
  const MODE_CLASS={local_first:'stamp local',full_remote:'stamp remote',full_local:'stamp locked'};
  const MODE_CONSEQUENCE={local_first:'Requests use the local engine first; only overflow to the paid provider when every lane is busy or a guard fires.',
                          full_remote:'Every completion goes straight to the paid overflow provider -- the local engine sits idle.',
                          full_local:'Never spends money: a request with no free local lane QUEUES for one instead of overflowing.'};
  const MODE_BODY={local_first:{local_only:0,force_remote:0},
                   full_remote:{local_only:0,force_remote:1},
                   full_local:{local_only:1,force_remote:0}};
  function renderMode(){
    const b=$('#modebadge');
    if(b){b.textContent=MODE_LABEL[MODE]||MODE;b.className=MODE_CLASS[MODE]||'stamp';}
    $$('#modeseg .segbtn').forEach(x=>x.classList.toggle('on',x.dataset.mode===MODE));
    $('#modeconsequence').textContent=MODE_CONSEQUENCE[MODE]||'';
  }
  async function setMode(m){
    if(!MODE_BODY[m])return;
    const prev=MODE;
    try{
      const r=await adminFetch('/gateway/config',{method:'POST',body:JSON.stringify(MODE_BODY[m])});
      const d=await r.json();
      MODE=(d.config&&d.config.mode)||m;
    }catch(e){MODE=prev;setBanner('mode','mode change failed: '+e,'err');}
    FR=(MODE==='full_remote')?1:0;renderMode();loadCfg();
  }
  $$('#modeseg .segbtn').forEach(x=>x.addEventListener('click',()=>setMode(x.dataset.mode)));

  setInterval(()=>{const s=lastOk?(Date.now()-lastOk)/1000:null;const lu=$('#updated'),ld=$('#live');if(!lu)return;
    if(s==null){lu.textContent='connecting...';return;}
    lu.textContent='live \u00b7 updated '+Math.round(s)+'s ago';
    ld.className='dot '+(s<8?'up':s<30?'warn':'down');
  },1000);

  let statsData=null,recentEvents=[];
  async function tickStats(){
    let s;try{const r=await fetch('/gateway/stats',{cache:'no-store'});s=await r.json();setBanner('stats',null);lastOk=Date.now();}
    catch(e){setBanner('stats','reconnecting... (showing last data)','warn');return;}
    statsData=s;$('#rm').textContent=s.remote_model||'--';
    const uh=Math.floor(s.uptime/3600),um=Math.floor(s.uptime%3600/60);
    const gpuAvg=(s.gpu||[]).length?Math.round((s.gpu.reduce((a,g)=>a+(+g.util||0),0)/s.gpu.length)):null;
    const modelShort=modelName.replace(/^.*\//,'').slice(0,26);
    const bk=s.backoff>0?`<span class=sub style=color:var(--amb)>OOM backoff ${s.backoff}s</span>`:'';
    setBanner('oom', s.backoff>0?('Local engine is in OOM backoff for another '+s.backoff+'s -- new requests overflow to the paid provider until it clears.'):null, 'warn');
    $('#strip').innerHTML=[
      `<div class=tile><div class=k><abbr title="Whether the local vLLM engine on :8001 is responding to /health.">Local engine</abbr></div><div class=v><span class="dot ${s.local_healthy?'up':'down'}"></span> ${s.local_healthy?'up':'DOWN'}</div><div class=sub title="${esc(modelName)}">${esc(modelShort||'--')}</div></div>`,
      `<div class=tile><div class=k><abbr title="Concurrent request slots (lanes) in use / total available. Waiting shown when requests are queued for one, broken out by class.">Lanes</abbr></div><div class=v class=mono>${s.inflight}<small>/${s.budget}</small>${s.waiting?` <span style=color:var(--amb)>+${s.waiting} waiting${s.waiting_by_class?` (${s.waiting_by_class.interactive||0} interactive, ${s.waiting_by_class.background||0} bg)`:''}</span>`:''}</div><div class=sub><abbr title="Reserved prompt plus bounded output tokens across all lanes / token-budget cap.">context reserved ${((s.inflight_reserved_tokens||0)/1000).toFixed(0)}K / ${((s.token_budget||0)/1000).toFixed(0)}K cap</abbr>${bk?' &middot; '+bk:''}</div></div>`,
      `<div class=tile><div class=k><abbr title="Share of requests served by the local engine vs sent to the overflow provider.">Served local</abbr></div><div class=v class=mono style=color:var(--grn)>${s.local_pct}<small>%</small></div><div class=sub>overflow ${s.remote_pct}% &middot; avg wait ${s.avg_wait}s</div></div>`,
      `<div class=tile><div class=k><abbr title="Live tokens/s from the engine's Prometheus metrics (prompt + generation combined). '--' if the engine is down or not yet scraped.">Tokens/s now</abbr></div><div class=v class=mono id=tps_now>--</div><div class=sub>time to first token <span id=ttft_line>--</span></div></div>`,
      `<div class=tile><div class=k><abbr title="Average GPU utilisation across all cards, as reported by nvidia-smi.">GPU util avg</abbr></div><div class=v class=mono>${gpuAvg==null?'--':gpuAvg+'<small>%</small>'}</div><div class=sub>${(s.gpu||[]).map((g,i)=>'GPU'+i+' '+((g.util==null?'?':g.util)+'%')).join(' &middot; ')||'no GPU data'}</div></div>`,
      `<div class=tile><div class=k><abbr title="Total requests since the gateway started, and current uptime.">Requests / uptime</abbr></div><div class=v class=mono>${s.total}</div><div class=sub>up ${uh}h${um}m &middot; peak lanes ${s.peak_inflight}</div></div>`,
    ].join('');
    // Routing mix
    const lp=s.local_pct,rp=s.remote_pct;
    $('#lrbar').innerHTML=`<i class=bl style=width:${lp}%></i><i class=br style=width:${rp}%></i>`;
    $('#lrtxt').innerHTML=`<span style=color:var(--grn)>&#9632;</span> local ${s.local} (${lp}%) &nbsp; <span style=color:var(--amb)>&#9632;</span> overflow ${s.remote} (${rp}%) &nbsp; peak lanes ${s.peak_inflight} &nbsp; uptime ${uh}h${um}m`;
    renderReasons(s.remote_reasons||{},s.remote||1);
    recentEvents=(s.events||[]);
    renderRecent();
    $('#tab_n_traffic').textContent=(s.remote||0)+' overflow';
  }

  // ---- WHY OVERFLOW: every reason gets a plain sentence + the live value of the setting
  // that causes it + a jump-to-setting link (items 11-20). Built from the exact strings
  // record_event() uses server-side -- see _route_completions() in keepalive-shim.py. ----
  const REASON_INFO={
    forced:      {why:'Full-remote mode is switched on -- every request goes straight to the paid provider.', field:'f_force_remote', label:v=>'force-remote = '+v, fix:'Switch the mode badge back to LOCAL FIRST.'},
    size:        {why:'Prompt + requested output was bigger than the single-request size cap.', field:'f_max_local_tokens', label:v=>'cap = '+Number(v).toLocaleString()+' tok', fix:'Raise the size cap in Settings if this box can actually hold it.'},
    'big-out':   {why:'Requested output tokens was at or above the big-output threshold.', field:'f_big_output', label:v=>'threshold = '+Number(v).toLocaleString()+' tok', fix:'Raise the big-output threshold if these should stay local.'},
    'big-prompt':{why:'The prompt was at or above the big-prompt threshold.', field:'f_big_prompt', label:v=>v>0?('threshold = '+Number(v).toLocaleString()+' tok'):'currently off (0)', fix:'Raise (or set) the big-prompt threshold.'},
    'local-down':{why:'The local engine was unhealthy when this request arrived.', field:'f_oom_backoff_secs', label:v=>'backoff = '+v+'s', fix:'Check the Local engine tile above and the engine logs.'},
    monster:     {why:'A huge prefill was already monopolizing the engine -- new arrivals overflow until it drains.', field:'f_monster_inflight', label:v=>'threshold = '+Number(v).toLocaleString()+' tok in flight', fix:'Raise the monster-inflight threshold, or expect this while a big job runs.'},
    'tiny-fast': {why:'This was a tiny/fast-lane request, but even the reserved tiny headroom was full.', field:'f_tiny_extra_lanes', label:v=>'extra tiny lanes = '+v, fix:'Add more tiny extra lanes.'},
    tokens:      {why:'A lane was free, but serving this would have exceeded the total in-flight context budget.', field:'f_token_budget', label:v=>'budget = '+Number(v).toLocaleString()+' tok', fix:'Raise the total in-flight context cap.'},
    prefill:     {why:'The engine already had about as much uncached prompt to prefill as the admission window allows (and this request was not small); it waited for a lane, then overflowed.', field:'f_prefill_admit_secs', label:v=>'window = '+v+' s of prefill', fix:'Raise the prefill admission window, or wait for the backlog to drain.'},
    'bg-yield':  {why:'Lanes existed, but they are reserved for interactive traffic -- this request was background.', field:'f_fg_reserved', label:v=>'reserved for interactive = '+v, fix:'Lower the reserved-for-interactive count, or accept background waits longer.'},
    cap:         {why:'All lanes were busy and this request waited past its queue timeout.', field:'f_local_budget', label:v=>'budget = '+v+' lane(s)', fix:'Raise Local budget (lanes), or raise the queue-wait timeout.'},
    failover:    {why:'Local accepted the request but errored or ran out of memory mid-flight, so it fell back to remote.', field:'f_oom_backoff_secs', label:v=>'backoff = '+v+'s', fix:'Check engine logs for the underlying crash.'},
    'force-remote':{why:'A deliberate full-remote maintenance window -- background traffic was rejected outright rather than billed.', field:'f_force_remote', label:v=>'force-remote = '+v, fix:'Turn off full-remote mode when the window ends.'},
  };
  function renderReasons(reasons,totalRemote){
    const entries=Object.entries(reasons).sort((a,b)=>b[1]-a[1]);
    if(!entries.length){$('#reasons').innerHTML='<span class=empty>No overflow yet -- everything has been served locally.</span>';return;}
    const max=entries[0][1];
    $('#reasons').innerHTML=entries.map(([key,n])=>{
      const info=REASON_INFO[key]||{why:'(undocumented reason: '+esc(key)+')',field:null,label:()=>'',fix:''};
      const val=info.field?CFG[info.field.slice(2)]:null;
      const pct=totalRemote?Math.round(100*n/totalRemote):0;
      const jump=info.field?`<a href="#" class=rz onclick="event.preventDefault();window.__gotoSetting('${info.field}')">${esc(info.field?info.label(val):'')} &rarr;</a>`:'';
      return '<div class=reasonrow><div><div class=rtitle>'+esc(key)+'</div><div class=rwhy>'+esc(info.why)+'</div>'+
        (info.fix?'<div class=rfix>&#128161; '+esc(info.fix)+'</div>':'')+
        '<div class=rbar><i style=width:'+Math.round(100*n/max)+'%></i></div></div>'+
        '<div class=rcount><b>'+n+'</b><div class=rz>'+pct+'% of overflow</div>'+(jump?'<div style="margin-top:3px">'+jump+'</div>':'')+'</div></div>';
    }).join('');
  }
  window.__gotoSetting=goToSetting;   // reasonrow links are built via innerHTML, so this needs a stable global

  function renderRecent(){
    const per=25,pg=paginate(recentEvents,'ev',per);
    $('#req_summary').textContent=pg.total? (pg.total+' events buffered (this process only) \u00b7 showing '+(pg.from+1)+'-'+pg.to) : 'No requests yet this uptime.';
    $('#ev').innerHTML=pg.slice.map(e=>`<tr><td class="rz mono">${fmtAgo(e.t)}</td><td>${esc(e.ep)}</td><td>${clientDisplay(e.client)}</td><td><span class="tag ${routeTag(e.d)}">${esc(e.d)}</span></td><td class=rz>${esc(e.r||'')}</td><td class=mono>${(e.ptok||0)}&rarr;${(e.maxtok||0)}${e.stream?' &#9889;':''}</td><td class=mono>${e.waited?e.waited+'s':''}</td></tr>`).join('')
      || '<tr><td colspan=7 class=empty>No requests logged yet this uptime.</td></tr>';
    wirePager('ev','#ev_prev','#ev_next','#ev_page',pg.pages,renderRecent);
  }

  // ==================== UNIFIED "NOW" LIST (items 1-10, 50-55) ====================
  // One list, one Kind column, one pager, one empty state -- instead of four hand-rolled
  // lists (live requests / agent lanes / frontier queue+windows / research) each with its
  // own pager and its own definition of "still running".
  const TERMINAL=new Set(['done','failed','error','degraded','cancelled']);
  let lanesData=null, winData=null, nFilter='';
  const N_STATE_LEGEND='<span class="sq run" style="vertical-align:-1px"></span> working &nbsp;'+
    '<span class="sq queued" style="vertical-align:-1px"></span> queued &nbsp;'+
    '<span class="sq stale" style="vertical-align:-1px"></span> stale (no heartbeat) &nbsp;'+
    '<span class="sq done" style="vertical-align:-1px"></span> done &nbsp;'+
    '<span class="sq blocked" style="vertical-align:-1px"></span> blocked';
  $('#n_state_legend').innerHTML=N_STATE_LEGEND;

  function buildUnified(){
    const rows=[];
    const L=lanesData||{};
    // requests (kind=request) -- in-flight HTTP calls, from /gateway/lanes .active
    (L.active||[]).forEach(a=>{
      const remote=a.route==='remote',queued=a.phase==='queued',held=a.phase==='held';
      const state=queued?'queued':held?'queued':remote?'stale':'run';
      // deadline estimate (item 7): how much of the configured wait is left before this
      // would overflow, computed client-side from the live config -- no backend change needed.
      let deadline=null;
      const guaranteedLocal=queued&&!a.bg&&CFG&&Number(CFG.interactive_never_overflow)===1;
      if(queued&&CFG&&!guaranteedLocal){
        const waitS=(a.bg?CFG.bg_wait_secs:CFG.local_wait_secs);
        if(waitS!=null)deadline=Math.max(0,Number(waitS)-(a.waited||0));
      }
      rows.push({kind:'request',who:clientDisplay(a.name,a.ip,a.ua),
        what:(a.preview?'"'+esc(a.preview)+'"':'<span class=rz>(no text preview)</span>')+(a.bg?' &middot; background':'')+(a.tiny?' &middot; tiny':'')+(a.model?' &middot; '+esc(a.model):''),
        state, badgeText: queued?'waiting for a lane'+(deadline!=null?' \u00b7 overflows in ~'+Math.round(deadline)+'s':(guaranteedLocal?' \u00b7 guaranteed local (never overflows)':'')):held?'holding for local (background) -- never billed':remote?'remote overflow \u00b7 '+esc(a.reason||''):a.phase==='local'?'local model'+(a.waited?' \u00b7 waited '+a.waited+'s':''):'routing',
        age:a.elapsed_s||0, meta:esc(a.ep)+(a.ptok?' \u00b7 '+Math.round(a.ptok/1000)+'K in':'')+(a.maxtok?' \u00b7 '+a.maxtok+' max out':'')+(a.stream?' \u00b7 \u26a1 stream':''),
        sortkey:(a.name||'')+' '+(a.preview||'')});
    });
    // agent lanes (kind=lane) -- long-running background workers (STATUS-file heartbeat)
    (L.lanes||[]).forEach(l=>{
      const st=l.status||'';const m=st.match(/^(RUNNING|DONE|BLOCKED)\s*\|\s*([^|]*)\|\s*(.*)$/);
      const stg=m?m[1]:(st?'RUNNING':'UNKNOWN');const text=m?m[3].trim():(st||('newest file: '+l.newest));
      const stale=stg==='RUNNING'&&l.age_s>900;
      const state=stg==='DONE'?'done':stg==='BLOCKED'?'blocked':stale?'stale':stg==='RUNNING'?'run':'empty';
      rows.push({kind:'lane',who:'<span class=role>'+esc(l.lane)+'</span>',what:esc(text),state,
        badgeText: state==='done'?'done':state==='blocked'?'blocked':state==='stale'?'no heartbeat for '+dur(l.age_s):state==='run'?'working':'idle',
        age:l.age_s||0, meta:esc((l.root||'').replace(/^\/home\/kevin\//,'~/')), sortkey:l.lane||''});
    });
    // research jobs (kind=research)
    (L.research||[]).forEach(j=>{
      const running=!TERMINAL.has(j.status),zero=/^0\//.test(j.claims||'');
      const state=running?'run':(j.status==='failed'||j.status==='error')?'blocked':(j.status==='degraded'||zero)?'stale':'done';
      rows.push({kind:'research',who:'<span class=rawid>'+esc((j.id||'').slice(-6))+'</span>',what:esc(j.q)+' &middot; '+esc(j.depth)+(j.agents?' \u00b7 '+esc(j.agents)+' agents':'')+(j.tokens?' \u00b7 '+Math.round(j.tokens/1000)+'K tok':''),state,
        badgeText: running?esc(j.phase||j.status)+(j.progress?' \u00b7 '+esc(j.progress):''):j.status==='degraded'?'degraded -- no usable evidence':zero?'0 claims survived':j.claims?esc(j.claims)+' claims':'',
        age:j.elapsed_s||0, meta:'<a href="/gateway/research/'+encodeURIComponent(j.id||'')+'" target=_blank rel=noopener>report &rarr;</a>', sortkey:j.q||'',
        href:'/gateway/research/'+encodeURIComponent(j.id||'')});
    });
    // frontier windows (kind=window) -- this now covers what used to be a SEPARATE "frontier
    // queue" list too (item 51): queued/running/done are all just window states.
    ((winData&&winData.windows)||[]).forEach(w=>{
      const state=w.state==='queued'?'queued':w.failed?'blocked':w.state==='running'?'run':'done';
      rows.push({kind:'window',who:'<span class=role>'+esc(w.name)+'</span>',what:esc(w.description||'(no header comment found)'),state,
        badgeText: w.state==='queued'?'queued':w.duration_s!=null?dur(w.duration_s)+(w.state==='running'?' so far':''):w.state,
        age:w.duration_s||0, meta:w.log_url?'<a href="'+w.log_url+'" target=_blank rel=noopener>log &rarr;</a>':'', sortkey:w.name||''});
    });
    return rows;
  }
  function kindLabel(k){return {request:'request',lane:'agent lane',window:'window',research:'research'}[k]||k;}
  function renderNow(){
    const all=buildUnified();
    const summary={request:0,lane:0,window:0,research:0};
    let longest=null;
    all.forEach(r=>{if(r.state==='run'){summary[r.kind]=(summary[r.kind]||0)+1;if(!longest||r.age>longest.age)longest=r;}});
    $('#now_summary').innerHTML=
      '<span><b>'+summary.request+'</b><span class=lbl>requests in flight</span></span>'+
      '<span><b>'+summary.lane+'</b><span class=lbl>agent lanes working</span></span>'+
      '<span><b>'+summary.window+'</b><span class=lbl>windows running</span></span>'+
      '<span><b>'+summary.research+'</b><span class=lbl>research running</span></span>';
    const longestEl=$('#longest');
    if(longest&&longest.age>180){longestEl.hidden=false;longestEl.innerHTML='\u23f1 longest still running: <b>'+kindLabel(longest.kind)+'</b> '+longest.who+' &mdash; '+dur(longest.age);}
    else longestEl.hidden=true;
    $('#tab_n_now').textContent=(summary.request+summary.lane+summary.window+summary.research)+' active';

    const showDone=$('#n_showdone').checked;
    let rows=all.filter(r=>(!nFilter||r.kind===nFilter)&&(showDone||!TERMINAL.has(r.state)&&r.state!=='done'));
    const s=SORT.now;
    if(s.key==='kind')rows=sortArr(rows,'kind',s.dir);
    else if(s.key==='who')rows=sortArr(rows,'sortkey',s.dir);
    else if(s.key==='age')rows=sortArr(rows,'age',s.dir);
    else rows=rows.slice().sort((a,b)=>(b.state==='run')-(a.state==='run')||b.age-a.age); // default: working-first, then oldest

    const per=25,pg=paginate(rows,'now',per);
    $('#n_rows').innerHTML=pg.slice.map(r=>{
      const inner='<span class="sq '+r.state+'"></span>';
      const body='<div class=tname>'+r.who+'</div><div class=tstat>'+r.what+'</div>';
      const badge='<span class="badge '+(r.state==='run'?'run':r.state==='done'?'ok':r.state==='blocked'?'err':'warn')+'">'+esc(r.badgeText)+'</span>';
      return '<tr><td>'+inner+'</td><td><span class="badge kind">'+kindLabel(r.kind)+'</span></td><td>'+body+'</td><td>'+badge+'</td><td class=mono>'+dur(r.age)+'</td><td class=tmeta>'+(r.meta||'')+'</td></tr>';
    }).join('')||('<tr><td colspan=6 class=empty>'+(all.length?'Nothing matches this filter right now.':'Nothing in flight -- gateway is idle. That is normal, not a problem.')+'</td></tr>');
    wirePager('now','#n_prev','#n_next','#n_page',pg.pages,renderNow);
    $('#n_err').textContent=[(lanesData&&lanesData.errors)||[],(winData&&winData.errors)||[]].flat().join(', ');
    // health checks (estate watchdog -- kept, but visually a separate box now, not the top of the page)
    const H=(lanesData&&lanesData.health)||{},HC=H.checks||[];
    $('#n_health').textContent=H.present?HC.length:'-';
    $('#health_line').innerHTML=H.present?('overall <b style="color:'+(H.overall==='ok'?'var(--grn)':H.overall==='warn'?'var(--amb)':'var(--red)')+'">'+esc(H.overall||'?')+'</b> \u00b7 last run '+(H.age_s!=null?ago(H.age_s):'?')+(H.paused?' \u00b7 <span class="badge warn">PAUSED</span>':'')):'estate watchdog not installed yet';
    $('#t_health').innerHTML=HC.map(c=>'<span title="'+esc(c.detail||'')+'"><span class="sq '+(c.status==='OK'?'done':c.status==='WARN'?'stale':c.status==='CRIT'?'blocked':'empty')+'" style="margin:0 5px 0 0;vertical-align:-1px"></span>'+esc(c.name)+' <span class=rz>'+esc((c.detail||'').slice(0,48))+'</span></span>').join('')||'<span class=empty>'+(H.present?'no checks reported':'no watchdog state yet')+'</span>';
    const ha=$('#health_alert');if(H.alert){ha.hidden=false;ha.textContent='ALERT \u00b7 '+H.alert.slice(0,400);}else{ha.hidden=true;}
  }
  $$('.chip[data-k]').forEach(c=>c.addEventListener('click',()=>{
    nFilter=c.dataset.k;PAGE.now=1;
    $$('.chip[data-k]').forEach(x=>x.classList.toggle('on',x===c));
    renderNow();
  }));
  $('#k_all').classList.add('on');
  $('#n_showdone').addEventListener('change',()=>{PAGE.now=1;renderNow();});
  wireSort('#n_table','now',renderNow);
  $('#n_csv').addEventListener('click',()=>{
    const rows=buildUnified().filter(r=>!nFilter||r.kind===nFilter);
    csvDownload('gateway-now.csv',[['kind','who','status','age_s'],...rows.map(r=>[r.kind,r.who.replace(/<[^>]+>/g,''),r.badgeText,Math.round(r.age)])]);
  });

  let busyLanes=false;
  async function tickLanes(){if(busyLanes)return;busyLanes=true;
    try{lanesData=await(await fetch('/gateway/lanes',{cache:'no-store'})).json();lastOk=Date.now();renderNow();}
    catch(e){setBanner('lanes','background-task feed unreachable: '+e,'err');}
    finally{busyLanes=false;}
  }
  let busyWin=false;
  async function tickWindows(){if(busyWin)return;busyWin=true;
    try{winData=await(await fetch('/gateway/windows',{cache:'no-store'})).json();renderNow();}
    catch(e){setBanner('windows','windows feed unreachable: '+e,'err');}
    finally{busyWin=false;}
  }

  // ==================== TELEMETRY (uses /gateway/telemetry) ====================
  function sparkline(id,series){
    const svg=document.getElementById(id);if(!svg)return;
    const W=300,H=56,pad=3;
    const lines=(series||[]).filter(s=>s&&s.length);
    if(!lines.length){svg.innerHTML='';return;}
    const n=Math.max(...lines.map(s=>s.length));
    const all=[].concat(...lines).filter(v=>v!=null&&isFinite(v));
    let lo=all.length?Math.min(...all):0, hi=all.length?Math.max(...all):1;
    lo=Math.min(0,lo);if(hi<=lo)hi=lo+1;
    const x=i=>pad+(W-2*pad)*(n<=1?0:i/(n-1));
    const y=v=>H-pad-(H-2*pad)*((v-lo)/(hi-lo));
    const cls=['s1','s2','s3'];
    svg.innerHTML=lines.map((s,li)=>{
      const pts=s.map((v,i)=>(v==null||!isFinite(v))?null:x(i).toFixed(1)+','+y(v).toFixed(1)).filter(Boolean).join(' ');
      return pts?'<polyline class="'+cls[li%3]+'" points="'+pts+'"></polyline>':'';
    }).join('');
    return {n,lo,hi};
  }
  // sample cadence is fixed server-side at 2s (TELEM_SAMPLE_SECS) -- used only to label the
  // window ("last ~Nm"), never to compute anything routing-relevant.
  const TELEM_SAMPLE_SECS=2;
  function sparkFoot(id,n,cur){
    const el=document.getElementById(id);if(!el)return;
    const win=n?dur(n*TELEM_SAMPLE_SECS):'--';
    el.textContent='last ~'+win+(cur!=null?' \u00b7 now '+cur:'');
  }
  function seriesOf(fast,path){return fast.map(s=>{let v=s;for(const k of path){v=v&&v[k];if(v==null)return null;}return v;});}
  let telemData=null,perClientList=[],errorFeed=[],pcWindow='uptime';
  function renderTelem(){
    if(!telemData)return;
    const t=telemData;
    const fast=(t.series&&t.series.fast)||[];
    const eng=(t.latest&&t.latest.engine)||{};
    const tpsNow=$('#tps_now'),ttftLine=$('#ttft_line');
    if(tpsNow){tpsNow.innerHTML=eng.ok&&(eng.prompt_tok_s!=null||eng.gen_tok_s!=null)?(((eng.prompt_tok_s||0)+(eng.gen_tok_s||0)).toFixed(0)):'--';}
    if(ttftLine){ttftLine.innerHTML=eng.ok&&eng.ttft_p50!=null?(eng.ttft_p50.toFixed(2)+'s / '+(eng.ttft_p95||0).toFixed(2)+'s'):'--';}
    $('#telem_engok').innerHTML=eng.ok?'<span class="dot up"></span> scraping OK':
      ('<span class="dot down"></span> '+(eng.age_s!=null?('stale '+Math.round(eng.age_s)+'s ago ('+esc(eng.err||'')+')'):('never scraped yet ('+esc(eng.err||'engine down')+')')));
    setBanner('engine', eng.ok?null:'Engine Prometheus scrape is failing -- telemetry sparklines are stale.', 'warn');
    $('#telem_tiles').innerHTML=[
      tile('Tokens/s now',eng.ok&&(eng.prompt_tok_s!=null||eng.gen_tok_s!=null)?(((eng.prompt_tok_s||0)+(eng.gen_tok_s||0)).toFixed(0)):'--'),
      tile('TTFT (time to first token) p50 <small>/ p95</small>',eng.ok&&eng.ttft_p50!=null?(eng.ttft_p50.toFixed(2)+'s <small>/ '+(eng.ttft_p95||0).toFixed(2)+'s</small>'):'--'),
      tile('KV-cache % (attention-cache memory used)',eng.ok&&eng.kv_cache_pct!=null?(eng.kv_cache_pct.toFixed(0)+'<small>%</small>'):'--'),
      tile('Engine running <small>/ waiting</small>',eng.ok?((eng.running==null?'--':eng.running)+' <small>/ '+(eng.waiting==null?'--':eng.waiting)+'</small>'):'--'),
    ].join('');
    let m;
    m=sparkline('sp_toks',[seriesOf(fast,['engine','prompt_tok_s']),seriesOf(fast,['engine','gen_tok_s'])]);sparkFoot('sf_toks',fast.length,eng.ok?((eng.prompt_tok_s||0).toFixed(0)+'p/'+(eng.gen_tok_s||0).toFixed(0)+'g tok/s'):null);
    m=sparkline('sp_ttft',[seriesOf(fast,['engine','ttft_p50']),seriesOf(fast,['engine','ttft_p95'])]);sparkFoot('sf_ttft',fast.length,eng.ttft_p50!=null?eng.ttft_p50.toFixed(2)+'s':null);
    m=sparkline('sp_tpot',[seriesOf(fast,['engine','tpot_p50']),seriesOf(fast,['engine','tpot_p95'])]);sparkFoot('sf_tpot',fast.length,eng.tpot_p50!=null?eng.tpot_p50.toFixed(2)+'s':null);
    m=sparkline('sp_kv',[seriesOf(fast,['engine','kv_cache_pct']),seriesOf(fast,['gateway','inflight'])]);sparkFoot('sf_kv',fast.length,eng.kv_cache_pct!=null?eng.kv_cache_pct.toFixed(0)+'%':null);
    const pw=(gi)=>fast.map(s=>{const g=s.gpu&&s.gpu[gi];return(g&&g.power_w!=null&&g.power_limit_w)?100*g.power_w/g.power_limit_w:null;});
    m=sparkline('sp_gpu0',[seriesOf(fast,['gpu',0,'util']),seriesOf(fast,['gpu',0,'temp_c']),pw(0)]);sparkFoot('sf_gpu0',fast.length,null);
    m=sparkline('sp_gpu1',[seriesOf(fast,['gpu',1,'util']),seriesOf(fast,['gpu',1,'temp_c']),pw(1)]);sparkFoot('sf_gpu1',fast.length,null);
    m=sparkline('sp_remote',[seriesOf(fast,['gateway','remote_share_pct'])]);sparkFoot('sf_remote',fast.length,null);
    const pc=t.per_client||{};
    perClientList=Object.keys(pc).sort((a,b)=>pc[b].requests-pc[a].requests).map(n=>({n:n,c:pc[n]}));
    renderPerClient();
    errorFeed=t.errors||[];
    renderErrorFeed();
  }
  function renderPerClient(){
    const per=25,pg=paginate(perClientList,'pc',per);
    $('#pc_filtered_note').textContent='uptime-scoped rollup (switch the window selector for a 24h view backed by the on-disk log)';
    $('#telem_clients').innerHTML=pg.slice.map(x=>{const n=x.n,c=x.c;
      return '<tr><td><a href="#" onclick="event.preventDefault();window.__filterHistory(\''+esc(n).replace(/'/g,"\\'")+'\')">'+clientDisplay(n)+'</a></td><td class=mono>'+c.requests+'</td><td class=mono>'+c.local+'</td><td class=mono>'+c.remote+
        '</td><td class=mono>'+c.tokens_out+(c.tokens_out_lb>0?' <span class=rz title="includes chunk-count lower bounds">(~)</span>':'')+
        '</td><td class=mono>'+(c.wait_avg_s!=null?c.wait_avg_s+'s':'--')+'</td><td class=mono>'+(c.ttft_avg_s!=null?c.ttft_avg_s+'s':'--')+
        '</td><td class=mono>'+c.errors+'</td><td class=mono>'+(c.cost_est_usd?'$'+c.cost_est_usd.toFixed(4):'--')+'</td></tr>';
    }).join('')||'<tr><td colspan=9 class=empty>No completed requests yet this uptime.</td></tr>';
    wirePager('pc','#pc_prev','#pc_next','#pc_page',pg.pages,renderPerClient);
  }
  window.__filterHistory=function(name){$('#h_client').value=name;hPage=1;tickHistory();applyTab('traffic',true);$('#h_client').scrollIntoView({block:'center'});};
  $('#pc_window').addEventListener('change',async e=>{
    pcWindow=e.target.value;
    if(pcWindow==='uptime'){renderPerClient();return;}
    try{const s=await(await fetch('/gateway/history/summary?hours=24',{cache:'no-store'})).json();
      const pc=s.per_client||{};
      perClientList=Object.keys(pc).sort((a,b)=>pc[b].requests-pc[a].requests).map(n=>({n:n,c:{requests:pc[n].requests,local:pc[n].local,remote:pc[n].remote,tokens_out:pc[n].tokens_out,errors:pc[n].errors,wait_avg_s:null,ttft_avg_s:null,cost_est_usd:null}}));
      $('#pc_filtered_note').textContent='last 24h, from the on-disk log';
      renderPerClient();
    }catch(e2){setBanner('history','24h summary unreachable: '+e2,'err');}
  });
  function renderErrorFeed(){
    const per=25,pg=paginate(errorFeed,'ef',per);
    $('#telem_errors').innerHTML=pg.slice.map(e=>'<tr><td class="rz mono">'+fmtAgo(e.t)+'</td><td>'+clientDisplay(e.client)+'</td><td>'+esc(e.ep)+
      '</td><td><span class="tag '+routeTag(e.route)+'">'+esc(e.route)+'</span></td><td class=rz>'+esc(e.reason)+
      '</td><td class=mono>'+(e.status||'')+'</td></tr>').join('')||'<tr><td colspan=6 class=empty>No errors recorded this uptime.</td></tr>';
    wirePager('ef','#ef_prev','#ef_next','#ef_page',pg.pages,renderErrorFeed);
  }
  async function tickTelemetry(){
    try{telemData=await(await fetch('/gateway/telemetry',{cache:'no-store'})).json();renderTelem();}
    catch(e){setBanner('telemetry','telemetry feed unreachable: '+e,'err');}
  }

  // ==================== HISTORY (server-side paginated /gateway/history) ====================
  let hPage=1,hTimer=null;
  function hLimit(){return parseInt($('#h_limit').value)||25;}
  async function tickHistory(){
    const cq=($('#h_client').value||'').trim(),rq=($('#h_route').value||'').trim();
    const limit=hLimit();
    const u='/gateway/history?limit='+limit+'&page='+hPage+(cq?'&client='+encodeURIComponent(cq):'')+(rq?'&route='+encodeURIComponent(rq):'');
    let h;try{h=await(await fetch(u,{cache:'no-store'})).json();}
    catch(e){$('#h_rows').innerHTML='<tr><td colspan=8 class=empty>history feed unreachable: '+esc(e)+'</td></tr>';return;}
    const items=h.items||[];
    $('#h_rows').innerHTML=items.map(r=>{
      const badge='<span class="tag '+routeTag(r.route)+'">'+esc(r.route||'?')+'</span>'+(r.reason?' <span class=rz>'+esc(r.reason)+'</span>':'');
      const out=r.outtok!=null?r.outtok:(r.outtok_lb!=null?'~'+r.outtok_lb:0);
      return '<tr><td class="rz mono">'+fmtAgo(r.t)+'</td><td>'+clientDisplay(r.client)+'</td><td>'+badge+
        '</td><td class=rz title="'+esc(r.preview||'')+'">'+esc((r.preview||'').slice(0,60))+
        '</td><td class=mono>'+(r.ptok||0)+'&rarr;'+out+
        '</td><td class=mono>'+(r.ttft!=null?r.ttft.toFixed(2)+'s':'--')+
        '</td><td class=mono>'+fmtDur(r.duration)+'</td><td class=mono>'+(r.status||'')+'</td></tr>';
    }).join('')||('<tr><td colspan=8 class=empty>No matching requests logged'+((cq||rq)?' for this filter.':' yet.')+'</td></tr>');
    const lg=h.log||{};
    $('#h_logstate').textContent='logged '+(lg.written||0)+' since restart'+(lg.dropped_cap?' \u00b7 '+lg.dropped_cap+' dropped (daily cap)':'')+(lg.dropped_queue?' \u00b7 '+lg.dropped_queue+' dropped (queue full)':'')+(lg.capped_today?' \u00b7 TODAY LOG AT SIZE CAP':'');
    $('#h_page').textContent=hPage;
    $('#h_prev').disabled=hPage<=1;
    $('#h_next').disabled=!h.has_more;
  }
  $('#h_client').addEventListener('input',()=>{clearTimeout(hTimer);hTimer=setTimeout(()=>{hPage=1;tickHistory();},300);});
  $('#h_route').addEventListener('input',()=>{clearTimeout(hTimer);hTimer=setTimeout(()=>{hPage=1;tickHistory();},300);});
  $('#h_limit').addEventListener('change',()=>{hPage=1;tickHistory();});
  $('#h_prev').addEventListener('click',()=>{if(hPage>1){hPage--;tickHistory();}});
  $('#h_next').addEventListener('click',()=>{hPage++;tickHistory();});

  // ==================== LOCAL MODELS + CONFIG ====================
  async function loadLocalModels(){
    try{const d=await(await fetch('/gateway/models/local')).json();
      $('#mdir').textContent=d.models_dir||'';
      $('#lm').innerHTML=(d.models||[]).map(m=>{
        const badge=m.live?'<span class="tag local">LIVE</span>':'';
        const btn=m.servable&&!m.live?`<button class="ghost lmsw" data-p="${esc(m.path)}" style="font-size:11px;padding:3px 10px">Switch</button>`:
                  (m.servable?'<span class=rz>current</span>':'<span class=rz style=color:var(--amb)>not servable</span>');
        return `<tr><td>${badge}</td><td class=mono>${esc(m.name)}</td><td class=mono>${esc(m.gb)} GB</td><td class=rz>${esc(m.desc)}</td><td>${btn}</td></tr>`;
      }).join('')||'<tr><td colspan=5 class=rz>no checkpoints found</td></tr>';
      $$('.lmsw').forEach(b=>b.addEventListener('click',()=>switchLocal(b.dataset.p)));
    }catch(e){}
  }
  async function switchLocal(p){
    if(!confirm('Switch the local engine to:\n\n'+p+'\n\nThe engine restarts. Traffic falls back to remote overflow until it is healthy.'))return;
    $('#lmmsg').textContent='switching... engine restarting';
    try{const r=await adminFetch('/gateway/models/local',{method:'POST',body:JSON.stringify({path:p})});
      const d=await r.json();
      $('#lmmsg').textContent=d.error?('X '+d.error):('OK switching to '+d.switching_to+' ('+d.quant+') -- '+d.note);
      setTimeout(loadLocalModels,15000);
    }catch(e){$('#lmmsg').textContent='X '+e;}
  }
  async function loadCfg(){
    try{const c=await(await fetch('/gateway/config')).json();
      CFG=c;
      FR=c.force_remote?1:0;
      MODE=c.mode||(c.local_only?'full_local':(c.force_remote?'full_remote':'local_first'));renderMode();
      for(const [k,v] of Object.entries(c)){const el=document.getElementById('f_'+k);if(el&&el.type!=='password')el.value=v;}
      $('#keystate').textContent=c.remote_key_display?('current: '+c.remote_key_display):'not set';
      const ab=$('#authbadge'),at=$('#authtext');
      if(c.admin_token_set){ab.textContent='admin-gated';ab.className='authbadge on';
        at.innerHTML='An admin token IS set. Saving settings or switching the local model from this page requires it -- you will be prompted once per browser and it is then remembered in this browser only (localStorage).';}
      else{ab.textContent='\u26a0 open (no admin token)';ab.className='authbadge off';
        at.innerHTML='<b style=color:var(--amb)>No admin token is set.</b> Anyone on the LAN who can reach this page can change settings or switch the local model with no login. Set <span class=mono>SHIM_ADMIN_TOKEN_FILE</span> (or <span class=mono>SHIM_ADMIN_TOKEN</span>) and restart the service to close this.';}
    }catch(e){}
  }
  async function saveCfg(){
    const b={};
    $$('input[id^=f_]').forEach(el=>{const k=el.id.slice(2);
      if(el.type==='password'){if(el.value.trim())b[k]=el.value.trim();}
      else if(el.value!=='')b[k]=el.value;
    });
    $('#savemsg').textContent='saving...';
    try{const r=await adminFetch('/gateway/config',{method:'POST',body:JSON.stringify(b)});
      const d=await r.json();$('#savemsg').textContent='OK saved: '+(d.changed||[]).join(', ');$('#f_remote_key').value='';loadCfg();}
    catch(e){$('#savemsg').textContent='X '+e;}
  }
  $('#save').addEventListener('click',saveCfg);
  $('#f_preset').addEventListener('change',e=>{const v=e.target.value;if(!v)return;const [b,m]=v.split('|');$('#f_remote_base').value=b;$('#f_remote_model').value=m;});

  // ==================== SHARED CLOCK (item 56): one setInterval instead of five, and
  // everything pauses while the tab is hidden instead of repainting invisibly forever. ====================
  const TASKS=[
    {every:1500, fn:tickStats, due:0},
    {every:2000, fn:tickLanes, due:0},
    {every:10000, fn:tickWindows, due:0},
    {every:2000, fn:tickTelemetry, due:0},
    {every:10000, fn:tickHistory, due:0},
    {every:15000, fn:loadModels, due:0},
  ];
  function clockTick(){
    if(document.hidden)return;
    const now=Date.now();
    TASKS.forEach(t=>{if(now>=t.due){t.due=now+t.every;t.fn();}});
  }
  setInterval(clockTick,500);
  document.addEventListener('visibilitychange',()=>{if(!document.hidden){TASKS.forEach(t=>t.due=0);clockTick();}});

  // ==================== BOOT ====================
  loadModels();loadCfg();loadLocalModels();
  tickStats();tickLanes();tickWindows();tickTelemetry();tickHistory();
})();
</script></body></html>"""

ALIASES_HTML = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Gateway aliases</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--bd:#30363d;--fg:#e6edf3;--dim:#8b949e;--blu:#58a6ff;--red:#f85149;--grn:#3fb950}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}.wrap{max-width:1000px;margin:auto;padding:20px}.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px;margin:14px 0}h1{font-size:20px}h2{font-size:15px;border-bottom:1px solid var(--bd);padding-bottom:8px}label{display:flex;flex-direction:column;gap:4px;color:var(--dim);font-size:12px}input{background:var(--bg);border:1px solid var(--bd);border-radius:6px;color:var(--fg);padding:8px;font:inherit}form .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}.wide{grid-column:1/-1}button{border:1px solid var(--bd);border-radius:6px;background:var(--blu);color:#06111f;padding:8px 14px;font-weight:600;cursor:pointer}.danger{background:transparent;color:var(--red)}table{width:100%;border-collapse:collapse}th,td{text-align:left;border-bottom:1px solid var(--bd);padding:8px}small,.muted{color:var(--dim)}#msg{margin:10px 0}.ok{color:var(--grn)}.err{color:var(--red)}
</style></head><body><div class=wrap><a href=/gateway/dashboard>&larr; gateway dashboard</a><h1>Gateway provider aliases</h1>
<p class=muted>Use <code>estate</code> for the dashboard mode, <code>estate-local</code> for local-only, and <code>estate-remote</code> for the default remote. Custom aliases always target their configured OpenAI-compatible endpoint. Context limits are enforced per endpoint.</p>
<div class=card><h2>Admin token</h2><label>Token (stored only in this browser)<input id=token type=password autocomplete=off></label></div>
<div class=card><h2>Create or update custom alias</h2><form id=form><div class=grid>
<label>Alias name <small>e.g. estate-openai</small><input id=name required pattern="[A-Za-z][A-Za-z0-9._-]{1,63}"></label>
<label>Model id<input id=model required></label><label class=wide>Base URL<input id=base type=url placeholder=https://api.openai.com/v1 required></label>
<label>API key <small>blank on update keeps existing key</small><input id=key type=password autocomplete=off></label>
<label>Context limit (tokens)<input id=context_limit type=number min=1024 value=128000 required></label>
<label>Maximum output (tokens)<input id=max_output type=number min=1 value=16384 required></label>
<label><span>Enabled</span><input id=enabled type=checkbox checked></label>
</div><p><button type=submit>Save alias</button></p></form><div id=msg></div></div>
<div class=card><h2>Configured aliases</h2><div id=list>Loading...</div></div>
<script>
const $=id=>document.getElementById(id), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const headers=()=>{const t=$('token').value.trim();return t?{'X-Admin-Token':t,'Content-Type':'application/json'}:{'Content-Type':'application/json'}};
if(localStorage.gatewayAdminToken){$('token').value=localStorage.gatewayAdminToken} $('token').onchange=()=>localStorage.gatewayAdminToken=$('token').value;
async function load(){const r=await fetch('/gateway/aliases',{cache:'no-store'}),d=await r.json();let h='<p class=muted>Built-ins: '+d.builtins.map(esc).join(', ')+'</p>';
if(!d.aliases.length)h+='<p class=muted>No custom aliases.</p>';else h+='<table><thead><tr><th>alias</th><th>endpoint</th><th>model</th><th>context</th><th>status</th><th></th></tr></thead><tbody>'+d.aliases.map(a=>'<tr><td><code>'+esc(a.name)+'</code></td><td>'+esc(a.base)+'</td><td>'+esc(a.model)+'</td><td>'+Number(a.context_limit).toLocaleString()+' / '+Number(a.max_output).toLocaleString()+'</td><td>'+ (a.enabled?'enabled':'disabled')+'</td><td><button class=danger data-del="'+esc(a.name)+'">Delete</button></td></tr>').join('')+'</tbody></table>';$('list').innerHTML=h;document.querySelectorAll('[data-del]').forEach(b=>b.onclick=async()=>{if(!confirm('Delete '+b.dataset.del+'?'))return;const r=await fetch('/gateway/aliases',{method:'DELETE',headers:headers(),body:JSON.stringify({name:b.dataset.del})});$('msg').textContent=(await r.json()).error||'Deleted';load()})}
$('form').onsubmit=async e=>{e.preventDefault();const d={name:$('name').value,base:$('base').value,model:$('model').value,context_limit:Number($('context_limit').value),max_output:Number($('max_output').value),enabled:$('enabled').checked};if($('key').value)d.key=$('key').value;const r=await fetch('/gateway/aliases',{method:'POST',headers:headers(),body:JSON.stringify(d)}),j=await r.json();$('msg').className=r.ok?'ok':'err';$('msg').textContent=j.error||('Saved '+j.saved);if(r.ok){$('key').value='';load()}};load();
</script></div></body></html>"""

async def _on_startup(app):
    global _REMOTE_DEAD_PERSIST
    _drain_startup_recover()
    _REMOTE_DEAD_PERSIST = True
    _remote_dead_load()
    _load_stats()
    if FLOW_RESTORE:
        try:
            _now = time.time()
            _rows = await asyncio.get_running_loop().run_in_executor(
                None, lambda: flow_routes_restore_blocking(_now, _PROCESS_STARTED))
            log.info("routing ring restored: %d rows from the request log", flow_routes_restore(_rows))
        except Exception as e:                                   # never block startup on a history read
            log.warning("routing ring restore failed: %s", e)
    try:                                                         # LF: per-client output history survives a restart
        _now2 = time.time()
        _oh = await asyncio.get_running_loop().run_in_executor(
            None, lambda: output_history_restore_blocking(_now2, _PROCESS_STARTED))
        log.info("output history restored: %d completions for %d clients", output_history_restore(_oh), len(_oh))
    except Exception as e:
        log.warning("output history restore failed: %s", e)
    try:                                                         # K4: which call types are short survives a restart
        _now3 = time.time()
        _mh = await asyncio.get_running_loop().run_in_executor(
            None, lambda: micro_history_restore_blocking(_now3, _PROCESS_STARTED))
        log.info("micro-lane history restored: %d completions -> %d signatures", micro_history_restore(_mh), len(_MICRO_HIST))
    except Exception as e:
        log.warning("micro-lane history restore failed: %s", e)
    _spend()   # R2: load (and, for a mid-day ledger, import today's recorded spend) before serving
    app["saver"] = asyncio.create_task(_stats_saver())
    await _hw_seed_slow()                                                  # Lane TL: refill the slow ring from disk
    app["telemetry_sampler"] = asyncio.create_task(_telemetry_sampler())   # TELEMETRY
    app["jsonl_flusher"] = asyncio.create_task(_jsonl_flusher())           # TELEMETRY-HISTORY

async def _on_cleanup(app):
    t = app.get("saver")
    if t:
        t.cancel()
    ts = app.get("telemetry_sampler")   # TELEMETRY
    if ts:
        ts.cancel()
    jf = app.get("jsonl_flusher")       # TELEMETRY-HISTORY
    if jf:
        jf.cancel()
    _save_stats()
    if _JSONL_PENDING:   # best-effort final flush at shutdown -- same posture as _save_stats()
        try:
            lines = list(_JSONL_PENDING)
            _JSONL_PENDING.clear()
            _flush_jsonl_blocking(lines, _JSONL_STATE["date"], _JSONL_STATE["path"],
                                   _JSONL_STATE["bytes"], _JSONL_STATE["capped"])
        except Exception as e:
            log.warning("jsonl final flush failed: %s", e)
    if _HW_PENDING:      # Lane TL: same posture for the hardware-history minute points
        try:
            recs = list(_HW_PENDING)
            _HW_PENDING.clear()
            _hw_append_blocking(recs)
        except Exception as e:
            log.warning("hw history final flush failed: %s", e)

async def _gateway_root_redirect(request):
    """A bare-function / returned-HTTPException handler is deprecated in aiohttp 3.14+
    (surfaced by the test suite's warnings) -- an async handler that RAISES is the
    supported form and won't break on a future aiohttp upgrade."""
    raise web.HTTPFound("/gateway/dashboard")


def make_app():
    app = web.Application(client_max_size=1024**3)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    app.router.add_get("/health", h_health)
    app.router.add_get("/health/local", h_health_local)
    app.router.add_get("/v1/models", h_models)
    app.router.add_post("/v1/chat/completions", handle_completions)
    app.router.add_post("/v1/completions", handle_completions)
    app.router.add_get("/gateway/stats", gateway_stats)
    app.router.add_get("/gateway/capacity", gateway_capacity)                 # CF: capacity-aware flow facts
    app.router.add_get("/gateway/offline", gateway_offline)                   # CF: planned local-offline window
    app.router.add_post("/gateway/offline", gateway_offline)
    app.router.add_delete("/gateway/offline", gateway_offline)
    app.router.add_get("/gateway/drain", gateway_drain)
    app.router.add_post("/gateway/drain", gateway_drain)
    app.router.add_delete("/gateway/drain", gateway_drain)
    app.router.add_get("/gateway/models/local", gateway_models_local)
    app.router.add_post("/gateway/models/local", gateway_models_local)
    app.router.add_get("/gateway/config", gateway_config)
    app.router.add_get("/gateway/config/effective", gateway_config_effective)   # lane CFG: typed effective config
    app.router.add_get("/gateway/internal-errors", gateway_internal_errors)    # lane SH: swallowed exceptions
    app.router.add_get("/gateway/spend", gateway_spend)                        # R2 spend authority
    app.router.add_post("/gateway/spend/reserve", gateway_spend_reserve)
    app.router.add_post("/gateway/spend/finalize", gateway_spend_finalize)
    app.router.add_post("/gateway/spend/recover", gateway_spend_recover)
    app.router.add_post("/gateway/config", gateway_config)
    app.router.add_get("/gateway/aliases", gateway_aliases)
    app.router.add_post("/gateway/aliases", gateway_aliases)
    app.router.add_delete("/gateway/aliases", gateway_aliases)
    app.router.add_get("/gateway/aliases/page", gateway_aliases_page)
    app.router.add_get("/gateway/lanes", gateway_lanes)
    app.router.add_get("/gateway/windows", gateway_windows)
    app.router.add_get("/gateway/windows/{name}/log", gateway_window_log)
    app.router.add_get("/gateway/telemetry", gateway_telemetry)   # TELEMETRY
    app.router.add_get("/gateway/telemetry/history", gateway_telemetry_history)   # Lane TL: persisted hw history
    app.router.add_get("/gateway/metrics", gateway_metrics)       # TELEMETRY
    app.router.add_get("/gateway/history", gateway_history)                   # TELEMETRY-HISTORY
    app.router.add_get("/gateway/history/summary", gateway_history_summary)   # TELEMETRY-HISTORY
    app.router.add_get("/gateway/research/{jid}", gateway_research_detail)
    app.router.add_get("/gateway/dashboard", gateway_dashboard)
    app.router.add_get("/gateway", _gateway_root_redirect)
    app.router.add_route("*", "/{tail:.*}", h_catchall)
    return app


if __name__ == "__main__":
    try:
        _claim_spend_authority()
    except RuntimeError as exc:
        log.critical("gateway refuses duplicate spend authority: %s", exc)
        raise SystemExit(75) from exc
    log.info("gateway-shim on :%d | local=%s | remote=%s model=%s | budget=%d big=%dtok backoff=%ds",
             PORT, LOCAL, REMOTE_BASE or "(none)", REMOTE_MODEL, BUDGET, BIG_TOKENS, OOM_BACKOFF)
    log.info("  tiny-lane<=%dtok +%d lanes | ft-concurrency-scale=%d | req-logging=%s",
             TINY_TOKENS, TINY_EXTRA_LANES, FT_CONCURRENCY_SCALE, LOG_REQUESTS)
    log.info("  guards: big-out>=%dtok big-prompt>=%dtok size-cap>=%dtok first-token-max=%.0fs local-out-cap=%dtok -> remote",
             BIG_OUTPUT, BIG_PROMPT, MAX_LOCAL_TOKENS, FIRST_TOKEN_MAX, LOCAL_MAX_OUT)
    # An accepted stream can be live for a full bounded model turn. A default
    # 60-second aiohttp shutdown would cancel it even if systemd waited longer.
    web.run_app(make_app(), host="0.0.0.0", port=PORT, shutdown_timeout=1800)
