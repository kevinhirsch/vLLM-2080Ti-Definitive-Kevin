#!/usr/bin/env python3
"""
gateway-shim — capacity-routing front door on :8000 for the 2x2080Ti box.

Local-first, remote-overflow, self-healing. Everything is passed through to the
local vLLM (:8001) DYNAMICALLY — no model names are hardcoded, so any model you
add to vLLM tomorrow just works. When local is at capacity or unhealthy, requests
transparently overflow to a remote OpenAI-compatible endpoint (DeepSeek).

Placement (per request):
  - Estimate cost in "units": moderate = 1, big (est. prompt >= SHIM_BIG_TOKENS) = 2.
  - Local budget = SHIM_LOCAL_BUDGET units (default 2) => "2 moderate OR 1 big",
    matching the box's chaos-tested envelope (44GB VRAM, ~1.7GB activation headroom).
  - If local is healthy AND admitting this request stays within budget => LOCAL.
    Otherwise => REMOTE (model rewritten to SHIM_REMOTE_MODEL).
Self-heal:
  - Any local OOM / EngineDead / 5xx / connection error => this request FAILS OVER
    to remote, AND local budget is cut to 1 for SHIM_OOM_BACKOFF_SECS, then recovers.
    So the box finds its own safe concurrency without anyone configuring the number.
Cost-first: a single request is always local (free); only real overflow costs money.

NEVER manages the vLLM process (that's vllm-qwen27b.service + its watchdog).
"""
import asyncio, os, sys, json, time, logging, collections, subprocess, re, hmac
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


PORT         = int(os.environ.get("SHIM_PORT", "8000"))
LOCAL        = os.environ.get("SHIM_UPSTREAM", "http://127.0.0.1:8001").rstrip("/")
# where live config edits (via the dashboard) are persisted so they survive a restart
SHIM_ENV_FILE = os.environ.get("SHIM_ENV_FILE", "/home/kevin/.local/share/vllm-qwen27b/shim.env")
REMOTE_BASE  = os.environ.get("SHIM_REMOTE_BASE", "").rstrip("/")
REMOTE_KEY   = os.environ.get("SHIM_REMOTE_KEY", "")
REMOTE_MODEL = os.environ.get("SHIM_REMOTE_MODEL", "deepseek-v4-flash")
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
EXACT_TOKENS      = os.environ.get("SHIM_EXACT_TOKENS", "1") not in ("0", "false", "")
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
CRASH_ADAPTIVE        = os.environ.get("SHIM_CRASH_ADAPTIVE", "0") not in ("0", "false", "")
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
BG_LOCAL_ONLY    = os.environ.get("SHIM_BG_LOCAL_ONLY", "1") not in ("0", "false", "")
BG_WAIT_LOCAL    = float(os.environ.get("SHIM_BG_WAIT_LOCAL_SECS", "120"))
BG_REJECT_RETRY_SECS = int(os.environ.get("SHIM_BG_REJECT_RETRY_SECS", "300"))
# Background turns don't need chain-of-thought: thinking mode makes a cron status report
# generate 2-3K reasoning tokens and hold a local lane for minutes. Injecting
# enable_thinking=false for LOCAL background requests cuts their lane-hold ~10x.
BG_NO_THINK = os.environ.get("SHIM_BG_NO_THINK", "1") not in ("0", "false", "")
# thinking-budget guard: below THINK_OFF_UNDER tokens disable thinking entirely, below
# THINK_LOW_UNDER downgrade to reasoning_effort=low. Measured starvation point ~1315 tok.
THINK_GUARD     = os.environ.get("SHIM_THINK_GUARD", "1") not in ("0", "false", "")
EMPTY_RETRY     = os.environ.get("SHIM_EMPTY_RETRY", "1") not in ("0", "false", "")
REP_GUARD       = os.environ.get("SHIM_REP_GUARD", "1") not in ("0", "false", "")
REP_MIN_PATTERN = int(os.environ.get("SHIM_REP_MIN_PATTERN", "8"))
REP_MAX_PATTERN = int(os.environ.get("SHIM_REP_MAX_PATTERN", "64"))
REP_MIN_COUNT   = int(os.environ.get("SHIM_REP_MIN_COUNT", "6"))
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
NONTHINK_PROFILE = os.environ.get("SHIM_NONTHINK_PROFILE", "1") not in ("0", "false", "")
NONTHINK_PP      = float(os.environ.get("SHIM_NONTHINK_PP", "1.5"))
NONTHINK_TOP_P   = float(os.environ.get("SHIM_NONTHINK_TOP_P", "0.80"))
NONTHINK_TEMP    = float(os.environ.get("SHIM_NONTHINK_TEMP", "0.7"))
NONTHINK_TOP_K   = int(os.environ.get("SHIM_NONTHINK_TOP_K", "20"))
# MASTER SWITCH: 1 = FULL REMOTE (every completion -> DeepSeek; local engine untouched —
# for maintenance/repro/debugging), 0 = normal local-first. Toggle live from the dashboard.
FORCE_REMOTE = 1 if os.environ.get("SHIM_FORCE_REMOTE", "0").lower() in ("1", "true", "on") else 0
# The third mode (Kevin 2026-09-10: "LOCAL FIRST // FULL REMOTE // FULL LOCAL"):
# 1 = FULL LOCAL — never overflow to the paid remote for ANY request; when every local lane is
# busy a request QUEUES for a lane instead of spending money ($0 guaranteed, latency unbounded).
# Better than unsetting the remote credentials because it is reversible from the dashboard and
# leaves the remote configured for the moment it is wanted again. It OUTRANKS FORCE_REMOTE: if
# both are somehow set, local wins, because the mode that cannot spend money is the safe one.
LOCAL_ONLY = 1 if os.environ.get("SHIM_LOCAL_ONLY", "0").lower() in ("1", "true", "on") else 0
# PEAK-AWARE overflow bias (2026-08-13, DeepSeek peak/off-peak pricing eff. Aug 16):
# during remote-provider PEAK hours (UTC ranges like "1-4,6-10"), BACKGROUND requests
# wait the full LOCAL_WAIT for a local lane instead of fast-overflowing at BG_WAIT —
# biasing robot busywork away from 2x-priced remote. Interactive routing unchanged.
PEAK_HOURS = os.environ.get("SHIM_PEAK_HOURS_UTC", "1-4,6-10")

def is_peak():
    try:
        h = time.gmtime().tm_hour
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


# ---------------- live config (editable from the dashboard, no restart) ----------------
# env-var name -> (global name, caster). Only these are runtime-tunable.
_CFG = {
    # remote overflow provider (any OpenAI-compatible endpoint)
    "SHIM_REMOTE_BASE":      ("REMOTE_BASE",  lambda v: str(v).rstrip("/")),
    "SHIM_REMOTE_KEY":       ("REMOTE_KEY",   str),
    "SHIM_REMOTE_MODEL":     ("REMOTE_MODEL", str),
    "SHIM_FORCE_REMOTE":     ("FORCE_REMOTE", lambda v: 1 if str(v).lower() in ("1","true","on") else 0),
    "SHIM_LOCAL_ONLY":       ("LOCAL_ONLY",   lambda v: 1 if str(v).lower() in ("1","true","on") else 0),
    # local capacity
    "SHIM_LOCAL_BUDGET":     ("BUDGET",       int),
    "SHIM_LOCAL_WAIT_SECS":  ("LOCAL_WAIT",   float),
    "SHIM_TOKEN_BUDGET":     ("TOKEN_BUDGET", int),
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
    "SHIM_BG_XCLIENTS":      ("BG_XCLIENTS", lambda v: [m.lower() for m in _parse_seq(v, _CFG_SEP["SHIM_BG_XCLIENTS"])]),
    "SHIM_POOL_TOKENS":      ("POOL_TOKENS",      int),
    "SHIM_BG_WAIT_SECS":     ("BG_WAIT",          float),
    "SHIM_BG_MARKERS":       ("BG_MARKERS", lambda v: [m for m in str(v).split("|") if m]),
    "SHIM_BG_LOCAL_ONLY":    ("BG_LOCAL_ONLY", lambda v: str(v).lower() not in ("0","false","")),
    "SHIM_BG_WAIT_LOCAL_SECS": ("BG_WAIT_LOCAL", float),
    "SHIM_BG_REJECT_RETRY_SECS": ("BG_REJECT_RETRY_SECS", int),
    "SHIM_PEAK_HOURS_UTC":   ("PEAK_HOURS",       str),
    # behaviour toggles
    "SHIM_BG_NO_THINK":      ("BG_NO_THINK",  lambda v: str(v).lower() not in ("0","false","")),
    "SHIM_THINK_GUARD":      ("THINK_GUARD",  lambda v: str(v).lower() not in ("0","false","")),
    "SHIM_EMPTY_RETRY":      ("EMPTY_RETRY",  lambda v: str(v).lower() not in ("0","false","")),
    "SHIM_REP_GUARD":        ("REP_GUARD",    lambda v: str(v).lower() not in ("0","false","")),
    "SHIM_REP_MIN_PATTERN":  ("REP_MIN_PATTERN", int),
    "SHIM_REP_MAX_PATTERN":  ("REP_MAX_PATTERN", int),
    "SHIM_REP_MIN_COUNT":    ("REP_MIN_COUNT", int),
    "SHIM_THINK_BUDGET_FRAC":("THINK_BUDGET_FRAC", float),
    "SHIM_THINK_BUDGET_MIN": ("THINK_BUDGET_MIN", int),
    "SHIM_THINK_BUDGET_MAX": ("THINK_BUDGET_MAX", int),
    "SHIM_THINK_OFF_UNDER":  ("THINK_OFF_UNDER", int),
    "SHIM_THINK_LOW_UNDER":  ("THINK_LOW_UNDER", int),
    "SHIM_NO_THINK_IPS":     ("NO_THINK_IPS", lambda v: set(_parse_seq(v, _CFG_SEP["SHIM_NO_THINK_IPS"]))),
    "SHIM_LOG_REQUESTS":     ("LOG_REQUESTS", lambda v: str(v).lower() not in ("0","false","")),
    # non-thinking sampling profile (EXP-026)
    "SHIM_NONTHINK_PROFILE": ("NONTHINK_PROFILE", lambda v: str(v).lower() not in ("0","false","")),
    "SHIM_NONTHINK_PP":      ("NONTHINK_PP",    float),
    "SHIM_NONTHINK_TOP_P":   ("NONTHINK_TOP_P", float),
    "SHIM_NONTHINK_TEMP":    ("NONTHINK_TEMP",  float),
    "SHIM_NONTHINK_TOP_K":   ("NONTHINK_TOP_K", int),
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
TOKEN_BUDGET     = int(os.environ.get("SHIM_TOKEN_BUDGET", "500000"))
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
LOG_REQUESTS      = os.environ.get("SHIM_LOG_REQUESTS", "1") not in ("0", "false", "")
LOG_PREVIEW_CHARS = int(os.environ.get("SHIM_LOG_PREVIEW_CHARS", "70"))
# vLLM-only params that a remote OpenAI endpoint would reject — stripped on overflow.
REMOTE_STRIP = ("chat_template_kwargs", "mamba_cache_mode", "guided_decoding_backend")
# Force non-thinking on the DeepSeek failover (see remap_for_remote). Disable with SHIM_REMOTE_NO_THINK=0.
REMOTE_NO_THINK = os.environ.get("SHIM_REMOTE_NO_THINK", "1") not in ("0", "false", "")

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s [gateway] %(levelname)s %(message)s")
log = logging.getLogger("gateway-shim")

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
_waiting  = 0           # requests currently blocked in the queue-first wait loop (backlog)
_backoff_until = 0.0
_health = {"ok": False, "at": 0.0}
REMOTE_ENABLED = bool(REMOTE_BASE and REMOTE_KEY)


def remote_ok():
    """May this request overflow to the PAID remote at all?

    One question, one place. FULL LOCAL (Kevin 2026-09-10) is exactly "the answer is no,
    for every request" -- so every routing decision asks this rather than reading
    REMOTE_ENABLED itself, and a request that would have overflowed waits for a local lane
    instead (the queue-first wait loop already treats "no remote" as an unbounded deadline).
    Reading the globals live is deliberate: both flags are hot-reloadable from the dashboard,
    so the mode changes without a restart and without dropping in-flight work."""
    return bool(REMOTE_ENABLED) and not LOCAL_ONLY


def routing_mode():
    """The dashboard's three states, derived from the same two flags the router uses so the
    badge can never disagree with the behaviour."""
    if LOCAL_ONLY:
        return "full_local"
    if REMOTE_ENABLED and FORCE_REMOTE:
        return "full_remote"
    return "local_first"


STATS_FILE = os.environ.get("SHIM_STATS_FILE", "/home/kevin/.local/share/vllm-qwen27b/gateway-stats.json")

# ---------------- live metrics (for the /gateway/dashboard status page) ----------------
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
                "written": 0, "dropped_cap": 0, "dropped_queue": 0, "last_err": None}
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
    "errors": 0, "cost_est_usd": 0.0})
_ERROR_FEED = collections.deque(maxlen=200)
_cpu_prev = {"t": 0.0, "total": 0, "idle": 0}
_telem_lr_prev = {"local": None, "remote": None}   # previous tick's cumulative local/remote counts


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


# ---- (b) GPU: extended nvidia-smi fields, off-loop refresh. _gpu_stats() (edited below)
# keeps its existing synchronous/cached contract for its existing callers unchanged. ----
_GPU_FIELDS = ("index,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total,"
               "temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,"
               "pcie.link.gen.current,fan.speed")


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
    data = []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=" + _GPU_FIELDS,
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=2).stdout.strip()
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
    kv = _fv(fam, "vllm:kv_cache_usage_perc")
    dq = _counter_rate(fam, prev_fam, "vllm:prefix_cache_queries_total", dt)
    dh = _counter_rate(fam, prev_fam, "vllm:prefix_cache_hits_total", dt)
    prefix_hit_rate = round(dh / dq, 4) if (dq and dh is not None and dq > 0) else None
    prompt_tok_s = _counter_rate(fam, prev_fam, "vllm:prompt_tokens_total", dt)
    gen_tok_s = _counter_rate(fam, prev_fam, "vllm:generation_tokens_total", dt)
    drafted = _counter_rate(fam, prev_fam, "vllm:spec_decode_num_draft_tokens_total", dt)
    accepted = _counter_rate(fam, prev_fam, "vllm:spec_decode_num_accepted_tokens_total", dt)
    spec_rate = round(accepted / drafted, 4) if (drafted and accepted is not None and drafted > 0) else None
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


async def _take_sample():
    loop = asyncio.get_running_loop()
    gpu, host, engine = await asyncio.gather(
        _gpu_stats_async(),
        loop.run_in_executor(None, _host_stats_blocking),
        _scrape_engine_metrics(),
    )
    return {
        "t": time.time(), "gpu": gpu, "host": host, "engine": engine,
        "gateway": {"inflight": _inflight, "budget": effective_budget(), "waiting": _waiting,
                    "backoff_s": max(0, int(_backoff_until - time.time())),
                    "local_healthy": _health.get("ok", False),
                    "remote_share_pct": _remote_share_delta()},
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
    }
    out["host"] = {k: _avg(s["host"].get(k) for s in samples) for k in
                   ("cpu_pct", "ram_used_gb", "ram_total_gb", "disk_free_gb", "models_disk_free_gb")}
    gpu_n = max((len(s["gpu"]) for s in samples), default=0)
    out["gpu"] = []
    for i in range(gpu_n):
        cards = [s["gpu"][i] for s in samples if i < len(s["gpu"])]
        if not cards:
            continue
        out["gpu"].append({k: _avg(c.get(k) for c in cards) for k in
                           ("util", "mem_util", "used", "free", "total", "temp_c", "power_w",
                            "power_limit_w", "clock_sm_mhz", "clock_mem_mhz", "fan_pct")})
    ok_samples = [s["engine"] for s in samples if s["engine"].get("ok")]
    out["engine"] = {"ok": bool(ok_samples)}
    if ok_samples:
        for k in ("running", "waiting", "kv_cache_pct", "prefix_hit_rate", "prompt_tok_s",
                  "gen_tok_s", "spec_accept_rate", "ttft_p50", "ttft_p95", "tpot_p50",
                  "tpot_p95", "e2e_p50", "e2e_p95"):
            out["engine"][k] = _avg(e.get(k) for e in ok_samples)
    return out


async def _telemetry_sampler():
    global _TELEM_TICK
    while True:
        await asyncio.sleep(TELEM_SAMPLE_SECS)
        try:
            sample = await _take_sample()
            _TELEM_FAST.append(sample)
            _TELEM_WINDOW.append(sample)
            _TELEM_TICK += 1
            if _TELEM_TICK % max(1, TELEM_SLOW_EVERY) == 0:
                ds = _downsample(_TELEM_WINDOW)
                if ds:
                    _TELEM_SLOW.append(ds)
                _TELEM_WINDOW.clear()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("telemetry sampler: %s", e)


def _remote_cost_estimate(ptok, outtok):
    """est. remote cost = tokens x configurable $/Mtok, peak/off-peak aware (reuses the
    EXISTING is_peak() rather than a second copy of the peak-hours logic)."""
    if not ptok and not outtok:
        return 0.0
    cin, cout = (REMOTE_COST_IN_PER_MTOK_PEAK, REMOTE_COST_OUT_PER_MTOK_PEAK) if is_peak() \
        else (REMOTE_COST_IN_PER_MTOK, REMOTE_COST_OUT_PER_MTOK)
    return round(((ptok or 0) / 1e6) * cin + ((outtok or 0) / 1e6) * cout, 6)


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
        _active_set(request, **kw)
    except Exception:
        pass


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
        status = getattr(resp, "status", None)
        if status is None:
            status = info.get("http_status")
        c = _PER_CLIENT[name]
        c["requests"] += 1
        if route in ("local", "remote"):
            c[route] += 1
        if outtok is not None:
            c["tokens_out"] += outtok; c["tokens_out_exact"] += outtok
        elif outtok_lb is not None:
            c["tokens_out"] += outtok_lb; c["tokens_out_lb"] += outtok_lb
        c["wait_sum"] += waited; c["wait_n"] += 1
        ttft = info.get("ttft")
        if ttft is not None:
            c["ttft_sum"] += ttft; c["ttft_n"] += 1
        req_cost = 0.0
        if route == "remote":
            req_cost = _remote_cost_estimate(
                info.get("ptok") or 0, outtok if outtok is not None else outtok_lb)
            c["cost_est_usd"] += req_cost
        if (status is not None and status >= 400) or reason == "failover":
            c["errors"] += 1
            _ERROR_FEED.appendleft({"t": round(now, 1), "client": name, "ep": info.get("ep"),
                                    "route": route, "reason": reason, "status": status,
                                    "preview": info.get("preview")})
        # (e) append-only JSONL request log -- see DESIGN.md (e) / REPORT.md. Never write on the
        # request path: this only appends a small dict to a bounded in-memory list;
        # _jsonl_flusher() (sibling to _stats_saver()) does the actual blocking file I/O off the
        # event loop. Reuses every local already computed above -- no new work on this path
        # beyond building one small dict and a bounds-checked list.append().
        _telemetry_log_enqueue({
            "t": round(now, 3), "client": name, "ip": info.get("ip"),
            "xclient": info.get("xclient"), "ua": info.get("ua"),
            "ep": info.get("ep"), "model": info.get("model"),
            "route": route, "reason": reason,
            "waited": round(waited, 3) if waited else 0.0,
            "ttft": ttft, "ptok": info.get("ptok"), "maxtok": info.get("maxtok"),
            "outtok": outtok, "outtok_lb": outtok_lb,
            # RESPONSE-SHAPE fields (shim-remote-observability lane, 2026-09-05). Populated
            # today for route=="remote" non-streaming responses only (_forward_remote ->
            # _note_remote_response -> classify_remote_response; see REPORT.md) -- always None
            # for route=="local" (out of scope for this lane: the local path already has its
            # own real-time empty-response check, _is_empty_thinking_response()/EMPTY_RETRY, it
            # just never persisted the verdict here) and for content_len/content_empty/
            # has_tool_calls on ANY streaming response (not reconstructable from a chunked
            # stream without reassembling every delta -- documented limitation, see REPORT.md).
            # finish_reason is the one exception that CAN show up on a streaming remote
            # response too: _relay()'s _scan_usage best-effort-scans the SSE trailer for it.
            "finish_reason": info.get("finish_reason"),
            "content_len": info.get("content_len"),
            "content_empty": info.get("content_empty"),
            "has_tool_calls": info.get("has_tool_calls"),
            "ptok_exact": info.get("ptok_exact"),
            "duration": round(now - info["t0"], 3) if info.get("t0") else None,
            "status": status, "stream": bool(info.get("stream")),
            "bg": bool(info.get("bg")), "tiny": bool(info.get("tiny")),
            "preview": info.get("preview"), "cost_est": req_cost,
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
                "written": 0, "dropped_cap": len(lines), "rotated": rotated}
    max_bytes = int(TELEMETRY_JSONL_MAX_MB * 1024 * 1024)
    written = dropped_cap = 0
    buf = []
    for rec in lines:
        try:
            s = json.dumps(rec, separators=(",", ":")) + "\n"
        except Exception:
            continue   # one broken record must not lose the rest of the batch
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
    return {"date": cur_date, "path": cur_path, "bytes": cur_bytes, "capped": cur_capped,
            "written": written, "dropped_cap": dropped_cap, "rotated": rotated}


def _telemetry_retention_sweep():
    """Off-loop, called at most once per day (only when _jsonl_flusher notices the UTC date
    rolled over) -- deletes request-log files older than TELEMETRY_RETENTION_DAYS. Narrowly
    scoped on purpose: only files matching 'requests-YYYYMMDD.jsonl' inside TELEMETRY_DIR, never
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
        if not (name.startswith("requests-") and name.endswith(".jsonl")):
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
            _JSONL_STATE["last_err"] = None
            if result.get("rotated"):
                await loop.run_in_executor(None, _telemetry_retention_sweep)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("jsonl flusher: %s", e)
            _JSONL_STATE["last_err"] = str(e)


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


def _history_summary_blocking(since_ts, max_files, hard_line_cap=500000):
    """Off-loop full read (forward this time -- aggregation needs every line in the window
    anyway, so the reverse tail-reader's early-exit trick buys nothing here) of each day-file
    touching [since_ts, now]. `hard_line_cap` is a last-resort circuit breaker for a
    misconfigured huge `hours` value, not expected to bite at the documented 200MB/day cap."""
    now = time.time()
    per_client = collections.defaultdict(lambda: {"requests": 0, "local": 0, "remote": 0,
                                                    "tokens_in": 0, "tokens_out": 0, "errors": 0})
    per_route = collections.defaultdict(lambda: {"requests": 0, "tokens_out": 0, "errors": 0})
    durations, ttfts = [], []
    files_scanned = []
    lines_seen = 0
    truncated = False
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
                    is_err = (status is not None and status >= 400) or rec.get("reason") == "failover"
                    outtok = rec.get("outtok")
                    if outtok is None:
                        outtok = rec.get("outtok_lb") or 0
                    pc = per_client[name]
                    pc["requests"] += 1
                    if route in ("local", "remote"):
                        pc[route] += 1
                    pc["tokens_in"] += rec.get("ptok") or 0
                    pc["tokens_out"] += outtok or 0
                    if is_err:
                        pc["errors"] += 1
                    pr = per_route[route]
                    pr["requests"] += 1
                    pr["tokens_out"] += outtok or 0
                    if is_err:
                        pr["errors"] += 1
                    if rec.get("duration") is not None:
                        durations.append(rec["duration"])
                    if rec.get("ttft") is not None:
                        ttfts.append(rec["ttft"])
        except OSError as e:
            log.warning("history summary: read %s failed: %s", fn, e)
        if truncated:
            break
    durations.sort()
    ttfts.sort()
    total_requests = sum(c["requests"] for c in per_client.values())
    return {
        "per_client": dict(per_client), "per_route": dict(per_route),
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
    _active_set(request, route=decision, reason=reason,
                phase=("done" if decision in ("local", "held") else "remote"))
    _stats["total"] += 1
    if decision == "local":
        _stats["local"] += 1
    elif decision == "held":
        _stats["held"] += 1
    elif decision == "rejected-bg":
        _stats["rejected_bg"] += 1
    else:
        _stats["remote"] += 1
        _remote_reasons[reason] += 1
    if waited and waited > 0:
        _stats["waited_total"] += waited
        _stats["waited_n"] += 1
    _stats["peak_inflight"] = max(_stats["peak_inflight"], _inflight)
    # TELEMETRY: pull whatever _relay()/_forward_remote()/the local-success branches have
    # already stashed on this request's live-registry entry (ttft/outtok are None for the
    # "record-before-forward" remote branches -- see DESIGN.md (c) for exactly why, and where
    # the accurate post-hoc numbers live instead: _telemetry_note_request(), fed from the same
    # _ACTIVE entry but read later, in handle_completions' finally, after the request finishes).
    _rinfo = _ACTIVE.get(id(request)) or {}
    _outtok, _outtok_lb = _rinfo.get("outtok"), _rinfo.get("outtok_lb")
    _events.appendleft({"t": round(time.time(), 1), "d": decision, "r": reason,
                        "client": _client_label(request), "units": units,
                        "waited": round(waited or 0, 1),
                        "ep": request.path.rsplit("/", 1)[-1],
                        "ptok": ptok, "maxtok": maxtok, "stream": stream,
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


# form-field key <-> env-var (what the dashboard sends)
_FIELD_ENV = {k.lower().replace("shim_", ""): k for k in _CFG}

def current_config(masked=True):
    """Every runtime-tunable knob, keyed by dashboard field name (env minus SHIM_)."""
    g = globals()
    out = {}
    for env, (gname, _) in _CFG.items():
        field = env.lower().replace("shim_", "")
        v = g[gname]
        if isinstance(v, (set, frozenset, list, tuple)):
            # Same separator table the env-file writer uses. This value is rendered straight
            # into the dashboard form, which POSTs it back into apply_config -- so if the two
            # ever disagree again, merely opening the page and pressing Save re-corrupts the
            # field. That is precisely how no_think_ips died.
            v = _fmt_seq(v, _CFG_SEP.get(env, _CFG_SEP_DEFAULT))
        elif isinstance(v, bool):
            v = 1 if v else 0
        out[field] = v
    k = g["REMOTE_KEY"]
    out["remote_key_display"] = ("set (" + k[:5] + "\u2026" + k[-4:] + ")") if (masked and k and len(k) > 12) else ("set" if k else "")
    out["remote_key_set"] = bool(k)
    out["admin_token_set"] = bool(SHIM_ADMIN_TOKEN)  # dashboard auth-state badge (never the value itself)
    out["mode"] = routing_mode()  # the three-way badge reads this, never its own guess
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
    g["REMOTE_ENABLED"] = bool(g["REMOTE_BASE"] and g["REMOTE_KEY"])
    if changed and _config_owner():
        _persist_config()
    return changed

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
    agree = sum(1 for k, v in tunable if str(os.environ.get(k, "\0")) == v)
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
                  if isinstance(g[gname], (list, tuple, set, frozenset)) else str(g[gname]))
            for env, (gname, _) in _CFG.items()}
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
    chars = 0
    for m in (j.get("messages") or []):
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
    vals = [v for v in (TINY_TOKENS, BIG_TOKENS, BIG_PROMPT, MAX_LOCAL_TOKENS) if v and v > 0]
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

def _tokenize_exact(text):
    """Ask the engine for the REAL token count instead of guessing chars/3.5.

    The estimator was measured +9% to +19% off on real code and 1.77x off on repetitive
    text, and every routing decision (tiny lane, big-prompt guard, local size cap) keys off
    it. /tokenize costs 10ms on a small prompt and 400ms on a 1.2M-char one -- 0.2-0.8% of
    those requests' own latency, i.e. cheapest exactly where precision matters most.

    Falls back to the estimate (and stops trying for a minute) if the endpoint misbehaves,
    so routing never depends on it being up."""
    global _tok_fail_until
    if time.time() < _tok_fail_until:
        return None
    try:
        req = urllib.request.Request(
            LOCAL.rstrip("/") + "/tokenize",
            json.dumps({"model": _local_model_name(), "prompt": text}).encode(),
            {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=TOKENIZE_TIMEOUT) as r:
            return int(json.load(r).get("count"))
    except Exception as e:
        _tok_fail_until = time.time() + 60
        log.warning("tokenize failed (%s) -> falling back to estimate for 60s", str(e)[:80])
        return None


def _local_model_name():
    return os.environ.get("SHIM_LOCAL_MODEL_NAME", "qwen-local")


def estimate_units(body):
    # A "big" request occupies the WHOLE current local budget (runs alone, nothing concurrent),
    # rather than a fixed 2. This means at budget=1 a single big request still runs LOCAL (safe:
    # 1 big alone never OOMs) instead of needlessly overflowing to DeepSeek, while at budget>=2 it
    # still blocks anything running alongside it.
    return effective_budget() if _est_tokens(body) >= BIG_TOKENS else 1


def first_token_timeout(body, concurrency=1):
    # Prefill throughput is shared across concurrent local requests, so time-to-first-token scales
    # with how many are in flight. Without this, a legit big prefill queued behind others is misread
    # as a "wedged backend" (false failover + a 120s budget=1 backoff cascade). concurrency=1 for
    # remote/uncontended calls preserves the original tight deadline.
    factor = max(1, concurrency) if FT_CONCURRENCY_SCALE else 1
    return min(FIRST_TOKEN_MAX, FIRST_TOKEN_BASE + (_est_tokens(body) / PREFILL_TPS) * factor)


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
    for m in (msgs[:2] + msgs[-1:]):
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
    return (_est_tokens(body) + mt) > MAX_LOCAL_TOKENS


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
    prepared = repetition_guard(nonthinking_sampling_profile(thinking_budget_guard(bound_local_output(body))))
    if _no_think_policy(request, background):
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
    if (j.get("thinking_token_budget") is not None
            or (j.get("chat_template_kwargs") or {}).get("enable_thinking") is not None):
        return body
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
        if MAX_LOCAL_TOKENS <= 0:
            raise ValueError("local output requires a finite token limit")
        return max(prompt, MAX_LOCAL_TOKENS) * count, count
    if isinstance(output, bool) or not isinstance(output, int) or output < 0:
        raise ValueError("local output token limit must be a positive integer")
    return (prompt + output) * count, count


def _memory_available(reservation):
    return TOKEN_BUDGET <= 0 or _inflight_reserved_tokens + reservation <= TOKEN_BUDGET


def wants_stream(body):
    try:
        return bool(json.loads(body).get("stream"))
    except Exception:
        return False


def remap_for_remote(body):
    """Rewrite the model to the remote model and strip vLLM-only params."""
    try:
        j = json.loads(body)
    except Exception:
        return body
    j["model"] = REMOTE_MODEL
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
                                foreign_tokens = max(0, int(kv_pct * POOL_TOKENS) - _inflight_tokens)
                            foreign_heavy = bool(foreign > 0 and BIG_TOKENS > 0 and foreign_tokens >= BIG_TOKENS)
                except Exception:
                    foreign = 0
                    foreign_tokens = 0
                    foreign_heavy = False
    except Exception:
        ok = False
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
    return await session.post(f"{base}{path}", data=body, headers=headers, timeout=to)


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


async def _relay(request, base, path, body, key, streaming, concurrency=1):
    """
    Forward to (base) and relay the response to the client.
    Returns ("ok", web.Response|StreamResponse) on success,
            ("fail", (status, text, is_oom)) if the upstream failed BEFORE the client was
            committed (so the caller may fail over) — including a streaming upstream that
            dies during prefill before producing any token (the FIRST-TOKEN GATE).
    """
    t_relay_start = time.time()   # TELEMETRY: TTFT reference point -- see DESIGN.md (c)
    session = aiohttp.ClientSession()
    try:
        if streaming:
            # bound time-to-response-headers for streaming so a backend that accepts the
            # connection but never responds (a wedge) fails over instead of hanging.
            up = await asyncio.wait_for(_open(session, base, path, body, key, streaming),
                                        timeout=first_token_timeout(body, concurrency))
        else:
            up = await _open(session, base, path, body, key, streaming)
    except asyncio.TimeoutError:
        await session.close()
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
            ct = up.headers.get("Content-Type", "application/json").split(";")[0]
            await session.close()
            return "ok", web.Response(body=data, status=up.status, content_type=ct)
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
            ct = up.headers.get("Content-Type", "application/json").split(";")[0]
            return "ok", web.Response(body=data, status=up.status, content_type=ct)
        finally:
            await session.close()

    # streaming with FIRST-TOKEN GATE: buffer upstream chunks WITHOUT committing to the
    # client until real generation has started. If the upstream dies before that (e.g. an
    # OOM crash during prefill), nothing reached the client yet -> return "fail" so the
    # caller fails the whole request over to remote seamlessly (no error surfaced).
    def _mk_resp():
        return web.StreamResponse(status=up.status, headers={
            "Content-Type": up.headers.get("Content-Type", "text/event-stream"),
            "Cache-Control": "no-cache", "Connection": "keep-alive"})

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
    _is_remote_relay = (base == REMOTE_BASE)

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
                if _exact_finish[0] is None and b'"finish_reason"' in ln:
                    ch = (json.loads(raw).get("choices") or [{}])[0]
                    fr = ch.get("finish_reason")
                    if fr is not None:
                        _exact_finish[0] = fr
                if _exact_outtok[0] is not None and _exact_finish[0] is not None:
                    break
        except Exception:
            pass

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
    deadline = first_token_timeout(body, concurrency)
    try:
        phase = await asyncio.wait_for(_read_until_commit(), timeout=deadline)
    except asyncio.TimeoutError:
        await session.close()
        return "fail", (up.status, f"no first token within {deadline:.0f}s (busy/wedged)", False)
    except Exception as e:
        await session.close()
        return "fail", (up.status, f"pre-commit stream error: {e}", True)

    # TELEMETRY: time to first streamed byte after the upstream POST was issued (excludes the
    # shim's own admission/queue wait, which is tracked separately as `waited`) -- see DESIGN.md
    # (c) for why this is the right spot and why non-streaming has no equivalent insertion point.
    if _t_first[0] is not None:
        _active_set(request, ttft=round(_t_first[0] - t_relay_start, 3))
    _scan_usage(buf)   # TELEMETRY: covers the (common, for short responses) case where the whole
                        # stream -- usage trailer included -- already arrived within the gate

    if phase == "clean_end":
        # upstream finished before any 'meaningful' chunk — deliver as-is (short/empty response),
        # don't fail over (avoids double-generating a real-but-tiny answer).
        resp = _mk_resp(); await resp.prepare(request)
        if buf:
            await resp.write(bytes(buf))
        await resp.write_eof(); await session.close()
        _outkw = {"outtok_lb": _chunk_ct[0]}
        if _exact_outtok[0] is not None:
            _outkw["outtok"] = _exact_outtok[0]
        if _is_remote_relay and _exact_finish[0] is not None:
            _outkw["finish_reason"] = _exact_finish[0]
        _active_set(request, **_outkw)
        return "ok", resp

    # committed to the client: flush buffered first chunk(s), then stream the rest. A mid-stream
    # error past this point can only be reported inline (client already receiving output).
    resp = _mk_resp(); await resp.prepare(request)
    await resp.write(bytes(buf))
    # TELEMETRY: exact output-token count when the client asked for stream_options.include_usage
    # and the usage trailer didn't already arrive within the gate phase above (long streams) --
    # same cheap per-chunk scan, applied to whatever's left of the stream.
    try:
        async for chunk in up.content.iter_any():
            await resp.write(chunk)
            _chunk_ct[0] += 1
            _scan_usage(chunk)
    except Exception as e:
        try:
            await resp.write(f'data: {{"error":{{"message":"stream interrupted: {e}"}}}}\n\n'.encode())
        except Exception:
            pass
    await resp.write_eof(); await session.close()
    _outkw = {"outtok_lb": _chunk_ct[0]}
    if _exact_outtok[0] is not None:
        _outkw["outtok"] = _exact_outtok[0]
    if _is_remote_relay and _exact_finish[0] is not None:
        _outkw["finish_reason"] = _exact_finish[0]
    _active_set(request, **_outkw)
    return "ok", resp


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
        empty_no_tool = cls["content_empty"] and not cls["has_tool_calls"]
        if empty_no_tool:
            log.warning("remote returned EMPTY content (finish_reason=%s, completion_tokens=%s)",
                        cls["finish_reason"], cls["completion_tokens"])
        ptok = _est_tokens(body)
        min_tok = int(os.environ.get("SHIM_FLIGHTREC_MIN_TOK", "15000"))
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


async def _forward_remote(request, path, body, streaming):
    if not remote_ok():
        return web.json_response(
            {"error": {"message": "local unavailable and no remote overflow configured"}}, status=503)
    kind, payload = await _relay(request, REMOTE_BASE, path, remap_for_remote(body), REMOTE_KEY, streaming)
    if kind == "ok":
        _note_payload_outcome(request, payload, streaming)   # TELEMETRY: see DESIGN.md (c)
        if not streaming:
            # TELEMETRY + FLIGHT RECORDER (shim-remote-observability lane, 2026-09-05): see
            # _note_remote_response()'s docstring. Streaming remote responses only get the
            # finish_reason capture inside _relay() above (best-effort, SSE-scan-based);
            # content-shape telemetry and the flight recorder are non-streaming-only.
            await _note_remote_response(request, payload, body)
        return payload
    status, text, _ = payload
    return web.json_response({"error": {"message": f"remote overflow failed: {status} {text}"}}, status=502)


async def handle_completions(request):
    """Register the request as live work for the dashboard, then run the router (below)."""
    body = await request.read()          # aiohttp caches the body; the router re-reads it for free
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
    info.update({"ep": request.path.rsplit("/", 1)[-1], "model": str(j.get("model") or "")[:40],
                 "ptok": _est_tokens(body), "maxtok": maxtok, "stream": wants_stream(body),
                 "t0": time.time(), "phase": "routing", "route": None, "reason": None,
                 "preview": _preview(body)[:100], "bg": is_background(body, request), "tiny": is_tiny(body)})
    _ACTIVE[id(request)] = info
    _resp = None
    try:
        _resp = await _route_completions(request)
        return _resp
    finally:
        _info = _ACTIVE.pop(id(request), None)
        if _info is not None:
            _telemetry_note_request(_info, _resp)   # TELEMETRY: per-client rollups + error feed


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


async def _route_completions(request):
    global _inflight, _waiting, _inflight_tokens, _inflight_reserved_tokens
    path = request.path
    body = await request.read()
    units = estimate_units(body)
    streaming = wants_stream(body)
    ptok = _est_tokens(body)
    try:
        maxtok = int(json.loads(body).get("max_tokens") or 0)
    except Exception:
        maxtok = 0
    ev = dict(ptok=ptok, maxtok=maxtok, stream=streaming)
    tiny = is_tiny(body)
    background = is_background(body, request)
    # BG-LOCAL-ONLY: set to the triggering reason ("local-down") once a background request has
    # been HELD for an engine-health recovery below. When set, a local success further down in
    # this function (tiny fast-lane / empty-retry / normal admission) is recorded as
    # route="held" instead of route="local" -- same response to the client either way, just
    # telemetry that distinguishes "recovered from a held outage" from an ordinary local hit.
    # None for every request that never went through the hold -- i.e. everything today already
    # covers, byte-for-byte the same recorded route as before.
    bg_held_reason = None

    if LOG_REQUESTS:
        try:
            model_req = json.loads(body).get("model", "?")
        except Exception:
            model_req = "?"
        log.info("REQ ip=%s ua=%r model=%s ptok=%d maxtok=%d stream=%s tiny=%s bg=%s preview=%r",
                 getattr(request, "remote", "?"), request.headers.get("User-Agent", "?")[:45],
                 model_req, ptok, maxtok, streaming, tiny, background, _preview(body))

    # MASTER SWITCH: full-remote mode (maintenance/debug) — everything -> DeepSeek
    if remote_ok() and FORCE_REMOTE:
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

    # size cap: too-big-for-this-box requests OOM local even at budget=1 -> send straight to remote
    if remote_ok() and not local_pin and over_local_cap(body):
        log.info("route %s est prompt+max > %d -> remote(size)", path, MAX_LOCAL_TOKENS)
        record_event("remote", "size", request, units, 0, **ev)
        return await _forward_remote(request, path, body, streaming)

    # big-OUTPUT requests (e.g. Hermes max_tokens=65536): long generations that saturate this slow
    # box and hang interactive clients -> straight to remote (DeepSeek serves them far faster).
    if remote_ok() and not local_pin and BIG_OUTPUT > 0 and maxtok >= BIG_OUTPUT:
        log.info("route %s maxtok=%d >= %d -> remote(big-out)", path, maxtok, BIG_OUTPUT)
        record_event("remote", "big-out", request, units, 0, **ev)
        return await _forward_remote(request, path, body, streaming)

    # big-PROMPT requests (e.g. a pi session whose context has grown huge): can't prefill within the
    # first-token cap on this slow box and would saturate/OOM local -> straight to remote up front.
    if remote_ok() and not local_pin and BIG_PROMPT > 0 and ptok >= BIG_PROMPT:
        log.info("route %s ptok=%d >= %d -> remote(big-prompt)", path, ptok, BIG_PROMPT)
        record_event("remote", "big-prompt", request, units, 0, **ev)
        return await _forward_remote(request, path, body, streaming)

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
        elif remote_ok():
            log.info("route %s local unhealthy -> remote(local-down)", path)
            record_event("remote", "local-down", request, units, 0, **ev)
            return await _forward_remote(request, path, body, streaming)

    # MONSTER-IN-FLIGHT bypass: a huge prefill is monopolizing engine steps; anything admitted
    # now would crawl (~1 tok per chunk-step). Route new arrivals remote until it drains.
    _foreign = _health.get("foreign", 0) if FOREIGN_LOAD_GUARD else 0
    _foreign_heavy = bool(_health.get("foreign_heavy", False)) if FOREIGN_LOAD_GUARD else False
    if remote_ok() and not local_pin and (
        (MONSTER_INFLIGHT > 0 and _inflight_tokens >= MONSTER_INFLIGHT)
        or _foreign_heavy
    ):
        log.info("route %s monster/foreign inflight tok=%d foreign=%d foreign_tok=%d -> remote(monster)",
                 path, _inflight_tokens, _foreign, int(_health.get("foreign_tokens", 0) or 0))
        record_event("remote", "monster", request, units, 0, **ev)
        return await _forward_remote(request, path, body, streaming)

    try:
        local_body = _prepare_local_body(request, body, background)
        reservation, sequences = local_memory_reservation(local_body)
    except (ValueError, TypeError, AttributeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
    if TOKEN_BUDGET > 0 and reservation > TOKEN_BUDGET:
        # Waiting cannot make a request larger than the entire pool admissible.
        if remote_ok() and not local_pin:
            record_event("remote", "tokens", request, units, 0, **ev)
            return await _forward_remote(request, path, body, streaming)
        return web.json_response({"error": "request exceeds local token reservation budget"}, status=503)
    units *= sequences
    reserved = False

    def claim_local():
        nonlocal reserved
        global _inflight, _inflight_tokens, _inflight_reserved_tokens
        _inflight += units
        _inflight_tokens += ptok
        _inflight_reserved_tokens += reservation
        reserved = True

    def release_local():
        nonlocal reserved
        global _inflight, _inflight_tokens, _inflight_reserved_tokens
        if reserved:
            _inflight -= units
            _inflight_tokens -= ptok
            _inflight_reserved_tokens -= reservation
            reserved = False

    # TINY fast-lane: small calls skip the queue, but never the KV memory limit.
    # BEYOND the big-request budget (bounded by TINY_EXTRA_LANES, staying within max-num-seqs), so a
    # trivial call never eats a 15s wait or gets starved during a budget=1 backoff. If even that
    # headroom is full, fast-overflow immediately (a tiny call on DeepSeek is cheap + fast).
    if tiny:
        if (_health["ok"] and (_inflight + units) <= (effective_budget() + TINY_EXTRA_LANES)
                and _memory_available(reservation)):
            claim_local()
            log.info("route %s TINY units=%d inflight=%d/%d(+%d) -> local(tiny)",
                     path, units, _inflight, effective_budget(), TINY_EXTRA_LANES)
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
                                 bg_held_reason or "tiny", request, units, 0, **ev)
                    return payload
                status, text, oom = payload
                if oom:
                    trigger_backoff(f"local {status}: {text[:120]}")
                log.warning("local(tiny) failed (%s) -> failover to remote", status)
                record_event("remote", "failover", request, units, 0, **ev)
                release_local()
                return await _forward_remote(request, path, body, streaming)
            finally:
                release_local()
        elif remote_ok():
            log.info("route %s TINY inflight=%d/%d(+%d) full -> remote(tiny-fast)",
                     path, _inflight, effective_budget(), TINY_EXTRA_LANES)
            record_event("remote", "tiny-fast" if _memory_available(reservation) else "tokens", request, units, 0, **ev)
            return await _forward_remote(request, path, body, streaming)
        # no remote configured -> fall through to the normal local wait loop

    # local UP: claim a slot, WAITING up to LOCAL_WAIT for capacity instead of instant-overflow.
    # PRIORITY LANES: background (cron/batch) may only fill lanes beyond FG_RESERVED — those
    # stay free so an interactive turn NEVER queues behind robot busywork — and background
    # waits only BG_WAIT before overflowing to the cheap remote.
    # The check-and-increment is done with no await in between, so it's race-free under asyncio.
    if background:
        lane_limit = max(1, effective_budget() - FG_RESERVED)
        _bgw = LOCAL_WAIT if is_peak() else BG_WAIT   # peak: bias bg toward local queueing
        deadline = time.time() + (_bgw if remote_ok() else 1e9)
    else:
        lane_limit = effective_budget()
        deadline = time.time() + (LOCAL_WAIT if remote_ok() else 1e9)
    admitted = False
    _local_reason = "-"      # "bg-big-idle" when the idle-engine rule admitted a big background request
    admitted_conc = 1        # local concurrency at admission -> scales the first-token deadline
    waited = 0.0
    queued = False
    try:
        while True:
            # big background request + idle engine: nothing in flight and nobody else queued
            # (this request counts itself in _waiting once queued) -> may take the whole budget.
            _bg_big_idle = (background and BG_BIG_LOCAL_WHEN_IDLE and units >= effective_budget()
                            and units <= effective_budget()
                            and _inflight == 0 and _waiting <= (1 if queued else 0))
            if _health["ok"] and ((_inflight + units) <= lane_limit or _bg_big_idle) \
                    and _memory_available(reservation):
                if _bg_big_idle and (_inflight + units) > lane_limit:
                    _stats["bg_big_idle_local"] = _stats.get("bg_big_idle_local", 0) + 1
                    _local_reason = "bg-big-idle"
                claim_local()
                admitted_conc = _inflight
                admitted = True
                break
            if time.time() >= deadline:
                break
            if not queued:                       # first time we couldn't get a slot -> we're backlogged
                queued = True
                _active_set(request, phase="queued")
                _waiting += 1
                _stats["peak_waiting"] = max(_stats["peak_waiting"], _waiting)
            await asyncio.sleep(SLOT_POLL)
            waited += SLOT_POLL
            await local_healthy()  # refresh cached health while waiting
    finally:
        if queued:
            _waiting -= 1

    if not admitted:
        # distinguish WHY we couldn't admit: lane-count vs total-context (size-aware) cap
        if (_inflight + units) <= lane_limit:
            reason = "tokens"
        elif background and (_inflight + units) <= effective_budget():
            reason = "bg-yield"      # lanes exist but are reserved for interactive
        else:
            reason = "cap"
        where = f"remote({reason})" if remote_ok() else "remote(none)"
        log.info("route %s units=%d inflight=%d/%d tok=%d/%d waited=%.1fs -> %s",
                 path, units, _inflight, effective_budget(), _inflight_tokens, TOKEN_BUDGET, waited, where)
        if queued:
            _stats["overflowed_after_wait"] += 1
        record_event("remote", reason, request, units, waited, **ev)
        return await _forward_remote(request, path, body, streaming)

    log.info("route %s units=%d inflight=%d/%d waited=%.1fs -> local%s",
             path, units, _inflight, effective_budget(), waited,
             "(bg-big-idle)" if _local_reason == "bg-big-idle" else "")
    _active_set(request, phase="local", route="local", waited=round(waited, 1))
    # FLIGHT RECORDER (RCA, 2026-08-13): persist big local-routed request bodies so the
    # next Xid-31 crash leaves a deterministic repro payload. Ring of 40 files, 0600.
    # Directory made configurable (SHIM_FLIGHTREC_DIR, default unchanged) by the
    # shim-remote-observability lane, 2026-09-05 -- see _flightrec_dir()'s docstring.
    try:
        if ptok >= int(os.environ.get("SHIM_FLIGHTREC_MIN_TOK", "15000")):
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
        record_event("remote", "failover", request, units, waited, **ev)
        release_local()
        return await _forward_remote(request, path, body, streaming)
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


async def h_models(request):
    # Pure passthrough of whatever vLLM serves (new models appear automatically).
    try:
        return await _passthrough(request)
    except Exception as e:
        log.warning("h_models: local unreachable (%r)", e)
        return web.json_response({"object": "list", "data": []}, status=503)


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
        "local_healthy": up,
        "budget": effective_budget(), "configured_budget": BUDGET,
        "inflight": _inflight, "peak_inflight": _stats["peak_inflight"],
        "waiting": _waiting, "peak_waiting": _stats["peak_waiting"],
        "overflowed_after_wait": _stats["overflowed_after_wait"],
        "inflight_tokens": _inflight_tokens, "token_budget": TOKEN_BUDGET,
        "inflight_reserved_tokens": _inflight_reserved_tokens,
        "backoff": max(0, int(_backoff_until - time.time())),
        "remote_model": REMOTE_MODEL, "remote_enabled": REMOTE_ENABLED,
        "local_only": bool(LOCAL_ONLY), "mode": routing_mode(),
        "local_wait": LOCAL_WAIT,
        "monster_inflight": MONSTER_INFLIGHT,
        "total": _stats["total"], "local": _stats["local"], "remote": _stats["remote"],
        "held": _stats["held"], "rejected_bg": _stats["rejected_bg"],
        "local_pct": round(100 * _stats["local"] / total, 1),
        "remote_pct": round(100 * _stats["remote"] / total, 1),
        "avg_wait": round(_stats["waited_total"] / (_stats["waited_n"] or 1), 1),
        "remote_reasons": dict(_remote_reasons),
        "gpu": _gpu_stats(),
        "events": list(_events)[:60],
    })


# ---- TELEMETRY (added): /gateway/telemetry (JSON) + /gateway/metrics (Prometheus text).
# See LANE/DESIGN.md (f). Both are read-only, additive routes with no interaction with the
# routing/failover path -- worst case (engine down, as observed live all session) they report
# a degraded field, never a 5xx of their own. ----
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
        }
    return web.json_response({
        "at": time.time(),
        "latest": latest,
        "series": {"fast": list(_TELEM_FAST), "slow": list(_TELEM_SLOW)},
        "per_client": per_client,
        "errors": list(_ERROR_FEED)[:60],
        "percentiles": pct,
        "engine_scrape": {"ok": _ENGINE_METRICS["ok"],
                          "age_s": round(time.time() - _ENGINE_METRICS["at"], 1) if _ENGINE_METRICS["at"] else None,
                          "err": _ENGINE_METRICS["err"]},
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
    duration+TTFT, tokens in/out, and error counts over the trailing `hours` (default 24, capped
    at 30 days). Cached for HISTORY_SUMMARY_CACHE_TTL seconds (default 5s): the History dashboard
    section polls this every 10s and a full scan of one-to-several ~200MB/day files on every
    single poll is real disk+CPU work not worth repeating for back-to-back callers -- see
    REPORT.md disk-math."""
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
        return web.json_response(data)
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
    return web.json_response(out)


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


async def gateway_dashboard(request):
    return web.Response(text=DASHBOARD_HTML, content_type="text/html")

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
 <span class=rz><abbr title="This gateway listens on :8000 and forwards to the local vLLM engine on :8001. It queues, serialises, and (when local is full) overflows to a paid remote provider.">:8000 capacity-routing gateway</abbr> &rarr; local engine :8001 &middot; overflow provider <b id=rm class=mono>--</b></span>
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
   <label><span class=k>How long should an interactive request wait for a free lane?</span><span class=hint>seconds, before overflowing to the paid provider</span>
    <input id=f_local_wait_secs type=number min=0 step=1></label>
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
   <label><span class=k>A request this big claims the whole budget alone</span><span class=hint>tokens (serialize-solo)</span>
    <input id=f_big_tokens type=number min=0 step=1000></label>
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
      `<div class=tile><div class=k><abbr title="Concurrent request slots (lanes) in use / total available. Waiting shown when requests are queued for one.">Lanes</abbr></div><div class=v class=mono>${s.inflight}<small>/${s.budget}</small>${s.waiting?` <span style=color:var(--amb)>+${s.waiting} waiting</span>`:''}</div><div class=sub><abbr title="Reserved prompt plus bounded output tokens across all lanes / token-budget cap.">context reserved ${((s.inflight_reserved_tokens||0)/1000).toFixed(0)}K / ${((s.token_budget||0)/1000).toFixed(0)}K cap</abbr>${bk?' &middot; '+bk:''}</div></div>`,
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
      if(queued&&CFG){
        const waitS=(a.bg?CFG.bg_wait_secs:CFG.local_wait_secs);
        if(waitS!=null)deadline=Math.max(0,Number(waitS)-(a.waited||0));
      }
      rows.push({kind:'request',who:clientDisplay(a.name,a.ip,a.ua),
        what:(a.preview?'"'+esc(a.preview)+'"':'<span class=rz>(no text preview)</span>')+(a.bg?' &middot; background':'')+(a.tiny?' &middot; tiny':'')+(a.model?' &middot; '+esc(a.model):''),
        state, badgeText: queued?'waiting for a lane'+(deadline!=null?' \u00b7 overflows in ~'+Math.round(deadline)+'s':''):held?'holding for local (background) -- never billed':remote?'remote overflow \u00b7 '+esc(a.reason||''):a.phase==='local'?'local model'+(a.waited?' \u00b7 waited '+a.waited+'s':''):'routing',
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

async def _on_startup(app):
    _load_stats()
    app["saver"] = asyncio.create_task(_stats_saver())
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
    app.router.add_get("/gateway/models/local", gateway_models_local)
    app.router.add_post("/gateway/models/local", gateway_models_local)
    app.router.add_get("/gateway/config", gateway_config)
    app.router.add_post("/gateway/config", gateway_config)
    app.router.add_get("/gateway/lanes", gateway_lanes)
    app.router.add_get("/gateway/windows", gateway_windows)
    app.router.add_get("/gateway/windows/{name}/log", gateway_window_log)
    app.router.add_get("/gateway/telemetry", gateway_telemetry)   # TELEMETRY
    app.router.add_get("/gateway/metrics", gateway_metrics)       # TELEMETRY
    app.router.add_get("/gateway/history", gateway_history)                   # TELEMETRY-HISTORY
    app.router.add_get("/gateway/history/summary", gateway_history_summary)   # TELEMETRY-HISTORY
    app.router.add_get("/gateway/research/{jid}", gateway_research_detail)
    app.router.add_get("/gateway/dashboard", gateway_dashboard)
    app.router.add_get("/gateway", _gateway_root_redirect)
    app.router.add_route("*", "/{tail:.*}", h_catchall)
    return app


if __name__ == "__main__":
    log.info("gateway-shim on :%d | local=%s | remote=%s model=%s | budget=%d big=%dtok backoff=%ds",
             PORT, LOCAL, REMOTE_BASE or "(none)", REMOTE_MODEL, BUDGET, BIG_TOKENS, OOM_BACKOFF)
    log.info("  tiny-lane<=%dtok +%d lanes | ft-concurrency-scale=%d | req-logging=%s",
             TINY_TOKENS, TINY_EXTRA_LANES, FT_CONCURRENCY_SCALE, LOG_REQUESTS)
    log.info("  guards: big-out>=%dtok big-prompt>=%dtok size-cap>=%dtok first-token-max=%.0fs local-out-cap=%dtok -> remote",
             BIG_OUTPUT, BIG_PROMPT, MAX_LOCAL_TOKENS, FIRST_TOKEN_MAX, LOCAL_MAX_OUT)
    web.run_app(make_app(), host="0.0.0.0", port=PORT)
