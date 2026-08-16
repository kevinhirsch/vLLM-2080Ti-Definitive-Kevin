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
import asyncio, os, sys, json, time, logging, collections, subprocess
import urllib.request
import aiohttp
from aiohttp import web

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
BG_WAIT     = float(os.environ.get("SHIM_BG_WAIT_SECS", "5"))
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
# MASTER SWITCH: 1 = FULL REMOTE (every completion -> DeepSeek; local engine untouched —
# for maintenance/repro/debugging), 0 = normal local-first. Toggle live from the dashboard.
FORCE_REMOTE = 1 if os.environ.get("SHIM_FORCE_REMOTE", "0").lower() in ("1", "true", "on") else 0
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
NO_THINK_IPS = {ip.strip() for ip in os.environ.get("SHIM_NO_THINK_IPS", "").split(",") if ip.strip()}
# A single request whose est. (prompt + max_tokens) exceeds this will OOM local even at
# budget=1 (context + generation peaks past this box's tiny free VRAM), so route it straight
# to remote. Prevents single-big-request OOM crashes. (2026-08-12: observed a solo OOM here.)
MAX_LOCAL_TOKENS = int(os.environ.get("SHIM_MAX_LOCAL_TOKENS", "80000"))
# Size-aware admission: cap TOTAL in-flight prompt tokens across local lanes (activation ∝
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

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s [gateway] %(levelname)s %(message)s")
log = logging.getLogger("gateway-shim")

_inflight = 0            # local capacity units currently in flight
_inflight_tokens = 0    # sum of est prompt tokens of in-flight local requests (size-aware cap)
_waiting  = 0           # requests currently blocked in the queue-first wait loop (backlog)
_backoff_until = 0.0
_health = {"ok": False, "at": 0.0}
REMOTE_ENABLED = bool(REMOTE_BASE and REMOTE_KEY)
STATS_FILE = os.environ.get("SHIM_STATS_FILE", "/home/kevin/.local/share/vllm-qwen27b/gateway-stats.json")

# ---------------- live metrics (for the /gateway/dashboard status page) ----------------
_stats = {"started": time.time(), "total": 0, "local": 0, "remote": 0,
          "waited_total": 0.0, "waited_n": 0, "peak_inflight": 0, "peak_waiting": 0,
          "overflowed_after_wait": 0}
_remote_reasons = collections.Counter()
_events = collections.deque(maxlen=200)   # most-recent-first ring buffer of routing decisions
_gpu_cache = {"at": 0.0, "data": []}

def _client_label(request):
    # harness self-id via X-Client/X-Title header, else source IP
    return request.headers.get("X-Client") or request.headers.get("X-Title") \
        or getattr(request, "remote", None) or "?"

def record_event(decision, reason, request, units, waited, ptok=0, maxtok=0, stream=False):
    _stats["total"] += 1
    if decision == "local":
        _stats["local"] += 1
    else:
        _stats["remote"] += 1
        _remote_reasons[reason] += 1
    if waited and waited > 0:
        _stats["waited_total"] += waited
        _stats["waited_n"] += 1
    _stats["peak_inflight"] = max(_stats["peak_inflight"], _inflight)
    _events.appendleft({"t": round(time.time(), 1), "d": decision, "r": reason,
                        "client": _client_label(request), "units": units,
                        "waited": round(waited or 0, 1),
                        "ep": request.path.rsplit("/", 1)[-1],
                        "ptok": ptok, "maxtok": maxtok, "stream": stream})

def _gpu_stats():
    now = time.time()
    if now - _gpu_cache["at"] < 1.5:
        return _gpu_cache["data"]
    data = []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=2).stdout.strip()
        for line in out.splitlines():
            u, f, g = [int(x) for x in line.split(", ")]
            data.append({"used": u, "free": f, "util": g})
    except Exception:
        pass
    _gpu_cache.update(at=now, data=data)
    return data

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


# ---------------- live config (editable from the dashboard, no restart) ----------------
# env-var name -> (global name, caster). Only these are runtime-tunable.
_CFG = {
    # remote overflow provider (any OpenAI-compatible endpoint)
    "SHIM_REMOTE_BASE":      ("REMOTE_BASE",  lambda v: str(v).rstrip("/")),
    "SHIM_REMOTE_KEY":       ("REMOTE_KEY",   str),
    "SHIM_REMOTE_MODEL":     ("REMOTE_MODEL", str),
    "SHIM_FORCE_REMOTE":     ("FORCE_REMOTE", lambda v: 1 if str(v).lower() in ("1","true","on") else 0),
    # local capacity
    "SHIM_LOCAL_BUDGET":     ("BUDGET",       int),
    "SHIM_LOCAL_WAIT_SECS":  ("LOCAL_WAIT",   float),
    "SHIM_TOKEN_BUDGET":     ("TOKEN_BUDGET", int),
    "SHIM_OOM_BACKOFF_SECS": ("OOM_BACKOFF",  int),
    # routing guards
    "SHIM_BIG_TOKENS":       ("BIG_TOKENS",       int),
    "SHIM_BIG_OUTPUT":       ("BIG_OUTPUT",       int),
    "SHIM_BIG_PROMPT":       ("BIG_PROMPT",       int),
    "SHIM_MAX_LOCAL_TOKENS": ("MAX_LOCAL_TOKENS", int),
    "SHIM_LOCAL_MAX_OUT":    ("LOCAL_MAX_OUT",    int),
    "SHIM_FIRST_TOKEN_MAX":  ("FIRST_TOKEN_MAX",  float),
    "SHIM_PREFILL_TPS":      ("PREFILL_TPS",      float),
    # lanes / priority
    "SHIM_TINY_TOKENS":      ("TINY_TOKENS",      int),
    "SHIM_TINY_EXTRA_LANES": ("TINY_EXTRA_LANES", int),
    "SHIM_FG_RESERVED":      ("FG_RESERVED",      int),
    "SHIM_BG_WAIT_SECS":     ("BG_WAIT",          float),
    "SHIM_BG_MARKERS":       ("BG_MARKERS", lambda v: [m for m in str(v).split("|") if m]),
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
    "SHIM_NO_THINK_IPS":     ("NO_THINK_IPS", lambda v: {i.strip() for i in str(v).split(",") if i.strip()}),
    "SHIM_LOG_REQUESTS":     ("LOG_REQUESTS", lambda v: str(v).lower() not in ("0","false","")),
}
# A single request whose est. (prompt + max_tokens) exceeds this will OOM local even at
# budget=1 (context + generation peaks past this box's tiny free VRAM), so route it straight
# to remote. Prevents single-big-request OOM crashes. (2026-08-12: observed a solo OOM here.)
MAX_LOCAL_TOKENS = int(os.environ.get("SHIM_MAX_LOCAL_TOKENS", "80000"))
# Size-aware admission: cap TOTAL in-flight prompt tokens across local lanes (activation ∝
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

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s [gateway] %(levelname)s %(message)s")
log = logging.getLogger("gateway-shim")

_inflight = 0            # local capacity units currently in flight
_inflight_tokens = 0    # sum of est prompt tokens of in-flight local requests (size-aware cap)
_waiting  = 0           # requests currently blocked in the queue-first wait loop (backlog)
_backoff_until = 0.0
_health = {"ok": False, "at": 0.0}
REMOTE_ENABLED = bool(REMOTE_BASE and REMOTE_KEY)
STATS_FILE = os.environ.get("SHIM_STATS_FILE", "/home/kevin/.local/share/vllm-qwen27b/gateway-stats.json")

# ---------------- live metrics (for the /gateway/dashboard status page) ----------------
_stats = {"started": time.time(), "total": 0, "local": 0, "remote": 0,
          "waited_total": 0.0, "waited_n": 0, "peak_inflight": 0, "peak_waiting": 0,
          "overflowed_after_wait": 0}
_remote_reasons = collections.Counter()
_events = collections.deque(maxlen=200)   # most-recent-first ring buffer of routing decisions
_gpu_cache = {"at": 0.0, "data": []}

def _client_label(request):
    # harness self-id via X-Client/X-Title header, else source IP
    return request.headers.get("X-Client") or request.headers.get("X-Title") \
        or getattr(request, "remote", None) or "?"

def record_event(decision, reason, request, units, waited, ptok=0, maxtok=0, stream=False):
    _stats["total"] += 1
    if decision == "local":
        _stats["local"] += 1
    else:
        _stats["remote"] += 1
        _remote_reasons[reason] += 1
    if waited and waited > 0:
        _stats["waited_total"] += waited
        _stats["waited_n"] += 1
    _stats["peak_inflight"] = max(_stats["peak_inflight"], _inflight)
    _events.appendleft({"t": round(time.time(), 1), "d": decision, "r": reason,
                        "client": _client_label(request), "units": units,
                        "waited": round(waited or 0, 1),
                        "ep": request.path.rsplit("/", 1)[-1],
                        "ptok": ptok, "maxtok": maxtok, "stream": stream})

def _gpu_stats():
    now = time.time()
    if now - _gpu_cache["at"] < 1.5:
        return _gpu_cache["data"]
    data = []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=2).stdout.strip()
        for line in out.splitlines():
            u, f, g = [int(x) for x in line.split(", ")]
            data.append({"used": u, "free": f, "util": g})
    except Exception:
        pass
    _gpu_cache.update(at=now, data=data)
    return data

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


# ---------------- live config (editable from the dashboard, no restart) ----------------
# env-var name -> (global name, caster). Only these are runtime-tunable.
# form-field key <-> env-var (what the dashboard sends)
_FIELD_ENV = {k.lower().replace("shim_", ""): k for k in _CFG}

def current_config(masked=True):
    """Every runtime-tunable knob, keyed by dashboard field name (env minus SHIM_)."""
    g = globals()
    out = {}
    for env, (gname, _) in _CFG.items():
        field = env.lower().replace("shim_", "")
        v = g[gname]
        if isinstance(v, (set, list)):
            v = ",".join(sorted(v)) if isinstance(v, set) else "|".join(v)
        elif isinstance(v, bool):
            v = 1 if v else 0
        out[field] = v
    k = g["REMOTE_KEY"]
    out["remote_key_display"] = ("set (" + k[:5] + "\u2026" + k[-4:] + ")") if (masked and k and len(k) > 12) else ("set" if k else "")
    out["remote_key_set"] = bool(k)
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
    if changed:
        _persist_config()
    return changed

def _persist_config():
    """Rewrite SHIM_ENV_FILE with current tunable values (atomic, mode 600)."""
    g = globals()
    vals = {env: str(g[gname]) for env, (gname, _) in _CFG.items()}
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
    if not EXACT_TOKENS:
        return int(chars / CHARS_PER_TOK)
    # A char count can only ever correspond to FEWER tokens than chars/MIN_CHARS_PER_TOK.
    # If even that pessimistic bound is under every decision threshold, the exact number
    # cannot change any routing decision, so skip the round-trip. This is an optimisation
    # with a proof, not a heuristic about the answer.
    if chars / MIN_CHARS_PER_TOK < _min_decision_threshold():
        return int(chars / CHARS_PER_TOK)
    exact = _tokenize_exact(_prompt_text(j))
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
    if "cron" in xc or "batch" in xc:
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
    mt = j.get("max_tokens")
    if not mt or int(mt) <= 0 or int(mt) > LOCAL_MAX_OUT:
        j["max_tokens"] = LOCAL_MAX_OUT
        return json.dumps(j).encode()
    return body


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
    return json.dumps(j).encode()


async def local_healthy():
    now = time.time()
    if now - _health["at"] < HEALTH_TTL:
        return _health["ok"]
    ok = False
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{LOCAL}/health", timeout=aiohttp.ClientTimeout(total=2)) as r:
                ok = (r.status == 200)
    except Exception:
        ok = False
    _health.update(ok=ok, at=now)
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
    """True once the upstream emitted real generated progress (a content/reasoning token
    or a finish_reason) — the safe point to commit the response to the client."""
    i = text.find('"reasoning_content":"')
    if i != -1 and text[i + 21:i + 22] not in ("", '"'):
        return True
    i = text.find('"content":"')
    if i != -1 and text[i + 11:i + 12] not in ("", '"'):
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

    async def _read_until_commit():
        nchunks = 0
        async for chunk in up.content.iter_any():
            buf.extend(chunk)
            nchunks += 1
            if (_looks_meaningful(buf.decode("utf-8", "replace"))
                    or len(buf) >= FIRST_GATE_MAX_BYTES or nchunks >= FIRST_GATE_MAX_CHUNKS):
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

    if phase == "clean_end":
        # upstream finished before any 'meaningful' chunk — deliver as-is (short/empty response),
        # don't fail over (avoids double-generating a real-but-tiny answer).
        resp = _mk_resp(); await resp.prepare(request)
        if buf:
            await resp.write(bytes(buf))
        await resp.write_eof(); await session.close()
        return "ok", resp

    # committed to the client: flush buffered first chunk(s), then stream the rest. A mid-stream
    # error past this point can only be reported inline (client already receiving output).
    resp = _mk_resp(); await resp.prepare(request)
    await resp.write(bytes(buf))
    try:
        async for chunk in up.content.iter_any():
            await resp.write(chunk)
    except Exception as e:
        try:
            await resp.write(f'data: {{"error":{{"message":"stream interrupted: {e}"}}}}\n\n'.encode())
        except Exception:
            pass
    await resp.write_eof(); await session.close()
    return "ok", resp


async def _forward_remote(request, path, body, streaming):
    if not REMOTE_ENABLED:
        return web.json_response(
            {"error": {"message": "local unavailable and no remote overflow configured"}}, status=503)
    kind, payload = await _relay(request, REMOTE_BASE, path, remap_for_remote(body), REMOTE_KEY, streaming)
    if kind == "ok":
        return payload
    status, text, _ = payload
    return web.json_response({"error": {"message": f"remote overflow failed: {status} {text}"}}, status=502)


async def handle_completions(request):
    global _inflight, _waiting, _inflight_tokens
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

    if LOG_REQUESTS:
        try:
            model_req = json.loads(body).get("model", "?")
        except Exception:
            model_req = "?"
        log.info("REQ ip=%s ua=%r model=%s ptok=%d maxtok=%d stream=%s tiny=%s bg=%s preview=%r",
                 getattr(request, "remote", "?"), request.headers.get("User-Agent", "?")[:45],
                 model_req, ptok, maxtok, streaming, tiny, background, _preview(body))

    # MASTER SWITCH: full-remote mode (maintenance/debug) — everything -> DeepSeek
    if REMOTE_ENABLED and FORCE_REMOTE:
        record_event("remote", "forced", request, units, 0, **ev)
        return await _forward_remote(request, path, body, streaming)

    # size cap: too-big-for-this-box requests OOM local even at budget=1 -> send straight to remote
    if REMOTE_ENABLED and over_local_cap(body):
        log.info("route %s est prompt+max > %d -> remote(size)", path, MAX_LOCAL_TOKENS)
        record_event("remote", "size", request, units, 0, **ev)
        return await _forward_remote(request, path, body, streaming)

    # big-OUTPUT requests (e.g. Hermes max_tokens=65536): long generations that saturate this slow
    # box and hang interactive clients -> straight to remote (DeepSeek serves them far faster).
    if REMOTE_ENABLED and BIG_OUTPUT > 0 and maxtok >= BIG_OUTPUT:
        log.info("route %s maxtok=%d >= %d -> remote(big-out)", path, maxtok, BIG_OUTPUT)
        record_event("remote", "big-out", request, units, 0, **ev)
        return await _forward_remote(request, path, body, streaming)

    # big-PROMPT requests (e.g. a pi session whose context has grown huge): can't prefill within the
    # first-token cap on this slow box and would saturate/OOM local -> straight to remote up front.
    if REMOTE_ENABLED and BIG_PROMPT > 0 and ptok >= BIG_PROMPT:
        log.info("route %s ptok=%d >= %d -> remote(big-prompt)", path, ptok, BIG_PROMPT)
        record_event("remote", "big-prompt", request, units, 0, **ev)
        return await _forward_remote(request, path, body, streaming)

    # local DOWN -> overflow immediately (waiting for a slot won't help a dead engine)
    if not await local_healthy():
        if REMOTE_ENABLED:
            log.info("route %s local unhealthy -> remote(local-down)", path)
            record_event("remote", "local-down", request, units, 0, **ev)
            return await _forward_remote(request, path, body, streaming)

    # TINY fast-lane: negligible-VRAM micro-calls skip the queue-first wait and use headroom slots
    # BEYOND the big-request budget (bounded by TINY_EXTRA_LANES, staying within max-num-seqs), so a
    # trivial call never eats a 15s wait or gets starved during a budget=1 backoff. If even that
    # headroom is full, fast-overflow immediately (a tiny call on DeepSeek is cheap + fast).
    if tiny:
        if _health["ok"] and (_inflight + units) <= (effective_budget() + TINY_EXTRA_LANES):
            _inflight += units
            _inflight_tokens += ptok
            log.info("route %s TINY units=%d inflight=%d/%d(+%d) -> local(tiny)",
                     path, units, _inflight, effective_budget(), TINY_EXTRA_LANES)
            try:
                # tiny fast-lane needs the same thinking guard as the main path: these are
                # exactly the small-max_tokens calls that get starved to an empty response.
                kind, payload = await _relay(request, LOCAL, path,
                                             repetition_guard(thinking_budget_guard(bound_local_output(body))),
                                             None, streaming, concurrency=1)
                if kind == "ok":
                    record_event("local", "tiny", request, units, 0, **ev)
                    return payload
                status, text, oom = payload
                if oom:
                    trigger_backoff(f"local {status}: {text[:120]}")
                log.warning("local(tiny) failed (%s) -> failover to remote", status)
                record_event("remote", "failover", request, units, 0, **ev)
                return await _forward_remote(request, path, body, streaming)
            finally:
                _inflight -= units
                _inflight_tokens -= ptok
        elif REMOTE_ENABLED:
            log.info("route %s TINY inflight=%d/%d(+%d) full -> remote(tiny-fast)",
                     path, _inflight, effective_budget(), TINY_EXTRA_LANES)
            record_event("remote", "tiny-fast", request, units, 0, **ev)
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
        deadline = time.time() + (_bgw if REMOTE_ENABLED else 1e9)
    else:
        lane_limit = effective_budget()
        deadline = time.time() + (LOCAL_WAIT if REMOTE_ENABLED else 1e9)
    admitted = False
    admitted_conc = 1        # local concurrency at admission -> scales the first-token deadline
    waited = 0.0
    queued = False
    try:
        while True:
            if _health["ok"] and (_inflight + units) <= lane_limit \
                    and (TOKEN_BUDGET <= 0 or _inflight_tokens + ptok <= TOKEN_BUDGET or _inflight == 0):
                _inflight += units
                _inflight_tokens += ptok
                admitted_conc = _inflight
                admitted = True
                break
            if time.time() >= deadline:
                break
            if not queued:                       # first time we couldn't get a slot -> we're backlogged
                queued = True
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
        where = f"remote({reason})" if REMOTE_ENABLED else "remote(none)"
        log.info("route %s units=%d inflight=%d/%d tok=%d/%d waited=%.1fs -> %s",
                 path, units, _inflight, effective_budget(), _inflight_tokens, TOKEN_BUDGET, waited, where)
        if queued:
            _stats["overflowed_after_wait"] += 1
        record_event("remote", reason, request, units, waited, **ev)
        return await _forward_remote(request, path, body, streaming)

    log.info("route %s units=%d inflight=%d/%d waited=%.1fs -> local",
             path, units, _inflight, effective_budget(), waited)
    # FLIGHT RECORDER (RCA, 2026-08-13): persist big local-routed request bodies so the
    # next Xid-31 crash leaves a deterministic repro payload. Ring of 40 files, 0600.
    if ptok >= int(os.environ.get("SHIM_FLIGHTREC_MIN_TOK", "15000")):
        try:
            fr = "/home/kevin/.local/share/vllm-qwen27b/flightrec"
            fn = f"{fr}/{int(time.time())}_{ptok}tok.json"
            with open(fn, "wb") as f:
                f.write(body)
            os.chmod(fn, 0o600)
            olds = sorted(os.listdir(fr))
            for o in olds[:-40]:
                os.unlink(os.path.join(fr, o))
        except Exception as e:
            log.warning("flightrec: %s", e)
    try:
        _lb = repetition_guard(thinking_budget_guard(bound_local_output(body)))
        if (background and BG_NO_THINK) or (getattr(request, "remote", None) in NO_THINK_IPS):
            _lb = strip_thinking(_lb)
        kind, payload = await _relay(request, LOCAL, path, _lb, None, streaming, concurrency=admitted_conc)
        if kind == "ok":
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
                    record_event("local", "empty-retry", request, units, waited, **ev)
                    return payload2
            record_event("local", "-", request, units, waited, **ev)
            return payload
        status, text, oom = payload
        if oom:
            trigger_backoff(f"local {status}: {text[:120]}")
        log.warning("local failed (%s) -> failover to remote", status)
        record_event("remote", "failover", request, units, waited, **ev)
        return await _forward_remote(request, path, body, streaming)
    finally:
        _inflight -= units
        _inflight_tokens -= ptok


# ---------------- passthrough (dynamic; no hardcoded models) ----------------
async def _passthrough(request):
    body = await request.read()
    async with aiohttp.ClientSession() as s:
        async with s.request(request.method, f"{LOCAL}{request.rel_url}", data=body or None,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30)) as up:
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
    if REMOTE_ENABLED and REMOTE_BASE:
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
        "backoff": max(0, int(_backoff_until - time.time())),
        "remote_model": REMOTE_MODEL, "remote_enabled": REMOTE_ENABLED,
        "local_wait": LOCAL_WAIT,
        "total": _stats["total"], "local": _stats["local"], "remote": _stats["remote"],
        "local_pct": round(100 * _stats["local"] / total, 1),
        "remote_pct": round(100 * _stats["remote"] / total, 1),
        "avg_wait": round(_stats["waited_total"] / (_stats["waited_n"] or 1), 1),
        "remote_reasons": dict(_remote_reasons),
        "gpu": _gpu_stats(),
        "events": list(_events)[:60],
    })

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
    try:
        fields = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)
    changed = apply_config(fields if isinstance(fields, dict) else {})
    log.info("config updated via dashboard: %s", ",".join(changed) or "(none)")
    return web.json_response({"changed": changed, "config": current_config(masked=True)})

async def gateway_dashboard(request):
    return web.Response(text=DASHBOARD_HTML, content_type="text/html")

DASHBOARD_HTML = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>vLLM Gateway Status</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--bd:#30363d;--fg:#e6edf3;--dim:#8b949e;--grn:#3fb950;--amb:#d29922;--red:#f85149;--blu:#58a6ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
.wrap{max-width:1000px;margin:0 auto;padding:18px}
h1{font-size:16px;margin:0 0 2px;font-weight:600}.sub{color:var(--dim);font-size:12px;margin-bottom:16px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:middle}
.up{background:var(--grn);box-shadow:0 0 6px var(--grn)}.down{background:var(--red);box-shadow:0 0 6px var(--red)}
.grid{display:grid;gap:10px}.g4{grid-template-columns:repeat(4,1fr)}.g2{grid-template-columns:repeat(2,1fr)}
@media(max-width:640px){.g4{grid-template-columns:repeat(2,1fr)}}
.card{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:12px 14px}
.k{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.04em}.v{font-size:22px;font-weight:600;margin-top:2px}
.v small{font-size:12px;color:var(--dim);font-weight:400}
.bar{height:8px;border-radius:4px;background:#21262d;overflow:hidden;display:flex;margin-top:8px}
.bar i{display:block;height:100%}.bl{background:var(--grn)}.br{background:var(--amb)}
table{width:100%;border-collapse:collapse;font-size:12px}th{text-align:left;color:var(--dim);font-weight:500;padding:6px 8px;border-bottom:1px solid var(--bd)}
td{padding:5px 8px;border-bottom:1px solid #21262d;white-space:nowrap}
.tag{padding:1px 7px;border-radius:10px;font-size:11px;font-weight:600}
.tl{background:rgba(63,185,80,.15);color:var(--grn)}.tr{background:rgba(210,153,34,.15);color:var(--amb)}
.rz{color:var(--dim)}.mono{font-variant-numeric:tabular-nums}h2{font-size:12px;color:var(--dim);text-transform:uppercase;letter-spacing:.04em;margin:20px 0 8px}
.pill{display:inline-block;background:#21262d;border-radius:10px;padding:1px 8px;margin:0 4px 4px 0;font-size:12px}
label{display:flex;flex-direction:column;gap:3px}input{background:#0d1117;border:1px solid var(--bd);color:var(--fg);border-radius:6px;padding:6px 8px;font:13px ui-monospace,monospace}
button{background:var(--blu);color:#04101f;border:0;border-radius:6px;padding:7px 16px;font-weight:600;cursor:pointer;font-size:13px}button:hover{opacity:.9}
</style></head><body><div class=wrap>
<h1>vLLM Gateway <span id=live></span> <span id=modebadge style='font-size:12px;padding:2px 10px;border-radius:10px;vertical-align:middle'></span> <button id=modebtn style='font-size:12px;padding:4px 12px;margin-left:6px'></button></h1>
<div class=sub>:8000 capacity-routing gateway → local :8001 · overflow <span id=rm></span> · refresh 1.5s · <span id=err style="color:var(--red)"></span></div>
<div class="grid g4" id=status></div>
<h2>Routing (local vs overflow)</h2><div class=card><div class=bar id=lrbar></div><div id=lrtxt class=sub style=margin-top:8px></div><div id=reasons></div></div>
<h2>GPU</h2><div class="grid g2" id=gpu></div>
<h2>Recent requests</h2><div class=card style=overflow-x:auto><table><thead><tr><th>time</th><th>ep</th><th>client</th><th>route</th><th>reason</th><th>size (in→out)</th><th>waited</th></tr></thead><tbody id=ev></tbody></table></div>
<h2>Local model — switch which checkpoint the engine serves</h2>
<div class=card>
<div class=sub>Scans <span id=mdir class=rz></span>. Any HF checkpoint works (not just Qwen). Switching restarts the engine (~40s warm / ~4-5 min cold); the gateway serves via remote overflow meanwhile. \u26a0\ufe0f no auth on this page.</div>
<table><thead><tr><th></th><th>model</th><th>size</th><th>format</th><th></th></tr></thead><tbody id=lm></tbody></table>
<div style=margin-top:8px><span id=lmmsg class=rz></span></div>
</div>
<h2>Settings — remote overflow provider &amp; routing</h2>
<div class=card>
<div class=sub>Applied live (no restart) and saved to shim.env. \u26a0\ufe0f no auth \u2014 anyone on the LAN can change these.</div>
<div class=k style=margin-bottom:6px>Provider preset <select id=f_preset style="background:#0d1117;border:1px solid var(--bd);color:var(--fg);border-radius:6px;padding:5px 8px">
 <option value="">— pick to autofill base+model —</option>
 <option value="https://api.deepseek.com|deepseek-v4-flash">DeepSeek v4-flash</option>
 <option value="https://api.deepseek.com|deepseek-v4-pro">DeepSeek v4-pro</option>
 <option value="https://api.minimax.io/v1|MiniMax-M3">MiniMax M3</option>
 <option value="https://dashscope-intl.aliyuncs.com/compatible-mode/v1|qwen3-coder-next">Qwen3-Coder-Next (DashScope intl)</option>
 <option value="https://dashscope-intl.aliyuncs.com/compatible-mode/v1|qwen3.7-flash">qwen3.7-flash (DashScope intl)</option>
 <option value="https://openrouter.ai/api/v1|minimax/minimax-m3">OpenRouter \u2192 MiniMax M3</option>
</select> <span class=rz>verify exact model id with your provider</span></div>
<div class="grid g2" style=gap:10px>
 <label class=k>Remote base URL<input id=f_remote_base placeholder=https://api.minimax.io/v1></label>
 <label class=k>Remote model<input id=f_remote_model placeholder=MiniMax-M3></label>
 <label class=k>Remote API key <span id=keystate class=rz></span><input id=f_remote_key type=password placeholder="leave blank to keep current"></label>
 <label class=k>Force full-remote (0/1)<input id=f_force_remote type=number min=0 max=1></label>
</div>
<h2 style=margin-top:16px>Local capacity</h2>
<div class="grid g2" style=gap:10px>
 <label class=k>Local budget (lanes)<input id=f_local_budget type=number min=1 max=8></label>
 <label class=k>Queue wait secs (interactive)<input id=f_local_wait_secs type=number min=0 step=1></label>
 <label class=k>Total in-flight ctx cap (tok)<input id=f_token_budget type=number min=0 step=50000></label>
 <label class=k>OOM backoff secs<input id=f_oom_backoff_secs type=number min=0 step=10></label>
</div>
<h2 style=margin-top:16px>Routing guards \u2014 what goes remote</h2>
<div class="grid g2" style=gap:10px>
 <label class=k>Big OUTPUT \u2265 tok \u2192 remote<input id=f_big_output type=number min=0 step=1000></label>
 <label class=k>Big PROMPT \u2265 tok \u2192 remote (0=off, crash guard)<input id=f_big_prompt type=number min=0 step=1000></label>
 <label class=k>Single-req size cap (tok)<input id=f_max_local_tokens type=number min=0 step=10000></label>
 <label class=k>Local output clamp (tok)<input id=f_local_max_out type=number min=0 step=1024></label>
 <label class=k>Serialize-solo threshold (big=full budget, tok)<input id=f_big_tokens type=number min=0 step=1000></label>
 <label class=k>First-token deadline cap (s)<input id=f_first_token_max type=number min=5 step=5></label>
 <label class=k>Prefill rate est (tok/s)<input id=f_prefill_tps type=number min=100 step=100></label>
</div>
<h2 style=margin-top:16px>Priority lanes &amp; behaviour</h2>
<div class="grid g2" style=gap:10px>
 <label class=k>Tiny fast-lane \u2264 tok<input id=f_tiny_tokens type=number min=0 step=100></label>
 <label class=k>Tiny extra lanes (beyond budget)<input id=f_tiny_extra_lanes type=number min=0 max=4></label>
 <label class=k>Lanes reserved for interactive<input id=f_fg_reserved type=number min=0 max=4></label>
 <label class=k>Background queue wait (s)<input id=f_bg_wait_secs type=number min=0 step=1></label>
 <label class=k>Background markers (|-sep)<input id=f_bg_markers placeholder="scheduled cron job"></label>
 <label class=k>Peak hours UTC (e.g. 1-4,6-10)<input id=f_peak_hours_utc></label>
 <label class=k>No-think for background (0/1)<input id=f_bg_no_think type=number min=0 max=1></label>
 <label class=k>No-think client IPs (comma-sep)<input id=f_no_think_ips placeholder="10.0.1.10,10.0.1.250"></label>
 <label class=k>Per-request logging (0/1)<input id=f_log_requests type=number min=0 max=1></label>
</div>
<div style=margin-top:12px><button id=save>Save settings</button> <span id=savemsg class=rz></span></div>
</div>
</div><script>
const $=s=>document.querySelector(s);let model="?";
async function models(){try{const r=await fetch('/v1/models');const d=await r.json();model=(d.data&&d.data[0]&&d.data[0].id)||"?";}catch(e){}}
function fmtAgo(t){const s=Math.max(0,Date.now()/1000-t);return s<60?s.toFixed(0)+'s':(s/60).toFixed(0)+'m';}
function tile(k,v){return `<div class=card><div class=k>${k}</div><div class=v>${v}</div></div>`;}
async function tick(){
 let s;try{const r=await fetch('/gateway/stats');s=await r.json();$('#err').textContent='';}catch(e){$('#err').textContent='reconnecting… (showing last data)';$('#live').innerHTML='<span class="dot down"></span>';return;}
 $('#live').innerHTML=`<span class="dot up"></span>`;$('#rm').textContent=s.remote_model;
 const uh=Math.floor(s.uptime/3600),um=Math.floor(s.uptime%3600/60);
 const bk=s.backoff>0?`<div class=k style=color:var(--amb)>OOM backoff ${s.backoff}s</div>`:'';
 $('#status').innerHTML=[
  `<div class=card><div class=k>Local engine</div><div class=v><span class="dot ${s.local_healthy?'up':'down'}"></span>${s.local_healthy?'up':'DOWN'}</div><div class=k style=margin-top:4px title="${model}">${model.slice(0,22)}</div></div>`,
  `<div class=card><div class=k>Lanes / backlog</div><div class=v class=mono>${s.inflight}<small>/${s.budget}</small>${s.waiting?` <span style=color:var(--amb)>+${s.waiting} queued</span>`:''}</div><div class=bar><i class=bl style=width:${100*s.inflight/Math.max(1,s.budget)}%></i>${s.waiting?`<i class=br style=width:${Math.min(100,100*s.waiting/Math.max(1,s.budget))}%></i>`:''}</div><div class=k style=margin-top:4px>ctx in flight ${((s.inflight_tokens||0)/1000).toFixed(0)}K/${((s.token_budget||0)/1000).toFixed(0)}K · peak backlog ${s.peak_waiting} · overflowed ${s.overflowed_after_wait}</div>${bk}</div>`,
  tile('Total requests',`<span class=mono>${s.total}</span>`),
  `<div class=card><div class=k>Served local</div><div class=v class=mono style=color:var(--grn)>${s.local_pct}<small>%</small></div><div class=k style=margin-top:4px>overflow ${s.remote_pct}% · avg wait ${s.avg_wait}s</div></div>`,
 ].join('');
 const lp=s.local_pct,rp=s.remote_pct;
 $('#lrbar').innerHTML=`<i class=bl style=width:${lp}%></i><i class=br style=width:${rp}%></i>`;
 $('#lrtxt').innerHTML=`<span style=color:var(--grn)>■</span> local ${s.local} (${lp}%) &nbsp; <span style=color:var(--amb)>■</span> overflow ${s.remote} (${rp}%) &nbsp; peak lanes ${s.peak_inflight} &nbsp; uptime ${uh}h${um}m`;
 $('#reasons').innerHTML=Object.keys(s.remote_reasons||{}).length?('overflow reasons: '+Object.entries(s.remote_reasons).map(([k,v])=>`<span class=pill>${k}: ${v}</span>`).join('')):'';
 $('#gpu').innerHTML=(s.gpu||[]).map((g,i)=>`<div class=card><div class=k>GPU ${i}</div><div class=v class=mono>${g.util}<small>% util</small></div><div class=bar><i class=bl style="width:${100*g.used/(g.used+g.free)}%;background:var(--blu)"></i></div><div class=k style=margin-top:4px>${(g.used/1024).toFixed(1)}G used · ${(g.free/1024).toFixed(1)}G free</div></div>`).join('')||'<div class=card><div class=k>no GPU data</div></div>';
 $('#ev').innerHTML=(s.events||[]).map(e=>`<tr><td class="rz mono">${fmtAgo(e.t)} ago</td><td>${e.ep}</td><td>${e.client}</td><td><span class="tag ${e.d=='local'?'tl':'tr'}">${e.d}</span></td><td class=rz>${e.r}</td><td class=mono>${(e.ptok||0)}→${(e.maxtok||0)}${e.stream?' ⚡':''}</td><td class=mono>${e.waited?e.waited+'s':''}</td></tr>`).join('');
}
let FR=0;
function renderMode(){const b=document.getElementById('modebadge'),t=document.getElementById('modebtn');
 if(FR){b.textContent='FULL REMOTE';b.style.background='rgba(210,153,34,.25)';b.style.color='var(--amb)';t.textContent='Switch to LOCAL';}
 else{b.textContent='LOCAL-FIRST';b.style.background='rgba(63,185,80,.2)';b.style.color='var(--grn)';t.textContent='Switch to FULL REMOTE';}}
async function toggleMode(){FR=FR?0:1;await fetch('/gateway/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({force_remote:FR})});renderMode();loadCfg();}
document.getElementById('modebtn').addEventListener('click',toggleMode);
document.getElementById('f_preset').addEventListener('change',e=>{const v=e.target.value;if(!v)return;const [b,m]=v.split('|');$('#f_remote_base').value=b;$('#f_remote_model').value=m;});
async function loadLocalModels(){
 try{const d=await(await fetch('/gateway/models/local')).json();
  $('#mdir').textContent=d.models_dir||'';
  $('#lm').innerHTML=(d.models||[]).map(m=>{
    const badge=m.live?'<span class="tag tl">LIVE</span>':'';
    const btn=m.servable&&!m.live?`<button class=lmsw data-p="${m.path}" style="font-size:11px;padding:3px 10px">Switch</button>`:
              (m.servable?'<span class=rz>current</span>':'<span class=rz style=color:var(--amb)>not servable</span>');
    return `<tr><td>${badge}</td><td class=mono>${m.name}</td><td class=mono>${m.gb} GB</td><td class=rz>${m.desc}</td><td>${btn}</td></tr>`;}).join('')
    ||'<tr><td colspan=5 class=rz>no checkpoints found</td></tr>';
  document.querySelectorAll('.lmsw').forEach(b=>b.addEventListener('click',()=>switchLocal(b.dataset.p)));
 }catch(e){}}
async function switchLocal(p){
 if(!confirm('Switch the local engine to:\n\n'+p+'\n\nThe engine restarts. Traffic falls back to remote overflow until it is healthy.'))return;
 $('#lmmsg').textContent='switching\u2026 engine restarting';
 try{const r=await fetch('/gateway/models/local',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:p})});
  const d=await r.json();
  $('#lmmsg').textContent=d.error?('\u2717 '+d.error):('\u2713 switching to '+d.switching_to+' ('+d.quant+') \u2014 '+d.note);
  setTimeout(loadLocalModels,15000);}catch(e){$('#lmmsg').textContent='\u2717 '+e;}}
setInterval(loadLocalModels,20000);
async function loadCfg(){try{const c=await(await fetch('/gateway/config')).json();
 FR=c.force_remote?1:0;renderMode();
 for(const [k,v] of Object.entries(c)){const el=document.getElementById('f_'+k);if(el&&el.type!=='password')el.value=v;}
 $('#keystate').textContent=c.remote_key_display?('\u00b7 '+c.remote_key_display):'\u00b7 not set';}catch(e){}}
async function saveCfg(){const b={};
 document.querySelectorAll('input[id^=f_]').forEach(el=>{const k=el.id.slice(2);
  if(el.type==='password'){if(el.value.trim())b[k]=el.value.trim();}
  else if(el.value!=='')b[k]=el.value;});
 $('#savemsg').textContent='saving\u2026';
 try{const r=await fetch('/gateway/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
  const d=await r.json();$('#savemsg').textContent='\u2713 saved: '+(d.changed||[]).join(', ');$('#f_remote_key').value='';loadCfg();}
 catch(e){$('#savemsg').textContent='\u2717 '+e;}}
document.getElementById('save').addEventListener('click',saveCfg);
models();loadCfg();loadLocalModels();tick();setInterval(tick,1500);setInterval(models,15000);
</script></body></html>"""

async def _on_startup(app):
    _load_stats()
    app["saver"] = asyncio.create_task(_stats_saver())

async def _on_cleanup(app):
    t = app.get("saver")
    if t:
        t.cancel()
    _save_stats()

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
    app.router.add_get("/gateway/dashboard", gateway_dashboard)
    app.router.add_get("/gateway", lambda r: web.HTTPFound("/gateway/dashboard"))
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
