# gateway-part: telemetry sampling: GPU (nvidia-smi) + host (/proc) stats, the vLLM /metrics Prometheus parser, DB2 per-request latency windows, Lane TL persistent hw/engine history (/gateway/telemetry/history), and _telemetry_sampler
# gateway-part: executed inside keepalive-shim.py's own namespace by _include_gateway_part() -- not an
# gateway-part: importable module. Names here are the shim's globals. See gateway_parts.py.

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


