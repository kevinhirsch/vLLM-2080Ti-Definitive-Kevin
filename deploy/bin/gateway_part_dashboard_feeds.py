# gateway-part: the dashboard's read-only feeds: research & lanes aggregator (/gateway/lanes, /gateway/research/*), windows drill-down (/gateway/windows*), and the dashboard page loader (dashboard_html, /gateway/dashboard)
# gateway-part: executed inside keepalive-shim.py's own namespace by _include_gateway_part() -- not an
# gateway-part: importable module. Names here are the shim's globals. See gateway_parts.py.
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

