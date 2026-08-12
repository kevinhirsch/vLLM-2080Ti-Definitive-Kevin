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
import aiohttp
from aiohttp import web

PORT         = int(os.environ.get("SHIM_PORT", "8000"))
LOCAL        = os.environ.get("SHIM_UPSTREAM", "http://127.0.0.1:8001").rstrip("/")
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
# Adaptive first-token deadline: base + est_prompt_tokens/prefill_rate. Big-context prefills
# legitimately take a while, so this scales with prompt size; a WEDGED backend (accepts the
# connection but never emits a token) still fails over in ~base seconds instead of hanging.
FIRST_TOKEN_BASE = float(os.environ.get("SHIM_FIRST_TOKEN_BASE", "15"))
PREFILL_TPS      = float(os.environ.get("SHIM_PREFILL_TPS", "500"))
# A single request whose est. (prompt + max_tokens) exceeds this will OOM local even at
# budget=1 (context + generation peaks past this box's tiny free VRAM), so route it straight
# to remote. Prevents single-big-request OOM crashes. (2026-08-12: observed a solo OOM here.)
MAX_LOCAL_TOKENS = int(os.environ.get("SHIM_MAX_LOCAL_TOKENS", "80000"))
DEFAULT_MAX_OUT  = int(os.environ.get("SHIM_DEFAULT_MAX_OUT", "8192"))
# vLLM-only params that a remote OpenAI endpoint would reject — stripped on overflow.
REMOTE_STRIP = ("chat_template_kwargs", "mamba_cache_mode", "guided_decoding_backend")

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s [gateway] %(levelname)s %(message)s")
log = logging.getLogger("gateway-shim")

_inflight = 0            # local capacity units currently in flight
_backoff_until = 0.0
_health = {"ok": False, "at": 0.0}
REMOTE_ENABLED = bool(REMOTE_BASE and REMOTE_KEY)

# ---------------- live metrics (for the /gateway/dashboard status page) ----------------
_stats = {"started": time.time(), "total": 0, "local": 0, "remote": 0,
          "waited_total": 0.0, "waited_n": 0, "peak_inflight": 0}
_remote_reasons = collections.Counter()
_events = collections.deque(maxlen=200)   # most-recent-first ring buffer of routing decisions
_gpu_cache = {"at": 0.0, "data": []}

def _client_label(request):
    # harness self-id via X-Client/X-Title header, else source IP
    return request.headers.get("X-Client") or request.headers.get("X-Title") \
        or getattr(request, "remote", None) or "?"

def record_event(decision, reason, request, units, waited):
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
                        "ep": request.path.rsplit("/", 1)[-1]})

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
    return int(chars / CHARS_PER_TOK)


def estimate_units(body):
    # A "big" request occupies the WHOLE current local budget (runs alone, nothing concurrent),
    # rather than a fixed 2. This means at budget=1 a single big request still runs LOCAL (safe:
    # 1 big alone never OOMs) instead of needlessly overflowing to DeepSeek, while at budget>=2 it
    # still blocks anything running alongside it.
    return effective_budget() if _est_tokens(body) >= BIG_TOKENS else 1


def first_token_timeout(body):
    return FIRST_TOKEN_BASE + _est_tokens(body) / PREFILL_TPS


def over_local_cap(body):
    try:
        mt = int(json.loads(body).get("max_tokens") or DEFAULT_MAX_OUT)
    except Exception:
        mt = DEFAULT_MAX_OUT
    return (_est_tokens(body) + mt) > MAX_LOCAL_TOKENS


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


async def _relay(request, base, path, body, key, streaming):
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
                                        timeout=first_token_timeout(body))
        else:
            up = await _open(session, base, path, body, key, streaming)
    except asyncio.TimeoutError:
        await session.close()
        return "fail", (0, "no response headers within first-token deadline (wedged backend)", True)
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
    # allowing a legit big-context prefill the time it actually needs.
    deadline = first_token_timeout(body)
    try:
        phase = await asyncio.wait_for(_read_until_commit(), timeout=deadline)
    except asyncio.TimeoutError:
        await session.close()
        return "fail", (up.status, f"no first token within {deadline:.0f}s (wedged backend)", True)
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
    global _inflight
    path = request.path
    body = await request.read()
    units = estimate_units(body)
    streaming = wants_stream(body)

    # size cap: too-big-for-this-box requests OOM local even at budget=1 -> send straight to remote
    if REMOTE_ENABLED and over_local_cap(body):
        log.info("route %s est prompt+max > %d -> remote(size)", path, MAX_LOCAL_TOKENS)
        record_event("remote", "size", request, units, 0)
        return await _forward_remote(request, path, body, streaming)

    # local DOWN -> overflow immediately (waiting for a slot won't help a dead engine)
    if not await local_healthy():
        if REMOTE_ENABLED:
            log.info("route %s local unhealthy -> remote(local-down)", path)
            record_event("remote", "local-down", request, units, 0)
            return await _forward_remote(request, path, body, streaming)

    # local UP: claim a slot, WAITING up to LOCAL_WAIT for capacity instead of instant-overflow.
    # The check-and-increment is done with no await in between, so it's race-free under asyncio.
    deadline = time.time() + (LOCAL_WAIT if REMOTE_ENABLED else 1e9)
    admitted = False
    waited = 0.0
    while True:
        if _health["ok"] and (_inflight + units) <= effective_budget():
            _inflight += units
            admitted = True
            break
        if time.time() >= deadline:
            break
        await asyncio.sleep(SLOT_POLL)
        waited += SLOT_POLL
        await local_healthy()  # refresh cached health while waiting

    if not admitted:
        where = "remote(cap)" if REMOTE_ENABLED else "remote(none)"
        log.info("route %s units=%d inflight=%d budget=%d waited=%.1fs -> %s",
                 path, units, _inflight, effective_budget(), waited, where)
        record_event("remote", "cap", request, units, waited)
        return await _forward_remote(request, path, body, streaming)

    log.info("route %s units=%d inflight=%d/%d waited=%.1fs -> local",
             path, units, _inflight, effective_budget(), waited)
    try:
        kind, payload = await _relay(request, LOCAL, path, body, None, streaming)
        if kind == "ok":
            record_event("local", "-", request, units, waited)
            return payload
        status, text, oom = payload
        if oom:
            trigger_backoff(f"local {status}: {text[:120]}")
        log.warning("local failed (%s) -> failover to remote", status)
        record_event("remote", "failover", request, units, waited)
        return await _forward_remote(request, path, body, streaming)
    finally:
        _inflight -= units


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
</style></head><body><div class=wrap>
<h1>vLLM Gateway <span id=live></span></h1>
<div class=sub>:8000 capacity-routing gateway → local :8001 · overflow <span id=rm></span> · refresh 1.5s · <span id=err style="color:var(--red)"></span></div>
<div class="grid g4" id=status></div>
<h2>Routing (local vs overflow)</h2><div class=card><div class=bar id=lrbar></div><div id=lrtxt class=sub style=margin-top:8px></div><div id=reasons></div></div>
<h2>GPU</h2><div class="grid g2" id=gpu></div>
<h2>Recent requests</h2><div class=card style=overflow-x:auto><table><thead><tr><th>time</th><th>ep</th><th>client</th><th>route</th><th>reason</th><th>waited</th></tr></thead><tbody id=ev></tbody></table></div>
</div><script>
const $=s=>document.querySelector(s);let model="?";
async function models(){try{const r=await fetch('/v1/models');const d=await r.json();model=(d.data&&d.data[0]&&d.data[0].id)||"?";}catch(e){}}
function fmtAgo(t){const s=Math.max(0,Date.now()/1000-t);return s<60?s.toFixed(0)+'s':(s/60).toFixed(0)+'m';}
function tile(k,v){return `<div class=card><div class=k>${k}</div><div class=v>${v}</div></div>`;}
async function tick(){
 let s;try{const r=await fetch('/gateway/stats');s=await r.json();$('#err').textContent='';}catch(e){$('#err').textContent='gateway unreachable';return;}
 $('#live').innerHTML=`<span class="dot up"></span>`;$('#rm').textContent=s.remote_model;
 const uh=Math.floor(s.uptime/3600),um=Math.floor(s.uptime%3600/60);
 const bk=s.backoff>0?`<div class=k style=color:var(--amb)>OOM backoff ${s.backoff}s</div>`:'';
 $('#status').innerHTML=[
  `<div class=card><div class=k>Local engine</div><div class=v><span class="dot ${s.local_healthy?'up':'down'}"></span>${s.local_healthy?'up':'DOWN'}</div><div class=k style=margin-top:4px title="${model}">${model.slice(0,22)}</div></div>`,
  `<div class=card><div class=k>Lanes in use</div><div class=v class=mono>${s.inflight}<small>/${s.budget}</small></div><div class=bar><i class=bl style=width:${100*s.inflight/Math.max(1,s.budget)}%></i></div>${bk}</div>`,
  tile('Total requests',`<span class=mono>${s.total}</span>`),
  `<div class=card><div class=k>Served local</div><div class=v class=mono style=color:var(--grn)>${s.local_pct}<small>%</small></div><div class=k style=margin-top:4px>overflow ${s.remote_pct}% · avg wait ${s.avg_wait}s</div></div>`,
 ].join('');
 const lp=s.local_pct,rp=s.remote_pct;
 $('#lrbar').innerHTML=`<i class=bl style=width:${lp}%></i><i class=br style=width:${rp}%></i>`;
 $('#lrtxt').innerHTML=`<span style=color:var(--grn)>■</span> local ${s.local} (${lp}%) &nbsp; <span style=color:var(--amb)>■</span> overflow ${s.remote} (${rp}%) &nbsp; peak lanes ${s.peak_inflight} &nbsp; uptime ${uh}h${um}m`;
 $('#reasons').innerHTML=Object.keys(s.remote_reasons||{}).length?('overflow reasons: '+Object.entries(s.remote_reasons).map(([k,v])=>`<span class=pill>${k}: ${v}</span>`).join('')):'';
 $('#gpu').innerHTML=(s.gpu||[]).map((g,i)=>`<div class=card><div class=k>GPU ${i}</div><div class=v class=mono>${g.util}<small>% util</small></div><div class=bar><i class=bl style="width:${100*g.used/(g.used+g.free)}%;background:var(--blu)"></i></div><div class=k style=margin-top:4px>${(g.used/1024).toFixed(1)}G used · ${(g.free/1024).toFixed(1)}G free</div></div>`).join('')||'<div class=card><div class=k>no GPU data</div></div>';
 $('#ev').innerHTML=(s.events||[]).map(e=>`<tr><td class="rz mono">${fmtAgo(e.t)} ago</td><td>${e.ep}</td><td>${e.client}</td><td><span class="tag ${e.d=='local'?'tl':'tr'}">${e.d}</span></td><td class=rz>${e.r}</td><td class=mono>${e.waited?e.waited+'s':''}</td></tr>`).join('');
}
models();tick();setInterval(tick,1500);setInterval(models,15000);
</script></body></html>"""

def make_app():
    app = web.Application(client_max_size=1024**3)
    app.router.add_get("/health", h_health)
    app.router.add_get("/v1/models", h_models)
    app.router.add_post("/v1/chat/completions", handle_completions)
    app.router.add_post("/v1/completions", handle_completions)
    app.router.add_get("/gateway/stats", gateway_stats)
    app.router.add_get("/gateway/dashboard", gateway_dashboard)
    app.router.add_get("/gateway", lambda r: web.HTTPFound("/gateway/dashboard"))
    app.router.add_route("*", "/{tail:.*}", h_catchall)
    return app


if __name__ == "__main__":
    log.info("gateway-shim on :%d | local=%s | remote=%s model=%s | budget=%d big=%dtok backoff=%ds",
             PORT, LOCAL, REMOTE_BASE or "(none)", REMOTE_MODEL, BUDGET, BIG_TOKENS, OOM_BACKOFF)
    web.run_app(make_app(), host="0.0.0.0", port=PORT)
