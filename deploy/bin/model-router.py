#!/usr/bin/env python3
"""
model-router — single front door on :8000 for the 2x2080Ti box.

Hermes/AZ send an OpenAI request naming a model; the router makes sure that
model is the one loaded on the internal upstream (:8001), swapping if needed
(44GB VRAM = one model at a time), then proxies the request. A swap reloads the
backend (~15-20s with GLM on NVMe); for streaming requests the router emits SSE
keep-alive comments during the load so clients don't time out.

Runs as the normal user — it launches BOTH models as child processes (process
groups), no sudo/systemctl at runtime:
  qwen    -> Qwen3.6-27B  (vLLM wrapper serve-tqk8v4-fg.sh, env-sourced, :8001)
  glm     -> GLM-4.5-Air  (llama-server, IQ2_M on NVMe, :8001)
  glm-q4  -> GLM-4.5-Air  (llama-server, Q4_K_M, quality)
"""
import asyncio, os, signal, subprocess, sys, time, json, logging
import aiohttp
from aiohttp import web

PORT = int(os.environ.get("ROUTER_PORT", "8000"))
UPSTREAM = os.environ.get("ROUTER_UPSTREAM", "http://127.0.0.1:8001")
D = "/home/kevin/.local/share/vllm-qwen27b"
QWEN_ENV = f"{D}/vllm-qwen27b.env"
QWEN_WRAPPER = f"{D}/serve-qwen-8001.sh"     # the vLLM Qwen serve, binds :8001
QWEN_PIDFILE, QWEN_LOG = f"{D}/qwen-router.pid", f"{D}/qwen-router.log"
GLM_PIDFILE, GLM_LOG = f"{D}/glm-router.pid", f"{D}/glm-router.log"
LLAMA_SERVER = "/home/kevin/llama.cpp/build/bin/llama-server"
GLM_IQ2 = "/home/kevin/Desktop/models/GLM-4.5-Air-UD-IQ2_M.gguf"   # NVMe (fast load)
GLM_Q4  = "/run/media/kevin/Master/models-archive/GLM-4.5-Air-GGUF/Q4_K_M/GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf"  # WD cold archive (Samsung /mnt/storage removed 2026-08-07; glm-q4 deprecated/slow)
LOADW = 260   # max seconds to wait for a backend to become healthy
LOG_ROTATE_MAX_BYTES = 20 * 1024 * 1024   # rotate a child log once it passes this size

ADVERTISED = ["qwen3.6:27b", "glm-4.5-air", "glm-4.5-air-q4"]
swap_lock = asyncio.Lock()

# The router process itself is exec'd directly as the systemd unit's main process
# (see serve-tqk8v4-fg.sh), so anything written to stdout lands in journalctl for
# free. Nothing logged this way before 2026-08-10 -- ensure_backend's restart
# decisions were completely invisible (see WEDGE-ROOTCAUSE-2026-08-10.md).
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                     format="%(asctime)s [router] %(levelname)s %(message)s")
log = logging.getLogger("model-router")

def backend_for(model: str) -> str:
    # GLM disabled 2026-08-10 (Kevin-approved): GLM model-swaps wedged the router
    # (health returns OK while completions hang) and took down every consumer
    # (Applicant scoring/drafting, Hermes). A swap kills qwen, holds swap_lock while
    # loading GLM from a cold archive, and if that load stalls the whole front door
    # blocks. Qwen3.6-27B is the daily driver; serve it for EVERY request so there
    # are no swaps and no lock wedge. To restore GLM: revert model-router.py.bak-20260810.
    return "qwen"

def sh(*cmd): return subprocess.run(list(cmd), capture_output=True, text=True)

def _live_pid(pidfile):
    try:
        pid = int(open(pidfile).read().strip()); os.kill(pid, 0); return pid
    except Exception:
        return None

def stop_proc(pidfile):
    pid = _live_pid(pidfile)
    if pid:
        log.warning("stop_proc: killing pid=%s (pidfile=%s)", pid, pidfile)
        try: os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:
            try: os.kill(pid, signal.SIGTERM)
            except Exception: pass
        time.sleep(3)
        try: os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception: pass
    try: os.remove(pidfile)
    except Exception: pass

def _open_log_append(path):
    """Open a child-process log in APPEND mode, rotating out a stale huge file first.
    Never truncate: qwen-router.log/glm-router.log are the only place the real
    engine's own stdout/stderr (CUDA/NCCL/cudagraph errors) ever lands, and opening
    in "wb" mode on every backend (re)start used to wipe them on every recovery --
    destroying exactly the evidence needed to root-cause a wedge. See
    WEDGE-ROOTCAUSE-2026-08-10.md.
    """
    try:
        if os.path.exists(path) and os.path.getsize(path) > LOG_ROTATE_MAX_BYTES:
            rotated = f"{path}.1"
            os.replace(path, rotated)
            log.info("_open_log_append: rotated %s -> %s (exceeded %d bytes)", path, rotated, LOG_ROTATE_MAX_BYTES)
    except Exception as e:
        log.warning("_open_log_append: rotation check failed for %s: %r", path, e)
    return open(path, "ab")

def start_qwen():
    f = _open_log_append(QWEN_LOG)
    p = subprocess.Popen(["bash", "-lc", f'set -a; source "{QWEN_ENV}"; set +a; exec "{QWEN_WRAPPER}"'],
                         stdout=f, stderr=f, start_new_session=True)
    open(QWEN_PIDFILE, "w").write(str(p.pid))
    log.warning("start_qwen: launched pid=%s (log=%s, append mode)", p.pid, QWEN_LOG)

def start_glm(gguf, ngl):
    f = _open_log_append(GLM_LOG)
    p = subprocess.Popen(
        [LLAMA_SERVER, "-m", gguf, "--host", "127.0.0.1", "--port", "8001",
         "-ngl", str(ngl), "-c", "65536", "--parallel", "1", "-t", "8", "--no-warmup", "--alias", "glm-4.5-air"],
        stdout=f, stderr=f, start_new_session=True)
    open(GLM_PIDFILE, "w").write(str(p.pid))
    log.warning("start_glm: launched pid=%s gguf=%s (log=%s, append mode)", p.pid, gguf, GLM_LOG)

async def upstream_ready(timeout=2):
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{UPSTREAM}/health", timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                return r.status == 200
    except Exception:
        return False

async def upstream_backend(timeout=2):
    """Which family is live on :8001 (or None)."""
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{UPSTREAM}/v1/models", timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                ids = [m["id"].lower() for m in (await r.json()).get("data", [])]
                return "glm" if any("glm" in i for i in ids) else "qwen"
    except Exception:
        return None

async def free_vram(timeout=50):
    t = time.time()
    while time.time() - t < timeout:
        out = sh("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits").stdout.split("\n")[0].strip()
        try:
            if int(out) < 1500: return
        except Exception: pass
        await asyncio.sleep(2)

async def wait_ready(timeout=LOADW):
    t = time.time()
    while time.time() - t < timeout:
        if await upstream_ready(): return True
        await asyncio.sleep(2)
    return False

# ---------- backend health cache / restart guardrails ----------
# WEDGE-ROOTCAUSE-2026-08-10.md: the previous ensure_backend() re-verified backend
# health on EVERY single request with a 2s-timeout probe, gated behind one global
# lock, and unconditionally killed + cold-reloaded the whole live engine on any one
# late/failed probe -- with zero check for in-flight work. Under this box's normal
# heavy load (max-num-seqs=2, long-context) that is a plainly-reachable false
# positive, and it was reproduced live during the investigation. Fixed by:
#   1. Caching a confirmed-healthy verdict for HEALTH_CACHE_TTL seconds so most
#      requests never probe the backend at all.
#   2. Requiring several consecutive HARD probe failures (connection errors / bad
#      status), spread over a real time window, before ever treating a LIVE process
#      as dead -- one slow/timed-out probe under load is normal, not evidence.
#   3. Never killing a backend while requests are in flight.
#   4. If nothing is running at all yet (cold boot, or the child process actually
#      exited), start it immediately -- there is no "busy healthy engine" to
#      protect in that case, and the router must still be able to boot the backend.
PROBE_TIMEOUT = 10          # was 2s; tolerate legitimate load (long prefill / max-num-seqs=2)
HEALTH_CACHE_TTL = 30       # seconds a confirmed-healthy verdict is trusted with zero probing
HARD_FAIL_THRESHOLD = 5     # consecutive hard probe failures required before considering a restart
HARD_FAIL_MIN_WINDOW = 60   # ...and they must span at least this many seconds (not a burst)

_health_cache = {"ok": False, "family": None, "checked_at": 0.0}
_fail_streak = 0
_fail_streak_started = None
_inflight = 0   # requests currently proxied to the real backend (any family)

def _pidfile_for(fam):
    return QWEN_PIDFILE if fam == "qwen" else GLM_PIDFILE

async def probe_backend(timeout=PROBE_TIMEOUT):
    """One real probe of :8001. Returns (ok, family, error_str_or_None)."""
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{UPSTREAM}/health", timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                if r.status != 200:
                    return False, None, f"/health -> HTTP {r.status}"
    except Exception as e:
        return False, None, f"/health probe error: {e!r}"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{UPSTREAM}/v1/models", timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                ids = [m["id"].lower() for m in (await r.json()).get("data", [])]
                fam = "glm" if any("glm" in i for i in ids) else "qwen"
                return True, fam, None
    except Exception as e:
        return False, None, f"/v1/models probe error: {e!r}"

async def ensure_backend(target: str) -> bool:
    global _fail_streak, _fail_streak_started
    fam = "glm" if target.startswith("glm") else "qwen"
    now = time.time()

    # Fast path: a recently-confirmed-healthy backend of the right family needs no
    # network probe at all. This alone is what stops per-request re-verification.
    age = now - _health_cache["checked_at"]
    if _health_cache["ok"] and _health_cache["family"] == fam and age < HEALTH_CACHE_TTL:
        log.info("ensure_backend(%s): cache hit (age=%.1fs < ttl=%ds) -- no probe", target, age, HEALTH_CACHE_TTL)
        return True

    async with swap_lock:
        # Another request may have refreshed the cache while we waited on the lock.
        now = time.time()
        age = now - _health_cache["checked_at"]
        if _health_cache["ok"] and _health_cache["family"] == fam and age < HEALTH_CACHE_TTL:
            log.info("ensure_backend(%s): cache hit after lock (age=%.1fs) -- no probe", target, age)
            return True

        ok, probed_fam, err = await probe_backend()
        now = time.time()

        if ok and probed_fam == fam:
            log.info("ensure_backend(%s): probe OK (family=%s) -- caching healthy for %ds", target, probed_fam, HEALTH_CACHE_TTL)
            _health_cache.update(ok=True, family=probed_fam, checked_at=now)
            _fail_streak, _fail_streak_started = 0, None
            return True

        do_restart, reason = False, None

        if not err:
            # Probe succeeded but it's the WRONG family -- something is definitely
            # alive and just answered, it's simply not what we want (legit swap
            # case; unreachable today since backend_for() always returns "qwen").
            if _inflight > 0:
                log.warning("ensure_backend(%s): wrong family loaded (%s) but %d request(s) "
                            "in flight -- refusing to kill, re-evaluating next request",
                            target, probed_fam, _inflight)
                return True
            do_restart, reason = True, f"wrong family loaded (got {probed_fam}, want {fam})"
        else:
            proc_alive = _live_pid(_pidfile_for(fam)) is not None
            if not proc_alive:
                # Nothing is running for this family at all (cold boot, or the
                # process exited on its own). No "busy healthy engine" to protect --
                # start it now, same as the router has always needed to.
                log.warning("ensure_backend(%s): probe failed (%s) and no live %s process -- starting fresh",
                            target, err, fam)
                do_restart, reason = True, f"no live {fam} process (probe: {err})"
            else:
                # A live process exists but failed to answer. This is exactly the
                # dangerous "busy, not dead" case from WEDGE-ROOTCAUSE-2026-08-10.md.
                # Require sustained evidence, over a real window, with zero in-flight
                # work, before ever touching it.
                _fail_streak += 1
                if _fail_streak_started is None:
                    _fail_streak_started = now
                streak_age = now - _fail_streak_started
                log.warning("ensure_backend(%s): probe FAILED (%s) on a LIVE process -- "
                            "fail_streak=%d over %.1fs, inflight=%d",
                            target, err, _fail_streak, streak_age, _inflight)
                sustained = _fail_streak >= HARD_FAIL_THRESHOLD and streak_age >= HARD_FAIL_MIN_WINDOW
                if not sustained:
                    log.info("ensure_backend(%s): NOT restarting -- need >=%d fails spanning >=%ds "
                             "(have %d over %.1fs); treating as busy, not dead",
                             target, HARD_FAIL_THRESHOLD, HARD_FAIL_MIN_WINDOW, _fail_streak, streak_age)
                    return True
                if _inflight > 0:
                    log.warning("ensure_backend(%s): sustained-failure threshold met but %d request(s) "
                                "in flight -- refusing to kill, re-evaluating next request",
                                target, _inflight)
                    return True
                do_restart, reason = True, (f"{_fail_streak} consecutive hard failures over "
                                             f"{streak_age:.1f}s, zero in-flight, last error: {err}")

        log.warning("ensure_backend(%s): RESTARTING backend -- %s", target, reason)
        _health_cache.update(ok=False, family=None, checked_at=0.0)
        _fail_streak, _fail_streak_started = 0, None
        stop_proc(QWEN_PIDFILE); stop_proc(GLM_PIDFILE); await free_vram()
        if target == "qwen":     start_qwen()
        elif target == "glm":    start_glm(GLM_IQ2, 34)   # -ngl reduced from 40 to leave VRAM for 64K KV
        elif target == "glm-q4": start_glm(GLM_Q4, 18)
        ready = await wait_ready()
        if ready:
            _health_cache.update(ok=True, family=fam, checked_at=time.time())
        log.warning("ensure_backend(%s): restart complete, ready=%s", target, ready)
        return ready

# ---------- HTTP ----------
async def h_health(request): return web.Response(text="OK")

async def h_models(request):
    return web.json_response({"object": "list",
        "data": [{"id": i, "object": "model", "owned_by": "local"} for i in ADVERTISED]})

async def h_proxy(request):
    global _inflight
    body = await request.read()
    model, streaming = None, False
    if body:
        try:
            j = json.loads(body); model = j.get("model"); streaming = bool(j.get("stream"))
        except Exception: pass
    target = backend_for(model)

    if streaming:
        resp = web.StreamResponse(status=200, headers={
            "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive"})
        await resp.prepare(request)
        task = asyncio.create_task(ensure_backend(target))
        while not task.done():
            try: await resp.write(b": model-router loading backend\n\n")
            except Exception: task.cancel(); return resp
            await asyncio.sleep(4)
        if not await task:
            await resp.write(b'data: {"error":{"message":"backend failed to load"}}\n\n')
            await resp.write_eof(); return resp
        # Counted as in-flight from here (real request reaching the backend) until
        # the upstream call finishes -- this is what lets ensure_backend refuse to
        # kill a backend that's actively serving generation traffic.
        _inflight += 1
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(f"{UPSTREAM}{request.path}", data=body,
                        headers={"Content-Type": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=None, sock_read=600)) as up:
                    async for chunk in up.content.iter_any():
                        await resp.write(chunk)
        except Exception as e:
            try: await resp.write(f'data: {{"error":{{"message":"upstream: {e}"}}}}\n\n'.encode())
            except Exception: pass
        finally:
            _inflight -= 1
        await resp.write_eof(); return resp

    if not await ensure_backend(target):
        return web.json_response({"error": {"message": "backend failed to load"}}, status=503)
    _inflight += 1
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(f"{UPSTREAM}{request.path}", data=body,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=600)) as up:
                data = await up.read()
                return web.Response(body=data, status=up.status,
                                    content_type=up.headers.get("Content-Type", "application/json").split(";")[0])
    finally:
        _inflight -= 1

async def h_catchall(request):
    body = await request.read()
    try:
        async with aiohttp.ClientSession() as s:
            async with s.request(request.method, f"{UPSTREAM}{request.rel_url}", data=body or None,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=120)) as up:
                return web.Response(body=await up.read(), status=up.status)
    except Exception as e:
        return web.json_response({"error": {"message": f"router: {e}"}}, status=502)

async def _warm(app):
    asyncio.create_task(ensure_backend("qwen"))

def make_app():
    app = web.Application(client_max_size=1024**3)
    app.router.add_get("/health", h_health)
    app.router.add_get("/v1/models", h_models)
    app.router.add_post("/v1/chat/completions", h_proxy)
    app.router.add_post("/v1/completions", h_proxy)
    app.router.add_route("*", "/{tail:.*}", h_catchall)
    app.on_startup.append(_warm)
    return app

if __name__ == "__main__":
    web.run_app(make_app(), host="0.0.0.0", port=PORT)
