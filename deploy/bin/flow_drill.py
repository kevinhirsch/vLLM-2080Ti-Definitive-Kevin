#!/usr/bin/env python3
"""Capacity-aware flow drill (lane CF, 2026-10-02): the REAL keepalive-shim.py in front of a stub engine and a stub
remote, driven by a synthetic mix shaped like the estate's (Halo sessions with a shared prefix and tool steps, a card
runner, robot background work, Kevin's occasional interactive call), with time compressed ~1:10.

Why a drill and not the live box: the live engine is shared by every lane and fenced for benchmarks; nothing here
touches it (all ports are private, every shim state file is in a tmp dir, no GPU is used).

  flow_drill.py compare [--reps 3] [--secs 120]     SHIM_FLOW_MODE=off vs enforce, same load, N reps each
  flow_drill.py modes   [--secs 180]                mode-switch timeline: local+remote -> planned offline window ->
                                                    back -> engine killed -> engine back; prints a per-5 s series
Exit code 0 always; the numbers are the product.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import tempfile
import time

import aiohttp
import logging
from aiohttp import web

for _n in ("aiohttp.server", "aiohttp.access", "asyncio"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

HERE = os.path.dirname(os.path.abspath(__file__))
SHIM = os.environ.get("DRILL_SHIM", os.path.join(HERE, "keepalive-shim.py"))
CHARS_PER_TOK = 3.5

# ---- time-compressed engine model (production: ~900 uncached prefill tok/s, ~5-40 tok/s per stream) ----
PREFILL_TPS = 600.0
DECODE_TPS = 40.0
MAX_RUNNING = 8
CACHE_TTL = 90.0


def tok(s):
    return int(len(s) / CHARS_PER_TOK)


class StubEngine:
    def __init__(self):
        self.lock = asyncio.Lock()               # one prefill at a time, FIFO (the engine's own queue)
        self.sem = asyncio.Semaphore(MAX_RUNNING)
        self.waiting = self.running = 0
        self.up = True
        self.cache = {}                          # prefix key -> time its prefill finished
        self.computed = self.cached = 0
        self.wasted = 0                          # prefill tokens computed for requests whose caller then left
        self.gen = 0
        self.queries = self.hits = 0

    async def health(self, r):
        return web.Response(text="OK") if self.up else web.Response(status=503, text="down")

    async def metrics(self, r):
        m = 'engine="0",model_name="stub"'
        t = (f'vllm:num_requests_running{{{m}}} {self.running}\nvllm:num_requests_waiting{{{m}}} {self.waiting}\n'
             f'vllm:kv_cache_usage_perc{{{m}}} 0.2\nvllm:prompt_tokens_by_source_total{{{m},source="local_compute"}} {self.computed}\n'
             f'vllm:prompt_tokens_by_source_total{{{m},source="local_cache_hit"}} {self.cached}\n'
             f'vllm:prefix_cache_queries_total{{{m}}} {self.queries}\nvllm:prefix_cache_hits_total{{{m}}} {self.hits}\n'
             f'vllm:generation_tokens_total{{{m}}} {self.gen}\nvllm:prompt_tokens_total{{{m}}} {self.computed + self.cached}\n')
        return web.Response(text=t)

    async def chat(self, request):
        if not self.up:
            return web.Response(status=503, text="down")
        j = await request.json()
        msgs = j.get("messages") or []
        key = hashlib.sha256(json.dumps(msgs[:1], sort_keys=True).encode()).hexdigest()[:12]
        shared = tok(json.dumps(msgs[:1]))
        ptok = sum(tok(m.get("content") or "") for m in msgs)
        now = time.time()
        credit = shared if (key in self.cache and now - self.cache[key] < CACHE_TTL) else 0
        computed = max(1, ptok - credit)
        out = int(j.get("_out") or 60)
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        spent = 0
        phase = "queued"
        self.waiting += 1
        try:
            async with self.sem:
                async with self.lock:
                    self.waiting -= 1
                    phase = "prefill"
                    self.running += 1
                    step = max(1, int(computed / 12))
                    while spent < computed:
                        n = min(step, computed - spent)
                        await asyncio.sleep(n / PREFILL_TPS)
                        spent += n
                    self.computed += computed
                    self.cached += credit
                    self.queries += ptok
                    self.hits += credit
                    self.cache[key] = time.time()
                    phase = "decode"
                # decode runs outside the prefill lock
                await resp.prepare(request)
                await resp.write(b'data: ' + json.dumps({"choices": [{"delta": {"content": "x"}}]}).encode() + b"\n\n")
                for _ in range(8):
                    await asyncio.sleep(out / 8 / DECODE_TPS)
                    await resp.write(b'data: ' + json.dumps({"choices": [{"delta": {"content": "y"}}]}).encode() + b"\n\n")
                self.gen += out
                await resp.write(b'data: ' + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {
                    "prompt_tokens": ptok, "completion_tokens": out, "prompt_tokens_details": {"cached_tokens": credit}}}).encode()
                                 + b"\n\ndata: [DONE]\n\n")
        except (asyncio.CancelledError, ConnectionResetError, ConnectionError):
            if phase == "queued":
                self.waiting -= 1
            elif phase == "prefill":
                self.computed += spent
                self.wasted += spent            # the engine computed this and the caller left before using it
            return resp
        finally:
            if phase in ("prefill", "decode"):
                self.running -= 1
        return resp


class StubRemote:
    def __init__(self):
        self.calls = 0
        self.by_client = {}

    async def chat(self, request):
        j = await request.json()
        self.calls += 1
        c = request.headers.get("X-Client", "?")
        self.by_client[c] = self.by_client.get(c, 0) + 1
        await asyncio.sleep(1.0)
        usage = {"prompt_tokens": 500, "completion_tokens": 40, "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 500}
        if j.get("stream"):
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            await resp.write(b'data: ' + json.dumps({"choices": [{"delta": {"content": "r"}}]}).encode() + b"\n\n")
            await resp.write(b'data: ' + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": usage}).encode()
                             + b"\n\ndata: [DONE]\n\n")
            return resp
        return web.json_response({"choices": [{"message": {"role": "assistant", "content": "remote"}, "finish_reason": "stop"}],
                                  "usage": usage, "model": "stub-remote"})


# ---- load: classes shaped like the estate (names matter: the gateway classifies by X-Client) ----
CLASSES = {
    # xclient, patience (s, caller timeout = declared deadline), shared prefix tokens, unique tokens
    "kevin": ("ubuntuide01 pi / OpenCode", 45, 700, 300),
    "halo": ("halo-hermes", 60, 3200, 1400),
    "runner": ("card-repair-engine", 70, 600, 500),
    "background": ("overseer-refine-overflow", 40, 300, 600),
}


def body_for(cls, session, step, rng):
    xc, _, shared, uniq = CLASSES[cls]
    sysmsg = ("S%s " % session) + ("lorem ipsum " * int(shared * CHARS_PER_TOK / 12))
    hist = ("u%d " % step) * 3
    u = hist + ("data " * int(uniq * (1 + 0.15 * min(step, 8)) * CHARS_PER_TOK / 5))
    return {"model": "estate", "stream": True, "max_tokens": 200, "_out": rng.choice((40, 60, 90)),
            "messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": u}]}


class Stats:
    def __init__(self):
        self.rows = []                          # (t, cls, outcome, ttft, total)
        self.cycles = 0
        self.t0 = time.time()

    def add(self, cls, outcome, ttft, total):
        self.rows.append((time.time() - self.t0, cls, outcome, ttft, total))


async def one_call(sess, base, cls, session, step, rng, stats, declare):
    xc, patience, _, _ = CLASSES[cls]
    h = {"X-Client": xc, "Content-Type": "application/json"}
    if declare:
        h["X-Gateway-Deadline-S"] = str(patience)
    t0 = time.time()
    ttft = None
    try:
        async with asyncio.timeout(patience):
            async with sess.post(base + "/v1/chat/completions", json=body_for(cls, session, step, rng), headers=h) as r:
                if r.status != 200:
                    await r.read()
                    stats.add(cls, "refused" if r.status in (429, 503) else "error", None, time.time() - t0)
                    return False, (float(r.headers.get("Retry-After") or 5))
                async for chunk in r.content.iter_any():
                    if ttft is None and chunk:
                        ttft = time.time() - t0
                stats.add(cls, "ok", ttft, time.time() - t0)
                return True, 0
    except (asyncio.TimeoutError, TimeoutError):
        stats.add(cls, "timeout", ttft, time.time() - t0)
        return False, 0
    except aiohttp.ClientError:
        stats.add(cls, "error", None, time.time() - t0)
        return False, 2


async def halo_session(sess, base, sid, stop, stats, rng, declare):
    step = 0
    streak = 0
    while time.time() < stop:
        ok, ra = await one_call(sess, base, "halo", sid, step, rng, stats, declare)
        if ok:
            streak += 1
            step += 1
            if streak % 3 == 0:
                stats.cycles += 1
            await asyncio.sleep(rng.uniform(2.0, 6.0))        # a tool step
        else:
            streak = 0
            await asyncio.sleep(max(1.0, ra) if ra else 1.0)


async def runner_worker(sess, base, wid, stop, stats, rng, declare):
    n = 0
    while time.time() < stop:
        ok, ra = await one_call(sess, base, "runner", "r%d-%d" % (wid, n // 6), n, rng, stats, declare)
        n += 1
        await asyncio.sleep(rng.uniform(0.5, 2.0) if ok else max(1.0, ra))


async def poisson(sess, base, cls, rate, stop, stats, rng, declare, tag):
    tasks = []
    n = 0
    while time.time() < stop:
        await asyncio.sleep(rng.expovariate(rate))
        n += 1
        tasks.append(asyncio.create_task(one_call(sess, base, cls, "%s%d" % (tag, n % 3), n, rng, stats, declare)))
    await asyncio.gather(*tasks, return_exceptions=True)


async def capacity_scale(sess, base, cls="background"):
    """The estate's capacity_flow.scale_from() arithmetic, inline: the share of local prefill capacity the higher classes
    leave free, damped when this class already queues past the deadline it is allowed."""
    try:
        async with sess.get(base + "/gateway/capacity") as r:
            f = await r.json()
    except Exception:
        return 1.0
    order = ["kevin", "halo", "runner", "background"]
    above = order[:order.index(cls)]
    util = sum(f["demand"][c]["uncached_prefill_s_per_min_5m"] for c in above) / 60.0
    spare = max(0.0, min(1.0, 1.0 - util))
    q = f["queue"][cls]
    dl, wait = q.get("default_deadline_s"), q.get("expected_wait_s")
    damp = 1.0 if (not dl or wait is None or wait <= dl) else dl / wait
    return spare * damp


async def poisson_follow(sess, base, cls, rate, stop, stats, rng, declare, tag):
    """Open-loop arrivals scaled by capacity_scale (refreshed every 3 s): the producer follows measured capacity."""
    tasks, n, scale, last = [], 0, 1.0, 0.0
    while time.time() < stop:
        if time.time() - last > 3:
            scale, last = await capacity_scale(sess, base, cls), time.time()
        await asyncio.sleep(rng.expovariate(max(rate * max(scale, 0.05), 1e-6)))
        if rng.random() > scale:                   # time-shift: this arrival is not produced now
            stats.deferred = getattr(stats, "deferred", 0) + 1
            continue
        n += 1
        tasks.append(asyncio.create_task(one_call(sess, base, cls, "%s%d" % (tag, n % 3), n, rng, stats, declare)))
    await asyncio.gather(*tasks, return_exceptions=True)


async def load(base, secs, seed, stats, declare=True, mix=None, follow=False):
    rng = random.Random(seed)
    stop = time.time() + secs
    mix = mix or {"halo": 3, "runner": 2, "bg_rate": 0.3, "kevin_rate": 0.06}
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as sess:
        tasks = [asyncio.create_task(halo_session(sess, base, "h%d" % i, stop, stats, random.Random(seed * 10 + i), declare))
                 for i in range(mix["halo"])]
        tasks += [asyncio.create_task(runner_worker(sess, base, i, stop, stats, random.Random(seed * 100 + i), declare))
                  for i in range(mix["runner"])]
        bg = poisson_follow if follow else poisson
        tasks.append(asyncio.create_task(bg(sess, base, "background", mix["bg_rate"], stop, stats, random.Random(seed + 1), declare, "b")))
        tasks.append(asyncio.create_task(poisson(sess, base, "kevin", mix["kevin_rate"], stop, stats, random.Random(seed + 2), declare, "k")))
        await asyncio.gather(*tasks, return_exceptions=True)


# ---- the rig ----
class Rig:
    def __init__(self, flow_mode, remote=True, tmp=None):
        self.flow_mode, self.remote = flow_mode, remote
        self.tmp = tmp or tempfile.mkdtemp(prefix="flow-drill-")
        self.engine, self.rem = StubEngine(), StubRemote()
        self.runners = []
        self.proc = None

    async def start(self):
        for stub, routes, attr in ((self.engine, [("/health", "health"), ("/metrics", "metrics")], "pe"),
                                   (self.rem, [], "pr")):
            app = web.Application()
            if stub is self.engine:
                for p, n in routes:
                    app.router.add_get(p, getattr(stub, n))
            app.router.add_post("/v1/chat/completions", stub.chat)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            setattr(self, attr, site._server.sockets[0].getsockname()[1])
            self.runners.append(runner)
        import socket
        s = socket.socket(); s.bind(("127.0.0.1", 0)); self.pg = s.getsockname()[1]; s.close()
        t = self.tmp
        env = {k: v for k, v in os.environ.items() if not k.startswith("SHIM_")}
        env.update({
            "SHIM_PORT": str(self.pg), "SHIM_UPSTREAM": f"http://127.0.0.1:{self.pe}",
            "SHIM_REMOTE_BASE": f"http://127.0.0.1:{self.pr}" if self.remote else "", "SHIM_REMOTE_KEY": "drill" if self.remote else "",
            "SHIM_REMOTE_MODEL": "deepseek-flash",
            "SHIM_SPEND_FILE": f"{t}/spend.json", "SHIM_SPEND_CLIENTS_FILE": f"{t}/spend-clients.json",
            "SHIM_STATS_FILE": f"{t}/stats.json", "SHIM_TELEMETRY_DIR": f"{t}/telemetry", "SHIM_ENV_FILE": f"{t}/shim.env",
            "SHIM_ALIASES_FILE": f"{t}/aliases.json", "SHIM_FLIGHTREC_DIR": f"{t}/flightrec",
            "SHIM_ADMIN_TOKEN_FILE": f"{t}/none.token", "GATEWAY_DRAIN_LEDGER": f"{t}/drains.jsonl",
            "SHIM_FLOW_EVENTS_FILE": f"{t}/incidents/capacity-mode.jsonl",
            "SHIM_EXACT_TOKENS": "0", "SHIM_LOCAL_BUDGET": "10", "SHIM_PREFILL_TPS": str(PREFILL_TPS),
            "SHIM_PREFIX_ALIGN_TOKENS": "100", "SHIM_PREFIX_HIT_MARGIN_TOKENS": "50", "SHIM_USE_COMPUTED_COST": "0",
            "SHIM_BIG_PROMPT": "0", "SHIM_BIG_OUTPUT": "0", "SHIM_MONSTER_INFLIGHT": "0", "SHIM_FOREIGN_LOAD_GUARD": "0",
            "SHIM_PERF_BREAKER_ENABLED": "0", "SHIM_PREDICTED_OCCUPANCY_SECS": "0", "SHIM_TELEM_SAMPLE_SECS": "1",
            "SHIM_LOCAL_WAIT_SECS": "4", "SHIM_BG_WAIT_SECS": "3", "SHIM_BG_WAIT_LOCAL_SECS": "25", "SHIM_FG_RESERVED": "2",
            "SHIM_TINY_TOKENS": "0", "SHIM_BG_LOCAL_ONLY": "0", "SHIM_INTERACTIVE_NEVER_OVERFLOW": "1",
            "SHIM_FLOW_MODE": self.flow_mode, "SHIM_FLOW_DEMAND_WINDOW_S": "40", "SHIM_FLOW_BACKLOG_S": "5", "SHIM_FLOW_DEADLINES": "kevin=0,halo=0,runner=70,background=40",
            "SHIM_FLOW_PREFIX_HOLD_MAX_S": "8", "SHIM_FLOW_STARVE_S": "30", "SHIM_FLOW_URGENT_SLACK_S": "4",
            "SHIM_FIRST_TOKEN_MAX": "600", "SHIM_LOCAL_FIRST_FIRST_TOKEN_MAX": "600", "SHIM_STREAM_IDLE_TIMEOUT_SECS": "600",
            "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": "",
        })
        self.log = open(f"{t}/shim.log", "w")
        self.proc = subprocess.Popen([sys.executable, SHIM], env=env, stdout=self.log, stderr=subprocess.STDOUT)
        self.base = f"http://127.0.0.1:{self.pg}"
        for _ in range(100):
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.get(self.base + "/health") as r:
                        if r.status == 200:
                            return
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(0.3)
        raise RuntimeError("shim did not start: " + open(f"{self.tmp}/shim.log").read()[-800:])

    async def get(self, path):
        async with aiohttp.ClientSession() as s:
            async with s.get(self.base + path) as r:
                return await r.json()

    async def call(self, path, method="POST", payload=None):
        async with aiohttp.ClientSession() as s:
            async with s.request(method, self.base + path, json=payload) as r:
                return await r.json()

    async def stop(self):
        if self.proc:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        for r in self.runners:
            await r.cleanup()


def pct(v, q):
    v = sorted(v)
    return round(v[min(len(v) - 1, int(q * len(v)))], 2) if v else None


def summarize(stats, rig, secs):
    out = {}
    for c in CLASSES:
        rows = [r for r in stats.rows if r[1] == c]
        ok = [r for r in rows if r[2] == "ok"]
        out[c] = {"ok": len(ok), "timeout": sum(1 for r in rows if r[2] == "timeout"),
                  "refused": sum(1 for r in rows if r[2] == "refused"), "error": sum(1 for r in rows if r[2] == "error"),
                  "ttft_p50": pct([r[3] for r in ok if r[3] is not None], .5), "ttft_p95": pct([r[3] for r in ok if r[3] is not None], .95)}
    e = rig.engine
    out["_engine"] = {"computed_tok": e.computed, "cached_tok": e.cached, "wasted_prefill_tok": e.wasted,
                      "cache_hit_share": round(e.cached / max(1, e.cached + e.computed), 3),
                      "wasted_share": round(e.wasted / max(1, e.computed), 3)}
    out["_flow"] = {"bg_deferred": getattr(stats, "deferred", 0), "halo_cycles": stats.cycles, "remote_calls": rig.rem.calls, "completed_per_min": round(60 * sum(out[c]["ok"] for c in CLASSES) / secs, 1),
                    "timeouts_per_hr": round(3600 * sum(out[c]["timeout"] for c in CLASSES) / secs)}
    return out


async def run_once(flow_mode, secs, seed, remote=False, declare=True, follow=False):
    rig = Rig(flow_mode, remote=remote)
    await rig.start()
    stats = Stats()
    try:
        await load(rig.base, secs, seed, stats, declare=declare, follow=follow)
        cap = await rig.get("/gateway/capacity")
        res = summarize(stats, rig, secs)
        res["_gateway"] = {"mode": cap.get("mode"), "refused": {c: cap["queue"][c]["refused_total"] for c in CLASSES},
                           "affinity": cap.get("affinity"), "counters": cap.get("counters")}
        return res
    finally:
        await rig.stop()


def fmt(res):
    lines = []
    for c in CLASSES:
        r = res[c]
        lines.append("  %-10s ok %3d  timeout %3d  refused %3d  ttft p50/p95 %s/%s" % (c, r["ok"], r["timeout"], r["refused"], r["ttft_p50"], r["ttft_p95"]))
    lines.append("  engine %s" % json.dumps(res["_engine"]))
    lines.append("  flow   %s" % json.dumps(res["_flow"]))
    return "\n".join(lines)


ARMS = (("off", "off", False), ("enforce", "enforce", False), ("enforce+follow", "enforce", True))


async def compare(reps, secs, arms=None):
    arms = [a for a in ARMS if (not arms or a[0] in arms)]
    allr = {a[0]: [] for a in arms}
    for rep in range(reps):
        for name, mode, follow in arms:
            t = time.time()
            r = await run_once(mode, secs, seed=100 + rep, follow=follow)
            allr[name].append(r)
            print("== rep %d  ARM=%s  (%.0fs)" % (rep + 1, name, time.time() - t))
            print(fmt(r), flush=True)
    print("\n== MEDIAN OVER %d REPS (min..max)" % reps)
    def med(name, getter):
        v = [getter(r) for r in allr[name] if getter(r) is not None]
        return (statistics.median(v), min(v), max(v)) if v else (None, None, None)
    keys = []
    for c in CLASSES:
        for key in ("ttft_p50", "ttft_p95", "ok", "timeout", "refused"):
            keys.append(("%s %s" % (c, key), (lambda c, key: lambda r: r[c][key])(c, key)))
    for key in ("wasted_prefill_tok", "cache_hit_share"):
        keys.append(("engine " + key, (lambda key: lambda r: r["_engine"][key])(key)))
    for key in ("halo_cycles", "completed_per_min", "timeouts_per_hr", "bg_deferred"):
        keys.append(("flow " + key, (lambda key: lambda r: r["_flow"][key])(key)))
    f = lambda x: "-" if x[0] is None else "%s (%s..%s)" % tuple(round(y, 2) for y in x)
    print("%-26s" % "metric" + "".join("%-26s" % a[0] for a in arms))
    for name, g in keys:
        print("%-26s" % name + "".join("%-26s" % f(med(a[0], g)) for a in arms))
    return allr


async def modes(secs):
    """Mode-switch timeline with real flow in enforce mode and a remote available."""
    rig = Rig("enforce", remote=True)
    await rig.start()
    stats = Stats()
    series = []
    t_start = time.time()
    seg = secs / 6.0
    events = []

    async def sampler():
        while time.time() - t_start < secs + 5:
            await asyncio.sleep(5)
            try:
                cap = await rig.get("/gateway/capacity")
            except Exception:
                continue
            ok_now = sum(1 for r in stats.rows if r[2] == "ok")
            series.append((round(time.time() - t_start), cap["mode"], {c: cap["queue"][c]["waiting"] for c in CLASSES},
                           cap["pressure"]["inflight"], rig.engine.waiting, rig.rem.calls, ok_now,
                           sum(1 for r in stats.rows if r[2] == "timeout")))

    async def script():
        await asyncio.sleep(seg)
        d = await rig.call("/gateway/offline", "POST", {"ttl_s": 600, "reason": "drill: speed benchmark", "by": "drill"})
        events.append((round(time.time() - t_start), "OFFLINE WINDOW OPENED (planned)"))
        lease = d["lease"]
        await asyncio.sleep(seg)
        await rig.call("/gateway/offline", "DELETE", {"lease": lease})
        events.append((round(time.time() - t_start), "OFFLINE WINDOW CLOSED"))
        await asyncio.sleep(seg)
        rig.engine.up = False
        events.append((round(time.time() - t_start), "ENGINE KILLED (unplanned)"))
        await asyncio.sleep(seg)
        rig.engine.up = True
        events.append((round(time.time() - t_start), "ENGINE BACK"))

    try:
        await asyncio.gather(load(rig.base, secs, 7, stats), sampler(), script())
        print("%5s  %-13s %-26s %6s %7s %6s %6s %6s" % ("t(s)", "mode", "waiting k/h/r/b", "infl", "eng.q", "remote", "done", "tmout"))
        for s in series:
            w = s[2]
            print("%5d  %-13s %-26s %6d %7d %6d %6d %6d" % (s[0], s[1], "%d/%d/%d/%d" % (w["kevin"], w["halo"], w["runner"], w["background"]),
                                                          s[3], s[4], s[5], s[6], s[7]))
        print("events:", events)
        cap = await rig.get("/gateway/capacity")
        print("mode_changes:", json.dumps([(e["from"], e["to"]) for e in reversed(cap["mode_changes"])]))
        print("remote_use 15m:", json.dumps(cap["remote_use"]["windows"]["15m"]))
        print(fmt(summarize(stats, rig, secs)))
        return series, events, cap
    finally:
        await rig.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["compare", "modes"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--secs", type=int, default=120)
    ap.add_argument("--arms", default="", help="comma list of: off, enforce, enforce+follow (default all)")
    a = ap.parse_args()
    asyncio.run(compare(a.reps, a.secs, [x for x in a.arms.split(",") if x]) if a.cmd == "compare" else modes(a.secs))


if __name__ == "__main__":
    main()
