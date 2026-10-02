"""localflow — a local Claude-Code-Workflows clone.

Deterministic multi-agent orchestration over the local Qwen3.8 engine, with
transparent overflow to a remote API (DeepSeek) via the keepalive-shim gateway
(:8000). A *workflow* is a Python module exposing `async def run(args)` that uses
the injected primitives:

    agent(prompt, *, schema=None, phase=None, model=None, label=None, system=None)
    parallel([thunk, ...])          # barrier; failed thunk -> None
    pipeline(items, stage1, stage2) # per-item stage chain, NO barrier
    phase(title) / log(msg)

Concurrency is capped at C (default 12). C > the shim's local budget (8) means
surplus agent runs deterministically overflow to DeepSeek "for the additional
load" — exactly the design ask. Every completed agent() appends to journal.jsonl
keyed by hash(prompt,opts); a rerun with --resume replays the unchanged prefix.

Run:  python localflow.py <workflow.py> [--concurrency N] [--journal PATH] [--resume] [--args JSON]
"""
from __future__ import annotations
import argparse, asyncio, hashlib, importlib.util, json, os, sys, time, urllib.request, urllib.error

GATEWAY = os.environ.get("LOCALFLOW_GATEWAY", "http://127.0.0.1:8000/v1")
# Direct engine base (NOT the gateway): the EXP-038 /tq/* fork routes are custom
# and the keepalive shim (:8000) does not proxy them yet, so fork_agents talks
# straight to the vLLM OpenAI server (:8001). No /v1 suffix — /tq/* live at root.
ENGINE_URL = os.environ.get("LOCALFLOW_ENGINE_URL", "http://127.0.0.1:8001")
MODEL = os.environ.get("LOCALFLOW_MODEL", "qwen-local")
DEFAULT_C = int(os.environ.get("LOCALFLOW_CONCURRENCY", "12"))
REQ_TIMEOUT = int(os.environ.get("LOCALFLOW_TIMEOUT", "600"))
MAX_AGENTS = int(os.environ.get("LOCALFLOW_MAX_AGENTS", "1000"))  # runaway backstop


def _now() -> float:
    return time.time()


class Orchestrator:
    """The singleton runtime the primitives delegate to."""

    def __init__(self, concurrency=DEFAULT_C, journal_path=None, resume=False, bg=True):
        self.C = concurrency
        self.sem = asyncio.Semaphore(concurrency)
        self.journal_path = journal_path
        self.bg = bg  # tag workflow load as background so it yields to interactive traffic
        self.cur_phase = "main"
        self.agent_count = 0
        self.spent_tokens = 0
        self._journal_cache = {}   # key -> result (loaded for resume)
        self._journal_fh = None
        if journal_path:
            if resume and os.path.exists(journal_path):
                self._load_journal(journal_path)
            self._journal_fh = open(journal_path, "a", buffering=1)

    # ---- journal / resume ----
    def _load_journal(self, path):
        n = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._journal_cache[rec["key"]] = rec["result"]
                    n += 1
                except Exception:
                    pass
        _emit(f"[resume] loaded {n} cached agent result(s) from {path}")

    def _journal_write(self, key, label, result):
        if self._journal_fh:
            self._journal_fh.write(json.dumps({"key": key, "label": label,
                                               "result": result, "ts": _now()}) + "\n")

    @staticmethod
    def _key(prompt, opts):
        h = hashlib.sha256()
        h.update(prompt.encode())
        h.update(json.dumps(opts, sort_keys=True, default=str).encode())
        return h.hexdigest()[:16]

    # ---- the HTTP call to the gateway (blocking; run in a thread) ----
    def _post(self, prompt, system, schema, model, temperature=None, max_tokens=None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model or MODEL,
            "messages": messages,
            "temperature": (float(os.environ.get("LOCALFLOW_TEMP", "0.6"))
                            if temperature is None else float(temperature)),
            "max_tokens": (int(os.environ.get("LOCALFLOW_MAX_TOKENS", "1536"))
                           if max_tokens is None else int(max_tokens)),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "out", "schema": schema, "strict": True},
            }
        headers = {"Content-Type": "application/json"}
        # shim routing tags: workflow load is background -> yields to interactive lanes
        headers["X-Client"] = "workflow-bg" if self.bg else "workflow"
        req = urllib.request.Request(GATEWAY + "/chat/completions",
                                     json.dumps(payload).encode(), headers)
        r = json.load(urllib.request.urlopen(req, timeout=REQ_TIMEOUT))
        usage = r.get("usage", {}) or {}
        text = (r["choices"][0]["message"]["content"] or "").strip()
        return text, usage

    async def agent(self, prompt, *, schema=None, phase=None, model=None,
                    label=None, system=None, retries=2,
                    temperature=None, max_tokens=None):
        self.agent_count += 1
        if self.agent_count > MAX_AGENTS:
            raise RuntimeError(f"agent cap {MAX_AGENTS} exceeded (runaway backstop)")
        ph = phase or self.cur_phase
        lbl = label or (prompt[:48].replace("\n", " "))
        opts = {"schema": schema, "model": model, "system": system, "phase": ph,
                "temperature": temperature, "max_tokens": max_tokens}
        key = self._key(prompt, opts)
        if key in self._journal_cache:
            _emit(f"  [{ph}] {lbl} … cached")
            return self._journal_cache[key]

        async with self.sem:
            last_err = None
            for attempt in range(retries + 1):
                t0 = _now()
                try:
                    text, usage = await asyncio.to_thread(
                        self._post, prompt, system, schema, model,
                        temperature, max_tokens)
                    self.spent_tokens += int(usage.get("total_tokens") or 0)
                    dt = _now() - t0
                    if schema is not None:
                        try:
                            result = json.loads(text)
                        except json.JSONDecodeError as je:
                            last_err = je
                            _emit(f"  [{ph}] {lbl} … bad-json retry {attempt+1}")
                            prompt = prompt + "\n\nReturn ONLY valid JSON matching the schema."
                            continue
                    else:
                        result = text
                    _emit(f"  [{ph}] {lbl} … ok ({dt:.0f}s, {usage.get('total_tokens','?')}tok)")
                    self._journal_write(key, lbl, result)
                    return result
                except urllib.error.HTTPError as e:
                    last_err = e
                    body = e.read().decode()[:120] if hasattr(e, "read") else ""
                    _emit(f"  [{ph}] {lbl} … HTTP {e.code} retry {attempt+1} {body}")
                    await asyncio.sleep(1.5 * (attempt + 1))
                except Exception as e:
                    last_err = e
                    _emit(f"  [{ph}] {lbl} … {type(e).__name__} retry {attempt+1}")
                    await asyncio.sleep(1.5 * (attempt + 1))
            _emit(f"  [{ph}] {lbl} … FAILED ({type(last_err).__name__})")
            return None

    # ---- EXP-038 "prefill once, fork N" (direct engine, custom /tq/* routes) ----
    def _tq_post(self, path, payload, engine_url=None):
        """Blocking POST to a custom /tq/* route on the DIRECT engine (:8001),
        not the gateway. Raises urllib HTTPError/URLError on failure."""
        base = (engine_url or ENGINE_URL).rstrip("/")
        req = urllib.request.Request(
            base + path, json.dumps(payload).encode(),
            {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as r:
            return json.load(r)

    async def fork_agents(self, shared_prompt, agent_specs, *,
                          engine_url=None, system=None, phase=None):
        """Prefill ``shared_prompt`` ONCE on the fork-enabled engine, then fan out
        ``len(agent_specs)`` continuations that all ride the shared KV prefix.

        Each spec is a dict of sampling overrides + optional ``label``:
        ``{temperature?, max_tokens?, top_p?, top_k?, seed?, min_tokens?,
        repetition_penalty?, label?}``. All children share the *same* prompt (the
        pinned prefix) and diverge only in sampling — the "same context, N
        samplers" shape (e.g. N verifiers at different temperatures). Returns a
        list of text results aligned with ``agent_specs``.

        Talks to LOCALFLOW_ENGINE_URL (direct :8001) because the /tq/* routes are
        custom and the keepalive shim (:8000) does not proxy them yet. Falls back
        TRANSPARENTLY to ``len(agent_specs)`` ordinary ``agent()`` calls when the
        engine lacks the routes (HTTP 404 / feature-gated) or is unreachable."""
        ph = phase or self.cur_phase
        # The fork prefix has no role separation; fold any system preamble in.
        full = f"{system}\n\n{shared_prompt}" if system else shared_prompt

        if self.agent_count + len(agent_specs) > MAX_AGENTS:
            raise RuntimeError(f"agent cap {MAX_AGENTS} exceeded (fork_agents preflight)")

        try:
            pin = await asyncio.to_thread(self._tq_post, "/tq/pin",
                                          {"prompt": full}, engine_url)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                _emit(f"  [{ph}] fork_agents: engine has no /tq routes (404) — "
                      f"falling back to {len(agent_specs)} agent() call(s)")
                return await self._fork_fallback(shared_prompt, agent_specs,
                                                 system=system, phase=ph)
            _emit(f"  [{ph}] fork_agents: /tq/pin HTTP {e.code} — fallback")
            return await self._fork_fallback(shared_prompt, agent_specs,
                                             system=system, phase=ph)
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            _emit(f"  [{ph}] fork_agents: engine {engine_url or ENGINE_URL} "
                  f"unreachable ({type(e).__name__}) — fallback")
            return await self._fork_fallback(shared_prompt, agent_specs,
                                             system=system, phase=ph)

        handle_id = pin.get("handle_id")
        _emit(f"  [{ph}] fork_agents: pinned {handle_id} "
              f"({pin.get('num_computed_tokens')}/{pin.get('prefix_len')} tok, "
              f"fully_prefilled={pin.get('fully_prefilled')})")
        # Build child sampling specs (whitelist keys the route accepts).
        _keys = ("temperature", "top_p", "top_k", "min_p", "seed", "min_tokens",
                 "repetition_penalty", "presence_penalty", "frequency_penalty",
                 "stop", "stop_token_ids", "ignore_eos")
        children = []
        for spec in agent_specs:
            child = {"max_tokens": int(spec.get("max_tokens", 256))}
            for k in _keys:
                if spec.get(k) is not None:
                    child[k] = spec[k]
            children.append(child)

        try:
            res = await asyncio.to_thread(
                self._tq_post, "/tq/fork",
                {"handle_id": handle_id, "children": children}, engine_url)
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:160] if hasattr(e, "read") else ""
            _emit(f"  [{ph}] fork_agents: /tq/fork HTTP {e.code} {body} — fallback")
            return await self._fork_fallback(shared_prompt, agent_specs,
                                             system=system, phase=ph)
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            _emit(f"  [{ph}] fork_agents: /tq/fork unreachable ({type(e).__name__}) — fallback")
            return await self._fork_fallback(shared_prompt, agent_specs,
                                             system=system, phase=ph)
        finally:
            # Always release the pin (idempotent) — covers success and the
            # except-fallback path above.
            await self._safe_unpin(handle_id, engine_url, ph)

        out = []
        forked = res.get("children", [])
        for i, spec in enumerate(agent_specs):
            ch = forked[i] if i < len(forked) else {}
            text = (ch.get("text") or "").strip()
            lbl = spec.get("label") or f"fork[{i}]"
            _emit(f"  [{ph}] {lbl} … forked "
                  f"({ch.get('num_output_tokens')}tok, "
                  f"cached={ch.get('num_cached_tokens')}, "
                  f"{ch.get('finish_reason')})")
            out.append(text or None)
        self.agent_count += len(out)
        return out

    async def _safe_unpin(self, handle_id, engine_url, ph):
        if not handle_id:
            return
        try:
            await asyncio.to_thread(self._tq_post, "/tq/unpin",
                                    {"handle_id": handle_id}, engine_url)
        except Exception as e:  # noqa: BLE001 - best effort; log a leak risk
            _emit(f"  [{ph}] fork_agents: unpin {handle_id} failed "
                  f"({type(e).__name__}) — possible KV leak until engine restart")

    async def _fork_fallback(self, shared_prompt, agent_specs, *,
                             system=None, phase=None):
        """N ordinary agent() calls — same shared prompt, per-spec sampling."""
        async def _one(spec, i):
            return await self.agent(
                shared_prompt, system=system,
                temperature=spec.get("temperature"),
                max_tokens=spec.get("max_tokens"),
                label=(spec.get("label") or f"fork-fallback[{i}]"),
                phase=phase)
        return await asyncio.gather(
            *[_one(s, i) for i, s in enumerate(agent_specs)])


# ---- module-level singleton + primitives injected into workflow scripts ----
_ORC: Orchestrator | None = None


def _emit(msg):
    print(msg, flush=True)


def phase(title):
    if _ORC:
        _ORC.cur_phase = title
    _emit(f"\n=== phase: {title} ===")


def log(msg):
    _emit(f"[log] {msg}")


async def agent(prompt, **kw):
    return await _ORC.agent(prompt, **kw)


async def fork_agents(shared_prompt, agent_specs, **kw):
    """Prefill ``shared_prompt`` once on the fork-enabled engine, then fan out
    ``len(agent_specs)`` sampling-divergent continuations that ride the shared KV
    prefix. Transparently falls back to N ordinary agent() calls if the engine
    lacks the /tq/* routes. See Orchestrator.fork_agents."""
    return await _ORC.fork_agents(shared_prompt, agent_specs, **kw)


async def parallel(thunks):
    """Barrier: await all thunks concurrently. A failing thunk -> None."""
    async def _guard(t):
        try:
            return await t()
        except Exception as e:
            _emit(f"  [parallel] thunk error: {type(e).__name__}: {e}")
            return None
    return await asyncio.gather(*[_guard(t) for t in thunks])


async def pipeline(items, *stages):
    """Per-item stage chain with NO barrier. Item A can be in stage 3 while B is
    still in stage 1. A stage that throws drops that item to None."""
    async def _run_item(item, idx):
        cur = item
        for si, stage in enumerate(stages):
            try:
                res = stage(cur, item, idx)
                cur = await res if asyncio.iscoroutine(res) else res
            except Exception as e:
                _emit(f"  [pipeline] item {idx} stage {si} error: {type(e).__name__}: {e}")
                return None
        return cur
    return await asyncio.gather(*[_run_item(it, i) for i, it in enumerate(items)])


def _load_workflow(path):
    spec = importlib.util.spec_from_file_location("wf_module", path)
    mod = importlib.util.module_from_spec(spec)
    # inject primitives so the script can use them without imports
    for name in ("agent", "fork_agents", "parallel", "pipeline", "phase", "log"):
        setattr(mod, name, globals()[name])
    spec.loader.exec_module(mod)
    return mod


async def _main_async(a):
    global _ORC
    _ORC = Orchestrator(concurrency=a.concurrency, journal_path=a.journal,
                        resume=a.resume, bg=not a.foreground)
    mod = _load_workflow(a.script)
    if not hasattr(mod, "run"):
        raise SystemExit(f"{a.script}: workflow must define `async def run(args)`")
    wf_args = json.loads(a.args) if a.args else None
    t0 = _now()
    _emit(f"[localflow] {a.script} C={a.concurrency} gateway={GATEWAY} "
          f"model={MODEL} bg={not a.foreground}")
    result = await mod.run(wf_args)
    dt = _now() - t0
    _emit(f"\n[localflow] done in {dt:.0f}s — {_ORC.agent_count} agent run(s), "
          f"~{_ORC.spent_tokens} tokens")
    print("\n===== RESULT =====")
    print(json.dumps(result, indent=2, default=str) if not isinstance(result, str) else result)
    return result


def main():
    p = argparse.ArgumentParser(description="localflow — local workflows orchestrator")
    p.add_argument("script", help="path to a workflow .py defining async def run(args)")
    p.add_argument("--concurrency", type=int, default=DEFAULT_C)
    p.add_argument("--journal", default=None, help="journal.jsonl path (enables resume)")
    p.add_argument("--resume", action="store_true", help="replay cached results from --journal")
    p.add_argument("--foreground", action="store_true",
                   help="tag as interactive (not bg) — competes with Hermes/pi traffic")
    p.add_argument("--args", default=None, help="JSON passed to run(args)")
    a = p.parse_args()
    if a.concurrency < 1:
        p.error("--concurrency must be >= 1")
    asyncio.run(_main_async(a))


if __name__ == "__main__":
    main()
