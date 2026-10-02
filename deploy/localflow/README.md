# localflow — local Claude-Code-Workflows clone

Deterministic multi-agent orchestration over the **local Qwen3.8 engine**, with
transparent **overflow to DeepSeek** via the keepalive-shim gateway (`:8000`).
A local, $0-mostly reproduction of the Claude Code `Workflow` tool.

Spec: `~/Desktop/frontier-lab/specs/SPEC-local-workflows-orchestrator.md`
(EXP-042). Backend dependency (moonshots): EXP-035/036/038.

## Model
A **workflow** is a Python file exposing `async def run(args)` that uses the
injected primitives (no imports needed — the runner injects them):

| primitive | meaning |
|---|---|
| `await agent(prompt, *, schema=None, phase=None, model=None, label=None, system=None)` | one subagent run → gateway `:8000`. No `schema` → text; with `schema` (JSON Schema) → validated dict via guided decoding + retry. Returns `None` on terminal failure. |
| `await parallel([thunk, ...])` | **barrier**: run all concurrently, await all. A failing thunk → `None` (`.filter`/`if v`). |
| `await pipeline(items, s1, s2, ...)` | per-item stage chain, **no barrier**. Each stage `fn(prev, item, idx)`; may be sync or async. A throwing stage drops that item → `None`. |
| `await fork_agents(shared_prompt, agent_specs)` | **prefill once, fork N** (EXP-038). Prefills `shared_prompt` a single time on the fork-enabled engine, then fans out `len(agent_specs)` continuations that ride the shared KV prefix, each diverging only in sampling (`{temperature?, max_tokens?, top_p?, seed?, label?}`). Returns a text list aligned with `agent_specs`. Falls back **transparently** to N ordinary `agent()` calls when the engine lacks the routes. |
| `phase(title)` / `log(msg)` | progress grouping / narrator line. |

## Fork (prefill once, fork N) — EXP-038
`fork_agents` targets **`LOCALFLOW_ENGINE_URL` (direct `:8001`)**, not the gateway:
the custom `/tq/pin` · `/tq/fork` · `/tq/unpin` routes are feature-gated
(`VLLM_TQ_GDN_SNAPSHOT=1`) and the keepalive shim (`:8000`) does not proxy them
yet. If the engine 404s (routes absent) or is unreachable, `fork_agents` degrades
silently to N independent `agent()` calls. See
`examples/fork_research.py` (2112+ token corpus → 4 temperature-divergent
verifiers → synthesize) and the vLLM-side design in
`~/Desktop/.ftree-gdnsnap/docs/exp038-http-routes.md`.

## Overflow (the core ask)
Agents hit the gateway `:8000`; the shim does local-first with DeepSeek overflow.
Concurrency cap **C defaults to 12 > the shim's local budget (`SHIM_LOCAL_BUDGET`,
see `env/shim.env.example`)** → surplus agent
runs deterministically overflow to DeepSeek "for the additional load". Workflow
traffic is tagged `X-Client: workflow-bg` so it **yields to interactive** Hermes/pi
traffic. Force interactive priority with `--foreground`.

## Run
```bash
python ~/localflow/localflow.py <workflow.py> \
    [--concurrency N] [--journal PATH] [--resume] [--foreground] [--args JSON]

# validate the runtime end-to-end (cheap):
python ~/localflow/localflow.py ~/localflow/examples/smoke.py --concurrency 4

# deep-research shape (EXP-043 seed):
python ~/localflow/localflow.py ~/localflow/examples/deep_research.py \
    --concurrency 12 --journal /tmp/dr.jsonl \
    --args '"tradeoffs of linear vs full attention for long context"'
```

## Journal / resume
With `--journal PATH`, every completed `agent()` appends `{key,label,result}` keyed
by `hash(prompt,opts)`. Re-run with `--resume` to replay the unchanged prefix and
only re-run new/changed calls — same as the Workflow tool's resume.

## Env
`LOCALFLOW_GATEWAY` (default `http://127.0.0.1:8000/v1`), `LOCALFLOW_ENGINE_URL`
(fork routes; default `http://127.0.0.1:8001`), `LOCALFLOW_MODEL` (`qwen-local`),
`LOCALFLOW_CONCURRENCY` (12), `LOCALFLOW_MAX_TOKENS` (1536), `LOCALFLOW_TEMP`
(0.6), `LOCALFLOW_TIMEOUT` (600), `LOCALFLOW_MAX_AGENTS` (1000).

## Status — P1 MVP
Implemented: `agent/parallel/pipeline/phase/log`, guided-JSON + retry, concurrency
cap → overflow, journal + resume, bg-yield tagging, runaway cap, token accounting.
Next (P2/P3): rich pi-backed subagent for coding/web, per-agent `model`/effort/
worktree isolation, CLI registry, budget ledger. See the spec.
