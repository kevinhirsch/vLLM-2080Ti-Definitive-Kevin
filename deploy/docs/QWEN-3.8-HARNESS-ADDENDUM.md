# Qwen 3.8 harness addendum — Claude-replacement stack notes

_2026-08-16 addendum to Qwen-AgentZero-Optimization.md (written for 3.6; its
principles stand — this covers only what the 3.8 era changes). Applies to pi,
Agent Zero, and both Hermes clients behind the `:8000` gateway._

## 1. Grammar-constrained tool calling is now ON the menu (Tier 1.2, unblocked)

The optimization doc's biggest lever — guided decoding — was parked because
xgrammar crashes with spec-decode + GPTQ-Int4 (vLLM #11484). Two things changed:

- Both serve scripts now pin `--structured-outputs-config '{"backend":"guidance"}'`
  (llguidance). The xgrammar trap can no longer be selected by "auto", including
  the day MTP comes back.
- Upstream v0.1.15 fixed **named `tool_choice`** response handling, and
  `bench_toolcalls.sh` qualifies it.

What clients get, today, with no client-side parser work:

| mechanism | when to use |
|---|---|
| `tool_choice={"type":"function","function":{"name":...}}` | The orchestrator already knows which tool must run (pi's read-before-edit loop, AZ's dispatch step). The call is schema-forced server-side — malformed JSON becomes structurally impossible. |
| `response_format` / `guided_json` | Structured reports from subagents (evidence blocks, verdicts). Grammar-enforced by llguidance. |
| `tool_choice="auto"` | Open-ended turns. Still parser-dependent (`qwen3_xml` + froggeric v22) — keep the client-side tolerance you already have. |

Caveat that stays true: guided decoding constrains *format*, not *judgment* —
keep the evidence gates (lint/test/git-diff fed back) from the optimization doc.

## 2. reasoning_effort: what it actually does (measured, don't re-litigate)

- `reasoning_effort` is a **prompt-level hint** injected by the template
  (v22-official injects `xhigh` by default). Measured 2026-08-14 on this box:
  it does **not bound** thinking — `max_tokens=4000` + `effort=low` still
  returned an empty answer.
- The **hard bound** is the shim's `thinking_token_budget` guard (live,
  default-on): thinking gets `THINK_BUDGET_FRAC` of the caller's `max_tokens`,
  floored/capped, and the guard no longer lets `reasoning_effort` bypass it
  (746d8b1). Measured: same answer, 7x faster at budget=200.

Client guidance:
- **Execution turns** (tool loops, edits): send `chat_template_kwargs:
  {"enable_thinking": false}` — or just a sane `max_tokens` and let the shim
  guard bound it. Don't rely on `effort=low`.
- **Planning/research turns**: leave thinking on; optionally raise quality with
  `effort=xhigh` *and* an explicit larger `max_tokens` so the budget guard
  scales with it. The v22 templates also accept `<|think_on|>`/`<|think_off|>`
  inline toggles and `auto_disable_thinking_with_tools=true` if you want
  thinking off exactly when tools are attached.

## 3. Client checklist for the 3.8 stack (state as of 746d8b1 + this branch)

- [x] Gateway/watchdog: model-agnostic, nothing to do.
- [ ] pi `~/.pi/agent/models.json` (VM at 10.0.1.12): point provider `local` at
  **`qwen-local`** (served since 746d8b1; survives every future swap). Same for
  any subagent defs pinning `qwen3.6:27b` — that alias still works but is
  legacy.
- [ ] AZ (Docker :5080): `ctx_length` 245000 still correct (engine 256000).
  Keep `enable_thinking:false` for execution profiles.
- [ ] Hermes ×2: they send `reasoning_effort=medium` — fine; the budget guard
  now applies to them too (they were bypassing it before 746d8b1).
- [ ] Sampling: server `gencfg` remains the source of truth (temp 0.6 / top_k 20
  / top_p 0.95). The official 3.8 instruct rec is T0.7/top_p 0.8/presence 1.5 —
  worth an A/B *after* MTP requalification, since temperature trades directly
  against MTP acceptance (0.71@T0.4 → 0.52@T0.8 on 3.6).

## 4. Subagent fan-out reality check

"Unlimited subagents" was verified 2026-08-12 (3 parallel + overflow, 0 OOM)
under budget=1. The 2026-08-15 campaign raised `SHIM_LOCAL_BUDGET` to 8 with
`max-num-seqs 8` — more parallel subagents run truly local now, but the
per-request budget math (`SHIM_MAX_LOCAL_TOKENS=256000`, big-prompt 240000)
is what keeps a fan-out from OOMing the box. If a burst regresses into
EngineCore OOMs, the first knob is budget back down (8→4), not max-num-seqs.

## 5. If MTP requalification ships (see QWEN-3.8-MTP-REQUALIFICATION.md)

Nothing changes for clients — same aliases, same gateway. Expect single-stream
decode to roughly return to the 3.6-era 80-100 tok/s band. Re-run
`bench_toolcalls.sh` once after adoption: guided decoding + MTP is the exact
combination the old xgrammar bug punished, and G3 is the regression net.
