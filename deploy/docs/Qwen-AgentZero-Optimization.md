# Making Agent Zero + Qwen 3.6-27B More Reliable / More "Claude Code-like"

Compiled 2026-07-17 from: the "45 days with Qwen 3.6" r/LocalLLM field report, deep research on Qwen3/vLLM serving, vLLM speculative decoding, Agent Zero internals, and Claude Code's engineering patterns. Cross-referenced against this box's live config (`10.0.1.225:8000`, `vllm-qwen27b.service`, Agent Zero in Docker on `localhost:5080`).

---

## The one principle everything reduces to

**Qwen 3.6-27B is "a good reader, a bad guesser."** Every independent source — the Reddit operator, Anthropic's Claude Code design, and the Qwen serving research — lands on the same thing:

> Reliability comes from **deterministic scaffolding around the model** (evidence, grammar constraints, hooks, gates, fresh facts), **NOT** from better prompting.

The Reddit operator's hard-won version, after weeks of whack-a-mole:
> *"Prose doesn't enforce anything. Only code does. A rule marked MANDATORY and used to gate the output worked 8/8. But the thing that actually worked every single time was never the prompt — it was a deterministic check outside the model (a `git diff` scan, a substring match) fed back as evidence. Rules it can argue with, it argues with. Evidence it can't fake, it accepts. Turn your soft rules into lint gates. The model will negotiate with a style guide. It won't negotiate with a failing build."*

Claude Code is that principle implemented in software: schema-validated tool calls, read-before-edit, hooks that force tests to run, permission gates, verify-by-running. So "make Agent Zero more Claude Code-like" and "make Qwen reliable" are the **same project**.

---

## Already done this session (all confirmed correct by the research)

- **Disabled Qwen native thinking** (`enable_thinking: false`) — Reddit + Qwen research both confirm reasoning **OFF** for coding/agentic. The model "loops within its reasoning context: *But wait… But wait… Actually… Actually…*" and Q4 quant amplifies it. Keep thinking for research/spec-writing only, off for execution.
- **Vision off** — server is `--language-model-only`; sending images caused an infinite `vision_load` loop (AZ issue #544).
- **ctx_length 245000** — matches server `--max-model-len 256000`. Undersized context → truncation → malformed output → loops was the #1 local-model failure.
- **Circuit breaker `max_consecutive_unusable_responses: 5`** — the v2.4 runaway-loop breaker.
- **Rate limits 0/unlimited** on all model slots — correct for a local endpoint (no external quota to protect).
- **Tool parser `qwen3_xml`** (server-side) — research confirms this is the *more robust* choice vs the `qwen3_coder` regex parser, which throws JSON parse errors on code-with-`<`/`>` arguments. Already optimal.

---

## TIER 1 — Highest leverage, do next

### 1. Patch the chat template (likely ROOT CAUSE of the "Message misformat" crashes)
The Reddit thread's single most-upvoted technical comment:
> *"The worse tool calling is because of a few bugs in the standard chat template. On huggingface **froggeric** has patched templates. Since I switched I've had almost no issues with tool calling for Qwen 3.6."* (corroborated by a second commenter)

This box has a `chat_template.jinja` (12 KB) in `~/Desktop/models/Qwen3.6-27B-GPTQ-Int4/`. We've been fixing the misformat loop *symptomatically* (thinking off, circuit breaker). This is the potential **source-level fix**.
- **Action:** Diff the current template against froggeric's patched Qwen 3.6 template on HuggingFace. If vLLM isn't being pointed at a known-good template, pass `--chat-template /path/to/patched.jinja` in `serve-tqk8v4-fg.sh`.
- **Why it matters:** a template bug corrupts tool-call formatting at generation time — no amount of AZ-side parsing tolerance fully fixes that.

### 2. Grammar / schema-constrained tool calling (the biggest single reliability lever)
Both the Claude Code research and the Qwen research name this #1. Claude Code's tool reliability comes from the **API validating tool calls against a JSON schema** — malformed JSON is structurally impossible. Agent Zero instead relies on Qwen hand-writing a JSON envelope in prose, which is exactly what breaks.
- **Action:** Move Agent Zero from prompt-parsed JSON toward vLLM's structured/guided decoding (`guided_json` / llguidance) so tool-call JSON is grammar-constrained to always-valid.
- **⚠️ Critical caveat for THIS box:** vLLM issue #11484 — **xgrammar guided decoding crashes when combined with speculative decoding on GPTQ-Int4 models.** You run GPTQ-Int4 **+** MTP. So naïve `guided_json` may crash the server. Mitigations: use the **`guidance` (llguidance) backend** instead of xgrammar, or test with MTP temporarily disabled. This is the "rewrite AZ's dispatch loop" change I flagged earlier — high value, non-trivial, and needs careful testing against your exact stack.

### 3. Turn soft rules into deterministic gates (evidence, not prose)
The Reddit operator's #1 lesson = Claude Code's hooks pattern. In Agent Zero terms, use the **extension/hook points** (`tool_execute_after`, `tool_execute_before`, `monologue_end`) to run real checks and feed results back as evidence:
- After any file edit → auto-run lint/tests/`git diff`, inject the result. The model accepts a failing build; it argues with a style guide.
- Gate "done" on an external check passing, not on the model's say-so (fixes "it grades its own homework and gives itself an A").
- **Action:** Add AZ extensions under `/a0/usr/agents/<profile>/extensions/python/tool_execute_after/` that shell out to your project's lint/test and append the output to the tool result.

### 4. Feed it fresh facts — context7 / current library docs (fixes the hallucination spiral)
Qwen 3.6's data cutoff is mid-2024. It guesses at fast-moving libraries and spirals ("asked to pin Tailwind 4.3.2, it pinned 3.x in 6/6 runs"). Multiple Reddit commenters and one 91%-one-shot-success operator fixed this with up-to-date docs.
- **Action:** Add the **context7 MCP server** (or generate your own docs from library HTML into the AZ knowledge base) so current API/library facts are in context. AZ has MCP client support (`mcp_servers` setting). One operator: *"generate your own documentation for third-party libraries from their HTML docs → 91% one-shot success on C++ tickets."*

---

## TIER 2 — Structural (Claude Code patterns → Agent Zero primitives)

### 5. Permission gating / denylist destructive commands
Reddit: *"Loves to trash the codebase. One unclear instruction and it will go ham — it deleted a sibling task's committed, verified deliverables and returned 'ok'."* Claude Code prevents this with allow/deny rules + confirm-before-irreversible.
- **Action:** AZ `tool_execute_before` hook that hard-blocks `rm -rf`, force-push, `curl | sh`, and edits to protected paths — independent of the model's judgment.

### 6. Read-before-edit, unique-match string replace
Claude Code's `Edit` requires the file to have been read first and the `old_string` to match uniquely, or it's rejected. Kills a whole class of hallucinated-rewrite / file-corruption failures.
- **Action:** If you build a Qwen-specific edit tool, enforce these two constraints.

### 7. Bash timeouts
Reddit: *"Sometimes the model executes a Bash command that never finishes, causing it to wait indefinitely."*
- **Action:** Wrap AZ's `code_execution_tool` calls with a hard timeout so a hung command can't stall the whole run.

### 8. Subagent context isolation (AZ already has the primitive)
Claude Code's biggest context win: subagents run in isolated context and return **only a condensed summary**, never their raw transcript. AZ's `call_subordinate` can do this — ensure subordinates return summaries, not raw tool dumps, so the orchestrator's context stays clean.

### 9. Plan mode
A read-only investigation phase that produces an approved plan before any edit — enforced by disabling write/exec tools during planning, not by asking the model to behave. AZ v2.3 shipped a built-in Orchestrator; pair it with scope-locking.

### 10. Repo-aware behavior via `AGENTS.md` + Projects
AZ already supports `AGENTS.md` (root + `agent.protocol.projects.agents_md.md`). Use a repo `AGENTS.md` + the `developer` profile + AZ Projects for Cursor-like repo awareness. Keep it short and high-signal (context rot hurts small models faster).

---

## TIER 3 — Cheap prompt/profile tweaks (low effort, real payoff)

- **"If this task is ambiguous, tell me why."** One line in the system prompt. A Reddit commenter: heads off endless loops by giving the model an out when it gets stuck. Cheap and effective.
- **MANDATORY + gated, not buried in a list.** The operator measured it: same rule buried in a list = 0/5; marked MANDATORY and used to gate output = 8/8.
- **"You MUST use search"** — explicit directives. Qwen ignores tooling unless directly ordered ("a full run with 0 search calls against an explicit directive").
- **Small, precise, scope-locked tasks.** Verbosity control: keep full HTML pages, verbose test/lint output *out* of context — trim tool output.
- **Consider the `tiny-local` profile** as a starting point (purpose-built for small/local models) or clone it into `/a0/usr/agents/` and tighten the JSON/communication prompts. Keep prompt overrides under `/a0/usr` (root dirs get overwritten on update).

---

## Serving-level (vLLM) tuning — genuine trade-offs, test on your box

### Sampling parameters
Current chat kwargs: `temperature 0.4, top_p 0.9, presence_penalty 0.1`. Qwen's **official instruct/agentic** recommendation: `temperature 0.7, top_p 0.80, top_k 20, presence_penalty 1.5`.
- **presence_penalty 0.1 → ~1.0–1.5 is the safe, high-value change.** Repetition/looping is Qwen 3.6's signature failure; presence_penalty is the intended lever (Qwen keeps `repetition_penalty` at 1.0/off). Higher pp too aggressively can cause "language mixing," so move to ~1.0 first.
- **temperature 0.4 → 0.7 is a REAL trade-off, not a free win.** Higher temp improves Qwen's quality per the official rec, **but lowers MTP acceptance** (~0.71 @ T=0.4 → ~0.52 @ T=0.8), which slows your token generation. Your 0.4 is actually good for MTP speed. Decide per priority; if you raise temp, re-check tok/s.
- **Never use greedy (temp=0)** — Qwen: "can lead to endless repetitions."
- Add **`top_k: 20`** (currently unset) — part of the official recommended set.

### Quantization: GPTQ-Int4 → consider AWQ-Int4 (Marlin) or Q8
- Qwen officially flags **GPTQ as problematic for Qwen3**; reasoning/tool tasks degrade more than generic benchmarks show. The purpose-built SM75 fork's *"best"* recipe uses **AWQ-Marlin**, not GPTQ.
- Reddit echoes it: *"meaningful quality delta between Q4 and Q8"*; the reasoning-loop behavior was attributed to *"the Q4 quant in action."*
- **Trade-off:** re-quantize/re-download the model; verify it fits 2×22 GB in TP2 with your 256K KV budget. Bigger lift, but potentially the biggest *quality* jump. Q8 may be tight on 44 GB with long context — AWQ-Int4 is the safer swap.

### MTP speculative tokens
You run `num_speculative_tokens: 3`. The general vLLM guidance is MTP sweet spot **K=1–2** (single MTP layer; K≥3 "reliability not guaranteed") — **but** the weicj SM75 fork you're on specifically validated **K=3** on this rig, so this is fork-blessed. If you ever see spec-decode instability, test **K=2** as the fallback. (Note: the "spec=4 → EngineDeadError" claim in one report could not be confirmed against a primary source.)

### Tool parser — already optimal
`qwen3_xml` is the robust choice. If you ever switch, do NOT use `qwen3_coder` (regex, breaks on code args). Harden your client parser regardless: `qwen3_xml` still emits ~6–7% invalid JSON on *parallel* tool calls (spurious braces).

---

## The 5 things to do first (if you do nothing else)

1. **Diff/patch the chat template** against froggeric's — potential source-level fix for the misformat crashes (Tier 1.1).
2. **Raise `presence_penalty` to ~1.0** + add `top_k: 20` — cheap, directly targets the looping (Serving/Sampling).
3. **Add deterministic post-edit hooks** (lint/test/git-diff fed back as evidence) — the highest-leverage *behavioral* change, and the Reddit operator's #1 lesson (Tier 1.3).
4. **Add context7 MCP** (or knowledge-base docs) for current library facts — kills the hallucination spiral (Tier 1.4).
5. **Denylist destructive commands** via a pre-exec hook — stops the "trashes the codebase" failure (Tier 2.5).

Then, as a larger project: **grammar-constrained tool calling** (Tier 1.2) — the biggest single reliability lever, but needs care because guided decoding + MTP + GPTQ-Int4 has a documented crash interaction (#11484); use the llguidance backend.

---

## Key sources
- Reddit field report: r/LocalLLM "I ran Qwen 3.6 locally for 45 days" (`/comments/1uyukbe/`) + comments (froggeric templates, reasoning-off, Q4/Q8, context7, evidence-over-prose).
- Qwen official model card: https://huggingface.co/Qwen/Qwen3.6-27B (sampling, thinking, YaRN, tool parser).
- vLLM: tool_calling, structured_outputs, speculative_decoding/mtp docs; issues #11484 (guided+spec+Int4 crash), #43713 (parallel-call JSON), #44676 (thinking→tool JSON corruption).
- SM75 stack matching this rig: https://github.com/weicj/vllm-2080ti-definitive · https://github.com/weicj/2080Ti-LLM-Toolbox
- Claude Code / Anthropic engineering: writing-tools-for-agents, effective-context-engineering, multi-agent-research-system, building-effective-agents; code.claude.com/docs (hooks, permission-modes, sub-agents, memory).
- Agent Zero: github.com/agent0ai/agent-zero, agent-zero.ai/p/docs (v2.4 circuit breaker, MCP, profiles, extensions).
- Get-current-library-facts: context7 MCP; claude-code-router (route CC harness to local models); QwenLM/qwen-code (Qwen-native agentic CLI).
