---
license: apache-2.0
tags:
  - jinja
  - chat-template
  - qwen
  - qwen3.5
  - qwen3.6
  - lm-studio
  - mlx
  - llama.cpp
  - tool-calling
  - thinking
---

# Fixed jinja chat templates for Qwen 3.5 & 3.6 (v15)

> **2026-05-13 Update (v15):** Three-part fix for agentic tool-loop failures. (1) **Two-tier error escalation:** replaced the brittle backwards-lookahead error detection with a fully forward-tracking `last_tool_failed` + `consecutive_failures` counter. On the first error the generation prompt is pre-seeded with a correction directive inside `<think>`; on the 2nd+ consecutive error the think block is bypassed and an out-of-band directive forces an immediate corrected action. (2) **Length-gated detection:** error signals are only read from short tool responses (< 500 chars), preventing false positives when reading code files containing `error`, `exception`, etc. in legitimate content. (3) **Static system prompt:** tool instructions are now fully unconditional, permanently eliminating the KV cache invalidation vector introduced in v14.
>
> **2026-05-12 Update (v14):** Cured tool amnesia loops and post-tool overthinking friction! Implemented **Smart Loop Preservation** to dynamically scan subsequent tool returns for error markers and conditionally preserve historical reasoning context during active tool failures. Broadened the system instruction scope to define `<think>` as a dual-purpose planning **or synthesis** space, completely eliminating indecisiveness post-tool retrieval.
>
> **2026-05-11 Update (v13):** Radical simplification and compatibility overhaul! Reverted tool schemas and assistant output formatting to standard JSON to natively fix downstream MCP parser crashes and C++ implicit enum coercion bugs. Removed the `ns_scan` history loop to permanently fix KV cache invalidation mid-conversation. Replaced global string replacement for hallucinated tags with a C++ safe, localized array-slicing method to prevent data-corruption on user code blocks.
>
> **2026-05-10 Update (v12):** Fixed agent stalls, parameter data-loss, and hallucination bugs! Restored dynamic tool instructions and the `<IMPORTANT>` formatting reminder block to stop grammar parser crashes.
>
> **2026-05-10 Update (v11):** Fixed agent looping and overthinking! Re-implemented `preserve_thinking` kwarg to properly strip reasoning blocks from history by default, and restored the reasoning bypass (`<think>\n\n</think>\n\n`).

These are drop-in Jinja templates that fix rendering errors, token waste, and missing features in the official Qwen chat templates. 

They are tested to work across LM Studio, llama.cpp, vLLM, MLX, oMLX, and any engine that supports HuggingFace Jinja templates.

---

## Why you need this
The official Qwen templates contain restrictions and Python-specific Jinja logic that break usage on many inference engines and agent frameworks. 

Here are the critical issues this template fixes:

| Problem | Impact | Fix |
|---|---|---|
| **1. Tool calls fail on C++ engines** | The `\|items` filter doesn't exist in `minijinja` (LM Studio, llama.cpp, MLX). Tool calls instantly crash the template. | Rewritten for strict C++ engine compatibility, natively dumping JSON schemas safely. |
| **2. Mid-conversation system crash** | Frameworks injecting mid-conversation steering instructions trigger a hard crash. | Native, chronological rendering for system messages anywhere. |
| **3. `developer` role rejected** | Modern APIs send the developer role; the official template rejects it. | Added full support for `"developer"`. |
| **4. Agentic retry stall & reasoning spiral** | Model correctly diagnoses a tool error in `<think>` but repeatedly emits the identical failing `<tool_call>`. At long context (60k+ tokens), the reasoning block degenerates into a 2000+ token repetition loop. | Two-tier escalation: (1) first error pre-seeds `<think>` with a correction directive; (2) on 2nd+ consecutive error, bypasses thinking entirely and injects an urgent out-of-band directive, making the degenerate loop physically impossible. |
| **5. Post-Tool Indecisive Overthinking** | Forced `<think>` block prefilling combined with narrow instructions causes the model to panic and debate internal prompt rules after fetching tool data. | Refactored instructions to define `<think>` as a dual-purpose space for planning **or synthesis**. |
| **6. Whitespace tag hallucinations** | Model hallucinates invalid boundaries (e.g., `</ think>`), swallowing conversational text. | C++ safe array-slicing isolates the reasoning block without corrupting user code snippets. |
| **7. No-user-query crash** | `raise_exception` crashes agentic loops, system-only contexts, or `/reset` flows. | Removed backwards history scanning entirely. |
| **8. Unclosed thinking before tool call** | Model calls a tool without closing its reasoning, bleeding XML tags into tool parsers. | Auto-injects closing tags before tool boundaries securely using array slicing. |
| **9. Thinking tool_call hallucination** | Model places `<tool_call>` inside `<think>` block because prompt forces `<think>\n` before a strict tool instruction. | Hoists system toggle to inject `<think>` natively into tool instructions. |
| **10. MCP Tool parsing crashes** | Downstream coding agents crash because tool parameters contain unescaped newlines inside custom XML wrappers. | Restored 100% standard JSON formatted tool calls (`{"name": "...", "arguments": {...}}`) natively. |
| **11. Cache invalidation on llama.cpp** | Mutating the initial system prompt based on future user toggles or thinking state breaks the prefix KV cache. | System prompt tool instructions are now fully unconditional and static. Thinking state is controlled exclusively via the generation prompt bypass. |
| **12. Reasoning bypass hallucinations** | When thinking is disabled, Qwen models inherently hallucinate reasoning tags anyway. | Injects an empty closed `<think>\n\n</think>\n\n` block to successfully force reasoning bypass. |
| **13. Jinja C++ crashes (UndefinedValue)** | Python negative indexing `[-1]` or implicit enum coercions crash on `minijinja`. | Replaced all brittle logic with native JSON iteration and safe Jinja strings. |
| **14. Empty thinking blocks spam** | Every past turn gets wrapped in empty `<think></think>` tags, wasting context and breaking caching. | Strictly skips empty blocks unconditionally. |

---

## Quick install

Choose your environment and update the template:

### LM Studio
1. Open your Qwen model in the right-side panel.
2. Scroll down to **Prompt Template**.
3. Replace the template with the contents of `qwen3.5/chat_template.jinja` or `qwen3.6/chat_template.jinja`.
4. Click **Save**.

### llama.cpp / koboldcpp
```bash
--jinja --chat-template-file qwen3.6/chat_template.jinja
```

### vLLM / TextGen
Replace the `"chat_template"` string in your `tokenizer_config.json` with the raw file contents.

### oMLX
Overwrite `chat_template.jinja` in your local model directory. Load with `--jinja`. Remove any `chat_template_kwargs` overrides because the template handles everything internally.

---

## Which file do I use?

| Template File | Supported Models |
|------|-----------| 
| [`qwen3.5/chat_template.jinja`](qwen3.5/chat_template.jinja) | Qwen3.5-35B-A3B, Qwen3.5-32B, Qwen3.5-14B, and all Qwen 3.5 variants. |
| [`qwen3.6/chat_template.jinja`](qwen3.6/chat_template.jinja) | Qwen3.6-27B, Qwen3.6-35B-A3B, and all Qwen 3.6 variants. |

> **Note:** The 3.6 template is a superset. It additionally handles `preserve_thinking`, `</thinking>` hallucination recovery, and interrupted thought streams. If you are on 3.6, always use the 3.6 file.

---

## The thinking toggle
You can control the model reasoning behavior. Insert `<|think_on|>` or `<|think_off|>` anywhere in your system or user prompt. 

The template natively intercepts the tag, removes it from the final context so the model never sees it, and flips the reasoning mode instantly.

**Fast answer, no reasoning:**
```text
System: You are a coding assistant. <|think_off|>
User: What's 2+2?
```

**Deep reasoning:**
```text
System: You are a coding assistant. <|think_on|>
User: Implement a red-black tree in Rust.
```
*(The tag syntax uses Qwen's control-token delimiters to guarantee it will never collide with legitimate text or file paths, unlike earlier community templates that used `/think`)*

---

## Preserving past thoughts

By default, Qwen models "forget" their previous `<think>` blocks in the chat history to prevent repetitive looping and save context tokens. 
If you are running an agentic workflow where the model *needs* to reference its past reasoning, you can enable the `preserve_thinking` flag in your engine's template kwargs:

```json
{
  "preserve_thinking": true
}
```
*(If your engine does not support passing kwargs, the template will default to standard Qwen behavior and strip past thoughts).*

---

## Pre-installed models

If you are using one of the following models, you already have an older version of this template installed.

- [froggeric/Qwen3.6-27B-MLX-8bit](https://huggingface.co/froggeric/Qwen3.6-27B-MLX-8bit)
- [froggeric/Qwen3.6-27B-MLX-4bit](https://huggingface.co/froggeric/Qwen3.6-27B-MLX-4bit)
- [froggeric/Qwen3.5-35B-A3B-Uncensored-FernflowerAI-MLX-8bit](https://huggingface.co/froggeric/Qwen3.5-35B-A3B-Uncensored-FernflowerAI-MLX-8bit)
- [froggeric/Qwen3.5-35B-A3B-Uncensored-FernflowerAI-MLX-4bit](https://huggingface.co/froggeric/Qwen3.5-35B-A3B-Uncensored-FernflowerAI-MLX-4bit)
- [froggeric/Qwen3.6-27B-Uncensored-Heretic-v2-MLX-8bit](https://huggingface.co/froggeric/Qwen3.6-27B-Uncensored-Heretic-v2-MLX-8bit)
- [froggeric/Qwen3.6-27B-Uncensored-Heretic-v2-MLX-6bit](https://huggingface.co/froggeric/Qwen3.6-27B-Uncensored-Heretic-v2-MLX-6bit)
- [froggeric/Qwen3.6-27B-Uncensored-Heretic-v2-MLX-4bit](https://huggingface.co/froggeric/Qwen3.6-27B-Uncensored-Heretic-v2-MLX-4bit)
- [froggeric/Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-8bit](https://huggingface.co/froggeric/Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-8bit)
- [froggeric/Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-6bit](https://huggingface.co/froggeric/Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-6bit)
- [froggeric/Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-4bit](https://huggingface.co/froggeric/Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-4bit)

---

<details>
<summary>Technical Details of the Critical Fixes</summary>

### 1. Two-Tier Agentic Error Escalation & Length-Gated Detection
When a tool call fails validation, the model's `<think>` block correctly diagnoses the problem. However, because the generation prompt was always identical (`<|im_start|>assistant\n<think>\n`), the model's attention was biased towards the cached token sequence for the previous (failing) tool call. At long context lengths (60k+ tokens), this compounds into a catastrophic **degenerate reasoning spiral** — the model loops inside `<think>` generating thousands of repetitive tokens without ever producing a corrected action.

A naive keyword-based error detector (`'error' in content`) also causes false positives when reading code files: JavaScript with `throw new Error(...)`, `catch(e)` blocks, or any file mentioning error-handling would incorrectly trigger the correction directive on a perfectly successful read, confusing the model into thinking its call failed.

v15 fixes this with two mechanisms:

**Detection:** Error signals are only read from **short** tool responses (`content | length < 500`). Framework error messages are always compact (30–300 chars). Any response large enough to contain file content is never flagged as an error, regardless of keywords present.

**Escalation:** A forward-tracked `consecutive_failures` counter drives a two-tier response:
- **Tier 1 (1st error):** Generation prompt prefix changes to `<think>\nThe previous tool call returned an error. I must diagnose the failure and retry with completely corrected arguments:\n`. Seeds reasoning at a different token position, breaking the cached attractor state.
- **Tier 2 (2nd+ consecutive errors):** Think block bypassed entirely (`<think>\n\n</think>\n\n`), preventing the degenerate spiral. An urgent out-of-band directive is injected into the output stream, forcing an immediate corrected tool call without intermediate reasoning.

The `consecutive_failures` counter resets to zero whenever a user message, assistant message, or successful (short, error-free) tool response is rendered.

### 2. Static System Prompt (KV Cache Safety)
v14 contained a conditional inside the `tool_instructions` set block: `{%- if ns_flags.enable_thinking %}`. Because `ns_flags.enable_thinking` is evaluated at system prompt render time (before the message loop), any `<|think_off|>` in a later message would change the value of the flag on the next inference call, producing a different system prompt prefix. Inference engines detect the changed prefix and invalidate the entire KV cache.

v15 removes this conditional entirely. The `<think>` instruction is now always present in the system prompt. Thinking state is controlled exclusively via the generation prompt bypass (`<think>\n\n</think>\n\n`), which is outside the KV-cached prefix and has no cache impact.

### 3. Universal Synthesis Guidance (Curing Overthinking)
By prefilling `<think>\n`, earlier templates forced reasoning post-tool retrieval, but strict guidelines demanding reasoning *only before tool calls* caused internal logic conflicts. This template seamlessly widens the operational scope of `<think>` to cover planning **or synthesis**, completely stabilizing post-tool behavior.

### 4. Tool calls on C++ engines
The official template iterates tool call arguments with `|items`:
`{%- for key, value in tool_call.arguments|items %}`

Python's Jinja supports `|items`. C++ runtimes (LM Studio, llama.cpp, MLX) do not, which produces a rendering error. This template uses native JSON serialization to safely inject tools.

### 5. Mid-conversation system messages crash
The official template hard-crashes if a `system` or `developer` message appears anywhere except the first position. This breaks agentic frameworks (Codex CLI, Docker Agent, oh-my-pi, OpenCode) that inject steering instructions mid-conversation. The fix natively renders these messages chronologically to preserve LLM recency bias while enforcing strict image-blocking checks.

### 6. Whitespace Tag Hallucination Isolation
Using global `.replace('</ think>', '</think>')` silently corrupts code blocks if the user queries about XML formatting. This template employs an entirely C++ safe array-slicing method (`content.split('<think>')`) to securely extract the reasoning content at strict boundaries without ever modifying user text.

### 7. Auto-close unclosed thinking before tool calls
The model sometimes starts a thinking block and immediately calls a tool without emitting the closing tag. The official template lets the unclosed thinking tag bleed into the tool call. The fixed templates detect this pattern and safely auto-inject the closing tag using standard Jinja `split` operations to guarantee 100% C++ compatibility.

### 8. KV Cache preservation (Immutable System Prompt)
Alteration of the *initial* system prompt's instructions completely drops the LLM KV prefix cache. This template isolates the system prompt entirely, preserving the cache, and relies strictly on generation bypass formatting (`<think>\n\n</think>\n\n`) to toggle thinking mid-conversation.
</details>

<details>
<summary>Comparison: Qwen 3.5 templates</summary>

| Feature | Official | LuffyTheFox | mod-ellary | Pneuny | **This (v15)** |
|---------|----------|-------------|------------|--------|----------------|
| Tool arguments | Fails | Fixed | Missing | Fixed | **Fixed (JSON native)** |
| Agentic Retry Stall & Reasoning Spiral | Stalls | Stalls | Stalls | Stalls | **Two-tier escalation system** |
| Post-Tool Overthinking | Broken | Broken | Broken | Broken | **Universal Synthesis** |
| `developer` role | Missing | Missing | Missing | Missing | **Added** |
| Thinking toggle | None | None | `/think` (system only) | None | **`<\|think_off\|>` anywhere** |
| Empty think in history | Broken | Broken | Tags omitted | Broken | **Pruned dynamically** |
| Mid-conversation system | Crashes | Crashes | Crashes | Crashes | **Fixed** |
| No-user-query crash | Crashes | Crashes | Crashes | Crashes | **Graceful fallback** |
| Auto-close thinking | Not handled | Not handled | Not handled | Not handled | **Engine-safe auto-inject** |
| KV cache stability | Breaks | Breaks | Breaks | Breaks | **Fully immutable prefix** |

</details>

<details>
<summary>Comparison: Qwen 3.6 template</summary>

| Feature | Official | **This (v15)** |
|---------|----------|----------------|
| Tool arguments | Fails (`\|items`) | **Fixed (JSON native)** |
| Agentic Retry Stall & Reasoning Spiral | Stalls | **Two-tier escalation system** |
| Post-Tool Overthinking | Spams/Stalls | **Universal Synthesis** |
| `developer` role | Missing | **Added** |
| Thinking toggle | None | **`<\|think_off\|>` anywhere** |
| Empty think in history | Spams empty blocks | **Pruned dynamically** |
| KV prefix caching | Breaks on dynamic history | **100% stable (Immutable)** |
| Mid-conversation system | Crashes | **Fixed** |
| `</thinking>` hallucination | Fails | **Detected and handled (C++ safe)** |
| Auto-close thinking before tool | Not handled | **Engine-safe auto-inject** |
| vLLM stop parsing | Crashes if thinking disabled | **Fixed natively** |
| Long-context tool adherence | Fails | **Dynamic reinforcement** |

</details>

---

## Authorship

| Role | Author |
|------|--------|
| Original models | Alibaba Cloud (Qwen team) |
| Template fixes | [froggeric](https://huggingface.co/froggeric) |

## License

Apache-2.0, inherited from Qwen.
