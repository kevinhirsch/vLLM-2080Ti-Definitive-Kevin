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
  - vllm
  - tool-calling
  - thinking
---

# Fixed jinja chat templates for Qwen 3.5 & 3.6 (v17)

<details open>
<summary><b>Update History & Changelog (v17)</b></summary>

> **2026-05-15 Update (v17):** Major architecture overhaul resolving edge cases in agentic tooling and KV Cache. (1) **Unified Template:** Consolidated Qwen 3.5 and Qwen 3.6 into a single `chat_template.jinja` file that handles all variants seamlessly. (2) **Fixed "Mutually Exclusive" Stopping Bug:** Changed the history-pruning logic from wiping the entire turn to safely array-slicing out just the raw tool tags (`content.split('<tool_call>')[0]`). This preserves the conversational text in the history, which cures the bug where the model would artificially abort its turn (output `<|im_end|>`) when it wanted to talk and use a tool simultaneously. (3) **100% KV Cache Hit Rate Restoration:** Fully normalized internal whitespace logic (`\n\n` -> `\n`) around think blocks and tool calls to exactly match the model's native autoregressive generation spacing. This perfectly synchronizes the template's rendered history with the cached generated tokens, completely eliminating the severe cache invalidation and full-prompt re-processing issues present in v16.

</details>

<details>
<summary><b>Update History & Changelog (v11-v16)</b></summary>

> **2026-05-14 Update (v16):** Four-part fix addressing community-reported regressions. (1) **Native XML tool format:** reverted from JSON back to the native `<function=name>` / `<parameter=x>` format the model was trained on, restoring full compatibility with vLLM's `qwen3_coder` parser and all inference engines that implement the Qwen tool protocol. (2) **`--reasoning off` respected in error paths:** when thinking is disabled (`enable_thinking=false` / `--reasoning off`), the error escalation directives are now injected as plain text without opening any `<think>` block, preventing degenerate prompts in no-reasoning sessions. (3) **Smarter false-positive detection:** short shell command results (starting with `$ `) and search results with timing footers (`Took X.Xs`) are now correctly excluded from error detection, preventing tool-retry loops when commands succeed but their output happens to contain the word `error`. (4) **`consecutive_failures` counter no longer resets on assistant messages**, allowing Tier 2 escalation to actually fire across multi-turn tool retry chains.
>
> **2026-05-13 Update (v15):** Three-part fix for agentic tool-loop failures. (1) **Two-tier error escalation:** replaced the brittle backwards-lookahead error detection with a fully forward-tracking `last_tool_failed` + `consecutive_failures` counter. On the first error the generation prompt is pre-seeded with a correction directive inside `<think>`; on the 2nd+ consecutive error the think block is bypassed and an out-of-band directive forces an immediate corrected action. (2) **Length-gated detection:** error signals are only read from short tool responses (< 500 chars), preventing false positives when reading code files containing `error`, `exception`, etc. in legitimate content. (3) **Static system prompt:** tool instructions are now fully unconditional, permanently eliminating the KV cache invalidation vector introduced in v14.
>
> **2026-05-12 Update (v14):** Cured tool amnesia loops and post-tool overthinking friction! Implemented **Smart Loop Preservation** to dynamically scan subsequent tool returns for error markers and conditionally preserve historical reasoning context during active tool failures. Broadened the system instruction scope to define `<think>` as a dual-purpose planning **or synthesis** space, completely eliminating indecisiveness post-tool retrieval.
>
> **2026-05-11 Update (v13):** Radical simplification and compatibility overhaul! Reverted tool schemas and assistant output formatting to standard JSON to natively fix downstream MCP parser crashes and C++ implicit enum coercion bugs. Removed the `ns_scan` history loop to permanently fix KV cache invalidation mid-conversation. Replaced global string replacement for hallucinated tags with a C++ safe, localized array-slicing method to prevent data-corruption on user code blocks.
>
> **2026-05-10 Update (v12):** Fixed agent stalls, parameter data-loss, and hallucination bugs! Restored dynamic tool instructions and the `<IMPORTANT>` formatting reminder block to stop grammar parser crashes.
>
> **2026-05-10 Update (v11):** Fixed agent looping and overthinking! Re-implemented `preserve_thinking` kwarg to properly strip reasoning blocks from history by default, and restored the reasoning bypass (`<think>\n\n</think>\n\n`).

</details>

This is a drop-in Jinja template that fixes rendering errors, KV cache invalidation, token waste, and missing features in the official Qwen chat templates.

It is tested to work across LM Studio, llama.cpp, vLLM, MLX, oMLX, and any engine that supports HuggingFace Jinja templates.

---

## Why you need this
The official Qwen templates contain restrictions and Python-specific Jinja logic that break usage on many inference engines and agent frameworks.

Here are the critical issues this template fixes:

| Category | Problem | Impact | Fix |
|---|---|---|---|
| **Agentic Loop** | **Premature Stalls (Stopping Bug)** | Model aborts its turn (`<\|im_end\|>`) when trying to combine conversation and a tool call. | Safely prunes history to preserve conversational text without duplicating tool calls. (v17) |
| **Agentic Loop** | **Retry Stall & Reasoning Spiral** | Model correctly diagnoses a tool error but repeatedly emits the identical failing `<tool_call>`. | Two-tier escalation: seeds `<think>` with correction directive; injects urgent out-of-band directive. |
| **Agentic Loop** | **Post-Tool Overthinking** | Forced `<think>` block prefilling causes model to panic and debate internal rules after fetching data. | Broadened instructions to define `<think>` as a dual-purpose space for planning *or synthesis*. |
| **Agentic Loop** | **False-Positive Error Detection** | Short shell commands (`$ grep`) or search outputs containing the word `error` trigger retry loops. | Strict guards exclude shell echoes and timing footers from error detection. |
| **Performance** | **KV Cache Invalidation** | History pruning and whitespace mismatch invalidates KV cache, causing full prompt re-processing every turn. | Strict `\n` whitespace normalization mirrors autoregressive outputs for a 100% KV Cache hit rate. (v17) |
| **Performance** | **Empty Thinking Blocks Spam** | Every past turn gets wrapped in empty `<think></think>` tags, wasting context and breaking caching. | Strictly skips empty blocks unconditionally. |
| **Compatibility** | **Tool Calls Fail on C++ Engines** | The `\|items` filter doesn't exist in `minijinja` (LM Studio, llama.cpp, MLX). | Rewritten for strict C++ engine compatibility using natively supported key iteration. |
| **Compatibility** | **Wrong Tool Call Format** | Qwen-native parsers (like vLLM's `qwen3_coder`) expect XML `<function=name>`. JSON format breaks them. | Restored native XML format while keeping C++ safety. |
| **Compatibility** | **Jinja C++ Crashes** | Python-specific filters (`map`, `first` on strings) crash on `minijinja`. | All filters replaced with universally compatible equivalents. |
| **Stability** | **Mid-Conversation System Crash** | Frameworks injecting mid-conversation steering instructions trigger a hard crash. | Native, chronological rendering for system messages anywhere in the history. |
| **Stability** | **No-User-Query Crash** | `raise_exception` crashes agentic loops or system-only contexts. | Removed backwards history scanning entirely. |
| **Stability** | **Unclosed Thinking Before Tool** | Model calls a tool without closing its reasoning, bleeding XML tags into tool parsers. | Auto-injects closing tags before tool boundaries securely. |
| **Edge Cases** | **`developer` Role Rejected** | Modern APIs send the developer role; the official template rejects it. | Added full support for `"developer"`. |
| **Edge Cases** | **`--reasoning off` Ignored** | When thinking is disabled, tool error escalation still opened a `<think>` block, corrupting the prompt. | Error escalation branches now fully respect `enable_thinking=false`. |
| **Edge Cases** | **Reasoning Bypass Hallucinations** | When thinking is disabled, Qwen models inherently hallucinate reasoning tags anyway. | Injects an empty closed `<think>\n\n</think>\n\n` block to successfully force reasoning bypass. |
| **Edge Cases** | **Whitespace Tag Hallucinations** | Model hallucinates invalid boundaries (e.g., `</ think>`), swallowing conversational text. | C++ safe array-slicing isolates the reasoning block without corrupting user code snippets. |

---

## Quick install

Choose your environment and update the template:

### LM Studio
1. Open your Qwen model in the right-side panel.
2. Scroll down to **Prompt Template**.
3. Replace the template with the contents of `chat_template.jinja`.
4. Click **Save**.

### llama.cpp / koboldcpp
```bash
--jinja --chat-template-file chat_template.jinja
```

### vLLM
Replace the `"chat_template"` string in your `tokenizer_config.json` with the raw file contents. Use the `qwen3_coder` tool parser:
```bash
--tool-call-parser qwen3_coder
```

### oMLX
Overwrite `chat_template.jinja` in your local model directory. Load with `--jinja`. Remove any `chat_template_kwargs` overrides because the template handles everything internally.

---

## Which file do I use?

Both Qwen 3.5 and Qwen 3.6 variants (including 35B, 32B, 27B, and 14B parameters) have been consolidated. You only need the single `chat_template.jinja` file at the root of the repository.

One-line versions (`chat_template_oneline.txt`) are pre-minified for engines that require a single-line template string.

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

### 1. KV Cache Safety & Autoregressive Whitespace Normalization (v17)
Llama.cpp and vLLM utilize prefix KV caching to speed up generation. Previous template versions wiped conversational text from tool-call turns and injected `\n\n` before tool boundaries. These manipulations created a mismatch between the strings generated autoregressively by the model and the history rendered by the template, forcing a complete cache invalidation and full prompt re-processing on every turn. v17 normalizes all whitespace injections to strict single `\n` boundaries and safely prunes tool tags without dropping conversation, achieving a 100% KV cache hit rate in multi-turn loops.

### 2. "Mutually Exclusive" Stopping Bug Resolution (v17)
By preserving conversational text alongside tool calls in the history (rather than aggressively wiping it), the model unlearns the artificial limitation that conversing and using a tool are mutually exclusive. It no longer aborts its turn with an early `<|im_end|>` when explaining its tool actions.

### 3. Native XML Tool Call Format (v16)
The model was trained with the XML-based tool call format used by Qwen3-Coder:
```xml
<tool_call>
<function=tool_name>
<parameter=param_name>
value
</parameter>
</function>
</tool_call>
```
v16 restored this format natively, making it compatible with all parsers while bypassing the `|items` crash by using C++ safe key iteration (`for args_name in tool_call.arguments`).

### 4. Two-Tier Agentic Error Escalation (v15)
When a tool call fails validation repeatedly, the model can enter a degenerate reasoning spiral. v15 leverages a two-tier escalation system driven by a forward-tracked `consecutive_failures` counter:
- **Tier 1 (1st error):** Generation prompt prefix changes to seed reasoning at a different token position, breaking the cached attractor state.
- **Tier 2 (2nd+ consecutive errors):** Think block bypassed entirely. An urgent out-of-band directive forces an immediate corrected action wrapped safely within the user `tool_response` block.

### 5. Universal Synthesis (v14)
Forced `<think>` block prefilling combined with narrow system instructions causes the model to panic and debate its own internal rules after fetching tool data. v14 broadens the system instruction scope to define `<think>` as a dual-purpose space for planning **or synthesis**, completely eliminating indecisiveness post-tool retrieval.

### 6. `enable_thinking=false` in Error Paths (v16)
When users set `--reasoning off` in llama.cpp, error escalation directives are now injected as plain text (no `<think>` wrapper) to prevent creating degenerate prompts the model couldn't resolve in no-reasoning mode.

### 7. Smart False-Positive Detection (v15/v16)
Added strict guards: responses starting with `$ ` (shell echoes) or containing `Took X.Xs` footers (search tools) are excluded from error detection, preventing successful terminal executions from triggering false-positive retry loops when their text contains "error".

### 8. minijinja Compatibility Constraints
Three Python-only Jinja2 filters crash on `minijinja` (the C++ runtime used by llama.cpp, LM Studio, and MLX). They have been completely removed and replaced with universally compatible equivalents:
- `\| items` -> `for key in mapping`
- `map('string')` -> `join('|')`
- `\| first` -> `'$ ' in content`

</details>

<details>
<summary>Comparison Matrix: Official vs Fixed vs Community</summary>

| Feature | Official Qwen Templates | LuffyTheFox | mod-ellary | Pneuny | **This Fixed Template (v17)** |
|---------|----------|-------------|------------|--------|----------------|
| Tool call format | XML (native) | JSON | JSON | JSON | **XML (native, qwen3_coder compatible)** |
| Tool arguments | Fails (`\|items`) | Fixed | Missing | Fixed | **Fixed (C++ safe XML)** |
| Agentic Retry Stall & Reasoning Spiral | Stalls | Stalls | Stalls | Stalls | **Two-tier escalation system** |
| Post-Tool Overthinking | Spams/Stalls | Broken | Broken | Broken | **Universal Synthesis** |
| Premature Stalls (Stopping Bug) | Stalls | Stalls | Stalls | Stalls | **Fixed via conversation preservation (v17)** |
| `--reasoning off` on tool errors | N/A | N/A | N/A | N/A | **Fully respected** |
| Shell/search false positives | N/A | N/A | N/A | N/A | **Guarded** |
| `developer` role | Missing | Missing | Missing | Missing | **Added** |
| Thinking toggle | None | None | `/think` (system only) | None | **`<\|think_off\|>` anywhere** |
| Empty think in history | Spams empty blocks | Broken | Tags omitted | Broken | **Pruned dynamically** |
| KV prefix caching | Breaks on dynamic history | Breaks | Breaks | Breaks | **100% stable (Immutable \n spacing) (v17)** |
| Mid-conversation system | Crashes | Crashes | Crashes | Crashes | **Fixed** |
| No-user-query crash | Crashes | Crashes | Crashes | Crashes | **Graceful fallback** |
| `</thinking>` hallucination | Fails | N/A | N/A | N/A | **Detected and handled (C++ safe)** |
| Auto-close thinking before tool | Not handled | Not handled | Not handled | Not handled | **Engine-safe auto-inject** |

</details>

---

## Running the test suite

```bash
python3 scripts/test_v17.py
```

Tests cover: XML tool format, tool instructions, thinking bypass, `<|think_off|>` / `<|think_on|>`, Tier 1 & 2 escalation, length-gated detection, shell/search false positives, `--reasoning off` + errors, counter reset, historical think stripping, `preserve_thinking`, developer role, mid-conversation system, tool response wrapping, and string argument passthrough.

---

## Authorship

| Role | Author |
|------|--------|
| Original models | Alibaba Cloud (Qwen team) |
| Template fixes | [froggeric](https://huggingface.co/froggeric) |

## License

Apache-2.0, inherited from Qwen.
