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

# Fixed jinja chat templates for Qwen 3.5 & 3.6 (v16)

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

These are drop-in Jinja templates that fix rendering errors, token waste, and missing features in the official Qwen chat templates.

They are tested to work across LM Studio, llama.cpp, vLLM, MLX, oMLX, and any engine that supports HuggingFace Jinja templates.

---

## Why you need this
The official Qwen templates contain restrictions and Python-specific Jinja logic that break usage on many inference engines and agent frameworks.

Here are the critical issues this template fixes:

| Problem | Impact | Fix |
|---|---|---|
| **1. Tool calls fail on C++ engines** | The `\|items` filter doesn't exist in `minijinja` (LM Studio, llama.cpp, MLX). Tool calls instantly crash the template. | Rewritten for strict C++ engine compatibility. |
| **2. Wrong tool call format** | vLLM `qwen3_coder` parser and other Qwen-native parsers expect `<function=name>` XML format. JSON format breaks them. | Restored native XML `<function=name>` / `<parameter=x>` format. |
| **3. Mid-conversation system crash** | Frameworks injecting mid-conversation steering instructions trigger a hard crash. | Native, chronological rendering for system messages anywhere. |
| **4. `developer` role rejected** | Modern APIs send the developer role; the official template rejects it. | Added full support for `"developer"`. |
| **5. Agentic retry stall & reasoning spiral** | Model correctly diagnoses a tool error in `<think>` but repeatedly emits the identical failing `<tool_call>`. At long context (60k+ tokens), the reasoning block degenerates into a 2000+ token repetition loop. | Two-tier escalation: (1) first error pre-seeds `<think>` with a correction directive; (2) on 2nd+ consecutive error, bypasses thinking entirely and injects an urgent out-of-band directive. |
| **6. `--reasoning off` ignored on tool errors** | When thinking is disabled, tool error escalation still opened a `<think>` block, corrupting the generation prompt. | Error escalation branches now fully respect `enable_thinking=false`. |
| **7. False-positive error detection** | Short shell command results (`$ grep …`) and search outputs (`Took 0.1s`) containing `error` in code identifiers trigger incorrect retry loops. | Added guards: responses starting with `$ ` or containing `Took ` footer are never flagged as errors. |
| **8. Post-Tool Indecisive Overthinking** | Forced `<think>` block prefilling combined with narrow instructions causes the model to panic and debate internal prompt rules after fetching tool data. | Refactored instructions to define `<think>` as a dual-purpose space for planning **or synthesis**. |
| **9. Whitespace tag hallucinations** | Model hallucinates invalid boundaries (e.g., `</ think>`), swallowing conversational text. | C++ safe array-slicing isolates the reasoning block without corrupting user code snippets. |
| **10. No-user-query crash** | `raise_exception` crashes agentic loops, system-only contexts, or `/reset` flows. | Removed backwards history scanning entirely. |
| **11. Unclosed thinking before tool call** | Model calls a tool without closing its reasoning, bleeding XML tags into tool parsers. | Auto-injects closing tags before tool boundaries securely using array slicing. |
| **12. Cache invalidation on llama.cpp** | Mutating the initial system prompt based on future user toggles or thinking state breaks the prefix KV cache. | System prompt tool instructions are now fully unconditional and static. |
| **13. Reasoning bypass hallucinations** | When thinking is disabled, Qwen models inherently hallucinate reasoning tags anyway. | Injects an empty closed `<think>\n\n</think>\n\n` block to successfully force reasoning bypass. |
| **14. Jinja C++ crashes** | Python-specific filters (`|items`, `map('string')`, `| first` on strings) crash on `minijinja`. | All filters replaced with universally compatible equivalents. |
| **15. Empty thinking blocks spam** | Every past turn gets wrapped in empty `<think></think>` tags, wasting context and breaking caching. | Strictly skips empty blocks unconditionally. |

---

## Quick install

Choose your environment and update the template:

### LM Studio
1. Open your Qwen model in the right-side panel.
2. Scroll down to **Prompt Template**.
3. Replace the template with the contents of `qwen3.5/chat_template-v16.jinja` or `qwen3.6/chat_template-v16.jinja`.
4. Click **Save**.

### llama.cpp / koboldcpp
```bash
--jinja --chat-template-file qwen3.6/chat_template-v16.jinja
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

| Template File | Supported Models |
|------|-----------|
| [`qwen3.5/chat_template-v16.jinja`](qwen3.5/chat_template-v16.jinja) | Qwen3.5-35B-A3B, Qwen3.5-32B, Qwen3.5-14B, and all Qwen 3.5 variants. |
| [`qwen3.6/chat_template-v16.jinja`](qwen3.6/chat_template-v16.jinja) | Qwen3.6-27B, Qwen3.6-35B-A3B, and all Qwen 3.6 variants. |

One-line versions (`*_oneline.txt`) are pre-minified for engines that require a single-line template string.

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

### 1. Native XML Tool Call Format (v16)
The model was trained with the XML-based tool call format used by Qwen3-Coder:
```
<tool_call>
<function=tool_name>
<parameter=param_name>
value
</parameter>
</function>
</tool_call>
```

v13 changed this to JSON (`{"name": "tool_name", "arguments": {...}}`) to fix MCP parser crashes. However, this broke vLLM's native `qwen3_coder` tool parser and all inference engines that implement the Qwen protocol natively. v16 restores the original XML format, making it compatible with all parsers again while retaining the JSON output for the tool schema presentation (which was always separate).

The key insight: the v12 XML renderer already used `for args_name in tool_call.arguments` (key iteration), which **is** supported by `minijinja`. The `|items` crash never required a JSON fallback — it only required avoiding that specific filter.

### 2. Two-Tier Agentic Error Escalation (v15, refined in v16)
When a tool call fails validation, the model's `<think>` block correctly diagnoses the problem. However, because the generation prompt was always identical, the model's attention was biased towards the cached token sequence for the previous (failing) tool call. At long context lengths (60k+ tokens), this compounds into a catastrophic **degenerate reasoning spiral**.

v15 introduced a two-tier escalation system driven by a forward-tracked `consecutive_failures` counter:
- **Tier 1 (1st error):** Generation prompt prefix changes to seed reasoning at a different token position, breaking the cached attractor state.
- **Tier 2 (2nd+ consecutive errors):** Think block bypassed entirely, preventing the degenerate spiral. An urgent out-of-band directive forces an immediate corrected action.

v16 fixes a bug where `consecutive_failures` was incorrectly reset on every **assistant** message, preventing Tier 2 from ever firing across a multi-turn retry chain. Now only **user messages** and **successful tool responses** reset the counter.

### 3. `enable_thinking=false` in Error Paths (v16)
The original error escalation always emitted `<think>\n...` regardless of whether thinking was enabled. When users set `--reasoning off` in llama.cpp (which passes `enable_thinking=false`), the Tier 1 hint still opened a `<think>` block, creating a degenerate prompt the model couldn't resolve while in no-reasoning mode.

v16 wraps all `<think>` emissions in the error path with `{%- if ns_flags.enable_thinking is not false %}`. When thinking is off:
- Tier 1 injects the correction directive as plain text (no `<think>` wrapper)
- Tier 2 skips the `<think>\n\n</think>\n\n` bypass prefix entirely

### 4. Smart False-Positive Detection (v15/v16)
A naive keyword detector (`'error' in content`) triggers on perfectly successful tool results that happen to contain error-related identifiers in code:
- `$ grep -n "error_message" file.go` → contains `error`
- Search results returning `661: "error_message": ""` → contains `error`

v15 added a length gate (`content | length < 500`). v16 adds two more guards:
- **`'$ ' not in content`**: Shell command echoes always start with `$ ` (dollar-space). This single check correctly identifies and excludes all shell tool output.
- **`'took ' not in content_lower`**: Search tools like `grep`, `ripgrep`, and CLI tools append `Took X.Xs` timing footers. This excludes them regardless of content.

Together these three guards produce zero false positives on all observed real-world tool output patterns.

### 5. Static System Prompt (KV Cache Safety, v15)
Tool instructions are fully unconditional and static, permanently eliminating the KV cache invalidation vector introduced in v14. Thinking state is controlled exclusively via the generation prompt bypass, which is outside the KV-cached prefix.

### 6. minijinja Compatibility Constraints
Three Python-only Jinja2 filters crash on `minijinja` (the C++ runtime used by llama.cpp, LM Studio, and MLX):

| Filter | Python Jinja2 | minijinja | Safe alternative |
|---|---|---|---|
| `\| items` | ✅ | ❌ | `for key in mapping` + `mapping[key]` |
| `map('string')` | ✅ | ❌ | `join('|')` directly |
| `\| first` on strings | ✅ | ❌ | `'$ ' in content` substring check |

All three are avoided in v16. The `| first` filter works for arrays in minijinja but not for strings; the replacement uses a simple `in` operator substring check which is universally supported.

</details>

<details>
<summary>Comparison: Qwen 3.5 templates</summary>

| Feature | Official | LuffyTheFox | mod-ellary | Pneuny | **This (v16)** |
|---------|----------|-------------|------------|--------|----------------|
| Tool call format | XML (native) | JSON | JSON | JSON | **XML (native, qwen3_coder compatible)** |
| Tool arguments | Fails | Fixed | Missing | Fixed | **Fixed (C++ safe XML)** |
| Agentic Retry Stall & Reasoning Spiral | Stalls | Stalls | Stalls | Stalls | **Two-tier escalation system** |
| Post-Tool Overthinking | Broken | Broken | Broken | Broken | **Universal Synthesis** |
| `--reasoning off` on tool errors | N/A | N/A | N/A | N/A | **Fully respected** |
| Shell/search false positives | N/A | N/A | N/A | N/A | **Guarded** |
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

| Feature | Official | **This (v16)** |
|---------|----------|----------------|
| Tool call format | XML (native) | **XML (native, qwen3_coder compatible)** |
| Tool arguments | Fails (`\|items`) | **Fixed (C++ safe XML)** |
| Agentic Retry Stall & Reasoning Spiral | Stalls | **Two-tier escalation system** |
| Post-Tool Overthinking | Spams/Stalls | **Universal Synthesis** |
| `--reasoning off` on tool errors | N/A | **Fully respected** |
| Shell/search false positives | N/A | **Guarded** |
| `developer` role | Missing | **Added** |
| Thinking toggle | None | **`<\|think_off\|>` anywhere** |
| Empty think in history | Spams empty blocks | **Pruned dynamically** |
| KV prefix caching | Breaks on dynamic history | **100% stable (Immutable)** |
| Mid-conversation system | Crashes | **Fixed** |
| `</thinking>` hallucination | Fails | **Detected and handled (C++ safe)** |
| Auto-close thinking before tool | Not handled | **Engine-safe auto-inject** |
| vLLM stop parsing | Crashes if thinking disabled | **Fixed natively** |

</details>

---

## Running the test suite

```bash
python3 scripts/test_v15.py          # test both variants
python3 scripts/test_v15.py qwen3.6  # test one variant
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
