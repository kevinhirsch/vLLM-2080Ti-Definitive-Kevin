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

# Fixed jinja chat templates for Qwen 3.5 & 3.6 (v12)

> **2026-05-10 Update (v12):** Fixed agent stalls, parameter data-loss, and hallucination bugs! Restored dynamic tool instructions and the `<IMPORTANT>` formatting reminder block to stop grammar parser crashes. Fixed a massive semantic data-loss bug in v10 by restoring parameter descriptions to the compacted tool schemas. Implemented a robust Pre-Parse Tag Normalization Pipeline to neutralize whitespace tag hallucinations (e.g., `</ think>`), preventing conversational text from being permanently trapped or deleted.
>
> **2026-05-10 Update (v11):** Fixed agent looping and overthinking! Re-implemented `preserve_thinking` kwarg to properly strip reasoning blocks from history by default, and restored the reasoning bypass (`<think>\n\n</think>\n\n`) when `enable_thinking` is false to prevent the model from hallucinating reasoning tags. Also removed the redundant end-of-prompt system message.
> 
> **2026-05-09 Update (v10):** Massive stability and performance overhaul! Halved tool prompt tokens via compaction, eliminated C++ engine `UndefinedValue` crashes by removing brittle string slices, fixed prefix cache invalidation in llama.cpp by ensuring chronological state tracking, resolved vLLM crashes with `<|think_off|>`, and fixed tool parameter omission over long contexts via dynamic instruction reinforcement.
> 
> **2026-05-08 Update (v9):** Fixed 9th bug: Thinking-tool-call hallucination. Refactored system prompt parsing to enable dynamic tool instructions. The template now actively teaches the model how to safely combine `<think>` blocks and `<tool_call>` boundaries.
> 
> **2026-05-07 Update (v8):** Fixed 8th bug: Mid-conversation system messages no longer crash the template. Compatibility restored for agent frameworks (OpenCode, Docker Agent, oh-my-pi). Re-engineered Jinja string parsing for C++ engine stability.

These are drop-in Jinja templates that fix rendering errors, token waste, and missing features in the official Qwen chat templates. 

They are tested to work across LM Studio, llama.cpp, vLLM, MLX, oMLX, and any engine that supports HuggingFace Jinja templates.

---

## Why you need this
The official Qwen templates contain restrictions and Python-specific Jinja logic that break usage on many inference engines and agent frameworks. 

Here are the 14 bugs this template fixes:

| Problem | Impact | Fix |
|---|---|---|
| **1. Tool calls fail on C++ engines** | The `\|items` filter doesn't exist in `minijinja` (LM Studio, llama.cpp, MLX). Tool calls instantly crash the template. | Rewritten for strict C++ engine compatibility. |
| **2. Mid-conversation system crash** | Frameworks injecting mid-conversation steering instructions trigger a hard crash. | Native, chronological rendering for system messages anywhere. |
| **3. `developer` role rejected** | Modern APIs send the developer role; the official template rejects it. | Added full support for `"developer"`. |
| **4. Empty thinking blocks spam** | Every past turn gets wrapped in empty `<think></think>` tags, wasting context and breaking caching. | Strictly skips empty blocks unconditionally. |
| **5. No way to toggle thinking** | The user is restricted to the model defaults. | Intercepts `<\|think_off\|>` and `<\|think_on\|>` tags natively. |
| **6. Whitespace tag hallucinations** | Model hallucinates invalid boundaries (e.g., `</ think>`), swallowing conversational text. | Pre-Parse Tag Normalization Pipeline natively neutralizes variants. |
| **7. No-user-query crash** | `raise_exception` crashes agentic loops, system-only contexts, or `/reset` flows. | Removed backwards history scanning entirely. |
| **8. Unclosed thinking before tool call** | Model calls a tool without closing its reasoning, bleeding XML tags into tool parsers. | Auto-injects closing tags before tool boundaries. |
| **9. Thinking tool_call hallucination** | Model places `<tool_call>` inside `<think>` block because prompt forces `<think>\n` before a strict tool instruction. | Hoists system toggle to inject `<think>` natively into tool instructions. |
| **10. Massive token waste on tools** | The template dumps raw JSON schemas for tools, wasting ~50% of the prompt context and increasing TTFT. | Tools are now compacted into typed one-line signatures while preserving semantic parameter descriptions. |
| **11. Cache invalidation on llama.cpp** | Conditionally hiding past `<think>` blocks based on future messages breaks KV cache. | Refactored history rendering to strictly chronological forward state tracking. |
| **12. Reasoning bypass hallucinations** | When thinking is disabled, Qwen models inherently hallucinate reasoning tags anyway. | Injects an empty closed `<think>\n\n</think>\n\n` block to successfully force reasoning bypass. |
| **13. Jinja C++ crashes (UndefinedValue)** | Python negative indexing `[-1]` when closing tags fails on `minijinja` engines. | Replaced all string slicing with safe Jinja `replace()` operations. |
| **14. Tool format drift & stalls** | Model leaks conversational text between reasoning and tool calls, stalling strict parsers. | Re-engineered dynamic instructions and restored `<IMPORTANT>` formatting reminder to the initial system prompt. |

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
<summary>Technical Details of the 9 Fixes</summary>

### 1. Tool calls on C++ engines
The official template iterates tool call arguments with `|items`:
`{%- for key, value in tool_call.arguments|items %}`

Python's Jinja supports `|items`. C++ runtimes (LM Studio, llama.cpp, MLX) do not, which produces a rendering error. This template uses direct dictionary key lookups instead. It also replaces `is sequence` with `is iterable`, removes Python-only `|safe` wrappers, and handles arguments returned as raw strings.

### 2. Mid-conversation system messages crash
The official template hard-crashes if a `system` or `developer` message appears anywhere except the first position. This breaks agentic frameworks (Codex CLI, Docker Agent, oh-my-pi, OpenCode) that inject steering instructions mid-conversation. The fix natively renders these messages chronologically to preserve LLM recency bias while enforcing strict image-blocking checks.

### 3. `developer` role
The OpenAI-compatible API spec sends `message.role == "developer"` for system-level instructions. The official Qwen template throws an exception. Both templates here accept `"developer"` and map it properly.

### 4. Empty thinking blocks
The official template wraps every past assistant turn in thinking tags, even when empty. When there is no reasoning content, those tags waste context tokens and break prefix caching. The 3.5 template checks `reasoning_content` before emitting. The 3.6 template checks `reasoning_content|trim|length > 0` and ties history visibility to the `<|think_off|>` override.

### 5. `</thinking>` hallucination (Qwen 3.6 only)
The Qwen 3.6 model sometimes generates `</thinking>` instead of the expected `</think>`. The official parser splits on `</think >` only and fails. The 3.6 template detects which closing tag was actually used and splits dynamically. It also handles interrupted generation by rescuing incomplete streams.

### 6. Arguments serialization
The official template serializes argument values with `|tojson` unconditionally, failing when the value is already a string. The fixed templates check the type first. Strings pass through as-is, and everything else goes through `|tojson`.

### 7. Auto-close unclosed thinking before tool calls
The model sometimes starts a thinking block and immediately calls a tool without emitting the closing tag. The official template lets the unclosed thinking tag bleed into the tool call. The fixed templates detect this pattern and safely auto-inject the closing tag using standard Jinja `split` operations to guarantee 100% C++ compatibility.

### 8. No-user-query exception
The official template scans the message list in reverse. If all messages are tool results, or there are no user messages, it fires `raise_exception('No user query found...')` and hard-crashes. The fix replaces the exception with a graceful fallback `{%- set ns.last_query_index = messages|length - 1 %}`, enabling agentic tool-calling chains to function perfectly.

### 9. Thinking tool_call hallucination
The official template appends `<think>\n` to the end of the generation prompt to initiate reasoning. However, its system instructions rigidly demand the model to output *only* `<tool_call>` with no suffix. This contradictory state causes the model to improperly nest its tool call inside the thinking block. This template utilizes a global pre-scan to evaluate the final `enable_thinking` state across the entire conversation history, guaranteeing it can dynamically inject a proper `<think>...</think>` usage example into the tool instructions exactly when reasoning is enabled.
</details>

<details>
<summary>Comparison: Qwen 3.5 templates</summary>

| Feature | Official | LuffyTheFox | mod-ellary | Pneuny | **This (v12)** |
|---------|----------|-------------|------------|--------|----------------|
| Tool arguments | Fails | Fixed | Missing | Fixed | **Fixed** |
| `\|safe` removed | Fails | Fixed | Missing | Fixed | **Fixed** |
| `developer` role | Missing | Missing | Missing | Missing | **Added** |
| Thinking toggle | None | None | `/think` (system only) | None | **`<\|think_off\|>` anywhere** |
| Empty think in history | Broken | Broken | Tags omitted | Broken | **Pruned dynamically** |
| Mid-conversation system | Crashes | Crashes | Crashes | Crashes | **Fixed** |
| No-user-query crash | Crashes | Crashes | Crashes | Crashes | **Graceful fallback** |
| Auto-close thinking | Not handled | Not handled | Not handled | Not handled | **Engine-safe auto-inject** |
| Tool token optimization | None | None | None | None | **~50% reduction** |
| Long-context tool adherence | Fails | Fails | Fails | Fails | **Dynamic reinforcement** |

</details>

<details>
<summary>Comparison: Qwen 3.6 template</summary>

| Feature | Official | **This (v12)** |
|---------|----------|----------------|
| Tool arguments | Fails (`\|items`) | **Fixed** |
| `\|safe` removed | Fails | **Fixed** |
| `developer` role | Missing | **Added** |
| Thinking toggle | None | **`<\|think_off\|>` anywhere** |
| Empty think in history | Spams empty blocks | **Pruned dynamically** |
| KV prefix caching | Breaks on dynamic history | **100% stable (Chronological)** |
| Mid-conversation system | Crashes | **Fixed** |
| `</thinking>` hallucination | Fails | **Detected and handled** |
| Auto-close thinking before tool | Not handled | **Engine-safe auto-inject** |
| vLLM stop parsing | Crashes if thinking disabled | **Fixed natively** |
| Tool token optimization | None | **~50% reduction** |
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
