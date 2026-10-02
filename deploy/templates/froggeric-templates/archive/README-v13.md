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

# Fixed jinja chat templates for Qwen 3.5 & 3.6 (v13)

> **2026-05-11 Update (v13):** Radical simplification and compatibility overhaul! Reverted tool schemas and assistant output formatting to standard JSON to natively fix downstream MCP parser crashes and C++ implicit enum coercion bugs. Removed the `ns_scan` history loop to permanently fix KV cache invalidation mid-conversation. Replaced global string replacement for hallucinated tags with a C++ safe, localized array-slicing method to prevent data-corruption on user code blocks.
>
> **2026-05-10 Update (v12):** Fixed agent stalls, parameter data-loss, and hallucination bugs! Restored dynamic tool instructions and the `<IMPORTANT>` formatting reminder block to stop grammar parser crashes.
>
> **2026-05-10 Update (v11):** Fixed agent looping and overthinking! Re-implemented `preserve_thinking` kwarg to properly strip reasoning blocks from history by default, and restored the reasoning bypass (`<think>\n\n</think>\n\n`).
> 
> **2026-05-09 Update (v10):** Massive stability and performance overhaul! Fixed prefix cache invalidation in llama.cpp by ensuring chronological state tracking, resolved vLLM crashes with `<|think_off|>`.

These are drop-in Jinja templates that fix rendering errors, token waste, and missing features in the official Qwen chat templates. 

They are tested to work across LM Studio, llama.cpp, vLLM, MLX, oMLX, and any engine that supports HuggingFace Jinja templates.

---

## Why you need this
The official Qwen templates contain restrictions and Python-specific Jinja logic that break usage on many inference engines and agent frameworks. 

Here are the 14 bugs this template fixes:

| Problem | Impact | Fix |
|---|---|---|
| **1. Tool calls fail on C++ engines** | The `\|items` filter doesn't exist in `minijinja` (LM Studio, llama.cpp, MLX). Tool calls instantly crash the template. | Rewritten for strict C++ engine compatibility, natively dumping JSON schemas safely. |
| **2. Mid-conversation system crash** | Frameworks injecting mid-conversation steering instructions trigger a hard crash. | Native, chronological rendering for system messages anywhere. |
| **3. `developer` role rejected** | Modern APIs send the developer role; the official template rejects it. | Added full support for `"developer"`. |
| **4. Empty thinking blocks spam** | Every past turn gets wrapped in empty `<think></think>` tags, wasting context and breaking caching. | Strictly skips empty blocks unconditionally. |
| **5. No way to toggle thinking** | The user is restricted to the model defaults. | Intercepts `<\|think_off\|>` and `<\|think_on\|>` tags natively. |
| **6. Whitespace tag hallucinations** | Model hallucinates invalid boundaries (e.g., `</ think>`), swallowing conversational text. | C++ safe array-slicing isolates the reasoning block without corrupting user code snippets. |
| **7. No-user-query crash** | `raise_exception` crashes agentic loops, system-only contexts, or `/reset` flows. | Removed backwards history scanning entirely. |
| **8. Unclosed thinking before tool call** | Model calls a tool without closing its reasoning, bleeding XML tags into tool parsers. | Auto-injects closing tags before tool boundaries securely using array slicing. |
| **9. Thinking tool_call hallucination** | Model places `<tool_call>` inside `<think>` block because prompt forces `<think>\n` before a strict tool instruction. | Hoists system toggle to inject `<think>` natively into tool instructions. |
| **10. MCP Tool parsing crashes** | Downstream coding agents crash because tool parameters contain unescaped newlines inside custom XML wrappers. | Restored 100% standard JSON formatted tool calls (`{"name": "...", "arguments": {...}}`) natively. |
| **11. Cache invalidation on llama.cpp** | Mutating the initial system prompt based on future user toggles breaks the prefix KV cache. | Replaced history mutation with structural bypass generation (`<think>\n\n</think>\n\n`), keeping the system prompt 100% immutable. |
| **12. Reasoning bypass hallucinations** | When thinking is disabled, Qwen models inherently hallucinate reasoning tags anyway. | Injects an empty closed `<think>\n\n</think>\n\n` block to successfully force reasoning bypass. |
| **13. Jinja C++ crashes (UndefinedValue)** | Python negative indexing `[-1]` or implicit enum coercions crash on `minijinja`. | Replaced all brittle logic with native JSON iteration and safe Jinja strings. |
| **14. Tool format drift & stalls** | Model leaks conversational text between reasoning and tool calls, stalling strict parsers. | Cleanly refactored dynamic tool instructions block with a standard JSON example. |

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
<summary>Technical Details of the 14 Fixes</summary>

### 1. Tool calls on C++ engines
The official template iterates tool call arguments with `|items`:
`{%- for key, value in tool_call.arguments|items %}`

Python's Jinja supports `|items`. C++ runtimes (LM Studio, llama.cpp, MLX) do not, which produces a rendering error. This template uses native JSON serialization to safely inject tools.

### 2. Mid-conversation system messages crash
The official template hard-crashes if a `system` or `developer` message appears anywhere except the first position. This breaks agentic frameworks (Codex CLI, Docker Agent, oh-my-pi, OpenCode) that inject steering instructions mid-conversation. The fix natively renders these messages chronologically to preserve LLM recency bias while enforcing strict image-blocking checks.

### 3. `developer` role
The OpenAI-compatible API spec sends `message.role == "developer"` for system-level instructions. The official Qwen template throws an exception. Both templates here accept `"developer"` and map it properly.

### 4. Empty thinking blocks
The official template wraps every past assistant turn in thinking tags, even when empty. When there is no reasoning content, those tags waste context tokens and break prefix caching. The 3.5 and 3.6 templates check `reasoning_content|trim|length > 0` and tie history visibility to the `<|think_off|>` override.

### 5. `</thinking>` hallucination
Qwen models sometimes generate `</thinking>` instead of the expected `</think>`. The official parser splits on `</think >` only and fails. This template dynamically detects the exact hallucinated variant used and structurally isolates it safely.

### 6. Whitespace Tag Hallucination Isolation
Using global `.replace('</ think>', '</think>')` silently corrupts code blocks if the user queries about XML formatting. This template employs an entirely C++ safe array-slicing method (`content.split('<think>')`) to securely extract the reasoning content at strict boundaries without ever modifying user text.

### 7. Auto-close unclosed thinking before tool calls
The model sometimes starts a thinking block and immediately calls a tool without emitting the closing tag. The official template lets the unclosed thinking tag bleed into the tool call. The fixed templates detect this pattern and safely auto-inject the closing tag using standard Jinja `split` operations to guarantee 100% C++ compatibility.

### 8. No-user-query exception
The official template scans the message list in reverse. If all messages are tool results, or there are no user messages, it fires `raise_exception('No user query found...')` and hard-crashes. The fix replaces the exception with a graceful fallback `{%- set ns.last_query_index = messages|length - 1 %}`, enabling agentic tool-calling chains to function perfectly.

### 9. Thinking tool_call hallucination
The official template appends `<think>\n` to the end of the generation prompt to initiate reasoning. However, its system instructions rigidly demand the model to output *only* `<tool_call>` with no suffix. This contradictory state causes the model to improperly nest its tool call inside the thinking block. This template utilizes a cleanly formatted instruction block that properly guides the model.

### 10. Downstream MCP Tool parsing crashes
Using a custom XML formatting for arguments (`<parameter=key>value</parameter>`) breaks downstream standard JSON parsers when values contain newlines. This template natively supports and enforces standard JSON formatting (`{"name": "...", "arguments": {...}}`) exactly as Llama.cpp and MCP frameworks expect.

### 11. KV Cache preservation (Immutable System Prompt)
Dynamically parsing `<|think_off|>` in the conversation history and altering the *initial* system prompt's instructions completely drops the LLM KV prefix cache. This template isolates the system prompt entirely, preserving the cache, and relies strictly on generation bypass formatting (`<think>\n\n</think>\n\n`) to toggle thinking mid-conversation.
</details>

<details>
<summary>Comparison: Qwen 3.5 templates</summary>

| Feature | Official | LuffyTheFox | mod-ellary | Pneuny | **This (v13)** |
|---------|----------|-------------|------------|--------|----------------|
| Tool arguments | Fails | Fixed | Missing | Fixed | **Fixed (JSON native)** |
| `\|safe` removed | Fails | Fixed | Missing | Fixed | **Fixed** |
| `developer` role | Missing | Missing | Missing | Missing | **Added** |
| Thinking toggle | None | None | `/think` (system only) | None | **`<\|think_off\|>` anywhere** |
| Empty think in history | Broken | Broken | Tags omitted | Broken | **Pruned dynamically** |
| Mid-conversation system | Crashes | Crashes | Crashes | Crashes | **Fixed** |
| No-user-query crash | Crashes | Crashes | Crashes | Crashes | **Graceful fallback** |
| Auto-close thinking | Not handled | Not handled | Not handled | Not handled | **Engine-safe auto-inject** |
| Tool token optimization | None | None | None | None | **JSON-native simplification** |
| Long-context tool adherence | Fails | Fails | Fails | Fails | **Dynamic reinforcement** |

</details>

<details>
<summary>Comparison: Qwen 3.6 template</summary>

| Feature | Official | **This (v13)** |
|---------|----------|----------------|
| Tool arguments | Fails (`\|items`) | **Fixed (JSON native)** |
| `\|safe` removed | Fails | **Fixed** |
| `developer` role | Missing | **Added** |
| Thinking toggle | None | **`<\|think_off\|>` anywhere** |
| Empty think in history | Spams empty blocks | **Pruned dynamically** |
| KV prefix caching | Breaks on dynamic history | **100% stable (Immutable)** |
| Mid-conversation system | Crashes | **Fixed** |
| `</thinking>` hallucination | Fails | **Detected and handled (C++ safe)** |
| Auto-close thinking before tool | Not handled | **Engine-safe auto-inject** |
| vLLM stop parsing | Crashes if thinking disabled | **Fixed natively** |
| Tool token optimization | None | **JSON-native simplification** |
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
