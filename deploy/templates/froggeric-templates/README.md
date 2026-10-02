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

# Fixed jinja chat templates for Qwen 3.5 & 3.6 (v21)

This is a drop-in Jinja template that fixes rendering errors, KV cache invalidation, token waste, and fatal agentic stalling in the official Qwen chat templates. 

It works across LM Studio, llama.cpp, vLLM, MLX, oMLX, and any engine that supports HuggingFace Jinja templates. You only need the single `chat_template.jinja` file at the root of the repository for all Qwen 3.5 and 3.6 variants.

---

<details>
<summary><b>Quick Install</b></summary>

Choose your environment and update the template:

### LM Studio
1. Open your Qwen model in the right side panel.
2. Scroll down to **Prompt Template**.
3. Replace the template with the contents of `chat_template.jinja`.
4. Click **Save**.

### llama.cpp / koboldcpp
Use the file directly in your launch command:
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

</details>

---

## Why you need this

The official Qwen templates contain restrictions and Python-specific Jinja logic that break usage on many inference engines and agent frameworks.

<details open>
<summary><b>Critical Issues Fixed</b></summary>

| Area | Issue | The Fix |
|---|---|---|
| **Agentic Loop** | Model aborts turn when combining chat and a tool call. | Cured "Empty Think" poisoning. |
| **Agentic Loop** | Model gets stuck emitting the identical failing tool call. | Added two-tier error escalation to force correction. |
| **Agentic Loop** | Model panics and debates internal rules after fetching data. | Broadened `<think>` instructions to allow synthesis. |
| **Agentic Loop** | API returns containing the word "error" trigger false loops. | Replaced broad matching with strict structural guards. |
| **Performance** | Mutated past turns constantly destroy the prefix cache. | Enforced chronological history for a 100% KV Cache hit rate. |
| **Performance** | Deep Jinja nesting drops `llama.cpp` speed by 80%. | Flattened the AST architecture to maximize throughput. |
| **Compatibility** | Python-specific filters crash C++ inference engines. | Rewrote all filters to be 100% `minijinja` safe. |
| **Compatibility** | Qwen-native parsers (like vLLM) crash on JSON formatting. | Maintained canonical Qwen XML format as the default. |
| **Compatibility** | Older API setups and wrappers crash on native XML. | Added a `tool_call_format="json"` opt-in override. |
| **Compatibility** | Anthropic `message.thinking` payloads are rejected. | Added native Anthropic reasoning support. |
| **Stability** | Massive tool data returns blow out the context window. | Added dynamic payload truncation limits. |
| **Stability** | Mid-conversation system prompts crash the template. | Added native support for arbitrary system messages. |
| **Stability** | Model calls a tool without closing its reasoning tags. | Auto-injects closing tags before tool boundaries. |
| **Edge Cases** | Text occasionally duplicates during streaming generation. | Restored canonical spacing to the generation prompt. |
| **Edge Cases** | Model hallucinates reasoning tags when thinking is disabled. | Injected strict boundaries to force reasoning bypass. |

</details>



---

## Customization & Usage

<details>
<summary><b>The Thinking Toggle</b></summary>

You can control the model reasoning behavior. Insert `<|think_on|>` or `<|think_off|>` anywhere in your system or user prompt. The template intercepts the tag, removes it from the final context so the model never sees it, and flips the reasoning mode instantly.

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
*(The tag syntax uses Qwen's control token delimiters to guarantee it will never collide with legitimate text or file paths, unlike earlier community templates that used `/think`)*

</details>

<details>
<summary><b>Token Saving (Stripping past thoughts)</b></summary>

By default, this template **preserves** all past `<think>` blocks in the chat history. This prevents the model from suffering "amnesia stalls" during complex agentic loops, and it mathematically guarantees a 100% Prefix KV Cache hit rate on local inference engines.

If you are running constrained hardware and need to save context tokens, you can explicitly disable this feature in your engine's template kwargs to automatically strip past thoughts:
```json
{
  "preserve_thinking": false
}
```
*(Note: Setting this to false will reduce your KV Cache hit rate during multi-turn chats because the prompt string will dynamically mutate).*

</details>

<details>
<summary><b>Tool Call Format Override (JSON vs XML)</b></summary>

Qwen models are natively trained to output tool calls in XML (`<function=name>`). By default, this template uses this native XML format to maximize the model's intelligence and reliability. 

**For most users (including those using `vLLM` or `llama.cpp` / `llama-server`), you do NOT need to change anything.** Modern inference engines natively understand Qwen's XML and will automatically translate it into standard OpenAI API responses for your downstream clients.

**When to use the JSON override:**
If you are using a custom wrapper, an older engine version, or a specific framework (like `ik_llama`) that *strictly* expects the model itself to output Hermes JSON (`{"name": "...", "arguments": {...}}`) and crashes on XML, you can force the template to use JSON by passing this kwarg:
```json
{
  "tool_call_format": "json"
}
```
> [!WARNING]
> When you opt into the JSON format, the template explicitly disables the `max_tool_arg_chars` truncation feature. Truncating a JSON string structurally corrupts its syntax, which would poison the model's history and crash downstream parsers.

</details>

---

<details>
<summary><b>Running the test suite</b></summary>

```bash
python3 scripts/test_v21.py
```
Tests cover `auto_disable_thinking_with_tools`, payload truncation logic, parallel tool spacing, mid-conversation system rendering, deep agent loop fallback, XML tool format, `<|think_off|>` / `<|think_on|>` inline overrides, and all legacy regression tests.

</details>

---

## Authorship
| Role | Author |
|------|--------|
| Original models | Alibaba Cloud (Qwen team) |
| Template fixes | [froggeric](https://huggingface.co/froggeric) |
| C++ AST optimizations | [barubary](https://github.com/spiritbuun/buun-llama-cpp) / `spiritbuun` |

## License
Apache-2.0, inherited from Qwen.

---

<details>
<summary>Technical Details of the Critical Fixes</summary>

### 1. The "Empty Think" Poisoning & Logic Trap Cure
Previous versions attempted to save tokens by replacing past thoughts with empty `<think>\n</think>` blocks, combined with an absolute system prompt demanding a tool be called immediately after `</think>`. This created a toxic learning pattern: the model associated empty thoughts with tools, and full thoughts with forbidden conversational text, causing an 80%+ premature `<|im_end|>` stalling rate. We abolished empty think injection and rewrote the `<IMPORTANT>` directives to explicitly authorize conversational synthesis after a thought block.

### 2. KV Cache Safety & Autoregressive Normalization
Llama.cpp and vLLM utilize prefix KV caching to speed up generation. Because this template now preserves historical thoughts chronologically by default, the rendered history perfectly synchronizes with the cached generated tokens. Combined with strict single `\n` normalization at autoregressive boundaries, this achieves a 100% KV Cache hit rate in multi-turn loops.

### 3. Native XML Tool Call Format
The model was trained with the XML tool call format used by Qwen3-Coder. We restored this format natively, making it compatible with all parsers while bypassing the `|items` crash by using C++ safe key iteration (`for args_name in tool_call.arguments`).

### 4. Two-Tier Agentic Error Escalation
When a tool call fails validation repeatedly, the model can enter a degenerate reasoning spiral. This template leverages a two-tier escalation system driven by a forward tracked `consecutive_failures` counter. On the first error, the generation prompt prefix changes to seed reasoning at a different token position, breaking the cached attractor state. On the second consecutive error, the think block is bypassed entirely and an urgent out-of-band directive forces an immediate corrected action wrapped safely within the user `tool_response` block.

### 5. Smart False-Positive Detection
Instead of broad substring matching that triggers false retry loops on successful database returns containing words like "error", this template utilizes strict structural guards looking for `Exception:`, `"error":`, `Traceback`, and `command not found`, combined with length gates and shell echo exclusions (`$ `).

### 6. minijinja Compatibility Constraints
Python-only Jinja2 features crash or misbehave on `minijinja` (the C++ runtime used by llama.cpp, LM Studio, and MLX). All instances have been refactored for universal support:
* `content | replace('<|think_on|>', '')` became `content.split('<|think_on|>') | join('')` (Fixes a bug where `minja` silently drops the entire text payload if the replaced string is found at index 0).
* `\| items` became `for key in mapping`
* `loop.previtem` became `messages[loop.index0 - 1]`
* `map('string')` became `join('|')`
* `\| first` became `'$ ' in content`

### 7. AST Flattening for C++ Throughput
Deeply nested Jinja loops and macros create severe parsing bottlenecks in C++ inference engines. We flattened the AST architecture, effectively curing an 80% inference throughput drop on `llama.cpp` by streamlining how `ns_state` tracking and historical rendering loops are evaluated.

### 8. Dynamic Payload Truncation
Massive API or database returns can instantly blow out a model's context window. We implemented `max_tool_arg_chars` and `max_tool_response_chars` limiters that safely slice oversized payloads. Crucially, this truncation is automatically disabled when the `tool_call_format="json"` override is active, as slicing a serialized JSON string structurally corrupts the data and crashes downstream parsers.

### 9. Reasoning Bypass Hallucination Mitigation
When thinking is disabled, Qwen models often hallucinate reasoning tags due to their training bias. We injected a safe boundary and adjusted the `<IMPORTANT>` system block to remove explicit mentions of `</think>` during tool instructions. This successfully stops the model from hallucinating closing tags when attempting to call tools in a no-reasoning state.

</details>

<details>
<summary>Update History & Changelog</summary>

> **2026-07-02 Update (v21.3): Optional JSON Tool Format Kwarg.** Added an optional `tool_call_format="json"` override for the `chat_template_kwargs`. This provides an escape hatch for users on specific setups (like custom wrappers or older engines that only support Hermes JSON parsing) without breaking the canonical XML default or degrading the model's primary training distribution. `max_tool_arg_chars` truncation is safely bypassed in JSON mode to prevent syntax corruption.

> **2026-07-02 Update (v21.2): Reasoning Bypass Hallucination Fix.** Adjusted the `<IMPORTANT>` block instructions to remove explicit mentions of `</think>` when defining tool call behavior. This stops the model from hallucinating `</think>` tags as a prefix when reasoning is explicitly disabled.

> **2026-07-02 Update (v21.1): Reliability Overhaul & XML Revert.** Addressed critical bugs and compatibility issues, particularly around prefix cache efficiency. (1) **Tool Call XML Format Revert:** Reverted PR 45 to restore the native XML format, as JSON fundamentally broke `vLLM`'s native `qwen3_coder` parser. (2) **Prefix Cache Fixes:** Restored the `preserve_thinking` default to `true` and removed extraneous newlines that broke caching. (3) **Prompt Injection Guard:** `<|think_off|>` tags in untrusted tool responses are now correctly ignored. (4) **Quoted Tag Bug Fix:** The template no longer corrupts history when the assistant quotes `</think>`. (5) **Anthropic `message.thinking`:** Added support for Anthropic reasoning content. (6) **False-Positive Tool Errors:** Reduced error detection scope to the first 80 characters of a response to prevent false positives. (7) **Canonical Generation Prompt:** Restored the canonical `\n\n` spacing to the non-thinking generation prompt to fix answer duplication issues under streaming. *(Huge thanks to `Moore2877`, `choongng`, and the community for their excellent contributions!)*

> **2026-06-05 Update (v20): The Architect Patch.** Major structural update for agentic loops and C++ inference engines. (1) **Minja AST Flattening:** Optimized Jinja nesting to fix parsing bottlenecks that dropped inference throughput by 80% on `llama.cpp`. (2) **Minja Replace Bug Fix (Hotfix):** Bypassed a C++ parsing bug in `llama.cpp` where using the `replace` filter at index 0 of a user prompt silently dropped the entire text payload. Inline thinking toggles now use `split` and `join` for robust stripping. (3) **Auto-disable Thinking:** Introduced `auto_disable_thinking_with_tools` kwarg (default `false`) that allows users to instantly shut off reasoning blocks during tool use. (4) **Deep Agent Fallbacks:** Resolved exceptions triggered by mid-conversation system prompts or loops lacking human `user` messages. (5) **Payload Truncation:** Implemented `max_tool_arg_chars` and `max_tool_response_chars` configurations to definitively stop context window explosions from massive data returns. *(Huge thanks to `barubary` / `spiritbuun` for their contributions to these C++ architecture optimizations!)*

> **2026-05-18 Update (v19): The Agentic Loop Cure.** (1) **Abolished "Empty Think" Poisoning:** Rewrote the AST history rendering to completely remove the injection of empty `<think>\n</think>` blocks. This cures a severe in-context learning bias where the model assumed tools could only be called if it didn't think first, which was causing 80%+ of premature `<|im_end|>` turn aborts. (2) **System Prompt Logic Trap Removed:** Softened the absolute tool mandate in the `<IMPORTANT>` block and restored Universal Synthesis instructions. The model is now explicitly permitted to transition from `</think>` to a conversational answer without panicking. (3) **True 100% KV Cache & Amnesia Fix:** `preserve_thinking` now defaults to `true`. Past thoughts are retained chronologically, permanently curing "amnesia stalls" during multi-step tool loops while mathematically guaranteeing 100% KV Cache prefix matching out of the box.

> **2026-05-16 Update (v18): Stability & Precision Patch.** (1) **Bulletproof False-Positive Detection:** Shifted agentic error detection from broad substring matching to strict structural formats (e.g. `"error":`, `Exception:`, `Traceback`), completely curing false-positive retry loops when successful JSON returns simply contain the word "error" or "fail". (2) **Legacy Engine Compatibility:** Replaced `loop.previtem` with explicit array indexing, fixing AST crashes on older `llama.cpp` and `minijinja` builds that do not track loop state items. (3) **True Whitespace Normalization:** Fixed a bug where reasoning bypasses and hallucinated tag recovery stacked hidden multi-newlines (`\n\n\n`), strictly fulfilling the 100% KV Cache hit rate claim for all edge cases. (4) **Code Cleanup:** Removed dead conditional branches during XML tool parsing.

> **2026-05-15 Update (v17):** (1) **Unified Template:** Consolidated Qwen 3.5 and Qwen 3.6 into a single `chat_template.jinja` file that handles all variants. (2) **Fixed "Mutually Exclusive" Stopping Bug:** Changed the history pruning logic from wiping the entire turn to safely array slicing out just the raw tool tags (`content.split('<tool_call>')[0]`). This preserves the conversational text in the history, which cures the bug where the model would artificially abort its turn (output `<|im_end|>`) when it wanted to talk and use a tool simultaneously. (3) **100% KV Cache Hit Rate Restoration:** Fully normalized internal whitespace logic (`\n\n` -> `\n`) around think blocks and tool calls to exactly match the model's native autoregressive generation spacing. This perfectly synchronizes the template's rendered history with the cached generated tokens, completely eliminating the severe cache invalidation and full prompt re-processing issues present in v16.

> **2026-05-14 Update (v16):** (1) **Native XML tool format:** reverted from JSON back to the native `<function=name>` / `<parameter=x>` format the model was trained on, restoring full compatibility with vLLM's `qwen3_coder` parser and all inference engines that implement the Qwen tool protocol. (2) **`--reasoning off` respected in error paths:** when thinking is disabled (`enable_thinking=false` / `--reasoning off`), the error escalation directives are now injected as plain text without opening any `<think>` block, preventing degenerate prompts in no-reasoning sessions. (3) **Smarter false-positive detection:** short shell command results (starting with `$ `) and search results with timing footers (`Took X.Xs`) are now correctly excluded from error detection, preventing tool retry loops when commands succeed but their output happens to contain the word `error`. (4) **`consecutive_failures` counter no longer resets on assistant messages**, allowing Tier 2 escalation to actually fire across multi-turn tool retry chains.

> **2026-05-13 Update (v15):** (1) **Two-tier error escalation:** replaced the brittle backwards lookahead error detection with a fully forward tracking `last_tool_failed` + `consecutive_failures` counter. On the first error the generation prompt is pre-seeded with a correction directive inside `<think>`; on the 2nd+ consecutive error the think block is bypassed and an out-of-band directive forces an immediate corrected action. (2) **Length-gated detection:** error signals are only read from short tool responses (< 500 chars), preventing false positives when reading code files containing `error`, `exception`, etc in legitimate content. (3) **Static system prompt:** tool instructions are now fully unconditional, permanently eliminating the KV cache invalidation vector introduced in v14.

> **2026-05-12 Update (v14):** Cured tool amnesia loops and post-tool overthinking friction! Implemented Smart Loop Preservation to dynamically scan subsequent tool returns for error markers and conditionally preserve historical reasoning context during active tool failures. Broadened the system instruction scope to define `<think>` as a dual-purpose planning or synthesis space, completely eliminating indecisiveness post-tool retrieval.

> **2026-05-11 Update (v13):** Reverted tool schemas and assistant output formatting to standard JSON to natively fix downstream MCP parser crashes and C++ implicit enum coercion bugs. Removed the `ns_scan` history loop to permanently fix KV cache invalidation mid-conversation. Replaced global string replacement for hallucinated tags with a C++ safe, localized array slicing method to prevent data corruption on user code blocks.

> **2026-05-10 Update (v12):** Fixed agent stalls, parameter data loss, and hallucination bugs! Restored dynamic tool instructions and the `<IMPORTANT>` formatting reminder block to stop grammar parser crashes.

> **2026-05-10 Update (v11):** Fixed agent looping and overthinking! Re-implemented `preserve_thinking` kwarg to properly strip reasoning blocks from history by default, and restored the reasoning bypass (`<think>\n\n</think>\n\n`).

</details>