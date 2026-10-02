# qwen38-evalkit — Baseline-v0 (prod, 2026-08-16)

**Tag:** `baseline-v0-20260816`
**Engine:** `qwen-local` via vLLM direct on `:8001` (never `:8000`, the capacity gateway)
**Chat template kwargs:** `{"enable_thinking": false}` for all items
**Started:** 2026-08-16T15:56:29Z **Finished:** 2026-08-16T16:09:10Z
**Wall time:** 760.85s (12m 41s)
**Raw results:** `/home/kevin/Desktop/qwen38-evalkit/results/baseline-v0-20260816/` (every item's full request/response saved as `<item_id>.json`; `summary.json` has the machine-readable scorecard)

## Headline

**92 / 100 passed (92%)**

## Per-category

| Category | Pass | Total | Pass rate |
|---|---|---|---|
| tool_call | 20 | 20 | 100% |
| code_exec | 25 | 25 | 100% |
| json_schema | 10 | 10 | 100% |
| instruction | 10 | 10 | 100% |
| loop_probe | 4 | 10 | 40% |
| long_ctx | 15 | 15 | 100% |
| agentic_chain | 8 | 10 | 80% |
| **Total** | **92** | **100** | **92%** |

## Engine stability / crash-tolerance protocol

- Engine health (`GET /v1/models` on `:8001`) checked before the run and spot-checked continuously through the long_ctx block: **HTTP 200 throughout, no interruption**.
- **Zero `engine-crash` items.** No HTTP error, timeout, or dropped connection occurred on any of the 100 items, including all 15 long_ctx items (8k–128k token prompts).
- Per-item wall time scaled as expected with prompt size in the long_ctx block: 8k≈4-5s, 16k≈9-10s, 32k≈18-19s, 64k≈43s, 96k≈64-65s, 128k≈87-88s.
- **Journal sweep (mandated "gold pattern" check):** grepped `journalctl -u vllm-qwen27b.service` and the kernel ring buffer (`journalctl -k`) case-insensitively for `block_table OOB`, `\bXid\b`, `MMU fault` across the full run window (15:56–16:09 UTC). **No matches in either log.** No per-item journal capture fired either (every long_ctx item ≥29k tokens has `journal_captured: false` in its raw result). The evidence directory `/home/kevin/Desktop/xid31-evidence-20260816/` was created in preparation but is empty — there was nothing to capture. This run did not reproduce the racy Xid31 fault or trip the armed guard; the engine survived the full long-context block cleanly.

## The 5 most informative failures

### 1. `loop_probe_008` — model self-truncated a "list 100" task at 92 items
Prompt: *"List exactly 100 distinct chemical element names... Output only the 100 numbered lines: no title, no intro, no closing remarks, no blank lines."*
Model output ends cleanly (not cut off by `max_tokens`; `finish_reason: "stop"`):
```
...
90. Thorium
91. Protactinium
92. Uranium
```
It simply emitted a stop token after item 92 instead of continuing to 100 — a genuine task-completion failure, not a formatting or scorer issue (there are 118 known elements, so it did not run out of valid names).

### 2. `loop_probe_004` — mirrored-repetition degeneration
Prompt: *"List exactly 100 distinct animal names..."* Scorer found 8 duplicate items. The duplicates aren't random — they're a **near-exact mirror** of an earlier stretch of the list re-emitted in reverse order:
- item 20 `gorilla` reappears as item 59; item 21 `chimpanzee` reappears as item 58; item 22 `orangutan` reappears as item 57
- item 82 `mesoplodon`-family whales reappear at 98-100 in reverse (`84 hyperoodon`→`98`, `83 ziphius`→`99`, `82 mesoplodon`→`100`)

This is a classic long-generation attention/repetition failure mode — worth watching across future quant changes since it's exactly the kind of degeneration a KV-cache or quantization regression could make worse.

### 3. `agentic_chain_002` — correct answer, wrong literal format (scorer brittleness, not a model error)
Task: *"What's the delivery ETA for order ORD-9001?"* with `track_package` returning `{"eta": "2026-08-20", "carrier": "UPS"}`.
Model's final answer: *"...The estimated delivery date (ETA) is **August 20, 2026**..."* — factually correct, uses the right tools in the right order, and cites the exact right date, but the check pattern is the literal digit string `2026-08-20`, which never appears verbatim (the model paraphrased into prose). This is flagged as **CHECKER-adversarial-relevant**: it's a false negative caused by an overly literal `answer_check.pattern`, not a model defect. Worth an item-spec fix (accept both date formats) rather than a model regression.

### 4. `agentic_chain_008` — defensible-but-wrong tool selection
Task: *"What is 500 USD worth in GBP right now?"* Expected tool: `get_exchange_rate(base="USD", quote="GBP")`. Model instead called:
```json
{"name": "convert_currency", "arguments": {"amount": 500, "from_currency": "USD", "to_currency": "GBP"}}
```
Both tools were offered; `convert_currency` is arguably the more directly responsive choice for an amount-conversion question, but the item's expected-call check requires `get_exchange_rate` specifically, so the chain aborts after step 0 (no tool_call parses as a match) and the remaining 2 steps score as cascading failures. Genuine model/spec mismatch worth a second look — either the model's tool judgment or the item's expected-tool choice.

### 5. `loop_probe_002` — single stray duplicate, otherwise perfect
Prompt: *"List exactly 100 distinct English words related to space and astronomy..."* Only 1 duplicate out of 100: `pulsar` appears at both line 4 and line 79, everywhere else distinct, in order, no stray text. This is the "near-miss" end of the same degeneration spectrum as #2 above but with no visible mirroring structure — a single isolated repeat rather than a block-repeat pattern, suggesting the failure mode isn't one single mechanism.

## Notes on scope / caveats

- This run only exercises the harness as currently built (100 items across 7 categories: `tool_call`(20), `code_exec`(25), `json_schema`(10), `instruction`(10), `loop_probe`(10), `long_ctx`(15), `agentic_chain`(10)).
- All `loop_probe` failures are duplicate-content or under-count issues; none were disqualified for stray commentary/extra lines (`no_extra_ok` was `True` on every failure) or wrong `finish_reason`.
- `code_exec` scoring is hardened against hardcoded-lookup-table solutions (server-side harness, never shown to the model) — all 25/25 passed the *real* hidden tests, not a memorized answer key.
- Raw per-item JSON (full request payload + full raw response + score breakdown) is preserved for every one of the 100 items at `/home/kevin/Desktop/qwen38-evalkit/results/baseline-v0-20260816/<item_id>.json` for independent re-scoring via `score_only.py`.
