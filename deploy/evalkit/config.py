"""
Shared configuration for the qwen38-evalkit.

Every runner script imports this instead of hardcoding endpoints, so there is
exactly one place that says which port we talk to.
"""

# NEVER point this at :8000 -- that's the capacity gateway (vllm-keepalive-shim),
# which mutates/reroutes requests (local-first, DeepSeek overflow, dynamic
# passthrough, a big_prompt guard, etc). We need the raw, unmutated engine
# behavior, so we always test directly against the vLLM OpenAI server.
BASE_URL = "http://localhost:8001"
MODEL_NAME = "qwen-local"

# Applied to every request by default (item/profile may override). Qwen3's
# chat template emits a <think>...</think> block unless this is set; for a
# deterministic, mechanically-scored eval we don't want reasoning tokens
# eating the max_tokens budget or needing to be stripped before checking.
DEFAULT_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

# Crash tolerance (see "Engine facts" in the brief): the engine can Xid31 and
# die on very large prompts. On any HTTP error/timeout we poll this endpoint
# until it comes back (or we give up).
HEALTH_PATH = "/v1/models"
CRASH_POLL_TIMEOUT_S = 300
CRASH_POLL_INTERVAL_S = 3

# systemd unit for the direct-:8001 engine (confirmed via `systemctl list-units`
# and `ps aux` on this box: vllm-qwen27b.service runs
# `python -m vllm.entrypoints.openai.api_server --port 8001 ...`).
JOURNAL_UNIT = "vllm-qwen27b.service"

# The exact guard log line lives at
# vllm/v1/attention/backends/turboquant_attn.py: "TQ continuation dequant
# block_table OOB (pre-launch): layer=... -- clamping to avoid MMU fault".
# We grep case-insensitively for the substring below, plus general Xid/MMU
# fault chatter in the kernel ring buffer, as a broader safety net.
JOURNAL_GOLD_PATTERNS = [r"block_table OOB", r"\bXid\b", r"MMU fault"]

# Any item whose prompt is expected to exceed this many tokens gets a
# journal capture attempt after it runs, win or lose (crash or clean), since
# a *survived* near-miss (the guard firing without a crash) is exactly the
# "gold" event described in the brief and would otherwise leave no trace in
# our own results directory.
LARGE_PROMPT_TOKEN_THRESHOLD = 29000

# Approximation mandated by the brief for building long_ctx corpora from
# source text without pulling in a tokenizer dependency.
CHARS_PER_TOKEN = 3.3

VLLM_SOURCE_DIR = "/home/kevin/Desktop/vLLM-2080Ti-Definitive/vllm"
