#!/usr/bin/env bash
# Tool-calling qualification for the Qwen 3.8 stack.
#
# Layers on tools/tool_call_smoke.py (auto tool choice) and adds the two cases
# that changed recently:
#   1. NAMED tool_choice — upstream v0.1.15 (PR #98) fixed named-tool-choice
#      response handling; pi can use this for schema-forced calls.
#   2. guided decoding with the "guidance" (llguidance) backend — the Tier-1
#      reliability lever from Qwen-AgentZero-Optimization.md. NEVER xgrammar on
#      this box (vLLM #11484: crashes with spec-decode + Int4). The requal serve
#      variant sets the backend server-side.
#
# Run against the engine (:8001). Usage:
#   ./bench_toolcalls.sh [BASE_URL] [MODEL] [N_AUTO]
set -uo pipefail
BASE="${1:-http://127.0.0.1:8001/v1}"
BASE="${BASE%/}"
MODEL="${2:-qwen-local}"
N="${3:-20}"
if ! [ "$N" -gt 0 ] 2>/dev/null; then
  echo "N_AUTO must be a positive integer (got: $N)" >&2; exit 2
fi
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SMOKE="$HERE/../../tools/tool_call_smoke.py"
FAIL=0

echo "== 1/3 auto tool choice x$N (tool_call_smoke.py) =="
AUTO_OK=0
for i in $(seq 1 "$N"); do
  if python3 "$SMOKE" --base-url "$BASE" --model "$MODEL" \
       --require-auto-weather-tool >/dev/null 2>&1; then
    AUTO_OK=$((AUTO_OK + 1))
  fi
done
echo "auto tool choice: $AUTO_OK/$N clean"
# bar: >=95% with ceiling division (19/20 at the default N=20; never a
# zero-success pass at small N)
[ "$AUTO_OK" -ge $(((N * 19 + 19) / 20)) ] || FAIL=1

echo "== 2/3 NAMED tool_choice (v0.1.15 fix) =="
python3 - "$BASE" "$MODEL" <<'PY' || FAIL=1
import json, sys, requests
base, model = sys.argv[1], sys.argv[2]
tools = [{"type": "function", "function": {
    "name": "read_file",
    "description": "Read a file from disk.",
    "parameters": {"type": "object",
                   "properties": {"path": {"type": "string"}},
                   "required": ["path"]}}}]
r = requests.post(f"{base}/chat/completions", timeout=180, json={
    "model": model, "max_tokens": 300, "temperature": 0.7,
    "messages": [{"role": "user",
                  "content": "Show me what's in /etc/hostname"}],
    "tools": tools,
    "tool_choice": {"type": "function", "function": {"name": "read_file"}},
    "chat_template_kwargs": {"enable_thinking": False},
})
r.raise_for_status()
msg = r.json()["choices"][0]["message"]
calls = msg.get("tool_calls") or []
assert calls, f"no tool_calls in response: {json.dumps(msg)[:400]}"
assert calls[0]["function"]["name"] == "read_file", calls[0]
args = json.loads(calls[0]["function"]["arguments"])
assert isinstance(args.get("path"), str) and args["path"], args
print(f"named tool_choice OK: read_file({args['path']!r})")
PY

echo "== 3/3 code-heavy arguments (the qwen3_coder-regex killer case) =="
python3 - "$BASE" "$MODEL" <<'PY' || FAIL=1
import json, sys, requests
base, model = sys.argv[1], sys.argv[2]
tools = [{"type": "function", "function": {
    "name": "write_file",
    "description": "Write content to a file.",
    "parameters": {"type": "object",
                   "properties": {"path": {"type": "string"},
                                  "content": {"type": "string"}},
                   "required": ["path", "content"]}}}]
SNIPPET = 'if (a < b && b > c) { printf("<ok>\\n"); }'
prompt = ("Use write_file to save this exact C snippet to /tmp/cmp.c:\n"
          + SNIPPET + "\nPreserve it byte-for-byte.")
ok = 0
for i in range(5):
    r = requests.post(f"{base}/chat/completions", timeout=180, json={
        "model": model, "max_tokens": 400, "temperature": 0.7,
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools, "tool_choice": "auto",
        "chat_template_kwargs": {"enable_thinking": False},
    })
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    try:
        # strict: right tool, right path, and the snippet byte-for-byte as the
        # decoded content (modulo leading/trailing whitespace — a trailing
        # newline is not a mangling failure; wrapped/mutated text is)
        call = calls[0]["function"]  # empty calls -> IndexError -> not ok
        args = json.loads(call["arguments"])
        if (call["name"] == "write_file"
                and args.get("path") == "/tmp/cmp.c"
                and args.get("content", "").strip() == SNIPPET):
            ok += 1
    except (json.JSONDecodeError, KeyError, IndexError):
        pass
print(f"code-args: {ok}/5 calls carried the exact snippet to the right path")
assert ok >= 4, "code-heavy args are being mangled by the parser"
PY

if [ "$FAIL" -ne 0 ]; then
  echo "TOOL-CALL QUALIFICATION: FAIL"; exit 1
fi
echo "TOOL-CALL QUALIFICATION: PASS"
