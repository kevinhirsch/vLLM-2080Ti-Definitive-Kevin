# deploy/bench — Qwen 3.8 qualification harness

Scripts that decide config changes with evidence instead of vibes. All of them
talk to the ENGINE (`:8001`) directly, not the gateway, so shim budgets and
guards don't shape the numbers. stdlib + `requests` only.

| script | what it decides |
|---|---|
| `bench_equivalence.py` | Is MTP lossless vs the MTP-off reference? (record/replay, greedy) |
| `bench_mtp_requal.py` | Does the 57-64K garble band stay clean under MTP? (context ladder) |
| `bench_decode.py` | The vLLM-Benchmarks.md reference matrix, as paste-ready table rows |
| `bench_toolcalls.sh` | Tool-calling quality: auto x20, named tool_choice, code-heavy args |
| `bench_lib.py` | Shared: streaming TTFT/decode split, /metrics acceptance scrape, garble heuristics, NRestarts |
| `RESULTS-TEMPLATE.md` | Fill-in sheet; completed runs get appended to vLLM-Benchmarks.md |

Order and pass bars: `deploy/docs/QWEN-3.8-MTP-REQUALIFICATION.md`.
