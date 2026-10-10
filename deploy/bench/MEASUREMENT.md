The measurement runner builds on upstream vLLM and vLLM 2080 Ti Definitive Edition (github.com/weicj). It measures client SSE arrival and mechanically scored synthetic tool contracts. Browser rendering and real tool effects require the separate Hermes fixture gate.

Offline checks and the stable 100-fixture core:

```sh
python3 deploy/evalkit/run_eval.py --validate-fixtures --profile deploy/evalkit/profiles/greedy.json
```

Prepare exact prompts with the current served template. This submits CPU-only `/tokenize` requests, with no generation:

```sh
python3 deploy/bench/measure_ux.py --prepare /tmp/baseline-32768.json --prompt-tokens 32768 --tokenize-base http://127.0.0.1:8001 --engine-pid PID --profiles greedy,halo-non-thinking
```

Run this commissioning command inside an exclusive controller window, with a drained live offline fence and `WINDOW_ENGINE_PID` from its identified boot:

```sh
python3 deploy/bench/measure_ux.py --base http://127.0.0.1:8001 --suite /tmp/baseline-32768.json --case prose_32768_512_greedy --concurrency 1,4 --reps 1 --cache-regime unique-prefix --out /tmp/direct-cold.json
```

Use `shared-prefix` and a fresh output path for warm commissioning. Cold prefix nonces are deterministic across arms; warm measurements explicitly prime the exact per-stream request and record actual cached-token usage. Both exclude the same short kernel warmup. Run paired repeated phases before drawing performance conclusions. Split suites to stay within each controller phase budget.

Gateway commissioning changes the base to `http://127.0.0.1:8000`, with `measurement: {phase: gateway-cold, ttl_s: 900}` on the controller step. It requires `WINDOW_BENCH_TOKEN_FILE` and `WINDOW_BENCH_IDENTITY_FILE`, submits `estate-local`, and verifies every local route/window identity receipt. Token contents are never saved. The token file must be private and owned by the current user.

Live quality uses `run_eval.py --tag FRESH --profile ... --engine-pid PID` inside the same fence. It verifies positive completion usage, finish reason, usable tool IDs/arguments, process/checkpoint/config identity, and unchanged runtime. Offline re-scoring repeats those checks. The standalone supplemental corpus remains optional and has category-prefixed long-context IDs.

First useful candidate means non-whitespace prose or a syntactically complete tool call; the tool fixture separately checks name and arguments. `first_correct_tool_contract_s` requires that contract to pass; it does not prove an executed effect. Gaps are SSE arrival events, potentially containing multiple tokens. Continuous server usage supplies counted generation rate when available; the first-bundle rate estimate is explicitly labeled when it is missing.

Speculative request metrics are captured when the engine enables summary/detailed observability. Cumulative counter deltas are wave diagnostics, with shared/per-request scope labeled. Accepted draft tokens per verify exclude the bonus token; the emitted-token estimate adds one and is labeled conditional on EOS/grammar/partial-step behavior. Native loaded-worker artifact hashes remain a controller frozen-build prerequisite.
