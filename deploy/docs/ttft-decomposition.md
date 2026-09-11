# TTFT decomposition telemetry

Card: `gw-ttft-decomposition-telemetry`. Answers "which part of a slow request was slow" --
admission (waiting for a local lane), the engine (queue + prefill), or decode (token
generation) -- per request, rolled up per traffic class.

## Why three fields, not the four the card asked for

The card's goal named four independently-measured phases: `admission_wait`, `engine_queue`,
`prefill_time`, `decode_time`. Only three are shim-observable:

- **admission_wait** -- the shim's own admission queue. Always known; this is the pre-existing
  `waited` field, duplicated under this name in the JSONL so the AC-named field is directly
  present (see `_telemetry_note_request()`).
- **queue_plus_prefill** -- from the moment the shim hands the request to the engine to the
  first streamed byte back. This is the pre-existing `ttft` field (kept under that name, not
  duplicated). It covers exactly what the card called `engine_queue + prefill_time` combined.
- **decode_time** -- derived: `duration - admission_wait - queue_plus_prefill`, clamped to 0
  against measurement noise. New field.

`engine_queue` and `prefill_time` cannot be split further without an engine change. vLLM tracks
them separately *internally* per request (`vllm/v1/engine/output_processor.py`'s
`do_tracing()`, off `RequestStateStats`: `queued_ts`, `scheduled_ts`, `first_token_ts`), but:

1. Those per-request values are only ever emitted as OpenTelemetry trace spans, gated behind
   `--otlp-traces-endpoint` (off in this deployment, and turning it on needs a trace collector
   plus an engine restart to add the flag).
2. The only *always-on* exposure is via Prometheus histograms
   (`vllm:request_queue_time_seconds`, `vllm:request_prefill_time_seconds`) -- process-wide
   counters, not attributable to one specific request when several are running concurrently
   (which is the whole point of the lane-based admission control this gateway does).
3. Surfacing the per-request values on the OpenAI-compatible HTTP response would need a patch
   to vLLM's serving layer -- an engine code change, which needs a restart to load.

Splitting the combined phase further belongs with the engine-restart bake-off window (the same
window as `gw-prefill-chunk-size-bakeoff` etc.), not this shim-only pass. Filed as a follow-up
rather than forced.

Also streaming-only: `ttft` (and therefore `queue_plus_prefill` and `decode_time`) is `None` for
non-streaming responses -- the shim has no equivalent measurement point on that path (see
`_relay()`'s own TELEMETRY comment). Reported as `null`, never `0`.

## Where it lives

- **Per request**: `requests-*.jsonl` rows carry `admission_wait`, `queue_plus_prefill` (read
  the existing `ttft` field), `decode_time`.
- **Rolled up**: `GET /gateway/history/summary?hours=1` (or any window) returns
  `latency_by_class`: `{interactive: {admission_wait: {p50, p90, n}, queue_plus_prefill: {...},
  decode_time: {...}}, background: {...}}`. Deliberately NOT on the hot `/gateway/stats` poll
  path (1-2s cadence) -- `/gateway/history/summary` already exists for exactly this kind of
  windowed disk-backed rollup (5s response cache, used today for the 24h per-client view), and
  a hot-path percentile would mean re-scanning JSONL every poll for no benefit over the existing
  on-demand/cached endpoint.
- **Pure logic** (independently testable, no server): `decompose_timing()` and
  `_latency_class_pctls()` in `keepalive-shim.py`; `deploy/bin/test_ttft_decomposition.py`.

## Real data

Live snapshot immediately after deploy (last hour, `curl -s
'http://10.0.1.225:8000/gateway/history/summary?hours=1' | python3 -m json.tool`), to confirm
the mechanism is producing real numbers, not just passing synthetic tests. This is necessarily
thin right after deploy -- most rows in the trailing hour predate the field existing, so `n` is
small. A representative full day (`?hours=24`, once one full day has elapsed since deploy) is
the follow-up check, not reproduced here in advance.

```
<<FILLED IN AFTER DEPLOY -- see the live curl output appended below at deploy time>>
```
