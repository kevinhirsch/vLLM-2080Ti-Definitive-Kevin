# Model Profile Routes

This document records the evidence rules behind the deployment profiles. The
current profile catalog, meanings, and measured throughput live in
[Profile Guide](../profiles/README.md).

`Profile` is the relative `.env` path selected by `launcher.sh`; the concrete
checkpoint is selected separately with `MODEL_DIR`.

## Evidence Rule

Full pass means a real request returns HTTP 200, finishes the stream, and the
Chinese quality smoke has no repetition collapse, broken output, garbling, or
clear wrong-response behavior.

A memory plateau only proves lower capacity risk. If quality smoke fails, the
route is not promoted as a profile even when capacity or synthetic throughput
looks usable.

Load-only, READY-only, health-only, small-window smoke, and empty-stream results
are not capacity evidence.

## KV Positioning

- FP16/default KV is the quality route.
- INT8 KV is the capacity / balance route; currently shipped only as
  `normal` / piecewise profiles.
- TQK8V4 is the TurboQuant compression route; currently shipped only for
  quality-passed `fast` profiles.
- TQ4NC had capacity experiments, but is not used in the current shipped
  profiles.

## Notes

- Profiles are organized as `profiles/<model>/<mode>/<weight>/<route>.env`.
- `normal` is the current recommended production route. `fast` keeps only
  high-performance routes that passed quality smoke. `safe` is the launcher
  eager fallback mode, not the current shipped profile directory.
- The same dual-2080-Ti runtime also validates Qwen3.x 35B MoE lanes. The
  shipped preset set now covers 256K `normal` and `aggressive` noMTP
  text-only lanes, 136K `normal` and `aggressive` noMTP text+image lanes, and
  a 178K `fast` MTP3 speed preset.
- FP8 + FP16KV `normal` is formally 256K and passes a long-prompt smoke at
  `262016/128`.
- FP8 + FP16KV `aggressive` is also validated at 256K. Its recorded throughput
  stays on the `4096/128` synthetic lane, while the near-full `262016/128`
  smoke is treated as a capacity proof because stream chunk coalescing can
  overstate long-run decode speed.
- FP8 + FP16KV text+image is validated at 136K in both `normal` and
  `aggressive`. Both passed `138240/128`; `139008/64` also passed at the edge,
  while `139136/32` exceeds the configured `139264` limit.
- FP8 + FP16KV `fast` is currently validated at 178K and passes a long-prompt
  smoke at `182144/128`.
- FP8 + TQK8V4 is validated at 256K for text-only and 240K for text-image.
  The image route uses GPU util 0.96.
- fast + INT8KV is not kept: capacity or synthetic throughput may pass, but
  Chinese quality smoke showed repeated or broken output.
- Legacy `fast` + INT8KV compatibility issues should be reported and fixed, but
  those fixes do not promote the route back into the shipped fast catalog.
- Throughput background is kept in
  [Qwen3.6 KV Throughput Sweep](qwen36-kv-throughput-sweep.md) and
  [MTP Task Sensitivity](mtp-task-sensitivity.md).
