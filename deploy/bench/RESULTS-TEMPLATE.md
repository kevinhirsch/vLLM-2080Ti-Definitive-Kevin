# Qwen 3.8 MTP re-qualification — results (fill in on HNET00)

Run date: ____  Engine: fork v0.1.15 merge @ ____  Model: Qwen3.8-27B-GPTQ-Int4
Variant: serve-qwen38-mtp-requal-fg.sh  MTP_K=__  Template: froggeric v22-official

## G1 losslessness (bench_equivalence.py) — PASS requires the reference prefix (up to 800 whitespace-normalized chars) to match identically per probe

| probe | common prefix (chars) | verdict |
|---|---|---|
| prose | | |
| code | | |
| ctx60k | | |
| toolish | | |

## G2 context ladder (bench_mtp_requal.py)

| ctx | needle | garble flags | acceptance | decode tok/s | TTFT s | NRestarts Δ | verdict |
|---|---|---|---|---|---|---|---|
| 8K | | | | | | | |
| 32K | | | | | | | |
| 57K | | | | | | | |
| 64K | | | | | | | |
| 96K | | | | | | | |
| 128K | | | | | | | |
| 200K | | | | | | | |

## G3 tool calls (bench_toolcalls.sh)

| case | result |
|---|---|
| auto x20 | /20 |
| named tool_choice | |
| code-args x5 | /5 |

## G4 reference matrix (bench_decode.py) — paste MTP-off baseline beside it

_Transcription note: this comparison table is assembled by hand from TWO runs
of `bench_decode.py` (one on the MTP-off engine, one on the requal variant —
each prints `point | TTFT | decode tok/s | aggregate | acceptance | garble |
NRestarts Δ` rows); copy each run's decode column into its column below. The
`acceptance`, `garble`, and `NRestarts Δ` columns are taken from the **MTP-on
run** (the MTP-off baseline has no acceptance and contributes only its decode
column). Run the MTP-on pass with `--min-single-tps 70` so the ship bar is
enforced by exit code, not just by eyeball._

| point | MTP-off tok/s | MTP-on tok/s | Δ | acceptance | garble | NRestarts Δ |
|---|---|---|---|---|---|---|
| 1x7.5K | | | | | | |
| 2x7.5K | | | | | | |
| 2x30K | | | | | | |
| 1x60K | | | | | | |
| 1x200K | | | | | | |

3.6-era anchors: 1x7.5K MTP3+graphs 83.7-105.7 tok/s; MTP-off ≈43.6; MTP3
acceptance 0.679 @T0.4. G4 ship bar: 1x7.5K ≥70 tok/s.

## G5 soak

Start NRestarts: __  End: __  Hours: __  Morning ladder: PASS / FAIL

## Decision

- [ ] ADOPT (drop-in stays; vLLM-Benchmarks.md + Obsidian "Qwen3.8 Live Config" updated)
- [ ] ROLLBACK (drop-in removed; failing JSON + journal filed in incident notes)

MTP_K shipped: __  Notes:
