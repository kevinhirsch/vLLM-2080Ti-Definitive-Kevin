# Qwen 3.8 MTP re-qualification — winning the decode speed back

_Prepared 2026-08-16 on the `claude/qwen-3.8-27b-tuning-isbc7k` branch, after
the upstream v0.1.15 merge. Companion to the 2026-08-15 campaign commit
(746d8b1) and the Obsidian notes "Qwen3.8 Live Config" / "Qwen3.8 Capacity
Tuning Backlog"._

## Why this exists

The 2026-08-15 campaign shipped Qwen3.8-27B-GPTQ-Int4 with **MTP speculative
decoding OFF** because MTP garbled output and crashed the engine in the
~57-64K context band. That bought clean 255K context at a ~40-60% single-stream
decode cost (3.6 reference: MTP3+cudagraphs 83.7-105.7 tok/s vs ~43.6 no-MTP).

Since then, three fixes landed that together cover **both halves** of the
incident:

| Fix | Where | What it covers |
|---|---|---|
| `40129ea` preserve hybrid Mamba **prefix-cache correctness with MTP** | upstream v0.1.15 (merged here) | The garble. We run prefix caching + MTP + GDN — this exact triple. Agentic (pi) loops re-extend a shared prefix every turn, so corruption compounds with context length — consistent with a ~57-64K onset. |
| `c256ad2` preserve **GDN state slot zero during decode** | upstream v0.1.15 (merged here) | Garble via linear-attention state clobbering during multi-token (spec-verify) decode steps. |
| `746d8b1` clamp negative draft-token ids | this fork (Kevin, 2026-08-15) | The crash: cudagraph replay let `-1` "no token" sentinels reach the embedding lookup → `cudaErrorIllegalAddress`. |

None of this is proven on the box yet — that is what this procedure does.
**MTP stays OFF for clients until every gate below is green.**

## What's in the kit

| File | Purpose |
|---|---|
| `deploy/bin/serve-qwen38-mtp-requal-fg.sh` | Serve variant: live config + MTP (env `MTP_K`, default 3) + llguidance structured outputs |
| `deploy/systemd/vllm-qwen27b.service.d/mtp-requal.conf.example` | Drop-in that activates the variant; delete to roll back |
| `deploy/bench/bench_equivalence.py` | Greedy MTP-on == MTP-off record/replay check (losslessness) |
| `deploy/bench/bench_mtp_requal.py` | Context ladder 8K→200K across the old failure band, garble + acceptance + crash detection |
| `deploy/bench/bench_decode.py` | Reference matrix (same points as vLLM-Benchmarks.md) |
| `deploy/bench/bench_toolcalls.sh` | Auto + named tool_choice + code-heavy args qualification |
| `deploy/bench/RESULTS-TEMPLATE.md` | Fill-in tables for all of the above |

## Procedure (run on HNET00, in this order)

**0. Prereq A — fresh reboot, and know about the Xid31 confounder.** The
2026-08-16 frontier session (branch `frontier-pastnative-20260816`, and
`fix-xid31-guard`) documents an **open, unattributed Xid 31 GPU memory-fault
hunt in the TurboQuant continuation dequant path** — the exact path this
ladder exercises at long context — and observed that the crash floor descends
across successive un-rebooted crash cycles. Therefore:

- Reboot the box before the ladder, and re-reboot after any crash before
  re-testing — otherwise a descending Xid31 floor masquerades as an MTP
  regression at ever-lower context.
- Enable `VLLM_TURBOQUANT_CONTINUATION_BOUNDS_CHECK=1` for the duration of
  qualification (env-gated reader guard from `fix-xid31-guard`; requires that
  branch's guard commits if not yet merged — skip if the running build
  predates them). NOTE: the engine runs under systemd, which does NOT inherit
  a shell `export`. The requal serve variant sets the env itself; for the
  step-1 MTP-off reference recording, either accept that it runs unguarded
  (it exercises the same TQ path at 60K) or add
  `Environment=VLLM_TURBOQUANT_CONTINUATION_BOUNDS_CHECK=1` to a
  `vllm-qwen27b.service.d` drop-in before recording.
- On any crash, check `dmesg -T | grep -i xid` FIRST. Paired Xid 31 on both
  GPUs = the pre-existing fault hunt, NOT automatically an MTP verdict; file it
  against the Xid31 investigation and re-run the stage after a reboot. Only a
  crash without Xid 31 (or a reproducible garble) counts against MTP.

**0. Prereq B — rebuild the engine on this branch.** The v0.1.15 merge touched
GDN/mamba kernels and the scheduler; the running venv predates it. (Note the
frontier experiments were run on the pre-merge base — the host session banked
its own identical v0.1.15 merge as `merged-v0115-regression` without promoting
it; this branch IS the promoted version, carrying the same upstream content.)

```bash
cd ~/Desktop/vLLM-2080Ti-Definitive && git fetch && git checkout claude/qwen-3.8-27b-tuning-isbc7k
./build.sh          # cold compile; engine start after this takes the usual 4-5 min
deploy/install.sh   # refreshes ~/.local/share/vllm-qwen27b/ scripts + templates
```

**1. Record the MTP-off reference** (engine as-is, before the drop-in):

```bash
cd ~/Desktop/vLLM-2080Ti-Definitive/deploy/bench
python3 bench_equivalence.py record --out ref-mtp-off.json
python3 bench_decode.py | tee decode-mtp-off.md     # baseline rows
```

**2. Activate the requal variant:**

```bash
sudo install -m644 ../systemd/vllm-qwen27b.service.d/mtp-requal.conf.example \
     /etc/systemd/system/vllm-qwen27b.service.d/mtp-requal.conf
sudo systemctl daemon-reload && sudo systemctl restart vllm-qwen27b
# wait for /health; cold start ~4-5 min
```

**3. Gates, in order — stop at the first red:**

| # | Gate | Command | Pass bar |
|---|---|---|---|
| G1 | Losslessness | `bench_equivalence.py check --ref ref-mtp-off.json` | all probes ≥200-char identical prefix |
| G2 | Failure-band ladder | `bench_mtp_requal.py` | every stage PASS: needle recalled, no garble flags, NRestarts Δ=0 |
| G3 | Tool calls | `bench_toolcalls.sh` | ≥19/20 auto; named OK; ≥4/5 code-args intact |
| G4 | Reference matrix | `bench_decode.py` | 1x7.5K ≥ **70 tok/s** (else MTP isn't paying for its risk); no garble; NRestarts Δ=0 |
| G5 | Soak | leave running overnight with the watchdog; re-run G2 next morning | NRestarts Δ=0 over the soak; morning ladder still green |

**4. Optional sweep once green:** `MTP_K=2` via the drop-in (`Environment=MTP_K=2`)
and re-run G2+G4. Ship the k with the best G4 number whose G2 acceptance stays
above ~0.45. Temperature interacts with acceptance (3.6 data: 0.71@T0.4 →
0.52@T0.8; the official 3.8 instruct rec is T0.7) — if acceptance is marginal,
try the sweep once at T0.4 to see how much headroom sampling costs.

**5. Adopt or roll back.**
- **Adopt:** keep the drop-in; record the numbers in `RESULTS-TEMPLATE.md`;
  append the outcome to vLLM-Benchmarks.md; update the Obsidian "Qwen3.8 Live
  Config" note. Clients need no changes (`qwen-local` alias unchanged).
- **Roll back:** `sudo rm .../mtp-requal.conf && sudo systemctl daemon-reload
  && sudo systemctl restart vllm-qwen27b` → MTP-off config J serves again.
  File the failing stage JSON + engine journal in the incident notes.

## Interactions to keep in mind

- **Guided decoding**: the variant pins `--structured-outputs-config
  '{"backend":"guidance"}'`. This is deliberate and load-bearing: xgrammar has
  the documented crash with spec-decode + Int4 (vLLM #11484), and named
  tool_choice exercises guided decoding server-side. G3 covers this combination
  under MTP; if G3 fails only under MTP, re-run with the drop-in's
  `Environment=MTP_K=0` equivalent (remove `--speculative-config`) to bisect.
- **Watchdog**: auto-discovers the served model — no change needed. The
  requal id (`qwen38-int4-tqk8v4-mtp3-requal`) shows up in `/v1/models`,
  which is your at-a-glance confirmation of which variant is live.
- **Shim**: no changes required; `qwen-local` stays the routing alias. The
  repetition guard (`SHIM_REP_GUARD`) stays ON — it's a second net under G2's
  garble detection, not a replacement for it.
- **The suffix-overlay / layered-spec experiments** (`VLLM_SUFFIX_OVERLAY`,
  env-gated, off) stay off during qualification — one variable at a time.
