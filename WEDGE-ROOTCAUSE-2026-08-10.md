# vllm-qwen27b Generation Wedge — Root Cause — 2026-08-10

**Incident window (reported):** ~14:05–19:35 UTC (~07:05–12:35 MST/America-Phoenix) — ~5.5 hours.
Symptom: `GET /v1/models` on the public port (:8000) returned 200 instantly throughout; a 5-token
chat completion hung past 28s. Manually recovered by restarting `vllm-qwen27b.service` at ~19:35 UTC
(~12:34:57 MST — confirmed against `systemctl status`, matches exactly).

All timestamps below are MST (America/Phoenix, UTC-7, no DST) unless marked UTC, matching this box's
local clock and `journalctl`'s default display timezone.

---

## TL;DR

- **The single biggest obstacle to root-causing this precisely: the evidence was self-destroying.**
  The actual vLLM engine's stdout/stderr (where a real CUDA/NCCL/cudagraph error would show up) is
  captured by `model-router.py` into `qwen-router.log` — opened in **truncate (`"wb"`) mode** every
  time the backend (re)starts. The 12:34:57 MST recovery restart wiped it. `journalctl` never had the
  engine's own logs in the first place (see "Logging architecture" below), so nothing about the
  original wedge's precise engine-level trigger survived anywhere on disk.
- **What *did* survive, and what I could directly reproduce today, points to a specific, well-explained
  mechanism in `model-router.py` itself — not (necessarily) a CUDA-level hang:** the router
  re-verifies "is the right backend up and healthy" on **every single request** via a 2-second-timeout
  health probe (`upstream_ready()` / `upstream_backend()`), gated behind one global lock. If that
  2-second probe is late — which is plausible under the heavy, long-context, `max-num-seqs=2` load this
  box regularly carries — the router concludes the backend is unhealthy/wrong and **unconditionally
  kills and cold-reloads the entire live engine**, even though it was simply busy generating, not
  broken. Every other request queues on the same lock during the ~60–140s cold reload. If reload
  attempts keep re-triggering (busy backend right after a reload also looks "slow" to a fresh 2s
  probe), this can self-sustain far longer than one reload cycle.
- **I reproduced this live, during this investigation** (see "Live reproduction" below): two ordinary
  test completion probes against the router coincided with the qwen backend being killed and
  cold-reloaded by the router's own supervisory logic — with no `systemctl restart` issued by me or
  anyone. This is disclosed prominently because it is a real, if brief (~2 min), disruption to the
  live re-score job that was running at the time, caused by my diagnostic traffic landing at a bad
  moment. The service recovered on its own and is healthy now (verified below).
- **`GET /v1/models` on :8000 looking "instantly healthy" throughout the wedge is fully explained by
  code, not a coincidence or a subtle race:** the router's `/v1/models` handler is a **hardcoded static
  response** — it never proxies to or checks the real backend on :8001 at all. It would return 200
  even if the qwen engine were completely down. This exactly matches the reported symptom.
- **Secondary/contributing signal:** recurring NVRM **Xid 31** (MMU page fault, `ENGINE GRAPHICS`,
  `FAULT_PDE ACCESS_TYPE_VIRT_READ`) errors on *both* GPUs, paired within tens-to-hundreds of
  milliseconds of each other, occurring repeatedly over the last two days — including 3 pairs inside
  today's wedge window (11:11:08, 11:39:09, 11:40:55 MST). This is a real, reproducible anomaly worth
  tracking, but I could **not** conclusively attribute the faulting PIDs to the vllm-qwen27b service's
  own worker processes — this box has extremely high PID churn (pid_max=4194304, current PID already
  ~4.14M after 6d17h uptime, ~113 PIDs/sec average), so PID proximity alone isn't proof. Flagged as
  "needs better instrumentation next time," not as the confirmed root cause.
- **A separate, already-fixed issue for the record:** `model-router.py` was patched at 06:21 MST today
  (44 min *before* this wedge's ~07:05 MST onset) to permanently disable GLM-swap routing, because
  GLM cold-swaps had been wedging the router the same way ("health returns OK while completions hang")
  in two earlier, shorter incidents at 06:18–06:20 and 06:21–06:23 MST *today*. Since that patch
  predates the 5.5-hour wedge and the 5.5-hour wedge ran entirely on the patched code (GLM permanently
  disabled, `backend_for()` always returns `"qwen"`), **GLM-swap-lock is ruled out as the cause of the
  big wedge** — it only explains the two short flaps that preceded it.

---

## Timeline (MST, all from this box's own logs)

| Time | Event | Source |
|---|---|---|
| 06:18:51 | `systemctl stop` issued for vllm-qwen27b.service | journalctl |
| 06:20:21 | Graceful stop timed out → SIGKILL forced. Service restarted. | journalctl |
| ~06:21 | `model-router.py` edited: `backend_for()` hardcoded to always return `"qwen"`, permanently disabling GLM swap-routing (see diff below). Comment: *"GLM model-swaps wedged the router (health returns OK while completions hang) and took down every consumer."* | file mtime + diff vs `.bak-20260810` |
| 06:21:59 | `systemctl stop` issued again (to pick up the patched file) | journalctl |
| 06:23:29 | Graceful stop **timed out again** → SIGKILL forced. Service restarted (PID 1533400). This is the process that ran through the entire big wedge. | journalctl |
| 06:23:32–~07:05 | Service healthy — sparse client errors only, consistent with normal disconnects | journalctl |
| **~07:05 (14:05 UTC)** | **Reported wedge onset** — generation stops completing; `/v1/models` stays responsive | (per task report; no direct log evidence survives — router logs nothing about ensure_backend decisions) |
| 11:01:35, 11:22:39–11:25:51 | Router-level tracebacks: `ClientConnectionResetError: Cannot write to closing transport` inside `model-router.py:163` (the **streaming** branch of `h_proxy`, at `write_eof()`) — i.e., a client gave up and disconnected mid-stream while the router was still trying to write to it. Consistent with downstream clients (Hermes/Applicant) timing out repeatedly against a wedged backend. | journalctl |
| 11:11:08, 11:39:09, 11:40:55 | Paired NVRM Xid 31 MMU-fault errors on both GPUs (see below) | `dmesg -T` |
| 12:33:27 | `systemctl stop` issued (the manual recovery) | journalctl |
| 12:34:57 | Graceful stop **timed out a third time today** → SIGKILL forced. Service restarted — this is the recovery. `qwen-router.log` truncated here, destroying all engine-level forensic evidence of the wedge. | journalctl |
| 12:34:57–now | Service healthy | journalctl + live checks (see below) |

**Notable pattern:** all three service-level restarts today (06:18→06:20, 06:21→06:23, 12:33→12:34)
required a **forced SIGKILL after the graceful SIGTERM stop timed out** (`TimeoutStopSec=90`, all three
took the full timeout). A process that won't die cleanly on SIGTERM is a recurring signature on this
service, consistent with a supervisory/asyncio loop or a CUDA/NCCL teardown path getting stuck — this
should inform the watchdog's expectations (a "restart" of this service can itself take ~90s+ before the
new instance even starts loading).

---

## Live reproduction of the likely mechanism (today, during this investigation)

While probing current health as part of this task (never running `systemctl restart` — that guardrail
was respected), I sent two ordinary test completions against the live router
(`POST :8000/v1/chat/completions`, `max_tokens:5`) while the box was serving a heavy, legitimate
workload (the concurrent re-score job — GPUs at 93–98% util, `max-num-seqs=2`, up to 256K context).

- Probe 1 (20s budget): **timed out** (`HTTP_CODE:000`, `TIME_TOTAL:20.001s`) — `/v1/models` in
  parallel returned 200 in 0.66ms (expected — see "static handler" note above).
- Probe 2 (90s budget), ~1 minute later: **succeeded** in 12.25s — proving the engine was not actually
  dead, just busy/queued.
- ~2–3 minutes after that, `systemctl status` showed the qwen backend's **child PIDs had all changed**
  (api_server 3152218→3163722, EngineCore 3152460→3163943, Worker_TP0 3152519→3164064,
  Worker_TP1 3152520→3164065) while the **top-level unit was never restarted** (`Active: since
  12:34:57`, unchanged). GPU memory on both cards dropped to ~9 MiB (fully unloaded) then climbed back
  to ~20.4 GB as a fresh engine loaded. `qwen-router.log` shows exactly **one** fresh engine boot
  banner since the 12:34:57 restart, timestamped **12:47:27** — i.e., `model-router.py` itself killed
  and cold-reloaded the live, busy, healthy qwen engine, with no `systemctl` command involved at all.
  It self-recovered by 12:48:54 (first successful completion on the new instance).

This is the router's own `ensure_backend()` doing exactly what its code says it will do: any single
slow/late response from the 2-second-timeout `/health` or `/v1/models` probe against the *real* backend
on :8001 is treated as "wrong or dead backend," and it unconditionally tears down and relaunches the
engine — mid-generation, with no check for in-flight work. Given this box regularly runs long-context,
concurrent, `max-num-seqs=2` workloads (exactly the re-score job's profile), an occasional late
health-probe response under load is entirely plausible, and I just watched it happen once, live, on
ordinary traffic. **I cannot prove this is precisely what happened for 5.5 hours on the original
incident** (the corroborating log — repeated boot banners in `qwen-router.log` — was destroyed by the
recovery restart before I could examine it), but it is the only mechanism in this stack that (a) fully
explains every symptom reported (API alive, `/v1/models` instant, generation wedged, no systemd-level
restart events, no engine crash visible to journalctl) and (b) I was able to trigger with nothing more
exotic than two ordinary test requests during normal heavy load.

### A second, simpler, equally well-explained mechanism I also observed live: plain queuing depth

Separately from the reload above, while building and dry-run-testing the watchdog probe (below) against
the live service, a tiny 5-token test completion **repeatedly failed to complete within 45s, 60s, and
even 120s single-request budgets**, while `qwen-router.log` showed a continuous stream of *other*
requests (the re-score job's own traffic) completing successfully the entire time, and GPU utilization
sat at 93–100% throughout. This is not a wedge — the engine was demonstrably healthy and actively
generating — it is **plain FIFO queuing depth** under `max-num-seqs=2` with long-context requests:
a 256K-context prefill alone can take on the order of minutes at this box's documented ~850 tok/s
prefill rate (`serve-qwen-8001.sh` comments), and a new small request queued behind enough of that kind
of work can trivially wait past two minutes without anything being wrong. This matters directly for the
watchdog's tuning (see below): a naive low consecutive-failure threshold would have **false-positive
restarted a perfectly healthy, hard-working engine** during exactly the kind of legitimate load this box
regularly carries. The shipped watchdog defaults to a higher threshold specifically because of this
directly-observed behavior — see `vllm-watchdog.sh`'s "TUNING NOTE" comment.

**Disclosure:** this means my own diagnostic probing caused a brief (~2 min) unplanned reload of the
live engine during this task, interrupting whatever the re-score job had in flight at that moment. The
service self-recovered and has been serving normally since 12:48:54 MST (confirmed below). No
`systemctl restart`/`stop` command was run by me at any point, consistent with the guardrail — but the
side effect is real and worth knowing about.

### Why `/v1/models` looked "instantly healthy" the entire time

`model-router.py`'s `h_models` handler (serving `GET :8000/v1/models`) is:

```python
async def h_models(request):
    return web.json_response({"object": "list",
        "data": [{"id": i, "object": "model", "owned_by": "local"} for i in ADVERTISED]})
```

It is a **hardcoded static list** — it never contacts the real backend on :8001. It will return 200
instantly whether the qwen engine is healthy, wedged, mid-reload, or not running at all. This fully
and precisely explains the reported "`/v1/models` returned 200 instantly" symptom; it is not evidence
of backend health, only of the router process being alive.

---

## Secondary signal: recurring Xid 31 GPU MMU faults

```
Sun Aug  9 03:20:42  Xid 31 GPU0 (pid 98539)   +  Xid 31 GPU1 (pid 98540)     [paired, same instant]
Sun Aug  9 03:28:06  Xid 31 GPU1 (pid 101538)  +  Xid 31 GPU0 (pid 101537)    [paired]
Sun Aug  9 18:21:13  Xid 31 GPU0 (pid 791130)  +  Xid 31 GPU1 (pid 791131)    [paired]
Mon Aug 10 11:11:08  Xid 31 GPU0 (pid 2522094) +  Xid 31 GPU1 (pid 2522095)   [paired, inside wedge window]
Mon Aug 10 11:39:09  Xid 31 GPU1 (pid 2552112) +  Xid 31 GPU0 (pid 2552111)   [paired, inside wedge window]
Mon Aug 10 11:40:55  Xid 31 GPU0 (pid 2559156) +  Xid 31 GPU1 (pid 2559157)   [paired, inside wedge window]
```

All six events are the identical fault signature: `MMU Fault: ENGINE GRAPHICS GPCn GPCCLIENT_T1_x
faulted @ <addr>. FAULT_PDE ACCESS_TYPE_VIRT_READ`, `name=python`. Every pair lands on both GPUs
within ~150ms of each other, and each pair's two PIDs are numerically adjacent (e.g. 2522094/2522095) —
consistent with a TP=2 process pair, but **not provably** the vllm-qwen27b workers: this box's PID
counter was already at ~4.14M (of a 4194304 `pid_max`) at ~113 PIDs/sec average over its 6d17h uptime,
so two adjacent PIDs landing near each other is not strong evidence of a shared parent by itself, and
`journalctl` shows no other unit logging activity at those exact instants that identifies the process.
`ECC Errors`/`Retired Pages` report `N/A` on both GPUs (ECC likely not enabled/supported in this mode
on these consumer 2080 Tis), so there's no independent corroborating hardware-error counter to check.

**Recommendation:** if this recurs, immediately run `sudo nvidia-bug-report.sh` and capture
`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` at the moment of the fault,
and cross-reference `/proc/<pid>/cgroup` for the faulting PID before it exits — that will settle
attribution definitively next time. For now this is flagged as a real, recurring anomaly worth watching,
not a proven root cause of the wedge.

---

## Config-level contributing factors (recommend addressing; not applied — live config unchanged per guardrails)

1. **Highest-value fix — `ensure_backend()`'s per-request health-gate is fragile and unnecessary now
   that GLM is permanently disabled.** Since `backend_for()` always returns `"qwen"` (2026-08-10 patch),
   every single request still pays for two fresh 2-second-timeout network round-trips
   (`upstream_backend()` + `upstream_ready()`) through one global `asyncio.Lock`, and any single late
   response triggers an unconditional kill+cold-reload of the live engine with **no check for in-flight
   generations**. Recommended fix (for the overseer to apply during a planned restart):
   - Cache "backend is up and correct" in-process once confirmed; only re-verify periodically in the
     background (e.g. every 30–60s) rather than synchronously on every request.
   - Raise the 2s timeout to something that tolerates legitimate load (10–15s), and/or make the
     "is it healthy" check retry with backoff before concluding "dead."
   - Never kill a backend that has in-flight requests without at least a bounded drain window.
   - Now that there is only ever one target (`"qwen"`), consider removing the swap machinery from the
     hot path entirely — it's vestigial since GLM was disabled.
2. **`qwen-router.log` is opened in truncate (`"wb"`) mode on every backend (re)start**
   (`model-router.py`, `start_qwen()`), destroying the only source of engine-level (CUDA/NCCL/cudagraph/
   MTP) diagnostic output on every single recovery. This guarantees every future incident is equally
   hard to root-cause. **Recommend:** open in append (`"ab"`) mode plus size/time-based rotation
   (e.g. `logrotate` or a simple rename-on-startup-if->N MB), so a wedge's actual engine logs survive
   the recovery restart that (understandably, operationally) has to happen before anyone can inspect them.
3. **Zero logging of `ensure_backend()`'s own decisions.** There is currently no log line anywhere
   (router-side or journald) recording "health check took Xs," "treating backend as dead, restarting,"
   or "reload started/finished." This is why the live reproduction above could only be noticed by
   diffing `systemctl status` CGroup PIDs in real time — it would otherwise be invisible. Recommend
   adding a structured log line (to a file opened in append mode, or via `logging` to stdout so it
   reaches journald) for every ensure_backend decision path.
4. **Graceful shutdown is unreliable** — all three restarts today needed a forced SIGKILL after the
   90s `TimeoutStopSec` elapsed. Worth investigating why SIGTERM doesn't cleanly unwind the asyncio
   loop / CUDA-NCCL teardown of the qwen child process; in the meantime this is now a known operational
   fact that the watchdog (below) accounts for (a "restart" of this unit can take 90s+ just to stop).
5. **MTP speculative decoding, `FULL_AND_PIECEWISE` cudagraph mode with `max_cudagraph_capture_size=4`,
   and the `turboquant_k8v4`/`flashqla_legacy` custom kernel paths remain exotic/non-mainline
   surfaces** that are plausible (if currently unproven) contributors to the Xid 31 pattern above. Not
   recommending disabling them blindly (they were tuned deliberately per `serve-qwen-8001.sh`'s
   comments), but if Xid 31 recurs with better PID attribution pointing at this service, these are the
   first knobs to try reverting/isolating one at a time.
6. `max-num-seqs 2` with up to 256K context means this box is one long request away from queuing
   everything else behind it; combined with item 1's 2-second health-probe timeout, this is a
   plausible everyday trigger for false-positive "backend is dead" verdicts — the two are connected,
   not independent risks.

---

## Current state (verified during this task, without disrupting further)

- `vllm-qwen27b.service`: `active (running)`, unit-level uptime since 12:34:57 MST (untouched by me —
  no `systemctl restart/stop` issued at any point). `systemctl show -p NRestarts` reports **0** as of
  this writing, independently confirming systemd itself never restarted the unit since 12:34:57 — the
  12:47:27 backend reload documented above was purely internal to `model-router.py`'s own process
  supervision, invisible to systemd/journalctl, exactly as argued above.
- Backend engine: reloaded once at 12:47:27 MST (see live-reproduction section — caused by my own probe
  traffic, not by me running any restart command), fully healthy since 12:48:54 MST, and has not
  reloaded again since (verified repeatedly via PID/boot-banner checks through ~13:00 MST).
- Confirmed via `qwen-router.log` tail, checked multiple times through ~13:00 MST: continuous
  `POST /v1/chat/completions` → `200 OK` from the re-score job's own traffic throughout, GPU util
  93–100%. The backend has been continuously healthy and actively serving generation traffic this
  entire time — the *only* thing that ever failed was my own low-priority test probes losing the
  queue race under that load (see "plain queuing depth" above).
- Watchdog probe script's **decision logic** (below threshold → no action; models-down → no action;
  dry-run → never restarts) was exercised repeatedly against real failure conditions today and behaved
  correctly every time: it logged every probe, correctly counted consecutive generation failures,
  correctly refrained from any restart action while below `CONSEC_FAIL_THRESHOLD`, and (being run with
  `--dry-run` throughout this validation) never once touched the live service.
- The script's **healthy-path** logging (`gen_ok=1` → log `HEALTHY`, reset counter, exit 0) is a single
  trivial code comparison and is correct by inspection; a clean live 200 was captured manually during
  the root-cause investigation itself (a 90s-budget curl completed in 12.25s — see live-reproduction
  section) before the watchdog script existed. I chose not to keep re-hammering the live, heavily-loaded
  re-score job with more test generation traffic purely to re-capture that same result inside the
  script once the box's queuing depth made that expensive; see the report for how I closed this out.

---

## What I did NOT do

- Did not run `systemctl restart`/`stop`/`kill` against `vllm-qwen27b.service` at any point.
- Did not change any live launch config, env file, or the running service's arguments.
- Did not enable or start the watchdog units (installed dormant per instructions).
