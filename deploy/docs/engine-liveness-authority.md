# Engine liveness authority (Lane LV, 2026-10-03)

**Problem (Lane AU audit item 7).** On HNET00, four or five owners each kept the vLLM engine alive in their own way. Each had its own probe, cooldown and kill/restart path. They did not know about each other's windows. Nobody owned the gaps between them.

24 h measured to 2026-10-03 08:10:
- journal: 104 `Started`, 73 `Stopping`, 14 SIGKILL exits;
- sudo: 85 `systemctl stop`, 79 `systemctl start`, 3 watchdog kill+restart, 5 timer stop/start pairs;
- events.jsonl: 77 `restart-requested`, 76 `restart-finished` (63 said `healthy=False`), 82 `engine-death`, 33 `engine-fault`;
- ledger: 36 FAULT rows, 78 planned-stop rows.

**Fix.** There is one authority, `deploy/bin/engine-actuator.py`, with a declared state machine. It owns:
- one lock, `restart.lock`, shared with planned restarts;
- one rate limit with backoff and one circuit breaker;
- one published state, `~/.local/share/vllm-qwen27b/liveness-state.json`.

Every other component either reads that state or asks the authority to act.

## 1. Inventory: every path that can stop, start, kill or restart the engine

| # | Path | Trigger | Own cooldown / limit | Fired (24 h) | After LV |
|---|------|---------|----------------------|--------------|----------|
| 1 | `vllm-watchdog.sh`, run by `vllm-qwen27b-watchdog.timer` every 60 s | generation wedge: models 200, 5 failed gen probes (2 with a kernel Xid), engine counters flat | 1800 s cooldown, 2 per hour (its own state file) | 3 wedge kills (SIGKILL, then `systemctl restart`), 840 OK / 112 failed probes | Detects only. A wedge goes to `engine-actuator.py recover --cause wedge`. The old rails remain only as a fallback for when the actuator is missing or crashes. |
| 2 | systemd `Restart=always`, `RestartSec=15`, `StartLimitIntervalSec=0` (drop-ins `restart-always.conf`, `no-start-limit.conf`) | any exit | 15 s pacing, no limit | ~20 auto-restarts; crash loops of 6 and 10 boots (22:11, 01:19) | Unchanged: it is the process supervisor. The authority reads it (`SubState=auto-restart` = BOOTING) and detects CRASH_LOOP from the fault ledger. |
| 3 | `engine-actuator.py restart` (Halo `engine_restart` MCP, lanes, windowctl) | discretionary, with a stated reason | one at a time (`restart.lock`), drain or offline window first | 77 planned restarts (S4 25, claude-up 22, claude-s2 14, S3 6, DFT 3, Halo 2, …) | Same, plus it is refused while someone else holds the engine or a release is in flight. The holder passes `--hold <lease>`. |
| 4 | Window scripts (frontier-queue `done/*.sh`, `projects/lanes/*/window.sh`, `tools/s4_v3_window.sh`) | a window runs | none. They stop the watchdog timer and restore it in a trap. | 8 direct stops; DFT's trap skipped the timer re-arm when the engine was healthy, so the timer stayed stopped 4.5 h | Take a TTL-bounded **hold** (`hold run … -- cmd`). Never stop the timer. Legacy windows are recognised as implicit holds (process pattern, frontier `RUNNING` flag), each with a bound. |
| 5 | `estate-watchdog.sh --fix`, cron every 5 min | checks only. Its `--fix` touches wg0 and the .10 route, **never the engine**. | per-fix cooldown + circuit | 0 engine actions | Reads `liveness-state.json` (new check `engine-liveness`). It is the dead man for the authority's clock: it re-arms the watchdog timer when it has been stopped ≥30 min and no window owns the engine. |
| 6 | `engine-fault-collector.py` (ExecStopPost, `--pre-kill`, `--reconcile`) | every engine death | none (records only) | 82 deaths classified | Records only. A death inside an engine hold is tagged `during_hold`; a stop inside one is attributed to the holder. |
| 7 | Gateway shim `local_healthy()` (`keepalive-shim.py`) | `/health` every `HEALTH_TTL` | — | routing only, never acts | Unchanged (GW2 and CFG own the shim). It is a reader of engine health for routing, not an actuator. |
| 8 | Operator: `vllm-stop.sh`, `vllm-restart.sh`, `switch-model.sh` | Kevin | — | 1 (kevin) | Stop takes an 8 h `kevin-desktop` hold, so the authority does not undo it. Restart releases that hold. |
| 9 | `warmup-after-start.sh` (ExecStartPost) and `announce-start` | every start | — | — | Unchanged (AU). It keeps the unit `activating` while warming. The authority's boot deadline (900 s) covers it. |

**Gaps nobody owned before LV:**
- An engine stopped and abandoned, e.g. a window killed before its trap ran. Nothing started it.
- A process whose API never came up, or stopped answering while the process lived. The watchdog only counts "models up, generation down".
- A watchdog timer left stopped.
- A planned-restart job left `draining` or `stopping` by a dead actuator.

## 2. The state machine

`classify()` is pure and is tested. The table below is published verbatim in `liveness-state.json` under `declared`, so the estate's generic progress invariant (`tools/stateful_objects.py`) can hold each state to its own declaration.

The order is: who owns the engine first, then health.

| State | Kind | Owner (who acts) | Deadline | Exits |
|-------|------|------------------|----------|-------|
| UP | resting | — | — | SUSPECT, UNRESPONSIVE, DOWN, PLANNED, HELD |
| SUSPECT | active | watchdog probes | 180 s | UP; wedge confirmed → `recover`; UNRESPONSIVE |
| BOOTING | active | systemd and the warm-up hook | 900 s | UP; STUCK_BOOT; DOWN |
| STOPPING | active | systemd (TimeoutStopSec, then SIGKILL) | 120 s | DOWN, BOOTING |
| DOWN | active | authority: `start` | 180 s grace | BOOTING (RECOVERING); BREAKER_OPEN |
| STUCK_BOOT | active | authority: `recover` | 0 | RECOVERING; BREAKER_OPEN |
| UNRESPONSIVE | active | authority: `recover` | 0 | RECOVERING; BREAKER_OPEN |
| RECOVERING | active | authority: verifies its own action | 960 s | UP (outcome ok); action failed → backoff or BREAKER_OPEN |
| PLANNED | active | planned restart (`restart.lock` holder) | 2400 s | UP or BOOTING; holder died → job reconciled `abandoned` |
| HELD | active | the holder | the hold's TTL (≤ 8 h) | release, TTL expiry, holder pid died |
| OFFLINE_WINDOW | active | the gateway offline-lease holder | gateway TTL (≤ 3600 s) | window closed or expired |
| CRASH_LOOP | active | Halo (hand-off `engine-crash-loop`) **and** a need of kind `decision` sent to Kevin (Discord + thread, deduped while open). Hand-offs have no consumer until Halo spec 02 lands. | — | UP; a Halo planned restart or rollback |
| BREAKER_OPEN | active | authority (half-open retry), plus a Halo hand-off and a need sent to Kevin | backoff (≤ 7200 s) | half-open attempt at `next_try`; UP; `reset-breaker` |
| PAUSED | active | Kevin (`LIVENESS_PAUSE` file) | 24 h (flagged after) | remove the file |

The watchdog neither probes nor counts in HELD, OFFLINE_WINDOW, PLANNED or PAUSED. This replaces "stop the timer during a window".

## 3. The gate: one rate limit, backoff and breaker for every automatic action

Automatic actions are `start` (DOWN after the grace period, or when a hold ends) and `recover` (SIGKILL the control group, then `restart --no-block`). Each one is appended to `liveness-actions.jsonl` with state `pending`. It is then **judged by its effect**:
- `ok` when `/health` answers on a boot that started after the action;
- `failed` when `/health` is still down after 960 s.

Rules:
- At most 2 automatic actions per hour, across all causes. This keeps the watchdog's long-standing rail.
- Base gap of 600 s between actions, doubling per consecutive failed action, capped at 7200 s.
- Nothing new while an earlier action is still being verified.
- After 3 consecutive failed actions the breaker opens (Halo hand-off). It half-opens for one attempt when the backoff has passed. An `ok` outcome, or `reset-breaker --by --reason`, closes it.
- Actions never block on the boot (`--no-block`), so a 60 s oneshot cannot be killed mid-recovery.

Knobs (env): `LIVENESS_AUTO_MAX_PER_HOUR`, `_AUTO_MIN_GAP_S`, `_BACKOFF_MAX_S`, `_BREAKER_FAILS`, `_BOOT_DEADLINE_S`, `_UNRESPONSIVE_S`, `_DOWN_GRACE_S`, `_CRASH_LOOP_FAULTS`, `_CRASH_LOOP_WINDOW_S`, `_HOLD_MAX_TTL_S`.

## 4. Holds (`liveness-holds.json`, flock-serialised)

```
engine-actuator.py hold acquire --kind engine|quiesce --by WHO --reason WHY --ttl S [--owner-pid PID]  -> {"lease": ...}
engine-actuator.py hold renew   --lease L --ttl S
engine-actuator.py hold release --lease L | --by WHO
engine-actuator.py hold run     --kind engine --by WHO --reason WHY --ttl S [--no-ensure-up] -- CMD ARGS
engine-actuator.py hold status
engine-actuator.py restart --hold L ...         # the holder restarts its own engine
```

- **engine**: a window owns the engine. The authority takes no automatic action, the watchdog does not probe, and planned restarts are allowed only to the holder.
- **quiesce**: a release is in flight (the gateway publish). Planned restarts are refused. Wedge recovery still runs, because a wedged engine would never let the drain finish.
- There is at most one hold per kind.
- TTL is 60 s to 8 h. Every hold expires by itself. `run` holds and `--owner-pid` holds are void the moment their owner dies (pid and start-time check).
- `hold run` releases in `finally` and then starts the engine at once if it is DOWN. This is the windows' old "always a healthy engine at exit" contract, now in one place.

## 5. The drain-disturbance class (gateway publish, GW2 08:01)

GW2's publish aborted after its drain with "Halo became active during gateway drain". The two incidents had finished their runs mid-drain and turned `repair-requested` with run `stopping`.

The cause was two different predicates for "a Halo run is live":
- the incident supervisor counts `stopping` as terminal (slot free; the next round waits on the start pause);
- `gateway_safe_publish.py` counted it as active, and missed live investigation runs.

The publisher now applies the supervisor's own lease-backed occupancy rule. A test fails if its copies of `TERMINAL_HALO_RUN_STATUSES`, `OPEN_STATES` or `ACTIVE_EXECUTION_STATES` drift from the supervisor's.

The publish also:
- takes the liveness **quiesce** hold, owned by its pid, so no planned engine restart can collide with the drain;
- marks the start-pause lease `phase: draining` when the drain opens.

The supervisor side is in Halo spec `05-…`. The pause still exempts restoration and control-rank-0 starts. During `phase: draining` it should hold even those, bounded by the lease.

## 6. What else reads the state

- `estate-watchdog.sh`, check `engine-liveness`:
  - stale for more than 300 s → CRIT (the authority's clock stopped);
  - CRASH_LOOP or BREAKER_OPEN → CRIT;
  - acting states → WARN.
- Halo's `engine_status`: estate spec 06 adds `liveness-state.json` to it.
- The estate progress invariant: spec 06 registers type `engine-liveness`.

## 7. Phase 2 (not built here; owners named)

- **Boot-success gate with rollback to the last-known-good profile** (AU's suggestion): N consecutive `boot-failed` faults with the same cause → revert `active-serve` and the override env to the last profile that reached UP. It belongs with RL's `release.py` (immutable releases) and `windowctl.py`. The authority already detects CRASH_LOOP and hands it to Halo.
- RL's `windowctl` takes an engine hold for the window's lifetime (agreed 10-03).
- Legacy window scripts migrate to `hold run`. Until then they are implicit holds:
  - a matching `bash …(_window|_driver|/window).sh` process, for at most 8 h of its runtime;
  - the frontier `RUNNING` flag, for at most 3 h.

## 8. Deploy and rollback

Deploy is a plain file copy to `~/.local/share/vllm-qwen27b/`, with the old copies archived first:
- `engine-actuator.py`, `vllm-watchdog.sh`, `engine-fault-collector.py`;
- AU's `warmup-after-start.sh`;
- `vllm-stop.sh` and `vllm-restart.sh`;
- `watchdog/estate-watchdog.sh`.

No unit change and no engine restart are needed. The next timer tick runs the authority.

Kill switch: `touch ~/.local/share/vllm-qwen27b/LIVENESS_PAUSE`. The authority then publishes state only, and the watchdog defers.

Rollback: copy the archived files back.
