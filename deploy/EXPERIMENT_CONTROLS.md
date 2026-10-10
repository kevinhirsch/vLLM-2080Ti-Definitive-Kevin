# Isolated engine measurements

The controller runs as a user unit; `vllm-qwen27b` and its engine watchdog are SYSTEM units. Estate Core is a USER service and can restart the engine directly. An actuator hold alone does not stop Core publication. Keep the retired overseer paused and use an explicit maintenance stop of the current producer for a trial.

Direct-engine commissioning can run from the reviewed, frozen checkout while using the existing state and leases:

```sh
ENGINE_ACTUATOR=/home/kevin/Desktop/wt-experiment-controls-20261010/deploy/bin/engine-actuator.py \
python3 /home/kevin/Desktop/wt-experiment-controls-20261010/deploy/bin/windowctl.py run /absolute/path/trial.json --results /absolute/path/results
```

The source actuator imports its sibling helper; the deadman receives the same explicit actuator path. Keep this checkout frozen until restoration. Alternatively deploy `deploy/bin/experiment_controls.py`, `windowctl.py`, and `engine-actuator.py` together beside their existing dependencies in `~/.local/share/vllm-qwen27b`, preserving a reviewed copy of the previous files. Atomic replacement of these scripts does not require an engine or gateway restart. Coordinate the producer pause and any publishers first. The gateway patch can be deployed separately through the qualified publisher; it also requires `experiment_controls.py` beside `keepalive-shim.py`.

Validate a reviewed JSON/YAML specification with:

```sh
python3 /home/kevin/.local/share/vllm-qwen27b/windowctl.py validate /absolute/path/trial.json
python3 /home/kevin/.local/share/vllm-qwen27b/windowctl.py submit /absolute/path/trial.json --wait
```

`run SPEC --results DIR` is the foreground alternative. Submit freezes the controller, helper dependencies and spec; its deadman uses the same explicit control paths and SYSTEM engine identity. Token values are never command arguments.

A direct-only specification can use:

```json
{
  "lane": "switch-review",
  "window": "direct-cold-warm-c1-c4",
  "reason": "isolated baseline throughput and restoration qualification",
  "max_s": 3600,
  "ttl_s": 1800,
  "wait_s": 120,
  "watch_user_units": ["estate-core.service"],
  "pause_user_units": ["estate-core.service"],
  "steps": [
    {
      "name": "measure",
      "run": ["/absolute/python", "/absolute/reviewed-probe.py", "REVIEWED_PROBE_ARGUMENTS"],
      "timeout_s": 900
    }
  ]
}
```

Replace the placeholder argv with the measurement agent's actual probe arguments before validation and submission. A direct step omits `measurement` and never calls `/gateway/measurement`. Every run with a live engine exports `WINDOW_ENGINE_PID` and `WINDOW_ENGINE_IDENTITY_FILE`; these contain observed PID and sanitized identity, not credentials. `WINDOW_NATIVE_MANIFEST` points to `snapshot.json` for the measurement agent's legacy disk-library hash gate; it does not attest mapped modules. A boot step accepts `boot: {tree: PATH, env: {...}, extra_args_append: "..."}` or a prebuilt release. Root, model path, selected CLI flags and requested environment must match the new process; rc=0 plus unchanged PID cannot pass. Scheduler identity includes `--long-prefill-token-threshold` and `VLLM_SCHED_*`.

After scoped endpoint deployment, add `measurement: {phase: "baseline-cold-c1", ttl_s: 180}` to a run/script step with `timeout_s` at most 3600. The controller grants the phase via admin-authenticated `/gateway/measurement`, writes a temporary 0600 token and nonsecret identity file, and exports `WINDOW_BENCH_TOKEN_FILE` and `WINDOW_BENCH_IDENTITY_FILE`. The probe sends `X-Window-Token` only to the local gateway and uses model `estate-local`. Each successful local response carries `X-Gateway-Route: local` and JSON `X-Window-Identity` with window, phase, PID generation, root, model, flags and fingerprint. Forged client labels grant no exemption. Admission still obeys health, capacity and publication fences.

The controller renews the same phase token at a 180-second TTL every 60 seconds, up to its fixed deadline (maximum 3600 seconds). Revocation, expiry, identity change or gateway restart denies queued token callers. Before every restart and at each measurement step's exit, revoke the phase, remove its token file and observe two zero gateway/direct-engine queue samples. Missing counts fail closed.

Restoration revokes phase admission, stops owned probe units (excluding its deadman), then verifies the gateway/direct-engine drain. This also cancels a direct probe that kept issuing requests after its controller died. It then puts back original override and model selector bytes, native libraries, root/model/flags, KV pool and SYSTEM timer states. A bounded direct generation must produce `CONTROL_OK`, finish normally, report completion tokens and keep the same runtime identity. Only then are USER service activity states restored, the offline window closed and the actuator hold released. Failed qualification preserves the maintenance stop, hold, offline fence and deadman for recovery. `summary.json.restore.ok` and `state.json.restored` must both be true; an offline test pass does not qualify a live trial.
