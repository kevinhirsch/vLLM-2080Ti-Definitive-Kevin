# deploy/env/host-local: the serving host's private ops files (examples only)

Rule (Kevin, 2026-10-21): **code public, config private.** These files sit beside the gateway and the engine
launchers in `~/.local/share/vllm-qwen27b` on the serving host. They name its network, clients, paths and models, so
only redacted examples with placeholders (documentation addresses from RFC 5737) are in git. The real files stay on
the host and in its nightly backup. `install.sh` never copies them, and `deploy/bin/live_drift_check.py` treats them as
host-local (`local` in `deploy/env/live-layer-allow.txt`), so they are never reported as untracked.

| Example | Real file | What it holds |
|---|---|---|
| `clients.map.example` | `clients.map` | `<ip or X-Client> = <friendly name>` for the gateway's dashboard and logs |
| `lanes.dirs.example` | `lanes.dirs` | lane-root directories for the gateway's lane heartbeats |
| `v02.override.env.example` | `v02.override.env` | the 0.2.x engine's release pin and rollback knobs, read by `serve-hauhaucs-v02.sh` |
| `v02.model.json.example` | `v02.model.json` | the served model and its names, written by `model_switch.py` |
| `gateway-aliases.example.json` | `gateway-aliases.json` | the gateway's model aliases, every `key` blank |

Secrets never enter git: `shim.env` (`SHIM_REMOTE_KEY`) and `gateway-aliases.json` (per-alias keys) stay on the host;
their redacted forms are `deploy/env/shim.env.example` and `gateway-aliases.example.json` here.
