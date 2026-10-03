# Engine releases, the window framework, and jobs in units (lane RL, 2026-10-03)

Leads: L104 (immutable releases), L105 (one window framework), L106 (jobs in systemd units), L107 (PYTHONPATH re-export bug).

## 1. Immutable engine releases (`deploy/bin/release.py`)

Before this, production booted `~/Desktop/wt-integrate`, a mutable git worktree that people also develop in. On
2026-10-03 there were three near-misses:

- an uncommitted fix was sitting in the prod tree;
- lane trees symlink `.venv`/`.deps` into it, so a lane boot could JIT-rebuild the `tq_gqa`/FlashQLA `.so` files inside the prod tree (torch `cpp_extension` keys `build.ninja` on the source path);
- an override PYTHONPATH was silently dropped.

Layout (`~/.local/share/vllm-releases/`, outside every worktree, on the same disk):

| path | what |
|---|---|
| `<id>/` | V02_ROOT. `git archive <sha>` (exactly the commit), plus the compiled `vllm/*.so`, `_version.py` and `third_party/triton_kernels` copied from `--from` (default wt-integrate; refused when csrc/CMake/setup.py differ from that tree's HEAD or the `.so` predate the last csrc commit), plus FlashQLA sources (patch hash recorded). Read-only except caches and JIT build dirs. |
| `<id>/.deps/tq_gqa_build`, `<id>/.deps/FlashQLA…/.torch_extensions_vllm_flashqla_legacy` | JIT `.so` prebuilt at the release's own path with no GPU visible. A second load must be a ninja no-op, so a boot rebuilds nothing. |
| `<id>/.venv -> ../venvs/<fp>` | Hard-linked snapshot of the source venv: about 28 MB of directories, not 7.2 GB. It drops the editable finders that point back at the dev tree, rewrites the shebangs, and has read-only directories (so no `pip install` into it). |
| `<id>/RELEASE.json` | Manifest: sha, `dirty=false`, sha256 of every `.so`, venv fingerprint, build env (nvcc/gcc/arch), created_at, size. |
| `<id>/RELEASE.files.sha256` | sha256 of every immutable file. This is what `verify` re-checks. |
| `current -> <id>` | What production boots. `serve-hauhaucs-v02.sh` resolves it once at boot (`readlink -f`). |
| `history.jsonl` | Every activation, rollback and auto-rollback. |

```
release.py build <sha> [--label k5] [--from TREE]     # ~6 min (two JIT builds, niced), ~0.5 GB
release.py list | verify [<id>|current] [--running] | env <id> | rm <id>
release.py activate <id> --reason R     # verify -> GPU gate -> flip -> drained restart via engine-actuator (own unit)
                                        # -> health -> engine cwd == release -> verify again -> KV pool; auto-rollback on failure
release.py rollback --reason R          # back to the previous release (or the legacy tree)
```

Lane A/B trees become releases. Build one with `release.py build <lane sha> --label <lane>`, and a window boots it with
`boot: {release: {sha: ..., label: ...}}`, which builds it before the window opens if it is missing. A `tree:` boot
whose JIT dirs are symlinks into another tree is refused.

## 2. One window framework (`deploy/bin/windowctl.py`)

A lane writes a declarative spec (see `deploy/windows/*.yaml`; the header of `windowctl.py` is the schema). The
framework owns everything else:

- **Gateway window:** opens the local-offline window (gateway-offline.py, lease recorded with its pid), renews it every 60 s, and always closes it.
- **Snapshot and exact restore** of:
  - the bytes of `v02.override.env`;
  - `snapshot_files`;
  - the states of the watched systemd timers (and `pause_timers`, restarted by the restore and by the dead-man);
  - the `.so` hashes of the production root (JIT build dirs are backed up and put back if a boot rebuilt them);
  - health;
  - the KV pool. A restore below the expected pool, after one retry, is a FAILED restore.
- **GPU gate before every boot:** there must be no non-engine compute app on either GPU. The framework waits `gpu_gate_wait_s`, then fails and names the owner unit and command.
- **Per boot:** records the KV pool, boot time and running root.
- **Xid watch:** takes an Xid baseline and checks for new Xids after every step; each new Xid is attributed to its owning unit, and `on_new_xid: abort` stops the window.
- **Spend:** at or above `spend_pause_usd` (default $20) the window pauses, meaning it stops and restores. The gateway's own $25/day cap is never touched.
- **Units:** every command runs in its own systemd user unit `<lane>-<window>-<step>`.
- **Dead-man:** a timer unit (`<lane>-<window>-deadman`) ticks every 2 minutes. If the window process is gone, it restores. If the process overran by 15 minutes, it sends TERM.
- **GPU busy signal:** published in `~/.local/share/vllm-qwen27b/gpu-busy.json` (see §4).
- **Conflicts:** `conflicts:` refuses to start while a matching process runs (an exact `/proc` scan that never matches itself).
- **Promotion:** `promote:` is the ONLY way a window changes the production default, and it happens only after every step passed.
- **Outputs:** a results dir with `summary.json`, `snapshot.json`, `state.json`, `window.log` and `steps/*.log`.

```
windowctl.py validate SPEC | run SPEC --dry-run [--set VAR=VAL] | submit SPEC --wait | status
```

To stop a running window: `systemctl --user stop win-<lane>-<window>`. The restore runs first, then the unit stops.

## 3. Jobs in their own units (`deploy/bin/unitrun.py`)

`unitrun.py run --lane L --job J [--timeout S] [--out F] -- CMD` runs the command as `systemd-run --user --unit=L-J --collect`:

- a second start of a running unit is refused;
- the exit status is written by the unit's own ExecStopPost, so it survives `--collect` and the caller dying;
- the pids in the unit's cgroup are sampled into `~/.local/share/lane-units/` (`<unit>.json`, `history.jsonl`).

Kill a job with `unitrun.py stop UNIT` or `unitrun.py stop --lane L`, never with a `pkill -f` pattern. `owner_of_pid()`
attributes a pid live from its cgroup, or from the registry once the pid is gone. `engine-fault-collector.py` uses it:
foreign Xids (pids outside the engine tree) are recorded in the ledger row as `foreign_xids` with the owning unit and
lane.

## 4. The shared GPU gate for lane benches

`~/projects/lanes/windows/gpuok.sh [GPU] [NEED_MiB]` (also `--wait S`) exits 1 while any of these hold:

- a live busy signal belongs to another window;
- `planned_offline` is True;
- the engine is unhealthy;
- free VRAM is below NEED.

Commands that a window runs itself get `WINDOW_ID` and pass. `gpuguard.py check|apps|busy|kv-pool` is the Python side.

## 5. PYTHONPATH bug (L107)

`serve-hauhaucs-v02.sh` (and `serve-profile-v02.sh`) exported `PYTHONPATH="$V02_ROOT:$FLASHQLA_ROOT"` after sourcing the
override, which silently dropped e.g. K3's `PYTHONPATH=wt-k3`. Now:

- an override that sets PYTHONPATH is prepended and logged;
- a PYTHONPATH inherited from the unit env is not mistaken for an override;
- the venv is resolved (`readlink -f`) so the JIT paths match the release prebuild;
- every boot logs `V02_ROOT=... (release id sha=...)` to the journal.

Both scripts now live on frontier `deploy/bin`. The integrate branch still carries its older copies, but the deployed
copies come from frontier.

## 6. Migration plan: production from wt-integrate to the release pointer

This is one planned window, run by WQ: `deploy/windows/rl-release-migration.yaml`. Nothing changes production before
it, and wt-integrate's working tree is never edited.

0. **Before the window (done 2026-10-03):** the release `bf46537a45-20261003T1501` was built from integrate HEAD and verified. If integrate HEAD has moved by window time, the framework builds a fresh release from the new HEAD before it opens the gateway window (`release: {tree_head: wt-integrate}`). The tree must be clean.
1. **preconditions:** the release verifies, its sha equals wt-integrate HEAD, and the tree is clean.
2. **deploy-serve:** install the new serve scripts, the attributing collector and `unitrun.py` into `~/.local/share/vllm-qwen27b/`. These are snapshot files, put back verbatim on any failure. With no `current` pointer they boot wt-integrate exactly as before.
3. **base-boot** (legacy tree, fresh boot, GPU gate) → **base-root** check → **base-probe:** quick.py, greedy probe, evalkit tool_call.
4. **rel-boot** (the release) → **rel-verify-running:** the engine's cwd is the release, and `verify` still passes after the boot, so nothing was rebuilt → **rel-probe** (same probes).
5. **tree-unchanged**, then **identity-gate** (`release_ab_probe.py compare`): the KV pools are equal, the greedy outputs are byte-identical on every prompt the base reproduced, decode is not slower, and evalkit is not worse.
6. **promote** (only if 1-5 all passed): `release.py activate <id> --no-restart` flips `current`, and the deployed files are kept. The framework's restore boot then puts production on the release and checks the root, the `.so` hashes, the KV pool and health.
7. **Any failure:** no promotion. The serve scripts, collector and `unitrun.py` are restored verbatim, the override is restored, and the legacy tree is rebooted.

Rollback at any later time: `python3 deploy/bin/release.py rollback --reason "..."`. This removes the pointer (back to
the legacy tree) or moves to the previous release, then does a drained restart through the engine actuator.

After the migration:

- production changes only through `release.py build` + `activate` (or a window's `promote`);
- wt-integrate is a dev tree again;
- prune old releases with `release.py rm` (it refuses the current release, the rollback target and the running engine's release).
