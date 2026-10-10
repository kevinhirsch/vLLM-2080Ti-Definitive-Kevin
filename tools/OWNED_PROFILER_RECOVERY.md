# Exact offline lease recovery for optional profiler windows

This is a source-only derivative of frozen bfeed710, built on the vLLM 2080 Ti Definitive Edition control framework and upstream vLLM (github.com/weicj attribution retained). It has not been activated or used for collection. The original bfeed source, manifests and first commissioning specs remain unchanged and unqualified.

The installed gateway is newer than this repository's historical gateway source. tools/patch-current-gateway-offline.py requires installed base SHA84efdf5beb10df20a941f911901885e89a4c77f46f923148588ad705def97b34 and produces a fresh separate candidate. It changes only _offline_status and gateway_offline. Never deploy this worktree's historical deploy/bin/keepalive-shim.py as the gateway candidate.

## Server and client contract

A POST body containing a lease always means extension of that exact currently active lease. An expired, absent or mismatched lease returns409; it never opens a replacement. POST without a lease continues to open ordinary planned-offline windows. GET and mutation responses advertise offline_lease_renewal=extend-only-v1.

The precondition and mutation run together after the handler's last await. Therefore DELETE followed by an already-started but not yet parsed renewal cannot reopen the fence. Client reads or a silently ignored renew_only field do not provide this property. The existing persisted planned-offline authority and admin token API remain the sole authority.

Instrumented controller preflight refuses unsupported gateways before taking snapshots or pausing producers. Default-off controller paths retain their original behavior. Gateway supplied-lease semantics necessarily affect all renewal clients: lost authority now refuses instead of becoming a new lease. The shipped ordinary gateway-offline.py open path omits the lease field and is compatible. Gateway activation still requires caller review and publisher/runtime qualification by root.

## Recovery ownership and ordering

The controller retains its ordinary renewal thread and joins it before restoration. Failure to join defers restoration and leaves the original lease/hold/deadman. It never pretends an in-flight request completed.

Instrumented restore/deadman take a private per-results flock and reread state.json under it. Stale, unreadable, already-restored or foreign authority refuses without HTTP, producer or state mutations. A competing deadman does not overwrite the owner state or summary. Exact authority is never adopted from a replacement response.

For this opt-in path, deadman reads the controller's start ticks and task state together. A matching zombie or dead task (`Z`, `X`, `x`) enters recovery; it cannot run its finally and is never treated as a living owner. It retains the exact recorded engine-hold capability because the deployed actuator's `_hold_owner_dead` still recognizes matching start ticks as valid, even for a zombie. The owned restore supplies that capability to the existing restart and releases it before offline close; it does not assume it is void. A reused or absent PID is gone and its owner-scoped hold is cleared as before. Unreadable/malformed process authority defers without signals or restoration. The uninstrumented legacy path remains unchanged.

The keeper refreshes the original lease before collector cleanup and renews it through long restoration. It joins before DELETE, obtains a final acknowledged exact extension, and performs the exact DELETE. No keeper POST occurs afterward. A409 is unqualified. Closed/restored state is published before releasing recovery ownership. Guardian publication is the sole instrumented state write during restore, including verified enter-failure; no outer Window/deadman write can overwrite a newer recovery owner after unlock. Missing instrumented configuration refuses legacy fallback. The helper is included in the immutable controller source snapshot.

A genuine collector/schema cleanup deferral retains original receipts and refreshes the same TTL for the next normal deadman attempt. This is not a new adapter, ownership migration, cleanup proof, or permission to alter a frozen manifest. Any recovery failure re-pauses only originally active, maintenance-declared producer units/timers; unknown ownership does not stop unrelated units. The engine hold is released while the offline lease is still active, so a hold refusal can retry without starting another engine recovery after admission reopened.

## Source proof limits

CPU contracts cover both handler POST/DELETE interleavings, expired/foreign/supplied-empty leases, ordinary opening and draining behavior; missing capability, keeper errors, in-flight joins, exact authority through cleanup deferral, cross-process/stale deadman refusal, private state publication, hold failure, and default-off restoration. Handler tests execute only the pinned installed function AST with fake dependencies. They do not import or run the installed gateway.

Real admin/API responses, disk persistence, actual MainPID/callback/session schema, Nsight helper lifetime, producer maintenance, success/abort/deadman recovery and fresh uninstrumented generation remain unqualified. A finite TTL is not infinite fencing during gateway unavailability. A failed extension forbids qualified resume; root must preserve supervised existing authority and inspect remaining lifetime. No first collector is authorized by this source receipt alone.

If exact DELETE succeeds server-side but its acknowledgement is lost, or its final status readback fails, automatic recovery deliberately remains unqualified. The old lease may already be closed; the helper never opens or adopts a replacement. Preserve this manual recovery limit rather than advertising automatic recovery from every transport failure.

See the source/provenance report in frontier-lab/reports/2026-10-10/source-audit/owned-profiler-recovery-readiness.
