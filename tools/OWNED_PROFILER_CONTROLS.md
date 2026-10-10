# Optional owned Nsight collection (source/CPU qualified only)

This isolated controller variant is based on frozen controls5039b5610c3edcc4bc88c70ebe90e1d236a293ed. It does not deploy, wrap the engine or relax MainPID/argv/root/model/native checks. It builds on the upstream vLLM, vLLM2080TiDefinitiveEdition and github.com/weicj control sources under Apache2.0. NVIDIA Nsight remains a separately installed proprietary tool; no vendor binaries are redistributed here.

Only a spec containing `owned_profiler` opts in. All literal paths and hashes are required; use the canonical Nsight2025.6.3 target-linux-x64 binary and resolved library files, not a version-changing symlink. Existing controls5039 and measurement7613 stay unchanged. This is an instrumentation diagnosis; instrumented specs cannot promote.

```yaml
owned_profiler:
  tool: /opt/nvidia/nsight-systems/2025.6.3/target-linux-x64/nsys
  tool_sha256: REPLACE_WITH_EXACT_SHA256
  libraries:
    /opt/nvidia/nsight-systems/2025.6.3/target-linux-x64/libToolsInjection64.so: REPLACE_WITH_EXACT_SHA256
  collection_s: 300
  ready_timeout_s: 30
  cleanup_timeout_s: 30
  graph: node:host-only
  injection_env:
    CUDA_INJECTION_SHM_ALLOWED: 'TRUE'
steps:
  - name: stop-engine
    engine: stop
  - name: collect
    profiler: start
  - name: trace-original
    boot:
      default: true
      env:
        CUDA_INJECTION_SHM_ALLOWED: 'TRUE'
  - name: diagnostic
    run: [python3, /ABSOLUTE/FROZEN/DIAGNOSTIC.py]
  - name: finish-collection
    profiler: stop
```

This is an interface example, not an executable qualified window; supply the normal lane/window/reason/results/offline/producer fields and exact source-qualified diagnostics. The library pathname above must be replaced if it is not an actual canonical installed file. Bind all relevant vendor libraries. The declared environment must also appear explicitly in every traced `boot.env`. Engine generation remains the ordinary controller-owned SYSTEM restart; only new same-UID processes born after collection start are eligible for documented system-wide CUDA collection. A competing system-wide collector can cause silent process-scope fallback. Presence of a ready callback is insufficient evidence of target coverage.

The generated collector argv has no application, `--trace=cuda-sw`, `--cuda-trace-scope=system-wide`, sample/cpuctxsw/backtraces disabled, graph or node host-only scope, bounded duration1..300s, `--stop-on-exit=false`, `--kill=none`, no overwrite and a unique `Halo_<UUID>` session. The transient unit has RuntimeMaxSec=collection_s+120. Before launching, the private manifest and state pointer are written; the callback verifies the manifest/helper/tool/library hashes, exact collector unit, MainPID/start/UID/full-argv hash, session ID and callback environment. PID sampling is inherited from existing `unitrun`; it is not injection ownership authority.

Success, abort and deadman use the same restoration path: revoke measurement, stop ordinary window jobs while excluding the exact collector/deadman, drain, then serialize typed collector cleanup. Instrumented window-prefix matching uses a separator boundary so `trace` cannot stop `trace2`. Shutdown addresses only the recorded session ID with kill=none, and unit stop addresses only its exact service after current ownership checks. Unknown schemas, changed IDs/PIDs/argv, escaped helpers, unreadable evidence, cleanup timeouts and malformed records defer while retaining the gateway recovery fence. A naturally completed collector with no owned session is cleanable. A surviving session after collector death is deferred because ownership cannot safely be reconstructed. Recovery is retryable if session shutdown completed before unit stop.

Any reserved collection forces restoration to freshly boot the exact original engine. Known instrumentation environment values are hashed, not copied into receipts. Process-tree membership is checked twice; mapped declared libraries, files under the canonical tool parent, and known injection/CUPTI/Nsight prefixes are checked by path/inode/device/hash. Original CUPTI mappings may be preserved exactly; existing injection libraries refuse baseline capture. Fresh env/mapping/session/helper checks run before and after the real generation canary, before producer timers/user units/admission resume. Original override/native bytes and normal identity/KV/canary gates remain required.

The session JSON adapter deliberately refuses unknown schemas. Installed help establishes `sessions list --output-format=json`; actual field names, callback-name value, collector MainPID argv, injection permissions, graph event coverage and bounded success/abort/deadman cleanup are not yet runtime qualified. Do not enable this variant until independent review and an owned offline commission. Require actual CUDA/module/launch/copy records for both expected TP worker generations, correlate them with exact loaded native/source/cubin evidence, and distinguish graph-level from node-level coverage. Do not infer performance or SM75 counter permissions from a callback or software trace. Profiling overhead and instrumented timings are excluded from gains/promotion evidence.

Collector/report output stays in a private0700 directory; manifests/receipts are0600. Raw traces/logs may contain newly launched same-UID process metadata and paths. Review before vault/external publication; prefer sanitized event and ownership receipts. Ending collection early via shutdown can cancel an active report; this code makes no report-completeness claim. To obtain a complete duration-bounded report, wait for normal completion before the stop step; the controller still qualifies cleanup and fresh restoration.

Private test invocation (no external tool/service calls):

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /ABSOLUTE/QUALIFIED/PYTHON -m pytest -q deploy/bin/test_owned_profiler.py
```

Source proofs are CPU/mock contracts, not actual NVIDIA protocol, full serving graph, MTP state or performance proof.
