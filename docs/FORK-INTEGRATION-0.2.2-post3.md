# Fork integration: weicj v0.2.2-post3 + Kevin's local layer (2026-10-02)

Branch `integrate-v0.2.2-post3` = upstream `weicj/vLLM-2080Ti-Definitive` `main` @ v0.2.2-post3 (vLLM 0.29.1rc0 base,
torch 2.13 + cu130, Python 3.12, GCC 15) plus:

* upstream PR #238 (avoid equal-block partial hash on TP routes), merged with the #241 refactor;
* ports of the fork's still-needed engine work (see the port ledger, vault note `vLLM Upstream Integration 2026-10-02`):
  cudagraph padded-prefix refresh, scheduler progress-invariant guard + SCHED-DROP + accepted-count preserve (lane EF),
  short-first interleave + prefill share (lane EF2), shm_broadcast deadlock fixes (weicj#124), GDN Triton config pins
  (weicj#107 part), Responses-API leniency (weicj#121/#123), EngineCore env re-hydrate, env knobs
  `VLLM_CUSTOM_ALLREDUCE_MAX_SIZE_MB` and `VLLM_ROPE_MAX_POSITION`;
* the operational layer unchanged from the 0.1.x fork: `deploy/` (capacity gateway, watchdogs, serve scripts, evalkit), research docs.

The 0.1.x fork history is **not** an ancestor of this branch (upstream's 0.2.x tree is unrelated history). The 0.1.x fork tip is
preserved as branch `frontier-pastnative-20260816`; nothing is deleted.

Serve with `deploy/bin/serve-hauhaucs-v02.sh` (same flags as `serve-hauhaucs.sh`; speculative config adds
`disable_eagle_block_drop` which replaces the old `VLLM_MAMBA_ALIGN_RETAIN_MTP_CACHE_BLOCK` patch).
