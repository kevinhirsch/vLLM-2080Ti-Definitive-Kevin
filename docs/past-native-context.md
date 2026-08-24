# Past-Native Context (EXP-017)

This document is the reproducible recipe for serving a Qwen3-Next hybrid model
*past its native context window* on 2×2080Ti-22GB. On this fork, Qwen3.8-27B
answered correctly at `352,247` prompt tokens — about `90,103` tokens past the
model's native `262,144` — with the needle placed near the far end of the
extended window.

This is an experimental capability, not a shipped profile. The recipe below is
complete enough to reproduce the result, and the caveats section states honestly
what has and has not been validated.

## Result

| Field | Value |
|---|---:|
| Model | Qwen3.8-27B (hybrid GDN + attention) |
| Native max position | `262,144` |
| Served prompt tokens | `352,247` |
| Tokens past native | `90,103` |
| Needle depth | `~88%` (far end of extended window) |
| Retrieval | correct, coherent continuation |
| Short-context gate | 5/5 pass |
| GPUs | 2×2080Ti-22GB (SM75), NVLink, TP2 |

The needle at `~88%` depth sits well past the native `262,144` boundary, so the
correct retrieval is direct evidence that position handling — not just KV
capacity — extended cleanly.

## Recipe

Three things are required together. Any one missing either crashes with an
illegal memory access or silently truncates position handling.

### 1. RoPE cache-sizing patch

`vllm/model_executor/models/qwen3_next.py`, in the attention block's
`get_rope(...)` call:

```python
self.rotary_emb = get_rope(
    head_size=self.head_dim,
    max_position=int(os.getenv("VLLM_ROPE_MAX_POSITION") or config.max_position_embeddings),  # EXP-017 rope ctx extension
    rope_parameters=config.rope_parameters,
    dual_chunk_attention_config=self.dual_chunk_attention_config,
)
```

Stock vLLM sizes the RoPE cos/sin cache from `config.max_position_embeddings`.
When the served window exceeds that, the position index runs past the end of the
cos/sin table and the attention kernel reads out of bounds — the observed
failure is a CUDA illegal memory access, not a graceful error. The override lets
`VLLM_ROPE_MAX_POSITION` size the cache to the *extended* window so every
in-flight position has a valid cos/sin entry.

The env-or-default form is deliberate: with the env unset, behavior is identical
to stock, so the patch is inert on normal routes.

### 2. Environment

```bash
VLLM_ROPE_MAX_POSITION=524288
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS=393216
```

- `VLLM_ROPE_MAX_POSITION=524288` sizes the cos/sin cache for the full 512K
  ladder, above the `352k` we actually served, so the cache is not the limiter.
- `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` clears the guard that otherwise rejects a
  `--max-model-len` above native.
- `VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS=393216` raises the
  TurboQuant continuation dequant workspace reserve from the default `262144`.
  The `355k`-token prefill needs about `520MB`, which overruns the default
  `512MB` workspace; without the bump the continuation dequant path fails.

### 3. Launch flags

```bash
--hf-overrides '{"rope_parameters":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144, ...}}' \
--max-model-len 393216
```

YaRN with `factor=2.0` over `original_max_position_embeddings=262144` gives a
`524,288` effective ceiling; `--max-model-len 393216` is the served window,
which leaves headroom above the measured `352k` result.

## Validation method

The core check is a needle-in-a-haystack retrieval placed *past* the native
boundary, not merely a long prompt that loads without error.

1. Build a prompt long enough to exceed native (`352,247` tokens here) with a
   unique needle token/sentence inserted at a known deep offset (`~88%`).
2. Submit through the normal serving path (TP2, chunked prefill).
3. Confirm the model retrieves the needle verbatim and continues coherently —
   not a truncated echo, not a repetition collapse.
4. Run a short-context gate (5 unrelated small prompts) to confirm the patched
   build has not regressed normal routes. Result: 5/5 pass.

A memory plateau or a clean load is *not* evidence here, by the same rule the
profile catalog uses: only a correct, coherent retrieval past native counts.

## Why hybrids make this clean

Qwen3-Next is a hybrid: most layers are Gated DeltaNet (GDN), which are
position-agnostic recurrent/state layers, and only a minority carry rotary
position embeddings. In this model **16 of 64 layers carry RoPE**; the other 48
are GDN and do not consult the cos/sin cache at all.

That is why past-native extension behaves so well with a single cache-sizing
change: three quarters of the depth is inherently insensitive to absolute
position, so the only thing that had to be made correct past native was the RoPE
cache in the attention quarter. There is no per-layer positional drift to
accumulate across the GDN stack. A pure-attention model of the same size would
be far more likely to degrade or hallucinate at this extension factor.

## Localization story (in brief)

The working recipe is the survivor of six iterations. The failures were
informative and localized the problem precisely:

1. Raise `--max-model-len` alone → rejected by the long-context guard.
2. Add `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` → loads, then CUDA illegal memory
   access during prefill.
3. The illegal access tracked the attention (not GDN) layers → RoPE cos/sin
   cache indexed out of bounds past native.
4. Add the `get_rope` cache-sizing override → illegal access gone, but the
   continuation dequant path failed on the largest prefill.
5. Trace the dequant failure to the TurboQuant continuation workspace: `355k`
   prefill needs `~520MB` against the default `512MB` reserve.
6. Raise `VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS=393216` →
   full prefill completes, needle retrieved correctly past native.

Each step converted a crash into a more specific crash until the surface was
fully covered.

## Caveats (honest status)

This is a validated experiment, not a promoted profile. Still pending:

- **Fresh-unique-prefix confirm.** The retrieval should be re-run with a
  freshly generated unique prefix to rule out any residual cache effect.
- **Full evalkit.** Only the needle protocol and a 5/5 short-context gate have
  run. A complete quality eval past native is not yet done.
- **Soak.** No long-duration stability soak at the extended window yet; the
  SM75 crash history on this hardware (see the Xid31 work) means soak matters
  before any promotion.

Treat the result as: the position-handling and workspace mechanics are correct
past native, and one deep-needle retrieval passed. Do not treat it as a
production long-context guarantee.

## Next steps

- Walk the YaRN factor ladder toward the full `524,288` ceiling and record the
  first factor where retrieval or coherence degrades.
- Run the full evalkit past native, not only the needle protocol.
- Run a stability soak at the extended window before considering promotion.
