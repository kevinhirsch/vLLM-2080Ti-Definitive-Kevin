"""fork_research.py — "prefill once, fork N" over the EXP-038 fork engine.

Builds one large (2112+ token) shared research corpus, prefills it ONCE on the
fork-enabled engine (POST /tq/pin), then forks 4 verifier agents that all ride
that shared KV prefix — each at a DIFFERENT sampling temperature (POST /tq/fork)
— and finally synthesizes their verdicts into one calibrated answer.

The whole point: the 2112-token corpus is prefilled a single time; the 4
verifiers pay ~zero prefill compute (they adopt the pinned prefix from cache) and
diverge only in sampling. If the engine lacks the /tq/* routes, fork_agents
falls back TRANSPARENTLY to 4 ordinary agent() calls (each re-sends the corpus).

Run (throwaway engine serving BOTH the /v1 and /tq/* surfaces):

    LOCALFLOW_ENGINE_URL=http://127.0.0.1:8099 \
    LOCALFLOW_GATEWAY=http://127.0.0.1:8099/v1 \
    python ~/localflow/localflow.py ~/localflow/examples/fork_research.py --concurrency 4

The corpus targets >2112 tokens (the Stage-2 cached-prefix figure). /tq/pin
reports prefix_len so you can confirm the target was met.
"""

# A claim the verifiers must adjudicate against the corpus below.
CLAIM = ("Linear-attention models can always replace full attention for "
         "long-context tasks with no loss in retrieval accuracy.")

# Building blocks for a long, self-consistent corpus. We repeat + index these to
# comfortably clear 2112 tokens of shared context (well over ~1600 words).
_FACTS = [
    "Full self-attention computes pairwise interactions between every pair of "
    "tokens, giving quadratic time and memory in the sequence length, which is "
    "the dominant cost driver for long-context inference.",
    "Linear-attention variants replace the softmax kernel with a feature map so "
    "that attention can be computed as a running summary, reducing the per-token "
    "cost to a constant with respect to the sequence length.",
    "Because a linear-attention state is a fixed-size summary, it must compress "
    "the entire past into that state; this compression is lossy and degrades "
    "exact retrieval of a specific earlier token more than full attention does.",
    "State-space and gated-delta-network models keep a recurrent state that is "
    "updated per token, so they stream long inputs cheaply but can forget "
    "precise positional detail that a full-attention head would preserve.",
    "Empirically, on needle-in-a-haystack retrieval, full attention holds near "
    "perfect accuracy across the context window while pure linear attention "
    "falls off as the needle moves further from the query.",
    "Hybrid architectures interleave a few full-attention layers among many "
    "linear or recurrent layers, recovering most retrieval accuracy while "
    "keeping the bulk of the compute sub-quadratic.",
    "Prefix caching lets a server reuse the key-value blocks of a shared prompt "
    "across many requests, so the expensive prefill of a long shared context is "
    "paid once and amortized over every continuation that reuses it.",
    "Forking a pinned prefix into several children with different sampling "
    "parameters is a way to explore multiple continuations of the same context "
    "without recomputing that context for each child.",
    "The compute saved by such a fork is proportional to the shared prefix "
    "length; the longer the shared corpus, the larger the fraction of total "
    "work that is done exactly once rather than N times.",
    "A fixed-size recurrent state caps memory growth, which is attractive for "
    "very long contexts, but the same cap is what bounds how much distinct "
    "detail can be recalled verbatim from far back in the sequence.",
    "Quantized KV caches reduce the memory footprint of full attention and can "
    "narrow, though not eliminate, the memory-cost gap between full and linear "
    "attention for long contexts on constrained hardware.",
    "Whether linear attention is 'good enough' is task-dependent: summarization "
    "and diffuse reasoning tolerate lossy compression far better than exact "
    "lookup, code navigation, or citation-grounded question answering.",
]


def _build_corpus() -> str:
    """Assemble a >2112-token shared corpus by numbering + lightly varying the
    fact blocks across several 'sections'. Deterministic (no RNG)."""
    sections = []
    for s in range(1, 7):  # 6 sections x 12 facts x ~35-45 tokens each
        lines = [f"## Section {s}: evidence group {s}"]
        for i, fact in enumerate(_FACTS, start=1):
            lines.append(f"[{s}.{i}] {fact}")
        sections.append("\n".join(lines))
    header = ("RESEARCH CORPUS — attention mechanisms for long context.\n"
              "The following numbered findings are the ONLY admissible evidence.\n")
    return header + "\n\n".join(sections)


async def run(args):
    phase("Corpus")
    corpus = _build_corpus()
    words = len(corpus.split())
    log(f"corpus ~{words} words (~{int(words / 0.75)} tok est; target > 2112)")

    shared = (
        corpus
        + "\n\n=== TASK ===\n"
        + f"Using ONLY the corpus above, judge this claim: \"{CLAIM}\"\n"
        + "Answer SUPPORTED / PARTIALLY SUPPORTED / REFUTED and justify in "
          "3-4 sentences citing section numbers like [3.5]."
    )

    phase("Fork verify")
    # Same shared prefix; 4 verifiers diverge ONLY in sampling temperature.
    specs = [
        {"temperature": 0.0, "max_tokens": 220, "label": "verifier@0.0"},
        {"temperature": 0.3, "max_tokens": 220, "label": "verifier@0.3"},
        {"temperature": 0.7, "max_tokens": 220, "label": "verifier@0.7"},
        {"temperature": 1.0, "max_tokens": 220, "label": "verifier@1.0"},
    ]
    verdicts = await fork_agents(shared, specs)
    for spec, v in zip(specs, verdicts):
        head = (v or "").replace("\n", " ")[:110]
        log(f"{spec['label']}: {head}")

    phase("Synthesize")
    joined = "\n\n".join(
        f"[{spec['label']}]\n{v}" for spec, v in zip(specs, verdicts) if v)
    if not joined:
        log("no verdicts returned — nothing to synthesize")
        return {"n_forks": len(specs), "verdicts": verdicts, "synthesis": None}

    synthesis = await agent(
        "Four verifier agents independently judged the SAME claim against the "
        "same evidence, each at a different sampling temperature. Synthesize "
        "their verdicts into ONE calibrated answer. State the consensus verdict, "
        "note any disagreement between the hotter and colder samples, and keep "
        f"it to ~5 sentences.\n\nClaim: {CLAIM}\n\nVerdicts:\n{joined}",
        label="synthesize", phase="Synthesize")

    return {
        "claim": CLAIM,
        "n_forks": len(specs),
        "verdicts": verdicts,
        "synthesis": synthesis,
    }
