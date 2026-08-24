"""deep_research.py — the deep-research shape as a local workflow (EXP-043 seed).

scope → fan-out N search-angle agents → extract falsifiable claims →
3-vote adversarial verify (2/3 refutes kills a claim) → synthesize a cited answer.

This mirrors the proven Claude-Code deep-research harness, running local-with-
overflow. P1 uses the model's own knowledge for the "search" step (no web tool
yet); the pi-backed web-fetch backend is P3 (see SPEC §8). Even knowledge-only,
it exercises every orchestration primitive at fan-out scale.

    python ~/localflow/localflow.py ~/localflow/examples/deep_research.py \
        --concurrency 12 --journal /tmp/dr.jsonl \
        --args '"What are the tradeoffs of linear-attention vs full-attention for long context?"'
"""

ANGLES_SCHEMA = {
    "type": "object",
    "properties": {"angles": {"type": "array", "items": {"type": "string"},
                              "minItems": 4, "maxItems": 6}},
    "required": ["angles"], "additionalProperties": False,
}
CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {"claims": {"type": "array",
                              "items": {"type": "string"}, "maxItems": 5}},
    "required": ["claims"], "additionalProperties": False,
}
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"refuted": {"type": "boolean"}, "why": {"type": "string"}},
    "required": ["refuted", "why"], "additionalProperties": False,
}


async def run(args):
    question = args or "What are the tradeoffs of linear vs full attention for long context?"
    log(f"question: {question}")

    phase("Scope")
    scoped = await agent(
        f"Decompose this research question into distinct search angles:\n{question}\n"
        "Return JSON {angles:[...]} — 4 to 6 crisp, non-overlapping angles.",
        schema=ANGLES_SCHEMA, label="decompose", phase="Scope")
    angles = (scoped or {}).get("angles", [])[:6]
    log(f"{len(angles)} angles")

    # Search + extract as a per-item pipeline (no barrier): each angle is
    # researched then mined for claims independently.
    phase("Search+Extract")
    def _search(angle, _item, idx):
        return agent(
            f"Research this angle of the question '{question}':\n{angle}\n"
            "Write a dense, factual paragraph (your best knowledge).",
            label=f"search[{idx}]", phase="Search+Extract")
    def _extract(brief, angle, idx):
        if not brief:
            return {"claims": []}
        return agent(
            f"From this research brief, extract up to 5 FALSIFIABLE claims "
            f"(each independently checkable):\n{brief}\nReturn JSON {{claims:[...]}}.",
            schema=CLAIMS_SCHEMA, label=f"extract[{idx}]", phase="Search+Extract")
    extracted = await pipeline(angles, _search, _extract)
    claims = []
    for e in extracted:
        if e:
            claims.extend(e.get("claims", []))
    # dedup (plain code, not an agent)
    seen, uniq = set(), []
    for c in claims:
        k = c.strip().lower()[:80]
        if k and k not in seen:
            seen.add(k); uniq.append(c)
    log(f"{len(uniq)} unique claims to verify")

    phase("Verify")
    async def _verify_one(claim):
        votes = await parallel([
            (lambda c=claim: agent(
                f"Try to REFUTE this claim. Default refuted=true if uncertain.\n"
                f"Claim: {c}\nReturn JSON {{refuted, why}}.",
                schema=VERDICT_SCHEMA, label="refute", phase="Verify"))
            for _ in range(3)
        ])
        refutes = sum(1 for v in votes if v and v.get("refuted"))
        return {"claim": claim, "survives": refutes < 2, "refutes": refutes}
    verified = await parallel([(lambda c=c: _verify_one(c)) for c in uniq])
    survivors = [v["claim"] for v in verified if v and v["survives"]]
    log(f"{len(survivors)}/{len(uniq)} claims survived 3-vote verification")

    phase("Synthesize")
    body = "\n".join(f"- {s}" for s in survivors)
    answer = await agent(
        f"Question: {question}\n\nVerified claims:\n{body}\n\n"
        "Write a clear, well-structured answer grounded ONLY in these claims.",
        label="synthesize", phase="Synthesize")
    return {"question": question, "angles": angles,
            "claims_total": len(uniq), "claims_survived": len(survivors),
            "answer": answer}
