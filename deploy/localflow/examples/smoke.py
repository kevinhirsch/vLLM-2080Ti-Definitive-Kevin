"""smoke.py — minimal localflow validation: 3-lane fan-out + schema synthesis.

Cheap, fast. Proves: agent() text calls, parallel() barrier, guided-JSON schema
output, concurrency, journal. Run:
    python ~/localflow/localflow.py ~/localflow/examples/smoke.py --concurrency 4
"""

SYNTH_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {"type": "string"},
        "one_line_reason": {"type": "string"},
    },
    "required": ["winner", "one_line_reason"],
    "additionalProperties": False,
}

LANES = [
    "In ONE sentence, argue why Python is the best language for a beginner.",
    "In ONE sentence, argue why Rust is the best language for a beginner.",
    "In ONE sentence, argue why JavaScript is the best language for a beginner.",
]


async def run(args):
    phase("Fan-out")
    takes = await parallel([
        (lambda p=p: agent(p, label=p[:24], phase="Fan-out"))
        for p in LANES
    ])
    failed = [i for i, t in enumerate(takes) if not t]
    if failed:
        raise RuntimeError(f"fan-out lane(s) failed: {failed}")
    for i, t in enumerate(takes):
        log(f"lane {i}: {t[:80]}")

    phase("Synthesize")
    joined = "\n".join(f"- {t}" for t in takes)
    verdict = await agent(
        "Three one-sentence arguments for a beginner's first language:\n"
        f"{joined}\n\nPick the single most persuasive one. Return JSON "
        "{winner, one_line_reason}.",
        schema=SYNTH_SCHEMA, label="synthesize", phase="Synthesize")
    return {"lanes": takes, "verdict": verdict}
