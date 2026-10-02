"""
Deterministic long-context corpus builder for long_ctx items.

Concatenates .py files from the vLLM source tree (sorted path order, so the
result is stable) up to a target char budget (target_tokens * CHARS_PER_TOKEN,
per the brief's ~3.3 chars/token approximation -- no tokenizer dependency),
plants three sentinel facts at 20%/50%/80% char-depth, and caches both the
corpus text and its metadata (sentinel ids/numbers) under corpus/ so repeated
runs reuse exactly the same prompt instead of rebuilding it.

Determinism: file gathering order is a sorted rglob (stable across runs on
an unchanged source tree) and sentinel numbers come from
random.Random(seed).randint(...) with a seed string derived only from
(target_tokens, depth) -- so even a from-scratch rebuild reproduces the same
corpus content and the same expected answers.
"""
import json
import os
import random
from pathlib import Path

from config import VLLM_SOURCE_DIR, CHARS_PER_TOKEN

CORPUS_DIR = Path(__file__).parent / "corpus"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEPTHS = (20, 50, 80)
SENTINEL_LOW, SENTINEL_HIGH = 100000, 999999


def _gather_source_text(max_chars):
    src = Path(VLLM_SOURCE_DIR)
    files = sorted(src.rglob("*.py"))
    if not files:
        raise RuntimeError(f"no .py files found under {src}")
    chunks = []
    total = 0
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        chunk = f"\n# ==== FILE: {f.relative_to(src)} ====\n" + text
        chunks.append(chunk)
        total += len(chunk)
        if total >= max_chars:
            break
    full = "".join(chunks)
    if len(full) < max_chars:
        # Source tree smaller than requested budget (only matters for huge
        # target sizes) -- pad deterministically by repeating.
        reps = (max_chars // max(len(full), 1)) + 1
        full = full * reps
    return full[:max_chars]


def _seeded_number(target_tokens, depth):
    rng = random.Random(f"sentinel-{target_tokens}-{depth}")
    return rng.randint(SENTINEL_LOW, SENTINEL_HIGH)


def build_corpus(target_tokens):
    """Return (text, meta), building+caching on first call for this size."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = CORPUS_DIR / f"corpus_{target_tokens}tok.txt"
    meta_path = CORPUS_DIR / f"corpus_{target_tokens}tok.meta.json"
    if txt_path.exists() and meta_path.exists():
        return txt_path.read_text(), json.loads(meta_path.read_text())

    max_chars = int(target_tokens * CHARS_PER_TOKEN)
    base = _gather_source_text(max_chars)

    inserts = []
    for depth in DEPTHS:
        offset = int(len(base) * depth / 100)
        nl = base.find("\n", offset)
        if nl == -1:
            nl = offset
        number = _seeded_number(target_tokens, depth)
        sid = f"SENTINEL-{target_tokens}-{depth}"
        line = f"\n# {sid}: the magic number is {number}\n"
        inserts.append((nl, line, sid, number, depth))

    # Insert highest offset first so earlier offsets aren't shifted by
    # already-inserted text.
    inserts.sort(key=lambda x: x[0], reverse=True)
    text = base
    for nl, line, _sid, _number, _depth in inserts:
        text = text[:nl] + line + text[nl:]

    sentinels = {
        str(depth): {"id": sid, "number": number}
        for _nl, _line, sid, number, depth in inserts
    }
    meta = {
        "target_tokens": target_tokens,
        "char_len": len(text),
        "approx_tokens": round(len(text) / CHARS_PER_TOKEN),
        "chars_per_token": CHARS_PER_TOKEN,
        "sentinels": sentinels,
        "source_dir": os.path.relpath(VLLM_SOURCE_DIR, REPO_ROOT),
    }
    txt_path.write_text(text)
    meta_path.write_text(json.dumps(meta, indent=2))
    return text, meta


if __name__ == "__main__":
    # Manual cache-warm / inspection: python3 corpus.py
    for tok in (8000, 16000, 32000, 64000, 96000, 128000):
        _text, m = build_corpus(tok)
        print(tok, "->", m["approx_tokens"], "approx tokens,", m["char_len"], "chars")
        for depth, s in m["sentinels"].items():
            print(f"   depth {depth}%: {s['id']} = {s['number']}")
