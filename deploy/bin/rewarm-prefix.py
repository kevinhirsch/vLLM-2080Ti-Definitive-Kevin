#!/usr/bin/env python3
"""rewarm-prefix.py -- after an engine (re)start, re-prefill the shared prefixes real clients use.

Every engine fault/restart wipes the prefix cache, so Halo's ~28.5K-token system+tools prefix (and the
pi / Hermes system prompts) must be prefilled again by the first real request of each family, and N
concurrent first requests prefill the SAME prefix N times in parallel. This replays the most recent flight-
recorded body of each family with max_tokens=1 straight to the engine (before/while traffic ramps), so the
cache is warm after ~30-45 s per family instead of paid on the clients' critical path.

Usage: rewarm-prefix.py [--engine URL] [--cap SECS] [--max-families N]
Never fails the caller (exit 0). Logs to warmup.log. Families, newest body each, Halo first.
"""
import argparse, glob, json, os, sys, time, urllib.request

FR = os.path.expanduser("~/.local/share/vllm-qwen27b/flightrec")
LOG = os.path.expanduser("~/.local/share/vllm-qwen27b/warmup.log")
FAMILIES = [  # (name, substring that identifies the first system/developer message)
    ("halo", "I am Halo, the Estate Overseer"),
    ("pi", "coding assistant operating inside pi"),
    ("hermes", "You are Hermes Agent"),
]


def say(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} rewarm: {msg}\n")


def first_text(body):
    m = body.get("messages") or []
    if not m:
        return ""
    c = m[0].get("content") or ""
    if isinstance(c, list):
        c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
    return c[:600]


def newest_per_family():
    best = {}
    for path in sorted(glob.glob(os.path.join(FR, "*.json")), reverse=True):  # epoch-prefixed => newest first
        try:
            body = json.load(open(path))
        except Exception:
            continue
        head = first_text(body)
        for name, needle in FAMILIES:
            if name not in best and needle in head:
                best[name] = (path, body)
        if len(best) == len(FAMILIES):
            break
    return best


def warm(engine, name, path, body, timeout):
    req = dict(body)
    for k in ("max_completion_tokens", "max_tokens", "stream", "stream_options", "store",
              "thinking_token_budget", "reasoning_effort", "n", "logprobs", "top_logprobs"):
        req.pop(k, None)
    req.update({"model": "qwen-local", "max_tokens": 1, "stream": False, "temperature": 0})
    data = json.dumps(req).encode()
    t0 = time.time()
    r = urllib.request.Request(engine + "/v1/chat/completions", data=data,
                               headers={"Content-Type": "application/json", "X-Client": "warmup-prefix"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            out = json.loads(resp.read())
        usage = out.get("usage") or {}
        say(f"{name}: ok {os.path.basename(path)} prompt_tokens={usage.get('prompt_tokens')} "
            f"cached={((usage.get('prompt_tokens_details') or {}).get('cached_tokens'))} in {time.time()-t0:.1f}s")
    except Exception as e:  # noqa: BLE001
        say(f"{name}: FAILED {os.path.basename(path)} after {time.time()-t0:.1f}s: {e!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="http://127.0.0.1:8001")
    ap.add_argument("--cap", type=float, default=300)
    ap.add_argument("--max-families", type=int, default=3)
    a = ap.parse_args()
    t0 = time.time()
    fams = newest_per_family()
    say(f"families found: {sorted(fams)}")
    # LRU order matters: the prefix pool is only ~195 pages (a 46K prompt holds ~50), so warm the least
    # valuable family first and Halo LAST so it is the most-recently-used survivor. Halo is replayed twice:
    # the second pass is a full prefix hit and so also exercises (and JIT-compiles) the continuation path.
    order = ["hermes", "pi", "halo"][-a.max_families:]
    for name in order:
        if name not in fams:
            continue
        for _pass in range(2 if name == "halo" else 1):
            left = a.cap - (time.time() - t0)
            if left < 30:
                say("cap reached; stopping")
                break
            warm(a.engine, name, *fams[name], timeout=left)
    say(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        say(f"crashed: {e!r}")
    sys.exit(0)
