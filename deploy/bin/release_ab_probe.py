#!/usr/bin/env python3
"""release_ab_probe.py -- the identity gate for switching production to a release (lane RL, 2026-10-03).

A release built from the sha the production tree runs must behave IDENTICALLY to that tree. The migration window boots
both (same config, same GPUs, foreign-app gate before each boot) and this tool decides:

  release_ab_probe.py probe --out FILE [--reps 2]    # greedy (temperature 0) outputs of fixed prompts on :8001
  release_ab_probe.py compare --results DIR --base base --cand release [--tok-tol 0.05] [--kv-tol 0.002]

Gates (all must hold; compare exits 1 otherwise and writes DIR/compare.json):
  * KV pool of the two boots (windowctl summary.json) equal within kv_tol (same code + same free memory = same pool);
  * greedy outputs: every prompt the BASE reproduced across its own reps must come out byte-identical on the candidate
    (prompts the base itself did not reproduce are reported, not gated), and at least 2 prompts must be comparable;
  * natural decode tok/s (quick.py) of the candidate >= base * (1 - tok_tol);
  * evalkit pass count of the candidate >= base (when both ran).
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evalkit"))
import measurement

PROMPTS = [
    {"chat": "List the first eight prime numbers, comma separated, then explain in one sentence why 1 is not prime."},
    {"chat": "Write a Python function that returns the n-th Fibonacci number iteratively. Code only."},
    {"chat": "Summarize the causes of the French Revolution in exactly four bullet points."},
    {"completion": "The quick brown fox jumps over the lazy dog. " * 40 + "In conclusion,"},
]


def _post(path, body, port=8001, timeout=600):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        response = c.getresponse()
        if response.status != 200:
            raise ValueError(f"Generation HTTP {response.status}")
        data = json.loads(response.read())
        check = data
        if path == "/v1/completions" and isinstance(data, dict):
            check = {**data, "choices": [{**choice, "message": {"content": choice.get("text")}}
                     for choice in data.get("choices", [])]}
        error = measurement.generation_error(check)
        if error:
            raise ValueError(f"Generation failed: {error}")
        return data
    finally:
        c.close()


def model_id(port=8001):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        c.request("GET", "/v1/models")
        response = c.getresponse()
        if response.status != 200:
            raise ValueError(f"Model discovery HTTP {response.status}")
        data = json.loads(response.read()).get("data")
        if not data or not data[0].get("id"):
            raise ValueError("Model discovery returned no model")
        return data[0]["id"]
    finally:
        c.close()


def probe(out, reps=2, max_tokens=128, port=8001, engine_pid=None, identity_file=None, native_manifest=None):
    engine_pid = engine_pid or os.environ.get('WINDOW_ENGINE_PID')
    if not engine_pid:
        raise ValueError('Identity probe requires the controller-identified engine PID')
    identity_file = identity_file or os.environ.get('WINDOW_ENGINE_IDENTITY_FILE')
    native_manifest = native_manifest or os.environ.get('WINDOW_NATIVE_MANIFEST')
    measurement.require_offline_window(drained=True)
    provenance = measurement.runtime_identity(int(engine_pid))
    native = measurement.verify_loaded_native(provenance, identity_file, native_manifest)
    measurement.verify_models_root(f'http://127.0.0.1:{port}', provenance)
    m = model_id(port)
    rows = []
    for i, p in enumerate(PROMPTS):
        texts = []
        completion_tokens = []
        for _ in range(reps):
            measurement.require_offline_window()
            measurement.verify_loaded_native(provenance, identity_file, native_manifest)
            if "chat" in p:
                r = _post("/v1/chat/completions", {"model": m, "messages": [{"role": "user", "content": p["chat"]}],
                                                   "max_tokens": max_tokens, "temperature": 0, "seed": 0,
                                                   "chat_template_kwargs": {"enable_thinking": False}}, port)
                texts.append(r["choices"][0]["message"].get("content") or "")
            else:
                r = _post("/v1/completions", {"model": m, "prompt": p["completion"], "max_tokens": max_tokens,
                                              "temperature": 0, "seed": 0}, port)
                texts.append(r["choices"][0].get("text") or "")
            completion_tokens.append(r["usage"]["completion_tokens"])
        rows.append({"i": i, "texts": texts, "completion_tokens": completion_tokens,
                     "generation_verified": True, "prompt_sha256": measurement.canonical_hash(p),
                     "self_consistent": len(set(texts)) == 1})
    if measurement.runtime_identity(int(engine_pid))['fingerprint'] != provenance['fingerprint']:
        raise ValueError('Engine identity changed during identity probe')
    res = {"model": m, "reps": reps, "max_tokens": max_tokens, "prompts": rows,
           "provenance": provenance, "native_attestation": native}
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    return res


def _mean(xs):
    xs = [float(x) for x in xs or []]
    return sum(xs) / len(xs) if xs else None


def _evalkit_passed(path):
    try:
        text = open(path, errors="replace").read()
    except OSError:
        return None
    m = re.findall(r"(\d+)\s*/\s*(\d+)\s+passed", text)
    return sum(int(a) for a, _ in m) if m else None


def compare(results, base="base", cand="release", tok_tol=0.05, kv_tol=0.002):
    def load(name):
        try:
            return json.load(open(os.path.join(results, name)))
        except (OSError, ValueError):
            return None
    summ = load("summary.json") or {}
    boots = {b.get("label"): b for b in summ.get("boots") or []}
    gates, facts = {}, {}
    kb, kc = (boots.get(base) or {}).get("kv_pool"), (boots.get(cand) or {}).get("kv_pool")
    facts["kv_pool"] = {base: kb, cand: kc}
    gates["kv_pool_equal"] = bool(kb and kc and abs(kc - kb) <= kb * kv_tol)
    pb, pc = load(f"probe_{base}.json"), load(f"probe_{cand}.json")
    comparable, mism, skipped = 0, [], []
    def verified(probe):
        rows = (probe or {}).get('prompts') or []
        return bool(rows) and all(row.get('generation_verified') is True and row.get('texts')
            and len(row.get('completion_tokens') or []) == len(row['texts'])
            and all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in row['completion_tokens'])
            and all(isinstance(text, str) and bool(text.strip()) for text in row['texts']) for row in rows)
    gates['generation_verified'] = verified(pb) and verified(pc)
    same_fixtures = bool(pb and pc and len(pb['prompts']) == len(pc['prompts']) and
                         [r['i'] for r in pb['prompts']] == [r['i'] for r in pc['prompts']] and
                         [r.get('prompt_sha256') for r in pb['prompts']] == [r.get('prompt_sha256') for r in pc['prompts']])
    if pb and pc:
        for rb, rc in zip(pb["prompts"], pc["prompts"]):
            if not rb["self_consistent"]:
                skipped.append(rb["i"])
                continue
            comparable += 1
            if not rc["texts"] or any(text != rb["texts"][0] for text in rc["texts"]):
                mism.append({"i": rb["i"], "base": rb["texts"][0][:160], cand: (rc["texts"][0] if rc["texts"] else "")[:160]})
    facts["greedy"] = {"comparable": comparable, "mismatch": mism, "base_not_self_consistent": skipped}
    gates["greedy_identical"] = same_fixtures and comparable >= 2 and not mism
    qb, qc = load(f"quick_{base}.json"), load(f"quick_{cand}.json")
    tb, tc = _mean((qb or {}).get("natural_tok_s")), _mean((qc or {}).get("natural_tok_s"))
    facts["natural_tok_s"] = {base: tb, cand: tc}
    gates["decode_not_slower"] = bool(tb and tc and tc >= tb * (1 - tok_tol))
    eb, ec = _evalkit_passed(os.path.join(results, f"evalkit_{base}.out")), _evalkit_passed(os.path.join(results, f"evalkit_{cand}.out"))
    facts["evalkit_passed"] = {base: eb, cand: ec}
    if eb is not None or ec is not None:
        gates["evalkit_not_worse"] = eb is not None and ec is not None and ec >= eb
    out = {"ok": all(gates.values()), "gates": gates, "facts": facts}
    with open(os.path.join(results, "compare.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("probe")
    p.add_argument("--out", required=True)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--engine-pid", type=int, default=os.environ.get('WINDOW_ENGINE_PID'))
    p.add_argument("--engine-identity-file", default=os.environ.get('WINDOW_ENGINE_IDENTITY_FILE'))
    p.add_argument("--native-manifest", default=os.environ.get('WINDOW_NATIVE_MANIFEST'))
    p = sp.add_parser("compare")
    p.add_argument("--results", required=True)
    p.add_argument("--base", default="base")
    p.add_argument("--cand", default="release")
    p.add_argument("--tok-tol", type=float, default=0.05)
    p.add_argument("--kv-tol", type=float, default=0.002)
    a = ap.parse_args(argv)
    if a.cmd == "probe":
        r = probe(a.out, a.reps, a.max_tokens, engine_pid=a.engine_pid,
                  identity_file=a.engine_identity_file, native_manifest=a.native_manifest)
        print(json.dumps({"model": r["model"], "self_consistent": [x["self_consistent"] for x in r["prompts"]]}))
        return 0
    r = compare(a.results, a.base, a.cand, a.tok_tol, a.kv_tol)
    print(json.dumps(r, indent=1))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
