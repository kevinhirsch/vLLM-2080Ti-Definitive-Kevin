#!/usr/bin/env python3
"""Docs and dashboard vs reality: config drift report -- lane CFG, 2026-10-03.

The vault notes that describe the LIVE config (Usage Contract, Qwen3.8 Live Config, Lane Guarantee, Agent
Configuration Standard, Pi Web UI, vLLM Upstream Integration) and the gateway dashboard's labels drifted from the
running system: a dashboard showed 500k while 922k was live, an audit found 26 of 63 dashboard items wrong or stale,
a ledger said VLLM_CUSTOM_ALLREDUCE_MAX_SIZE_MB=32 was live when no layer set it. This tool reads each claim and
compares it with a MEASURED fact:

  gateway   GET /gateway/config (live knob values), /gateway/capacity (live KV pool, token budget, prefill),
            falling back to shim.env when the gateway is down
  engine    engine_config_check.py: the api_server's /proc environ + argv (authoritative), the resolved chain
  dashboard gateway_dashboard.html: every form field f_<x> must be a live config field, every cfg.<x> the page
            reads must exist, and a hint that says "N = ... (default)" must match the schema default

Claims are (a) a registry of precise regexes per note, scoped to the note's CURRENT section, and (b) every
KEY=VALUE for SHIM_/VLLM_/V02_ keys inside those scopes. Read-only; prints text or --json / --markdown.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gateway_config_schema as GS  # noqa: E402

VAULT = Path(os.environ.get("CFG_VAULT_MEMORY", "/home/kevin/Obsidian/Memory"))
GATEWAY = os.environ.get("CFG_GATEWAY", "http://127.0.0.1:8000")
DASHBOARD = HERE / "gateway_dashboard.html"

# scope: ("all",) | ("description",) | ("section", heading_regex) -> from the LAST matching heading to the next
# '## ' heading | ("line", regex) -> lines matching regex.  kind: int | float | str | present
CLAIMS = [
    # Local LLM Usage Contract (rewritten to the live engine 2026-10-02 23:40: whole note is "current")
    ("Local LLM Usage Contract", ("all",), r"KV pool:\*\* \*\*([\d,]+) tokens", "cap.kv_pool_live", "int"),
    ("Local LLM Usage Contract", ("all",), r"`--max-num-seqs (\d+)`", "argv.--max-num-seqs", "int"),
    ("Local LLM Usage Contract", ("all",), r"util (0\.\d+)", "argv.--gpu-memory-utilization", "float"),
    ("Local LLM Usage Contract", ("all",), r"`max_model_len` ([\d,]+)", "argv.--max-model-len", "int"),
    ("Local LLM Usage Contract", ("all",), r"gateway budget (\d+) \(", "gw.local_budget", "int"),
    ("Local LLM Usage Contract", ("all",), r"\(\+(\d+) tiny", "gw.tiny_extra_lanes", "int"),
    ("Local LLM Usage Contract", ("all",), r"(\d+) reserved for foreground", "gw.fg_reserved", "int"),
    ("Local LLM Usage Contract", ("all",), r"clamps local generation to ([\d,]+)", "gw.local_max_out", "int"),
    ("Local LLM Usage Contract", ("all",), r"first-token caps are (\d+) s \(default\)", "gw.first_token_max", "float"),
    ("Local LLM Usage Contract", ("all",), r"/ (\d+) s \(local-first\)", "gw.local_first_first_token_max", "float"),
    ("Local LLM Usage Contract", ("all",), r"Micro-calls \(≤ ([\d,]+) tokens\)", "gw.tiny_tokens", "int"),
    ("Local LLM Usage Contract", ("all",), r"prompt > \*\*([\d,]+)\*\* tokens → DeepSeek", "gw.local_context_limit", "int"),
    ("Local LLM Usage Contract", ("all",), r"predicted-uncached prompt > ([\d,]+) tokens", "gw.big_prompt", "int"),
    ("Local LLM Usage Contract", ("all",), r"spend cap \$(\d+)/day", "gw.spend_cap_usd", "float"),
    ("Local LLM Usage Contract", ("all",), r"· (\d+) local lanes ·", "gw.local_budget", "int"),
    ("Local LLM Usage Contract", ("all",), r"(--language-model-only)", "flag.--language-model-only", "present"),
    # Qwen3.8 Live Config: frontmatter description + the newest CURRENT block
    ("Qwen3.8 Live Config", ("description",), r"KV pool ([\d,]*\d)", "cap.kv_pool_live", "int"),
    ("Qwen3.8 Live Config", ("section", r"^## CURRENT as of"), r"KV pool: ([\d,]+) tokens", "cap.kv_pool_live", "int"),
    ("Qwen3.8 Live Config", ("section", r"^## CURRENT as of"), r"`--max-num-seqs (\d+)`", "argv.--max-num-seqs", "int"),
    ("Qwen3.8 Live Config", ("section", r"^## CURRENT as of"), r"util (0\.\d+)", "argv.--gpu-memory-utilization", "float"),
    ("Qwen3.8 Live Config", ("section", r"^## CURRENT as of"), r"MNBT (\d+)", "argv.--max-num-batched-tokens", "int"),
    ("Qwen3.8 Live Config", ("section", r"^## CURRENT as of"), r"`max_model_len` ([\d,]+)", "argv.--max-model-len", "int"),
    ("Qwen3.8 Live Config", ("section", r"^## CURRENT as of"), r"`SHIM_POOL_TOKENS` \(([\d,]+)\)", "gw.pool_tokens", "int"),
    ("Qwen3.8 Live Config", ("section", r"^## CURRENT as of"), r"`SHIM_TOKEN_BUDGET` \(([\d,]+)\)", "gw.token_budget", "int"),
    # Local Context Lane Guarantee: the "Current numbers" section
    ("Local Context Lane Guarantee", ("section", r"^## Current numbers"), r"KV pool ([\d,]+) tokens", "cap.kv_pool_live", "int"),
    ("Local Context Lane Guarantee", ("section", r"^## Current numbers"), r"`--max-num-seqs (\d+)`", "argv.--max-num-seqs", "int"),
    ("Local Context Lane Guarantee", ("section", r"^## Current numbers"), r"admission budget \*\*(\d+)\*\*", "gw.local_budget", "int"),
    ("Local Context Lane Guarantee", ("section", r"^## Current numbers"), r"PREDICTED-UNCACHED tokens > ([\d,]+)", "gw.big_prompt", "int"),
    ("Local Context Lane Guarantee", ("section", r"^## Current numbers"), r"`SHIM_POOL_TOKENS` ([\d,]+)", "gw.pool_tokens", "int"),
    ("Local Context Lane Guarantee", ("section", r"^## Current numbers"), r"`SHIM_TOKEN_BUDGET` ([\d,]+)", "gw.token_budget", "int"),
    # Local vLLM Endpoint And Pi Web UI: the "Current:" line
    ("Local vLLM Endpoint And Pi Web UI", ("line", r"^Do not act on the 08-15 state"), r"Current:.*?(\d+) seqs", "argv.--max-num-seqs", "int"),
    ("Local vLLM Endpoint And Pi Web UI", ("line", r"^Do not act on the 08-15 state"), r"Current:.*?KV pool ([\d,]*\d)", "cap.kv_pool_live", "int"),
    ("Local vLLM Endpoint And Pi Web UI", ("line", r"^Do not act on the 08-15 state"), r"Current:.*?gateway budget (\d+)", "gw.local_budget", "int"),
    # Estate Agent Configuration Standard
    ("Estate Agent Configuration Standard", ("all",), r"\| output cap \| gateway `local_max_out` \| \*\*([\d,]+)\*\*", "gw.local_max_out", "int"),
    # vLLM Upstream Integration 2026-10-02 (port ledger rows that claim live state)
    ("vLLM Upstream Integration 2026-10-02", ("all",), r"VLLM_CUSTOM_ALLREDUCE_MAX_SIZE_MB` \(default = upstream 8 MiB\); live env sets (\d+)",
     "env.VLLM_CUSTOM_ALLREDUCE_MAX_SIZE_MB", "int"),
    ("vLLM Upstream Integration 2026-10-02", ("all",), r"serve scripts already pin `VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=(\d)`",
     "env.VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH", "int"),
]
# KEY=VALUE claims are additionally harvested inside these scopes
KV_SCOPES = {
    "Qwen3.8 Live Config": ("section", r"^## CURRENT as of"),
    "Local LLM Usage Contract": ("all",),
    "Local Context Lane Guarantee": ("section", r"^## Current numbers"),
    "Estate Agent Configuration Standard": ("all",),
}
UI_ONLY_FIELDS = {"preset"}      # dashboard helpers that are not config fields (the preset picker fills base+model)
_KV = re.compile(r"\b((?:SHIM|VLLM|V02)_[A-Z0-9_]+)\s*=\s*`?([A-Za-z0-9_.:/-]+)")


# ------------------------------------------------------------------ facts
def _get(path, timeout=5):
    try:
        with urllib.request.urlopen(GATEWAY + path, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def gather_facts(engine=True):
    facts, notes = {}, []
    cfg = _get("/gateway/config")
    if cfg:
        for k, v in cfg.items():
            facts["gw." + k] = v
    else:
        notes.append("gateway /gateway/config unreachable: gateway facts from shim.env")
        entries, _ = GS.parse_env_file(os.path.expanduser("~/.local/share/vllm-qwen27b/shim.env"))
        for k, (_, v) in GS.last_wins(entries).items():
            if k.startswith("SHIM_") and not GS.SCHEMA.get(k, GS.Key(k, "str", None)).secret:
                facts["gw." + k[5:].lower()] = v
    cap = _get("/gateway/capacity")
    if cap:
        cm = cap.get("capacity_model") or {}
        pool = cm.get("kv_pool_tokens") or {}
        tb = cm.get("token_budget") or {}
        facts["cap.kv_pool_live"] = pool.get("live") or pool.get("effective")
        facts["cap.token_budget_effective"] = tb.get("effective")
        facts["cap.prefill_pure"] = (cap.get("throughput") or {}).get("prefill_pure_tok_s")
    else:
        notes.append("gateway /gateway/capacity unreachable")
    if engine:
        try:
            import engine_config_check as E
            chain = E.build_chain()
            pid = chain["main_pid"]
            env, _ = E.proc_env(pid) if pid else ({}, None)
            argv = E.proc_argv(pid) or []
            for k, v in (env or {}).items():
                if k.startswith(("VLLM_", "V02_")):
                    facts["env." + k] = v
            facts["env.__pid__"] = pid
            if not pid or env is None:
                notes.append("engine not running (MainPID %s): engine claims unresolved" % pid)
                facts.pop("env.__pid__", None)
                env, argv = {}, []
            for flag, val in E._flags(argv).items():
                facts["argv." + flag] = val
                facts["flag." + flag] = True
            for k, v in (chain.get("final_env") or {}).items():
                if k.startswith("V02_"):
                    facts.setdefault("env." + k, v)
            # an unset V02_* knob means the serve script's own ${V02_X:-default}
            try:
                script = Path(chain.get("serve_script") or "").read_text()
                for k, d in re.findall(r"\$\{(V02_\w+):-([^}]*)\}", script):
                    facts.setdefault("env." + k, d)
            except OSError:
                pass
        except Exception as e:
            notes.append("engine facts unavailable: %r" % (e,))
    return facts, notes


# ------------------------------------------------------------------ scoping
def note_text(name):
    p = VAULT / (name + ".md")
    try:
        return p.read_text()
    except OSError:
        return None


def scoped(text, scope):
    if scope[0] == "all":
        return text
    if scope[0] == "description":
        m = re.search(r"^description:\s*(.+)$", text, re.M)
        return m.group(1) if m else ""
    if scope[0] == "line":
        return "\n".join(l for l in text.splitlines() if re.search(scope[1], l))
    if scope[0] == "section":
        heads = [m for m in re.finditer(scope[1], text, re.M)]
        if not heads:
            return ""
        start = heads[-1].start()
        nxt = re.search(r"^## ", text[start + 3:], re.M)
        return text[start: start + 3 + nxt.start()] if nxt else text[start:]
    return text


def _num(s):
    try:
        return float(str(s).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def compare(doc, live, kind):
    if kind == "present":
        return None if live is None else live is True
    if live is None:
        return None
    if kind in ("int", "float"):
        a, b = _num(doc), _num(live)
        if b is None:
            return False                         # e.g. live "auto" vs a documented number
        return a is not None and abs(a - b) < 1e-9
    return str(doc).strip().lower() == str(live).strip().lower()


def check_notes(facts):
    rows = []
    for note, scope, rx, fact, kind in CLAIMS:
        text = note_text(note)
        if text is None:
            rows.append(dict(source=note, claim=rx, fact=fact, doc=None, live=None, status="note-missing"))
            continue
        for m in re.finditer(rx, scoped(text, scope)):
            doc = m.group(1)
            live = facts.get(fact, "<absent>" if fact.startswith("env.") and facts.get("env.__pid__") else None)
            if fact.startswith(("env.", "argv.", "flag.")) and not facts.get("env.__pid__"):
                live = None                              # engine down: nothing to compare against
            ok = compare(doc, None if live == "<absent>" else live, kind) if live != "<absent>" else False
            rows.append(dict(source=note, claim=m.group(0)[:90], fact=fact, doc=doc, live=live,
                             status="ok" if ok else ("unresolved" if ok is None else "MISMATCH")))
    for note, scope in KV_SCOPES.items():
        text = note_text(note)
        if not text:
            continue
        for m in _KV.finditer(scoped(text, scope)):
            k, v = m.group(1), m.group(2)
            if k.startswith("SHIM_"):
                key = GS.SCHEMA.get(k)
                if key is not None and key.secret:
                    continue
                fact = "gw." + k[5:].lower()
                live = facts.get(fact)
                if live is None:
                    continue                         # not a dashboard-tunable field: no live view to compare
                ok = GS.same_value(k, str(v), str(live)) if key else str(v) == str(live)
            else:
                fact = "env." + k
                if not facts.get("env.__pid__"):
                    continue
                live = facts.get(fact, "<absent>")
                ok = str(v) == str(live)
            rows.append(dict(source=note, claim=m.group(0)[:90], fact=fact, doc=v, live=live,
                             status="ok" if ok else "MISMATCH"))
    return rows


# ------------------------------------------------------------------ dashboard
def check_dashboard(facts, path=DASHBOARD):
    rows = []
    try:
        html = Path(path).read_text()
    except OSError as e:
        return [dict(source="dashboard", claim=str(e), status="unresolved")]
    live_fields = {k[3:] for k in facts if k.startswith("gw.")}
    if not live_fields:
        return [dict(source="dashboard", claim="gateway config unavailable", status="unresolved")]
    write_only = set(re.findall(r"id=f_([a-z0-9_]+) type=password", html))     # secrets: never echoed back, by design
    for f in sorted(set(re.findall(r"id=f_([a-z0-9_]+)", html))):
        if f in write_only or f in UI_ONLY_FIELDS:
            continue
        if f not in live_fields:
            rows.append(dict(source="dashboard", claim="form field f_%s" % f, fact="gw." + f, doc="input", live="<no such config field>",
                             status="MISMATCH"))
    for f in sorted(set(re.findall(r"\b(?:cfg|CFG)\.([a-z_][a-z0-9_]*)", html))):
        if f not in live_fields:
            rows.append(dict(source="dashboard", claim="page reads cfg.%s" % f, fact="gw." + f, doc="read",
                             live="<not in /gateway/config>", status="MISMATCH"))
    # hints that state a default: "<label>...<span class=hint>N = ... (default) &middot; ...</span><input id=f_x"
    for m in re.finditer(r'<span class=hint>(.*?)</span>\s*<input id=f_([a-z0-9_]+)', html, re.S):
        hint, field = m.group(1), m.group(2)
        key = GS.SCHEMA.get("SHIM_" + field.upper())
        if key is None or key.default is None:
            continue
        for seg in re.split(r"&middot;|·", hint):
            dm = re.match(r"\s*(-?\d+(?:\.\d+)?)\s*=.*\bdefault\b", seg)
            if dm and _num(dm.group(1)) != _num(key.default):
                rows.append(dict(source="dashboard", claim="hint for f_%s: %s" % (field, re.sub(r"<[^>]+>", "", seg).strip()[:90]),
                                 fact="schema default SHIM_%s" % field.upper(), doc=dm.group(1), live=key.default, status="MISMATCH"))
    # configured numbers the page shows next to a live counterpart
    pairs = [("gw.pool_tokens", "cap.kv_pool_live", "KV pool row (shows configured SHIM_POOL_TOKENS)")]
    for conf, live, label in pairs:
        a, b = facts.get(conf), facts.get(live)
        if a is not None and b is not None and _num(a) != _num(b):
            rows.append(dict(source="dashboard", claim=label, fact=live, doc=a, live=b, status="MISMATCH"))
    return rows


def render_text(rows, notes):
    for n in notes:
        print("NOTE", n)
    w = max([len(r.get("source", "")) for r in rows] + [10])
    for r in rows:
        print("%-10s %-*s %-34s doc=%-14s live=%s   [%s]" % (r["status"], w, r.get("source", ""), (r.get("fact") or "")[:34],
                                                          str(r.get("doc"))[:14], str(r.get("live"))[:40], r.get("claim", "")[:80]))
    bad = [r for r in rows if r["status"] == "MISMATCH"]
    print("%d claims checked, %d mismatches, %d unresolved" % (len(rows), len(bad), sum(1 for r in rows if r["status"] == "unresolved")))


def render_markdown(rows):
    out = ["| source | claim | doc | live (measured) | fact |", "|---|---|---|---|---|"]
    for r in rows:
        if r["status"] != "MISMATCH":
            continue
        out.append("| %s | %s | %s | %s | `%s` |" % (r.get("source"), r.get("claim", "").replace("|", "\\|"), r.get("doc"),
                                                   r.get("live"), r.get("fact")))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="docs/dashboard vs live config drift report (read-only)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--markdown", action="store_true", help="mismatches as a markdown table")
    ap.add_argument("--no-engine", action="store_true")
    ap.add_argument("--all", action="store_true", help="also print ok rows")
    a = ap.parse_args(argv)
    facts, notes = gather_facts(engine=not a.no_engine)
    rows = check_notes(facts) + check_dashboard(facts)
    if a.json:
        print(json.dumps(dict(notes=notes, rows=rows), indent=1, default=str))
    elif a.markdown:
        print(render_markdown(rows))
    else:
        render_text(rows if a.all else [r for r in rows if r["status"] != "ok"], notes)
    return 1 if any(r["status"] == "MISMATCH" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
