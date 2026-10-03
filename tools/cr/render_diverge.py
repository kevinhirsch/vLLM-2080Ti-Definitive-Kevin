#!/usr/bin/env python3
"""Lane CR: render predecessor/continuation bodies with the SERVED chat template and report the first
token where the continuation's prompt stops extending the predecessor's (template-level divergence).
usage: render_diverge.py DIR [B]   (uses captured flight-recorder bodies; read-only)"""
import glob, json, os, sys, hashlib
from transformers import AutoTokenizer
MODEL = "/home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven"
TPL = open(os.path.expanduser("~/.local/share/vllm-qwen27b/chat_template-froggeric-v22-official.jinja")).read()
tok = AutoTokenizer.from_pretrained(MODEL)
d = sys.argv[1]; B = int(sys.argv[2]) if len(sys.argv) > 2 else 1856

def render(j):
    kw = dict(j.get("chat_template_kwargs") or {})
    msgs = j["messages"]
    for m in msgs:   # vLLM parses tool_call arguments JSON strings into dicts before templating
        for tc in m.get("tool_calls") or []:
            f = tc.get("function") or {}
            if isinstance(f.get("arguments"), str):
                try: f["arguments"] = json.loads(f["arguments"])
                except Exception: pass
    s = tok.apply_chat_template(msgs, tools=j.get("tools"), chat_template=TPL, tokenize=False,
                                add_generation_prompt=True, **kw)
    return s

items = []
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    try: j = json.load(open(f))
    except Exception: continue
    items.append((os.path.basename(f), [json.dumps(m, sort_keys=True) for m in j["messages"]], j))
stats = {"pairs": 0, "pure_token_append": 0, "diverged": 0}
for i, (fn, ms, j) in enumerate(items):
    best = None
    for fn2, ms2, j2 in items[:i]:
        if len(ms2) < len(ms) and ms[:len(ms2)] == ms2 and (best is None or len(ms2) > len(best[1])):
            best = (fn2, ms2, j2)
    if not best: continue
    a = tok.encode(render(json.loads(json.dumps(best[2]))), add_special_tokens=False)
    b = tok.encode(render(json.loads(json.dumps(j))), add_special_tokens=False)
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]: k += 1
    stats["pairs"] += 1
    gen_prompt = len(a) - k
    if gen_prompt <= 8:
        stats["pure_token_append"] += 1; tag = "append"
    else:
        stats["diverged"] += 1; tag = "DIVERGED"
    E = (len(a) - 4) // B * B
    print(f"{fn} <- {best[0]}: pred_tok={len(a)} lcp={k} pred_tail_unmatched={len(a)-k} E={E} lcp>=E:{k>=E} {tag}")
    if tag == "DIVERGED":
        print("   pred :", repr(tok.decode(a[k:k+40])))
        print("   cont :", repr(tok.decode(b[k:k+40])))
print(json.dumps(stats))
