"""Build token-pair corpora for lookup_opportunity.py. CPU only. Sources: evalkit result files (request+response),
plus a SYNTHETIC code-edit upper bound built from repo source files (clearly labelled)."""
import glob, json, os, random, sys, hashlib, difflib
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven")
enc = lambda s: tok.encode(s, add_special_tokens=False)
out = open(sys.argv[1], "w"); random.seed(0)
R = "/home/kevin/Desktop/qwen38-evalkit/results"
seen = set(); per = {}
files = sorted(glob.glob(R + "/*/*_[0-9][0-9][0-9].json"))
random.shuffle(files)
for f in files:
    try: d = json.load(open(f))
    except Exception: continue
    if "request" not in d or not isinstance(d.get("result"), dict): continue
    resp = (d["result"].get("response") or {})
    try: m = resp["choices"][0]["message"]
    except Exception: continue
    text = (m.get("content") or "")
    for tc in m.get("tool_calls") or []:
        text += json.dumps(tc.get("function", {}))
    if len(text) < 20: continue
    h = hashlib.md5(text.encode()).hexdigest()
    if h in seen: continue
    seen.add(h)
    cat = d["category"]
    if per.get(cat, 0) >= 150: continue
    per[cat] = per.get(cat, 0) + 1
    rq = d["request"]
    prompt = "\n".join(x.get("content") if isinstance(x.get("content"), str) else json.dumps(x.get("content")) for x in rq.get("messages", []))
    if rq.get("tools"): prompt = json.dumps(rq["tools"]) + "\n" + prompt
    p = enc(prompt)
    if len(p) > 40000: p = p[-40000:]
    out.write(json.dumps({"src": "evalkit:" + cat, "prompt": p, "resp": enc(text)}) + "\n")
# synthetic code-edit: prompt = file + instruction, response = file with ~3 edited line blocks (upper bound for 'rewrite this file')
srcs = [p for p in glob.glob("/home/kevin/Desktop/wt-fr5-suffix/vllm/v1/**/*.py", recursive=True) if 3000 < os.path.getsize(p) < 9000]
random.shuffle(srcs)
for p in srcs[:40]:
    lines = open(p).read().split("\n")
    new = list(lines)
    for _ in range(3):
        i = random.randrange(len(new)); new[i] = new[i] + "  # edited"
        if i + 1 < len(new): del new[i + 1]
    out.write(json.dumps({"src": "SYNTHETIC:code_rewrite_3_edits", "prompt": enc("Apply the edits and output the whole file.\n" + "\n".join(lines)), "resp": enc("\n".join(new))}) + "\n")
# vault-note proxy for multi_bench summarisation: no real outputs exist, so skip (measured acceptance used instead)
print(per, len(srcs[:40]))
