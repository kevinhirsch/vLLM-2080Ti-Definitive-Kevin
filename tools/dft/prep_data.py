#!/usr/bin/env python3
"""Lane DFT step 2 (CPU): build the distillation corpus from recorded estate traffic.
Sources: (A) flight-recorded request bodies (gateway flightrec + incident dumps), one longest body per session;
         (B) Estate Chats mirrors (Halo/Hermes sessions, assistant turns written by the local model);
         (C) a few vault prose notes (regulariser for natural text).
Splits are BY SESSION; sessions that appear in the frozen estate-pass set (s2-speed/fr) are VAL only.
Outputs: data/seqs/<id>.npz (ids int32, w float16 per-token loss weight), data/manifest.json, data/gen_prompts.jsonl (on-policy generation prompts)."""
import json, glob, os, re, hashlib, random, sys, collections
import numpy as np
from transformers import AutoTokenizer
HOME = os.path.expanduser("~")
M = "/home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven"
TPL = open(f"{HOME}/.local/share/vllm-qwen27b/chat_template-froggeric-v22-official.jinja").read()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(f"{OUT}/seqs", exist_ok=True)
tok = AutoTokenizer.from_pretrained(M)
rng = random.Random(20261002)
CAP, WIN, W_ASSIST, W_OTHER = 24576, 3072, 1.0, 0.3
IM_S, IM_E = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
AS = tok("assistant\n", add_special_tokens=False)["input_ids"]

def render(messages, tools=None, kw=None, gen=False):
    s = tok.apply_chat_template(messages, tools=tools or None, tokenize=False, add_generation_prompt=gen, chat_template=TPL, **(kw or {}))
    return s

def weights(ids):
    w = np.full(len(ids), W_OTHER, dtype=np.float16)
    i, n = 0, len(ids)
    while i < n:
        if ids[i] == IM_S and ids[i + 1: i + 1 + len(AS)] == AS:
            j = i + 1 + len(AS)
            e = j
            while e < n and ids[e] != IM_E:
                e += 1
            w[j: min(e + 1, n)] = W_ASSIST
            i = e
        i += 1
    return w

def pick_windows(w, n):
    if n <= WIN:
        return [[0, n]]
    ends = list(range(WIN, n + 1, 512))
    if ends[-1] != n:
        ends.append(n)
    cs = np.concatenate([[0], np.cumsum(w.astype(np.float32) >= 0.99)])
    dens = [(cs[e] - cs[e - WIN], e) for e in ends]
    best = max(dens)[1]
    cand = [e for d, e in dens if d >= 0.1 * WIN and abs(e - best) >= WIN]
    out = [[best - WIN, best]]
    if cand:
        e = rng.choice(cand); out.append([e - WIN, e])
    return out

def session_key(msgs):
    fu = next((x for x in msgs if x["role"] == "user"), None)
    c = fu["content"] if fu else ""
    c = c if isinstance(c, str) else json.dumps(c)
    return hashlib.md5(c[:1500].encode()).hexdigest()[:12]

def body_files():
    fs = glob.glob(f"{HOME}/.local/share/vllm-qwen27b/flightrec/*.json") + glob.glob(f"{HOME}/.local/share/vllm-qwen27b/incidents/*/flightrec/*.json")
    return fs

def ntok(p):
    m = re.search(r"_(\d+)tok\.json$", p)
    return int(m.group(1)) if m else 0

# ---- (A) bodies
fr_keys = set()
for p in glob.glob(f"{HOME}/projects/lanes/s2-speed/fr/*.json"):
    try: fr_keys.add(session_key(json.load(open(p))["messages"]))
    except Exception: pass
best, small = {}, collections.defaultdict(list)
for p in body_files():
    try: d = json.load(open(p))
    except Exception: continue
    m = d.get("messages")
    if not m: continue
    k = session_key(m)
    t = ntok(p)
    if k not in best or t > best[k][0]: best[k] = (t, p)
    if 600 < t <= 7000: small[k].append((t, p))
print("sessions", len(best), "fr sessions", len(fr_keys & set(best)))
keys = sorted(best)
rng.shuffle(keys)
val_keys = set(k for k in keys if k in fr_keys)
nonfr = [k for k in keys if k not in fr_keys]
val_keys |= set(nonfr[: max(6, len(nonfr) // 8)])
manifest = []
def add_seq(sid, src, split, text=None, ids=None):
    if ids is None:
        ids = tok(text, add_special_tokens=False)["input_ids"]
    ids = ids[:CAP]
    if len(ids) < 700: return None
    w = weights(ids)
    if float((w >= 0.99).mean()) < 0.01: return None
    wins = pick_windows(w, len(ids))
    end = max(e for s, e in wins)
    ids, w = ids[:end], w[:end]
    np.savez(f"{OUT}/seqs/{sid}.npz", ids=np.array(ids, dtype=np.int32), w=w)
    manifest.append(dict(id=sid, src=src, split=split, n=len(ids), windows=wins))
    return len(ids)
skipped = 0
for k in keys:
    t, p = best[k]
    d = json.load(open(p))
    try:
        s = render(d["messages"], d.get("tools"), d.get("chat_template_kwargs"))
    except Exception as e:
        skipped += 1; continue
    add_seq(f"body_{k}", "flightrec", "val" if k in val_keys else "train", text=s)
print("bodies added", sum(1 for m in manifest if m["src"] == "flightrec"), "render-skipped", skipped)

# ---- (B) Estate Chats
chat_files = glob.glob(f"{HOME}/Obsidian/Estate Chats/*/*.md")
rng.shuffle(chat_files)
tot, nchat = 0, 0
gen_chat = []
for f in chat_files:
    if tot > 760_000: break
    t = open(f, errors="ignore").read()
    if "## Turns" not in t: continue
    body = t.split("## Turns", 1)[1]
    if len(body) < 3000 or "pong" in os.path.basename(f).lower(): continue
    parts = re.split(r"^### \d\d:\d\d:\d\d · (\w+)\s*$", body, flags=re.M)
    msgs = []
    for i in range(1, len(parts) - 1, 2):
        role, c = parts[i], re.sub(r"\n_meta: [^\n]*\s*$", "", parts[i + 1].strip())
        if role not in ("user", "assistant", "tool") or not c: continue
        msgs.append({"role": role, "content": c[:12000]})
    if not any(m["role"] == "assistant" for m in msgs): continue
    try: s = render(msgs)
    except Exception: continue
    sid = "chat_" + hashlib.md5(f.encode()).hexdigest()[:10]
    n = add_seq(sid, "estate-chat", "val" if nchat % 9 == 0 else "train", text=s)
    if n: tot += n; nchat += 1
    if n and nchat % 9 != 1 and len(gen_chat) < 200:   # (nchat%9==0 before increment => val)
        ai = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i > 0]
        rng.shuffle(ai)
        for i in ai[:3]:
            try: ps = render(msgs[:i], kw={"enable_thinking": rng.random() < 0.7}, gen=True)
            except Exception: continue
            ln = len(ps) // 3.6
            if 1500 < ln < 8000:
                gen_chat.append(dict(id="genchat_" + sid + f"_{i}", tok_est=int(ln), prompt=ps)); break
print("chats", nchat, tot)

# ---- (C) vault prose
notes = glob.glob(f"{HOME}/Obsidian/Memory/*.md")
rng.shuffle(notes)
nn_ = 0
for f in notes:
    if nn_ >= 36: break
    t = open(f, errors="ignore").read()
    if not (4000 < len(t) < 14000): continue
    title = os.path.basename(f)[:-3]
    s = render([{"role": "user", "content": f"Write the vault note titled '{title}'."}, {"role": "assistant", "content": t}], kw={"enable_thinking": False})
    if add_seq("note_" + hashlib.md5(f.encode()).hexdigest()[:10], "vault-prose", "val" if nn_ % 9 == 0 else "train", text=s): nn_ += 1
print("notes", nn_)

# ---- on-policy generation prompts (short bodies of train sessions only)
gen = []
for k in keys:
    if k in val_keys or not small.get(k): continue
    t, p = sorted(small[k])[-1]
    d = json.load(open(p))
    try: s = render(d["messages"], d.get("tools"), d.get("chat_template_kwargs"), gen=True)
    except Exception: continue
    gen.append(dict(id=f"gen_{k}", tok_est=t, prompt=s, kw=d.get("chat_template_kwargs") or {}))
gen = gen + gen_chat
rng.shuffle(gen)
with open(f"{OUT}/gen_prompts.jsonl", "w") as f:
    for g in gen[:120]: f.write(json.dumps(g) + "\n")
print("gen prompts", min(len(gen), 120))

json.dump(manifest, open(f"{OUT}/manifest.json", "w"))
for sp in ("train", "val"):
    ms = [m for m in manifest if m["split"] == sp]
    print(sp, "seqs", len(ms), "prefill tokens", sum(m["n"] for m in ms), "window positions", sum(e - s for m in ms for s, e in m["windows"]),
          dict(collections.Counter(m["src"] for m in ms)))
