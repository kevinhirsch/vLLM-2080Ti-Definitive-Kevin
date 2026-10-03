#!/usr/bin/env python3
"""Lane DFT (inside the offline window, LIVE engine :8001 up): on-policy generation. Replays truncated real estate conversations
(data/gen_prompts.jsonl) through /v1/completions at the production sampling settings, 12-way concurrent, and appends each
prompt+generation as a new sequence (generated tokens weight 1.0) to data/manifest.json. 12% go to the val split.
usage: gen_onpolicy.py [--n 100] [--max-tokens 1400] [--budget-s 1200]"""
import argparse, json, os, sys, time, threading, http.client, random
import numpy as np
from transformers import AutoTokenizer
HERE = os.path.dirname(os.path.abspath(__file__)); D = f"{HERE}/data"
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100); ap.add_argument("--max-tokens", type=int, default=1400)
ap.add_argument("--budget-s", type=float, default=1200); ap.add_argument("--conc", type=int, default=12)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained("/home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven")
P = [json.loads(l) for l in open(f"{D}/gen_prompts.jsonl")][: a.n]
c = http.client.HTTPConnection("127.0.0.1", 8001, timeout=10); c.request("GET", "/v1/models"); MODEL = json.loads(c.getresponse().read())["data"][0]["id"]
res, lock, t0 = [], threading.Lock(), time.time()
q = list(P)
def worker():
    while True:
        with lock:
            if not q or time.time() - t0 > a.budget_s: return
            g = q.pop()
        try:
            cn = http.client.HTTPConnection("127.0.0.1", 8001, timeout=900)
            cn.request("POST", "/v1/completions", json.dumps(dict(model=MODEL, prompt=g["prompt"], max_tokens=a.max_tokens, temperature=0.6, top_p=0.95, top_k=20,
                       skip_special_tokens=False)), {"Content-Type": "application/json", "X-Client": "dft-gen"})
            r = json.loads(cn.getresponse().read()); txt = r["choices"][0]["text"]
            with lock: res.append((g, txt, r["usage"]))
        except Exception as e:
            print("gen err", repr(e)[:100], flush=True)
ths = [threading.Thread(target=worker) for _ in range(a.conc)]; [t.start() for t in ths]; [t.join() for t in ths]
man = json.load(open(f"{D}/manifest.json")); man = [m for m in man if not m["id"].startswith("gen_")]
IM_E = tok.convert_tokens_to_ids("<|im_end|>"); rng = random.Random(7); ntok = 0
for i, (g, txt, u) in enumerate(res):
    pids = tok(g["prompt"], add_special_tokens=False)["input_ids"]; gids = tok(txt, add_special_tokens=False)["input_ids"]
    if len(gids) < 24: continue
    ids = pids + gids
    w = np.concatenate([np.full(len(pids), 0.3), np.full(len(gids), 1.0)]).astype(np.float16)
    n = len(ids); wins = [[max(0, n - 3072), n]]
    sid = "gen_" + g["id"].split("_", 1)[1][:40] + f"_{i}"
    np.savez(f"{D}/seqs/{sid}.npz", ids=np.array(ids, dtype=np.int32), w=w)
    man.append(dict(id=sid, src="onpolicy", split="val" if rng.random() < 0.12 else "train", n=n, windows=wins)); ntok += len(gids)
json.dump(man, open(f"{D}/manifest.json", "w"))
print(f"GEN DONE {len(res)} completions, {ntok} generated tokens, {time.time() - t0:.0f}s", flush=True)
