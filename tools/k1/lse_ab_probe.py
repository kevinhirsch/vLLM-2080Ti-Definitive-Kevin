#!/usr/bin/env python3
"""K1 LSE-fix A/B probe (direct :8001, temp 0). Graded long-context recall where the prefix-combine path is active
(seq_len >= 20,480): natural text (evalkit corpus) with 12 keyed facts spread over the context, ask for all of them.

Per document: facts recalled (0-12), the answer text, and the top-1 logprob of each generated token; compare arms with
--compare a.json b.json (token agreement, recall delta, mean |dlogprob| over the common prefix).
"""

from __future__ import annotations

import argparse
import http.client
import json
import random
import time

CORPUS = "/home/kevin/Desktop/qwen38-evalkit/corpus/corpus_128000tok.txt"
DOCS = [(30000, 1), (52000, 2), (95000, 3), (127000, 4)]  # (nominal tokens, seed); real ~26K/44K/81K/108K
NFACTS = 12


def build(tokens, seed):
    text = open(CORPUS).read()
    words = text.split()
    n = int(tokens / 3.64)  # corpus_128000tok: 35,197 words = 128,045 tokens
    r = random.Random(seed)
    start = r.randint(0, max(0, len(words) - n - 1))
    body = words[start:start + n]
    while len(body) < n:
        body += words[: n - len(body)]
    facts = []
    for i in range(NFACTS):
        code = f"{r.randint(1000, 9999)}-{r.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{r.randint(10, 99)}"
        facts.append((f"locker {i + 1}", code))
    # place facts at depths 0.04 .. 0.96 (most of them past the 20,480-token prefix-combine threshold)
    for i, (name, code) in reversed(list(enumerate(facts))):
        pos = int(len(body) * (0.04 + 0.92 * i / (NFACTS - 1)))
        body.insert(pos, f"(Registry note: the combination for {name} is {code}.)")
    prompt = (" ".join(body) + "\n\nThe text above contains registry notes giving the combination for lockers 1 to "
              f"{NFACTS}. List every locker and its combination, one per line, as 'locker N: CODE'. Nothing else.")
    return prompt, facts


def ask(prompt, max_tokens):
    c = http.client.HTTPConnection("127.0.0.1", 8001, timeout=3600)
    body = dict(model="qwen-local", messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens,
                temperature=0, logprobs=True, top_logprobs=1, chat_template_kwargs={"enable_thinking": False})
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json", "X-Client": "bench-k1"})
    d = json.loads(c.getresponse().read())
    ch = d["choices"][0]
    toks = [(t["token"], t["logprob"]) for t in ((ch.get("logprobs") or {}).get("content") or [])]
    return dict(text=ch["message"].get("content") or "", tokens=toks, prompt_tokens=d["usage"]["prompt_tokens"],
                secs=round(time.time() - t0, 1))


def run(arm, out):
    res = dict(arm=arm, when=time.strftime("%Y-%m-%d %H:%M:%S"), docs=[])
    for tokens, seed in DOCS:
        prompt, facts = build(tokens, seed)
        a = ask(prompt, 260)
        recall = sum(1 for _, code in facts if code in a["text"])
        a.update(seed=seed, target_tokens=tokens, recall=recall)
        res["docs"].append(a)
        print(json.dumps(dict(arm=arm, prompt_tokens=a["prompt_tokens"], recall=f"{recall}/{NFACTS}", secs=a["secs"])),
              flush=True)
    with open(out, "w") as f:
        json.dump(res, f)


def compare(pa, pb):
    A, B = json.load(open(pa)), json.load(open(pb))
    for da, db in zip(A["docs"], B["docs"]):
        ta, tb = da["tokens"], db["tokens"]
        same = 0
        for (x, _), (y, _) in zip(ta, tb):
            if x != y:
                break
            same += 1
        dl = [abs(x[1] - y[1]) for x, y in zip(ta[:same], tb[:same])]
        print(json.dumps(dict(prompt_tokens=da["prompt_tokens"], recall=(da["recall"], db["recall"]),
                              identical_prefix_tokens=f"{same}/{max(len(ta), len(tb))}",
                              mean_abs_dlogprob=round(sum(dl) / max(1, len(dl)), 4), arms=(A["arm"], B["arm"]))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
    else:
        run(a.arm, a.out)
