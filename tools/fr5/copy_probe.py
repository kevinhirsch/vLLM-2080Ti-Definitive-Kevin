"""FR5 GPU-window probe (client only, talks to the engine on :8001; changes nothing on the engine).
Sends copy-heavy and control prompts at C1, greedy, records per-request MTP acceptance (from /metrics deltas) and
the full prompt/response text so lookup_opportunity.py can be run on REAL model outputs afterwards.
Workloads: rewrite (edit one thing in a source file, output the whole file), quote (reproduce matching lines of a log
verbatim), summarize (control, same as multi_bench). Usage: copy_probe.py <label> <out.json> <out_pairs.jsonl>"""
import glob, json, os, random, statistics as st, sys, time, urllib.request
sys.path.insert(0, "/home/kevin/projects/lanes/coexec")
from qual_bench import stream, BASE  # noqa: E402

def metrics():
    txt = urllib.request.urlopen(f"{BASE}/metrics", timeout=10).read().decode()
    o = {"drafts": 0.0, "accepted": 0.0, "pos": {}}
    for line in txt.splitlines():
        v = line.rsplit(" ", 1)[-1]
        if line.startswith("vllm:spec_decode_num_drafts_total{"): o["drafts"] += float(v)
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total{"): o["accepted"] += float(v)
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos"):
            pos = line.split('position="')[1].split('"')[0]; o["pos"][pos] = o["pos"].get(pos, 0.0) + float(v)
    return o

def delta(a, b):
    d = b["drafts"] - a["drafts"]
    return {"drafts": d, "accepted_per_draft": (b["accepted"] - a["accepted"]) / d if d else None,
            "pos_rate": {k: (b["pos"][k] - a["pos"].get(k, 0)) / d for k in b["pos"]} if d else {}}

def reqs():
    random.seed(1)
    src = sorted(p for p in glob.glob("/home/kevin/Desktop/wt-fr5-suffix/vllm/v1/**/*.py", recursive=True) if 2500 < os.path.getsize(p) < 6500)
    random.shuffle(src); out = []
    for p in src[:8]:
        t = open(p).read()
        out.append(("rewrite", f"Here is a Python file.\n\n```python\n{t}\n```\n\nOutput the complete file again, unchanged except that every comment starting with '#' gets the suffix ' (reviewed)'. Output only the file in one code block.", 1400))
    for p in src[8:14]:
        t = open(p).read().split("\n")
        log = "\n".join(f"{i+1:04d}: {l}" for i, l in enumerate(t))
        out.append(("quote", f"Tool output (grep -n of a source file):\n{log}\n\nReproduce verbatim every line above that contains the word 'def ' or 'class ' (keep the line-number prefix), then add one sentence naming the file's purpose.", 700))
    notes = sorted(p for p in glob.glob("/home/kevin/Obsidian/Memory/*.md") if 8000 <= os.path.getsize(p) <= 12000)[:6]
    for p in notes:
        out.append(("summarize", "Summarize the following note in detail, section by section.\n\n" + open(p, errors="replace").read(), 384))
    return out

def main():
    label, outp, pairs = sys.argv[1:4]
    res = {"label": label, "t0": time.time(), "items": []}
    pf = open(pairs, "w")
    for kind, prompt, mt in reqs():
        body = {"model": "qwen-local", "chat_template_kwargs": {"enable_thinking": False}, "temperature": 0, "seed": 0,
                "max_tokens": mt, "messages": [{"role": "user", "content": prompt}]}
        stream(dict(body, max_tokens=1))              # warm prefix cache so decode is measured alone
        m0 = metrics(); t0 = time.perf_counter()
        # non-stream call to get the text; acceptance delta covers exactly this request (C1, sequential)
        r = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(r, timeout=600)); wall = time.perf_counter() - t0
        m1 = metrics(); text = d["choices"][0]["message"]["content"] or ""
        n = d["usage"]["completion_tokens"]
        it = {"kind": kind, "completion_tokens": n, "tok_s_wall": n / wall, **delta(m0, m1)}
        res["items"].append(it); pf.write(json.dumps({"src": "live:" + kind, "prompt_text": prompt, "resp_text": text}) + "\n")
        print(kind, n, round(it["tok_s_wall"], 1), it["accepted_per_draft"], flush=True)
    for kind in ("rewrite", "quote", "summarize"):
        xs = [i for i in res["items"] if i["kind"] == kind]
        res[kind] = {"n": len(xs), "tok_s_median": st.median(i["tok_s_wall"] for i in xs),
                     "accepted_per_draft_mean": st.fmean(i["accepted_per_draft"] for i in xs if i["accepted_per_draft"])}
    json.dump(res, open(outp, "w"), indent=1)

if __name__ == "__main__": main()
