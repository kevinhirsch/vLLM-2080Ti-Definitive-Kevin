#!/usr/bin/env python3
"""Lane DFT offline eval: per-position (0..2) acceptance of MTP weight files on HELD-OUT windows (val split), base vs tuned, using the
target hidden states dumped by extract_hidden.py.  For each position k: top1 = P(draft == target greedy) [teacher-forced], sampled = E[p_target(draft)]
under the production sampling distribution (T=.6, top_k 20, top_p .95) = the rejection-sampling acceptance, chain_* = conditional on the earlier
drafts accepted and on-greedy-path recorded tokens (the engine's per-position conditional acceptance).  Weights are loaded exactly as the engine
does (bf16 file -> fp16 compute); --int4-sim also evaluates the U2b int4 version of each file.
usage: eval_mtp.py --hid-dir H --mtp base=PATH tuned=PATH [--int4-sim] [--out J]"""
import argparse, json, os, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C, evalcore as E
ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=f"{HERE}/data/manifest.json"); ap.add_argument("--seq-dir", default=f"{HERE}/data/seqs")
ap.add_argument("--hid-dir", required=True); ap.add_argument("--mtp", nargs="+", required=True, help="label=path/to/model-mtp.safetensors")
ap.add_argument("--split", default="val"); ap.add_argument("--int4-sim", action="store_true"); ap.add_argument("--out", default="")
ap.add_argument("--max-windows", type=int, default=0); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--cpu", action="store_true")
a = ap.parse_args()
dev = torch.device("cpu") if a.cpu else torch.device("cuda", a.gpu)
if not a.cpu: torch.cuda.set_device(dev)
man = json.load(open(a.manifest)); wins = E.list_windows(man, a.hid_dir, a.split)
if a.max_windows: wins = wins[: a.max_windows]
fr = E.Frozen(dev)
loaded = [E.load_window(a.seq_dir, a.hid_dir, m, wi, off) for m, wi, off in wins]
srcs = {}
for m, wi, off in wins: srcs[m["src"]] = srcs.get(m["src"], 0) + 1
print(f"{len(wins)} held-out windows {srcs}", flush=True)
results = {}
for spec in a.mtp:
    label, path = spec.split("=", 1)
    for variant in (["bf16"] + (["int4"] if a.int4_sim else [])):
        blk = C.MTPBlock().to(dev).load_ckpt(path)
        # the engine casts the bf16 file to fp16 (--dtype half); autocast reproduces fp16 compute with identical weight values
        blk.half().float()  # round-trip through fp16 weights
        if variant == "int4": C.int4_sim_(blk)
        blk.eval(); tot = torch.zeros(E.K, 2, 6, device=dev); t0 = time.time()
        for w in loaded: tot += E.window_metrics(blk, fr, w)
        s = E.summarize(tot); results[f"{label}:{variant}"] = s
        for pop in ("assistant", "all"):
            d = s[pop]
            print(f"{label:>8}:{variant:<5} [{pop:9}] " + " | ".join(f"pos{k}: top1 {d[f'pos{k}']['top1']:.3f} samp {d[f'pos{k}']['sampled']:.3f} chain {d[f'pos{k}']['chain_sampled']:.3f}" for k in range(3)) + f" | accept_len {d['accept_len_sampled']:.3f} (n={d['pos0']['n']})", flush=True)
        del blk; torch.cuda.empty_cache()
if a.out: json.dump(results, open(a.out, "w"), indent=1)
