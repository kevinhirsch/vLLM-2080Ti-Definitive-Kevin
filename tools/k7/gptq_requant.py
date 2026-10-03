#!/usr/bin/env python
"""Lane K7: GPTQ re-quantization of the abliterated HauhauCS W4A16 model into ROTATED symmetric per-channel int4 weights
for the Turing W4A4 kernels (k7 w4a4_gemm per-token and w4a4g group-scaled both take per-output-channel weight scales).

Source = shipped int4 dequant (no bf16 abliterated weights exist locally); original files are only read.
Per Marlin linear: W_rot = W @ blockdiag(H128)^T; Hessian H = X_rot^T X_rot from the BASELINE (W4A16, fp32) forward of the
calibration windows (ref.pt windows 6..11 by default, disjoint from the gate's eval windows 0..5); GPTQ (act-order,
1% damp, symmetric per-channel scales from an MSE-clip search) -> int4 codes + fp32 scale.
CPU pipeline: the main process streams the calibration forward layer by layer and feeds (W_rot, H) jobs to --workers
single-thread GPTQ processes (1 thread each is ~10x faster than 6 threads on this contended box); --device cuda runs
GPTQ inline on the GPU instead (window use).
Output: <out>/layer_XX.safetensors with "<name>.codes" (int8, 2 codes/byte, cutlass int4b_t order) and "<name>.scale",
meta.json, stats.jsonl (calib-Hessian output SQNR RTN vs GPTQ per linear).  Resumable: finished layers are skipped
(the calib residual stream is checkpointed in <out>/stream.pt).
Usage: CUDA_VISIBLE_DEVICES= python tools/k7/gptq_requant.py --workers 4      (CPU, no engine impact)
       CUDA_VISIBLE_DEVICES=1 python tools/k7/gptq_requant.py --device cuda   (inside an engine-off window)"""
import argparse, json, os, sys, time, types
import torch
import torch.multiprocessing as mp
sys.path.insert(0, "/home/kevin/Desktop/wt-integrate/tools/u2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_dump as R
import rotquant as RQ


def _job(args):
    name, wr, H, dev = args
    if mp.current_process().name != "MainProcess":  # pool worker: 1 thread each (10x faster than 6 on this box)
        torch.set_num_threads(1)
    d = torch.device(dev)
    wr, H = wr.to(d), H.to(d)
    c, s = RQ.gptq_sym(wr, H, 4, 0)
    cr, sr = RQ.sym_quant(wr, 4, 0)
    def oerr(dq):
        E = dq - wr
        return ((E @ H) * E).sum().item() / ((wr @ H) * wr).sum().item()
    import math
    st = {"out_sqnr_gptq_db": -10 * math.log10(oerr(RQ.dequant(c, s, 0))), "out_sqnr_rtn_db": -10 * math.log10(oerr(RQ.dequant(cr, sr, 0)))}
    return name, RQ.pack_s4(c.cpu()), s.float().cpu().contiguous(), st


def build_calib(ref, n_windows, win, seed, eval_windows, log):
    """Calibration windows DISJOINT from the gate's eval windows (32-gram overlap <= 10%): recorded estate flightrec bodies
    (chat-templated, like ref_dump), the evalkit code corpus, and vault Memory prose.  Mix ~60/25/15."""
    import glob, random
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(R.MODEL_DIR)
    tpl = open(R.CHAT_TEMPLATE).read()
    rng = random.Random(seed)
    used = set()
    for e in eval_windows:
        used |= R._ngram_hashes(e.tolist())
    pools = {"estate": [], "code": [], "prose": []}
    for f in sorted(glob.glob(f"{R.FLIGHTREC}/*.json")) + sorted(glob.glob("/home/kevin/.local/share/vllm-qwen27b/flightrec/*.json")):
        try:
            d = json.load(open(f))
            t = tok.apply_chat_template(d["messages"], tools=d.get("tools"), tokenize=False, add_generation_prompt=True,
                                        chat_template=tpl, **(d.get("chat_template_kwargs") or {}))
        except Exception:
            continue
        pools["estate"].append(tok(t, add_special_tokens=False)["input_ids"])
    pools["code"].append(tok(open(R.CORPUS).read(), add_special_tokens=False)["input_ids"])
    for f in sorted(glob.glob("/home/kevin/Obsidian/Memory/*.md"))[:400]:
        pools["prose"].append(tok(open(f, errors="ignore").read(), add_special_tokens=False)["input_ids"])
    cands = {k: [] for k in pools}
    for k, docs in pools.items():
        for ids_ in docs:
            for o in range(0, max(0, len(ids_) - win), win // 2):
                cands[k].append(ids_[o:o + win])
        rng.shuffle(cands[k])
    want = {"estate": int(n_windows * 0.6), "code": int(n_windows * 0.25)}
    want["prose"] = n_windows - want["estate"] - want["code"]
    out = []
    for k, n in want.items():
        got = 0
        for c in cands[k]:
            if got >= n:
                break
            hs = R._ngram_hashes(c)
            if len(hs & used) > 0.1 * max(1, len(hs)):
                continue
            used |= hs; out.append(c); got += 1
        log(f"calib {k}: {got}/{n} windows (candidates {len(cands[k])})")
    return torch.tensor(out, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="/home/kevin/projects/lanes/u2-quant/ref.pt")
    ap.add_argument("--out", default="/home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A4rot128-gptq-k7")
    ap.add_argument("--calib", default="build", help="'build' (disjoint estate/code/prose set) or 'a:b' ref.pt windows")
    ap.add_argument("--calib-windows", type=int, default=128)
    ap.add_argument("--calib-win", type=int, default=512)
    ap.add_argument("--calib-batch", type=int, default=8, help="windows per forward batch")
    ap.add_argument("--eval-windows", default="0:6", help="ref.pt windows the gate evaluates on (excluded from calib)")
    ap.add_argument("--hb", type=int, default=128)
    ap.add_argument("--layers", type=int, default=R.N_LAYERS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fwd-threads", type=int, default=4)
    a = ap.parse_args()
    from safetensors.torch import save_file
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(a.fwd_threads)
    T0 = time.time(); log = lambda *s: print(f"[{time.time()-T0:6.0f}s]", *s, flush=True)
    ref = torch.load(a.ref)
    e0, e1 = map(int, a.eval_windows.split(":"))
    if a.calib == "build":
        ids = build_calib(ref, a.calib_windows, a.calib_win, 4321, ref["ids"][e0:e1], log)
        ckinds = "estate/code/prose mix"
    else:
        c0, c1 = map(int, a.calib.split(":")); ids = ref["ids"][c0:c1]; ckinds = [m["kind"] for m in ref["manifest"][c0:c1]]
    B, T = ids.shape
    log(f"calibration: {B} windows x {T} tokens = {B*T} tokens")
    json.dump({"source": R.MODEL_DIR, "calib": a.calib, "calib_tokens": B * T, "calib_kinds": ckinds, "eval_windows_excluded": [e0, e1],
               "hb": a.hb, "scheme": "W @ blockdiag(Hadamard_hb)^T; symmetric int4 per output channel; GPTQ act-order damp 0.01",
               "pack": "int8 [N,K/2], low nibble = even k (cutlass int4b_t)", "by": "Lane K7 2026-10-03"},
              open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    CAP, NAMES = {}, {}
    _lin = torch.nn.functional.linear

    def lin(x, w, b=None):
        n = NAMES.get(id(w))
        if n is not None:
            xs = RQ.block_had(x.reshape(-1, x.shape[-1]).float(), a.hb)
            H = xs.T @ xs
            CAP[n] = CAP[n] + H if n in CAP else H
        return _lin(x, w, b)

    R.F = types.SimpleNamespace(**{k: getattr(torch.nn.functional, k) for k in dir(torch.nn.functional) if not k.startswith("__")})
    R.F.linear = lin
    dev = torch.device(a.device)
    if dev.type == "cuda":  # window mode: whole calib forward + GPTQ on the GPU
        _rope = R.rope_cos_sin
        R.rope_cos_sin = lambda T_: tuple(t.to(dev) for t in _rope(T_))
    sp = os.path.join(a.out, "stream.pt")
    start = 0
    if os.path.exists(sp):
        st = torch.load(sp); x, start = st["x"], st["next_layer"]
        log(f"resume at layer {start}")
    else:
        x = R.load_embed()[ids.reshape(-1)].float()
    x = x.to(dev)
    pool = mp.get_context("spawn").Pool(a.workers) if a.device == "cpu" and a.workers > 1 else None
    pending = []  # (layer, AsyncResult list, lnames)

    def flush(block):
        while pending and (block or all(r.ready() for r in pending[0][1])):
            li, rs, _ = pending.pop(0)
            out, stats = {}, {}
            for r in rs:
                n, codes, scale, stt = r.get()
                out[f"{n}.codes"], out[f"{n}.scale"], stats[n] = codes, scale, stt
            save_file(out, os.path.join(a.out, f"layer_{li:02d}.safetensors"))
            with open(os.path.join(a.out, "stats.jsonl"), "a") as fh:
                fh.write(json.dumps({"layer": li, **stats}) + "\n")
            log(f"layer {li:2d} saved  " + " ".join(f"{n}:{v['out_sqnr_rtn_db']:.1f}->{v['out_sqnr_gptq_db']:.1f}dB" for n, v in stats.items()))

    for i in range(start, a.layers):
        tl = time.time()
        W = {k: v.to(dev) for k, v in R.load_layer_weights(i).items()}
        lnames = ["gate_proj", "up_proj", "down_proj"] + (["q_proj", "k_proj", "v_proj", "o_proj"] if R.LAYER_TYPES[i] == "full_attention"
                                                        else ["in_proj_qkv", "in_proj_z", "out_proj"])
        NAMES.clear(); CAP.clear()
        for n in lnames:
            NAMES[id(W[n])] = n
        with torch.no_grad():
            cb = a.calib_batch
            xn = torch.cat([R.decoder_layer(x[b0 * T:min(B, b0 + cb) * T], i, W, min(B, b0 + cb) - b0, T) for b0 in range(0, B, cb)])
        jobs = [(n, RQ.rotate_weight(W[n], a.hb).cpu() if pool else RQ.rotate_weight(W[n], a.hb), CAP[n].cpu() if pool else CAP[n], a.device) for n in lnames]
        if pool is None:
            rs = [types.SimpleNamespace(get=(lambda r=_job(j): r), ready=lambda: True) for j in jobs]
        else:
            rs = [pool.apply_async(_job, (j,)) for j in jobs]
        pending.append((i, rs, lnames))
        x = xn
        del W, jobs
        flush(block=False)
        while len(pending) > max(1, a.workers // 2):  # bound RAM: Hessians of in-flight layers
            flush(block=True)
        if not pending:
            torch.save({"x": x.cpu(), "next_layer": i + 1}, sp)
        log(f"layer {i:2d} fwd+submit {time.time()-tl:5.1f}s (in flight {len(pending)})")
    flush(block=True)
    torch.save({"x": x.cpu(), "next_layer": a.layers}, sp)
    if pool:
        pool.close(); pool.join()
    log("done")


if __name__ == "__main__":
    main()
