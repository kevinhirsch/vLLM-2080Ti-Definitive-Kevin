#!/usr/bin/env python3
"""Lane DFT: distil the target's sampling distribution into the engine's MTP drafter block (only `mtp.*` trains; lm_head, embeddings and
the target are frozen and shared). Chain-aware (training-time-test) loss over the 3 draft steps, teacher-forced inputs, teacher =
target top-k distribution at the production sampling settings (T=0.6, top_k=20, top_p=0.95), position weights favour assistant tokens.
torchrun --nproc_per_node=2 train_mtp.py --hid-dir H --out-dir OUT [--budget-min 45]   (single process also works)
Writes OUT/model-mtp.safetensors (bf16, the best-val checkpoint, same key layout as the original) + OUT/train_log.json. Never touches the original weights."""
import argparse, json, math, os, sys, time, random
import numpy as np
import torch, torch.distributed as dist
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C, evalcore as E
ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=f"{HERE}/data/manifest.json"); ap.add_argument("--seq-dir", default=f"{HERE}/data/seqs")
ap.add_argument("--hid-dir", required=True); ap.add_argument("--out-dir", required=True)
ap.add_argument("--init", default=os.path.join(C.MODEL_DIR, "model-mtp.safetensors"))
ap.add_argument("--budget-min", type=float, default=45); ap.add_argument("--lr", type=float, default=1.5e-5)
ap.add_argument("--accum", type=int, default=2); ap.add_argument("--val-every-min", type=float, default=9)
ap.add_argument("--max-val-windows", type=int, default=60); ap.add_argument("--warmup", type=int, default=12)
ap.add_argument("--selftest", action="store_true", help="CPU/gloo tiny-shape smoke test of the whole loop (no GPU, synthetic data)")
ap.add_argument("--alphas", default="0.51,0.31,0.18"); ap.add_argument("--max-train-windows", type=int, default=0)
a = ap.parse_args()
rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
dev = torch.device("cpu") if a.selftest else torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
if not a.selftest: torch.cuda.set_device(dev)
if world > 1: dist.init_process_group("gloo" if a.selftest else "nccl")
torch.manual_seed(1234 + rank); random.seed(99)
log = lambda *x: print(f"[r{rank} {time.strftime('%H:%M:%S')}]", *x, flush=True) if rank == 0 else None
os.makedirs(a.out_dir, exist_ok=True)
if a.selftest:
    man = []; E.TMIN = 8; C.ROT = 8
    tr = [({"id": f"tr{i}", "windows": [[0, 96]]}, 0, 0) for i in range(24)]; va = [({"id": f"va{i}", "windows": [[0, 96]]}, 0, 0) for i in range(6)]
    def _lw(seq_dir, hid_dir, entry, widx, off):
        g = torch.Generator().manual_seed(abs(hash(entry["id"])) % 10000); T = 96
        return dict(id=entry["id"], ids=torch.randint(0, 200, (T,), generator=g), w=torch.where(torch.rand(T, generator=g) < .5, 1.0, .3), H=torch.randn(T, 64, generator=g))
    E.load_window = _lw
else:
    man = json.load(open(a.manifest))
    tr = E.list_windows(man, a.hid_dir, "train"); va = E.list_windows(man, a.hid_dir, "val")
random.Random(5).shuffle(va); va = va[: a.max_val_windows]
if a.max_train_windows: tr = tr[: a.max_train_windows]
log(f"train windows {len(tr)} ({sum(m['windows'][wi][1]-m['windows'][wi][0] for m,wi,_ in tr)} positions), val windows {len(va)}")
if a.selftest:
    class _FR: pass
    fr = _FR(); fr.lm = torch.randn(200, 64); fr.emb = torch.randn(200, 64); fr.dev = dev; fr.dtype = torch.float32
    ang = torch.rand(512, 4) * 6.28; fr.cache = torch.cat([ang.cos(), ang.sin()], -1); fr.logits = lambda h: h @ fr.lm.T
    torch.manual_seed(7); blk = C.MTPBlock(h=64, i=96, nh=4, nkv=2, hd=16, rot=8)
    for p_ in blk.parameters(): torch.nn.init.normal_(p_, std=0.2) if p_.ndim == 2 else torch.nn.init.normal_(p_, std=0.1)
    blk = blk.to(dev)
else:
    fr = E.Frozen(dev)
    blk = C.MTPBlock().to(dev).load_ckpt(a.init)
for p in blk.parameters(): p.requires_grad_(True)
opt = torch.optim.AdamW(blk.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 12, enabled=not a.selftest)
alphas = tuple(float(x) for x in a.alphas.split(","))

def val_eval():
    blk.eval()
    tot = torch.zeros(E.K, 2, 6, device=dev)
    for m, wi, off in va[rank::world]:
        tot += E.window_metrics(blk, fr, E.load_window(a.seq_dir, a.hid_dir, m, wi, off))
    if world > 1: dist.all_reduce(tot)
    blk.train()
    return E.summarize(tot)

def score(s): return s["assistant"]["accept_len_sampled"]
hist = []
v0 = val_eval(); best = (score(v0), 0, {k: x.detach().cpu().clone() for k, x in blk.state_dict().items()})
hist.append(dict(step=0, minutes=0, val=v0)); log("VAL step0 (base):", json.dumps({k: round(v0["assistant"][f"pos{i}"]["chain_sampled"], 4) for i, k in enumerate(["p0", "p1", "p2"])}), "len", round(score(v0), 4))
t0 = time.time(); step = 0; epoch = 0; last_val = t0; ema = {}
budget = a.budget_min * 60
done = False
while not done:
    order = list(range(len(tr))); random.Random(100 + epoch).shuffle(order)
    my = order[rank::world]
    for i in range(0, len(my) - a.accum + 1, a.accum):
        frac = (time.time() - t0) / budget
        if frac >= 1: done = True; break
        lr = a.lr * min(1.0, (step + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups: g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for j in range(a.accum):
            m, wi, off = tr[my[i + j]]
            win = E.load_window(a.seq_dir, a.hid_dir, m, wi, off)
            with torch.autocast("cuda", dtype=torch.float16, enabled=not a.selftest):
                loss, ex = E.window_loss(blk, fr, win, alphas=alphas)
            scaler.scale(loss / a.accum).backward()
            for kx, vx in ex.items(): ema[kx] = 0.95 * ema.get(kx, vx) + 0.05 * vx
        if world > 1:
            flat = torch.cat([p.grad.reshape(-1) for p in blk.parameters()])
            dist.all_reduce(flat); flat /= world
            o = 0
            for p in blk.parameters():
                n = p.numel(); p.grad.copy_(flat[o: o + n].view_as(p)); o += n
            del flat
        scaler.unscale_(opt); gn = torch.nn.utils.clip_grad_norm_(blk.parameters(), 1.0)
        scaler.step(opt); scaler.update(); step += 1
        if step % 5 == 0: log(f"ep{epoch} step {step} lr {lr:.2e} gn {float(gn):.2f} scale {scaler.get_scale():.0f} min {(time.time()-t0)/60:.1f}", {k: round(v, 3) for k, v in ema.items()})
        if (time.time() - last_val) / 60 >= a.val_every_min:
            v = val_eval(); last_val = time.time(); s = score(v)
            hist.append(dict(step=step, minutes=(time.time() - t0) / 60, val=v))
            log("VAL step", step, "p0/p1/p2 chain_sampled", [round(v["assistant"][f"pos{q}"]["chain_sampled"], 4) for q in range(3)], "len", round(s, 4), "(best %.4f)" % best[0])
            if s > best[0]: best = (s, step, {k: x.detach().cpu().clone() for k, x in blk.state_dict().items()})
    epoch += 1
v = val_eval(); s = score(v); hist.append(dict(step=step, minutes=(time.time() - t0) / 60, val=v))
log("FINAL VAL step", step, "len", round(s, 4), "best", round(best[0], 4), "at step", best[1])
if s > best[0]: best = (s, step, {k: x.detach().cpu().clone() for k, x in blk.state_dict().items()})
if rank == 0:
    blk.load_state_dict({k: x.to(dev) for k, x in best[2].items()})
    blk.export_ckpt(os.path.join(a.out_dir, "model-mtp.safetensors"))
    json.dump(dict(args=vars(a), best_step=best[1], best_score=best[0], hist=hist, steps=step, epochs=epoch, train_windows=len(tr)), open(os.path.join(a.out_dir, "train_log.json"), "w"), indent=1)
    log("WROTE", os.path.join(a.out_dir, "model-mtp.safetensors"))
if world > 1: dist.barrier(); dist.destroy_process_group()
