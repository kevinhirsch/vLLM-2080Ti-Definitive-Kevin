"""Shared train/eval core: window loading, teacher targets, student chain forward, losses and per-step acceptance metrics."""
import json, os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import common as C

K = 3
TMIN = 48          # ignore origins with < TMIN tokens of window context (the engine's drafter would have the full history)


def load_int4_head(mdir, dev, dt, rows=8192):
    """Dequantised U2b int4 lm_head from <model>-u2cache (the exact values the stack's Marlin head multiplies by)."""
    import glob
    from safetensors.torch import load_file
    from vllm.model_executor.layers.quantization import u2_headquant as Q
    fs = sorted(glob.glob(os.path.join(Q.cache_dir_for(mdir), "lm_head.weight.*.safetensors")), key=os.path.getmtime)
    if not fs: return None
    t = load_file(fs[-1]); out = torch.empty(t["weight_shape"][0].item(), t["weight_shape"][1].item(), dtype=dt, device=dev)
    for r in range(0, out.shape[0], rows):
        e = min(r + rows, out.shape[0])
        sub = {"weight_packed": t["weight_packed"][r:e], "weight_scale": t["weight_scale"][r:e], "weight_shape": torch.tensor([e - r, out.shape[1]]),
               "weight_zero_point": t["weight_zero_point"][r // 8: e // 8]}
        out[r:e] = Q.dequantize_linear(sub).to(dev, dt)
    return out


class Frozen:
    """lm_head / embedding / rope cache shared by every block variant (fp16, on device)."""
    def __init__(self, dev, mdir=C.MODEL_DIR):
        p = os.path.join(mdir, "model.safetensors")
        dt = torch.float16 if torch.device(dev).type == "cuda" else torch.float32
        self.lm = None
        if os.environ.get("DFT_INT4_HEAD", "1") == "1":     # production stack serves an int4 lm_head: teacher AND student use it so acceptance matches the engine
            self.lm = load_int4_head(mdir, dev, dt)
        if self.lm is None:
            self.lm = C.read_tensor(p, "lm_head.weight").to(dev, dt)                              # (V,H)
        self.emb = C.read_tensor(p, "model.language_model.embed_tokens.weight").to(dev, dt)
        self.cache = C.load_rope_cache(dev)
        self.dev = dev; self.dtype = dt

    def logits(self, h):
        return h.to(self.lm.dtype) @ self.lm.T


def load_window(seq_dir, hid_dir, entry, widx, hid_off):
    z = np.load(os.path.join(seq_dir, entry["id"] + ".npz"))
    s, e = entry["windows"][widx]
    H = np.load(os.path.join(hid_dir, entry["id"] + ".npy"), mmap_mode="r")[hid_off: hid_off + (e - s)]
    return dict(id=entry["id"], ids=torch.from_numpy(z["ids"][s:e].astype(np.int64)), w=torch.from_numpy(z["w"][s:e].astype(np.float32)),
                H=torch.from_numpy(np.ascontiguousarray(H)))


def list_windows(manifest, hid_dir, split):
    out = []
    for m in manifest:
        if m["split"] != split or not os.path.exists(os.path.join(hid_dir, m["id"] + ".npy")): continue
        off = 0
        for wi, (s, e) in enumerate(m["windows"]):
            out.append((m, wi, off)); off += e - s
    return out


def prep(fr, win):
    dev = fr.dev
    H = win["H"].to(dev, fr.dtype); ids = win["ids"].to(dev); w = win["w"].to(dev)
    emb = fr.emb[ids]
    with torch.no_grad():
        tidx, tp = C.teacher_topk(fr.logits, H)
    return H, emb, ids, w, tidx, tp


def _chunk_loss(m, lmT, idx, p, wt):
    lg = (m @ lmT).float()
    lse = torch.logsumexp(lg, -1)
    g = lg.gather(1, idx)
    ce = -(p * (g - lse[:, None])).sum(-1)
    return (ce * wt).sum(), lg.argmax(-1)


def window_loss(blk, fr, win, alphas=(0.51, 0.31, 0.18), frac_assist=0.7, frac_other=0.2, chunk=512, gen=None):
    """Differentiable weighted distillation loss for one window (autocast fp16 outside)."""
    H, emb, ids, w, tidx, tp = prep(fr, win)
    T = H.shape[0]
    outs = C.chain_forward(blk, H, emb, fr.cache, K=K)
    pw = torch.where(w >= 0.99, frac_assist, frac_other)
    sel = (torch.rand(T, device=H.device, generator=gen) < pw).float()
    total, wsum, extra = 0.0, 0.0, {}
    for k, m in enumerate(outs, 1):
        L = m.shape[0]
        t = torch.arange(L, device=H.device)
        p = t + k                       # teacher position
        ok = (t >= TMIN) & (p + 1 < T)
        wt = sel[(p + 1).clamp(max=T - 1)] * ok.float()
        if wt.sum() == 0: continue
        keep = wt > 0
        mk, ik, pk, wk = m[keep], tidx[p[keep]], tp[p[keep]], wt[keep]
        lsum, corr = 0.0, 0.0
        for s in range(0, mk.shape[0], chunk):
            l, am = checkpoint(_chunk_loss, mk[s: s + chunk], fr.lm.T, ik[s: s + chunk], pk[s: s + chunk], wk[s: s + chunk], use_reentrant=False)
            lsum = lsum + l
            corr += float((am == ik[s: s + chunk, 0]).float().sum())
        total = total + alphas[k - 1] * lsum / wk.sum()
        extra[f"ce{k}"] = float(lsum.detach()) / float(wk.sum()); extra[f"acc{k}"] = corr / float(wk.sum())
    return total, extra


def dev_type(t): return t.device.type


@torch.no_grad()
def window_metrics(blk, fr, win, chunk=1024):
    """Teacher-forced and chain-conditional acceptance statistics. Returns tensor [K, 2(pop: all, assistant), 6]:
    n, greedy_ok, sampled_accept_prob, n_chain, chain_greedy_ok, chain_sampled_prob  (sums, to be all-reduced)."""
    H, emb, ids, w, tidx, tp = prep(fr, win)
    T = H.shape[0]
    with torch.autocast("cuda", dtype=torch.float16, enabled=(dev_type(H) == "cuda")):
        outs = C.chain_forward(blk, H, emb, fr.cache, K=K)
    tgt = tidx[:, 0]                                            # target greedy token at every position
    res = torch.zeros(K, 2, 6, device=H.device)
    chain_ok = torch.ones(T, dtype=torch.bool, device=H.device)  # per origin: all earlier drafts accepted AND recorded tokens on the greedy path
    drafts = []
    for k, m in enumerate(outs, 1):
        L = m.shape[0]
        am = torch.cat([(m[s: s + chunk].to(fr.lm.dtype) @ fr.lm.T).argmax(-1) for s in range(0, L, chunk)])
        drafts.append(am)
        t = torch.arange(L, device=H.device); p = t + k
        ok = (t >= TMIN) & (p + 1 < T)
        g = (am == tgt[p]).float()
        sp = ((tidx[p] == am[:, None]).float() * tp[p]).sum(-1)
        cm = chain_ok[:L] & ok
        for pop, sel in ((0, ok), (1, ok & (w[(p + 1).clamp(max=T - 1)] >= 0.99))):
            c2 = cm & sel
            res[k - 1, pop] = torch.stack([sel.sum(), (g * sel).sum(), (sp * sel).sum(), c2.sum(), (g * c2).sum(), (sp * c2).sum()]).float()
        # extend the chain condition to step k+1: draft_k accepted AND recorded input token of step k+1 (X[t+k+1]) equals the target greedy at t+k
        nxt = torch.zeros(T, dtype=torch.bool, device=H.device)
        recorded_on_path = ids[(p + 1).clamp(max=T - 1)] == tgt[p]
        nxt[:L] = (am == tgt[p]) & recorded_on_path
        chain_ok = chain_ok & nxt
    return res


def summarize(res):
    """res: [K,2,6] summed tensor -> nested dict of rates + expected acceptance length proxy."""
    r = res.double().cpu().numpy()
    out = {}
    for pop, name in ((0, "all"), (1, "assistant")):
        d = {}
        a_s = []
        for k in range(K):
            n, g, sp, nc, cg, csp = r[k, pop]
            d[f"pos{k}"] = dict(n=int(n), top1=g / max(n, 1), sampled=sp / max(n, 1), chain_n=int(nc), chain_top1=cg / max(nc, 1), chain_sampled=csp / max(nc, 1))
            a_s.append(csp / max(nc, 1))
        # expected tokens per verification step if the chain rates hold (sampling acceptance, conditional chain)
        d["accept_len_sampled"] = 1 + a_s[0] + a_s[0] * a_s[1] + a_s[0] * a_s[1] * a_s[2]
        out[name] = d
    return out
