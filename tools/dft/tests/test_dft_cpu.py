"""CPU unit tests for Lane DFT (no GPU). Run: CUDA_VISIBLE_DEVICES= wt-integrate/.venv/bin/python tests/test_dft_cpu.py"""
import os, sys, math, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common as C
torch.manual_seed(0)

def tiny():
    b = C.MTPBlock(h=64, i=96, nh=4, nkv=2, hd=16, rot=8)
    for p in b.parameters():
        torch.nn.init.normal_(p, std=0.2) if p.ndim == 2 else torch.nn.init.normal_(p, std=0.1)
    return b

def test_chain_vs_naive():
    # tiny block; module-level ROT constant is used by apply_rope -> patch for the tiny test
    C.ROT = 8
    blk = tiny(); T, K = 20, 3
    ang = torch.rand(64, 4) * 6.28
    cache = torch.cat([ang.cos(), ang.sin()], dim=-1)           # (P, 8)
    Hs = torch.randn(T, 64); emb = torch.randn(T, 64)
    outs = C.chain_forward(blk, Hs, emb, cache, K=K)
    g = blk.nh // blk.nkv
    def attn(q, keys, vals):   # q (nh,hd); keys list of (nkv,hd)
        Kt = torch.stack(keys); Vt = torch.stack(vals)          # S,nkv,hd
        Kt = Kt.repeat_interleave(g, 1); Vt = Vt.repeat_interleave(g, 1)
        s = torch.einsum("hd,shd->hs", q, Kt) / math.sqrt(blk.hd)
        return torch.einsum("hs,shd->hd", s.softmax(-1), Vt)
    maxerr = 0.0
    with torch.no_grad():
        # step-1 keys for every position (written from TRUE hidden), positions 0..T-2
        k1, v1 = [], []
        for s in range(T - 1):
            _, k, v, _, _ = blk.front(Hs[s:s+1], emb[s+1:s+2], torch.tensor([s]), cache)
            k1.append(k[0]); v1.append(v[0])
        for t in range(T - 1):
            keys, vals = list(k1[: t + 1]), list(v1[: t + 1])
            hid = Hs[t:t+1]
            for step in range(1, K + 1):
                if t + step >= T: break
                pos = torch.tensor([t + step - 1])
                q, k, v, gate, res = blk.front(hid, emb[t+step:t+step+1], pos, cache)
                if step > 1:
                    keys.append(k[0]); vals.append(v[0])
                o = attn(q[0], keys, vals)
                out = blk.back(o.unsqueeze(0), gate, res)
                maxerr = max(maxerr, (out[0] - outs[step-1][t]).abs().max().item())
                hid = out
    assert maxerr < 2e-5, maxerr
    print("chain_vs_naive OK maxerr", maxerr)

def test_rope_and_norm_vs_vllm():
    C.ROT = 64
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    with set_current_vllm_config(VllmConfig()):
        r = get_rope(head_size=256, max_position=4096, rope_parameters=C.ROPE_PARAMS)
        n = GemmaRMSNorm(256, eps=1e-6)
    cache = r.cos_sin_cache[:4096].float()
    L = 37; pos = torch.randint(0, 3000, (L,))
    q = torch.randn(L, 24 * 256); k = torch.randn(L, 4 * 256)
    q2, k2 = r.forward_native(pos, q.clone(), k.clone())
    mine_q = C.apply_rope(q.view(L, 24, 256), pos, cache).reshape(L, -1)
    mine_k = C.apply_rope(k.view(L, 4, 256), pos, cache).reshape(L, -1)
    assert torch.allclose(q2, mine_q, atol=1e-5) and torch.allclose(k2, mine_k, atol=1e-5), ((q2-mine_q).abs().max(),)
    with torch.no_grad():
        n.weight.copy_(torch.randn(256) * 0.1)
    x = torch.randn(L, 24, 256) * 3
    ref = n.forward_native(x.view(-1, 256)).view(L, 24, 256)
    assert torch.allclose(ref, C.gnorm(x, n.weight), atol=1e-5)
    print("rope/norm vs vLLM OK")

def test_teacher():
    lg = torch.randn(10, 100) * 3
    idx, p = C.teacher_topk(lambda h: h, lg)
    assert torch.allclose(p.sum(-1), torch.ones(10), atol=1e-5)
    assert ((p > 0).sum(-1) <= 20).all()
    assert (idx[:, 0] == lg.argmax(-1)).all()
    print("teacher OK")

def test_ckpt_roundtrip():
    import tempfile
    from safetensors import safe_open
    src = os.path.join(C.MODEL_DIR, "model-mtp.safetensors")
    blk = C.MTPBlock(); blk.load_ckpt(src)
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "model-mtp.safetensors"); blk.export_ckpt(out)
        with safe_open(src, "pt") as a, safe_open(out, "pt") as b:
            assert set(a.keys()) == set(b.keys())
            for k in a.keys():
                assert a.get_tensor(k).dtype == b.get_tensor(k).dtype and torch.equal(a.get_tensor(k), b.get_tensor(k)), k
    print("ckpt roundtrip identical OK")


def test_loss_and_metrics_tiny():
    """end-to-end window_loss / window_metrics on tiny CPU shapes: loss decreases when the tiny block is trained for a few steps"""
    import evalcore as E
    C.ROT = 8
    class FR: pass
    fr = FR(); V = 200; fr.lm = torch.randn(V, 64); fr.emb = torch.randn(V, 64); fr.dev = torch.device("cpu"); fr.dtype = torch.float32
    ang = torch.rand(512, 4) * 6.28; fr.cache = torch.cat([ang.cos(), ang.sin()], -1)
    fr.logits = lambda h: h @ fr.lm.T
    blk = tiny(); T = 160
    win = dict(id="t", ids=torch.randint(0, V, (T,)), w=torch.where(torch.rand(T) < 0.5, 1.0, 0.3), H=torch.randn(T, 64))
    opt = torch.optim.Adam(blk.parameters(), lr=3e-3)
    l0 = None
    for it in range(25):
        opt.zero_grad(); loss, ex = E.window_loss(blk, fr, win, frac_assist=1.0, frac_other=1.0); loss.backward(); opt.step()
        if l0 is None: l0 = float(loss)
    assert float(loss) < l0 * 0.9, (l0, float(loss))
    res = E.window_metrics(blk, fr, win)
    s = E.summarize(res); assert s["all"]["pos0"]["n"] > 50
    print("loss/metrics tiny OK", round(l0, 3), "->", round(float(loss), 3), "train top1", round(s["all"]["pos0"]["top1"], 3))

if __name__ == "__main__":
    for f in (test_chain_vs_naive, test_rope_and_norm_vs_vllm, test_teacher, test_ckpt_roundtrip, test_loss_and_metrics_tiny):
        f()
    print("ALL PASS")
