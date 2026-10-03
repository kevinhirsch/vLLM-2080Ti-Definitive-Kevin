"""Lane K8: offload the big fp32 matmuls of the CPU reference harness to a spare slice of GPU1 (the prod engine owns the
rest).  Hard VRAM cap ~300 MiB: weights stream in row blocks, tokens in chunks.  Math stays fp32 (no TF32 on Turing)."""
import os, torch
import torch.nn.functional as F

_dev = torch.device(os.environ.get("K8_GPU", "cuda:1"))
_orig_linear = F.linear
ROWS, TOK = 1024, 1024


_last = [0.0]


def window_guard():
    """Coordinator rule 2026-10-03: never hold GPU while a planned-offline window is open. Checked at most every 8 s."""
    import time, json, urllib.request
    if time.time() - _last[0] < 8:
        return
    _last[0] = time.time()
    try:
        d = json.load(urllib.request.urlopen("http://localhost:8000/gateway/capacity", timeout=4))
        if d.get("planned_offline"):
            print("WINDOW OPEN (planned_offline) -> exiting, GPU released:", d.get("why"), flush=True)
            os._exit(75)
    except Exception as e:
        print("capacity check failed (continuing):", e, flush=True)


def linear(x, W, b=None):
    if (W.dim() != 2 or W.numel() < 4_000_000 or x.dim() != 2 or x.dtype != torch.float32 or W.dtype != torch.float32
            or b is not None):
        return _orig_linear(x, W, b)
    window_guard()
    n_in = W.shape[1]
    tok = max(256, min(x.shape[0], (48 << 20) // (4 * n_in)))
    rows = max(256, (24 << 20) // (4 * n_in))
    out = torch.empty(x.shape[0], W.shape[0], dtype=torch.float32)
    for t0 in range(0, x.shape[0], tok):
        xg = x[t0:t0 + tok].to(_dev)
        for r0 in range(0, W.shape[0], rows):
            wg = W[r0:r0 + rows].to(_dev)
            out[t0:t0 + tok, r0:r0 + rows] = _orig_linear(xg, wg).cpu()
    return out


def install():
    _thread_guard()
    F.linear = linear
    torch.cuda.set_per_process_memory_fraction(min(0.02, 1.0), _dev) if False else None


def gdn_core(fn, q, k, v, g, beta, TP=None, hs=4):
    """run HF torch_chunk_gated_delta_rule head-chunked on the GPU. returns core [B,T,H,V] (and S_tp [B,H,K,V] at TP)."""
    B, T, H, _ = q.shape
    outs, sts = [], []
    for h0 in range(0, H, hs):
        sl = slice(h0, h0 + hs)
        a = [t[:, :, sl].to(_dev) for t in (q, k, v)]
        gg, bb = g[:, :, sl].to(_dev), beta[:, :, sl].to(_dev)
        core, _ = fn(*a, g=gg, beta=bb, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
        outs.append(core.cpu())
        if TP is not None:
            _, st = fn(*[t[:, :TP] for t in a], g=gg[:, :TP], beta=bb[:, :TP], initial_state=None, output_final_state=True,
                       use_qk_l2norm_in_kernel=True)
            sts.append(st.cpu())
    return torch.cat(outs, 2), (torch.cat(sts, 1) if TP is not None else None)


def sdpa_causal(q, k, v, scale, hs=4):
    """q,k,v [B,H,T,hd] cpu -> [B,H,T,hd] using GPU head chunks"""
    outs = []
    for h0 in range(0, q.shape[1], hs):
        sl = slice(h0, h0 + hs)
        outs.append(F.scaled_dot_product_attention(q[:, sl].to(_dev), k[:, sl].to(_dev), v[:, sl].to(_dev), is_causal=True, scale=scale).cpu())
    return torch.cat(outs, 1)


def _thread_guard(period=2.0):
    """Independent watchdog thread: runs the shared gpuok.sh gate (NEED=0 so our own VRAM does not trip it) every
    `period` s and hard-exits (os._exit 75) the moment a window/boot/busy signal appears, whatever the compute path is doing."""
    import threading, subprocess, time

    def loop():
        while True:
            time.sleep(period)
            try:
                rc = subprocess.run([os.path.expanduser("~/projects/lanes/windows/gpuok.sh"), "1", "0"], capture_output=True, timeout=20)
                if rc.returncode == 1:
                    print("K8 GUARD thread: gpuok refused -> exiting, GPU released:", rc.stdout.decode()[:200], flush=True)
                    os._exit(75)
            except Exception as e:
                print("K8 guard check error (continuing):", e, flush=True)
    threading.Thread(target=loop, daemon=True).start()
