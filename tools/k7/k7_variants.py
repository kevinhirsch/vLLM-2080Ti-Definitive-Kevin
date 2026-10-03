"""Lane K7 variants for Lane LP's shared gate (wt-lp/tools/lp/variant_gate.py), loaded with
    VG_PLUGINS=/home/kevin/Desktop/wt-k7/tools/k7/k7_variants.py python tools/lp/variant_gate.py --variants k7_pc,...
Rotation y = W x = (W H^T)(H x), H = blockdiag Hadamard(K7_HB, default 128) on the input dim (= k7 act_quant_h128 kernel).
Weights: shipped W4 dequant -> rotate -> symmetric int4 RTN with per-row/group MSE clip (rotquant.sym_quant).
  k7_a4x   exact rotated fp32 weights, per-token int4 act               (activation error alone)
  k7_w4r   per-channel re-quantized rotated weights, fp32 act           (weight double-quant error alone)
  k7_pc    W4A4 per-channel W x per-token A on every Marlin linear     (= the built CUTLASS s4 kernel)
  k7_pcin  k7_pc on input projections only (gate/up, in_proj_qkv/z, q/k/v); down/out/o stay W4A16
  k7_pcnd  k7_pc everywhere except mlp.down_proj (W4A16)
  k7_g128  W4A4 g128 W x g128 A (needs a group-epilogue kernel)
  k7_g128in  k7_g128 on input projections only
Per-linear activation SQNR (rotated basis) is dumped at exit to $K7_STATS (default ~/projects/lanes/k7/vg_stats.json)."""
import atexit, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rotquant as RQ

HB = int(os.environ.get("K7_HB", "128"))
CLIP = float(os.environ.get("K7_ACT_CLIP", "1.0"))
IN_PROJ = ("gate_proj", "up_proj", "in_proj_qkv", "in_proj_z", "q_proj", "k_proj", "v_proj")
_cache = {"layer": None, "w": {}}
STATS = {}


def _wq(n, info, kind):
    """Rotated (and re-quantized) weight on the device of info["w"].  Caches only CPU int8 codes + fp32 scales per layer,
    so the GPU mode of variant_gate never accumulates per-layer weight copies on the GPU."""
    w = info["w"]
    layer = n.split(".")[0]
    if _cache["layer"] != layer:
        _cache["layer"], _cache["w"] = layer, {}
    wr = RQ.rotate_weight(w, HB)
    if kind == "rx":
        return wr
    key = (n, kind)
    g = 128 if kind == "g128" else 0
    if key not in _cache["w"]:
        if kind == "gpc":
            c, s = _gptq_load(n)
        else:
            c, s = RQ.sym_quant(wr, 4, g)
        c, s = c.cpu(), s.cpu()
        d = RQ.dequant(c.to(wr.device), s.to(wr.device), g)
        STATS.setdefault(n, {})[f"w_{kind}_sqnr_db"] = (10 * torch.log10(wr.pow(2).sum() / (d - wr).pow(2).sum())).item()
        _cache["w"][key] = (c, s)
        return d
    c, s = _cache["w"][key]
    return RQ.dequant(c.to(wr.device), s.to(wr.device), g)


GPTQ_DIR = os.environ.get("K7_GPTQ_DIR", "")
_gq = {}


def _gptq_load(n):
    """n = 'L<i>.<linear>' -> (codes int8 [N,K], scale fp32 [N]) from gptq_requant.py output."""
    from safetensors import safe_open
    li, ln = n.split(".", 1)
    ln, _, r0 = ln.partition("#r")
    f = os.path.join(GPTQ_DIR, f"layer_{int(li[1:]):02d}.safetensors")
    import time as _t
    t0 = _t.time()
    while not os.path.exists(f) or _t.time() - os.path.getmtime(f) < 5:  # gate may trail a running gptq_requant.py
        if _t.time() - t0 > float(os.environ.get("K7_GPTQ_WAIT_S", "3600")):
            raise FileNotFoundError(f)
        _t.sleep(5)
    with safe_open(f, "pt") as h:
        c, s = RQ.unpack_s4(h.get_tensor(f"{ln}.codes")), h.get_tensor(f"{ln}.scale")
    if r0:
        r0 = int(r0); c, s = c[r0:r0 + 2048], s[r0:r0 + 2048]  # must match variant_gate --gpu-rows (default 2048)
    return c, s


def _aq(n, x, grp):
    xr = RQ.block_had(x.half().float(), HB)
    xa = RQ.act_quant(x.half().float(), 4, grp, HB, CLIP)
    st = STATS.setdefault(n, {})
    k = f"a_{'g128' if grp else 'pt'}_sqnr_db"
    if k not in st:
        st[k] = (10 * torch.log10(xr.pow(2).sum() / (xa - xr).pow(2).sum().clamp(min=1e-30))).item()
        st["absmax_over_rms"] = (x.abs().amax(-1) / x.pow(2).mean(-1).sqrt().clamp(min=1e-8)).median().item()
        st["absmax_over_rms_rot"] = (xr.abs().amax(-1) / xr.pow(2).mean(-1).sqrt().clamp(min=1e-8)).median().item()
    return xa


AG_BITS = int(os.environ.get("K7_AG_SBITS", "11"))


def _aq_g_int(n, x):
    """Activation path of the group-scaled W4A4 kernel (k7 w4a4g): block-Hadamard, per-(row,128) int4 codes, and the group
    scale stored as an integer multiplier of the row-max group scale (s_int = round(s_g / s_max * 2^AG_BITS)) so the kernel
    folds it into the int32 accumulator with one IMAD per group (no float epilogue per group)."""
    xr = RQ.block_had(x.half().float(), HB)
    xg = xr.reshape(xr.shape[0], -1, 128)
    sg = (xg.abs().amax(-1, keepdim=True) * CLIP / 7).clamp_min(1e-12)
    q = torch.clamp(torch.round(xg / sg), -8, 7)
    smax = sg.amax(1, keepdim=True)
    s_int = torch.round(sg / smax * 2 ** AG_BITS).clamp_min(1)
    xa = (q * s_int * smax / 2 ** AG_BITS).reshape(xr.shape)
    st = STATS.setdefault(n, {})
    if "a_gint_sqnr_db" not in st:
        st["a_gint_sqnr_db"] = (10 * torch.log10(xr.pow(2).sum() / (xa - xr).pow(2).sum().clamp(min=1e-30))).item()
    return xa


def v_k7_agpc(n, x, info):
    return _aq_g_int(n, x) @ _wq(n, info, "pc").T


def v_k7_agpcin(n, x, info):
    return v_k7_agpc(n, x, info) if _short(n) in IN_PROJ else _w4a16(x, info)


# GPTQ weights (K7_GPTQ_DIR = gptq_requant.py output) instead of RTN
def v_k7_gpc(n, x, info):
    return _aq(n, x, 0) @ _wq(n, info, "gpc").T


def v_k7_gagpc(n, x, info):
    return _aq_g_int(n, x) @ _wq(n, info, "gpc").T


def v_k7_gagpcin(n, x, info):
    return v_k7_gagpc(n, x, info) if _short(n) in IN_PROJ else _w4a16(x, info)


def v_k7_gw4r(n, x, info):  # GPTQ weight error alone (fp activations)
    return RQ.block_had(x.half().float(), HB) @ _wq(n, info, "gpc").T


def _short(n):
    return n.split(".", 1)[1].split("#", 1)[0]


def v_k7_a4x(n, x, info):
    return _aq(n, x, 0) @ _wq(n, info, "rx").T


def v_k7_w4r(n, x, info):
    return RQ.block_had(x.half().float(), HB) @ _wq(n, info, "pc").T


def v_k7_pc(n, x, info):
    return _aq(n, x, 0) @ _wq(n, info, "pc").T


def _w4a16(x, info):
    return x.half().float() @ info["w"].T


def v_k7_pcin(n, x, info):
    return v_k7_pc(n, x, info) if _short(n) in IN_PROJ else _w4a16(x, info)


def v_k7_pcnd(n, x, info):
    return _w4a16(x, info) if _short(n) == "down_proj" else v_k7_pc(n, x, info)


def v_k7_g128(n, x, info):
    return _aq(n, x, 128) @ _wq(n, info, "g128").T


def v_k7_g128in(n, x, info):
    return v_k7_g128(n, x, info) if _short(n) in IN_PROJ else _w4a16(x, info)


def _dump():
    if not STATS:
        return
    by = {}
    for n, s in STATS.items():
        for k, v in s.items():
            by.setdefault(_short(n), {}).setdefault(k, []).append(v)
    med = {t: {k: sorted(v)[len(v) // 2] for k, v in d.items()} for t, d in by.items()}
    mn = {t: {k: min(v) for k, v in d.items() if "sqnr" in k} for t, d in by.items()}
    p = os.environ.get("K7_STATS", "/home/kevin/projects/lanes/k7/vg_stats.json")
    json.dump({"hb": HB, "act_clip": CLIP, "by_type_median": med, "by_type_min": mn, "per_linear": STATS}, open(p, "w"), indent=1)
    print("K7 STATS by type (median):", json.dumps(med), flush=True)


atexit.register(_dump)


def register(VARIANTS):
    for k, v in list(globals().items()):
        if k.startswith("v_k7_"):
            VARIANTS[k[2:]] = v
