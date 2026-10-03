#!/usr/bin/env python
"""Lane U2 unit check (CPU only, no GPU, no engine): int4 head/MTP quantizer vs the compressed-tensors library.

Checks
  1. layout: our pack of (q, zp) is bit-identical to compressed_tensors pack_to_int32 (packed_dim 1 / 0);
  2. decompress: the stock CT PackedQuantizationCompressor.decompress of OUR tensors == our dequantize_linear;
  3. format equality on a REAL checkpoint layer: our dequantize_linear on the shipped down_proj tensors
     == CT decompress of the same tensors (so our reading of the format is the library's);
  4. real lm_head rows + real MTP linears: relative error / SQNR of the int4 reconstruction, RTN vs MSE-clip;
  5. env gating: ignore_filter / wants() are identity with the env unset.
Run: CUDA_VISIBLE_DEVICES= python tools/u2/test_headquant.py
"""
import importlib.util, os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("u2hq", os.path.join(HERE, "../../vllm/model_executor/layers/quantization/u2_headquant.py"))
hq = importlib.util.module_from_spec(spec); spec.loader.exec_module(hq)
from compressed_tensors.compressors.pack_quantized import PackedQuantizationCompressor
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

M = "/home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven/"
args = QuantizationArgs(num_bits=4, type="int", symmetric=False, strategy="group", group_size=128, observer="memoryless_minmax")
scheme = QuantizationScheme(targets=["Linear"], weights=args)
ok = True
def check(name, cond, extra=""):
    global ok
    ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, extra, flush=True)

def ct_decompress(t):
    sd = {k: v for k, v in t.items()}
    out = PackedQuantizationCompressor.decompress(sd, scheme)
    return out["weight"].float()

# 0. env gating
for k in ("VLLM_U2_INT4_HEAD", "VLLM_U2_INT4_MTP"): os.environ.pop(k, None)
ign = ["lm_head", "re:.*mtp.*", "re:mtp\\..*", "model.visual.x"]
check("gating: env unset -> ignore untouched, wants() False", hq.ignore_filter(ign) == ign and not hq.wants("lm_head.weight") and not hq.wants("mtp.fc.weight"))
os.environ["VLLM_U2_INT4_HEAD"] = "1"
check("gating: HEAD only drops lm_head", hq.ignore_filter(ign) == ["re:.*mtp.*", "re:mtp\\..*", "model.visual.x"] and hq.wants("lm_head.weight") and not hq.wants("mtp.fc.weight"))
os.environ["VLLM_U2_INT4_HEAD"] = "0"; os.environ["VLLM_U2_INT4_MTP"] = "1"
check("gating: MTP only drops mtp regexes", hq.ignore_filter(ign) == ["lm_head", "model.visual.x"] and hq.wants("mtp.layers.0.mlp.up_proj.weight") and not hq.wants("mtp.layers.0.input_layernorm.weight") and not hq.wants("lm_head.weight"))
os.environ.pop("VLLM_U2_INT4_MTP")

# 1+2 synthetic
torch.manual_seed(0)
W = (torch.randn(256, 512) * 0.02).bfloat16()
t = hq.quantize_linear(W)
dq = hq.dequantize_linear(t)
check("shapes/dtypes", t["weight_packed"].shape == (256, 64) and t["weight_packed"].dtype == torch.int32 and t["weight_scale"].shape == (256, 4) and t["weight_scale"].dtype == torch.bfloat16 and t["weight_zero_point"].shape == (32, 4) and t["weight_zero_point"].dtype == torch.int32 and t["weight_shape"].dtype == torch.int64)
# independent re-derivation of q/zp from our dequant is impossible; instead rebuild q,zp via CT unpack and re-pack with CT
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
q = unpack_from_int32(t["weight_packed"], 4, torch.Size([256, 512]))
zp = unpack_from_int32(t["weight_zero_point"], 4, torch.Size([256, 4]), packed_dim=0)
check("pack == CT pack_to_int32 (weights)", torch.equal(pack_to_int32(q.to(torch.int8), 4), t["weight_packed"]))
check("pack == CT pack_to_int32 (zero points, dim0)", torch.equal(pack_to_int32(zp.to(torch.int8), 4, packed_dim=0), t["weight_zero_point"]))
check("q range [-8,7], zp range [-8,7]", int(q.min()) >= -8 and int(q.max()) <= 7 and int(zp.min()) >= -8 and int(zp.max()) <= 7)
ct = ct_decompress(t)
check("CT decompress(ours) == our dequantize_linear (CT returns scale dtype=bf16)", torch.equal(ct, dq.bfloat16().float()), f"maxabs diff vs fp32 {(ct-dq).abs().max():.3e} (bf16 ulp)")
rel = ((dq - W.float()).norm() / W.float().norm()).item()
check("synthetic gaussian int4 rel err sane (<0.12)", rel < 0.12, f"rel={rel:.4f}")

# 3 real shipped layer: our reader == CT reader
h = safe_open(M + "model.safetensors", "pt")
p = "model.language_model.layers.3.mlp.down_proj"
ship = {s: h.get_tensor(f"{p}.{s}") for s in ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape")}
a, b = hq.dequantize_linear(ship), ct_decompress(ship)
check("shipped layer: our dequantize == CT decompress (to bf16 rounding)", torch.equal(a.bfloat16().float(), b), f"{tuple(a.shape)}")

# 4 real lm_head rows, MTP linears: quality of reconstruction
def stats(name, Wb, grid):
    t0 = time.time(); tt = hq.quantize_linear(Wb, grid=grid); d = hq.dequantize_linear(tt)
    e = (d - Wb.float()).pow(2).sum().item(); s = Wb.float().pow(2).sum().item()
    return 10 * torch.log10(torch.tensor(s / e)).item(), time.time() - t0
lm = h.get_slice("lm_head.weight")
rows = torch.cat([h.get_tensor("lm_head.weight")[r:r+2048] for r in (0, 100000, 240000)]) if False else None
full = h.get_tensor("lm_head.weight")
for lo in (0, 120000, 246272):
    Wb = full[lo:lo + 2048]
    s1, _ = stats("lm_head", Wb, (1.0,)); s2, dt = stats("lm_head", Wb, hq.CLIP_GRID)
    check(f"lm_head rows {lo}: MSE-clip SQNR >= plain RTN", s2 >= s1 - 1e-6, f"RTN {s1:.2f} dB -> clip {s2:.2f} dB ({dt:.1f}s/2048 rows)")
m = safe_open(M + "model-mtp.safetensors", "pt")
for n in ("mtp.layers.0.self_attn.k_proj.weight", "mtp.layers.0.mlp.down_proj.weight", "mtp.fc.weight"):
    Wb = m.get_tensor(n)[:2048]
    s2, dt = stats(n, Wb, hq.CLIP_GRID)
    check(f"{n} quantizes (in%128==0)", True, f"SQNR {s2:.2f} dB")
# 6 symmetric (embedding path): our layout == what the CT embedding triton kernel reads (nibble-8)*scale
Wm = (torch.randn(128, 512) * 0.02).bfloat16()
te = hq.quantize_linear(Wm, symmetric=True)
check("sym: no zero_point tensor emitted", "weight_zero_point" not in te and set(te) == {"weight_packed", "weight_scale", "weight_shape"})
de = hq.dequantize_linear(te)
sh = torch.arange(8, dtype=torch.int32) * 4
qk = (((te["weight_packed"].unsqueeze(-1) >> sh) & 15).reshape(128, 512) - 8).float()
kern = (qk.reshape(128, 4, 128) * te["weight_scale"].float().unsqueeze(-1)).reshape(128, 512)  # same arithmetic as _dequant_gather_kernel
check("sym: dequantize_linear == embedding-kernel arithmetic", torch.equal(de, kern))
check("sym rel err sane (<0.14)", ((de - Wm.float()).norm() / Wm.float().norm()).item() < 0.14, f"{((de-Wm.float()).norm()/Wm.float().norm()).item():.4f}")
emb = h.get_tensor("model.language_model.embed_tokens.weight")[:2048]
tt = hq.quantize_linear(emb, symmetric=True); dd = hq.dequantize_linear(tt)
check("real embed rows int4-sym", True, f"SQNR {10*torch.log10(emb.float().pow(2).sum()/(dd-emb.float()).pow(2).sum()).item():.2f} dB")
os.environ["VLLM_U2_INT4_EMBED"] = "1"
check("gating: EMBED wants embed name, nothing else", hq.wants("model.language_model.embed_tokens.weight") and not hq.wants("lm_head.weight") and hq.ignore_filter(ign) == ign)
os.environ.pop("VLLM_U2_INT4_EMBED")
print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
