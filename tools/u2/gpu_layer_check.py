#!/usr/bin/env python
"""Lane U2 GPU unit check of the REAL engine path for an int4 lm_head: real CompressedTensorsConfig (VLLM_U2_INT4_HEAD=1, so
`lm_head` leaves `ignore` and ParallelLMHead falls back to the Linear scheme), real ParallelLMHead with the real VocabParallelEmbedding.weight_loader
under an emulated TP=2 (both ranks), real Marlin repack + gptq_marlin_gemm, vs a dequantized fp32 matmul.  Toy vocab (4096 rows of the real lm_head),
so it needs ~0.35 GB of GPU incl. context: safe next to a live engine if >=1 GiB is free (checked below).  Also runs fine inside an S3 window.
Run: VLLM_U2_INT4_HEAD=1 python tools/u2/gpu_layer_check.py [--device 1] [--min-free-mib 1200]
"""
import argparse, importlib.util, json, os, subprocess, sys, tempfile
os.environ["VLLM_U2_INT4_HEAD"] = "1"
os.environ["VLLM_U2_INT4_EMBED"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ap = argparse.ArgumentParser(); ap.add_argument("--device", type=int, default=1); ap.add_argument("--min-free-mib", type=int, default=1200)
ap.add_argument("--vocab", type=int, default=4096); ap.add_argument("--ref", default="/home/kevin/projects/lanes/u2-quant/ref.pt")
ap.add_argument("--model", default="/home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven")
a = ap.parse_args()
free = int(subprocess.check_output(["nvidia-smi", "-i", str(a.device), "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).decode().strip())
if free < a.min_free_mib:
    sys.exit(f"REFUSED: GPU{a.device} has only {free} MiB free (< {a.min_free_mib}); not risking the engine")
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.device)
os.environ["VLLM_U2_CACHE_DIR"] = tempfile.mkdtemp(prefix="u2chk_")
import torch
torch.cuda.set_per_process_memory_fraction(0.04)  # ~0.9 GiB hard cap on this process's tensors
from safetensors import safe_open
from vllm.config import VllmConfig, set_current_vllm_config
import vllm.model_executor.layers.vocab_parallel_embedding as vpe
import vllm.model_executor.parameter as vparam
from vllm.model_executor.layers.quantization import u2_headquant as hq
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig, CompressedTensorsLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

qc = json.load(open(f"{a.model}/config.json"))["quantization_config"]
cfg = CompressedTensorsConfig.from_config(qc)
assert "lm_head" not in cfg.ignore, "ignore_filter did not drop lm_head"
print("PASS config: lm_head no longer ignored; mtp regexes kept:", [i for i in cfg.ignore if "mtp" in i])

V, D = a.vocab, 5120
Wb = safe_open(f"{a.model}/model.safetensors", "pt").get_tensor("lm_head.weight")[:V]
class L: info = staticmethod(lambda f, *x: None); warning = info
tensors = {}
for name, t in hq.wrap_weights(iter([("lm_head.weight", Wb)]), a.model, L):
    tensors[name.split("lm_head.")[1]] = t
print("PASS wrap_weights ->", {k: (tuple(v.shape), str(v.dtype).replace("torch.", "")) for k, v in tensors.items()})
Wdq = hq.dequantize_linear({k: tensors[k] for k in ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape")})

ref = torch.load(a.ref)["H"] if os.path.exists(a.ref) else None
x = (ref.reshape(-1, D)[16:16 + 8] if ref is not None else torch.randn(8, D)).half().cuda()
want = (x.float().cpu() @ Wdq.T)  # fp32 reference of the int4 head

shards = []
for rank in range(2):
    vpe.get_tensor_model_parallel_rank = lambda r=rank: r
    vpe.get_tensor_model_parallel_world_size = lambda: 2
    vparam.get_tensor_model_parallel_rank = lambda r=rank: r
    vparam.get_tensor_model_parallel_world_size = lambda: 2
    with set_current_vllm_config(VllmConfig()):
        head = ParallelLMHead(V, D, quant_config=cfg, prefix="lm_head", params_dtype=torch.float16)
        assert isinstance(head.quant_method, CompressedTensorsLinearMethod), type(head.quant_method)
        head.to("cuda")
        for k, v in tensors.items():
            p = getattr(head, k)
            p.weight_loader(p, v) if hasattr(p, "weight_loader") else head.weight_loader(p, v)
        head.quant_method.process_weights_after_loading(head)
        y = head.quant_method.apply(head, x).float().cpu()
    shards.append(y)
    print(f"PASS rank {rank}: CompressedTensorsLinearMethod + Marlin ran, out {tuple(y.shape)}")
got = torch.cat(shards, 1)
err = (got - want).abs().max().item(); rel = ((got - want).norm() / want.norm()).item()
print(f"logits int4-head(Marlin, TP2 shard-loaded) vs fp32 dequant ref: max|d|={err:.4f} rel L2={rel:.2e}  argmax agree={(got.argmax(1)==want.argmax(1)).float().mean().item():.2f}")
assert rel < 5e-3, "Marlin output differs from dequantized reference"
# ---- embedding arm: real VocabParallelEmbedding + CT embedding method, TP2 emulated, vs dequantized lookup
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_embedding import CompressedTensorsEmbeddingWNA16Int
E = safe_open(f"{a.model}/model.safetensors", "pt").get_tensor("model.language_model.embed_tokens.weight")[:V]
et = {}
for name, t in hq.wrap_weights(iter([("model.language_model.embed_tokens.weight", E)]), a.model, L):
    et[name.split("embed_tokens.")[1]] = t
assert set(et) == {"weight_packed", "weight_scale", "weight_shape"}, set(et)
Edq = hq.dequantize_linear(et).bfloat16().float()
ids = torch.tensor([0, 5, 77, 2047, 2048, 3000, 4095], device="cuda")
outs = []
for rank in range(2):
    vpe.get_tensor_model_parallel_rank = lambda r=rank: r
    vpe.get_tensor_model_parallel_world_size = lambda: 2
    vparam.get_tensor_model_parallel_rank = lambda r=rank: r
    vparam.get_tensor_model_parallel_world_size = lambda: 2
    with set_current_vllm_config(VllmConfig()):
        emb = VocabParallelEmbedding(V, D, quant_config=cfg, params_dtype=torch.float16)
        assert isinstance(emb.quant_method, CompressedTensorsEmbeddingWNA16Int), type(emb.quant_method)
        emb.to("cuda")
        for k, v in et.items():
            p = getattr(emb, k); p.weight_loader(p, v)
        lo, hi = rank * V // 2, (rank + 1) * V // 2
        inr = (ids >= lo) & (ids < hi)
        o = emb.quant_method.embedding(emb, (ids - lo).clamp(0, V // 2 - 1)).float().cpu()
        outs.append(torch.where(inr.cpu()[:, None], o, torch.zeros(())))
got = outs[0] + outs[1]
want = Edq[ids.cpu()]
print(f"PASS embedding int4 lookup (TP2 shard-loaded) vs dequant ref: max|d|={(got-want).abs().max():.2e}")
assert (got - want).abs().max() < 2e-3
print("ALL PASS; torch max mem MiB", torch.cuda.max_memory_allocated() / 2**20)
