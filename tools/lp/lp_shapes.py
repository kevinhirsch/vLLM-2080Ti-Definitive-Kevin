"""Real checkpoint per-rank TP=2 linear shapes (rank-0 slices) for LP benches."""
import torch
from safetensors import safe_open

MD = "/home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven/model.safetensors"
P = "model.language_model.layers"
f = safe_open(MD, "pt")
g = lambda n: f.get_tensor(n)


def col(names, rows):  # column-parallel: rank-0 rows of each, concatenated (merged layer)
    out = {"weight_packed": [], "weight_scale": [], "weight_zero_point": []}
    for n, r in zip(names, rows):
        out["weight_packed"].append(g(n + ".weight_packed")[:r]); out["weight_scale"].append(g(n + ".weight_scale")[:r])
        out["weight_zero_point"].append(g(n + ".weight_zero_point")[: r // 8])
    t = {k: torch.cat(v, 0) for k, v in out.items()}
    return t, t["weight_packed"].shape[0], t["weight_packed"].shape[1] * 8


def row(n, k):  # row-parallel: rank-0 input slice
    t = {"weight_packed": g(n + ".weight_packed")[:, : k // 8], "weight_scale": g(n + ".weight_scale")[:, : k // 128],
         "weight_zero_point": g(n + ".weight_zero_point")[:, : k // 128]}
    return {k2: v.contiguous() for k2, v in t.items()}, t["weight_packed"].shape[0], k


SHAPES = {
    "gate_up": (lambda: col([f"{P}.3.mlp.gate_proj", f"{P}.3.mlp.up_proj"], [8704, 8704]), 64),
    "down": (lambda: row(f"{P}.3.mlp.down_proj", 8704), 64),
    "gdn_qkvz": (lambda: col([f"{P}.0.linear_attn.in_proj_qkv", f"{P}.0.linear_attn.in_proj_z"], [5120, 3072]), 48),
    "gdn_out": (lambda: row(f"{P}.0.linear_attn.out_proj", 3072), 48),
    "attn_qkv": (lambda: col([f"{P}.3.self_attn.q_proj", f"{P}.3.self_attn.k_proj", f"{P}.3.self_attn.v_proj"], [6144, 512, 512]), 16),
    "attn_o": (lambda: row(f"{P}.3.self_attn.o_proj", 3072), 16),
}
