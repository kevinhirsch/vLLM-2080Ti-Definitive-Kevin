"""Lane U2 helper: build a REAL compressed-tensors WNA16 (Marlin) layer from pack-quantized tensors, exactly as the
engine does for any quantized Linear / ParallelLMHead, so a microbench or unit check exercises the production kernel path.
Needs a CUDA device.  Memory: ~ (N*K/2 bytes) + scales; keep N*K small next to a live engine."""
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import CompressedTensorsWNA16
from vllm.platforms import current_platform


import vllm.model_executor.parameter as _vparam
_vparam.get_tensor_model_parallel_rank = lambda: 0  # standalone: no distributed init
_vparam.get_tensor_model_parallel_world_size = lambda: 1


class _Holder(torch.nn.Module):
    pass


def make_marlin_layer(t: dict, N: int, K: int, device="cuda", dtype=torch.float16, group=128, symmetric=False):
    """t: output of u2_headquant.quantize_linear (weight_packed [N,K/8], weight_scale [N,K/g], weight_zero_point [N/8,K/g])."""
    scheme = CompressedTensorsWNA16(strategy="group", num_bits=4, group_size=group, symmetric=symmetric, layer_name="u2.test")
    layer = _Holder()

    def loader(param, w):  # un-sharded copy; the engine's weight_loader does the same for tp=1
        param.data.copy_(w)

    with set_current_vllm_config(VllmConfig()):
        scheme.create_weights(layer, output_size=N, input_size=K, output_partition_sizes=[N],
                              input_size_per_partition=K, params_dtype=dtype, weight_loader=loader)
        layer.to(device)
        for k in [k for k in ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape") if k in t]:
            getattr(layer, k).data.copy_(t[k].to(getattr(layer, k).dtype))
        scheme.process_weights_after_loading(layer)
    return layer, scheme
