# SPDX-License-Identifier: Apache-2.0
"""Lane K3 (L78): overlap the tensor-parallel all-reduce of a RowParallelLinear
with the GEMM that produces it.

For a large prefill chunk (e.g. 3632 tokens x 5120 fp16 = 37 MB per all-reduce,
128 of them per forward) the all-reduce is serialized behind its GEMM. Nothing
physical forces that: the GEMM output rows are independent, so the all-reduce
of rows [0, m) can ride NVLink on a high-priority side stream while the GEMM
computes rows [m, n).

Env-gated, default OFF:
  VLLM_K3_AR_OVERLAP=1            enable
  VLLM_K3_AR_OVERLAP_CHUNKS=2     row chunks per layer (>=2)
  VLLM_K3_AR_OVERLAP_MIN_TOKENS=1024
                                  below this (all decode / CUDA-graph sizes)
                                  the plain GEMM + all-reduce path is used

The op is opaque to torch.compile; below the token threshold or while a CUDA
graph is being captured it runs exactly the original two calls.
"""

from __future__ import annotations

import os
import weakref

import torch

from vllm.utils.torch_utils import direct_register_custom_op

_LAYERS: dict[str, "weakref.ref"] = {}
_COMM_STREAMS: dict[int, torch.cuda.Stream] = {}


def enabled() -> bool:
    return os.environ.get("VLLM_K3_AR_OVERLAP", "0") == "1"


def _chunks() -> int:
    return max(2, int(os.environ.get("VLLM_K3_AR_OVERLAP_CHUNKS", "2")))


def _min_tokens() -> int:
    return int(os.environ.get("VLLM_K3_AR_OVERLAP_MIN_TOKENS", "1024"))


def register_layer(name: str, layer: torch.nn.Module) -> None:
    _LAYERS[name] = weakref.ref(layer)


def _comm_stream(device: torch.device) -> torch.cuda.Stream:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    s = _COMM_STREAMS.get(idx)
    if s is None:
        # highest priority so the (few-SM) all-reduce kernel is scheduled ahead
        # of queued GEMM thread blocks.
        s = torch.cuda.Stream(device=idx, priority=-1)
        _COMM_STREAMS[idx] = s
    return s


def all_reduce_into(part: torch.Tensor, out: torch.Tensor, stream) -> None:
    """All-reduce ``part`` into ``out`` (same shape) on the current stream.

    Uses the same backend choice as CudaCommunicator.all_reduce for this size
    (custom IPC all-reduce if it accepts the size, else NCCL), but writes into
    the caller's buffer so no concatenation copy is needed.
    """
    from vllm.distributed.parallel_state import get_tp_group

    dc = get_tp_group().device_communicator
    ca = getattr(dc, "ca_comm", None)
    if ca is not None and not ca.disabled and ca.should_custom_ar(part):
        ca.all_reduce(part, out=out, registered=False)
        return
    nccl = dc.pynccl_comm
    nccl.all_reduce(part, out, stream=stream)


def plain(layer, x: torch.Tensor, bias) -> torch.Tensor:
    from vllm.distributed.communication_op import tensor_model_parallel_all_reduce

    return tensor_model_parallel_all_reduce(layer.quant_method.apply(layer, x, bias))


def k3_row_linear_ar_impl(x: torch.Tensor, layer_name: str) -> torch.Tensor:
    layer = _LAYERS[layer_name]()
    bias = None if (layer.tp_rank > 0 or layer.skip_bias_add) else layer.bias
    n = x.shape[0]
    k = _chunks()
    if (
        n < _min_tokens()
        or n < 8 * k
        or torch.cuda.is_current_stream_capturing()
        or x.dim() != 2
    ):
        return plain(layer, x, bias)

    main = torch.cuda.current_stream()
    comm = _comm_stream(x.device)
    out_dim = layer.output_size
    out = torch.empty((n, out_dim), dtype=x.dtype, device=x.device)
    # 8-row aligned chunk boundaries (16-byte multiple for the custom AR)
    step = ((n + k - 1) // k + 7) // 8 * 8
    a = 0
    while a < n:
        b = min(n, a + step)
        part = layer.quant_method.apply(layer, x[a:b], bias)
        ev = torch.cuda.Event()
        ev.record(main)
        comm.wait_event(ev)
        with torch.cuda.stream(comm):
            part.record_stream(comm)
            all_reduce_into(part, out[a:b], comm)
        a = b
    main.wait_stream(comm)
    return out


def k3_row_linear_ar_fake(x: torch.Tensor, layer_name: str) -> torch.Tensor:
    layer = _LAYERS[layer_name]()
    return x.new_empty((x.shape[0], layer.output_size))


direct_register_custom_op(
    op_name="k3_row_linear_ar",
    op_func=k3_row_linear_ar_impl,
    fake_impl=k3_row_linear_ar_fake,
)
