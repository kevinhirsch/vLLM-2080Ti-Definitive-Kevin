# SPDX-License-Identifier: Apache-2.0
"""PLE n-gram table kept in its checkpoint file on the host; only the selected rows reach the GPU.

Qwen3.8-Flash-Next's PLE table has 320,001,536 rows of 160 values (16 hashed rows per token, one PLE layer):
102 GB in fp16. Quantized packs shrink it (GGUF IQ4_NL: 90 B/row, 26.8 GiB; EXL3 ``exl3_ngram_trellis`` K=4:
82 B/row, 24.4 GiB), but it still fits neither VRAM next to the weights nor pinned host memory. So the table
stays in the memory-mapped checkpoint (page cache / NVMe) and only the selected rows move:

1. ``start_prefetch`` (top of the forward, eager break): n-gram ids, computed on the GPU, are copied to a pinned
   host buffer on a side stream; a worker thread waits for that copy, gathers the packed rows from the mmap into
   pinned staging (the CPU only copies bytes), uploads them and decodes them on the GPU, all on the side stream.
2. ``forward`` (the PLE layer, one layer later) joins the worker and the side stream.

Decoders: GGUF IQ4_NL rows -> the ggml IQ4_NL dequant kernel; EXL3 trellis rows -> ``exllamav3_ext.ngram_dequant``
(mul1 codebook * row scale + per-head bias, the head found from the row id and the head offsets).
Every tensor-parallel rank gathers all rows itself (both read the same page-cache pages); no ETP all-reduce.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import struct
from typing import ClassVar

import numpy as np
import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture, eager_break_during_capture
from vllm.logger import init_logger
from vllm.model_executor.utils import set_weight_attrs

from .ngram_embedding import Qwen4ExpPLEEmbedding, Qwen4ExpPLEEmbeddingMethod

logger = init_logger(__name__)

IQ4_NL = 20
IQ4_NL_BLOCK = (32, 18)  # elements, bytes

# A finished future: start_prefetch's inline gather (during breakable-graph capture) leaves nothing to wait for.
_DONE: concurrent.futures.Future = concurrent.futures.Future()
_DONE.set_result(None)


class HostRowGather:
    """Byte-row gather from a memory-mapped table (CPU only copies bytes)."""

    def __init__(self, path: str, offset: int, rows: int, row_bytes: int) -> None:
        self.path, self.offset, self.rows, self.row_bytes = path, offset, rows, row_bytes
        self.table = np.memmap(path, dtype=np.uint8, mode="r", offset=offset, shape=(rows, row_bytes))

    def gather(self, ids: np.ndarray, out: np.ndarray) -> np.ndarray:
        """``out[i] = table[ids[i]]``; ids outside the table raise (they would mean a hash/layout bug)."""
        n = ids.shape[0]
        if n and (ids.min() < 0 or ids.max() >= self.rows):
            raise IndexError(f"PLE row id out of range [0, {self.rows}): {ids.min()}..{ids.max()}")
        np.take(self.table, ids, axis=0, out=out[:n])
        return out[:n]


class Qwen4ExpPLEHostTableMethod(Qwen4ExpPLEEmbeddingMethod):
    """Base: the table bytes stay in the checkpoint file; ``embedding_cls`` gathers + decodes rows."""

    path: str
    offset: int
    rows: int
    row_dim: int
    row_bytes: int

    @property
    def embedding_cls(self):
        return Qwen4ExpPLEHostEmbedding

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if input_size_per_partition != self.row_dim:
            raise ValueError(f"PLE head dim {input_size_per_partition} != table row dim {self.row_dim}")
        if output_size > self.rows:
            raise ValueError(f"PLE vocab {output_size} > table rows {self.rows}")
        # Placeholder: the table is never materialised as a parameter.
        weight = nn.Parameter(torch.empty(0, input_size_per_partition, dtype=params_dtype), requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)

    def dequantize(self, layer: nn.Module, embeddings: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
        return embeddings.to(output_dtype)

    def decode(self, layer: nn.Module, rows_dev: torch.Tensor, ids_dev: torch.Tensor, num_tokens: int,
               out_dim: int, dtype: torch.dtype) -> torch.Tensor:
        raise NotImplementedError


class Qwen4ExpPLEGGUFTableMethod(Qwen4ExpPLEHostTableMethod):
    """GGUF ``per_layer_token_embd`` (IQ4_NL), selected by ``GGUFConfig.get_qwen4exp_ple_method``."""

    def __init__(self, path: str, offset: int, rows: int, row_dim: int, ggml_type: int) -> None:
        if ggml_type != IQ4_NL:
            raise NotImplementedError(f"GGUF PLE table type {ggml_type}: only IQ4_NL ({IQ4_NL}) is supported")
        if row_dim % IQ4_NL_BLOCK[0]:
            raise ValueError(f"PLE row dim {row_dim} is not a multiple of the IQ4_NL block")
        self.path, self.offset, self.rows, self.row_dim = path, offset, rows, row_dim
        self.row_bytes = row_dim // IQ4_NL_BLOCK[0] * IQ4_NL_BLOCK[1]

    def decode(self, layer, rows_dev, ids_dev, num_tokens, out_dim, dtype):
        from vllm.model_executor.layers.quantization import gguf_ops

        return gguf_ops.ggml_dequantize(rows_dev, IQ4_NL, num_tokens, out_dim, dtype)


def read_safetensors_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


class Qwen4ExpPLEEXL3TableMethod(Qwen4ExpPLEHostTableMethod):
    """turboderp's ``exl3_ngram_trellis`` table (``ngram_embedding.safetensors``): int16 rows of 1 scale word +
    160*K/16 trellis words, stored as contiguous ``shard_<i>.trellis`` tensors. The small aux tensors
    (head_bias / head_offsets / head_vocab_sizes / layer_multipliers) load as ordinary parameters."""

    TABLE = "ngram_embedding"

    def __init__(self, path: str, prefix_hint: str = "") -> None:
        header, data_start = read_safetensors_header(path)
        meta = header.get("__metadata__", {})
        if meta.get("format") != "exl3_ngram_trellis":
            raise ValueError(f"{path}: not an exl3_ngram_trellis table ({meta.get('format')!r})")
        self.bits = int(meta["K"])
        self.row_dim = int(meta["row_dim"])
        self.rows = int(meta["rows"])
        shards = sorted(
            ((int(k.rsplit(".shard_", 1)[1].split(".")[0]), v) for k, v in header.items()
             if ".shard_" in k and k.endswith(".trellis")),
            key=lambda kv: kv[0],
        )
        if not shards or shards[0][1]["dtype"] != "I16":
            raise ValueError(f"{path}: no int16 trellis shards")
        words = shards[0][1]["shape"][1]
        expect = 0
        for i, (idx, v) in enumerate(shards):
            if idx != i or v["shape"][1] != words or (i and v["data_offsets"][0] != expect):
                raise ValueError(f"{path}: trellis shards are not one contiguous table (shard {idx})")
            expect = v["data_offsets"][1]
        if sum(v["shape"][0] for _, v in shards) != self.rows:
            raise ValueError(f"{path}: shard rows do not add up to {self.rows}")
        self.path = path
        self.offset = data_start + shards[0][1]["data_offsets"][0]
        self.row_bytes = words * 2
        self.words = words
        self.num_heads = int(next(v for k, v in header.items() if k.endswith(".head_bias"))["shape"][0])

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs) -> None:
        super().create_weights(layer, input_size_per_partition, output_partition_sizes, input_size, output_size,
                               params_dtype, **extra_weight_attrs)
        aux = {
            "head_bias": torch.zeros(self.num_heads, self.row_dim, dtype=torch.float16),
            "head_offsets": torch.zeros(self.num_heads, dtype=torch.int64),
            "head_vocab_sizes": torch.zeros(self.num_heads, dtype=torch.int64),
            "layer_multipliers": torch.zeros(3, dtype=torch.int64),
        }
        for name, t in aux.items():
            p = nn.Parameter(t, requires_grad=False)
            layer.register_parameter(name, p)

    def decode(self, layer, rows_dev, ids_dev, num_tokens, out_dim, dtype):
        from exllamav3.ext import exllamav3_ext

        heads = (torch.searchsorted(layer.head_offsets, ids_dev, right=True) - 1).clamp_(0, self.num_heads - 1)
        out = torch.empty(ids_dev.shape[0], self.row_dim, dtype=torch.float16, device=rows_dev.device)
        exllamav3_ext.ngram_dequant(rows_dev.view(torch.int16), self.bits, heads.to(torch.int32),
                                    layer.head_bias, out)
        return out.view(num_tokens, out_dim).to(dtype)


def exl3_ple_method_for(model_dir: str) -> Qwen4ExpPLEEXL3TableMethod | None:
    path = os.path.join(model_dir, "ngram_embedding.safetensors")
    return Qwen4ExpPLEEXL3TableMethod(path) if os.path.exists(path) else None


class Qwen4ExpPLEHostEmbedding(Qwen4ExpPLEEmbedding):
    supports_prefetch: ClassVar[bool] = True

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: Qwen4ExpPLEHostTableMethod,
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
        quant_config=None,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            embedding_method=embedding_method,
            num_ngram_heads=num_ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
        )
        if self.etp_data_parallel_size != 1:
            raise NotImplementedError("host-table PLE does not support ETP across data parallel ranks")
        m = embedding_method
        self.method = m
        self.gatherer = HostRowGather(m.path, m.offset, m.rows, m.row_bytes)
        self.num_ngram_heads = num_ngram_heads
        self._output_dim = num_ngram_heads * embedding_dim
        self._params_dtype = params_dtype
        max_rows = max(1, max_total_tokens) * num_ngram_heads
        device = torch.empty(0).device  # the model-init device context (cuda:<rank>)
        pin = device.type == "cuda"
        self._ids_host = torch.empty(max_rows, dtype=torch.int64, device="cpu", pin_memory=pin)
        self._ids_dev = torch.empty(max_rows, dtype=torch.int64, device=device)
        self._rows_host = torch.empty(max_rows, m.row_bytes, dtype=torch.uint8, device="cpu", pin_memory=pin)
        self._rows_dev = torch.empty(max_rows, m.row_bytes, dtype=torch.uint8, device=device)
        self._prefetch_out = torch.empty(max(1, max_total_tokens), self._output_dim, dtype=params_dtype,
                                         device=device)
        self._stream = torch.cuda.Stream(device=device)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="ple-gather")
        self._pending: concurrent.futures.Future | None = None
        self._pending_tokens = 0
        logger.info(
            "PLE table from %s @%d: %d rows x %d B (%s), host gather, %d heads/token",
            m.path, m.offset, m.rows, m.row_bytes, type(m).__name__, num_ngram_heads,
        )

    def allocate_embedding_weight(self, num_embeddings: int, embedding_dim: int, dtype: torch.dtype):
        return torch.empty(0, embedding_dim, dtype=dtype)

    def _gather_and_upload(self, n_rows: int, num_tokens: int, ready: torch.cuda.Event) -> None:
        ready.synchronize()  # ids are in pinned host memory
        self.gatherer.gather(self._ids_host[:n_rows].numpy(), self._rows_host.numpy())
        with torch.cuda.stream(self._stream):
            self._rows_dev[:n_rows].copy_(self._rows_host[:n_rows], non_blocking=True)
            emb = self.method.decode(self, self._rows_dev[:n_rows], self._ids_dev[:n_rows], num_tokens,
                                     self._output_dim, self._params_dtype)
            self._prefetch_out[:num_tokens].copy_(emb)

    @eager_break_during_capture
    def start_prefetch(self, hidden_states: torch.Tensor, ngram_ids: torch.Tensor) -> None:
        num_tokens = ngram_ids.shape[0]
        n_rows = ngram_ids.numel()
        if self._pending is not None:
            self._pending.result()
        self._stream.wait_stream(torch.cuda.current_stream())
        ready = torch.cuda.Event()
        with torch.cuda.stream(self._stream):
            flat = ngram_ids.reshape(-1)
            self._ids_dev[:n_rows].copy_(flat)
            self._ids_host[:n_rows].copy_(flat, non_blocking=True)
            ready.record(self._stream)
        ngram_ids.record_stream(self._stream)
        self._pending_tokens = num_tokens
        if BreakableCUDAGraphCapture.is_active():
            # This call is an eager break between two captured segments. Capture runs in global mode, so the gather
            # thread's CUDA calls (event sync, side-stream upload and decode) during the next segment's capture
            # invalidated it (W2b: cudaErrorStreamCaptureInvalidated, first seen at layer 2's HC mix). Gather inline
            # while capturing. Replays run this break with no capture active and keep the overlapped thread.
            self._gather_and_upload(n_rows, num_tokens, ready)
            self._pending = _DONE
        else:
            self._pending = self._executor.submit(self._gather_and_upload, n_rows, num_tokens, ready)

    @eager_break_during_capture
    def _finalize(self, output: torch.Tensor) -> None:
        if self._pending is None:
            raise RuntimeError("host-table PLE forward without start_prefetch")
        self._pending.result()
        self._pending = None
        torch.cuda.current_stream().wait_stream(self._stream)
        n = min(self._pending_tokens, output.shape[0])
        output[:n].copy_(self._prefetch_out[:n])
        if n < output.shape[0]:  # padded rows (no ids were hashed for them)
            output[n:].zero_()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self._prefetch_out.new_empty((hidden_states.shape[0], self._output_dim))
        self._finalize(output)
        return output
