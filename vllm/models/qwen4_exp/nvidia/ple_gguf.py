# SPDX-License-Identifier: Apache-2.0
"""PLE n-gram table read from a GGUF file on the host (IQ4_NL rows), dequantized on the GPU.

Qwen3.8-Flash-Next's PLE table has 320,001,536 rows of 160 values (16 hashed rows per token, one PLE layer):
102 GB in fp16, 26.8 GiB as the GGUF's IQ4_NL. It fits neither in VRAM nor pinned host memory next to the
experts, so the table stays in the memory-mapped GGUF (page cache / NVMe) and only the selected rows move:

1. ``start_prefetch`` (top of the forward, eager break): the n-gram ids are computed on the GPU, copied to a
   pinned host buffer on a side stream; a worker thread waits for that copy, gathers the 90-byte IQ4_NL rows
   from the mmap into pinned staging (the CPU only copies bytes, no CPU math), copies them to the GPU and
   dequantizes them with the ggml IQ4_NL kernel into the prefetch buffer, all on the side stream.
2. ``forward`` (the PLE layer, one layer later): joins the worker and the side stream.

Every tensor-parallel rank gathers all rows itself (both ranks read the same page-cache pages), so the
ETP all-reduce of the pinned-host path is not needed.
"""

from __future__ import annotations

import concurrent.futures
from typing import ClassVar

import numpy as np
import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.logger import init_logger
from vllm.model_executor.utils import set_weight_attrs

from .ngram_embedding import Qwen4ExpPLEEmbedding, Qwen4ExpPLEEmbeddingMethod

logger = init_logger(__name__)

IQ4_NL = 20
IQ4_NL_BLOCK = (32, 18)  # elements, bytes


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


class Qwen4ExpPLEGGUFTableMethod(Qwen4ExpPLEEmbeddingMethod):
    """Selected by ``GGUFConfig.get_qwen4exp_ple_method``: table bytes stay in the GGUF."""

    def __init__(self, path: str, offset: int, rows: int, row_dim: int, ggml_type: int) -> None:
        if ggml_type != IQ4_NL:
            raise NotImplementedError(f"GGUF PLE table type {ggml_type}: only IQ4_NL ({IQ4_NL}) is supported")
        if row_dim % IQ4_NL_BLOCK[0]:
            raise ValueError(f"PLE row dim {row_dim} is not a multiple of the IQ4_NL block")
        self.path, self.offset, self.rows, self.row_dim = path, offset, rows, row_dim
        self.row_bytes = row_dim // IQ4_NL_BLOCK[0] * IQ4_NL_BLOCK[1]

    @property
    def embedding_cls(self):
        return Qwen4ExpPLEGGUFHostEmbedding

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
            raise ValueError(f"PLE head dim {input_size_per_partition} != GGUF row dim {self.row_dim}")
        if output_size > self.rows:
            raise ValueError(f"PLE vocab {output_size} > GGUF table rows {self.rows}")
        # Placeholder: the table is never materialised as a parameter.
        weight = nn.Parameter(torch.empty(0, input_size_per_partition, dtype=params_dtype), requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)

    def dequantize(self, layer: nn.Module, embeddings: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
        return embeddings.to(output_dtype)


class Qwen4ExpPLEGGUFHostEmbedding(Qwen4ExpPLEEmbedding):
    supports_prefetch: ClassVar[bool] = True

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: Qwen4ExpPLEGGUFTableMethod,
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
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
            raise NotImplementedError("GGUF host PLE does not support ETP across data parallel ranks")
        m = embedding_method
        self.gatherer = HostRowGather(m.path, m.offset, m.rows, m.row_bytes)
        self.num_ngram_heads = num_ngram_heads
        self._output_dim = num_ngram_heads * embedding_dim
        self._params_dtype = params_dtype
        max_rows = max(1, max_total_tokens) * num_ngram_heads
        device = torch.device("cuda", torch.cuda.current_device())
        self._ids_host = torch.empty(max_rows, dtype=torch.int64, pin_memory=True)
        self._rows_host = torch.empty(max_rows, m.row_bytes, dtype=torch.uint8, pin_memory=True)
        self._rows_dev = torch.empty(max_rows, m.row_bytes, dtype=torch.uint8, device=device)
        self._prefetch_out = torch.empty(max(1, max_total_tokens), self._output_dim, dtype=params_dtype,
                                         device=device)
        self._stream = torch.cuda.Stream(device=device)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="ple-gather")
        self._pending: concurrent.futures.Future | None = None
        self._pending_tokens = 0
        logger.info(
            "PLE table from GGUF %s @%d: %d rows x %d B (IQ4_NL), host gather, %d heads/token",
            m.path, m.offset, m.rows, m.row_bytes, num_ngram_heads,
        )

    def allocate_embedding_weight(self, num_embeddings: int, embedding_dim: int, dtype: torch.dtype):
        return torch.empty(0, embedding_dim, dtype=dtype)

    def _gather_and_upload(self, n_rows: int, num_tokens: int, ready: torch.cuda.Event) -> None:
        ready.synchronize()  # ids are in pinned host memory
        ids = self._ids_host[:n_rows].numpy()
        self.gatherer.gather(ids, self._rows_host.numpy())
        from vllm.model_executor.layers.quantization import gguf_ops

        with torch.cuda.stream(self._stream):
            self._rows_dev[:n_rows].copy_(self._rows_host[:n_rows], non_blocking=True)
            emb = gguf_ops.ggml_dequantize(
                self._rows_dev[:n_rows], IQ4_NL, num_tokens, self._output_dim, self._params_dtype
            )
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
            self._ids_host[:n_rows].copy_(ngram_ids.reshape(-1), non_blocking=True)
            ready.record(self._stream)
        ngram_ids.record_stream(self._stream)
        self._pending_tokens = num_tokens
        self._pending = self._executor.submit(self._gather_and_upload, n_rows, num_tokens, ready)

    @eager_break_during_capture
    def _finalize(self, output: torch.Tensor) -> None:
        if self._pending is None:
            raise RuntimeError("GGUF PLE forward without start_prefetch")
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
