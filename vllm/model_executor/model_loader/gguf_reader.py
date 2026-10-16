# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A small, pure-Python GGUF (llama.cpp) header reader.

The ``gguf`` pip package cannot read the Qwen4Exp checkpoints because
they use ggml type 42 (``Q2_0``), which the package does not know about.  This
module therefore parses the GGUF header (key/value pairs plus the tensor
table) by hand, using only the standard library and numpy.  It never reads
the tensor *data* while parsing, so it works on header-only files.
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# ggml tensor types
# ---------------------------------------------------------------------------
# ggml type id -> (name, block_elems, block_bytes)
#
# ``block_elems`` is the number of elements stored per block and
# ``block_bytes`` is the number of bytes used per block.  For
# element-wise types (F32, F16, BF16, ...) the block is a single element.
# ``nbytes = (numel // block_elems) * block_bytes``.
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    8: ("Q8_0", 32, 34),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    42: ("Q2_0", 64, 18),
}

# gguf scalar value type -> (struct format, size in bytes)
_SCALAR: dict[int, tuple[str, int]] = {
    0: ("B", 1),
    1: ("b", 1),
    2: ("H", 2),
    3: ("h", 2),
    4: ("I", 4),
    5: ("i", 4),
    6: ("f", 4),
    7: ("?", 1),
    10: ("Q", 8),
    11: ("q", 8),
    12: ("d", 8),
}


def _read_u32(f: Any) -> int:
    return struct.unpack("<I", f.read(4))[0]


def _read_u64(f: Any) -> int:
    return struct.unpack("<Q", f.read(8))[0]


def _read_string(f: Any) -> str:
    length = _read_u64(f)
    return f.read(length).decode("utf-8", "replace")


def _read_value(f: Any, type_id: int) -> Any:
    """Read one GGUF value (scalar, string, or nested array).

    All arrays are kept in full, including integer arrays and
    (potentially large) string arrays.
    """
    if type_id in _SCALAR:
        fmt, size = _SCALAR[type_id]
        return struct.unpack("<" + fmt, f.read(size))[0]
    if type_id == 8:
        return _read_string(f)
    if type_id == 9:
        elem_type = _read_u32(f)
        count = _read_u64(f)
        return [_read_value(f, elem_type) for _ in range(count)]
    raise ValueError(f"unknown GGUF value type {type_id}")


# ---------------------------------------------------------------------------
# tensor metadata
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GGUFTensorInfo:
    """Metadata describing one tensor in a GGUF file.

    ``dims`` is stored in ggml order (``ne0`` first, i.e. the innermost /
    fastest-varying dimension first), exactly as written in the
    file.  ``offset`` is the *absolute* byte offset of the tensor data
    within the file (``data_start`` plus the relative offset).
    """

    name: str
    dims: tuple[int, ...]
    ggml_type: int
    type_name: str
    offset: int
    nbytes: int
    path: str

    @property
    def shape(self) -> tuple[int, ...]:
        """Tensor shape in torch/numpy (row-major) order."""
        return tuple(reversed(self.dims))

    @property
    def is_float(self) -> bool:
        """Whether the tensor is stored as a floating-point type."""
        return self.ggml_type in (0, 1, 30)


# ---------------------------------------------------------------------------
# single shard
# ---------------------------------------------------------------------------
class GGUFFile:
    """A single GGUF shard: parsed header plus tensor table."""

    def __init__(self, path: str):
        self.path = path
        self.version: int = 0
        self.kv: dict[str, Any] = {}
        self.tensors: dict[str, GGUFTensorInfo] = {}
        self.data_start: int = 0
        self._parse()

    def _parse(self) -> None:
        with open(self.path, "rb") as f:
            if f.read(4) != b"GGUF":
                raise ValueError(f"not a GGUF file: {self.path}")
            self.version = _read_u32(f)
            tensor_count = _read_u64(f)
            kv_count = _read_u64(f)

            # key/value pairs
            for _ in range(kv_count):
                key = _read_string(f)
                type_id = _read_u32(f)
                self.kv[key] = _read_value(f, type_id)

            # tensor table
            raw_tensors: list[tuple] = []
            for _ in range(tensor_count):
                name = _read_string(f)
                ndims = _read_u32(f)
                dims = tuple(_read_u64(f) for _ in range(ndims))
                ggml_type = _read_u32(f)
                rel_offset = _read_u64(f)
                type_name, block_elems, block_bytes = GGML_TYPES.get(
                    ggml_type, (f"T{ggml_type}", 1, 0)
                )
                numel = 1
                for d in dims:
                    numel *= d
                nbytes = (numel // block_elems) * block_bytes
                raw_tensors.append(
                    (name, dims, ggml_type, type_name, rel_offset, nbytes)
                )
            header_end = f.tell()

        alignment = self.kv.get("general.alignment", 32)
        self.data_start = ((header_end + alignment - 1) // alignment) * alignment

        for name, dims, ggml_type, type_name, rel_offset, nbytes in raw_tensors:
            self.tensors[name] = GGUFTensorInfo(
                name=name,
                dims=dims,
                ggml_type=ggml_type,
                type_name=type_name,
                offset=self.data_start + rel_offset,
                nbytes=nbytes,
                path=self.path,
            )

    def raw(self, name: str) -> np.ndarray:
        """Return the raw bytes of a tensor as a uint8 memmap."""
        info = self.tensors[name]
        return np.memmap(
            self.path,
            dtype=np.uint8,
            mode="r",
            offset=info.offset,
            shape=(info.nbytes,),
        )

    def array(self, name: str) -> np.ndarray:
        """Return a floating-point tensor as a memmap, reshaped to shape.

        F32/F16 are mapped to their native numpy dtypes; BF16 is
        exposed as raw uint16.  Raises ``ValueError`` for quantized
        (non-float) types.
        """
        info = self.tensors[name]
        if not info.is_float:
            raise ValueError(f"tensor {name!r} is not a float type")
        if info.ggml_type == 0:
            dtype = np.float32
        elif info.ggml_type == 1:
            dtype = np.float16
        else:  # BF16 (30)
            dtype = np.uint16
        return np.memmap(
            self.path,
            dtype=dtype,
            mode="r",
            offset=info.offset,
            shape=info.shape,
        )


# ---------------------------------------------------------------------------
# split (multi-shard) handling
# ---------------------------------------------------------------------------
_SPLIT_RE = re.compile(r"-(\d+)-of-(\d+)\.gguf$")


def split_paths(path: str) -> list[str]:
    """Return all existing shard paths for a (possibly split) GGUF.

    Given one shard path such as ``X-00001-of-00002.gguf``, this returns the
    paths of every existing shard in order, preserving the zero
    padding width of the shard index.  A path that does not match the
    split pattern is returned unchanged as a single-element list.
    """
    match = _SPLIT_RE.search(path)
    if match is None:
        return [path]
    index_start, index_end = match.span(1)
    count = int(match.group(2))
    width = len(match.group(1))
    paths: list[str] = []
    for i in range(1, count + 1):
        candidate = path[:index_start] + str(i).zfill(width) + path[index_end:]
        if os.path.exists(candidate):
            paths.append(candidate)
    return paths


# ---------------------------------------------------------------------------
# multi-shard model
# ---------------------------------------------------------------------------
class GGUFModel:
    """A (possibly split) GGUF model.

    Opens every shard referenced by ``path`` (via :func:`split_paths`),
    merges their tensor tables (erroring on duplicate names), and
    exposes the key/value pairs of the first shard as ``kv``.
    """

    def __init__(self, path: str):
        self.path = path
        self.files: list[GGUFFile] = []
        self.kv: dict[str, Any] = {}
        self.tensors: dict[str, GGUFTensorInfo] = {}
        self._open_shards()

    def _open_shards(self) -> None:
        for shard_path in split_paths(self.path):
            file = GGUFFile(shard_path)
            self.files.append(file)
            for name, info in file.tensors.items():
                if name in self.tensors:
                    raise ValueError(f"duplicate tensor name {name!r}")
                self.tensors[name] = info
        if self.files:
            self.kv = self.files[0].kv

    def raw(self, name: str) -> np.ndarray:
        """Return the raw bytes of a tensor, from its owning shard."""
        for file in self.files:
            if name in file.tensors:
                return file.raw(name)
        raise KeyError(f"tensor {name!r} not found")

    def array(self, name: str) -> np.ndarray:
        """Return a floating-point tensor, from its owning shard."""
        for file in self.files:
            if name in file.tensors:
                return file.array(name)
        raise KeyError(f"tensor {name!r} not found")


__all__ = [
    "GGML_TYPES",
    "GGUFTensorInfo",
    "GGUFFile",
    "GGUFModel",
    "split_paths",
]
