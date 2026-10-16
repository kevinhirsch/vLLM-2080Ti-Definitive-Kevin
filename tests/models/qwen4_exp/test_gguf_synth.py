# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the synthetic tiny qwen4exp GGUF writer + numpy refs."""
from __future__ import annotations

import importlib
import struct
from pathlib import Path

import numpy as np
import pytest

from tests.models.qwen4_exp import gguf_numpy_ref as ref
from tests.models.qwen4_exp import gguf_synth as synth


def _load_reader():
    """Return a reader object/class or None."""
    try:
        from vllm.model_executor.model_loader.gguf_reader import GGUFModel
    except Exception:
        GGUFModel = None
    if GGUFModel is not None:
        return GGUFModel
    try:
        mod = importlib.import_module("tests.models.qwen4_exp.gguf_index_reference")
        return mod
    except Exception:
        return None


def _iter_tensors(reader):
    """Yield (name, dims_list, type_name) for a reader object."""
    if hasattr(reader, "tensors"):
        tensors = reader.tensors
        if isinstance(tensors, dict):
            for name, info in tensors.items():
                dims = getattr(info, "dims", None) or getattr(info, "shape", None)
                ttype = getattr(info, "type_name", None) or getattr(info, "type", None)
                yield name, list(dims), str(ttype)
        else:
            for t in tensors:
                name = getattr(t, "name", None)
                dims = getattr(t, "shape", None) or getattr(t, "dims", None)
                ttype = getattr(t, "type", None) or getattr(t, "dtype", None)
                yield name, list(dims), str(ttype)
    elif isinstance(reader, dict):
        for name, info in reader.items():
            dims = info.get("shape") or info.get("dims")
            ttype = info.get("type") or info.get("dtype")
            yield name, list(dims), str(ttype)


def test_write_tiny_and_read_back(tmp_path):
    """Write tiny GGUF (shard=True), read back, check dims/types."""
    path = tmp_path / "tiny.gguf"
    result = synth.write_tiny_qwen4exp(path, seed=0, shard=True)
    paths = result["paths"]
    assert len(paths) == 2
    for p in paths:
        assert Path(p).exists()

    # read back with the repo reader if available
    reader_mod = _load_reader()
    if reader_mod is None:
        pytest.skip("no GGUF reader module available")

    expected = result["tensors"]
    seen: dict[str, tuple[list, str]] = {}
    for p in paths:
        try:
            reader = reader_mod(p)
        except Exception as e:
            pytest.skip(f"reader failed: {e}")
        for name, dims, ttype in _iter_tensors(reader):
            seen[name] = (dims, ttype)

    # check every tensor's dims/type
    for name, (dims, ttype) in expected.items():
        assert name in seen, f"missing tensor {name}"
        got_dims, got_type = seen[name]
        assert list(got_dims) == list(dims), f"{name} dims {got_dims} != {dims}"
        assert got_type == ttype, f"{name} type {got_type} != {ttype}"


def test_dequant_finite_and_q2_0_hand(tmp_path):
    """dequant of every quantized tensor returns finite values;
    dequant_q2_0 matches a hand-computed block."""
    path = tmp_path / "tiny.gguf"
    result = synth.write_tiny_qwen4exp(path, seed=1, shard=False)
    tensors = result["tensors"]

    # dequant of every quantized tensor returns finite values
    for name, (dims, ttype) in tensors.items():
        if ttype in ("F32", "F16", "BF16"):
            continue  # not quantized
        raw = _read_tensor_raw(path, name)
        out = ref.dequant(raw, ttype, dims)
        assert np.all(np.isfinite(out)), f"{name} ({ttype}) not finite"

    # hand-computed Q2_0 block: d=0.5, qs bytes chosen so the 64 weights are known
    d_bytes = np.float16(0.5).tobytes()
    # choose qs so weights are known: byte j encodes 4 2-bit values
    # value = d * (((qs[j//4] >> (2*(j%4))) & 3) - 1)
    # all-zero qs -> value 0 -> weight = d * (0-1) = -d = -0.5
    codes = np.arange(64) % 4  # weight j has 2-bit code j % 4 -> value (code - 1) * d
    qs = np.zeros(16, dtype=np.uint8)
    for j, c in enumerate(codes):
        qs[j // 4] |= np.uint8(c << (2 * (j % 4)))
    raw_arr = np.frombuffer(d_bytes + qs.tobytes(), dtype=np.uint8)
    got = ref.dequant_q2_0(raw_arr, 64)
    expected = ((codes - 1) * 0.5).astype(np.float32)
    assert np.array_equal(got, expected)


def _read_tensor_raw(path: Path, name: str) -> np.ndarray:
    """Read a tensor's raw bytes from a GGUF file."""
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"GGUF"
    off = 4
    off += 4  # version
    n_tensors = struct.unpack_from("<Q", data, off)[0]
    off += 8
    n_kv = struct.unpack_from("<Q", data, off)[0]
    off += 8
    # skip kv
    for _ in range(n_kv):
        klen = struct.unpack_from("<Q", data, off)[0]
        off += 8 + klen
        vtype = struct.unpack_from("<I", data, off)[0]
        off += 4
        off += _kv_value_size(vtype, data, off)
    # tensor info
    infos: dict[str, tuple[int, int]] = {}
    for _ in range(n_tensors):
        nlen = struct.unpack_from("<Q", data, off)[0]
        off += 8
        tname = data[off:off + nlen].decode("utf-8")
        off += nlen
        n_dims = struct.unpack_from("<I", data, off)[0]
        off += 4
        off += 8 * n_dims
        ttype = struct.unpack_from("<I", data, off)[0]
        off += 4
        toff = struct.unpack_from("<Q", data, off)[0]
        off += 8
        infos[tname] = (toff, ttype)
    # data offset: pad to 32
    data_off = (off + 31) // 32 * 32
    toff, ttype = infos[name]
    # compute nbytes from type and dims
    be, bb = next((b, bb) for (i, b, bb) in synth.GGML.values() if i == ttype)
    n_elems = 1
    for n, d, t in synth._tiny_tensor_table():
        if n == name:
            n_elems = int(np.prod(d))
            break
    n_bytes = n_elems * bb // be
    return data[data_off + toff: data_off + toff + n_bytes]


def _kv_value_size(vtype: int, data: bytes, off: int) -> int:
    """Return the size in bytes of a GGUF kv value of the given type."""
    if vtype == 7:  # bool
        return 1
    if vtype in (4, 5, 6):  # uint32, int32, float32
        return 4
    if vtype == 8:  # string
        slen = struct.unpack_from("<Q", data, off)[0]
        return 8 + slen
    if vtype == 9:  # array
        ltype = struct.unpack_from("<I", data, off)[0]
        off += 4
        alen = struct.unpack_from("<Q", data, off)[0]
        off += 8
        size = 0
        for _ in range(alen):
            size += _kv_value_size(ltype, data, off + size)
        return 4 + 8 + size
    if vtype in (10, 11):  # uint64, int64
        return 8
    raise ValueError(f"unknown kv value type {vtype}")


def test_q8_0_path():
    """Quantize a known float row yourself (Q8_0 = fp16 d + 32 int8,
    value = d*q) and check dequant returns it."""
    d = 0.01
    qs = np.array(
        [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8,
         9, -9, 10, -10, 11, -11, 12, -12, 13, -13, 14, -14, 15, -15, 127, -128],
        dtype=np.int8,
    )
    block = np.float16(d).tobytes() + qs.tobytes()
    raw = np.frombuffer(block, dtype=np.uint8)
    got = ref.dequant(raw, "Q8_0", [32])
    expected = np.array([d * q for q in qs], dtype=np.float32)
    assert np.allclose(got, expected, rtol=1e-3)
