#!/usr/bin/env python3
"""Build a Qwen4Exp model on the META device from a config dir (no GPU, no weights) and print its parameters.

Catches config/shape/import errors before a GPU window and dumps the vLLM parameter names that a
weight loader must fill. Usage: meta_build.py CONFIG_DIR [--quantization gguf] [--json OUT]
"""
import argparse
import json
import os
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config_dir")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--hf-overrides", default=None, help="JSON dict")
    ap.add_argument("--json", default=None)
    ap.add_argument("--load-check", action="store_true",
                    help="feed every checkpoint tensor name (meta tensors with the header shapes/dtypes) through "
                         "model.load_weights and report names that fail or parameters left unloaded")
    a = ap.parse_args()

    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.utils.torch_utils import set_default_torch_dtype

    kw = dict(model=a.config_dir, dtype=a.dtype, load_format="dummy", enforce_eager=True,
              max_model_len=4096, max_num_batched_tokens=4096, gpu_memory_utilization=0.5,
              mamba_cache_mode="align", language_model_only=bool(os.environ.get("NM9_LMO")),
              engram_config={"cpu_offload": False})
    if a.quantization:
        kw["quantization"] = a.quantization
    if a.hf_overrides:
        kw["hf_overrides"] = json.loads(a.hf_overrides)
    if os.environ.get("NM9_FAKE_SM75"):  # plugins that validate the device at registration (vllm-exl3-turing)
        import vllm_exl3.turing as _t

        _t.validate_runtime = lambda: _t.load_extension()
    vllm_config = EngineArgs(**kw).create_engine_config()
    # No GPU here: layers that ask the platform for "the current device" get the meta device.
    from vllm.platforms import current_platform
    type(current_platform).current_device = classmethod(lambda cls: torch.device("meta"))
    torch.cuda.current_device = lambda: torch.device("meta")

    class _NoStream:  # aux streams / events are created eagerly by some layers
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, name):
            return lambda *a, **k: None

    torch.cuda.Stream = _NoStream

    def _no_pin(fn):
        def wrapped(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return fn(*args, **kwargs)
        return wrapped

    for name in ("empty", "zeros", "ones", "full", "tensor", "arange"):
        setattr(torch, name, _no_pin(getattr(torch, name)))
    torch.Tensor.pin_memory = lambda self, *a, **k: self
    torch.cuda.Event = _NoStream
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                     distributed_init_method=f"file://{tempfile.mktemp()}", backend="gloo")
        initialize_model_parallel(1, 1)
        with set_default_torch_dtype(vllm_config.model_config.dtype), torch.device("meta"):
            model = initialize_model(vllm_config=vllm_config)
    import collections

    qm = collections.Counter()
    for mname, mod in model.named_modules():
        m = getattr(mod, "quant_method", None)
        if m is not None:
            qm[(type(mod).__name__, type(m).__name__)] += 1
            if os.environ.get("NM9_SHOW_QM") and ".layers.0." in mname:
                print("QM", mname, type(mod).__name__, type(m).__name__)
    print("quant methods:", dict(qm))
    if a.load_check:
        with set_current_vllm_config(vllm_config):
            load_check(model, a.config_dir)
    rows = [(n, list(p.shape), str(p.dtype)) for n, p in model.named_parameters()]
    bufs = [(n, list(b.shape), str(b.dtype)) for n, b in model.named_buffers()]
    total = sum(torch.Size(s).numel() for _, s, _ in rows)
    print(f"{type(model).__name__}: {len(rows)} params, {total / 1e6:.2f} M elements, {len(bufs)} buffers")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"params": rows, "buffers": bufs}, f, indent=1)
    else:
        for r in rows:
            print(*r)


_ST_DTYPES = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32, "I16": torch.int16,
              "I32": torch.int32, "I64": torch.int64, "U8": torch.uint8, "I8": torch.int8, "F8_E4M3": torch.float8_e4m3fn}


def _checkpoint_meta_tensors(model_dir):
    """(name, meta tensor) for every tensor in every *.safetensors of the dir (headers only, no data read),
    honouring the same skip rules as the default loader (VLLM_EXL3_NGRAM_STREAM trellis shards)."""
    import glob
    import struct

    from vllm.model_executor.model_loader.weight_utils import _should_skip_streamed_exl3_ngram_trellis

    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for name, info in header.items():
            if name == "__metadata__" or _should_skip_streamed_exl3_ngram_trellis(name):
                continue
            # a real CPU tensor with zero strides: loaders can copy out of it, no memory is used
            yield name, torch.zeros((), dtype=_ST_DTYPES[info["dtype"]]).expand(info["shape"])


def load_check(model, model_dir):
    """Run the model's own load_weights over every checkpoint name; report failures and unloaded params."""
    failures = []
    loaded = set()

    def gen():
        for name, t in _checkpoint_meta_tensors(model_dir):
            yield name, t

    try:
        loaded = set(model.load_weights(gen()))
    except Exception as e:  # noqa: BLE001 - the first failing name is the finding
        import traceback

        failures.append(repr(e)[:400])
        traceback.print_exc(limit=8)
    params = {n for n, _ in model.named_parameters()}
    unloaded = sorted(p for p in params if p not in loaded and ".ple.ple_embedding.ngram_embedding.weight" not in p)
    print("loaded expert names (sample):", sorted(n for n in loaded if ".experts." in n)[:6])
    print(f"LOAD-CHECK: {len(loaded)} loaded, {len(unloaded)} parameters unloaded, {len(failures)} failures")
    import re as _re
    import collections as _c
    print("unloaded by pattern:", dict(_c.Counter(_re.sub(r"\.\d+\.", ".N.", u) for u in unloaded).most_common(40)))
    for u in unloaded[:10]:
        print("  UNLOADED", u)
    for f in failures:
        print("  FAILURE", f)
    if failures or unloaded:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
