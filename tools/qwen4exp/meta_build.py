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
    a = ap.parse_args()

    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.utils.torch_utils import set_default_torch_dtype

    kw = dict(model=a.config_dir, dtype=a.dtype, load_format="dummy", enforce_eager=True,
              max_model_len=4096, max_num_batched_tokens=4096, gpu_memory_utilization=0.5,
              mamba_cache_mode="align", language_model_only=False,
              engram_config={"cpu_offload": False})
    if a.quantization:
        kw["quantization"] = a.quantization
    if a.hf_overrides:
        kw["hf_overrides"] = json.loads(a.hf_overrides)
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


if __name__ == "__main__":
    main()
