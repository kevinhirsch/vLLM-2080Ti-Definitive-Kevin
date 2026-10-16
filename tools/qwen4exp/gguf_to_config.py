#!/usr/bin/env python3
"""Make a vLLM model directory for a Qwen3.8-Flash-Next (qwen4exp) GGUF: config.json (+ tokenizer files).

    gguf_to_config.py FIRST_SHARD.gguf OUT_DIR --tokenizer-dir DIR

config.json is built from the GGUF header alone (``gguf_qwen4exp_config.qwen4exp_text_config``) and carries
``quantization_config = {"quant_method": "gguf", "gguf_file": <abs path>}``, so
``vllm serve OUT_DIR --load-format gguf`` needs nothing else. The tokenizer files are copied from a Qwen3.5/3.8
checkpoint with the same 248320-token vocabulary (the GGUF's own tokenizer is not converted).
"""
import argparse
import json
import os
import shutil

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json")


def main() -> None:
    from vllm.model_executor.model_loader.gguf_qwen4exp_config import qwen4exp_text_config
    from vllm.model_executor.model_loader.gguf_reader import GGUFModel

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gguf")
    ap.add_argument("out_dir")
    ap.add_argument("--tokenizer-dir", required=True)
    a = ap.parse_args()
    path = os.path.abspath(a.gguf)
    gm = GGUFModel(path)
    cfg = qwen4exp_text_config(gm)
    cfg.update({
        "architectures": ["Qwen4ExpForCausalLM"],
        "model_type": "qwen4_exp_text",
        "torch_dtype": "float16",
        "quantization_config": {"quant_method": "gguf", "gguf_file": path},
    })
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    for name in TOKENIZER_FILES:
        src = os.path.join(a.tokenizer_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(a.out_dir, name))
    print(os.path.join(a.out_dir, "config.json"))


if __name__ == "__main__":
    main()
