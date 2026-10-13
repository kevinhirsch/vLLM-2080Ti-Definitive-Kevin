# SPDX-License-Identifier: Apache-2.0
"""E16 P5/P9: loader of the tree-commit CUDA extension (tree_commit_ext/tree_commit_sm75.cu).

Standalone (imports only os/torch): release.py --jit-ext prebuilds it with
``tree_commit_loader.py:VLLM_TREE_COMMIT_BUILD_DIR``.  Same pattern as K5's
gdn_sm75_cuda loader: a prebuilt .so whose source stamp matches is imported
directly (no ninja / cpp_extension rebuild check at boot); otherwise JIT-built
once and stamped.  Never loaded for the first time inside CUDA-graph capture.
"""

from __future__ import annotations

import os

import torch

_EXT = None
_NAME = "tree_commit_sm75"
_CFLAGS = ["-O3", "-gencode=arch=compute_75,code=sm_75"]
_SRC = os.path.join(os.path.dirname(__file__), "tree_commit_ext", "tree_commit_sm75.cu")


def _capturing() -> bool:
    return torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing()


def _source_stamp() -> str:
    import hashlib

    h = hashlib.sha256()
    with open(_SRC, "rb") as f:
        h.update(f.read())
    h.update(repr((_CFLAGS, torch.__version__, torch.version.cuda)).encode())
    return h.hexdigest()


def _import_prebuilt(build_dir: str):
    """The release-prebuilt ``<dir>/tree_commit_sm75.so`` when its stamp
    matches this source (same pattern as K5's loader): imported directly, no
    ninja / torch rebuild check at boot.  None otherwise."""
    so = os.path.join(build_dir, _NAME + ".so")
    try:
        with open(os.path.join(build_dir, _NAME + ".stamp")) as f:
            if f.read().strip() != _source_stamp() or not os.path.exists(so):
                return None
    except OSError:
        return None
    import importlib.machinery
    import importlib.util

    loader = importlib.machinery.ExtensionFileLoader(_NAME, so)
    spec = importlib.util.spec_from_file_location(_NAME, so, loader=loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _load():
    """Load the commit extension: stamped prebuilt import, else JIT build +
    stamp (release build time).  Never first inside CUDA-graph capture:
    TreeCommit.warmup loads and launches it at runner init."""
    global _EXT
    if _EXT is None:
        if _capturing():
            raise RuntimeError(
                "tree commit extension must be loaded before CUDA graph capture "
                "(TreeCommit.warmup at runner init)"
            )
        bd = os.environ.get(
            "VLLM_TREE_COMMIT_BUILD_DIR", os.path.expanduser("~/.cache/vllm-tree-commit")
        )
        _EXT = _import_prebuilt(bd)
        if _EXT is None:
            from torch.utils.cpp_extension import load

            os.makedirs(bd, exist_ok=True)
            _EXT = load(
                name=_NAME, sources=[_SRC], extra_cuda_cflags=_CFLAGS,
                build_directory=bd, verbose=False,
            )
            try:
                with open(os.path.join(bd, _NAME + ".stamp"), "w") as f:
                    f.write(_source_stamp() + "\n")
            except OSError:
                pass
    return _EXT
