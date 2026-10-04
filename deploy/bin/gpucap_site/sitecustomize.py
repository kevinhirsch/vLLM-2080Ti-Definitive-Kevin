"""gpucap sitecustomize (lane RL, 2026-10-03): enforce GPU_CAP_MIB on any torch process launched through gpucap.py run.

Loaded automatically because gpucap.py puts this directory first on PYTHONPATH. Nothing happens at import: when torch is
imported, torch.cuda._lazy_init is wrapped so that the FIRST real CUDA initialization (never earlier -- an eager init
would break fork-based workers) caps the caching allocator on every visible device with
torch.cuda.set_per_process_memory_fraction(GPU_CAP_MIB / device total). Allocations past the cap raise OOM in the
process instead of squeezing production. Non-torch CUDA processes are covered by gpucap.py's watcher.
Chains to any other sitecustomize found later on sys.path.
"""
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

_CAP = os.environ.get("GPU_CAP_MIB")


def _apply(torch):
    try:
        cap = float(_CAP) * (1 << 20)
        for i in range(torch.cuda.device_count()):
            total = float(torch.cuda.get_device_properties(i).total_memory)
            torch.cuda.set_per_process_memory_fraction(max(0.001, min(1.0, cap / total)), i)
        if os.environ.get("GPU_CAP_VERBOSE"):
            sys.stderr.write(f"gpucap: allocator capped at {_CAP} MiB on {torch.cuda.device_count()} device(s)\n")
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"gpucap: could not apply GPU_CAP_MIB={_CAP}: {e!r}\n")


def _wrap(torch):
    cuda = torch.cuda
    orig = cuda._lazy_init
    state = {"done": False}

    def _lazy_init(*a, **k):
        r = orig(*a, **k)
        if not state["done"]:
            state["done"] = True
            _apply(torch)
        return r
    cuda._lazy_init = _lazy_init


class _TorchHook(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name != "torch":
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec("torch")
        finally:
            pass
        if spec is None or spec.loader is None:
            return None
        self._real = spec.loader
        spec.loader = self
        return spec

    def create_module(self, spec):
        return self._real.create_module(spec)

    def exec_module(self, module):
        self._real.exec_module(module)
        try:
            _wrap(module)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"gpucap: torch hook failed: {e!r}\n")


if _CAP:
    if "torch" in sys.modules:
        _wrap(sys.modules["torch"])
    else:
        sys.meta_path.insert(0, _TorchHook())

# chain to the next sitecustomize on the path (venvs and distros ship their own)
_here = os.path.dirname(os.path.abspath(__file__))
for _p in sys.path:
    if _p and os.path.abspath(_p) != _here and os.path.isfile(os.path.join(_p, "sitecustomize.py")):
        _spec = importlib.util.spec_from_file_location("_chained_sitecustomize", os.path.join(_p, "sitecustomize.py"))
        try:
            _spec.loader.exec_module(importlib.util.module_from_spec(_spec))
        except Exception:  # noqa: BLE001
            pass
        break
