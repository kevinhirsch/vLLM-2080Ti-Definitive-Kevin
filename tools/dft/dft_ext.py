"""vLLM worker extension (loaded with worker_extension_cls='dft_ext.DumpExt'): captures the TARGET model's post-final-norm hidden
states (exactly what the MTP drafter is fed) for every token the engine prefills, and dumps chosen windows to disk from TP rank 0."""
import os
import numpy as np
import torch

class DumpExt:
    def dft_arm(self):
        m = self.model_runner.get_model()
        if not hasattr(self, "_dft_hook"):
            self._dft_on = False
            def hook(mod, inp, out):
                if self._dft_on:
                    h = out[0] if isinstance(out, tuple) else out
                    assert h.shape[-1] == 5120, h.shape
                    self._dft_buf.append(h.detach().to(torch.float16, copy=True))
            self._dft_hook = m.register_forward_hook(hook)
        self._dft_buf = []
        self._dft_on = True
        return type(m).__name__

    def dft_dump(self, path, spans, expect_n):
        self._dft_on = False
        h = torch.cat(self._dft_buf, 0) if self._dft_buf else torch.empty(0, 5120)
        self._dft_buf = []
        n = int(h.shape[0])
        if n != expect_n:
            return {"ok": False, "n": n, "expect": expect_n}
        if self.rank == 0:
            out = torch.cat([h[s:e] for s, e in spans], 0).cpu().numpy()
            tmp = path + ".tmp.npy"
            np.save(tmp, out)
            os.replace(tmp, path)
        del h
        return {"ok": True, "n": n}
