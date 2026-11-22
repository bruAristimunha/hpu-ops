"""Habana bug (Gaudi2, bridge 1.21.4, PT_HPU_LAZY_MODE=0): compiling a Conv2d with symbolic batch at batch 128
and then running batch 16 returns wrong values (diff ~ 4.06 on Conv2d, ~ 0.42 on MiniPreLocal / SignalJEPA_PreLocal).

    PT_HPU_LAZY_MODE=0 python3 repro/issue42_conv2d_dynamic_shrink.py

Root cause in ``habana_frameworks/torch/dynamo/compile_backend/recipe_compiler.py``:
- Lines 94-99: when ``var_range.upper > MAX_UPPER_SIZE`` (default unbounded symbolic batch), ``get_input_symbolic``
  clamps ``[min_shape, max_shape]`` to ``[val, 2 * val]``, compiling a Synapse recipe valid only for ``[128, 256]``.
- Lines 308-352: ``HabanaGraphModule.__call__`` launches ``self._recipe_id`` without checking input bounds, and
  Synapse's in-process recipe cache matches the identical symbolic JIT IR across ``torch._dynamo.reset()``, so a
  ``(16, 128)`` run followed by ``torch._dynamo.reset()`` and ``(128, 16)`` silently reuses the ``[128, 256]``
  recipe at batch 16.

Verified workarounds: compile with ``dynamic=False`` or run ``(128, 16)`` in a fresh process.
"""
import habana_frameworks.torch.core  # noqa: F401
import habana_frameworks.torch.hpu as hthpu
import torch
import torch.nn as nn


class MiniPreLocal(nn.Module):
    def __init__(self, c=19):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, 10, stride=5, bias=False),
            nn.GELU(),
            nn.Conv1d(8, 16, 3, stride=2, bias=False),
            nn.GELU(),
            nn.Conv1d(16, 32, 3, stride=2, bias=False),
            nn.GELU(),
        )
        self.spatial = nn.Conv2d(c, 4, (1, 1))
        self.head = nn.Linear(32 * 4 * 9, 4)

    def forward(self, x):
        b, c, t = x.shape
        y = self.conv(x.reshape(b * c, 1, t))
        y = y.reshape(b, c, 32, -1)
        y = self.spatial(y).flatten(1)
        return self.head(y)


def check_conv2d_dynamic_shrink():
    torch.manual_seed(0)
    m = nn.Conv2d(19, 4, (1, 1)).eval()
    xc = {b: torch.randn(b, 19, 32, 9) for b in (16, 128)}
    with torch.no_grad():
        ref = {b: m(xc[b]) for b in xc}
    c = torch.compile(m.to("hpu"), backend="hpu_backend", dynamic=True)
    for b in (128, 16):
        with torch.no_grad():
            out = c(xc[b].to("hpu"))
        hthpu.synchronize()
        diff = float((out.cpu() - ref[b]).abs().max())
        print(f"Conv2d dynamic=True batch {b:3d}: max_abs_diff={diff:.3e}", flush=True)


def check_miniprelocal_reset():
    torch.manual_seed(0)
    m = MiniPreLocal(19).eval()
    xc = {b: torch.randn(b, 19, 200) for b in (16, 128)}
    with torch.no_grad():
        ref = {b: m(xc[b]) for b in xc}
    mh = m.to("hpu")
    xh = {b: v.to("hpu") for b, v in xc.items()}
    for tag, kwargs in (("default", {}), ("dynamic=False", {"dynamic": False})):
        for order in ((16, 128), (128, 16)):
            torch._dynamo.reset()
            c = torch.compile(mh, backend="hpu_backend", **kwargs)
            for b in order:
                with torch.no_grad():
                    out = c(xh[b])
                hthpu.synchronize()
                diff = float((out.cpu() - ref[b]).abs().max())
                print(f"MiniPreLocal {tag:13s} order={order} batch {b:3d}: max_abs_diff={diff:.3e}", flush=True)


if __name__ == "__main__":
    check_conv2d_dynamic_shrink()
    check_miniprelocal_reset()
