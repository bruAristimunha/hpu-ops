"""Habana bug (Gaudi2, bridge 1.21.4, eager and lazy): ``amax`` and ``max`` drop NaN, found for hpu-ops #21.

    PT_HPU_LAZY_MODE=0 python amax_nan.py     # and PT_HPU_LAZY_MODE=1

torch returns NaN for every row with a NaN (``amax``/``max`` propagate NaN). Measured on Gaudi2:
    [[nan, 1, 2], [2, 1, nan], [1, nan, inf], [1, 2, 3]] -> amax(-1) = [-inf, 2, inf, 3]   (torch: [nan, nan, nan, 3])
A leading NaN even gives -inf, below every value of the row. ``sum``, ``log_softmax`` and Habana's own ``logsumexp``
propagate NaN. Workaround: let NaN reach the result through an op that propagates it (hpu_ops' logsumexp shifts by
the max and sums), or mask with ``isnan().any()``.
"""
import habana_frameworks.torch.core as htcore
import torch

x = torch.tensor([[float("nan"), 1.0, 2.0], [2.0, 1.0, float("nan")], [1.0, float("nan"), float("inf")], [1.0, 2.0, 3.0]])
for name, fn in (("amax", lambda t: t.amax(-1)), ("max", lambda t: t.max(-1).values)):
    out = fn(x.to("hpu"))
    htcore.mark_step()
    got, want = out.cpu(), fn(x)
    ok = torch.equal(got.nan_to_num(), want.nan_to_num()) and torch.equal(got.isnan(), want.isnan())
    print(f"{name}: hpu {got.tolist()} cpu {want.tolist()} -> {'OK' if ok else 'BUG'}", flush=True)
