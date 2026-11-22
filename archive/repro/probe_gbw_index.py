"""Gaudi probe: every ``aten.index.Tensor`` call (below autograd) in one GBWAtt train step, with its shapes and the
bridge's CPU-fallback delta around it.
    PT_HPU_LAZY_MODE=0 python3 repro/probe_gbw_index.py"""
import json
import os
import sys
import traceback

import torch
from habana_frameworks.torch.hpu.metrics import metric_global
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
from model_steps import batch  # noqa: E402


def fb():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


class Index(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is not torch.ops.aten.index.Tensor:
            return func(*args, **(kwargs or {}))
        before = fb()
        out = func(*args, **(kwargs or {}))
        after = fb()
        self_, idx = args
        where = [f"{f.filename.split('site-packages/')[-1]}:{f.lineno}" for f in traceback.extract_stack()[-12:-2]]
        print("INDEX " + json.dumps({
            "self": [list(self_.shape), str(self_.dtype), str(self_.device)],
            "indices": [None if i is None else [list(i.shape), str(i.dtype), str(i.device)] for i in idx],
            "fallback_delta": {k: after[k] - before.get(k, 0) for k in after if after[k] - before.get(k, 0)},
            "where": where}), flush=True)
        return out


def main():
    import hpu_complex
    import hpu_ops
    hpu_ops.enable()
    hpu_complex.enable()
    from spd_diag import protocol as P
    info, X, y = batch(22, 32)
    torch.manual_seed(0)
    model = P.build("GBWAtt", 22, info, None).to("hpu")
    xb = model.fixed_input(X).to("hpu") if hasattr(model, "fixed_input") else X.to("hpu")
    if hasattr(model, "fit_graph"):
        model.fit_graph(X)
    out = model(xb)  # warm-up outside the mode
    hpu_ops._STATE["calls"].clear()
    with Index():
        out = model(xb)
        out = out[0] if isinstance(out, tuple) else out
        torch.nn.functional.cross_entropy(out, y.to("hpu")).backward()
    print("CALLS", dict(hpu_ops._STATE["calls"]), flush=True)
    import habana_frameworks.torch.hpu as hthpu
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for step in range(3):  # the bridge's fallback delta per phase of a plain train step
        phases = {}
        a = fb()
        opt.zero_grad(set_to_none=True)
        out = model(xb)
        out = out[0] if isinstance(out, tuple) else out
        loss = torch.nn.functional.cross_entropy(out, y.to("hpu"))
        hthpu.synchronize()
        b = fb()
        loss.backward()
        hthpu.synchronize()
        c = fb()
        opt.step()
        hthpu.synchronize()
        d = fb()
        for name, (u, v) in {"forward": (a, b), "backward": (b, c), "optimizer": (c, d)}.items():
            phases[name] = {k: v[k] - u.get(k, 0) for k in v if v[k] - u.get(k, 0) and "index" in k}
        print("PHASES", step, json.dumps(phases), flush=True)


if __name__ == "__main__":
    main()
