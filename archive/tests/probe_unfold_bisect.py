"""On a Gaudi card: where TensorCSPNet's training step makes ``aten::unfold`` fall back (bisect of the step).
    PT_HPU_LAZY_MODE=0|1 python3 tests/probe_unfold_bisect.py [--no-conv-weight-grad]
Runs warm-up steps, then records the bridge's CPU-fallback ops of the forward, the backward and the optimizer step
separately, and of each module's forward (hooks). ``--no-conv-weight-grad`` turns off hpu_ops' one-channel strided
weight-gradient rewrite (``conv._one_channel_strided`` answers False), to tell its ``Tensor.unfold`` apart."""
import os
import sys

import habana_frameworks.torch.core as htcore
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"


def fb():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def delta(a, b):
    return {k: b[k] - a.get(k, 0) for k in b if b[k] - a.get(k, 0)}


def mark():
    if LAZY:
        htcore.mark_step()


def main():
    import hpu_ops
    from hpu_ops import conv
    from spd_diag import protocol as P
    from spd_diag.cache import open_cache

    if "--no-conv-weight-grad" in sys.argv:
        conv._one_channel_strided = lambda *a: False
    P._enable_device("hpu")
    print("hpu_ops", hpu_ops.__file__, "lazy", LAZY, "args", sys.argv[1:], flush=True)
    cache = open_cache("BNCI2014_001")
    sd = cache.subject(cache.subjects[0], "X")
    X = torch.as_tensor(sd.X[:32], dtype=torch.float32)
    X = X / X.std()
    y = torch.as_tensor(sd.y[:32]).long().to("hpu")
    torch.manual_seed(0)
    model = P.build("TensorCSPNet", X.shape[1], cache.info, None).to("hpu")
    xb = X.to("hpu")
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    per_module = {}

    def hook(name):
        def pre(mod, args):
            mark()
            per_module[name] = fb()

        def post(mod, args, out):
            mark()
            d = delta(per_module[name], fb())
            if d:
                print("  forward of", name, type(mod).__name__, d, flush=True)

        return pre, post

    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(xb), y).backward()
        mark()
        opt.step()
        mark()
    for name, mod in model.named_modules():
        if name and not list(mod.children()):
            pre, post = hook(name)
            mod.register_forward_pre_hook(pre)
            mod.register_forward_hook(post)
    opt.zero_grad(set_to_none=True)
    f0 = fb()
    loss = torch.nn.functional.cross_entropy(model(xb), y)
    mark()
    f1 = fb()
    hpu_ops._STATE["calls"].clear()
    loss.backward()
    mark()
    f2 = fb()
    opt.step()
    mark()
    f3 = fb()
    print("forward", delta(f0, f1), "\nbackward", delta(f1, f2), "rewrites", dict(hpu_ops._STATE["calls"]),
          "\noptimizer", delta(f2, f3), flush=True)


if __name__ == "__main__":
    main()
