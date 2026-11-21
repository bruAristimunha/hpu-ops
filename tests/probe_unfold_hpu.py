"""On a Gaudi card: ``Tensor.unfold`` with hpu_ops' rewrite, adversarial cases, eager and lazy.
    PT_HPU_LAZY_MODE=0|1 python3 tests/probe_unfold_hpu.py
Each case: values against the CPU, sizes/strides/offset against the CPU's view, a write into the input seen through
the view, input gradient (and second order) against the CPU, and the bridge's CPU-fallback ops it made. Prints one
``CASE`` line per case and a ``SUMMARY``; exits 1 if any case is wrong or ``aten::unfold`` fell back."""
import os
import sys

import habana_frameworks.torch.core as htcore
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

import hpu_ops

hpu_ops.enable()
LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"


def fb():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def mark():
    if LAZY:
        htcore.mark_step()


ERRORS = {}  # name -> relative errors of the last case, printed with it


def close(a, b, what="values"):
    """Relative error (norm) of the card's ``a`` against the CPU's ``b``, within 1e-5 (2e-2 for 16-bit)."""
    if a.shape != b.shape:
        return False
    low = a.dtype in (torch.bfloat16, torch.float16) or "16" in what
    a, b = a.cpu().double(), b.double()
    err = ((a - b).norm() / b.norm().clamp_min(1e-30)).item() if b.numel() else 0.0
    ERRORS[what] = float(f"{err:.2e}")
    return err <= (2e-2 if low else 1e-5)


def case(name, make, chain):
    """``make()`` builds the CPU input; ``chain(t)`` applies the unfolds."""
    f0 = fb()
    row = {}
    try:
        x = make()
        xh = make_on_hpu(x, name)
        ERRORS.clear()
        want, got = chain(x), chain(xh)
        mark()
        row["values"] = close(got, want)
        row["layout"] = (tuple(got.shape), got.stride(), got.storage_offset() - xh.storage_offset()) == (
            tuple(want.shape), want.stride(), want.storage_offset() - x.storage_offset())
        if x.numel() and 0 not in x.stride():
            xh.add_(1)
            mark()
            row["view"] = close(got, chain(x + 1))
            x = x + 1  # the card's input now holds x + 1
        if x.dim() and x.is_floating_point():
            xg, xhg = x.detach().double().requires_grad_(), xh.detach().clone().requires_grad_()
            (g,) = torch.autograd.grad(chain(xg).sin().square().sum(), xg, create_graph=True)
            (gg,) = torch.autograd.grad(g.square().sum(), xg)
            (gh,) = torch.autograd.grad(chain(xhg).sin().square().sum(), xhg, create_graph=True)
            (ggh,) = torch.autograd.grad(gh.square().sum(), xhg)
            mark()
            tag = "16" if xh.dtype in (torch.bfloat16, torch.float16) else ""
            row["grad"] = close(gh, g, "grad" + tag)
            row["grad2"] = close(ggh, gg, "grad2" + tag)
    except Exception as e:  # noqa: BLE001
        row["error"] = repr(e)[:200]
    mark()
    d = fb()
    row["fallbacks"] = {k: d[k] - f0.get(k, 0) for k in d if d[k] - f0.get(k, 0)}
    print("CASE", name, row, "rel_err", dict(ERRORS), flush=True)
    return row


def write_case(name, write, card_want=None):
    """``write(t)`` writes through an unfold view of ``t``; the base and the view must equal the CPU's after it (or
    both raise)."""
    f0 = fb()
    torch.manual_seed(2)
    x = torch.randn(4, 13)
    xh = x.to("hpu")
    x0 = x.clone()
    row = {}
    try:
        view = write(x)
        want_err = None
    except RuntimeError as e:
        view, want_err = None, str(e)[:60]
    try:
        viewh = write(xh)
        mark()
        got_err = None
    except RuntimeError as e:
        viewh, got_err = None, str(e)[:60]
    row["same_error"] = (want_err is None) == (got_err is None)
    if card_want is not None and got_err is None:
        row["base_as_eager_card"] = close(xh, card_want(x0), "base")
    elif want_err is None and got_err is None:
        row["base"] = close(xh, x, "base")
        row["view"] = close(viewh, view, "view")
    mark()
    d = fb()
    row["fallbacks"] = {k: d[k] - f0.get(k, 0) for k in d if d[k] - f0.get(k, 0)}
    print("CASE write", name, row, "cpu_error", want_err, "card_error", got_err, flush=True)
    return row


def make_on_hpu(x, name):
    """The same layout on the card: the CPU input's base moved, then the same view of it."""
    base = x._base if x._base is not None else x
    bh = base.to("hpu")
    return bh.as_strided(x.shape, x.stride(), x.storage_offset())


def main():
    print("hpu_ops", hpu_ops.__file__, "lazy", LAZY, "HPU kernel for unfold (ours in lazy)",
          torch._C._dispatch_has_kernel_for_dispatch_key("aten::unfold", "HPU"), flush=True)
    torch.manual_seed(0)
    b = torch.randn(4, 7, 13)
    cases = [
        ("contiguous", lambda: b.clone(), lambda t: t.unfold(2, 4, 3)),
        ("step>size", lambda: b.clone(), lambda t: t.unfold(2, 3, 5)),
        ("odd 13,5,2", lambda: b.clone(), lambda t: t.unfold(2, 5, 2)),
        ("size==dim", lambda: b.clone(), lambda t: t.unfold(1, 7, 1)),
        ("size 1", lambda: b.clone(), lambda t: t.unfold(0, 1, 1)),
        ("neg dim -1", lambda: b.clone(), lambda t: t.unfold(-1, 5, 2)),
        ("neg dim -3", lambda: b.clone(), lambda t: t.unfold(-3, 2, 2)),
        ("transposed", lambda: b.clone().transpose(0, 2), lambda t: t.unfold(1, 3, 2)),
        ("offset view", lambda: b.clone()[1:, 2:, 3:], lambda t: t.unfold(2, 4, 3)),
        ("strided slice", lambda: b.clone()[:, ::2, 1::3], lambda t: t.unfold(1, 2, 1)),
        ("1-d", lambda: torch.arange(11.0), lambda t: t.unfold(0, 4, 4)),
        ("0-d", lambda: torch.tensor(3.0), lambda t: t.unfold(0, 1, 1)),
        ("twice (conv2d patches)", lambda: b.clone(), lambda t: t.unfold(1, 3, 1).unfold(2, 8, 4)),
        ("bfloat16", lambda: b.clone().bfloat16(), lambda t: t.unfold(2, 4, 3)),
        ("int32", lambda: torch.arange(40, dtype=torch.int32).reshape(4, 10), lambda t: t.unfold(1, 3, 2)),
    ]
    rows = [case(*c) for c in cases]

    # writes THROUGH the view (and a view of it): base and view against the CPU's
    writes = [
        ("fill first of non-overlapping", lambda t: t.unfold(1, 3, 3).select(-1, 0).fill_(1)),
        ("add first of overlapping", lambda t: t.unfold(1, 3, 1)[..., 0].add_(1)),
        ("mul view of view", lambda t: t.unfold(1, 4, 2)[:, 1::2].select(-1, 1).mul_(3)),
        ("copy_ whole non-overlapping", lambda t: t.unfold(1, 3, 3).copy_(torch.ones_like(t.unfold(1, 3, 3)))),
        ("transposed base", lambda t: t.t().unfold(0, 3, 3).select(-1, 0).fill_(2)),
    ]
    for name, write in writes:
        rows.append(write_case(name, write))
    # undefined in torch: the CPU adds once per window holding an element, the card once per element, in eager
    # mode with Habana's own unfold too (repro/unfold_overlapping_inplace.py); lazy must match eager's card result
    rows.append(write_case("overlapping in-place add_ (undefined)", lambda t: t.unfold(1, 3, 1).add_(1),
                           card_want=lambda x: x + 1))

    # TensorCSPNet's temporal block: hpu-ops' weight gradient, and its double backward
    f0 = fb()
    torch.manual_seed(1)
    conv = torch.nn.Conv2d(1, 4, (3, 2 * 9), stride=(1, 9))
    x = torch.randn(5, 1, 3, 54, requires_grad=True)
    (g,) = torch.autograd.grad(conv(x).square().sum(), conv.weight, create_graph=True)
    (gg,) = torch.autograd.grad(g.square().sum(), x)
    convh = conv.to("hpu")
    xh = x.detach().to("hpu").requires_grad_()
    (gh,) = torch.autograd.grad(convh(xh).square().sum(), convh.weight, create_graph=True)
    (ggh,) = torch.autograd.grad(gh.square().sum(), xh)
    mark()
    d = fb()
    ERRORS.clear()
    row = {"weight_grad": close(gh, g.detach(), "weight_grad"), "double_backward": close(ggh, gg, "double"),
           "rel_err": dict(ERRORS), "rewrites":
           dict(hpu_ops._STATE["calls"]),
           "fallbacks": {k: d[k] - f0.get(k, 0) for k in d if d[k] - f0.get(k, 0)}}
    print("CASE conv temporal block", row, flush=True)
    rows.append(row)

    # compiled (hpu_backend): eager mode only; on bridge 1.21.4 the backend fails to import in lazy mode
    if LAZY:
        print("CASE compiled unavailable in lazy mode", flush=True)
        return summary(rows)
    try:
        f = torch.compile(lambda t: (t.unfold(2, 4, 3) * 2).sum(-1), backend="hpu_backend")
        xh = b.to("hpu")
        got = [f(xh) for _ in range(2)]
        mark()
        row = {"values": all(close(o, (b.unfold(2, 4, 3) * 2).sum(-1)) for o in got)}
    except Exception as e:  # noqa: BLE001
        row = {"error": repr(e)[:200]}
    print("CASE compiled", row, flush=True)
    rows.append(row)
    return summary(rows)


def summary(rows):
    calls = hpu_ops._STATE["calls"]["unfold"]
    rows.append({"rewrite_calls_as_expected": calls > 0 if LAZY else calls == 0})
    print("CASE rewrite calls", calls, "expected", ">0 (lazy)" if LAZY else "0 (eager: Habana's kernel)",
          rows[-1], flush=True)
    def is_bad(r):
        wrong = not all(v for v in r.values() if isinstance(v, bool))
        return "error" in r or wrong or "aten::unfold" in r.get("fallbacks", {})

    bad = [r for r in rows if is_bad(r)]
    print(f"SUMMARY {len(rows)} cases, {len(bad)} bad", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
