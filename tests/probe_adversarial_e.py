"""Gaudi probe, review round 2 (adversarial), team E: tries to break the #20 index rewrites and the #19 conv split.
    PT_HPU_LAZY_MODE=0|1 python3 tests/probe_adversarial_e.py [ops|perf|compile]
Every case runs natively first (hpu_ops not enabled: origin/main's behaviour for these ops), then with
hpu_ops.enable(), and both are compared with CPU float64. Prints one ``PROBE {json}`` line per case."""
import json
import os
import sys
import time
import traceback
import warnings

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch
import torch.nn.functional as F
from habana_frameworks.torch.hpu.metrics import metric_global

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "0") != "0"
HPU = torch.device("hpu")


def mark():
    if LAZY:
        htcore.mark_step()


def fb():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def show(case, **row):
    print("PROBE " + json.dumps({"case": case, "lazy": LAZY, "bridge": "1.21.4", **row}, default=str), flush=True)


def to_cpu(out):
    if isinstance(out, (tuple, list)):
        return [to_cpu(o) for o in out]
    return None if out is None else out.detach().cpu()


def err(a, b):
    if isinstance(a, list):
        return [err(x, y) for x, y in zip(a, b)]
    if a is None or b is None:
        return None if a is None and b is None else "one is None"
    if a.shape != b.shape:
        return f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    if a.numel() == 0:
        return 0.0
    if a.dtype == torch.bool or not (a.is_floating_point() or a.is_complex()):
        return int((a != b).sum())
    a, b = a.double(), b.double()
    same_nan = torch.equal(a.isnan(), b.isnan())
    d = (a - b).abs().nan_to_num(0.0)
    return {"max_abs": float(d.max()), "nan_match": same_nan}


def run(fn, make):
    """fn(*make(device)) on the card; returns (cpu outputs, fallback delta, cpu warnings) or the exception."""
    torch.manual_seed(0)  # same data as the CPU float64 reference
    args = make(HPU)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f0 = fb()
        try:
            out = fn(*args)
            mark()
            out = to_cpu(out)
            hthpu.synchronize()
        except Exception as e:  # noqa: BLE001
            return {"raised": repr(e)[:400], "at": traceback.format_exc().splitlines()[-3:-1]}
        f1 = fb()
    return {"out": out, "fb": {k: f1[k] - f0.get(k, 0) for k in f1 if f1[k] - f0.get(k, 0)},
            "warn": sorted({str(w.message)[:120] for w in caught if "runs on the CPU" in str(w.message)})}


def case(name, fn, make, ref=True):
    """Native then hpu_ops run of fn on the card, each compared with fn on CPU float64 (if ref)."""
    import hpu_ops
    only = os.environ.get("CASE")  # run one case (substring) in its own process: a failed lazy graph poisons the next
    if only and only not in name:
        return
    want = None
    if ref:
        try:
            torch.manual_seed(0)
            want = to_cpu(fn(*[a.double() if torch.is_tensor(a) and a.is_floating_point() else a
                               for a in make(torch.device("cpu"))]))
        except Exception as e:  # noqa: BLE001
            want = None
            show(name, phase="cpu64", raised=repr(e)[:300])
    rows = {}
    for phase in os.environ.get("PHASES", "native,hpu_ops").split(","):
        if phase == "hpu_ops":
            hpu_ops.enable()
            hpu_ops._STATE["calls"].clear()
        r = run(fn, make)
        if "out" in r:
            out = r.pop("out")
            r["err_vs_cpu64"] = err(out, want) if want is not None else None
            r["sample"] = (out[0] if isinstance(out, list) else out)
            r["sample"] = r["sample"].reshape(-1)[:4].tolist() if r["sample"] is not None else None
        if phase == "hpu_ops":
            r["calls"] = dict(hpu_ops._STATE["calls"])
            UNDO()
        rows[phase] = r
    show(name, **rows)


def UNDO():
    """Undo enable(): reload the dispatcher state so the next case's native run is native."""
    import hpu_ops
    undo = hpu_ops._STATE.pop("undo", None)
    if callable(undo):
        undo()


def grad_of(f):
    def g(x, *rest):
        x = x.detach().requires_grad_()
        y = f(x, *rest)
        (gx,) = torch.autograd.grad(y.float().pow(2).sum(), (x,))
        return [y, gx]
    return g


def ops_cases():
    tri = torch.triu_indices(6, 6)

    def vech(dev, dtype=torch.float32):
        torch.manual_seed(0)
        return (torch.randn(4, 3, 6, 6).to(dtype).to(dev), tri[0].to(dev), tri[1].to(dev))

    idx = lambda x, a, b: x[..., a, b]  # noqa: E731
    case("index_2d vech f32 fwd+grad (device idx)", grad_of(idx), vech)
    case("index_2d vech host idx fwd+grad", grad_of(idx), lambda d: (vech(d)[0], tri[0], tri[1]))
    case("index_2d vech bf16 fwd+grad", grad_of(idx), lambda d: vech(d, torch.bfloat16), ref=False)
    case("index_2d int64 data", idx, lambda d: (torch.arange(4 * 36).reshape(4, 6, 6).to(d), tri[0].to(d), tri[1].to(d)))
    case("index_2d int32 data", idx,
         lambda d: (torch.arange(4 * 36, dtype=torch.int32).reshape(4, 6, 6).to(d), tri[0].to(d), tri[1].to(d)))
    case("index_2d bool data", idx,
         lambda d: ((torch.arange(4 * 36).reshape(4, 6, 6) % 3 == 0).to(d), tri[0].to(d), tri[1].to(d)))
    case("index_2d int32 indices broadcast (3,1)x(4,) batch 1", grad_of(idx),
         lambda d: (torch.randn(1, 5, 6).to(d), torch.tensor([[0], [4], [-1]], dtype=torch.int32).to(d),
                    torch.tensor([1, -6, 5, 0], dtype=torch.int32).to(d)))
    case("index_2d transposed offset view", grad_of(lambda x, a, b: x[1:, :, 1:].transpose(-1, -2)[..., a, b]),
         lambda d: (torch.randn(3, 7, 8).to(d), tri[0][:15].to(d), tri[1][:15].to(d)))
    case("index_2d 1x1 trailing", grad_of(idx),
         lambda d: (torch.randn(5, 1, 1).to(d), torch.tensor([0, -1, 0]).to(d), torch.tensor([0, 0, -1]).to(d)))
    case("index_2d empty index", grad_of(idx),
         lambda d: (torch.randn(2, 4, 4).to(d), torch.empty(0, dtype=torch.long).to(d),
                    torch.empty(0, dtype=torch.long).to(d)))
    case("index_2d empty batch", grad_of(idx), lambda d: (torch.randn(0, 6, 6).to(d), tri[0].to(d), tri[1].to(d)))
    case("index_2d duplicate idx grad", grad_of(idx),
         lambda d: (torch.randn(2, 4, 4).to(d), torch.tensor([0, 0, 3, 3]).to(d), torch.tensor([1, 1, 2, -2]).to(d)))
    case("index_2d OOB device idx fwd+grad (no CPU ref)", grad_of(idx),
         lambda d: (torch.arange(24.0).reshape(2, 3, 4).to(d), torch.tensor([0, 1]).to(d), torch.tensor([4, 1]).to(d)),
         ref=False)
    case("index_2d large 64x64 vech, batch 256", grad_of(idx),
         lambda d: (torch.randn(256, 64, 64).to(d), *[t.to(d) for t in torch.triu_indices(64, 64)]))

    mask_put = lambda x, m, v: x.clone().index_put_([m], v)  # noqa: E731
    acc_put = lambda x, m, v: x.clone().index_put_([m], v, accumulate=True)  # noqa: E731

    def mk(shape, dtype=torch.float32, mshape=None):
        def make(d):
            torch.manual_seed(1)
            x = (torch.randn(*shape) * 3).to(dtype)
            m = torch.randn(*(mshape or shape)) > 0
            return (x.to(d), m.to(d), torch.tensor(-2, dtype=dtype).to(d))
        return make

    case("bool index_put_ f32", mask_put, mk((8, 18)))
    case("bool index_put_ accumulate f32", acc_put, mk((8, 18)))
    case("bool index_put_ leading mask (8,) on (8,18)", mask_put, mk((8, 18), mshape=(8,)))
    case("bool index_put_ int64 self", mask_put, mk((8, 18), torch.int64))
    case("bool index_put_ accumulate int64 self", acc_put, mk((8, 18), torch.int64))
    case("bool index_put_ int32 self", mask_put, mk((8, 18), torch.int32))
    case("bool index_put_ bf16 self", mask_put, mk((8, 18), torch.bfloat16), ref=False)
    case("bool index_put_ bool self", mask_put,
         lambda d: (torch.zeros(4, 5, dtype=torch.bool).to(d), (torch.randn(4, 5) > 0).to(d), torch.tensor(True).to(d)))
    case("bool index_put_ cpu 0-d value", mask_put,
         lambda d: (torch.randn(4, 5).to(d), (torch.randn(4, 5) > 0).to(d), torch.tensor(7.0)))
    case("bool index_put_ empty self", mask_put,
         lambda d: (torch.randn(0, 5).to(d), torch.zeros(0, 5, dtype=torch.bool).to(d), torch.tensor(1.0).to(d)))
    case("bool index_put_ transposed self", lambda x, m, v: x.t().index_put_([m], v).t().clone(),
         lambda d: (torch.randn(5, 4).to(d), (torch.randn(4, 5) > 0).to(d), torch.tensor(9.0).to(d)))
    case("bool index_put_ on a column-slice view (non-dense self)",
         lambda x, m, v: x.clone()[:, 1:].index_put_([m], v),
         lambda d: (torch.randn(8, 19).to(d), (torch.randn(8, 18) > 0).to(d), torch.tensor(0.0).to(d)))

    def slice_in_backward(x):
        x = x.detach().requires_grad_()
        y = (x * 2.0)[:, 1:]
        y = y.clone()[:, ::1]
        z = (x * 3.0)
        zz = z[:, 1:]
        zz.index_put_([zz <= 0], torch.tensor(0.0, device=x.device))
        (gx,) = torch.autograd.grad((z * z).sum() + y.sum(), (x,))
        return [z, gx]

    case("bool index_put_ on a slice view, train step", slice_in_backward, lambda d: (torch.randn(8, 19).to(d),))
    case("bool index_put value near overflow", mask_put,
         lambda d: (torch.randn(4, 5).to(d), (torch.randn(4, 5) > 0).to(d), torch.tensor(3.3e38).to(d)))
    case("bool index_put_ subnormal value (FTZ?)", mask_put,
         lambda d: (torch.randn(4, 5).to(d), (torch.randn(4, 5) > 0).to(d), torch.tensor(1e-40).to(d)), ref=False)

    def custom_backward(x):
        class Thr(torch.autograd.Function):
            @staticmethod
            def forward(ctx, a):
                ctx.save_for_backward(a)
                return a.clamp_min(0)

            @staticmethod
            def backward(ctx, g):
                (a,) = ctx.saved_tensors
                s = g.clone()
                s[a <= 0] = 0
                return s
        x = x.detach().requires_grad_()
        y = Thr.apply(x)
        (gx,) = torch.autograd.grad((y * y).sum(), (x,))
        return [y, gx]

    case("bool mask assign inside autograd.Function.backward", custom_backward, lambda d: (torch.randn(8, 18).to(d),))

    def conv_case(name, ishape, wshape, stride=1, padding=0, dilation=1, groups=1, dtype=torch.float32, ref=True):
        f = {3: F.conv1d, 4: F.conv2d, 5: F.conv3d}[len(ishape)]

        def fn(x, w, b):
            x, w, b = (t.detach().requires_grad_() for t in (x, w, b))
            y = f(x, w, b, stride, padding, dilation, groups)
            gs = torch.autograd.grad(y.float().pow(2).sum(), (x, w, b), allow_unused=True)
            return [y, *gs]

        def make(d):
            torch.manual_seed(2)
            return tuple(t.to(dtype).to(d) for t in (torch.randn(*ishape), torch.randn(*wshape) / wshape[-1] ** 0.5,
                                                      torch.randn(wshape[0])))
        case(name, fn, make, ref=ref)

    # BREAK (lazy, bridge 1.21.4): a native k=256 conv over L >= 300 fails setAssociatedDimAndSize; k=255 compiles.
    # The split's 256-wide chunks therefore fail in lazy, and k == 256 itself is not taken (it is not > MAX_KERNEL).
    conv_case("conv1d k=256 exactly, L=1000", (4, 8, 1000), (16, 8, 256))
    conv_case("conv1d k=351 (STaRNet-like)", (4, 8, 1000), (16, 8, 351))
    conv_case("conv2d (4,1089) (TensorCSPNet-like)", (2, 4, 4, 1500), (8, 4, 4, 1089))
    conv_case("conv1d k=257 empty batch", (0, 2, 400), (3, 2, 257))
    conv_case("conv1d k=300 batch 1, output length 1", (1, 2, 300), (3, 2, 300))
    conv_case("conv1d k=300 stride 3 pad 150 dil 2 groups 2", (2, 4, 900), (4, 2, 300), 3, 150, 2, 2)
    conv_case("conv1d k=600 padding > input (chunks in padding)", (1, 1, 100), (2, 1, 600), 1, 300)
    conv_case("conv1d k=351 bf16", (4, 8, 1000), (16, 8, 351), dtype=torch.bfloat16)
    conv_case("conv2d (300,300) both dims split", (1, 1, 310, 310), (1, 1, 300, 300))
    conv_case("conv1d k=257 depthwise 64 ch", (2, 64, 500), (64, 1, 257), groups=64)

    def conv_noncontig(x, w, b):
        return F.conv1d(x.transpose(-1, -2)[..., 3:], w, b)

    case("conv1d k=300 non-contiguous offset input", conv_noncontig,
         lambda d: (torch.randn(2, 900, 4).to(d), torch.randn(5, 4, 300).to(d) / 17, torch.randn(5).to(d)))

    def conv_inference(x, w, b):
        with torch.inference_mode():
            return F.conv1d(x, w, b).clone()

    case("conv1d k=351 inference_mode", conv_inference,
         lambda d: (torch.randn(2, 8, 1000).to(d), torch.randn(16, 8, 351).to(d) / 19, torch.randn(16).to(d)))


def lazy_twice():
    """The same rewritten graph built and run twice in a row in one process (lazy cache reuse, races)."""
    import hpu_ops
    hpu_ops.enable()
    tri = torch.triu_indices(6, 6)
    outs = []
    for rep in range(3):
        torch.manual_seed(0)
        x = torch.randn(4, 3, 6, 6, device=HPU, requires_grad=True)
        y = x @ x.transpose(-1, -2)
        v = y[..., tri[0].to(HPU), tri[1].to(HPU)]
        s = v.clone()
        s[s <= 0] = 0
        w = F.conv1d(s.reshape(4, 3, 21).repeat(1, 1, 20), torch.ones(2, 3, 300, device=HPU) / 300)
        (w.sum() + s.sum()).backward()
        mark()
        outs.append((float(w.sum().cpu()), float(x.grad.abs().sum().cpu())))
    show("rewritten graph 3 runs in a row", outs=outs, equal=len(set(outs)) == 1, calls=dict(hpu_ops._STATE["calls"]))


def timed(fn, args, n=30):
    for _ in range(5):
        keep = fn(*args)
    mark()
    hthpu.synchronize()
    keep = []
    t = time.perf_counter()
    for _ in range(n):
        keep.append(fn(*args))
        mark()
    hthpu.synchronize()
    return round((time.perf_counter() - t) / n * 1e3, 4)


def perf():
    """Per-call cost the Python kernels add to ops they do not rewrite (every conv, every 2-D gather), same process."""
    import hpu_ops
    x = torch.randn(32, 22, 500, device=HPU)
    w = torch.randn(16, 22, 25, device=HPU)
    g = torch.randn(64, 64, device=HPU)
    i0, i1 = torch.randint(64, (512,), device=HPU), torch.randint(64, (512,), device=HPU)
    big = torch.randn(4, 8, 1000, device=HPU)
    wb = torch.randn(16, 8, 351, device=HPU)
    tri = torch.triu_indices(22, 22)
    tri_d = tri.to(HPU)
    v = torch.randn(32, 22, 22, device=HPU, requires_grad=True)

    def vech_step(t):
        (g,) = torch.autograd.grad(t[..., tri[0], tri[1]].square().sum(), (t,))
        return g

    def vech_step_dev(t):
        (g,) = torch.autograd.grad(t[..., tri_d[0], tri_d[1]].square().sum(), (t,))
        return g

    rows = {}
    for phase in os.environ.get("PHASES", "native,hpu_ops").split(","):
        if phase == "hpu_ops":
            hpu_ops.enable()
        rows[phase] = {"conv1d k=25 (not split)": timed(F.conv1d, (x, w)),
                       "x[i0,i1] 64x64 gather": timed(lambda a, b, c: a[b, c], (g, i0, i1)),
                       "vech host idx fwd+bwd 32x22x22": timed(vech_step, (v,)),
                       "vech device idx fwd+bwd 32x22x22": timed(vech_step_dev, (v,)),
                       "conv1d k=351": timed(F.conv1d, (big, wb)) if phase == "hpu_ops" else None}
    show("per-call ms (mean of 30 synced calls)", **rows)


def compiled():
    """torch.compile(backend='hpu_backend') of the rewritten ops, forward and backward, against CPU float64."""
    import hpu_ops
    hpu_ops.enable()
    tri = torch.triu_indices(6, 6)

    def f(x, w):
        y = x @ x.transpose(-1, -2)
        v = y[..., tri[0], tri[1]].clone()
        v[v <= 0] = 0
        return F.conv1d(v.reshape(4, 3, 21).repeat(1, 1, 20), w)

    torch.manual_seed(0)
    x0, w0 = torch.randn(4, 3, 6, 6), torch.randn(2, 3, 300) / 17
    xc, wc = x0.double().requires_grad_(), w0.double().requires_grad_()
    yc = f(xc, wc)
    yc.pow(2).sum().backward()
    for dyn in (False, None):
        torch._dynamo.reset()
        hpu_ops._STATE["calls"].clear()
        try:
            cf = torch.compile(f, backend="hpu_backend", dynamic=dyn)
            outs = []
            for rep in range(2):
                xh, wh = x0.to(HPU).requires_grad_(), w0.to(HPU).requires_grad_()
                y = cf(xh, wh)
                y.pow(2).sum().backward()
                outs.append([err(y.detach().cpu(), yc.detach()), err(xh.grad.cpu(), xc.grad),
                             err(wh.grad.cpu(), wc.grad)])
            show(f"compile hpu_backend dynamic={dyn}", errs=outs, calls=dict(hpu_ops._STATE["calls"]))
        except Exception as e:  # noqa: BLE001
            show(f"compile hpu_backend dynamic={dyn}", raised=repr(e)[:500])


def conv_lazy():
    """Which part of a split large-kernel convolution step fails (VARIANT: fwd | wgrad | xgrad | full)."""
    import hpu_ops
    hpu_ops.enable()
    variant = os.environ.get("VARIANT", "full")
    shape = os.environ.get("SHAPE", "1d")
    torch.manual_seed(2)
    if shape == "1d":
        x0, w0, f = torch.randn(4, 8, 1000), torch.randn(16, 8, 351) / 19, F.conv1d
    else:
        x0, w0, f = torch.randn(2, 4, 4, 1500), torch.randn(8, 4, 4, 1089) / 33, F.conv2d
    x = x0.to(HPU).requires_grad_(variant in ("xgrad", "full"))
    w = w0.to(HPU).requires_grad_(variant in ("wgrad", "full"))
    try:
        y = f(x, w)
        outs = [y]
        if variant != "fwd":
            wants = [t for t in (x, w) if t.requires_grad]
            outs += list(torch.autograd.grad(y.pow(2).sum(), wants))
        mark()
        outs = [o.cpu() for o in outs]
        xc, wc = x0.double().requires_grad_(x.requires_grad), w0.double().requires_grad_(w.requires_grad)
        yc = f(xc, wc)
        ref = [yc]
        if variant != "fwd":
            ref += list(torch.autograd.grad(yc.pow(2).sum(), [t for t in (xc, wc) if t.requires_grad]))
        show(f"split conv lazy {shape} {variant}", ok=True, errs=[err(a, b) for a, b in zip(outs, ref)],
             calls=dict(hpu_ops._STATE["calls"]))
    except Exception as e:  # noqa: BLE001
        show(f"split conv lazy {shape} {variant}", ok=False, raised=repr(e)[:300], calls=dict(hpu_ops._STATE["calls"]))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "ops"
    import hpu_ops
    if what == "ops":
        ops_cases()
    elif what == "twice":
        lazy_twice()
    elif what == "perf":
        perf()
    elif what == "convlazy":
        conv_lazy()
    elif what == "compile":
        compiled()
