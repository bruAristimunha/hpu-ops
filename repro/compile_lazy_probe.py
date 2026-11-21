"""torch.compile(backend="hpu_backend") under PT_HPU_LAZY_MODE=0|1, with and without hpu_ops / hpu_complex.
    PT_HPU_LAZY_MODE=1 python3 repro/compile_lazy_probe.py            # every case, one subprocess each
    PT_HPU_LAZY_MODE=1 python3 repro/compile_lazy_probe.py mlp none   # one case in this process
Models: mlp (Linear-ReLU-Linear), conv (Conv2d, the 4-D convolution Habana's weight permutation pass targets),
SPDNet (spd_diag's, on a real BNCI2014_001 batch). Plugins: none, ops (hpu_ops.enable()), ops+complex (also
hpu_complex.enable()). Each case: compile, three train steps (forward, backward, Adam); one ``CASE {json}`` line
with the status, the exception and its last frames."""
import json
import os
import subprocess
import sys
import traceback

MODELS = ("mlp", "conv", "SPDNet")
PLUGINS = ("none", "ops", "ops+complex")


def build(name):
    import torch
    torch.manual_seed(0)
    if name == "mlp":
        return (torch.nn.Sequential(torch.nn.Linear(22, 64), torch.nn.ReLU(), torch.nn.Linear(64, 4)),
                torch.randn(32, 22), torch.randint(0, 4, (32,)))
    if name == "conv":
        return (torch.nn.Sequential(torch.nn.Conv2d(1, 8, (1, 25)), torch.nn.ReLU(), torch.nn.Flatten(),
                                    torch.nn.Linear(8 * 22 * 76, 4)),
                torch.randn(32, 1, 22, 100), torch.randint(0, 4, (32,)))
    from spd_diag import protocol as P
    from spd_diag.cache import open_cache
    cache = open_cache("BNCI2014_001")
    sd = cache.subject(cache.subjects[0], "X")
    X = torch.as_tensor(sd.X[:32], dtype=torch.float32)
    return P.build(name, X.shape[1], cache.info, None), X / X.std(), torch.as_tensor(sd.y[:32]).long()


def one(name, plugins):
    row = {"model": name, "plugins": plugins, "lazy": os.environ.get("PT_HPU_LAZY_MODE")}
    try:
        if plugins != "none":
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            import hpu_ops
            row["hpu_ops"] = hpu_ops.__file__
            hpu_ops.enable()
        if plugins == "ops+complex":
            import hpu_complex
            hpu_complex.enable()
        import habana_frameworks.torch.core as htcore
        import torch
        model, X, y = build(name)
        hpu = torch.device("hpu")
        model, X, y = model.to(hpu), X.to(hpu), y.to(hpu)
        opt = torch.optim.Adam(model.parameters(), 1e-3)
        compiled = torch.compile(model, backend="hpu_backend")
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            out = compiled(X)
            out = out[0] if isinstance(out, tuple) else out
            loss = torch.nn.functional.cross_entropy(out, y)
            loss.backward()
            opt.step()
            htcore.mark_step()
        row.update(status="ok", loss=round(loss.item(), 4))
    except Exception as e:  # noqa: BLE001
        row.update(status="error", error=f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}",
                   where=[f"{f.filename.split('site-packages/')[-1]}:{f.lineno} {f.name}"
                          for f in traceback.extract_tb(e.__traceback__)][-5:])
    print("CASE " + json.dumps(row), flush=True)


def main():
    if len(sys.argv) == 3:
        return one(*sys.argv[1:])
    for name in MODELS:
        for plugins in PLUGINS:
            out = subprocess.run([sys.executable, os.path.abspath(__file__), name, plugins], capture_output=True,
                                 text=True, timeout=900)
            lines = [line for line in out.stdout.splitlines() if line.startswith("CASE ")]
            print(lines[-1] if lines else "CASE " + json.dumps(
                {"model": name, "plugins": plugins, "status": "crash", "rc": out.returncode,
                 "tail": (out.stdout + out.stderr).splitlines()[-6:]}), flush=True)


if __name__ == "__main__":
    main()
