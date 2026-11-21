"""The non-linalg findings of the max-effort review of main (``gate/code-review-hpu-ops.txt``), each with the review's
exact input, on the CPU standing in for HPU. Each test fails on main (bda890b) and passes with the fix."""
import os
import pathlib
import subprocess
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
aten = torch.ops.aten
NAN, INF = float("nan"), float("inf")


def _run(code, env=None):
    """``code`` in a fresh interpreter with this checkout first on the path; returns its stdout."""
    environ = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "tests")]), **(env or {}))
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=environ, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout.strip()


# ---- (2) quantile's import-time probes read the default dtype and device


PROBES = "import hpu_ops.quantile as q; print(q._ranks_in_float64(), q._length_limited())"


@pytest.mark.parametrize("setup", ["torch.set_default_dtype(torch.bfloat16)", "torch.set_default_dtype(torch.float16)",
                                   "torch.set_default_dtype(torch.float64)", "torch.set_default_device('meta')"])
def test_import_under_another_default_dtype_or_device(setup):
    """``torch.set_default_dtype(torch.bfloat16); import hpu_ops`` raised (quantile rejects bf16), and the meta default
    device raised on ``item()``. The probes give the default process's answers."""
    want = _run(PROBES)
    assert _run(f"import torch; {setup}; import hpu_ops; {PROBES}") == want


# ---- (5) compiled inference_mode graphs left padding="same" and unbatched large-kernel convolutions whole


def _conv_widths(layer, x, context):
    import torch.nn as nn  # noqa: F401
    from test_cpu import compiled, cpu_as_hpu
    from hpu_ops import _dispatch, conv

    nodes = []
    with cpu_as_hpu(lambda d: _dispatch.chain([conv.install(d), conv.install_large_kernel(d)])), context():
        got = compiled(layer, nodes, dynamic=False)(x)
    widths = [tuple(n.args[1].meta["val"].shape[2:]) for n in nodes if n.target is aten.convolution.default]
    return got, widths


def _conv_cases():
    import torch.nn as nn

    torch.manual_seed(0)
    return {
        "conv1d k351 same": (nn.Conv1d(1, 2, 351, padding="same"), torch.randn(2, 1, 400)),
        "conv1d k350 same (asymmetric)": (nn.Conv1d(1, 2, 350, padding="same"), torch.randn(2, 1, 400)),
        "conv1d k300 same dilation 2": (nn.Conv1d(2, 2, 300, padding="same", dilation=2), torch.randn(1, 2, 700)),
        "conv2d (1,301) same": (nn.Conv2d(1, 4, (1, 301), padding="same"), torch.randn(2, 1, 3, 400)),
        "conv1d k351 unbatched": (nn.Conv1d(1, 2, 351), torch.randn(1, 400)),
        "conv2d (351,1) unbatched": (nn.Conv2d(1, 4, (351, 1)), torch.randn(1, 400, 3)),
        "conv1d k351 same unbatched": (nn.Conv1d(1, 2, 351, padding="same"), torch.randn(1, 400)),
    }


@pytest.mark.parametrize("name", list(_conv_cases()))
@pytest.mark.parametrize("context", [torch.inference_mode, torch.no_grad])
def test_compiled_inference_mode_splits_same_and_unbatched_large_kernels(monkeypatch, name, context):
    """The review's inputs: ``nn.Conv1d(1, 2, 351, padding='same')`` on (2, 1, 400), ``nn.Conv2d(1, 4, (1, 301),
    padding='same')``, ``nn.Conv1d(1, 2, 351)`` on an unbatched (1, 400): compiled under ``inference_mode()`` the graph
    kept one 351-wide convolution (Gaudi2's MME rejects it). Every graph convolution is now at most 256 wide and the
    values are the layer's."""
    layer, x = _conv_cases()[name]
    with torch.no_grad():
        want = layer(x)
    got, widths = _conv_widths(layer, x, context)
    assert widths and all(max(w) <= 256 for w in widths), widths
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
