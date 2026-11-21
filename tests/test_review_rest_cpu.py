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
