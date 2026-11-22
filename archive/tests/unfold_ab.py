"""On a Gaudi card: ``tests/linalg_steps.py`` with or without hpu_ops' ``Tensor.unfold`` rewrite, in one tree.
    python3 tests/unfold_ab.py before|after [--script model_parity.py] [script args...]
``before`` drops ``composite.install_unfold`` from ``hpu_ops.FIXES`` before ``enable()`` (the code before the fix);
``after`` runs every fix. The script (default ``linalg_steps.py``) is one of tests/."""
import os
import runpy
import sys

import hpu_ops
from hpu_ops import composite

side = sys.argv.pop(1)
script = "linalg_steps.py"
if sys.argv[1:2] == ["--script"]:
    script = sys.argv[2]
    del sys.argv[1:3]
assert side in ("before", "after"), side
if side == "before":
    hpu_ops.FIXES = tuple(f for f in hpu_ops.FIXES if f is not composite.install_unfold)
print("UNFOLD_AB", side, "install_unfold in FIXES", composite.install_unfold in hpu_ops.FIXES, flush=True)
runpy.run_path(os.path.join(os.path.dirname(__file__), script), run_name="__main__")
