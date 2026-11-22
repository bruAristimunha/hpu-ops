"""Run a script with hpu_ops' indexing rewrites (install_index, install_mask_fill) left out of enable() (same process baseline for the fix).
    python3 repro/without_fix.py <script.py> [args...]"""
import runpy
import sys

import hpu_ops

hpu_ops.FIXES = tuple(f for f in hpu_ops.FIXES if f.__name__ not in ("install_index", "install_mask_fill"))
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
