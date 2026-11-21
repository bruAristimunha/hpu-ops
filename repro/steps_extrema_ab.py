"""tests/linalg_steps.py --baseline with hpu_ops.extrema off, on without the sort family, or on (an A/B in one
checkout, one arm per process; alternate the arms: repro/run_steps_extrema_ab.sh).
    PT_HPU_LAZY_MODE=0|1 python3 repro/steps_extrema_ab.py off|nosort|on [linalg_steps args]"""
import pathlib
import runpy
import sys

import hpu_ops
from hpu_ops import extrema

arm = sys.argv.pop(1)
if arm == "off":
    hpu_ops.FIXES = tuple(f for f in hpu_ops.FIXES if f is not extrema.install)
elif arm == "nosort":
    extrema._sortable = lambda x: False
else:
    import os

    os.environ["HPU_OPS_EXACT_SORT"] = "1"  # the measured "on" arm had the sort family on
print(f"ARM {arm}", flush=True)
sys.argv = [sys.argv[0], "--baseline"] + sys.argv[1:]
runpy.run_path(str(pathlib.Path(__file__).parents[1] / "tests" / "linalg_steps.py"), run_name="__main__")
