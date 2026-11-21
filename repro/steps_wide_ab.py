"""tests/linalg_steps.py --baseline with hpu_ops.wide's copy checks on or off (one arm per process; alternate them),
and how many eager copy_ calls to the card the checks see.
    PT_HPU_LAZY_MODE=0 python3 repro/steps_wide_ab.py on|off [linalg_steps args]"""
import atexit
import pathlib
import runpy
import sys

import hpu_ops
from hpu_ops import wide

arm = sys.argv.pop(1)
if arm == "off":
    hpu_ops.FIXES = tuple(f for f in hpu_ops.FIXES if f is not wide.install_copy_checks)
else:
    seen = [0]
    checks = wide.copy_checks

    def counted(self, src, device_type):
        seen[0] += 1
        return checks(self, src, device_type)

    wide.copy_checks = counted
    atexit.register(lambda: print(f"COPIES checked {seen[0]}", flush=True))
print(f"ARM {arm}", flush=True)
sys.argv = [sys.argv[0], "--baseline"] + sys.argv[1:]
runpy.run_path(str(pathlib.Path(__file__).parents[1] / "tests" / "linalg_steps.py"), run_name="__main__")
