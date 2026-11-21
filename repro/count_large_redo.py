"""tests/linalg_steps.py with hpu_ops.linalg's large-route check spied on: per call of ``_large_untrusted``, the
matrices' shape and condition estimate, and how many were redone on the CPU. Eager.
    PT_HPU_LAZY_MODE=0 python3 repro/count_large_redo.py --models EEGSPDNet"""
import atexit
import pathlib
import runpy
import sys

import torch

from hpu_ops import linalg

calls = []
real = linalg._large_untrusted


def spy(A, Ainv):
    bad = real(A, Ainv)
    c2 = (A * A).sum((-2, -1)) * (Ainv * Ainv).sum((-2, -1))
    calls.append((tuple(A.shape), float(c2.sqrt().max()), int(bad.sum())))
    return bad


linalg._large_untrusted = spy
atexit.register(lambda: print(f"REDO calls={len(calls)} redone={sum(c[2] for c in calls)} "
                              f"shapes={sorted(set(c[0] for c in calls))} cond_max={max((c[1] for c in calls), default=0):.3g}"
                              f" cond_min={min((c[1] for c in calls), default=0):.3g}", flush=True))
sys.argv = [sys.argv[0], "--baseline"] + sys.argv[1:]
runpy.run_path(str(pathlib.Path(__file__).parents[1] / "tests" / "linalg_steps.py"), run_name="__main__")
