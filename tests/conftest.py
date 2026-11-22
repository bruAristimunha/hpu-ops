"""pytest collects the CPU suite only: ``tests/test_*.py``, which run without a card (the CPU stands in for HPU).
The Gaudi-only scripts in this directory (probes, benches, step timers, model checks) are run on the card by hand and
never collected, also if one is named like a test."""
import pytest
import torch

from hpu_ops import linalg, tpc

GAUDI_ONLY = ["probe_*.py", "bench_*.py", "check_*.py", "gaudi_*.py", "run_on_hpu.py", "linalg_on_hpu.py",
              "linalg_percall.py", "linalg_steps.py", "model_parity.py", "model_steps.py",
              "index_im2col_modes.py", "sdpa_checks.py", "tri_solve_*.py", "adversarial_*.py"]
collect_ignore_glob = GAUDI_ONLY


@pytest.fixture
def gj_ref(monkeypatch):
    """``linalg._GJ`` set to ``tpc.gj_solve_reference``."""
    monkeypatch.setattr(linalg, "_GJ", tpc.gj_solve_reference)


@pytest.fixture
def svd_hook(monkeypatch):
    """``linalg._SVD`` set to ``svd_reference`` on real tensors and torch's traceable SVD under fake tensors;
    collects ``(shape, compute_uv)`` per call."""
    calls = []

    def hook(A, compute_uv):
        calls.append((tuple(A.shape), compute_uv))
        if not linalg._is_fake(A):
            return tpc.svd_reference(A.detach(), compute_uv=compute_uv)
        if not compute_uv:
            return A.new_empty(0), torch.linalg.svdvals(A.detach()), A.new_empty(0)
        return torch.linalg.svd(A.detach(), full_matrices=False)

    monkeypatch.setattr(linalg, "_SVD", hook)
    return calls
