"""pytest collects the CPU suite only: ``tests/test_*.py``, which run without a card (the CPU stands in for HPU).
The Gaudi-only scripts in this directory (probes, benches, step timers, model checks) are run on the card by hand and
never collected, also if one is named like a test."""
GAUDI_ONLY = ["probe_*.py", "bench_*.py", "check_*.py", "gaudi_*.py", "run_on_hpu.py", "linalg_on_hpu.py",
              "linalg_percall.py", "linalg_steps.py", "model_parity.py", "model_steps.py", "index_steps.py",
              "index_im2col_modes.py", "sdpa_checks.py", "tri_solve_*.py", "unfold_ab.py", "adversarial_*.py"]
collect_ignore_glob = GAUDI_ONLY
