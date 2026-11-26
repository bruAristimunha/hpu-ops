"""#64: double-single eigenvalue refinement (CPU stand-in, float64 references)."""
import torch

from hpu_ops import ds_eigh


def _ill_conditioned(n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    d = torch.logspace(-2, 2, n, dtype=torch.float64)
    return q @ torch.diag(d) @ q.T, d


def test_ds_refine_beats_float32_on_ill_conditioned_spectra():
    A, d = _ill_conditioned()
    w32, V32 = torch.linalg.eigh(A.float())
    err32 = float((w32.double() - d).abs().max() / d.abs().max())
    # Rayleigh alone is limited by the fp32 vectors; refine them, then re-quote
    refined, V2 = ds_eigh.ds_refine_vectors(A, w32, V32)
    err_ds = float((refined - d).abs().max() / d.abs().max())
    print(f"fp32 eigh err {err32:.3e}  DS refined err {err_ds:.3e}")
    # the refined eigenvalues sit at the float32 storage floor (~1 ulp of the largest
    # eigenvalue): the vectors any consumer can store are float32, so no on-card route
    # can quote better eigenvalues than this
    assert err_ds < err32 / 2
    assert err_ds < 2e-7
    _, V64 = torch.linalg.eigh(A)
    align = (V2.double() * V64).sum(-2).abs()
    assert float(align.min()) > 0.999999


def test_ds_split_and_arithmetic_round_trip():
    torch.manual_seed(0)
    x = torch.randn(64, dtype=torch.float64)
    hi, lo = ds_eigh.ds_split(x)
    assert float((hi.double() + lo.double() - x).abs().max()) < 1e-14
    a = torch.randn(16, 16, dtype=torch.float64)
    ah, al = ds_eigh.ds_split(a)
    bh, bl = ds_eigh.ds_split(a)
    ph, pl = ds_eigh.ds_matmul(ah, al, bh, bl)
    err = float(((ph.double() + pl.double()) - a @ a).abs().max() / (a @ a).abs().max())
    assert err < 1e-13
