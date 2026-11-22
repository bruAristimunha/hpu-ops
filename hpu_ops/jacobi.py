"""Block Jacobi ``eigh`` on the card past the eigh kernels' sizes: two-sided, each pair of ``b``-wide blocks
diagonalized exactly by an eigh kernel (``tpc.eigh_big`` on 2b = 128 blocks on HPU), all pairs of a round in one
launch, the rotations applied as batched matmuls; round-robin pairing (``eigh._schedule``) over the blocks.

Sweeps run until the off-diagonal part is below ``TOL`` of the matrix (Frobenius, the worst matrix of the batch:
one host read per sweep), at most ``max_sweeps(k)``; a traced call (no host reads) runs the cap. Measured (CPU
float64-exact sub-solver, n = 512 and 1024): 16-wide blocks and 5 fixed sweeps left 1e-3 / 1e-2 eigenvalue error
(code review of #51); 64-wide blocks reach CPU float32's error in 5 (n = 512) to 8 (n = 1024) sweeps.

Padding to a whole number of block pairs uses a value no eigenvalue of the input reaches, ``2 ||A||_inf + 1`` (a
row-sum bound: no squares, so no overflow for entries near 1e19): the padded coordinates' subproblems are then
block-diagonal with distinct eigenvalues, their rotations never mix them with the input's, and the padded eigenvalues
sort last and are dropped.
"""
import torch

BLOCK = 64  # 2 * BLOCK = tpc.BIG_EIGH_MAXN
TOL = 1e-6
# Block-cyclic sweeps on the running S (off(S) / ||S|| reaches float32's rounding level by sweep 5, n <= 128), then
# POLISH sweeps each from S = V^T A V recomputed with V re-orthogonalized (two Newton-Schulz steps), and a last
# re-orthogonalization and Rayleigh-Ritz S = V^T A V. Without the polish the rotations' rounding left V orthogonal to
# 5e-6 only and the eigenvalues 10-30x CPU float32's error (n = 64, 128, CPU float32 sub-solver); with it, 0.6-2x.
SWEEPS = 5  # traced calls; eager ones stop by TOL, at most max_sweeps(k)
POLISH = 1


def max_sweeps(k):
    """The sweep cap for ``k`` blocks: cyclic Jacobi converges quadratically once close; 6 plus a few per doubling."""
    return 6 + 2 * max(1, (k - 1).bit_length())


def _off(S):
    """The worst matrix's off-diagonal Frobenius norm relative to its own (scaled first: no overflow)."""
    S = S / S.abs().amax((-2, -1), keepdim=True).clamp_min(torch.finfo(S.dtype).tiny)
    total = torch.linalg.vector_norm(S, dim=(-2, -1))
    diag = torch.linalg.vector_norm(torch.diagonal(S, dim1=-2, dim2=-1), dim=-1)
    return ((total.square() - diag.square()).clamp_min(0).sqrt() / total.clamp_min(torch.finfo(S.dtype).tiny)).amax()


def _pairs(k, device):
    """The round-robin schedule of ``k`` (even) blocks: ``[k - 1, k]`` block orders, the pairs of a round adjacent."""
    from .eigh import _schedule

    return _schedule(k, device).flatten(-2)


def _block_perm(order, b):
    """The coordinate permutation that lays out the blocks in ``order`` (``[k]``), ``b`` wide each."""
    return (order[:, None] * b + torch.arange(b, device=order.device)).flatten()


def _rotate_pairs(S, V, perm, eigh, b):
    """One round: ``S`` (``[B, m, m]``, symmetric) and ``V`` (``[B, m, m]``, its columns rotated too) in the layout
    ``perm`` (the pairs adjacent), each pair's ``2b x 2b`` diagonal block diagonalized by ``eigh``, then ``S <- J^T S
    J``, ``V <- V J`` with ``J`` the block-diagonal rotation. ``V`` may be None."""
    B, m = S.shape[0], S.shape[-1]
    h, w = m // (2 * b), 2 * b
    Sp = S.index_select(-2, perm).index_select(-1, perm)
    blocks = torch.diagonal(Sp.view(B, h, w, h, w), dim1=1, dim2=3).permute(0, 3, 1, 2).reshape(B * h, w, w)
    _, J = eigh(blocks.contiguous())
    J = J.reshape(B, h, w, w)
    Sp = torch.einsum("bixy,bixjz->biyjz", J, Sp.view(B, h, w, h, w))
    Sp = torch.einsum("biyjz,bjzw->biyjw", Sp, J).reshape(B, m, m)
    inv = perm.argsort()
    S = Sp.index_select(-2, inv).index_select(-1, inv)
    if V is not None:
        Vp = V.index_select(-1, perm).view(B, -1, h, w)
        V = torch.einsum("brjz,bjzw->brjw", Vp, J).reshape(B, -1, m).index_select(-1, inv)
    return S, V


def _orthonormalize(V, steps=2):
    """Two Newton-Schulz steps ``V <- V (3 I - V^T V) / 2``: an orthogonal ``V`` to rounding, from 1e-5 away."""
    eye = torch.eye(V.shape[-1], dtype=V.dtype, device=V.device)
    for _ in range(steps):
        V = V @ (1.5 * eye - 0.5 * (V.mT @ V))
    return V


def _sweep(S, V, k, eigh, b):
    for order in _pairs(k, S.device):
        S, V = _rotate_pairs(S, V, _block_perm(order, b), eigh, b)
    return S, V


def block_eigh(A, eigh, compute_v=True, b=BLOCK, sweeps=None, polish=POLISH, tol=TOL):
    """``(w, V)`` of the symmetric ``A[..., n, n]`` (both triangles read): ``w`` ascending, the columns of ``V``
    orthonormal eigenvectors (None unless ``compute_v``). ``eigh`` takes ``[B, 2b, 2b]`` and returns ``(w, V)``."""
    batch, n = A.shape[:-2], A.shape[-1]
    S = A.reshape(-1, n, n)
    B = S.shape[0]
    k = 2 * -(-n // (2 * b))
    m = k * b
    if m > n:  # pad with a value past the spectrum: the padded eigenvalues sort last
        norm = S.abs().sum(-1).amax(-1)
        big = torch.where(norm > 0, (2 * norm).clamp_max(torch.finfo(S.dtype).max), 1.0)
        pad = torch.diag_embed(big[:, None].expand(B, m - n))
        S = torch.cat([torch.cat([S, S.new_zeros(B, n, m - n)], -1), torch.cat([S.new_zeros(B, m - n, n), pad], -1)], -2)
    A0 = S
    V = torch.eye(m, dtype=S.dtype, device=S.device).expand(B, m, m)  # also for eigenvalues alone: the polish needs it
    from .linalg import _strict, _tracing

    traced = _tracing(S)
    cap = (SWEEPS if traced else max_sweeps(k)) if sweeps is None else sweeps
    for _ in range(cap):
        S, V = _sweep(S, V, k, eigh, b)
        S = (S + S.mT) / 2
        if sweeps is None and not traced:
            off = _off(S)
            with _strict.host_read("linalg_eigh: block Jacobi convergence", off):
                if float(off) <= tol:
                    break
    for _ in range(polish):
        V = _orthonormalize(V)
        S = V.mT @ (A0 @ V)
        S, V = _sweep((S + S.mT) / 2, V, k, eigh, b)
    V = _orthonormalize(V)
    S = V.mT @ (A0 @ V)
    w, idx = torch.diagonal(S, dim1=-2, dim2=-1).sort(-1)
    if S.device.type == "hpu":
        from . import tpc

        tpc._flush()  # lazy mode: the sort and the gather in their own graph (see Eigh.backward)
    if not compute_v:
        V = None
    w = w[..., :n]
    if V is None:
        return w.reshape(*batch, n), None
    V = V.gather(-1, idx[..., None, :].expand(B, m, m))[..., :n, :n]
    return w.reshape(*batch, n), V.reshape(*batch, n, n)
