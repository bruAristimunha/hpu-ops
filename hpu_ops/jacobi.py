"""Block Jacobi on the card for the sizes past the Jacobi kernels' (``tpc.EIGH_MAXN``, ``tpc.SMALL_MAXN``): ``eigh``
of a symmetric matrix (two-sided) and ``svd`` (one-sided, on the Gram matrix of column pairs), each pair of
``BLOCK``-wide blocks diagonalized exactly by the ``2 * BLOCK = 32`` eigh kernel, all pairs of a round in one launch,
the rotations applied as batched matmuls. Round-robin pairing (``eigh._schedule``) over the blocks, a fixed number of
sweeps (``SWEEPS``), so nothing reads the device: no host sync, one shape per step.

Padding to a whole number of block pairs uses a value no eigenvalue (singular value) of the input reaches, ``2
||A||_F + 1``: the padded coordinates' subproblems are then block-diagonal with distinct eigenvalues, their rotations
never mix them with the input's, and the padded eigenvalues sort last (first, for singular values) and are dropped.
"""
import torch

BLOCK = 16  # 2 * BLOCK = tpc.EIGH_MAXN
# Block-cyclic sweeps on the running S (off(S) / ||S|| reaches float32's rounding level by sweep 5, n <= 128), then
# POLISH sweeps each from S = V^T A V recomputed with V re-orthogonalized (two Newton-Schulz steps), and a last
# re-orthogonalization and Rayleigh-Ritz S = V^T A V. Without the polish the rotations' rounding left V orthogonal to
# 5e-6 only and the eigenvalues 10-30x CPU float32's error (n = 64, 128, CPU float32 sub-solver); with it, 0.6-2x.
SWEEPS = 5
POLISH = 1


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


def block_eigh(A, eigh, compute_v=True, b=BLOCK, sweeps=SWEEPS, polish=POLISH):
    """``(w, V)`` of the symmetric ``A[..., n, n]`` (both triangles read): ``w`` ascending, the columns of ``V``
    orthonormal eigenvectors (None unless ``compute_v``). ``eigh`` takes ``[B, 2b, 2b]`` and returns ``(w, V)``."""
    batch, n = A.shape[:-2], A.shape[-1]
    S = A.reshape(-1, n, n)
    B = S.shape[0]
    k = 2 * -(-n // (2 * b))
    m = k * b
    if m > n:  # pad with a value past the spectrum: the padded eigenvalues sort last
        big = 2 * torch.linalg.vector_norm(S, dim=(-2, -1)) + 1
        pad = torch.diag_embed(big[:, None].expand(B, m - n))
        S = torch.cat([torch.cat([S, S.new_zeros(B, n, m - n)], -1), torch.cat([S.new_zeros(B, m - n, n), pad], -1)], -2)
    A0 = S
    V = torch.eye(m, dtype=S.dtype, device=S.device).expand(B, m, m)  # also for eigenvalues alone: the polish needs it
    for _ in range(sweeps):
        S, V = _sweep(S, V, k, eigh, b)
        S = (S + S.mT) / 2
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


def block_svd(A, eigh, compute_uv=True, b=BLOCK, sweeps=SWEEPS, polish=POLISH):
    """``(U, S, Vh)`` (thin) of ``A[..., m, n]``: one-sided block Jacobi on the columns (rows if m < n), each pair's
    Gram matrix diagonalized by ``eigh`` and the pair rotated by its eigenvectors; ``S`` descending, the column norms
    at the end, ``U = A V / S`` (a singular value below ``n eps S_max`` gives a zero column) re-orthogonalized. ``U``
    and ``Vh`` are None unless ``compute_uv``."""
    if A.shape[-2] < A.shape[-1]:
        U, S, Vh = block_svd(A.mT, eigh, compute_uv, b, sweeps, polish)
        return (None, S, None) if not compute_uv else (Vh.mT, S, U.mT)
    batch, m, n = A.shape[:-2], A.shape[-2], A.shape[-1]
    X = A.reshape(-1, m, n)
    B = X.shape[0]
    k = 2 * -(-n // (2 * b))
    p = k * b
    if p > n:  # pad with [[A, 0], [0, c I]], c past every singular value: the padded ones sort first
        c = 2 * torch.linalg.vector_norm(X, dim=(-2, -1)) + 1
        pad = torch.diag_embed(c[:, None].expand(B, p - n))
        X = torch.cat([torch.cat([X, X.new_zeros(B, m, p - n)], -1), torch.cat([X.new_zeros(B, p - n, n), pad], -1)], -2)
    rows = X.shape[-2]
    X0 = X
    V = torch.eye(p, dtype=X.dtype, device=X.device).expand(B, p, p)
    h, w = p // (2 * b), 2 * b
    for sweep in range(sweeps + polish):
        if sweep >= sweeps:  # polish: V re-orthogonalized, X = A V recomputed from the input
            V = _orthonormalize(V)
            X = X0 @ V
        for order in _pairs(k, X.device):
            perm = _block_perm(order, b)
            Xp = X.index_select(-1, perm).view(B, rows, h, w)
            G = torch.einsum("brix,briy->bixy", Xp, Xp).reshape(B * h, w, w)
            _, J = eigh(((G + G.mT) / 2).contiguous())
            J = J.reshape(B, h, w, w)
            inv = perm.argsort()
            X = torch.einsum("brix,bixy->briy", Xp, J).reshape(B, rows, p).index_select(-1, inv)
            V = torch.einsum("brix,bixy->briy", V.index_select(-1, perm).view(B, p, h, w), J).reshape(B, p, p)
            V = V.index_select(-1, inv)
    V = _orthonormalize(V)
    X = X0 @ V
    norms = torch.linalg.vector_norm(X, dim=-2)
    S, idx = norms.sort(-1, descending=True)
    S = S[..., p - n:]
    idx = idx[..., p - n:]
    if not compute_uv:
        return None, S.reshape(*batch, n), None
    V = V.gather(-1, idx[..., None, :].expand(B, p, n))[..., :n, :]
    U = X.gather(-1, idx[..., None, :].expand(B, rows, n))[..., :m, :]
    # a singular value at rounding level (below n eps S_max) gives a zero column: dividing by it would blow the
    # rounding up (the projector of _complete_orthogonal has exact zeros), and the Newton-Schulz steps keep zeros
    tiny = S <= S[..., :1] * (n * torch.finfo(S.dtype).eps)
    U = torch.where(tiny[..., None, :], 0.0, U / torch.where(tiny, 1.0, S)[..., None, :])
    # A V / S loses orthogonality where S is small (7e-6 for a 134 x 128 Gaussian, 1e-4 for a square one): a
    # first-order Cholesky-QR step U <- U (I - diag(E) / 2 - triu(E, 1)), E = U^T U - I, which, S sorted descending,
    # corrects each column by the larger singular values' columns only (a symmetric Newton-Schulz step let the
    # inaccurate columns rotate the accurate ones); a zero column stays zero
    E = U.mT @ U - torch.eye(n, dtype=U.dtype, device=U.device)
    U = U - U @ (torch.triu(E, 1) + torch.diag_embed(torch.diagonal(E, dim1=-2, dim2=-1)) / 2)
    return U.reshape(*batch, m, n), S.reshape(*batch, n), V.mT.reshape(*batch, n, n)
