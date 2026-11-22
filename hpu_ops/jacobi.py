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
        big = 2 * S.abs().sum(-1).amax(-1) + 1
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


SVD_BLOCK = 32  # 2 * SVD_BLOCK = tpc.BIG_SVD_MAXN (64)


def _rotate_svd_pairs(X, V, perm, tsqr_r, svd_big, b):
    """One Gram-free block Jacobi SVD round: ``X`` (``[B, p, p]``) and ``V`` (``[B, p, p]``) in column order
    ``perm``, each ``p x 2b`` panel reduced to ``Rp`` (``[B * h, 2b, 2b]``) by ``tsqr_r`` without forming ``X^T X``,
    solved by ``svd_big(Rp) -> (_, _, V_p_t)``, then ``X <- X J``, ``V <- V J`` directly."""
    B, p = X.shape[0], X.shape[-1]
    h, w = p // (2 * b), 2 * b
    Xp = X.index_select(-1, perm).view(B, p, h, w)
    Rp = tsqr_r(Xp.permute(0, 2, 1, 3).reshape(B * h, p, w).contiguous())
    _, _, V_p_t = svd_big(Rp.contiguous())
    J = V_p_t.mT.reshape(B, h, w, w)
    inv = perm.argsort()
    X = torch.einsum("brjz,bjzw->brjw", Xp, J).reshape(B, p, p).index_select(-1, inv)
    Vp = V.index_select(-1, perm).view(B, p, h, w)
    V = torch.einsum("brjz,bjzw->brjw", Vp, J).reshape(B, p, p).index_select(-1, inv)
    return X, V


def _sweep_svd(X, V, k, tsqr_r, svd_big, b):
    for order in _pairs(k, X.device):
        X, V = _rotate_svd_pairs(X, V, _block_perm(order, b), tsqr_r, svd_big, b)
    return X, V


def _cpu_svd_big(X):
    with torch._C._AutoDispatchBelowAutograd():
        return torch.ops.aten._linalg_svd.default(X, False, True)


def _cpu_svdvals_big(X):
    with torch._C._AutoDispatchBelowAutograd():
        return torch.ops.aten._linalg_svd.default(X, False, False)[1]


def _svd_solvers(device, svd_big, svdvals_big, tsqr_r, tsqr_qr, block_qr):
    if device.type == "hpu":
        from . import tpc

        return (tpc.jacobi_svd_big if svd_big is None else svd_big,
                tpc.jacobi_svdvals_big if svdvals_big is None else svdvals_big,
                tpc.tsqr_r if tsqr_r is None else tsqr_r,
                tpc.tsqr_qr if tsqr_qr is None else tsqr_qr,
                tpc.block_qr if block_qr is None else block_qr,
                tpc.BIG_SVD_MAXN, tpc.BIG_SVD_MAXM, tpc.BIG_SVDVALS_MAXN, tpc.BIG_SVDVALS_MAXM)
    return (_cpu_svd_big if svd_big is None else svd_big,
            _cpu_svdvals_big if svdvals_big is None else svdvals_big,
            (lambda X: torch.linalg.qr(X, mode="r").R) if tsqr_r is None else tsqr_r,
            (lambda X: torch.linalg.qr(X, mode="reduced")) if tsqr_qr is None else tsqr_qr,
            (lambda X: torch.linalg.qr(X, mode="reduced")) if block_qr is None else block_qr,
            64, 128, 128, 128)


def block_svd_gram_free(A, compute_uv=True, full_matrices=False, sweeps=None, polish=POLISH, b=SVD_BLOCK,
                        svd_big=None, svdvals_big=None, tsqr_r=None, tsqr_qr=None, block_qr=None):
    """Gram-free on-card SVD ``(U, S, Vh)`` for ``A[..., m, n]`` past ``tpc.SMALL_MAXN``.

    Tier 1 (``min(m, n) <= 64``): direct ``jacobi_svd_big`` / ``jacobi_svdvals_big`` (with ``tsqr`` pre-reduction when
    the tall side exceeds ``128``). Tier 2 (``min(m, n) > 64``): ``block_qr`` pre-reduction for ``m > n``, then
    Gram-free block one-sided Jacobi directly on column panels via ``tsqr_r`` + ``jacobi_svd_big`` (no ``X^T X``,
    fixed sweep schedule with zero host reads, ``X = X0 @ V`` polish sweep, and ``block_qr`` cleanup on ``U``).
    """
    batch, m, n = A.shape[:-2], A.shape[-2], A.shape[-1]
    k_min = min(m, n)
    full = full_matrices and m != n
    if A.numel() == 0:
        S = A.new_empty(*batch, k_min)
        if not compute_uv:
            return A.new_empty(0), S, A.new_empty(0)
        return A.new_empty(*batch, m, m if full else k_min), S, A.new_empty(*batch, n if full else k_min, n)

    if m < n:
        U_t, S, Vh_t = block_svd_gram_free(
            A.mT, compute_uv=compute_uv, full_matrices=full_matrices, sweeps=sweeps, polish=polish, b=b,
            svd_big=svd_big, svdvals_big=svdvals_big, tsqr_r=tsqr_r, tsqr_qr=tsqr_qr, block_qr=block_qr)
        if not compute_uv:
            return A.new_empty(0), S, A.new_empty(0)
        return Vh_t.mT, S, U_t.mT

    _svd_big, _svdvals_big, _tsqr_r, _tsqr_qr, _block_qr, maxn, maxm, _vals_maxn, vals_maxm = _svd_solvers(
        A.device, svd_big, svdvals_big, tsqr_r, tsqr_qr, block_qr)

    sigma = A.abs().amax(dim=(-2, -1), keepdim=True)
    safe_sigma = torch.where(sigma > 0, sigma, torch.ones_like(sigma))
    As = A / safe_sigma

    def rescale(s):
        return torch.where(sigma.squeeze(-1) == 0, 0.0, s * safe_sigma.squeeze(-1))

    if n <= maxn:
        if not compute_uv:
            S = _svdvals_big(As if m <= vals_maxm else _tsqr_r(As))
            return A.new_empty(0), rescale(S), A.new_empty(0)
        if m <= maxm:
            U, S, Vh = _svd_big(As)
        else:
            Q, R = _tsqr_qr(As)
            Ur, S, Vh = _svd_big(R)
            U = Q @ Ur
        S = rescale(S)
        if full:
            from .linalg import _complete_orthogonal

            U = _complete_orthogonal(U, S)
        return U, S, Vh

    if m > n:
        Q0, R0 = _block_qr(As)
    else:
        Q0, R0 = None, As

    X0 = R0.reshape(-1, n, n)
    B = X0.shape[0]
    k = 2 * -(-n // (2 * b))
    p = k * b
    if p > n:
        max_sum = torch.maximum(X0.abs().sum(-1).amax(-1), X0.abs().sum(-2).amax(-1))
        c_pad = 2 * max_sum + 1
        pad = torch.diag_embed(c_pad[:, None].expand(B, p - n))
        X0 = torch.cat([torch.cat([X0, X0.new_zeros(B, n, p - n)], -1),
                        torch.cat([X0.new_zeros(B, p - n, n), pad], -1)], -2)

    X = X0.clone()
    V = torch.eye(p, dtype=X0.dtype, device=X0.device).expand(B, p, p).clone()
    sw = (8 if n <= 128 else 12 if n <= 256 else 14) if sweeps is None else sweeps
    for _ in range(sw):
        X, V = _sweep_svd(X, V, k, _tsqr_r, _svd_big, b)
    for _ in range(polish):
        V = _orthonormalize(V)
        X = X0 @ V
        X, V = _sweep_svd(X, V, k, _tsqr_r, _svd_big, b)
    V = _orthonormalize(V)

    if X.device.type == "hpu":
        from . import tpc

        tpc._flush()
    S_all = X.norm(dim=-2)
    S_all, idx = S_all.sort(-1, descending=True)
    X = X.gather(-1, idx[..., None, :].expand_as(X))[..., :n, p - n:]
    V = V.gather(-1, idx[..., None, :].expand_as(V))[..., :n, p - n:]
    S_norm = S_all[..., p - n:]
    S = rescale(S_norm.reshape(*batch, n))
    if not compute_uv:
        return A.new_empty(0), S, A.new_empty(0)

    tiny = torch.finfo(X.dtype).tiny
    U_raw = torch.where(S_norm.unsqueeze(-2) > 0, X / S_norm.clamp_min(tiny).unsqueeze(-2), 0.0)
    U_r, Ru = _block_qr(U_raw.contiguous())
    d = torch.diagonal(Ru, dim1=-2, dim2=-1)
    U_r = (U_r * torch.where(d < 0, -1.0, 1.0).unsqueeze(-2)).reshape(*batch, n, n)
    U = Q0 @ U_r if Q0 is not None else U_r
    Vh = V.mT.reshape(*batch, n, n)
    if full:
        from .linalg import _complete_orthogonal

        U = _complete_orthogonal(U, S)
    return U, S, Vh
