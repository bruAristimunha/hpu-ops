"""On a Gaudi card: the LUT-free M3B-A TPC kernels (``jacobi_svd_big``, ``jacobi_svdvals_big``, ``hh_qr``,
``tsqr_r``, ``tsqr_qr``, ``block_qr``) against their float32 FTZ-faithful CPU replicas in ``kernel_replica``.
    PT_HPU_LAZY_MODE=0|1 python3 tests/check_svd_big_replica.py

Checks both numerical/replica parity (including ``1e-30`` graded, ``1e-25``, ``1e25``, zeros, rank-deficient,
``64x64``, ``128x64``, ``128x128``, ``256x64``) and strict on-card execution: ``FallbackOps == 0`` and
``hpu_ops route == 0`` on every case.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import warnings

import numpy as np
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

import hpu_ops
import kernel_replica as kr

warnings.filterwarnings("ignore")
hpu_ops.enable()
from hpu_ops import linalg as ops_linalg  # noqa: E402
from hpu_ops import tpc  # noqa: E402

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
CPU_ROUTE = []


def _install_route_spy():
    for name in ("_on_cpu", "_out_on_cpu"):
        inner = getattr(ops_linalg, name)

        def counted(*args, _inner=inner, _name=name, **kwargs):
            CPU_ROUTE.append(_name)
            return _inner(*args, **kwargs)

        setattr(ops_linalg, name, counted)


def sync():
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.hpu as hthpu

    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def _fb_ops():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


@contextlib.contextmanager
def watch():
    rec = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f0, r0 = _fb_ops(), len(CPU_ROUTE)
        try:
            yield rec
        finally:
            rec["fallback"] = {k: v - f0.get(k, 0) for k, v in _fb_ops().items() if v - f0.get(k, 0)}
            rec["cpu_route"] = len(CPU_ROUTE) - r0
            rec["cpu_warn"] = [str(w.message) for w in caught if "runs on the CPU" in str(w.message)]


def _graded(n, lo_exp):
    vals = torch.tensor([10.0 ** (lo_exp * i / max(1, n - 1)) for i in range(n)], dtype=torch.float32)
    return torch.diag(vals)


def _rank_def(m, n, r, g):
    return (torch.randn(m, r, generator=g) @ torch.randn(r, n, generator=g)) / max(1, r)


def svd_cases():
    g = torch.Generator().manual_seed(0)
    yield "graded_1e-30_64", _graded(64, -30)
    yield "tiny_1e-25_64", torch.randn(64, 64, generator=g) * 1e-25
    yield "huge_1e25_64", torch.randn(64, 64, generator=g) * 1e25
    yield "zeros_64x64", torch.zeros(64, 64)
    yield "rank5_64x64", _rank_def(64, 64, 5, g)
    yield "randn_40x40", torch.randn(40, 40, generator=g)
    yield "randn_64x64", torch.randn(64, 64, generator=g)
    yield "randn_128x64", torch.randn(128, 64, generator=g)
    yield "randn_64x33", torch.randn(64, 33, generator=g)
    yield "randn_33x64", torch.randn(33, 64, generator=g)


def svdvals_only_cases():
    g = torch.Generator().manual_seed(1)
    yield "graded_1e-30_128", _graded(128, -30)
    yield "tiny_1e-25_128", torch.randn(128, 128, generator=g) * 1e-25
    yield "huge_1e25_128", torch.randn(128, 128, generator=g) * 1e25
    yield "zeros_128x128", torch.zeros(128, 128)
    yield "rank10_128x128", _rank_def(128, 128, 10, g)
    yield "randn_128x128", torch.randn(128, 128, generator=g)


def qr_cases():
    g = torch.Generator().manual_seed(2)
    yield "graded_1e-30_64", _graded(64, -30)
    yield "tiny_1e-25_64", torch.randn(64, 64, generator=g) * 1e-25
    yield "huge_1e25_64", torch.randn(64, 64, generator=g) * 1e25
    yield "zeros_64x64", torch.zeros(64, 64)
    yield "rank5_64x64", _rank_def(64, 64, 5, g)
    yield "randn_64x64", torch.randn(64, 64, generator=g)
    yield "randn_128x64", torch.randn(128, 64, generator=g)
    yield "randn_256x64", torch.randn(256, 64, generator=g)


def tsqr_cases():
    g = torch.Generator().manual_seed(3)
    yield "graded_1e-30_256x64", torch.cat([_graded(64, -30), torch.zeros(192, 64)], dim=0)
    yield "tiny_1e-25_256x64", torch.randn(256, 64, generator=g) * 1e-25
    yield "huge_1e25_256x64", torch.randn(256, 64, generator=g) * 1e25
    yield "zeros_256x64", torch.zeros(256, 64)
    yield "rank5_256x64", _rank_def(256, 64, 5, g)
    yield "randn_256x64", torch.randn(256, 64, generator=g)
    yield "randn_320x48", torch.randn(320, 48, generator=g)


def block_qr_cases():
    g = torch.Generator().manual_seed(4)
    yield "randn_128x128", torch.randn(128, 128, generator=g)
    yield "randn_256x128", torch.randn(256, 128, generator=g)
    yield "zeros_128x128", torch.zeros(128, 128)
    yield "rank10_128x128", _rank_def(128, 128, 10, g)
    yield "tiny_1e-25_128x128", torch.randn(128, 128, generator=g) * 1e-25
    yield "huge_1e25_128x128", torch.randn(128, 128, generator=g) * 1e25


def _check_svd_case(name, a, op_name):
    a_np = a.numpy()
    scale = float(np.abs(a_np).max()) or 1.0
    ref_s = torch.linalg.svdvals(a.double()).numpy()
    floor = (1e-6 * float(ref_s.max())) if ref_s.size else 1.0
    safe_denom = np.maximum(ref_s, floor if floor > 0.0 else 1.0)
    ok = True
    modes = ("svdvals",) if op_name == "jacobi_svdvals_big" else ("thin", "svdvals")
    for mode in modes:
        compute_uv = mode != "svdvals"
        _, want_s, _ = kr.jacobi_svd_big(a_np, compute_uv=compute_uv, full_matrices=False)
        h = a[None].to("hpu")
        with watch() as rec:
            if not compute_uv:
                got = (tpc.jacobi_svdvals_big(h, sort=True),)
            else:
                got = tpc.jacobi_svd_big(h, sort=True)
            sync()
        got_np = [g[0].cpu().numpy() for g in got]
        got_s = got_np[1] if compute_uv else got_np[0]
        s_diff = float(np.abs(got_s - want_s).max()) if got_s.size else 0.0
        sv_rel_err = float(np.max(np.abs(got_s.astype(np.float64) - ref_s) / safe_denom)) if got_s.size else 0.0
        row = dict(op=op_name, case=name, mode=mode, lazy=LAZY, S_diff=s_diff, sv_rel_err=sv_rel_err,
                   fallback=rec["fallback"], cpu_route=rec["cpu_route"])
        case_ok = (not rec["fallback"]) and (rec["cpu_route"] == 0) and (not rec["cpu_warn"])
        case_ok &= s_diff <= 2e-5 * scale
        if compute_uv:
            U, Sg, Vh = (x.astype(np.float64) for x in got_np)
            k = Sg.shape[0]
            row["recon"] = float(np.abs((U[:, :k] * Sg) @ Vh[:k] - a_np).max() / scale)
            row["orth"] = float(max(np.abs(U.T @ U - np.eye(U.shape[1])).max(),
                                    np.abs(Vh @ Vh.T - np.eye(Vh.shape[0])).max()))
            case_ok &= row["recon"] < 1e-5 and row["orth"] < 1e-5
        row["ok"] = case_ok
        ok &= case_ok
        print("RESULT " + json.dumps(row), flush=True)
    return ok


def _check_qr_case(name, a, op_name):
    a_np = a.numpy()
    m, n = a_np.shape
    k = min(m, n)
    scale = float(np.abs(a_np).max()) or 1.0
    ref_s = torch.linalg.svdvals(a.double()).numpy()
    ok = True
    h = a[None].to("hpu")
    with watch() as rec:
        if op_name == "hh_qr":
            Q_h, R_h = tpc.hh_qr(h)
            got_q, got_r = Q_h[0], R_h[0]
        elif op_name == "tsqr_r":
            got_q, got_r = None, tpc.tsqr_r(h)[0]
        elif op_name == "tsqr_qr":
            Q_h, R_h = tpc.tsqr_qr(h)
            got_q, got_r = Q_h[0], R_h[0]
        elif op_name == "block_qr":
            Q_h, R_h = tpc.block_qr(h)
            got_q, got_r = Q_h[0], R_h[0]
        else:
            raise ValueError(op_name)
        sync()
    r_np = got_r.cpu().numpy().astype(np.float64)
    if op_name == "hh_qr":
        _, r_rep = kr.hh_qr_kernel(a_np, mode="qr")
    elif op_name == "tsqr_r":
        r_rep = kr.tsqr_r_replica(a_np)
    elif op_name == "tsqr_qr":
        _, r_rep = kr.tsqr_qr_replica(a_np)
    else:
        _, r_rep = kr.block_qr_replica(a_np)
    r_sv = np.linalg.svd(r_np, compute_uv=False)
    rep_sv = np.linalg.svd(r_rep.astype(np.float64), compute_uv=False)
    sv_diff_rep = float(np.abs(r_sv - rep_sv).max()) if r_sv.size else 0.0
    sv_diff_ref = float(np.abs(r_sv - ref_s).max() / scale) if r_sv.size else 0.0
    tril_err = float(np.abs(np.tril(r_np, -1)).max() / scale) if r_np.size else 0.0
    row = dict(op=op_name, case=name, lazy=LAZY, sv_diff_rep=sv_diff_rep, sv_err=sv_diff_ref,
               tril_err=tril_err, fallback=rec["fallback"], cpu_route=rec["cpu_route"])
    case_ok = (not rec["fallback"]) and (rec["cpu_route"] == 0) and (not rec["cpu_warn"])
    case_ok &= sv_diff_rep <= 2e-5 * scale and sv_diff_ref < 1e-5 and tril_err < 1e-5
    if got_q is not None:
        q_np = got_q.cpu().numpy().astype(np.float64)
        row["recon"] = float(np.abs(q_np @ r_np - a_np).max() / scale)
        row["orth"] = float(np.abs(q_np.T @ q_np - np.eye(k)).max())
        case_ok &= row["recon"] < 1e-5 and row["orth"] < 1e-5
    row["ok"] = case_ok
    ok &= case_ok
    print("RESULT " + json.dumps(row), flush=True)
    return ok


def main():
    _install_route_spy()
    assert tpc.ready(), "hpu_ops.tpc.ready() is False"
    ok = True
    for name, a in svd_cases():
        ok &= _check_svd_case(name, a, "jacobi_svd_big")
    for name, a in svdvals_only_cases():
        ok &= _check_svd_case(name, a, "jacobi_svdvals_big")
    for name, a in qr_cases():
        ok &= _check_qr_case(name, a, "hh_qr")
    for name, a in tsqr_cases():
        ok &= _check_qr_case(name, a, "tsqr_r")
        ok &= _check_qr_case(name, a, "tsqr_qr")
    for name, a in block_qr_cases():
        ok &= _check_qr_case(name, a, "block_qr")
    print("SUMMARY " + ("ok" if ok else "MISMATCH"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
