"""On a Gaudi card: ``tpc._blocked``'s ops one at a time, each synced, to find the one whose launch fails, and the
synced ms of each (summed over the panels; the last of ``--reps`` runs printed).
    PT_HPU_LAZY_MODE=0 python3 tests/probe_blocked_ops.py N BATCH [--reps 3]"""
import json
import sys

import torch

from hpu_ops import tpc

tpc.install_kernel_path()
import habana_frameworks.torch.hpu as hthpu  # noqa: E402


TIMES = {}


def st(name, f):
    import time

    try:
        hthpu.synchronize()
        t = time.perf_counter()
        out = f()
        hthpu.synchronize()
        TIMES[name] = TIMES.get(name, 0) + (time.perf_counter() - t) * 1e3
        return out
    except Exception as e:  # noqa: BLE001
        print("FAIL", name, repr(e)[:150], flush=True)
        raise SystemExit(1)


n, b = int(sys.argv[1]), int(sys.argv[2])
reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 3
assert tpc.ready()
A0 = torch.randn(b, n, n).to("hpu").mT.contiguous()


def run(A):
  perm = tpc._arange(n, A, b)
  M = A
  Ds, infos, s = [], [], 0
  for c in tpc._panel_widths(n):
      e = s + c
      W = st("W", lambda: tpc._fresh(M[:, s:, s:e].contiguous()))
      rows, X, D, info, key = st("panel", lambda: tpc._panel(W))
      order = st("order", lambda: key.argsort(-1).to(torch.float32))
      perm = st("perm", lambda: tpc._cat([perm[:, :s], perm[:, s:].gather(-1, order.long())], -1))
      Ds.append(D)
      infos.append(st("info", lambda: torch.where(info > 0, info + s, float("inf"))))
      tail = st("tail", lambda: tpc._rows(M[:, s:], order))
      G = st("G", lambda: X @ tail[:, :c])
      G = st("Gcat", lambda: tpc._cat([G[..., :s], X, G[..., e:]], -1))
      rest = st("rest", lambda: tpc._cat([M[:, :s], tail[:, c:]], 1))
      C = st("C", lambda: rest[..., s:e])
      rest = st("upd", lambda: tpc._cat([rest[..., :s], torch.zeros_like(C), rest[..., e:]], -1) - C @ G)
      M = st("M", lambda: tpc._cat([rest[:, :s], G, rest[:, s:]], 1))

      s = e
  D = st("Dcat", lambda: torch.cat(Ds, -1))
  info = st("amin", lambda: torch.stack(infos, -1).amin(-1))
  info = st("isinf", lambda: torch.where(torch.isinf(info), 0.0, info))
  par = st("parity", lambda: tpc._parity(perm))
  Sx = st("S", lambda: torch.stack([par, info], -1))
  inv = st("argsort", lambda: perm.argsort(-1))
  X = st("gather", lambda: M.gather(-1, inv[:, None, :].expand(b, n, n)))


for _ in range(reps):
    TIMES.clear()
    run(A0)
print("OK all", json.dumps({k: round(v, 3) for k, v in TIMES.items()}), flush=True)
