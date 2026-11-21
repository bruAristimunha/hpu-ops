"""Gaudi probe (issue #21): how NaN passes through the reductions of the logsumexp rewrite, and what Habana's bincount
does with negative input, eager or lazy (PT_HPU_LAZY_MODE). torch on the CPU: NaN propagates through amax, max,
log_softmax, logsumexp; bincount of a negative value raises."""
import habana_frameworks.torch.core as htcore
import torch

hpu = torch.device("hpu")
nan, inf = float("nan"), float("inf")


def show(name, fn, x):
    try:
        out = fn(x.to(hpu))
        htcore.mark_step()
        print(f"{name:34s} hpu {out.cpu().tolist()}  cpu {fn(x).tolist()}", flush=True)
    except RuntimeError as e:
        print(f"{name:34s} hpu raises {str(e)[:80]}  cpu {fn(x).tolist()}", flush=True)


def main():
    x = torch.tensor([[nan, 1.0, 2.0], [2.0, 1.0, nan], [1.0, nan, inf], [1.0, 2.0, 3.0]])
    show("amax(-1)", lambda t: t.amax(-1), x)
    show("max(-1).values", lambda t: t.max(-1).values, x)
    show("log_softmax(-1)", lambda t: torch.log_softmax(t, -1), x)
    show("amax(log_softmax)", lambda t: torch.log_softmax(t, -1).amax(-1), x)
    show("logsumexp native", lambda t: torch.logsumexp(t, -1), x)
    show("isnan.any", lambda t: t.isnan().any(-1), x)
    show("nan_to_num(nan=0,keep inf)", lambda t: t.nan_to_num(0.0, inf, -inf), torch.tensor([nan, inf, -inf, 1.0]))
    show("sum(-1)", lambda t: t.sum(-1), x)
    try:
        neg = torch.tensor([0, 1, -1, 2], device=hpu)
        out = torch.bincount(neg)
        htcore.mark_step()
        print("bincount([0, 1, -1, 2]) hpu", out.cpu().tolist(), "(cpu raises)", flush=True)
    except RuntimeError as e:
        print("bincount negative hpu raises", str(e)[:80], flush=True)


if __name__ == "__main__":
    main()
