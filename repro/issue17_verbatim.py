"""hpu-ops #17's reproducer as filed (PT_HPU_LAZY_MODE=1): torch.quantile of a 4096 vector, hpu_ops and hpu_complex
enabled. Before the fix: FATAL 'Trying to resize tensor with non-existing TensorImpl'."""
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import hpu_complex
import hpu_ops
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

hpu_ops.enable()
hpu_complex.enable()
hpu = torch.device("hpu")
x = torch.randn(4096, device=hpu)
q = torch.quantile(x, 0.9)
htcore.mark_step()
print("QUANTILE", float(q.cpu()), "CPU", float(torch.quantile(x.cpu(), 0.9)))
hthpu.synchronize()
print("FALLBACKS", dict(metric_global("cpu_fallback").stats()).get("FallbackOps"))
print("CALLS", dict(hpu_ops._dispatch._STATE["calls"]))
