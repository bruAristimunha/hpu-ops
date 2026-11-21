"""Where eager and lazy copy_ / _to_copy resolve on Habana's bridge (no card op): PT_HPU_LAZY_MODE=0|1 python ..."""
import habana_frameworks.torch.core  # noqa: F401
import torch

for op in ("aten::copy_", "aten::_to_copy", "aten::_copy_from", "aten::_copy_from_and_resize", "aten::histc",
           "aten::histc.out", "aten::bucketize.Tensor"):
    table = torch._C._dispatch_dump_table(op)
    print(op, [line for line in table.splitlines() if line.startswith(("HPU", "AutogradHPU", "CompositeExplicit"))])
