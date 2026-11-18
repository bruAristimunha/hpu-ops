"""CUDA's autocast op lists for Habana's bridge. Before Python starts: ``eval "$(python -m hpu_ops.autocast)"``.

Under ``torch.autocast("hpu", torch.bfloat16)``, Habana's default lists (``hpu_ops/autocast_helpers.h``, the same in
bridges 1.21 to 1.24) run layer_norm, group_norm, batch_norm, instance_norm, softmax, log_softmax, mean, mul and
dropout in bfloat16. CUDA runs the norms and softmax in float32, and the others in their input dtype. Habana reads
its two lists once, when its library loads, from the files that ``PT_HPU_AUTOCAST_LOWER_PRECISION_OPS_LIST`` and
``PT_HPU_AUTOCAST_FP32_OPS_LIST`` name. These files hold torch's CUDA lists (``ATen/autocast_mode.h``): the lower
precision one, and the float32 one with the ops that CUDA runs with a float32 output dtype (softmax, sum, ...).
"""
import pathlib

LISTS = {"PT_HPU_AUTOCAST_LOWER_PRECISION_OPS_LIST": "autocast_cuda_lower.txt",
         "PT_HPU_AUTOCAST_FP32_OPS_LIST": "autocast_cuda_fp32.txt"}


def exports():
    """The shell lines that point Habana at the CUDA lists."""
    here = pathlib.Path(__file__).resolve().parent
    return [f"export {name}={here / file}" for name, file in LISTS.items()]


if __name__ == "__main__":
    print("\n".join(exports()))
