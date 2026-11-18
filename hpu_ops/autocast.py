"""CUDA's autocast op lists for Habana's bridge. Before Python starts: ``eval "$(python -m hpu_ops.autocast)"``.

Under ``torch.autocast("hpu", torch.bfloat16)``, Habana's default lists (``hpu_ops/autocast_helpers.h``, the same in
bridges 1.21 to 1.24) run layer_norm, group_norm, batch_norm, instance_norm, softmax, log_softmax, mean, mul and
dropout in bfloat16, even from float32 inputs. CUDA runs the norms and softmax in float32, and the others in their
input dtype. Habana reads its two lists once, when its library loads, from the files that
``PT_HPU_AUTOCAST_LOWER_PRECISION_OPS_LIST`` and ``PT_HPU_AUTOCAST_FP32_OPS_LIST`` name.

These files hold torch's CUDA lists (``ATen/autocast_mode.h``). The lower precision one is CUDA's. The float32 one has
CUDA's float32 ops; the ops CUDA runs with a float32 output dtype (softmax, sum, norm, ...); the ops CUDA promotes to
the widest input dtype, which Habana's fixed promote list lacks (float32 is that dtype for mixed inputs, and makes
mixed inputs run at all); and binary_cross_entropy, which CUDA refuses under autocast.
"""
import pathlib
import shlex

FILES = {"PT_HPU_AUTOCAST_LOWER_PRECISION_OPS_LIST": "autocast_cuda_lower.txt",
         "PT_HPU_AUTOCAST_FP32_OPS_LIST": "autocast_cuda_fp32.txt"}
# Habana's promote list, fixed in its source (hpu_ops/autocast_helpers.h)
HABANA_PROMOTE = {"add", "addcmul", "addcdiv", "cat", "div", "exp", "mul", "pow", "sub", "iadd", "truediv", "stack"}
REFUSED_BY_CUDA = {"binary_cross_entropy"}


def paths():
    """Habana's setting name -> the file with CUDA's list."""
    here = pathlib.Path(__file__).resolve().parent
    return {name: str(here / file) for name, file in FILES.items()}


def exports():
    """The shell lines that point Habana at the CUDA lists."""
    return [f"export {name}={shlex.quote(path)}" for name, path in paths().items()]


if __name__ == "__main__":
    print("\n".join(exports()))
