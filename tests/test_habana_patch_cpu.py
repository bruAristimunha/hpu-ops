"""The #63 vendor patch: fill_missing_output_strides on plain FX graphs (CPU)."""
import torch

from hpu_ops import habana_patch


class _M(torch.nn.Module):
    def forward(self, x, y):
        return x @ y


def _graph_with_missing():
    gm = torch.fx.symbolic_trace(_M())
    nodes = list(gm.graph.nodes)
    ph = [n for n in nodes if n.op == "placeholder"]
    ph[0].meta["val"] = torch.randn(4, 4).t()  # non-contiguous fake value
    ph[1].meta["val"] = torch.randn(4, 4)
    ph[1].meta["output_strides"] = [(999,)]  # vendor-computed: must not be touched
    return gm, ph


def test_fills_only_missing_from_fake_value():
    gm, ph = _graph_with_missing()
    assert habana_patch.fill_missing_output_strides(gm) == 1
    assert ph[0].meta["output_strides"] == [tuple(ph[0].meta["val"].stride())]
    assert ph[1].meta["output_strides"] == [(999,)]


def test_no_fake_value_no_fill():
    gm = torch.fx.symbolic_trace(_M())
    assert habana_patch.fill_missing_output_strides(gm) == 0


def test_patch_recipe_compiler_noop_off_habana():
    assert habana_patch.patch_recipe_compiler() is False  # no habana_frameworks here
