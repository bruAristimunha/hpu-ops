"""``im2col`` (``F.unfold``) and ``col2im`` (``F.fold``) on HPU.

Habana's eager stack lacks ``aten::im2col`` and ``aten::col2im``, and lazy mode falls back to the CPU for both.
``Tensor.unfold`` falls back in lazy mode, ``unfold_backward`` falls back in both modes, grouped convolutions crash
Synapse when ``kh > 256`` (``MAtt`` uses ``k=(257, 1)``), and 6-D reshapes / ``expand`` fail under
``torch.compile(..., dynamic=True)``.

Two on-card paths are used with autocast off, differentiated directly by autograd:

- **Tier 1 (4-D non-overlapping fast path)**: when ``dilation == [1, 1]``, ``padding == [0, 0]``,
  ``stride == kernel_size``, and either ``(W == 1 and kw == 1)`` or ``(H == 1 and kh == 1)`` (covering all five
  affected ``spd_learn`` models: ``SSGCNet``, ``MAtt``, ``GyroAtt``, ``SPDMetaFormer``, ``CorAtt``). Uses <= 4-D
  ``slice`` / ``reshape`` / ``transpose`` / ``F.pad`` with exact native gradients and zero index tensors.
- **Tier 2 (general exact path)**: all other valid configurations. Computes flat spatial indices (on CPU + single
  ``.to(device)`` in eager/lazy, or on ``device`` when tracing) and runs ``index_select`` for ``im2col`` and
  ``index_add`` for ``col2im`` (casting ``bool`` through ``float32`` for ``index_add``), avoiding ``expand`` so
  ``torch.compile(..., fullgraph=True, dynamic=True)`` succeeds.

Results are tensors of their own (contiguous, never a view of the input), as ATen's. An empty batch or channel dim
builds no index. The spatial index (``kh * kw * oh * ow`` int64 elements) is built on the host and moved once, or on
the device while ``torch.compile`` traces, and is bounded by ``MAX_INDEX``: larger calls keep the native op. An
``.out`` that aliases the input raises ATen's overlap error before anything is written.

Invalid inputs and unsupported dtypes (e.g. ``int32``, ``int64``) pass through to ATen so native exceptions are
preserved.
"""
import torch
import torch.nn.functional as F
from torch._C import DispatchKey

from ._dispatch import count, register_kernels, tracing

aten = torch.ops.aten
MAX_INDEX = 1 << 24  # spatial index elements (kh * kw * oh * ow, int64: 128 MiB); larger calls keep the native op
OVERLAP = ("unsupported operation: some elements of the input tensor and the written-to tensor refer to a single "
           "memory location. Please clone() the tensor before performing the operation.")
SUPPORTED_DTYPES = frozenset((
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
    torch.bool,
))


def _valid_params(kernel_size, dilation, padding, stride):
    return (
        len(kernel_size) == 2
        and len(dilation) == 2
        and len(padding) == 2
        and len(stride) == 2
        and all(k > 0 for k in kernel_size)
        and all(d > 0 for d in dilation)
        and all(p >= 0 for p in padding)
        and all(s > 0 for s in stride)
    )


def _out_size(height, width, kh, kw, dh, dw, ph, pw, sh, sw):
    oh = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    return oh, ow


def _valid_im2col(self, kernel_size, dilation, padding, stride):
    if self.ndim not in (3, 4) or self.dtype not in SUPPORTED_DTYPES:
        return False
    if not _valid_params(kernel_size, dilation, padding, stride):
        return False
    non_batch = self.shape[1:] if self.ndim == 4 else self.shape
    if any(dim <= 0 for dim in non_batch):
        return False
    _, height, width = non_batch
    oh, ow = _out_size(height, width, *kernel_size, *dilation, *padding, *stride)
    return oh > 0 and ow > 0 and kernel_size[0] * kernel_size[1] * oh * ow <= MAX_INDEX


def _valid_col2im(self, output_size, kernel_size, dilation, padding, stride):
    if self.ndim not in (2, 3) or self.dtype not in SUPPORTED_DTYPES:
        return False
    if len(output_size) != 2 or not _valid_params(kernel_size, dilation, padding, stride):
        return False
    height, width = output_size
    if height <= 0 or width <= 0:
        return False
    kh, kw = kernel_size
    c_kw, length = self.shape[-2], self.shape[-1]
    if c_kw <= 0 or c_kw % (kh * kw) != 0:
        return False
    oh, ow = _out_size(height, width, kh, kw, *dilation, *padding, *stride)
    return oh > 0 and ow > 0 and length == oh * ow and kh * kw * length <= MAX_INDEX


def _is_4d_non_overlap(height, width, kernel_size, dilation, padding, stride):
    kh, kw = kernel_size
    return (
        tuple(dilation) == (1, 1)
        and tuple(padding) == (0, 0)
        and tuple(stride) == (kh, kw)
        and ((width == 1 and kw == 1) or (height == 1 and kh == 1))
    )


def _tracing(t):
    """True while a tracer sees ``t`` (``_dispatch.tracing``: dynamo, AOT autograd's fake and functional tensors, a
    fake or proxy mode): build index tensors on the device, in the graph, instead of on the host."""
    return tracing(t)


def _fresh(res, src):
    """``res`` as a tensor of its own, as ATen's im2col and col2im return: the fast path's reshapes can be views of
    ``src``."""
    res = res.contiguous()
    return res.clone() if torch._C._is_alias_of(res, src) else res


def _flat_indices(height, width, kh, kw, dh, dw, ph, pw, sh, sw, device, tracing):
    hp, wp = height + 2 * ph, width + 2 * pw
    oh, ow = _out_size(height, width, kh, kw, dh, dw, ph, pw, sh, sw)
    idx_device = device if tracing else torch.device("cpu")
    ky = torch.arange(kh, device=idx_device, dtype=torch.int64) * dh
    kx = torch.arange(kw, device=idx_device, dtype=torch.int64) * dw
    oy = torch.arange(oh, device=idx_device, dtype=torch.int64) * sh
    ox = torch.arange(ow, device=idx_device, dtype=torch.int64) * sw
    rows = ky[:, None, None, None] + oy[None, None, :, None]
    cols = kx[None, :, None, None] + ox[None, None, None, :]
    idx = (rows * wp + cols).reshape(-1).to(device)
    return idx, hp, wp, oh, ow


def im2col(self, kernel_size, dilation, padding, stride):
    """On-card ``im2col``: 4-D non-overlapping fast path or ``index_select`` general path."""
    x = self.unsqueeze(0) if self.ndim == 3 else self
    batch, channels, height, width = x.shape
    kh, kw = kernel_size
    if batch * channels == 0:  # no index to build; the empty result stays connected to self for autograd
        oh, ow = _out_size(height, width, kh, kw, *dilation, *padding, *stride)
        res = x.reshape(-1)[:0].reshape(batch, channels * kh * kw, oh * ow).clone()
        return res.squeeze(0) if self.ndim == 3 else res
    with torch.autocast(self.device.type, enabled=False):
        if _is_4d_non_overlap(height, width, kernel_size, dilation, padding, stride):
            if width == 1 and kw == 1:
                oh = height // kh
                if height > oh * kh:
                    x = x[:, :, : oh * kh, :]
                res = x.reshape(batch, channels, oh, kh).transpose(2, 3).reshape(batch, channels * kh, oh)
            else:
                ow = width // kw
                if width > ow * kw:
                    x = x[:, :, :, : ow * kw]
                res = x.reshape(batch, channels, ow, kw).transpose(2, 3).reshape(batch, channels * kw, ow)
        else:
            dh, dw = dilation
            ph, pw = padding
            sh, sw = stride
            idx, hp, wp, oh, ow = _flat_indices(height, width, kh, kw, dh, dw, ph, pw, sh, sw, self.device,
                                                _tracing(self))
            xp = F.pad(x, (pw, pw, ph, ph)) if (ph or pw) else x
            res = xp.reshape(batch * channels, hp * wp).index_select(1, idx).reshape(
                batch, channels * kh * kw, oh * ow
            )
    res = _fresh(res, self)
    return res.squeeze(0) if self.ndim == 3 else res


def col2im(self, output_size, kernel_size, dilation, padding, stride):
    """On-card ``col2im``: 4-D non-overlapping fast path or ``index_add`` general path."""
    x = self.unsqueeze(0) if self.ndim == 2 else self
    batch, c_kw, length = x.shape
    height, width = output_size
    kh, kw = kernel_size
    channels = c_kw // (kh * kw)
    if batch * channels == 0:  # no index to build; the empty result stays connected to self for autograd
        out = x.reshape(-1)[:0].reshape(batch, channels, height, width).clone()
        return out.squeeze(0) if self.ndim == 2 else out
    with torch.autocast(self.device.type, enabled=False):
        if _is_4d_non_overlap(height, width, kernel_size, dilation, padding, stride):
            if width == 1 and kw == 1:
                oh = height // kh
                out = x.reshape(batch, channels, kh, oh).transpose(2, 3).reshape(batch, channels, oh * kh, 1)
                if height > oh * kh:
                    out = F.pad(out, (0, 0, 0, height - oh * kh))
            else:
                ow = width // kw
                out = x.reshape(batch, channels, kw, ow).transpose(2, 3).reshape(batch, channels, 1, ow * kw)
                if width > ow * kw:
                    out = F.pad(out, (0, width - ow * kw, 0, 0))
        else:
            dh, dw = dilation
            ph, pw = padding
            sh, sw = stride
            idx, hp, wp, _, _ = _flat_indices(height, width, kh, kw, dh, dw, ph, pw, sh, sw, self.device,
                                              _tracing(self))
            src = x.reshape(batch * channels, length * kh * kw)
            if x.dtype == torch.bool:
                acc = src.new_zeros((batch * channels, hp * wp), dtype=torch.float32).index_add(
                    1, idx, src.to(torch.float32)
                ).bool()
            else:
                acc = src.new_zeros((batch * channels, hp * wp)).index_add(1, idx, src)
            out = acc.reshape(batch, channels, hp, wp)
            if ph or pw:
                out = out[:, :, ph : height + ph, pw : width + pw]  # a crop of the padded sum: copied below
    out = _fresh(out, self)
    return out.squeeze(0) if self.ndim == 2 else out


def _cpu_fallback(op, self, *args, out=None):
    cpu_self = self.to("cpu")
    if out is None:
        return op(cpu_self, *args).to(self.device)
    if out.device != self.device:
        raise RuntimeError(
            f"Expected all tensors to be on the same device, but found at least two devices, "
            f"{self.device} and {out.device}!"
        )
    cpu_out = out.to("cpu")
    op(cpu_self, *args, out=cpu_out)
    out.resize_(cpu_out.shape).copy_(cpu_out)
    return out


def _im2col_kernel(device_type, op, fallback=None):
    def kernel(self, kernel_size, dilation, padding, stride):
        if self.device.type == device_type and _valid_im2col(self, kernel_size, dilation, padding, stride):
            count("im2col")
            return im2col(self, kernel_size, dilation, padding, stride)
        if fallback is not None:
            return op._op_dk(fallback, self, kernel_size, dilation, padding, stride)
        return _cpu_fallback(op, self, kernel_size, dilation, padding, stride)

    return kernel


def _im2col_out_kernel(device_type, op):
    def kernel(self, kernel_size, dilation, padding, stride, *, out):
        if (
            self.device.type == device_type
            and out.device == self.device
            and out.dtype == self.dtype
            and _valid_im2col(self, kernel_size, dilation, padding, stride)
        ):
            if torch._C._is_alias_of(out, self):
                raise RuntimeError(OVERLAP)  # before resizing out, which is self
            count("im2col")
            res = im2col(self, kernel_size, dilation, padding, stride)
            out.resize_(res.shape).copy_(res)
            return out
        return _cpu_fallback(op, self, kernel_size, dilation, padding, stride, out=out)

    return kernel


def _col2im_kernel(device_type, op, fallback=None):
    def kernel(self, output_size, kernel_size, dilation, padding, stride):
        if self.device.type == device_type and _valid_col2im(self, output_size, kernel_size, dilation, padding, stride):
            count("col2im")
            return col2im(self, output_size, kernel_size, dilation, padding, stride)
        if fallback is not None:
            return op._op_dk(fallback, self, output_size, kernel_size, dilation, padding, stride)
        return _cpu_fallback(op, self, output_size, kernel_size, dilation, padding, stride)

    return kernel


def _col2im_out_kernel(device_type, op):
    def kernel(self, output_size, kernel_size, dilation, padding, stride, *, out):
        if (
            self.device.type == device_type
            and out.device == self.device
            and out.dtype == self.dtype
            and _valid_col2im(self, output_size, kernel_size, dilation, padding, stride)
        ):
            if torch._C._is_alias_of(out, self):
                raise RuntimeError(OVERLAP)  # before resizing out, which is self
            count("col2im")
            res = col2im(self, output_size, kernel_size, dilation, padding, stride)
            out.resize_(res.shape).copy_(res)
            return out
        return _cpu_fallback(op, self, output_size, kernel_size, dilation, padding, stride, out=out)

    return kernel


def install(device_type):
    """Rewrite ``aten::im2col`` and ``aten::col2im`` (``.default`` and ``.out``) on ``device_type``. Returns an undo."""
    im_default, im_out = aten.im2col.default, aten.im2col.out
    col_default, col_out = aten.col2im.default, aten.col2im.out
    return register_kernels(device_type, [
        (im_default, "Autograd{}", _im2col_kernel(device_type, im_default, DispatchKey.Autograd)),
        (im_default, "{}", _im2col_kernel(device_type, im_default)),
        (im_out, "{}", _im2col_out_kernel(device_type, im_out)),
        (col_default, "Autograd{}", _col2im_kernel(device_type, col_default, DispatchKey.Autograd)),
        (col_default, "{}", _col2im_kernel(device_type, col_default)),
        (col_out, "{}", _col2im_out_kernel(device_type, col_out)),
    ])
