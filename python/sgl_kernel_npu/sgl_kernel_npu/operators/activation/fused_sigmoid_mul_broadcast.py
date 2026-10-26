"""Public fused_sigmoid_mul_broadcast operator with explicit implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("activation.fused_sigmoid_mul_broadcast", "auto")
@register_impl("activation.fused_sigmoid_mul_broadcast", "triton")
def _triton_impl(x, gate):
    import torch
    import triton

    from ...implementation.triton.activation.fused_sigmoid_mul import (
        _fused_sigmoid_mul_broadcast_kernel,
    )
    from ...utils.triton_utils import get_device_properties

    _validate_inputs(x, gate)
    bs, hidden_dim = x.shape
    x = x.contiguous()
    gate = gate.contiguous()
    # Allocate after contiguous() so a transposed/strided x still yields a
    # contiguous output matching the kernel's row-major writes.
    out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    _, num_vectorcore = get_device_properties()
    n_rows = min(bs, num_vectorcore)

    _fused_sigmoid_mul_broadcast_kernel[(n_rows,)](
        out,
        gate,
        x,
        bs,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


@register_impl("activation.fused_sigmoid_mul_broadcast", "torch_native")
def _torch_native_impl(x, gate):
    from ...implementation.torch_native.activation.fused_sigmoid_mul_broadcast import (
        fused_sigmoid_mul_broadcast,
    )

    _validate_inputs(x, gate)
    return fused_sigmoid_mul_broadcast(x, gate.reshape(-1, 1))


@register_operator("activation.fused_sigmoid_mul_broadcast")
def fused_sigmoid_mul_broadcast(x, gate, *, impl="auto"):
    """Dispatch to triton by default; use impl="torch_native" for the reference.

    Preserve the existing input, output, and in-place behavior.
    """
    return get_impl_by_name("activation.fused_sigmoid_mul_broadcast", impl)(x, gate)


def _validate_inputs(x, gate):
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D, got ndim={x.ndim}, shape={x.shape}")

    bs, hidden_dim = x.shape

    # Accept gate as (N,) or (N, 1).  Passing it directly avoids the squeeze()
    # Python overhead that showed up in micro-benchmarks on Ascend NPU.
    if gate.ndim == 2:
        if gate.shape[0] != bs or gate.shape[1] != 1:
            raise ValueError(f"2-D gate must have shape ({bs}, 1), got {gate.shape}")
    elif gate.ndim != 1 or gate.shape[0] != bs:
        raise ValueError(f"gate must be ({bs},) or ({bs}, 1), got {gate.shape}")
