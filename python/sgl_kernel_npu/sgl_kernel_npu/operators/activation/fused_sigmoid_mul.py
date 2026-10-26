"""Same-shape fused sigmoid multiplication."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("activation.fused_sigmoid_mul", "auto")
@register_impl("activation.fused_sigmoid_mul", "triton")
def _triton_impl(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Compute ``x * sigmoid(gate)`` in one fused AIV kernel (same-shape).

    Args:
        x: input tensor, any shape.
        gate: gating tensor, must have the same shape as ``x``.

    Returns:
        A tensor with the same shape as ``x``.
    """
    import triton

    from ...implementation.triton.activation.fused_sigmoid_mul import (
        _ELEM_BLOCK_SIZE,
        _fused_sigmoid_mul_kernel,
    )
    from ...utils.triton_utils import get_device_properties

    if x.shape != gate.shape:
        raise ValueError(
            f"x and gate must have the same shape, got {x.shape} vs {gate.shape}"
        )

    x = x.contiguous()
    gate = gate.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out

    _, num_vectorcore = get_device_properties()
    num_blocks = min(triton.cdiv(n, _ELEM_BLOCK_SIZE), num_vectorcore)
    num_blocks = max(num_blocks, 1)

    _fused_sigmoid_mul_kernel[(num_blocks,)](
        x,
        gate,
        out,
        n,
        BLOCK_SIZE=_ELEM_BLOCK_SIZE,
    )
    return out


@register_impl("activation.fused_sigmoid_mul", "torch_native")
def _torch_native_impl(x, gate):
    from ...implementation.torch_native.activation.fused_sigmoid_mul import (
        fused_sigmoid_mul,
    )

    if x.shape != gate.shape:
        raise ValueError(
            f"x and gate must have the same shape, got {x.shape} vs {gate.shape}"
        )
    return fused_sigmoid_mul(x, gate)


@register_operator("activation.fused_sigmoid_mul")
def fused_sigmoid_mul(x, gate, *, impl="auto"):
    """Compute x * sigmoid(gate), preserving x's dtype and shape."""
    return get_impl_by_name("activation.fused_sigmoid_mul", impl)(x, gate)
