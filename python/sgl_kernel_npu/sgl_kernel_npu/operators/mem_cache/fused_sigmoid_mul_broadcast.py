"""Public fused_sigmoid_mul_broadcast operator with explicit implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("mem_cache.fused_sigmoid_mul_broadcast", "auto")
@register_impl("mem_cache.fused_sigmoid_mul_broadcast", "triton")
def _triton_impl(x, gate):
    from ...implementation.triton.mem_cache.fused_sigmoid_mul import (
        fused_sigmoid_mul_broadcast,
    )

    return fused_sigmoid_mul_broadcast(x, gate)


@register_impl("mem_cache.fused_sigmoid_mul_broadcast", "torch_native")
def _torch_native_impl(x, gate):
    from ...implementation.torch_native.mem_cache.fused_sigmoid_mul_broadcast import (
        fused_sigmoid_mul_broadcast,
    )

    return fused_sigmoid_mul_broadcast(x, gate)


@register_operator("mem_cache.fused_sigmoid_mul_broadcast")
def fused_sigmoid_mul_broadcast(x, gate, *, impl="auto"):
    """Dispatch to triton by default; use impl="torch_native" for the reference.

    Preserve the existing input, output, and in-place behavior.
    """
    return get_impl_by_name("mem_cache.fused_sigmoid_mul_broadcast", impl)(x, gate)
