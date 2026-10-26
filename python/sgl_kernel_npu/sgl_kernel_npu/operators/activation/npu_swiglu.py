"""Public npu_swiglu operator with explicit implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("activation.npu_swiglu", "auto")
@register_impl("activation.npu_swiglu", "torch_npu")
def _torch_npu_impl(input, dim=-1):
    import torch_npu

    return torch_npu.npu_swiglu(input, dim)


@register_impl("activation.npu_swiglu", "torch_native")
def _torch_native_impl(input, dim=-1):
    from ...implementation.torch_native.activation.npu_swiglu import npu_swiglu

    return npu_swiglu(input, dim)


@register_operator("activation.npu_swiglu")
def npu_swiglu(input, dim=-1, *, impl="auto"):
    """Dispatch to torch_npu by default; use impl="torch_native" for the reference.

    Preserve the existing input, output, and in-place behavior.
    """
    return get_impl_by_name("activation.npu_swiglu", impl)(input, dim)
