"""Public npu_fast_gelu operator with explicit implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("activation.npu_fast_gelu", "auto")
@register_impl("activation.npu_fast_gelu", "torch_npu")
def _torch_npu_impl(input):
    import torch_npu

    return torch_npu.npu_fast_gelu(input)


@register_impl("activation.npu_fast_gelu", "torch_native")
def _torch_native_impl(input):
    from ...implementation.torch_native.activation.npu_fast_gelu import npu_fast_gelu

    return npu_fast_gelu(input)


@register_operator("activation.npu_fast_gelu")
def npu_fast_gelu(input, *, impl="auto"):
    """Dispatch to torch_npu by default; use impl="torch_native" for the reference.

    Preserve the existing input, output, and in-place behavior.
    """
    return get_impl_by_name("activation.npu_fast_gelu", impl)(input)
