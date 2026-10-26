"""Public npu_dtype_cast operator with explicit implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("data_move.npu_dtype_cast", "auto")
@register_impl("data_move.npu_dtype_cast", "torch_npu")
def _torch_npu_impl(input, dtype):
    import torch_npu

    return torch_npu.npu_dtype_cast(input, dtype)


@register_impl("data_move.npu_dtype_cast", "torch_native")
def _torch_native_impl(input, dtype):
    import torch

    from ...implementation.torch_native.data_move.npu_dtype_cast import npu_dtype_cast

    if not isinstance(dtype, torch.dtype) or "float4" in str(dtype):
        raise NotImplementedError(
            "torch_native dtype_cast requires a standard unpacked torch.dtype; use torch_npu for packed FP4"
        )
    return npu_dtype_cast(input, dtype)


@register_operator("data_move.npu_dtype_cast")
def npu_dtype_cast(input, dtype, *, impl="auto"):
    """Dispatch to torch_npu by default; use impl="torch_native" for the reference.

    Preserve the existing input, output, and in-place behavior.
    """
    return get_impl_by_name("data_move.npu_dtype_cast", impl)(input, dtype)
