"""Public npu_format_cast operator with explicit implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("data_move.npu_format_cast", "auto")
@register_impl("data_move.npu_format_cast", "torch_npu")
def _torch_npu_impl(input, acl_format, *, customize_dtype=None, input_dtype=None):
    import torch_npu

    kwargs = {}
    if customize_dtype is not None:
        kwargs["customize_dtype"] = customize_dtype
    if input_dtype is not None:
        kwargs["input_dtype"] = input_dtype
    return torch_npu.npu_format_cast(input, acl_format, **kwargs)


@register_impl("data_move.npu_format_cast", "torch_native")
def _torch_native_impl(input, acl_format, *, customize_dtype=None, input_dtype=None):
    from ...implementation.torch_native.data_move.npu_format_cast import npu_format_cast

    return npu_format_cast(
        input, acl_format, customize_dtype=customize_dtype, input_dtype=input_dtype
    )


@register_operator("data_move.npu_format_cast")
def npu_format_cast(
    input, acl_format, *, customize_dtype=None, input_dtype=None, impl="auto"
):
    """Dispatch to torch_npu by default; use impl="torch_native" for the reference.

    torch_native supports base dense formats only; private storage descriptors and packed custom types require torch_npu.
    """
    return get_impl_by_name("data_move.npu_format_cast", impl)(
        input, acl_format, customize_dtype=customize_dtype, input_dtype=input_dtype
    )
