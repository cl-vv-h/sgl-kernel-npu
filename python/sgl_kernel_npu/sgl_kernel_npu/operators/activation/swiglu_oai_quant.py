"""Public swiglu_oai_quant operator with explicit implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("activation.swiglu_oai_quant", "auto")
@register_impl("activation.swiglu_oai_quant", "triton")
def _triton_impl(
    x, alpha, limit, need_quant=True, group_list=None, group_list_type=None
):
    from ...implementation.triton.activation.swiglu_oai_quant import swiglu_oai_quant

    return swiglu_oai_quant(x, alpha, limit, need_quant, group_list, group_list_type)


@register_impl("activation.swiglu_oai_quant", "torch_native")
def _torch_native_impl(
    x, alpha, limit, need_quant=True, group_list=None, group_list_type=None
):
    from ...implementation.torch_native.activation.swiglu_oai_quant import (
        swiglu_oai_quant,
    )

    return swiglu_oai_quant(x, alpha, limit, need_quant, group_list, group_list_type)


@register_operator("activation.swiglu_oai_quant")
def swiglu_oai_quant(
    x,
    alpha,
    limit,
    need_quant=True,
    group_list=None,
    group_list_type=None,
    *,
    impl="auto"
):
    """Dispatch to triton by default; use impl="torch_native" for the reference.

    Scale is unwritten when need_quant=False; inactive capacity rows are also unspecified, as in the original kernel.
    """
    return get_impl_by_name("activation.swiglu_oai_quant", impl)(
        x, alpha, limit, need_quant, group_list, group_list_type
    )
