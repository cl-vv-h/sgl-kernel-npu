"""Public swiglu_group_quant operator with explicit implementation selection."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl(
    "activation.swiglu_group_quant", "auto", supported_generations=[Generation.ARCH35]
)
@register_impl(
    "activation.swiglu_group_quant",
    "ascendc",
    supported_generations=[Generation.ARCH35],
)
def _ascendc_impl(
    x,
    topk_weight=None,
    group_index=None,
    dst_type=None,
    quant_mode=1,
    group_size=128,
    round_scale=False,
    ue8m0_scale=False,
    output_origin=False,
    group_list_type=0,
    clamp_value=0.0,
):
    import torch

    return torch.ops.npu.swiglu_group_quant(
        x,
        topk_weight,
        group_index,
        dst_type,
        quant_mode,
        group_size,
        round_scale,
        ue8m0_scale,
        output_origin,
        group_list_type,
        clamp_value,
    )


@register_impl("activation.swiglu_group_quant", "torch_native")
def _torch_native_impl(
    x,
    topk_weight=None,
    group_index=None,
    dst_type=None,
    quant_mode=1,
    group_size=128,
    round_scale=False,
    ue8m0_scale=False,
    output_origin=False,
    group_list_type=0,
    clamp_value=0.0,
):
    from ...implementation.torch_native.activation.swiglu_group_quant import (
        swiglu_group_quant,
    )

    return swiglu_group_quant(
        x,
        topk_weight,
        group_index,
        dst_type,
        quant_mode,
        group_size,
        round_scale,
        ue8m0_scale,
        output_origin,
        group_list_type,
        clamp_value,
    )


@register_operator("activation.swiglu_group_quant")
def swiglu_group_quant(
    x,
    topk_weight=None,
    group_index=None,
    dst_type=None,
    quant_mode=1,
    group_size=128,
    round_scale=False,
    ue8m0_scale=False,
    output_origin=False,
    group_list_type=0,
    clamp_value=0.0,
    *,
    impl="auto"
):
    """Dispatch to ascendc by default; use impl="torch_native" for the reference.

    Inactive capacity rows and y_origin when not written by the kernel are unspecified. torch_native is an eager numerical reference.
    """
    return get_impl_by_name("activation.swiglu_group_quant", impl)(
        x,
        topk_weight,
        group_index,
        dst_type,
        quant_mode,
        group_size,
        round_scale,
        ue8m0_scale,
        output_origin,
        group_list_type,
        clamp_value,
    )
