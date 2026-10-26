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
    from math import prod

    import torch

    from ...implementation.torch_native.activation.swiglu_group_quant import (
        swiglu_group_quant,
    )

    dtype = torch.float8_e4m3fn if dst_type is None else dst_type
    if (
        x.ndim < 1
        or x.dtype not in (torch.float16, torch.bfloat16)
        or not x.is_contiguous()
    ):
        raise ValueError("x must be a contiguous FP16/BF16 tensor")
    if quant_mode not in (1, 2, 3) or group_list_type != 0:
        raise ValueError(
            "supported quant_mode values are 1/2/3; group_list_type must be 0"
        )
    if dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("dst_type must be FP8 E4M3FN or E5M2")
    alignment = 128 if quant_mode == 2 else 256
    if x.shape[-1] == 0 or x.shape[-1] % alignment:
        raise ValueError(f"last dimension must be a positive multiple of {alignment}")
    if quant_mode == 3 and ue8m0_scale and not round_scale:
        raise NotImplementedError(
            "torch_native requires round_scale=True for a coherent E8M0 mode-3 scale buffer"
        )
    rows, cols = prod(x.shape[:-1]), x.shape[-1] // 2
    valid = rows
    if group_index is not None and group_index.numel():
        # VFProcessGroupIndex sums the entries; do not substitute the last entry.
        valid = max(0, min(rows, int(group_index.sum().item())))
    y = torch.empty((*x.shape[:-1], cols), dtype=dtype, device=x.device)
    origin = torch.empty((*x.shape[:-1], cols), dtype=x.dtype, device=x.device)
    scale_shape = (
        (*x.shape[:-1], (cols + 63) // 64, 2)
        if quant_mode == 2
        else (*x.shape[:-1], (cols + 127) // 128)
    )
    scale_dtype = (
        torch.float8_e8m0fnu
        if quant_mode == 2 or (quant_mode == 3 and ue8m0_scale)
        else torch.float32
    )
    scale = torch.empty(scale_shape, dtype=scale_dtype, device=x.device)
    flat = x.reshape(rows, x.shape[-1])
    weight = None
    if valid and topk_weight is not None and topk_weight.numel():
        weight = topk_weight.reshape(-1)[:valid].unsqueeze(-1)

    return swiglu_group_quant(
        flat,
        weight,
        quant_mode,
        round_scale,
        ue8m0_scale,
        output_origin,
        clamp_value,
        dtype=dtype,
        rows=rows,
        cols=cols,
        valid=valid,
        y=y,
        scale=scale,
        origin=origin,
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
    impl="auto",
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
