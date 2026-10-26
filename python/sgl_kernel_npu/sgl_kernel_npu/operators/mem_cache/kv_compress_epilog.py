"""Public kv_compress_epilog operator with explicit implementation selection."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl(
    "mem_cache.kv_compress_epilog", "auto", supported_generations=[Generation.ARCH35]
)
@register_impl(
    "mem_cache.kv_compress_epilog", "ascendc", supported_generations=[Generation.ARCH35]
)
def _ascendc_impl(
    kv_compress_cache,
    x,
    slot_mapping,
    quant_group_size,
    quant_mode,
    round_scale_flag,
    layout,
):
    import torch

    return torch.ops.npu.kv_compress_epilog(
        kv_compress_cache,
        x,
        slot_mapping,
        quant_group_size,
        quant_mode,
        round_scale_flag,
        layout,
    )


@register_impl("mem_cache.kv_compress_epilog", "torch_native")
def _torch_native_impl(
    kv_compress_cache,
    x,
    slot_mapping,
    quant_group_size,
    quant_mode,
    round_scale_flag,
    layout,
):
    from ...implementation.torch_native.mem_cache.kv_compress_epilog import (
        kv_compress_epilog,
    )

    return kv_compress_epilog(
        kv_compress_cache,
        x,
        slot_mapping,
        quant_group_size,
        quant_mode,
        round_scale_flag,
        layout,
    )


@register_operator("mem_cache.kv_compress_epilog")
def kv_compress_epilog(
    kv_compress_cache,
    x,
    slot_mapping,
    quant_group_size,
    quant_mode,
    round_scale_flag,
    layout,
    *,
    impl="auto"
):
    """Dispatch to ascendc by default; use impl="torch_native" for the reference.

    Preserve the existing input, output, and in-place behavior.
    """
    return get_impl_by_name("mem_cache.kv_compress_epilog", impl)(
        kv_compress_cache,
        x,
        slot_mapping,
        quant_group_size,
        quant_mode,
        round_scale_flag,
        layout,
    )
