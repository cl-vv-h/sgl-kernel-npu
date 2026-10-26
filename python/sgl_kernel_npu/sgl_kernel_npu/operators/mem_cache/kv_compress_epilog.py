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
    import torch

    from ...implementation.torch_native.mem_cache.kv_compress_epilog import (
        kv_compress_epilog,
    )

    if x.ndim != 2 or x.dtype != torch.bfloat16 or x.shape[1] <= 64:
        raise ValueError(
            "x must be a BF16 matrix with a quantized prefix and 64 RoPE columns"
        )
    if quant_mode not in (1, 2) or layout not in (1, 2):
        raise ValueError("quant_mode and layout must be 1 or 2")
    if layout == 2 and quant_mode != 2:
        raise ValueError("layout=2 only supports MXFP8")
    if quant_mode == 2 and quant_group_size not in (64, 128):
        raise ValueError("MXFP8 quant_group_size must be 64 or 128")
    cache = (
        kv_compress_cache.squeeze(2)
        if kv_compress_cache.ndim == 4
        else kv_compress_cache
    )
    dtype = cache.dtype
    if dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("cache must use FP8 E4M3FN or E5M2")
    rows, cols = x.shape[0], x.shape[1] - 64
    width = 128 if quant_mode == 1 else quant_group_size

    if not cache.is_contiguous():
        raise NotImplementedError(
            "torch_native packed-cache reference requires contiguous storage"
        )
    if slot_mapping.numel() != rows:
        raise ValueError("slot_mapping length must match x rows")
    target = cache.view(torch.uint8).reshape(cache.shape[0], -1)
    slots = slot_mapping.tolist()
    groups = (cols + width - 1) // width
    scale_bytes = groups * 4 if quant_mode == 1 else groups
    value_width = cols + 128
    padded_width = ((value_width + scale_bytes + 127) // 128) * 128
    scale_width = ((scale_bytes + 7) // 8) * 8
    block_size = 1 if layout == 1 else cache.shape[1]
    if layout == 1 and target.shape[-1] < padded_width:
        raise ValueError("cache row is too short for the packed layout")
    if layout == 2 and target.shape[-1] < block_size * (value_width + scale_width):
        raise ValueError("cache block is too short for the packed layout")
    if any(
        slot != -1 and not 0 <= slot < cache.shape[0] * block_size for slot in slots
    ):
        raise ValueError("slot_mapping index is out of bounds")
    return kv_compress_epilog(
        x,
        target,
        slots,
        quant_mode,
        round_scale_flag,
        layout,
        dtype=dtype,
        cols=cols,
        width=width,
        block_size=block_size,
        value_width=value_width,
        scale_width=scale_width,
        padded_width=padded_width,
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
