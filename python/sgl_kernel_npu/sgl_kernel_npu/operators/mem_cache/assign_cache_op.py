"""Public assign_cache_op operator with explicit implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("mem_cache.assign_cache_op", "auto")
@register_impl("mem_cache.assign_cache_op", "ascendc")
def _ascendc_impl(out, src, dst_start_idx, dst_end_idx, src_start_idx, src_end_idx):
    import torch

    return torch.ops.npu.assign_cache_op(
        out, src, dst_start_idx, dst_end_idx, src_start_idx, src_end_idx
    )


@register_impl("mem_cache.assign_cache_op", "torch_native")
def _torch_native_impl(
    out, src, dst_start_idx, dst_end_idx, src_start_idx, src_end_idx
):
    from ...implementation.torch_native.mem_cache.assign_cache_op import assign_cache_op

    return assign_cache_op(
        out, src, dst_start_idx, dst_end_idx, src_start_idx, src_end_idx
    )


@register_operator("mem_cache.assign_cache_op")
def assign_cache_op(
    out, src, dst_start_idx, dst_end_idx, src_start_idx, src_end_idx, *, impl="auto"
):
    """Dispatch to ascendc by default; use impl="torch_native" for the reference.

    Preserve the existing input, output, and in-place behavior.
    """
    return get_impl_by_name("mem_cache.assign_cache_op", impl)(
        out, src, dst_start_idx, dst_end_idx, src_start_idx, src_end_idx
    )
