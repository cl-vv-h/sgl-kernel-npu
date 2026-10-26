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

    inputs = (out, src, dst_start_idx, dst_end_idx, src_start_idx, src_end_idx)
    if any(t.ndim < 1 for t in inputs):
        return False
    # Preserve the existing host-side batch checks, including its src size check.
    if not (
        out.shape[0]
        == src.shape[0]
        == dst_start_idx.shape[0]
        == dst_end_idx.shape[0]
        == src_start_idx.shape[0]
        == src_end_idx.shape[0]
    ):
        return False

    intervals = [
        (int(dst_start_idx[i].item()), int(dst_end_idx[i].item()))
        for i in range(out.shape[0])
    ]
    starts = [start for start, _ in intervals]
    src_starts = [int(src_start_idx[i].item()) for i in range(out.shape[0])]
    src_ends = [int(src_end_idx[i].item()) for i in range(out.shape[0])]
    return assign_cache_op(out, src.reshape(-1), starts, src_starts, src_ends)


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
