"""Public alloc_extend operator with explicit implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("mem_cache.alloc_extend", "auto")
@register_impl("mem_cache.alloc_extend", "ascendc")
def _ascendc_impl(
    pre_lens, seq_lens, last_loc, free_pages, page_size, out_indices, values
):
    import torch

    return torch.ops.npu.alloc_extend(
        pre_lens, seq_lens, last_loc, free_pages, page_size, out_indices, values
    )


@register_impl("mem_cache.alloc_extend", "torch_native")
def _torch_native_impl(
    pre_lens, seq_lens, last_loc, free_pages, page_size, out_indices, values
):
    from ...implementation.torch_native.mem_cache.alloc_extend import alloc_extend

    if page_size <= 0:
        raise ValueError("page_size must be positive")
    pre = [int(pre_lens[i].item()) for i in range(pre_lens.numel())]
    end = [int(seq_lens[i].item()) for i in range(pre_lens.numel())]
    if any(after < before for before, after in zip(pre, end)):
        raise ValueError("seq_lens must be at least pre_lens")
    return alloc_extend(pre, end, last_loc, free_pages, page_size, out_indices, values)


@register_operator("mem_cache.alloc_extend")
def alloc_extend(
    pre_lens,
    seq_lens,
    last_loc,
    free_pages,
    page_size,
    out_indices,
    values,
    *,
    impl="auto"
):
    """Dispatch to ascendc by default; use impl="torch_native" for the reference.

    Preserve the existing input, output, and in-place behavior.
    """
    return get_impl_by_name("mem_cache.alloc_extend", impl)(
        pre_lens, seq_lens, last_loc, free_pages, page_size, out_indices, values
    )
