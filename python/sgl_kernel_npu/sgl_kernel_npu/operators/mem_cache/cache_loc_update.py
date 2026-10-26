"""Public cache_loc_update operator."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("mem_cache.cache_loc_update", "auto")
@register_impl("mem_cache.cache_loc_update", "ascendc")
def _ascendc_impl(
    req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc
):
    import torch

    return torch.ops.npu.cache_loc_update(
        req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc
    )


@register_impl("mem_cache.cache_loc_update", "torch_native")
def _torch_native_impl(
    req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc
):
    from ...implementation.torch_native.mem_cache.cache_locations import (
        cache_loc_update,
    )
    from ._cache_locations import prepare_cache_locations

    intervals = prepare_cache_locations(
        req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc
    )
    cache_loc_update(token_pool, out_cache_loc.reshape(-1), intervals)
    return out_cache_loc


@register_operator("mem_cache.cache_loc_update")
def cache_loc_update(
    req_pool_indices,
    token_pool,
    start_offset,
    end_offset,
    out_cache_loc,
    *,
    impl="auto"
):
    """Select ascendc by default, or torch_native for the reference."""
    return get_impl_by_name("mem_cache.cache_loc_update", impl)(
        req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc
    )
