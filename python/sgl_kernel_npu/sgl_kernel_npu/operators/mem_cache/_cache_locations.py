"""Input preparation shared by the two cache-location reference adapters."""

import torch


def prepare_cache_locations(
    req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc
):
    if (
        req_pool_indices.dtype not in (torch.int32, torch.int64)
        or token_pool.dtype != torch.int32
        or out_cache_loc.dtype != torch.int32
        or start_offset.dtype != torch.int64
        or end_offset.dtype != torch.int64
    ):
        raise ValueError(
            "expected int32/int64 request indices, int32 pools, and int64 offsets"
        )
    if (
        token_pool.ndim != 2
        or req_pool_indices.ndim != 1
        or start_offset.ndim != 1
        or end_offset.ndim != 1
    ):
        raise ValueError("expected a 2-D token pool and 1-D indices/offsets")
    if (
        req_pool_indices.numel() != start_offset.numel()
        or req_pool_indices.numel() != end_offset.numel()
    ):
        raise ValueError("request indices and offsets must have equal lengths")
    if not token_pool.is_contiguous() or not out_cache_loc.is_contiguous():
        raise ValueError("token_pool and out_cache_loc must be contiguous")
    intervals = []
    offset = 0
    for row, start, end in zip(
        req_pool_indices.cpu().tolist(),
        start_offset.cpu().tolist(),
        end_offset.cpu().tolist(),
    ):
        if (
            not 0 <= row < token_pool.shape[0]
            or not 0 <= start <= end <= token_pool.shape[1]
        ):
            raise ValueError("request row or token interval is outside token_pool")
        intervals.append((row, start, end, offset))
        offset += end - start
    if offset > out_cache_loc.numel():
        raise ValueError("out_cache_loc is smaller than the packed token intervals")
    return intervals
