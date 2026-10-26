"""Packed cache-location scatter/gather, with the native in-place return identities."""


def cache_loc_assign(token_pool, out_cache_loc, intervals):
    for row, start, end, offset in intervals:
        token_pool[row, start:end].copy_(out_cache_loc[offset : offset + end - start])
    return token_pool


def cache_loc_update(token_pool, out_cache_loc, intervals):
    for row, start, end, offset in intervals:
        out_cache_loc[offset : offset + end - start].copy_(token_pool[row, start:end])
    return out_cache_loc
