"""Eager page-allocation reference; writes both caller-owned output buffers."""

import torch


def alloc_extend(
    pre_lens, seq_lens, last_loc, free_pages, page_size, out_indices, values
):
    output_offset = 0
    page_offset = 0
    for i in range(len(pre_lens)):
        pre, end = pre_lens[i], seq_lens[i]
        length = end - pre
        partial = min(length, (-pre) % page_size)
        out_indices[output_offset : output_offset + partial].copy_(
            last_loc[i] + 1 + torch.arange(partial, device=out_indices.device)
        )
        output_offset += partial
        remaining = length - partial
        pages = (remaining + page_size - 1) // page_size
        positions = torch.arange(remaining, device=out_indices.device)
        slots = free_pages[page_offset + positions // page_size] * page_size
        out_indices[output_offset : output_offset + remaining].copy_(
            slots + positions % page_size
        )
        output_offset += remaining
        page_offset += pages
    values.reshape(-1)[0] = page_offset
