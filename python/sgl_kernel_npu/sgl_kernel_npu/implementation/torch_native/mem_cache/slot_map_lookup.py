"""Cache-hit reference using the Python adapter's int32 outputs."""

import torch


def slot_map_lookup(slot_map, req_indices, topk_indices, block_dim=0):
    if slot_map.dtype != torch.int32:
        raise ValueError("slot_map must be int32")
    if topk_indices.shape[-1] != 2048:
        raise ValueError("slot_map_lookup requires topk=2048")
    pos = torch.full_like(topk_indices, -1, dtype=torch.int32)
    req = req_indices.long().reshape(-1, 1).expand_as(topk_indices)
    idx = topk_indices.long()
    valid = (
        (req >= 0) & (req < slot_map.shape[0]) & (idx >= 0) & (idx < slot_map.shape[1])
    )
    pos[valid] = slot_map[req[valid], idx[valid]]
    return (pos >= 0).to(torch.int32), pos
