"""Cache-hit reference using the Python adapter's int32 outputs."""

import torch


def slot_map_lookup(slot_map, req, idx, valid, pos):
    pos[valid] = slot_map[req[valid], idx[valid]]
    return (pos >= 0).to(torch.int32), pos
