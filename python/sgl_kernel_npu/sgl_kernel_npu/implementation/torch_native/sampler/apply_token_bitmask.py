import torch


def apply_token_bitmask(logits, bitmask, rows, cols):
    words = bitmask[rows[:, None], cols[None, :] // 32].to(torch.int64)
    allowed = ((words >> (cols % 32)) & 1).bool()
    logits[rows] = logits[rows].masked_fill(~allowed, float("-inf"))
    return logits
