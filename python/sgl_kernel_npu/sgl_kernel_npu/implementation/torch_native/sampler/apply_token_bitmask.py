import torch


def apply_token_bitmask(logits, bitmask, indices=None):
    rows = (
        torch.arange(logits.shape[0], device=logits.device)
        if indices is None or indices.numel() == 0
        else indices.long()
    )
    cols = torch.arange(logits.shape[1], device=logits.device)
    words = bitmask[rows[:, None], cols[None, :] // 32].to(torch.int64)
    allowed = ((words >> (cols % 32)) & 1).bool()
    logits[rows] = logits[rows].masked_fill(~allowed, float("-inf"))
    return logits
