import torch.nn.functional as F


def gelu(input, approximate="none"):
    return F.gelu(input, approximate=approximate)
