import torch


def npu_fast_gelu(input):
    return (input.float() * torch.sigmoid(1.702 * input.float())).to(input.dtype)
