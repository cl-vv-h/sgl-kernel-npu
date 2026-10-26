import torch


def fused_sigmoid_mul(x, gate):
    return (x.float() * torch.sigmoid(gate.float())).to(x.dtype).contiguous()
