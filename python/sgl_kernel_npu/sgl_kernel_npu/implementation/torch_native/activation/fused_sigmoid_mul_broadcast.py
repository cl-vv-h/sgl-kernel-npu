import torch


def fused_sigmoid_mul_broadcast(x, gate):
    return (x.float() * torch.sigmoid(gate.float())).to(x.dtype).contiguous()
