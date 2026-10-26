import torch.nn.functional as F


def npu_swiglu(input, dim):
    gate, up = input.float().chunk(2, dim=dim)
    return (F.silu(gate) * up).to(input.dtype)
