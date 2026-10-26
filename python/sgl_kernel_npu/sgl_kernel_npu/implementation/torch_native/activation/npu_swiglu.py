import torch.nn.functional as F


def npu_swiglu(input, dim=-1):
    if input.shape[dim] % 2:
        raise ValueError("SwiGLU split dimension must be even")
    gate, up = input.float().chunk(2, dim=dim)
    return (F.silu(gate) * up).to(input.dtype)
