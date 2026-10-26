import torch


def fused_sigmoid_mul_broadcast(x, gate):
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D, got ndim={x.ndim}, shape={x.shape}")
    if gate.shape not in ((x.shape[0],), (x.shape[0], 1)):
        raise ValueError(
            f"gate must be ({x.shape[0]},) or ({x.shape[0]}, 1), got {gate.shape}"
        )
    return (
        (x.float() * torch.sigmoid(gate.float().reshape(-1, 1)))
        .to(x.dtype)
        .contiguous()
    )
