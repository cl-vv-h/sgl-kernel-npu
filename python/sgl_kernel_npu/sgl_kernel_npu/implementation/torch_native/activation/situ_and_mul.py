import torch


def situ_and_mul(x, out, beta, inv_beta, linear_beta, inv_linear_beta):
    gate, up = x.float().chunk(2, dim=-1)
    activated = beta * torch.tanh(gate * inv_beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up * inv_linear_beta)
    out.copy_((activated * up).to(out.dtype))
