"""SiTU followed by per-32-element MXFP8 quantization in ordinary PyTorch."""

import torch

from .._quantization import mx_quantize


def situ_mxfp8_quant(hidden_states, valid, beta, linear_beta, payload, scales):
    if valid:
        gate, up = hidden_states[:valid].float().chunk(2, dim=-1)
        result = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
        result = (result * (linear_beta * torch.tanh(up / linear_beta))).to(
            torch.bfloat16
        )
        quant, scale = mx_quantize(result, torch.float8_e4m3fn, torch.bfloat16)
        payload[:valid].copy_(quant)
        scales[:valid].copy_(scale)
    return payload, scales
