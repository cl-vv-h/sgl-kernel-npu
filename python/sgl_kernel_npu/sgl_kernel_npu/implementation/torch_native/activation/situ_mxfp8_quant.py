"""SiTU followed by per-32-element MXFP8 quantization in ordinary PyTorch."""

import torch

from .._quantization import mx_quantize


def situ_mxfp8_quant(
    hidden_states, group_list, group_list_type=1, beta=4.0, linear_beta=25.0
):
    if hidden_states.ndim != 2 or hidden_states.shape[1] != 6144:
        raise ValueError(
            "situ_mxfp8_quant expects hidden_states shape [capacity, 6144]"
        )
    if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous BF16")
    if hidden_states.shape[0] == 0 or group_list.ndim != 1 or group_list.numel() == 0:
        raise ValueError("capacity and group_list must be nonempty")
    if group_list.dtype not in (torch.int32, torch.int64):
        raise ValueError("group_list must use int32 or int64")
    if group_list_type not in (0, 1) or beta <= 0 or linear_beta <= 0:
        raise ValueError("invalid group_list_type or SiTU coefficients")
    capacity = hidden_states.shape[0]
    count = group_list[-1] if group_list_type == 0 else group_list.sum()
    valid = max(0, min(capacity, int(count.item())))
    payload = torch.empty(
        (capacity, 3072), dtype=torch.float8_e4m3fn, device=hidden_states.device
    )
    scales = torch.empty(
        (capacity, 48, 2), dtype=torch.float8_e8m0fnu, device=hidden_states.device
    )
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
