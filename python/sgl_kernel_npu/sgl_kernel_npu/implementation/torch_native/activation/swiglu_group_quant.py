"""Eager reference for the three existing SwiGLU group-quantization modes."""

from math import prod

import torch

from .._quantization import blocks, expanded, fp8_coefficient, mx_quantize


def swiglu_group_quant(
    x,
    topk_weight=None,
    group_index=None,
    dst_type=None,
    quant_mode=1,
    group_size=128,
    round_scale=False,
    ue8m0_scale=False,
    output_origin=False,
    group_list_type=0,
    clamp_value=0.0,
):
    dtype = torch.float8_e4m3fn if dst_type is None else dst_type
    if (
        x.ndim < 1
        or x.dtype not in (torch.float16, torch.bfloat16)
        or not x.is_contiguous()
    ):
        raise ValueError("x must be a contiguous FP16/BF16 tensor")
    if quant_mode not in (1, 2, 3) or group_list_type != 0:
        raise ValueError(
            "supported quant_mode values are 1/2/3; group_list_type must be 0"
        )
    if dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("dst_type must be FP8 E4M3FN or E5M2")
    alignment = 128 if quant_mode == 2 else 256
    if x.shape[-1] == 0 or x.shape[-1] % alignment:
        raise ValueError(f"last dimension must be a positive multiple of {alignment}")
    if quant_mode == 3 and ue8m0_scale and not round_scale:
        raise NotImplementedError(
            "torch_native requires round_scale=True for a coherent E8M0 mode-3 scale buffer"
        )
    rows, cols = prod(x.shape[:-1]), x.shape[-1] // 2
    valid = rows
    if group_index is not None and group_index.numel():
        # VFProcessGroupIndex sums the entries; do not substitute the last entry.
        valid = max(0, min(rows, int(group_index.sum().item())))
    y = torch.empty((*x.shape[:-1], cols), dtype=dtype, device=x.device)
    origin = torch.empty((*x.shape[:-1], cols), dtype=x.dtype, device=x.device)
    scale_shape = (
        (*x.shape[:-1], (cols + 63) // 64, 2)
        if quant_mode == 2
        else (*x.shape[:-1], (cols + 127) // 128)
    )
    scale_dtype = (
        torch.float8_e8m0fnu
        if quant_mode == 2 or (quant_mode == 3 and ue8m0_scale)
        else torch.float32
    )
    scale = torch.empty(scale_shape, dtype=scale_dtype, device=x.device)
    if valid == 0:
        return y, scale, origin
    gate, up = x.reshape(rows, -1)[:valid].float().chunk(2, -1)
    if clamp_value != 0:
        gate, up = gate.clamp(max=clamp_value), up.clamp(-clamp_value, clamp_value)
    value = gate / (1.0 + torch.exp(-gate)) * up
    if topk_weight is not None and topk_weight.numel():
        value = value * topk_weight.reshape(-1)[:valid].float().unsqueeze(-1)
    if quant_mode == 2:
        quant, scales = mx_quantize(value, dtype, x.dtype)
    else:
        maxima = blocks(value, 128).abs().amax(-1)
        if quant_mode == 1:
            scales = maxima * fp8_coefficient(dtype)
        else:
            scales = maxima.clamp_min(1e-4) * fp8_coefficient(torch.float8_e4m3fn)
            if round_scale:
                bits = scales.contiguous().view(torch.int32)
                exponent = ((bits >> 23) & 255) - 127 + ((bits & 0x7FFFFF) != 0).int()
                scales = torch.pow(2.0, exponent.float())
        divisor = expanded(scales, 128, cols)
        quant = (value / torch.where(divisor == 0, 1.0, divisor)).to(dtype)
        if quant_mode == 3 and ue8m0_scale:
            scales = (
                ((scales.contiguous().view(torch.int32) >> 23) & 255)
                .to(torch.uint8)
                .view(torch.float8_e8m0fnu)
            )
    y.reshape(rows, cols)[:valid].copy_(quant)
    scale.reshape(rows, -1)[:valid].copy_(scales.reshape(valid, -1))
    # Only mode 3 writes y_origin; other modes allocate it but never fill it.
    if quant_mode == 3 and output_origin:
        origin.reshape(rows, cols)[:valid].copy_(value.to(x.dtype))
    return y, scale, origin
