"""Eager reference for the three existing SwiGLU group-quantization modes."""

import torch

from .._quantization import blocks, expanded, fp8_coefficient, mx_quantize


def swiglu_group_quant(
    x,
    topk_weight,
    quant_mode,
    round_scale,
    ue8m0_scale,
    output_origin,
    clamp_value,
    *,
    dtype,
    rows,
    cols,
    valid,
    y,
    scale,
    origin
):
    if valid == 0:
        return y, scale, origin
    gate, up = x[:valid].float().chunk(2, -1)
    if clamp_value != 0:
        gate, up = gate.clamp(max=clamp_value), up.clamp(-clamp_value, clamp_value)
    value = gate / (1.0 + torch.exp(-gate)) * up
    if topk_weight is not None:
        value = value * topk_weight.float()
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
