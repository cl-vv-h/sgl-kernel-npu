"""PyTorch reference for packed FP8 values, BF16 RoPE bytes and scale storage."""

import torch

from .._quantization import blocks, expanded, fp8_coefficient, scale_values


def kv_compress_epilog(
    x,
    target,
    slots,
    quant_mode,
    round_scale_flag,
    layout,
    *,
    dtype,
    cols,
    width,
    block_size,
    value_width,
    scale_width,
    padded_width
):
    maxima = blocks(x[:, :cols].float(), width).abs().amax(-1)
    if quant_mode == 1:
        scales = maxima * fp8_coefficient(dtype)
        divisor = expanded(scales, width, cols)
        scaled = x[:, :cols].float() / torch.where(divisor == 0, 1.0, divisor)
        scale_bytes = scales.contiguous().view(torch.uint8)
    else:
        ratio = maxima.clamp_min(1e-4) * (1.0 / torch.finfo(dtype).max)
        exponents = (ratio.contiguous().view(torch.int32) >> 23) & 255
        if round_scale_flag:
            exponents = exponents + (exponents != 0).int()
        scale_bytes = exponents.to(torch.uint8)
        scaled = x[:, :cols].float() / expanded(scale_values(scale_bytes), width, cols)
    quant_bytes = scaled.to(dtype).contiguous().view(torch.uint8)
    rope_bytes = x[:, -64:].contiguous().view(torch.uint8)
    if layout == 1:
        contents = (
            torch.cat((quant_bytes, rope_bytes, scale_bytes), -1)
            if quant_mode == 1
            else torch.cat((rope_bytes, quant_bytes, scale_bytes), -1)
        )
        for i, slot in enumerate(slots):
            if slot == -1:
                continue
            target[slot, : contents.shape[-1]].copy_(contents[i])
            target[slot, contents.shape[-1] : padded_width].zero_()
    else:
        for i, slot in enumerate(slots):
            if slot == -1:
                continue
            block, offset = divmod(slot, block_size)
            begin = offset * value_width
            target[block, begin : begin + value_width].copy_(
                torch.cat((quant_bytes[i], rope_bytes[i]))
            )
            begin = block_size * value_width + offset * scale_width
            target[block, begin : begin + scale_bytes.shape[-1]].copy_(scale_bytes[i])
            target[block, begin + scale_bytes.shape[-1] : begin + scale_width].zero_()
