"""PyTorch reference for packed FP8 values, BF16 RoPE bytes and scale storage."""

import torch

from .._quantization import blocks, expanded, fp8_coefficient, scale_values


def kv_compress_epilog(
    kv_compress_cache,
    x,
    slot_mapping,
    quant_group_size,
    quant_mode,
    round_scale_flag,
    layout,
):
    if x.ndim != 2 or x.dtype != torch.bfloat16 or x.shape[1] <= 64:
        raise ValueError(
            "x must be a BF16 matrix with a quantized prefix and 64 RoPE columns"
        )
    if quant_mode not in (1, 2) or layout not in (1, 2):
        raise ValueError("quant_mode and layout must be 1 or 2")
    if layout == 2 and quant_mode != 2:
        raise ValueError("layout=2 only supports MXFP8")
    if quant_mode == 2 and quant_group_size not in (64, 128):
        raise ValueError("MXFP8 quant_group_size must be 64 or 128")
    cache = (
        kv_compress_cache.squeeze(2)
        if kv_compress_cache.ndim == 4
        else kv_compress_cache
    )
    dtype = cache.dtype
    if dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("cache must use FP8 E4M3FN or E5M2")
    rows, cols = x.shape[0], x.shape[1] - 64
    width = 128 if quant_mode == 1 else quant_group_size
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
    # A byte view preserves caller-owned storage and the native row/block strides.
    storage = cache.view(torch.uint8)
    if not cache.is_contiguous():
        raise NotImplementedError(
            "torch_native packed-cache reference requires contiguous storage"
        )
    if slot_mapping.numel() != rows:
        raise ValueError("slot_mapping length must match x rows")
    if layout == 1:
        target = storage.reshape(cache.shape[0], -1)
        contents = (
            torch.cat((quant_bytes, rope_bytes, scale_bytes), -1)
            if quant_mode == 1
            else torch.cat((rope_bytes, quant_bytes, scale_bytes), -1)
        )
        padded_width = ((contents.shape[-1] + 127) // 128) * 128
        if target.shape[-1] < padded_width:
            raise ValueError("cache row is too short for the packed layout")
        for i, slot in enumerate(slot_mapping.tolist()):
            if slot == -1:
                continue
            if not 0 <= slot < target.shape[0]:
                raise ValueError("slot_mapping index is out of bounds")
            target[slot, : contents.shape[-1]].copy_(contents[i])
            target[slot, contents.shape[-1] : padded_width].zero_()
    else:
        block_size = cache.shape[1]
        target = storage.reshape(cache.shape[0], -1)
        value_width = cols + 128
        scale_width = ((scale_bytes.shape[-1] + 7) // 8) * 8
        if target.shape[-1] < block_size * (value_width + scale_width):
            raise ValueError("cache block is too short for the packed layout")
        for i, slot in enumerate(slot_mapping.tolist()):
            if slot == -1:
                continue
            if not 0 <= slot < cache.shape[0] * block_size:
                raise ValueError("slot_mapping index is out of bounds")
            block, offset = divmod(slot, block_size)
            begin = offset * value_width
            target[block, begin : begin + value_width].copy_(
                torch.cat((quant_bytes[i], rope_bytes[i]))
            )
            begin = block_size * value_width + offset * scale_width
            target[block, begin : begin + scale_bytes.shape[-1]].copy_(scale_bytes[i])
            target[block, begin + scale_bytes.shape[-1] : begin + scale_width].zero_()
