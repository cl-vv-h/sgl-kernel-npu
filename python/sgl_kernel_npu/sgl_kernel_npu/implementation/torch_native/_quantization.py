"""PyTorch-only helpers for the existing FP8 reference contracts.

Constants and exponent arithmetic follow the corresponding AscendC kernels.
These eager references may synchronize for group lengths and are not graph kernels.
"""

import struct

import torch
import torch.nn.functional as F


def fp8_coefficient(dtype):
    bits = 0x37924925 if dtype == torch.float8_e5m2 else 0x3B124925
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def blocks(value, size):
    return F.pad(value, (0, (-value.shape[-1]) % size)).reshape(
        value.shape[0], (value.shape[-1] + size - 1) // size, size
    )


def scale_values(scale_bytes):
    return torch.pow(2.0, scale_bytes.float() - 127)


def expanded(scales, size, columns):
    return scales.repeat_interleave(size, dim=-1)[..., :columns]


def mx_quantize(value, dtype, src_dtype, *, saturate=True):
    rounded = value.to(src_dtype)
    if src_dtype == torch.float16:
        fields = (blocks(rounded, 32).view(torch.int16).int() & 0x7C00) >> 10
        exponents = (fields + 112) << 7
        exponents = torch.where(fields == 0, 0, exponents)
        exponents = torch.where(fields == 31, 0x7F80, exponents)
    else:
        exponents = (blocks(rounded.float(), 32).view(torch.int32) >> 16) & 0x7F80
    lower = 0x0780 if dtype == torch.float8_e5m2 else 0x0400
    maximum = exponents.amax(-1)
    raw = ((maximum.clamp_min(lower) - lower) >> 7).to(torch.uint8)
    raw = torch.where(maximum == 0x7F80, 255, raw).to(torch.uint8)
    divisor = expanded(scale_values(raw), 32, value.shape[-1])
    quant = rounded.float() / divisor
    quant = torch.where(expanded(raw == 255, 32, value.shape[-1]), float("nan"), quant)
    if src_dtype == torch.bfloat16:
        quant = quant.to(src_dtype).float()
    if saturate:
        bound = torch.finfo(dtype).max
        quant = quant.clamp(-bound, bound)
    # The output scale storage is aligned to pairs of E8M0 bytes.
    packed = F.pad(raw, (0, raw.shape[-1] % 2)).contiguous().view(torch.float8_e8m0fnu)
    return quant.to(dtype), packed.reshape(raw.shape[0], -1, 2)
