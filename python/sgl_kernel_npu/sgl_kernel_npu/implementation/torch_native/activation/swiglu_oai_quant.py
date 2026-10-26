"""Reference preserving capacity buffers and the optional int8 epilogue."""

import torch


def swiglu_oai_quant(flat, alpha, limit, need_quant, valid, out, scales):
    if valid:
        gate, up = flat[:valid].float().chunk(2, -1)
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
        value = gate * torch.sigmoid(alpha * gate) * (up + 1.0)
        if need_quant:
            scale = value.abs().amax(-1) / 127
            quant = (value / scale[:, None]).to(flat.dtype).float()
            quant = (
                torch.nan_to_num(quant, nan=0.0).trunc().clamp(-128, 127).to(torch.int8)
            )
            out[:valid].copy_(quant)
            scales[:valid].copy_(scale)
        else:
            out[:valid].copy_(value.to(flat.dtype))
    return out, scales
