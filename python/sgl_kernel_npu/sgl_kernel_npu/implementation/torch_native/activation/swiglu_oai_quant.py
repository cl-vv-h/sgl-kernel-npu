"""Reference preserving capacity buffers and the optional int8 epilogue."""

import torch


def swiglu_oai_quant(
    x, alpha, limit, need_quant=True, group_list=None, group_list_type=None
):
    flat = x.reshape(-1, x.shape[-1])
    rows, columns = flat.shape
    if columns % 2:
        raise ValueError("the last dimension must contain equally sized gate/up halves")
    valid = rows
    if group_list is not None:
        if group_list_type not in (0, 1):
            raise ValueError(f"group_list_type must be 0 or 1, got {group_list_type}")
        if group_list.dtype not in (torch.int32, torch.int64):
            raise ValueError("group_list dtype must be torch.int32 or torch.int64")
        count = group_list[-1] if group_list_type == 0 else group_list.sum()
        valid = int(count.item())
        if not 0 <= valid <= rows:
            raise ValueError("group_list row count exceeds input capacity")
    out = torch.empty(
        (rows, columns // 2),
        dtype=torch.int8 if need_quant else x.dtype,
        device=x.device,
    )
    scales = torch.empty(rows, dtype=torch.float32, device=x.device)
    if valid:
        gate, up = flat[:valid].float().chunk(2, -1)
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
        value = gate * torch.sigmoid(alpha * gate) * (up + 1.0)
        if need_quant:
            scale = value.abs().amax(-1) / 127
            quant = (value / scale[:, None]).to(x.dtype).float()
            quant = (
                torch.nan_to_num(quant, nan=0.0).trunc().clamp(-128, 127).to(torch.int8)
            )
            out[:valid].copy_(quant)
            scales[:valid].copy_(scale)
        else:
            out[:valid].copy_(value.to(x.dtype))
    return out.reshape(*x.shape[:-1], columns // 2), scales
