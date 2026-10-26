"""Tensor-backed indexed copies; opaque device pointers require the native kernel."""

from math import prod


def unidex_copy(
    src,
    dst,
    src_index,
    dst_index,
    valid_mask,
    src_address_ndims,
    dst_address_ndims,
    block_dim=8,
    src_ptr=None,
    dst_ptr=None,
):
    if src_ptr not in (None, src.data_ptr()) or dst_ptr not in (None, dst.data_ptr()):
        raise NotImplementedError(
            "torch_native cannot dereference device-visible shared-memory pointers; use ascendc"
        )
    for name, tensor, dims in (
        ("src", src, src_address_ndims),
        ("dst", dst, dst_address_ndims),
    ):
        if dims <= 0 or dims >= tensor.ndim:
            raise ValueError(f"invalid {name}_address_ndims: {dims}")
    if src.dtype != dst.dtype:
        raise ValueError("src and dst must have the same dtype")
    width = prod(src.shape[src_address_ndims:])
    if width != prod(dst.shape[dst_address_ndims:]):
        raise ValueError("src and dst logical rows must have the same byte size")
    if not (src_index.numel() == dst_index.numel() == valid_mask.numel()):
        raise ValueError(
            "src_index, dst_index, and valid_mask must have the same length"
        )
    source = src.view(-1, width)
    target = dst.view(-1, width)
    si, di = src_index.reshape(-1).long(), dst_index.reshape(-1).long()
    mask = valid_mask.reshape(-1).bool()
    mask = (
        mask & (si >= 0) & (si < source.shape[0]) & (di >= 0) & (di < target.shape[0])
    )
    target[di[mask].to(dst.device)] = source[si[mask].to(src.device)].to(dst.device)
    return dst
