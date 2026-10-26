"""Page-based KV transfer with optional indexer data and scale caches."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ._types import TransferDirection, TransferFlag, direction_value, flags_value


def _buffers(
    device_k,
    host_k,
    device_v,
    host_v,
    device_index_k,
    host_index_k,
    device_index_k_scale,
    host_index_k_scale,
):
    import torch

    # The native host reads size(4) even for an absent V; give it a valid empty 5-D view.
    if device_v.numel() == 0:
        device_v = device_v.reshape(0, 0, 0, 0, 0)
    if host_v.numel() == 0:
        host_v = host_v.reshape(0, 0, 0, 0, 0)
    buffers = [(device_k, host_k, device_v, host_v)]
    if device_index_k is not None and host_index_k is not None:
        empty = torch.empty((0, 0, 0, 0, 0))
        buffers.append((device_index_k, host_index_k, empty, empty))
    if device_index_k_scale is not None and host_index_k_scale is not None:
        if device_index_k_scale.ndim == 4:
            device_index_k_scale = device_index_k_scale.unsqueeze(-1)
        empty = torch.empty((0, 0, 0, 0, 0))
        buffers.append((device_index_k_scale, host_index_k_scale, empty, empty))
    return buffers


@register_impl("data_move.transfer_kv_dim_exchange", "auto")
@register_impl("data_move.transfer_kv_dim_exchange", "csrc")
def _csrc_impl(
    device_indices,
    host_indices,
    device_k,
    host_k,
    device_v,
    host_v,
    device_index_k=None,
    host_index_k=None,
    device_index_k_scale=None,
    host_index_k_scale=None,
    page_size=128,
    direction=TransferDirection.H2D,
    flags=TransferFlag.FAST2D,
):
    import torch

    direction, flags = direction_value(direction), flags_value(flags)
    buffers = _buffers(
        device_k,
        host_k,
        device_v,
        host_v,
        device_index_k,
        host_index_k,
        device_index_k_scale,
        host_index_k_scale,
    )
    for dk, hk, dv, hv in buffers:
        torch.ops.npu.transfer_kv_dim_exchange(
            dk, hk, dv, hv, device_indices, host_indices, page_size, direction, flags
        )


@register_impl("data_move.transfer_kv_dim_exchange", "torch_native")
def _torch_native_impl(
    device_indices,
    host_indices,
    device_k,
    host_k,
    device_v,
    host_v,
    device_index_k=None,
    host_index_k=None,
    device_index_k_scale=None,
    host_index_k_scale=None,
    page_size=128,
    direction=TransferDirection.H2D,
    flags=TransferFlag.FAST2D,
):
    from ...implementation.torch_native.data_move.transfer_kv_dim_exchange import (
        transfer_kv_dim_exchange,
    )
    from ._transfer_utils import index_pairs, kv_views, validate_pairs

    direction = direction_value(direction)
    flags_value(flags)
    if not isinstance(page_size, int) or page_size <= 0:
        raise ValueError("page_size must be a positive integer")
    if (
        device_indices.ndim != 1
        or host_indices.ndim != 1
        or device_indices.numel() % page_size
    ):
        raise ValueError("KV token indices must be 1-D and contain complete pages")
    tokens = index_pairs(device_indices, host_indices)
    # The ACL implementation uses the first token index of each page-sized group.
    if any(d < 0 or h < 0 for d, h in tokens[::page_size]):
        raise ValueError("page-start token indices must be nonnegative")
    pages = [(d // page_size, h // page_size) for d, h in tokens[::page_size]]
    buffers = _buffers(
        device_k,
        host_k,
        device_v,
        host_v,
        device_index_k,
        host_index_k,
        device_index_k_scale,
        host_index_k_scale,
    )
    views = []
    for dk, hk, dv, hv in buffers:
        views.append(kv_views(dk, hk, page_size))
        validate_pairs(pages, dk.shape[1], hk.shape[0])
        if dv.numel() and hv.numel():
            views.append(kv_views(dv, hv, page_size))
            if (
                dv.dtype != dk.dtype
                or dv.shape[:4] != dk.shape[:4]
                or hv.shape[:4] != hk.shape[:4]
            ):
                raise ValueError(
                    "V buffers must share K's dtype, heads, layers and page counts"
                )
    return transfer_kv_dim_exchange(views, pages, direction)


@register_operator("data_move.transfer_kv_dim_exchange")
def transfer_kv_dim_exchange(
    device_indices,
    host_indices,
    device_k,
    host_k,
    device_v,
    host_v,
    device_index_k=None,
    host_index_k=None,
    device_index_k_scale=None,
    host_index_k_scale=None,
    page_size=128,
    direction=TransferDirection.H2D,
    flags=TransferFlag.FAST2D,
    *,
    impl="auto"
):
    """Keep the historical Python signature; choose csrc or torch_native with impl."""
    return get_impl_by_name("data_move.transfer_kv_dim_exchange", impl)(
        device_indices,
        host_indices,
        device_k,
        host_k,
        device_v,
        host_v,
        device_index_k,
        host_index_k,
        device_index_k_scale,
        host_index_k_scale,
        page_size,
        direction,
        flags,
    )
