"""Restore one layer of host state into device slots."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ._types import TransferFlag, flags_value


@register_impl("data_move.submit_state_per_layer_h2d", "auto")
@register_impl("data_move.submit_state_per_layer_h2d", "csrc")
def _csrc_impl(src, dst, src_indices, dst_indices, layer_id, flags=TransferFlag.FAST2D):
    import torch

    return torch.ops.npu.transfer_state_per_layer_direct_pf_lf(
        src, dst, src_indices, dst_indices, layer_id, flags_value(flags)
    )


@register_impl("data_move.submit_state_per_layer_h2d", "torch_native")
def _torch_native_impl(
    src, dst, src_indices, dst_indices, layer_id, flags=TransferFlag.FAST2D
):
    from ...implementation.torch_native.data_move.state_transfer import per_layer_h2d
    from ._transfer_utils import index_pairs, per_layer_views, validate_pairs

    flags_value(flags)
    pairs = index_pairs(dst_indices, src_indices)
    if not pairs:
        return None
    source, destination = per_layer_views(src, dst, layer_id)
    validate_pairs(pairs, destination.shape[0], source.shape[0])
    return per_layer_h2d(source, destination, pairs)


@register_operator("data_move.submit_state_per_layer_h2d")
def submit_state_per_layer_h2d(
    src,
    dst,
    src_indices,
    dst_indices,
    layer_id,
    flags=TransferFlag.FAST2D,
    *,
    impl="auto"
):
    """Enqueue in-place H2D copies for layer_id; caller owns synchronization."""
    return get_impl_by_name("data_move.submit_state_per_layer_h2d", impl)(
        src, dst, src_indices, dst_indices, layer_id, flags
    )


transfer_state_per_layer_direct_pf_lf = submit_state_per_layer_h2d
