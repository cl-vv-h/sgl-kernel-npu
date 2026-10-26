"""Mamba state movement between layer-first and host-slot-first storage."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ._types import TransferDirection, direction_value


@register_impl("data_move.transfer_mamba_state", "auto")
@register_impl("data_move.transfer_mamba_state", "csrc")
def _csrc_impl(
    device_buf, host_buf, device_indices, host_indices, direction=TransferDirection.H2D
):
    import torch

    return torch.ops.npu.transfer_mamba_state(
        device_buf, host_buf, device_indices, host_indices, direction_value(direction)
    )


@register_impl("data_move.transfer_mamba_state", "torch_native")
def _torch_native_impl(
    device_buf, host_buf, device_indices, host_indices, direction=TransferDirection.H2D
):
    from ...implementation.torch_native.data_move.state_transfer import mamba_state
    from ._transfer_utils import index_pairs, mamba_views

    direction = direction_value(direction)
    device, host = mamba_views(device_buf, host_buf)
    pairs = index_pairs(device_indices, host_indices, device.shape[1], host.shape[0])
    return mamba_state(device, host, pairs, direction)


@register_operator("data_move.transfer_mamba_state")
def transfer_mamba_state(
    device_buf,
    host_buf,
    device_indices,
    host_indices,
    direction=TransferDirection.H2D,
    *,
    impl="auto"
):
    """Copy selected slots in place; auto uses the original ACL extension."""
    return get_impl_by_name("data_move.transfer_mamba_state", impl)(
        device_buf, host_buf, device_indices, host_indices, direction
    )
