"""Transfer all layers of each state component from device slots to host slots."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ._types import TransferFlag, flags_value


@register_impl("data_move.submit_state_all_layer_d2h", "auto")
@register_impl("data_move.submit_state_all_layer_d2h", "csrc")
def _csrc_impl(
    device_states, host_states, device_indices, host_indices, flags=TransferFlag.FAST2D
):
    import torch

    return torch.ops.npu.transfer_state_all_layer_direct_lf_pf(
        list(device_states),
        list(host_states),
        device_indices,
        host_indices,
        flags_value(flags),
    )


@register_impl("data_move.submit_state_all_layer_d2h", "torch_native")
def _torch_native_impl(
    device_states, host_states, device_indices, host_indices, flags=TransferFlag.FAST2D
):
    from ...implementation.torch_native.data_move.state_transfer import all_layer_d2h
    from ._transfer_utils import all_layer_views, index_pairs, validate_pairs

    flags_value(flags)
    device_states, host_states = list(device_states), list(host_states)
    if not device_states or len(device_states) != len(host_states):
        raise ValueError(
            "device_states and host_states must be nonempty matching component lists"
        )
    pairs = index_pairs(device_indices, host_indices)
    if not pairs:
        return None
    views = [all_layer_views(d, h) for d, h in zip(device_states, host_states)]
    layers = views[0][0].shape[0]
    if any(d.shape[0] != layers for d, _ in views):
        raise ValueError("state components must have the same layer count")
    validate_pairs(
        pairs, min(d.shape[1] for d, _ in views), min(h.shape[0] for _, h in views)
    )
    return all_layer_d2h(views, pairs)


@register_operator("data_move.submit_state_all_layer_d2h")
def submit_state_all_layer_d2h(
    device_states,
    host_states,
    device_indices,
    host_indices,
    flags=TransferFlag.FAST2D,
    *,
    impl="auto"
):
    """Enqueue in-place D2H copies; the caller owns stream synchronization."""
    return get_impl_by_name("data_move.submit_state_all_layer_d2h", impl)(
        device_states, host_states, device_indices, host_indices, flags
    )


# Preserve the historical public spelling without creating a second registry entry.
transfer_state_all_layer_direct_lf_pf = submit_state_all_layer_d2h
