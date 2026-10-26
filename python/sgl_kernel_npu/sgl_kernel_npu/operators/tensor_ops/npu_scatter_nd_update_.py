"""Public npu_scatter_nd_update_ operator with explicit implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("tensor_ops.npu_scatter_nd_update_", "auto")
@register_impl("tensor_ops.npu_scatter_nd_update_", "torch_npu")
def _torch_npu_impl(input, indices, updates):
    import torch_npu

    return torch_npu.npu_scatter_nd_update_(input, indices, updates)


@register_impl("tensor_ops.npu_scatter_nd_update_", "torch_native")
def _torch_native_impl(input, indices, updates):
    from ...implementation.torch_native.tensor_ops.npu_scatter_nd_update_ import (
        npu_scatter_nd_update_,
    )

    depth = indices.shape[-1]
    if depth == 0 or depth > input.ndim:
        raise ValueError("indices depth must be in [1, input.ndim]")
    coordinates = indices.long().reshape(-1, depth)
    index = tuple(coordinates[:, d] for d in range(depth))
    values = updates.reshape(coordinates.shape[0], *input.shape[depth:])
    return npu_scatter_nd_update_(input, index, values)


@register_operator("tensor_ops.npu_scatter_nd_update_")
def npu_scatter_nd_update_(input, indices, updates, *, impl="auto"):
    """Dispatch to torch_npu by default; use impl="torch_native" for the reference.

    Preserve the existing input, output, and in-place behavior.
    """
    return get_impl_by_name("tensor_ops.npu_scatter_nd_update_", impl)(
        input, indices, updates
    )
