"""Public npu_reduce_mean_batch_invariant operator with explicit implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("tensor_ops.npu_reduce_mean_batch_invariant", "auto")
@register_impl("tensor_ops.npu_reduce_mean_batch_invariant", "batch_invariant_ops")
def _batch_invariant_ops_impl(input, dim, keepdim=False):
    import batch_invariant_ops  # noqa: F401 -- registers the dispatcher namespace.
    import torch

    return torch.ops.batch_invariant_ops.npu_reduce_mean_batch_invariant(
        input, dim, keepdim
    )


@register_impl("tensor_ops.npu_reduce_mean_batch_invariant", "torch_native")
def _torch_native_impl(input, dim, keepdim=False):
    from ...implementation.torch_native.tensor_ops.npu_reduce_mean_batch_invariant import (
        npu_reduce_mean_batch_invariant,
    )

    return npu_reduce_mean_batch_invariant(input, dim, keepdim)


@register_operator("tensor_ops.npu_reduce_mean_batch_invariant")
def npu_reduce_mean_batch_invariant(input, dim, keepdim=False, *, impl="auto"):
    """Dispatch to batch_invariant_ops by default; use impl="torch_native" for the reference.

    Preserve the existing input, output, and in-place behavior.
    """
    return get_impl_by_name("tensor_ops.npu_reduce_mean_batch_invariant", impl)(
        input, dim, keepdim
    )
