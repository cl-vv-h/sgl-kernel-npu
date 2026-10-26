"""Public npu_prefetch operator with explicit implementation selection."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl(
    "data_move.npu_prefetch", "auto", supported_generations=[Generation.ARCH22]
)
@register_impl(
    "data_move.npu_prefetch", "torch_npu", supported_generations=[Generation.ARCH22]
)
def _torch_npu_impl(input, dependency, max_size, offset=0):
    import torch_npu

    return torch_npu.npu_prefetch(input, dependency, max_size, offset)


@register_impl("data_move.npu_prefetch", "torch_native")
def _torch_native_impl(input, dependency, max_size, offset=0):
    from ...implementation.torch_native.data_move.npu_prefetch import npu_prefetch

    return npu_prefetch(input, dependency, max_size, offset)


@register_operator("data_move.npu_prefetch")
def npu_prefetch(input, dependency, max_size, offset=0, *, impl="auto"):
    """Dispatch to torch_npu by default; use impl="torch_native" for the reference.

    torch_native cannot issue an NPU cache-management operation and raises NotImplementedError.
    """
    return get_impl_by_name("data_move.npu_prefetch", impl)(
        input, dependency, max_size, offset
    )
