"""Public gelu operator with explicit implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("mem_cache.gelu", "auto")
@register_impl("mem_cache.gelu", "torch_native")
def _torch_native_impl(input, approximate="none"):
    from ...implementation.torch_native.mem_cache.gelu import gelu

    return gelu(input, approximate)


@register_operator("mem_cache.gelu")
def gelu(input, approximate="none", *, impl="auto"):
    """Dispatch to torch_native by default; use impl="torch_native" for the reference.

    Preserve the existing input, output, and in-place behavior.
    """
    return get_impl_by_name("mem_cache.gelu", impl)(input, approximate)
