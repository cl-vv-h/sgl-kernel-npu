"""Public apply_token_bitmask operator with explicit implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("sampler.apply_token_bitmask", "auto")
@register_impl("sampler.apply_token_bitmask", "ascendc")
def _ascendc_impl(logits, bitmask, indices=None):
    import torch

    return torch.ops.npu.apply_token_bitmask(logits, bitmask, indices)


@register_impl("sampler.apply_token_bitmask", "torch_native")
def _torch_native_impl(logits, bitmask, indices=None):
    import torch

    from ...implementation.torch_native.sampler.apply_token_bitmask import (
        apply_token_bitmask,
    )

    rows = (
        torch.arange(logits.shape[0], device=logits.device)
        if indices is None or indices.numel() == 0
        else indices.long()
    )
    cols = torch.arange(logits.shape[1], device=logits.device)

    return apply_token_bitmask(logits, bitmask, rows, cols)


@register_operator("sampler.apply_token_bitmask")
def apply_token_bitmask(logits, bitmask, indices=None, *, impl="auto"):
    """Dispatch to ascendc by default; use impl="torch_native" for the reference.

    Preserve the existing input, output, and in-place behavior.
    """
    return get_impl_by_name("sampler.apply_token_bitmask", impl)(
        logits, bitmask, indices
    )
