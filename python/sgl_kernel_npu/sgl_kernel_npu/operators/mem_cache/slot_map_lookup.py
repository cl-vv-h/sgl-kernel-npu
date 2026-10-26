"""Public slot_map_lookup operator with explicit implementation selection."""

from typing import Tuple

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl(
    "mem_cache.slot_map_lookup", "auto", supported_generations=[Generation.ARCH22]
)
@register_impl(
    "mem_cache.slot_map_lookup", "ascendc", supported_generations=[Generation.ARCH22]
)
def _ascendc_impl(
    slot_map: torch.Tensor,
    req_indices: torch.Tensor,
    topk_indices: torch.Tensor,
    block_dim: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return cache-hit flags and slot positions for ``topk_indices``."""
    token_on_device = torch.empty_like(topk_indices, dtype=torch.int32)
    device_token_pos = torch.empty_like(topk_indices, dtype=torch.int32)
    torch.ops.npu.slot_map_lookup(
        slot_map,
        req_indices,
        topk_indices,
        token_on_device,
        device_token_pos,
        block_dim,
    )
    return token_on_device, device_token_pos


@register_impl("mem_cache.slot_map_lookup", "torch_native")
def _torch_native_impl(slot_map, req_indices, topk_indices, block_dim=0):
    from ...implementation.torch_native.mem_cache.slot_map_lookup import slot_map_lookup

    return slot_map_lookup(slot_map, req_indices, topk_indices, block_dim)


@register_operator("mem_cache.slot_map_lookup")
def slot_map_lookup(slot_map, req_indices, topk_indices, block_dim=0, *, impl="auto"):
    """Dispatch to ascendc by default; use impl="torch_native" for the reference.

    Allocates and returns two int32 tensors, matching the existing Python adapter.
    """
    return get_impl_by_name("mem_cache.slot_map_lookup", impl)(
        slot_map, req_indices, topk_indices, block_dim
    )
