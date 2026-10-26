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
    import torch

    from ...implementation.torch_native.mem_cache.slot_map_lookup import slot_map_lookup

    if slot_map.dtype != torch.int32:
        raise ValueError("slot_map must be int32")
    if topk_indices.shape[-1] != 2048:
        raise ValueError("slot_map_lookup requires topk=2048")
    pos = torch.full_like(topk_indices, -1, dtype=torch.int32)
    req = req_indices.long().reshape(-1, 1).expand_as(topk_indices)
    idx = topk_indices.long()
    valid = (
        (req >= 0) & (req < slot_map.shape[0]) & (idx >= 0) & (idx < slot_map.shape[1])
    )

    return slot_map_lookup(slot_map, req, idx, valid, pos)


@register_operator("mem_cache.slot_map_lookup")
def slot_map_lookup(slot_map, req_indices, topk_indices, block_dim=0, *, impl="auto"):
    """Dispatch to ascendc by default; use impl="torch_native" for the reference.

    Allocates and returns two int32 tensors, matching the existing Python adapter.
    """
    return get_impl_by_name("mem_cache.slot_map_lookup", impl)(
        slot_map, req_indices, topk_indices, block_dim
    )
