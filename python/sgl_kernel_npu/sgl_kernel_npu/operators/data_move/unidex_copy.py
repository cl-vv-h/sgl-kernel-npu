"""Public unidex_copy operator with explicit implementation selection."""

from math import prod
from typing import Optional, Tuple

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


def _infer_rows_and_block_bytes(
    tensor: torch.Tensor, address_ndims: int, name: str
) -> Tuple[int, int]:
    if address_ndims <= 0 or address_ndims >= tensor.dim():
        raise ValueError(
            f"{name}_address_ndims must be in [1, {tensor.dim() - 1}], "
            f"got {address_ndims}"
        )

    rows = prod(tensor.shape[:address_ndims])
    block_elements = prod(tensor.shape[address_ndims:])
    return int(rows), int(block_elements * tensor.element_size())


@register_impl(
    "data_move.unidex_copy", "auto", supported_generations=[Generation.ARCH22]
)
@register_impl(
    "data_move.unidex_copy", "ascendc", supported_generations=[Generation.ARCH22]
)
def _ascendc_impl(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_index: torch.Tensor,
    dst_index: torch.Tensor,
    valid_mask: torch.Tensor,
    src_address_ndims: int,
    dst_address_ndims: int,
    block_dim: int = 8,
    src_ptr: Optional[int] = None,
    dst_ptr: Optional[int] = None,
) -> torch.Tensor:
    """Copy selected logical rows from ``src`` into ``dst`` in place.

    ``src_ptr`` and ``dst_ptr`` may override the Tensor addresses with
    device-visible shared-memory addresses. The caller owns those allocations
    and must keep them alive until work on the current NPU stream completes.
    """
    src_rows, src_block_bytes = _infer_rows_and_block_bytes(
        src, src_address_ndims, "src"
    )
    dst_rows, dst_block_bytes = _infer_rows_and_block_bytes(
        dst, dst_address_ndims, "dst"
    )
    if src_block_bytes != dst_block_bytes:
        raise ValueError(
            "src and dst logical rows must have the same byte size, got "
            f"{src_block_bytes} and {dst_block_bytes}"
        )
    if (
        src_index.numel() != dst_index.numel()
        or src_index.numel() != valid_mask.numel()
    ):
        raise ValueError(
            "src_index, dst_index, and valid_mask must have the same length"
        )
    if src.dtype != dst.dtype:
        raise ValueError(
            f"src and dst must have the same dtype, got {src.dtype} and {dst.dtype}"
        )

    torch.ops.npu.unidex_copy(
        src,
        dst,
        src_index,
        dst_index,
        valid_mask,
        src_rows,
        dst_rows,
        src_block_bytes,
        src_index.numel(),
        block_dim,
        src_ptr,
        dst_ptr,
    )
    return dst


@register_impl("data_move.unidex_copy", "torch_native")
def _torch_native_impl(
    src,
    dst,
    src_index,
    dst_index,
    valid_mask,
    src_address_ndims,
    dst_address_ndims,
    block_dim=8,
    src_ptr=None,
    dst_ptr=None,
):
    from ...implementation.torch_native.data_move.unidex_copy import unidex_copy

    return unidex_copy(
        src,
        dst,
        src_index,
        dst_index,
        valid_mask,
        src_address_ndims,
        dst_address_ndims,
        block_dim,
        src_ptr,
        dst_ptr,
    )


@register_operator("data_move.unidex_copy")
def unidex_copy(
    src,
    dst,
    src_index,
    dst_index,
    valid_mask,
    src_address_ndims,
    dst_address_ndims,
    block_dim=8,
    src_ptr=None,
    dst_ptr=None,
    *,
    impl="auto",
):
    """Dispatch to ascendc by default; use impl="torch_native" for the reference.

    Uses the existing unidex_copy_inplace Python contract and returns dst. The torch_native path requires tensor-backed addresses.
    """
    return get_impl_by_name("data_move.unidex_copy", impl)(
        src,
        dst,
        src_index,
        dst_index,
        valid_mask,
        src_address_ndims,
        dst_address_ndims,
        block_dim,
        src_ptr,
        dst_ptr,
    )


# Compatibility spelling; both names refer to one registered public function.
unidex_copy_inplace = unidex_copy
