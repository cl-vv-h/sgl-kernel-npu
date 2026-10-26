"""SiTU gating with optional grouped token capacity."""

from typing import Optional

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator


def _validate_cumulative_storage(group_list, group_list_type):
    # The existing kernel reads the total at pointer + shape[0], not at shape[0]-1.
    # Require a backed sentinel instead of permitting an out-of-storage access.
    if group_list is not None and group_list_type == 0:
        if group_list.ndim != 1 or not group_list.is_contiguous():
            raise ValueError("cumulative group_list must be a contiguous 1-D tensor")
        required = group_list.storage_offset() + group_list.shape[0] + 1
        if (
            group_list.untyped_storage().nbytes() // group_list.element_size()
            < required
        ):
            raise ValueError(
                "cumulative group_list requires a backed total-row sentinel at shape[0]"
            )


@register_impl("activation.situ_and_mul", "auto")
@register_impl("activation.situ_and_mul", "triton")
def _triton_impl(
    x,
    group_list=None,
    group_list_type=None,
    beta: float = 4.0,
    linear_beta: Optional[float] = 25.0,
):
    """SituAndMul activation with optional MoE group_list.

    Args:
        x: ``[..., 2d]`` tensor (gate | up halves along the last dim).
        group_list: per-expert token counts (count) or cumulative sum (cusum).
            ``None`` = dense / shared-expert path: process ALL rows of ``x`` (no dispatch
            padding). Required only for the routed-MoE path.
        group_list_type: 0 = cusum, 1 = count. Ignored when ``group_list is None``.
        beta: SituAndMul beta (soft-saturation bound on the gate path).
        linear_beta: optional soft-saturation bound on the up path; ``None`` leaves ``up``.

    Returns:
        ``[..., d]`` tensor. With ``group_list``: only the first ``sum(group_list)`` rows
        are written (rest is padding). Without: all rows written.
    """
    from ...implementation.triton.activation.situ import _situ_and_mul_kernel
    from ...utils.triton_utils import get_device_properties

    _validate_cumulative_storage(group_list, group_list_type)
    has_group_list = group_list is not None
    if has_group_list and group_list_type not in (0, 1):
        raise ValueError(f"group_list_type must be 0 or 1, but got {group_list_type}")
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"x last dim must be even, but got {x.shape[-1]}")

    # The JIT addresses x as a flat, row-major [rows, 2d] buffer.
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    s, h = x_2d.shape
    out = torch.empty((s, h // 2), dtype=x.dtype, device=x.device)

    if has_group_list:
        num_experts = group_list.shape[0]
        if group_list.dtype == torch.int64:
            num_experts_algin = (num_experts + 7) // 8 * 8
        elif group_list.dtype == torch.int32:
            num_experts_algin = (num_experts + 15) // 16 * 16
        else:
            raise ValueError(
                f"group_list dtype must be torch.int32 or torch.int64, "
                f"but got {group_list.dtype}"
            )
        group_list_arg = group_list
        num_experts_arg = num_experts
        num_experts_algin_arg = num_experts_algin
        gl_type_arg = group_list_type
    else:
        # dense / shared: kernel skips the group_list block (HAS_GROUP_LIST=False),
        # so these are never read -- pass harmless dummies.
        group_list_arg = x_2d
        num_experts_arg = 1
        num_experts_algin_arg = 1
        gl_type_arg = 0

    do_linear_beta = linear_beta is not None
    linear_beta_v = linear_beta if do_linear_beta else 1.0

    _, num_vectorcore = get_device_properties()
    _situ_and_mul_kernel[(num_vectorcore,)](
        x_2d,
        group_list_arg,
        out,
        TOTAL_COLS=h,
        HALF_COLS=h // 2,
        NUM_EXPERTS=num_experts_arg,
        NUM_EXPERTS_ALGIN=num_experts_algin_arg,
        GROUP_LIST_TYPE=gl_type_arg,
        N_ROWS=s,
        NUM_CORES=num_vectorcore,
        HAS_GROUP_LIST=has_group_list,
        BETA=beta,
        INV_BETA=1.0 / beta,
        DO_LINEAR_BETA=do_linear_beta,
        LINEAR_BETA=linear_beta_v,
        INV_LINEAR_BETA=(1.0 / linear_beta_v) if do_linear_beta else 1.0,
    )
    return out.reshape(*x.shape[:-1], h // 2)


@register_impl("activation.situ_and_mul", "torch_native")
def _torch_native_impl(
    x, group_list=None, group_list_type=None, beta=4.0, linear_beta=25.0
):
    from ...implementation.torch_native.activation.situ_and_mul import situ_and_mul

    if x.ndim == 0 or x.shape[-1] == 0 or x.shape[-1] % 2:
        raise ValueError("x last dimension must be positive and even")
    _validate_cumulative_storage(group_list, group_list_type)
    x_2d = x.reshape(-1, x.shape[-1])
    rows = x_2d.shape[0]
    if group_list is not None:
        if group_list_type not in (0, 1):
            raise ValueError("group_list_type must be 0 or 1")
        if group_list.ndim != 1 or group_list.dtype not in (torch.int32, torch.int64):
            raise ValueError("group_list must be a 1-D int32 or int64 tensor")
        if not group_list.is_contiguous():
            raise ValueError("group_list must be contiguous")
        if group_list_type == 0:
            storage_view = group_list.as_strided((group_list.shape[0] + 1,), (1,))
            rows = int(storage_view[-1].item())
        else:
            rows = int(group_list.sum().item())
        if not 0 <= rows <= x_2d.shape[0]:
            raise ValueError("group_list total must fit within x's row capacity")
    inv_beta = 1.0 / beta
    inv_linear_beta = 1.0 / linear_beta if linear_beta is not None else None
    out = torch.empty((x_2d.shape[0], x.shape[-1] // 2), dtype=x.dtype, device=x.device)
    situ_and_mul(x_2d[:rows], out[:rows], beta, inv_beta, linear_beta, inv_linear_beta)
    return out.reshape(*x.shape[:-1], x.shape[-1] // 2)


@register_operator("activation.situ_and_mul")
def situ_and_mul(
    x, group_list=None, group_list_type=None, beta=4.0, linear_beta=25.0, *, impl="auto"
):
    """Process dense inputs or valid grouped rows; trailing capacity is unspecified."""
    return get_impl_by_name("activation.situ_and_mul", impl)(
        x, group_list, group_list_type, beta, linear_beta
    )
