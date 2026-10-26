"""Public swiglu_oai_quant operator with explicit implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("activation.swiglu_oai_quant", "auto")
@register_impl("activation.swiglu_oai_quant", "triton")
def _triton_impl(
    x, alpha, limit, need_quant=True, group_list=None, group_list_type=None
):
    import torch

    from ...implementation.triton.activation.swiglu_oai_quant import (
        _COL_BLOCK_SIZE,
        _INT8_MAX,
        _INT32_ALIGN_ELEMS,
        _INT64_ALIGN_ELEMS,
        _swiglu_oai_quant_kernel,
    )
    from ...utils.triton_utils import get_device_properties

    orig_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    s, h = x.shape
    # x is [gate | up] concatenated on the last dim; each half is one projection.
    half_cols = h // 2
    out_dtype = torch.int8 if need_quant else x.dtype
    out = torch.empty((s, half_cols), dtype=out_dtype, device=x.device)
    scale = torch.empty((s,), dtype=torch.float32, device=x.device)
    _, num_vectorcore = get_device_properties()

    if group_list is not None:
        # ── MoE grouped path ──────────────────────────────────────────
        if group_list_type not in (0, 1):
            raise ValueError(f"group_list_type must be 0 or 1, got {group_list_type}")
        num_experts = group_list.shape[0]
        # group_list loads must be 64-byte aligned on NPU; pad by index dtype.
        if group_list.dtype == torch.int64:
            num_experts_algin = _align_up(num_experts, _INT64_ALIGN_ELEMS)
        elif group_list.dtype == torch.int32:
            num_experts_algin = _align_up(num_experts, _INT32_ALIGN_ELEMS)
        else:
            raise ValueError(
                f"group_list dtype must be torch.int32 or torch.int64, "
                f"got {group_list.dtype}"
            )

        _swiglu_oai_quant_kernel[(num_vectorcore,)](
            x,
            group_list,
            out,
            scale,
            alpha,
            limit,
            0,  # total_rows: derived from group_list inside the kernel
            TOTAL_COLS=h,
            HALF_COLS=half_cols,
            COL_BLOCK_SIZE=_COL_BLOCK_SIZE,
            NUM_EXPERTS=num_experts,
            NUM_EXPERTS_ALGIN=num_experts_algin,
            GROUP_LIST_TYPE=group_list_type,
            NUM_CORES=num_vectorcore,
            DTYPE_MAX=_INT8_MAX,
            SCALE=need_quant,
            HAS_GROUP_LIST=True,
            multibuffer=True,
        )
    else:
        # ── Dense MLP path ────────────────────────────────────────────
        _swiglu_oai_quant_kernel[(num_vectorcore,)](
            x,
            None,  # group_list_ptr: unused in dense mode (HAS_GROUP_LIST=False)
            out,
            scale,
            alpha,
            limit,
            s,  # total_rows
            TOTAL_COLS=h,
            HALF_COLS=half_cols,
            COL_BLOCK_SIZE=_COL_BLOCK_SIZE,
            NUM_EXPERTS=1,
            NUM_EXPERTS_ALGIN=1,
            GROUP_LIST_TYPE=0,
            NUM_CORES=num_vectorcore,
            DTYPE_MAX=_INT8_MAX,
            SCALE=need_quant,
            HAS_GROUP_LIST=False,
            multibuffer=True,
        )

    return out.reshape(*orig_shape[:-1], -1), scale


@register_impl("activation.swiglu_oai_quant", "torch_native")
def _torch_native_impl(
    x, alpha, limit, need_quant=True, group_list=None, group_list_type=None
):
    import torch

    from ...implementation.torch_native.activation.swiglu_oai_quant import (
        swiglu_oai_quant,
    )

    flat = x.reshape(-1, x.shape[-1])
    rows, columns = flat.shape
    if columns % 2:
        raise ValueError("the last dimension must contain equally sized gate/up halves")
    valid = rows
    if group_list is not None:
        if group_list_type not in (0, 1):
            raise ValueError(f"group_list_type must be 0 or 1, got {group_list_type}")
        if group_list.dtype not in (torch.int32, torch.int64):
            raise ValueError("group_list dtype must be torch.int32 or torch.int64")
        count = group_list[-1] if group_list_type == 0 else group_list.sum()
        valid = int(count.item())
        if not 0 <= valid <= rows:
            raise ValueError("group_list row count exceeds input capacity")
    out = torch.empty(
        (rows, columns // 2),
        dtype=torch.int8 if need_quant else x.dtype,
        device=x.device,
    )
    scales = torch.empty(rows, dtype=torch.float32, device=x.device)

    swiglu_oai_quant(flat, alpha, limit, need_quant, valid, out, scales)
    return out.reshape(*x.shape[:-1], columns // 2), scales


@register_operator("activation.swiglu_oai_quant")
def swiglu_oai_quant(
    x,
    alpha,
    limit,
    need_quant=True,
    group_list=None,
    group_list_type=None,
    *,
    impl="auto",
):
    """Dispatch to triton by default; use impl="torch_native" for the reference.

    Scale is unwritten when need_quant=False; inactive capacity rows are also unspecified, as in the original kernel.
    """
    return get_impl_by_name("activation.swiglu_oai_quant", impl)(
        x, alpha, limit, need_quant, group_list, group_list_type
    )


def _align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment
